// src/utils/solver.js
// Solver 枚举引擎 + Pareto 过滤
// 给定模型，枚举 GPU × TP × PP × EP × quant × framework

import { GPU_LIST } from '../data/gpus/index.js'
import { QUANT_MAP, FRAMEWORK_MAP, INTERCONNECT_MAP } from '../data/constants.js'
import { aggregateGpuSlots, calcAll, getQuantBytes } from './calc.js'
import { resolveCpuMemBwOption } from '../data/runtime.js'
import {
  getDefaultGpuMemoryUtilization,
  normalizeGpuMemoryUtilization,
} from './runtime.js'

// Solver 使用的量化列表（跳过 fp32，保留 bf16/fp8/int8/int6/int5/int4）
export const SOLVER_QUANTS = QUANT_MAP.filter(q => q.id !== 'fp32' && q.id !== 'int3' && q.id !== 'int2')

// 量化质量下限选项
export const QUANT_FLOOR_OPTIONS = [
  { id: 'none', label_zh: '不限',       label_en: 'Any',        minQuality: null },
  { id: 'int4', label_zh: '≥ INT4',     label_en: '≥ INT4',     minQuality: 'ok' },
  { id: 'int8', label_zh: '≥ INT8/FP8', label_en: '≥ INT8',     minQuality: 'good' },
  { id: 'bf16', label_zh: '仅 BF16',    label_en: 'BF16 only',  minQuality: 'great' },
]

// Quant floors are precision filters, not broad quality labels. Several formats share
// a quality label, so filtering by quality would incorrectly include INT5/INT6 for
// the INT8 floor and FP8 for the BF16-only floor.
const QUANT_FLOOR_IDS = {
  none: new Set(SOLVER_QUANTS.map(q => q.id)),
  int4: new Set(['bf16', 'fp8', 'int8', 'int6', 'int5', 'int4']),
  int8: new Set(['bf16', 'fp8', 'int8']),
  bf16: new Set(['bf16']),
}

const SOLVER_GPU_COUNTS = [1, 2, 4, 8]

function getDefaultPcieInterconnect(gpu) {
  const generation = Number(gpu?.pcie_gen)
  const id = Number.isFinite(generation) && generation <= 3
    ? 'pcie3'
    : generation >= 5
      ? 'pcie5'
      : 'pcie4'
  return INTERCONNECT_MAP.find(i => i.id === id)
    ?? INTERCONNECT_MAP.find(i => i.id === 'pcie4')
}

function getPpOptions(model, totalGpuCount) {
  if (totalGpuCount < 2 || (model?.params ?? 0) < 30) return [1]
  return SOLVER_GPU_COUNTS.filter(n => n <= totalGpuCount && totalGpuCount % n === 0)
}

function getEpOptions(model, stageGpuCount) {
  if (model?.type !== 'moe' || model?.experts == null || stageGpuCount < 2) return [1]
  const maxEp = Math.min(model.experts, stageGpuCount)
  const options = [1]
  for (const n of SOLVER_GPU_COUNTS) {
    if (n > 1 && n <= maxEp && model.experts % n === 0 && stageGpuCount % n === 0) options.push(n)
  }
  return options
}

function makeResultKey(result) {
  return [
    result.gpu.id,
    result.gpuCount,
    result.ppCount,
    result.epCount,
    result.quant.id,
  ].join('|')
}

function pickPreferredResult(current, next) {
  if (!current) return next
  if (next.decodeSpeed !== current.decodeSpeed) return next.decodeSpeed > current.decodeSpeed ? next : current
  if (next.ttft !== current.ttft) return next.ttft < current.ttft ? next : current
  if (next.vramNeeded !== current.vramNeeded) return next.vramNeeded < current.vramNeeded ? next : current
  return next.framework.id < current.framework.id ? next : current
}

function pruneEquivalentResults(results) {
  const bestByKey = new Map()
  for (const result of results) {
    const key = makeResultKey(result)
    bestByKey.set(key, pickPreferredResult(bestByKey.get(key), result))
  }
  return [...bestByKey.values()]
}

function isCancelled(shouldCancel) {
  return shouldCancel?.() === true
}

function validateWorkload(model, { ctx, batch, promptLen, outputLen }) {
  if (model?.localInference === false) {
    return {
      code: 'model_local_inference_unsupported',
      modelId: model.id,
      modelName: model.name,
    }
  }
  const values = { ctx, batch, promptLen, outputLen }
  for (const [field, value] of Object.entries(values)) {
    if (!Number.isFinite(value) || value <= 0 || !Number.isInteger(value)) {
      return { code: 'invalid_workload', field, value }
    }
  }

  const maxCtx = Number(model?.max_ctx)
  if (Number.isFinite(maxCtx) && maxCtx > 0 && ctx > maxCtx) {
    return {
      code: 'context_exceeds_model_limit',
      ctx,
      maxCtx,
      modelId: model?.id,
      modelName: model?.name,
    }
  }

  const sequenceLength = promptLen + outputLen
  if (sequenceLength > ctx) {
    return {
      code: 'workload_exceeds_context',
      ctx,
      promptLen,
      outputLen,
      sequenceLength,
    }
  }

  return null
}

/**
 * 计算 MoE 模型的 non-expert 参数量（与 calc.js 保持一致）
 * @param {object} model - 模型对象
 * @returns {number|null} - non-expert 参数量（B），非 MoE 返回 null
 */
function calcNonExpertParams(model) {
  if (model.type !== 'moe' || !model.experts || !model.active_params) return null
  const denom = (model.experts_per_token ?? 1) - model.experts
  if (denom === 0) return model.active_params
  const ne = (model.params * (model.experts_per_token ?? 1) - model.experts * model.active_params) / denom
  return Math.max(0, ne)
}

/**
 * 根据 gpuCount 和 GPU 的 nvlink_bw 自动选择互联方式
 */
export function autoInterconnect(gpu, gpuCount) {
  if (gpuCount === 1) return getDefaultPcieInterconnect(gpu)
  if (gpu?.unitKind === 'system') {
    // Catalog system entries already include their internal NVLink/NVSwitch
    // fabric. Multiple units need an explicit inter-system network.
    return INTERCONNECT_MAP.find(i => i.id === 'ib_ndr')
      ?? INTERCONNECT_MAP.find(i => i.scope === 'inter')
      ?? getDefaultPcieInterconnect(gpu)
  }
  const duplexBw = Number(gpu?.nvlink_bw)
  if (Number.isFinite(duplexBw) && duplexBw > 0 && !gpu?.mixedGpu) {
    return INTERCONNECT_MAP.find(i => (
      i.id.startsWith('nvlink')
      && Number(i.duplexBw ?? i.bw * 2) === duplexBw
    )) ?? {
      id: `nvlink_${duplexBw}`,
      label: `NVLink (${duplexBw} GB/s duplex)`,
      bw: duplexBw / 2,
      duplexBw,
      scope: 'intra',
      derived: true,
    }
  }
  return getDefaultPcieInterconnect(gpu)
}

function isDiscreteSolverGpu(gpu) {
  return !['system', 'cpu'].includes(gpu?.unitKind)
}

function replacementInterconnect(gpu, gpuCount, preferred) {
  if (!preferred) return autoInterconnect(gpu, gpuCount)
  if (preferred.id?.startsWith('pcie')) {
    const requestedGeneration = Number(String(preferred.id).replace('pcie', ''))
    const supportedGeneration = Number(gpu?.pcie_gen) || 4
    return requestedGeneration <= supportedGeneration
      ? preferred
      : getDefaultPcieInterconnect(gpu)
  }
  if (gpuCount <= 1) return getDefaultPcieInterconnect(gpu)
  if (preferred.scope === 'inter') return preferred
  const gpuDuplexBw = Number(gpu?.nvlink_bw)
  const preferredDuplexBw = Number(preferred.duplexBw ?? preferred.bw * 2)
  return Number.isFinite(gpuDuplexBw) && gpuDuplexBw === preferredDuplexBw
    ? preferred
    : autoInterconnect(gpu, gpuCount)
}

function capSharedMemoryGpu(gpu, cpuMemBw, sysRam) {
  if (!gpu?.sharedMemory) return gpu
  const ram = Number(sysRam)
  const availableRam = Number.isFinite(ram) && ram > 0 ? ram * 0.9 : 57.6
  const memoryBw = Number(cpuMemBw?.bw)
  // The estimator's shared-memory allocation control is integer GB. Keep
  // solver candidates reproducible by using the same granularity instead of
  // recommending (for example) 57.6 GB and reopening at the 16 GB default.
  const sharedVram = gpu.vram > 0
    ? Math.min(gpu.vram, availableRam)
    : Math.max(1, Math.floor(availableRam))
  return {
    ...gpu,
    vram: sharedVram,
    bw: Number.isFinite(memoryBw) && memoryBw > 0 ? Math.min(gpu.bw, memoryBw) : gpu.bw,
    sharedMemoryConfigured: true,
    catalogVram: gpu.vram,
    sharedVram,
  }
}

function applyGpuMemoryUtilization(gpu, value, framework = null) {
  if (!gpu) return gpu
  const fallback = getDefaultGpuMemoryUtilization(framework, gpu)
  const usableRatio = normalizeGpuMemoryUtilization(value, fallback)
  return {
    ...gpu,
    usableRatio,
  }
}

/**
 * 判断框架是否支持该 GPU 厂商
 */
function frameworkSupportsGpu(framework, gpu) {
  if (!framework.vendors) return true
  return framework.vendors.includes(gpu.vendor)
}

/**
 * 模式 A：给定模型，枚举所有 GPU × gpuCount × quant × framework 组合
 *
 * @param {object} opts
 * @param {object} opts.model           - 模型对象
 * @param {number} opts.maxGpuCount     - 最大 GPU 数量（1/2/4/8）
 * @param {string|null} opts.vendorFilter - GPU 厂商过滤（null = 全部）
 * @param {boolean} opts.excludeDatacenterGpu - 是否排除数据中心卡（默认 false）
 * @param {string} opts.quantFloor      - 量化质量下限 id
 * @param {number|null} opts.minDecodeSpeed - 最低 decode 速度（tok/s）
 * @param {number|null} opts.maxTtft    - 最大 TTFT（ms）
 * @param {number} opts.ctx             - 上下文长度
 * @param {number} opts.batch           - 并发数
 * @param {number} opts.promptLen       - Prompt 长度
 * @param {number} opts.outputLen       - 输出长度
 * @param {boolean} opts.disableYield   - 是否禁用分批让出主线程（默认 false）
 * @param {function} opts.onProgress    - 进度回调 (done, total)
 * @param {function} opts.shouldCancel  - 取消检查函数
 * @returns {Promise<{ results: SolverResult[], cancelled: boolean }>}
 */
export async function solveForModel(opts) {
  const {
    model, maxGpuCount = 4, vendorFilter = null, excludeDatacenterGpu = false,
    quantFloor = 'none', minDecodeSpeed = null, maxTtft = null,
    ctx = 4096, batch = 1, promptLen = 512, outputLen = 256,
    cpuMemBw = resolveCpuMemBwOption('ddr5_4800'), cpuTflops = null,
    gpuMemoryUtilization = null, sysRam = 64,
    disableYield = false,
    onProgress,
    shouldCancel,
  } = opts

  const validationError = validateWorkload(model, { ctx, batch, promptLen, outputLen })
  if (validationError) return { results: [], cancelled: false, validationError }

  const totalGpuCounts = SOLVER_GPU_COUNTS.filter(n => n <= maxGpuCount)

  // 过滤量化列表。未知 floor 安全地回退到“不限”。
  const allowedQuantIds = QUANT_FLOOR_IDS[quantFloor] ?? QUANT_FLOOR_IDS.none
  const quants = SOLVER_QUANTS.filter(q => allowedQuantIds.has(q.id))

  // 过滤 GPU 列表
  const gpus = GPU_LIST.filter(g => {
    if (!isDiscreteSolverGpu(g)) return false
    if (vendorFilter && vendorFilter !== 'all') {
      if (g.vendor !== vendorFilter) return false
    }
    if (excludeDatacenterGpu && g.tier === 'datacenter') return false
    return true
  }).map(gpu => capSharedMemoryGpu(gpu, cpuMemBw, sysRam))

  // 过滤框架（跳过 theory，只保留实际框架）
  const frameworks = FRAMEWORK_MAP.filter(f => f.id !== 'theory')

  // 预计算 MoE 模型的 non-expert 参数量
  const nonExpertParams = calcNonExpertParams(model)
  const totalExpertParams = nonExpertParams != null ? model.params - nonExpertParams : null

  // 预估最低显存需求（EP+TP 最大分片下的 INT4 每卡需求），用于第一层剪枝
  const maxPossibleEp = (model.type === 'moe' && model.experts)
    ? Math.min(model.experts, maxGpuCount)
    : 1
  const minWeightPerCardGB = (nonExpertParams != null)
    ? ((nonExpertParams + (model.params - nonExpertParams) / maxPossibleEp) * (QUANT_MAP.find(q => q.id === 'int4')?.gguf_bytes ?? 0.615)) / maxGpuCount
    : (model.params * (QUANT_MAP.find(q => q.id === 'int4')?.gguf_bytes ?? 0.615)) / maxGpuCount

  // 构建任务列表（提前剪枝）
  const tasks = []
  for (const gpu of gpus) {
    // 第一层剪枝：单卡显存连 EP+TP 最大分片下的 INT4 都装不下，跳过该 GPU
    const usableVram = gpu.vram * (gpu.usableRatio ?? 1.0)
    if (usableVram < minWeightPerCardGB * 0.8) continue

    // Apple Silicon 的统一内存属于一台设备，不能按离散 GPU 数量横向叠加。
    const deviceGpuCounts = gpu.vendor === 'apple' || gpu.sharedMemory
      ? totalGpuCounts.filter(n => n === 1)
      : totalGpuCounts

    for (const totalGpuCount of deviceGpuCounts) {
      // 提前剪枝：极小模型（< 1B）跳过多卡配置（1 卡足够，多卡无意义）
      if (model.params < 1 && totalGpuCount > 1) continue

      for (const ppCount of getPpOptions(model, totalGpuCount)) {
        const stageGpuCount = totalGpuCount / ppCount
        const interconnect = autoInterconnect(gpu, totalGpuCount)
        const epOptions = getEpOptions(model, stageGpuCount)

        for (const quant of quants) {
          for (const framework of frameworks) {
            if (!frameworkSupportsGpu(framework, gpu)) continue
            const runtimeGpu = applyGpuMemoryUtilization(
              gpu,
              gpuMemoryUtilization,
              framework,
            )
            const runtimeUsableVram = runtimeGpu.vram
              * (runtimeGpu.usableRatio ?? 1)

            // 提前剪枝：极小模型（< 1B）只保留高效框架
            if (model.params < 1 && !['vllm', 'sglang', 'llamacpp', 'mlx', 'llamacpp_metal'].includes(framework.id)) continue

            const quantBytes = getQuantBytes(quant, runtimeGpu, framework)

            // EP 循环提前到 quant 同层，让剪枝能感知 EP 分片。
            for (const epCount of epOptions) {
              const tpCount = stageGpuCount / epCount
              let perCardWeightGB
              if (epCount > 1 && totalExpertParams != null) {
                // MoE + EP: dense 部分按 TP 分片，expert 部分按 EP × TP 分片。
                perCardWeightGB = (
                  nonExpertParams / (ppCount * tpCount)
                  + totalExpertParams / (ppCount * epCount * tpCount)
                ) * quantBytes
              } else {
                // Dense 或 EP=1: 参数先按 PP stage，再按 stage 内 TP 分片。
                perCardWeightGB = (model.params * quantBytes) / ppCount / tpCount
              }

              // 与单卡可用显存比较（而非总显存）
              if (perCardWeightGB > runtimeUsableVram * 1.1) continue

              tasks.push({
                gpu: runtimeGpu,
                totalGpuCount,
                stageGpuCount,
                tpCount,
                ppCount,
                epCount,
                interconnect,
                quant,
                framework,
              })
            }
          }
        }
      }
    }
  }

  const total = tasks.length
  const results = []
  const BATCH_SIZE = 50

  for (let i = 0; i < tasks.length; i += BATCH_SIZE) {
    if (isCancelled(shouldCancel)) return { results: [], cancelled: true }
    const chunk = tasks.slice(i, i + BATCH_SIZE)
    for (const task of chunk) {
      if (isCancelled(shouldCancel)) return { results: [], cancelled: true }
      try {
        const r = calcAll({
          gpu: task.gpu,
          gpuCount: task.totalGpuCount,
          ppCount: task.ppCount,
          epCount: task.epCount,
          interconnect: task.interconnect,
          model,
          quant: task.quant,
          ctx,
          batch,
          promptLen,
          outputLen,
          framework: task.framework,
          flashAttention: true,
          cpuMemBw,
          cpuTflops,
          sysRam,
        })

        if (!(r.fitOk ?? r.vramOk)) continue
        if (minDecodeSpeed != null && r.singleToks < minDecodeSpeed) continue
        if (maxTtft != null && r.ttft > maxTtft) continue

        const row = {
          gpu: task.gpu,
          gpuCount: r.totalGpuCount ?? task.totalGpuCount,
          totalGpuCount: r.totalGpuCount ?? task.totalGpuCount,
          stageGpuCount: r.stageGpuCount ?? task.stageGpuCount,
          tpCount: r.tpCount ?? task.tpCount,
          ppCount: r.ppCount ?? task.ppCount,
          epCount: r.epCount ?? task.epCount,
          interconnect: task.interconnect,
          quant: task.quant,
          framework: task.framework,
          vramNeeded: r.displayNeeded ?? r.totalNeeded,
          decodeSpeed: r.singleToks,
          ttft: r.ttft,
          totalVram: r.totalVram,
          vramPct: r.vramPct,
          tpEfficiency: r.tpEfficiency,
          bottleneck: r.bottleneck,
        }
        row.insightKeys = generateInsightKeys(row)
        results.push(row)
      } catch {
        // 忽略计算错误
      }
    }

    onProgress?.(Math.min(i + BATCH_SIZE, total), total)
    // 让出主线程（某些环境 requestIdleCallback 可能导致卡住，可按需禁用）
    if (!disableYield) {
      await yieldToMain()
    }
  }

  return { results: computePareto(pruneEquivalentResults(results)), cancelled: false }
}

/**
 * 为每条结果生成洞察建议
 * @param {object} row - 单条 solver 结果
 * @returns {string[]} - i18n keys
 */
function generateInsightKeys(row) {
  const keys = []

  // 显存余量检查
  if (row.vramPct > 85) {
    keys.push('solver.insight_vram_tight')
  }
  if (row.vramPct < 30) {
    keys.push('solver.insight_vram_room')
  }

  // 多卡效率检查
  if ((row.tpCount ?? row.gpuCount) > 1 && row.tpEfficiency != null) {
    if (row.tpEfficiency < 0.75) {
      keys.push('solver.insight_interconnect_low')
    } else if (row.tpEfficiency > 0.95) {
      keys.push('solver.insight_scale_excellent')
    }
  }

  // 量化质量检查
  if (row.quant?.quality === 'ok' || row.quant?.quality === 'poor') {
    keys.push('solver.insight_quant_quality')
  }

  // 瓶颈检查
  if (row.bottleneck === 'compute') {
    keys.push('solver.insight_compute_bottleneck')
  }

  return keys
}

/**
 * 计算 Pareto 前沿
 * 目标：速度越大越好，显存越小越好，GPU 数量越少越好
 *
 * @param {object[]} results
 */
function computePareto(results) {
  if (results.length === 0) return []

  const entries = results.map((result, index) => ({
    result,
    index,
    speed: result.decodeSpeed,
    vram: result.vramNeeded,
    gpuCount: result.totalGpuCount ?? result.gpuCount,
  })).sort((a, b) =>
    b.speed - a.speed ||
    a.vram - b.vram ||
    a.gpuCount - b.gpuCount
  )

  const dominated = new Array(results.length).fill(false)
  // For all strictly faster speed groups, retain the lowest VRAM seen for each
  // physical GPU count. GPU counts have very low cardinality, so the prefix
  // dominance query is both exact and inexpensive.
  const fasterBestVramByGpuCount = new Map()

  let groupStart = 0
  while (groupStart < entries.length) {
    let groupEnd = groupStart + 1
    while (groupEnd < entries.length && entries[groupEnd].speed === entries[groupStart].speed) {
      groupEnd++
    }

    const speedGroup = entries.slice(groupStart, groupEnd)
    let minGpuAtLowerVram = Infinity
    let vramStart = 0

    while (vramStart < speedGroup.length) {
      let vramEnd = vramStart + 1
      while (vramEnd < speedGroup.length && speedGroup[vramEnd].vram === speedGroup[vramStart].vram) {
        vramEnd++
      }

      const sameVram = speedGroup.slice(vramStart, vramEnd)
      const minGpuAtSameVram = sameVram[0].gpuCount

      for (const entry of sameVram) {
        const dominatedByFaster = [...fasterBestVramByGpuCount].some(
          ([gpuCount, bestVram]) => gpuCount <= entry.gpuCount && bestVram <= entry.vram
        )
        const dominatedAtSameSpeed =
          minGpuAtLowerVram <= entry.gpuCount ||
          minGpuAtSameVram < entry.gpuCount

        dominated[entry.index] = dominatedByFaster || dominatedAtSameSpeed
      }

      minGpuAtLowerVram = Math.min(minGpuAtLowerVram, minGpuAtSameVram)
      vramStart = vramEnd
    }

    for (const entry of speedGroup) {
      const previous = fasterBestVramByGpuCount.get(entry.gpuCount)
      if (previous == null || entry.vram < previous) {
        fasterBestVramByGpuCount.set(entry.gpuCount, entry.vram)
      }
    }

    groupStart = groupEnd
  }

  return entries.map(entry => ({
    ...entry.result,
    isPareto: !dominated[entry.index],
  }))
}

/**
 * 让出主线程，避免阻塞 UI
 */
function yieldToMain() {
  return new Promise(resolve => {
    let done = false
    const finish = () => {
      if (done) return
      done = true
      resolve()
    }

    if (typeof requestIdleCallback !== 'undefined') {
      // Some environments may throttle/skip idle callbacks; timer ensures progress.
      requestIdleCallback(finish, { timeout: 50 })
      setTimeout(finish, 60)
    } else {
      setTimeout(finish, 0)
    }
  })
}

/**
 * 升级路径求解器：给定当前配置，枚举最小改动方案以达到目标速度
 *
 * @param {object} opts
 * @param {object} opts.currentGpu      - 当前 GPU 对象
 * @param {number} opts.currentGpuCount - 当前 GPU 数量
 * @param {object} opts.currentQuant    - 当前量化精度
 * @param {object} opts.model           - 模型对象
 * @param {number} opts.targetSpeed     - 目标速度（tok/s）
 * @param {number} opts.ctx             - 上下文长度
 * @param {number} opts.batch           - 并发数
 * @param {number} opts.promptLen       - Prompt 长度
 * @param {number} opts.outputLen       - 输出长度
 * @param {function} opts.onProgress    - 进度回调
 * @param {function} opts.shouldCancel  - 取消检查函数
 * @returns {Promise<{ results: Array, cancelled: boolean }>}
 */
export async function solveUpgrade(opts) {
  const {
    currentGpu: suppliedCurrentGpu,
    currentGpuCount: suppliedCurrentGpuCount,
    gpuSlots: suppliedGpuSlots,
    currentQuant, model,
    targetSpeed, ctx = 4096, batch = 1, promptLen = 512, outputLen = 256,
    currentFramework = null,
    currentInterconnect = null,
    flashAttention = true,
    kvCacheQuant = null,
    prefixCacheHit = 0,
    cpuOffload = false,
    pcieBw = null,
    pcieWidth = null,
    pureCpu = false,
    cpuMemBw = resolveCpuMemBwOption('ddr5_4800'),
    cpuTflops = null,
    gpuMemoryUtilization = null,
    sysRam = 64,
    speculativeDecoding = false,
    acceptanceRate = 0.7,
    draftLen = 4,
    draftModelParams = 1,
    ppCount = 1,
    epCount = 1,
    imageCount = 0,
    nglCount = null,
    onProgress, shouldCancel,
  } = opts

  const validationError = validateWorkload(model, { ctx, batch, promptLen, outputLen })
  if (validationError) return { results: [], cancelled: false, validationError }
  const gpuSlots = (suppliedGpuSlots?.length
    ? suppliedGpuSlots
    : suppliedCurrentGpu
      ? [{ gpu: suppliedCurrentGpu, count: suppliedCurrentGpuCount ?? 1 }]
      : []
  ).filter(slot => slot?.gpu).map(slot => ({
    gpu: capSharedMemoryGpu(slot.gpu, cpuMemBw, sysRam),
    count: Math.max(1, Math.round(Number(slot.count) || 1)),
  }))
  const currentGpuCount = gpuSlots.reduce((sum, slot) => sum + slot.count, 0)
  const currentGpu = gpuSlots.length > 1
    ? aggregateGpuSlots(gpuSlots)
    : gpuSlots[0]?.gpu ?? suppliedCurrentGpu
  if (!currentGpu || !currentQuant || currentGpuCount < 1) {
    return {
      results: [],
      cancelled: false,
      validationError: { code: 'invalid_current_configuration' },
    }
  }
  if (currentGpu?.vendor === 'apple' && currentGpuCount !== 1) {
    return {
      results: [],
      cancelled: false,
      validationError: {
        code: 'apple_multi_device_unsupported',
        gpuCount: currentGpuCount,
      },
    }
  }

  const upgradePaths = []
  const frameworks = currentFramework
    ? [currentFramework]
    : FRAMEWORK_MAP.filter(f => f.id !== 'theory')
  const runtimeArgs = {
    flashAttention,
    kvCacheQuant,
    prefixCacheHit,
    cpuOffload,
    pcieBw,
    pcieWidth,
    pureCpu,
    cpuMemBw,
    cpuTflops,
    gpuMemoryUtilization,
    sysRam,
    speculativeDecoding,
    acceptanceRate,
    draftLen,
    draftModelParams,
    ppCount,
    epCount,
    imageCount,
    nglCount,
  }
  const currentCapacity = gpuSlots.reduce(
    (sum, slot) => sum + Number(slot.gpu?.vram ?? 0) * slot.count,
    0,
  )

  // 策略 1: 增加同型号 GPU（2x, 4x, 8x）
  const addGpuCounts = pureCpu
    || currentGpu?.vendor === 'apple'
    || currentGpu?.sharedMemory
    || gpuSlots.some(slot => !isDiscreteSolverGpu(slot.gpu))
    ? []
    : [currentGpuCount * 2, currentGpuCount * 4, currentGpuCount * 8]
  for (const newCount of addGpuCounts) {
    if (isCancelled(shouldCancel)) return { results: [], cancelled: true }
    if (newCount > 8) continue
    const interconnect = currentInterconnect ?? autoInterconnect(currentGpu, newCount)
    const scale = newCount / currentGpuCount
    const nextGpuSlots = gpuSlots.map(slot => ({
      gpu: slot.gpu,
      count: Math.max(1, Math.round(slot.count * scale)),
    }))

    for (const framework of frameworks) {
      if (isCancelled(shouldCancel)) return { results: [], cancelled: true }
      if (!frameworkSupportsGpu(framework, currentGpu)) continue
      const runtimeGpu = applyGpuMemoryUtilization(
        currentGpu,
        gpuMemoryUtilization,
        framework,
      )

      try {
        const r = calcAll({
          gpu: runtimeGpu,
          gpuCount: newCount,
          interconnect,
          model,
          quant: currentQuant,
          ctx,
          batch,
          promptLen,
          outputLen,
          framework,
          ...runtimeArgs,
        })

        if ((r.fitOk ?? r.vramOk) && r.singleToks >= targetSpeed) {
          upgradePaths.push({
            type: 'add_gpu',
            gpu: runtimeGpu,
            gpuSlots: nextGpuSlots,
            gpuCount: r.totalGpuCount ?? newCount,
            totalGpuCount: r.totalGpuCount ?? newCount,
            stageGpuCount: r.stageGpuCount ?? newCount,
            tpCount: r.tpCount ?? newCount,
            ppCount: r.ppCount ?? 1,
            epCount: r.epCount ?? 1,
            interconnect,
            quant: currentQuant,
            framework,
            vramNeeded: r.displayNeeded ?? r.totalNeeded,
            decodeSpeed: r.singleToks,
            ttft: r.ttft,
            totalVram: r.totalVram,
            vramPct: r.vramPct,
            tpEfficiency: r.tpEfficiency,
            bottleneck: r.bottleneck,
            changeKey: 'solver.change_add_gpu',
            changeParams: { count: newCount, gpu: currentGpu.name },
            relativeCapacity: newCount / currentGpuCount,
            runtime: runtimeArgs,
          })
        }
      } catch {
        // 忽略计算错误
      }
    }
  }

  // 策略 2: 更改量化（包括通常能提升速度/降低内存的低精度格式）
  const alternativeQuants = SOLVER_QUANTS.filter(q => q.id !== currentQuant.id)
  for (const quant of alternativeQuants) {
    if (isCancelled(shouldCancel)) return { results: [], cancelled: true }
    const interconnect = currentInterconnect ?? autoInterconnect(currentGpu, currentGpuCount)

    for (const framework of frameworks) {
      if (isCancelled(shouldCancel)) return { results: [], cancelled: true }
      if (!frameworkSupportsGpu(framework, currentGpu)) continue
      const runtimeGpu = applyGpuMemoryUtilization(
        currentGpu,
        gpuMemoryUtilization,
        framework,
      )

      try {
        const r = calcAll({
          gpu: runtimeGpu,
          gpuCount: currentGpuCount,
          interconnect,
          model,
          quant,
          ctx,
          batch,
          promptLen,
          outputLen,
          framework,
          ...runtimeArgs,
        })

        if ((r.fitOk ?? r.vramOk) && r.singleToks >= targetSpeed) {
          upgradePaths.push({
            type: 'upgrade_quant',
            gpu: runtimeGpu,
            gpuSlots,
            gpuCount: r.totalGpuCount ?? currentGpuCount,
            totalGpuCount: r.totalGpuCount ?? currentGpuCount,
            stageGpuCount: r.stageGpuCount ?? currentGpuCount,
            tpCount: r.tpCount ?? currentGpuCount,
            ppCount: r.ppCount ?? 1,
            epCount: r.epCount ?? 1,
            interconnect,
            quant,
            framework,
            vramNeeded: r.displayNeeded ?? r.totalNeeded,
            decodeSpeed: r.singleToks,
            ttft: r.ttft,
            totalVram: r.totalVram,
            vramPct: r.vramPct,
            tpEfficiency: r.tpEfficiency,
            bottleneck: r.bottleneck,
            changeKey: 'solver.change_quant',
            changeParams: { quant: quant.label },
            relativeCapacity: 1,
            runtime: runtimeArgs,
          })
        }
      } catch {
        // 忽略计算错误
      }
    }
  }

  // 策略 3: 用同厂商离散 GPU 替换所有当前卡。包含更快的同显存卡。
  const canReplaceGpu = !pureCpu && gpuSlots.every(slot => isDiscreteSolverGpu(slot.gpu))
  const sameVendorGpus = canReplaceGpu
    ? GPU_LIST.filter(g =>
        isDiscreteSolverGpu(g)
        && g.vendor === currentGpu.vendor
        && !gpuSlots.some(slot => slot.gpu.id === g.id)
      ).sort((a, b) => a.vram - b.vram || b.bw - a.bw)
    : []

  for (const newGpu of sameVendorGpus) {
    if (isCancelled(shouldCancel)) return { results: [], cancelled: true }
    const candidateGpu = capSharedMemoryGpu(newGpu, cpuMemBw, sysRam)
    const interconnect = replacementInterconnect(candidateGpu, currentGpuCount, currentInterconnect)
    const nextGpuSlots = [{ gpu: candidateGpu, count: currentGpuCount }]

    for (const framework of frameworks) {
      if (isCancelled(shouldCancel)) return { results: [], cancelled: true }
      if (!frameworkSupportsGpu(framework, candidateGpu)) continue
      const runtimeGpu = applyGpuMemoryUtilization(
        candidateGpu,
        gpuMemoryUtilization,
        framework,
      )

      try {
        const r = calcAll({
          gpu: runtimeGpu,
          gpuCount: currentGpuCount,
          interconnect,
          model,
          quant: currentQuant,
          ctx,
          batch,
          promptLen,
          outputLen,
          framework,
          ...runtimeArgs,
        })

        if ((r.fitOk ?? r.vramOk) && r.singleToks >= targetSpeed) {
          upgradePaths.push({
            type: 'upgrade_gpu',
            gpu: runtimeGpu,
            gpuSlots: nextGpuSlots,
            gpuCount: r.totalGpuCount ?? currentGpuCount,
            totalGpuCount: r.totalGpuCount ?? currentGpuCount,
            stageGpuCount: r.stageGpuCount ?? currentGpuCount,
            tpCount: r.tpCount ?? currentGpuCount,
            ppCount: r.ppCount ?? ppCount,
            epCount: r.epCount ?? epCount,
            interconnect,
            quant: currentQuant,
            framework,
            vramNeeded: r.displayNeeded ?? r.totalNeeded,
            decodeSpeed: r.singleToks,
            ttft: r.ttft,
            totalVram: r.totalVram,
            vramPct: r.vramPct,
            tpEfficiency: r.tpEfficiency,
            bottleneck: r.bottleneck,
            changeKey: 'solver.change_gpu',
            changeParams: { gpu: candidateGpu.name, count: currentGpuCount },
            relativeCapacity: currentCapacity > 0
              ? candidateGpu.vram * currentGpuCount / currentCapacity
              : 1,
            runtime: runtimeArgs,
          })
        }
      } catch {
        // 忽略计算错误
      }
    }
  }

  onProgress?.(upgradePaths.length, upgradePaths.length)
  await yieldToMain()

  const rankedPaths = computePareto(pruneEquivalentResults(upgradePaths))

  // Capacity is a hardware-size comparison only; it is not a price estimate.
  rankedPaths.sort((a, b) => {
    if (a.relativeCapacity !== b.relativeCapacity) {
      return a.relativeCapacity - b.relativeCapacity
    }
    return b.decodeSpeed - a.decodeSpeed
  })

  for (const path of rankedPaths) {
    path.insightKeys = generateInsightKeys(path)
  }

  return { results: rankedPaths, cancelled: isCancelled(shouldCancel) }
}
