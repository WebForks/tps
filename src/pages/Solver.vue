<script setup>
import { ref, computed, watch, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter, useRoute } from 'vue-router'
import Header from '../components/layout/Header.vue'
import ModelPicker from '../components/config/ModelPicker.vue'
import ResultRowContent from '../components/result/SolverResultRow.vue'
import { ALL_MODELS, resolveModelId } from '../data/models/index.js'
import { GPU_LIST } from '../data/gpus/index.js'
import { FRAMEWORK_MAP, INTERCONNECT_MAP, QUANT_MAP } from '../data/constants.js'
import {
  KV_CACHE_MAP,
  PCIE_BW_OPTIONS,
  PCIE_WIDTH_OPTIONS,
  createCpuMemBwOption,
  normalizeCpuMemMeasuredBandwidth,
  normalizeCpuTflops,
  normalizeRamCapacity,
  resolveCpuMemBwOption,
} from '../data/runtime.js'
import { aggregateGpuSlots } from '../utils/calc.js'
import { readUrlState, resolveUrlState } from '../utils/useUrlState.js'
import {
  solveForModel,
  solveUpgrade,
  QUANT_FLOOR_OPTIONS,
} from '../utils/solver.js'
import { fmtGB, fmtToks, fmtMs } from '../utils/format.js'

const { t, locale } = useI18n()
const router = useRouter()
const route = useRoute()

const _p = route.query
const hasInitialQuery = Object.keys(_p).length > 0
const resolvedInitialUrl = hasInitialQuery
  ? resolveUrlState(readUrlState())
  : null

const LIMITS = {
  targetSpeed: { min: 1, max: 100000, def: 100 },
  maxGpuCount: { min: 1, max: 8, def: 4 },
  minDecodeSpeed: { min: 1, max: 100000 },
  maxTtft: { min: 1, max: 600000 },
  ctx: { min: 512, max: 10_485_760, def: 4096 },
  batch: { min: 1, max: 256, def: 1 },
  promptLen: { min: 1, max: 262144, def: 512 },
  outputLen: { min: 1, max: 131072, def: 256 },
}

function clampInt(value, { min, max, def }) {
  if (value === '' || value == null) return def
  const n = Number(value)
  if (!Number.isFinite(n)) return def
  const rounded = Math.round(n)
  return Math.min(max, Math.max(min, rounded))
}

function clampOptionalPositive(value, { min, max }) {
  if (value === '' || value == null) return ''
  const n = Number(value)
  if (!Number.isFinite(n)) return ''
  const rounded = Math.round(n)
  if (rounded < min) return ''
  return Math.min(max, rounded)
}

function addCustomModelQuery(query, selectedModel) {
  if (selectedModel?.id !== 'custom') return query
  Object.assign(query, {
    cmn: selectedModel.name,
    cmt: selectedModel.type,
    cmp: selectedModel.params,
    cma: selectedModel.type === 'moe' ? selectedModel.active_params : undefined,
    cml: selectedModel.layers,
    cmkv: selectedModel.kv_heads,
    cmqh: selectedModel.query_heads,
    cmhd: selectedModel.head_dim,
    cmhs: selectedModel.hidden_size,
    cmctx: selectedModel.max_ctx,
    cmexp: selectedModel.type === 'moe' ? selectedModel.experts : undefined,
    cmk: selectedModel.type === 'moe' ? selectedModel.experts_per_token : undefined,
    cmmla: selectedModel.mla_ratio,
  })
  return query
}

// ── 升级模式检测 ──────────────────────────────────────
const isUpgradeMode = ref(_p.upgrade === '1')

function parseCurrentGpuSlots() {
  let remaining = 512
  const parsed = []
  for (const entry of String(_p.gpus ?? '').split(',')) {
    if (remaining <= 0) break
    const [id, count] = entry.split(':')
    const gpu = GPU_LIST.find(item => item.id === id && item.unitKind !== 'cpu') ?? null
    if (!gpu) continue
    const normalizedCount = clampInt(count, { min: 1, max: remaining, def: 1 })
    parsed.push({ gpu, count: normalizedCount })
    remaining -= normalizedCount
  }
  if (parsed.length) return parsed

  const gpu = GPU_LIST.find(item => item.id === _p.gpu && item.unitKind !== 'cpu')
  return gpu
    ? [{ gpu, count: clampInt(_p.n, { min: 1, max: 512, def: 1 }) }]
    : []
}

const currentGpuSlots = ref(parseCurrentGpuSlots())
const currentGpuCount = computed(() => currentGpuSlots.value.reduce(
  (sum, slot) => sum + slot.count,
  0,
))
const currentGpu = computed(() => {
  if (!currentGpuSlots.value.length) return null
  return currentGpuSlots.value.length === 1
    ? currentGpuSlots.value[0].gpu
    : aggregateGpuSlots(currentGpuSlots.value)
})

const currentQuant = ref(QUANT_MAP.find(q => q.id === _p.quant) ?? QUANT_MAP.find(q => q.id === 'bf16'))
const targetSpeed = ref(clampInt(_p.target, LIMITS.targetSpeed))

// ── 模式 A：给定模型 ──────────────────────────────────
const restoredCustomModel = _p.model === 'custom'
  ? resolvedInitialUrl?.model
  : null
const modelA = ref(
  restoredCustomModel
  ?? ALL_MODELS.find(m => m.id === resolveModelId(_p.model))
  ?? ALL_MODELS.find(m => m.id === 'deepseek_r1')
  ?? ALL_MODELS[0]
)
const maxGpuCount = ref(clampInt(_p.maxg, LIMITS.maxGpuCount))
const vendorFilter = ref(['all', 'nvidia', 'amd', 'apple', 'intel', 'domestic'].includes(_p.vendor) ? _p.vendor : 'all')
const quantFloor = ref(QUANT_FLOOR_OPTIONS.some(q => q.id === _p.qf) ? _p.qf : 'none')
// 默认过滤数据中心卡（消费级优先）；仅当显式传入 excl_dc=0 时展示全部
const excludeDatacenterGpu = ref(_p.excl_dc !== '0')
const minDecodeSpeedA = ref(clampOptionalPositive(_p.minds && _p.minds !== 'none' ? _p.minds : '', LIMITS.minDecodeSpeed))
const maxTtft = ref(clampOptionalPositive(_p.mttft && _p.mttft !== 'none' ? _p.mttft : '', LIMITS.maxTtft))

// ── 共用参数 ──────────────────────────────────────────
const ctx = ref(clampInt(_p.ctx ?? LIMITS.ctx.def, LIMITS.ctx))
const batch = ref(clampInt(_p.b ?? LIMITS.batch.def, LIMITS.batch))
const promptLen = ref(clampInt(_p.pl ?? LIMITS.promptLen.def, LIMITS.promptLen))
const outputLen = ref(clampInt(_p.ol ?? LIMITS.outputLen.def, LIMITS.outputLen))

function parseOffloadMode(value) {
  if (value === '1' || value === 'on') return 'on'
  if (value === '0' || value === 'off') return 'off'
  return 'auto'
}

const currentFramework = ref(
  resolvedInitialUrl?.framework
  ?? FRAMEWORK_MAP.find(option => option.id === _p.fw)
  ?? null
)
const currentInterconnect = ref(
  resolvedInitialUrl?.interconnect
  ?? INTERCONNECT_MAP.find(option => option.id === _p.ic)
  ?? null
)
const flashAttention = ref(resolvedInitialUrl?.flashAttention ?? _p.fa !== '0')
const kvCacheQuant = ref(
  resolvedInitialUrl?.kvCacheQuant
  ?? KV_CACHE_MAP.find(option => option.id === _p.kv)
  ?? KV_CACHE_MAP[0]
)
const prefixCacheHit = ref(Math.min(90, Math.max(0, Number(_p.pc) || 0)))
const cpuOffloadMode = ref(resolvedInitialUrl?.cpuOffloadMode ?? parseOffloadMode(_p.co))
const pureCpu = ref(resolvedInitialUrl?.pureCpu ?? _p.pcpu === '1')
const cpuOffload = computed(() => cpuOffloadMode.value === 'on' && !pureCpu.value)
const pcieBw = ref(
  resolvedInitialUrl?.pcieBw
  ?? PCIE_BW_OPTIONS.find(option => option.id === _p.pcie)
  ?? PCIE_BW_OPTIONS[1]
)
const pcieWidth = ref(
  resolvedInitialUrl?.pcieWidth
  ?? PCIE_WIDTH_OPTIONS.find(option => option.id === _p.pw)
  ?? PCIE_WIDTH_OPTIONS[1]
)
const baseCpuMemBw = resolvedInitialUrl?.cpuMemBw
  ?? resolveCpuMemBwOption(_p.cmb)
  ?? resolveCpuMemBwOption('ddr5_4800')
const cpuMemBw = ref(createCpuMemBwOption(
  baseCpuMemBw.generation,
  baseCpuMemBw.transferRate,
  clampInt(_p.cmc ?? baseCpuMemBw.channels, { min: 1, max: 16, def: 2 }),
  normalizeCpuMemMeasuredBandwidth(_p.cmm, baseCpuMemBw.measuredBw),
))
const cpuTflops = ref(resolvedInitialUrl?.cpuTflops ?? normalizeCpuTflops(_p.ctf))
const gpuMemoryUtilization = ref(
  resolvedInitialUrl?.gpuMemoryUtilization
  ?? (_p.gmu == null ? null : Math.min(1, Math.max(0.5, Number(_p.gmu) || 0.9)))
)
const sysRam = ref(normalizeRamCapacity(resolvedInitialUrl?.sysRam ?? _p.sr, 64))
const speculativeDecoding = ref(_p.sd === '1')
const acceptanceRate = ref(Math.min(0.9, Math.max(0.3, Number(_p.ar) || 0.7)))
const draftLen = ref(clampInt(_p.dl, { min: 2, max: 8, def: 4 }))
const draftModelParams = ref(Math.min(32, Math.max(0.1, Number(_p.dmp) || 1)))
const ppCount = ref(clampInt(_p.pp, { min: 1, max: 512, def: 1 }))
const epCount = ref(clampInt(_p.ep, { min: 1, max: 512, def: 1 }))
const imageCount = ref(clampInt(_p.img, { min: 0, max: 8, def: 0 }))
const nglCount = ref(_p.ngl == null
  ? null
  : clampInt(_p.ngl, { min: 0, max: 1000, def: 0 }))

watch([modelA, draftModelParams, nglCount], ([selectedModel, draftParams, ngl]) => {
  const draftCap = Math.max(
    0.5,
    Math.min(32, Number(selectedModel?.active_params ?? selectedModel?.params ?? 8) * 0.5),
  )
  const normalizedDraft = Math.min(draftCap, Math.max(0.1, Number(draftParams) || 1))
  if (draftParams !== normalizedDraft) draftModelParams.value = normalizedDraft

  if (ngl != null && Number.isFinite(Number(selectedModel?.layers))) {
    const normalizedNgl = Math.min(
      Math.max(0, Math.round(Number(ngl))),
      Math.max(0, Math.round(Number(selectedModel.layers))),
    )
    if (ngl !== normalizedNgl) nglCount.value = normalizedNgl
  }
}, { immediate: true })

// ── 求解状态 ──────────────────────────────────────────
const solving = ref(false)
const progress = ref(0)
const progressTotal = ref(0)
const results = ref([])
const hasRun = ref(false)
const wasCancelled = ref(false)
const validationError = ref(null)
const visibleNonParetoCount = ref(50)
const activeRunToken = ref(null)

const modelContextMax = computed(() => {
  const maxCtx = Number(modelA.value?.max_ctx)
  return Number.isFinite(maxCtx) && maxCtx > 0
    ? Math.min(LIMITS.ctx.max, maxCtx)
    : LIMITS.ctx.max
})

const validationMessage = computed(() => {
  const error = validationError.value
  if (!error) return ''

  const isZh = locale.value === 'zh'
  if (error.code === 'model_local_inference_unsupported') {
    return t('solver.api_only_model_validation', { model: error.modelName || error.modelId || '—' })
  }
  if (error.code === 'context_exceeds_model_limit') {
    return isZh
      ? `上下文长度 ${error.ctx.toLocaleString()} 超过了 ${error.modelName || '该模型'} 的上限 ${error.maxCtx.toLocaleString()}。`
      : `Context length ${error.ctx.toLocaleString()} exceeds the ${error.modelName || 'model'} limit of ${error.maxCtx.toLocaleString()}.`
  }
  if (error.code === 'workload_exceeds_context') {
    return isZh
      ? `Prompt 与输出共 ${error.sequenceLength.toLocaleString()} tokens，超过了 ${error.ctx.toLocaleString()} 的上下文长度。`
      : `Prompt plus output is ${error.sequenceLength.toLocaleString()} tokens, exceeding the ${error.ctx.toLocaleString()}-token context.`
  }
  if (error.code === 'apple_multi_device_unsupported') {
    return isZh
      ? 'Apple Silicon 统一内存配置只能按一台设备计算。'
      : 'Apple Silicon unified memory configurations can only be calculated as one device.'
  }
  if (error.code === 'invalid_workload') {
    return isZh
      ? `${error.field} 必须是大于零的有效数字。`
      : `${error.field} must be a valid number greater than zero.`
  }

  return isZh ? '输入参数无效，请检查后重试。' : 'The input configuration is invalid. Please review it and try again.'
})

// ── 结果排序 ──────────────────────────────────────────
const sortBy = ref('speed') // speed | vram | gpuCount | speedPerGpu

const sortedResults = computed(() => {
  const list = [...results.value]
  switch (sortBy.value) {
    case 'vram':        return list.sort((a, b) => a.vramNeeded - b.vramNeeded)
    case 'gpuCount':    return list.sort((a, b) => (a.totalGpuCount ?? a.gpuCount) - (b.totalGpuCount ?? b.gpuCount))
    case 'speedPerGpu': return list.sort((a, b) => (b.decodeSpeed / (b.totalGpuCount ?? b.gpuCount)) - (a.decodeSpeed / (a.totalGpuCount ?? a.gpuCount)))
    default:            return list.sort((a, b) => b.decodeSpeed - a.decodeSpeed)
  }
})

const paretoResults  = computed(() => sortedResults.value.filter(r => r.isPareto))
const nonParetoResults = computed(() => sortedResults.value.filter(r => !r.isPareto))
const visibleNonParetoResults = computed(() => nonParetoResults.value.slice(0, visibleNonParetoCount.value))

// ── 求解入口 ──────────────────────────────────────────
async function runWithTimeout(solver, params, runToken, timeoutMs) {
  const attemptToken = { cancelled: false }
  let timeoutId

  try {
    return await Promise.race([
      solver({
        ...params,
        shouldCancel: () => runToken.cancelled || attemptToken.cancelled,
      }),
      new Promise((_, reject) => {
        timeoutId = setTimeout(() => {
          attemptToken.cancelled = true
          reject(new Error('solver-timeout'))
        }, timeoutMs)
      }),
    ])
  } finally {
    if (timeoutId != null) clearTimeout(timeoutId)
  }
}

async function runSolver() {
  if (solving.value) return
  const normalizedTargetSpeed = clampInt(targetSpeed.value, LIMITS.targetSpeed)
  const normalizedMaxGpuCount = clampInt(maxGpuCount.value, LIMITS.maxGpuCount)
  if (targetSpeed.value !== normalizedTargetSpeed) targetSpeed.value = normalizedTargetSpeed
  if (maxGpuCount.value !== normalizedMaxGpuCount) maxGpuCount.value = normalizedMaxGpuCount
  const runToken = { cancelled: false }
  const HARD_TIMEOUT_MS = 4000

  activeRunToken.value = runToken
  solving.value = true
  progress.value = 0
  progressTotal.value = 0
  results.value = []
  hasRun.value = true
  wasCancelled.value = false
  validationError.value = null
  visibleNonParetoCount.value = 50

  try {
    const commonParams = {
      ctx: clampInt(ctx.value, LIMITS.ctx),
      batch: clampInt(batch.value, LIMITS.batch),
      promptLen: clampInt(promptLen.value, LIMITS.promptLen),
      outputLen: clampInt(outputLen.value, LIMITS.outputLen),
      flashAttention: flashAttention.value,
      kvCacheQuant: kvCacheQuant.value,
      prefixCacheHit: prefixCacheHit.value,
      cpuOffload: cpuOffload.value,
      pcieBw: pcieBw.value,
      pcieWidth: pcieWidth.value,
      pureCpu: pureCpu.value,
      cpuMemBw: cpuMemBw.value,
      cpuTflops: cpuTflops.value,
      gpuMemoryUtilization: gpuMemoryUtilization.value,
      sysRam: sysRam.value,
      speculativeDecoding: speculativeDecoding.value,
      acceptanceRate: acceptanceRate.value,
      draftLen: draftLen.value,
      draftModelParams: draftModelParams.value,
      ppCount: ppCount.value,
      epCount: epCount.value,
      imageCount: imageCount.value,
      nglCount: nglCount.value,
      onProgress: (done, total) => {
        progress.value = done
        progressTotal.value = total
      },
      shouldCancel: () => runToken.cancelled,
    }

    let outcome = null

    if (isUpgradeMode.value && currentGpu.value && currentQuant.value) {
      const upgradeParams = {
        currentGpu: currentGpu.value,
        currentGpuCount: currentGpuCount.value,
        gpuSlots: currentGpuSlots.value,
        currentQuant: currentQuant.value,
        currentFramework: currentFramework.value,
        currentInterconnect: currentInterconnect.value,
        model: modelA.value,
        targetSpeed: normalizedTargetSpeed,
      }
      const fullUpgradeParams = { ...upgradeParams, ...commonParams }

      try {
        outcome = await runWithTimeout(solveUpgrade, fullUpgradeParams, runToken, HARD_TIMEOUT_MS)
      } catch {
        outcome = await solveUpgrade(fullUpgradeParams)
      }

      // 兜底：若首轮意外返回空结果，进行一次无进度回调的重算
      if (!outcome?.cancelled && !outcome?.validationError && (outcome?.results?.length ?? 0) === 0) {
        const retry = await solveUpgrade(fullUpgradeParams)
        if (!retry?.cancelled && (retry?.results?.length ?? 0) > 0) {
          outcome = retry
        }
      }
    } else {
      const solverParams = {
        model: modelA.value,
        maxGpuCount: normalizedMaxGpuCount,
        vendorFilter: vendorFilter.value,
        excludeDatacenterGpu: excludeDatacenterGpu.value,
        quantFloor: quantFloor.value,
        minDecodeSpeed: clampOptionalPositive(minDecodeSpeedA.value, LIMITS.minDecodeSpeed) || null,
        maxTtft: clampOptionalPositive(maxTtft.value, LIMITS.maxTtft) || null,
        disableYield: true,
      }
      const fullSolverParams = { ...solverParams, ...commonParams }

      try {
        outcome = await runWithTimeout(solveForModel, fullSolverParams, runToken, HARD_TIMEOUT_MS)
      } catch {
        outcome = await solveForModel(fullSolverParams)
      }

      // 兜底：若首轮意外返回空结果，进行一次无进度回调的重算
      if (!outcome?.cancelled && !outcome?.validationError && (outcome?.results?.length ?? 0) === 0) {
        const retry = await solveForModel(fullSolverParams)
        if (!retry?.cancelled && (retry?.results?.length ?? 0) > 0) {
          outcome = retry
        }
      }
    }

    if (activeRunToken.value !== runToken) {
      if ((outcome?.results?.length ?? 0) > 0 && results.value.length === 0) {
        results.value = outcome.results
      }
      return
    }
    validationError.value = outcome?.validationError ?? null
    if (!outcome || outcome.cancelled) {
      wasCancelled.value = true
      return
    }
    results.value = outcome.results
  } catch (err) {
    console.error('[solver] runSolver failed', err)
    if (activeRunToken.value === runToken) {
      results.value = []
      wasCancelled.value = false
      validationError.value = null
    }
  } finally {
    if (activeRunToken.value === runToken) {
      activeRunToken.value = null
    }
    solving.value = false
  }
}

function cancelSolver() {
  if (activeRunToken.value) activeRunToken.value.cancelled = true
}

function showMoreResults() {
  visibleNonParetoCount.value += 50
}

// ── 跳转到主计算器 ────────────────────────────────────
function useConfig(row) {
  const rowSlots = row.gpuSlots?.length
    ? row.gpuSlots
    : [{ gpu: row.gpu, count: row.gpuCount }]
  const offloadMode = row.runtime
    ? cpuOffloadMode.value
    : 'off'
  const usesCpuMemory = cpuOffload.value
    || pureCpu.value
    || rowSlots.some(slot => slot.gpu?.sharedMemory)
  const sharedVram = rowSlots.find(slot =>
    slot.gpu?.sharedMemory
    && slot.gpu?.sharedMemoryConfigured
    && Number(slot.gpu?.catalogVram ?? 0) <= 0
  )?.gpu?.sharedVram
  const cpuMemBaseId = cpuMemBw.value?.generation && cpuMemBw.value?.transferRate
    ? `${cpuMemBw.value.generation}_${cpuMemBw.value.transferRate}`
    : cpuMemBw.value?.id
  const query = addCustomModelQuery({
    gpus: rowSlots.map(slot => `${slot.gpu.id}:${slot.count}`).join(','),
    ic: row.interconnect?.id,
    model: modelA.value.id,
    quant: row.quant.id,
    fw: row.framework.id,
    fa: flashAttention.value ? undefined : '0',
    kv: kvCacheQuant.value?.id !== 'auto' ? kvCacheQuant.value?.id : undefined,
    pc: prefixCacheHit.value > 0 ? prefixCacheHit.value : undefined,
    co: offloadMode === 'auto' ? undefined : offloadMode,
    pcie: cpuOffload.value ? pcieBw.value?.id : undefined,
    pw: cpuOffload.value ? pcieWidth.value?.id : undefined,
    pcpu: pureCpu.value ? '1' : undefined,
    cmb: usesCpuMemory ? cpuMemBaseId : undefined,
    cmc: usesCpuMemory && cpuMemBw.value?.channels !== 2
      ? cpuMemBw.value.channels
      : undefined,
    cmm: usesCpuMemory && cpuMemBw.value?.measuredBw != null
      ? cpuMemBw.value.measuredBw
      : undefined,
    ctf: cpuTflops.value != null ? cpuTflops.value : undefined,
    gmu: gpuMemoryUtilization.value != null
      ? gpuMemoryUtilization.value
      : undefined,
    sr: usesCpuMemory && sysRam.value !== 64 ? sysRam.value : undefined,
    sv: sharedVram != null ? Math.max(1, Math.floor(sharedVram)) : undefined,
    sd: speculativeDecoding.value ? '1' : undefined,
    ar: speculativeDecoding.value && acceptanceRate.value !== 0.7 ? acceptanceRate.value : undefined,
    dl: speculativeDecoding.value && draftLen.value !== 4 ? draftLen.value : undefined,
    dmp: speculativeDecoding.value && draftModelParams.value !== 1 ? draftModelParams.value : undefined,
    pp: row.ppCount > 1 ? row.ppCount : undefined,
    ep: row.epCount > 1 ? row.epCount : undefined,
    img: imageCount.value > 0 ? imageCount.value : undefined,
    ngl: nglCount.value != null ? nglCount.value : undefined,
    ctx: ctx.value,
    b: batch.value,
    pl: promptLen.value,
    ol: outputLen.value,
  }, modelA.value)
  router.push({
    path: '/',
    query,
  })
}

function progressPct() {
  if (!progressTotal.value) return 0
  return Math.round((progress.value / progressTotal.value) * 100)
}

// ── 常量 ──────────────────────────────────────────────
const GPU_COUNT_OPTIONS = [1, 2, 4, 8]
const VENDOR_OPTIONS = [
  { id: 'all',      label_zh: '全部',   label_en: 'All'      },
  { id: 'nvidia',   label_zh: 'NVIDIA', label_en: 'NVIDIA'   },
  { id: 'amd',      label_zh: 'AMD',    label_en: 'AMD'      },
  { id: 'apple',    label_zh: 'Apple',  label_en: 'Apple'    },
  { id: 'intel',    label_zh: 'Intel',  label_en: 'Intel'    },
  { id: 'domestic', label_zh: '国产',   label_en: 'Domestic' },
]
const SORT_OPTIONS = computed(() => [
  { id: 'speed',       label: t('solver.sort_speed')       },
  { id: 'vram',        label: t('solver.sort_vram')        },
  { id: 'gpuCount',    label: t('solver.sort_gpu_count')   },
  { id: 'speedPerGpu', label: t('solver.sort_speed_per_gpu') },
])

watch(
  [modelA, maxGpuCount, vendorFilter, excludeDatacenterGpu, quantFloor, minDecodeSpeedA, maxTtft,
   ctx, batch, promptLen, outputLen, isUpgradeMode, currentGpuSlots, currentQuant, targetSpeed,
   currentFramework, currentInterconnect, flashAttention, kvCacheQuant, prefixCacheHit,
   cpuOffloadMode, pcieBw, pcieWidth, pureCpu, cpuMemBw, cpuTflops, gpuMemoryUtilization,
   sysRam, speculativeDecoding,
   acceptanceRate, draftLen, draftModelParams, ppCount, epCount, imageCount, nglCount],
  () => {
    const model = modelA.value
    const maxg = maxGpuCount.value
    const vendor = vendorFilter.value
    const excl_dc = excludeDatacenterGpu.value
    const qf = quantFloor.value
    const minds = minDecodeSpeedA.value
    const mttft = maxTtft.value
    const ctxValue = ctx.value
    const batchValue = batch.value
    const promptValue = promptLen.value
    const outputValue = outputLen.value
    const upgradeMode = isUpgradeMode.value
    const curQuant = currentQuant.value
    const tgtSpeed = targetSpeed.value
    validationError.value = null
    const normalizedMinds = clampOptionalPositive(minds, LIMITS.minDecodeSpeed)
    const normalizedMttft = clampOptionalPositive(mttft, LIMITS.maxTtft)
    const normalizedCtx = clampInt(ctxValue, LIMITS.ctx)
    const normalizedBatch = clampInt(batchValue, LIMITS.batch)
    const normalizedPrompt = clampInt(promptValue, LIMITS.promptLen)
    const normalizedOutput = clampInt(outputValue, LIMITS.outputLen)
    const normalizedMaxg = clampInt(maxg, LIMITS.maxGpuCount)
    const normalizedTarget = clampInt(tgtSpeed, LIMITS.targetSpeed)

    if (minDecodeSpeedA.value !== normalizedMinds) minDecodeSpeedA.value = normalizedMinds
    if (maxTtft.value !== normalizedMttft) maxTtft.value = normalizedMttft
    if (ctx.value !== normalizedCtx) ctx.value = normalizedCtx
    if (batch.value !== normalizedBatch) batch.value = normalizedBatch
    if (promptLen.value !== normalizedPrompt) promptLen.value = normalizedPrompt
    if (outputLen.value !== normalizedOutput) outputLen.value = normalizedOutput
    if (maxGpuCount.value !== normalizedMaxg) maxGpuCount.value = normalizedMaxg
    if (targetSpeed.value !== normalizedTarget) targetSpeed.value = normalizedTarget

    const query = {}
    if (model?.id) query.model = model.id
    addCustomModelQuery(query, model)
    if (upgradeMode) {
      query.upgrade = '1'
      if (currentGpuSlots.value.length) {
        query.gpus = currentGpuSlots.value
          .map(slot => `${slot.gpu.id}:${slot.count}`)
          .join(',')
      }
      if (curQuant?.id) query.quant = curQuant.id
      if (normalizedTarget !== LIMITS.targetSpeed.def) query.target = String(normalizedTarget)
      if (currentFramework.value?.id) query.fw = currentFramework.value.id
      if (currentInterconnect.value?.id) query.ic = currentInterconnect.value.id
      if (!flashAttention.value) query.fa = '0'
      if (kvCacheQuant.value?.id && kvCacheQuant.value.id !== 'auto') query.kv = kvCacheQuant.value.id
      if (prefixCacheHit.value > 0) query.pc = String(prefixCacheHit.value)
      if (cpuOffloadMode.value !== 'auto') query.co = cpuOffloadMode.value
      if (cpuOffload.value) {
        if (pcieBw.value?.id) query.pcie = pcieBw.value.id
        if (pcieWidth.value?.id) query.pw = pcieWidth.value.id
      }
      if (pureCpu.value) query.pcpu = '1'
      const usesCpuMemory = cpuOffload.value
        || pureCpu.value
        || currentGpuSlots.value.some(slot => slot.gpu?.sharedMemory)
      if (usesCpuMemory) {
        if (cpuMemBw.value?.generation && cpuMemBw.value?.transferRate) {
          query.cmb = `${cpuMemBw.value.generation}_${cpuMemBw.value.transferRate}`
        } else if (cpuMemBw.value?.id) {
          query.cmb = cpuMemBw.value.id
        }
        if (cpuMemBw.value?.channels !== 2) query.cmc = String(cpuMemBw.value.channels)
        if (cpuMemBw.value?.measuredBw != null) query.cmm = String(cpuMemBw.value.measuredBw)
        if (sysRam.value !== 64) query.sr = String(sysRam.value)
      }
      if (cpuTflops.value != null) query.ctf = String(cpuTflops.value)
      if (gpuMemoryUtilization.value != null) {
        query.gmu = String(gpuMemoryUtilization.value)
      }
      if (speculativeDecoding.value) {
        query.sd = '1'
        if (acceptanceRate.value !== 0.7) query.ar = String(acceptanceRate.value)
        if (draftLen.value !== 4) query.dl = String(draftLen.value)
        if (draftModelParams.value !== 1) query.dmp = String(draftModelParams.value)
      }
      if (ppCount.value !== 1) query.pp = String(ppCount.value)
      if (epCount.value !== 1) query.ep = String(epCount.value)
      if (imageCount.value !== 0) query.img = String(imageCount.value)
      if (nglCount.value != null) query.ngl = String(nglCount.value)
    } else {
      if (normalizedMaxg !== LIMITS.maxGpuCount.def) query.maxg = String(normalizedMaxg)
      if (vendor !== 'all') query.vendor = vendor
      if (!excl_dc) query.excl_dc = '0'
      if (qf !== 'none') query.qf = qf
      if (normalizedMinds !== '') query.minds = String(normalizedMinds)
      if (normalizedMttft !== '') query.mttft = String(normalizedMttft)
    }
    if (normalizedCtx !== LIMITS.ctx.def) query.ctx = String(normalizedCtx)
    if (normalizedBatch !== LIMITS.batch.def) query.b = String(normalizedBatch)
    if (normalizedPrompt !== LIMITS.promptLen.def) query.pl = String(normalizedPrompt)
    if (normalizedOutput !== LIMITS.outputLen.def) query.ol = String(normalizedOutput)
    router.replace({ query })
  },
  { immediate: true }
)

onMounted(() => {
  if (hasInitialQuery) runSolver()
})
</script>

<template>
  <div class="min-h-screen bg-gray-50">
    <Header />

    <main class="pt-14 sm:pt-16 max-w-7xl mx-auto px-3 sm:px-4 pb-16">
      <!-- 页面标题 -->
      <div class="py-6 sm:py-8">
        <div class="flex items-center gap-3 mb-2">
          <h1 class="text-2xl sm:text-3xl font-bold text-gray-900">{{ t('solver.title') }}</h1>
          <span v-if="isUpgradeMode" class="px-3 py-1 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-700">
            {{ t('solver.upgrade_mode') }}
          </span>
        </div>
        <p v-if="!isUpgradeMode" class="mt-1 text-sm text-gray-500">{{ t('solver.subtitle') }}</p>
        <p v-else class="mt-1 text-sm text-gray-600">
          {{ t('solver.upgrade_current_config', {
            gpu: currentGpu?.name || '—',
            count: currentGpuCount,
            quant: currentQuant?.label || '—',
            target: targetSpeed
          }) }}
        </p>
        <RouterLink to="/ranking" class="inline-flex mt-3 text-sm font-medium text-emerald-700 hover:text-emerald-800">
          {{ t('solver.reverse_link') }}
        </RouterLink>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-[380px_1fr] gap-6">
        <!-- ── 左侧：输入面板 ── -->
        <div class="space-y-4">

          <div class="bg-white rounded-xl border border-gray-200 p-4">
            <h2 class="text-sm font-semibold text-gray-700 mb-3">{{ t('model.title') }}</h2>
            <ModelPicker v-model:model="modelA" />
          </div>

          <div class="bg-white rounded-xl border border-gray-200 p-4 space-y-4">
            <h2 class="text-sm font-semibold text-gray-700">{{ t('solver.constraints') }}</h2>

            <!-- 最大 GPU 数量 -->
            <div>
              <label class="block text-xs text-gray-500 mb-1.5">{{ t('solver.max_gpu_count') }}</label>
              <div class="flex gap-2">
                <button
                  v-for="n in GPU_COUNT_OPTIONS" :key="n"
                  @click="maxGpuCount = n"
                  class="flex-1 py-1.5 rounded-md text-sm font-medium border transition-colors"
                  :class="maxGpuCount === n
                    ? 'bg-emerald-600 text-white border-emerald-600'
                    : 'bg-white text-gray-600 border-gray-300 hover:bg-gray-50'"
                >{{ n }}</button>
              </div>
            </div>

            <!-- GPU 厂商过滤 -->
            <div>
              <label class="block text-xs text-gray-500 mb-1.5">{{ t('solver.vendor_filter') }}</label>
              <div class="flex flex-wrap gap-1.5">
                <button
                  v-for="v in VENDOR_OPTIONS" :key="v.id"
                  @click="vendorFilter = v.id"
                  class="px-2.5 py-1 rounded-md text-xs font-medium border transition-colors"
                  :class="vendorFilter === v.id
                    ? 'bg-emerald-600 text-white border-emerald-600'
                    : 'bg-white text-gray-600 border-gray-300 hover:bg-gray-50'"
                >{{ locale === 'zh' ? v.label_zh : v.label_en }}</button>
              </div>
            </div>

            <!-- GPU 层级过滤 -->
            <div>
              <label class="block text-xs text-gray-500 mb-1.5">{{ locale === 'zh' ? '显卡类型' : 'GPU Type' }}</label>
              <div class="flex gap-1.5">
                <button
                  @click="excludeDatacenterGpu = false"
                  class="flex-1 px-2.5 py-1 rounded-md text-xs font-medium border transition-colors"
                  :class="!excludeDatacenterGpu
                    ? 'bg-emerald-600 text-white border-emerald-600'
                    : 'bg-white text-gray-600 border-gray-300 hover:bg-gray-50'"
                >{{ locale === 'zh' ? '全部' : 'All' }}</button>
                <button
                  @click="excludeDatacenterGpu = true"
                  class="flex-1 px-2.5 py-1 rounded-md text-xs font-medium border transition-colors"
                  :class="excludeDatacenterGpu
                    ? 'bg-emerald-600 text-white border-emerald-600'
                    : 'bg-white text-gray-600 border-gray-300 hover:bg-gray-50'"
                >{{ locale === 'zh' ? '消费级' : 'Consumer' }}</button>
              </div>
              <p class="text-xs text-gray-400 mt-1.5">{{ locale === 'zh' ? '消费级：排除数据中心卡(H100/A100等)' : 'Consumer: Exclude datacenter GPUs (H100/A100/etc)' }}</p>
            </div>

            <!-- 量化质量下限 -->
            <div>
              <label class="block text-xs text-gray-500 mb-1.5">{{ t('solver.quant_floor') }}</label>
              <div class="flex flex-wrap gap-1.5">
                <button
                  v-for="q in QUANT_FLOOR_OPTIONS" :key="q.id"
                  @click="quantFloor = q.id"
                  class="px-2.5 py-1 rounded-md text-xs font-medium border transition-colors"
                  :class="quantFloor === q.id
                    ? 'bg-emerald-600 text-white border-emerald-600'
                    : 'bg-white text-gray-600 border-gray-300 hover:bg-gray-50'"
                >{{ locale === 'zh' ? q.label_zh : q.label_en }}</button>
              </div>
            </div>

            <!-- 最低 Decode 速度 -->
            <div>
              <label class="block text-xs text-gray-500 mb-1.5">
                {{ t('solver.min_decode_speed') }}
                <span class="text-gray-400 font-normal">{{ t('solver.optional') }}</span>
              </label>
              <div class="relative">
                <input v-model.number="minDecodeSpeedA" type="number" min="0" :max="LIMITS.minDecodeSpeed.max" placeholder="—"
                  class="w-full px-3 py-1.5 pr-14 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent" />
                <span class="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-gray-400">tok/s</span>
              </div>
            </div>

            <!-- TTFT 上限 -->
            <div>
              <label class="block text-xs text-gray-500 mb-1.5">
                {{ t('solver.max_ttft') }}
                <span class="text-gray-400 font-normal">{{ t('solver.optional') }}</span>
              </label>
              <div class="relative">
                <input v-model.number="maxTtft" type="number" min="0" :max="LIMITS.maxTtft.max" placeholder="—"
                  class="w-full px-3 py-1.5 pr-10 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent" />
                <span class="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-gray-400">ms</span>
              </div>
            </div>
          </div>

          <!-- 共用推理参数 -->
          <div class="bg-white rounded-xl border border-gray-200 p-4 space-y-3">
            <h2 class="text-sm font-semibold text-gray-700">{{ t('solver.inference_params') }}</h2>
            <div class="grid grid-cols-2 gap-3">
              <div>
                <label class="block text-xs text-gray-500 mb-1">{{ t('run.ctx') }}</label>
                <input v-model.number="ctx" type="number" :min="LIMITS.ctx.min" :max="modelContextMax"
                  class="w-full px-2.5 py-1.5 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-emerald-500" />
              </div>
              <div>
                <label class="block text-xs text-gray-500 mb-1">{{ t('run.batch') }}</label>
                <input v-model.number="batch" type="number" :min="LIMITS.batch.min" :max="LIMITS.batch.max"
                  class="w-full px-2.5 py-1.5 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-emerald-500" />
              </div>
              <div>
                <label class="block text-xs text-gray-500 mb-1">{{ t('run.prompt') }}</label>
                <input v-model.number="promptLen" type="number" :min="LIMITS.promptLen.min" :max="LIMITS.promptLen.max"
                  class="w-full px-2.5 py-1.5 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-emerald-500" />
              </div>
              <div>
                <label class="block text-xs text-gray-500 mb-1">{{ t('run.output') }}</label>
                <input v-model.number="outputLen" type="number" :min="LIMITS.outputLen.min" :max="LIMITS.outputLen.max"
                  class="w-full px-2.5 py-1.5 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-emerald-500" />
              </div>
            </div>
          </div>

          <!-- 求解按钮 -->
          <button
            v-if="!solving"
            @click="runSolver"
            class="w-full py-3 rounded-xl text-sm font-semibold transition-colors bg-emerald-600 hover:bg-emerald-700 text-white shadow-sm"
          >
            <span>{{ t('solver.run_btn') }}</span>
          </button>
          <button
            v-else
            @click="cancelSolver"
            class="w-full py-3 rounded-xl text-sm font-semibold transition-colors bg-rose-50 text-rose-700 border border-rose-200 hover:bg-rose-100"
          >
            <span class="flex items-center justify-center gap-2">
              <svg class="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
              </svg>
              {{ t('solver.running') }} {{ progressTotal ? `${progressPct()}%` : '' }} · {{ t('solver.cancel_btn') }}
            </span>
          </button>
        </div>

        <!-- ── 右侧：结果面板 ── -->
        <div>
          <!-- 进度条 -->
          <div v-if="solving" class="mb-4">
            <div class="flex justify-between text-xs text-gray-500 mb-1">
              <span>{{ t('solver.progress_label') }}</span>
              <span>{{ progress }} / {{ progressTotal }}</span>
            </div>
            <div class="h-2 bg-gray-200 rounded-full overflow-hidden">
              <div
                class="h-full bg-emerald-500 rounded-full transition-all duration-300"
                :style="{ width: progressPct() + '%' }"
              />
            </div>
          </div>

          <!-- 空状态 -->
          <div v-if="!hasRun && !solving" class="flex flex-col items-center justify-center h-64 text-gray-400">
            <svg viewBox="0 0 64 64" class="w-16 h-16 mb-3 opacity-30" fill="none">
              <circle cx="32" cy="32" r="28" stroke="currentColor" stroke-width="3"/>
              <path d="M20 32h24M32 20v24" stroke="currentColor" stroke-width="3" stroke-linecap="round"/>
            </svg>
            <p class="text-sm">{{ t('solver.empty_hint') }}</p>
          </div>

          <!-- 无结果 -->
          <div v-else-if="hasRun && !solving && results.length === 0"
            class="flex flex-col items-center justify-center h-64 text-gray-400">
            <svg viewBox="0 0 64 64" class="w-16 h-16 mb-3 opacity-30" fill="none">
              <circle cx="32" cy="32" r="28" stroke="currentColor" stroke-width="3"/>
              <path d="M22 22l20 20M42 22L22 42" stroke="currentColor" stroke-width="3" stroke-linecap="round"/>
            </svg>
            <p class="text-sm" :class="{ 'text-amber-700': validationMessage }">
              {{ validationMessage || (wasCancelled ? t('solver.cancelled') : t('solver.no_results')) }}
            </p>
          </div>

          <!-- 结果列表 -->
          <template v-else-if="results.length > 0">
            <!-- 统计 + 排序 -->
            <div class="flex flex-wrap items-center justify-between gap-3 mb-4">
              <div class="text-sm text-gray-600">
                {{ t('solver.result_count', { total: results.length, pareto: paretoResults.length }) }}
              </div>
              <div class="flex flex-wrap items-center gap-2">
                <span class="text-xs text-gray-500">{{ t('solver.sort_by') }}</span>
                <div class="flex flex-wrap gap-1">
                  <button
                    v-for="opt in SORT_OPTIONS" :key="opt.id"
                    @click="sortBy = opt.id"
                    class="px-2 py-1 rounded text-xs font-medium border transition-colors"
                    :class="sortBy === opt.id
                      ? 'bg-emerald-600 text-white border-emerald-600'
                      : 'bg-white text-gray-600 border-gray-300 hover:bg-gray-50'"
                  >{{ opt.label }}</button>
                </div>
              </div>
            </div>

            <!-- Pareto 前沿 -->
            <div v-if="paretoResults.length > 0" class="mb-2">
              <div class="flex items-center gap-2 mb-2">
                <span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-amber-100 text-amber-700 border border-amber-200">
                  <svg viewBox="0 0 16 16" class="w-3 h-3" fill="currentColor" aria-hidden="true">
                    <path d="M8 1l1.9 3.8 4.2.6-3 2.9.7 4.2L8 10.4l-3.8 2 .7-4.2-3-2.9 4.2-.6z"/>
                  </svg>
                  {{ t('solver.pareto_label') }}
                </span>
                <span class="text-xs text-gray-400">{{ t('solver.pareto_hint') }}</span>
              </div>
              <div class="space-y-2">
                <div
                  v-for="(row, i) in paretoResults" :key="'p' + i"
                  class="bg-white rounded-xl border border-amber-200 p-3 sm:p-4 hover:border-amber-300 transition-colors"
                >
                  <ResultRowContent :row="row" :is-pareto="true" @use="useConfig(row)" />
                </div>
              </div>
            </div>

            <!-- 其余方案 -->
            <div v-if="nonParetoResults.length > 0">
              <div class="text-xs text-gray-400 mb-2 mt-4">{{ t('solver.other_results') }}</div>
              <div class="space-y-2">
                <div
                  v-for="(row, i) in visibleNonParetoResults" :key="'n' + i"
                  class="bg-white rounded-xl border border-gray-200 p-3 sm:p-4 hover:border-gray-300 transition-colors"
                >
                  <ResultRowContent :row="row" :is-pareto="false" @use="useConfig(row)" />
                </div>
              </div>
              <div v-if="nonParetoResults.length > visibleNonParetoCount" class="flex flex-col items-center gap-2 mt-3">
                <p class="text-xs text-gray-400 text-center">
                  {{ t('solver.truncated', { n: nonParetoResults.length - visibleNonParetoCount }) }}
                </p>
                <button
                  @click="showMoreResults"
                  class="px-3 py-1.5 rounded-lg text-xs font-medium border border-gray-300 bg-white text-gray-700 hover:bg-gray-50"
                >
                  {{ t('solver.show_more') }}
                </button>
              </div>
            </div>
          </template>
        </div>
      </div>
    </main>
  </div>
</template>
