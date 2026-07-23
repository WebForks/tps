<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter, useRoute } from 'vue-router'
import TopBar from '../components/layout/TopBar.vue'
import GpuConfig from '../components/config/GpuConfig.vue'
import { GPU_LIST } from '../data/gpus/index.js'
import { ALL_MODELS } from '../data/models/index.js'
import { QUANT_MAP, FRAMEWORK_MAP, INTERCONNECT_MAP } from '../data/constants.js'
import { calcAll, aggregateGpuSlots } from '../utils/calc.js'
import { fmtParams, fmtGB, fmtToks, fmtMs, fmtCtx, isNew } from '../utils/format.js'
import {
  PCIE_BW_OPTIONS,
  PCIE_WIDTH_OPTIONS,
  CPU_MEM_CHANNEL_OPTIONS,
  CPU_MEM_GENERATIONS,
  CPU_MEM_TRANSFER_RATE_PRESETS,
  RAM_CAPACITY_OPTIONS,
  createCpuMemBwOption,
  normalizeCpuMemMeasuredBandwidth,
  normalizeRamCapacity,
  resolveCpuMemBwOption,
} from '../data/runtime.js'
import {
  getDefaultGpuMemoryUtilization,
  normalizeGpuMemoryUtilization,
} from '../utils/runtime.js'

const { t } = useI18n()
const router = useRouter()
const route = useRoute()

// ── URL 解析 ────────────────────────────────────────
const _p = route.query
const WORKLOAD_LIMITS = Object.freeze({
  ctx: { min: 512, max: 10_485_760, fallback: 16_384 },
  batch: { min: 1, max: 256, fallback: 1 },
  promptLen: { min: 1, max: 10_485_760, fallback: 1_024 },
  outputLen: { min: 1, max: 10_485_760, fallback: 1_024 },
})

function boundedInt(value, fallback, min, max) {
  const number = Number(value)
  return Number.isFinite(number)
    ? Math.min(max, Math.max(min, Math.round(number)))
    : fallback
}

function parseGpuSlots(query) {
  if (query.gpus) {
    let remaining = 512
    const parsed = []
    for (const s of String(query.gpus).split(',')) {
      if (remaining <= 0) break
      const [id, count] = s.split(':')
      const gpu = GPU_LIST.find(g => g.id === id && g.unitKind !== 'cpu') ?? null
      if (!gpu) continue
      const normalizedCount = boundedInt(count, 1, 1, remaining)
      parsed.push({ gpu, count: normalizedCount })
      remaining -= normalizedCount
    }
    if (parsed.length) return parsed
  }
  const gpu = GPU_LIST.find(g => g.id === query.gpu && g.unitKind !== 'cpu')
  if (gpu) return [{ gpu, count: boundedInt(query.n, 1, 1, 512) }]
  return null
}

// ── GPU 配置 ─────────────────────────────────────────
const gpuSlots    = ref(parseGpuSlots(_p) ?? [{ gpu: GPU_LIST.find(g => g.id === 'h100_sxm') ?? GPU_LIST[0], count: 1 }])
const interconnect = ref(
  INTERCONNECT_MAP.find(i => i.id === _p.ic)
  ?? INTERCONNECT_MAP.find(i => i.id === 'pcie4')
  ?? INTERCONNECT_MAP[0]
)
const ctx         = ref(boundedInt(_p.ctx, WORKLOAD_LIMITS.ctx.fallback, WORKLOAD_LIMITS.ctx.min, WORKLOAD_LIMITS.ctx.max))
const batch       = ref(boundedInt(_p.b, WORKLOAD_LIMITS.batch.fallback, WORKLOAD_LIMITS.batch.min, WORKLOAD_LIMITS.batch.max))
const promptLen   = ref(boundedInt(_p.pl, WORKLOAD_LIMITS.promptLen.fallback, WORKLOAD_LIMITS.promptLen.min, WORKLOAD_LIMITS.promptLen.max))
const outputLen   = ref(boundedInt(_p.ol, WORKLOAD_LIMITS.outputLen.fallback, WORKLOAD_LIMITS.outputLen.min, WORKLOAD_LIMITS.outputLen.max))
const framework   = ref(FRAMEWORK_MAP.find(f => f.id === _p.fw) ?? FRAMEWORK_MAP.find(f => f.id === 'theory'))
const gpuCount    = computed(() => gpuSlots.value.reduce((s, g) => s + g.count, 0))
const sharedVram  = ref(boundedInt(_p.sv, 16, 1, 512))

const pcieBw = ref(PCIE_BW_OPTIONS.find(option => option.id === _p.pcie) ?? PCIE_BW_OPTIONS[1])
const pcieWidth = ref(PCIE_WIDTH_OPTIONS.find(option => option.id === _p.pw) ?? PCIE_WIDTH_OPTIONS[1])
const initialCpuMemBw = resolveCpuMemBwOption(_p.cmb) ?? resolveCpuMemBwOption('ddr5_4800')
const cpuMemBw = ref(createCpuMemBwOption(
  initialCpuMemBw.generation,
  initialCpuMemBw.transferRate,
  boundedInt(_p.cmc, initialCpuMemBw.channels ?? 2, 1, 16),
  normalizeCpuMemMeasuredBandwidth(_p.cmm, initialCpuMemBw.measuredBw),
))
const sysRam = ref(normalizeRamCapacity(_p.sr, 64))
const cpuMemGeneration = computed(() => CPU_MEM_GENERATIONS.find(
  generation => generation.id === cpuMemBw.value?.generation
) ?? CPU_MEM_GENERATIONS[2])
const cpuMemRatePresets = computed(() => (
  CPU_MEM_TRANSFER_RATE_PRESETS[cpuMemGeneration.value.id] ?? []
))

function setCpuMemGeneration(generation) {
  cpuMemBw.value = createCpuMemBwOption(
    generation.id,
    generation.defaultTransferRate,
    cpuMemBw.value?.channels ?? 2,
  )
}

function setCpuMemTransferRate(value) {
  const next = createCpuMemBwOption(
    cpuMemGeneration.value.id,
    value,
    cpuMemBw.value?.channels ?? 2,
  )
  if (next) cpuMemBw.value = next
}

function setCpuMemChannels(channels) {
  const next = createCpuMemBwOption(
    cpuMemGeneration.value.id,
    cpuMemBw.value?.transferRate,
    channels,
  )
  if (next) cpuMemBw.value = next
}

function setCpuMemMeasuredBandwidth(value) {
  const next = createCpuMemBwOption(
    cpuMemGeneration.value.id,
    cpuMemBw.value?.transferRate,
    cpuMemBw.value?.channels ?? 2,
    normalizeCpuMemMeasuredBandwidth(value),
  )
  if (next) cpuMemBw.value = next
}

function setSystemRam(value) {
  sysRam.value = normalizeRamCapacity(value, sysRam.value ?? 64)
}

function normalizeWorkloadInteger(target, event) {
  const limits = WORKLOAD_LIMITS[target]
  const source = {
    ctx,
    batch,
    promptLen,
    outputLen,
  }[target]
  const normalized = boundedInt(source.value, limits.fallback, limits.min, limits.max)
  if (target === 'ctx') ctx.value = normalized
  else if (target === 'batch') batch.value = normalized
  else if (target === 'promptLen') promptLen.value = normalized
  else if (target === 'outputLen') outputLen.value = normalized
  if (event?.target) event.target.value = String(normalized)
}

function setFramework(id) {
  const selected = FRAMEWORK_MAP.find(option => option.id === id)
  if (selected) framework.value = selected
}

const workloadTokens = computed(() => promptLen.value + outputLen.value)
const workloadFitsContext = computed(() => workloadTokens.value <= ctx.value)

const effectiveGpu = computed(() => {
  const slots = gpuSlots.value.map(s => {
    let g = s.gpu
    if (g?.sharedMemory && g?.vram === 0) g = { ...g, vram: sharedVram.value }
    return { ...s, gpu: g }
  })
  return slots.length === 1 ? slots[0].gpu : aggregateGpuSlots(slots)
})
const usesUnifiedMemory = computed(() => Boolean(effectiveGpu.value?.unifiedMemory))
const usesConventionalSharedMemory = computed(() => (
  Boolean(effectiveGpu.value?.sharedMemory)
  && !usesUnifiedMemory.value
))
const gpuMemoryUtilization = computed(() => normalizeGpuMemoryUtilization(
  null,
  getDefaultGpuMemoryUtilization(framework.value, effectiveGpu.value),
))
const calculationGpu = computed(() => effectiveGpu.value
  ? { ...effectiveGpu.value, usableRatio: gpuMemoryUtilization.value }
  : null
)

// ── 排序 & 基础筛选 ──────────────────────────────────
const sortBy          = ref(['speed','prefill','vram','params','vram_free','efficiency'].includes(_p.sort) ? _p.sort : 'speed')
const filterType      = ref(['all','dense','moe'].includes(_p.type) ? _p.type : 'all')
const showOnlyRunnable = ref(_p.runnable === '0' ? false : true)

// ── 新增筛选条件 ─────────────────────────────────────
// 参数量范围：all / le7 / 7to30 / 30to100 / gt100
const PARAM_RANGES = [
  { id: 'all',    label: () => t('ranking.filter_params_all') },
  { id: 'le7',    label: () => '≤ 7B'   },
  { id: '7to30',  label: () => '7–30B'  },
  { id: '30to100',label: () => '30–100B'},
  { id: 'gt100',  label: () => '> 100B' },
]
const filterParams = ref(_p.params ?? 'all')

// 最低速度：0 = 不限
const SPEED_THRESHOLDS = [
  { id: '0',   label: () => t('ranking.filter_min_speed_all'), value: 0   },
  { id: '10',  label: () => '≥ 10 tok/s',  value: 10  },
  { id: '20',  label: () => '≥ 20 tok/s',  value: 20  },
  { id: '50',  label: () => '≥ 50 tok/s',  value: 50  },
  { id: '100', label: () => '≥ 100 tok/s', value: 100 },
]
const filterMinSpeed = ref(_p.minspeed ?? '0')

// 最低量化精度：'' = 不限，否则为 quant.id
const QUANT_FLOORS = [
  { id: '',     label: () => t('ranking.filter_min_quant_all') },
  { id: 'int4', label: () => 'INT4+'  },
  { id: 'int8', label: () => 'INT8+'  },
  { id: 'fp8',  label: () => 'FP8+'   },
  { id: 'bf16', label: () => 'BF16'   },
]
const filterMinQuant = ref(_p.minquant ?? '')

// 隐藏需要 CPU 卸载的模型
const hideOffload = ref(_p.hideoffload === '1')

// 显示 legacy 模型（默认隐藏）
const filterLegacy = ref(_p.legacy === '1')

// 是否有激活的筛选（用于显示"已筛选"标记）
const hasActiveFilters = computed(() =>
  filterParams.value !== 'all' ||
  filterMinSpeed.value !== '0' ||
  filterMinQuant.value !== '' ||
  hideOffload.value ||
  filterLegacy.value
)

function resetFilters() {
  filterParams.value = 'all'
  filterMinSpeed.value = '0'
  filterMinQuant.value = ''
  hideOffload.value = false
  filterLegacy.value = false
}

// ── URL 同步 ─────────────────────────────────────────
watch(
  [gpuSlots, interconnect, ctx, batch, promptLen, outputLen, framework, sortBy, filterType, showOnlyRunnable, sharedVram,
   pcieBw, pcieWidth, cpuMemBw, sysRam,
   filterParams, filterMinSpeed, filterMinQuant, hideOffload, filterLegacy],
  ([slots, ic, c, b, pl, ol, fw, sort, type, runnable, sv, pb, pw, cmb, sr, params, minspeed, minquant, offload, legacy]) => {
    const query = {}
    const hasUnifiedMemory = slots?.some(slot => slot.gpu?.unifiedMemory)
    const hasSharedMemory = slots?.some(slot => slot.gpu?.sharedMemory)
    const usesCpuMemorySettings = !hasUnifiedMemory
    const usesPcieOffloadSettings = !hasUnifiedMemory && !hasSharedMemory
    if (slots?.length) query.gpus = slots.map(s => `${s.gpu.id}:${s.count}`).join(',')
    if (ic?.id) query.ic = ic.id
    if (c !== WORKLOAD_LIMITS.ctx.fallback) query.ctx = c
    if (b !== WORKLOAD_LIMITS.batch.fallback) query.b = b
    if (pl !== WORKLOAD_LIMITS.promptLen.fallback) query.pl = pl
    if (ol !== WORKLOAD_LIMITS.outputLen.fallback) query.ol = ol
    if (fw?.id && fw.id !== 'theory') query.fw = fw.id
    if (sort !== 'speed') query.sort = sort
    if (type !== 'all') query.type = type
    if (!runnable) query.runnable = '0'
    if (hasSharedMemory && !hasUnifiedMemory && sv !== 16) query.sv = sv
    if (usesPcieOffloadSettings && pb?.id && pb.id !== 'gen4') query.pcie = pb.id
    if (usesPcieOffloadSettings && pw?.id && pw.id !== 'x8') query.pw = pw.id
    const cpuMemBaseId = cmb?.generation && cmb?.transferRate
      ? `${cmb.generation}_${cmb.transferRate}`
      : cmb?.id
    if (
      usesCpuMemorySettings
      &&
      cpuMemBaseId
      && (
        cpuMemBaseId !== 'ddr5_4800'
        || cmb?.channels !== 2
        || cmb?.measuredBw != null
      )
    ) query.cmb = cpuMemBaseId
    if (usesCpuMemorySettings && cmb?.channels != null && cmb.channels !== 2) query.cmc = cmb.channels
    if (usesCpuMemorySettings && cmb?.measuredBw != null) query.cmm = cmb.measuredBw
    if (usesCpuMemorySettings && sr !== 64) query.sr = sr
    if (params !== 'all') query.params = params
    if (minspeed !== '0') query.minspeed = minspeed
    if (minquant !== '') query.minquant = minquant
    if (offload) query.hideoffload = '1'
    if (legacy) query.legacy = '1'
    router.replace({ query })
  },
  { immediate: true },
)

// ── 计算所有模型结果（requestIdleCallback 分批，避免 UI 卡顿）────────
const allModelResults = ref([])
const calcProgress = ref(0)
const calcTotal = ref(0)
const CALC_BATCH_SIZE = 8
let _calcVersion = 0
let _pendingCalcHandle = null
let _pendingCalcUsesIdleCallback = false

function _cancelScheduledCalc() {
  if (_pendingCalcHandle == null) return
  if (
    _pendingCalcUsesIdleCallback
    && typeof cancelIdleCallback !== 'undefined'
  ) {
    cancelIdleCallback(_pendingCalcHandle)
  } else {
    clearTimeout(_pendingCalcHandle)
  }
  _pendingCalcHandle = null
}

function _scheduleCalcBatch(callback) {
  if (typeof requestIdleCallback !== 'undefined') {
    _pendingCalcUsesIdleCallback = true
    _pendingCalcHandle = requestIdleCallback(deadline => {
      _pendingCalcHandle = null
      callback(deadline)
    }, { timeout: 250 })
  } else {
    _pendingCalcUsesIdleCallback = false
    _pendingCalcHandle = setTimeout(() => {
      _pendingCalcHandle = null
      callback(null)
    }, 0)
  }
}

function _calculateModel(model, config) {
  let bestQuant = null
  let bestResult = null

  for (const quant of QUANT_MAP) {
    try {
      const commonArgs = {
        ...config.args,
        model,
        quant,
      }
      let result = calcAll({ ...commonArgs, cpuOffload: false, pcieBw: null })

      if (
        config.allowCpuOffload
        && !result.fitOk
        && !result.vramOk
        && model.type === 'moe'
        && model.active_params
      ) {
        const offloadResult = calcAll({
          ...commonArgs,
          cpuOffload: true,
          pcieBw: config.offloadPcieBw,
        })
        if (offloadResult.fitOk) result = offloadResult
      }

      if (result.fitOk && (!bestResult || quant.bytes > bestQuant.bytes)) {
        bestQuant = quant
        bestResult = result
      }
    } catch {
      // Unsupported catalog/runtime combinations must not abort the ranking.
    }
  }

  return {
    model,
    quant: bestQuant,
    result: bestResult,
    canRun: Boolean(bestResult),
    cpuOffload: bestResult?.cpuOffload ?? false,
  }
}

function _compareModelResults(a, b) {
  if (sortBy.value === 'params') {
    return b.model.params - a.model.params
      || a.model.name.localeCompare(b.model.name)
  }
  if (!a.result || !b.result) {
    if (Boolean(a.result) !== Boolean(b.result)) return a.result ? -1 : 1
    return a.model.name.localeCompare(b.model.name)
  }

  let difference = 0
  if (sortBy.value === 'speed') {
    difference = b.result.singleToks - a.result.singleToks
  } else if (sortBy.value === 'prefill') {
    difference = b.result.prefillToks - a.result.prefillToks
  } else if (sortBy.value === 'vram') {
    difference = a.result.totalNeeded - b.result.totalNeeded
  } else if (sortBy.value === 'vram_free') {
    difference = (b.result.totalVram - b.result.totalNeeded)
      - (a.result.totalVram - a.result.totalNeeded)
  } else if (sortBy.value === 'efficiency') {
    difference = (b.result.tokPerJoule ?? 0) - (a.result.tokPerJoule ?? 0)
  }
  return difference || a.model.name.localeCompare(b.model.name)
}

function _runCalcBatch(models, start, version, config, results, deadline) {
  if (version !== _calcVersion) return

  const hardEnd = Math.min(start + CALC_BATCH_SIZE, models.length)
  let index = start
  while (index < hardEnd) {
    // Always calculate at least one model, then yield when the idle budget is
    // nearly exhausted.
    if (
      index > start
      && deadline
      && !deadline.didTimeout
      && deadline.timeRemaining() < 2
    ) {
      break
    }
    results.push(_calculateModel(models[index], config))
    index += 1
  }

  if (version !== _calcVersion) return
  calcProgress.value = index
  allModelResults.value = results.slice()

  if (index < models.length) {
    _scheduleCalcBatch(nextDeadline => {
      _runCalcBatch(models, index, version, config, results, nextDeadline)
    })
  }
}

function _startCalculation() {
  _cancelScheduledCalc()
  const version = ++_calcVersion
  allModelResults.value = []
  calcProgress.value = 0

  if (!calculationGpu.value || !framework.value) {
    calcTotal.value = 0
    return
  }

  const models = ALL_MODELS.filter(model =>
    model.localInference !== false
    && (filterType.value === 'all' || model.type === filterType.value)
  )
  calcTotal.value = models.length
  const config = {
    args: {
      gpu: calculationGpu.value,
      gpuCount: gpuCount.value,
      interconnect: interconnect.value,
      ctx: ctx.value,
      batch: batch.value,
      promptLen: promptLen.value,
      outputLen: outputLen.value,
      framework: framework.value,
      flashAttention: true,
      kvCacheQuant: null,
      prefixCacheHit: 0,
      speculativeDecoding: false,
      acceptanceRate: 0.7,
      draftLen: 4,
      pureCpu: false,
      cpuMemBw: cpuMemBw.value,
      sysRam: sysRam.value,
      pcieWidth: pcieWidth.value,
    },
    offloadPcieBw: pcieBw.value,
    allowCpuOffload: !(
      calculationGpu.value.unifiedMemory
      || calculationGpu.value.sharedMemory
      || calculationGpu.value.invalidMemoryMix
    ),
  }
  const results = []
  _scheduleCalcBatch(deadline => {
    _runCalcBatch(models, 0, version, config, results, deadline)
  })
}

// Only calcAll inputs restart the idle job. Sorting and post-filters apply to
// partial/completed rows immediately without repeating the expensive work.
watch(
  [calculationGpu, gpuCount, interconnect, ctx, batch, promptLen, outputLen, framework,
   filterType, pcieBw, pcieWidth, cpuMemBw, sysRam],
  _startCalculation,
  { immediate: true }
)

onUnmounted(() => {
  _calcVersion += 1
  _cancelScheduledCalc()
})

// ── 后置筛选与排序 ────────────────────────────────────
const modelResults = computed(() => {
  let list = allModelResults.value.filter(Boolean)

  if (showOnlyRunnable.value) {
    list = list.filter(item => item.canRun)
  }

  // 过滤 legacy 模型（默认隐藏）
  if (!filterLegacy.value) {
    list = list.filter(item => item.model.status !== 'legacy')
  }

  // 参数量范围
  if (filterParams.value !== 'all') {
    list = list.filter(item => {
      const p = item.model.params
      if (filterParams.value === 'le7')     return p <= 7
      if (filterParams.value === '7to30')   return p > 7 && p <= 30
      if (filterParams.value === '30to100') return p > 30 && p <= 100
      if (filterParams.value === 'gt100')   return p > 100
      return true
    })
  }

  // 最低速度
  const speedThreshold = SPEED_THRESHOLDS.find(s => s.id === filterMinSpeed.value)?.value ?? 0
  if (speedThreshold > 0) {
    list = list.filter(item => item.result && item.result.singleToks >= speedThreshold)
  }

  // 最低量化精度
  if (filterMinQuant.value) {
    const floorQuant = QUANT_MAP.find(q => q.id === filterMinQuant.value)
    if (floorQuant) {
      list = list.filter(item => item.quant && item.quant.bytes >= floorQuant.bytes)
    }
  }

  // 隐藏 CPU 卸载
  if (hideOffload.value) {
    list = list.filter(item => !item.cpuOffload)
  }

  list.sort(_compareModelResults)
  return list
})

// ── 虚拟化：分页渲染（每次加载 50 条，滚动到底部追加）──
const PAGE_SIZE = 50
const visibleCount = ref(PAGE_SIZE)
const sentinelRef = ref(null)
let observer = null

// 筛选条件变化时重置可见数量
watch(modelResults, () => { visibleCount.value = PAGE_SIZE })

const visibleResults = computed(() => modelResults.value.slice(0, visibleCount.value))
const hasMore = computed(() => visibleCount.value < modelResults.value.length)

function loadMore() {
  visibleCount.value = Math.min(visibleCount.value + PAGE_SIZE, modelResults.value.length)
}

onMounted(() => {
  observer = new IntersectionObserver(
    (entries) => { if (entries[0].isIntersecting && hasMore.value) loadMore() },
    { rootMargin: '200px' }
  )
  if (sentinelRef.value) observer.observe(sentinelRef.value)
})

onUnmounted(() => { observer?.disconnect() })

// ── 跳转 ─────────────────────────────────────────────
function useThisModel(modelData) {
  const unifiedMemory = Boolean(effectiveGpu.value?.unifiedMemory)
  const usesCpuMemory = !unifiedMemory && Boolean(
    modelData.cpuOffload
    || effectiveGpu.value?.sharedMemory
  )
  const cpuMemBaseId = cpuMemBw.value?.generation && cpuMemBw.value?.transferRate
    ? `${cpuMemBw.value.generation}_${cpuMemBw.value.transferRate}`
    : cpuMemBw.value?.id
  const query = {
    gpus:  gpuSlots.value.map(s => `${s.gpu.id}:${s.count}`).join(','),
    ic:    interconnect.value?.id ?? undefined,
    model: modelData.model.id,
    quant: modelData.quant?.id ?? 'bf16',
    fw:    framework.value.id,
    ctx:   ctx.value,
    b:     batch.value !== 1 ? batch.value : undefined,
    pl:    promptLen.value,
    ol:    outputLen.value,
    co:    !unifiedMemory && modelData.cpuOffload ? 'on' : undefined,
    pcie:  !unifiedMemory && modelData.cpuOffload ? pcieBw.value.id : undefined,
    pw:    !unifiedMemory && modelData.cpuOffload ? pcieWidth.value.id : undefined,
    cmb:   usesCpuMemory ? cpuMemBaseId : undefined,
    cmc:   usesCpuMemory && cpuMemBw.value.channels !== 2 ? cpuMemBw.value.channels : undefined,
    cmm:   usesCpuMemory && cpuMemBw.value.measuredBw != null ? cpuMemBw.value.measuredBw : undefined,
    gmu:   gpuMemoryUtilization.value,
    sr:    usesCpuMemory && sysRam.value !== 64 ? sysRam.value : undefined,
    sv:    effectiveGpu.value?.sharedMemory && sharedVram.value !== 16 ? sharedVram.value : undefined,
  }
  router.push({ path: '/', query })
}
</script>

<template>
  <div class="min-h-screen bg-gray-50 overflow-x-hidden pt-12 sm:pt-14 pb-20 sm:pb-8">
    <TopBar />

    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      <!-- 标题 -->
      <div class="bg-white rounded-xl border border-gray-200 p-6">
        <h1 class="text-2xl font-bold text-gray-900 mb-2">{{ t('ranking.title') }}</h1>
        <p class="text-sm text-gray-600">{{ t('ranking.subtitle') }}</p>
      </div>

      <!-- GPU 配置 -->
      <div class="bg-white rounded-xl border border-gray-200 p-4">
        <GpuConfig v-model:gpuSlots="gpuSlots" v-model:interconnect="interconnect" v-model:sharedVram="sharedVram" />
      </div>

      <!-- Every result uses this explicit workload. -->
      <section class="rounded-xl border border-gray-200 bg-white p-4 space-y-4">
        <div>
          <h2 class="text-sm font-semibold uppercase tracking-wider text-gray-700">
            {{ t('ranking.workload') }}
          </h2>
          <p class="mt-1 text-xs leading-relaxed text-gray-500">{{ t('ranking.workload_tip') }}</p>
        </div>

        <div class="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
          <label class="block">
            <span class="mb-1.5 block text-xs font-medium text-gray-500">{{ t('run.ctx') }}</span>
            <input
              v-model.number="ctx"
              @change="normalizeWorkloadInteger('ctx', $event)"
              type="number"
              :min="WORKLOAD_LIMITS.ctx.min"
              :max="WORKLOAD_LIMITS.ctx.max"
              step="1"
              class="w-full rounded-lg border border-gray-200 bg-gray-50 px-3 py-2 text-sm text-gray-800 focus:outline-none focus:ring-1 focus:ring-emerald-500"
            />
          </label>
          <label class="block">
            <span class="mb-1.5 block text-xs font-medium text-gray-500">{{ t('run.batch') }}</span>
            <input
              v-model.number="batch"
              @change="normalizeWorkloadInteger('batch', $event)"
              type="number"
              :min="WORKLOAD_LIMITS.batch.min"
              :max="WORKLOAD_LIMITS.batch.max"
              step="1"
              class="w-full rounded-lg border border-gray-200 bg-gray-50 px-3 py-2 text-sm text-gray-800 focus:outline-none focus:ring-1 focus:ring-emerald-500"
            />
          </label>
          <label class="block">
            <span class="mb-1.5 block text-xs font-medium text-gray-500">{{ t('run.prompt') }}</span>
            <input
              v-model.number="promptLen"
              @change="normalizeWorkloadInteger('promptLen', $event)"
              type="number"
              :min="WORKLOAD_LIMITS.promptLen.min"
              :max="WORKLOAD_LIMITS.promptLen.max"
              step="1"
              class="w-full rounded-lg border border-gray-200 bg-gray-50 px-3 py-2 text-sm text-gray-800 focus:outline-none focus:ring-1 focus:ring-emerald-500"
            />
          </label>
          <label class="block">
            <span class="mb-1.5 block text-xs font-medium text-gray-500">{{ t('run.output') }}</span>
            <input
              v-model.number="outputLen"
              @change="normalizeWorkloadInteger('outputLen', $event)"
              type="number"
              :min="WORKLOAD_LIMITS.outputLen.min"
              :max="WORKLOAD_LIMITS.outputLen.max"
              step="1"
              class="w-full rounded-lg border border-gray-200 bg-gray-50 px-3 py-2 text-sm text-gray-800 focus:outline-none focus:ring-1 focus:ring-emerald-500"
            />
          </label>
          <label class="block sm:col-span-2 lg:col-span-1">
            <span class="mb-1.5 block text-xs font-medium text-gray-500">{{ t('run.framework') }}</span>
            <select
              :value="framework.id"
              @change="setFramework($event.target.value)"
              class="w-full rounded-lg border border-gray-200 bg-gray-50 px-3 py-2 text-sm text-gray-800 focus:outline-none focus:ring-1 focus:ring-emerald-500"
            >
              <option v-for="option in FRAMEWORK_MAP" :key="option.id" :value="option.id">
                {{ t(`framework.${option.id}`) }}
              </option>
            </select>
          </label>
        </div>

        <p
          class="rounded-lg border px-3 py-2 text-xs"
          :class="workloadFitsContext
            ? 'border-emerald-100 bg-emerald-50 text-emerald-700'
            : 'border-red-200 bg-red-50 text-red-700'"
        >
          {{ t('ranking.workload_total', {
            total: workloadTokens.toLocaleString(),
            context: ctx.toLocaleString(),
          }) }}
          <span v-if="!workloadFitsContext" class="font-semibold">
            {{ t('ranking.workload_invalid') }}
          </span>
        </p>
      </section>

      <!-- Unified-memory devices already carry their own capacity and bandwidth. -->
      <section
        v-if="!usesUnifiedMemory"
        class="rounded-xl border border-gray-200 bg-white p-4 space-y-4"
      >
        <div>
          <h2 class="text-sm font-semibold uppercase tracking-wider text-gray-700">
            {{ t(usesConventionalSharedMemory
              ? 'ranking.shared_memory_assumptions'
              : 'ranking.offload_assumptions') }}
          </h2>
          <p class="mt-1 text-xs leading-relaxed text-gray-500">
            {{ t(usesConventionalSharedMemory
              ? 'ranking.shared_memory_assumptions_tip'
              : 'ranking.offload_assumptions_tip') }}
          </p>
        </div>

        <div class="grid gap-4 lg:grid-cols-2">
          <div class="space-y-3 rounded-lg bg-gray-50 p-3">
            <div>
              <label class="mb-1.5 block text-xs font-medium text-gray-500">{{ t('run.ram_generation') }}</label>
              <div class="flex flex-wrap gap-1.5">
                <button
                  v-for="generation in CPU_MEM_GENERATIONS"
                  :key="generation.id"
                  @click="setCpuMemGeneration(generation)"
                  class="rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors"
                  :class="cpuMemGeneration.id === generation.id
                    ? 'border-emerald-600 bg-emerald-600 text-white'
                    : 'border-gray-200 bg-white text-gray-600 hover:bg-gray-100'"
                >{{ generation.label }}</button>
              </div>
            </div>

            <div>
              <label class="mb-1.5 block text-xs font-medium text-gray-500">{{ t('run.ram_transfer_rate') }}</label>
              <div class="flex flex-wrap gap-1.5">
                <button
                  v-for="rate in cpuMemRatePresets"
                  :key="rate"
                  @click="setCpuMemTransferRate(rate)"
                  class="rounded-lg border px-2 py-1 text-xs transition-colors"
                  :class="cpuMemBw.transferRate === rate
                    ? 'border-emerald-600 bg-emerald-600 text-white'
                    : 'border-gray-200 bg-white text-gray-600 hover:bg-gray-100'"
                >{{ rate }}</button>
                <input
                  :value="cpuMemBw.transferRate"
                  @change="setCpuMemTransferRate($event.target.value)"
                  type="number"
                  :min="cpuMemGeneration.minTransferRate"
                  :max="cpuMemGeneration.maxTransferRate"
                  step="1"
                  class="w-24 rounded-lg border border-gray-200 bg-white px-2 py-1 text-center text-xs text-gray-700 focus:outline-none focus:ring-1 focus:ring-emerald-500"
                  :aria-label="t('run.ram_transfer_rate_custom')"
                />
                <span class="self-center text-xs text-gray-400">MT/s</span>
              </div>
            </div>

            <div>
              <label class="mb-1.5 block text-xs font-medium text-gray-500">{{ t('run.ram_channels') }}</label>
              <div class="flex flex-wrap gap-1.5">
                <button
                  v-for="channels in CPU_MEM_CHANNEL_OPTIONS"
                  :key="channels"
                  @click="setCpuMemChannels(channels)"
                  class="rounded-lg border px-2.5 py-1 text-xs transition-colors"
                  :class="cpuMemBw.channels === channels
                    ? 'border-emerald-600 bg-emerald-600 text-white'
                    : 'border-gray-200 bg-white text-gray-600 hover:bg-gray-100'"
                >{{ channels }}</button>
              </div>
              <p class="mt-1.5 text-xs text-emerald-700">
                {{ t('run.ram_theoretical_bw_channels', { channels: cpuMemBw.channels, bw: cpuMemBw.theoreticalBw.toFixed(1) }) }}
              </p>
            </div>

            <div>
              <label class="mb-1.5 block text-xs font-medium text-gray-500">{{ t('run.ram_measured_bw') }}</label>
              <div class="flex items-center gap-2">
                <input
                  :value="cpuMemBw.measuredBw ?? ''"
                  @change="setCpuMemMeasuredBandwidth($event.target.value)"
                  type="number"
                  min="0.1"
                  max="10000"
                  step="0.1"
                  :placeholder="t('run.ram_measured_bw_placeholder')"
                  class="w-40 rounded-lg border border-gray-200 bg-white px-2 py-1.5 text-xs text-gray-700 focus:outline-none focus:ring-1 focus:ring-emerald-500"
                />
                <span class="text-xs text-gray-400">GB/s</span>
              </div>
            </div>
          </div>

          <div class="space-y-3 rounded-lg bg-gray-50 p-3">
            <div>
              <label class="mb-1.5 block text-xs font-medium text-gray-500">{{ t('run.sys_ram') }}</label>
              <div class="flex flex-wrap gap-1.5">
                <button
                  v-for="capacity in RAM_CAPACITY_OPTIONS"
                  :key="capacity"
                  @click="setSystemRam(capacity)"
                  class="rounded-lg border px-2 py-1 text-xs transition-colors"
                  :class="sysRam === capacity
                    ? 'border-emerald-600 bg-emerald-600 text-white'
                    : 'border-gray-200 bg-white text-gray-600 hover:bg-gray-100'"
                >{{ capacity }} GB</button>
                <input
                  :value="sysRam"
                  @change="setSystemRam($event.target.value)"
                  type="number"
                  min="8"
                  max="4096"
                  step="1"
                  class="w-24 rounded-lg border border-gray-200 bg-white px-2 py-1 text-center text-xs text-gray-700 focus:outline-none focus:ring-1 focus:ring-emerald-500"
                  :aria-label="t('run.sys_ram_custom')"
                />
              </div>
            </div>

            <div v-if="!usesConventionalSharedMemory">
              <label class="mb-1.5 block text-xs font-medium text-gray-500">{{ t('run.pcie_bw') }}</label>
              <div class="flex flex-wrap gap-1.5">
                <button
                  v-for="option in PCIE_BW_OPTIONS"
                  :key="option.id"
                  @click="pcieBw = option"
                  class="rounded-lg border px-2.5 py-1 text-xs transition-colors"
                  :class="pcieBw.id === option.id
                    ? 'border-emerald-600 bg-emerald-600 text-white'
                    : 'border-gray-200 bg-white text-gray-600 hover:bg-gray-100'"
                >{{ option.label }}</button>
              </div>
            </div>

            <div v-if="!usesConventionalSharedMemory">
              <label class="mb-1.5 block text-xs font-medium text-gray-500">{{ t('run.pcie_width') }}</label>
              <div class="flex flex-wrap gap-1.5">
                <button
                  v-for="option in PCIE_WIDTH_OPTIONS"
                  :key="option.id"
                  @click="pcieWidth = option"
                  class="rounded-lg border px-3 py-1 text-xs transition-colors"
                  :class="pcieWidth.id === option.id
                    ? 'border-emerald-600 bg-emerald-600 text-white'
                    : 'border-gray-200 bg-white text-gray-600 hover:bg-gray-100'"
                >{{ option.label }}</button>
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- 筛选和排序 -->
      <div class="bg-white rounded-xl border border-gray-200 p-4 space-y-3">
        <!-- 第一行：模型类型 + 排序 + 仅可运行 + 重置 -->
        <div class="flex flex-wrap gap-3 items-center">
          <div class="flex items-center gap-2">
            <label class="text-xs font-medium text-gray-500 whitespace-nowrap">{{ t('ranking.filter_type') }}</label>
            <div class="flex gap-1">
              <button
                v-for="opt in [
                  { id: 'all',   label: t('ranking.filter_all') },
                  { id: 'dense', label: t('ranking.filter_dense') },
                  { id: 'moe',   label: t('ranking.filter_moe') },
                ]"
                :key="opt.id"
                @click="filterType = opt.id"
                :class="['px-2.5 py-1 rounded-lg text-xs font-medium border transition-colors',
                  filterType === opt.id
                    ? 'bg-emerald-600 border-emerald-600 text-white shadow-sm'
                    : 'bg-gray-50 border-gray-200 text-gray-600 hover:bg-gray-100']"
              >{{ opt.label }}</button>
            </div>
          </div>

          <div class="flex items-center gap-2">
            <label class="text-xs font-medium text-gray-500 whitespace-nowrap">{{ t('ranking.sort_by') }}</label>
            <select v-model="sortBy" class="text-xs border border-gray-300 rounded-lg px-2.5 py-1.5 bg-gray-50 focus:outline-none focus:ring-1 focus:ring-emerald-500">
              <option value="speed">{{ t('ranking.sort_speed') }}</option>
              <option value="prefill">{{ t('ranking.sort_prefill') }}</option>
              <option value="vram">{{ t('ranking.sort_vram') }}</option>
              <option value="vram_free">{{ t('ranking.sort_vram_free') }}</option>
              <option value="efficiency">{{ t('ranking.sort_efficiency') }}</option>
              <option value="params">{{ t('ranking.sort_params') }}</option>
            </select>
          </div>

          <label class="flex items-center gap-1.5 cursor-pointer select-none">
            <input type="checkbox" v-model="showOnlyRunnable" class="w-3.5 h-3.5 accent-emerald-500 rounded" />
            <span class="text-xs text-gray-600">{{ t('ranking.show_only_runnable') }}</span>
          </label>

          <label class="flex items-center gap-1.5 cursor-pointer select-none">
            <input type="checkbox" v-model="hideOffload" class="w-3.5 h-3.5 accent-emerald-500 rounded" />
            <span class="text-xs text-gray-600">{{ t('ranking.filter_hide_offload') }}</span>
          </label>

          <label class="flex items-center gap-1.5 cursor-pointer select-none">
            <input type="checkbox" v-model="filterLegacy" class="w-3.5 h-3.5 accent-emerald-500 rounded" />
            <span class="text-xs text-gray-600">{{ t('ranking.filter_show_legacy') }}</span>
          </label>

          <div class="ml-auto flex items-center gap-2">
            <button
              v-if="hasActiveFilters"
              @click="resetFilters"
              class="text-xs px-2.5 py-1 rounded-lg border border-gray-300 text-gray-500 hover:bg-gray-100 transition-colors"
            >{{ t('ranking.filter_reset') }}</button>
            <span class="text-xs text-gray-500">
              <span v-if="hasActiveFilters" class="text-emerald-600 font-medium">{{ t('ranking.active_filters') }} · </span>
              {{ t('ranking.total_models', { count: modelResults.length }) }}
            </span>
          </div>
        </div>

        <!-- 第二行：参数量 + 最低速度 + 最低量化 -->
        <div class="flex flex-wrap gap-3 items-center pt-2 border-t border-gray-100">
          <!-- 参数量范围 -->
          <div class="flex items-center gap-2">
            <label class="text-xs font-medium text-gray-500 whitespace-nowrap">{{ t('ranking.filter_params') }}</label>
            <div class="flex gap-1 flex-wrap">
              <button
                v-for="opt in PARAM_RANGES"
                :key="opt.id"
                @click="filterParams = opt.id"
                :class="['px-2.5 py-1 rounded-lg text-xs font-medium border transition-colors',
                  filterParams === opt.id
                    ? 'bg-emerald-600 border-emerald-600 text-white shadow-sm'
                    : 'bg-gray-50 border-gray-200 text-gray-600 hover:bg-gray-100']"
              >{{ opt.label() }}</button>
            </div>
          </div>

          <!-- 最低速度 -->
          <div class="flex items-center gap-2">
            <label class="text-xs font-medium text-gray-500 whitespace-nowrap">{{ t('ranking.filter_min_speed') }}</label>
            <div class="flex gap-1 flex-wrap">
              <button
                v-for="opt in SPEED_THRESHOLDS"
                :key="opt.id"
                @click="filterMinSpeed = opt.id"
                :class="['px-2.5 py-1 rounded-lg text-xs font-medium border transition-colors',
                  filterMinSpeed === opt.id
                    ? 'bg-emerald-600 border-emerald-600 text-white shadow-sm'
                    : 'bg-gray-50 border-gray-200 text-gray-600 hover:bg-gray-100']"
              >{{ opt.label() }}</button>
            </div>
          </div>

          <!-- 最低量化精度 -->
          <div class="flex items-center gap-2">
            <label class="text-xs font-medium text-gray-500 whitespace-nowrap">{{ t('ranking.filter_min_quant') }}</label>
            <div class="flex gap-1 flex-wrap">
              <button
                v-for="opt in QUANT_FLOORS"
                :key="opt.id"
                @click="filterMinQuant = opt.id"
                :class="['px-2.5 py-1 rounded-lg text-xs font-medium border transition-colors',
                  filterMinQuant === opt.id
                    ? 'bg-emerald-600 border-emerald-600 text-white shadow-sm'
                    : 'bg-gray-50 border-gray-200 text-gray-600 hover:bg-gray-100']"
              >{{ opt.label() }}</button>
            </div>
          </div>
        </div>
      </div>

      <!-- 计算进度条（分批计算期间显示） -->
      <div v-if="calcTotal > 0 && calcProgress < calcTotal" class="bg-white rounded-xl border border-gray-200 px-4 py-2 flex items-center gap-3">
        <div class="flex-1 bg-gray-100 rounded-full h-1.5 overflow-hidden">
          <div
            class="bg-emerald-500 h-1.5 rounded-full transition-all duration-200"
            :style="{ width: (calcProgress / calcTotal * 100).toFixed(1) + '%' }"
          />
        </div>
        <span class="text-xs text-gray-400 whitespace-nowrap">{{ calcProgress }} / {{ calcTotal }}</span>
      </div>

      <!-- 模型列表 -->
      <!-- 桌面端：表格视图 -->
      <div class="hidden sm:block bg-white rounded-xl border border-gray-200 overflow-hidden">
        <div class="overflow-x-auto">
          <table class="w-full">
            <thead class="bg-gray-50 border-b border-gray-200">
              <tr>
                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">{{ t('ranking.table_model') }}</th>
                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">{{ t('ranking.table_type') }}</th>
                <th class="px-4 py-3 text-right text-xs font-semibold text-gray-700 uppercase tracking-wider">{{ t('ranking.table_params') }}</th>
                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">{{ t('ranking.table_best_quant') }}</th>
                <th class="px-4 py-3 text-right text-xs font-semibold text-gray-700 uppercase tracking-wider">{{ t('ranking.table_vram') }}</th>
                <th class="px-4 py-3 text-right text-xs font-semibold text-gray-700 uppercase tracking-wider">{{ t('ranking.table_vram_free') }}</th>
                <th class="px-4 py-3 text-right text-xs font-semibold text-gray-700 uppercase tracking-wider">{{ t('ranking.table_speed') }}</th>
                <th class="px-4 py-3 text-right text-xs font-semibold text-gray-700 uppercase tracking-wider">{{ t('ranking.table_efficiency') }}</th>
                <th class="px-4 py-3 text-center text-xs font-semibold text-gray-700 uppercase tracking-wider">{{ t('ranking.table_status') }}</th>
                <th class="px-4 py-3 text-center text-xs font-semibold text-gray-700 uppercase tracking-wider">{{ t('ranking.table_action') }}</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-gray-200">
              <tr
                v-for="item in visibleResults"
                :key="item.model.id"
                class="hover:bg-gray-50 transition-colors"
                :class="{ 'opacity-50': !item.canRun }"
              >
                <td class="px-4 py-3">
                  <div class="text-sm font-medium text-gray-900">{{ item.model.name }}</div>
                  <div v-if="item.cpuOffload" class="text-[10px] text-amber-600 mt-0.5">CPU offload</div>
                </td>
                <td class="px-4 py-3">
                  <span
                    :class="item.model.type === 'moe' ? 'bg-amber-100 text-amber-700' : 'bg-emerald-100 text-emerald-700'"
                    class="text-xs font-medium px-2 py-0.5 rounded-full"
                  >
                    {{ item.model.type === 'moe' ? 'MoE' : 'Dense' }}
                  </span>
                </td>
                <td class="px-4 py-3 text-right text-sm text-gray-900">{{ fmtParams(item.model.params) }}</td>
                <td class="px-4 py-3">
                  <span v-if="item.quant" class="text-sm text-gray-700">{{ item.quant.label }}</span>
                  <span v-else class="text-sm text-gray-400">—</span>
                </td>
                <td class="px-4 py-3 text-right">
                  <span v-if="item.result" class="text-sm text-gray-900">{{ fmtGB(item.result.totalNeeded) }}</span>
                  <span v-else class="text-sm text-gray-400">—</span>
                </td>
                <td class="px-4 py-3 text-right">
                  <span v-if="item.result" class="text-sm text-gray-700">{{ fmtGB(item.result.totalVram - item.result.totalNeeded) }}</span>
                  <span v-else class="text-sm text-gray-400">—</span>
                </td>
                <td class="px-4 py-3 text-right">
                  <span v-if="item.result" class="text-sm font-medium text-gray-900">{{ fmtToks(item.result.singleToks) }}</span>
                  <span v-else class="text-sm text-gray-400">—</span>
                </td>
                <td class="px-4 py-3 text-right">
                  <span v-if="item.result && item.result.tokPerJoule" class="text-sm text-gray-700" :title="`${item.result.tokPerJoule.toFixed(6)} tok/J`">
                    {{ (item.result.tokPerJoule * 1000).toFixed(1) }}
                  </span>
                  <span v-else class="text-sm text-gray-400">—</span>
                </td>
                <td class="px-4 py-3 text-center">
                  <span
                    v-if="item.canRun"
                    class="inline-flex items-center text-xs font-medium px-2 py-0.5 rounded-full bg-green-100 text-green-700"
                  >
                    ✓ {{ t('ranking.status_runnable') }}
                  </span>
                  <span
                    v-else
                    class="inline-flex items-center text-xs font-medium px-2 py-0.5 rounded-full bg-red-100 text-red-700"
                  >
                    ✗ {{ t('ranking.status_oom') }}
                  </span>
                </td>
                <td class="px-4 py-3 text-center">
                  <button
                    v-if="item.canRun"
                    @click="useThisModel(item)"
                    class="text-xs px-3 py-1 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg transition-colors"
                  >
                    {{ t('ranking.use_config') }}
                  </button>
                  <span v-else class="text-xs text-gray-400">—</span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <!-- 加载更多哨兵（桌面端） -->
        <div ref="sentinelRef" class="h-4" />
        <div v-if="hasMore" class="text-center py-3 text-xs text-gray-400">
          {{ t('ranking.loading_more', { shown: visibleResults.length, total: modelResults.length }) }}
        </div>
      </div>

      <!-- 移动端：卡片视图 -->
      <div class="sm:hidden space-y-3">
        <div
          v-for="item in visibleResults"
          :key="item.model.id"
          class="bg-white rounded-xl border border-gray-200 overflow-hidden shadow-sm"
          :class="{ 'opacity-60': !item.canRun }"
        >
          <!-- 顶部色条：可运行绿 / OOM 红 -->
          <div :class="item.canRun ? 'bg-emerald-500' : 'bg-red-400'" class="h-1 w-full" />

          <div class="p-4">
            <!-- 标题行 -->
            <div class="flex items-start justify-between gap-2 mb-2">
              <div class="flex-1 min-w-0">
                <div class="flex items-center gap-1.5 flex-wrap mb-1">
                  <span
                    :class="item.model.type === 'moe' ? 'bg-amber-100 text-amber-700' : 'bg-emerald-100 text-emerald-700'"
                    class="text-[10px] font-bold px-1.5 py-0.5 rounded flex-shrink-0"
                  >{{ item.model.type === 'moe' ? 'MoE' : 'Dense' }}</span>
                  <span v-if="isNew(item.model.released)" class="text-[10px] font-bold px-1.5 py-0.5 rounded bg-red-100 text-red-600 flex-shrink-0">NEW</span>
                  <span v-if="item.cpuOffload" class="text-[10px] font-medium px-1.5 py-0.5 rounded bg-amber-50 text-amber-600 border border-amber-200 flex-shrink-0">CPU offload</span>
                </div>
                <h3 class="text-base font-bold text-gray-900 leading-tight">{{ item.model.name }}</h3>
                <!-- 参数量 / 上下文 / 激活参数 -->
                <div class="flex items-center gap-2 mt-1 flex-wrap">
                  <span class="text-xs text-gray-500">{{ fmtParams(item.model.params) }}</span>
                  <span class="text-gray-300 text-xs">·</span>
                  <span class="text-xs text-gray-500">{{ fmtCtx(item.model.max_ctx) }} ctx</span>
                  <template v-if="item.model.type === 'moe' && item.model.active_params">
                    <span class="text-gray-300 text-xs">·</span>
                    <span class="text-xs text-amber-600">{{ fmtParams(item.model.active_params) }} active</span>
                  </template>
                </div>
              </div>
              <!-- 状态 badge -->
              <span
                :class="item.canRun ? 'bg-emerald-100 text-emerald-700' : 'bg-red-100 text-red-600'"
                class="flex-shrink-0 text-xs font-semibold px-2.5 py-1 rounded-full"
              >
                {{ item.canRun ? '✓ ' + t('ranking.status_runnable') : '✗ ' + t('ranking.status_oom') }}
              </span>
            </div>

            <!-- 速度大卡（可运行时显示）-->
            <div v-if="item.result" class="rounded-xl bg-gradient-to-r from-emerald-50 to-teal-50 border border-emerald-100 px-4 py-3 mb-3 flex items-center justify-between">
              <div>
                <div class="text-[10px] text-emerald-600 font-medium uppercase tracking-wide mb-0.5">{{ t('ranking.table_speed') }}</div>
                <div class="text-2xl font-bold text-emerald-700 leading-none">{{ fmtToks(item.result.singleToks) }}</div>
                <div class="text-[10px] text-emerald-500 mt-0.5">单请求</div>
              </div>
              <div class="text-right">
                <div class="text-[10px] text-gray-500 mb-0.5">TPOT</div>
                <div class="text-sm font-semibold text-gray-700">{{ fmtMs(item.result.tpot) }}</div>
                <div class="text-[10px] text-gray-400 mt-1">TTFT</div>
                <div class="text-sm font-semibold text-gray-700">{{ fmtMs(item.result.ttft) }}</div>
              </div>
            </div>

            <!-- 4 格数据网格 -->
            <div class="grid grid-cols-4 gap-1.5 mb-3">
              <div class="bg-gray-50 rounded-lg p-2 text-center">
                <div class="text-[10px] text-gray-400 mb-0.5">量化</div>
                <div class="text-xs font-semibold text-gray-800 truncate">
                  {{ item.quant?.label ?? '—' }}
                </div>
              </div>
              <div class="bg-gray-50 rounded-lg p-2 text-center">
                <div class="text-[10px] text-gray-400 mb-0.5">显存</div>
                <div class="text-xs font-semibold text-gray-800">
                  {{ item.result ? fmtGB(item.result.totalNeeded) : '—' }}
                </div>
              </div>
              <div class="bg-gray-50 rounded-lg p-2 text-center">
                <div class="text-[10px] text-gray-400 mb-0.5">显存%</div>
                <div
                  class="text-xs font-semibold"
                  :class="item.result && item.result.vramPct > 95 ? 'text-red-600' : item.result && item.result.vramPct > 80 ? 'text-amber-600' : 'text-gray-800'"
                >
                  {{ item.result ? item.result.vramPct.toFixed(0) + '%' : '—' }}
                </div>
              </div>
              <div class="bg-gray-50 rounded-lg p-2 text-center">
                <div class="text-[10px] text-gray-400 mb-0.5">瓶颈</div>
                <div class="text-xs font-semibold text-gray-800">
                  {{ item.result ? (item.result.bottleneck === 'bandwidth' ? 'BW' : 'Compute') : '—' }}
                </div>
              </div>
            </div>

            <!-- 操作按钮 -->
            <button
              v-if="item.canRun"
              @click="useThisModel(item)"
              class="w-full py-2.5 bg-emerald-600 active:bg-emerald-700 text-white text-sm font-semibold rounded-xl transition-colors"
            >
              {{ t('ranking.use_config') }}
            </button>
            <div v-else class="w-full py-2.5 bg-gray-100 text-gray-400 text-sm font-medium rounded-xl text-center">
              {{ t('ranking.status_oom') }}
            </div>
          </div>
        </div>
        <!-- 加载更多哨兵（移动端） -->
        <div v-if="hasMore" class="text-center py-3 text-xs text-gray-400">
          {{ t('ranking.loading_more', { shown: visibleResults.length, total: modelResults.length }) }}
        </div>
      </div>
    </div>
  </div>
</template>
