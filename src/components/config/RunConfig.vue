<script setup>
import { computed, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { QUANT_MAP } from '../../data/constants.js'
import {
  KV_CACHE_MAP,
  PREFIX_CACHE_OPTIONS,
  PCIE_BW_OPTIONS,
  PCIE_WIDTH_OPTIONS,
  CPU_MEM_GENERATIONS,
  CPU_MEM_TRANSFER_RATE_PRESETS,
  CPU_MEM_CHANNEL_OPTIONS,
  CPU_MEM_CHANNELS_MIN,
  CPU_MEM_CHANNELS_MAX,
  CPU_MEM_MEASURED_BW_MAX_GBS,
  CPU_TFLOPS_MIN,
  CPU_TFLOPS_MAX,
  RAM_CAPACITY_OPTIONS,
  RAM_CAPACITY_MIN_GB,
  RAM_CAPACITY_MAX_GB,
  createCpuMemBwOption,
  isKvCacheSupported,
  normalizeCpuMemChannels,
  normalizeCpuMemMeasuredBandwidth,
  normalizeCpuTflops,
  normalizeRamCapacity,
  normalizeKvCacheOption,
} from '../../data/runtime.js'
import { fmtCtx } from '../../utils/format.js'
import {
  getDefaultGpuMemoryUtilization,
  isWeightQuantSupported,
  normalizeGpuMemoryUtilization,
  normalizeWeightQuantOption,
  supportsRuntimeFeature,
  usesFp16ForCombinedPrecision,
} from '../../utils/runtime.js'
import TipIcon from '../ui/TipIcon.vue'

const { t } = useI18n()

const props = defineProps({
  model: { type: Object, default: null },
  framework: { type: Object, default: null },
  gpu: { type: Object, default: null },
  gpuCount: { type: Number, default: 1 },
  gpuSharedMemory: { type: Boolean, default: false },
})

const quant = defineModel('quant', { required: true })
const ctx = defineModel('ctx', { required: true })
const batch = defineModel('batch', { required: true })
const promptLen = defineModel('promptLen', { required: true })
const outputLen = defineModel('outputLen', { required: true })
const flashAttention = defineModel('flashAttention', { required: true })
const kvCacheQuant = defineModel('kvCacheQuant', { required: true })
const prefixCacheHit = defineModel('prefixCacheHit', { required: true })
const cpuOffload = defineModel('cpuOffload', { required: true })
const pcieBw = defineModel('pcieBw', { required: true })
const pureCpu = defineModel('pureCpu', { required: true })
const cpuMemBw = defineModel('cpuMemBw', { required: true })
const speculativeDecoding = defineModel('speculativeDecoding', { required: true })
const acceptanceRate = defineModel('acceptanceRate', { required: true })
const draftLen = defineModel('draftLen', { required: true })
const draftModelParams = defineModel('draftModelParams', { default: 1 })
const ppCount = defineModel('ppCount', { required: true })
const imageCount = defineModel('imageCount', { required: true })
const nglCount       = defineModel('nglCount', { default: null })
const epCount        = defineModel('epCount', { default: 1 })
const pcieWidth      = defineModel('pcieWidth', { required: true })
const sysRam         = defineModel('sysRam', { required: true })
const gpuMemoryUtilization = defineModel('gpuMemoryUtilization', { default: null })
const cpuTflops      = defineModel('cpuTflops', { default: null })
const cpuOffloadMode = defineModel('cpuOffloadMode', { default: 'auto' })

// PP 显示条件：至少 2 张卡且模型参数 >= 30B
const ppSupported = computed(() =>
  !pureCpu.value
  && !cpuOffload.value
  && supportsRuntimeFeature(props.framework, 'pp')
  && props.gpuCount >= 2
  && (props.model?.params ?? 0) >= 30
)
const ppOptions = computed(() => {
  if (!ppSupported.value) return [1]
  return [1, 2, 4, 8, 16].filter(n => n <= props.gpuCount && props.gpuCount % n === 0)
})
const stageGpuCount = computed(() =>
  Math.max(1, props.gpuCount / Math.max(1, ppCount.value))
)
// Keep URL/restored topology valid when model or GPU count changes.
watch([ppSupported, ppOptions], ([supported, options]) => {
  if (!supported || !options.includes(ppCount.value)) ppCount.value = 1
}, { immediate: true })

// EP 显示条件：MoE 模型且有 experts 字段，至少 2 张卡
const epSupported = computed(() =>
  !pureCpu.value &&
  !cpuOffload.value &&
  supportsRuntimeFeature(props.framework, 'ep') &&
  props.model?.type === 'moe' &&
  props.model?.experts != null &&
  props.gpuCount >= 2
)
// EP 可选值：1（不启用）+ experts 的因子（不超过 gpuCount）
const epOptions = computed(() => {
  if (!epSupported.value) return [1]
  const experts = props.model.experts
  const maxEp = Math.min(experts, stageGpuCount.value)
  const options = [1]
  for (const n of [2, 4, 8, 16, 32, 64, 128, 256]) {
    if (n <= maxEp && experts % n === 0 && stageGpuCount.value % n === 0) options.push(n)
  }
  return options
})
watch([epSupported, epOptions], ([supported, options]) => {
  if (!supported || !options.includes(epCount.value)) epCount.value = 1
}, { immediate: true })
const speculativeSupported = computed(() =>
  supportsRuntimeFeature(props.framework, 'speculative')
)
const maxDraftModelParams = computed(() =>
  Math.max(0.5, Math.min(32, (props.model?.active_params ?? props.model?.params ?? 8) * 0.5))
)
const supportsImages = computed(() => {
  const tags = (props.model?.tags ?? []).map(tag => String(tag).toLowerCase())
  return props.model?.vision_seq_tokens != null
    || tags.includes('vision')
    || tags.includes('image')
    || (tags.includes('multimodal') && !tags.includes('audio'))
})
const visionTokensPerImage = computed(() => {
  const tokens = Number(props.model?.vision_seq_tokens)
  return Number.isFinite(tokens) && tokens > 0 ? Math.round(tokens) : 1024
})
const visionTokensWereInferred = computed(() =>
  supportsImages.value && !(
    Number.isFinite(Number(props.model?.vision_seq_tokens))
    && Number(props.model?.vision_seq_tokens) > 0
  )
)
watch([draftModelParams, maxDraftModelParams], ([value]) => {
  const numeric = Number(value)
  const normalized = Number.isFinite(numeric)
    ? Math.min(maxDraftModelParams.value, Math.max(0.1, numeric))
    : 1
  if (numeric !== normalized) draftModelParams.value = normalized
}, { immediate: true })

// 当切换到不支持的框架时，自动关闭 Speculative Decoding
watch(() => props.framework, (newFramework) => {
  if (newFramework && !speculativeSupported.value && speculativeDecoding.value) {
    speculativeDecoding.value = false
  }
}, { immediate: true })

const isLlamaCppFramework = computed(() => props.framework?.id === 'llamacpp')
const offloadSupported = computed(() =>
  !props.gpuSharedMemory
  && supportsRuntimeFeature(props.framework, 'cpuOffload')
  && (props.model?.type === 'moe' || isLlamaCppFramework.value)
)
const automaticOffloadSupported = computed(() =>
  offloadSupported.value && props.model?.type === 'moe'
)
const pureCpuSupported = computed(() =>
  !props.gpu?.unifiedMemory
  &&
  supportsRuntimeFeature(props.framework, 'pureCpu')
)
const effectiveNgl = computed(() => nglCount.value ?? Math.floor((props.model?.layers ?? 32) / 2))
const usesCpuMemoryBandwidth = computed(() =>
  pureCpu.value || cpuOffload.value || Boolean(props.gpu?.sharedMemory)
)
const usesCpuCompute = computed(() =>
  pureCpu.value
  || (
    cpuOffload.value
    && isLlamaCppFramework.value
    && props.model?.type !== 'moe'
  )
)

const activeCpuMemGeneration = computed(() => {
  const id = cpuMemBw.value?.generation ?? String(cpuMemBw.value?.id ?? '').split('_')[0]
  return CPU_MEM_GENERATIONS.find(option => option.id === id) ??
    CPU_MEM_GENERATIONS.find(option => option.id === 'ddr5')
})
const cpuMemTransferRatePresets = computed(() =>
  CPU_MEM_TRANSFER_RATE_PRESETS[activeCpuMemGeneration.value.id] ?? []
)
const cpuMemBandwidthDisplay = computed(() =>
  Number(cpuMemBw.value?.bw ?? 0).toFixed(1)
)
const cpuMemTheoreticalBandwidthDisplay = computed(() =>
  Number(cpuMemBw.value?.theoreticalBw ?? cpuMemBw.value?.bw ?? 0).toFixed(1)
)
const activeCpuMemChannels = computed(() =>
  normalizeCpuMemChannels(cpuMemBw.value?.channels)
)

const gpuMemoryUtilizationPercent = computed(() =>
  Math.round(normalizeGpuMemoryUtilization(
    gpuMemoryUtilization.value,
    getDefaultGpuMemoryUtilization(props.framework, props.gpu),
  ) * 100)
)

let lastAutomaticGpuMemoryUtilization = getDefaultGpuMemoryUtilization(
  props.framework,
  props.gpu,
)
watch(
  [() => props.framework?.id, () => props.gpu?.id],
  () => {
    const next = getDefaultGpuMemoryUtilization(props.framework, props.gpu)
    const current = Number(gpuMemoryUtilization.value)
    if (
      gpuMemoryUtilization.value == null
      || !Number.isFinite(current)
      || current === lastAutomaticGpuMemoryUtilization
    ) {
      gpuMemoryUtilization.value = next
      lastAutomaticGpuMemoryUtilization = next
    }
  },
  { immediate: true },
)

function selectGpuMemoryUtilization(percent, event = null) {
  gpuMemoryUtilization.value = normalizeGpuMemoryUtilization(Number(percent) / 100)
  if (event?.target) event.target.value = gpuMemoryUtilizationPercent.value
}

function selectCpuMemGeneration(generationId) {
  const generation = CPU_MEM_GENERATIONS.find(option => option.id === generationId)
  if (!generation) return
  const option = createCpuMemBwOption(
    generation.id,
    generation.defaultTransferRate,
    activeCpuMemChannels.value,
  )
  if (option) cpuMemBw.value = option
}

function selectCpuMemTransferRate(value, event = null) {
  const option = createCpuMemBwOption(
    activeCpuMemGeneration.value.id,
    value,
    activeCpuMemChannels.value,
  )
  if (option) cpuMemBw.value = option
  // Reflect clamping/rounding back into the native input immediately.
  if (event?.target) event.target.value = cpuMemBw.value?.transferRate ?? ''
}

function selectCpuMemChannels(value, event = null) {
  const channels = normalizeCpuMemChannels(value, activeCpuMemChannels.value)
  const option = createCpuMemBwOption(
    activeCpuMemGeneration.value.id,
    cpuMemBw.value?.transferRate,
    channels,
  )
  if (option) cpuMemBw.value = option
  if (event?.target) event.target.value = cpuMemBw.value?.channels ?? ''
}

function selectCpuMemMeasuredBandwidth(value, event = null) {
  const measured = normalizeCpuMemMeasuredBandwidth(value)
  const option = createCpuMemBwOption(
    activeCpuMemGeneration.value.id,
    cpuMemBw.value?.transferRate,
    activeCpuMemChannels.value,
    measured,
  )
  if (option) cpuMemBw.value = option
  if (event?.target) event.target.value = cpuMemBw.value?.measuredBw ?? ''
}

function selectCpuTflops(value, event = null) {
  cpuTflops.value = normalizeCpuTflops(value)
  if (event?.target) event.target.value = cpuTflops.value ?? ''
}

function selectSysRam(value, event = null) {
  sysRam.value = normalizeRamCapacity(value, sysRam.value ?? 64)
  if (event?.target) event.target.value = sysRam.value
}

function kvCacheOptionSupported(option) {
  return isKvCacheSupported(props.framework, option)
}

function selectKvCacheOption(option) {
  if (kvCacheOptionSupported(option)) kvCacheQuant.value = option
}

function quantOptionSupported(option) {
  return isWeightQuantSupported(
    props.framework,
    props.gpu,
    option,
    { pureCpu: pureCpu.value },
  )
}

function selectQuantOption(option) {
  if (quantOptionSupported(option)) quant.value = option
}

function quantOptionTitle(option) {
  const supportNote = option?.id === 'bf16' && usesFp16ForCombinedPrecision(props.gpu)
    ? t('run.bf16_fp16_fallback')
    : null
  if (!quantOptionSupported(option)) {
    return t('run.quant_format_unsupported', { format: option.label })
  }
  const qualityNote = option.ppl_loss != null
    ? (option.ppl_loss === 0
        ? t('run.quant_ppl_lossless')
        : t('run.quant_ppl_loss', { loss: option.ppl_loss.toFixed(2) }))
    : ''
  return [supportNote, qualityNote].filter(Boolean).join(' ')
}

watch(
  [
    () => props.framework?.id,
    () => props.gpu?.id,
    () => quant.value?.id,
    pureCpu,
  ],
  () => {
    const normalized = normalizeWeightQuantOption(
      props.framework,
      props.gpu,
      quant.value,
      QUANT_MAP,
      { pureCpu: pureCpu.value },
    )
    if (normalized && normalized.id !== quant.value?.id) quant.value = normalized
  },
  { immediate: true },
)

// URL/session state may contain a cache format selected under a different
// framework. Normalize immediately and whenever either side changes.
watch(
  [() => props.framework?.id, () => kvCacheQuant.value?.id],
  () => {
    const normalized = normalizeKvCacheOption(props.framework, kvCacheQuant.value)
    if (normalized?.id !== kvCacheQuant.value?.id) kvCacheQuant.value = normalized
  },
  { immediate: true }
)

watch(offloadSupported, supported => {
  if (!supported) {
    if (cpuOffload.value) cpuOffload.value = false
  } else if (cpuOffloadMode.value === 'on' && !cpuOffload.value) {
    cpuOffload.value = true
  }
}, { immediate: true })

watch(pureCpuSupported, supported => {
  if (!supported && pureCpu.value) pureCpu.value = false
}, { immediate: true })

// MoE + llamacpp + offload 时清除 nglCount（不需要 NGL 滑块）
watch([cpuOffload, () => props.framework, () => props.model], ([co, fw, m]) => {
  const isLlamaCppHybridDense = co && fw?.id === 'llamacpp' && m?.type !== 'moe'
  if (!isLlamaCppHybridDense) nglCount.value = null
})

// These modes are alternatives, including when state is restored externally
// rather than changed through the three mode buttons below.
watch(pureCpu, enabled => {
  if (enabled && cpuOffload.value) cpuOffload.value = false
})
watch(cpuOffload, enabled => {
  if (enabled && pureCpu.value) pureCpu.value = false
})

const effectiveCpuOffloadMode = computed(() => {
  if (cpuOffloadMode.value === 'auto') {
    return automaticOffloadSupported.value ? 'auto' : 'off'
  }
  if (cpuOffloadMode.value === 'on') {
    return offloadSupported.value ? 'on' : 'off'
  }
  return 'off'
})

function selectComputeMode(mode) {
  if (mode === 'cpu') {
    if (!pureCpuSupported.value) return
    cpuOffloadMode.value = 'off'
    cpuOffload.value = false
    pureCpu.value = true
    return
  }

  pureCpu.value = false
  if (mode === 'auto' && automaticOffloadSupported.value) {
    cpuOffloadMode.value = 'auto'
    return
  }
  if (mode === 'on' && offloadSupported.value) {
    cpuOffloadMode.value = 'on'
    cpuOffload.value = true
    return
  }
  cpuOffloadMode.value = 'off'
  cpuOffload.value = false
}

const BASE_CTX_OPTIONS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 10485760]
const BATCH_OPTIONS = [1, 2, 4, 8, 16, 32, 64, 128, 256]

const ctxOptions = computed(() => {
  const maxCtx = props.model?.max_ctx
  if (!maxCtx) return BASE_CTX_OPTIONS
  const filtered = BASE_CTX_OPTIONS.filter(v => v <= maxCtx)
  if (filtered.length === 0 || filtered[filtered.length - 1] !== maxCtx) {
    filtered.push(maxCtx)
  }
  return filtered
})
</script>

<template>
  <section class="bg-white rounded-xl border border-gray-200 p-4 space-y-4">
    <h2 class="text-sm font-semibold text-gray-700 uppercase tracking-wider">{{ t('run.title') }}</h2>

    <!-- 量化精度 -->
    <div>
      <label class="flex items-center gap-1 text-xs text-gray-500 mb-2">{{ t('run.quant') }}<TipIcon :text="t('run.quant_tip')" /></label>
      <div class="flex flex-wrap gap-1.5">
        <button
          v-for="q in QUANT_MAP"
          :key="q.id"
          @click="selectQuantOption(q)"
          :disabled="!quantOptionSupported(q)"
          :title="quantOptionTitle(q)"
          :class="[
            'px-2.5 py-1.5 rounded-lg text-xs font-medium border transition-colors',
            !quantOptionSupported(q)
              ? 'bg-gray-100 border-gray-200 text-gray-300 cursor-not-allowed'
              : quant.id === q.id
              ? 'bg-emerald-600 border-emerald-600 text-white shadow-sm'
              : 'bg-gray-50 border-gray-200 text-gray-600 hover:bg-gray-100 hover:border-gray-300'
          ]"
        >
          {{ q.label }}
          <span
            v-if="q.ppl_loss != null && q.ppl_loss > 0"
            :class="['ml-0.5 text-[10px] font-normal', quant.id === q.id ? 'text-emerald-200' : 'text-gray-400']"
          >+{{ q.ppl_loss }}</span>
        </button>
      </div>
      <p
        v-if="quant?.id === 'bf16' && usesFp16ForCombinedPrecision(props.gpu)"
        class="mt-1.5 text-xs text-amber-700 bg-amber-50 rounded px-2 py-1.5 border border-amber-200"
      >
        {{ t('run.bf16_fp16_fallback') }}
      </p>
    </div>

    <div v-if="!pureCpu">
      <label class="flex items-center justify-between text-xs text-gray-500 mb-2">
        <span>{{ t('run.gpu_memory_budget') }}</span>
        <span class="text-emerald-700 font-medium">{{ gpuMemoryUtilizationPercent }}%</span>
      </label>
      <input
        type="range"
        :value="gpuMemoryUtilizationPercent"
        @input="selectGpuMemoryUtilization($event.target.value)"
        min="50"
        max="100"
        step="1"
        class="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-emerald-500"
      />
      <p class="mt-1.5 text-xs text-slate-500">
        {{ t('run.gpu_memory_budget_hint') }}
      </p>
    </div>

    <!-- 上下文长度 -->
    <div>
      <label class="flex items-center justify-between text-xs text-gray-500 mb-2">
        <span class="flex items-center gap-1">{{ t('run.ctx') }}<TipIcon :text="t('run.ctx_tip')" /></span>
        <span class="text-emerald-700 font-medium">{{ fmtCtx(ctx) }}</span>
      </label>
      <select
        :value="ctx"
        @change="ctx = Number($event.target.value)"
        class="w-full bg-gray-50 border border-gray-300 rounded-lg px-3 py-2 text-sm text-gray-900 focus:outline-none focus:ring-1 focus:ring-emerald-500"
      >
        <option v-for="v in ctxOptions" :key="v" :value="v">{{ fmtCtx(v) }}</option>
      </select>
    </div>

    <!-- 并发数 -->
    <div>
      <label class="flex items-center justify-between text-xs text-gray-500 mb-2">
        <span class="flex items-center gap-1">{{ t('run.batch') }}<TipIcon :text="t('run.batch_tip')" /></span>
        <span class="text-emerald-700 font-medium">{{ batch }}</span>
      </label>

      <div class="flex gap-1.5 flex-wrap">
        <button
          v-for="n in BATCH_OPTIONS"
          :key="n"
          @click="batch = n"
          :class="[
            'px-2.5 py-1 rounded-lg text-xs font-medium border transition-colors',
            batch === n
              ? 'bg-emerald-600 border-emerald-600 text-white shadow-sm'
              : 'bg-gray-50 border-gray-200 text-gray-600 hover:bg-gray-100 hover:border-gray-300'
          ]"
        >{{ n }}</button>
      </div>
    </div>

    <!-- Prompt / Output 长度 -->
    <div class="grid grid-cols-1 gap-3">
      <div>
        <label class="flex items-center justify-between text-xs text-gray-500 mb-1">
          <span class="flex items-center gap-1">{{ t('run.prompt') }}<TipIcon :text="t('run.prompt_tip')" /></span>
          <span class="text-emerald-700">{{ promptLen }}</span>
        </label>
        <input
          v-model.number="promptLen"
          type="number"
          min="1"
          step="64"
          class="w-full bg-gray-50 border border-gray-300 rounded-lg px-2 py-1.5 text-sm text-gray-900 focus:outline-none focus:ring-1 focus:ring-emerald-500"
        />
      </div>
      <div>
        <label class="flex items-center justify-between text-xs text-gray-500 mb-1">
          <span class="flex items-center gap-1">{{ t('run.output') }}<TipIcon :text="t('run.output_tip')" /></span>
          <span class="text-emerald-700">{{ outputLen }}</span>
        </label>
        <input
          v-model.number="outputLen"
          type="number"
          min="1"
          step="64"
          class="w-full bg-gray-50 border border-gray-300 rounded-lg px-2 py-1.5 text-sm text-gray-900 focus:outline-none focus:ring-1 focus:ring-emerald-500"
        />
      </div>
    </div>

    <!-- Attention / KV / Prefix -->
    <div class="space-y-3 pt-1 border-t border-gray-100">
      <div>
        <label class="flex items-center gap-1 text-xs text-gray-500 mb-2">{{ t('run.flash_attention') }}<TipIcon :text="t('run.flash_attention_tip')" /></label>
        <div class="grid grid-cols-2 gap-1.5">
          <button
            @click="flashAttention = true"
            :class="[
              'px-2.5 py-1.5 rounded-lg text-xs font-medium border transition-colors',
              flashAttention
                ? 'bg-emerald-600 border-emerald-600 text-white shadow-sm'
                : 'bg-gray-50 border-gray-200 text-gray-600 hover:bg-gray-100 hover:border-gray-300'
            ]"
          >{{ t('run.enabled') }}</button>
          <button
            @click="flashAttention = false"
            :class="[
              'px-2.5 py-1.5 rounded-lg text-xs font-medium border transition-colors',
              !flashAttention
                ? 'bg-emerald-600 border-emerald-600 text-white shadow-sm'
                : 'bg-gray-50 border-gray-200 text-gray-600 hover:bg-gray-100 hover:border-gray-300'
            ]"
          >{{ t('run.disabled') }}</button>
        </div>
      </div>

      <div>
        <label class="flex items-center gap-1 text-xs text-gray-500 mb-2">{{ t('run.kv_cache_quant') }}<TipIcon :text="t('run.kv_cache_quant_tip')" /></label>
        <div class="flex flex-wrap gap-1.5">
          <button
            v-for="option in KV_CACHE_MAP"
            :key="option.id"
            @click="selectKvCacheOption(option)"
            :disabled="!kvCacheOptionSupported(option)"
            :title="!kvCacheOptionSupported(option) ? t('run.kv_cache_format_unsupported') : ''"
            :class="[
              'px-2.5 py-1.5 rounded-lg text-xs font-medium border transition-colors',
              !kvCacheOptionSupported(option)
                ? 'bg-gray-100 border-gray-200 text-gray-400 cursor-not-allowed opacity-60'
                : kvCacheQuant.id === option.id
                ? 'bg-emerald-600 border-emerald-600 text-white shadow-sm'
                : 'bg-gray-50 border-gray-200 text-gray-600 hover:bg-gray-100 hover:border-gray-300'
            ]"
          >
            {{ option.label }}
          </button>
        </div>
      </div>

      <div>
        <label class="flex items-center justify-between text-xs text-gray-500 mb-2">
          <span class="flex items-center gap-1">{{ t('run.prefix_cache') }}<TipIcon :text="t('run.prefix_cache_tip')" /></span>
          <span class="text-emerald-700 font-medium">{{ prefixCacheHit }}%</span>
        </label>
        <div class="flex gap-1.5 flex-wrap">
          <button
            v-for="n in PREFIX_CACHE_OPTIONS"
            :key="n"
            @click="prefixCacheHit = n"
            :class="[
              'px-2.5 py-1 rounded-lg text-xs font-medium border transition-colors',
              prefixCacheHit === n
                ? 'bg-emerald-600 border-emerald-600 text-white shadow-sm'
                : 'bg-gray-50 border-gray-200 text-gray-600 hover:bg-gray-100 hover:border-gray-300'
            ]"
          >{{ n }}%</button>
        </div>
      </div>

      <!-- 计算模式：GPU / GPU+CPU Offload / 纯 CPU -->
      <div>
        <label class="flex items-center gap-1 text-xs text-gray-500 mb-2">
          {{ t('run.compute_mode') }}
          <TipIcon v-if="cpuOffload" :text="t('run.cpu_offload_tip')" />
          <TipIcon v-else-if="pureCpu" :text="t('run.cpu_mem_bw_tip')" />
        </label>
        <div class="flex gap-1.5 flex-wrap">
          <!-- GPU -->
          <button
            @click="selectComputeMode('off')"
            :class="[
              'px-2.5 py-1.5 rounded-lg text-xs font-medium border transition-colors',
              effectiveCpuOffloadMode === 'off' && !pureCpu
                ? 'bg-emerald-600 border-emerald-600 text-white shadow-sm'
                : 'bg-gray-50 border-gray-200 text-gray-600 hover:bg-gray-100 hover:border-gray-300'
            ]"
          >{{ t('run.compute_mode_gpu') }}</button>
          <!-- GPU + CPU Offload（MoE 模型或 llama.cpp 框架显示）-->
          <button
            v-if="automaticOffloadSupported"
            @click="selectComputeMode('auto')"
            :class="[
              'px-2.5 py-1.5 rounded-lg text-xs font-medium border transition-colors',
              effectiveCpuOffloadMode === 'auto' && !pureCpu
                ? 'bg-emerald-600 border-emerald-600 text-white shadow-sm'
                : 'bg-gray-50 border-gray-200 text-gray-600 hover:bg-gray-100 hover:border-gray-300'
            ]"
          >{{ t('run.compute_mode_offload_auto') }}</button>
          <button
            v-if="offloadSupported"
            @click="selectComputeMode('on')"
            :class="[
              'px-2.5 py-1.5 rounded-lg text-xs font-medium border transition-colors',
              effectiveCpuOffloadMode === 'on' && cpuOffload && !pureCpu
                ? 'bg-emerald-600 border-emerald-600 text-white shadow-sm'
                : 'bg-gray-50 border-gray-200 text-gray-600 hover:bg-gray-100 hover:border-gray-300'
            ]"
          >{{ t('run.compute_mode_offload') }}</button>
          <!-- 纯 CPU -->
          <button
            v-if="pureCpuSupported"
            @click="selectComputeMode('cpu')"
            :class="[
              'px-2.5 py-1.5 rounded-lg text-xs font-medium border transition-colors',
              pureCpu
                ? 'bg-emerald-600 border-emerald-600 text-white shadow-sm'
                : 'bg-gray-50 border-gray-200 text-gray-600 hover:bg-gray-100 hover:border-gray-300'
            ]"
          >{{ t('run.compute_mode_cpu') }}</button>
        </div>

        <!-- llama.cpp + dense：NGL 分层 -->
        <template v-if="cpuOffload && !pureCpu && isLlamaCppFramework && props.model?.type !== 'moe'">
          <label class="flex items-center justify-between text-xs text-gray-500 mt-2 mb-1.5">
            <span class="flex items-center gap-1">{{ t('run.ngl_count') }}<TipIcon :text="t('run.ngl_count_tip')" /></span>
            <span class="text-emerald-700 font-medium">{{ effectiveNgl }} / {{ props.model?.layers ?? '?' }} {{ t('run.layers') }}</span>
          </label>
          <input
            type="range"
            :value="effectiveNgl"
            @input="nglCount = Number($event.target.value)"
            min="0"
            :max="props.model?.layers ?? 64"
            step="1"
            class="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-emerald-500"
          />
          <p class="mt-1.5 text-xs text-slate-500 bg-slate-50 rounded px-2 py-1.5 border border-slate-200">
            ℹ️ {{ t('run.llamacpp_hybrid_note') }}
          </p>
        </template>

        <!-- llamacpp + MoE 或非 llamacpp：expert PCIe offload -->
        <template v-if="cpuOffload && !pureCpu && (!isLlamaCppFramework || props.model?.type === 'moe')">
          <label class="flex items-center gap-1 text-xs text-gray-500 mt-2 mb-1.5">{{ t('run.pcie_bw') }}<TipIcon :text="t('run.pcie_bw_tip')" /></label>
          <div class="flex gap-1.5 flex-wrap">
            <button
              v-for="option in PCIE_BW_OPTIONS"
              :key="option.id"
              @click="pcieBw = option"
              :class="[
                'px-2.5 py-1.5 rounded-lg text-xs font-medium border transition-colors',
                pcieBw.id === option.id
                  ? 'bg-emerald-600 border-emerald-600 text-white shadow-sm'
                  : 'bg-gray-50 border-gray-200 text-gray-600 hover:bg-gray-100 hover:border-gray-300'
              ]"
            >
              {{ option.label }} <span class="opacity-70">({{ option.bw }} GB/s)</span>
            </button>
          </div>
          <label class="flex items-center gap-1 text-xs text-gray-500 mt-2 mb-1.5">{{ t('run.pcie_width') }}<TipIcon :text="t('run.pcie_width_tip')" /></label>
          <div class="flex gap-1.5 flex-wrap">
            <button
              v-for="option in PCIE_WIDTH_OPTIONS"
              :key="option.id"
              @click="pcieWidth = option"
              :class="[
                'px-2.5 py-1.5 rounded-lg text-xs font-medium border transition-colors',
                pcieWidth.id === option.id
                  ? 'bg-emerald-600 border-emerald-600 text-white shadow-sm'
                  : 'bg-gray-50 border-gray-200 text-gray-600 hover:bg-gray-100 hover:border-gray-300'
              ]"
            >{{ option.label }}</button>
          </div>
        </template>

        <!-- CPU memory speed used by every CPU-backed path -->
        <template v-if="usesCpuMemoryBandwidth">
          <label class="flex items-center gap-1 text-xs text-gray-500 mt-3 mb-1.5">
            {{ t('run.ram_generation') }}<TipIcon :text="t('run.ram_generation_tip')" />
          </label>
          <div class="flex gap-1.5 flex-wrap">
            <button
              v-for="generation in CPU_MEM_GENERATIONS"
              :key="generation.id"
              @click="selectCpuMemGeneration(generation.id)"
              :class="[
                'px-2.5 py-1.5 rounded-lg text-xs font-medium border transition-colors',
                activeCpuMemGeneration.id === generation.id
                  ? 'bg-emerald-600 border-emerald-600 text-white shadow-sm'
                  : 'bg-gray-50 border-gray-200 text-gray-600 hover:bg-gray-100 hover:border-gray-300'
              ]"
            >{{ generation.label }}</button>
          </div>

          <label class="flex items-center justify-between text-xs text-gray-500 mt-2 mb-1.5">
            <span class="flex items-center gap-1">
              {{ t('run.ram_transfer_rate') }}<TipIcon :text="t('run.ram_transfer_rate_tip')" />
            </span>
            <span class="text-emerald-700 font-medium">{{ cpuMemBw?.transferRate }} MT/s</span>
          </label>
          <div class="flex gap-1.5 flex-wrap">
            <button
              v-for="rate in cpuMemTransferRatePresets"
              :key="rate"
              @click="selectCpuMemTransferRate(rate)"
              :class="[
                'px-2.5 py-1 rounded-lg text-xs font-medium border transition-colors',
                cpuMemBw?.transferRate === rate
                  ? 'bg-emerald-600 border-emerald-600 text-white shadow-sm'
                  : 'bg-gray-50 border-gray-200 text-gray-600 hover:bg-gray-100 hover:border-gray-300'
              ]"
            >{{ rate }}</button>
          </div>
          <label class="block text-xs text-gray-500 mt-2 mb-1">{{ t('run.ram_transfer_rate_custom') }}</label>
          <div class="flex items-center gap-2">
            <input
              type="number"
              :value="cpuMemBw?.transferRate"
              @change="selectCpuMemTransferRate($event.target.value, $event)"
              :min="activeCpuMemGeneration.minTransferRate"
              :max="activeCpuMemGeneration.maxTransferRate"
              step="1"
              class="w-32 bg-gray-50 border border-gray-300 rounded-lg px-2 py-1.5 text-sm text-gray-900 focus:outline-none focus:ring-1 focus:ring-emerald-500"
            />
            <span class="text-xs text-gray-500">MT/s</span>
          </div>
          <label class="flex items-center justify-between text-xs text-gray-500 mt-2 mb-1.5">
            <span>{{ t('run.ram_channels') }}</span>
            <span class="text-emerald-700 font-medium">{{ activeCpuMemChannels }}</span>
          </label>
          <div class="flex gap-1.5 flex-wrap">
            <button
              v-for="channels in CPU_MEM_CHANNEL_OPTIONS"
              :key="channels"
              @click="selectCpuMemChannels(channels)"
              :class="[
                'px-2.5 py-1 rounded-lg text-xs font-medium border transition-colors',
                activeCpuMemChannels === channels
                  ? 'bg-emerald-600 border-emerald-600 text-white shadow-sm'
                  : 'bg-gray-50 border-gray-200 text-gray-600 hover:bg-gray-100 hover:border-gray-300'
              ]"
            >{{ channels }}</button>
            <input
              type="number"
              :value="activeCpuMemChannels"
              @change="selectCpuMemChannels($event.target.value, $event)"
              :min="CPU_MEM_CHANNELS_MIN"
              :max="CPU_MEM_CHANNELS_MAX"
              step="1"
              :aria-label="t('run.ram_channels_custom')"
              class="w-20 bg-gray-50 border border-gray-300 rounded-lg px-2 py-1 text-xs text-gray-900"
            />
          </div>
          <label class="block text-xs text-gray-500 mt-2 mb-1">
            {{ t('run.ram_measured_bw') }}
          </label>
          <div class="flex items-center gap-2">
            <input
              type="number"
              :value="cpuMemBw?.measuredBw ?? ''"
              @change="selectCpuMemMeasuredBandwidth($event.target.value, $event)"
              min="0.1"
              :max="CPU_MEM_MEASURED_BW_MAX_GBS"
              step="0.1"
              :placeholder="t('run.ram_measured_bw_placeholder')"
              class="w-40 bg-gray-50 border border-gray-300 rounded-lg px-2 py-1.5 text-sm text-gray-900 focus:outline-none focus:ring-1 focus:ring-emerald-500"
            />
            <span class="text-xs text-gray-500">GB/s</span>
          </div>
          <p class="mt-1.5 text-xs text-slate-600 bg-slate-50 rounded px-2 py-1.5 border border-slate-200">
            {{ t('run.ram_bandwidth_summary', {
              theoretical: cpuMemTheoreticalBandwidthDisplay,
              applied: cpuMemBandwidthDisplay,
              kind: t(`run.bandwidth_kind_${cpuMemBw?.bandwidthKind ?? 'theoretical'}`),
            }) }}
            {{ t('run.ram_effective_bw_note') }}
          </p>
        </template>

        <!-- System RAM capacity is needed by every CPU-backed mode -->
        <template v-if="cpuOffload || pureCpu || props.gpu?.sharedMemory">
          <label class="flex items-center gap-1 text-xs text-gray-500 mt-2 mb-1.5">{{ t('run.sys_ram') }}<TipIcon :text="t('run.sys_ram_tip')" /></label>
          <div class="flex gap-1.5 flex-wrap">
            <button
              v-for="n in RAM_CAPACITY_OPTIONS"
              :key="n"
              @click="selectSysRam(n)"
              :class="[
                'px-2.5 py-1 rounded-lg text-xs font-medium border transition-colors',
                sysRam === n
                  ? 'bg-emerald-600 border-emerald-600 text-white shadow-sm'
                  : 'bg-gray-50 border-gray-200 text-gray-600 hover:bg-gray-100 hover:border-gray-300'
              ]"
            >{{ n }} GB</button>
          </div>
          <label class="block text-xs text-gray-500 mt-2 mb-1">{{ t('run.sys_ram_custom') }}</label>
          <div class="flex items-center gap-2">
            <input
              type="number"
              :value="sysRam"
              @change="selectSysRam($event.target.value, $event)"
              :min="RAM_CAPACITY_MIN_GB"
              :max="RAM_CAPACITY_MAX_GB"
              step="1"
              class="w-32 bg-gray-50 border border-gray-300 rounded-lg px-2 py-1.5 text-sm text-gray-900 focus:outline-none focus:ring-1 focus:ring-emerald-500"
            />
            <span class="text-xs text-gray-500">GB</span>
          </div>
        </template>

        <template v-if="usesCpuCompute">
          <label class="block text-xs text-gray-500 mt-2 mb-1">
            {{ t('run.cpu_tflops') }}
          </label>
          <div class="flex items-center gap-2">
            <input
              type="number"
              :value="cpuTflops ?? ''"
              @change="selectCpuTflops($event.target.value, $event)"
              :min="CPU_TFLOPS_MIN"
              :max="CPU_TFLOPS_MAX"
              step="0.01"
              :placeholder="t('run.cpu_tflops_placeholder')"
              class="w-32 bg-gray-50 border border-gray-300 rounded-lg px-2 py-1.5 text-sm text-gray-900 focus:outline-none focus:ring-1 focus:ring-emerald-500"
            />
            <span class="text-xs text-gray-500">TFLOPS</span>
          </div>
          <p class="mt-1.5 text-xs text-slate-500 bg-slate-50 rounded px-2 py-1.5 border border-slate-200">
            ℹ️ {{ t('run.cpu_prefill_note') }}
          </p>
        </template>
      </div>

      <!-- Speculative Decoding -->
      <div>
        <label class="flex items-center gap-1 text-xs text-gray-500 mb-2">
          {{ t('run.speculative_decoding') }}<TipIcon :text="t('run.speculative_decoding_tip')" />
          <span v-if="!speculativeSupported" class="text-red-600 text-[10px] ml-1">({{ t('run.framework_not_supported') }})</span>
        </label>
        <div class="grid grid-cols-2 gap-1.5 mb-2">
          <button
            @click="speculativeDecoding = true"
            :disabled="!speculativeSupported"
            :class="[
              'px-2.5 py-1.5 rounded-lg text-xs font-medium border transition-colors',
              !speculativeSupported
                ? 'bg-gray-100 border-gray-300 text-gray-400 cursor-not-allowed'
                : speculativeDecoding
                  ? 'bg-emerald-600 border-emerald-600 text-white shadow-sm'
                  : 'bg-gray-50 border-gray-200 text-gray-600 hover:bg-gray-100 hover:border-gray-300'
            ]"
          >{{ t('run.enabled') }}</button>
          <button
            @click="speculativeDecoding = false"
            :disabled="!speculativeSupported"
            :class="[
              'px-2.5 py-1.5 rounded-lg text-xs font-medium border transition-colors',
              !speculativeSupported
                ? 'bg-gray-100 border-gray-300 text-gray-400 cursor-not-allowed'
                : !speculativeDecoding
                  ? 'bg-emerald-600 border-emerald-600 text-white shadow-sm'
                  : 'bg-gray-50 border-gray-200 text-gray-600 hover:bg-gray-100 hover:border-gray-300'
            ]"
          >{{ t('run.disabled') }}</button>
        </div>
        <template v-if="speculativeDecoding && speculativeSupported">
          <div class="space-y-2">
            <div>
              <label class="flex items-center justify-between text-xs text-gray-500 mb-1">
                <span class="flex items-center gap-1">{{ t('run.acceptance_rate') }}<TipIcon :text="t('run.acceptance_rate_tip')" /></span>
                <span class="text-emerald-700 font-medium">{{ (acceptanceRate * 100).toFixed(0) }}%</span>
              </label>
              <input
                type="range"
                v-model.number="acceptanceRate"
                min="0.3"
                max="0.9"
                step="0.05"
                class="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-emerald-500"
              />
            </div>
            <div>
              <label class="flex items-center justify-between text-xs text-gray-500 mb-1">
                <span class="flex items-center gap-1">{{ t('run.draft_model_params') }}<TipIcon :text="t('run.draft_model_params_tip')" /></span>
                <span class="text-emerald-700 font-medium">{{ Number(draftModelParams).toFixed(1) }}B</span>
              </label>
              <div class="flex items-center gap-1.5 flex-wrap">
                <button
                  v-for="size in [0.5, 1, 3, 7]"
                  :key="size"
                  @click="draftModelParams = Math.min(size, maxDraftModelParams)"
                  :disabled="size > maxDraftModelParams"
                  :class="[
                    'px-2.5 py-1 rounded-lg text-xs font-medium border transition-colors',
                    size > maxDraftModelParams
                      ? 'bg-gray-100 border-gray-200 text-gray-300 cursor-not-allowed'
                      : Number(draftModelParams) === size
                        ? 'bg-emerald-600 border-emerald-600 text-white shadow-sm'
                        : 'bg-gray-50 border-gray-200 text-gray-600 hover:bg-gray-100'
                  ]"
                >{{ size }}B</button>
                <input
                  type="number"
                  v-model.number="draftModelParams"
                  min="0.1"
                  :max="maxDraftModelParams"
                  step="0.1"
                  class="w-20 bg-gray-50 border border-gray-300 rounded-lg px-2 py-1 text-xs text-gray-900"
                />
              </div>
            </div>
            <div>
              <label class="flex items-center justify-between text-xs text-gray-500 mb-1">
                <span class="flex items-center gap-1">{{ t('run.draft_len') }}<TipIcon :text="t('run.draft_len_tip')" /></span>
                <span class="text-emerald-700 font-medium">{{ draftLen }} tok</span>
              </label>
              <input
                type="range"
                v-model.number="draftLen"
                min="2"
                max="8"
                step="1"
                class="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-emerald-500"
              />
            </div>
            <div class="text-xs text-emerald-700 bg-emerald-50 rounded px-2 py-1.5 border border-emerald-200">
              {{ t('run.speculative_speedup', { speedup: ((1 - Math.pow(Math.min(0.999, acceptanceRate), draftLen + 1)) / (1 - Math.min(0.999, acceptanceRate))).toFixed(2) }) }}
            </div>
          </div>
        </template>
      </div>
    </div>

    <!-- Pipeline Parallel -->
    <div class="pt-1 border-t border-gray-100">
      <label class="flex items-center justify-between text-xs text-gray-500 mb-2">
        <span class="flex items-center gap-1">
          {{ t('run.pp_count') }}<TipIcon :text="t('run.pp_count_tip')" />
          <span v-if="!ppSupported" class="text-orange-500 text-[10px] ml-1">({{ t('run.pp_not_applicable') }})</span>
        </span>
        <span class="text-emerald-700 font-medium">PP{{ ppCount }}</span>
      </label>
      <div class="flex gap-1.5 flex-wrap">
        <button
          v-for="n in ppOptions"
          :key="n"
          @click="ppCount = n"
          :class="[
            'px-2.5 py-1 rounded-lg text-xs font-medium border transition-colors',
            ppCount === n
                ? 'bg-emerald-600 border-emerald-600 text-white shadow-sm'
                : 'bg-gray-50 border-gray-200 text-gray-600 hover:bg-gray-100 hover:border-gray-300'
          ]"
        >PP{{ n }}</button>
      </div>
    </div>

    <!-- Expert Parallel（仅 MoE 模型 + 多卡时显示）-->
    <div v-if="epSupported" class="pt-1 border-t border-gray-100">
      <label class="flex items-center justify-between text-xs text-gray-500 mb-2">
        <span class="flex items-center gap-1">
          {{ t('run.ep_count') }}<TipIcon :text="t('run.ep_count_tip')" />
        </span>
        <span class="text-emerald-700 font-medium">EP{{ epCount }}</span>
      </label>
      <div class="flex gap-1.5 flex-wrap">
        <button
          v-for="n in epOptions"
          :key="n"
          @click="epCount = n"
          :class="[
            'px-2.5 py-1 rounded-lg text-xs font-medium border transition-colors',
            epCount === n
              ? 'bg-emerald-600 border-emerald-600 text-white shadow-sm'
              : 'bg-gray-50 border-gray-200 text-gray-600 hover:bg-gray-100 hover:border-gray-300'
          ]"
        >EP{{ n }}</button>
      </div>
    </div>

    <!-- Vision Image Count (only for multimodal models) -->
    <div v-if="supportsImages" class="pt-1 border-t border-gray-100">
      <label class="flex items-center justify-between text-xs text-gray-500 mb-2">
        <span class="flex items-center gap-1">{{ t('run.image_count') }}<TipIcon :text="t('run.image_count_tip')" /></span>
        <span class="text-emerald-700 font-medium">{{ imageCount }} {{ t('run.images') }}</span>
      </label>
      <div class="flex gap-1.5 flex-wrap">
        <button
          v-for="n in [0, 1, 2, 4, 8]"
          :key="n"
          @click="imageCount = n"
          :class="[
            'px-2.5 py-1 rounded-lg text-xs font-medium border transition-colors',
            imageCount === n
              ? 'bg-emerald-600 border-emerald-600 text-white shadow-sm'
              : 'bg-gray-50 border-gray-200 text-gray-600 hover:bg-gray-100 hover:border-gray-300'
          ]"
        >{{ n === 0 ? t('run.no_image') : n }}</button>
      </div>
      <p v-if="imageCount > 0" class="text-xs text-purple-600 bg-purple-50 rounded px-2 py-1.5 border border-purple-200 mt-2">
        {{ t('run.vision_patch_info', {
          tokens: (visionTokensPerImage * imageCount).toLocaleString(),
          per: visionTokensPerImage.toLocaleString(),
        }) }}
        <span v-if="visionTokensWereInferred"> {{ t('run.vision_patch_inferred') }}</span>
      </p>
    </div>
  </section>
</template>
