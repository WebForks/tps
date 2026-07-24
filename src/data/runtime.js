export const KV_CACHE_MAP = [
  { id: 'auto', label: 'Auto', bytes: null },
  { id: 'fp16', label: 'FP16', bytes: 2.0 },
  { id: 'fp8', label: 'FP8', bytes: 1.0 },
  { id: 'int8', label: 'INT8', bytes: 1.0 },
  { id: 'int4', label: 'INT4', bytes: 0.5 },
]

// Framework-level KV cache formats exposed by this estimator.
//
// This deliberately distinguishes INT4 from framework-specific FP4/NVFP4
// cache formats. A framework supporting only FP4 must not silently accept the
// generic INT4 option because their storage and hardware requirements differ.
// `auto` and `fp16` represent the framework/model-native full-precision cache.
const DEFAULT_KV_CACHE_SUPPORT = Object.freeze(['auto', 'fp16'])

export const KV_CACHE_SUPPORT_BY_FRAMEWORK = Object.freeze({
  theory:          Object.freeze(['auto', 'fp16', 'fp8', 'int8', 'int4']),
  // Current trtllm-serve exposes auto/fp8 (and NVFP4, which is deliberately
  // not represented by the generic INT4 estimator option).
  trtllm:          Object.freeze(['auto', 'fp16', 'fp8']),
  sglang:          Object.freeze(['auto', 'fp16', 'fp8']),
  vllm:            Object.freeze(['auto', 'fp16', 'fp8']),
  // LMDeploy exposes KV quantization through --quant-policy.
  lmdeploy:        Object.freeze(['auto', 'fp16', 'fp8', 'int8', 'int4']),
  tgi:             Object.freeze(['auto', 'fp16', 'fp8']),
  exllamav2:       Object.freeze(['auto', 'fp16', 'int8', 'int4']),
  // mlx_lm.server does not currently expose the generate API's KV
  // quantization controls as a stable server option.
  mlx:             Object.freeze(['auto', 'fp16']),
  llamacpp_metal:  Object.freeze(['auto', 'fp16', 'int8', 'int4']),
  llamacpp:        Object.freeze(['auto', 'fp16', 'int8', 'int4']),
})

function getId(value) {
  return typeof value === 'string' ? value : value?.id
}

export function getSupportedKvCacheIds(framework) {
  return KV_CACHE_SUPPORT_BY_FRAMEWORK[getId(framework)] ?? DEFAULT_KV_CACHE_SUPPORT
}

export function isKvCacheSupported(framework, kvCache) {
  return getSupportedKvCacheIds(framework).includes(getId(kvCache) ?? 'auto')
}

export function normalizeKvCacheOption(framework, kvCache) {
  const option = KV_CACHE_MAP.find(item => item.id === getId(kvCache))
  return option && isKvCacheSupported(framework, option)
    ? option
    : KV_CACHE_MAP[0]
}

export const PREFIX_CACHE_OPTIONS = [0, 25, 50, 75, 90]

// PCIe bandwidth presets for MoE CPU offload mode.
// Like INTERCONNECT_MAP, `bw` is one-way bandwidth because each transfer-time
// formula models a payload moving in one direction.
// bw = x16 单向理论峰值 (GB/s)：gen3=16, gen4=32, gen5=64
// calc.js 再乘以所选链路宽度的 ratio（x4/x8/x16）。
export const PCIE_BW_OPTIONS = [
  { id: 'gen3', label: 'PCIe 3.0', bw: 16 },  // x16 单向峰值 16 GB/s → x8 实际 8 GB/s
  { id: 'gen4', label: 'PCIe 4.0', bw: 32 },  // x16 单向峰值 32 GB/s → x8 实际 16 GB/s
  { id: 'gen5', label: 'PCIe 5.0', bw: 64 },  // x16 单向峰值 64 GB/s → x8 实际 32 GB/s
]

// CPU memory configuration for pure-CPU and llama.cpp hybrid inference.
//
// `bw` is the dual-channel *theoretical* bandwidth in decimal GB/s:
//   transfer rate (MT/s) × 8 bytes/transfer × channels / 1000
//
// Do not apply an "observed bandwidth" discount here. calc.js applies the
// selected framework's decode-efficiency factor separately, so discounting
// these values as well would count utilization twice.
export const CPU_MEM_CHANNELS = 2
export const CPU_MEM_CHANNEL_OPTIONS = [1, 2, 4, 6, 8, 12]
export const CPU_MEM_CHANNELS_MIN = 1
export const CPU_MEM_CHANNELS_MAX = 16
export const CPU_MEM_MEASURED_BW_MAX_GBS = 10000
export const CPU_TFLOPS_MIN = 0.01
export const CPU_TFLOPS_MAX = 1000

export const CPU_MEM_GENERATIONS = [
  { id: 'ddr3', label: 'DDR3', defaultTransferRate: 1600, minTransferRate: 800,  maxTransferRate: 3200 },
  { id: 'ddr4', label: 'DDR4', defaultTransferRate: 3200, minTransferRate: 1600, maxTransferRate: 6000 },
  { id: 'ddr5', label: 'DDR5', defaultTransferRate: 4800, minTransferRate: 3200, maxTransferRate: 16000 },
]

export const CPU_MEM_TRANSFER_RATE_PRESETS = {
  ddr3: [800, 1066, 1333, 1600, 1866, 2133],
  ddr4: [1600, 1866, 2133, 2400, 2666, 2933, 3200, 3600],
  ddr5: [4000, 4400, 4800, 5200, 5600, 6000, 6400, 7200, 8000],
}

export function normalizeCpuMemTransferRate(generation, value) {
  const config = CPU_MEM_GENERATIONS.find(option => option.id === generation)
  if (value == null || (typeof value === 'string' && value.trim() === '')) return null
  const numeric = Number(value)
  if (!config || !Number.isFinite(numeric)) return null
  return Math.min(config.maxTransferRate, Math.max(config.minTransferRate, Math.round(numeric)))
}

export function calcCpuMemTheoreticalBandwidth(transferRate, channels = CPU_MEM_CHANNELS) {
  const rate = Number(transferRate)
  const channelCount = Number(channels)
  if (!Number.isFinite(rate) || rate <= 0 || !Number.isFinite(channelCount) || channelCount <= 0) return null
  return Number((rate * 8 * channelCount / 1000).toFixed(3))
}

export function normalizeCpuMemChannels(value, fallback = CPU_MEM_CHANNELS) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return fallback
  return Math.min(
    CPU_MEM_CHANNELS_MAX,
    Math.max(CPU_MEM_CHANNELS_MIN, Math.round(numeric)),
  )
}

export function normalizeCpuMemMeasuredBandwidth(value, fallback = null) {
  if (value == null || (typeof value === 'string' && value.trim() === '')) return fallback
  const numeric = Number(value)
  if (!Number.isFinite(numeric) || numeric <= 0) return fallback
  return Number(Math.min(CPU_MEM_MEASURED_BW_MAX_GBS, numeric).toFixed(3))
}

/**
 * Optional peak CPU FP16/BF16 throughput used for compute-bound work. The
 * selected framework efficiency is applied later by calc.js, just as it is
 * for the GPU catalog's peak-throughput fields.
 * `null` intentionally means unknown so callers can retain the estimator's
 * conservative fallback instead of inventing a desktop-CPU capability.
 */
export function normalizeCpuTflops(value, fallback = null) {
  if (value == null || (typeof value === 'string' && value.trim() === '')) return fallback
  const numeric = Number(value)
  if (!Number.isFinite(numeric) || numeric <= 0) return fallback
  return Number(Math.min(CPU_TFLOPS_MAX, Math.max(CPU_TFLOPS_MIN, numeric)).toFixed(3))
}

export function createCpuMemBwOption(
  generation,
  transferRate,
  channels = CPU_MEM_CHANNELS,
  measuredBandwidth = null,
) {
  const config = CPU_MEM_GENERATIONS.find(option => option.id === generation)
  const rate = normalizeCpuMemTransferRate(generation, transferRate)
  const channelCount = normalizeCpuMemChannels(channels)
  const theoreticalBw = calcCpuMemTheoreticalBandwidth(rate, channelCount)
  const measuredBw = normalizeCpuMemMeasuredBandwidth(measuredBandwidth)
  if (!config || rate == null || theoreticalBw == null) return null

  const channelSuffix = channelCount === CPU_MEM_CHANNELS ? '' : `_c${channelCount}`
  const measuredSuffix = measuredBw == null ? '' : `_m${measuredBw}`

  return {
    id: `${config.id}_${rate}${channelSuffix}${measuredSuffix}`,
    label: `${config.label}-${rate}`,
    generation: config.id,
    transferRate: rate,
    channels: channelCount,
    theoreticalBw,
    measuredBw,
    // calc.js consumes `bw`. Keep that interface stable while making the
    // selected source explicit for UI/export consumers.
    bw: measuredBw ?? theoreticalBw,
    bandwidthKind: measuredBw == null ? 'theoretical' : 'measured',
  }
}

export const CPU_MEM_BW_OPTIONS = CPU_MEM_GENERATIONS.flatMap(generation =>
  CPU_MEM_TRANSFER_RATE_PRESETS[generation.id].map(rate =>
    createCpuMemBwOption(generation.id, rate)
  )
)

export function resolveCpuMemBwOption(id) {
  if (!id) return null
  const preset = CPU_MEM_BW_OPTIONS.find(option => option.id === id)
  if (preset) return preset

  // Custom rates/channels/measured bandwidth use one stable URL-safe form:
  // ddr5_4800, ddr5_4800_c4, or ddr5_4800_c4_m180.5.
  const match = /^(ddr[345])_(\d+)(?:_c(\d+))?(?:_m(\d+(?:\.\d+)?))?$/.exec(
    String(id).toLowerCase(),
  )
  return match
    ? createCpuMemBwOption(
        match[1],
        Number(match[2]),
        match[3] == null ? CPU_MEM_CHANNELS : Number(match[3]),
        match[4] == null ? null : Number(match[4]),
      )
    : null
}

// PCIe 插槽宽度，决定 CPU Offload 模式下 expert 权重的实际传输带宽
// ratio = 实际带宽 / PCIE_BW_OPTIONS 中存储的 x16 理论峰值
// 台式机主板常见：x8（多卡），x16（主槽单卡）；服务器/HEDT 可跑 x16
export const PCIE_WIDTH_OPTIONS = [
  { id: 'x4',  label: 'x4',  ratio: 0.25 },
  { id: 'x8',  label: 'x8',  ratio: 0.5  },
  { id: 'x16', label: 'x16', ratio: 1.0  },
]

export function getDefaultPcieWidth(gpuCount = 1) {
  const preferredId = Number(gpuCount) > 1 ? 'x8' : 'x16'
  return PCIE_WIDTH_OPTIONS.find(option => option.id === preferredId)
    ?? PCIE_WIDTH_OPTIONS[0]
}

// 系统内存（RAM）容量预设，用于 CPU Offload / 纯 CPU 模式下的 OOM 校验
export const RAM_CAPACITY_OPTIONS = [16, 32, 48, 64, 96, 128, 192, 256, 384, 512]
export const RAM_CAPACITY_MIN_GB = 8
export const RAM_CAPACITY_MAX_GB = 4096

export function normalizeRamCapacity(value, fallback = null) {
  if (value == null || (typeof value === 'string' && value.trim() === '')) return fallback
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return fallback
  return Math.min(RAM_CAPACITY_MAX_GB, Math.max(RAM_CAPACITY_MIN_GB, Math.round(numeric)))
}
