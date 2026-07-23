import { KV_CACHE_MAP, isKvCacheSupported } from '../data/runtime.js'

const ALL_WEIGHT_QUANT_IDS = Object.freeze([
  'fp32',
  'bf16',
  'fp8',
  'int8',
  'int6',
  'int5',
  'int4',
  'int3',
  'int2',
])

const DEFAULT_CAPABILITY = Object.freeze({
  vendors: null,
  weightQuants: Object.freeze(['bf16']),
  topology: Object.freeze({
    tp: false,
    pp: false,
    ep: false,
    cpuOffload: false,
    pureCpu: false,
    speculative: false,
  }),
  command: false,
})

function capability({
  vendors = null,
  weightQuants,
  topology = {},
  command = true,
  commandMessage = null,
}) {
  return Object.freeze({
    vendors: vendors == null ? null : Object.freeze([...vendors]),
    weightQuants: Object.freeze([...weightQuants]),
    topology: Object.freeze({
      tp: false,
      pp: false,
      ep: false,
      cpuOffload: false,
      pureCpu: false,
      speculative: false,
      ...topology,
    }),
    command,
    commandMessage,
  })
}

/**
 * One capability source for configuration UI, calculation validation, and
 * command generation. A capability means that the estimator's generic option
 * has a documented runtime representation; it does not assert that every
 * model architecture has a compatible checkpoint in that format.
 */
export const FRAMEWORK_RUNTIME_CAPABILITIES = Object.freeze({
  theory: capability({
    vendors: null,
    weightQuants: ALL_WEIGHT_QUANT_IDS,
    topology: {
      tp: true,
      pp: true,
      ep: true,
      cpuOffload: true,
      pureCpu: true,
      speculative: true,
    },
    command: false,
    commandMessage: 'Theory mode is an estimate and has no deployment command.',
  }),
  trtllm: capability({
    vendors: ['nvidia'],
    weightQuants: ['bf16', 'fp8', 'int8', 'int4'],
    // TensorRT-LLM's moe_expert_parallel_size is nested inside its model
    // parallel world. The estimator's EP control is an independent
    // DP-attention dimension, so exposing it here would validate a different
    // memory layout than trtllm-serve actually launches.
    topology: { tp: true, pp: true, speculative: true },
  }),
  sglang: capability({
    vendors: ['nvidia', 'amd'],
    weightQuants: ['fp32', 'bf16', 'fp8', 'int8', 'int4'],
    topology: { tp: true, pp: true, ep: true, speculative: true },
  }),
  vllm: capability({
    vendors: ['nvidia', 'amd'],
    weightQuants: ['fp32', 'bf16', 'fp8', 'int8', 'int4'],
    topology: { tp: true, pp: true, ep: true, speculative: true },
  }),
  lmdeploy: capability({
    vendors: ['nvidia'],
    weightQuants: ['bf16', 'fp8', 'int8', 'int4'],
    topology: { tp: true },
  }),
  tgi: capability({
    vendors: ['nvidia', 'amd'],
    weightQuants: ['bf16', 'fp8', 'int8', 'int4'],
    topology: { tp: true },
  }),
  exllamav2: capability({
    vendors: ['nvidia'],
    // The estimator's INT6/5/3/2 options are GGUF Q*_K formats, not EXL2
    // bpw selections. Do not treat similarly sized formats as interchangeable.
    weightQuants: ['bf16', 'int8', 'int4'],
    topology: { tp: true },
    command: false,
    commandMessage:
      'ExLlamaV2 does not ship an HTTP server module. Use a maintained server such as TabbyAPI.',
  }),
  mlx: capability({
    vendors: ['apple'],
    // Q6_K/Q5_K/Q3_K/Q2_K are GGUF formats. The generic UI currently has no
    // separate MLX 2/3/6-bit format records, so expose only unambiguous values.
    weightQuants: ['fp32', 'bf16', 'int8', 'int4'],
    topology: {},
  }),
  llamacpp_metal: capability({
    vendors: ['apple'],
    weightQuants: ['fp32', 'bf16', 'int8', 'int6', 'int5', 'int4', 'int3', 'int2'],
    topology: { tp: true },
  }),
  llamacpp: capability({
    vendors: null,
    weightQuants: ['fp32', 'bf16', 'int8', 'int6', 'int5', 'int4', 'int3', 'int2'],
    topology: { tp: true, cpuOffload: true, pureCpu: true },
  }),
})

function getId(value) {
  return typeof value === 'string' ? value : value?.id
}

function positiveFinite(value) {
  const numeric = Number(value)
  return Number.isFinite(numeric) && numeric > 0
}

function integerAtLeastOne(value, fallback = 1) {
  const numeric = Number(value)
  return Number.isFinite(numeric) && Number.isInteger(numeric) && numeric >= 1
    ? numeric
    : fallback
}

export function getFrameworkRuntimeCapability(framework) {
  return FRAMEWORK_RUNTIME_CAPABILITIES[getId(framework)] ?? DEFAULT_CAPABILITY
}

export function supportsRuntimeFeature(framework, feature) {
  return Boolean(getFrameworkRuntimeCapability(framework).topology[feature])
}

/**
 * Infer only architecture families needed for runtime format gates. Catalog
 * records may override this with `architecture` or `gpuArchitecture`.
 */
export function inferGpuArchitecture(gpu) {
  if (!gpu) return null
  const explicit = gpu.architecture ?? gpu.gpuArchitecture
  if (explicit) return String(explicit).toLowerCase()

  const vendor = String(gpu.vendor ?? '').toLowerCase()
  const id = String(gpu.baseGpuId ?? gpu.id ?? '').toLowerCase()
  const name = String(gpu.name ?? '').toLowerCase()
  const value = `${id} ${name}`

  if (vendor === 'nvidia') {
    if (/(?:^|[_\s])(gb300|gb200|b300|b200|b100|rtx50|rtx 50|blackwell|gb10)/.test(value)) return 'blackwell'
    if (/(?:^|[_\s])(h200|h100|h800|h20|hopper)/.test(value)) return 'hopper'
    if (/(?:^|[_\s])(rtx40|rtx 40|l40|l4|ada)/.test(value)) return 'ada'
    if (/(?:^|[_\s])(rtx30|rtx 30|a100|a800|a40|a30|a10|a2|ampere)/.test(value)) return 'ampere'
    if (/(?:^|[_\s])(rtx20|rtx 20|gtx16|gtx 16|t4|turing)/.test(value)) return 'turing'
    if (/(?:^|[_\s])(v100|volta)/.test(value)) return 'volta'
    if (/(?:^|[_\s])(gtx10|gtx 10|p100|pascal)/.test(value)) return 'pascal'
    if (/(?:^|[_\s])(gtx9|gtx 9|maxwell)/.test(value)) return 'maxwell'
  }

  if (vendor === 'amd') {
    if (/(mi3\d\d|cdna3)/.test(value)) return 'cdna3'
    if (/(mi2\d\d|cdna2)/.test(value)) return 'cdna2'
    if (/(rx9\d\d\d|rdna4)/.test(value)) return 'rdna4'
    if (/(rx7\d\d\d|rdna3)/.test(value)) return 'rdna3'
    if (/(rx6\d\d\d|rdna2)/.test(value)) return 'rdna2'
  }

  if (vendor === 'apple') {
    const match = /apple[_\s-]*m(\d+)/.exec(value)
    return match ? `apple-m${match[1]}` : 'apple-silicon'
  }

  return null
}

const NVIDIA_ARCH_LEVEL = Object.freeze({
  maxwell: 1,
  pascal: 2,
  volta: 3,
  turing: 4,
  ampere: 5,
  ada: 6,
  hopper: 7,
  blackwell: 8,
})

function nvidiaArchitectureAtLeast(gpu, minimum) {
  const actual = NVIDIA_ARCH_LEVEL[inferGpuArchitecture(gpu)]
  const required = NVIDIA_ARCH_LEVEL[minimum]
  return actual != null && required != null && actual >= required
}

export function usesFp16ForCombinedPrecision(gpu) {
  if (gpu?.nativeBf16 === false) return true
  if (gpu?.nativeBf16 === true) return false
  if (gpu?.vendor !== 'nvidia') return false

  const architectureLevel = NVIDIA_ARCH_LEVEL[inferGpuArchitecture(gpu)]
  return architectureLevel != null
    && architectureLevel < NVIDIA_ARCH_LEVEL.ampere
}

/**
 * Hardware gate for formats whose memory footprint would otherwise be paired
 * with a made-up BF16 throughput fallback.
 */
export function isNativeWeightFormatSupported(gpu, quant, framework, options = {}) {
  const quantId = getId(quant)
  const frameworkId = getId(framework)
  if (!gpu || options.pureCpu || frameworkId === 'theory') return true

  // GGUF and MLX software quantization can dequantize low-bit weights even
  // when the device lacks a same-width tensor-core throughput field.
  if (['llamacpp', 'llamacpp_metal', 'mlx', 'exllamav2'].includes(frameworkId)) {
    return quantId !== 'fp8'
  }

  if (quantId === 'fp8') {
    // Requiring an explicit catalog throughput prevents, for example, FP8 on
    // Maxwell/Pascal or on an aggregate that silently inherited BF16.
    if (!positiveFinite(gpu.fp8)) return false
    if (gpu.vendor === 'nvidia') {
      // TGI documents FP8 weights for H100-class hardware and newer. Other
      // runtimes expose kernels on Ada as well.
      const minimum = frameworkId === 'tgi' ? 'hopper' : 'ada'
      return nvidiaArchitectureAtLeast(gpu, minimum)
    }
    return true
  }

  if (quantId === 'int8' || quantId === 'int4') {
    if (gpu.vendor === 'nvidia') {
      const minimum = frameworkId === 'trtllm' ? 'ampere' : 'turing'
      return nvidiaArchitectureAtLeast(gpu, minimum)
    }
    // Generic INT8/INT4 does not identify an AMD checkpoint/kernel format.
    // Require an explicit native metric rather than treating BF16 fallback as
    // native low-precision support.
    return positiveFinite(gpu[quantId])
  }

  if (quantId === 'fp32') return positiveFinite(gpu.fp32)
  if (quantId === 'bf16') return positiveFinite(gpu.bf16)
  return true
}

export function isWeightQuantSupported(framework, gpu, quant, options = {}) {
  const frameworkId = getId(framework)
  const quantId = getId(quant)
  const capability = getFrameworkRuntimeCapability(framework)
  if (!frameworkId || !quantId || !capability.weightQuants.includes(quantId)) return false

  // TGI's documented ROCm image currently excludes AWQ and exposes BF16/FP16
  // kernels; CUDA-only FP8 cache/weight flags must not leak into AMD commands.
  if (frameworkId === 'tgi' && gpu?.vendor === 'amd' && quantId !== 'bf16') return false

  return isNativeWeightFormatSupported(gpu, quant, framework, options)
}

export function getSupportedWeightQuantIds(framework, gpu = null, options = {}) {
  return getFrameworkRuntimeCapability(framework).weightQuants.filter(id =>
    isWeightQuantSupported(framework, gpu, id, options),
  )
}

export function normalizeWeightQuantOption(framework, gpu, quant, quantOptions, options = {}) {
  const candidates = Array.isArray(quantOptions) ? quantOptions : []
  if (isWeightQuantSupported(framework, gpu, quant, options)) return quant
  const supportedIds = getSupportedWeightQuantIds(framework, gpu, options)
  return candidates.find(option => option.id === 'bf16' && supportedIds.includes(option.id))
    ?? candidates.find(option => supportedIds.includes(option.id))
    ?? null
}

export function normalizeGpuMemoryUtilization(value, fallback = 0.9) {
  const fallbackNumber = Number(fallback)
  const safeFallback = Number.isFinite(fallbackNumber) ? fallbackNumber : 0.9
  if (value == null || (typeof value === 'string' && value.trim() === '')) {
    return Math.min(1, Math.max(0.05, safeFallback))
  }
  const numeric = Number(value)
  const selected = Number.isFinite(numeric) ? numeric : safeFallback
  return Math.min(1, Math.max(0.05, selected))
}

export function getDefaultGpuMemoryUtilization(framework, gpu = null) {
  const frameworkId = getId(framework)
  if (['trtllm', 'sglang', 'vllm', 'lmdeploy', 'tgi'].includes(frameworkId)) {
    return 0.9
  }
  return normalizeGpuMemoryUtilization(gpu?.usableRatio, 1)
}

/**
 * The catalog's combined BF16/FP16 option is still usable on pre-Ampere
 * NVIDIA cards, but those cards execute the FP16 path. Keep that distinction
 * visible rather than implying native BF16 support.
 */
export function getWeightQuantSupportNote(gpu, quant, translate = null) {
  if (getId(quant) === 'bf16' && usesFp16ForCombinedPrecision(gpu)) {
    if (typeof translate === 'function') return translate('run.bf16_fp16_fallback')
    return 'This GPU uses FP16 for the combined BF16/FP16 option; it does not support native BF16.'
  }
  return null
}

export function getEffectiveSharedMemoryBandwidth(gpu, cpuMemBw) {
  const gpuBandwidth = Number(gpu?.bw)
  const memoryBandwidth = Number(
    cpuMemBw?.measuredBw ?? cpuMemBw?.bw ?? cpuMemBw,
  )
  if (!Number.isFinite(memoryBandwidth) || memoryBandwidth <= 0) {
    return Number.isFinite(gpuBandwidth) && gpuBandwidth > 0 ? gpuBandwidth : null
  }
  if (!gpu?.sharedMemory || !Number.isFinite(gpuBandwidth) || gpuBandwidth <= 0) {
    return memoryBandwidth
  }
  return Math.min(gpuBandwidth, memoryBandwidth)
}

/**
 * Return a strict two-segment Hugging Face model repository, or null. An
 * organization page is intentionally not a model repository.
 */
export function getHuggingFaceRepoId(model) {
  const direct = model?.hfRepo ?? model?.repoId
  if (typeof direct === 'string' && /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(direct.trim())) {
    return direct.trim()
  }

  const raw = model?.links?.hf
  if (typeof raw !== 'string' || !raw.trim()) return null

  try {
    const parsed = new URL(raw)
    if (!['huggingface.co', 'www.huggingface.co'].includes(parsed.hostname.toLowerCase())) {
      return null
    }
    const segments = parsed.pathname.split('/').filter(Boolean)
    if (segments.length < 2) return null
    const [owner, repository] = segments
    if (
      !/^[A-Za-z0-9_.-]+$/.test(owner)
      || !/^[A-Za-z0-9_.-]+$/.test(repository)
      || ['models', 'organizations', 'datasets', 'spaces'].includes(owner.toLowerCase())
    ) {
      return null
    }
    return `${owner}/${repository}`
  } catch {
    return null
  }
}

const SERIALIZED_WEIGHT_FORMATS = Object.freeze(['fp8', 'int8', 'int4'])

function getMappedRuntimeRepository(model, quantId, frameworkId) {
  const mappings = [
    model?.runtimeRepositories,
    model?.quantizedRepositories,
    model?.checkpointRepositories,
  ]
  for (const mapping of mappings) {
    if (!mapping || typeof mapping !== 'object') continue
    const entry = mapping[quantId]
    if (typeof entry === 'string') return entry
    if (entry && typeof entry === 'object') {
      const candidate = entry[frameworkId] ?? entry.repo ?? entry.hf ?? entry.default
      if (typeof candidate === 'string') return candidate
    }
  }
  return null
}

/**
 * Resolve the repository that actually contains the selected weight format.
 * Full-precision, llama.cpp local GGUF, and MLX conversion workflows may start
 * from the base repository. Server runtimes must name a separate serialized
 * low-bit checkpoint explicitly; a base repo plus an AWQ/GPTQ flag is not an
 * AWQ/GPTQ checkpoint.
 */
export function getRuntimeModelRepoId(model, quant, framework) {
  const quantId = getId(quant)
  const frameworkId = getId(framework)
  if (
    !SERIALIZED_WEIGHT_FORMATS.includes(quantId)
    || ['theory', 'llamacpp', 'llamacpp_metal', 'mlx'].includes(frameworkId)
    || (frameworkId === 'tgi' && ['fp8', 'int8'].includes(quantId))
  ) {
    return getHuggingFaceRepoId(model)
  }

  const mapped = getMappedRuntimeRepository(model, quantId, frameworkId)
  return mapped ? getHuggingFaceRepoId({ repoId: mapped }) : null
}

export function getDraftModelRepoId(value) {
  return typeof value === 'string'
    ? getHuggingFaceRepoId({ repoId: value })
    : getHuggingFaceRepoId(value)
}

function reason(code, message) {
  return Object.freeze({ code, message })
}

function supportsSglangDpAttention(model) {
  const identity = [
    model?.architecture,
    model?.modelType,
    model?.id,
    model?.name,
  ].filter(Boolean).join(' ').toLowerCase()
  return /deepseek.*(?:v?2|v?3|r1)|qwen[\s_.-]*(?:2|3)/.test(identity)
}

/**
 * Validate the parts of a runtime configuration represented by the shared
 * capability matrix. The calculator can call this with `forCommand: false`;
 * command generation uses `forCommand: true` to additionally require a local
 * checkpoint and a supported CLI.
 */
export function getRuntimeCompatibility({
  framework,
  gpu = null,
  model = null,
  quant = null,
  kvCacheQuant = KV_CACHE_MAP[0],
  gpuCount = 1,
  ppCount = 1,
  epCount = 1,
  cpuOffload = false,
  pureCpu = false,
  speculativeDecoding = false,
  draftModelRepo = null,
  speculativeConfigPath = null,
  forCommand = false,
} = {}) {
  const frameworkId = getId(framework)
  const capability = getFrameworkRuntimeCapability(framework)
  const reasons = []

  if (!frameworkId || !FRAMEWORK_RUNTIME_CAPABILITIES[frameworkId]) {
    reasons.push(reason('unknown-framework', 'The selected runtime is not recognized.'))
  }

  if (frameworkId !== 'theory' && (model?.localInference === false || model?.availability === 'api-only')) {
    reasons.push(
      reason(
        'local-inference-unavailable',
        'This catalog entry is API-only and has no downloadable local-inference checkpoint.',
      ),
    )
  }

  if (frameworkId !== 'theory' && gpu?.mixedGpu) {
    if (gpu.invalidMemoryMix || gpu.sharedMemory || gpu.unifiedMemory) {
      reasons.push(
        reason(
          'mixed-gpu-memory-unsupported',
          'Shared/unified-memory devices cannot be combined with other GPUs as independent memory shards.',
        ),
      )
    } else if (gpu.mixedVendors || gpu.componentVendors?.length !== 1) {
      reasons.push(
        reason(
          'mixed-gpu-vendor-unsupported',
          'Mixed GPU estimation requires every card to use the same hardware vendor and runtime backend.',
        ),
      )
    } else if (
      gpu.mixedArchitectures
      || !gpu.mixedGpuEstimateSupported
      || gpu.componentArchitectures?.length !== 1
    ) {
      reasons.push(
        reason(
          'mixed-gpu-architecture-unsupported',
          'Mixed GPU estimation requires a known, compatible architecture family on every card.',
        ),
      )
    } else if (forCommand) {
      reasons.push(
        reason(
          'mixed-gpu-command-unsupported',
          'Estimation is available for these mixed GPUs, but rank-to-device placement is deployment-specific, so a safe one-line launch command cannot be generated.',
        ),
      )
    }
  }

  if (
    !pureCpu
    && gpu?.vendor
    && capability.vendors
    && !capability.vendors.includes(gpu.vendor)
  ) {
    reasons.push(
      reason(
        'vendor-unsupported',
        `The selected runtime does not support ${gpu.vendor} hardware.`,
      ),
    )
  }
  if (
    !pureCpu
    && frameworkId === 'trtllm'
    && !nvidiaArchitectureAtLeast(gpu, 'ampere')
  ) {
    reasons.push(
      reason(
        'gpu-architecture-unsupported',
        'Current TensorRT-LLM releases support Ampere, Ada, Hopper, and Blackwell GPUs; Turing and older GPUs are not supported.',
      ),
    )
  }
  if (
    frameworkId === 'tgi'
    && gpu?.vendor === 'amd'
    && !['cdna2', 'cdna3'].includes(inferGpuArchitecture(gpu))
  ) {
    reasons.push(
      reason(
        'amd-hardware-unsupported',
        'The documented TGI ROCm image is supported on AMD Instinct MI210, MI250, and MI300-class GPUs.',
      ),
    )
  }

  if (quant && !isWeightQuantSupported(framework, gpu, quant, { pureCpu })) {
    reasons.push(
      reason(
        'weight-format-unsupported',
        `${getId(quant)?.toUpperCase() ?? 'This weight format'} is not supported by the selected runtime and hardware.`,
      ),
    )
  }

  if (!isKvCacheSupported(framework, kvCacheQuant)) {
    reasons.push(
      reason(
        'kv-format-unsupported',
        `${getId(kvCacheQuant)?.toUpperCase() ?? 'This KV format'} is not supported by the selected runtime.`,
      ),
    )
  }
  if (
    frameworkId === 'tgi'
    && gpu?.vendor === 'amd'
    && getId(kvCacheQuant) === 'fp8'
  ) {
    reasons.push(
      reason(
        'kv-format-hardware-unsupported',
        'The TGI ROCm image does not expose the CUDA FP8 KV-cache path.',
      ),
    )
  }

  const tp = integerAtLeastOne(gpuCount)
  const pp = integerAtLeastOne(ppCount)
  const ep = integerAtLeastOne(epCount)
  if (tp > 1 && !capability.topology.tp) {
    reasons.push(reason('tp-unsupported', 'Tensor parallelism is not supported by the selected runtime.'))
  }
  if (pp > 1 && !capability.topology.pp) {
    reasons.push(reason('pp-unsupported', 'Pipeline parallelism is not supported by the selected runtime.'))
  }
  if (ep > 1 && !capability.topology.ep) {
    reasons.push(reason('ep-unsupported', 'Expert parallelism is not supported by the selected runtime.'))
  }
  if (ep > 1 && model?.type !== 'moe') {
    reasons.push(reason('ep-model-unsupported', 'Expert parallelism requires a Mixture-of-Experts model.'))
  }
  if (
    frameworkId === 'sglang'
    && ep > 1
    && !supportsSglangDpAttention(model)
  ) {
    reasons.push(
      reason(
        'ep-model-family-unsupported',
        'SGLang DP-attention expert parallelism is only enabled for its documented DeepSeek and Qwen MoE model families.',
      ),
    )
  }
  if (frameworkId === 'sglang' && pp > 1 && speculativeDecoding) {
    reasons.push(
      reason(
        'topology-feature-conflict',
        'SGLang does not support speculative decoding together with pipeline parallelism.',
      ),
    )
  }
  if (cpuOffload && !capability.topology.cpuOffload) {
    reasons.push(
      reason(
        'offload-unsupported',
        'This estimator offload mode has no equivalent in the selected runtime.',
      ),
    )
  }
  if (pureCpu && !capability.topology.pureCpu) {
    reasons.push(reason('cpu-unsupported', 'Pure-CPU serving is not supported by the selected runtime.'))
  }
  if (pureCpu && gpu?.unifiedMemory) {
    reasons.push(
      reason(
        'unified-memory-cpu-estimate-unsupported',
        'Pure-CPU mode is not modeled for unified-memory systems; use the shared-memory GPU path so the chip memory bandwidth is applied.',
      ),
    )
  }
  if (speculativeDecoding && !capability.topology.speculative) {
    reasons.push(
      reason(
        'speculative-unsupported',
        'Draft-model speculative decoding is not supported by the selected runtime.',
      ),
    )
  }

  if (forCommand) {
    if (!capability.command) {
      reasons.push(
        reason(
          'command-unavailable',
          capability.commandMessage ?? 'No maintained deployment command is available for this runtime.',
        ),
      )
    }
    if (model?.localInference !== false && model?.availability !== 'api-only' && !getHuggingFaceRepoId(model)) {
      reasons.push(
        reason(
          'model-repository-invalid',
          'A valid Hugging Face model repository is required; organization pages and internal catalog IDs are not runnable checkpoints.',
        ),
      )
    } else if (
      model?.localInference !== false
      && model?.availability !== 'api-only'
      && !getRuntimeModelRepoId(model, quant, framework)
    ) {
      reasons.push(
        reason(
          'quantized-checkpoint-required',
          `${getId(quant)?.toUpperCase() ?? 'The selected'} weights require an explicit repository containing that serialized checkpoint format.`,
        ),
      )
    }
    if (gpu?.unitKind === 'system') {
      const physicalCount = integerAtLeastOne(gpu.physicalGpuCount, 1)
      reasons.push(
        reason(
          'aggregate-topology-required',
          `This ${physicalCount}-GPU aggregate system requires deployment-specific rank, host, and network topology; a safe one-line launch command cannot be generated.`,
        ),
      )
    }
    if (
      speculativeDecoding
      && ['vllm', 'sglang'].includes(frameworkId)
      && !getDraftModelRepoId(draftModelRepo)
    ) {
      reasons.push(
        reason(
          'draft-model-repository-required',
          'Speculative serving requires an explicit downloadable draft-model repository.',
        ),
      )
    }
    if (
      speculativeDecoding
      && frameworkId === 'trtllm'
      && (
        typeof speculativeConfigPath !== 'string'
        || !/^[A-Za-z0-9_./-]+$/.test(speculativeConfigPath)
      )
    ) {
      reasons.push(
        reason(
          'speculative-config-required',
          'TensorRT-LLM speculative serving requires a safe path to a completed LLM-API YAML configuration.',
        ),
      )
    }
  }

  return Object.freeze({
    supported: reasons.length === 0,
    reasons: Object.freeze(reasons),
    capability,
  })
}

export function isRuntimeConfigurationSupported(config) {
  return getRuntimeCompatibility(config).supported
}

export function getRuntimeCompatibilityMessage(compatibility) {
  return compatibility?.reasons?.[0]?.message ?? null
}
