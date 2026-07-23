import { QUANT_MAP, FRAMEWORK_MAP } from '../src/data/constants.js'
import {
  KV_CACHE_MAP,
  calcCpuMemTheoreticalBandwidth,
  createCpuMemBwOption,
  normalizeCpuTflops,
  resolveCpuMemBwOption,
} from '../src/data/runtime.js'
import {
  generateCmd,
  getCommandCompatibility,
} from '../src/utils/cmdGen.js'
import {
  getDefaultGpuMemoryUtilization,
  getEffectiveSharedMemoryBandwidth,
  getRuntimeCompatibilityMessage,
  getWeightQuantSupportNote,
  isWeightQuantSupported,
} from '../src/utils/runtime.js'

function assert(condition, message) {
  if (!condition) throw new Error(message)
}

function byId(list, id) {
  const value = list.find(item => item.id === id)
  assert(value, `Missing fixture ${id}`)
  return value
}

function framework(id) {
  return byId(FRAMEWORK_MAP, id)
}

function quant(id) {
  return byId(QUANT_MAP, id)
}

function kv(id = 'auto') {
  return byId(KV_CACHE_MAP, id)
}

const baseModel = {
  id: 'fixture-8b',
  name: 'Fixture 8B',
  type: 'dense',
  params: 8,
  links: { hf: 'https://huggingface.co/example-org/fixture-8b' },
}
const moeModel = {
  ...baseModel,
  id: 'fixture-moe',
  name: 'Fixture MoE',
  architecture: 'Qwen3MoeForCausalLM',
  type: 'moe',
  experts: 8,
}
const nvidia = {
  id: 'rtx4090',
  name: 'RTX 4090',
  vendor: 'nvidia',
  architecture: 'ada',
  bf16: 82.6,
  int8: 330,
  int4: 660,
  bw: 1008,
}
const hopper = {
  id: 'h100',
  name: 'H100',
  vendor: 'nvidia',
  architecture: 'hopper',
  bf16: 989,
  fp8: 1979,
  int8: 1979,
  int4: 3958,
  bw: 3350,
}
const amd = {
  id: 'mi300x',
  name: 'MI300X',
  vendor: 'amd',
  architecture: 'cdna3',
  bf16: 1307,
  fp8: 2614,
  bw: 5300,
}
const apple = {
  id: 'apple_m4_max',
  name: 'Apple M4 Max',
  vendor: 'apple',
  architecture: 'apple-m4',
  bf16: 28,
  bw: 546,
  sharedMemory: true,
  unifiedMemory: true,
}

function config(overrides = {}) {
  return {
    model: baseModel,
    gpu: nvidia,
    gpuCount: 1,
    ppCount: 1,
    epCount: 1,
    ctx: 8192,
    batch: 1,
    promptLen: 1536,
    outputLen: 256,
    quant: quant('bf16'),
    kvCacheQuant: kv(),
    prefixCacheHit: 0,
    speculativeDecoding: false,
    draftLen: 4,
    cpuOffload: false,
    pureCpu: false,
    nglCount: null,
    gpuMemoryUtilization: 0.87,
    ...overrides,
  }
}

const lmdeploy = generateCmd(framework('lmdeploy'), config())
assert(
  lmdeploy?.startsWith('lmdeploy serve api_server example-org/fixture-8b'),
  'A valid BF16 LMDeploy configuration did not produce a command',
)

const baseInt4Config = config({ quant: quant('int4') })
assert(
  generateCmd(framework('vllm'), baseInt4Config) === null,
  'vLLM fabricated an AWQ command from a base checkpoint',
)
const missingQuantizedRepo = getCommandCompatibility(framework('vllm'), baseInt4Config)
assert(
  missingQuantizedRepo.reasons.some(item => item.code === 'quantized-checkpoint-required'),
  'Missing serialized INT4 checkpoint did not return a structured reason',
)

const awqModel = {
  ...baseModel,
  runtimeRepositories: {
    int4: 'example-org/fixture-8b-awq',
  },
}
const int8Model = {
  ...baseModel,
  runtimeRepositories: {
    int8: 'example-org/fixture-8b-int8',
  },
}
const vllmInt4 = generateCmd(
  framework('vllm'),
  config({ model: awqModel, quant: quant('int4') }),
)
assert(
  vllmInt4?.includes('vllm serve example-org/fixture-8b-awq')
    && vllmInt4.includes('--quantization awq')
    && !vllmInt4.includes('vllm serve example-org/fixture-8b '),
  'vLLM did not use the explicit AWQ repository',
)
assert(
  generateCmd(framework('lmdeploy'), config({
    model: int8Model,
    quant: quant('int8'),
  }))?.includes('--backend pytorch'),
  'LMDeploy INT8 command did not select its documented SmoothQuant backend',
)

const vllm = generateCmd(framework('vllm'), config({
  model: moeModel,
  gpuCount: 2,
  epCount: 2,
  speculativeDecoding: true,
  draftModelRepo: 'example-org/fixture-draft',
}))
assert(
  vllm?.includes('--tensor-parallel-size 2')
    && vllm.includes('--data-parallel-size 2')
    && vllm.includes('--enable-expert-parallel')
    && vllm.includes('--spec-model example-org/fixture-draft')
    && vllm.includes('--spec-tokens 4')
    && vllm.includes('--gpu-memory-utilization 0.87')
    && !vllm.includes('--tensor-parallel-size 4')
    && !vllm.includes('--expert-parallel-size'),
  'vLLM command uses stale EP, speculative, or memory flags',
)
assert(
  generateCmd(framework('vllm'), config({ speculativeDecoding: true })) === null
    && getCommandCompatibility(
      framework('vllm'),
      config({ speculativeDecoding: true }),
    ).reasons.some(item => item.code === 'draft-model-repository-required'),
  'vLLM fabricated a speculative command without a draft-model repository',
)

const sglang = generateCmd(framework('sglang'), config({
  model: moeModel,
  gpuCount: 2,
  ppCount: 2,
  epCount: 2,
  kvCacheQuant: kv('fp8'),
}))
assert(
  sglang?.includes('--tp-size 4')
    && sglang.includes('--dp-size 2')
    && sglang.includes('--ep-size 4')
    && sglang.includes('--enable-dp-attention')
    && sglang.includes('--pp-size 2')
    && sglang.includes('--disable-overlap-schedule')
    && sglang.includes('--kv-cache-dtype fp8_e4m3')
    && sglang.includes('--mem-fraction-static 0.87')
    && !sglang.includes('--enable-prefix-caching'),
  'SGLang command does not map TP, DP-attention, EP, and PP consistently',
)
const unsupportedSglangEpModel = {
  ...moeModel,
  id: 'fixture-unsupported-moe',
  architecture: undefined,
}
assert(
  generateCmd(framework('sglang'), config({
    model: unsupportedSglangEpModel,
    gpuCount: 2,
    epCount: 2,
  })) === null,
  'SGLang emitted a DP-attention EP command for an undocumented model family',
)

const sglangSpeculative = generateCmd(framework('sglang'), config({
  speculativeDecoding: true,
  draftModelRepo: 'example-org/fixture-draft',
}))
assert(
  sglangSpeculative?.includes('--speculative-draft-model-path example-org/fixture-draft')
    && sglangSpeculative.includes('--speculative-num-draft-tokens 4'),
  'SGLang standalone speculative command uses stale flags',
)
const sglangPpSpecConfig = config({
  gpuCount: 2,
  ppCount: 2,
  speculativeDecoding: true,
  draftModelRepo: 'example-org/fixture-draft',
})
assert(
  generateCmd(framework('sglang'), sglangPpSpecConfig) === null
    && getCommandCompatibility(
      framework('sglang'),
      sglangPpSpecConfig,
    ).reasons.some(item => item.code === 'topology-feature-conflict'),
  'SGLang emitted an invalid PP plus speculative-decoding command',
)

const tgiPromptBudget = generateCmd(framework('tgi'), config({
  batch: 3,
  promptLen: 1536,
}))
assert(
  tgiPromptBudget?.includes('--max-input-tokens 1536')
    && tgiPromptBudget.includes('--max-batch-prefill-tokens 4608')
    && tgiPromptBudget.includes('--cuda-memory-fraction 0.87')
    && !tgiPromptBudget.includes('--max-batch-prefill-tokens 24576'),
  'TGI sized its prefill budget from context capacity instead of prompt length',
)
assert(
  generateCmd(framework('tgi'), config({
    quant: quant('int8'),
  }))?.includes('--quantize bitsandbytes'),
  'TGI rejected its documented on-load INT8 quantization for a base checkpoint',
)
assert(
  generateCmd(framework('tgi'), config({
    gpu: hopper,
    quant: quant('fp8'),
  }))?.includes('--quantize fp8'),
  'TGI rejected its documented on-load FP8 quantization on Hopper',
)
assert(
  !isWeightQuantSupported(
    framework('tgi'),
    { ...nvidia, fp8: 165 },
    quant('fp8'),
  ),
  'TGI accepted FP8 weights on Ada even though its documented path requires H100-class hardware',
)

const tgiAmd = generateCmd(framework('tgi'), config({
  gpu: amd,
  quant: quant('bf16'),
}))
assert(
  tgiAmd?.includes('--device=/dev/kfd')
    && tgiAmd.includes('--device=/dev/dri')
    && tgiAmd.includes('-rocm'),
  'TGI AMD command did not use the ROCm container/device mapping',
)
assert(
  generateCmd(framework('tgi'), config({
    gpu: amd,
    quant: quant('int4'),
    model: awqModel,
  })) === null,
  'TGI emitted an unsupported AMD AWQ command',
)
assert(
  generateCmd(framework('tgi'), config({
    gpu: amd,
    kvCacheQuant: kv('fp8'),
  })) === null,
  'TGI emitted its CUDA-only FP8 KV-cache flag for AMD',
)
assert(
  generateCmd(framework('tgi'), config({
    gpu: {
      ...amd,
      id: 'rx7900xtx',
      name: 'Radeon RX 7900 XTX',
      architecture: 'rdna3',
    },
  })) === null,
  'TGI emitted a ROCm command for undocumented Radeon hardware',
)

const llamaCpu = generateCmd(framework('llamacpp'), config({
  quant: quant('int5'),
  kvCacheQuant: kv('int4'),
  batch: 4,
  pureCpu: true,
}))
assert(
  llamaCpu?.includes('--ctx-size 32768')
    && llamaCpu.includes('--parallel 4')
    && llamaCpu.includes('--n-gpu-layers 0')
    && llamaCpu.includes('--device none'),
  'llama.cpp did not use total context or explicit CPU-only flags',
)

const llamaMultiGpu = generateCmd(framework('llamacpp'), config({
  quant: quant('int5'),
  gpuCount: 2,
}))
assert(
  llamaMultiGpu?.includes('--split-mode row')
    && !llamaMultiGpu.includes('--split-mode layer'),
  'llama.cpp did not use row-parallel splitting for a tensor-sharded multi-GPU estimate',
)

const mlx = generateCmd(framework('mlx'), config({
  gpu: apple,
  quant: quant('int4'),
}))
const mlxSections = mlx?.split('# 2. Serve the converted model') ?? []
assert(
  mlxSections.length === 2
    && mlxSections[0].includes('--quantize')
    && mlxSections[0].includes('--q-bits 4')
    && mlxSections[1].includes('mlx_lm.server --model ./mlx_model')
    && !mlxSections[1].includes('--quantize')
    && !mlxSections[1].includes('--max-tokens'),
  'MLX mixed conversion-only flags into the server command',
)
assert(
  !isWeightQuantSupported(framework('mlx'), apple, quant('int5')),
  'GGUF Q5_K leaked into the MLX runtime capability',
)
assert(
  generateCmd(framework('llamacpp'), config({
    gpu: apple,
    pureCpu: true,
  })) === null,
  'Unified-memory hardware produced a CPU-only command using an unrelated DDR estimate',
)

const trt = generateCmd(framework('trtllm'), config({
  gpu: hopper,
  gpuCount: 2,
  ppCount: 2,
  kvCacheQuant: kv('fp8'),
}))
assert(
  trt?.startsWith('trtllm-serve example-org/fixture-8b')
    && trt.includes('--tp_size 2')
    && trt.includes('--pp_size 2')
    && trt.includes('--kv_cache_dtype fp8')
    && !trt.includes('trtllm-build')
    && !trt.includes('python -m'),
  'TensorRT-LLM command does not use current trtllm-serve flags',
)
const trtEpConfig = config({
  model: moeModel,
  gpu: hopper,
  gpuCount: 2,
  epCount: 2,
})
assert(
  generateCmd(framework('trtllm'), trtEpConfig) === null
    && getCommandCompatibility(
      framework('trtllm'),
      trtEpConfig,
    ).reasons.some(item => item.code === 'ep-unsupported'),
  'TensorRT-LLM emitted a command for the estimator\'s incompatible independent EP topology',
)

const exllamaCompatibility = getCommandCompatibility(
  framework('exllamav2'),
  config(),
)
assert(
  !exllamaCompatibility.supported
    && getRuntimeCompatibilityMessage(exllamaCompatibility)?.includes('TabbyAPI')
    && generateCmd(framework('exllamav2'), config()) === null,
  'ExLlamaV2 fabricated an obsolete HTTP server command',
)

const aggregateConfig = config({
  gpu: {
    ...hopper,
    id: 'nvl72',
    name: 'GB200 NVL72',
    unitKind: 'system',
    physicalGpuCount: 72,
  },
})
const aggregateCompatibility = getCommandCompatibility(framework('vllm'), aggregateConfig)
assert(
  generateCmd(framework('vllm'), aggregateConfig) === null
    && aggregateCompatibility.reasons.some(item => item.code === 'aggregate-topology-required'),
  'Aggregate hardware produced a misleading one-line topology command',
)

for (const invalidModel of [
  { ...baseModel, localInference: false },
  { ...baseModel, availability: 'api-only' },
  { ...baseModel, links: { hf: 'https://huggingface.co/example-org' } },
]) {
  assert(
    generateCmd(framework('vllm'), config({ model: invalidModel })) === null,
    'An API-only or invalid model repository produced a command',
  )
}

const turing = {
  ...nvidia,
  id: 'rtx2080ti_22gb',
  name: 'RTX 2080 Ti 22GB',
  architecture: 'turing',
  nativeBf16: false,
}
assert(
  isWeightQuantSupported(framework('vllm'), turing, quant('bf16'))
    && getWeightQuantSupportNote(turing, quant('bf16'))?.includes('FP16'),
  'The combined BF16/FP16 path did not surface the Turing FP16 fallback',
)
for (const runtimeId of ['vllm', 'sglang', 'lmdeploy', 'tgi']) {
  const command = generateCmd(framework(runtimeId), config({ gpu: turing }))
  assert(
    command?.includes('--dtype float16') && !command.includes('--dtype bfloat16'),
    `${runtimeId} did not emit FP16 for the combined BF16/FP16 option on Turing`,
  )
}
const trtTuringCompatibility = getCommandCompatibility(
  framework('trtllm'),
  config({ gpu: turing }),
)
assert(
  !trtTuringCompatibility.supported
    && trtTuringCompatibility.reasons.some(
      item => item.code === 'gpu-architecture-unsupported',
    ),
  'TensorRT-LLM incorrectly accepted an unsupported Turing GPU',
)
assert(
  !isWeightQuantSupported(
    framework('vllm'),
    { ...nvidia, id: 'gtx980', architecture: 'maxwell', fp8: null },
    quant('fp8'),
  ),
  'Maxwell hardware incorrectly passed the FP8 capability gate',
)

assert(
  calcCpuMemTheoreticalBandwidth(4800, 4) === 153.6,
  'DDR5 channel-count bandwidth calculation is incorrect',
)
const measuredMemory = createCpuMemBwOption('ddr5', 6000, 4, 181.25)
assert(
  measuredMemory?.theoreticalBw === 192
    && measuredMemory.bw === 181.25
    && measuredMemory.bandwidthKind === 'measured',
  'Measured memory bandwidth did not override the theoretical value',
)
assert(
  resolveCpuMemBwOption(measuredMemory.id)?.bw === 181.25,
  'Custom memory configuration did not round-trip through its stable ID',
)
assert(
  getEffectiveSharedMemoryBandwidth({ bw: 546, sharedMemory: true }, measuredMemory) === 181.25,
  'Shared-memory GPU bandwidth was not capped by system memory',
)
assert(
  getDefaultGpuMemoryUtilization(framework('vllm'), nvidia) === 0.9
    && getDefaultGpuMemoryUtilization(framework('llamacpp'), { ...nvidia, usableRatio: 0.96 }) === 0.96,
  'Runtime GPU memory-utilization defaults are incorrect',
)
assert(
  normalizeCpuTflops('') === null
    && normalizeCpuTflops(2.75) === 2.75,
  'Optional CPU throughput normalization is incorrect',
)

console.log('Runtime/command regression passed.')
