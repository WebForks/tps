import {
  aggregateGpuSlots,
  calcAll,
  calcBatchSweep,
  getQuantBytes,
  getWarnings,
} from '../src/utils/calc.js'
import { getDisplayVramBreakdown } from '../src/utils/vramBreakdown.js'
import { GPU_LIST } from '../src/data/gpus/index.js'
import { ALL_MODELS } from '../src/data/models/index.js'
import {
  FRAMEWORK_MAP,
  INTERCONNECT_MAP,
  QUANT_MAP,
} from '../src/data/constants.js'
import {
  KV_CACHE_MAP,
  PCIE_BW_OPTIONS,
  PCIE_WIDTH_OPTIONS,
  calcCpuMemTheoreticalBandwidth,
  createCpuMemBwOption,
  isKvCacheSupported,
} from '../src/data/runtime.js'
import { generateCmd, getCommandCompatibility } from '../src/utils/cmdGen.js'

function assert(condition, message) {
  if (!condition) throw new Error(message)
}

function approx(actual, expected, tolerance, message) {
  assert(
    Number.isFinite(actual) && Math.abs(actual - expected) <= tolerance,
    `${message}: expected ${expected}, got ${actual}`,
  )
}

function byId(list, id, label) {
  const value = list.find(item => item.id === id)
  assert(value, `${label ?? 'Fixture'} "${id}" is missing`)
  return value
}

function assertFiniteResult(result, label) {
  const fields = [
    'targetWeightGB',
    'targetKvGB',
    'perCardNeeded',
    'perCardVram',
    'cpuRamNeededGB',
    'effectiveToks',
    'singleToks',
    'prefillToks',
    'effectiveTpot',
    'ttft',
    'totalLatency',
    'arithmeticIntensity',
  ]
  for (const field of fields) {
    assert(
      Number.isFinite(result[field]) && result[field] >= 0,
      `${label}: ${field} must be finite and non-negative (got ${result[field]})`,
    )
  }
}

const rtx4090 = byId(GPU_LIST, 'rtx4090', 'GPU')
const rtx4090Mod = byId(GPU_LIST, 'rtx4090_48g_mod', 'GPU')
const rtx4080Super = byId(GPU_LIST, 'rtx4080s', 'GPU')
const rtx4080SuperMod = byId(GPU_LIST, 'rtx4080s_32g_mod', 'GPU')
const rtx3090 = byId(GPU_LIST, 'rtx3090', 'GPU')
const rtx3080 = byId(GPU_LIST, 'rtx3080', 'GPU')
const rtx3080Mod = byId(GPU_LIST, 'rtx3080_20g_mod', 'GPU')
const rtx2080Ti = byId(GPU_LIST, 'rtx2080ti', 'GPU')
const rtx2080TiMod = byId(GPU_LIST, 'rtx2080ti_22g_mod', 'GPU')
const h100 = byId(GPU_LIST, 'h100_sxm', 'GPU')
const h200 = byId(GPU_LIST, 'h200_sxm', 'GPU')
const t4 = byId(GPU_LIST, 't4', 'GPU')
const appleM4 = byId(GPU_LIST, 'apple_m4_16g', 'GPU')
const ryzenAiMax395 = byId(GPU_LIST, 'ryzen_ai_max_395', 'GPU')
const ryzenAiMax395Windows = byId(GPU_LIST, 'ryzen_ai_max_395_win', 'GPU')
const llama8b = byId(ALL_MODELS, 'llama3_8b', 'Model')
const llama70b = byId(ALL_MODELS, 'llama3_70b', 'Model')
const mixtral = byId(ALL_MODELS, 'mixtral_8x7b', 'Model')
const deepseekR1 = byId(ALL_MODELS, 'deepseek_r1', 'Model')
const qwen3_235b = byId(ALL_MODELS, 'qwen3_235b', 'Model')
const deepseekV3 = byId(ALL_MODELS, 'deepseek_v3', 'Model')
const qwen36 = byId(ALL_MODELS, 'qwen3_6_27b', 'Model')
const jambaLarge = byId(ALL_MODELS, 'jamba_1_5_large', 'Model')
const glm47 = byId(ALL_MODELS, 'glm4_7', 'Model')
const glm47Flash = byId(ALL_MODELS, 'glm4_7_flash', 'Model')
const mistralLarge3 = byId(ALL_MODELS, 'mistral_large_3', 'Model')
const gemma431 = byId(ALL_MODELS, 'gemma4_31b', 'Model')
const mamba28 = byId(ALL_MODELS, 'mamba_2_8b', 'Model')
const codestralMamba = byId(ALL_MODELS, 'codestral_mamba_7b', 'Model')
const mistralSmall4 = byId(ALL_MODELS, 'mistral_small_4', 'Model')
const kimiK2 = byId(ALL_MODELS, 'kimi_k2', 'Model')
const kimiK25 = byId(ALL_MODELS, 'kimi_k2_5', 'Model')
const lfm2 = byId(ALL_MODELS, 'liquidai_lfm2_8b', 'Model')
const graniteMoe = byId(ALL_MODELS, 'granite_moe_3b', 'Model')
const openMoe = byId(ALL_MODELS, 'openmoe_34b', 'Model')
const yuan51 = byId(ALL_MODELS, 'yuan2_51b', 'Model')
const kimiK15 = byId(ALL_MODELS, 'kimi_k1_5', 'Model')
const int4 = byId(QUANT_MAP, 'int4', 'Quantization')
const int6 = byId(QUANT_MAP, 'int6', 'Quantization')
const bf16 = byId(QUANT_MAP, 'bf16', 'Quantization')
const fp8 = byId(QUANT_MAP, 'fp8', 'Quantization')
const fp32 = byId(QUANT_MAP, 'fp32', 'Quantization')
const theory = byId(FRAMEWORK_MAP, 'theory', 'Framework')
const vllm = byId(FRAMEWORK_MAP, 'vllm', 'Framework')
const sglang = byId(FRAMEWORK_MAP, 'sglang', 'Framework')
const llamaCpp = byId(FRAMEWORK_MAP, 'llamacpp', 'Framework')
const mlx = byId(FRAMEWORK_MAP, 'mlx', 'Framework')
const lmdeploy = byId(FRAMEWORK_MAP, 'lmdeploy', 'Framework')
const tgi = byId(FRAMEWORK_MAP, 'tgi', 'Framework')
const pcie4 = byId(INTERCONNECT_MAP, 'pcie4', 'Interconnect')
const fp16Kv = byId(KV_CACHE_MAP, 'fp16', 'KV cache')
const int4Kv = byId(KV_CACHE_MAP, 'int4', 'KV cache')
const pcie5Host = byId(PCIE_BW_OPTIONS, 'gen5', 'PCIe host link')
const pcieX16 = byId(PCIE_WIDTH_OPTIONS, 'x16', 'PCIe width')

assert(
  new Set(GPU_LIST.map(gpu => gpu.id)).size === GPU_LIST.length,
  'GPU catalog contains duplicate IDs',
)

// China-market memory modifications keep the parent GPU die, so compute
// throughput must not increase with VRAM capacity. Bandwidth is independently
// checked from each seller-advertised data rate and bus width.
const modifiedGpuCases = [
  {
    gpu: rtx4090Mod,
    parent: rtx4090,
    vram: 48,
    dataRate: 21,
    busBits: 384,
    tdp: 450,
    nativeBf16: true,
  },
  {
    gpu: rtx4080SuperMod,
    parent: rtx4080Super,
    vram: 32,
    dataRate: 23,
    busBits: 256,
    tdp: 320,
    nativeBf16: true,
  },
  {
    gpu: rtx3080Mod,
    parent: rtx3080,
    vram: 20,
    dataRate: 19,
    busBits: 320,
    tdp: 350,
    nativeBf16: true,
  },
  {
    gpu: rtx2080TiMod,
    parent: rtx2080Ti,
    vram: 22,
    dataRate: 14,
    busBits: 352,
    tdp: 260,
    nativeBf16: false,
  },
]

for (const { gpu, parent, vram, dataRate, busBits, tdp, nativeBf16 } of modifiedGpuCases) {
  assert(gpu.modified === true && gpu.official === false, `${gpu.id}: unofficial marker regressed`)
  assert(gpu.baseGpuId === parent.id, `${gpu.id}: parent GPU link regressed`)
  assert(gpu.vram === vram, `${gpu.id}: VRAM capacity regressed`)
  assert(gpu.memoryDataRateGbps === dataRate, `${gpu.id}: memory data rate regressed`)
  assert(gpu.memoryBusBits === busBits, `${gpu.id}: memory bus width regressed`)
  approx(gpu.bw, dataRate * busBits / 8, 1e-12, `${gpu.id}: memory bandwidth`)
  assert(gpu.bf16 === parent.bf16, `${gpu.id}: BF16/FP16 throughput differs from parent die`)
  assert(gpu.int8 === parent.int8, `${gpu.id}: INT8 throughput differs from parent die`)
  assert(gpu.int4 === parent.int4, `${gpu.id}: INT4 throughput differs from parent die`)
  assert(gpu.tdp === tdp, `${gpu.id}: seller-reported TDP regressed`)
  assert(gpu.nativeBf16 === nativeBf16, `${gpu.id}: native BF16 capability regressed`)
}

const common = {
  gpu: rtx4090,
  gpuCount: 1,
  interconnect: pcie4,
  model: llama8b,
  quant: int4,
  ctx: 8192,
  batch: 1,
  promptLen: 512,
  outputLen: 128,
  framework: llamaCpp,
  flashAttention: true,
  sysRam: 128,
}

// Capacity examples used in the FAQ must include runtime reserve, KV cache,
// activations, and overhead—not just nominal cluster VRAM.
const deepseekR1OnEightH100 = calcAll({
  ...common,
  gpu: h100,
  gpuCount: 8,
  interconnect: byId(INTERCONNECT_MAP, 'nvlink4', 'Interconnect'),
  model: deepseekR1,
  quant: fp8,
  framework: theory,
})
assert(
  !deepseekR1OnEightH100.vramOk && !deepseekR1OnEightH100.fitOk,
  '8×80GB H100 incorrectly fit DeepSeek-R1 at FP8',
)
const qwen235OnFourH100 = calcAll({
  ...common,
  gpu: h100,
  gpuCount: 4,
  interconnect: byId(INTERCONNECT_MAP, 'nvlink4', 'Interconnect'),
  model: qwen3_235b,
  quant: bf16,
  framework: theory,
})
assert(
  !qwen235OnFourH100.vramOk && !qwen235OnFourH100.fitOk,
  '4×80GB H100 incorrectly fit Qwen3-235B at BF16',
)

const unofficialGpuEstimate = calcAll({ ...common, gpu: rtx4090Mod })
assert(
  unofficialGpuEstimate.unofficialGpu
    && unofficialGpuEstimate.accuracyTier === 'low'
    && getWarnings(unofficialGpuEstimate).some(warning => warning.key === 'unofficial_gpu_specs'),
  'Unofficial modified-GPU uncertainty was not surfaced',
)
const derivedAppleEstimate = calcAll({
  ...common,
  gpu: appleM4,
  quant: byId(QUANT_MAP, 'bf16', 'Quant'),
  framework: mlx,
})
assert(
  derivedAppleEstimate.derivedGpuSpecs
    && derivedAppleEstimate.accuracyTier === 'mid'
    && getWarnings(derivedAppleEstimate).some(
      warning => warning.key === 'derived_gpu_specs',
    ),
  'Derived Apple compute/power specifications were shown with high confidence',
)
const proprietaryParameterEstimate = calcAll({
  ...common,
  model: { ...llama8b, id: 'parameter_estimate_fixture', parameterEstimate: true },
})
assert(
  proprietaryParameterEstimate.parameterEstimate
    && proprietaryParameterEstimate.accuracyTier === 'low'
    && getWarnings(proprietaryParameterEstimate).some(
      warning => warning.key === 'parameter_estimate',
    ),
  'Estimated proprietary parameter count was shown with normal confidence',
)

assert(
  glm47.layers === 92
    && glm47.query_heads === 96
    && glm47.experts === 160
    && glm47.experts_per_token === 8,
  'GLM-4.7 architecture data regressed',
)
assert(
  glm47Flash.layers === 47
    && glm47Flash.kv_lora_rank === 512
    && glm47Flash.experts === 64,
  'GLM-4.7-Flash architecture data regressed',
)
assert(
  mistralLarge3.layers === 61
    && mistralLarge3.experts === 128
    && mistralLarge3.kv_lora_rank === 512
    && mistralLarge3.qk_nope_head_dim === 128
    && mistralLarge3.v_head_dim === 128,
  'Mistral Large 3 architecture data regressed',
)
assert(
  mistralSmall4.max_ctx === 1048576
    && mistralSmall4.qk_nope_head_dim === 64
    && mistralSmall4.v_head_dim === 128,
  'Mistral Small 4 architecture data regressed',
)
assert(
  kimiK2.kv_heads === 64
    && kimiK2.qk_nope_head_dim === 128
    && kimiK2.v_head_dim === 128
    && kimiK25.tags.includes('vision'),
  'Kimi K2 architecture data regressed',
)
assert(
  lfm2.linear_attention_layers === 18
    && lfm2.experts === 32
    && lfm2.kv_heads === 8,
  'LFM2 architecture data regressed',
)
assert(
  graniteMoe.layers === 32
    && graniteMoe.experts === 40
    && graniteMoe.experts_per_token === 8,
  'Granite MoE architecture data regressed',
)
assert(
  openMoe.layers === 32
    && openMoe.experts === 32
    && openMoe.query_heads === 24,
  'OpenMoE architecture data regressed',
)
assert(
  yuan51.type === 'dense'
    && yuan51.hidden_size === 8192
    && yuan51.query_heads === 64,
  'Yuan 2.0 51B architecture data regressed',
)

// DDR rate is a data rate (MT/s). A 64-bit channel transfers 8 bytes, and the
// estimator intentionally models a conventional two-channel memory bus.
const ddr3 = createCpuMemBwOption('ddr3', 1600)
const ddr4 = createCpuMemBwOption('ddr4', 3200)
const ddr5 = createCpuMemBwOption('ddr5', 4800)
assert(ddr3 && ddr4 && ddr5, 'DDR fixtures could not be created')
approx(ddr3.bw, 25.6, 1e-12, 'Dual-channel DDR3-1600 bandwidth')
approx(ddr4.bw, 51.2, 1e-12, 'Dual-channel DDR4-3200 bandwidth')
approx(ddr5.bw, 76.8, 1e-12, 'Dual-channel DDR5-4800 bandwidth')
approx(calcCpuMemTheoreticalBandwidth(6400, 2), 102.4, 1e-12, 'DDR5-6400 bandwidth')

const cpuDdr3 = calcAll({ ...common, pureCpu: true, cpuMemBw: ddr3 })
const cpuDdr4 = calcAll({ ...common, pureCpu: true, cpuMemBw: ddr4 })
const cpuDdr5 = calcAll({ ...common, pureCpu: true, cpuMemBw: ddr5 })
const cpuDefaultDdr = calcAll({ ...common, pureCpu: true, cpuMemBw: null })
approx(cpuDdr4.singleToks / cpuDdr3.singleToks, 2, 1e-10, 'DDR4/DDR3 CPU decode scaling')
approx(cpuDdr5.singleToks / cpuDdr3.singleToks, 3, 1e-10, 'DDR5/DDR3 CPU decode scaling')
approx(cpuDefaultDdr.cpuMemBwGBs, 76.8, 1e-12, 'Default CPU bandwidth')
assert(cpuDdr5.cpuMemBwLabel === 'DDR5-4800', 'Selected DDR label was not propagated')

// CPU-only calculations must not depend on whichever display GPU remains in
// the selector while the CPU mode is active.
const cpuOnAppleSelection = calcAll({
  ...common,
  gpu: appleM4,
  pureCpu: true,
  cpuMemBw: ddr5,
})
approx(
  cpuOnAppleSelection.singleToks,
  cpuDdr5.singleToks,
  1e-12,
  'CPU-only speed changed with selected GPU',
)
approx(
  cpuOnAppleSelection.cpuRamNeededGB,
  cpuDdr5.cpuRamNeededGB,
  1e-12,
  'CPU-only memory changed with selected GPU',
)

// A llama.cpp hybrid with zero GPU layers is physically the same decode path
// as CPU-only inference. GPU-specific llama.cpp calibration must not leak into
// the host-memory portion of a hybrid run.
const hybridAllCpu = calcAll({
  ...common,
  cpuOffload: true,
  pureCpu: false,
  cpuMemBw: ddr5,
  nglCount: 0,
})
approx(
  hybridAllCpu.singleToks,
  cpuDdr5.singleToks,
  1e-12,
  'Zero-layer llama.cpp hybrid does not match CPU-only decode',
)

// A full-layer llama.cpp "hybrid" is exactly the ordinary all-GPU path. NGL
// must not erase GPU compute or collective work when no layers remain on CPU.
const normalAllGpu = calcAll({ ...common, cpuOffload: false })
const hybridAllGpu = calcAll({
  ...common,
  cpuOffload: true,
  cpuMemBw: ddr5,
  nglCount: llama8b.layers,
})
approx(hybridAllGpu.perCardNeeded, normalAllGpu.perCardNeeded, 1e-12, 'All-GPU NGL memory')
approx(hybridAllGpu.singleToks, normalAllGpu.singleToks, 1e-12, 'All-GPU NGL decode')
approx(hybridAllGpu.prefillToks, normalAllGpu.prefillToks, 1e-9, 'All-GPU NGL Prefill')
approx(
  hybridAllGpu.decodeComputeLimit,
  normalAllGpu.decodeComputeLimit,
  1e-9,
  'All-GPU NGL decode compute',
)
assert(!hybridAllGpu.cpuComputeIsUpperBound, 'All-GPU NGL was marked as a CPU upper bound')

// CPU TFLOPS are optional. Without them, CPU-backed throughput is explicitly a
// bandwidth-only upper bound; with them, the CPU compute roof is enforced.
assert(
  cpuDdr5.cpuDecodeIsUpperBound && cpuDdr5.cpuPrefillIsUpperBound,
  'Missing CPU compute was not marked as an upper bound',
)
const cpuComputeLimited = calcAll({
  ...common,
  pureCpu: true,
  cpuMemBw: ddr5,
  cpuTflops: 0.001,
})
assert(cpuComputeLimited.cpuComputeProvided, 'Explicit CPU TFLOPS were not accepted')
assert(
  !cpuComputeLimited.cpuComputeIsUpperBound,
  'Explicit CPU TFLOPS remained marked as an upper bound',
)
assert(
  cpuComputeLimited.singleToks < cpuDdr5.singleToks,
  'CPU compute roof did not constrain decode',
)
assert(
  Number.isFinite(cpuComputeLimited.decodeComputeLimit),
  'CPU decode compute ceiling was not surfaced',
)
const unusedCpuCompute = calcAll({
  ...common,
  cpuTflops: 2.5,
})
assert(
  !unusedCpuCompute.cpuComputeProvided && unusedCpuCompute.cpuTflops == null,
  'GPU-only inference reported an unused CPU compute value as active',
)

// Conventional integrated GPUs share the selected DDR pool. Their effective
// bandwidth is capped by that DDR speed, and their user allocation cannot
// exceed the same usable system-RAM capacity.
const sharedGpuFixture = {
  ...h100,
  id: 'shared_gpu_fixture',
  vram: 0,
  bw: 200,
  bwUtilization: 1,
  usableRatio: 1,
  sharedMemory: true,
}
const sharedGpuFit = calcAll({
  ...common,
  gpu: sharedGpuFixture,
  framework: theory,
  cpuMemBw: ddr4,
  sysRam: 32,
})
approx(sharedGpuFit.effectiveGpuBwPerCard, 51.2, 1e-12, 'Shared-GPU DDR bandwidth cap')
assert(sharedGpuFit.fitOk, 'Valid shared-memory integrated GPU did not fit')
assert(
  sharedGpuFit.sharedSystemRamNeededGB > 0
    && sharedGpuFit.cpuRamNeededGB === 0,
  'Shared model residence was duplicated into the CPU-offload pool',
)
approx(sharedGpuFit.sharedAllocationGB, 28.8, 1e-12, 'Default shared-pool allocation')
approx(sharedGpuFit.sharedPoolAvailableGB, 28.8, 1e-12, 'Effective shared-pool capacity')
assert(
  !sharedGpuFit.cpuComputeProvided && sharedGpuFit.cpuTflops == null,
  'Shared-GPU inference incorrectly reported CPU compute as active',
)
const invalidSharedAllocation = calcAll({
  ...common,
  gpu: { ...sharedGpuFixture, vram: 64 },
  framework: theory,
  cpuMemBw: ddr4,
  sysRam: 32,
})
assert(
  !invalidSharedAllocation.ramOk && !invalidSharedAllocation.fitOk,
  'Shared-GPU allocation larger than usable system RAM passed fit',
)
assert(
  invalidSharedAllocation.sharedAllocationExceedsRam
    && invalidSharedAllocation.sharedAllocationGB === 64
    && invalidSharedAllocation.sharedAllocationExcessGB > 0,
  'Shared allocation overflow was not surfaced explicitly',
)
const invalidSharedWarningKeys = getWarnings(invalidSharedAllocation)
  .filter(warning => warning.level === 'error')
  .map(warning => warning.key)
assert(
  invalidSharedWarningKeys.includes('shared_allocation_exceeds_ram')
    && !invalidSharedWarningKeys.includes('vram_oom')
    && !invalidSharedWarningKeys.includes('cpu_ram_oom'),
  `Shared allocation overflow emitted misleading duplicate OOMs: ${invalidSharedWarningKeys.join(', ')}`,
)
const multipliedSharedDevice = calcAll({
  ...common,
  gpu: { ...sharedGpuFixture, vram: 16 },
  gpuCount: 2,
  framework: theory,
  cpuMemBw: ddr4,
  sysRam: 64,
})
assert(
  !multipliedSharedDevice.memoryDeviceTopologyOk
    && !multipliedSharedDevice.topologyOk
    && !multipliedSharedDevice.fitOk,
  'Shared-memory hardware was incorrectly stackable as multiple devices',
)
const mixedSharedGpu = aggregateGpuSlots([
  { gpu: { ...sharedGpuFixture, vram: 16 }, count: 1 },
  { gpu: h100, count: 1 },
])
const mixedSharedResult = calcAll({
  ...common,
  gpu: mixedSharedGpu,
  gpuCount: 2,
  framework: theory,
  cpuMemBw: ddr4,
  sysRam: 64,
})
assert(
  mixedSharedGpu.sharedMemory
    && mixedSharedGpu.invalidMemoryMix
    && !mixedSharedResult.memoryDeviceTopologyOk
    && !mixedSharedResult.fitOk,
  'Mixed aggregation silently erased shared-memory device semantics',
)
const multipliedUnifiedDevice = calcAll({
  ...common,
  gpu: appleM4,
  gpuCount: 2,
  quant: bf16,
  framework: mlx,
})
assert(
  !multipliedUnifiedDevice.memoryDeviceTopologyOk
    && !multipliedUnifiedDevice.topologyOk
    && !multipliedUnifiedDevice.fitOk,
  'Unified-memory hardware was incorrectly stackable as multiple devices',
)
for (const unifiedGpu of [ryzenAiMax395, ryzenAiMax395Windows]) {
  const multipliedUnifiedResult = calcAll({
    ...common,
    gpu: unifiedGpu,
    gpuCount: 2,
    quant: bf16,
    framework: theory,
  })
  assert(
    unifiedGpu.unifiedMemory
      && !multipliedUnifiedResult.memoryDeviceTopologyOk
      && !multipliedUnifiedResult.topologyOk
      && !multipliedUnifiedResult.fitOk,
    `${unifiedGpu.id}: unified-memory system was incorrectly stackable`,
  )
}

// System RAM reserves 10% and is a hard fit gate for every CPU-backed mode.
const cpuBf16SmallRam = calcAll({
  ...common,
  quant: bf16,
  pureCpu: true,
  cpuMemBw: ddr5,
  sysRam: 16,
})
const cpuBf16LargeRam = calcAll({
  ...common,
  quant: bf16,
  pureCpu: true,
  cpuMemBw: ddr5,
  sysRam: 32,
})
approx(cpuBf16SmallRam.availableSysRamGB, 14.4, 1e-12, 'System RAM reserve')
assert(!cpuBf16SmallRam.ramOk && !cpuBf16SmallRam.fitOk, 'RAM OOM did not fail fit')
assert(cpuBf16LargeRam.ramOk && cpuBf16LargeRam.fitOk, 'Valid CPU RAM configuration failed fit')

// MoE expert offload is constrained by both host DDR and PCIe. A faster link
// cannot make the host read experts faster than installed memory can supply.
const offloadBase = {
  ...common,
  model: mixtral,
  cpuOffload: true,
  pcieBw: pcie5Host,
  pcieWidth: pcieX16,
  sysRam: 128,
}
const offloadDdr3 = calcAll({ ...offloadBase, cpuMemBw: ddr3 })
const offloadDdr5Fast = calcAll({
  ...offloadBase,
  cpuMemBw: createCpuMemBwOption('ddr5', 8000),
})
approx(offloadDdr3.offloadTransferBw, 25.6 * 0.85, 1e-12, 'DDR-capped offload link')
approx(offloadDdr5Fast.offloadTransferBw, 64 * 0.85, 1e-12, 'PCIe-capped offload link')
assert(offloadDdr5Fast.singleToks > offloadDdr3.singleToks, 'DDR speed did not affect MoE offload')
assert(offloadDdr3.fitOk && offloadDdr5Fast.fitOk, 'Valid MoE offload fixture did not fit')

const offloadTwoGpu = calcAll({
  ...offloadBase,
  gpuCount: 2,
  cpuMemBw: createCpuMemBwOption('ddr5', 16000),
})
approx(
  offloadTwoGpu.offloadTransferBw,
  2 * 64 * 0.85,
  1e-12,
  'Tensor-parallel PCIe links were not aggregated',
)

const offloadUnknownRam = calcAll({
  ...offloadBase,
  cpuMemBw: ddr5,
  sysRam: null,
})
assert(!offloadUnknownRam.ramOk && !offloadUnknownRam.fitOk, 'Unknown offload RAM passed fit')

// Fit is a conjunction of memory, context, topology, runtime compatibility,
// and supported cache format—not a VRAM-only answer.
const badContext = calcAll({
  ...common,
  ctx: 4096,
  promptLen: 4000,
  outputLen: 200,
})
assert(!badContext.workloadContextOk && !badContext.fitOk, 'Oversized workload passed context fit')

const badWorkloadInput = calcAll({
  ...common,
  batch: 0,
})
assert(
  !badWorkloadInput.workloadInputOk && !badWorkloadInput.fitOk,
  'Invalid workload input was silently normalized into a runnable result',
)

const badTopology = calcAll({
  ...common,
  gpu: h100,
  gpuCount: 3,
  ppCount: 2,
  framework: vllm,
})
assert(!badTopology.topologyOk && !badTopology.fitOk, 'Non-divisible PP topology passed fit')

const tooManyPipelineStages = calcAll({
  ...common,
  gpu: h100,
  gpuCount: 4,
  ppCount: 4,
  model: {
    ...llama8b,
    id: 'two_layer_pipeline_fixture',
    layers: 2,
  },
  framework: theory,
})
assert(
  !tooManyPipelineStages.topologyOk && !tooManyPipelineStages.fitOk,
  'Pipeline topology admitted empty stages beyond the decoder layer count',
)

const badFramework = calcAll({ ...common, framework: mlx })
assert(!badFramework.frameworkOk && !badFramework.fitOk, 'Unsupported GPU/framework pair passed fit')

const badKvCache = calcAll({
  ...common,
  gpu: h100,
  framework: vllm,
  kvCacheQuant: int4Kv,
})
assert(!badKvCache.kvCacheSupported && !badKvCache.fitOk, 'Unsupported KV cache passed fit')

const badSpeculativeRuntime = calcAll({
  ...common,
  speculativeDecoding: true,
})
assert(
  !badSpeculativeRuntime.speculativeOk && !badSpeculativeRuntime.fitOk,
  'Unsupported speculative runtime passed fit',
)

const sglangSpeculative = calcAll({
  ...common,
  gpu: h100,
  quant: bf16,
  framework: sglang,
  speculativeDecoding: true,
  draftModelParams: 1,
})
assert(
  sglangSpeculative.speculativeOk
    && sglangSpeculative.runtimeConfigurationSupported
    && sglangSpeculative.fitOk,
  'SGLang speculative decoding was rejected by stale calculator-only rules',
)

const unsupportedVllmGguf = calcAll({
  ...common,
  gpu: h100,
  framework: vllm,
  quant: int6,
})
assert(
  !unsupportedVllmGguf.weightQuantSupported
    && !unsupportedVllmGguf.runtimeConfigurationSupported
    && !unsupportedVllmGguf.fitOk,
  'A llama.cpp-only GGUF quantization passed vLLM runtime fit',
)

const badDenseOffload = calcAll({
  ...common,
  framework: vllm,
  cpuOffload: true,
  cpuMemBw: ddr5,
})
assert(!badDenseOffload.denseOffloadOk && !badDenseOffload.fitOk, 'Unsupported dense offload passed fit')

const badOffloadEp = calcAll({
  ...offloadBase,
  gpu: h100,
  gpuCount: 2,
  epCount: 2,
  cpuMemBw: ddr5,
})
assert(!badOffloadEp.offloadParallelOk && !badOffloadEp.fitOk, 'Offload + EP passed fit')

const badOffloadPp = calcAll({
  ...offloadBase,
  gpu: h100,
  gpuCount: 2,
  ppCount: 2,
  cpuMemBw: ddr5,
})
assert(!badOffloadPp.offloadParallelOk && !badOffloadPp.fitOk, 'Offload + PP passed fit')

const badUnifiedOffload = calcAll({
  ...offloadBase,
  gpu: appleM4,
  model: mixtral,
  framework: mlx,
  cpuMemBw: ddr5,
})
assert(
  !badUnifiedOffload.offloadMemoryArchitectureOk && !badUnifiedOffload.fitOk,
  'Unified-memory CPU offload passed fit',
)

const warningKeys = new Set(getWarnings(badKvCache).map(warning => warning.key))
assert(warningKeys.has('kv_cache_unsupported'), 'KV compatibility warning is missing')

// Latency identities are exact within the estimator's own definitions.
const gpuResult = calcAll({
  ...common,
  gpu: h100,
  framework: vllm,
  quant: bf16,
})
approx(gpuResult.effectiveTpot, 1000 / gpuResult.singleToks, 1e-9, 'TPOT reciprocal')
approx(gpuResult.ttft, gpuResult.prefillLatency + gpuResult.effectiveTpot, 1e-9, 'TTFT identity')
approx(
  gpuResult.totalLatency,
  gpuResult.ttft + (common.outputLen - 1) * gpuResult.effectiveTpot,
  1e-8,
  'Total latency identity',
)
const llamaKvWritePerTokenGB = 2 * 2 * llama8b.layers * llama8b.kv_heads * llama8b.head_dim / 1e9
approx(
  gpuResult.kvTrafficGB,
  llamaKvWritePerTokenGB * (gpuResult.avgDecodeSeqLen + 1),
  1e-12,
  'Decode KV traffic omitted the new-token write',
)
assert(gpuResult.effectiveToks <= gpuResult.bwLimit * (1 + 1e-10), 'Decode exceeded BW ceiling')
assert(
  gpuResult.decodeComputeLimit == null
    || gpuResult.effectiveToks <= gpuResult.decodeComputeLimit * (1 + 1e-10),
  'Decode exceeded compute ceiling',
)
assert(
  gpuResult.computeLimit == null
    || gpuResult.prefillToks <= gpuResult.computeLimit * (1 + 1e-10),
  'Prefill exceeded raw compute ceiling',
)

// Decode and Prefill can use different hardware paths for the same weight
// format. INT4 decode uses INT4 throughput, while this catalog intentionally
// models INT4 Prefill through the BF16 path.
const computeFixtureGpu = {
  ...h100,
  id: 'compute_fixture',
  bw: 1e9,
  bwUtilization: 1,
  fp32: 2,
  bf16: 10,
  int4: 100,
}
const bf16ComputePath = calcAll({
  ...common,
  gpu: computeFixtureGpu,
  framework: theory,
  quant: bf16,
  ctx: 1024,
  promptLen: 512,
  outputLen: 128,
})
const int4ComputePath = calcAll({
  ...common,
  gpu: computeFixtureGpu,
  framework: theory,
  quant: int4,
  ctx: 1024,
  promptLen: 512,
  outputLen: 128,
})
const fp32ComputePath = calcAll({
  ...common,
  gpu: computeFixtureGpu,
  framework: theory,
  quant: fp32,
  ctx: 1024,
  promptLen: 512,
  outputLen: 128,
})
approx(
  int4ComputePath.decodeComputeLimit / bf16ComputePath.decodeComputeLimit,
  10,
  1e-10,
  'Decode used the Prefill FLOPS path',
)
approx(
  int4ComputePath.computeLimit / bf16ComputePath.computeLimit,
  1,
  1e-10,
  'INT4 Prefill did not use its configured BF16 path',
)
approx(
  fp32ComputePath.decodeComputeLimit / bf16ComputePath.decodeComputeLimit,
  0.2,
  1e-10,
  'FP32 decode ignored the GPU FP32 throughput',
)
approx(
  fp32ComputePath.computeLimit / bf16ComputePath.computeLimit,
  0.2,
  1e-10,
  'FP32 Prefill ignored the GPU FP32 throughput',
)

// Missing FP32 data is unsupported, not an invitation to silently substitute
// half of BF16 throughput.
const unsupportedFp32 = calcAll({
  ...common,
  gpu: t4,
  framework: theory,
  quant: fp32,
  ctx: 1024,
  promptLen: 512,
  outputLen: 128,
})
assert(
  !unsupportedFp32.computePrecisionSupported && !unsupportedFp32.fitOk,
  'GPU without explicit FP32 throughput was treated as supported',
)
assertFiniteResult(unsupportedFp32, 'Unsupported FP32 result')
assert(
  getWarnings(unsupportedFp32).some(warning => warning.key === 'compute_precision_unsupported'),
  'Unsupported FP32 warning is missing',
)

const mlaAttentionFixture = {
  ...llama8b,
  id: 'mla_attention_fixture',
  params: 0.001,
  layers: 1,
  query_heads: 1,
  kv_heads: 1,
  head_dim: 128,
  hidden_size: 128,
  kv_lora_rank: 64,
  qk_nope_head_dim: 128,
  qk_rope_head_dim: 64,
  v_head_dim: 128,
  max_ctx: 131072,
}
const mlaAttentionResult = calcAll({
  ...common,
  gpu: computeFixtureGpu,
  model: mlaAttentionFixture,
  framework: theory,
  quant: bf16,
  ctx: 131072,
  promptLen: 65536,
  outputLen: 2,
})
const expectedMlaDecodeFlops = 2 * mlaAttentionFixture.params * 1e9
  + 2
    * mlaAttentionFixture.query_heads
    * (
      mlaAttentionFixture.qk_nope_head_dim
      + mlaAttentionFixture.qk_rope_head_dim
      + mlaAttentionFixture.v_head_dim
    )
    * mlaAttentionResult.avgDecodeSeqLen
approx(
  mlaAttentionResult.decodeComputeLimit,
  computeFixtureGpu.bf16 * 1e12 / expectedMlaDecodeFlops,
  1e-6,
  'MLA Decode attention dimensions',
)

const cachedPrefill = calcAll({
  ...common,
  gpu: h100,
  framework: vllm,
  prefixCacheHit: 90,
})
assert(gpuResult.prefillKvWriteGB > 0, 'Prefill KV writes were not counted')
assert(cachedPrefill.prefillKvWriteGB < gpuResult.prefillKvWriteGB, 'Prefix cache did not reduce KV writes')

// Prefix hits remove queries from Prefill, but the cached prefix remains
// attended. Non-Flash attention therefore uses new_queries Ã— all_keys, not
// new_queries squared.
const prefixActivation = calcAll({
  ...common,
  gpu: h100,
  framework: theory,
  quant: bf16,
  ctx: 8192,
  promptLen: 8191,
  outputLen: 1,
  prefixCacheHit: 90,
  flashAttention: false,
})
const prefixQueries = 8191 - Math.round(8191 * 0.9)
const expectedPrefixActivationGB = (
  prefixQueries * llama8b.hidden_size * 2 * 8
  + (llama8b.hidden_size / llama8b.head_dim) * prefixQueries * 8191 * 2
) / 1e9
approx(
  prefixActivation.activationGB,
  expectedPrefixActivationGB,
  1e-12,
  'Prefix-cache attention working set',
)

// Pure recurrent models have no quadratic softmax score matrix, regardless of
// the Flash Attention toggle.
const recurrentFlash = calcAll({
  ...common,
  gpu: h100,
  model: mamba28,
  framework: theory,
  ctx: 8192,
  promptLen: 8191,
  outputLen: 1,
  flashAttention: true,
})
const recurrentNoFlash = calcAll({
  ...common,
  gpu: h100,
  model: mamba28,
  framework: theory,
  ctx: 8192,
  promptLen: 8191,
  outputLen: 1,
  flashAttention: false,
})
approx(
  recurrentNoFlash.activationGB,
  recurrentFlash.activationGB,
  1e-12,
  'Recurrent model allocated a quadratic attention score buffer',
)

// Long Prefill is executed in chunks. Every chunk rereads dense weights and
// pays collective startup latency, even though payload bytes remain linear in
// the total token count.
const chunkedPrefillModel = {
  ...llama8b,
  id: 'chunked_prefill_fixture',
  prefill_chunk_size: 2048,
}
const chunkedPrefill = calcAll({
  ...common,
  gpu: h100,
  gpuCount: 2,
  model: chunkedPrefillModel,
  framework: theory,
  ctx: 16384,
  promptLen: 8192,
  outputLen: 1,
})
assert(chunkedPrefill.prefillWeightPasses === 4, 'Prefill chunk count is wrong')
approx(
  chunkedPrefill.prefillWeightReadGB,
  llama8b.params * getQuantBytes(int4, h100, theory) * 4,
  1e-12,
  'Prefill did not reread dense weights per chunk',
)
const chunkedTpRingBytes = 2 * (2 - 1) / 2
  * llama8b.hidden_size
  * 8192
  * 2
const expectedChunkedTpSeconds = 2 * llama8b.layers * (
  chunkedTpRingBytes / (pcie4.bw * 1e9)
  + 1e-6 * 2 * (2 - 1) * 4
)
approx(
  chunkedPrefill.prefillCommunicationSeconds,
  expectedChunkedTpSeconds,
  1e-15,
  'Prefill collective startup was not paid per chunk',
)

// Sliding-window caches still write one K/V entry per uncached token after the
// cache is full; capacity growth alone would incorrectly report zero traffic.
const localAttentionModel = {
  ...llama8b,
  id: 'local_attention_fixture',
  layers: 2,
  local_layers: 2,
  sliding_window: 128,
  max_ctx: 4096,
}
const saturatedLocalPrefill = calcAll({
  ...common,
  gpu: h100,
  model: localAttentionModel,
  framework: vllm,
  ctx: 1024,
  promptLen: 512,
  outputLen: 128,
  prefixCacheHit: 90,
})
const localKvWritePerTokenGB = 2 * 2
  * localAttentionModel.layers
  * localAttentionModel.kv_heads
  * localAttentionModel.head_dim
  / 1e9
approx(
  saturatedLocalPrefill.prefillKvWriteGB,
  localKvWritePerTokenGB * saturatedLocalPrefill.effectivePromptLen,
  1e-12,
  'Sliding-window Prefill KV overwrite traffic',
)

function clippedCausalAverage(cachedTokens, newTokens, window) {
  let sum = 0
  for (let index = 1; index <= newTokens; index += 1) {
    sum += Math.min(cachedTokens + index, window)
  }
  return sum / newTokens
}

const clippedLocalDecode = calcAll({
  ...common,
  gpu: h100,
  model: localAttentionModel,
  framework: theory,
  ctx: 512,
  promptLen: 64,
  outputLen: 128,
})
approx(
  clippedLocalDecode.avgLocalDecodeSeqLen,
  clippedCausalAverage(63, 128, 128),
  1e-12,
  'Exact clipped sliding-window Decode mean',
)
const clippedLocalPrefill = calcAll({
  ...common,
  gpu: h100,
  model: localAttentionModel,
  framework: theory,
  ctx: 512,
  promptLen: 200,
  outputLen: 1,
  prefixCacheHit: 50,
})
approx(
  clippedLocalPrefill.prefillAverageLocalAttendedTokens,
  clippedCausalAverage(100, 100, 128),
  1e-12,
  'Exact clipped sliding-window Prefill mean',
)

// Gemma 4 uses different head counts/dimensions for global and local layers.
// Each cache class must be sized and TP-sharded independently.
const gemmaKv = calcAll({
  ...common,
  gpu: h100,
  gpuCount: 16,
  model: gemma431,
  framework: theory,
  quant: bf16,
  ctx: 131072,
  promptLen: 128,
  outputLen: 64,
  kvCacheQuant: fp16Kv,
})
const gemmaGlobalKvGB = 2 * 1 * 2
  * (gemma431.layers - gemma431.local_layers)
  * gemma431.global_kv_heads
  * gemma431.global_head_dim
  * 131072
  / 1e9
const gemmaLocalKvGB = 2 * 1 * 2
  * gemma431.local_layers
  * gemma431.kv_heads
  * gemma431.head_dim
  * gemma431.sliding_window
  / 1e9
approx(gemmaKv.targetGlobalSequenceKvGB, gemmaGlobalKvGB, 1e-12, 'Gemma global KV')
approx(gemmaKv.targetLocalSequenceKvGB, gemmaLocalKvGB, 1e-12, 'Gemma local KV')
approx(
  gemmaKv.perCardKvGB,
  gemmaGlobalKvGB / gemma431.global_kv_heads
    + gemmaLocalKvGB / gemma431.kv_heads,
  1e-12,
  'Gemma global/local KV sharding',
)

// Contiguous PP stages must use the actual mixed-attention pattern. With a
// 5-local/1-global Gemma pattern and PP4, the 15-layer stages alternate between
// 13L+2G and 12L+3G; the global-heavy stage limits long-context KV capacity.
const patternedGemmaModel = {
  ...gemma431,
  id: 'patterned_gemma_pp_fixture',
  layer_pattern: ['local', 'local', 'local', 'local', 'local', 'global'],
}
const patternedGemmaPp = calcAll({
  ...common,
  gpu: h100,
  gpuCount: 4,
  model: patternedGemmaModel,
  framework: theory,
  quant: bf16,
  ctx: 131072,
  promptLen: 128,
  outputLen: 64,
  ppCount: 4,
  kvCacheQuant: fp16Kv,
})
assert(
  patternedGemmaPp.ppComponentLayoutExact
    && !patternedGemmaPp.ppComponentLayoutConservative,
  'Explicit mixed-attention PP pattern was not used',
)
assert(
  patternedGemmaPp.perCardKvStageGlobalLayers === 3
    && patternedGemmaPp.perCardKvStageLocalLayers === 12,
  'Wrong mixed-attention stage selected for KV capacity',
)
approx(
  patternedGemmaPp.perCardGlobalSequenceKvGB,
  gemmaGlobalKvGB * 3 / 10,
  1e-12,
  'Patterned PP global KV capacity',
)
approx(
  patternedGemmaPp.perCardLocalSequenceKvGB,
  gemmaLocalKvGB * 12 / 50,
  1e-12,
  'Patterned PP local KV capacity',
)
const gemmaTrafficStages = [
  { global: 2, local: 13 },
  { global: 3, local: 12 },
  { global: 2, local: 13 },
  { global: 3, local: 12 },
]
const limitingGemmaTrafficStage = gemmaTrafficStages.reduce((largest, stage) => {
  const traffic = patternedGemmaPp.logicalGlobalSequenceKvTrafficGB * stage.global / 10
    + patternedGemmaPp.logicalLocalSequenceKvTrafficGB * stage.local / 50
  return traffic > largest.traffic ? { ...stage, traffic } : largest
}, { global: 0, local: 0, traffic: -Infinity })
approx(
  patternedGemmaPp.pipelineAggregateGpuKvTrafficGB,
  limitingGemmaTrafficStage.traffic * 4,
  1e-12,
  'Patterned PP component Decode traffic',
)
approx(
  patternedGemmaPp.pipelineAggregateGpuGlobalSequenceKvTrafficGB,
  patternedGemmaPp.logicalGlobalSequenceKvTrafficGB
    * limitingGemmaTrafficStage.global
    / 10
    * 4,
  1e-12,
  'Patterned PP global Decode traffic',
)
approx(
  patternedGemmaPp.pipelineAggregateGpuLocalSequenceKvTrafficGB,
  patternedGemmaPp.logicalLocalSequenceKvTrafficGB
    * limitingGemmaTrafficStage.local
    / 50
    * 4,
  1e-12,
  'Patterned PP local Decode traffic',
)

// If only aggregate counts are known, independently ceiling each component by
// the largest stage. This is deliberately conservative and explicitly flagged.
const conservativeGemmaPp = calcAll({
  ...common,
  gpu: h100,
  gpuCount: 4,
  model: { ...gemma431, id: 'count_only_gemma_pp_fixture' },
  framework: theory,
  quant: bf16,
  ctx: 131072,
  promptLen: 128,
  outputLen: 64,
  ppCount: 4,
  kvCacheQuant: fp16Kv,
})
assert(
  conservativeGemmaPp.ppComponentLayoutConservative,
  'Count-only mixed PP layout was not marked conservative',
)
approx(
  conservativeGemmaPp.perCardGlobalSequenceKvGB,
  gemmaGlobalKvGB,
  1e-12,
  'Conservative PP global-component ceiling',
)
approx(
  conservativeGemmaPp.perCardLocalSequenceKvGB,
  gemmaLocalKvGB * 15 / 50,
  1e-12,
  'Conservative PP local-component ceiling',
)
assert(
  conservativeGemmaPp.perCardKvGB >= patternedGemmaPp.perCardKvGB,
  'Count-only PP ceiling understated the known patterned layout',
)

// Recurrent state follows its explicit value-head sharding limit, rather than
// falling back to ordinary attention/query heads.
const codestralMambaTp64 = calcAll({
  ...common,
  gpu: h100,
  gpuCount: 64,
  model: codestralMamba,
  framework: theory,
  quant: bf16,
  ctx: 4096,
  promptLen: 256,
  outputLen: 64,
})
assert(
  codestralMamba.linear_num_value_heads === 128
    && codestralMambaTp64.recurrentKvShardCount === 64,
  'Explicit recurrent value-head sharding limit was ignored',
)
approx(
  codestralMambaTp64.perCardRecurrentKvGB,
  codestralMambaTp64.targetRecurrentKvGB / 64,
  1e-12,
  'Codestral Mamba recurrent-state sharding',
)

// PP uses the selected physical GPU count for memory, compute, and power, and
// single-microbatch Prefill must pay a pipeline-fill bubble.
const ppResult = calcAll({
  ...common,
  gpu: h100,
  gpuCount: 4,
  model: llama70b,
  framework: vllm,
  ppCount: 2,
})
assert(
  ppResult.totalGpuCount === 4
    && ppResult.stageGpuCount === 2
    && ppResult.tpCount === 2,
  'PP topology fields are inconsistent',
)
approx(ppResult.totalPower, h100.tdp * 4 / 1000, 1e-12, 'Cluster power')
assert(ppResult.ppBubbleEff < 1, 'Decode PP bubble was not applied')
assert(ppResult.prefillPpBubbleEff < 1, 'Prefill PP bubble was not applied')

const slowEpInterconnect = { id: 'slow_ep_fixture', label: 'Slow EP', bw: 1, scope: 'intra' }
const ep2Communication = calcAll({
  ...common,
  gpu: h100,
  gpuCount: 2,
  model: mixtral,
  framework: theory,
  epCount: 2,
  interconnect: slowEpInterconnect,
})
const ep4Communication = calcAll({
  ...common,
  gpu: h100,
  gpuCount: 4,
  model: mixtral,
  framework: theory,
  epCount: 4,
  interconnect: slowEpInterconnect,
})
assert(
  ep4Communication.epEfficiency < ep2Communication.epEfficiency,
  'EP all-to-all payload incorrectly shrank as more remote ranks were added',
)

// Uneven PP partitions are constrained by the largest integer stage, not by a
// fractional layers/stage average. 61 layers over four stages means 16 layers
// on the limiting card.
const unevenPp = calcAll({
  ...common,
  gpu: h200,
  gpuCount: 4,
  model: mistralLarge3,
  quant: int6,
  framework: theory,
  ctx: 1024,
  promptLen: 128,
  outputLen: 64,
  ppCount: 4,
})
const unevenPpFactor = 16 / (mistralLarge3.layers / 4)
assert(
  unevenPp.largestStageLayers === 16,
  'Uneven PP did not select the largest integer stage',
)
approx(unevenPp.ppStageImbalance, unevenPpFactor, 1e-12, 'Uneven PP stage factor')
approx(
  unevenPp.perCardWeightGB,
  mistralLarge3.params * int6.bytes / 4 * unevenPpFactor,
  1e-12,
  'Uneven PP largest-stage weight memory',
)
assert(
  unevenPp.perCardNeeded > h200.vram && !unevenPp.vramOk,
  'Fractional PP average produced a false H200 fit',
)

// EP shards experts but replicates non-expert weights. The memory-bandwidth
// ceiling must reflect those replicated reads even when per-card storage fits.
const ep1 = calcAll({
  ...common,
  gpu: h100,
  gpuCount: 4,
  model: mixtral,
  framework: vllm,
  ctx: 4096,
  epCount: 1,
})
const ep2 = calcAll({
  ...common,
  gpu: h100,
  gpuCount: 4,
  model: mixtral,
  framework: vllm,
  ctx: 4096,
  epCount: 2,
})
assert(ep2.bwLimit < ep1.bwLimit, 'EP replicated dense-weight reads were not counted')
assert(ep2.perCardNeeded < mixtral.params * getQuantBytes(int4, h100, vllm), 'EP did not shard storage')

for (const model of [glm47, glm47Flash, mistralLarge3, lfm2, graniteMoe, openMoe]) {
  const result = calcAll({
    ...common,
    gpu: h100,
    gpuCount: 2,
    model,
    framework: vllm,
    ctx: Math.min(4096, model.max_ctx),
    promptLen: 256,
    outputLen: 64,
    epCount: 2,
  })
  assert(result.modelDataOk, `${model.id}: corrected MoE decomposition was rejected`)
}
const uncertainKimiEp = calcAll({
  ...common,
  gpu: h100,
  gpuCount: 2,
  model: kimiK15,
  framework: vllm,
  epCount: 2,
})
assert(
  !uncertainKimiEp.modelDataOk && !uncertainKimiEp.fitOk,
  'Unpublished Kimi MoE decomposition was treated as exact',
)

// Explicit MLA cache dimensions: 61 × (512 latent + 64 RoPE) × 131072
// tokens × 2 bytes = 9.210691584 GB.
const deepseekKv = calcAll({
  ...common,
  gpu: h100,
  model: deepseekV3,
  framework: vllm,
  quant: bf16,
  ctx: 131072,
  kvCacheQuant: fp16Kv,
})
approx(deepseekKv.targetKvGB, 9.210691584, 1e-9, 'DeepSeek V3 MLA cache')
assert(!deepseekKv.mlaApproximate, 'Explicit MLA model was marked approximate')
const deepseekTp8 = calcAll({
  ...common,
  gpu: h100,
  gpuCount: 8,
  model: deepseekV3,
  framework: vllm,
  quant: bf16,
  ctx: 131072,
  kvCacheQuant: fp16Kv,
})
approx(
  deepseekTp8.perCardKvGB,
  deepseekKv.targetKvGB,
  1e-9,
  'MLA cache was incorrectly sharded by plain tensor parallelism',
)
approx(
  deepseekTp8.aggregateGpuKvTrafficGB,
  deepseekTp8.logicalKvTrafficGB * 8,
  1e-9,
  'MLA tensor-parallel cache replication traffic',
)

// Qwen hybrid recurrent state remains FP32 when the ordinary attention KV
// cache is quantized.
const qwenFp16 = calcAll({
  ...common,
  gpu: h100,
  model: qwen36,
  framework: theory,
  ctx: 4096,
  kvCacheQuant: fp16Kv,
})
const qwenInt4 = calcAll({
  ...common,
  gpu: h100,
  model: qwen36,
  framework: theory,
  ctx: 4096,
  kvCacheQuant: int4Kv,
})
const qwenStateGB = 48 * (
  48 * 128 * 128
  + (2 * 16 * 128 + 48 * 128) * 4
) * 4 / 1e9
const qwenSequenceFp16GB = 2 * 16 * 4 * 256 * 4096 * 2 / 1e9
const qwenSequenceInt4GB = 2 * 16 * 4 * 256 * 4096 * 0.5 / 1e9
approx(qwenFp16.targetKvGB, qwenStateGB + qwenSequenceFp16GB, 1e-12, 'Qwen FP16 cache')
approx(qwenInt4.targetKvGB, qwenStateGB + qwenSequenceInt4GB, 1e-12, 'Qwen INT4 cache')
const qwenTp8 = calcAll({
  ...common,
  gpu: h100,
  gpuCount: 8,
  model: qwen36,
  framework: theory,
  ctx: 4096,
  kvCacheQuant: fp16Kv,
})
approx(
  qwenTp8.perCardKvGB,
  qwenSequenceFp16GB / 4 + qwenStateGB / 8,
  1e-12,
  'Hybrid-attention KV/state sharding',
)

// Jamba 1.5 Large has 1 attention layer per 8 layers (9 attention + 63
// Mamba layers); recurrent state is fixed-size while attention cache scales.
const jambaKv = calcAll({
  ...common,
  gpu: h100,
  model: jambaLarge,
  framework: theory,
  ctx: 4096,
  kvCacheQuant: fp16Kv,
})
const jambaSequenceGB = 2 * 9 * 8 * 128 * 4096 * 2 / 1e9
const jambaStateGB = 63 * 8192 * 2 * (16 + 4) * 4 / 1e9
approx(jambaKv.targetKvGB, jambaSequenceGB + jambaStateGB, 1e-12, 'Jamba hybrid cache')

// VLM weights reside once, decoder FLOPs use the explicit text-only count, and
// each image pays a separate encoder Prefill pass.
const visionAccountingModel = {
  ...llama8b,
  id: 'vision_accounting_fixture',
  params: 12,
  text_params: 10,
  vision_encoder_params: 2,
  params_scope: 'total',
  vision_seq_tokens: 100,
  tags: ['vision'],
}
const visionAccounting = calcAll({
  ...common,
  gpu: h100,
  model: visionAccountingModel,
  framework: theory,
  imageCount: 2,
})
const visionQuantBytes = getQuantBytes(int4, h100, theory)
approx(visionAccounting.targetWeightGB, 12 * visionQuantBytes, 1e-12, 'VLM resident weights')
approx(
  visionAccounting.prefillWeightReadGB,
  10 * visionQuantBytes,
  1e-12,
  'VLM decoder-only Prefill weight read',
)
approx(
  visionAccounting.visionEncoderWeightReadGB,
  2 * visionQuantBytes * 2,
  1e-12,
  'VLM per-image encoder weight read',
)
approx(
  visionAccounting.visionEncoderFlops,
  2 * 2 * 1e9 * 100 * 2,
  1e-3,
  'VLM encoder Prefill FLOPs',
)
assert(
  !visionAccounting.visionAccountingApproximate,
  'Explicit VLM parameter split was marked approximate',
)
const pipelineVisionAccounting = calcAll({
  ...common,
  gpu: h100,
  gpuCount: 4,
  ppCount: 2,
  model: visionAccountingModel,
  framework: theory,
  imageCount: 1,
})
approx(
  pipelineVisionAccounting.perCardWeightGB,
  (
    visionAccountingModel.text_params / pipelineVisionAccounting.tpCount / 2
    + visionAccountingModel.vision_encoder_params / pipelineVisionAccounting.tpCount
  ) * visionQuantBytes,
  1e-12,
  'PP incorrectly divided non-decoder VLM weights across decoder stages',
)
const inferredVisionSplit = calcAll({
  ...common,
  gpu: h100,
  model: {
    ...visionAccountingModel,
    id: 'vision_inferred_split_fixture',
    text_params: undefined,
  },
  framework: theory,
  imageCount: 1,
})
approx(
  inferredVisionSplit.decoderModelParams,
  10,
  1e-12,
  'Declared-total VLM decoder inference',
)
approx(
  inferredVisionSplit.targetWeightGB,
  12 * visionQuantBytes,
  1e-12,
  'Inferred VLM split double-counted resident encoder weights',
)
assert(
  inferredVisionSplit.visionAccountingApproximate,
  'Inferred VLM parameter split was not marked approximate',
)

// Speculative decoding includes draft weights/KV, target verification,
// accepted-token yield, and draft Prefill. It is allowed to be a slowdown.
const normalDecode = calcAll({
  ...common,
  gpu: h100,
  framework: vllm,
})
const slowSpeculative = calcAll({
  ...common,
  gpu: h100,
  framework: vllm,
  speculativeDecoding: true,
  acceptanceRate: 0.3,
  draftLen: 8,
  draftModelParams: 7,
})
assert(slowSpeculative.weightGB > normalDecode.weightGB, 'Draft weights were not allocated')
assert(slowSpeculative.kvGB > normalDecode.kvGB, 'Draft KV cache was not allocated')
assert(slowSpeculative.ttft > normalDecode.ttft, 'Draft Prefill was not included in TTFT')
assert(slowSpeculative.speculativeSpeedup < 1, 'Slow speculative fixture was forced to a speedup')
assert(
  slowSpeculative.effectiveToks <= slowSpeculative.bwLimit * (1 + 1e-10),
  'Speculative decode exceeded its BW ceiling',
)
approx(
  slowSpeculative.expectedAcceptedTokens,
  (1 - 0.3 ** 9) / (1 - 0.3),
  1e-12,
  'Expected accepted speculative tokens',
)

// Breakdown components use one consistent display scope and do not double
// count activation memory inside runtime overhead.
const gpuBreakdown = getDisplayVramBreakdown(gpuResult)
approx(
  Object.values(gpuBreakdown).reduce((sum, value) => sum + value, 0),
  gpuResult.displayNeeded,
  1e-9,
  'GPU memory breakdown total',
)
const cpuBreakdown = getDisplayVramBreakdown(cpuDdr5)
approx(
  Object.values(cpuBreakdown).reduce((sum, value) => sum + value, 0),
  cpuDdr5.cpuRamNeededGB,
  1e-9,
  'CPU memory breakdown total',
)

approx(getQuantBytes(int4, rtx4090, llamaCpp), 0.615, 1e-12, 'GGUF INT4 size')
approx(getQuantBytes(int4, rtx4090, vllm), 0.55, 1e-12, 'AWQ/GPTQ INT4 size')

const commandConfig = {
  model: llama8b,
  gpuCount: 1,
  ppCount: 1,
  epCount: 1,
  ctx: 8192,
  batch: 1,
  quant: int4,
  kvCacheQuant: int4Kv,
  prefixCacheHit: 0,
  speculativeDecoding: false,
  draftLen: 4,
  cpuOffload: false,
  pureCpu: false,
  nglCount: null,
}
const llamaCommand = generateCmd(llamaCpp, commandConfig)
assert(
  llamaCommand.includes('--cache-type-k q4_0')
    && llamaCommand.includes('--cache-type-v q4_0'),
  'llama.cpp command ignored INT4 KV cache',
)
const lmdeployUnsupportedCommand = generateCmd(lmdeploy, commandConfig)
assert(
  lmdeployUnsupportedCommand === null
    && getCommandCompatibility(lmdeploy, commandConfig).reasons.some(
      item => item.code === 'quantized-checkpoint-required',
    ),
  'LMDeploy fabricated an AWQ command from a base BF16 repository',
)
const lmdeployCommand = generateCmd(lmdeploy, {
  ...commandConfig,
  quant: bf16,
})
assert(
  lmdeployCommand.includes('lmdeploy serve api_server')
    && lmdeployCommand.includes('--quant-policy 4')
    && !lmdeployCommand.includes('--cache-quant-policy'),
  'LMDeploy BF16 command ignored INT4 KV cache',
)
assert(
  isKvCacheSupported(lmdeploy, byId(KV_CACHE_MAP, 'fp8', 'KV cache')),
  'LMDeploy rejected its supported FP8 cache policy',
)
const lmdeployFp8Command = generateCmd(lmdeploy, {
  ...commandConfig,
  quant: bf16,
  kvCacheQuant: byId(KV_CACHE_MAP, 'fp8', 'KV cache'),
})
assert(
  lmdeployFp8Command.includes('--quant-policy fp8'),
  'LMDeploy command ignored FP8 KV cache',
)
const tgiCommand = generateCmd(tgi, {
  ...commandConfig,
  quant: bf16,
  kvCacheQuant: byId(KV_CACHE_MAP, 'fp8', 'KV cache'),
})
assert(tgiCommand.includes('--kv-cache-dtype fp8_e4m3fn'), 'TGI command ignored FP8 KV cache')

const mixedAmpere = aggregateGpuSlots([
  { gpu: rtx3090, count: 2 },
  { gpu: rtx3080Mod, count: 2 },
])
assert(
  mixedAmpere.mixedGpu
    && mixedAmpere.mixedGpuEstimateSupported
    && !mixedAmpere.mixedVendors
    && !mixedAmpere.mixedArchitectures
    && mixedAmpere.vendor === 'nvidia'
    && mixedAmpere.architecture === 'ampere',
  'Compatible mixed Ampere GPUs were not classified as estimable',
)
assert(
  mixedAmpere.vram === 20
    && mixedAmpere.bw === rtx3080Mod.bw * rtx3080Mod.bwUtilization
    && mixedAmpere.bf16 === rtx3080Mod.bf16,
  'Mixed Ampere estimate did not use the smallest/slowest card limits',
)
assert(
  mixedAmpere.modified && mixedAmpere.official === false,
  'Mixed aggregation erased modified-GPU confidence metadata',
)
const mixedAmpereEstimate = calcAll({
  ...common,
  gpu: mixedAmpere,
  gpuCount: 4,
  quant: bf16,
  framework: vllm,
})
assert(
  mixedAmpereEstimate.runtimeConfigurationSupported
    && mixedAmpereEstimate.frameworkOk
    && mixedAmpereEstimate.heterogeneousGpuEstimate
    && mixedAmpereEstimate.accuracyTier === 'low'
    && getWarnings(mixedAmpereEstimate).some(warning => warning.key === 'mixed_gpu_estimate')
    && getWarnings(mixedAmpereEstimate).some(warning => warning.key === 'unofficial_gpu_specs'),
  'Compatible mixed Ampere runtime estimate was not enabled with low-confidence warnings',
)
const mixedAmpereCommandConfig = {
  model: llama8b,
  gpu: mixedAmpere,
  gpuCount: 4,
  ppCount: 1,
  epCount: 1,
  ctx: common.ctx,
  batch: common.batch,
  promptLen: common.promptLen,
  outputLen: common.outputLen,
  quant: bf16,
  kvCacheQuant: byId(KV_CACHE_MAP, 'auto', 'KV cache'),
  cpuOffload: false,
  pureCpu: false,
  speculativeDecoding: false,
}
const mixedAmpereCommandCompatibility = getCommandCompatibility(
  vllm,
  mixedAmpereCommandConfig,
)
assert(
  !mixedAmpereCommandCompatibility.supported
    && mixedAmpereCommandCompatibility.reasons.some(
      reason => reason.code === 'mixed-gpu-command-unsupported',
    )
    && generateCmd(vllm, mixedAmpereCommandConfig) === null,
  'Mixed-GPU estimation incorrectly generated a topology-specific launch command',
)

const mixedArchitecture = aggregateGpuSlots([
  { gpu: rtx3090, count: 1 },
  { gpu: rtx4090, count: 1 },
])
const mixedArchitectureEstimate = calcAll({
  ...common,
  gpu: mixedArchitecture,
  gpuCount: 2,
  quant: bf16,
  framework: vllm,
})
assert(
  mixedArchitecture.mixedArchitectures
    && !mixedArchitecture.mixedGpuEstimateSupported
    && !mixedArchitectureEstimate.runtimeConfigurationSupported
    && mixedArchitectureEstimate.runtimeCompatibilityReasonCodes.includes(
      'mixed-gpu-architecture-unsupported',
    ),
  'Mixed architecture families were incorrectly accepted as one runtime estimate',
)

const mixed = aggregateGpuSlots([
  {
    gpu: {
      id: 'fast',
      name: 'Fast',
      vendor: 'nvidia',
      tier: 'consumer',
      vram: 24,
      bw: 100,
      bwUtilization: 0.8,
      usableRatio: 0.9,
      bf16: 80,
      fp8: 160,
      int8: 160,
      int4: 320,
      tdp: 400,
    },
    count: 1,
  },
  {
    gpu: {
      id: 'slow',
      name: 'Slow',
      vendor: 'nvidia',
      tier: 'consumer',
      vram: 16,
      bw: 50,
      bwUtilization: 0.5,
      usableRatio: 0.8,
      bf16: 40,
      fp8: 80,
      int8: 80,
      int4: 160,
      tdp: 200,
    },
    count: 1,
  },
])
approx(mixed.bw, 25, 1e-12, 'Mixed-GPU straggler bandwidth')
approx(mixed.bf16, 40, 1e-12, 'Mixed-GPU straggler compute')
approx(mixed.tdp, 300, 1e-12, 'Mixed-GPU average TDP')
assert(mixed.vram === 16 && mixed.usableRatio === 0.8, 'Mixed-GPU memory limits are wrong')

const crossedMemoryLimits = aggregateGpuSlots([
  {
    gpu: {
      id: 'large_low_reserve',
      name: 'Large low reserve',
      vendor: 'nvidia',
      vram: 24,
      usableRatio: 0.5,
      bw: 100,
      bwUtilization: 1,
      bf16: 10,
      fp8: 10,
      int8: 10,
      int4: 10,
      tdp: 1,
    },
    count: 1,
  },
  {
    gpu: {
      id: 'small_high_reserve',
      name: 'Small high reserve',
      vendor: 'nvidia',
      vram: 16,
      usableRatio: 0.9,
      bw: 100,
      bwUtilization: 1,
      bf16: 10,
      fp8: 10,
      int8: 10,
      int4: 10,
      tdp: 1,
    },
    count: 1,
  },
])
approx(
  crossedMemoryLimits.vram * crossedMemoryLimits.usableRatio,
  Math.min(24 * 0.5, 16 * 0.9),
  1e-12,
  'Mixed-GPU aggregation combined VRAM and reserve limits from different cards',
)

const invalidSweep = calcBatchSweep({
  ...common,
  framework: mlx,
}, [1, 2])
assert(invalidSweep.every(point => !point.fitOk), 'Batch sweep ignored full fit status')

// Broad catalog sweep: malformed or incomplete data must never leak NaN or
// Infinity into charts, exports, rankings, or solver comparisons.
let catalogCases = 0
for (const model of ALL_MODELS) {
  const maxContext = Number.isFinite(Number(model.max_ctx)) ? Number(model.max_ctx) : 4096
  const testContext = Math.max(128, Math.min(4096, maxContext))
  const testPrompt = Math.max(1, Math.min(256, testContext - 2))
  const testOutput = Math.max(1, Math.min(64, testContext - testPrompt))
  const result = calcAll({
    gpu: h100,
    gpuCount: 1,
    interconnect: pcie4,
    model,
    quant: bf16,
    ctx: testContext,
    batch: 1,
    promptLen: testPrompt,
    outputLen: testOutput,
    framework: theory,
    flashAttention: true,
  })
  assertFiniteResult(result, `Model ${model.id}`)
  assert(typeof result.fitOk === 'boolean', `Model ${model.id}: fitOk is not boolean`)
  assert(
    result.singleToksMin <= result.singleToks
      && result.singleToks <= result.singleToksMax,
    `Model ${model.id}: decode uncertainty range is not ordered`,
  )
  assert(
    result.effectiveToks <= result.bwLimit * (1 + 1e-10),
    `Model ${model.id}: decode exceeded its bandwidth ceiling`,
  )
  if (result.decodeComputeLimit != null) {
    assert(
      result.effectiveToks <= result.decodeComputeLimit * (1 + 1e-10),
      `Model ${model.id}: decode exceeded its compute ceiling`,
    )
  }
  const largerBatch = calcAll({
    gpu: h100,
    gpuCount: 1,
    interconnect: pcie4,
    model,
    quant: bf16,
    ctx: testContext,
    batch: 2,
    promptLen: testPrompt,
    outputLen: testOutput,
    framework: theory,
    flashAttention: true,
  })
  assert(
    largerBatch.targetKvGB >= result.targetKvGB,
    `Model ${model.id}: KV capacity fell when batch size increased`,
  )
  catalogCases += 1
}

for (const gpu of GPU_LIST) {
  const result = calcAll({
    ...common,
    gpu,
    framework: theory,
    ctx: 1024,
    promptLen: 256,
    outputLen: 64,
  })
  assertFiniteResult(result, `GPU ${gpu.id}`)
  catalogCases += 1
}

console.log(`Calculation regression passed: ${catalogCases} catalog cases plus focused invariants`)
