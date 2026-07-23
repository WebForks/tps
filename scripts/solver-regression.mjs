import { ALL_MODELS } from '../src/data/models/index.js'
import { GPU_LIST } from '../src/data/gpus/index.js'
import { FRAMEWORK_MAP, INTERCONNECT_MAP, QUANT_MAP } from '../src/data/constants.js'
import {
  KV_CACHE_MAP,
  PCIE_WIDTH_OPTIONS,
  createCpuMemBwOption,
  resolveCpuMemBwOption,
} from '../src/data/runtime.js'
import { autoInterconnect, solveForModel, solveUpgrade } from '../src/utils/solver.js'

function assert(condition, message) {
  if (!condition) throw new Error(message)
}

const model = ALL_MODELS.find(item => item.id === 'llama3_8b')
assert(model, 'Llama 3 8B test model is missing')

const common = {
  model,
  maxGpuCount: 1,
  vendorFilter: 'nvidia',
  excludeDatacenterGpu: true,
  ctx: 4096,
  batch: 1,
  promptLen: 512,
  outputLen: 256,
  disableYield: true,
}

const allowedQuantsByFloor = {
  bf16: ['bf16'],
  int8: ['bf16', 'fp8', 'int8'],
  int4: ['bf16', 'fp8', 'int8', 'int6', 'int5', 'int4'],
}

const requiredQuantsByFloor = {
  bf16: ['bf16'],
  int8: ['bf16', 'int8'],
  int4: ['bf16', 'int8', 'int6', 'int5', 'int4'],
}

for (const [floor, allowedIds] of Object.entries(allowedQuantsByFloor)) {
  const outcome = await solveForModel({ ...common, quantFloor: floor })
  const actualIds = [...new Set(outcome.results.map(row => row.quant.id))].sort()
  assert(
    actualIds.length > 0 && actualIds.every(id => allowedIds.includes(id)),
    `${floor} quant floor admitted an unsupported format: ${actualIds.join(', ')}`,
  )
  assert(
    requiredQuantsByFloor[floor].every(id => actualIds.includes(id)),
    `${floor} quant floor omitted a supported format: ${actualIds.join(', ')}`,
  )
}

const defaultRuntimeBudgetOutcome = await solveForModel({
  ...common,
  quantFloor: 'bf16',
})
assert(
  defaultRuntimeBudgetOutcome.results.length > 0
    && defaultRuntimeBudgetOutcome.results.every(row =>
      ['trtllm', 'sglang', 'vllm', 'lmdeploy', 'tgi'].includes(row.framework.id)
        ? row.gpu.usableRatio === 0.9
        : true
    ),
  'Solver did not apply each server runtime’s default GPU-memory budget',
)

const appleOutcome = await solveForModel({
  ...common,
  maxGpuCount: 8,
  vendorFilter: 'apple',
  excludeDatacenterGpu: false,
  quantFloor: 'none',
})
assert(appleOutcome.results.length > 0, 'Apple solver test returned no results')
assert(
  appleOutcome.results.every(row =>
    row.gpuCount === 1 &&
    row.totalGpuCount === 1 &&
    row.ppCount === 1
  ),
  'Apple solver returned a multi-device configuration',
)

const sharedMemoryBandwidth = resolveCpuMemBwOption('ddr5_4800')
const sharedMemoryOutcome = await solveForModel({
  ...common,
  maxGpuCount: 8,
  vendorFilter: 'amd',
  excludeDatacenterGpu: true,
  quantFloor: 'int4',
  cpuMemBw: sharedMemoryBandwidth,
  sysRam: 64,
})
const sharedMemoryRows = sharedMemoryOutcome.results.filter(row => row.gpu.sharedMemory)
assert(sharedMemoryRows.length > 0, 'Shared-memory GPUs were dropped from solver results')
assert(
  sharedMemoryRows.every(row =>
    row.gpuCount === 1 &&
    row.gpu.bw <= sharedMemoryBandwidth.bw &&
    row.gpu.sharedVram === 57 &&
    row.gpu.catalogVram === 0
  ),
  'Shared-memory GPU solver results did not preserve their reproducible RAM allocation/bandwidth',
)

const modelContextOutcome = await solveForModel({
  ...common,
  ctx: model.max_ctx + 1,
})
assert(
  modelContextOutcome.validationError?.code === 'context_exceeds_model_limit',
  'Model context limit was not validated',
)

const workloadContextOutcome = await solveForModel({
  ...common,
  promptLen: 4000,
  outputLen: 200,
})
assert(
  workloadContextOutcome.validationError?.code === 'workload_exceeds_context',
  'Prompt plus output context was not validated',
)

const fractionalWorkloadOutcome = await solveForModel({
  ...common,
  batch: 1.5,
})
assert(
  fractionalWorkloadOutcome.validationError?.code === 'invalid_workload',
  'Solver accepted a fractional token/batch workload that calcAll cannot run',
)

const apiOnlyModel = ALL_MODELS.find(item => item.localInference === false)
assert(apiOnlyModel, 'API-only model fixture is missing')
const apiOnlyOutcome = await solveForModel({ ...common, model: apiOnlyModel })
assert(
  apiOnlyOutcome.validationError?.code === 'model_local_inference_unsupported',
  'Solver recommended local hardware for an API-only model',
)

const h800 = GPU_LIST.find(gpu => gpu.id === 'h800')
const h800Interconnect = autoInterconnect(h800, 2)
assert(
  h800Interconnect?.id === 'nvlink_400' &&
  h800Interconnect.bw === 200 &&
  h800Interconnect.duplexBw === 400,
  'Solver did not derive NVLink bandwidth from the selected GPU metadata',
)
const pcie3Gpu = GPU_LIST.find(gpu => gpu.pcie_gen === 3)
assert(autoInterconnect(pcie3Gpu, 1)?.id === 'pcie3', 'PCIe 3 GPU did not use PCIe 3')
const aggregateSystem = GPU_LIST.find(gpu => gpu.unitKind === 'system')
assert(
  autoInterconnect(aggregateSystem, 2)?.scope === 'inter',
  'Aggregate systems incorrectly reused their internal NVLink between systems',
)

const appleGpu = GPU_LIST.find(gpu => gpu.vendor === 'apple')
const appleUpgradeOutcome = await solveUpgrade({
  currentGpu: appleGpu,
  currentGpuCount: 2,
  currentQuant: QUANT_MAP.find(quant => quant.id === 'int8'),
  model,
  targetSpeed: 1,
  ctx: 4096,
  batch: 1,
  promptLen: 512,
  outputLen: 256,
})
assert(
  appleUpgradeOutcome.validationError?.code === 'apple_multi_device_unsupported',
  'Apple upgrade solver accepted multiple devices',
)

const nvidiaUpgradeOutcome = await solveUpgrade({
  currentGpu: GPU_LIST.find(gpu => gpu.id === 'rtx4090'),
  currentGpuCount: 1,
  currentQuant: QUANT_MAP.find(quant => quant.id === 'int4'),
  model,
  targetSpeed: 1,
  ctx: 4096,
  batch: 1,
  promptLen: 512,
  outputLen: 256,
})
assert(nvidiaUpgradeOutcome.results.length > 0, 'NVIDIA upgrade solver returned no results')
assert(
  nvidiaUpgradeOutcome.results.every(row =>
    row.gpuCount === row.totalGpuCount &&
    row.stageGpuCount === row.totalGpuCount &&
    row.tpCount === row.totalGpuCount &&
    row.ppCount === 1 &&
    row.epCount === 1
  ),
  'Upgrade solver emitted inconsistent topology fields',
)
assert(
  nvidiaUpgradeOutcome.results.every(row =>
    row.costMultiplier == null &&
    Number.isFinite(row.relativeCapacity) &&
    Array.isArray(row.insightKeys)
  ),
  'Upgrade solver still presents VRAM capacity as monetary cost or emits untranslated insight text',
)

const rtx3090 = GPU_LIST.find(gpu => gpu.id === 'rtx3090')
const rtx4090 = GPU_LIST.find(gpu => gpu.id === 'rtx4090')
const bf16 = QUANT_MAP.find(quant => quant.id === 'bf16')
const sameVramUpgrade = await solveUpgrade({
  currentGpu: rtx3090,
  currentGpuCount: 1,
  currentQuant: bf16,
  model,
  targetSpeed: 1,
  ctx: 4096,
  batch: 1,
  promptLen: 512,
  outputLen: 256,
})
assert(
  sameVramUpgrade.results.some(row => row.type === 'upgrade_gpu' && row.gpu.id === rtx4090.id),
  'Upgrade solver omitted a faster same-VRAM replacement',
)
assert(
  sameVramUpgrade.results.some(row =>
    row.type === 'upgrade_quant' && row.quant.bytes < bf16.bytes
  ),
  'Upgrade solver omitted performance-improving lower precision',
)

const mixedUpgrade = await solveUpgrade({
  gpuSlots: [
    { gpu: rtx3090, count: 1 },
    { gpu: rtx4090, count: 1 },
  ],
  currentGpu: rtx3090,
  currentGpuCount: 1,
  currentQuant: bf16,
  currentFramework: FRAMEWORK_MAP.find(framework => framework.id === 'theory'),
  currentInterconnect: INTERCONNECT_MAP.find(option => option.id === 'pcie4'),
  model,
  targetSpeed: 1,
  ctx: 4096,
  batch: 1,
  promptLen: 512,
  outputLen: 256,
  kvCacheQuant: KV_CACHE_MAP.find(option => option.id === 'int8'),
  prefixCacheHit: 25,
  pcieWidth: PCIE_WIDTH_OPTIONS.find(option => option.id === 'x16'),
  cpuMemBw: createCpuMemBwOption('ddr4', 3200, 2, 88.5),
  cpuTflops: 12.5,
  gpuMemoryUtilization: 0.8,
  sysRam: 128,
})
assert(mixedUpgrade.results.length > 0, 'Mixed-GPU upgrade solver returned no results')
assert(
  mixedUpgrade.results
    .filter(row => row.type === 'upgrade_quant')
    .every(row => (
      row.gpuSlots?.length === 2 &&
      row.gpuSlots[0].gpu.id === 'rtx3090' &&
      row.gpuSlots[1].gpu.id === 'rtx4090'
    )),
  'Upgrade solver dropped current GPU slots from quantization recommendations',
)
assert(
  mixedUpgrade.results.every(row =>
    row.runtime?.kvCacheQuant?.id === 'int8' &&
    row.runtime?.prefixCacheHit === 25 &&
    row.runtime?.pcieWidth?.id === 'x16' &&
    row.runtime?.cpuMemBw?.measuredBw === 88.5 &&
    row.runtime?.cpuTflops === 12.5 &&
    row.runtime?.gpuMemoryUtilization === 0.8 &&
    row.runtime?.sysRam === 128
  ),
  'Upgrade solver did not preserve runtime/KV/PCIe/RAM settings',
)

const pureCpuUpgrade = await solveUpgrade({
  currentGpu: rtx3090,
  currentGpuCount: 1,
  currentQuant: bf16,
  currentFramework: FRAMEWORK_MAP.find(framework => framework.id === 'llamacpp'),
  model,
  targetSpeed: 0.1,
  ctx: 4096,
  batch: 1,
  promptLen: 512,
  outputLen: 256,
  pureCpu: true,
  cpuMemBw: resolveCpuMemBwOption('ddr5_4800'),
  sysRam: 64,
})
assert(pureCpuUpgrade.results.length > 0, 'Pure-CPU upgrade solver returned no quantization options')
assert(
  pureCpuUpgrade.results.every(row => row.type === 'upgrade_quant' && row.runtime?.pureCpu === true),
  'Pure-CPU upgrade solver recommended irrelevant GPU hardware changes',
)

const parallelModel = ALL_MODELS.find(item => item.id === 'llama3_70b')
const parallelOutcome = await solveForModel({
  ...common,
  model: parallelModel,
  maxGpuCount: 4,
  excludeDatacenterGpu: false,
  quantFloor: 'none',
})
assert(
  parallelOutcome.results.some(row => row.ppCount > 1),
  'Parallel solver test returned no PP configurations',
)
assert(
  parallelOutcome.results.every(row =>
    row.gpuCount === row.totalGpuCount &&
    row.stageGpuCount === row.totalGpuCount / row.ppCount &&
    row.tpCount === row.stageGpuCount / row.epCount &&
    row.totalGpuCount % row.ppCount === 0 &&
    row.stageGpuCount % row.epCount === 0
  ),
  'Parallel solver emitted an invalid GPU/PP/EP topology',
)

const moeModel = ALL_MODELS.find(item => item.id === 'mixtral_8x7b')
const moeOutcome = await solveForModel({
  ...common,
  model: moeModel,
  maxGpuCount: 8,
  excludeDatacenterGpu: false,
  quantFloor: 'int4',
})
assert(
  moeOutcome.results.some(row => row.epCount > 1),
  'MoE solver test returned no EP configurations',
)
assert(
  moeOutcome.results.every(row =>
    row.stageGpuCount === row.totalGpuCount / row.ppCount &&
    row.tpCount === row.stageGpuCount / row.epCount &&
    row.stageGpuCount % row.epCount === 0 &&
    moeModel.experts % row.epCount === 0
  ),
  'MoE solver emitted an invalid PP/EP/TP topology',
)

const paretoOutcome = await solveForModel({
  ...common,
  maxGpuCount: 8,
  vendorFilter: 'all',
  excludeDatacenterGpu: false,
  quantFloor: 'none',
})
assert(paretoOutcome.results.length > 2000, 'Pareto regression set did not exceed 2,000 rows')
assert(
  paretoOutcome.results.every(row => !['system', 'cpu'].includes(row.gpu?.unitKind)),
  'Solver enumerated aggregate systems or CPU catalog entries as individual GPUs',
)

function dominates(candidate, row) {
  const candidateGpuCount = candidate.totalGpuCount ?? candidate.gpuCount
  const rowGpuCount = row.totalGpuCount ?? row.gpuCount
  return (
    candidate.decodeSpeed >= row.decodeSpeed &&
    candidate.vramNeeded <= row.vramNeeded &&
    candidateGpuCount <= rowGpuCount &&
    (
      candidate.decodeSpeed > row.decodeSpeed ||
      candidate.vramNeeded < row.vramNeeded ||
      candidateGpuCount < rowGpuCount
    )
  )
}

for (let index = 0; index < paretoOutcome.results.length; index++) {
  const row = paretoOutcome.results[index]
  const expectedPareto = !paretoOutcome.results.some(
    (candidate, candidateIndex) => candidateIndex !== index && dominates(candidate, row),
  )
  assert(row.isPareto === expectedPareto, `Incorrect Pareto label at result ${index}`)
}

console.log(
  `Solver regression passed: ${paretoOutcome.results.length} rows, ` +
  `${paretoOutcome.results.filter(row => row.isPareto).length} Pareto`,
)
