import { GPU_LIST } from '../src/data/gpus/index.js'
import { ALL_MODELS } from '../src/data/models/index.js'
import { FRAMEWORK_MAP, INTERCONNECT_MAP, QUANT_MAP } from '../src/data/constants.js'
import {
  KV_CACHE_MAP,
  PCIE_WIDTH_OPTIONS,
  createCpuMemBwOption,
} from '../src/data/runtime.js'
import { calcAll } from '../src/utils/calc.js'
import { generateMarkdown } from '../src/utils/exportMd.js'

function assert(condition, message) {
  if (!condition) throw new Error(message)
}

function byId(list, id, label) {
  const value = list.find(item => item.id === id)
  assert(value, `${label} fixture "${id}" is missing`)
  return value
}

const gpu = byId(GPU_LIST, 'rtx4090', 'GPU')
const model = byId(ALL_MODELS, 'qwen3_8b', 'Model')
const interconnect = byId(INTERCONNECT_MAP, 'pcie4', 'Interconnect')
const kvCacheQuant = byId(KV_CACHE_MAP, 'auto', 'KV cache')
const t = key => key

const vllm = byId(FRAMEWORK_MAP, 'vllm', 'Framework')
const bf16 = byId(QUANT_MAP, 'bf16', 'Quantization')
const serverGpu = { ...gpu, usableRatio: 0.9 }
const serverArgs = {
  gpu: serverGpu,
  gpuCount: 1,
  interconnect,
  model,
  quant: bf16,
  framework: vllm,
  ctx: 4096,
  batch: 1,
  promptLen: 1024,
  outputLen: 256,
  flashAttention: true,
  kvCacheQuant,
  prefixCacheHit: 0,
  cpuOffload: false,
  sysRam: 64,
}
const serverResult = calcAll(serverArgs)
const serverReport = generateMarkdown({
  ...serverArgs,
  result: serverResult,
  gpuMemoryUtilization: 0.9,
  t,
  locale: 'en',
})

assert(
  serverReport.includes('| GPU Memory Budget | 90.0% |'),
  'Markdown export must preserve the selected GPU memory utilization',
)
assert(
  /\| Q6_K \|[^]*?\| ⚠️ Unsupported by runtime \| — \|/.test(serverReport),
  'Markdown quantization comparison must distinguish unsupported runtime formats from OOM',
)

const llamaCpp = byId(FRAMEWORK_MAP, 'llamacpp', 'Framework')
const int4 = byId(QUANT_MAP, 'int4', 'Quantization')
const cpuMemBw = createCpuMemBwOption('ddr4', 3600, 4, 123.4)
const cpuArgs = {
  gpu,
  gpuCount: 1,
  interconnect,
  model,
  quant: int4,
  framework: llamaCpp,
  ctx: 4096,
  batch: 1,
  promptLen: 1024,
  outputLen: 256,
  flashAttention: true,
  kvCacheQuant,
  prefixCacheHit: 0,
  cpuOffload: false,
  pureCpu: true,
  cpuMemBw,
  cpuTflops: 2.5,
  pcieWidth: byId(PCIE_WIDTH_OPTIONS, 'x16', 'PCIe width'),
  sysRam: 128,
}
const cpuResult = calcAll(cpuArgs)
const cpuReport = generateMarkdown({
  ...cpuArgs,
  result: cpuResult,
  t,
  locale: 'en',
})

assert(
  cpuReport.includes('DDR4-3600 · 4 channels · 123.4 GB/s (measured)'),
  'Markdown export must preserve RAM generation, speed, channel count, and measured bandwidth',
)
assert(
  cpuReport.includes('| Peak CPU Compute | 2.500 TFLOPS |'),
  'Markdown export must preserve the user-provided peak CPU throughput',
)

const sharedCatalogGpu = GPU_LIST.find(item =>
  item.sharedMemory && item.unitKind !== 'cpu'
)
assert(sharedCatalogGpu, 'Shared-memory GPU fixture is missing')
const sharedGpu = {
  ...sharedCatalogGpu,
  vram: 64,
  usableRatio: 1,
}
const sharedArgs = {
  gpu: sharedGpu,
  gpuCount: 1,
  interconnect,
  model,
  quant: int4,
  framework: byId(FRAMEWORK_MAP, 'theory', 'Framework'),
  ctx: 4096,
  batch: 1,
  promptLen: 1024,
  outputLen: 256,
  flashAttention: true,
  kvCacheQuant,
  prefixCacheHit: 0,
  cpuOffload: false,
  pureCpu: false,
  cpuMemBw: createCpuMemBwOption('ddr4', 3200),
  cpuTflops: 2.5,
  sysRam: 32,
}
const sharedResult = calcAll(sharedArgs)
const sharedReport = generateMarkdown({
  ...sharedArgs,
  result: sharedResult,
  t,
  locale: 'en',
})
assert(
  sharedReport.includes('| GPU Shared-Pool Allocation | 64.0 GB |')
    && sharedReport.includes('| Usable System RAM | 28.8 GB |')
    && sharedReport.includes('| Effective Shared Pool | 28.8 GB |'),
  'Markdown export did not distinguish shared allocation from usable system RAM',
)
assert(
  sharedReport.includes('GPU shared-pool allocation exceeds usable system RAM by 35.2 GB')
    && !sharedReport.includes('Cluster Usable VRAM')
    && !sharedReport.includes('insufficient by -')
    && !sharedReport.includes('insufficient by 0.0 GB')
    && !sharedReport.includes('| Peak CPU Compute |'),
  'Shared-pool export contains misleading VRAM/RAM or unused CPU-compute claims',
)

console.log('Export regression checks passed')
