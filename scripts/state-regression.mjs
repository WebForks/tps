import { GPU_LIST } from '../src/data/gpus/index.js'
import { ALL_MODELS } from '../src/data/models/index.js'
import { FRAMEWORK_MAP, INTERCONNECT_MAP, QUANT_MAP } from '../src/data/constants.js'
import { createCpuMemBwOption, KV_CACHE_MAP } from '../src/data/runtime.js'
import { ref } from 'vue'
import { resolveUrlState, readUrlState, watchUrlState } from '../src/utils/useUrlState.js'
import { readFileSync } from 'node:fs'

function assert(condition, message) {
  if (!condition) throw new Error(message)
}

const storage = new Map()
globalThis.sessionStorage = {
  getItem: key => storage.get(key) ?? null,
  setItem: (key, value) => storage.set(key, String(value)),
}
globalThis.window = {
  location: {
    search: '?co=off&n=999&ctx=99999999&pl=99999999&ol=99999999&dl=99&img=99&gmu=0.77&ctf=12.5',
    href: 'https://example.test/?co=off&n=999&ctx=99999999&pl=99999999&ol=99999999&dl=99&img=99&gmu=0.77&ctf=12.5',
    pathname: '/',
  },
  history: {
    state: { position: 7, replaced: true },
    replaceState() {},
  },
}

const bounded = readUrlState()
assert(bounded.cpuOffloadMode === 'off', 'Explicit offload-off state was not restored')
assert(bounded.cpuOffload === false, 'Legacy boolean offload state was not preserved')
assert(bounded.gpuCount === 512, 'GPU count URL bound does not match the 512 UI maximum')
assert(bounded.ctx === 10_485_760, 'Context URL bound does not match the UI maximum')
assert(bounded.promptLen === 10_485_760, 'Prompt URL bound does not match the UI maximum')
assert(bounded.outputLen === 10_485_760, 'Output URL bound does not match the UI maximum')
assert(bounded.draftLen === 8, 'Draft length URL bound does not match the UI maximum')
assert(bounded.imageCount === 8, 'Image count URL bound does not match the UI maximum')
assert(bounded.gpuMemoryUtilization === 0.77, 'GPU memory utilization was not restored')
assert(bounded.cpuTflops === 12.5, 'CPU TFLOPS was not restored')

const auto = resolveUrlState({ cpuOffloadMode: 'auto' })
assert(auto.cpuOffloadMode === 'auto' && auto.cpuOffload === null, 'Auto offload state collapsed to a boolean')

const alias = resolveUrlState({ modelId: 'gemma_1_7b' })
assert(alias.model?.id === 'gemma_7b', 'Legacy model ID alias was not resolved')

const custom = resolveUrlState({
  modelId: 'custom',
  customModel: {
    id: 'custom',
    name: 'Restored MoE',
    type: 'moe',
    params: 42,
    active_params: 7,
    layers: 48,
    kv_heads: 8,
    head_dim: 128,
    hidden_size: 4096,
    max_ctx: 65536,
  },
  ctx: 65536,
})
assert(custom.model?.name === 'Restored MoE', 'Custom model state was not restored')
assert(custom.ctx === 65536, 'Custom model context was not restored')

window.location.search = '?model=custom&cmt=moe&cmp=42&cma=7&cmexp=8&cmk=2'
const customMoeFromUrl = resolveUrlState(readUrlState())
assert(customMoeFromUrl.model?.experts === 8, 'Custom MoE total experts were not restored')
assert(customMoeFromUrl.model?.experts_per_token === 2, 'Custom MoE experts-per-token were not restored')

window.location.search = '?model=custom&cmt=dense&cmp=7&cmexp=8&cmk=2'
const customDenseFromUrl = resolveUrlState(readUrlState())
assert(
  customDenseFromUrl.model?.experts == null
    && customDenseFromUrl.model?.experts_per_token == null,
  'Dense custom models retained MoE-only expert metadata',
)

const h800 = GPU_LIST.find(gpu => gpu.id === 'h800')
assert(h800?.nvlink_bw === 400, 'Dynamic NVLink test GPU is missing')
const dynamicNvlink = resolveUrlState({
  gpuSlots: 'h800:2',
  interconnectId: 'nvlink_400',
})
assert(dynamicNvlink.interconnect?.bw === 200, 'Dynamic NVLink one-way bandwidth was not restored')
assert(dynamicNvlink.interconnect?.duplexBw === 400, 'Dynamic NVLink duplex metadata was not restored')

const cappedSlots = resolveUrlState({
  gpuSlots: 'rtx3090:400,rtx4090:400',
})
assert(
  cappedSlots.gpuSlots.reduce((sum, slot) => sum + slot.count, 0) === 512,
  'Aggregate GPU slot count was not capped at 512',
)

const cpuMemory = resolveUrlState({
  cpuMemBwId: 'ddr4_3200',
  cpuMemChannels: 4,
  cpuMemMeasuredBw: 88.5,
})
assert(cpuMemory.cpuMemBw?.channels === 4, 'Custom CPU memory channel count was not restored')
assert(cpuMemory.cpuMemBw?.theoreticalBw === 102.4, 'CPU memory bandwidth did not account for channel count')
assert(cpuMemory.cpuMemBw?.measuredBw === 88.5, 'Measured CPU memory bandwidth was not restored')
assert(cpuMemory.cpuMemBw?.bw === 88.5, 'Measured CPU memory bandwidth was not used')

let capturedUrl = null
let capturedHistoryState = null
window.history.replaceState = (state, _title, url) => {
  capturedHistoryState = state
  capturedUrl = String(url)
}
const sharedGpu = GPU_LIST.find(gpu => gpu.sharedMemory && gpu.unitKind !== 'cpu')
const sharedState = {
  gpuSlots: ref([{ gpu: sharedGpu, count: 1 }]),
  interconnect: ref(INTERCONNECT_MAP.find(option => option.id === 'pcie4')),
  model: ref(ALL_MODELS.find(item => item.id === 'llama3_8b')),
  quant: ref(QUANT_MAP.find(option => option.id === 'int4')),
  ctx: ref(4096),
  batch: ref(1),
  promptLen: ref(512),
  outputLen: ref(256),
  framework: ref(FRAMEWORK_MAP.find(option => option.id === 'llamacpp')),
  flashAttention: ref(true),
  kvCacheQuant: ref(KV_CACHE_MAP[0]),
  prefixCacheHit: ref(0),
  cpuOffload: ref(false),
  cpuOffloadMode: ref('off'),
  pcieBw: ref(null),
  pcieWidth: ref(null),
  pureCpu: ref(false),
  cpuMemBw: ref(createCpuMemBwOption('ddr4', 3200, 4, 88.5)),
  cpuTflops: ref(12.5),
  gpuMemoryUtilization: ref(0.8),
  sysRam: ref(128),
  sharedVram: ref(16),
  speculativeDecoding: ref(false),
  acceptanceRate: ref(0.7),
  draftLen: ref(4),
  draftModelParams: ref(1),
  ppCount: ref(1),
  epCount: ref(1),
  imageCount: ref(0),
  nglCount: ref(null),
}
watchUrlState(sharedState)
const persistedShared = new URL(capturedUrl)
assert(persistedShared.searchParams.get('cmb') === 'ddr4_3200', 'Shared GPU lost RAM type/speed')
assert(persistedShared.searchParams.get('cmc') === '4', 'Shared GPU lost RAM channels')
assert(persistedShared.searchParams.get('cmm') === '88.5', 'Shared GPU lost measured RAM bandwidth')
assert(persistedShared.searchParams.get('ctf') === '12.5', 'Shared GPU lost CPU throughput')
assert(persistedShared.searchParams.get('sr') === '128', 'Shared GPU lost system RAM capacity')
assert(persistedShared.searchParams.get('gmu') === '0.8', 'GPU memory utilization was not persisted')
assert(
  capturedHistoryState === window.history.state,
  'URL synchronization overwrote Vue Router history metadata',
)

capturedUrl = null
watchUrlState({
  ...sharedState,
  gpuSlots: ref([{
    gpu: GPU_LIST.find(gpu => gpu.id === 'rtx4090'),
    count: 1,
  }]),
  cpuOffload: ref(false),
  cpuOffloadMode: ref('off'),
  nglCount: ref(5),
})
const persistedManualNgl = new URL(capturedUrl)
assert(
  persistedManualNgl.searchParams.get('ngl') === '5'
    && persistedManualNgl.searchParams.get('co') === 'off',
  'Manual llama.cpp NGL was still coupled to the MoE CPU-offload state',
)

const resultPanelSource = readFileSync(
  new URL('../src/components/result/ResultPanel.vue', import.meta.url),
  'utf8',
)
assert(
  /<VramCard[\s\S]{0,400}:readonly="readonly"/.test(resultPanelSource),
  'ResultPanel does not pass readonly state to VramCard',
)

const modelPickerSource = readFileSync(
  new URL('../src/components/config/ModelPicker.vue', import.meta.url),
  'utf8',
)
assert(
  modelPickerSource.includes('customModel.experts')
    && modelPickerSource.includes('customModel.experts_per_token')
    && modelPickerSource.includes('experts < expertsPerToken'),
  'Custom MoE expert fields or validation are missing from ModelPicker',
)

const solverPageSource = readFileSync(
  new URL('../src/pages/Solver.vue', import.meta.url),
  'utf8',
)
assert(
  solverPageSource.includes('targetSpeed = ref(clampInt(_p.target, LIMITS.targetSpeed))')
    && solverPageSource.includes('maxGpuCount = ref(clampInt(_p.maxg, LIMITS.maxGpuCount))')
    && solverPageSource.includes('targetSpeed: { min: 1, max: 100000, def: 100 }')
    && solverPageSource.includes('maxGpuCount: { min: 1, max: 8, def: 4 }'),
  'Solver URL target/maxg parsing is not finite and bounded',
)

const rankingPageSource = readFileSync(
  new URL('../src/pages/Ranking.vue', import.meta.url),
  'utf8',
)
assert(
  !/\b(?:cpuTflops|normalizeCpuTflops)\b/.test(rankingPageSource)
    && !/\bctf\b/.test(rankingPageSource),
  'Ranking still exposes or forwards the unused CPU TFLOPS setting',
)
assert(
  rankingPageSource.includes('v-model.number="promptLen"')
    && rankingPageSource.includes('v-model.number="outputLen"')
    && rankingPageSource.includes('query.pl = pl')
    && rankingPageSource.includes('query.ol = ol')
    && rankingPageSource.includes('pl:    promptLen.value')
    && rankingPageSource.includes('ol:    outputLen.value'),
  'Ranking workload inputs are hidden, not URL-persisted, or not forwarded',
)
assert(
  rankingPageSource.includes('requestIdleCallback')
    && rankingPageSource.includes('results.push(_calculateModel(models[index], config))')
    && rankingPageSource.includes('version !== _calcVersion')
    && !rankingPageSource.includes('function _buildRawResults'),
  'Ranking calculations are not performed in cancellable idle-time batches',
)
assert(
  rankingPageSource.includes('v-if="!usesUnifiedMemory"')
    && rankingPageSource.includes('usesConventionalSharedMemory')
    && rankingPageSource.includes('config.allowCpuOffload')
    && rankingPageSource.includes('usesPcieOffloadSettings = !hasUnifiedMemory && !hasSharedMemory'),
  'Ranking does not distinguish unified memory from conventional shared-memory GPUs',
)

console.log('URL/state regression passed')
