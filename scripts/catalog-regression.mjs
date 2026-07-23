import {
  ALL_MODELS,
  COMMUNITY_MODELS,
  DENSE_MODELS,
  MODEL_ID_ALIASES,
  MOE_MODELS,
  resolveModelId,
} from '../src/data/models/index.js'
import { GPU_LIST } from '../src/data/gpus/index.js'
import { existsSync, readFileSync } from 'node:fs'

function assert(condition, message) {
  if (!condition) throw new Error(message)
}

function assertUnique(items, getKey, label) {
  const seen = new Map()
  for (const item of items) {
    const key = getKey(item)
    assert(key, `${label} contains an empty key`)
    assert(!seen.has(key), `${label} contains duplicate "${key}"`)
    seen.set(key, item)
  }
}

for (const [label, models] of [
  ['DENSE_MODELS', DENSE_MODELS],
  ['MOE_MODELS', MOE_MODELS],
  ['COMMUNITY_MODELS', COMMUNITY_MODELS],
]) {
  assertUnique(models, model => model.id, label)
}
assert(
  DENSE_MODELS.every(model => model.type !== 'moe'),
  'DENSE_MODELS contains a model marked as MoE',
)
assert(
  MOE_MODELS.every(model => model.type === 'moe'),
  'MOE_MODELS contains a model not marked as MoE',
)

assertUnique(ALL_MODELS, model => model.id, 'ALL_MODELS IDs')
assertUnique(
  ALL_MODELS,
  model => model.name.toLowerCase().replace(/[^a-z0-9]+/g, ''),
  'ALL_MODELS normalized display names',
)
assertUnique(GPU_LIST, gpu => gpu.id, 'GPU_LIST IDs')

const modelIds = new Set(ALL_MODELS.map(model => model.id))
const modelsById = new Map(ALL_MODELS.map(model => [model.id, model]))
const gpusById = new Map(GPU_LIST.map(gpu => [gpu.id, gpu]))
for (const [legacyId, canonicalId] of Object.entries(MODEL_ID_ALIASES)) {
  assert(!modelIds.has(legacyId), `Legacy model ID "${legacyId}" is still in ALL_MODELS`)
  assert(modelIds.has(canonicalId), `Alias target "${canonicalId}" is missing`)
  assert(resolveModelId(legacyId) === canonicalId, `Alias "${legacyId}" does not resolve`)
}

for (const model of ALL_MODELS) {
  for (const field of ['params', 'layers', 'hidden_size', 'max_ctx']) {
    assert(Number(model[field]) > 0, `${model.id}.${field} must be positive`)
  }
  const isRecurrent = ['mamba', 'mamba2', 'rwkv', 'ssm'].includes(model.architecture)
  if (!isRecurrent) {
    assert(Number(model.kv_heads) > 0, `${model.id}.kv_heads must be positive`)
    assert(Number(model.head_dim) > 0, `${model.id}.head_dim must be positive`)
  }
  if (model.vision_encoder_params != null) {
    assert(Number(model.vision_encoder_params) > 0, `${model.id}.vision_encoder_params must be positive`)
    assert(
      ['total', 'text'].includes(model.params_scope ?? model.parameter_scope),
      `${model.id} must declare whether params includes its vision encoder`,
    )
    if ((model.params_scope ?? model.parameter_scope) === 'total') {
      assert(
        model.vision_encoder_params < model.params,
        `${model.id}.vision_encoder_params must be smaller than total params`,
      )
    }
  }

  const hf = model.links?.hf
  if (model.localInference !== false && hf) {
    const pathParts = new URL(hf).pathname.split('/').filter(Boolean)
    assert(
      pathParts.length >= 2,
      `${model.id} has an organization page instead of a downloadable Hugging Face repository`,
    )
  }
  if (model.availability === 'api-only') {
    assert(model.localInference === false, `${model.id} API-only model must disable local inference`)
    assert(model.parameterEstimate === true, `${model.id} API-only specs must be disclosed as estimates`)
  }
}

for (const gpu of GPU_LIST) {
  for (const field of ['bw', 'tdp']) {
    assert(Number(gpu[field]) > 0, `${gpu.id}.${field} must be positive`)
  }
  assert(Number(gpu.vram) >= 0, `${gpu.id}.vram must not be negative`)
  if (gpu.vram === 0) {
    assert(gpu.sharedMemory === true, `${gpu.id} with zero dedicated VRAM must use shared memory`)
  }
  if (gpu.unitKind === 'system') {
    assert(
      Number.isInteger(gpu.physicalGpuCount) && gpu.physicalGpuCount > 1,
      `${gpu.id} aggregate system must declare physicalGpuCount`,
    )
  }
  if (gpu.modified) {
    assert(gpu.official === false, `${gpu.id} modified card must be marked unofficial`)
    assert(gpu.baseGpuId, `${gpu.id} modified card must identify its base GPU`)
    assert(gpu.specConfidence, `${gpu.id} modified card must disclose spec confidence`)
    assert(gpu.sources?.some(source => source.url), `${gpu.id} modified card must include a seller source`)
    const advertisedBw = gpu.memoryDataRateGbps * gpu.memoryBusBits / 8
    assert(
      Math.abs(advertisedBw - gpu.bw) < 0.01,
      `${gpu.id} bandwidth does not match memory data rate and bus width`,
    )
    const baseGpu = gpusById.get(gpu.baseGpuId)
    assert(baseGpu, `${gpu.id} references a missing base GPU`)
    for (const field of ['bf16', 'int8', 'int4']) {
      assert(
        gpu[field] === baseGpu[field],
        `${gpu.id}.${field} must match its unmodified base GPU`,
      )
    }
  }
}

for (const appleGpu of GPU_LIST.filter(gpu => gpu.vendor === 'apple')) {
  assert(appleGpu.computeEstimate === true, `${appleGpu.id} must disclose derived compute`)
  assert(appleGpu.tdpEstimate === true, `${appleGpu.id} must disclose estimated TDP`)
  assert(appleGpu.specConfidence === 'derived', `${appleGpu.id} must have derived spec confidence`)
}
assert(
  !GPU_LIST.some(gpu => /^apple_m[45]_ultra/.test(gpu.id)),
  'Unannounced M4/M5 Ultra hardware must not appear in the catalog',
)
assert(gpusById.has('apple_m3_ultra_96g'), 'Official 60-core M3 Ultra configuration is missing')

const kimiK26 = modelsById.get('kimi_k2_6')
assert(kimiK26?.kv_lora_rank === 512, 'Kimi K2.6 MLA metadata is missing')
assert(kimiK26?.vision_encoder_params === 0.4, 'Kimi K2.6 MoonViT metadata is missing')
assert(kimiK26?.vision_seq_tokens === 4096, 'Kimi K2.6 visual-token limit is stale')
const glm51 = modelsById.get('glm5_1')
assert(glm51?.max_ctx === 202752, 'GLM-5.1 context length is stale')
assert(glm51?.kv_lora_rank === 512, 'GLM-5.1 MLA metadata is missing')

const rtx2080Ti = gpusById.get('rtx2080ti')
const rtx2080TiMod = gpusById.get('rtx2080ti_22g_mod')
for (const gpu of [rtx2080Ti, rtx2080TiMod]) {
  assert(gpu?.nativeBf16 === false, `${gpu?.id ?? 'RTX 2080 Ti'} must disclose no native BF16`)
  assert(gpu?.bf16 === 56.9, `${gpu?.id ?? 'RTX 2080 Ti'} FP16 fallback throughput is stale`)
  assert(gpu?.int8 === 227.7 && gpu?.int4 === 455.4, `${gpu?.id ?? 'RTX 2080 Ti'} Tensor TOPS are stale`)
}

const ryzenAiMaxLinux = gpusById.get('ryzen_ai_max_395')
const ryzenAiMaxWindows = gpusById.get('ryzen_ai_max_395_win')
assert(
  ryzenAiMaxLinux?.unifiedMemory
    && ryzenAiMaxLinux.vram === 112
    && ryzenAiMaxLinux.bw === 256,
  'Ryzen AI MAX+ 395 unified-memory/Linux allocation metadata is stale',
)
assert(
  ryzenAiMaxWindows?.unifiedMemory
    && ryzenAiMaxWindows.vram === 96
    && ryzenAiMaxWindows.bw === 256,
  'Ryzen AI MAX+ 395 Windows VGM metadata is stale',
)
for (const gpu of [ryzenAiMaxLinux, ryzenAiMaxWindows]) {
  assert(
    gpu?.computeEstimate === true && gpu?.specConfidence === 'derived',
    `${gpu?.id ?? 'Ryzen AI MAX+ 395'} must disclose derived BF16 compute`,
  )
}

const documentationMarkers = {
  'README.md': {
    models: `| **模型** | ${ALL_MODELS.length} 个规范模型`,
    hardware: `| **GPU** | ${GPU_LIST.length} 个型号`,
    split: `Dense ${DENSE_MODELS.length} + MoE ${MOE_MODELS.length}`,
    range: '0.13B - 1.6T',
  },
  'README.en.md': {
    models: `| **Models** | ${ALL_MODELS.length} canonical models`,
    hardware: `| **GPUs** | ${GPU_LIST.length} models`,
    split: `Dense ${DENSE_MODELS.length} + MoE ${MOE_MODELS.length}`,
    range: '0.13B - 1.6T',
  },
  'Docs.md': {
    models: `**模型数量**: ${ALL_MODELS.length} 个`,
    hardware: `**GPU 数量**: ${GPU_LIST.length} 个`,
  },
  'public/llms.txt': {
    models: `支持 ${ALL_MODELS.length} 个规范模型`,
    hardware: `覆盖 ${GPU_LIST.length} 个硬件条目`,
  },
  'index.html': {
    models: `"${ALL_MODELS.length} 个规范模型"`,
    hardware: `"${GPU_LIST.length} 个硬件条目`,
  },
}

for (const [path, markers] of Object.entries(documentationMarkers)) {
  const content = readFileSync(new URL(`../${path}`, import.meta.url), 'utf8')
  assert(content.includes(markers.models), `${path} has a stale model count`)
  assert(content.includes(markers.hardware), `${path} has a stale hardware count`)
  if (markers.split) assert(content.includes(markers.split), `${path} has stale Dense/MoE counts`)
  if (markers.range) assert(content.includes(markers.range), `${path} has a stale model parameter range`)
}

for (const path of ['public/llms.txt', 'index.html']) {
  const content = readFileSync(new URL(`../${path}`, import.meta.url), 'utf8')
  assert(
    !/\d+\s*[–-]\s*\d+\s*(?:token\/s|TPS)\b/i.test(content),
    `${path} contains a stale hard-coded throughput range`,
  )
  for (const staleClaim of [
    '8 张 H100 SXM（FP8',
    '4 张 H100 SXM（BF16',
    '理论上界约',
    '精确估算',
    '精确计算',
  ]) {
    assert(!content.includes(staleClaim), `${path} contains stale FAQ claim "${staleClaim}"`)
  }
}

const indexHtml = readFileSync(new URL('../index.html', import.meta.url), 'utf8')
const jsonLdBlocks = [...indexHtml.matchAll(
  /<script\s+type=["']application\/ld\+json["']>([\s\S]*?)<\/script>/gi,
)]
assert(jsonLdBlocks.length > 0, 'index.html is missing structured JSON-LD data')
for (const [, json] of jsonLdBlocks) {
  try {
    JSON.parse(json)
  } catch (error) {
    throw new Error(`index.html contains invalid JSON-LD: ${error.message}`)
  }
}
assert(
  !existsSync(new URL('../public/manifest.json', import.meta.url)),
  'public/manifest.json duplicates the Vite PWA-generated manifest',
)

console.log(
  `Catalog regression passed: ${ALL_MODELS.length} canonical models, `
  + `${GPU_LIST.length} hardware entries, ${Object.keys(MODEL_ID_ALIASES).length} legacy aliases.`,
)
