/**
 * 回归基准：对比计算器输出与公开 benchmark 参考值
 * 运行：node scripts/benchmark-regression.mjs
 */
import { calcAll, getQuantBytes } from '../src/utils/calc.js'
import { GPU_LIST } from '../src/data/gpus/index.js'
import { ALL_MODELS } from '../src/data/models/index.js'
import { QUANT_MAP, FRAMEWORK_MAP, INTERCONNECT_MAP } from '../src/data/constants.js'

const model8b = ALL_MODELS.find(m => m.id === 'llama3_8b')
const model70b = ALL_MODELS.find(m => m.id === 'llama3_3_70b')
const int4 = QUANT_MAP.find(q => q.id === 'int4')
const bf16 = QUANT_MAP.find(q => q.id === 'bf16')
const mlx = FRAMEWORK_MAP.find(f => f.id === 'mlx')
const metal = FRAMEWORK_MAP.find(f => f.id === 'llamacpp_metal')
const llama = FRAMEWORK_MAP.find(f => f.id === 'llamacpp')
const mixtral = ALL_MODELS.find(m => m.id === 'mixtral_8x7b')
const vllm = FRAMEWORK_MAP.find(f => f.id === 'vllm')
const ic = INTERCONNECT_MAP[0]

// Only retain observations that are physically possible for the exact model,
// serialization and hardware selected below. Several former rows mixed results
// from a smaller/effectively lighter model or quantization with an 8B model:
//
// - M4 MLX 28, M4 Pro MLX 93, M4 Max MLX 160
// - M3 Pro MLX 35, M4 Max 36GB MLX 138
// - M4 Pro llama.cpp/Metal 77, M4 Pro Mixtral MLX 45
// - RTX 4090 BF16 vLLM 95
//
// Each exceeds the raw weight-streaming roof for the configuration represented
// by the calculator (before KV reads or runtime overhead), so treating it as a
// calibration target would force the estimator to violate hardware bandwidth.
// Those measurements may be valid for another model/packing/build, but they are
// not valid assertions for these rows.
const CASES = [
  { label: 'M3 Max 64G MLX 8B', gpu: 'apple_m3_max_64g', model: model8b, quant: int4, fw: mlx, batch: 1, real: 50 },
  { label: 'M1 Max 32G MLX 8B', gpu: 'apple_m1_max_32g', model: model8b, quant: int4, fw: mlx, batch: 1, real: 38 },
  // Public Metal observations vary substantially by llama.cpp build and exact
  // Q4 packing. Keep this as a broad sanity check, not a tight calibration point.
  { label: 'M4 16G metal 8B', gpu: 'apple_m4_16g', model: model8b, quant: int4, fw: metal, batch: 1, real: 22.5, tolerance: 0.40 },
  { label: '4090 llama.cpp INT4 b1', gpu: 'rtx4090', model: model8b, quant: int4, fw: llama, batch: 1, real: 127 },
  { label: 'M2 Max 32G MLX 8B', gpu: 'apple_m2_max_32g', model: model8b, quant: int4, fw: mlx, batch: 1, real: 45 },
  { label: '4090 INT4 vLLM b1', gpu: 'rtx4090', model: model8b, quant: int4, fw: vllm, batch: 1, real: 125 },
  { label: '4090 INT4 vLLM b32', gpu: 'rtx4090', model: model8b, quant: int4, fw: vllm, batch: 32, real: 1100, metric: 'decodeToks' },
  { label: '8xH100 70B INT4 per-card VRAM', gpu: 'h100_sxm', model: model70b, quant: int4, fw: vllm, batch: 1, gpuCount: 8, metric: 'perCardNeeded', real: 5.8, tolerance: 0.15 },
]

function getRawThroughputRoof(c, gpu) {
  if (!['singleToks', 'decodeToks'].includes(c.metric ?? 'singleToks')) return null

  const gpuCount = c.gpuCount ?? 1
  const batch = c.batch ?? 1
  const weightGB = c.model.params * getQuantBytes(c.quant, gpu, c.fw)
  // A synchronized decode step can reuse one weight read across the batch.
  const bandwidthRoof = gpu.bw * gpuCount / weightGB * batch
  const flopsKey = c.quant.flops_key ?? 'bf16'
  const tflopsPerGpu = gpu[flopsKey] ?? gpu.bf16
  const computeRoof = tflopsPerGpu * gpuCount * 1e12 / (2 * c.model.params * 1e9)
  return Math.min(bandwidthRoof, computeRoof)
}

function runCase(c) {
  const gpu = GPU_LIST.find(g => g.id === c.gpu)
  const physicalRoof = getRawThroughputRoof(c, gpu)
  if (physicalRoof != null && c.real > physicalRoof * 1.001) {
    throw new Error(
      `${c.label}: reference ${c.real} exceeds raw hardware roof ${physicalRoof.toFixed(1)}`,
    )
  }

  const r = calcAll({
    gpu, gpuCount: c.gpuCount ?? 1, interconnect: ic,
    model: c.model, quant: c.quant, ctx: 8192, batch: c.batch,
    promptLen: 512, outputLen: 128, framework: c.fw, flashAttention: true,
  })
  const metric = c.metric ?? 'singleToks'
  const got = r[metric]
  const err = Math.abs(got / c.real - 1)
  const tol = c.tolerance ?? 0.18
  const ok = err <= tol
  return {
    ...c,
    got: +got.toFixed(1),
    errPct: +(err * 100).toFixed(1),
    ok,
    tol: +(tol * 100).toFixed(0),
    physicalRoof,
  }
}

const results = CASES.map(runCase)
const failed = results.filter(r => !r.ok)

console.log('TPS Calculator benchmark regression\n')
for (const r of results) {
  const mark = r.ok ? 'OK' : 'FAIL'
  console.log(`[${mark}] ${r.label}: ${r.got} vs ${r.real} (${r.errPct}% err, tol ${r.tol}%)`)
}

console.log(`\n${results.length - failed.length}/${results.length} passed`)
if (failed.length) {
  process.exitCode = 1
}
