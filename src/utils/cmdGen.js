import {
  getDraftModelRepoId,
  getRuntimeModelRepoId,
  getRuntimeCompatibility,
  normalizeGpuMemoryUtilization,
  usesFp16ForCombinedPrecision,
} from './runtime.js'

function positiveInteger(value, fallback = 1) {
  const numeric = Number(value)
  return Number.isFinite(numeric) && numeric >= 1
    ? Math.max(1, Math.round(numeric))
    : fallback
}

function formatRatio(value, fallback = 0.9) {
  return normalizeGpuMemoryUtilization(value, fallback).toFixed(2)
}

/**
 * Format shell arguments as a readable copyable command. Notes are comments,
 * not continuations, so the final argument never ends in a backslash.
 */
function formatCmd(parts, notes = []) {
  const cmd = parts.length <= 3
    ? parts.join(' ')
    : parts.join(' \\\n  ')
  return notes.length ? `${cmd}\n${notes.join('\n')}` : cmd
}

function quantizationNote(format) {
  return `# Note: ${format} requires a checkpoint serialized in that format; replace the model repository if needed.`
}

function usesFp16Fallback(config) {
  return config.quant?.id === 'bf16'
    && usesFp16ForCombinedPrecision(config.gpu)
}

function genVllm(hfModel, config) {
  const {
    gpuCount,
    ppCount,
    epCount,
    ctx,
    batch,
    quant,
    kvCacheQuant,
    prefixCacheHit,
    speculativeDecoding,
    draftLen,
    draftModelRepo,
    gpuMemoryUtilization,
  } = config
  const tp = positiveInteger(gpuCount)
  const ep = positiveInteger(epCount)
  const parts = [`vllm serve ${hfModel}`]
  const notes = []

  // The estimator's EP dimension replicates attention/non-expert weights while
  // sharding experts over TP × EP. In vLLM that layout is TP × DP with EP
  // enabled; EP_SIZE is then derived as TP_SIZE × DP_SIZE.
  if (tp > 1) parts.push(`--tensor-parallel-size ${tp}`)
  if (ep > 1) parts.push(`--data-parallel-size ${ep}`)
  if (positiveInteger(ppCount) > 1) {
    parts.push(`--pipeline-parallel-size ${positiveInteger(ppCount)}`)
  }
  if (ep > 1) parts.push('--enable-expert-parallel')
  parts.push(`--max-model-len ${positiveInteger(ctx, 8192)}`)
  parts.push(`--max-num-seqs ${positiveInteger(batch)}`)
  parts.push(`--gpu-memory-utilization ${formatRatio(gpuMemoryUtilization)}`)

  if (quant.id === 'fp32') {
    parts.push('--dtype float')
  } else if (quant.id === 'bf16') {
    parts.push(usesFp16Fallback(config) ? '--dtype float16' : '--dtype bfloat16')
  } else if (quant.id === 'fp8') {
    parts.push('--quantization fp8')
  } else if (quant.id === 'int8') {
    parts.push('--quantization compressed-tensors')
    notes.push(quantizationNote('Compressed-Tensors INT8'))
  } else if (quant.id === 'int4') {
    parts.push('--quantization awq')
    notes.push(quantizationNote('AWQ INT4'))
    notes.push('# Note: use --quantization gptq for a GPTQ checkpoint.')
  }

  if (kvCacheQuant.id === 'fp8') parts.push('--kv-cache-dtype fp8')
  if (prefixCacheHit > 0) parts.push('--enable-prefix-caching')

  if (speculativeDecoding) {
    parts.push(`--spec-model ${getDraftModelRepoId(draftModelRepo)}`)
    parts.push(`--spec-tokens ${positiveInteger(draftLen, 4)}`)
  }

  return formatCmd(parts, notes)
}

function genSglang(hfModel, config) {
  const {
    gpuCount,
    ppCount,
    epCount,
    ctx,
    batch,
    quant,
    kvCacheQuant,
    speculativeDecoding,
    draftLen,
    draftModelRepo,
    gpuMemoryUtilization,
  } = config
  const parts = ['python -m sglang.launch_server', `--model-path ${hfModel}`]
  const notes = []
  const tp = positiveInteger(gpuCount)
  const pp = positiveInteger(ppCount)
  const ep = positiveInteger(epCount)
  const stageWorldSize = tp * ep

  // SGLang treats tp_size as the stage world size. DP attention divides that
  // world into replicated attention groups, while ep_size shards experts over
  // the full stage, matching the estimator's independent TP × EP dimensions.
  if (stageWorldSize > 1) parts.push(`--tp-size ${stageWorldSize}`)
  if (ep > 1) {
    parts.push(`--dp-size ${ep}`)
    parts.push(`--ep-size ${stageWorldSize}`)
    parts.push('--enable-dp-attention')
  }
  if (pp > 1) {
    parts.push(`--pp-size ${pp}`)
    // SGLang validates PP only with its overlap scheduler disabled.
    parts.push('--disable-overlap-schedule')
  }
  parts.push(`--context-length ${positiveInteger(ctx, 8192)}`)
  parts.push(`--max-running-requests ${positiveInteger(batch)}`)
  parts.push(`--mem-fraction-static ${formatRatio(gpuMemoryUtilization)}`)

  if (quant.id === 'fp32') {
    parts.push('--dtype float32')
  } else if (quant.id === 'bf16') {
    parts.push(usesFp16Fallback(config) ? '--dtype float16' : '--dtype bfloat16')
  } else if (quant.id === 'fp8') {
    parts.push('--quantization fp8')
  } else if (quant.id === 'int8') {
    parts.push('--quantization w8a8_int8')
    notes.push(quantizationNote('W8A8 INT8'))
  } else if (quant.id === 'int4') {
    parts.push('--quantization awq')
    notes.push(quantizationNote('AWQ INT4'))
  }

  if (kvCacheQuant.id === 'fp8') parts.push('--kv-cache-dtype fp8_e4m3')
  // RadixAttention prefix caching is enabled by default. `prefixCacheHit` is
  // a workload assumption, not a switch that needs an enable flag.

  if (speculativeDecoding) {
    parts.push('--speculative-algorithm STANDALONE')
    parts.push(`--speculative-draft-model-path ${getDraftModelRepoId(draftModelRepo)}`)
    parts.push(`--speculative-num-draft-tokens ${positiveInteger(draftLen, 4)}`)
  }

  parts.push('--host 0.0.0.0')
  parts.push('--port 30000')
  return formatCmd(parts, notes)
}

function genLmdeploy(hfModel, config) {
  const {
    gpuCount,
    ctx,
    batch,
    quant,
    kvCacheQuant,
    prefixCacheHit,
  } = config
  const parts = [`lmdeploy serve api_server ${hfModel}`]
  const notes = []

  if (positiveInteger(gpuCount) > 1) parts.push(`--tp ${positiveInteger(gpuCount)}`)
  parts.push(`--session-len ${positiveInteger(ctx, 8192)}`)
  parts.push(`--max-batch-size ${positiveInteger(batch)}`)

  if (quant.id === 'bf16') {
    parts.push(usesFp16Fallback(config) ? '--dtype float16' : '--dtype bfloat16')
  } else if (quant.id === 'int4') {
    parts.push('--model-format awq')
    notes.push(quantizationNote('AWQ INT4'))
    notes.push('# Note: use --model-format gptq when that matches the checkpoint.')
  } else if (quant.id === 'int8') {
    parts.push('--backend pytorch')
    notes.push(quantizationNote('a supported pre-quantized INT8 format'))
  } else if (quant.id === 'fp8') {
    parts.push('--model-format fp8')
  }

  if (kvCacheQuant.id === 'int4') {
    parts.push('--quant-policy 4')
  } else if (kvCacheQuant.id === 'int8') {
    parts.push('--quant-policy 8')
  } else if (kvCacheQuant.id === 'fp8') {
    parts.push('--quant-policy fp8')
  }
  if (prefixCacheHit > 0) parts.push('--enable-prefix-caching')
  parts.push('--server-port 23333')
  return formatCmd(parts, notes)
}

function genTgi(hfModel, config) {
  const {
    gpuCount,
    ctx,
    batch,
    promptLen,
    quant,
    kvCacheQuant,
    gpu,
    gpuMemoryUtilization,
  } = config
  const vendor = gpu?.vendor ?? config.gpuVendor ?? 'nvidia'
  const parts = vendor === 'amd'
    ? [
        'docker run --rm',
        '--cap-add=SYS_PTRACE',
        '--security-opt seccomp=unconfined',
        '--device=/dev/kfd',
        '--device=/dev/dri',
        '--group-add video',
        '--ipc=host',
        '-p 8080:80',
        '-v $PWD/data:/data',
        'ghcr.io/huggingface/text-generation-inference:3.3.5-rocm',
      ]
    : [
        'docker run --rm --gpus all',
        '--ipc=host',
        '-p 8080:80',
        '-v $PWD/data:/data',
        'ghcr.io/huggingface/text-generation-inference:latest',
      ]

  parts.push(`--model-id ${hfModel}`)
  parts.push(`--num-shard ${positiveInteger(gpuCount)}`)
  parts.push(`--max-total-tokens ${positiveInteger(ctx, 8192)}`)
  parts.push(`--max-input-tokens ${positiveInteger(promptLen, 1024)}`)
  parts.push(`--cuda-memory-fraction ${formatRatio(gpuMemoryUtilization)}`)
  parts.push(
    `--max-batch-prefill-tokens ${
      positiveInteger(batch) * positiveInteger(promptLen, 1024)
    }`,
  )

  if (quant.id === 'int8') {
    parts.push('--quantize bitsandbytes')
  } else if (quant.id === 'int4') {
    parts.push('--quantize awq')
  } else if (quant.id === 'fp8') {
    parts.push('--quantize fp8')
  } else if (quant.id === 'bf16') {
    parts.push(usesFp16Fallback(config) ? '--dtype float16' : '--dtype bfloat16')
  }
  if (kvCacheQuant.id === 'fp8') parts.push('--kv-cache-dtype fp8_e4m3fn')

  return formatCmd(parts)
}

const GGUF_QUANT_MAP = Object.freeze({
  fp32: { suffix: 'F32' },
  bf16: { suffix: 'F16' },
  int8: { suffix: 'Q8_0' },
  int6: { suffix: 'Q6_K' },
  int5: { suffix: 'Q5_K_M' },
  int4: { suffix: 'Q4_K_M' },
  int3: { suffix: 'Q3_K_M' },
  int2: { suffix: 'Q2_K' },
})

function calcNgl(model, cpuOffload, pureCpu, nglCount) {
  if (pureCpu) return 0
  if (cpuOffload && model.type === 'moe') return 'all'
  if (cpuOffload && model.type !== 'moe' && nglCount != null) {
    return Math.max(0, Math.round(Number(nglCount) || 0))
  }
  return 'all'
}

function safeLocalModelName(model) {
  const source = String(model.id ?? model.name ?? 'model')
  return source.replace(/[^A-Za-z0-9_.-]+/g, '-').replace(/^-+|-+$/g, '') || 'model'
}

function genLlamacpp(config) {
  const {
    model,
    ctx,
    batch,
    quant,
    kvCacheQuant,
    cpuOffload,
    pureCpu,
    nglCount,
    gpuCount,
  } = config
  const ggufInfo = GGUF_QUANT_MAP[quant.id]
  const parallel = positiveInteger(batch)
  const perSlotContext = positiveInteger(ctx, 8192)
  const totalContext = perSlotContext * parallel
  const modelPath = `./models/${safeLocalModelName(model)}-${ggufInfo.suffix}.gguf`
  const parts = [
    'llama-server',
    `--model ${modelPath}`,
    `--ctx-size ${totalContext}`,
    `--parallel ${parallel}`,
    `--n-gpu-layers ${calcNgl(model, cpuOffload, pureCpu, nglCount)}`,
  ]
  const notes = [
    `# Note: --ctx-size is total KV capacity (${perSlotContext} tokens × ${parallel} parallel slots).`,
  ]

  if (cpuOffload && model.type === 'moe') {
    parts.push('--cpu-moe')
    notes.push('# Note: --cpu-moe keeps MoE expert weights in system RAM.')
  }
  if (pureCpu) {
    parts.push('--device none')
    parts.push('--threads <CPU_THREADS>')
  } else if (positiveInteger(gpuCount) > 1) {
    // The estimator models simultaneous tensor sharding, so use llama.cpp's
    // row-parallel split instead of its pipelined layer split.
    parts.push('--split-mode row')
    notes.push('# Note: row-parallel splitting matches the estimator; use --tensor-split to override automatic GPU proportions.')
  }

  const cacheType = {
    fp16: 'f16',
    int8: 'q8_0',
    int4: 'q4_0',
  }[kvCacheQuant.id]
  if (cacheType) {
    parts.push(`--cache-type-k ${cacheType}`)
    parts.push(`--cache-type-v ${cacheType}`)
  }
  parts.push('--host 0.0.0.0')
  parts.push('--port 8080')

  if (quant.id === 'bf16') {
    notes.push('# Note: the generated GGUF path uses F16 for the combined BF16/FP16 estimator option.')
  }
  if (quant.id === 'int4') {
    notes.push('# Note: adjust Q4_K_M to the quantization suffix in the downloaded GGUF filename.')
  }
  return formatCmd(parts, notes)
}

function genMlx(hfModel, config) {
  const { quant } = config
  const conversion = [
    'mlx_lm.convert',
    `--hf-path ${hfModel}`,
    '--mlx-path ./mlx_model',
  ]
  const notes = [
    '# Note: mlx_lm.server does not expose a startup context-limit option; clients must stay within the model context.',
    '# Note: MLX-LM server is intended for local development, not production exposure.',
  ]

  if (quant.id === 'fp32') {
    conversion.push('--dtype float32')
  } else if (quant.id === 'bf16') {
    conversion.push('--dtype bfloat16')
  } else {
    const bits = Number(quant.id.replace('int', ''))
    conversion.push('--quantize')
    conversion.push(`--q-bits ${bits}`)
  }

  return [
    '# 1. Convert the Hugging Face checkpoint to MLX format',
    formatCmd(conversion),
    '',
    '# 2. Serve the converted model',
    'mlx_lm.server --model ./mlx_model --port 8080',
    ...notes,
  ].join('\n')
}

function genTrtllm(hfModel, config) {
  const {
    gpuCount,
    ppCount,
    ctx,
    batch,
    quant,
    kvCacheQuant,
    speculativeDecoding,
    speculativeConfigPath,
    gpuMemoryUtilization,
  } = config
  const parts = [
    `trtllm-serve ${hfModel}`,
    `--max_batch_size ${positiveInteger(batch)}`,
    `--max_seq_len ${positiveInteger(ctx, 8192)}`,
    `--tp_size ${positiveInteger(gpuCount)}`,
  ]
  const notes = []

  if (positiveInteger(ppCount) > 1) parts.push(`--pp_size ${positiveInteger(ppCount)}`)
  if (kvCacheQuant.id === 'fp8') parts.push('--kv_cache_dtype fp8')
  if (speculativeDecoding) {
    parts.push(`--config ${speculativeConfigPath}`)
    notes.push('# Note: define the draft-model speculative decoding method in the TensorRT-LLM YAML config.')
  }
  parts.push('--host 0.0.0.0')
  parts.push('--port 8000')

  if (quant.id !== 'bf16') {
    notes.push(
      `# Note: trtllm-serve loads quantization from checkpoint metadata; use a ${quant.id.toUpperCase()} checkpoint.`,
    )
  }
  notes.push(
    `# Note: the estimator reserves ${(normalizeGpuMemoryUtilization(gpuMemoryUtilization) * 100).toFixed(0)}% of each GPU; tune TensorRT-LLM cache settings if the checkpoint leaves a different amount free.`,
  )
  return formatCmd(parts, notes)
}

export const FRAMEWORK_DOCS = Object.freeze({
  vllm: 'https://docs.vllm.ai/en/latest/cli/serve/',
  sglang: 'https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md',
  lmdeploy: 'https://lmdeploy.readthedocs.io/en/latest/api/cli.html',
  tgi: 'https://huggingface.co/docs/text-generation-inference/reference/launcher',
  exllamav2: 'https://github.com/theroyallab/tabbyAPI',
  llamacpp: 'https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md',
  llamacpp_metal: 'https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md',
  mlx: 'https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md',
  trtllm: 'https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html',
})

export function getFrameworkDocsUrl(frameworkId) {
  return FRAMEWORK_DOCS[frameworkId] ?? null
}

export function getCommandCompatibility(framework, config) {
  return getRuntimeCompatibility({
    framework,
    ...config,
    forCommand: true,
  })
}

/**
 * Generate a command only when the selected runtime, hardware, topology,
 * formats, and downloadable model repository describe a supported deployment.
 */
export function generateCmd(framework, config) {
  if (!framework || !config?.model || !config?.quant || !config?.kvCacheQuant) return null
  const compatibility = getCommandCompatibility(framework, config)
  if (!compatibility.supported) return null

  const hfModel = getRuntimeModelRepoId(config.model, config.quant, framework)
  switch (framework.id) {
    case 'vllm':
      return genVllm(hfModel, config)
    case 'sglang':
      return genSglang(hfModel, config)
    case 'lmdeploy':
      return genLmdeploy(hfModel, config)
    case 'tgi':
      return genTgi(hfModel, config)
    case 'llamacpp':
    case 'llamacpp_metal':
      return genLlamacpp(config)
    case 'mlx':
      return genMlx(hfModel, config)
    case 'trtllm':
      return genTrtllm(hfModel, config)
    default:
      return null
  }
}
