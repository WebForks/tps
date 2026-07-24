// src/utils/calc.js
import { getAttentionSummary, getAttentionType, getTotalHeads } from './model.js'
import { getRuntimeCompatibility, inferGpuArchitecture } from './runtime.js'

const GB = 1e9
const MS = 1000
const DEFAULT_HEAD_DIM = 128
const DEFAULT_HIDDEN_SIZE = 4096
const DEFAULT_CPU_MEMORY_BW_GBS = 76.8 // dual-channel DDR5-4800

function finiteNumber(value, fallback) {
  if (value == null || (typeof value === 'string' && value.trim() === '')) return fallback
  const number = Number(value)
  return Number.isFinite(number) ? number : fallback
}

function positiveNumber(value, fallback, minimum = Number.EPSILON) {
  return Math.max(minimum, finiteNumber(value, fallback))
}

function positiveInteger(value, fallback = 1) {
  return Math.max(1, Math.round(finiteNumber(value, fallback)))
}

function isPositiveIntegerValue(value) {
  const numeric = Number(value)
  return Number.isFinite(numeric) && numeric >= 1 && Number.isInteger(numeric)
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value))
}

function getFrameworkRange(framework, key, fallback = 1) {
  const mid = positiveNumber(framework?.[key], fallback)
  return {
    min: positiveNumber(framework?.[`${key}Min`], mid),
    mid,
    max: positiveNumber(framework?.[`${key}Max`], mid),
  }
}

function normalizeRange(range) {
  const values = [range.min, range.mid, range.max].sort((a, b) => a - b)
  return { min: values[0], mid: values[1], max: values[2] }
}

function getTopology(gpuCount, ppCount, epCount, model) {
  const gpuCountValid = isPositiveIntegerValue(gpuCount)
  const ppInputValid = isPositiveIntegerValue(ppCount)
  const epInputValid = isPositiveIntegerValue(epCount)
  const totalGpuCount = positiveInteger(gpuCount)
  const requestedPp = positiveInteger(ppCount)
  const requestedEp = positiveInteger(epCount)
  const modelLayers = positiveInteger(model?.layers)
  // A pipeline rank needs at least one decoder layer. Allowing more pipeline
  // stages than layers creates empty ranks and can make the apparent per-card
  // memory *increase* as TP ranks are traded for unusable PP ranks.
  const ppValid = requestedPp <= totalGpuCount
    && requestedPp <= modelLayers
    && totalGpuCount % requestedPp === 0
  const effectivePp = ppValid ? requestedPp : 1
  const stageGpuCount = totalGpuCount / effectivePp
  const epApplicable = model?.type === 'moe'
  const epValid = requestedEp === 1 || (
    epApplicable
    && requestedEp <= stageGpuCount
    && stageGpuCount % requestedEp === 0
    && (!model?.experts || model.experts % requestedEp === 0)
  )
  const effectiveEp = epValid ? requestedEp : 1
  const tpCount = stageGpuCount / effectiveEp

  return {
    totalGpuCount,
    ppCount: effectivePp,
    epCount: effectiveEp,
    tpCount,
    stageGpuCount,
    topologyOk: gpuCountValid && ppInputValid && epInputValid && ppValid && epValid,
    requestedPp,
    requestedEp,
  }
}

function getMoeInfo(model) {
  if (model?.type !== 'moe') {
    return {
      nonExpertParams: model?.params ?? 0,
      expertParams: 0,
      activeExpertParams: 0,
      reliable: true,
    }
  }

  const total = positiveNumber(model.params, 1)
  const active = clamp(positiveNumber(model.active_params, total), 0, total)
  const experts = positiveInteger(model.experts, 1)
  const topK = clamp(positiveInteger(model.experts_per_token, 1), 1, experts)
  let nonExpertParams = finiteNumber(model.non_expert_params, NaN)
  let reliable = Number.isFinite(nonExpertParams)
    && nonExpertParams >= 0
    && nonExpertParams <= active
    && nonExpertParams <= total

  if (!reliable && topK !== experts) {
    const derived = (total * topK - experts * active) / (topK - experts)
    if (Number.isFinite(derived) && derived >= 0 && derived <= active && derived <= total) {
      nonExpertParams = derived
      reliable = true
    }
  }

  if (!reliable && topK === experts && Math.abs(active - total) / total < 0.05) {
    nonExpertParams = total
    reliable = true
  }

  // Rounded catalog totals can make the exact MoE identity impossible. Keep the
  // estimate finite, but mark decomposition-dependent modes as unsupported.
  if (!reliable) {
    nonExpertParams = Math.min(active, Math.max(0.25, active * 0.30))
  }

  const expertParams = Math.max(0, total - nonExpertParams)
  const activeExpertParams = Math.max(0, active - nonExpertParams)
  return { nonExpertParams, expertParams, activeExpertParams, reliable }
}

function getVisionParameterInfo(model) {
  const catalogParams = positiveNumber(model?.params, 1)
  const encoderValue = finiteNumber(model?.vision_encoder_params, NaN)
  const visionEncoderParams = Number.isFinite(encoderValue) && encoderValue > 0
    ? encoderValue
    : 0
  const textValue = finiteNumber(model?.text_params, NaN)
  const hasExplicitTextParams = Number.isFinite(textValue) && textValue > 0
  const rawScope = String(
    model?.params_scope
    ?? model?.parameter_scope
    ?? '',
  ).toLowerCase()
  const paramsScope = ['text', 'decoder'].includes(rawScope)
    ? 'text'
    : ['total', 'resident', 'includes_vision'].includes(rawScope)
      ? 'total'
      : 'unknown'

  // `params` is treated as the resident total unless the catalog explicitly
  // declares that it contains only the text decoder. This guarantees that
  // encoder weights are never silently added twice.
  const residentParams = paramsScope === 'text'
    ? catalogParams + visionEncoderParams
    : catalogParams
  // A declared total plus a known encoder gives a defensible decoder estimate.
  // It remains approximate because projectors/other multimodal components may
  // also be included. With unknown scope, retain the conservative legacy cost.
  const decoderParams = hasExplicitTextParams
    ? textValue
    : paramsScope === 'total' && visionEncoderParams > 0
      ? Math.max(Number.EPSILON, catalogParams - visionEncoderParams)
      : catalogParams
  const knownComponentGap = Math.abs(
    residentParams - (decoderParams + visionEncoderParams),
  )
  const visionAccountingApproximate = visionEncoderParams > 0
    && (
      !hasExplicitTextParams
      || paramsScope === 'unknown'
      || knownComponentGap > Math.max(0.01, residentParams * 0.01)
    )

  return {
    residentParams,
    decoderParams,
    visionEncoderParams,
    paramsScope,
    hasExplicitTextParams,
    visionAccountingApproximate,
  }
}

function isPureRecurrent(model) {
  if (['mamba', 'rwkv', 'ssm'].includes(model?.architecture)) return true
  const id = String(model?.id ?? '')
  return /(^|_)(mamba|rwkv)/i.test(id) || /^codestral_mamba/i.test(id) || /^falcon_mamba/i.test(id)
}

function getLayerBreakdown(model) {
  const layers = positiveInteger(model?.layers)
  if (isPureRecurrent(model)) {
    return { layers, globalLayers: 0, localLayers: 0, linearLayers: layers }
  }

  const explicitLinear = finiteNumber(model?.linear_attention_layers, NaN)
  const ratio = finiteNumber(model?.mamba_ratio, NaN)
  const linearLayers = Number.isFinite(explicitLinear)
    ? clamp(Math.round(explicitLinear), 0, layers)
    : Number.isFinite(ratio)
      ? layers - clamp(Math.round(layers * ratio), 0, layers)
      : 0
  const softmaxLayers = Math.max(0, layers - linearLayers)

  if (model?.sliding_window != null && model?.local_layers != null && model.sliding_window > 0) {
    const localLayers = clamp(Math.round(model.local_layers), 0, softmaxLayers)
    return {
      layers,
      globalLayers: softmaxLayers - localLayers,
      localLayers,
      linearLayers,
    }
  }

  return { layers, globalLayers: softmaxLayers, localLayers: 0, linearLayers }
}

function normalizeLayerKind(value) {
  const kind = String(value ?? '').toLowerCase()
  if (['global', 'full', 'full_attention', 'softmax'].includes(kind)) return 'global'
  if (['local', 'sliding', 'sliding_window', 'local_attention'].includes(kind)) return 'local'
  if (['linear', 'recurrent', 'mamba', 'ssm', 'gated_delta'].includes(kind)) return 'linear'
  return null
}

function getExplicitLayerKinds(model, layerBreakdown) {
  const { layers, globalLayers, localLayers, linearLayers } = layerBreakdown
  const explicitKinds = Array.isArray(model?.layer_types)
    ? model.layer_types
    : null
  let kinds = explicitKinds?.length === layers
    ? explicitKinds.map(normalizeLayerKind)
    : null

  if (!kinds || kinds.some(kind => kind == null)) {
    const pattern = [
      model?.layer_pattern,
      model?.attention_layer_pattern,
      model?.attention_pattern,
    ].find(value => Array.isArray(value) && value.length > 0)
    if (pattern) {
      const normalizedPattern = pattern.map(normalizeLayerKind)
      if (normalizedPattern.every(kind => kind != null)) {
        kinds = Array.from(
          { length: layers },
          (_, index) => normalizedPattern[index % normalizedPattern.length],
        )
      }
    }
  }

  if (!kinds || kinds.some(kind => kind == null)) {
    const interval = positiveInteger(
      model?.full_attention_interval
      ?? model?.global_attention_interval,
      0,
    )
    const hasInterval = finiteNumber(
      model?.full_attention_interval
      ?? model?.global_attention_interval,
      NaN,
    ) > 0
    if (hasInterval) {
      const offset = clamp(
        Math.round(finiteNumber(
          model?.full_attention_offset
          ?? model?.global_attention_offset,
          interval - 1,
        )),
        0,
        interval - 1,
      )
      const nonGlobalKind = linearLayers > 0
        ? 'linear'
        : localLayers > 0
          ? 'local'
          : 'global'
      kinds = Array.from(
        { length: layers },
        (_, index) => index % interval === offset ? 'global' : nonGlobalKind,
      )
    }
  }

  if (!kinds) return null
  const counts = kinds.reduce(
    (result, kind) => ({ ...result, [kind]: result[kind] + 1 }),
    { global: 0, local: 0, linear: 0 },
  )
  if (
    counts.global !== globalLayers
    || counts.local !== localLayers
    || counts.linear !== linearLayers
  ) {
    return null
  }
  return kinds
}

function getPipelineComponentLayout(model, layerBreakdown, ppCount) {
  const pipelineStages = positiveInteger(ppCount)
  const { layers, globalLayers, localLayers, linearLayers } = layerBreakdown
  const baseStageLayers = Math.floor(layers / pipelineStages)
  const extraStageCount = layers % pipelineStages
  const stageSizes = Array.from(
    { length: pipelineStages },
    (_, stageIndex) => baseStageLayers + (stageIndex < extraStageCount ? 1 : 0),
  )
  const kinds = getExplicitLayerKinds(model, layerBreakdown)
  if (kinds) {
    let offset = 0
    const stages = stageSizes.map((stageLayers, stageIndex) => {
      const stageKinds = kinds.slice(offset, offset + stageLayers)
      offset += stageLayers
      return {
        stageIndex,
        layers: stageLayers,
        globalLayers: stageKinds.filter(kind => kind === 'global').length,
        localLayers: stageKinds.filter(kind => kind === 'local').length,
        linearLayers: stageKinds.filter(kind => kind === 'linear').length,
      }
    })
    return { exact: true, conservative: false, stages, stageSizes }
  }

  // Counts alone do not reveal which contiguous PP stage receives each layer
  // kind. Bound every component by the most of that kind that can physically
  // fit on the largest stage. Summing these independent ceilings can overstate
  // a realizable stage, but can never understate the limiting card.
  const largestStageLayers = Math.max(...stageSizes)
  return {
    exact: false,
    conservative: pipelineStages > 1
      && [globalLayers, localLayers, linearLayers].filter(count => count > 0).length > 1,
    stages: null,
    stageSizes,
    componentCeilings: {
      globalLayers: Math.min(globalLayers, largestStageLayers),
      localLayers: Math.min(localLayers, largestStageLayers),
      linearLayers: Math.min(linearLayers, largestStageLayers),
    },
  }
}

function getStageComponentAllocation(
  layout,
  layerBreakdown,
  totals,
  layerTotalGB = 0,
) {
  const components = [
    ['globalGB', 'globalLayers'],
    ['localGB', 'localLayers'],
    ['recurrentGB', 'linearLayers'],
  ]
  const componentValue = (totalKey, countKey, stageCount) => {
    const total = Math.max(0, finiteNumber(totals?.[totalKey], 0))
    const count = layerBreakdown[countKey]
    return count > 0 ? total * stageCount / count : 0
  }

  if (layout.exact) {
    const allocations = layout.stages.map(stage => {
      const allocation = {
        stageIndex: stage.stageIndex,
        stageLayers: stage.layers,
        globalLayers: stage.globalLayers,
        localLayers: stage.localLayers,
        linearLayers: stage.linearLayers,
      }
      for (const [totalKey, countKey] of components) {
        allocation[totalKey] = componentValue(totalKey, countKey, stage[countKey])
      }
      allocation.componentTotalGB = allocation.globalGB
        + allocation.localGB
        + allocation.recurrentGB
      allocation.layerGB = Math.max(0, finiteNumber(layerTotalGB, 0))
        * stage.layers
        / layerBreakdown.layers
      allocation.totalGB = allocation.componentTotalGB + allocation.layerGB
      return allocation
    })
    return allocations.reduce((largest, allocation) =>
      allocation.totalGB > largest.totalGB ? allocation : largest,
    )
  }

  const allocation = {
    stageIndex: null,
    stageLayers: Math.max(...layout.stageSizes),
    ...layout.componentCeilings,
  }
  for (const [totalKey, countKey] of components) {
    allocation[totalKey] = componentValue(
      totalKey,
      countKey,
      layout.componentCeilings[countKey],
    )
  }
  allocation.componentTotalGB = allocation.globalGB
    + allocation.localGB
    + allocation.recurrentGB
  allocation.layerGB = Math.max(0, finiteNumber(layerTotalGB, 0))
    * allocation.stageLayers
    / layerBreakdown.layers
  allocation.totalGB = allocation.componentTotalGB + allocation.layerGB
  return allocation
}

function getLinearStateGB(model, linearLayers, batch, defaultBytes) {
  if (linearLayers <= 0) return 0

  const stateBytes = positiveNumber(model?.linear_state_bytes, defaultBytes)
  let elementsPerLayer = finiteNumber(model?.linear_state_elements_per_layer, NaN)
  if (!Number.isFinite(elementsPerLayer)) {
    if (/rwkv/i.test(String(model?.architecture ?? model?.id ?? ''))) {
      const hidden = positiveNumber(model?.hidden_size, DEFAULT_HIDDEN_SIZE)
      const headSize = positiveNumber(model?.rwkv_head_size, 64)
      elementsPerLayer = hidden * (headSize + 4)
    } else if (isPureRecurrent(model) || Number.isFinite(Number(model?.mamba_ratio))) {
      const hidden = positiveNumber(model?.hidden_size, DEFAULT_HIDDEN_SIZE)
      const expansion = positiveNumber(model?.ssm_expansion, 2)
      const stateSize = positiveNumber(model?.ssm_state_size, 16)
      const convKernel = positiveNumber(model?.ssm_conv_kernel, 4)
      elementsPerLayer = hidden * expansion * (stateSize + convKernel)
    } else {
      const keyHeads = positiveInteger(model?.linear_num_key_heads, 16)
      const valueHeads = positiveInteger(model?.linear_num_value_heads, keyHeads)
      const keyDim = positiveInteger(model?.linear_key_head_dim, 128)
      const valueDim = positiveInteger(model?.linear_value_head_dim, 128)
      const convKernel = positiveInteger(model?.linear_conv_kernel_dim, 4)
      const recurrent = valueHeads * keyDim * valueDim
      const convolution = (2 * keyHeads * keyDim + valueHeads * valueDim) * convKernel
      elementsPerLayer = recurrent + convolution
    }
  }

  return linearLayers * elementsPerLayer * batch * stateBytes / GB
}

function getSequenceKvComponents(model, layerBreakdown, tokens, batch, bytesPerElement) {
  const { globalLayers, localLayers } = layerBreakdown
  if (globalLayers + localLayers === 0) {
    return { globalGB: 0, localGB: 0, totalGB: 0 }
  }

  const kvLoraRank = finiteNumber(model?.kv_lora_rank, NaN)
  const ropeDim = finiteNumber(model?.qk_rope_head_dim, 0)
  if (Number.isFinite(kvLoraRank) && kvLoraRank > 0) {
    const globalTokens = tokens
    const localTokens = Math.min(tokens, positiveInteger(model?.sliding_window, tokens))
    const bytesPerTokenLayer = batch
      * bytesPerElement
      * (kvLoraRank + Math.max(0, ropeDim))
    const globalGB = bytesPerTokenLayer * globalLayers * globalTokens / GB
    const localGB = bytesPerTokenLayer * localLayers * localTokens / GB
    return { globalGB, localGB, totalGB: globalGB + localGB }
  }

  const globalHeads = positiveInteger(model?.global_kv_heads ?? model?.kv_heads)
  const globalDim = positiveInteger(model?.global_head_dim ?? model?.head_dim, DEFAULT_HEAD_DIM)
  const localHeads = positiveInteger(model?.kv_heads)
  const localDim = positiveInteger(model?.head_dim, DEFAULT_HEAD_DIM)
  const localTokens = Math.min(tokens, positiveInteger(model?.sliding_window, tokens))
  let globalGB = 2 * batch * bytesPerElement
    * globalLayers * globalHeads * globalDim * tokens / GB
  let localGB = 2 * batch * bytesPerElement
    * localLayers * localHeads * localDim * localTokens / GB

  // Backward compatibility for catalog entries that have not yet been upgraded
  // with explicit MLA latent dimensions.
  const legacyMlaRatio = finiteNumber(model?.mla_ratio, NaN)
  if (Number.isFinite(legacyMlaRatio) && legacyMlaRatio > 0) {
    globalGB *= legacyMlaRatio
    localGB *= legacyMlaRatio
  }
  return { globalGB, localGB, totalGB: globalGB + localGB }
}

function getSequenceKvGB(model, layerBreakdown, tokens, batch, bytesPerElement) {
  return getSequenceKvComponents(
    model,
    layerBreakdown,
    tokens,
    batch,
    bytesPerElement,
  ).totalGB
}

function getKvBreakdown(model, tokens, batch, bytesPerElement) {
  const layers = getLayerBreakdown(model)
  const sequence = getSequenceKvComponents(
    model,
    layers,
    tokens,
    batch,
    bytesPerElement,
  )
  // Recurrent/SSM state is normally retained in FP32 even when the ordinary
  // attention KV cache is quantized. Architectures with a different cache
  // dtype can override this through `linear_state_bytes`.
  const recurrentGB = getLinearStateGB(model, layers.linearLayers, batch, 4)
  return {
    ...layers,
    globalSequenceGB: sequence.globalGB,
    localSequenceGB: sequence.localGB,
    sequenceGB: sequence.totalGB,
    recurrentGB,
    totalGB: sequence.totalGB + recurrentGB,
    mlaApproximate: Boolean(model?.mla_ratio) && !Number.isFinite(Number(model?.kv_lora_rank)),
    recurrentApproximate: layers.linearLayers > 0
      && !Number.isFinite(Number(model?.linear_state_elements_per_layer))
      && !Number.isFinite(Number(model?.linear_num_value_heads))
      && !Number.isFinite(Number(model?.ssm_state_size))
      && !Number.isFinite(Number(model?.rwkv_head_size)),
  }
}

function getClippedCausalAverage(cachedTokens, newTokens, window = Infinity) {
  const cached = Math.max(0, finiteNumber(cachedTokens, 0))
  const count = positiveInteger(newTokens)
  const finiteWindow = finiteNumber(window, Infinity)
  if (!Number.isFinite(finiteWindow) || finiteWindow <= 0) {
    return cached + (count + 1) / 2
  }

  const cappedWindow = Math.max(1, finiteWindow)
  const growingTokens = clamp(Math.floor(cappedWindow - cached), 0, count)
  const growingSum = growingTokens * cached
    + growingTokens * (growingTokens + 1) / 2
  const cappedSum = (count - growingTokens) * cappedWindow
  return (growingSum + cappedSum) / count
}

function getActivationGB({
  model,
  batch,
  queryTokens,
  attendedTokens,
  flashAttention,
  totalHeads,
  layerBreakdown,
}) {
  const hidden = positiveInteger(model?.hidden_size, DEFAULT_HIDDEN_SIZE)
  const chunk = Math.min(
    positiveInteger(queryTokens),
    positiveInteger(model?.prefill_chunk_size, 2048),
  )
  const elementBytes = 2
  const layerBuffers = batch * chunk * hidden * elementBytes * 8 / GB
  const softmaxLayers = (layerBreakdown?.globalLayers ?? 0)
    + (layerBreakdown?.localLayers ?? 0)
  if (flashAttention || softmaxLayers === 0) return layerBuffers

  const globalAttended = positiveInteger(attendedTokens)
  const localAttended = Math.min(
    globalAttended,
    positiveInteger(model?.sliding_window, globalAttended),
  )
  // Attention-score buffers are reused one layer at a time. Use the largest
  // working set among global and local layers rather than summing every layer.
  const peakAttended = (layerBreakdown?.globalLayers ?? 0) > 0
    ? globalAttended
    : localAttended
  const attentionScores = batch
    * positiveInteger(totalHeads)
    * chunk
    * peakAttended
    * elementBytes
    / GB
  return layerBuffers + attentionScores
}

function getMoeTouchedParams(model, moe, tokenCount) {
  if (model?.type !== 'moe') return positiveNumber(model?.params, 1)
  if (!moe.reliable || !model?.experts || !model?.experts_per_token) {
    return Math.min(model.params, positiveNumber(model.active_params, model.params) * Math.max(1, tokenCount))
  }

  const experts = positiveInteger(model.experts)
  const topK = clamp(positiveInteger(model.experts_per_token), 1, experts)
  const distinctExperts = experts * (1 - Math.pow(1 - topK / experts, Math.max(1, tokenCount)))
  return Math.min(model.params, moe.nonExpertParams + moe.expertParams * distinctExperts / experts)
}

/**
 * Bytes streamed by one synchronized target-model pass.
 *
 * Expert parallelism shards routed experts, but the dense/non-expert portion
 * is replicated once per EP group. Counting only globally unique parameters
 * would therefore overstate the effective aggregate bandwidth of EP layouts.
 */
function getMoeWeightReadGB(model, moe, tokenCount, quantBytes, topology) {
  const touchedParams = getMoeTouchedParams(model, moe, tokenCount)
  if (model?.type !== 'moe' || topology.epCount <= 1 || !moe.reliable) {
    return touchedParams * quantBytes
  }

  const touchedExpertParams = Math.max(0, touchedParams - moe.nonExpertParams)
  return (
    moe.nonExpertParams * topology.epCount
    + touchedExpertParams
  ) * quantBytes
}

function getAttentionDimensions(model, global = false) {
  const fallback = positiveInteger(
    global ? model?.global_head_dim ?? model?.head_dim : model?.head_dim,
    DEFAULT_HEAD_DIM,
  )
  const explicitQk = finiteNumber(
    global ? model?.global_qk_head_dim ?? model?.qk_head_dim : model?.qk_head_dim,
    NaN,
  )
  const nopeDim = finiteNumber(model?.qk_nope_head_dim, NaN)
  const ropeDim = finiteNumber(model?.qk_rope_head_dim, NaN)
  const qkDim = Number.isFinite(explicitQk)
    ? positiveInteger(explicitQk, fallback)
    : Number.isFinite(nopeDim) || Number.isFinite(ropeDim)
      ? positiveInteger(
          Math.max(0, finiteNumber(nopeDim, 0)) + Math.max(0, finiteNumber(ropeDim, 0)),
          fallback,
        )
      : fallback
  const valueDim = positiveInteger(
    global ? model?.global_v_head_dim ?? model?.v_head_dim : model?.v_head_dim,
    fallback,
  )
  return { qkDim, valueDim }
}

function getDecodeAttentionFlops(
  model,
  layerBreakdown,
  totalHeads,
  attendedTokens,
  localAttendedTokens = attendedTokens,
) {
  const globalHeads = positiveInteger(model?.query_heads ?? model?.num_attention_heads ?? totalHeads)
  const globalDimensions = getAttentionDimensions(model, true)
  const localDimensions = getAttentionDimensions(model, false)
  return 2 * globalHeads * (
    layerBreakdown.globalLayers
      * (globalDimensions.qkDim + globalDimensions.valueDim)
      * attendedTokens
    + layerBreakdown.localLayers
      * (localDimensions.qkDim + localDimensions.valueDim)
      * localAttendedTokens
  )
}

function getPrefillAttentionFlopsPerToken({
  model,
  layerBreakdown,
  totalHeads,
  cachedTokens,
  newTokens,
}) {
  const queryHeads = positiveInteger(model?.query_heads ?? model?.num_attention_heads ?? totalHeads)
  const globalDimensions = getAttentionDimensions(model, true)
  const localDimensions = getAttentionDimensions(model, false)
  const averageAttended = Math.max(1, cachedTokens + (newTokens + 1) / 2)
  const localAttended = getClippedCausalAverage(
    cachedTokens,
    newTokens,
    finiteNumber(model?.sliding_window, Infinity),
  )
  return 2 * queryHeads * (
    layerBreakdown.globalLayers
      * (globalDimensions.qkDim + globalDimensions.valueDim)
      * averageAttended
    + layerBreakdown.localLayers
      * (localDimensions.qkDim + localDimensions.valueDim)
      * localAttended
  )
}

function getFlashAttentionBoostRange({ enabled, promptLen, headDim = DEFAULT_HEAD_DIM }) {
  if (!enabled) return { min: 1, mid: 1, max: 1 }
  const dimensionScale = clamp(headDim / DEFAULT_HEAD_DIM, 0.5, 2)
  const logScale = clamp(
    Math.log2(Math.max(2048, promptLen) / 2048) / Math.log2(65536 / 2048),
    0,
    1,
  )
  const range = {
    min: 1.08 + (1.55 - 1.08) * logScale,
    mid: 1.12 + (2.00 - 1.12) * logScale,
    max: 1.18 + (2.45 - 1.18) * logScale,
  }
  return {
    min: Math.max(1, range.min * dimensionScale),
    mid: Math.max(1, range.mid * dimensionScale),
    max: Math.max(1, range.max * dimensionScale),
  }
}

function getBatchSchedulingEfficiency(batch, framework) {
  if ((framework?.schedulingMode ?? 'continuous') !== 'continuous' || batch <= 8) return 1
  return 1 / (1 + (batch - 8) * 0.048)
}

function getAppleDecodeBwScale(gpu) {
  if (gpu?.vendor !== 'apple') return 1
  if (gpu.decodeBwScale != null) return gpu.decodeBwScale
  const id = gpu.id ?? ''
  if (/apple_m[45]/.test(id)) return 1
  if (/apple_m3/.test(id)) return 0.76
  if (/apple_m2/.test(id)) return 0.58
  if (/apple_m1/.test(id)) {
    if (/_max_/.test(id)) return 0.49
    if (/_pro_/.test(id)) return 0.63
  }
  return 1
}

function adjustFramework(gpu, model, framework, pureCpu) {
  let adjusted = framework
  const appliesModelSizeScaling = framework?.modelSizeScaling
    && Array.isArray(framework.modelSizeScaling)
    && (
      (pureCpu && framework.id === 'llamacpp')
      || (!pureCpu && gpu?.vendor !== 'apple')
    )
  if (appliesModelSizeScaling) {
    const scaling = framework.modelSizeScaling.find(item => model.params < item.maxParams)
    if (scaling) {
      adjusted = {
        ...framework,
        decode: scaling.decode ?? framework.decode,
        decodeMin: scaling.decodeMin ?? framework.decodeMin ?? framework.decode,
        decodeMax: scaling.decodeMax ?? framework.decodeMax ?? framework.decode,
      }
    }
  }

  if (!pureCpu && gpu?.vendor === 'nvidia' && framework?.id === 'llamacpp' && model.params < 15) {
    adjusted = {
      ...adjusted,
      decode: Math.max(adjusted.decode, 0.76),
      decodeMin: Math.max(adjusted.decodeMin ?? 0, 0.70),
      decodeMax: Math.max(adjusted.decodeMax ?? 0, 0.84),
    }
  }

  if (!pureCpu && gpu?.vendor === 'apple' && framework?.id === 'llamacpp_metal') {
    const id = gpu.id ?? ''
    const relativeToMlx = /_(max|ultra)_/.test(id) ? 0.74 : /_pro_/.test(id) ? 0.83 : 0.795
    const target = 0.90 * relativeToMlx
    adjusted = {
      ...adjusted,
      decode: target,
      decodeMin: target * 0.90,
      decodeMax: Math.min(0.92, target * 1.08),
    }
  }
  return adjusted
}

function getCpuBandwidth(cpuMemBw) {
  if (typeof cpuMemBw === 'number') {
    return positiveNumber(cpuMemBw, DEFAULT_CPU_MEMORY_BW_GBS)
  }
  return positiveNumber(cpuMemBw?.bw, DEFAULT_CPU_MEMORY_BW_GBS)
}

function getCpuTflops(cpuTflops, quant) {
  const rawValue = typeof cpuTflops === 'object' && cpuTflops !== null
    ? (
        cpuTflops[quant?.id]
        ?? cpuTflops[quant?.flops_key]
        ?? cpuTflops.tflops
      )
    : cpuTflops
  const value = finiteNumber(rawValue, NaN)
  return Number.isFinite(value) && value > 0 ? value : null
}

function getInterconnectLatencySeconds(interconnect) {
  return interconnect?.scope === 'inter' ? 4e-6 : 1e-6
}

function getTpCommunicationSeconds({
  topology,
  interconnect,
  model,
  tokenBatch,
  collectiveCount = 1,
}) {
  if (topology.tpCount <= 1 || !interconnect) return 0
  const tp = topology.tpCount
  const layersPerStage = Math.ceil(positiveInteger(model?.layers) / topology.ppCount)
  const hidden = positiveInteger(model?.hidden_size, DEFAULT_HIDDEN_SIZE)
  const ringBytes = 2 * (tp - 1) / tp * hidden * Math.max(1, tokenBatch) * 2
  const oneWayBw = positiveNumber(interconnect.bw, 1) * GB
  const collective = ringBytes / oneWayBw
    + getInterconnectLatencySeconds(interconnect)
      * 2
      * (tp - 1)
      * positiveInteger(collectiveCount)
  return 2 * layersPerStage * collective
}

function getEpCommunicationSeconds({
  topology,
  interconnect,
  model,
  tokenBatch,
  collectiveCount = 1,
}) {
  if (topology.epCount <= 1 || !interconnect || model?.type !== 'moe') return 0
  const layersPerStage = Math.ceil(positiveInteger(model?.layers) / topology.ppCount)
  const hidden = positiveInteger(model?.hidden_size, DEFAULT_HIDDEN_SIZE)
  const topK = positiveInteger(model?.experts_per_token)
  // With uniformly distributed routes, one rank keeps 1 / EP of its tokens
  // locally and sends the remaining (EP - 1) / EP to its peers. Dividing by
  // EP alone incorrectly makes all-to-all payload shrink toward zero as the
  // expert-parallel group grows.
  const remoteRouteFraction = (topology.epCount - 1) / topology.epCount
  const bytesPerRank = Math.max(1, tokenBatch)
    * topK
    * hidden
    * 2
    * remoteRouteFraction
  const exchange = bytesPerRank / (positiveNumber(interconnect.bw, 1) * GB)
    + getInterconnectLatencySeconds(interconnect) * positiveInteger(collectiveCount)
  return 2 * layersPerStage * exchange
}

function getPpCommunicationSeconds({
  topology,
  interconnect,
  model,
  tokenBatch,
  collectiveCount = 1,
}) {
  if (topology.ppCount <= 1 || !interconnect) return 0
  const hidden = positiveInteger(model?.hidden_size, DEFAULT_HIDDEN_SIZE)
  const transferSeconds = hidden * Math.max(1, tokenBatch) * 2
    / (positiveNumber(interconnect.bw, 1) * GB)
  return (topology.ppCount - 1) * (
    transferSeconds
    + getInterconnectLatencySeconds(interconnect) * positiveInteger(collectiveCount)
  )
}

function getMoeDispatchSeconds({ gpu, framework, model, batch, topology }) {
  if (model?.type !== 'moe' || !model?.experts || model.experts <= 1) return 0
  const activeExperts = positiveInteger(model.experts_per_token)
  const dispatchUs = gpu?.vendor === 'apple' ? framework?.appleMoeDispatchUs : framework?.cudaMoeDispatchUs
  if (dispatchUs == null) return 0
  const modeScale = {
    top1_routed: 0.20,
    routed: 0.55,
    shared_routed: 0.70,
    parallel_dense_routed: 1,
  }[model.moe_execution ?? (activeExperts <= 1 ? 'top1_routed' : 'routed')] ?? 0.55
  const fragmentCount = Math.max(1, activeExperts - 1)
  const fanout = Math.sqrt(Math.max(1, model.experts / 128))
  const batchScale = 1 / Math.sqrt(Math.max(1, batch))
  const backendScale = framework?.id === 'mlx' ? 0.20 : gpu?.vendor === 'apple' ? 0.45 : 1
  const stageLayers = Math.ceil(
    positiveInteger(model.layers) / positiveInteger(topology?.ppCount, 1),
  )
  return stageLayers * fragmentCount * dispatchUs * modeScale * fanout
    * batchScale * backendScale / 1e6
}

/**
 * Estimate memory fit and inference performance.
 *
 * `gpuCount` always means total physical GPUs. `ppCount` divides those GPUs
 * into pipeline stages; `epCount` divides each stage into expert groups; the
 * remaining per-stage dimension is tensor parallelism.
 */
export function calcAll({
  gpu,
  gpuCount = 1,
  interconnect,
  model,
  quant,
  ctx,
  batch,
  promptLen,
  outputLen,
  framework,
  flashAttention = true,
  kvCacheQuant = null,
  prefixCacheHit = 0,
  cpuOffload = false,
  pcieBw = null,
  pcieWidth = null,
  pureCpu = false,
  cpuMemBw = null,
  cpuTflops = null,
  sysRam = null,
  nglCount = null,
  speculativeDecoding = false,
  acceptanceRate = 0.7,
  draftLen = 4,
  draftModelParams = null,
  ppCount = 1,
  epCount = 1,
  imageCount = 0,
}) {
  if (!gpu || !model || !quant || !framework) {
    throw new TypeError('calcAll requires gpu, model, quant, and framework')
  }

  const workloadInputOk = [ctx, batch, promptLen, outputLen]
    .every(isPositiveIntegerValue)
  const safeCtx = positiveInteger(ctx, 4096)
  const safeBatch = positiveInteger(batch)
  const safePromptLen = positiveInteger(promptLen, 512)
  const safeOutputLen = positiveInteger(outputLen, 128)
  const safeImageCount = Math.max(0, Math.round(finiteNumber(imageCount, 0)))
  const effectiveCpuOffload = Boolean(cpuOffload) && !pureCpu
  const topology = getTopology(gpuCount, ppCount, epCount, model)
  const singleDeviceMemory = Boolean(gpu.unifiedMemory || gpu.sharedMemory)
  // Unified/shared-memory products describe one physical memory pool, not a
  // stackable accelerator card. This also catches an invalid mixed-slot
  // aggregate whose shared-memory semantics were propagated below.
  const memoryDeviceTopologyOk = !singleDeviceMemory
    || (
      topology.totalGpuCount === 1
      && !gpu.invalidMemoryMix
    )
  const isEP = topology.epCount > 1
  const usesLlamaCppGpuLayers = !pureCpu && framework.id === 'llamacpp'
  const isMoeOffload = effectiveCpuOffload && model.type === 'moe'
  const unsupportedDenseOffload = effectiveCpuOffload && model.type !== 'moe'
  const visionParameters = getVisionParameterInfo(model)
  const decoderModel = visionParameters.decoderParams === model.params
    ? model
    : { ...model, params: visionParameters.decoderParams }
  const auxiliaryResidentParams = Math.max(
    0,
    visionParameters.residentParams - visionParameters.decoderParams,
  )
  const moe = getMoeInfo(decoderModel)
  const decompositionOk = !(isEP || isMoeOffload) || moe.reliable
  const runtimeCompatibility = getRuntimeCompatibility({
    framework,
    gpu,
    model,
    quant,
    kvCacheQuant,
    // The shared runtime matrix models TP, PP, and EP as independent
    // dimensions. calcAll receives the physical total, so pass the resolved
    // per-stage TP width instead of treating every GPU as a TP rank.
    gpuCount: topology.tpCount,
    ppCount: topology.requestedPp,
    epCount: topology.requestedEp,
    cpuOffload: effectiveCpuOffload,
    pureCpu,
    speculativeDecoding,
    forCommand: false,
  })
  const runtimeCompatibilityReasonCodes = runtimeCompatibility.reasons
    .map(item => item.code)
  const runtimeReasonCodes = new Set(runtimeCompatibilityReasonCodes)
  const frameworkOk = ![
    'unknown-framework',
    'vendor-unsupported',
    'amd-hardware-unsupported',
  ].some(code => runtimeReasonCodes.has(code))
  const speculativeOk = !runtimeReasonCodes.has('speculative-unsupported')
  const kvCacheSupported = !runtimeCompatibilityReasonCodes.some(code =>
    code === 'kv-format-unsupported'
    || code === 'kv-format-hardware-unsupported'
  )
  const weightQuantSupported = !runtimeReasonCodes.has('weight-format-unsupported')
  const runtimeTopologySupported = !runtimeCompatibilityReasonCodes.some(code =>
    [
      'tp-unsupported',
      'pp-unsupported',
      'ep-unsupported',
      'ep-model-unsupported',
    ].includes(code)
  )
  const localInferenceSupported = !runtimeReasonCodes.has('local-inference-unavailable')
  const runtimeConfigurationSupported = runtimeCompatibility.supported
  const cpuParallelOk = !pureCpu || (topology.requestedPp === 1 && topology.requestedEp === 1)
  const offloadMemoryArchitectureOk = !effectiveCpuOffload
    || !(gpu.unifiedMemory || gpu.sharedMemory)
  // Host offload combined with PP/EP requires per-node RAM, PCIe, and routing
  // topology that this UI does not collect. Reject those combinations instead
  // of double-counting transfers or reporting an optimistic result.
  const offloadParallelOk = !effectiveCpuOffload
    || (topology.requestedPp === 1 && topology.requestedEp === 1)
  const modeOk = !unsupportedDenseOffload
    && frameworkOk
    && speculativeOk
    && cpuParallelOk
    && offloadParallelOk
    && offloadMemoryArchitectureOk

  const gpuAdjustedFramework = adjustFramework(gpu, model, framework, false)
  // CPU efficiency must be independent of the display GPU and must not inherit
  // CUDA/Metal-specific calibration used by a hybrid GPU portion.
  const cpuAdjustedFramework = adjustFramework({ vendor: 'cpu' }, model, framework, true)
  const adjustedFramework = pureCpu ? cpuAdjustedFramework : gpuAdjustedFramework
  const decodeRange = normalizeRange(getFrameworkRange(adjustedFramework, 'decode'))
  const cpuDecodeRange = normalizeRange(getFrameworkRange(cpuAdjustedFramework, 'decode'))
  const prefillRange = normalizeRange(getFrameworkRange(adjustedFramework, 'prefill'))
  const totalHeads = getTotalHeads(model) ?? positiveInteger(model.kv_heads)
  const attentionType = getAttentionType(model)
  const attentionSummary = getAttentionSummary(model)
  const layerBreakdown = getLayerBreakdown(model)
  const modelLayers = positiveInteger(model.layers)
  const averageStageLayers = modelLayers / topology.ppCount
  const largestStageLayers = Math.ceil(averageStageLayers)
  const ppStageImbalance = largestStageLayers / averageStageLayers
  const pipelineComponentLayout = getPipelineComponentLayout(
    model,
    layerBreakdown,
    topology.ppCount,
  )

  const prefixHitRatio = clamp(finiteNumber(prefixCacheHit, 0) / 100, 0, 0.99)
  const cachedTextTokens = Math.round(safePromptLen * prefixHitRatio)
  const uncachedTextTokens = Math.max(1, safePromptLen - cachedTextTokens)
  const tags = Array.isArray(model.tags) ? model.tags.map(tag => String(tag).toLowerCase()) : []
  const supportsImages = model.vision_seq_tokens != null
    || tags.includes('vision')
    || tags.includes('image')
    || (tags.includes('multimodal') && !tags.includes('audio'))
  const visionTokensWereInferred = safeImageCount > 0
    && supportsImages
    && !Number.isFinite(Number(model.vision_seq_tokens))
  const visionAccountingApproximate = safeImageCount > 0
    && supportsImages
    && (
      visionParameters.visionAccountingApproximate
      || visionParameters.visionEncoderParams <= 0
    )
  const visionTokensPerImage = supportsImages
    ? Math.max(1, Math.round(finiteNumber(model.vision_seq_tokens, 1024)))
    : 0
  const visionPatchTokens = visionTokensPerImage * safeImageCount
  const effectivePromptLen = uncachedTextTokens + visionPatchTokens
  const promptTokensWithVision = safePromptLen + visionPatchTokens
  const workloadTokens = promptTokensWithVision + safeOutputLen
  const modelContextOk = !model.max_ctx || safeCtx <= model.max_ctx
  const workloadContextOk = workloadTokens <= safeCtx
  const contextOk = modelContextOk && workloadContextOk
  const avgDecodeSeqLen = clamp(
    getClippedCausalAverage(
      promptTokensWithVision - 1,
      safeOutputLen,
      Infinity,
    ),
    1,
    safeCtx,
  )
  const avgLocalDecodeSeqLen = getClippedCausalAverage(
    promptTokensWithVision - 1,
    safeOutputLen,
    finiteNumber(model?.sliding_window, Infinity),
  )

  const sizingFramework = pureCpu ? { id: 'llamacpp' } : framework
  const quantBytes = getQuantBytes(quant, gpu, sizingFramework)
  const kvBytesPerElement = positiveNumber(kvCacheQuant?.bytes, positiveNumber(quant.kv_bytes, 2))
  const resolvedKvCacheLabel = kvCacheQuant?.id && kvCacheQuant.id !== 'auto'
    ? kvCacheQuant.label
    : kvBytesPerElement >= 4
      ? 'Auto (FP32)'
      : 'Auto (FP16/BF16)'
  const targetWeightGB = visionParameters.residentParams * quantBytes
  const targetKv = getKvBreakdown(model, safeCtx, safeBatch, kvBytesPerElement)

  const safeDraftLen = clamp(positiveInteger(draftLen, 4), 1, 32)
  const alpha = clamp(finiteNumber(acceptanceRate, 0.7), 0.01, 0.999)
  const inferredDraftParams = Math.min(
    7,
    Math.max(
      0.5,
      positiveNumber(decoderModel.active_params, decoderModel.params) * 0.125,
    ),
  )
  const suppliedDraftParams = typeof draftModelParams === 'object'
    ? finiteNumber(draftModelParams?.params, NaN)
    : finiteNumber(draftModelParams, NaN)
  const effectiveDraftParams = speculativeDecoding
    ? positiveNumber(suppliedDraftParams, inferredDraftParams)
    : 0
  const draftWasInferred = speculativeDecoding && !Number.isFinite(suppliedDraftParams)
  const draftWeightGB = effectiveDraftParams * quantBytes
  const draftScale = speculativeDecoding
    ? clamp(Math.sqrt(effectiveDraftParams / positiveNumber(model.params, 1)), 0, 1)
    : 0
  const draftKvGB = targetKv.totalGB * draftScale

  const activationGB = getActivationGB({
    model,
    batch: safeBatch,
    queryTokens: effectivePromptLen,
    attendedTokens: promptTokensWithVision,
    flashAttention,
    totalHeads,
    layerBreakdown,
  }) * (1 + draftScale)

  // llama.cpp's automatic NGL is a fit decision, not "half the layers".
  // Estimate it against the limiting per-card VRAM so mixed GPU sets retain
  // the same conservative equal-shard assumption used by the main calculator.
  const autoNgl = usesLlamaCppGpuLayers
    ? (() => {
        const perCardVramBudget = positiveNumber(gpu.vram, 0.001)
          * clamp(finiteNumber(gpu.usableRatio, 1), 0.01, 1)
        const tp = Math.max(1, topology.tpCount)
        const kvShardCount = Math.min(
          tp,
          Math.max(1, positiveInteger(model.kv_heads ?? model.query_heads ?? totalHeads)),
        )
        const fullGpuTargetWeightGB = isMoeOffload
          ? (moe.nonExpertParams + auxiliaryResidentParams) * quantBytes
          : targetWeightGB
        for (let layers = modelLayers; layers >= 0; layers -= 1) {
          const ratio = layers / modelLayers
          const perCardWeight = (
            fullGpuTargetWeightGB * ratio + draftWeightGB
          ) / tp
          const perCardKv = (
            targetKv.totalGB * ratio + draftKvGB
          ) / kvShardCount
          const perCardActivation = activationGB / tp
          const perCardRuntime = Math.max(0.75, Math.min(perCardWeight * 0.03, 4))
          if (
            perCardWeight + perCardKv + perCardActivation + perCardRuntime
            <= perCardVramBudget
          ) {
            return layers
          }
        }
        return 0
      })()
    : null
  const effectiveNgl = pureCpu
    ? 0
    : usesLlamaCppGpuLayers
      ? clamp(
          nglCount == null ? autoNgl : Math.round(finiteNumber(nglCount, autoNgl)),
          0,
          modelLayers,
        )
      : modelLayers
  const gpuLayerRatio = pureCpu ? 0 : effectiveNgl / modelLayers
  const isLlamaCppHybrid = usesLlamaCppGpuLayers && effectiveNgl < modelLayers
  const gpuComputeRequired = !pureCpu && (!isLlamaCppHybrid || gpuLayerRatio > 0)
  const decodeTflopsPerCard = getDecodeTflops(gpu, quant)
  const prefillTflopsPerCard = getPrefillTflops(gpu, quant)
  const computePrecisionSupported = !gpuComputeRequired || (
    Number.isFinite(decodeTflopsPerCard)
    && decodeTflopsPerCard > 0
    && Number.isFinite(prefillTflopsPerCard)
    && prefillTflopsPerCard > 0
  )
  // Keep unsupported configurations numerically finite for charts/exports,
  // while fitOk and a dedicated flag make the unsupported precision explicit.
  const totalDecodeTflops = (Number.isFinite(decodeTflopsPerCard)
    ? decodeTflopsPerCard
    : 1e-12) * topology.totalGpuCount
  const totalPrefillTflops = (Number.isFinite(prefillTflopsPerCard)
    ? prefillTflopsPerCard
    : 1e-12) * topology.totalGpuCount
  const resolvedCpuTflops = getCpuTflops(cpuTflops, quant)
  const cpuComputeNeeded = pureCpu || (isLlamaCppHybrid && gpuLayerRatio < 1)
  const cpuComputeIsUpperBound = cpuComputeNeeded && resolvedCpuTflops == null

  const decoderWeightGB = visionParameters.decoderParams * quantBytes
  const auxiliaryResidentWeightGB = auxiliaryResidentParams * quantBytes
  const gpuDecoderWeightGB = pureCpu
    ? 0
    : isMoeOffload
      ? moe.nonExpertParams * quantBytes * gpuLayerRatio
      : decoderWeightGB * gpuLayerRatio
  const gpuAuxiliaryWeightGB = pureCpu
    ? 0
    : auxiliaryResidentWeightGB * gpuLayerRatio
  const gpuTargetWeightGB = gpuDecoderWeightGB + gpuAuxiliaryWeightGB
  const gpuTargetGlobalSequenceKvGB = pureCpu
    ? 0
    : targetKv.globalSequenceGB * gpuLayerRatio
  const gpuTargetLocalSequenceKvGB = pureCpu
    ? 0
    : targetKv.localSequenceGB * gpuLayerRatio
  const gpuTargetSequenceKvGB = gpuTargetGlobalSequenceKvGB
    + gpuTargetLocalSequenceKvGB
  const gpuTargetRecurrentKvGB = pureCpu ? 0 : targetKv.recurrentGB * gpuLayerRatio
  const gpuTargetKvGB = gpuTargetSequenceKvGB + gpuTargetRecurrentKvGB
  const gpuDraftWeightGB = pureCpu ? 0 : draftWeightGB
  const gpuDraftGlobalSequenceKvGB = pureCpu
    ? 0
    : targetKv.globalSequenceGB * draftScale
  const gpuDraftLocalSequenceKvGB = pureCpu
    ? 0
    : targetKv.localSequenceGB * draftScale
  const gpuDraftSequenceKvGB = gpuDraftGlobalSequenceKvGB
    + gpuDraftLocalSequenceKvGB
  const gpuDraftRecurrentKvGB = pureCpu ? 0 : targetKv.recurrentGB * draftScale
  const gpuDraftKvGB = gpuDraftSequenceKvGB + gpuDraftRecurrentKvGB
  const weightGB = gpuTargetWeightGB + gpuDraftWeightGB
  const kvGB = gpuTargetKvGB + gpuDraftKvGB

  const mlaCacheIsReplicated = finiteNumber(model.kv_lora_rank, NaN) > 0
  const globalSequenceKvShardLimit = Math.max(
    1,
    mlaCacheIsReplicated
      // MLA stores one compressed latent vector shared by every query head.
      // Plain tensor parallelism therefore replicates this cache; sharding it
      // requires a distinct context-parallel mode that this UI does not expose.
      ? 1
      : positiveInteger(model.global_kv_heads ?? model.kv_heads, totalHeads),
  )
  const localSequenceKvShardLimit = Math.max(
    1,
    mlaCacheIsReplicated ? 1 : positiveInteger(model.kv_heads, totalHeads),
  )
  const globalSequenceKvShardCount = Math.min(
    topology.tpCount,
    globalSequenceKvShardLimit,
  )
  const localSequenceKvShardCount = Math.min(
    topology.tpCount,
    localSequenceKvShardLimit,
  )
  const recurrentKvShardLimit = Math.max(
    1,
    positiveInteger(
      model.linear_num_value_heads
      ?? model.query_heads
      ?? totalHeads,
    ),
  )
  const recurrentKvShardCount = Math.min(topology.tpCount, recurrentKvShardLimit)
  const globalSequenceKvReplicationFactor = topology.epCount
    * topology.tpCount
    / globalSequenceKvShardCount
  const localSequenceKvReplicationFactor = topology.epCount
    * topology.tpCount
    / localSequenceKvShardCount
  const recurrentKvReplicationFactor = topology.epCount
    * topology.tpCount
    / recurrentKvShardCount
  const fullRankDecoderWeightGB = (() => {
    if (pureCpu) return 0
    if (isEP && model.type === 'moe') {
      return (
        moe.nonExpertParams / topology.tpCount
        + moe.expertParams / (topology.epCount * topology.tpCount)
      ) * quantBytes
    }
    return gpuDecoderWeightGB / topology.tpCount
  })()
  // Vision encoders/projectors are not decoder layers and therefore cannot be
  // divided by the decoder's PP layer ratio. Conservatively reserve the whole
  // TP shard on the limiting stage; otherwise PP under-reports VLM fit.
  const fullRankAuxiliaryWeightGB = pureCpu
    ? 0
    : gpuAuxiliaryWeightGB / topology.tpCount
  // The draft model is dense: TP/PP shard it, while independent EP groups
  // retain replicas just like the target model's non-expert layers.
  const fullRankDraftWeightGB = pureCpu
    ? 0
    : gpuDraftWeightGB / topology.tpCount
  const perCardKvStage = getStageComponentAllocation(
    pipelineComponentLayout,
    layerBreakdown,
    {
      globalGB: (
        gpuTargetGlobalSequenceKvGB + gpuDraftGlobalSequenceKvGB
      ) / globalSequenceKvShardCount,
      localGB: (
        gpuTargetLocalSequenceKvGB + gpuDraftLocalSequenceKvGB
      ) / localSequenceKvShardCount,
      recurrentGB: (
        gpuTargetRecurrentKvGB + gpuDraftRecurrentKvGB
      ) / recurrentKvShardCount,
    },
    fullRankDecoderWeightGB + fullRankDraftWeightGB,
  )
  const selectedStageLayerRatio = perCardKvStage.stageLayers / modelLayers
  const perCardTargetWeightGB = fullRankDecoderWeightGB * selectedStageLayerRatio
    + fullRankAuxiliaryWeightGB
  const perCardDraftWeightGB = fullRankDraftWeightGB * selectedStageLayerRatio
  const perCardWeightGB = perCardKvStage.layerGB + fullRankAuxiliaryWeightGB
  const perCardGlobalSequenceKvGB = perCardKvStage.globalGB
  const perCardLocalSequenceKvGB = perCardKvStage.localGB
  const perCardRecurrentKvGB = perCardKvStage.recurrentGB
  const perCardKvGB = perCardKvStage.componentTotalGB
  const perCardActivationGB = pureCpu ? 0 : activationGB / Math.max(1, topology.tpCount)
  const perCardOverheadGB = pureCpu
    ? 0
    : Math.max(0.75, Math.min(perCardWeightGB * 0.03, 4))
  const overheadGB = perCardOverheadGB + perCardActivationGB
  const perCardNeeded = perCardWeightGB + perCardKvGB + perCardActivationGB + perCardOverheadGB
  const requestedSysRam = finiteNumber(sysRam, NaN)
  const availableSysRamGB = Number.isFinite(requestedSysRam)
    ? Math.max(0, requestedSysRam * 0.90)
    : null
  const configuredGpuVram = finiteNumber(gpu.vram, 0)
  const allocatedGpuVram = gpu.sharedMemory
    && configuredGpuVram <= 0
    && availableSysRamGB != null
    ? availableSysRamGB
    : configuredGpuVram
  const perCardVram = positiveNumber(allocatedGpuVram, 0.001)
    * clamp(finiteNumber(gpu.usableRatio, 1), 0.01, 1)
  const clusterNeeded = perCardNeeded * topology.totalGpuCount
  const totalVram = perCardVram * topology.totalGpuCount

  const cpuWeightGB = pureCpu
    ? targetWeightGB + draftWeightGB
    : isLlamaCppHybrid || isMoeOffload
      ? Math.max(0, targetWeightGB - gpuTargetWeightGB)
      : 0
  const cpuKvGB = pureCpu
    ? targetKv.totalGB + draftKvGB
    : isLlamaCppHybrid
      ? targetKv.totalGB * (1 - gpuLayerRatio)
      : 0
  const cpuRuntimeGB = cpuWeightGB + cpuKvGB > 0
    ? pureCpu
      ? Math.max(0.75, Math.min(cpuWeightGB * 0.03, 4)) + activationGB
      : Math.max(0.5, Math.min(cpuWeightGB * 0.01, 2))
        + (isLlamaCppHybrid ? activationGB : 0)
    : 0
  const cpuRamNeededGB = cpuWeightGB + cpuKvGB + cpuRuntimeGB
  const sharedSystemMemory = Boolean(gpu.sharedMemory) && !pureCpu
  const sharedRuntimeGB = Math.max(
    cpuRuntimeGB,
    activationGB + perCardOverheadGB,
  )
  const sharedSystemRamNeededGB = sharedSystemMemory
    ? targetWeightGB
      + draftWeightGB
      + targetKv.totalGB
      + draftKvGB
      + sharedRuntimeGB
    : 0
  const systemRamNeededGB = sharedSystemMemory
    ? sharedSystemRamNeededGB
    : cpuRamNeededGB
  const usesSystemRam = systemRamNeededGB > 0
  let ramOk = !usesSystemRam || (
    availableSysRamGB != null
    && systemRamNeededGB <= availableSysRamGB
  )
  let vramOk = pureCpu || perCardNeeded <= perCardVram
  const unifiedMemory = Boolean(gpu.unifiedMemory || gpu.sharedMemory)
  const sharedAllocationGB = sharedSystemMemory
    ? allocatedGpuVram * topology.totalGpuCount
    : null
  const sharedAllocationExceedsRam = sharedSystemMemory
    && availableSysRamGB != null
    && sharedAllocationGB > availableSysRamGB
  const sharedAllocationExcessGB = sharedAllocationExceedsRam
    ? sharedAllocationGB - availableSysRamGB
    : 0
  const sharedPoolAvailableGB = sharedSystemMemory
    ? Math.min(totalVram, availableSysRamGB ?? 0)
    : null
  if (gpu.unifiedMemory && cpuRamNeededGB > 0 && !pureCpu) {
    const systemAvailable = availableSysRamGB ?? totalVram
    const unifiedAvailable = gpu.unifiedMemory
      ? Math.min(totalVram, systemAvailable)
      : systemAvailable
    const unifiedNeeded = clusterNeeded + cpuRamNeededGB
    const unifiedOk = unifiedNeeded <= unifiedAvailable
    vramOk = vramOk && unifiedOk
    ramOk = ramOk && unifiedOk
  }
  if (sharedSystemMemory) {
    // Integrated GPUs and the CPU consume the same physical DRAM pool. The
    // user-selected GPU allocation and usable system RAM are both hard caps,
    // while resident weights/KV/runtime are counted only once in that pool.
    const allocationFitsSystemRam = availableSysRamGB != null
      && !sharedAllocationExceedsRam
    const sharedOk = allocationFitsSystemRam
      && sharedSystemRamNeededGB <= sharedPoolAvailableGB
    vramOk = vramOk && sharedOk
    ramOk = ramOk && sharedOk
  }
  const fitOk = vramOk
    && ramOk
    && workloadInputOk
    && contextOk
    && topology.topologyOk
    && memoryDeviceTopologyOk
    && decompositionOk
    && modeOk
    && kvCacheSupported
    && runtimeConfigurationSupported
    && computePrecisionSupported

  const cpuBw = getCpuBandwidth(cpuMemBw)
  const gpuMemoryBw = sharedSystemMemory
    ? Math.min(positiveNumber(gpu.bw, 1), cpuBw)
    : positiveNumber(gpu.bw, 1)
  const rawGpuBwPerCard = gpuMemoryBw
    * clamp(finiteNumber(gpu.bwUtilization, 0.80), 0.01, 1)
    * getAppleDecodeBwScale(gpu)
  const totalGpuBw = rawGpuBwPerCard
    * (sharedSystemMemory ? 1 : topology.totalGpuCount)
  const pcieOneWayBw = positiveNumber(pcieBw?.bw, 1)
    * clamp(finiteNumber(pcieWidth?.ratio, 0.5), 0.01, 1)
  // TP shards can use one host link per GPU concurrently. System DDR remains
  // a shared ceiling, so adding GPUs cannot exceed the selected RAM bandwidth.
  const aggregatePcieOneWayBw = pcieOneWayBw * topology.totalGpuCount
  const offloadTransferBw = Math.min(cpuBw, aggregatePcieOneWayBw) * 0.85
  const activeParams = decoderModel.type === 'moe'
    ? positiveNumber(decoderModel.active_params, decoderModel.params)
    : positiveNumber(decoderModel.params, 1)
  const decodeAttentionFlops = getDecodeAttentionFlops(
    model,
    layerBreakdown,
    totalHeads,
    avgDecodeSeqLen,
    avgLocalDecodeSeqLen,
  )
  const decodeFlopsPerToken = 2 * activeParams * GB + decodeAttentionFlops
  const targetWeightReadGB = getMoeWeightReadGB(
    decoderModel,
    moe,
    safeBatch,
    quantBytes,
    topology,
  )
  const decodeGlobalKvAtAverage = getSequenceKvComponents(
    model,
    layerBreakdown,
    avgDecodeSeqLen,
    safeBatch,
    kvBytesPerElement,
  )
  const decodeLocalKvAtAverage = getSequenceKvComponents(
    model,
    layerBreakdown,
    avgLocalDecodeSeqLen,
    safeBatch,
    kvBytesPerElement,
  )
  const decodeRecurrentKv = getKvBreakdown(
    model,
    1,
    safeBatch,
    kvBytesPerElement,
  )
  // Every decode step reads the existing sequence cache and writes the new
  // token's K/V entry. Recurrent state is fixed-size and is read and written.
  const sequenceKvWritePerStep = getSequenceKvComponents(
    model,
    layerBreakdown,
    1,
    safeBatch,
    kvBytesPerElement,
  )
  const logicalGlobalSequenceKvTrafficGB = decodeGlobalKvAtAverage.globalGB
    + sequenceKvWritePerStep.globalGB
  const logicalLocalSequenceKvTrafficGB = decodeLocalKvAtAverage.localGB
    + sequenceKvWritePerStep.localGB
  const logicalSequenceKvTrafficGB = logicalGlobalSequenceKvTrafficGB
    + logicalLocalSequenceKvTrafficGB
  const logicalRecurrentKvTrafficGB = 2 * decodeRecurrentKv.recurrentGB
  const logicalKvTrafficGB = logicalSequenceKvTrafficGB + logicalRecurrentKvTrafficGB
  const aggregateGpuGlobalSequenceKvTrafficGB = logicalGlobalSequenceKvTrafficGB
    * globalSequenceKvReplicationFactor
  const aggregateGpuLocalSequenceKvTrafficGB = logicalLocalSequenceKvTrafficGB
    * localSequenceKvReplicationFactor
  const aggregateGpuRecurrentKvTrafficGB = logicalRecurrentKvTrafficGB
    * recurrentKvReplicationFactor
  const aggregateGpuKvTrafficGB = (
    aggregateGpuGlobalSequenceKvTrafficGB
    + aggregateGpuLocalSequenceKvTrafficGB
    + aggregateGpuRecurrentKvTrafficGB
  )
  const decodeKvTrafficStage = getStageComponentAllocation(
    pipelineComponentLayout,
    layerBreakdown,
    {
      globalGB: aggregateGpuGlobalSequenceKvTrafficGB,
      localGB: aggregateGpuLocalSequenceKvTrafficGB,
      recurrentGB: aggregateGpuRecurrentKvTrafficGB,
    },
    targetWeightReadGB,
  )
  const pipelineAggregateGpuGlobalSequenceKvTrafficGB = decodeKvTrafficStage.globalGB
    * topology.ppCount
  const pipelineAggregateGpuLocalSequenceKvTrafficGB = decodeKvTrafficStage.localGB
    * topology.ppCount
  const pipelineAggregateGpuRecurrentKvTrafficGB = decodeKvTrafficStage.recurrentGB
    * topology.ppCount
  const pipelineAggregateGpuKvTrafficGB = decodeKvTrafficStage.componentTotalGB
    * topology.ppCount
  const aggregateGpuDraftWeightReadGB = draftWeightGB * topology.epCount
  const modeledKvTrafficGB = pureCpu
    ? logicalKvTrafficGB
    : isLlamaCppHybrid
      ? pipelineAggregateGpuKvTrafficGB * gpuLayerRatio
        + logicalKvTrafficGB * (1 - gpuLayerRatio)
      : pipelineAggregateGpuKvTrafficGB
  const decodeBytesPerStep = targetWeightReadGB * (pureCpu ? 1 : ppStageImbalance)
    + modeledKvTrafficGB
  const batchSchedulingEfficiency = getBatchSchedulingEfficiency(safeBatch, adjustedFramework)
  const ppBubbleEff = topology.ppCount > 1
    ? safeBatch / (safeBatch + topology.ppCount - 1)
    : 1
  const gpuStageTimeFactor = pureCpu ? 1 : ppStageImbalance

  function getTargetCommunicationSeconds(tokenBatch, collectiveCount = 1) {
    if (pureCpu) return 0
    const gpuFraction = isLlamaCppHybrid ? gpuLayerRatio : 1
    return gpuFraction * (getTpCommunicationSeconds({
      topology,
      interconnect,
      model,
      tokenBatch,
      collectiveCount,
    }) + getEpCommunicationSeconds({
      topology,
      interconnect,
      model,
      tokenBatch,
      collectiveCount,
    }) + getPpCommunicationSeconds({
      topology,
      interconnect,
      model,
      tokenBatch,
      collectiveCount,
    }))
  }

  const commonCommSeconds = getTargetCommunicationSeconds(safeBatch)
  const moeDispatchSeconds = pureCpu
    ? 0
    : getMoeDispatchSeconds({
        gpu,
        framework: adjustedFramework,
        model,
        batch: safeBatch,
        topology,
      })

  function getTargetMemorySeconds(
    decodeFactor,
    cpuDecodeFactor = decodeFactor,
    verificationTokens = 1,
  ) {
    const factor = positiveNumber(decodeFactor, 1)
    const cpuFactor = positiveNumber(cpuDecodeFactor, factor)
    const weightReadGB = getMoeWeightReadGB(
      decoderModel,
      moe,
      safeBatch * verificationTokens,
      quantBytes,
      topology,
    )
    const logicalTrafficGB = logicalKvTrafficGB * verificationTokens
    const stageTraffic = getStageComponentAllocation(
      pipelineComponentLayout,
      layerBreakdown,
      {
        globalGB: aggregateGpuGlobalSequenceKvTrafficGB * verificationTokens,
        localGB: aggregateGpuLocalSequenceKvTrafficGB * verificationTokens,
        recurrentGB: aggregateGpuRecurrentKvTrafficGB * verificationTokens,
      },
      weightReadGB,
    )
    const stageWeightTrafficGB = stageTraffic.layerGB * topology.ppCount
    const aggregateGpuTrafficGB = stageTraffic.componentTotalGB * topology.ppCount
    const selectedStageFactor = stageTraffic.stageLayers
      / modelLayers
      * topology.ppCount
    if (pureCpu) {
      return (weightReadGB + logicalTrafficGB) / (cpuBw * cpuFactor)
    }
    if (isLlamaCppHybrid) {
      if (isMoeOffload) {
        const touchedParams = getMoeTouchedParams(
          decoderModel,
          moe,
          safeBatch * verificationTokens,
        )
        const expertParams = Math.max(0, touchedParams - moe.nonExpertParams)
        const expertSeconds = expertParams
          * quantBytes
          * selectedStageFactor
          * gpuLayerRatio
          / offloadTransferBw
        const gpuSeconds = gpuLayerRatio > 0
          ? (
              moe.nonExpertParams
              * quantBytes
              * selectedStageFactor
              * gpuLayerRatio
              + aggregateGpuTrafficGB * gpuLayerRatio
            ) / (totalGpuBw * factor)
          : 0
        const cpuSeconds = (1 - gpuLayerRatio) > 0
          ? (
              weightReadGB * (1 - gpuLayerRatio)
              + logicalTrafficGB * (1 - gpuLayerRatio)
            ) / (cpuBw * cpuFactor)
          : 0
        return expertSeconds + gpuSeconds + cpuSeconds
      }
      const gpuSeconds = gpuLayerRatio > 0
        ? (
            weightReadGB * gpuLayerRatio
            + aggregateGpuTrafficGB * gpuLayerRatio
          ) / (totalGpuBw * factor)
        : 0
      const cpuSeconds = (1 - gpuLayerRatio) > 0
        ? (
            weightReadGB * (1 - gpuLayerRatio)
            + logicalTrafficGB * (1 - gpuLayerRatio)
          ) / (cpuBw * cpuFactor)
        : 0
      return gpuSeconds + cpuSeconds
    }
    if (isMoeOffload) {
      const touchedParams = getMoeTouchedParams(
        decoderModel,
        moe,
        safeBatch * verificationTokens,
      )
      const expertParams = Math.max(0, touchedParams - moe.nonExpertParams)
      const expertSeconds = expertParams
        * quantBytes
        * selectedStageFactor
        / offloadTransferBw
      const gpuSeconds = (
        moe.nonExpertParams * quantBytes * selectedStageFactor
        + aggregateGpuTrafficGB
      )
        / (totalGpuBw * factor)
      return expertSeconds + gpuSeconds
    }
    return (stageWeightTrafficGB + aggregateGpuTrafficGB)
      / (totalGpuBw * factor)
  }

  function getTargetComputeSeconds(
    decodeFactor,
    cpuDecodeFactor = decodeFactor,
    verificationTokens = 1,
  ) {
    const gpuUtilization = clamp(
      (positiveNumber(decodeFactor, 1) + prefillRange.mid) / 2,
      0.01,
      1,
    )
    const cpuUtilization = clamp(
      (positiveNumber(cpuDecodeFactor, decodeFactor) + prefillRange.mid) / 2,
      0.01,
      1,
    )
    const totalFlops = decodeFlopsPerToken * safeBatch * verificationTokens
    if (pureCpu) {
      return resolvedCpuTflops == null
        ? 0
        : totalFlops / (resolvedCpuTflops * 1e12 * cpuUtilization)
    }
    if (isLlamaCppHybrid) {
      const gpuSeconds = gpuLayerRatio > 0
        ? totalFlops * gpuLayerRatio
          / (totalDecodeTflops * 1e12 * gpuUtilization)
        : 0
      const cpuSeconds = resolvedCpuTflops != null && gpuLayerRatio < 1
        ? totalFlops * (1 - gpuLayerRatio)
          / (resolvedCpuTflops * 1e12 * cpuUtilization)
        : 0
      return (gpuSeconds + cpuSeconds) * gpuStageTimeFactor
    }
    return totalFlops
      / (totalDecodeTflops * 1e12 * gpuUtilization)
      * gpuStageTimeFactor
  }

  function getDraftStep(decodeFactor) {
    if (!speculativeDecoding) {
      return { seconds: 0, memorySeconds: 0, computeSeconds: 0 }
    }
    const draftStageTraffic = pureCpu
      ? null
      : getStageComponentAllocation(
          pipelineComponentLayout,
          layerBreakdown,
          {
            globalGB: aggregateGpuGlobalSequenceKvTrafficGB * draftScale,
            localGB: aggregateGpuLocalSequenceKvTrafficGB * draftScale,
            recurrentGB: aggregateGpuRecurrentKvTrafficGB * draftScale,
          },
          aggregateGpuDraftWeightReadGB,
        )
    const draftKvTrafficGB = pureCpu
      ? logicalKvTrafficGB * draftScale
      : draftStageTraffic.componentTotalGB * topology.ppCount
    const draftReadGB = pureCpu
      ? draftWeightGB + draftKvTrafficGB
      : draftStageTraffic.layerGB * topology.ppCount + draftKvTrafficGB
    const memorySeconds = pureCpu
      ? draftReadGB / (cpuBw * positiveNumber(decodeFactor, 1))
      : draftReadGB
        / (totalGpuBw * positiveNumber(decodeFactor, 1))
    const draftFlops = 2 * effectiveDraftParams * GB * safeBatch
    const draftUtilization = clamp(
      (positiveNumber(decodeFactor, 1) + prefillRange.mid) / 2,
      0.01,
      1,
    )
    const computeSeconds = pureCpu
      ? resolvedCpuTflops == null
        ? 0
        : draftFlops / (resolvedCpuTflops * 1e12 * draftUtilization)
      : draftFlops
        / (
          totalDecodeTflops
          * 1e12
          * draftUtilization
        )
        * gpuStageTimeFactor
    return {
      seconds: Math.max(memorySeconds, computeSeconds),
      memorySeconds,
      computeSeconds,
    }
  }

  const expectedAcceptedTokens = speculativeDecoding
    ? (1 - Math.pow(alpha, safeDraftLen + 1)) / (1 - alpha)
    : 1

  function getDecodeStep(decodeFactor, cpuDecodeFactor = decodeFactor) {
    const memorySeconds = getTargetMemorySeconds(decodeFactor, cpuDecodeFactor)
    const computeSeconds = getTargetComputeSeconds(decodeFactor, cpuDecodeFactor)
    const baseSeconds = Math.max(memorySeconds, computeSeconds)
    const schedulingSeconds = baseSeconds / batchSchedulingEfficiency / ppBubbleEff
    const normalSeconds = schedulingSeconds + commonCommSeconds + moeDispatchSeconds

    if (!speculativeDecoding) {
      return {
        seconds: normalSeconds,
        memorySeconds,
        computeSeconds,
        normalSeconds,
      }
    }

    const verificationTokens = safeDraftLen + 1
    const verificationMemorySeconds = getTargetMemorySeconds(
      decodeFactor,
      cpuDecodeFactor,
      verificationTokens,
    )
    const verificationComputeSeconds = getTargetComputeSeconds(
      decodeFactor,
      cpuDecodeFactor,
      verificationTokens,
    )
    const verificationSeconds = Math.max(
      verificationMemorySeconds,
      verificationComputeSeconds,
    ) / batchSchedulingEfficiency / ppBubbleEff
    const draftStep = getDraftStep(decodeFactor)
    const draftSeconds = safeDraftLen * draftStep.seconds
      / batchSchedulingEfficiency / ppBubbleEff
    // Verification is one batched target-model pass. Its payload grows with
    // the candidate sequence, but collective startup latency is paid once.
    const communicationSeconds = getTargetCommunicationSeconds(safeBatch * verificationTokens)
      + getMoeDispatchSeconds({
        gpu,
        framework: adjustedFramework,
        model,
        batch: safeBatch * verificationTokens,
        topology,
      })
    const cycleSeconds = verificationSeconds + draftSeconds + communicationSeconds
    return {
      seconds: cycleSeconds / expectedAcceptedTokens,
      memorySeconds: (
        verificationMemorySeconds
        + safeDraftLen * draftStep.memorySeconds
      ) / expectedAcceptedTokens,
      computeSeconds: (
        verificationComputeSeconds
        + safeDraftLen * draftStep.computeSeconds
      ) / expectedAcceptedTokens,
      normalSeconds,
    }
  }

  const stepMin = getDecodeStep(decodeRange.min, cpuDecodeRange.min)
  const stepMid = getDecodeStep(decodeRange.mid, cpuDecodeRange.mid)
  const stepMax = getDecodeStep(decodeRange.max, cpuDecodeRange.max)
  const effectiveToksMin = safeBatch / Math.max(stepMin.seconds, 1e-12)
  const effectiveToks = safeBatch / Math.max(stepMid.seconds, 1e-12)
  const effectiveToksMax = safeBatch / Math.max(stepMax.seconds, 1e-12)
  const singleToksMin = effectiveToksMin / safeBatch
  const singleToks = effectiveToks / safeBatch
  const singleToksMax = effectiveToksMax / safeBatch
  const effectiveTpot = stepMid.seconds * MS
  const tpot = effectiveTpot
  const speculativeSpeedup = speculativeDecoding
    ? stepMid.normalSeconds / Math.max(stepMid.seconds, 1e-12)
    : 1

  const bwLimit = safeBatch / Math.max(stepMid.memorySeconds, 1e-12)
  const decodeComputeLimit = stepMid.computeSeconds > 0
    ? safeBatch / stepMid.computeSeconds
    : null
  const rooflineDecodeBytesGB = speculativeDecoding
    ? (
        getMoeWeightReadGB(
          decoderModel,
          moe,
          safeBatch * (safeDraftLen + 1),
          quantBytes,
          topology,
        ) * gpuStageTimeFactor
        + modeledKvTrafficGB * (safeDraftLen + 1)
        + safeDraftLen * (
          (
            pureCpu
              ? draftWeightGB
              : aggregateGpuDraftWeightReadGB * gpuStageTimeFactor
          )
          + modeledKvTrafficGB * draftScale
        )
      ) / expectedAcceptedTokens
    : decodeBytesPerStep
  const rooflineDecodeFlops = speculativeDecoding
    ? (
        decodeFlopsPerToken * safeBatch * (safeDraftLen + 1)
        + 2 * effectiveDraftParams * GB * safeBatch * safeDraftLen
      ) / expectedAcceptedTokens
    : decodeFlopsPerToken * safeBatch
  const arithmeticIntensity = rooflineDecodeFlops
    / (Math.max(rooflineDecodeBytesGB, 1e-12) * GB)
  // Effective ridge point uses the same framework/utilization assumptions as
  // the timing model. This keeps the plotted crossing and bottleneck label
  // mathematically identical, including offload paths.
  const ridgePoint = stepMid.computeSeconds > 0
    ? arithmeticIntensity * stepMid.memorySeconds / stepMid.computeSeconds
    : null
  const roofline = ridgePoint ? arithmeticIntensity / ridgePoint : null
  const bottleneck = stepMid.memorySeconds >= stepMid.computeSeconds ? 'bandwidth' : 'compute'

  const flashRange = getFlashAttentionBoostRange({
    enabled: flashAttention,
    promptLen: effectivePromptLen,
    headDim: getAttentionDimensions(model).qkDim,
  })
  const softmaxLayerRatio = (layerBreakdown.globalLayers + layerBreakdown.localLayers)
    / Math.max(1, layerBreakdown.layers)
  const scaledFlashRange = {
    min: 1 + (flashRange.min - 1) * softmaxLayerRatio,
    mid: 1 + (flashRange.mid - 1) * softmaxLayerRatio,
    max: 1 + (flashRange.max - 1) * softmaxLayerRatio,
  }
  const prefillAttentionExtra = getPrefillAttentionFlopsPerToken({
    model,
    layerBreakdown,
    totalHeads,
    cachedTokens: cachedTextTokens,
    newTokens: effectivePromptLen,
  })
  const prefillAverageAttendedTokens = getClippedCausalAverage(
    cachedTextTokens,
    effectivePromptLen,
    Infinity,
  )
  const prefillAverageLocalAttendedTokens = getClippedCausalAverage(
    cachedTextTokens,
    effectivePromptLen,
    finiteNumber(model?.sliding_window, Infinity),
  )
  const prefillFlopsPerToken = 2 * activeParams * GB + prefillAttentionExtra
  const totalPrefillTokens = effectivePromptLen * safeBatch
  const totalPrefillFlops = prefillFlopsPerToken * totalPrefillTokens
  const prefillChunkSize = positiveInteger(model?.prefill_chunk_size, 2048)
  const prefillWeightPasses = Math.max(
    1,
    Math.ceil(effectivePromptLen / prefillChunkSize),
  )
  let remainingPrefillTokens = effectivePromptLen
  let prefillWeightReadGB = 0
  let prefillExpertWeightReadGB = 0
  for (let chunkIndex = 0; chunkIndex < prefillWeightPasses; chunkIndex += 1) {
    const chunkQueries = Math.min(prefillChunkSize, remainingPrefillTokens)
    const chunkTokenBatch = chunkQueries * safeBatch
    const touchedParams = getMoeTouchedParams(decoderModel, moe, chunkTokenBatch)
    prefillWeightReadGB += getMoeWeightReadGB(
      decoderModel,
      moe,
      chunkTokenBatch,
      quantBytes,
      topology,
    )
    prefillExpertWeightReadGB += Math.max(0, touchedParams - moe.nonExpertParams)
      * quantBytes
    remainingPrefillTokens -= chunkQueries
  }
  const prefillDenseWeightReadGB = moe.nonExpertParams
    * quantBytes
    * prefillWeightPasses
  const prefillKvAtEnd = getKvBreakdown(
    model,
    promptTokensWithVision,
    safeBatch,
    kvBytesPerElement,
  )
  // Cache writes occur for every uncached token even after a sliding-window
  // cache reaches steady state and starts overwriting old entries.
  const prefillSequenceKvWritePerToken = getSequenceKvComponents(
    model,
    layerBreakdown,
    1,
    safeBatch,
    kvBytesPerElement,
  )
  const logicalPrefillGlobalSequenceKvTrafficGB = prefillSequenceKvWritePerToken.globalGB
    * effectivePromptLen
  const logicalPrefillLocalSequenceKvTrafficGB = prefillSequenceKvWritePerToken.localGB
    * effectivePromptLen
  const logicalPrefillSequenceKvTrafficGB = logicalPrefillGlobalSequenceKvTrafficGB
    + logicalPrefillLocalSequenceKvTrafficGB
  const logicalPrefillRecurrentKvTrafficGB = 2
    * prefillKvAtEnd.recurrentGB
    * effectivePromptLen
  const logicalPrefillKvTrafficGB = logicalPrefillSequenceKvTrafficGB
    + logicalPrefillRecurrentKvTrafficGB
  const aggregateGpuPrefillGlobalSequenceKvTrafficGB =
    logicalPrefillGlobalSequenceKvTrafficGB * globalSequenceKvReplicationFactor
  const aggregateGpuPrefillLocalSequenceKvTrafficGB =
    logicalPrefillLocalSequenceKvTrafficGB * localSequenceKvReplicationFactor
  const aggregateGpuPrefillRecurrentKvTrafficGB =
    logicalPrefillRecurrentKvTrafficGB * recurrentKvReplicationFactor
  const aggregateGpuPrefillKvTrafficGB = (
    aggregateGpuPrefillGlobalSequenceKvTrafficGB
    + aggregateGpuPrefillLocalSequenceKvTrafficGB
    + aggregateGpuPrefillRecurrentKvTrafficGB
  )
  const prefillKvTrafficStage = getStageComponentAllocation(
    pipelineComponentLayout,
    layerBreakdown,
    {
      globalGB: aggregateGpuPrefillGlobalSequenceKvTrafficGB,
      localGB: aggregateGpuPrefillLocalSequenceKvTrafficGB,
      recurrentGB: aggregateGpuPrefillRecurrentKvTrafficGB,
    },
    prefillWeightReadGB,
  )
  const pipelineAggregateGpuPrefillGlobalSequenceKvTrafficGB =
    prefillKvTrafficStage.globalGB * topology.ppCount
  const pipelineAggregateGpuPrefillLocalSequenceKvTrafficGB =
    prefillKvTrafficStage.localGB * topology.ppCount
  const pipelineAggregateGpuPrefillRecurrentKvTrafficGB =
    prefillKvTrafficStage.recurrentGB * topology.ppCount
  const pipelineAggregateGpuPrefillKvTrafficGB =
    prefillKvTrafficStage.componentTotalGB
    * topology.ppCount
  const pipelinePrefillWeightReadGB = prefillKvTrafficStage.layerGB
    * topology.ppCount
  const prefillSelectedStageFactor = prefillKvTrafficStage.stageLayers
    / modelLayers
    * topology.ppCount
  const prefillKvWriteGB = pureCpu
    ? logicalPrefillKvTrafficGB
    : isLlamaCppHybrid
      ? pipelineAggregateGpuPrefillKvTrafficGB * gpuLayerRatio
        + logicalPrefillKvTrafficGB * (1 - gpuLayerRatio)
      : pipelineAggregateGpuPrefillKvTrafficGB
  const prefillMicrobatches = safeBatch * Math.max(
    1,
    prefillWeightPasses,
  )
  const prefillPpBubbleEff = topology.ppCount > 1
    ? prefillMicrobatches / (prefillMicrobatches + topology.ppCount - 1)
    : 1
  const prefillCommunicationSeconds = getTargetCommunicationSeconds(
    totalPrefillTokens,
    prefillWeightPasses,
  )

  function getDraftPrefillSeconds(prefillFactor, flashBoost) {
    if (!speculativeDecoding) return 0
    const utilization = clamp(prefillFactor * flashBoost, 0.01, 1)
    const draftPrefillStageTraffic = pureCpu
      ? null
      : getStageComponentAllocation(
          pipelineComponentLayout,
          layerBreakdown,
          {
            globalGB: aggregateGpuPrefillGlobalSequenceKvTrafficGB * draftScale,
            localGB: aggregateGpuPrefillLocalSequenceKvTrafficGB * draftScale,
            recurrentGB: aggregateGpuPrefillRecurrentKvTrafficGB * draftScale,
          },
          aggregateGpuDraftWeightReadGB * prefillWeightPasses,
        )
    const draftKvTrafficGB = pureCpu
      ? logicalPrefillKvTrafficGB * draftScale
      : draftPrefillStageTraffic.componentTotalGB * topology.ppCount
    const draftMemoryGB = pureCpu
      ? draftWeightGB * prefillWeightPasses + draftKvTrafficGB
      : draftPrefillStageTraffic.layerGB * topology.ppCount + draftKvTrafficGB
    const memorySeconds = pureCpu
      ? draftMemoryGB / (cpuBw * decodeRange.mid)
      : draftMemoryGB
        / (totalGpuBw * decodeRange.mid)
    const draftPrefillFlops = 2 * effectiveDraftParams * GB * totalPrefillTokens
    const computeSeconds = pureCpu
      ? resolvedCpuTflops == null
        ? 0
        : draftPrefillFlops / (resolvedCpuTflops * 1e12 * utilization)
      : draftPrefillFlops
        / (totalPrefillTflops * 1e12 * utilization)
        * gpuStageTimeFactor
    return Math.max(memorySeconds, computeSeconds)
  }

  function getTargetPrefillComputeSeconds(utilization) {
    const safeUtilization = clamp(utilization, 0.01, 1)
    if (pureCpu) {
      return resolvedCpuTflops == null
        ? 0
        : totalPrefillFlops / (resolvedCpuTflops * 1e12 * safeUtilization)
    }
    if (isLlamaCppHybrid) {
      const gpuSeconds = gpuLayerRatio > 0
        ? totalPrefillFlops * gpuLayerRatio
          / (totalPrefillTflops * 1e12 * safeUtilization)
        : 0
      const cpuSeconds = resolvedCpuTflops != null && gpuLayerRatio < 1
        ? totalPrefillFlops * (1 - gpuLayerRatio)
          / (resolvedCpuTflops * 1e12 * safeUtilization)
        : 0
      return (gpuSeconds + cpuSeconds) * gpuStageTimeFactor
    }
    return totalPrefillFlops
      / (totalPrefillTflops * 1e12 * safeUtilization)
      * gpuStageTimeFactor
  }

  const visionEncoderImages = supportsImages
    && visionParameters.visionEncoderParams > 0
    ? safeImageCount
    : 0
  const visionEncoderWeightReadGB = visionParameters.visionEncoderParams
    * quantBytes
    * visionEncoderImages
  const visionEncoderFlops = 2
    * visionParameters.visionEncoderParams
    * GB
    * visionTokensPerImage
    * visionEncoderImages
  const visionGpuBw = rawGpuBwPerCard
    * (sharedSystemMemory ? 1 : topology.stageGpuCount)
  const visionGpuPrefillTflops = (Number.isFinite(prefillTflopsPerCard)
    ? prefillTflopsPerCard
    : 1e-12) * topology.stageGpuCount

  function getVisionEncoderStep(prefillFactor) {
    if (visionEncoderImages <= 0) {
      return { seconds: 0, memorySeconds: 0, computeSeconds: 0 }
    }
    const utilization = clamp(prefillFactor, 0.01, 1)
    let memorySeconds
    let computeSeconds
    if (pureCpu) {
      memorySeconds = visionEncoderWeightReadGB / (cpuBw * cpuDecodeRange.mid)
      computeSeconds = resolvedCpuTflops == null
        ? 0
        : visionEncoderFlops / (resolvedCpuTflops * 1e12 * utilization)
    } else if (isLlamaCppHybrid) {
      memorySeconds = visionEncoderWeightReadGB * gpuLayerRatio
        / (visionGpuBw * decodeRange.mid)
        + visionEncoderWeightReadGB * (1 - gpuLayerRatio)
          / (cpuBw * cpuDecodeRange.mid)
      computeSeconds = visionEncoderFlops * gpuLayerRatio
        / (visionGpuPrefillTflops * 1e12 * utilization)
        + (resolvedCpuTflops != null
          ? visionEncoderFlops * (1 - gpuLayerRatio)
            / (resolvedCpuTflops * 1e12 * utilization)
          : 0)
    } else {
      memorySeconds = visionEncoderWeightReadGB / (visionGpuBw * decodeRange.mid)
      computeSeconds = visionEncoderFlops
        / (visionGpuPrefillTflops * 1e12 * utilization)
    }
    return {
      seconds: Math.max(memorySeconds, computeSeconds),
      memorySeconds,
      computeSeconds,
    }
  }

  function getPrefillSeconds(prefillFactor, flashBoost) {
    const utilization = clamp(prefillFactor * flashBoost, 0.01, 1)
    let memorySeconds
    if (pureCpu) {
      memorySeconds = (prefillWeightReadGB + logicalPrefillKvTrafficGB)
        / (cpuBw * decodeRange.mid)
    } else if (isLlamaCppHybrid) {
      const gpuExpertMemorySeconds = isMoeOffload
        ? prefillExpertWeightReadGB
          * prefillSelectedStageFactor
          * gpuLayerRatio
          / offloadTransferBw
        : 0
      const gpuMemoryGB = (
        (isMoeOffload ? prefillDenseWeightReadGB : prefillWeightReadGB)
          + pipelineAggregateGpuPrefillKvTrafficGB
      ) * gpuLayerRatio
      const cpuMemoryGB = (
        prefillWeightReadGB + logicalPrefillKvTrafficGB
      ) * (1 - gpuLayerRatio)
      memorySeconds = gpuExpertMemorySeconds
        + gpuMemoryGB / (totalGpuBw * decodeRange.mid)
        + cpuMemoryGB / (cpuBw * cpuDecodeRange.mid)
    } else if (isMoeOffload) {
      memorySeconds = prefillExpertWeightReadGB
        * prefillSelectedStageFactor
        / offloadTransferBw
        + (
          prefillDenseWeightReadGB * prefillSelectedStageFactor
          + pipelineAggregateGpuPrefillKvTrafficGB
        )
          / (totalGpuBw * decodeRange.mid)
    } else {
      memorySeconds = (
        pipelinePrefillWeightReadGB
        + pipelineAggregateGpuPrefillKvTrafficGB
      )
        / (totalGpuBw * decodeRange.mid)
    }
    const computeSeconds = getTargetPrefillComputeSeconds(utilization)
    const targetSeconds = Math.max(memorySeconds, computeSeconds) / prefillPpBubbleEff
    const draftSeconds = getDraftPrefillSeconds(prefillFactor, flashBoost)
      / prefillPpBubbleEff
    const visionSeconds = getVisionEncoderStep(prefillFactor).seconds
    return targetSeconds + draftSeconds + visionSeconds + prefillCommunicationSeconds
  }

  const prefillSecondsMin = getPrefillSeconds(prefillRange.min, scaledFlashRange.min)
  const prefillSeconds = getPrefillSeconds(prefillRange.mid, scaledFlashRange.mid)
  const prefillSecondsMax = getPrefillSeconds(prefillRange.max, scaledFlashRange.max)
  const prefillToksMin = totalPrefillTokens / Math.max(prefillSecondsMin, 1e-12)
  const prefillToks = totalPrefillTokens / Math.max(prefillSeconds, 1e-12)
  const prefillToksMax = totalPrefillTokens / Math.max(prefillSecondsMax, 1e-12)
  const prefillLatency = prefillSeconds * MS
  const ttft = prefillLatency + effectiveTpot
  const totalLatency = ttft + Math.max(0, safeOutputLen - 1) * effectiveTpot

  const tpBaseSeconds = Math.max(stepMid.memorySeconds, stepMid.computeSeconds)
    / batchSchedulingEfficiency / ppBubbleEff
  const targetGpuFraction = isLlamaCppHybrid ? gpuLayerRatio : 1
  const tpCommSeconds = pureCpu
    ? 0
    : getTpCommunicationSeconds({
        topology,
        interconnect,
        model,
        tokenBatch: safeBatch,
      }) * targetGpuFraction
  const epCommSeconds = pureCpu
    ? 0
    : getEpCommunicationSeconds({
        topology,
        interconnect,
        model,
        tokenBatch: safeBatch,
      }) * targetGpuFraction
  const tpEfficiency = tpBaseSeconds / Math.max(tpBaseSeconds + tpCommSeconds, 1e-12)
  const epEfficiency = tpBaseSeconds / Math.max(tpBaseSeconds + epCommSeconds, 1e-12)
  const ppP2pMs = pureCpu
    ? 0
    : getPpCommunicationSeconds({
        topology,
        interconnect,
        model,
        tokenBatch: safeBatch,
      }) * targetGpuFraction * MS

  const totalPower = pureCpu
    ? null
    : positiveNumber(gpu.tdp, 0) * topology.totalGpuCount / 1000
  const tokPerJoule = totalPower > 0 ? effectiveToks / (totalPower * 1000) : null
  const vramPct = pureCpu ? 0 : perCardNeeded / perCardVram * 100
  const displayNeeded = pureCpu ? 0 : perCardNeeded
  const displayVram = pureCpu ? 0 : perCardVram
  const totalNeeded = clusterNeeded
  const effectiveBw = pureCpu ? cpuBw : totalGpuBw
  const prefillAttentionFactor = prefillFlopsPerToken / (2 * activeParams * GB)
  const peakPrefillComputeSeconds = getTargetPrefillComputeSeconds(1)
    + getVisionEncoderStep(1).computeSeconds
  const computeLimit = peakPrefillComputeSeconds > 0
    ? totalPrefillTokens / peakPrefillComputeSeconds
    : null
  const visionEncoderPrefillSeconds = getVisionEncoderStep(prefillRange.mid).seconds
  const parameterEstimate = Boolean(model.parameterEstimate)
  const unofficialGpu = Boolean(gpu.modified || gpu.official === false)
  const heterogeneousGpuEstimate = Boolean(gpu.mixedGpu)
  const derivedGpuSpecs = Boolean(
    gpu.computeEstimate
    || gpu.tdpEstimate
    || gpu.specConfidence === 'derived'
  )
  const accuracyTier = (
    targetKv.mlaApproximate
    || targetKv.recurrentApproximate
    || pureCpu
    || cpuComputeIsUpperBound
    || visionAccountingApproximate
    || !runtimeConfigurationSupported
    || !computePrecisionSupported
    || pipelineComponentLayout.conservative
    || heterogeneousGpuEstimate
    || parameterEstimate
    || unofficialGpu
    || draftWasInferred
    || speculativeDecoding
  )
    ? 'low'
    : (
        topology.totalGpuCount > 1
        || effectiveCpuOffload
        || model.type === 'moe'
        || visionPatchTokens > 0
        || derivedGpuSpecs
      )
      ? 'mid'
      : 'high'

  return {
    // Memory and fit
    weightGB,
    kvGB,
    overheadGB,
    activationGB,
    targetWeightGB,
    targetKvGB: targetKv.totalGB,
    targetGlobalSequenceKvGB: targetKv.globalSequenceGB,
    targetLocalSequenceKvGB: targetKv.localSequenceGB,
    targetRecurrentKvGB: targetKv.recurrentGB,
    draftWeightGB,
    draftKvGB,
    perCardWeightGB,
    perCardKvGB,
    perCardGlobalSequenceKvGB,
    perCardLocalSequenceKvGB,
    perCardRecurrentKvGB,
    perCardKvStageIndex: perCardKvStage.stageIndex,
    perCardKvStageGlobalLayers: perCardKvStage.globalLayers,
    perCardKvStageLocalLayers: perCardKvStage.localLayers,
    perCardKvStageLinearLayers: perCardKvStage.linearLayers,
    perCardOverheadGB,
    perCardActivationGB,
    totalNeeded,
    totalVram,
    perCardNeeded,
    perCardVram,
    displayNeeded,
    displayVram,
    clusterNeeded,
    vramScope: pureCpu ? 'not_applicable' : topology.totalGpuCount > 1 ? 'per_card' : 'total',
    vramOk,
    vramPct,
    ramOk,
    fitOk,
    availableSysRamGB,
    cpuWeightGB,
    cpuKvGB,
    cpuRamNeededGB,
    systemRamNeededGB,
    sharedSystemRamNeededGB,
    sharedAllocationGB,
    sharedAllocationExceedsRam,
    sharedAllocationExcessGB,
    sharedPoolAvailableGB,
    sharedSystemMemory,
    usesSystemRam,
    sysRam: Number.isFinite(requestedSysRam) ? requestedSysRam : null,
    unifiedMemory,

    // Topology and validation
    gpuCount: topology.totalGpuCount,
    totalGpuCount: topology.totalGpuCount,
    stageGpuCount: topology.stageGpuCount,
    tpCount: topology.tpCount,
    ppCount: topology.ppCount,
    epCount: topology.epCount,
    isEP,
    averageStageLayers,
    largestStageLayers,
    ppStageImbalance,
    ppComponentLayoutExact: pipelineComponentLayout.exact,
    ppComponentLayoutConservative: pipelineComponentLayout.conservative,
    globalSequenceKvShardCount,
    localSequenceKvShardCount,
    recurrentKvShardCount,
    topologyOk: topology.topologyOk && memoryDeviceTopologyOk,
    memoryDeviceTopologyOk,
    requestedPpCount: topology.requestedPp,
    requestedEpCount: topology.requestedEp,
    contextOk,
    workloadInputOk,
    modelContextOk,
    workloadContextOk,
    workloadTokens,
    modeOk,
    frameworkOk,
    speculativeOk,
    cpuParallelOk,
    offloadParallelOk,
    offloadMemoryArchitectureOk,
    denseOffloadOk: !unsupportedDenseOffload,
    kvCacheSupported,
    weightQuantSupported,
    runtimeTopologySupported,
    localInferenceSupported,
    runtimeConfigurationSupported,
    runtimeCompatibilityReasonCodes,
    runtimeCompatibilityMessage: runtimeCompatibility.reasons[0]?.message ?? null,
    modelDataOk: decompositionOk,
    computePrecisionSupported,

    // Throughput
    bwLimit,
    computeLimit,
    decodeComputeLimit,
    decodeToks: effectiveToks,
    decodeToksMin: effectiveToksMin,
    decodeToksMax: effectiveToksMax,
    effectiveToks,
    effectiveToksMin,
    effectiveToksMax,
    singleToks,
    singleToksMin,
    singleToksMax,
    prefillToks,
    prefillToksMin,
    prefillToksMax,
    prefillWeightReadGB,
    prefillWeightPasses,
    prefillCommunicationSeconds,
    prefillKvWriteGB,
    pipelineAggregateGpuPrefillKvTrafficGB,
    pipelineAggregateGpuPrefillGlobalSequenceKvTrafficGB,
    pipelineAggregateGpuPrefillLocalSequenceKvTrafficGB,
    pipelineAggregateGpuPrefillRecurrentKvTrafficGB,
    visionEncoderWeightReadGB,
    visionEncoderFlops,
    visionEncoderPrefillSeconds,
    kvReadGB: modeledKvTrafficGB,
    kvTrafficGB: modeledKvTrafficGB,
    logicalKvTrafficGB,
    aggregateGpuKvTrafficGB,
    pipelineAggregateGpuKvTrafficGB,
    pipelineAggregateGpuGlobalSequenceKvTrafficGB,
    pipelineAggregateGpuLocalSequenceKvTrafficGB,
    pipelineAggregateGpuRecurrentKvTrafficGB,
    logicalGlobalSequenceKvTrafficGB,
    logicalLocalSequenceKvTrafficGB,
    avgDecodeSeqLen,
    avgLocalDecodeSeqLen,
    prefillAverageAttendedTokens,
    prefillAverageLocalAttendedTokens,
    effectiveBw,
    arithmeticIntensity,
    ridgePoint,
    roofline,
    bottleneck,

    // Latency and communication
    prefillLatency,
    ttft,
    tpot,
    effectiveTpot,
    totalLatency,
    effectiveTotalLatency: totalLatency,
    tpEfficiency,
    epEfficiency,
    ppBubbleEff,
    prefillPpBubbleEff,
    ppP2pMs,

    // Configuration details
    flashAttention,
    flashFactorMin: scaledFlashRange.min,
    flashFactorMax: scaledFlashRange.max,
    flashFactor: scaledFlashRange.mid,
    prefixCacheHit: Math.round(prefixHitRatio * 100),
    effectivePromptLen,
    attentionType,
    totalHeads,
    attentionSummary,
    prefillAttentionFactor,
    kvCacheLabel: resolvedKvCacheLabel,
    pureCpu: Boolean(pureCpu),
    cpuOffload: effectiveCpuOffload,
    isLlamaCppHybrid,
    isMoeOffload,
    cpuMemBwLabel: usesSystemRam
      ? (cpuMemBw?.label ?? 'DDR5-4800')
      : null,
    cpuMemBwGBs: usesSystemRam ? cpuBw : null,
    cpuTflops: cpuComputeNeeded ? resolvedCpuTflops : null,
    cpuComputeProvided: cpuComputeNeeded && resolvedCpuTflops != null,
    cpuComputeIsUpperBound,
    cpuDecodeIsUpperBound: cpuComputeIsUpperBound,
    offloadTransferBw: isMoeOffload ? offloadTransferBw : null,
    pcieBwLabel: isMoeOffload && pcieBw ? pcieBw.label : null,
    pcieWidthLabel: isMoeOffload && pcieWidth ? pcieWidth.label : null,
    autoNgl,
    effectiveNgl: usesLlamaCppGpuLayers || pureCpu ? effectiveNgl : null,
    nglCount: usesLlamaCppGpuLayers || pureCpu ? effectiveNgl : null,
    gpuLayerRatio,
    speculativeDecoding: Boolean(speculativeDecoding),
    speculativeSpeedup,
    acceptanceRate: alpha,
    draftLen: safeDraftLen,
    draftModelParams: effectiveDraftParams,
    draftWasInferred,
    expectedAcceptedTokens,
    imageCount: safeImageCount,
    visionPatchTokens,
    visionTokensPerImage,
    visionTokensWereInferred,
    visionAccountingApproximate,
    visionParamsScope: visionParameters.paramsScope,
    residentModelParams: visionParameters.residentParams,
    decoderModelParams: visionParameters.decoderParams,
    visionEncoderParams: visionParameters.visionEncoderParams,
    parameterEstimate,
    unofficialGpu,
    heterogeneousGpuEstimate,
    derivedGpuSpecs,
    mlaApproximate: targetKv.mlaApproximate,
    recurrentApproximate: targetKv.recurrentApproximate,
    cpuPrefillIsUpperBound: cpuComputeIsUpperBound,
    totalPower,
    powerIsPartial: Boolean(isLlamaCppHybrid || isMoeOffload),
    tokPerJoule,
    gpuVendor: gpu.vendor,
    gpuBw: gpu.bw,
    effectiveGpuBwPerCard: rawGpuBwPerCard,
    modelType: model.type,
    modelParams: model.params,
    modelExpertsPerToken: model.experts_per_token ?? null,
    accuracyTier,
  }
}

/** GGUF/llama.cpp use serialized GGUF bytes per parameter. */
export function getQuantBytes(quant, gpu, framework) {
  const useGguf = framework?.id === 'llamacpp' || framework?.id === 'llamacpp_metal'
  return positiveNumber(useGguf ? quant?.gguf_bytes ?? quant?.bytes : quant?.bytes, 0.5)
}

function getTflops(gpu, quant, keyName) {
  if (quant?.id === 'fp32') {
    const fp32 = finiteNumber(gpu?.fp32, NaN)
    return Number.isFinite(fp32) && fp32 > 0 ? fp32 : null
  }
  const key = quant?.[keyName] ?? quant?.flops_key ?? 'bf16'
  return positiveNumber(gpu?.[key], positiveNumber(gpu?.bf16, 1))
}

function getDecodeTflops(gpu, quant) {
  return getTflops(gpu, quant, 'flops_key')
}

function getPrefillTflops(gpu, quant) {
  return getTflops(gpu, quant, 'prefill_flops_key')
}

export function getWarnings(result) {
  const warnings = []
  if (!result) return warnings

  if (result.sharedAllocationExceedsRam) {
    warnings.push({
      level: 'error',
      key: 'shared_allocation_exceeds_ram',
      allocation: Number(result.sharedAllocationGB ?? 0).toFixed(1),
      available: Number(result.availableSysRamGB ?? 0).toFixed(1),
      diff: Number(result.sharedAllocationExcessGB ?? 0).toFixed(1),
    })
  }

  if (!result.vramOk) {
    const vramDeficit = Math.max(0, result.displayNeeded - result.displayVram)
    // Shared-pool allocation validation can fail even while the model itself
    // is smaller than the selected GPU budget. Do not emit a second
    // "VRAM short by 0.0 GB" error in that case.
    if (vramDeficit > 1e-9) {
      warnings.push({
        level: 'error',
        key: 'vram_oom',
        diff: vramDeficit.toFixed(1),
      })
    }
  } else if (!result.pureCpu && result.vramPct > 95) {
    warnings.push({ level: 'warn', key: 'vram_high' })
  }

  if (!result.ramOk) {
    const systemRamNeededGB = result.systemRamNeededGB ?? result.cpuRamNeededGB
    const ramDeficit = result.sysRam == null
      ? null
      : Math.max(0, systemRamNeededGB - result.availableSysRamGB)
    // Keep the unknown-capacity warning, and keep real model-residency OOMs.
    // Allocation-only failures already have the dedicated warning above.
    if (ramDeficit == null || ramDeficit > 1e-9) {
      warnings.push({
        level: 'error',
        key: result.sysRam == null ? 'cpu_ram_unknown' : 'cpu_ram_oom',
        diff: ramDeficit == null ? null : ramDeficit.toFixed(1),
        needed: systemRamNeededGB.toFixed(1),
      })
    }
  }
  if (!result.contextOk) {
    warnings.push({
      level: 'error',
      key: result.modelContextOk ? 'workload_context_invalid' : 'model_context_invalid',
      needed: result.workloadTokens,
    })
  }
  if (!result.workloadInputOk) warnings.push({ level: 'error', key: 'workload_input_invalid' })
  if (!result.topologyOk) warnings.push({ level: 'error', key: 'topology_invalid' })
  if (!result.frameworkOk) warnings.push({ level: 'error', key: 'framework_unsupported' })
  if (!result.speculativeOk) warnings.push({ level: 'error', key: 'speculative_unsupported' })
  if (!result.cpuParallelOk) warnings.push({ level: 'error', key: 'cpu_parallel_unsupported' })
  if (!result.offloadParallelOk) warnings.push({ level: 'error', key: 'offload_ep_unsupported' })
  if (!result.offloadMemoryArchitectureOk) warnings.push({ level: 'error', key: 'offload_shared_memory_unsupported' })
  if (!result.denseOffloadOk) warnings.push({ level: 'error', key: 'mode_invalid' })
  if (!result.kvCacheSupported) warnings.push({ level: 'error', key: 'kv_cache_unsupported' })
  if (!result.weightQuantSupported) warnings.push({ level: 'error', key: 'weight_quant_unsupported' })
  if (!result.runtimeTopologySupported) {
    warnings.push({ level: 'error', key: 'runtime_topology_unsupported' })
  }
  if (!result.localInferenceSupported) {
    warnings.push({ level: 'error', key: 'local_inference_unavailable' })
  }
  if (
    !result.runtimeConfigurationSupported
    && result.frameworkOk
    && result.speculativeOk
    && result.kvCacheSupported
    && result.weightQuantSupported
    && result.runtimeTopologySupported
    && result.localInferenceSupported
  ) {
    warnings.push({ level: 'error', key: 'runtime_configuration_unsupported' })
  }
  if (!result.modelDataOk) warnings.push({ level: 'error', key: 'moe_data_invalid' })
  if (!result.computePrecisionSupported) {
    warnings.push({ level: 'error', key: 'compute_precision_unsupported' })
  }
  if (result.mlaApproximate) warnings.push({ level: 'warn', key: 'mla_approximate' })
  if (result.recurrentApproximate) warnings.push({ level: 'warn', key: 'recurrent_approximate' })
  if (result.draftWasInferred) warnings.push({ level: 'info', key: 'draft_model_inferred' })
  if (result.speculativeDecoding) warnings.push({ level: 'info', key: 'draft_model_approximate' })
  if (result.cpuDecodeIsUpperBound) warnings.push({ level: 'info', key: 'cpu_decode_upper_bound' })
  if (result.cpuPrefillIsUpperBound) warnings.push({ level: 'info', key: 'cpu_prefill_upper_bound' })
  if (result.visionPatchTokens > 0) warnings.push({ level: 'info', key: 'vision_compute_approximate' })
  if (result.visionAccountingApproximate) {
    warnings.push({ level: 'warn', key: 'vision_accounting_approximate' })
  }
  if (result.ppComponentLayoutConservative) {
    warnings.push({ level: 'info', key: 'pp_component_layout_conservative' })
  }
  if (result.heterogeneousGpuEstimate) {
    warnings.push({ level: 'warn', key: 'mixed_gpu_estimate' })
  }
  if (result.parameterEstimate) warnings.push({ level: 'warn', key: 'parameter_estimate' })
  if (result.unofficialGpu) warnings.push({ level: 'warn', key: 'unofficial_gpu_specs' })
  if (result.derivedGpuSpecs && !result.unofficialGpu) {
    warnings.push({ level: 'warn', key: 'derived_gpu_specs' })
  }
  if (result.visionTokensWereInferred) warnings.push({ level: 'warn', key: 'vision_tokens_inferred' })
  if (result.powerIsPartial) warnings.push({ level: 'info', key: 'power_partial' })
  if (result.activationGB > 2) {
    warnings.push({ level: 'info', key: 'activation_high', gb: result.activationGB.toFixed(1) })
  }
  if (result.tpEfficiency < 0.7) warnings.push({ level: 'warn', key: 'tp_comm' })
  if ((result.singleToksMin ?? result.singleToks) < 20) warnings.push({ level: 'warn', key: 'slow_single' })
  if (result.bottleneck === 'bandwidth' && result.roofline != null && result.roofline < 0.1) {
    warnings.push({ level: 'info', key: 'bw_bottleneck' })
  }
  if (result.totalPower != null && result.totalPower > 10) {
    warnings.push({ level: 'info', key: 'high_power', power: result.totalPower.toFixed(1) })
  }
  return warnings
}

export function calcBatchSweep(
  params,
  batches = [1, 2, 4, 8, 16, 32, 64, 128, 256],
) {
  return batches.map(batch => {
    try {
      const result = calcAll({ ...params, batch })
      return {
        batch,
        decodeToks: result.effectiveToks,
        effectiveToks: result.effectiveToks,
        singleToks: result.singleToks,
        tpot: result.effectiveTpot,
        ttft: result.ttft,
        totalLatency: result.totalLatency,
        vramOk: result.vramOk,
        ramOk: result.ramOk,
        fitOk: result.fitOk,
        ppBubbleEff: result.ppBubbleEff,
        bottleneck: result.bottleneck,
      }
    } catch {
      return { batch, error: true }
    }
  })
}

/**
 * Aggregate heterogeneous GPUs assuming equal-sized model shards. With equal
 * sharding, the slowest card determines each synchronized step, so aggregate
 * bandwidth/compute is N times the slowest card rather than the sum of peaks.
 */
export function aggregateGpuSlots(slots) {
  if (!slots?.length) return null
  if (slots.length === 1) return slots[0].gpu

  const expanded = slots.flatMap(slot =>
    Array.from({ length: positiveInteger(slot.count) }, () => slot.gpu),
  )
  const count = expanded.length
  const componentVendors = [
    ...new Set(expanded.map(item => String(item?.vendor ?? '').toLowerCase()).filter(Boolean)),
  ]
  const inferredArchitectures = expanded.map(item => inferGpuArchitecture(item))
  const componentArchitectures = [...new Set(inferredArchitectures.filter(Boolean))]
  const allArchitecturesKnown = inferredArchitectures.every(Boolean)
  const hasSharedMemory = expanded.some(item => item?.sharedMemory)
  const hasUnifiedMemory = expanded.some(item => item?.unifiedMemory)
  const hasSingleDeviceMemory = hasSharedMemory || hasUnifiedMemory
  const mixedVendors = componentVendors.length !== 1
  const mixedArchitectures = !allArchitecturesKnown || componentArchitectures.length !== 1
  const mixedGpuEstimateSupported = !hasSingleDeviceMemory
    && !mixedVendors
    && !mixedArchitectures
  const minEffectiveBw = Math.min(...expanded.map(item =>
    positiveNumber(item.bw, 1)
      * clamp(finiteNumber(item.bwUtilization, 0.8), 0.01, 1)
      * getAppleDecodeBwScale(item),
  ))
  const limitingMemoryGpu = expanded.reduce((limiting, item) => {
    const usableCapacity = positiveNumber(item.vram, 0.001)
      * clamp(finiteNumber(item.usableRatio, 1), 0.01, 1)
    const limitingUsableCapacity = positiveNumber(limiting.vram, 0.001)
      * clamp(finiteNumber(limiting.usableRatio, 1), 0.01, 1)
    return usableCapacity < limitingUsableCapacity ? item : limiting
  })
  const minMetric = key => Math.min(...expanded.map(item =>
    positiveNumber(item[key], positiveNumber(item.bf16, 1)),
  ))
  const minOptionalMetric = key => {
    const values = expanded.map(item => finiteNumber(item[key], NaN))
    return values.every(value => Number.isFinite(value) && value > 0)
      ? Math.min(...values)
      : null
  }
  const totalTdp = expanded.reduce((sum, item) => sum + positiveNumber(item.tdp, 0), 0)

  return {
    // Keep VRAM and its reserve ratio from the same physical card. Combining
    // the smallest raw VRAM with another card's smallest ratio invents a
    // capacity below every card in the heterogeneous set.
    vram: positiveNumber(limitingMemoryGpu.vram, 0.001),
    bw: minEffectiveBw,
    fp32: minOptionalMetric('fp32'),
    bf16: minMetric('bf16'),
    fp8: minMetric('fp8'),
    int8: minMetric('int8'),
    int4: minMetric('int4'),
    tdp: totalTdp / count,
    bwUtilization: 1,
    usableRatio: clamp(finiteNumber(limitingMemoryGpu.usableRatio, 1), 0.01, 1),
    nvlink_bw: null,
    vendor: mixedVendors ? 'mixed' : componentVendors[0],
    architecture: mixedArchitectures ? null : componentArchitectures[0],
    tier: slots[0].gpu.tier,
    id: 'mixed',
    name: slots.map(slot => `${slot.gpu.name} ×${slot.count}`).join(' + '),
    mixedGpu: true,
    mixedVendors,
    mixedArchitectures,
    mixedGpuEstimateSupported,
    componentVendors,
    componentArchitectures,
    componentSlots: slots.map(slot => ({
      id: slot.gpu.id,
      name: slot.gpu.name,
      count: positiveInteger(slot.count),
      vram: positiveNumber(slot.gpu.vram, 0.001),
    })),
    modified: expanded.some(item => item?.modified),
    official: expanded.every(item => item?.official !== false),
    computeEstimate: expanded.some(item => item?.computeEstimate),
    tdpEstimate: expanded.some(item => item?.tdpEstimate),
    specConfidence: expanded.some(item => item?.specConfidence === 'derived')
      ? 'derived'
      : undefined,
    // Never erase shared-pool semantics when an invalid URL or caller mixes a
    // unified/shared-memory device with another slot. calcAll will reject this
    // aggregate instead of treating it as ordinary discrete VRAM.
    sharedMemory: hasSharedMemory,
    unifiedMemory: hasUnifiedMemory,
    invalidMemoryMix: hasSingleDeviceMemory && count > 1,
  }
}
