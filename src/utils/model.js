export function getTotalHeads(model) {
  const explicitHeads = Number(
    model?.query_heads
    ?? model?.num_attention_heads
    ?? model?.attention_heads,
  )
  if (Number.isFinite(explicitHeads) && explicitHeads > 0) {
    return Math.round(explicitHeads)
  }

  const hiddenSize = Number(model?.hidden_size)
  const headDim = Number(model?.head_dim)
  if (!Number.isFinite(hiddenSize) || !Number.isFinite(headDim) || headDim <= 0) return null
  const totalHeads = hiddenSize / headDim
  if (!Number.isFinite(totalHeads) || totalHeads <= 0) return null
  return Math.round(totalHeads)
}

export function getAttentionType(model) {
  if (Number(model?.kv_lora_rank) > 0) return 'mla'
  const kvHeads = Number(model?.kv_heads)
  const totalHeads = getTotalHeads(model)
  if (!Number.isFinite(kvHeads) || kvHeads <= 0 || !totalHeads) return 'unknown'
  if (kvHeads === 1) return 'mqa'
  if (kvHeads >= totalHeads) return 'mha'
  return 'gqa'
}

export function getAttentionSummary(model) {
  const type = getAttentionType(model)
  const kvHeads = Number(model?.kv_heads)
  const totalHeads = getTotalHeads(model)
  if (type === 'mla' && totalHeads) {
    const latent = Math.max(0, Number(model?.kv_lora_rank) || 0)
      + Math.max(0, Number(model?.qk_rope_head_dim) || 0)
    return `MLA (${totalHeads} heads, ${latent}D cache)`
  }
  if (!totalHeads || !Number.isFinite(kvHeads) || kvHeads <= 0) return '—'

  const typeLabel = {
    mha: 'MHA',
    gqa: 'GQA',
    mqa: 'MQA',
    mla: 'MLA',
    unknown: 'Unknown',
  }[type]

  return `${typeLabel} (${totalHeads}/${kvHeads})`
}
