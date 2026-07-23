// Jamba 52B: Hybrid Transformer-Mamba MoE, 12B active
// Source: https://huggingface.co/ai21labs/Jamba-v0.1
// Architecture: one attention layer per eight layers; the rest are Mamba.
export default {
  id: 'jamba_52b',
  released: '2024-03',
  name: 'Jamba 52B',
  type: 'moe',
  params: 52,
  active_params: 12,
  experts: 16,
  experts_per_token: 2,
  layers: 32,
  mamba_ratio: 0.125, // fraction of layers using full attention
  query_heads: 32,
  kv_heads: 8,
  head_dim: 128,
  ssm_expansion: 2,
  ssm_state_size: 16,
  ssm_conv_kernel: 4,
  linear_state_bytes: 4,
  hidden_size: 4096,
  max_ctx: 256000,
  tags: ['chat'],
  links: {
    ollama: null,
    hf: 'https://huggingface.co/ai21labs/Jamba-v0.1',
    ms: null
  }
}
