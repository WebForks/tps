// Jamba 1.5 Large: Hybrid Transformer-Mamba MoE, 94B active, 256K context
// Source: https://huggingface.co/ai21labs/AI21-Jamba-Large-1.5
// Architecture: one attention layer per eight layers; the rest are Mamba.
export default {
  id: 'jamba_1_5_large',
  released: '2024-08',
  name: 'Jamba 1.5 Large',
  type: 'moe',
  params: 398,
  active_params: 94,
  experts: 16,
  experts_per_token: 2,
  layers: 72,
  mamba_ratio: 0.125, // fraction of layers using full attention
  query_heads: 64,
  kv_heads: 8,
  head_dim: 128,
  ssm_expansion: 2,
  ssm_state_size: 16,
  ssm_conv_kernel: 4,
  linear_state_bytes: 4,
  hidden_size: 8192,
  max_ctx: 256000,
  tags: ['chat'],
  links: {
    ollama: null,
    hf: 'https://huggingface.co/ai21labs/AI21-Jamba-Large-1.5',
    ms: null
  }
}
