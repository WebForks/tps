// LiquidAI LFM2 8B: MoE with 1.5B active, optimized for on-device
// Source: https://huggingface.co/LiquidAI/LFM2-8B-A1B
export default {
  id: 'liquidai_lfm2_8b',
  released: '2025-02',
  name: 'LiquidAI LFM2 8B',
  type: 'moe',
  params: 8.3,
  active_params: 1.5,
  experts: 32,
  experts_per_token: 4,
  non_expert_params: 0.55,
  layers: 24,
  query_heads: 32,
  kv_heads: 8,
  head_dim: 64,
  linear_attention_layers: 18,
  linear_state_elements_per_layer: 6144,
  linear_state_bytes: 2,
  hidden_size: 2048,
  max_ctx: 128000,
  tags: ['chat'],

  links: {
    ollama: null,
    hf: 'https://huggingface.co/LiquidAI/LFM2-8B-A1B',
    ms: null
  }
}
