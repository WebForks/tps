// Mistral Large 3: 675B MoE with 41B active parameters
// Source: https://mistral.ai/news/mistral-3/
export default {
  id: 'mistral_large_3',
  released: '2025-12',
  name: 'Mistral Large 3 (675B MoE)',
  type: 'moe',
  params: 675,
  active_params: 41,
  experts: 128,
  experts_per_token: 4,
  layers: 61,
  query_heads: 128,
  kv_heads: 128,
  head_dim: 192,
  kv_lora_rank: 512,
  qk_nope_head_dim: 128,
  qk_rope_head_dim: 64,
  v_head_dim: 128,
  hidden_size: 7168,
  max_ctx: 294912,
  tags: ['chat', 'vision', 'multimodal'],
  links: {
    ollama: 'ollama pull mistral-large-3',
    hf: 'https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512',
  },
}
