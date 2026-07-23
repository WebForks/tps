// GLM-5.1: 754B MoE, ~40B active, 256 experts (8 active), 200K context, GlmMoeDSA architecture
// Released: March 2026
// Source: https://huggingface.co/zai-org/GLM-5.1
export default {
  id: 'glm5_1',
  name: 'GLM-5.1',
  type: 'moe',
  params: 754,
  active_params: 40,
  experts: 256,
  experts_per_token: 8,
  kv_lora_rank: 512,
  qk_nope_head_dim: 192,
  qk_rope_head_dim: 64,
  v_head_dim: 256,
  query_heads: 64,
  layers: 78,
  kv_heads: 64,
  head_dim: 64,
  hidden_size: 6144,
  max_ctx: 202752,
  tags: ['chat', 'multilingual'],
  released: '2026-03',
  links: {
    hf: 'https://huggingface.co/zai-org/GLM-5.1',
  },
}
