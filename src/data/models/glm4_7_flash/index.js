// GLM-4.7-Flash: 30B MoE with 3B active, lightweight deployment
// Source: https://huggingface.co/zai-org/GLM-4.7-Flash
export default {
  id: 'glm4_7_flash',
  released: '2025-12',
  name: 'GLM-4.7-Flash',
  type: 'moe',
  params: 30,
  active_params: 3,
  experts: 64,
  experts_per_token: 4,
  non_expert_params: 1.2,
  layers: 47,
  query_heads: 20,
  kv_heads: 20,
  head_dim: 256,
  kv_lora_rank: 512,
  qk_nope_head_dim: 192,
  qk_rope_head_dim: 64,
  v_head_dim: 256,
  hidden_size: 2048,
  max_ctx: 202752,
  tags: ['chat', 'multilingual'],

  links: {
    ollama: null,
    hf: 'https://huggingface.co/zai-org/GLM-4.7-Flash',
    ms: 'https://modelscope.cn/models/ZhipuAI/glm-4.7-flash'
  }
}
