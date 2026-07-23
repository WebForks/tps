// OpenMoE 34B: Open-source MoE research model
// Source: https://github.com/XueFuzhao/OpenMoE
export default {
  id: 'openmoe_34b',
  released: '2024-02',
  name: 'OpenMoE 34B',
  type: 'moe',
  params: 34,
  active_params: 6.5,
  experts: 32,
  experts_per_token: 2,
  layers: 32,
  query_heads: 24,
  kv_heads: 24,
  head_dim: 128,
  hidden_size: 3072,
  max_ctx: 2048,
  tags: ['chat'],

  links: {
    ollama: null,
    hf: 'https://huggingface.co/OrionZheng/openmoe-34b-200B',
    ms: null
  }
}
