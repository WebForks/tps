// DeepSeek R1-0528: latest reasoning model checkpoint
// Released: May 2025
// Source: https://huggingface.co/deepseek-ai/DeepSeek-R1-0528
export default {
  id: 'deepseek_r1_0528',
  released: '2025-05',
  name: 'DeepSeek R1-0528',
  type: 'moe',
  params: 671,
  active_params: 37,
  experts: 256,
  experts_per_token: 8,
  moe_execution: 'shared_routed',
  kv_lora_rank: 512,
  qk_nope_head_dim: 128,
  qk_rope_head_dim: 64,
  v_head_dim: 128,
  query_heads: 128,
  layers: 61,
  kv_heads: 128,
  head_dim: 128,
  hidden_size: 7168,
  max_ctx: 131072,
  tags: ['chat', 'multilingual', 'reasoning'],
  links: {
    hf: 'https://huggingface.co/deepseek-ai/DeepSeek-R1-0528',
    ms: 'https://modelscope.cn/models/deepseek-ai/DeepSeek-R1-0528',
  },
}
