// Kimi K2.5: 原生多模态版本，基于K2-Base继续预训练，约15T混合视觉+文本token
// MoE + MLA, 61层, 384路由专家 top-8, 1.04T总参数, 32B激活, 256K上下文
// Sources:
// - https://huggingface.co/moonshotai/Kimi-K2.5
// - https://huggingface.co/moonshotai/Kimi-K2.5/blob/main/config.json
// - https://huggingface.co/moonshotai/Kimi-K2.5/blob/main/preprocessor_config.json
export default {
  id: 'kimi_k2_5',
  released: '2026-01',
  name: 'Kimi K2.5 (MoE)',
  type: 'moe',
  params: 1000,
  active_params: 32,
  experts: 384,
  experts_per_token: 8,
  kv_lora_rank: 512,
  qk_nope_head_dim: 128,
  qk_rope_head_dim: 64,
  v_head_dim: 128,
  query_heads: 64,
  layers: 61,
  kv_heads: 64,
  head_dim: 128,
  hidden_size: 7168,
  max_ctx: 262144,
  // MoonViT is 400M parameters. The processor accepts at most 16,384 input
  // patches and merges each 2x2 block, yielding at most 4,096 visual tokens.
  vision_encoder_params: 0.4,
  params_scope: 'total',
  vision_seq_tokens: 4096,
  tags: ['chat', 'multilingual', 'vision', 'multimodal'],
  links: {
    hf: 'https://huggingface.co/moonshotai/Kimi-K2.5',
    ms: 'https://modelscope.cn/models/moonshotai/Kimi-K2.5',
  },
}
