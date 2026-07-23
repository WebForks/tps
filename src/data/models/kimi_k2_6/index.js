// Kimi K2.6: 1T MoE, 32B active, MLA attention, 256K context
// Released: April 2026
// Sources:
// - https://huggingface.co/moonshotai/Kimi-K2.6
// - https://huggingface.co/moonshotai/Kimi-K2.6/blob/main/config.json
// - https://huggingface.co/moonshotai/Kimi-K2.6/blob/main/preprocessor_config.json
export default {
  id: 'kimi_k2_6',
  name: 'Kimi K2.6 (MoE)',
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
  released: '2026-04',
  links: {
    hf: 'https://huggingface.co/moonshotai/Kimi-K2.6',
    ms: 'https://modelscope.cn/models/moonshotai/Kimi-K2.6',
  },
}
