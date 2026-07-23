// Qwen3.6 27B: dense, 64 layers, hybrid Gated DeltaNet(linear) + standard attention
// full_attention_interval=4: 48 linear layers (no KV cache) + 16 full attention layers
// Source: https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/config.json
export default {
  id: 'qwen3_6_27b',
  name: 'Qwen3.6 27B',
  type: 'dense',
  params: 27,
  layers: 64,
  query_heads: 24,
  kv_heads: 4,           // full attention KV heads
  head_dim: 256,
  linear_attention_layers: 48,
  linear_num_key_heads: 16,
  linear_num_value_heads: 48,
  linear_key_head_dim: 128,
  linear_value_head_dim: 128,
  linear_conv_kernel_dim: 4,
  linear_state_bytes: 4, // mamba_ssm_dtype=float32
  hidden_size: 5120,
  max_ctx: 262144,
  tags: ['chat', 'multilingual', 'vision', 'multimodal'],
  released: '2026-04',
  links: {
    hf: 'https://huggingface.co/Qwen/Qwen3.6-27B',
    ms: 'https://modelscope.cn/models/Qwen/Qwen3.6-27B',
  },
}
