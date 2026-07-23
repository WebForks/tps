// Qwen3.6 35B A3B: MoE, 35B total / 3B active, hybrid Gated DeltaNet(linear) + standard attention
// full_attention_interval=4: 30 linear layers (no KV cache) + 10 full attention layers
// Source: https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json
export default {
  id: 'qwen3_6_35b_a3b',
  name: 'Qwen3.6 35B A3B (MoE)',
  type: 'moe',
  params: 35,
  active_params: 3,
  experts: 256,
  experts_per_token: 8,
  layers: 40,
  query_heads: 16,
  kv_heads: 2,           // full attention KV heads
  head_dim: 256,
  linear_attention_layers: 30,
  linear_num_key_heads: 16,
  linear_num_value_heads: 32,
  linear_key_head_dim: 128,
  linear_value_head_dim: 128,
  linear_conv_kernel_dim: 4,
  linear_state_bytes: 4, // mamba_ssm_dtype=float32
  hidden_size: 2048,
  max_ctx: 262144,
  tags: ['chat', 'multilingual', 'vision', 'multimodal'],
  released: '2026-04',
  links: {
    hf: 'https://huggingface.co/Qwen/Qwen3.6-35B-A3B',
    ms: 'https://modelscope.cn/models/Qwen/Qwen3.6-35B-A3B',
  },
}
