// Codestral Mamba 7B: Mamba2 SSM architecture, code-specific, 256K ctx
// Source: https://huggingface.co/mistralai/Mamba-Codestral-7B-v0.1
export default {
  id: 'codestral_mamba_7b',
  released: '2024-07',
  name: 'Codestral Mamba 7B',
  type: 'dense',
  architecture: 'mamba',
  params: 7.3,
  layers: 64,
  kv_heads: 0, // Mamba uses SSM, not traditional attention
  head_dim: 64,
  hidden_size: 4096,
  linear_num_value_heads: 128,
  ssm_expansion: 2,
  ssm_state_size: 128,
  ssm_conv_kernel: 4,
  max_ctx: 262144,
  tags: ['chat', 'code'],

  links: {
    ollama: 'ollama pull codestral-mamba:7b',
    hf: 'https://huggingface.co/mistralai/Mamba-Codestral-7B-v0.1',
  },
}
