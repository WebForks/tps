// AMD 集成显卡（APU / Ryzen AI）
// 集成显卡共享系统内存，vram=0 表示由用户选择共享分配，bw 为内存带宽。
// Ryzen AI MAX+ 395 使用 256-bit LPDDR5X-8000（256 GB/s）统一内存。
// AMD 公布 Windows 最多 96 GB VGM，并公布 128 GB 系统最多 112 GB 可供
// GPU 分配的开发配置；BF16 峰值不是 AMD 公布规格，因此明确标为推导值。
// https://www.amd.com/en/newsroom/press-releases/2025-1-6-amd-announces-expanded-consumer-and-commercial-ai-.html
// https://www.amd.com/en/developer/resources/technical-articles/2025/amd-ryzen-ai-max-395--a-leap-forward-in-generative-ai-performanc.html
export default [
  { id: 'ryzen_9_7950x',       name: 'Ryzen 9 7950X',         vendor: 'amd', tier: 'consumer', released: '2022-09', vram: 0,  bw: 83,  bwUtilization: 0.80, bf16: 1,    int8: null, int4: null, nvlink_bw: null, tdp: 170, sharedMemory: true, unitKind: 'cpu' },
  { id: 'ryzen_ai_max_395',     name: 'Ryzen AI MAX+ 395 (112GB GPU allocation)', vendor: 'amd', tier: 'consumer', released: '2025-01', vram: 112, bw: 256, bwUtilization: 0.83, bf16: 36.9, int8: null, int4: null, nvlink_bw: null, tdp: 120, unifiedMemory: true, memoryModel: 'unified', computeEstimate: true, specConfidence: 'derived' },
  { id: 'ryzen_ai_max_395_win', name: 'Ryzen AI MAX+ 395 (Windows, 96GB VGM)',     vendor: 'amd', tier: 'consumer', released: '2025-01', vram: 96,  bw: 256, bwUtilization: 0.83, bf16: 36.9, int8: null, int4: null, nvlink_bw: null, tdp: 120, unifiedMemory: true, memoryModel: 'unified', computeEstimate: true, specConfidence: 'derived' },
  { id: 'radeon_890m',      name: 'Radeon 890M',       vendor: 'amd', tier: 'consumer', released: '2024-07', vram: 0,  bw: 102, bwUtilization: 0.80, bf16: 5,  int8: null, int4: null, nvlink_bw: null, tdp: 28,  sharedMemory: true },
  { id: 'radeon_880m',      name: 'Radeon 880M',       vendor: 'amd', tier: 'consumer', released: '2024-07', vram: 0,  bw: 89,  bwUtilization: 0.80, bf16: 4,  int8: null, int4: null, nvlink_bw: null, tdp: 28,  sharedMemory: true },
  { id: 'radeon_780m',      name: 'Radeon 780M',       vendor: 'amd', tier: 'consumer', released: '2023-02', vram: 0,  bw: 89,  bwUtilization: 0.80, bf16: 4,  int8: null, int4: null, nvlink_bw: null, tdp: 28,  sharedMemory: true },
  { id: 'radeon_760m',      name: 'Radeon 760M',       vendor: 'amd', tier: 'consumer', released: '2023-02', vram: 0,  bw: 64,  bwUtilization: 0.80, bf16: 3,  int8: null, int4: null, nvlink_bw: null, tdp: 28,  sharedMemory: true },
  { id: 'radeon_680m',      name: 'Radeon 680M',       vendor: 'amd', tier: 'consumer', released: '2022-01', vram: 0,  bw: 51,  bwUtilization: 0.80, bf16: 2,  int8: null, int4: null, nvlink_bw: null, tdp: 28,  sharedMemory: true },
  { id: 'radeon_660m',      name: 'Radeon 660M',       vendor: 'amd', tier: 'consumer', released: '2022-01', vram: 0,  bw: 38,  bwUtilization: 0.80, bf16: 2,  int8: null, int4: null, nvlink_bw: null, tdp: 15,  sharedMemory: true },
  { id: 'vega8',            name: 'Vega 8',            vendor: 'amd', tier: 'consumer', released: '2018-02', vram: 0,  bw: 38,  bwUtilization: 0.80, bf16: 1,  int8: null, int4: null, nvlink_bw: null, tdp: 15,  sharedMemory: true },
  { id: 'vega7',            name: 'Vega 7',            vendor: 'amd', tier: 'consumer', released: '2020-01', vram: 0,  bw: 38,  bwUtilization: 0.80, bf16: 1,  int8: null, int4: null, nvlink_bw: null, tdp: 15,  sharedMemory: true },
]
