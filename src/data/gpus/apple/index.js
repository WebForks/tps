// Apple Silicon（统一内存架构）
// 数据来源：Apple 官方规格 / Wikipedia
// 当前产品核对：
// - M5 Pro / M5 Max:
//   https://support.apple.com/en-us/126318
// - M4 Max / M3 Ultra Mac Studio:
//   https://support.apple.com/en-us/122211
// 仅收录 Apple 已发布的芯片和可购买内存配置；不收录传闻中的 Ultra。
// bf16 为 GPU FP32 TFLOPS × 2 估算（Apple 未公布官方 BF16 数字）
// unifiedMemory: true — 使用系统内存，usableRatio 为 GPU 可用比例（约 85%）
// decodeBwScale: 同带宽不同 GPU 核数 / 代际的有效 decode 带宽系数（缺省由 calc.js 按芯片推断）
// tdp 为整机 SoC 功耗代理估算，非独立 GPU 功耗；Apple 不公布芯片 TDP
// bwUtilization 实测校准（llama.cpp #4167，LLaMA 7B Q4_0 tg128）：
//   Max 芯片（单 die）：0.90（实测误差 <10%）
//   M5 Max 40-core：0.99（实测 102.93-104.03 tok/s，对应 build 8e672ef）
//   Ultra 芯片（双 die UltraFusion）：0.67（跨 die 内存读取有额外延迟，实测误差 <5%）
//   Pro/base 芯片：0.82（无足够实测数据校准）
export default [
  // ── M5 Max ──────────────────────────────────────────
  // 40-core GPU, 614 GB/s
  { id: 'apple_m5_max_128g', name: 'Apple M5 Max (128GB, 40-core GPU)', vendor: 'apple', tier: 'consumer', released: '2026-03', vram: 128, bw: 614, bwUtilization: 0.99, bf16: 27.0, int8: 54.0, int4: null, nvlink_bw: null, tdp: 70, unifiedMemory: true, usableRatio: 0.85, gpuCores: 40 },
  { id: 'apple_m5_max_64g',  name: 'Apple M5 Max (64GB, 40-core GPU)',  vendor: 'apple', tier: 'consumer', released: '2026-03', vram: 64,  bw: 614, bwUtilization: 0.99, bf16: 27.0, int8: 54.0, int4: null, nvlink_bw: null, tdp: 70, unifiedMemory: true, usableRatio: 0.85, gpuCores: 40 },
  { id: 'apple_m5_max_48g',  name: 'Apple M5 Max (48GB, 40-core GPU)',  vendor: 'apple', tier: 'consumer', released: '2026-03', vram: 48,  bw: 614, bwUtilization: 0.99, bf16: 27.0, int8: 54.0, int4: null, nvlink_bw: null, tdp: 70, unifiedMemory: true, usableRatio: 0.85, gpuCores: 40 },
  // 32-core GPU, 460 GB/s
  { id: 'apple_m5_max_36g',  name: 'Apple M5 Max (36GB, 32-core GPU)',  vendor: 'apple', tier: 'consumer', released: '2026-03', vram: 36,  bw: 460, bwUtilization: 0.90, bf16: 21.6, int8: 43.2, int4: null, nvlink_bw: null, tdp: 62, unifiedMemory: true, usableRatio: 0.85, decodeBwScale: 1.0, gpuCores: 32 },
  // ── M5 Pro ──────────────────────────────────────────
  // 20-core GPU, 307 GB/s
  { id: 'apple_m5_pro_24g',  name: 'Apple M5 Pro (24GB, 16-core GPU)',  vendor: 'apple', tier: 'consumer', released: '2026-03', vram: 24,  bw: 307, bwUtilization: 0.82, bf16: 10.8, int8: 21.6, int4: null, nvlink_bw: null, tdp: 38, unifiedMemory: true, usableRatio: 0.85, decodeBwScale: 0.77, gpuCores: 16 },
  { id: 'apple_m5_pro_48g',  name: 'Apple M5 Pro (48GB, 20-core GPU)',  vendor: 'apple', tier: 'consumer', released: '2026-03', vram: 48,  bw: 307, bwUtilization: 0.82, bf16: 13.5, int8: 27.0, int4: null, nvlink_bw: null, tdp: 46, unifiedMemory: true, usableRatio: 0.85, decodeBwScale: 1.0, gpuCores: 20 },
  { id: 'apple_m5_pro_64g',  name: 'Apple M5 Pro (64GB, 20-core GPU)',  vendor: 'apple', tier: 'consumer', released: '2026-03', vram: 64,  bw: 307, bwUtilization: 0.82, bf16: 13.5, int8: 27.0, int4: null, nvlink_bw: null, tdp: 46, unifiedMemory: true, usableRatio: 0.85, decodeBwScale: 1.0, gpuCores: 20 },
  // ── M5 ──────────────────────────────────────────────
  // 10-core GPU, 153.6 GB/s
  { id: 'apple_m5_32g',      name: 'Apple M5 (32GB)',      vendor: 'apple', tier: 'consumer', released: '2025-10', vram: 32,  bw: 154, bwUtilization: 0.82, bf16: 6.0,  int8: 12.0, int4: null, nvlink_bw: null, tdp: 22, unifiedMemory: true, usableRatio: 0.85 },
  { id: 'apple_m5_24g',      name: 'Apple M5 (24GB)',      vendor: 'apple', tier: 'consumer', released: '2025-10', vram: 24,  bw: 154, bwUtilization: 0.82, bf16: 6.0,  int8: 12.0, int4: null, nvlink_bw: null, tdp: 22, unifiedMemory: true, usableRatio: 0.85 },
  { id: 'apple_m5_16g',      name: 'Apple M5 (16GB)',      vendor: 'apple', tier: 'consumer', released: '2025-10', vram: 16,  bw: 154, bwUtilization: 0.82, bf16: 6.0,  int8: 12.0, int4: null, nvlink_bw: null, tdp: 22, unifiedMemory: true, usableRatio: 0.85 },

  // ── M4 Max ──────────────────────────────────────────
  // 40-core GPU, 546 GB/s
  { id: 'apple_m4_max_128g', name: 'Apple M4 Max (128GB, 40-core GPU)', vendor: 'apple', tier: 'consumer', released: '2024-11', vram: 128, bw: 546, bwUtilization: 0.90, bf16: 21.2, int8: 42.4, int4: null, nvlink_bw: null, tdp: 70, unifiedMemory: true, usableRatio: 0.85, decodeBwScale: 1.0, gpuCores: 40 },
  { id: 'apple_m4_max_64g',  name: 'Apple M4 Max (64GB, 40-core GPU)',  vendor: 'apple', tier: 'consumer', released: '2024-11', vram: 64,  bw: 546, bwUtilization: 0.90, bf16: 21.2, int8: 42.4, int4: null, nvlink_bw: null, tdp: 70, unifiedMemory: true, usableRatio: 0.85, decodeBwScale: 1.0, gpuCores: 40 },
  { id: 'apple_m4_max_48g',  name: 'Apple M4 Max (48GB, 40-core GPU)',  vendor: 'apple', tier: 'consumer', released: '2024-11', vram: 48,  bw: 546, bwUtilization: 0.90, bf16: 21.2, int8: 42.4, int4: null, nvlink_bw: null, tdp: 70, unifiedMemory: true, usableRatio: 0.85, decodeBwScale: 1.0, gpuCores: 40 },
  { id: 'apple_m4_max_36g',  name: 'Apple M4 Max (36GB, 32-core GPU)', vendor: 'apple', tier: 'consumer', released: '2024-11', vram: 36,  bw: 410, bwUtilization: 0.90, bf16: 16.0, int8: 32.0, int4: null, nvlink_bw: null, tdp: 62, unifiedMemory: true, usableRatio: 0.85, decodeBwScale: 1.09, gpuCores: 32 },
  // ── M4 Pro ──────────────────────────────────────────
  // 20-core GPU, 273 GB/s
  { id: 'apple_m4_pro_64g',  name: 'Apple M4 Pro (64GB, 20-core GPU)',  vendor: 'apple', tier: 'consumer', released: '2024-11', vram: 64,  bw: 273, bwUtilization: 0.82, bf16: 10.6, int8: 21.2, int4: null, nvlink_bw: null, tdp: 46, unifiedMemory: true, usableRatio: 0.85, decodeBwScale: 1.0, gpuCores: 20 },
  { id: 'apple_m4_pro_48g',  name: 'Apple M4 Pro (48GB, 20-core GPU)',  vendor: 'apple', tier: 'consumer', released: '2024-11', vram: 48,  bw: 273, bwUtilization: 0.82, bf16: 10.6, int8: 21.2, int4: null, nvlink_bw: null, tdp: 46, unifiedMemory: true, usableRatio: 0.85, decodeBwScale: 1.0, gpuCores: 20 },
  { id: 'apple_m4_pro_24g',  name: 'Apple M4 Pro (24GB, 16-core GPU)',  vendor: 'apple', tier: 'consumer', released: '2024-11', vram: 24,  bw: 273, bwUtilization: 0.82, bf16: 8.5,  int8: 17.0, int4: null, nvlink_bw: null, tdp: 38, unifiedMemory: true, usableRatio: 0.85, decodeBwScale: 0.77, gpuCores: 16 },
  // ── M4 ──────────────────────────────────────────────
  // 10-core GPU, 120 GB/s
  { id: 'apple_m4_32g',      name: 'Apple M4 (32GB)',      vendor: 'apple', tier: 'consumer', released: '2024-11', vram: 32,  bw: 120, bwUtilization: 0.82, bf16: 4.6,  int8: 9.2,  int4: null, nvlink_bw: null, tdp: 22, unifiedMemory: true, usableRatio: 0.85 },
  { id: 'apple_m4_16g',      name: 'Apple M4 (16GB)',      vendor: 'apple', tier: 'consumer', released: '2024-11', vram: 16,  bw: 120, bwUtilization: 0.82, bf16: 4.6,  int8: 9.2,  int4: null, nvlink_bw: null, tdp: 22, unifiedMemory: true, usableRatio: 0.85 },

  // ── M3 Ultra ─────────────────────────────────────────
  // Base 60-core GPU configuration, 819 GB/s
  { id: 'apple_m3_ultra_96g',  name: 'Apple M3 Ultra (96GB, 60-core GPU)',  vendor: 'apple', tier: 'consumer', released: '2025-03', vram: 96,  bw: 819, bwUtilization: 0.67, bf16: 25.8, int8: 51.6, int4: null, nvlink_bw: null, tdp: 120, unifiedMemory: true, usableRatio: 0.85, gpuCores: 60 },
  // 80-core GPU, 819 GB/s
  { id: 'apple_m3_ultra_512g', name: 'Apple M3 Ultra (512GB, 80-core GPU)', vendor: 'apple', tier: 'consumer', released: '2025-03', vram: 512, bw: 819, bwUtilization: 0.67, bf16: 34.4, int8: 68.8, int4: null, nvlink_bw: null, tdp: 140, unifiedMemory: true, usableRatio: 0.85, gpuCores: 80 },
  { id: 'apple_m3_ultra_256g', name: 'Apple M3 Ultra (256GB, 80-core GPU)', vendor: 'apple', tier: 'consumer', released: '2025-03', vram: 256, bw: 819, bwUtilization: 0.67, bf16: 34.4, int8: 68.8, int4: null, nvlink_bw: null, tdp: 140, unifiedMemory: true, usableRatio: 0.85, gpuCores: 80 },
  // ── M3 Max ──────────────────────────────────────────
  // 40-core GPU, 400 GB/s
  { id: 'apple_m3_max_128g', name: 'Apple M3 Max (128GB)', vendor: 'apple', tier: 'consumer', released: '2023-11', vram: 128, bw: 400, bwUtilization: 0.90, bf16: 17.2, int8: 34.4, int4: null, nvlink_bw: null, tdp: 78, unifiedMemory: true, usableRatio: 0.85 },
  { id: 'apple_m3_max_96g',  name: 'Apple M3 Max (96GB)',  vendor: 'apple', tier: 'consumer', released: '2023-11', vram: 96,  bw: 400, bwUtilization: 0.90, bf16: 17.2, int8: 34.4, int4: null, nvlink_bw: null, tdp: 78, unifiedMemory: true, usableRatio: 0.85 },
  // 30-core GPU, 300 GB/s
  { id: 'apple_m3_max_64g',  name: 'Apple M3 Max (64GB, 30-core GPU)',  vendor: 'apple', tier: 'consumer', released: '2023-11', vram: 64,  bw: 300, bwUtilization: 0.90, bf16: 12.9, int8: 25.8, int4: null, nvlink_bw: null, tdp: 78, unifiedMemory: true, usableRatio: 0.85, decodeBwScale: 0.87, gpuCores: 30 },
  // ── M3 Pro ──────────────────────────────────────────
  // 18-core GPU, 153 GB/s
  { id: 'apple_m3_pro_36g',  name: 'Apple M3 Pro (36GB, 18-core GPU)',  vendor: 'apple', tier: 'consumer', released: '2023-11', vram: 36,  bw: 153, bwUtilization: 0.82, bf16: 6.2,  int8: 12.4, int4: null, nvlink_bw: null, tdp: 27, unifiedMemory: true, usableRatio: 0.85, decodeBwScale: 1.28, gpuCores: 18 },
  { id: 'apple_m3_pro_18g',  name: 'Apple M3 Pro (18GB, 18-core GPU)',  vendor: 'apple', tier: 'consumer', released: '2023-11', vram: 18,  bw: 153, bwUtilization: 0.82, bf16: 6.2,  int8: 12.4, int4: null, nvlink_bw: null, tdp: 27, unifiedMemory: true, usableRatio: 0.85, decodeBwScale: 1.28, gpuCores: 18 },
  // ── M3 ──────────────────────────────────────────────
  // 10-core GPU, 100 GB/s
  { id: 'apple_m3_24g',      name: 'Apple M3 (24GB)',      vendor: 'apple', tier: 'consumer', released: '2023-11', vram: 24,  bw: 100, bwUtilization: 0.82, bf16: 4.1,  int8: 8.2,  int4: null, nvlink_bw: null, tdp: 20, unifiedMemory: true, usableRatio: 0.85 },
  { id: 'apple_m3_16g',      name: 'Apple M3 (16GB)',      vendor: 'apple', tier: 'consumer', released: '2023-11', vram: 16,  bw: 100, bwUtilization: 0.82, bf16: 4.1,  int8: 8.2,  int4: null, nvlink_bw: null, tdp: 20, unifiedMemory: true, usableRatio: 0.85 },
  { id: 'apple_m3_8g',       name: 'Apple M3 (8GB)',       vendor: 'apple', tier: 'consumer', released: '2023-11', vram: 8,   bw: 100, bwUtilization: 0.82, bf16: 4.1,  int8: 8.2,  int4: null, nvlink_bw: null, tdp: 20, unifiedMemory: true, usableRatio: 0.85 },

  // ── M2 Ultra ─────────────────────────────────────────
  // 76-core GPU, 800 GB/s
  { id: 'apple_m2_ultra_192g', name: 'Apple M2 Ultra (192GB)', vendor: 'apple', tier: 'consumer', released: '2023-06', vram: 192, bw: 800, bwUtilization: 0.67, bf16: 27.2, int8: 54.4, int4: null, nvlink_bw: null, tdp: 150, unifiedMemory: true, usableRatio: 0.85 },
  // ── M2 Max ──────────────────────────────────────────
  // 38-core GPU, 400 GB/s
  { id: 'apple_m2_max_96g',  name: 'Apple M2 Max (96GB)',  vendor: 'apple', tier: 'consumer', released: '2023-01', vram: 96,  bw: 400, bwUtilization: 0.90, bf16: 13.6, int8: 27.2, int4: null, nvlink_bw: null, tdp: 100, unifiedMemory: true, usableRatio: 0.85 },
  { id: 'apple_m2_max_64g',  name: 'Apple M2 Max (64GB)',  vendor: 'apple', tier: 'consumer', released: '2023-01', vram: 64,  bw: 400, bwUtilization: 0.90, bf16: 13.6, int8: 27.2, int4: null, nvlink_bw: null, tdp: 100, unifiedMemory: true, usableRatio: 0.85 },
  { id: 'apple_m2_max_32g',  name: 'Apple M2 Max (32GB)',  vendor: 'apple', tier: 'consumer', released: '2023-01', vram: 32,  bw: 400, bwUtilization: 0.90, bf16: 13.6, int8: 27.2, int4: null, nvlink_bw: null, tdp: 100, unifiedMemory: true, usableRatio: 0.85 },
  // ── M2 Pro ──────────────────────────────────────────
  // 19-core GPU, 200 GB/s
  { id: 'apple_m2_pro_32g',  name: 'Apple M2 Pro (32GB)',  vendor: 'apple', tier: 'consumer', released: '2023-01', vram: 32,  bw: 200, bwUtilization: 0.82, bf16: 6.8,  int8: 13.6, int4: null, nvlink_bw: null, tdp: 67, unifiedMemory: true, usableRatio: 0.85 },
  { id: 'apple_m2_pro_16g',  name: 'Apple M2 Pro (16GB)',  vendor: 'apple', tier: 'consumer', released: '2023-01', vram: 16,  bw: 200, bwUtilization: 0.82, bf16: 6.8,  int8: 13.6, int4: null, nvlink_bw: null, tdp: 67, unifiedMemory: true, usableRatio: 0.85 },

  // ── M1 Ultra ─────────────────────────────────────────
  // 64-core GPU, 800 GB/s (2 × M1 Max)
  { id: 'apple_m1_ultra_128g', name: 'Apple M1 Ultra (128GB)', vendor: 'apple', tier: 'consumer', released: '2022-03', vram: 128, bw: 800, bwUtilization: 0.67, bf16: 21.0, int8: 42.0, int4: null, nvlink_bw: null, tdp: 150, unifiedMemory: true, usableRatio: 0.85 },
  // ── M1 Max ──────────────────────────────────────────
  // 32-core GPU, 400 GB/s
  { id: 'apple_m1_max_64g',  name: 'Apple M1 Max (64GB, 32-core GPU)',  vendor: 'apple', tier: 'consumer', released: '2021-10', vram: 64,  bw: 400, bwUtilization: 0.90, bf16: 10.4, int8: 20.8, int4: null, nvlink_bw: null, tdp: 92, unifiedMemory: true, usableRatio: 0.85, decodeBwScale: 0.49, gpuCores: 32 },
  { id: 'apple_m1_max_32g',  name: 'Apple M1 Max (32GB, 32-core GPU)',  vendor: 'apple', tier: 'consumer', released: '2021-10', vram: 32,  bw: 400, bwUtilization: 0.90, bf16: 10.4, int8: 20.8, int4: null, nvlink_bw: null, tdp: 92, unifiedMemory: true, usableRatio: 0.85, decodeBwScale: 0.49, gpuCores: 32 },
  // ── M1 Pro ──────────────────────────────────────────
  // 16-core GPU, 200 GB/s
  { id: 'apple_m1_pro_32g',  name: 'Apple M1 Pro (32GB, 16-core GPU)',  vendor: 'apple', tier: 'consumer', released: '2021-10', vram: 32,  bw: 200, bwUtilization: 0.82, bf16: 5.2,  int8: 10.4, int4: null, nvlink_bw: null, tdp: 67, unifiedMemory: true, usableRatio: 0.85, decodeBwScale: 0.63, gpuCores: 16 },
  { id: 'apple_m1_pro_16g',  name: 'Apple M1 Pro (16GB, 16-core GPU)',  vendor: 'apple', tier: 'consumer', released: '2021-10', vram: 16,  bw: 200, bwUtilization: 0.82, bf16: 5.2,  int8: 10.4, int4: null, nvlink_bw: null, tdp: 67, unifiedMemory: true, usableRatio: 0.85, decodeBwScale: 0.63, gpuCores: 16 },
  // ── M1 ──────────────────────────────────────────────
  // 8-core GPU, 68 GB/s
  { id: 'apple_m1_16g',      name: 'Apple M1 (16GB)',      vendor: 'apple', tier: 'consumer', released: '2020-11', vram: 16,  bw: 68,  bwUtilization: 0.82, bf16: 2.6,  int8: 5.2,  int4: null, nvlink_bw: null, tdp: 20, unifiedMemory: true, usableRatio: 0.85 },
  { id: 'apple_m1_8g',       name: 'Apple M1 (8GB)',       vendor: 'apple', tier: 'consumer', released: '2020-11', vram: 8,   bw: 68,  bwUtilization: 0.82, bf16: 2.6,  int8: 5.2,  int4: null, nvlink_bw: null, tdp: 20, unifiedMemory: true, usableRatio: 0.85 },
]
