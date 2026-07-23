// Unofficial China-market VRAM modifications.
//
// These are not NVIDIA SKUs, and specifications can vary between sellers,
// board revisions, and VBIOS images. The entries below describe the specific
// configurations corroborated by the linked seller pages, not every card sold
// under the same marketplace name.
//
// Seller evidence (checked 2026-07-22):
// - RTX 2080 Ti 22GB: TU102, 22GB GDDR6, 14Gb/s, 352-bit, 260W
//   https://vietnamese.alibaba.com/product-detail/RTX-2080ti-Graphics-Card-GDDR6-22GB-1601193422287.html
// - RTX 3080 20GB: GA102, 20GB GDDR6X, 19Gb/s, 320-bit, 350W
//   https://portuguese.alibaba.com/product-detail/Turbo-rtx3080-20gb-pc-gaming-complete-62345032975.html
// - RTX 4080 SUPER 32GB: 32GB GDDR6X, 23Gb/s, 256-bit, 10,240 CUDA
//   cores, 736GB/s, and 320W. Some sellers shorten the name to "RTX 4080",
//   but those core and memory specifications identify the SUPER model.
//   https://www.alibaba.com/product-detail/RTX-4080-32GB-GDDR6X-Turbo-Graphics_1601740628258.html
// - RTX 4090 48GB: AD102, 48GB GDDR6X, 21Gb/s, 384-bit, 16,384 CUDA
//   cores, and 450W
//   https://chinese.alibaba.com/product-detail/RTX4090-48G-Brand-new-original-turbo-1601585571444.html
//
// `bw` is data rate * bus width / 8. A memory-capacity modification does not
// add shader/Tensor cores, so compute fields intentionally match the stock
// parent GPU. In this catalog `bf16` is the dense FP16/BF16-equivalent
// throughput used by the estimator. Turing cannot execute native BF16, so the
// RTX 2080 Ti value is its FP16 fallback and `nativeBf16` is false.
export default [
  {
    id: 'rtx4090_48g_mod',
    name: 'RTX 4090 48GB (China Mod)',
    vendor: 'nvidia',
    tier: 'consumer',
    released: null,
    vram: 48,
    bw: 1008,
    bwUtilization: 0.78,
    bf16: 165,
    int8: 330,
    int4: 661,
    nvlink_bw: null,
    pcie_gen: 4,
    tdp: 450,
    modified: true,
    official: false,
    specConfidence: 'seller-listed',
    sources: [
      { label: 'Alibaba seller listing', url: 'https://chinese.alibaba.com/product-detail/RTX4090-48G-Brand-new-original-turbo-1601585571444.html' },
    ],
    baseGpuId: 'rtx4090',
    memoryType: 'GDDR6X',
    memoryBusBits: 384,
    memoryDataRateGbps: 21,
    nativeBf16: true,
  },
  {
    id: 'rtx4080s_32g_mod',
    name: 'RTX 4080 SUPER 32GB (China Mod)',
    vendor: 'nvidia',
    tier: 'consumer',
    released: null,
    vram: 32,
    bw: 736,
    bwUtilization: 0.78,
    bf16: 104,
    int8: 208,
    int4: 416,
    nvlink_bw: null,
    pcie_gen: 4,
    tdp: 320,
    modified: true,
    official: false,
    specConfidence: 'seller-listed',
    sources: [
      { label: 'Alibaba seller listing', url: 'https://www.alibaba.com/product-detail/RTX-4080-32GB-GDDR6X-Turbo-Graphics_1601740628258.html' },
    ],
    baseGpuId: 'rtx4080s',
    memoryType: 'GDDR6X',
    memoryBusBits: 256,
    memoryDataRateGbps: 23,
    nativeBf16: true,
  },
  {
    id: 'rtx3080_20g_mod',
    name: 'RTX 3080 20GB (China Mod)',
    vendor: 'nvidia',
    tier: 'consumer',
    released: null,
    vram: 20,
    bw: 760,
    bwUtilization: 0.75,
    bf16: 60,
    int8: 119,
    int4: 238,
    nvlink_bw: null,
    pcie_gen: 4,
    tdp: 350,
    modified: true,
    official: false,
    specConfidence: 'seller-listed',
    sources: [
      { label: 'Alibaba seller listing', url: 'https://portuguese.alibaba.com/product-detail/Turbo-rtx3080-20gb-pc-gaming-complete-62345032975.html' },
    ],
    baseGpuId: 'rtx3080',
    memoryType: 'GDDR6X',
    memoryBusBits: 320,
    memoryDataRateGbps: 19,
    nativeBf16: true,
  },
  {
    id: 'rtx2080ti_22g_mod',
    name: 'RTX 2080 Ti 22GB (China Mod)',
    vendor: 'nvidia',
    tier: 'consumer',
    released: null,
    vram: 22,
    bw: 616,
    bwUtilization: 0.80,
    bf16: 56.9,
    int8: 227.7,
    int4: 455.4,
    nvlink_bw: null,
    pcie_gen: 3,
    tdp: 260,
    modified: true,
    official: false,
    specConfidence: 'seller-listed',
    sources: [
      { label: 'Alibaba seller listing', url: 'https://vietnamese.alibaba.com/product-detail/RTX-2080ti-Graphics-Card-GDDR6-22GB-1601193422287.html' },
    ],
    baseGpuId: 'rtx2080ti',
    memoryType: 'GDDR6',
    memoryBusBits: 352,
    memoryDataRateGbps: 14,
    nativeBf16: false,
    fp16EquivalentTflops: 56.9,
  },
]
