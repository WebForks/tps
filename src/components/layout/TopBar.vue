<script setup>
import Header from './Header.vue'
import Footer from './Footer.vue'

defineProps({
  result: Object, model: Object,
  gpu: Object, gpuCount: Number, interconnect: Object,
  quant: Object, framework: Object,
  ctx: Number, batch: Number, promptLen: Number, outputLen: Number,
  flashAttention: Boolean, kvCacheQuant: Object,
  prefixCacheHit: Number, cpuOffload: Boolean, pcieBw: Object,
  pcieWidth: Object, pureCpu: Boolean, cpuMemBw: Object,
  cpuTflops: Number, gpuMemoryUtilization: Number, sysRam: Number,
})
</script>

<template>
  <Header
    :result="result"
    :model="model"
    :gpu="gpu"
    :gpu-count="gpuCount"
    :interconnect="interconnect"
    :quant="quant"
    :framework="framework"
    :ctx="ctx"
    :batch="batch"
    :prompt-len="promptLen"
    :output-len="outputLen"
    :flash-attention="flashAttention"
    :kv-cache-quant="kvCacheQuant"
    :prefix-cache-hit="prefixCacheHit"
    :cpu-offload="cpuOffload"
    :pcie-bw="pcieBw"
    :pcie-width="pcieWidth"
    :pure-cpu="pureCpu"
    :cpu-mem-bw="cpuMemBw"
    :cpu-tflops="cpuTflops"
    :gpu-memory-utilization="gpuMemoryUtilization"
    :sys-ram="sysRam"
  />
  <Footer />
</template>
