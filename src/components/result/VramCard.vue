<script setup>
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { fmtGB, fmtPct } from '../../utils/format.js'
import { getDisplayVramBreakdown } from '../../utils/vramBreakdown.js'

const { t } = useI18n()
const props = defineProps({ result: Object, quantMatrix: Array, currentQuantId: String, readonly: Boolean, gpuCount: { type: Number, default: 1 } })
const emit = defineEmits(['selectQuant'])
const showMatrix = ref(true) // 默认展开

const isPureCpu = computed(() => Boolean(props.result?.pureCpu))
const isSharedPool = computed(() => Boolean(props.result?.sharedSystemMemory))
const displayNeeded = computed(() => {
  if (isPureCpu.value) return props.result?.cpuRamNeededGB ?? 0
  if (isSharedPool.value) return props.result?.sharedSystemRamNeededGB ?? 0
  return props.result?.displayNeeded ?? props.result?.totalNeeded ?? 0
})
const displayVram = computed(() => {
  if (isPureCpu.value) return props.result?.availableSysRamGB ?? 0
  if (isSharedPool.value) return props.result?.sharedPoolAvailableGB ?? 0
  return props.result?.displayVram ?? props.result?.totalVram ?? 0
})
const isPerCard = computed(() =>
  !isSharedPool.value
  && (props.result?.vramScope ?? (props.gpuCount > 1 ? 'per_card' : 'total')) === 'per_card'
)
const memoryOk = computed(() => {
  if (isPureCpu.value) return Boolean(props.result?.ramOk)
  if (isSharedPool.value) return Boolean(props.result?.vramOk && props.result?.ramOk)
  return Boolean(props.result?.vramOk)
})
const usesSecondarySystemRam = computed(() =>
  !isPureCpu.value && (props.result?.cpuRamNeededGB ?? 0) > 0
)
const secondaryRamPct = computed(() => {
  const needed = props.result?.cpuRamNeededGB ?? 0
  const available = props.result?.availableSysRamGB ?? 0
  if (needed <= 0) return 0
  return available > 0 ? needed / available * 100 : 100
})
const sharedAllocationPct = computed(() => {
  const allocation = props.result?.sharedAllocationGB ?? 0
  const available = props.result?.availableSysRamGB ?? 0
  if (allocation <= 0) return 0
  return available > 0 ? allocation / available * 100 : 100
})
const memoryPct = computed(() => {
  if (!props.result) return 0
  if (!isPureCpu.value && !isSharedPool.value) return props.result.vramPct ?? 0
  if (displayNeeded.value <= 0) return 0
  return displayVram.value > 0 ? displayNeeded.value / displayVram.value * 100 : 100
})

const barColor = computed(() => {
  if (!props.result) return 'bg-emerald-500'
  if (!memoryOk.value) return 'bg-red-500'
  if (memoryPct.value > 95) return 'bg-yellow-400'
  return 'bg-emerald-500'
})

const barWidth = computed(() => {
  if (!props.result) return '0%'
  return Math.min(100, memoryPct.value).toFixed(1) + '%'
})

const vramRating = computed(() => {
  if (!props.result) return null
  if (!memoryOk.value)      return { dot: 'bg-red-500',    label: t('result.vram_rating_oom'),         desc: t('result.vram_rating_oom_desc') }
  if (memoryPct.value > 95) return { dot: 'bg-yellow-400', label: t('result.vram_rating_tight'),       desc: t('result.vram_rating_tight_desc') }
  return                           { dot: 'bg-green-500',  label: t('result.vram_rating_comfortable'), desc: isPureCpu.value || isSharedPool.value ? t('result.ram_rating_comfortable_desc') : t('result.vram_rating_comfortable_desc') }
})

const pieData = computed(() => {
  if (!props.result) return []
  const { weightGB, kvGB, overheadGB, activationGB } = getDisplayVramBreakdown(props.result)
  return [
    { label: t('result.weight'), value: weightGB, color: 'bg-emerald-500' },
    { label: t('result.kv'),     value: kvGB,     color: 'bg-amber-500' },
    { label: t('result.activation'), value: activationGB, color: 'bg-blue-500' },
    { label: t('result.overhead'), value: overheadGB, color: 'bg-slate-500' },
  ]
})
</script>

<template>
  <div class="bg-white rounded-xl border border-gray-200 p-4 space-y-3">
    <div class="flex items-center justify-between">
      <h3 class="text-sm font-semibold text-gray-700">{{ isPureCpu ? t('result.system_ram') : isSharedPool ? t('gpu.shared_vram_label') : t('result.vram') }}</h3>
      <span v-if="result" :class="['text-xs font-medium px-2 py-0.5 rounded-full', memoryOk ? 'bg-emerald-100 text-emerald-700' : 'bg-red-100 text-red-700']">
        {{ memoryOk ? (isPureCpu || isSharedPool ? t('result.ram_ok') : t('result.vram_ok')) : (isPureCpu || isSharedPool ? t('result.ram_oom') : t('result.vram_oom')) }}
      </span>
    </div>

    <div v-if="result && isSharedPool" class="space-y-1 rounded-lg border border-gray-100 bg-gray-50 px-3 py-2">
      <div class="flex items-center justify-between text-xs">
        <span class="font-medium text-gray-600">{{ t('gpu.shared_vram_label') }}</span>
        <span :class="result.sharedAllocationExceedsRam ? 'text-red-600' : 'text-emerald-700'">
          {{ result.sharedAllocationExceedsRam ? t('result.ram_oom') : t('result.ram_ok') }}
        </span>
      </div>
      <div class="flex justify-between text-xs text-gray-500">
        <span>{{ fmtGB(result.sharedAllocationGB ?? 0) }} / {{ fmtGB(result.availableSysRamGB ?? 0) }}</span>
        <span>{{ fmtPct(sharedAllocationPct) }}</span>
      </div>
      <div class="h-1.5 bg-gray-200 rounded-full overflow-hidden">
        <div
          :class="['h-full rounded-full', result.sharedAllocationExceedsRam ? 'bg-red-500' : sharedAllocationPct > 95 ? 'bg-yellow-400' : 'bg-emerald-500']"
          :style="{ width: Math.min(100, sharedAllocationPct).toFixed(1) + '%' }"
        />
      </div>
      <div class="flex items-center justify-between text-xs pt-1 text-gray-500">
        <span>{{ t('result.available_ram') }}</span>
        <span class="font-medium text-gray-700">{{ fmtGB(result.availableSysRamGB ?? 0) }}</span>
      </div>
    </div>

    <!-- 显存体验评级 -->
    <div v-if="vramRating" class="flex items-center gap-2 px-3 py-2 bg-gray-50 rounded-lg border border-gray-100">
      <div :class="['w-2.5 h-2.5 rounded-full flex-shrink-0', vramRating.dot]" />
      <span class="text-sm font-semibold text-gray-800">{{ vramRating.label }}</span>
      <span class="text-xs text-gray-400">{{ vramRating.desc }}</span>
    </div>

    <!-- 进度条 -->
    <div v-if="result" class="space-y-1">
      <div v-if="isPerCard" class="text-xs text-blue-600 bg-blue-50 rounded px-2 py-1 border border-blue-100">
        {{ t('result.vram_per_card_note', { count: result.tpCount ?? gpuCount }) }}
      </div>
      <div class="flex justify-between text-xs text-gray-500">
        <span>{{ fmtGB(displayNeeded) }} / {{ fmtGB(displayVram) }}</span>
        <span>{{ fmtPct(memoryPct) }}</span>
      </div>
      <div class="h-2 bg-gray-200 rounded-full overflow-hidden">
        <div
          :class="['h-full rounded-full transition-all duration-500', barColor]"
          :style="{ width: barWidth }"
        />
      </div>
    </div>

    <div v-if="result && usesSecondarySystemRam" class="space-y-1 rounded-lg border border-gray-100 bg-gray-50 px-3 py-2">
      <div class="flex items-center justify-between text-xs">
        <span class="font-medium text-gray-600">{{ t('result.system_ram_usage') }}</span>
        <span :class="result.ramOk ? 'text-emerald-700' : 'text-red-600'">
          {{ result.ramOk ? t('result.ram_ok') : t('result.ram_oom') }}
        </span>
      </div>
      <div class="flex justify-between text-xs text-gray-500">
        <span>{{ fmtGB(result.cpuRamNeededGB) }} / {{ fmtGB(result.availableSysRamGB ?? 0) }}</span>
        <span>{{ fmtPct(secondaryRamPct) }}</span>
      </div>
      <div class="h-1.5 bg-gray-200 rounded-full overflow-hidden">
        <div
          :class="['h-full rounded-full', result.ramOk ? secondaryRamPct > 95 ? 'bg-yellow-400' : 'bg-emerald-500' : 'bg-red-500']"
          :style="{ width: Math.min(100, secondaryRamPct).toFixed(1) + '%' }"
        />
      </div>
    </div>

    <!-- 构成列表 -->
    <div v-if="result" class="space-y-1.5">
      <div v-for="item in pieData" :key="item.label" class="flex items-center justify-between text-xs">
        <div class="flex items-center gap-2">
          <div :class="['w-2 h-2 rounded-full', item.color]" />
          <span class="text-gray-500">{{ item.label }}</span>
        </div>
        <span class="text-gray-800 font-medium">{{ fmtGB(item.value) }}</span>
      </div>
      <div class="flex items-center justify-between text-xs pt-1 border-t border-gray-200">
        <span class="text-gray-500">{{ isPureCpu || isSharedPool ? t('result.available_ram') : isPerCard ? t('result.available_per_card') : t('result.available') }}</span>
        <span class="text-gray-800 font-medium">{{ fmtGB(displayVram) }}</span>
      </div>
      <div v-if="isPerCard && result.clusterNeeded != null" class="flex items-center justify-between text-xs text-gray-400">
        <span>{{ t('result.cluster_vram_total') }}</span>
        <span>{{ fmtGB(result.clusterNeeded) }} / {{ fmtGB(result.totalVram) }}</span>
      </div>
    </div>
    <!-- 理论估算注释 -->
    <p v-if="result" class="text-xs text-gray-400 leading-snug">{{ t('result.vram_theory_note') }}</p>

    <!-- 量化对比矩阵 -->
    <div v-if="quantMatrix?.length" class="border-t border-gray-100 pt-3 space-y-1">
      <button
        @click="showMatrix = !showMatrix"
        class="w-full flex items-center justify-between text-sm font-semibold text-gray-700 hover:text-gray-900 transition-colors px-1 pb-2"
      >
        <span>{{ t('result.quant_comparison') }}</span>
        <svg
          :class="['w-4 h-4 transition-transform', showMatrix ? 'rotate-180' : '']"
          fill="none" stroke="currentColor" viewBox="0 0 24 24"
        >
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      <div v-show="showMatrix" class="space-y-1">
        <div class="grid grid-cols-5 text-xs text-gray-400 font-medium px-1 pb-1">
          <span>{{ t('result.quant_matrix_quant') }}</span>
          <span class="text-right">{{ isPureCpu || isSharedPool ? t('result.quant_matrix_ram') : t('result.quant_matrix_vram') }}</span>
          <span class="text-center">{{ t('result.quant_matrix_status') }}</span>
          <span class="text-right">{{ t('result.quant_matrix_speed') }}</span>
          <span class="text-right">{{ t('result.quant_matrix_ppl') }}</span>
        </div>
        <div
          v-for="row in quantMatrix" :key="row.quant.id"
          @click="!props.readonly && row.configurationOk && emit('selectQuant', row.quant)"
          class="grid grid-cols-5 text-xs px-2 py-1.5 rounded items-center transition-colors"
          :class="[
            row.quant.id === currentQuantId ? 'bg-emerald-600 text-white' : '',
            !props.readonly && row.configurationOk && row.quant.id !== currentQuantId ? 'cursor-pointer hover:bg-gray-100' : '',
            !row.configurationOk ? 'opacity-70 cursor-not-allowed' : '',
            props.readonly ? 'cursor-default' : '',
          ]"
        >
          <span class="font-medium" :class="row.quant.id === currentQuantId ? 'text-white' : 'text-gray-700'">{{ row.quant.label }}</span>
          <span class="text-right" :class="row.quant.id === currentQuantId ? 'text-white' : 'text-gray-600'">{{ fmtGB(row.vramGB) }}</span>
          <span class="text-center">
            <span
              v-if="!row.configurationOk"
              :class="row.quant.id === currentQuantId ? 'text-amber-200' : 'text-amber-600'"
            >{{ t('result.quant_matrix_unsupported') }}</span>
            <span v-else-if="row.memoryOk" :class="row.quant.id === currentQuantId ? 'text-emerald-200' : 'text-emerald-600'">✓ {{ fmtPct(row.memoryPct) }}</span>
            <span v-else-if="row.cpuOffloadFeasible" :class="row.quant.id === currentQuantId ? 'text-amber-200' : 'text-amber-500'" :title="t('result.quant_matrix_offloadable_tip')">⚡ {{ fmtGB(row.offloadVramGB) }}</span>
            <span v-else :class="row.quant.id === currentQuantId ? 'text-red-200' : 'text-red-500'">✗ OOM</span>
          </span>
          <span class="text-right" :class="row.quant.id === currentQuantId ? 'text-white' : 'text-gray-500'">
            {{ row.fitOk ? (row.effectiveToks ?? row.decodeToks).toFixed(1) + ' tok/s' : '—' }}
          </span>
          <span class="text-right" :class="row.quant.id === currentQuantId ? 'text-emerald-100' : 'text-gray-400'">
            {{ row.quant.ppl_loss != null ? (row.quant.ppl_loss === 0 ? '—' : '+' + row.quant.ppl_loss.toFixed(2)) : '?' }}
          </span>
        </div>
        <p class="text-xs text-gray-400 leading-snug pt-1">{{ t('result.vram_theory_note') }}</p>
      </div>
    </div>
  </div>
</template>
