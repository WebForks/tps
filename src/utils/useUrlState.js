// src/utils/useUrlState.js
import { computed, watch } from 'vue'
import { GPU_LIST } from '../data/gpus/index.js'
import { ALL_MODELS, resolveModelId } from '../data/models/index.js'
import { QUANT_MAP, INTERCONNECT_MAP, FRAMEWORK_MAP } from '../data/constants.js'
import {
  KV_CACHE_MAP,
  PCIE_BW_OPTIONS,
  PCIE_WIDTH_OPTIONS,
  createCpuMemBwOption,
  normalizeCpuMemMeasuredBandwidth,
  normalizeCpuTflops,
  normalizeRamCapacity,
  resolveCpuMemBwOption,
} from '../data/runtime.js'

const SESSION_KEY = 'tps_estimator_query'
const LEGACY_SESSION_KEY = 'tps_calc_query'
const MAX_GPU_COUNT = 512
const MAX_CONTEXT = 10_485_760

function readNumber(params, key, { min = -Infinity, max = Infinity, integer = false } = {}) {
  if (!params.has(key)) return null
  const number = Number(params.get(key))
  if (!Number.isFinite(number)) return null
  const normalized = integer ? Math.round(number) : number
  return Math.min(max, Math.max(min, normalized))
}

function getSavedSearch() {
  return sessionStorage.getItem(SESSION_KEY) ?? sessionStorage.getItem(LEGACY_SESSION_KEY) ?? ''
}

function getParams() {
  return new URLSearchParams(window.location.search)
}

function readCustomModel(params) {
  if (params.get('model') !== 'custom') return null

  const type = params.get('cmt') === 'moe' ? 'moe' : 'dense'
  const paramsB = readNumber(params, 'cmp', { min: 0.01, max: 10_000 }) ?? 7
  const activeParams = type === 'moe'
    ? readNumber(params, 'cma', { min: 0.01, max: paramsB }) ?? paramsB
    : paramsB
  const model = {
    id: 'custom',
    name: String(params.get('cmn') || 'Custom Model').slice(0, 120),
    type,
    params: paramsB,
    active_params: activeParams,
    layers: readNumber(params, 'cml', { min: 1, max: 1_000, integer: true }) ?? 32,
    kv_heads: readNumber(params, 'cmkv', { min: 1, max: 1_024, integer: true }) ?? 8,
    head_dim: readNumber(params, 'cmhd', { min: 1, max: 4_096, integer: true }) ?? 128,
    hidden_size: readNumber(params, 'cmhs', { min: 1, max: 1_000_000, integer: true }) ?? 4096,
    max_ctx: readNumber(params, 'cmctx', { min: 512, max: MAX_CONTEXT, integer: true }) ?? 32768,
    links: { ollama: null, hf: null, ms: null },
    tags: [],
  }

  const queryHeads = readNumber(params, 'cmqh', { min: 1, max: 1_024, integer: true })
  if (queryHeads != null) model.query_heads = queryHeads
  const mlaRatio = readNumber(params, 'cmmla', { min: 0.001, max: 1 })
  if (mlaRatio != null) model.mla_ratio = mlaRatio
  if (type === 'moe') {
    const experts = readNumber(params, 'cmexp', { min: 1, max: 10_000, integer: true })
    const topK = readNumber(params, 'cmk', {
      min: 1,
      max: experts ?? 10_000,
      integer: true,
    })
    if (experts != null) model.experts = experts
    if (topK != null) model.experts_per_token = topK
  }
  return model
}

function readCpuOffloadMode(params) {
  if (!params.has('co')) return 'auto'
  const value = String(params.get('co')).toLowerCase()
  if (value === '1' || value === 'on') return 'on'
  if (value === '0' || value === 'off') return 'off'
  return value === 'auto' ? 'auto' : 'auto'
}

function resolveInterconnect(id, gpuSlots) {
  const shared = INTERCONNECT_MAP.find(item => item.id === id)
  if (shared) return shared
  if (id === 'pcie3') {
    return { id: 'pcie3', label: 'PCIe 3.0', bw: 16, scope: 'intra', duplexBw: 32 }
  }

  const match = /^nvlink_(\d+(?:\.\d+)?)$/.exec(String(id ?? ''))
  const primary = gpuSlots?.[0]?.gpu
  const requestedDuplexBw = match ? Number(match[1]) : null
  const sameGpu = gpuSlots?.every(slot => slot.gpu?.id === primary?.id)
  if (
    requestedDuplexBw != null
    && sameGpu
    && Number(primary?.nvlink_bw) === requestedDuplexBw
  ) {
    return {
      id: `nvlink_${requestedDuplexBw}`,
      label: `NVLink (${requestedDuplexBw} GB/s duplex)`,
      bw: requestedDuplexBw / 2,
      scope: 'intra',
      duplexBw: requestedDuplexBw,
      derived: true,
    }
  }
  return null
}

function setParams(updates) {
  const url = new URL(window.location.href)
  for (const [k, v] of Object.entries(updates)) {
    if (v == null) url.searchParams.delete(k)
    else url.searchParams.set(k, v)
  }
  // Vue Router stores its navigation position and scroll metadata in the
  // history entry. Replacing it with `{}` breaks later router navigation.
  window.history.replaceState(window.history.state, '', url.toString())
  // 同步保存到 sessionStorage，供切换路由后恢复
  sessionStorage.setItem(SESSION_KEY, url.search)
  sessionStorage.setItem(LEGACY_SESSION_KEY, url.search)
}

/** 从 URL 读取初始状态，URL 无参数时回退到 sessionStorage */
export function readUrlState() {
  let search = window.location.search
  if (!search || search === '?') {
    search = getSavedSearch()
    // 把 sessionStorage 的 query 恢复到 URL
    if (search) {
      window.history.replaceState(
        window.history.state,
        '',
        window.location.pathname + search,
      )
    }
  }
  const p = new URLSearchParams(search)
  return {
    gpuSlots:       p.get('gpus') ?? null,          // "id1:n1,id2:n2"
    gpuId:          p.get('gpu'),                    // legacy fallback
    gpuCount:       readNumber(p, 'n', { min: 1, max: MAX_GPU_COUNT, integer: true }), // legacy fallback
    interconnectId: p.get('ic'),
    modelId:        p.get('model'),
    customModel:    readCustomModel(p),
    quantId:        p.get('quant'),
    ctx:            readNumber(p, 'ctx', { min: 512, max: MAX_CONTEXT, integer: true }),
    batch:          readNumber(p, 'b', { min: 1, max: 256, integer: true }),
    promptLen:      readNumber(p, 'pl', { min: 1, max: MAX_CONTEXT, integer: true }),
    outputLen:      readNumber(p, 'ol', { min: 1, max: MAX_CONTEXT, integer: true }),
    frameworkId:    p.get('fw'),
    flashAttention: p.has('fa')   ? p.get('fa') !== '0'  : null,
    kvCacheQuantId: p.get('kv'),
    prefixCacheHit: readNumber(p, 'pc', { min: 0, max: 90 }),
    cpuOffloadMode: readCpuOffloadMode(p),
    cpuOffload:     p.has('co')
      ? ['1', 'on'].includes(String(p.get('co')).toLowerCase())
      : null,
    pcieBwId:       p.get('pcie'),
    pcieWidthId:    p.get('pw'),
    pureCpu:        p.has('pcpu') ? p.get('pcpu') === '1' : null,
    cpuMemBwId:     p.get('cmb'),
    cpuMemChannels: readNumber(p, 'cmc', { min: 1, max: 16, integer: true }),
    cpuMemMeasuredBw: normalizeCpuMemMeasuredBandwidth(p.get('cmm')),
    cpuTflops:      normalizeCpuTflops(p.get('ctf')),
    gpuMemoryUtilization: readNumber(p, 'gmu', { min: 0.5, max: 1 }),
    sysRam:         p.has('sr')   ? normalizeRamCapacity(p.get('sr')) : null,
    sharedVram:     readNumber(p, 'sv', { min: 1, max: 512, integer: true }),
    speculativeDecoding: p.has('sd')  ? p.get('sd') === '1' : null,
    acceptanceRate:      readNumber(p, 'ar', { min: 0.3, max: 0.9 }),
    draftLen:            readNumber(p, 'dl', { min: 2, max: 8, integer: true }),
    draftModelParams:    readNumber(p, 'dmp', { min: 0.1, max: 32 }),
    ppCount:             readNumber(p, 'pp', { min: 1, max: 256, integer: true }),
    epCount:             readNumber(p, 'ep', { min: 1, max: 256, integer: true }),
    imageCount:          readNumber(p, 'img', { min: 0, max: 8, integer: true }),
    nglCount:            readNumber(p, 'ngl', { min: 0, max: 1000, integer: true }),
  }
}

/** 解析初始值到对应对象 */
export function resolveUrlState(init) {
  // gpuSlots 优先；无则回退旧 gpu+gpuCount 参数
  let gpuSlots = null
  if (init.gpuSlots) {
    let remainingGpuCount = MAX_GPU_COUNT
    const parsed = []
    for (const s of init.gpuSlots.split(',')) {
      if (remainingGpuCount <= 0) break
      const [id, count] = s.split(':')
      const parsedCount = Number(count)
      const gpu = GPU_LIST.find(g => g.id === id && g.unitKind !== 'cpu') ?? null
      if (!gpu) continue
      const normalizedCount = Number.isFinite(parsedCount)
        ? Math.min(remainingGpuCount, Math.max(1, Math.round(parsedCount)))
        : 1
      parsed.push({
        gpu,
        count: normalizedCount,
      })
      remainingGpuCount -= normalizedCount
    }
    if (parsed.length) gpuSlots = parsed
  }
  if (!gpuSlots && init.gpuId) {
    const gpu = GPU_LIST.find(g => g.id === init.gpuId && g.unitKind !== 'cpu')
    if (gpu) gpuSlots = [{ gpu, count: init.gpuCount ?? 1 }]
  }
  // CPU-only owns the entire model execution path, so an old/manually-edited
  // URL that enables both modes resolves deterministically to CPU-only.
  const pureCpu = init.pureCpu
  const requestedOffloadMode = ['auto', 'on', 'off'].includes(init.cpuOffloadMode)
    ? init.cpuOffloadMode
    : init.cpuOffload == null
      ? 'auto'
      : init.cpuOffload
        ? 'on'
        : 'off'
  const cpuOffloadMode = pureCpu ? 'off' : requestedOffloadMode
  const cpuOffload = cpuOffloadMode === 'auto' ? null : cpuOffloadMode === 'on'
  const resolvedModel = ALL_MODELS.find(m => m.id === resolveModelId(init.modelId))
    ?? (init.modelId === 'custom' ? init.customModel : null)
  const resolvedCtx = init.ctx != null && resolvedModel?.max_ctx
    ? Math.min(init.ctx, resolvedModel.max_ctx)
    : init.ctx

  const baseCpuMemBw = resolveCpuMemBwOption(init.cpuMemBwId)
  const cpuMemBw = baseCpuMemBw && (
    init.cpuMemChannels != null
    || init.cpuMemMeasuredBw != null
  )
    ? createCpuMemBwOption(
        baseCpuMemBw.generation,
        baseCpuMemBw.transferRate,
        init.cpuMemChannels ?? baseCpuMemBw.channels,
        init.cpuMemMeasuredBw ?? baseCpuMemBw.measuredBw,
      )
    : baseCpuMemBw
  const resolvedNgl = init.nglCount != null && Number.isFinite(Number(resolvedModel?.layers))
    ? Math.min(init.nglCount, Math.max(0, Math.round(Number(resolvedModel.layers))))
    : init.nglCount
  const draftCap = Math.max(
    0.5,
    Math.min(32, Number(resolvedModel?.active_params ?? resolvedModel?.params ?? 8) * 0.5),
  )

  return {
    gpuSlots,
    interconnect: resolveInterconnect(init.interconnectId, gpuSlots),
    model:        resolvedModel,
    quant:        QUANT_MAP.find(q => q.id === init.quantId) ?? null,
    ctx:          resolvedCtx,
    batch:        init.batch,
    promptLen:    init.promptLen,
    outputLen:    init.outputLen,
    framework:    FRAMEWORK_MAP.find(f => f.id === init.frameworkId) ?? null,
    flashAttention: init.flashAttention,
    kvCacheQuant: KV_CACHE_MAP.find(k => k.id === init.kvCacheQuantId) ?? null,
    prefixCacheHit: init.prefixCacheHit,
    cpuOffloadMode,
    cpuOffload,
    pcieBw:       PCIE_BW_OPTIONS.find(p => p.id === init.pcieBwId) ?? null,
    pcieWidth:    PCIE_WIDTH_OPTIONS.find(w => w.id === init.pcieWidthId) ?? null,
    pureCpu,
    cpuMemBw,
    cpuTflops:      init.cpuTflops,
    gpuMemoryUtilization: init.gpuMemoryUtilization,
    sharedVram:   init.sharedVram,
    sysRam:       init.sysRam,
    speculativeDecoding: init.speculativeDecoding,
    acceptanceRate:      init.acceptanceRate,
    draftLen:            init.draftLen,
    draftModelParams:    init.draftModelParams == null
      ? null
      : Math.min(draftCap, Math.max(0.1, init.draftModelParams)),
    ppCount:             init.ppCount,
    epCount:             init.epCount,
    imageCount:          init.imageCount,
    nglCount:            resolvedNgl,
  }
}

/** 监听所有 ref，变化时同步写入 URL 和 sessionStorage */
export function watchUrlState({
  gpuSlots, interconnect, model, quant, ctx, batch,
  promptLen, outputLen, framework, flashAttention,
  kvCacheQuant, prefixCacheHit, cpuOffload, cpuOffloadMode, pcieBw, pcieWidth, pureCpu, cpuMemBw,
  cpuTflops, gpuMemoryUtilization, sysRam, sharedVram,
  speculativeDecoding, acceptanceRate, draftLen, draftModelParams,
  ppCount, epCount, imageCount, nglCount,
}) {
  const effectiveCpuOffloadMode = cpuOffloadMode ?? computed(() => (
    cpuOffload.value == null ? 'auto' : cpuOffload.value ? 'on' : 'off'
  ))
  const cpuTflopsSource = cpuTflops ?? computed(() => null)
  const gpuMemoryUtilizationSource = gpuMemoryUtilization ?? computed(() => null)
  watch(
    [gpuSlots, interconnect, model, quant, ctx, batch,
     promptLen, outputLen, framework, flashAttention,
     kvCacheQuant, prefixCacheHit, cpuOffload, effectiveCpuOffloadMode, pcieBw, pcieWidth, pureCpu, cpuMemBw,
     cpuTflopsSource, gpuMemoryUtilizationSource, sysRam, sharedVram,
     speculativeDecoding, acceptanceRate, draftLen, draftModelParams,
     ppCount, epCount, imageCount, nglCount],
    ([slots, ic, m, q, c, b, pl, ol, fw, fa, kv, pc, co, coMode, pb, pw, pcpu, cmb, ctf, gmu, sr, sv, sd, ar, dl, dmp, pp, ep, img, ngl]) => {
      const normalizedRam = normalizeRamCapacity(sr)
      const hasSharedMemoryGpu = slots?.some(slot => slot.gpu?.sharedMemory)
      const usesCpuMemoryBandwidth = pcpu || co || hasSharedMemoryGpu
      const draftCap = Math.max(
        0.5,
        Math.min(32, Number(m?.active_params ?? m?.params ?? 8) * 0.5),
      )
      const normalizedDraftParams = Number.isFinite(Number(dmp))
        ? Math.min(draftCap, Math.max(0.1, Number(dmp)))
        : 1
      if (dmp !== normalizedDraftParams) {
        draftModelParams.value = normalizedDraftParams
        return
      }
      const normalizedNgl = ngl != null && Number.isFinite(Number(m?.layers))
        ? Math.min(Math.max(0, Math.round(Number(ngl))), Math.max(0, Math.round(Number(m.layers))))
        : ngl
      if (ngl !== normalizedNgl) {
        nglCount.value = normalizedNgl
        return
      }
      const custom = m?.id === 'custom' ? m : null
      const cpuMemBaseId = cmb?.generation && cmb?.transferRate
        ? `${cmb.generation}_${cmb.transferRate}`
        : cmb?.id ?? null
      const measuredCpuMemBw = normalizeCpuMemMeasuredBandwidth(cmb?.measuredBw)
      const normalizedCpuTflops = normalizeCpuTflops(ctf)
      const numericGpuMemoryUtilization = Number(gmu)
      const normalizedGpuMemoryUtilization = Number.isFinite(numericGpuMemoryUtilization)
        ? Math.min(1, Math.max(0.5, numericGpuMemoryUtilization))
        : null
      setParams({
        gpus:  slots?.length ? slots.map(s => `${s.gpu.id}:${s.count}`).join(',') : null,
        gpu:   null,
        n:     null,
        ic:    ic?.id  ?? null,
        model: m?.id   ?? null,
        cmn:   custom?.name ?? null,
        cmt:   custom?.type ?? null,
        cmp:   custom?.params ?? null,
        cma:   custom?.type === 'moe' ? custom?.active_params ?? null : null,
        cml:   custom?.layers ?? null,
        cmkv:  custom?.kv_heads ?? null,
        cmqh:  custom?.query_heads ?? null,
        cmhd:  custom?.head_dim ?? null,
        cmhs:  custom?.hidden_size ?? null,
        cmctx: custom?.max_ctx ?? null,
        cmexp: custom?.type === 'moe' ? custom?.experts ?? null : null,
        cmk:   custom?.type === 'moe' ? custom?.experts_per_token ?? null : null,
        cmmla: custom?.mla_ratio ?? null,
        quant: q?.id   ?? null,
        ctx:   c,
        b:     b !== 1 ? b : null,
        pl:    pl,
        ol:    ol,
        fw:    fw?.id  ?? null,
        fa:    fa === false ? '0' : null,
        kv:    kv?.id !== 'auto' ? kv?.id : null,
        pc:    pc > 0 ? pc : null,
        co:    pcpu || coMode === 'auto' ? null : coMode,
        pcie:  pb?.id  ?? null,
        pw:    co && !pcpu ? (pw?.id ?? null) : null,
        pcpu:  pcpu ? '1' : null,
        cmb:   usesCpuMemoryBandwidth ? cpuMemBaseId : null,
        cmc:   usesCpuMemoryBandwidth && cmb?.channels != null && cmb.channels !== 2
          ? cmb.channels
          : null,
        cmm:   usesCpuMemoryBandwidth ? measuredCpuMemBw : null,
        ctf:   normalizedCpuTflops,
        gmu:   normalizedGpuMemoryUtilization,
        sr:    usesCpuMemoryBandwidth && normalizedRam != null && normalizedRam !== 64 ? normalizedRam : null,
        sv:    sv != null && sv !== 16 ? sv : null,
        sd:    sd ? '1' : null,
        ar:    sd && ar != null && ar !== 0.7 ? ar : null,
        dl:    sd && dl != null && dl !== 4   ? dl : null,
        dmp:   sd && dmp != null && dmp !== 1 ? dmp : null,
        pp:    pp != null && pp !== 1 ? pp : null,
        ep:    ep != null && ep !== 1 ? ep : null,
        img:   img != null && img !== 0 ? img : null,
        ngl:   (co && fw?.id === 'llamacpp' && ngl != null) ? ngl : null,
      })
    },
    { immediate: true, deep: true }
  )
}
