// src/utils/exportMd.js
// 将当前计算结果导出为 Markdown 报告
import { fmtToks, fmtToksRange, fmtGB, fmtMs, fmtPct, fmtParams, fmtCtx } from './format.js'
import { calcAll, getWarnings } from './calc.js'
import { QUANT_MAP } from '../data/constants.js'
import {
  CPU_MEM_BW_OPTIONS,
  PCIE_BW_OPTIONS,
  PCIE_WIDTH_OPTIONS,
  createCpuMemBwOption,
} from '../data/runtime.js'
import { getDisplayVramBreakdown } from './vramBreakdown.js'
import { normalizeGpuMemoryUtilization } from './runtime.js'

/**
 * 生成 Markdown 报告字符串
 * @param {object} opts
 * @param {object} opts.gpu
 * @param {number} opts.gpuCount
 * @param {object} opts.interconnect
 * @param {object} opts.model
 * @param {object} opts.quant
 * @param {object} opts.framework
 * @param {number} opts.ctx
 * @param {number} opts.batch
 * @param {number} opts.promptLen
 * @param {number} opts.outputLen
 * @param {boolean} opts.flashAttention
 * @param {object} opts.kvCacheQuant
 * @param {number} opts.prefixCacheHit
 * @param {boolean} opts.cpuOffload
 * @param {object|null} opts.pcieBw
 * @param {object|null} opts.pcieWidth
 * @param {boolean} opts.pureCpu
 * @param {object|null} opts.cpuMemBw
 * @param {number|null} opts.cpuTflops
 * @param {number|null} opts.gpuMemoryUtilization
 * @param {number|null} opts.sysRam
 * @param {object} opts.result        - calcAll() 返回值
 * @param {function} opts.t           - i18n t()
 * @param {string} opts.locale        - 'zh' | 'en'
 */
export function generateMarkdown({
  gpu, gpuCount, interconnect, model, quant, framework,
  ctx, batch, promptLen, outputLen, flashAttention, kvCacheQuant,
  prefixCacheHit, cpuOffload, pcieBw, pcieWidth: selectedPcieWidth,
  pureCpu: selectedPureCpu, cpuMemBw: selectedCpuMemBw,
  cpuTflops, gpuMemoryUtilization, sysRam,
  result, t, locale,
}) {
  const isZh = locale === 'zh'
  const now = new Date().toLocaleString(isZh ? 'zh-CN' : 'en-US', {
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit',
  })

  const lines = []
  const h1 = result.pureCpu
    ? (isZh ? '# CPU 推理速度估算报告' : '# CPU Inference Speed Estimation Report')
    : (isZh ? '# GPU 推理速度估算报告' : '# GPU Inference Speed Estimation Report')
  const site = 'tps.bunai.cc'
  const tpCount = result.tpCount ?? gpuCount
  const effectiveCpuOffload = result.cpuOffload ?? Boolean(cpuOffload)
  const fallbackCpuMemBw = (() => {
    const preset = CPU_MEM_BW_OPTIONS.find(option => option.label === result.cpuMemBwLabel)
    if (preset || !result.cpuMemBwLabel) return preset ?? null
    const match = /^DDR([345])-(\d+)$/i.exec(result.cpuMemBwLabel)
    return match ? createCpuMemBwOption(`ddr${match[1]}`, Number(match[2])) : null
  })()
  const cpuMemBw = selectedCpuMemBw ?? fallbackCpuMemBw
  const pcieWidth = selectedPcieWidth
    ?? PCIE_WIDTH_OPTIONS.find(option => option.label === result.pcieWidthLabel)
    ?? null
  const effectiveGpuMemoryUtilization = normalizeGpuMemoryUtilization(
    gpuMemoryUtilization ?? gpu?.usableRatio,
    gpu?.usableRatio ?? 1,
  )
  const calculationGpu = gpu
    ? { ...gpu, usableRatio: effectiveGpuMemoryUtilization }
    : gpu
  const runtimeCalcOptions = {
    pcieWidth,
    pureCpu: selectedPureCpu ?? result.pureCpu ?? false,
    cpuMemBw,
    cpuTflops: cpuTflops ?? result.cpuTflops ?? null,
    sysRam: sysRam ?? result.sysRam ?? null,
    nglCount: result.nglCount ?? null,
    ppCount: result.ppCount ?? 1,
    epCount: result.epCount ?? 1,
    imageCount: result.imageCount ?? 0,
    draftModelParams: result.draftModelParams ?? null,
  }
  const effectiveTpot = result.effectiveTpot != null && Number.isFinite(Number(result.effectiveTpot))
    ? Number(result.effectiveTpot)
    : (Number(result.singleToks) > 0 ? 1000 / Number(result.singleToks) : Number(result.tpot) || 0)
  const effectiveTotalLatency = result.effectiveTotalLatency != null && Number.isFinite(Number(result.effectiveTotalLatency))
    ? Number(result.effectiveTotalLatency)
    : Number(result.ttft) + Math.max(0, outputLen - 1) * effectiveTpot
  const getMemoryMetrics = calculation => {
    if (calculation.pureCpu) {
      const needed = calculation.cpuRamNeededGB ?? 0
      const available = calculation.availableSysRamGB ?? 0
      return {
        needed,
        available,
        ok: Boolean(calculation.ramOk),
        pct: available > 0 ? needed / available * 100 : needed > 0 ? 100 : 0,
        systemRam: true,
      }
    }
    if (calculation.sharedSystemMemory) {
      const needed = calculation.sharedSystemRamNeededGB ?? 0
      const available = calculation.sharedPoolAvailableGB ?? 0
      return {
        needed,
        available,
        ok: Boolean(calculation.vramOk && calculation.ramOk),
        pct: available > 0 ? needed / available * 100 : needed > 0 ? 100 : 0,
        systemRam: true,
        sharedPool: true,
      }
    }
    return {
      needed: calculation.displayNeeded ?? calculation.perCardNeeded ?? calculation.totalNeeded ?? 0,
      available: calculation.displayVram ?? calculation.perCardVram ?? calculation.totalVram ?? 0,
      ok: Boolean(calculation.vramOk),
      pct: calculation.vramPct ?? 0,
      systemRam: false,
    }
  }
  const memory = getMemoryMetrics(result)
  const cpuMemoryDescription = (() => {
    if (!cpuMemBw && !result.cpuMemBwLabel) return '—'
    const label = cpuMemBw?.label ?? result.cpuMemBwLabel
    const bandwidth = cpuMemBw?.bw ?? result.cpuMemBwGBs
    const channelText = cpuMemBw?.channels != null
      ? ` · ${cpuMemBw.channels} ${isZh ? '通道' : cpuMemBw.channels === 1 ? 'channel' : 'channels'}`
      : ''
    const bandwidthText = Number.isFinite(Number(bandwidth))
      ? ` · ${Number(bandwidth).toFixed(1)} GB/s`
      : ''
    const kindText = cpuMemBw?.bandwidthKind
      ? ` (${cpuMemBw.bandwidthKind === 'measured'
          ? (isZh ? '实测' : 'measured')
          : (isZh ? '理论' : 'theoretical')})`
      : ''
    return `${label}${channelText}${bandwidthText}${kindText}`
  })()

  lines.push(h1)
  lines.push('')
  lines.push(isZh ? `> 生成时间：${now} ｜ ${site}` : `> Generated: ${now} | ${site}`)
  lines.push('')

  // ── 1. 配置摘要 ──────────────────────────────────────
  lines.push(isZh ? '## 配置摘要' : '## Configuration Summary')
  lines.push('')
  const gpuStr = gpuCount > 1
    ? `${gpu.name} × ${gpuCount}${interconnect ? ` (${interconnect.label})` : ''}`
    : gpu.name
  const modeLabel = result.pureCpu
    ? (isZh ? '纯 CPU' : 'CPU only')
    : effectiveCpuOffload
      ? (isZh ? 'GPU + CPU 卸载' : 'GPU + CPU offload')
      : (isZh ? 'GPU 推理' : 'GPU inference')
  const frameworkLabel = framework.labelKey ? t(framework.labelKey) : (framework.label ?? framework.id)
  lines.push(`| ${isZh ? '项目' : 'Item'} | ${isZh ? '值' : 'Value'} |`)
  lines.push('|---|---|')
  lines.push(`| ${isZh ? '计算模式' : 'Compute Mode'} | ${modeLabel} |`)
  if (!result.pureCpu) {
    lines.push(`| GPU | ${gpuStr} |`)
    lines.push(`| ${isZh ? '并行配置' : 'Parallelism'} | TP ${tpCount} · PP ${result.ppCount ?? 1} · EP ${result.epCount ?? 1} |`)
    if (result.sharedSystemMemory) {
      lines.push(`| ${isZh ? 'GPU 共享内存分配' : 'GPU Shared-Pool Allocation'} | ${fmtGB(result.sharedAllocationGB ?? 0)} |`)
      lines.push(`| ${isZh ? '可用系统内存' : 'Usable System RAM'} | ${fmtGB(result.availableSysRamGB ?? 0)} |`)
      lines.push(`| ${isZh ? '有效共享内存池' : 'Effective Shared Pool'} | ${fmtGB(result.sharedPoolAvailableGB ?? 0)} |`)
    } else {
      lines.push(`| ${isZh ? '集群可用显存' : 'Cluster Usable VRAM'} | ${fmtGB(result.totalVram)} |`)
    }
    lines.push(`| ${isZh ? 'GPU 显存预算' : 'GPU Memory Budget'} | ${(effectiveGpuMemoryUtilization * 100).toFixed(1)}% |`)
    lines.push(`| ${isZh ? '建模 GPU 总带宽' : 'Modeled Aggregate GPU BW'} | ${Number(result.effectiveBw).toFixed(1)} GB/s |`)
    lines.push(`| ${isZh ? 'BF16 峰值总算力' : 'Aggregate BF16 Peak'} | ${gpu.bf16 * gpuCount} TFLOPS |`)
  }
  lines.push(`| ${isZh ? '量化精度' : 'Quantization'} | ${quant.label} |`)
  lines.push(`| ${isZh ? '推理框架' : 'Framework'} | ${frameworkLabel} |`)
  lines.push('')

  // ── 2. 模型信息 ──────────────────────────────────────
  lines.push(isZh ? '## 模型信息' : '## Model Information')
  lines.push('')
  lines.push(`| ${isZh ? '项目' : 'Item'} | ${isZh ? '值' : 'Value'} |`)
  lines.push('|---|---|')
  lines.push(`| ${isZh ? '名称' : 'Name'} | ${model.name} |`)
  lines.push(`| ${isZh ? '类型' : 'Type'} | ${model.type === 'moe' ? 'MoE' : 'Dense'} |`)
  lines.push(`| ${isZh ? '总参数' : 'Total Params'} | ${fmtParams(model.params)} |`)
  if (model.type === 'moe' && model.active_params) {
    lines.push(`| ${isZh ? '活跃参数' : 'Active Params'} | ${fmtParams(model.active_params)} |`)
  }
  lines.push(`| ${isZh ? '最大上下文' : 'Max Context'} | ${fmtCtx(model.max_ctx)} |`)
  lines.push(`| Attention | ${result.attentionSummary} |`)
  lines.push(`| ${isZh ? '层数' : 'Layers'} | ${model.layers} |`)
  lines.push(`| Hidden Size | ${model.hidden_size} |`)
  lines.push('')

  // ── 3. 运行参数 ──────────────────────────────────────
  lines.push(isZh ? '## 运行参数' : '## Runtime Parameters')
  lines.push('')
  lines.push(`| ${isZh ? '项目' : 'Item'} | ${isZh ? '值' : 'Value'} |`)
  lines.push('|---|---|')
  lines.push(`| ${isZh ? '上下文长度' : 'Context Length'} | ${fmtCtx(ctx)} tokens |`)
  lines.push(`| ${isZh ? '并发数' : 'Batch Size'} | ${batch} |`)
  lines.push(`| Prompt ${isZh ? '长度' : 'Length'} | ${promptLen.toLocaleString()} tokens |`)
  lines.push(`| ${isZh ? '输出长度' : 'Output Length'} | ${outputLen.toLocaleString()} tokens |`)
  lines.push(`| Flash Attention | ${flashAttention ? (isZh ? '开启' : 'Enabled') : (isZh ? '关闭' : 'Disabled')} |`)
  lines.push(`| KV Cache ${isZh ? '量化' : 'Quant'} | ${result.kvCacheLabel} |`)
  lines.push(`| Prefix Cache ${isZh ? '命中率' : 'Hit Rate'} | ${prefixCacheHit}% |`)
  if (result.pureCpu) {
    lines.push(`| ${isZh ? '系统内存' : 'System RAM'} | ${result.sysRam != null ? fmtGB(result.sysRam) : '—'} |`)
    lines.push(`| ${isZh ? 'CPU 内存' : 'CPU Memory'} | ${cpuMemoryDescription} |`)
  } else if (result.isMoeOffload) {
    lines.push(`| MoE CPU Offload | ${isZh ? '开启' : 'Enabled'} (${result.pcieBwLabel ?? ''}) |`)
    if (result.pcieWidthLabel) {
      lines.push(`| PCIe ${isZh ? '插槽宽度' : 'Link Width'} | ${result.pcieWidthLabel} |`)
    }
    if (result.cpuMemBwLabel) {
      lines.push(`| ${isZh ? 'CPU 内存' : 'CPU Memory'} | ${cpuMemoryDescription} |`)
    }
    if (result.offloadTransferBw) {
      lines.push(`| ${isZh ? '有效卸载链路上限' : 'Effective Offload Link Ceiling'} | ${result.offloadTransferBw.toFixed(1)} GB/s |`)
    }
    if (result.sysRam != null) {
      lines.push(`| ${isZh ? '系统内存' : 'System RAM'} | ${fmtGB(result.sysRam)} |`)
    }
  } else if (result.isLlamaCppHybrid) {
    lines.push(`| llama.cpp CPU Offload | ${isZh ? '开启' : 'Enabled'}${result.nglCount != null ? ` · ${result.nglCount} GPU layers` : ''} |`)
    if (result.cpuMemBwLabel) {
      lines.push(`| ${isZh ? 'CPU 内存' : 'CPU Memory'} | ${cpuMemoryDescription} |`)
    }
    if (result.sysRam != null) {
      lines.push(`| ${isZh ? '系统内存' : 'System RAM'} | ${fmtGB(result.sysRam)} |`)
    }
  } else if (result.sharedSystemMemory) {
    lines.push(`| ${isZh ? '安装系统内存' : 'Installed System RAM'} | ${result.sysRam != null ? fmtGB(result.sysRam) : '—'} |`)
    lines.push(`| ${isZh ? 'GPU 共享内存分配' : 'GPU Shared-Pool Allocation'} | ${fmtGB(result.sharedAllocationGB ?? 0)} |`)
    lines.push(`| ${isZh ? '可用系统内存' : 'Usable System RAM'} | ${fmtGB(result.availableSysRamGB ?? 0)} |`)
    lines.push(`| ${isZh ? 'CPU 内存' : 'CPU Memory'} | ${cpuMemoryDescription} |`)
  }
  if (result.cpuComputeProvided && result.cpuTflops != null) {
    lines.push(`| ${isZh ? 'CPU 峰值算力' : 'Peak CPU Compute'} | ${Number(result.cpuTflops).toFixed(3)} TFLOPS |`)
  }
  if ((result.imageCount ?? 0) > 0) {
    lines.push(`| ${isZh ? '图像数量' : 'Image Count'} | ${result.imageCount} |`)
  }
  if (result.speculativeDecoding) {
    const inferredDraft = result.draftWasInferred ? (isZh ? '（推断）' : ' (inferred)') : ''
    lines.push(`| ${isZh ? '投机解码' : 'Speculative Decoding'} | ${isZh ? '开启' : 'Enabled'} · ${result.draftModelParams.toFixed(1)}B draft${inferredDraft} · ${result.draftLen} tok · ${isZh ? '接受率' : 'acceptance'} ${(result.acceptanceRate * 100).toFixed(0)}% · ${isZh ? '净吞吐系数' : 'net throughput factor'} ×${result.speculativeSpeedup.toFixed(2)} |`)
  }
  lines.push('')

  // ── 4. 内存分析 ──────────────────────────────────────
  lines.push(result.pureCpu || result.sharedSystemMemory
    ? (isZh ? '## 系统内存分析' : '## System RAM Analysis')
    : (isZh ? '## 显存分析' : '## VRAM Analysis'))
  lines.push('')
  const needed = memory.needed
  const avail = memory.available
  const memoryDeficit = Math.max(0, needed - avail)
  const memoryStatus = memory.ok
    ? result.pureCpu || result.sharedSystemMemory
      ? (isZh ? '✅ 系统内存充足' : '✅ System RAM OK')
      : (isZh ? '✅ 显存充足' : '✅ VRAM OK')
    : result.sharedAllocationExceedsRam
      ? (
          isZh
            ? `❌ GPU 共享内存分配超过可用系统内存 ${Number(result.sharedAllocationExcessGB ?? 0).toFixed(1)} GB`
            : `❌ GPU shared-pool allocation exceeds usable system RAM by ${Number(result.sharedAllocationExcessGB ?? 0).toFixed(1)} GB`
        )
    : result.pureCpu || result.sharedSystemMemory
      ? (isZh ? `❌ 系统内存不足 ${memoryDeficit.toFixed(1)} GB` : `❌ System RAM insufficient by ${memoryDeficit.toFixed(1)} GB`)
      : (isZh ? `❌ 显存不足 ${memoryDeficit.toFixed(1)} GB` : `❌ VRAM insufficient by ${memoryDeficit.toFixed(1)} GB`)
  lines.push(`**${isZh ? '状态' : 'Status'}**: ${memoryStatus}`)
  if (result.vramScope === 'per_card' && !result.sharedSystemMemory) {
    lines.push(`> ${isZh ? `Tensor Parallel ×${tpCount}：以下为每卡显存` : `Tensor Parallel ×${tpCount}: per-GPU VRAM shown below`}`)
  }
  lines.push('')
  // 内存评级
  let vramRatingStr
  if (!memory.ok)              vramRatingStr = isZh ? '🔴 不足 — 无法运行'       : '🔴 Insufficient — Cannot run'
  else if (memory.pct > 95)    vramRatingStr = isZh ? '🟡 紧张 — 接近上限'       : '🟡 Tight — Near the limit'
  else                         vramRatingStr = isZh ? '🟢 宽裕 — 内存充足'       : '🟢 Comfortable — Plenty of headroom'
  lines.push(`**${isZh ? '体验评级' : 'Rating'}**: ${vramRatingStr}`)
  lines.push('')
  const vramDenom = avail || 1
  const vramBreakdown = getDisplayVramBreakdown(result)
  lines.push(`| ${isZh ? '项目' : 'Item'} | ${isZh ? (result.pureCpu || result.sharedSystemMemory ? '内存' : '显存') : 'Memory'} | ${isZh ? '占比' : 'Ratio'} |`)
  lines.push('|---|---|---|')
  lines.push(`| ${isZh ? '模型权重' : 'Model Weights'} | ${fmtGB(vramBreakdown.weightGB)} | ${fmtPct(vramBreakdown.weightGB / vramDenom * 100)} |`)
  lines.push(`| KV Cache | ${fmtGB(vramBreakdown.kvGB)} | ${fmtPct(vramBreakdown.kvGB / vramDenom * 100)} |`)
  if (vramBreakdown.activationGB > 0) {
    lines.push(`| ${isZh ? '激活内存' : 'Activation Mem'} | ${fmtGB(vramBreakdown.activationGB)} | ${fmtPct(vramBreakdown.activationGB / vramDenom * 100)} |`)
  }
  lines.push(`| ${isZh ? '系统开销' : 'Overhead'} | ${fmtGB(vramBreakdown.overheadGB)} | ${fmtPct(vramBreakdown.overheadGB / vramDenom * 100)} |`)
  lines.push(`| **${result.pureCpu || result.sharedSystemMemory ? (isZh ? '系统内存需求' : 'System RAM Needed') : result.vramScope === 'per_card' ? (isZh ? '每卡需求' : 'Per-GPU Needed') : (isZh ? '总需求' : 'Total Needed')}** | **${fmtGB(needed)}** | **${fmtPct(memory.pct)}** |`)
  lines.push(`| ${result.pureCpu || result.sharedSystemMemory ? (isZh ? '有效可用系统内存' : 'Effective Usable System RAM') : result.vramScope === 'per_card' ? (isZh ? '每卡可用' : 'Per-GPU Available') : (isZh ? '可用显存' : 'Available')} | ${fmtGB(avail)} | — |`)
  if (result.sharedSystemMemory) {
    lines.push(`| ${isZh ? 'GPU 共享内存分配' : 'GPU Shared-Pool Allocation'} | ${fmtGB(result.sharedAllocationGB ?? 0)} | — |`)
    lines.push(`| ${isZh ? '可用系统内存' : 'Usable System RAM'} | ${fmtGB(result.availableSysRamGB ?? 0)} | ${result.sharedAllocationExceedsRam ? '❌' : '✅'} |`)
  }
  if (result.vramScope === 'per_card' && !result.sharedSystemMemory && result.clusterNeeded != null) {
    lines.push(`| ${isZh ? '集群合计' : 'Cluster Total'} | ${fmtGB(result.clusterNeeded)} / ${fmtGB(result.totalVram)} | — |`)
  }
  if (!result.pureCpu && !result.sharedSystemMemory && result.cpuRamNeededGB > 0) {
    const ramAvailable = result.availableSysRamGB ?? 0
    const ramStatus = result.ramOk ? '✅' : '❌'
    lines.push(`| ${isZh ? '系统内存需求' : 'System RAM Needed'} | ${fmtGB(result.cpuRamNeededGB)} / ${fmtGB(ramAvailable)} | ${ramStatus} |`)
  }
  lines.push('')

  // 量化对比矩阵
  lines.push(isZh ? '### 量化对比矩阵' : '### Quantization Comparison')
  lines.push('')
  lines.push(isZh
    ? '> 理论估算，不代表该量化精度有对应的发布版本。'
    : '> Theoretical estimates. Does not imply a quantized release exists for this model.')
  lines.push('')
  lines.push(`| ${isZh ? '量化' : 'Quant'} | ${result.pureCpu || result.sharedSystemMemory ? (isZh ? '内存需求' : 'RAM') : (isZh ? '显存需求' : 'VRAM')} | ${isZh ? '状态' : 'Status'} | ${isZh ? '预估速度' : 'Est. Speed'} |`)
  lines.push('|---|---|---|---|')
  const _speculativeDecoding = result.speculativeDecoding
  const _acceptanceRate = result.acceptanceRate
  const _draftLen = result.draftLen
  for (const q of QUANT_MAP) {
    try {
      const isCurrent = q.id === quant.id
      const r = isCurrent
        ? result
        : calcAll({
            gpu: calculationGpu, gpuCount, interconnect, model, quant: q, ctx, batch,
            promptLen, outputLen, framework, flashAttention, kvCacheQuant,
            prefixCacheHit, cpuOffload: effectiveCpuOffload, pcieBw,
            ...runtimeCalcOptions,
            speculativeDecoding: _speculativeDecoding, acceptanceRate: _acceptanceRate, draftLen: _draftLen,
          })
      const label = isCurrent ? `**${q.label}**` : q.label
      const rowMemory = getMemoryMetrics(r)
      const vram = isCurrent ? `**${fmtGB(rowMemory.needed)}**` : fmtGB(rowMemory.needed)
      // OOM 时检查 MoE CPU offload 可行性
      let status
      const configurationOk = Boolean(
        r.workloadInputOk
        && r.contextOk
        && r.topologyOk
        && r.modeOk
        && r.kvCacheSupported
        && r.runtimeConfigurationSupported
        && r.computePrecisionSupported
        && r.modelDataOk
      )
      if (!configurationOk) {
        status = isZh ? '⚠️ 当前运行时不支持' : '⚠️ Unsupported by runtime'
      } else if (r.fitOk) {
        status = `✅ ${fmtPct(rowMemory.pct)}`
      } else if (!r.ramOk) {
        status = isZh ? `❌ 系统内存不足` : `❌ System RAM insufficient`
      } else if (!result.pureCpu && !effectiveCpuOffload && model.type === 'moe' && model.active_params) {
        try {
          const fallbackPcie = pcieBw ?? PCIE_BW_OPTIONS.find(x => x.id === 'gen4')
          const ro = calcAll({
            gpu: calculationGpu, gpuCount, interconnect, model, quant: q, ctx, batch,
            promptLen, outputLen, framework, flashAttention, kvCacheQuant,
            prefixCacheHit, cpuOffload: true, pcieBw: fallbackPcie,
            ...runtimeCalcOptions,
            pureCpu: false,
            speculativeDecoding: _speculativeDecoding, acceptanceRate: _acceptanceRate, draftLen: _draftLen,
          })
          status = ro.fitOk
            ? `⚡ ${fmtGB(ro.displayNeeded ?? ro.totalNeeded)} ${isZh ? '(可卸载)' : '(offloadable)'}`
            : (isZh ? `❌ 无法运行` : `❌ Cannot run`)
        } catch { status = isZh ? `❌ 无法运行` : `❌ Cannot run` }
      } else {
        status = isZh ? `❌ 无法运行` : `❌ Cannot run`
      }
      const speed = r.fitOk ? `${r.effectiveToks.toFixed(1)} tok/s` : '—'
      lines.push(`| ${label} | ${vram} | ${status} | ${speed} |`)
    } catch { /* skip */ }
  }
  lines.push('')

  // ── 5. 速度与延迟 ──────────────────────────────────────
  lines.push(isZh ? '## 速度与延迟' : '## Speed & Latency')
  lines.push('')
  // 不可运行提示
  if (!result.fitOk) {
    const fitFailure = !result.ramOk
      ? (isZh ? '系统内存不足' : 'System RAM is insufficient')
      : !result.vramOk
        ? (isZh ? '显存不足' : 'VRAM is insufficient')
        : (isZh ? '当前上下文、并行拓扑或运行模式无效' : 'The context, parallel topology, or runtime mode is invalid')
    lines.push(isZh
      ? `> ⚠️ **${fitFailure}，以下速度仅为理论估算值，当前配置实际无法运行。**`
      : `> ⚠️ **${fitFailure}; the speed values below are theoretical and this configuration cannot run.**`)
    lines.push('')
  }
  // 速度评级（基于单请求速度）
  const toks = result.singleToks
  let speedRatingStr
  if (!result.fitOk)   speedRatingStr = isZh ? '⛔ 当前配置无法运行'             : '⛔ Current configuration cannot run'
  else if (toks >= 60) speedRatingStr = isZh ? '🟢 极快 — 适合实时对话'         : '🟢 Blazing — Real-time chat ready'
  else if (toks >= 30) speedRatingStr = isZh ? '🟡 流畅 — 适合普通使用'         : '🟡 Smooth — Great for everyday use'
  else if (toks >= 15) speedRatingStr = isZh ? '🟠 可用 — 轻度使用'             : '🟠 Usable — Light usage'
  else                 speedRatingStr = isZh ? '🔴 较慢 — 建议换量化或升级硬件' : '🔴 Slow — Consider quantization or better hardware'
  lines.push(`**${isZh ? '体验评级' : 'Rating'}**: ${speedRatingStr}`)
  lines.push('')

  // Decode
  lines.push(`### Decode ${isZh ? '速度' : 'Speed'} (${result.bottleneck === 'bandwidth'
    ? (isZh ? '带宽瓶颈' : 'Bandwidth Bound')
    : (isZh ? '算力瓶颈' : 'Compute Bound')})`)
  lines.push('')
  lines.push(`| ${isZh ? '指标' : 'Metric'} | ${isZh ? '值' : 'Value'} |`)
  lines.push('|---|---|')
  lines.push(`| ${isZh ? 'Decode 带宽上限' : 'Decode BW Ceiling'} | ${fmtToks(result.bwLimit)} |`)
  if (result.decodeComputeLimit != null) {
    lines.push(`| ${isZh ? 'Decode 算力上限' : 'Decode Compute Ceiling'} | ${fmtToks(result.decodeComputeLimit)} |`)
  }
  lines.push(`| ${isZh ? '实际吞吐（总）' : 'Actual Throughput (Total)'} | ${fmtToksRange(result.effectiveToksMin ?? result.decodeToksMin, result.effectiveToksMax ?? result.decodeToksMax)} |`)
  lines.push(`| ${isZh ? '单请求速度' : 'Single Request'} | ${fmtToksRange(result.singleToksMin, result.singleToksMax)} |`)
  lines.push(`| Decode ${isZh ? 'KV/状态流量' : 'KV/State Traffic'} | ${fmtGB(result.kvReadGB)}/step |`)
  if (result.tpEfficiency < 1) {
    lines.push(`| TP ${isZh ? '通信效率' : 'Comm Efficiency'} | ${fmtPct(result.tpEfficiency * 100)} |`)
  }
  if (result.epEfficiency < 1) {
    lines.push(`| EP ${isZh ? '通信效率' : 'Comm Efficiency'} | ${fmtPct(result.epEfficiency * 100)} |`)
  }
  if (result.ppBubbleEff < 1) {
    lines.push(`| PP ${isZh ? '流水线效率' : 'Pipeline Efficiency'} | ${fmtPct(result.ppBubbleEff * 100)} |`)
  }
  lines.push('')

  // Prefill
  lines.push(isZh ? '### Prefill 速度' : '### Prefill Speed')
  lines.push('')
  lines.push(`| ${isZh ? '指标' : 'Metric'} | ${isZh ? '值' : 'Value'} |`)
  lines.push('|---|---|')
  if (result.computeLimit != null) {
    lines.push(`| ${isZh ? 'Prefill 算力上限' : 'Prefill Compute Ceiling'} | ${fmtToks(result.computeLimit)} |`)
  }
  lines.push(`| ${isZh ? '实际吞吐' : 'Actual Throughput'} | ${fmtToksRange(result.prefillToksMin, result.prefillToksMax)} |`)
  lines.push(`| FlashAttention ${isZh ? '系数' : 'Boost'} | ×${result.flashFactorMin.toFixed(1)} ~ ×${result.flashFactorMax.toFixed(1)} |`)
  lines.push(`| ${isZh ? '有效 Prompt' : 'Effective Prompt'} | ${result.effectivePromptLen.toLocaleString()} tokens |`)
  lines.push('')

  // Roofline
  const bottleneckLabel = result.bottleneck === 'bandwidth'
    ? (isZh ? '带宽瓶颈' : 'Bandwidth Bound')
    : (isZh ? '算力瓶颈' : 'Compute Bound')
  const hasPhysicalRoofline = Number(result.decodeComputeLimit) > 0
    && Number(result.arithmeticIntensity) > 0
    && Number(result.ridgePoint) > 0
  const physicalRooflineRatio = hasPhysicalRoofline
    ? Number(result.arithmeticIntensity) / Number(result.ridgePoint)
    : null
  lines.push(hasPhysicalRoofline
    ? `**${isZh ? '瓶颈类型' : 'Bottleneck'}**: ${bottleneckLabel}　　**Roofline ${isZh ? '比' : 'Ratio'} (AI / Ridge)**: ${physicalRooflineRatio.toFixed(2)}`
    : `**${isZh ? '瓶颈类型' : 'Bottleneck'}**: ${bottleneckLabel}　　**Roofline**: ${isZh ? '不可用' : 'Unavailable'}`)
  lines.push('')

  // 延迟
  lines.push(isZh ? '### 延迟' : '### Latency')
  lines.push('')
  lines.push(`| ${isZh ? '指标' : 'Metric'} | ${isZh ? '值' : 'Value'} |`)
  lines.push('|---|---|')
  lines.push(`| TTFT (${isZh ? '首 Token 延迟' : 'Time to First Token'}) | ${fmtMs(result.ttft)} |`)
  lines.push(`| TPOT (${isZh ? '生成延迟' : 'Time per Output Token'}) | ${fmtMs(effectiveTpot)} |`)
  lines.push(`| ${isZh ? '总延迟' : 'Total Latency'} | ${fmtMs(effectiveTotalLatency)} |`)
  if (!result.pureCpu && Number.isFinite(result.totalPower)) {
    lines.push(`| ${isZh ? '总功耗' : 'Total Power'} | ${result.totalPower.toFixed(1)} kW${gpu.unifiedMemory ? (isZh ? '（SoC TDP）' : ' (SoC TDP)') : ''} |`)
  }
  lines.push('')

  // ── 6. 警告与建议 ──────────────────────────────────────
  const warnings = getWarnings(result, t)
  if (warnings.length > 0) {
    lines.push(isZh ? '## 警告与建议' : '## Warnings & Suggestions')
    lines.push('')
    const levelIcon = { error: '❌', warn: '⚠️', info: 'ℹ️' }
    for (const w of warnings) {
      const text = t(`warning.${w.key}`, w)
      lines.push(`- ${levelIcon[w.level] ?? '•'} ${text}`)
    }
    lines.push('')
  }

  // ── 尾注 ──────────────────────────────────────────────
  lines.push('---')
  lines.push('')
  lines.push(isZh
    ? `*本报告由 [${site}](https://${site}) 生成，数据为理论估算值，实际性能受硬件状态、驱动版本、模型实现等因素影响。*`
    : `*This report is generated by [${site}](https://${site}). Values are theoretical estimates; actual performance may vary.*`)
  lines.push('')

  return lines.join('\n')
}

/**
 * 触发浏览器下载 .md 文件
 */
export function downloadMarkdown(content, filename) {
  const blob = new Blob([content], { type: 'text/markdown;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

/**
 * 生成文件名：tps-report-{model}-{gpu}-{quant}-{date}.md
 */
export function buildFilename(model, gpu, quant) {
  const date = new Date().toISOString().slice(0, 10)
  const slug = (s) => String(s).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')
  return `tps-${slug(model.name)}-${slug(gpu.name)}-${slug(quant.id)}-${date}.md`
}
