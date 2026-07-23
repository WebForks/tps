function nonNegative(value) {
  const numeric = Number(value)
  return Number.isFinite(numeric) ? Math.max(0, numeric) : 0
}

/**
 * Return the four non-overlapping components for the same scope as the VRAM
 * requirement shown to the user.
 *
 * Legacy results only expose cluster-level weights/KV and an `overheadGB`
 * value that already includes activations. In that case, preserve the known
 * weight/KV ratio and scale it to the displayed requirement. Newer results
 * expose exact per-card components.
 */
export function getDisplayVramBreakdown(result) {
  if (!result) {
    return { weightGB: 0, kvGB: 0, overheadGB: 0, activationGB: 0 }
  }

  if (result.pureCpu) {
    const weightGB = nonNegative(result.cpuWeightGB)
    const kvGB = nonNegative(result.cpuKvGB)
    const runtimeGB = Math.max(0, nonNegative(result.cpuRamNeededGB) - weightGB - kvGB)
    const activationGB = Math.min(nonNegative(result.activationGB), runtimeGB)
    return {
      weightGB,
      kvGB,
      overheadGB: Math.max(0, runtimeGB - activationGB),
      activationGB,
    }
  }

  const isPerCard = result.vramScope === 'per_card'
  const scopedValues = [
    result.perCardWeightGB,
    result.perCardKvGB,
    result.perCardOverheadGB,
    result.perCardActivationGB,
  ]

  if (isPerCard && scopedValues.every(value => Number.isFinite(Number(value)))) {
    return {
      weightGB: nonNegative(result.perCardWeightGB),
      kvGB: nonNegative(result.perCardKvGB),
      overheadGB: nonNegative(result.perCardOverheadGB),
      activationGB: nonNegative(result.perCardActivationGB),
    }
  }

  const activationGB = nonNegative(result.activationGB)
  const overheadGB = Math.max(0, nonNegative(result.overheadGB) - activationGB)
  const displayedNeeded = nonNegative(result.displayNeeded ?? result.totalNeeded)
  const componentBudget = Math.max(0, displayedNeeded - overheadGB - activationGB)
  const rawWeight = nonNegative(result.weightGB)
  const rawKv = nonNegative(result.kvGB)
  const rawWeightKv = rawWeight + rawKv

  return {
    weightGB: rawWeightKv > 0 ? componentBudget * rawWeight / rawWeightKv : componentBudget,
    kvGB: rawWeightKv > 0 ? componentBudget * rawKv / rawWeightKv : 0,
    overheadGB,
    activationGB,
  }
}
