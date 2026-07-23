import { readFileSync, readdirSync, statSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import en from '../src/i18n/en.js'
import zh from '../src/i18n/zh.js'

function assert(condition, message) {
  if (!condition) throw new Error(message)
}

function flattenKeys(value, prefix = '', keys = new Set()) {
  for (const [key, child] of Object.entries(value)) {
    const fullKey = prefix ? `${prefix}.${key}` : key
    if (child && typeof child === 'object' && !Array.isArray(child)) {
      flattenKeys(child, fullKey, keys)
    } else {
      keys.add(fullKey)
    }
  }
  return keys
}

function collectSourceFiles(directory, files = []) {
  for (const name of readdirSync(directory)) {
    const file = path.join(directory, name)
    if (statSync(file).isDirectory()) collectSourceFiles(file, files)
    else if (/\.(?:js|vue)$/.test(name)) files.push(file)
  }
  return files
}

const englishKeys = flattenKeys(en)
const chineseKeys = flattenKeys(zh)

for (const key of englishKeys) {
  assert(chineseKeys.has(key), `Chinese translation is missing "${key}"`)
}
for (const key of chineseKeys) {
  assert(englishKeys.has(key), `English translation is missing "${key}"`)
}

const usedStaticKeys = new Set()
for (const file of collectSourceFiles(fileURLToPath(new URL('../src', import.meta.url)))) {
  const source = readFileSync(file, 'utf8')
  for (const match of source.matchAll(/(?:\bt|\$t)\(\s*['"]([^'"`]+)['"]/g)) {
    // A trailing dot is a prefix used with string concatenation, not a full
    // translation key (for example, `result.${warning.key}`).
    if (!match[1].endsWith('.')) usedStaticKeys.add(match[1])
  }
}

for (const key of usedStaticKeys) {
  assert(englishKeys.has(key), `Static translation key "${key}" is missing`)
}

const calculationSource = readFileSync(
  fileURLToPath(new URL('../src/utils/calc.js', import.meta.url)),
  'utf8',
)
const warningFunction = calculationSource.slice(
  calculationSource.indexOf('export function getWarnings'),
  calculationSource.indexOf('export function calcBatchSweep'),
)
for (const match of warningFunction.matchAll(/\bkey:\s*'([^']+)'/g)) {
  assert(
    englishKeys.has(`warning.${match[1]}`),
    `Warning translation "warning.${match[1]}" is missing`,
  )
}

console.log(
  `i18n regression passed: ${englishKeys.size} synchronized keys, `
  + `${usedStaticKeys.size} statically referenced keys.`,
)
