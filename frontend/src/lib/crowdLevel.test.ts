import { describe, expect, it } from 'vitest'

import { CROWD_THRESHOLDS, crowdLevel, crowdLevelInfo, crowdRuleText } from './crowdLevel'

describe('crowdLevel', () => {
  it('calls a nearly empty scene quiet', () => {
    expect(crowdLevel(0)).toBe('quiet')
    expect(crowdLevel(8.9)).toBe('quiet')
  })

  it('starts busy at the lower cut-off', () => {
    expect(crowdLevel(CROWD_THRESHOLDS.busy)).toBe('busy')
    expect(crowdLevel(14.9)).toBe('busy')
  })

  it('starts crowded at the upper cut-off', () => {
    expect(crowdLevel(CROWD_THRESHOLDS.crowded)).toBe('crowded')
    expect(crowdLevel(40)).toBe('crowded')
  })

  it('returns null when there is no count', () => {
    expect(crowdLevel(null)).toBeNull()
    expect(crowdLevel(undefined)).toBeNull()
    expect(crowdLevel(Number.NaN)).toBeNull()
  })

  it('gives every level a word and a meaning', () => {
    for (const level of ['quiet', 'busy', 'crowded'] as const) {
      const info = crowdLevelInfo(level)
      expect(info.word.length).toBeGreaterThan(0)
      expect(info.meaning.length).toBeGreaterThan(0)
      expect(info.panel.length).toBeGreaterThan(0)
    }
  })

  it('states the cut-offs in plain words', () => {
    expect(crowdRuleText()).toBe('Quiet under 9 people · Busy 9–14 · Crowded 15 or more')
  })
})
