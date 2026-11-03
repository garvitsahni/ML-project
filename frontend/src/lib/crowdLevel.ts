export type CrowdLevel = 'quiet' | 'busy' | 'crowded'

/**
 * Simple, stated cut-offs on the estimated number of people in view.
 *
 * They come from the spread of the project's own window table (3,222 windows):
 * the 33rd percentile is ~9 people and the 67th is ~14. This is a plain rule,
 * not a trained classifier — there is no counting ground truth in the data.
 */
export const CROWD_THRESHOLDS = { busy: 9, crowded: 15 } as const

export interface CrowdLevelInfo {
  level: CrowdLevel
  word: string
  meaning: string
  /** Position on the three-step meter: 0 quiet, 1 busy, 2 crowded. */
  index: number
  /** Solid colour for rules, meter fills and chart bars. */
  hex: string
  panel: string
  word_class: string
  chip: string
  bar: string
}

const LEVELS: Record<CrowdLevel, CrowdLevelInfo> = {
  quiet: {
    level: 'quiet',
    word: 'Quiet',
    meaning: 'Few people in view.',
    index: 0,
    hex: '#3f6b52',
    panel: 'bg-card',
    word_class: 'text-quiet',
    chip: 'bg-[#eef3ee] text-quiet ring-[#d7e2da]',
    bar: 'bg-quiet',
  },
  busy: {
    level: 'busy',
    word: 'Busy',
    meaning: 'A moderate crowd.',
    index: 1,
    hex: '#a1731f',
    panel: 'bg-card',
    word_class: 'text-busy',
    chip: 'bg-[#f8f1e2] text-busy ring-[#eadfc6]',
    bar: 'bg-busy',
  },
  crowded: {
    level: 'crowded',
    word: 'Crowded',
    meaning: 'A lot of people in view.',
    index: 2,
    hex: '#b4442e',
    panel: 'bg-card',
    word_class: 'text-crowded',
    chip: 'bg-[#fdf1ee] text-crowded ring-[#f0d3c9]',
    bar: 'bg-crowded',
  },
}

export function crowdLevel(people: number | null | undefined): CrowdLevel | null {
  if (people === null || people === undefined || Number.isNaN(people)) return null
  if (people >= CROWD_THRESHOLDS.crowded) return 'crowded'
  if (people >= CROWD_THRESHOLDS.busy) return 'busy'
  return 'quiet'
}

export function crowdLevelInfo(level: CrowdLevel): CrowdLevelInfo {
  return LEVELS[level]
}

export function crowdRuleText(): string {
  return `Quiet under ${CROWD_THRESHOLDS.busy} people · Busy ${CROWD_THRESHOLDS.busy}–${
    CROWD_THRESHOLDS.crowded - 1
  } · Crowded ${CROWD_THRESHOLDS.crowded} or more`
}
