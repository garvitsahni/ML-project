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
    panel: 'bg-emerald-50 ring-emerald-200',
    word_class: 'text-emerald-700',
    chip: 'bg-emerald-100 text-emerald-800 ring-emerald-200',
    bar: 'bg-emerald-500',
  },
  busy: {
    level: 'busy',
    word: 'Busy',
    meaning: 'A moderate crowd.',
    panel: 'bg-amber-50 ring-amber-200',
    word_class: 'text-amber-700',
    chip: 'bg-amber-100 text-amber-900 ring-amber-200',
    bar: 'bg-amber-500',
  },
  crowded: {
    level: 'crowded',
    word: 'Crowded',
    meaning: 'A lot of people in view.',
    panel: 'bg-rose-50 ring-rose-200',
    word_class: 'text-rose-700',
    chip: 'bg-rose-100 text-rose-800 ring-rose-200',
    bar: 'bg-rose-500',
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
