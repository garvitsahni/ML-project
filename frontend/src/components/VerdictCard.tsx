import { crowdLevelInfo, crowdRuleText, type CrowdLevel } from '../lib/crowdLevel'

interface VerdictCardProps {
  level: CrowdLevel
  people: number
  caption?: string
  compact?: boolean
}

export function VerdictCard({ level, people, caption, compact = false }: VerdictCardProps) {
  const info = crowdLevelInfo(level)
  const shown = Math.round(people)

  return (
    <div
      className={`rounded-2xl p-6 ring-1 ${info.panel}`}
      role="status"
      aria-live="polite"
    >
      <p className="text-xs font-semibold uppercase tracking-widest text-slate-500">
        Crowd level
      </p>
      <p
        className={`mt-1 font-black uppercase leading-none tracking-tight ${info.word_class} ${
          compact ? 'text-4xl' : 'text-5xl sm:text-6xl'
        }`}
      >
        {info.word}
      </p>
      <p className="mt-3 text-base text-slate-700">
        About <span className="font-semibold tabular-nums text-slate-900">{shown}</span>{' '}
        {shown === 1 ? 'person' : 'people'}
        <span className="text-slate-500"> (estimate)</span>
      </p>
      {caption && <p className="mt-1 text-sm text-slate-500">{caption}</p>}
      <p className="mt-3 text-sm text-slate-600">{info.meaning}</p>
      <p className="mt-4 border-t border-slate-900/10 pt-3 text-xs text-slate-500">
        {crowdRuleText()}
      </p>
    </div>
  )
}
