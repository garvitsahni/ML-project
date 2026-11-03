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
      className="rounded-xl border border-line bg-card p-6"
      style={{ borderLeft: `3px solid ${info.hex}` }}
      role="status"
      aria-live="polite"
    >
      <p className="text-[11px] font-bold uppercase tracking-[0.16em] text-faint">
        Crowd level
      </p>
      <p
        className={`mt-1 font-black uppercase leading-[0.95] tracking-[-0.035em] ${
          info.word_class
        } ${compact ? 'text-4xl' : 'text-5xl sm:text-6xl'}`}
      >
        {info.word}
      </p>
      {caption && <p className="mt-1.5 text-sm text-muted">{caption}</p>}
      <p className="mt-3 text-base text-ink">
        About <span className="font-bold tabular-nums">{shown}</span>{' '}
        {shown === 1 ? 'person' : 'people'}{' '}
        <span className="text-faint">(estimate)</span>
      </p>
      <p className="mt-3 text-sm text-muted">{info.meaning}</p>

      <div className="mt-4 flex gap-1" aria-hidden>
        {[0, 1, 2].map((step) => (
          <span
            key={step}
            className={`h-1.5 flex-1 rounded-full ${
              step === info.index ? info.bar : 'bg-soft'
            }`}
          />
        ))}
      </div>

      <p className="mt-3 border-t border-soft pt-3 text-xs text-faint">
        {crowdRuleText()}
      </p>
    </div>
  )
}
