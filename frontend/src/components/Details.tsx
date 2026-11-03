import type { ReactNode } from 'react'
import { ChevronDown } from 'lucide-react'

export function Details({
  summary = 'See the numbers',
  children,
}: {
  summary?: string
  children: ReactNode
}) {
  return (
    <details className="group rounded-lg border border-line bg-card px-4">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 py-3 text-sm font-medium text-ink transition-colors hover:text-accent">
        <span>{summary}</span>
        <ChevronDown
          size={15}
          aria-hidden
          className="shrink-0 text-faint transition-transform group-open:rotate-180"
        />
      </summary>
      <div className="space-y-5 border-t border-soft pb-4 pt-4">{children}</div>
    </details>
  )
}
