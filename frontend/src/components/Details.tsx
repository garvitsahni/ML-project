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
    <details className="group rounded-xl bg-white ring-1 ring-slate-200">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-4 py-3 text-sm font-medium text-slate-700 hover:bg-slate-50">
        <span>{summary}</span>
        <ChevronDown
          size={16}
          aria-hidden
          className="shrink-0 text-slate-400 transition-transform group-open:rotate-180"
        />
      </summary>
      <div className="space-y-4 border-t border-slate-200 p-4">{children}</div>
    </details>
  )
}
