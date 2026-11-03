import { useEffect, useState } from 'react'

import { getModel } from '../lib/api'
import type { ModelInfo } from '../types'

export function ModelBadge() {
  const [model, setModel] = useState<ModelInfo | null>(null)
  const [error, setError] = useState(false)

  useEffect(() => {
    let active = true
    getModel()
      .then((info) => {
        if (active) setModel(info)
      })
      .catch(() => {
        if (active) setError(true)
      })
    return () => {
      active = false
    }
  }, [])

  if (error) {
    return (
      <span className="inline-flex items-center rounded-full bg-amber-50 px-3 py-1 text-xs font-medium text-amber-700 ring-1 ring-inset ring-amber-200">
        API unreachable
      </span>
    )
  }
  if (!model) {
    return (
      <span className="inline-flex items-center rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-500">
        Checking model…
      </span>
    )
  }
  if (!model.available) {
    return (
      <span
        className="inline-flex items-center rounded-full bg-amber-50 px-3 py-1 text-xs font-medium text-amber-700 ring-1 ring-inset ring-amber-200"
        title={model.note ?? model.caveat}
      >
        Detector only — no anomaly model
      </span>
    )
  }
  return (
    <span
      className="inline-flex items-center rounded-full bg-emerald-50 px-3 py-1 text-xs font-medium text-emerald-700 ring-1 ring-inset ring-emerald-200"
      title={`threshold ${model.threshold ?? '—'} · ${model.feature_names?.length ?? 0} features`}
    >
      Model: {model.algorithm} · trained {model.trained_at}
    </span>
  )
}
