import { AlertCircle, Download } from 'lucide-react'

import type {
  FeatureValue,
  ImageDetectionResult,
  VideoDetectionResult,
  VideoWindow,
} from '../types'

function downloadJson(data: unknown, filename: string) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  anchor.click()
  URL.revokeObjectURL(url)
}

function format(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return value.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })
}

function Card({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-slate-50 p-3 ring-1 ring-inset ring-slate-200">
      <p className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</p>
      <p className="mt-1 text-xl font-semibold text-slate-900">{value}</p>
    </div>
  )
}

function DownloadButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="inline-flex items-center gap-2 rounded-md bg-brand-600 px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-brand-700"
    >
      <Download size={16} aria-hidden />
      Download JSON
    </button>
  )
}

function GridHeatmap({ counts, size }: { counts: number[][]; size: number }) {
  const max = Math.max(1, ...counts.flat())
  return (
    <div>
      <p className="mb-2 text-sm font-medium text-slate-700">
        Density grid ({size}×{size}) — persons per cell
      </p>
      <div
        className="grid gap-1"
        style={{ gridTemplateColumns: `repeat(${size}, minmax(0, 1fr))` }}
      >
        {counts.flatMap((row, rowIndex) =>
          row.map((count, colIndex) => (
            <div
              key={`${rowIndex}-${colIndex}`}
              className="flex h-14 items-center justify-center rounded text-sm font-semibold text-slate-800"
              style={{ backgroundColor: `rgba(37, 99, 235, ${0.08 + 0.5 * (count / max)})` }}
              title={`row ${rowIndex + 1}, column ${colIndex + 1}: ${count}`}
            >
              {count}
            </div>
          )),
        )}
      </div>
    </div>
  )
}

function FeatureTable({ features }: { features: FeatureValue[] }) {
  return (
    <div className="overflow-hidden rounded-lg ring-1 ring-inset ring-slate-200">
      <table className="min-w-full divide-y divide-slate-200 text-sm">
        <thead className="bg-slate-50">
          <tr>
            <th className="px-3 py-2 text-left font-medium text-slate-600">Feature</th>
            <th className="px-3 py-2 text-right font-medium text-slate-600">Value</th>
            <th className="px-3 py-2 text-left font-medium text-slate-600">Unit</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100 bg-white">
          {features.map((feature) => (
            <tr key={feature.name}>
              <td className="px-3 py-2 font-mono text-xs text-slate-700">{feature.name}</td>
              <td className="px-3 py-2 text-right tabular-nums text-slate-900">
                {format(feature.value, 3)}
              </td>
              <td className="px-3 py-2 text-slate-500">{feature.unit}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function ImageResultsPanel({ result }: { result: ImageDetectionResult }) {
  return (
    <section className="space-y-5 rounded-xl bg-white p-6 shadow-sm ring-1 ring-slate-200">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">Detection results</h2>
          <p className="text-sm text-slate-500">
            {result.metadata.model_name} · {result.image.width}×{result.image.height}px ·{' '}
            {format(result.metadata.inference_time_s, 3)} s
          </p>
        </div>
        <DownloadButton onClick={() => downloadJson(result, 'photo-analysis.json')} />
      </div>

      <div className="grid gap-3 sm:grid-cols-3">
        <Card label="Detected persons (estimate)" value={String(result.count)} />
        <Card label="Confidence threshold" value={format(result.metadata.confidence, 2)} />
        <Card label="Inference size" value={`${result.metadata.imgsz}px`} />
      </div>

      <img
        src={result.annotated_image}
        alt={`Annotated upload with ${result.count} detected persons`}
        className="w-full rounded-lg ring-1 ring-slate-200"
      />

      <GridHeatmap counts={result.grid_counts} size={result.grid_size} />
      <FeatureTable features={result.static_features} />

      <div className="flex items-start gap-2 rounded-lg bg-blue-50 p-3 text-sm text-blue-800 ring-1 ring-inset ring-blue-200">
        <AlertCircle size={18} className="mt-0.5 shrink-0" aria-hidden />
        <p>{result.model_note}</p>
      </div>
      <p className="text-xs text-slate-500">{result.metadata.caveat}</p>
    </section>
  )
}

function ScoreBar({ window }: { window: VideoWindow }) {
  const prediction = window.prediction
  if (!prediction) return null
  const width = Math.max(2, Math.round(prediction.score * 100))
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-20 shrink-0 tabular-nums text-slate-500">
        {window.window_start}:{window.window_end}
      </span>
      <div className="h-2 flex-1 overflow-hidden rounded bg-slate-100">
        <div
          className={prediction.label === 1 ? 'h-full bg-red-500' : 'h-full bg-emerald-500'}
          style={{ width: `${width}%` }}
        />
      </div>
      <span className="w-12 shrink-0 text-right tabular-nums text-slate-600">
        {format(prediction.score, 2)}
      </span>
    </div>
  )
}

export function VideoResultsPanel({ result }: { result: VideoDetectionResult }) {
  const { stats, model, windows, video } = result
  const scored = windows.filter((window) => window.prediction !== null)
  const topWindows = [...scored]
    .sort((a, b) => (b.prediction?.score ?? 0) - (a.prediction?.score ?? 0))
    .slice(0, 5)

  return (
    <section className="space-y-5 rounded-xl bg-white p-6 shadow-sm ring-1 ring-slate-200">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">Analysis results</h2>
          <p className="text-sm text-slate-500">
            {video.name} · {video.width}×{video.height}px · {stats.frames_processed} frames
            processed · window {stats.window}, stride {stats.window_stride}
          </p>
        </div>
        <DownloadButton onClick={() => downloadJson(result, 'video-analysis.json')} />
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Card label="Windows" value={String(stats.total_windows)} />
        <Card
          label="Flagged anomalous"
          value={stats.anomaly_windows === null ? '—' : String(stats.anomaly_windows)}
        />
        <Card
          label="Anomaly rate"
          value={
            stats.anomaly_rate === null ? '—' : `${(stats.anomaly_rate * 100).toFixed(1)}%`
          }
        />
        <Card label="Processing time" value={`${format(stats.processing_time_s, 1)} s`} />
      </div>

      {model.available ? (
        <div className="space-y-3">
          <div className="flex flex-wrap items-center gap-2 text-sm text-slate-600">
            <span className="rounded-full bg-emerald-50 px-3 py-1 text-xs font-medium text-emerald-700 ring-1 ring-inset ring-emerald-200">
              {model.algorithm} · threshold {format(model.threshold, 2)}
            </span>
            <span>trained {model.trained_at}</span>
          </div>
          <div>
            <p className="mb-2 text-sm font-medium text-slate-700">
              Anomaly score per window (P(anomaly))
            </p>
            <div className="space-y-1">
              {windows.slice(0, 40).map((window) => (
                <ScoreBar key={window.window_start} window={window} />
              ))}
            </div>
            {windows.length > 40 && (
              <p className="mt-2 text-xs text-slate-500">
                Showing the first 40 of {windows.length} windows; download JSON for all.
              </p>
            )}
          </div>
          {topWindows.length > 0 && (
            <div>
              <p className="mb-2 text-sm font-medium text-slate-700">Highest-scoring windows</p>
              <ul className="space-y-1 text-sm text-slate-600">
                {topWindows.map((window) => (
                  <li key={window.window_start}>
                    <span className="font-mono text-xs">
                      {window.window_start}:{window.window_end}
                    </span>{' '}
                    — {window.prediction?.headline} ({format(window.prediction?.score, 2)}):{' '}
                    {window.prediction?.note}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      ) : (
        <div className="rounded-lg bg-amber-50 p-3 text-sm text-amber-800 ring-1 ring-inset ring-amber-200">
          {model.note ?? 'No compatible model — showing detector/feature output only.'}
        </div>
      )}

      <div className="overflow-hidden rounded-lg ring-1 ring-inset ring-slate-200">
        <table className="min-w-full divide-y divide-slate-200 text-sm">
          <thead className="bg-slate-50">
            <tr>
              <th className="px-3 py-2 text-left font-medium text-slate-600">Window</th>
              <th className="px-3 py-2 text-right font-medium text-slate-600">Persons</th>
              <th className="px-3 py-2 text-right font-medium text-slate-600">Speed px/f</th>
              <th className="px-3 py-2 text-right font-medium text-slate-600">Dir. consistency</th>
              <th className="px-3 py-2 text-right font-medium text-slate-600">Anomaly score</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 bg-white">
            {windows.slice(0, 60).map((window) => (
              <tr key={window.window_start}>
                <td className="px-3 py-2 font-mono text-xs text-slate-700">
                  {window.window_start}:{window.window_end}
                </td>
                <td className="px-3 py-2 text-right tabular-nums">
                  {format(window.features.person_count, 1)}
                </td>
                <td className="px-3 py-2 text-right tabular-nums">
                  {format(window.features.mean_speed_pxf, 2)}
                </td>
                <td className="px-3 py-2 text-right tabular-nums">
                  {format(window.features.directional_consistency, 2)}
                </td>
                <td className="px-3 py-2 text-right tabular-nums">
                  {format(window.prediction?.score ?? null, 2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {windows.length > 60 && (
        <p className="text-xs text-slate-500">
          Table shows the first 60 of {windows.length} windows; download JSON for all.
        </p>
      )}
      <p className="text-xs text-slate-500">{model.caveat}</p>
    </section>
  )
}
