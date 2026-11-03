import { Download } from 'lucide-react'

import { crowdLevel, crowdLevelInfo } from '../lib/crowdLevel'
import { featureLabel } from '../lib/labels'
import type {
  FeatureValue,
  ImageDetectionResult,
  VideoDetectionResult,
  VideoWindow,
} from '../types'
import { Details } from './Details'
import { VerdictCard } from './VerdictCard'

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

function humanUnit(unit: string): string {
  const map: Record<string, string> = {
    'count (detector estimate)': 'estimated count',
    'persons / megapixel': 'people per megapixel',
    'fraction 0-1': 'share (0–1)',
    'normalized 0-~1': 'relative (0–1)',
    'normalized 0-1': 'relative (0–1)',
    'px/frame': 'pixels per frame',
  }
  return map[unit] ?? unit
}

function DownloadButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="inline-flex items-center gap-2 rounded-md bg-slate-800 px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-slate-900"
    >
      <Download size={16} aria-hidden />
      Download data
    </button>
  )
}

function GridHeatmap({ counts, size }: { counts: number[][]; size: number }) {
  const max = Math.max(1, ...counts.flat())
  return (
    <div>
      <p className="mb-1 text-sm font-medium text-slate-700">
        Where people are standing ({size}×{size} grid)
      </p>
      <p className="mb-3 text-xs text-slate-500">
        Each square counts the people the detector found in that part of the picture.
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
              title={`row ${rowIndex + 1}, column ${colIndex + 1}: ${count} people`}
            >
              {count}
            </div>
          )),
        )}
      </div>
    </div>
  )
}

function FeatureList({ features }: { features: FeatureValue[] }) {
  return (
    <div>
      <p className="mb-1 text-sm font-medium text-slate-700">What we measured</p>
      <p className="mb-3 text-xs text-slate-500">
        These numbers describe the picture. None of them is a head count from the dataset.
      </p>
      <dl className="grid gap-2 sm:grid-cols-2">
        {features.map((feature) => {
          const { label, help } = featureLabel(feature.name)
          return (
            <div
              key={feature.name}
              className="rounded-lg bg-slate-50 p-3 ring-1 ring-inset ring-slate-200"
            >
              <dt className="text-sm font-medium text-slate-800">{label}</dt>
              <dd className="mt-1">
                <span className="text-lg font-semibold tabular-nums text-slate-900">
                  {format(feature.value, 2)}
                </span>
                <span className="ml-2 text-xs text-slate-500">
                  {humanUnit(feature.unit)}
                </span>
              </dd>
              {help && <p className="mt-1 text-xs text-slate-500">{help}</p>}
            </div>
          )
        })}
      </dl>
    </div>
  )
}

export function ImageResultsPanel({ result }: { result: ImageDetectionResult }) {
  const level = crowdLevel(result.count) ?? 'quiet'
  return (
    <section className="space-y-5">
      <VerdictCard level={level} people={result.count} caption="in this photo" />

      <figure className="overflow-hidden rounded-xl bg-white ring-1 ring-slate-200">
        <img
          src={result.annotated_image}
          alt={`Photo with ${result.count} detected people outlined`}
          className="w-full"
        />
        <figcaption className="px-4 py-2 text-xs text-slate-500">
          The people the detector found are outlined in green.
        </figcaption>
      </figure>

      <Details>
        <GridHeatmap counts={result.grid_counts} size={result.grid_size} />
        <FeatureList features={result.static_features} />
        <div className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-200 pt-4">
          <p className="max-w-md text-xs text-slate-500">{result.metadata.caveat}</p>
          <DownloadButton onClick={() => downloadJson(result, 'photo-analysis.json')} />
        </div>
      </Details>
    </section>
  )
}

function averagePeople(windows: VideoWindow[]): number | null {
  const counts = windows
    .map((window) => window.features.person_count)
    .filter((value): value is number => typeof value === 'number' && !Number.isNaN(value))
  if (counts.length === 0) return null
  return counts.reduce((sum, value) => sum + value, 0) / counts.length
}

function WindowStrip({ windows }: { windows: VideoWindow[] }) {
  return (
    <div>
      <p className="mb-1 text-sm font-medium text-slate-700">Crowd level over the clip</p>
      <p className="mb-3 text-xs text-slate-500">
        Each block is one 30-frame window, from the start of the clip to the end. Hover a
        block to see its numbers.
      </p>
      <div className="flex gap-0.5">
        {windows.map((window) => {
          const level = crowdLevel(window.features.person_count)
          const info = level ? crowdLevelInfo(level) : null
          const people = window.features.person_count
          return (
            <div
              key={window.window_start}
              className={`h-3 flex-1 rounded-sm ${info?.bar ?? 'bg-slate-300'}`}
              title={`Frames ${window.window_start}–${window.window_end}: ${
                info?.word ?? 'no data'
              }${people !== null && people !== undefined ? ` · about ${Math.round(people)} people` : ''}`}
            />
          )
        })}
      </div>
      <div className="mt-3 flex flex-wrap gap-3 text-xs text-slate-600">
        {(['quiet', 'busy', 'crowded'] as const).map((level) => {
          const info = crowdLevelInfo(level)
          return (
            <span key={level} className="inline-flex items-center gap-1.5">
              <span className={`h-2.5 w-2.5 rounded-sm ${info.bar}`} aria-hidden />
              {info.word}
            </span>
          )
        })}
      </div>
    </div>
  )
}

export function VideoResultsPanel({ result }: { result: VideoDetectionResult }) {
  const { stats, windows, video } = result
  const average = averagePeople(windows)
  const level = crowdLevel(average)

  return (
    <section className="space-y-5">
      {level && average !== null ? (
        <VerdictCard
          level={level}
          people={average}
          caption={`average across ${stats.total_windows} ${
            stats.total_windows === 1 ? 'window' : 'windows'
          }`}
        />
      ) : (
        <div className="rounded-2xl bg-slate-50 p-6 ring-1 ring-slate-200">
          <p className="text-lg font-semibold text-slate-900">No people found</p>
          <p className="mt-1 text-sm text-slate-600">
            The detector did not find anyone in this clip. Try a clearer or closer video.
          </p>
        </div>
      )}

      <div className="rounded-xl bg-white p-4 ring-1 ring-slate-200">
        <p className="text-sm text-slate-600">
          {video.name} · {video.width}×{video.height}px · {stats.frames_processed} frames
          checked
        </p>
      </div>

      <WindowStrip windows={windows} />

      <Details summary={`See the numbers (${stats.total_windows} windows)`}>
        <div className="overflow-hidden rounded-lg ring-1 ring-inset ring-slate-200">
          <table className="min-w-full divide-y divide-slate-200 text-sm">
            <thead className="bg-slate-50">
              <tr>
                <th className="px-3 py-2 text-left font-medium text-slate-600">Frames</th>
                <th className="px-3 py-2 text-right font-medium text-slate-600">People</th>
                <th className="px-3 py-2 text-left font-medium text-slate-600">Level</th>
                <th className="px-3 py-2 text-right font-medium text-slate-600">Speed</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 bg-white">
              {windows.slice(0, 60).map((window) => {
                const windowLevel = crowdLevel(window.features.person_count)
                const info = windowLevel ? crowdLevelInfo(windowLevel) : null
                return (
                  <tr key={window.window_start}>
                    <td className="px-3 py-2 font-mono text-xs text-slate-700">
                      {window.window_start}–{window.window_end}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums text-slate-900">
                      {window.features.person_count === null ||
                      window.features.person_count === undefined
                        ? '—'
                        : Math.round(window.features.person_count)}
                    </td>
                    <td className="px-3 py-2">
                      {info ? (
                        <span
                          className={`inline-flex rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${info.chip}`}
                        >
                          {info.word}
                        </span>
                      ) : (
                        '—'
                      )}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums text-slate-700">
                      {format(window.features.mean_speed_pxf, 2)} px/frame
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        {windows.length > 60 && (
          <p className="text-xs text-slate-500">
            Showing the first 60 of {windows.length} windows. Download the data for all of
            them.
          </p>
        )}
        <div className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-200 pt-4">
          <p className="max-w-md text-xs text-slate-500">
            Counts are detector estimates. This app describes the crowd — it does not score
            anomalies.
          </p>
          <DownloadButton onClick={() => downloadJson(result, 'video-analysis.json')} />
        </div>
      </Details>
    </section>
  )
}
