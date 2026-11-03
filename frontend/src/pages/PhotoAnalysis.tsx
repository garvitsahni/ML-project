import { useEffect, useRef, useState } from 'react'

import { Details } from '../components/Details'
import { ImageResultsPanel } from '../components/ResultsPanel'
import { UploadDropzone } from '../components/UploadDropzone'
import { detectImage, friendlyError } from '../lib/api'
import type { ImageDetectionResult } from '../types'

const STEPS = [
  'Upload a photo of a crowd.',
  'Press “Check crowd level”.',
  'Read the one-word answer.',
]

export default function PhotoAnalysis() {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [result, setResult] = useState<ImageDetectionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [conf, setConf] = useState(0.05)
  const [imgsz, setImgsz] = useState(640)
  const previewRef = useRef<string | null>(null)

  useEffect(() => {
    return () => {
      if (previewRef.current) URL.revokeObjectURL(previewRef.current)
    }
  }, [])

  const handleFile = (selected: File) => {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current)
    const url = URL.createObjectURL(selected)
    previewRef.current = url
    setPreviewUrl(url)
    setFile(selected)
    setResult(null)
    setError(null)
  }

  const handleDetect = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      setResult(await detectImage(file, { conf, imgsz }))
    } catch (caught) {
      setError(friendlyError(caught))
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-8">
      <header>
        <p className="text-[11px] font-bold uppercase tracking-[0.16em] text-accent">
          Photo analysis
        </p>
        <h1 className="mt-1 text-2xl font-bold tracking-tight text-ink sm:text-3xl">
          How crowded is this photo?
        </h1>
        <p className="mt-2 max-w-xl text-muted">
          Upload a photo. We count the people we can see and give you a simple answer.
        </p>
      </header>

      <div className="grid gap-8 lg:grid-cols-2">
        <section className="space-y-5 self-start rounded-xl border border-line bg-card p-6">
          <UploadDropzone
            file={file}
            onFileSelect={handleFile}
            acceptedTypes="image/jpeg,image/png,image/tiff,image/bmp"
            maxSize={10 * 1024 * 1024}
            hint="JPG, PNG, TIF or BMP · up to 10 MB"
          />

          {previewUrl && (
            <img
              src={previewUrl}
              alt="The photo you chose"
              className="w-full rounded-lg border border-line"
            />
          )}

          <button
            type="button"
            onClick={handleDetect}
            disabled={!file || loading}
            className="w-full rounded-lg bg-ink px-4 py-3 text-base font-semibold text-paper transition-colors hover:bg-black disabled:cursor-not-allowed disabled:bg-soft disabled:text-faint"
          >
            {loading ? 'Checking…' : 'Check crowd level'}
          </button>

          <Details summary="Advanced settings">
            <label className="block text-sm">
              <span className="font-medium text-ink">
                Detector sensitivity ({conf.toFixed(2)})
              </span>
              <input
                type="range"
                min={0.01}
                max={0.5}
                step={0.01}
                value={conf}
                onChange={(event) => setConf(Number(event.target.value))}
                className="mt-2 w-full accent-ink"
              />
              <span className="mt-1 block text-xs leading-relaxed text-faint">
                Lower finds more people, including ones it is unsure about. Leave it alone
                unless you have a reason.
              </span>
            </label>
            <label className="block text-sm">
              <span className="font-medium text-ink">Image size</span>
              <select
                value={imgsz}
                onChange={(event) => setImgsz(Number(event.target.value))}
                className="mt-2 w-full rounded-md border border-line bg-card px-3 py-2 text-ink"
              >
                <option value={320}>Small (fast)</option>
                <option value={640}>Normal</option>
                <option value={1280}>Large (slower, finer detail)</option>
              </select>
            </label>
          </Details>

          <p className="text-xs leading-relaxed text-faint">
            Runs on this computer. The photo is not sent to the internet.
          </p>
        </section>

        <div className="space-y-5">
          {error && (
            <div
              className="rounded-lg border border-[#f0d3c9] bg-[#fdf1ee] p-4 text-sm text-[#8f2f1c]"
              role="alert"
            >
              <p className="font-semibold">That didn’t work</p>
              <p className="mt-1 leading-relaxed">{error}</p>
            </div>
          )}

          {result ? (
            <ImageResultsPanel result={result} />
          ) : (
            !error && (
              <div className="flex h-full min-h-[20rem] flex-col justify-center rounded-xl border border-dashed border-line bg-card p-8">
                <p className="text-lg font-semibold text-ink">Three steps</p>
                <ol className="mt-5 space-y-4">
                  {STEPS.map((step, index) => (
                    <li key={step} className="flex items-start gap-3">
                      <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full border border-line bg-paper text-xs font-semibold text-ink">
                        {index + 1}
                      </span>
                      <span className="pt-0.5 text-sm text-muted">{step}</span>
                    </li>
                  ))}
                </ol>
                <p className="mt-7 border-t border-soft pt-4 text-xs leading-relaxed text-faint">
                  You will get one word — Quiet, Busy or Crowded — and the estimated number
                  of people.
                </p>
              </div>
            )
          )}
        </div>
      </div>
    </div>
  )
}
