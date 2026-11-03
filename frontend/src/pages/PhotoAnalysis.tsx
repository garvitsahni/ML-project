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
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold text-slate-900 sm:text-3xl">
          How crowded is this photo?
        </h1>
        <p className="mt-1 text-slate-600">
          Upload a photo. We count the people we can see and give you a simple answer.
        </p>
      </header>

      <div className="grid gap-6 lg:grid-cols-2">
        <section className="space-y-4 rounded-xl bg-white p-6 shadow-sm ring-1 ring-slate-200">
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
              className="w-full rounded-lg ring-1 ring-slate-200"
            />
          )}

          <button
            type="button"
            onClick={handleDetect}
            disabled={!file || loading}
            className="w-full rounded-lg bg-brand-600 px-4 py-3 text-base font-semibold text-white transition-colors hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-slate-400"
          >
            {loading ? 'Checking…' : 'Check crowd level'}
          </button>

          <Details summary="Advanced settings">
            <label className="block text-sm">
              <span className="font-medium text-slate-700">
                Detector sensitivity ({conf.toFixed(2)})
              </span>
              <input
                type="range"
                min={0.01}
                max={0.5}
                step={0.01}
                value={conf}
                onChange={(event) => setConf(Number(event.target.value))}
                className="mt-2 w-full accent-brand-600"
              />
              <span className="mt-1 block text-xs text-slate-500">
                Lower finds more people, including ones it is unsure about. Leave it alone
                unless you have a reason.
              </span>
            </label>
            <label className="block text-sm">
              <span className="font-medium text-slate-700">Image size</span>
              <select
                value={imgsz}
                onChange={(event) => setImgsz(Number(event.target.value))}
                className="mt-2 w-full rounded-md border border-slate-300 px-3 py-2"
              >
                <option value={320}>Small (fast)</option>
                <option value={640}>Normal</option>
                <option value={1280}>Large (slower, finer detail)</option>
              </select>
            </label>
          </Details>

          <p className="text-xs text-slate-500">
            Runs on this computer. The photo is not sent to the internet.
          </p>
        </section>

        <div className="space-y-4">
          {error && (
            <div
              className="rounded-lg bg-red-50 p-4 text-sm text-red-700 ring-1 ring-inset ring-red-200"
              role="alert"
            >
              <p className="font-medium">That didn’t work</p>
              <p className="mt-1">{error}</p>
            </div>
          )}

          {result ? (
            <ImageResultsPanel result={result} />
          ) : (
            !error && (
              <div className="flex h-full min-h-[20rem] flex-col justify-center rounded-xl border border-dashed border-slate-300 bg-white p-8">
                <p className="text-lg font-semibold text-slate-800">Three steps</p>
                <ol className="mt-4 space-y-3">
                  {STEPS.map((step, index) => (
                    <li key={step} className="flex items-start gap-3">
                      <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-brand-50 text-sm font-semibold text-brand-700 ring-1 ring-inset ring-brand-100">
                        {index + 1}
                      </span>
                      <span className="text-sm text-slate-600">{step}</span>
                    </li>
                  ))}
                </ol>
                <p className="mt-6 text-xs text-slate-500">
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
