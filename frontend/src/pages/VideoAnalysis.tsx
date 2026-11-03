import { useEffect, useRef, useState } from 'react'

import { Details } from '../components/Details'
import { VideoResultsPanel } from '../components/ResultsPanel'
import { UploadDropzone } from '../components/UploadDropzone'
import { detectVideo, friendlyError } from '../lib/api'
import type { VideoDetectionResult } from '../types'

const STEPS = [
  'Upload a short clip.',
  'Press “Check crowd level”.',
  'Read the level, then see how it changes over the clip.',
]

export default function VideoAnalysis() {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [result, setResult] = useState<VideoDetectionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [conf, setConf] = useState(0.05)
  const [imgsz, setImgsz] = useState(640)
  const [maxFrames, setMaxFrames] = useState(150)
  const [stride, setStride] = useState(1)
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

  const handleAnalyze = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      setResult(await detectVideo(file, { conf, imgsz, maxFrames, stride }))
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
          How crowded is this video?
        </h1>
        <p className="mt-1 text-slate-600">
          Upload a short clip. We count people over time and show how the crowd changes.
        </p>
      </header>

      <div className="grid gap-6 lg:grid-cols-2">
        <section className="space-y-4 rounded-xl bg-white p-6 shadow-sm ring-1 ring-slate-200">
          <UploadDropzone
            file={file}
            onFileSelect={handleFile}
            acceptedTypes="video/mp4,video/x-msvideo,video/quicktime,video/x-matroska,video/webm"
            maxSize={200 * 1024 * 1024}
            hint="MP4, AVI, MOV, MKV or WEBM · up to 200 MB"
          />

          {previewUrl && (
            <video
              src={previewUrl}
              controls
              className="w-full rounded-lg bg-black ring-1 ring-slate-200"
            />
          )}

          <button
            type="button"
            onClick={handleAnalyze}
            disabled={!file || loading}
            className="w-full rounded-lg bg-brand-600 px-4 py-3 text-base font-semibold text-white transition-colors hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-slate-400"
          >
            {loading ? 'Checking the clip…' : 'Check crowd level'}
          </button>
          {loading && (
            <p className="text-center text-xs text-slate-500">
              This can take a few seconds. Keep this tab open.
            </p>
          )}

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
                Lower finds more people, including ones it is unsure about.
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
            <label className="block text-sm">
              <span className="font-medium text-slate-700">
                How much of the clip to check ({maxFrames} frames)
              </span>
              <input
                type="range"
                min={30}
                max={600}
                step={10}
                value={maxFrames}
                onChange={(event) => setMaxFrames(Number(event.target.value))}
                className="mt-2 w-full accent-brand-600"
              />
              <span className="mt-1 block text-xs text-slate-500">
                Checking more frames takes longer but covers more of the clip.
              </span>
            </label>
            <label className="block text-sm">
              <span className="font-medium text-slate-700">
                Skip frames between checks ({stride})
              </span>
              <input
                type="range"
                min={1}
                max={5}
                step={1}
                value={stride}
                onChange={(event) => setStride(Number(event.target.value))}
                className="mt-2 w-full accent-brand-600"
              />
              <span className="mt-1 block text-xs text-slate-500">
                Higher is faster but coarser. Leave at 1 for the best detail.
              </span>
            </label>
          </Details>

          <p className="text-xs text-slate-500">
            Runs on this computer. The clip is not sent to the internet. A clip needs at
            least 30 frames to produce a result.
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
            <VideoResultsPanel result={result} />
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
                  You will get one word for the whole clip and a coloured strip showing each
                  moment.
                </p>
              </div>
            )
          )}
        </div>
      </div>
    </div>
  )
}
