import { useEffect, useRef, useState } from 'react'

import { ModelBadge } from '../components/ModelBadge'
import { VideoResultsPanel } from '../components/ResultsPanel'
import { UploadDropzone } from '../components/UploadDropzone'
import { detectVideo, errorMessage } from '../lib/api'
import type { VideoDetectionResult } from '../types'

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
      setError(errorMessage(caught))
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Video Analysis</h1>
          <p className="text-sm text-slate-600">
            Detect → track → optical flow → 30-frame sliding windows. Anomaly scores appear
            only when a compatible trained bundle exists.
          </p>
        </div>
        <ModelBadge />
      </header>

      <div className="grid gap-6 lg:grid-cols-2">
        <section className="space-y-4 rounded-xl bg-white p-6 shadow-sm ring-1 ring-slate-200">
          <UploadDropzone
            file={file}
            onFileSelect={handleFile}
            acceptedTypes="video/mp4,video/x-msvideo,video/quicktime,video/x-matroska,video/webm"
            maxSize={200 * 1024 * 1024}
            hint="MP4, AVI, MOV, MKV, or WEBM"
          />

          {previewUrl && (
            <video
              src={previewUrl}
              controls
              className="w-full rounded-lg bg-black ring-1 ring-slate-200"
            />
          )}

          <div className="grid gap-4 sm:grid-cols-2">
            <label className="block text-sm">
              <span className="font-medium text-slate-700">Confidence ({conf.toFixed(2)})</span>
              <input
                type="range"
                min={0.01}
                max={0.5}
                step={0.01}
                value={conf}
                onChange={(event) => setConf(Number(event.target.value))}
                className="mt-2 w-full accent-brand-600"
              />
            </label>
            <label className="block text-sm">
              <span className="font-medium text-slate-700">Inference size</span>
              <select
                value={imgsz}
                onChange={(event) => setImgsz(Number(event.target.value))}
                className="mt-2 w-full rounded-md border border-slate-300 px-3 py-2"
              >
                <option value={320}>320px</option>
                <option value={640}>640px</option>
                <option value={1280}>1280px</option>
              </select>
            </label>
            <label className="block text-sm">
              <span className="font-medium text-slate-700">
                Max frames ({maxFrames})
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
            </label>
            <label className="block text-sm">
              <span className="font-medium text-slate-700">Frame stride ({stride})</span>
              <input
                type="range"
                min={1}
                max={5}
                step={1}
                value={stride}
                onChange={(event) => setStride(Number(event.target.value))}
                className="mt-2 w-full accent-brand-600"
              />
            </label>
          </div>

          <button
            type="button"
            onClick={handleAnalyze}
            disabled={!file || loading}
            className="w-full rounded-md bg-brand-600 px-4 py-2 font-medium text-white transition-colors hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-slate-400"
          >
            {loading ? 'Processing frames…' : 'Analyze video'}
          </button>
          <p className="text-xs text-slate-500">
            CPU cost is roughly 20–50 ms per processed frame. A clip needs at least 30
            processed frames to produce one window. Speeds are px/frame — never m/s.
          </p>
        </section>

        <div className="space-y-4">
          {error && (
            <div className="rounded-lg bg-red-50 p-4 text-sm text-red-700 ring-1 ring-inset ring-red-200" role="alert">
              <p className="font-medium">Analysis failed</p>
              <p className="mt-1">{error}</p>
            </div>
          )}
          {result ? (
            <VideoResultsPanel result={result} />
          ) : (
            !error && (
              <div className="flex h-full min-h-[18rem] flex-col items-center justify-center rounded-xl border border-dashed border-slate-300 bg-white p-8 text-center">
                <span className="text-5xl" aria-hidden>
                  🎬
                </span>
                <p className="mt-3 font-medium text-slate-700">No results yet</p>
                <p className="text-sm text-slate-500">
                  Upload a crowd video and run analysis to see window features and, when a
                  model is available, per-window anomaly scores.
                </p>
              </div>
            )
          )}
        </div>
      </div>
    </div>
  )
}
