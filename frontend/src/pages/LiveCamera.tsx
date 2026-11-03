import { useCallback, useEffect, useRef, useState } from 'react'
import { Camera, CameraOff, ScanLine } from 'lucide-react'

import { ModelBadge } from '../components/ModelBadge'
import { detectFrame, errorMessage } from '../lib/api'
import type { FrameDetectionResult } from '../types'

export default function LiveCamera() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const overlayRef = useRef<HTMLCanvasElement>(null)
  const captureRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const busyRef = useRef(false)

  const [streaming, setStreaming] = useState(false)
  const [analyzing, setAnalyzing] = useState(false)
  const [auto, setAuto] = useState(true)
  const [result, setResult] = useState<FrameDetectionResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [conf, setConf] = useState(0.05)
  const [imgsz, setImgsz] = useState(640)

  const drawOverlay = useCallback((data: FrameDetectionResult | null) => {
    const video = videoRef.current
    const canvas = overlayRef.current
    if (!video || !canvas) return
    const rect = video.getBoundingClientRect()
    canvas.width = Math.max(1, Math.round(rect.width))
    canvas.height = Math.max(1, Math.round(rect.height))
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    if (!data) return
    const scaleX = canvas.width / data.image.width
    const scaleY = canvas.height / data.image.height
    ctx.strokeStyle = '#22c55e'
    ctx.lineWidth = 2
    for (const box of data.boxes) {
      ctx.strokeRect(
        box.x1 * scaleX,
        box.y1 * scaleY,
        (box.x2 - box.x1) * scaleX,
        (box.y2 - box.y1) * scaleY,
      )
    }
  }, [])

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setStreaming(false)
    setResult(null)
    drawOverlay(null)
  }, [drawOverlay])

  useEffect(() => {
    const onResize = () => drawOverlay(result)
    window.addEventListener('resize', onResize)
    return () => {
      window.removeEventListener('resize', onResize)
      streamRef.current?.getTracks().forEach((track) => track.stop())
    }
  }, [drawOverlay, result])

  const startCamera = async () => {
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
      setStreaming(true)
    } catch (caught) {
      setError(`Camera unavailable: ${errorMessage(caught)}`)
      stopCamera()
    }
  }

  const captureAndAnalyze = useCallback(async () => {
    const video = videoRef.current
    const canvas = captureRef.current
    if (!video || !canvas || video.videoWidth === 0 || busyRef.current) return
    busyRef.current = true
    setAnalyzing(true)
    try {
      const scale = Math.min(1, 960 / Math.max(video.videoWidth, video.videoHeight))
      canvas.width = Math.max(1, Math.round(video.videoWidth * scale))
      canvas.height = Math.max(1, Math.round(video.videoHeight * scale))
      const ctx = canvas.getContext('2d')
      if (!ctx) throw new Error('Could not read the camera frame')
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
      const blob = await new Promise<Blob | null>((resolve) =>
        canvas.toBlob(resolve, 'image/jpeg', 0.9),
      )
      if (!blob) throw new Error('Could not encode the camera frame')
      const data = await detectFrame(blob, { conf, imgsz })
      setResult(data)
      setError(null)
      drawOverlay(data)
    } catch (caught) {
      setError(errorMessage(caught))
    } finally {
      busyRef.current = false
      setAnalyzing(false)
    }
  }, [conf, imgsz, drawOverlay])

  useEffect(() => {
    if (!streaming || !auto) return
    const id = window.setInterval(() => {
      void captureAndAnalyze()
    }, 1500)
    return () => window.clearInterval(id)
  }, [streaming, auto, captureAndAnalyze])

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Live Camera</h1>
          <p className="text-sm text-slate-600">
            Browser camera → single-frame detection. A single frame has no temporal
            features, so no anomaly score is shown here — use Video Analysis for that.
          </p>
        </div>
        <ModelBadge />
      </header>

      <div className="grid gap-6 lg:grid-cols-3">
        <section className="space-y-4 lg:col-span-2">
          <div className="overflow-hidden rounded-xl bg-slate-900 shadow-sm">
            <div className="flex items-center justify-between border-b border-slate-700 px-4 py-3">
              <h2 className="font-medium text-white">Camera feed</h2>
              <button
                type="button"
                onClick={streaming ? stopCamera : startCamera}
                className={[
                  'inline-flex items-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium text-white transition-colors',
                  streaming ? 'bg-red-600 hover:bg-red-700' : 'bg-emerald-600 hover:bg-emerald-700',
                ].join(' ')}
              >
                {streaming ? <CameraOff size={16} aria-hidden /> : <Camera size={16} aria-hidden />}
                {streaming ? 'Stop camera' : 'Start camera'}
              </button>
            </div>

            <div className="relative bg-black">
              <video
                ref={videoRef}
                className="block w-full"
                playsInline
                muted
                onLoadedMetadata={() => drawOverlay(result)}
              />
              <canvas
                ref={overlayRef}
                className="pointer-events-none absolute inset-0 h-full w-full"
                aria-hidden
              />
              {!streaming && (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-slate-400">
                  <Camera size={48} aria-hidden />
                  <p className="text-sm">Camera is off</p>
                  <p className="text-xs">Click “Start camera” and allow browser access</p>
                </div>
              )}
            </div>

            <div className="flex items-center justify-between gap-3 border-t border-slate-700 px-4 py-3">
              <label className="flex items-center gap-2 text-sm text-slate-200">
                <input
                  type="checkbox"
                  checked={auto}
                  onChange={(event) => setAuto(event.target.checked)}
                  disabled={!streaming}
                  className="accent-brand-500"
                />
                Auto-analyze every 1.5 s
              </label>
              <button
                type="button"
                onClick={() => void captureAndAnalyze()}
                disabled={!streaming || analyzing}
                className="inline-flex items-center gap-2 rounded-md bg-brand-600 px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-slate-600"
              >
                <ScanLine size={16} aria-hidden />
                {analyzing ? 'Analyzing…' : 'Analyze frame'}
              </button>
            </div>
          </div>
          <canvas ref={captureRef} className="hidden" aria-hidden />

          {error && (
            <div className="rounded-lg bg-red-50 p-4 text-sm text-red-700 ring-1 ring-inset ring-red-200" role="alert">
              {error}
            </div>
          )}
        </section>

        <aside className="space-y-4">
          <div className="rounded-xl bg-white p-5 shadow-sm ring-1 ring-slate-200">
            <h2 className="mb-3 font-medium text-slate-900">Detection settings</h2>
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
            <label className="mt-4 block text-sm">
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
          </div>

          <div className="rounded-xl bg-white p-5 shadow-sm ring-1 ring-slate-200">
            <h2 className="mb-3 font-medium text-slate-900">Latest frame</h2>
            {result ? (
              <dl className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <dt className="text-slate-500">Persons (estimate)</dt>
                  <dd className="font-semibold text-slate-900">{result.count}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-slate-500">Inference time</dt>
                  <dd className="tabular-nums text-slate-900">
                    {result.metadata.inference_time_s.toFixed(3)} s
                  </dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-slate-500">Frame size</dt>
                  <dd className="tabular-nums text-slate-900">
                    {result.image.width}×{result.image.height}
                  </dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-slate-500">Detector</dt>
                  <dd className="text-slate-900">{result.metadata.model_name}</dd>
                </div>
              </dl>
            ) : (
              <p className="text-sm text-slate-500">
                No frame analyzed yet. Start the camera, then analyze a frame.
              </p>
            )}
          </div>

          <p className="text-xs text-slate-500">
            Counts are detector estimates — no counting ground truth exists. Camera frames
            never leave your machine except to the local API process.
          </p>
        </aside>
      </div>
    </div>
  )
}
