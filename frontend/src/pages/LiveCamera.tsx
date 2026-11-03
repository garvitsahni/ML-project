import { useCallback, useEffect, useRef, useState } from 'react'
import { Camera, CameraOff, ScanLine } from 'lucide-react'

import { Details } from '../components/Details'
import { VerdictCard } from '../components/VerdictCard'
import { crowdLevel } from '../lib/crowdLevel'
import { detectFrame, friendlyError } from '../lib/api'
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
    ctx.strokeStyle = '#3f6b52'
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
      setError(
        `We could not open the camera. Allow camera access in your browser, then try again. (${friendlyError(
          caught,
        )})`,
      )
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
      setError(friendlyError(caught))
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

  const level = result ? crowdLevel(result.count) : null

  return (
    <div className="space-y-8">
      <header>
        <p className="text-[11px] font-bold uppercase tracking-[0.16em] text-accent">
          Live camera
        </p>
        <h1 className="mt-1 text-2xl font-bold tracking-tight text-ink sm:text-3xl">
          How crowded is this room?
        </h1>
        <p className="mt-2 max-w-xl text-muted">
          Point your camera at a crowd. The level updates while the camera runs.
        </p>
      </header>

      <div className="grid gap-8 lg:grid-cols-3">
        <section className="space-y-5 lg:col-span-2">
          <div className="overflow-hidden rounded-xl border border-ink bg-ink">
            <div className="flex items-center justify-between border-b border-white/10 px-4 py-3">
              <h2 className="text-sm font-semibold tracking-tight text-paper">Camera</h2>
              <button
                type="button"
                onClick={streaming ? stopCamera : startCamera}
                className={[
                  'inline-flex items-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium text-white transition-colors',
                  streaming ? 'bg-crowded hover:bg-[#9a3826]' : 'bg-quiet hover:bg-[#34573f]',
                ].join(' ')}
              >
                {streaming ? (
                  <CameraOff size={15} aria-hidden />
                ) : (
                  <Camera size={15} aria-hidden />
                )}
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
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-white/50">
                  <Camera size={42} aria-hidden />
                  <p className="text-sm">Camera is off</p>
                  <p className="text-xs">Press “Start camera” and allow access</p>
                </div>
              )}
            </div>

            <div className="flex items-center justify-between gap-3 border-t border-white/10 px-4 py-3">
              <label className="flex items-center gap-2 text-sm text-paper/80">
                <input
                  type="checkbox"
                  checked={auto}
                  onChange={(event) => setAuto(event.target.checked)}
                  disabled={!streaming}
                  className="accent-paper"
                />
                Check automatically
              </label>
              <button
                type="button"
                onClick={() => void captureAndAnalyze()}
                disabled={!streaming || analyzing}
                className="inline-flex items-center gap-2 rounded-md bg-paper px-3 py-2 text-sm font-medium text-ink transition-colors hover:bg-white disabled:cursor-not-allowed disabled:bg-white/20 disabled:text-white/50"
              >
                <ScanLine size={15} aria-hidden />
                {analyzing ? 'Checking…' : 'Check now'}
              </button>
            </div>
          </div>
          <canvas ref={captureRef} className="hidden" aria-hidden />

          {error && (
            <div
              className="rounded-lg border border-[#f0d3c9] bg-[#fdf1ee] p-4 text-sm text-[#8f2f1c]"
              role="alert"
            >
              {error}
            </div>
          )}
        </section>

        <aside className="space-y-5">
          {result && level ? (
            <VerdictCard
              level={level}
              people={result.count}
              caption="in the latest frame"
              compact
            />
          ) : (
            <div className="rounded-xl border border-line bg-card p-5 text-sm leading-relaxed text-muted">
              Start the camera to see the crowd level here.
            </div>
          )}

          {result && (
            <p className="text-xs text-faint">
              Checked in {result.metadata.inference_time_s.toFixed(2)} s · frame{' '}
              {result.image.width}×{result.image.height}
            </p>
          )}

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
                Lower finds more people, including ones it is unsure about.
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
            Counts are estimates from the detector, not a head count. The camera feed stays
            on this computer.
          </p>
        </aside>
      </div>
    </div>
  )
}
