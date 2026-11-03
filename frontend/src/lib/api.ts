import type {
  DetectionOptions,
  FrameDetectionResult,
  ImageDetectionResult,
  ModelInfo,
  VideoDetectionResult,
  VideoOptions,
} from '../types'

const API_BASE = (import.meta.env.VITE_API_URL ?? '').replace(/\/$/, '')

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init)
  if (!response.ok) {
    let detail = `Request failed (HTTP ${response.status})`
    try {
      const body = (await response.json()) as { detail?: unknown }
      if (typeof body.detail === 'string') detail = body.detail
    } catch {
      // non-JSON error body: keep the status message
    }
    throw new Error(detail)
  }
  return (await response.json()) as T
}

function detectionForm(file: Blob, options: DetectionOptions): FormData {
  const form = new FormData()
  form.append('file', file)
  form.append('conf', String(options.conf))
  form.append('imgsz', String(options.imgsz))
  return form
}

export function getModel(): Promise<ModelInfo> {
  return request<ModelInfo>('/api/model')
}

export function detectImage(
  file: File,
  options: DetectionOptions,
): Promise<ImageDetectionResult> {
  return request<ImageDetectionResult>('/api/detect-image', {
    method: 'POST',
    body: detectionForm(file, options),
  })
}

export function detectFrame(
  frame: Blob,
  options: DetectionOptions,
): Promise<FrameDetectionResult> {
  const form = detectionForm(frame, options)
  return request<FrameDetectionResult>('/api/detect-frame', {
    method: 'POST',
    body: form,
  })
}

export function detectVideo(
  file: File,
  options: VideoOptions,
): Promise<VideoDetectionResult> {
  const form = detectionForm(file, options)
  form.append('max_frames', String(options.maxFrames))
  form.append('stride', String(options.stride))
  return request<VideoDetectionResult>('/api/detect-video', {
    method: 'POST',
    body: form,
  })
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

/** Turn a raw failure into something a first-time user can act on. */
export function friendlyError(error: unknown): string {
  const raw = errorMessage(error)
  if (/failed to fetch|networkerror|load failed|network request failed/i.test(raw)) {
    return 'We could not reach the analysis service. Make sure the server is running, then try again.'
  }
  if (/HTTP 413/.test(raw)) {
    return 'That file is too large. Try a smaller photo or a shorter clip.'
  }
  if (/HTTP 415/.test(raw)) {
    return 'That file type is not supported. Try a JPG, PNG, or MP4.'
  }
  if (/HTTP 422/.test(raw)) {
    return 'We could not read that file. Try a different photo or clip.'
  }
  if (/HTTP 5\d\d/.test(raw)) {
    return 'Something went wrong on the server. Try again in a moment.'
  }
  return raw
}
