export interface Box {
  x1: number
  y1: number
  x2: number
  y2: number
  confidence: number
}

export interface ImageInfo {
  width: number
  height: number
}

export interface FeatureValue {
  name: string
  value: number | null
  unit: string
  limitation: string
  group: string
}

export interface DetectionMetadata {
  model_name: string
  inference_time_s: number
  confidence: number
  imgsz: number
  caveat: string
}

export interface ImageDetectionResult {
  count: number
  boxes: Box[]
  grid_counts: number[][]
  grid_size: number
  static_features: FeatureValue[]
  image: ImageInfo
  annotated_image: string
  metadata: DetectionMetadata
  model_note: string
}

export interface FrameDetectionResult {
  count: number
  boxes: Box[]
  image: ImageInfo
  metadata: DetectionMetadata
}

export interface Prediction {
  label: number
  score: number
  headline: string
  note: string
}

export interface VideoWindow {
  window_start: number
  window_end: number
  features: Record<string, number | null>
  prediction: Prediction | null
}

export interface VideoStats {
  frames_processed: number
  total_windows: number
  anomaly_windows: number | null
  anomaly_rate: number | null
  processing_time_s: number
  window: number
  window_stride: number
}

export interface ModelInfo {
  available: boolean
  algorithm?: string | null
  trained_at?: string | null
  threshold?: number | null
  feature_names?: string[]
  caveat: string
  note?: string | null
}

export interface VideoInfo {
  name: string
  width: number
  height: number
}

export interface VideoDetectionResult {
  stats: VideoStats
  windows: VideoWindow[]
  model: ModelInfo
  video: VideoInfo
}

export interface DetectionOptions {
  conf: number
  imgsz: number
}

export interface VideoOptions extends DetectionOptions {
  maxFrames: number
  stride: number
}
