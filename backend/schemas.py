"""Pydantic response models for the CrowdSense AI HTTP API.

Every field is either a measured value or explicitly ``None`` — the API never
invents counts, ground truth, or metrics.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


class ImageInfo(BaseModel):
    width: int
    height: int


class FeatureValue(BaseModel):
    name: str
    value: float | None = None
    unit: str
    limitation: str
    group: str


class DetectionMetadata(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    inference_time_s: float
    confidence: float
    imgsz: int
    caveat: str


class ImageDetectionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    count: int = Field(description="Detector estimate — no counting ground truth exists")
    boxes: list[Box]
    grid_counts: list[list[int]]
    grid_size: int
    static_features: list[FeatureValue]
    image: ImageInfo
    annotated_image: str = Field(description="JPEG data URL of the annotated frame")
    metadata: DetectionMetadata
    model_note: str


class FrameDetectionResponse(BaseModel):
    count: int
    boxes: list[Box]
    image: ImageInfo
    metadata: DetectionMetadata


class Prediction(BaseModel):
    label: int
    score: float = Field(description="P(anomaly) from the trained bundle")
    headline: str
    note: str


class VideoWindow(BaseModel):
    window_start: int
    window_end: int
    features: dict[str, float | None]
    prediction: Prediction | None = None


class VideoStats(BaseModel):
    frames_processed: int
    total_windows: int
    anomaly_windows: int | None = Field(
        default=None, description="None when no compatible model is available"
    )
    anomaly_rate: float | None = None
    processing_time_s: float
    window: int
    window_stride: int


class ModelInfo(BaseModel):
    available: bool
    algorithm: str | None = None
    trained_at: str | None = None
    threshold: float | None = None
    feature_names: list[str] = Field(default_factory=list)
    caveat: str
    note: str | None = Field(default=None, description="Why a model is unavailable")


class VideoInfo(BaseModel):
    name: str
    width: int
    height: int


class VideoDetectionResponse(BaseModel):
    stats: VideoStats
    windows: list[VideoWindow]
    model: ModelInfo
    video: VideoInfo


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
