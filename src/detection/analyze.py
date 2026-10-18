"""Single-image crowd analysis: count, spatial density map, annotated frame.

Density here is a DETECTOR-BASED ESTIMATE over the image area — not a
ground-truth-calibrated count (no counting dataset is used in this project).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.detection.detector import DetectionResult, PersonDetector, draw_detections
from src.preprocessing.image import ImageMetadata, validate_image


@dataclass(frozen=True)
class ImageAnalysis:
    count: int
    density_map: np.ndarray  # float32, same HxW as input, Gaussian KDE over box centers
    grid_counts: np.ndarray  # (grid, grid) int counts of box centers per cell
    annotated: np.ndarray  # BGR image with boxes drawn
    detection_result: DetectionResult
    inference_time_s: float
    model_name: str
    meta: ImageMetadata | None = None


def _density_map(shape_hw: tuple[int, int], centers: list[tuple[float, float]]) -> np.ndarray:
    h, w = shape_hw
    if not centers:
        return np.zeros((h, w), dtype=np.float32)
    points = np.zeros((h, w), dtype=np.uint8)
    for cx, cy in centers:
        x, y = int(round(cx)), int(round(cy))
        if 0 <= x < w and 0 <= y < h:
            points[y, x] = 255
    sigma = max(4.0, min(h, w) / 32.0)
    k = max(3, int(sigma) * 6 + 1)
    return cv2.GaussianBlur(points.astype(np.float32) / 255.0, (k, k), sigma)


def _grid_counts(shape_hw: tuple[int, int], centers: list[tuple[float, float]], grid: int) -> np.ndarray:
    counts = np.zeros((grid, grid), dtype=np.int64)
    h, w = shape_hw
    for cx, cy in centers:
        col = min(grid - 1, max(0, int(cx / w * grid)))
        row = min(grid - 1, max(0, int(cy / h * grid)))
        counts[row, col] += 1
    return counts


def analyze_frame(
    frame: np.ndarray,
    detector: PersonDetector | None = None,
    grid_size: int = 4,
) -> ImageAnalysis:
    """Analyze an in-memory BGR frame (no disk metadata)."""
    import time as _time

    if detector is None:
        detector = PersonDetector()
    t0 = _time.perf_counter()
    result = detector.detect(frame)
    centers = [d.center for d in result.detections]
    h, w = frame.shape[:2]
    return ImageAnalysis(
        count=len(result.detections),
        density_map=_density_map((h, w), centers),
        grid_counts=_grid_counts((h, w), centers, grid_size),
        annotated=draw_detections(frame, result.detections),
        detection_result=result,
        inference_time_s=_time.perf_counter() - t0,
        model_name=result.model_name,
    )


def analyze_image(
    path: Path | str,
    detector: PersonDetector | None = None,
    grid_size: int = 4,
) -> ImageAnalysis:
    """Validate + analyze a photo from disk (the Photo page path)."""
    frame, meta = validate_image(path)
    analysis = analyze_frame(frame, detector=detector, grid_size=grid_size)
    return ImageAnalysis(
        count=analysis.count,
        density_map=analysis.density_map,
        grid_counts=analysis.grid_counts,
        annotated=analysis.annotated,
        detection_result=analysis.detection_result,
        inference_time_s=analysis.inference_time_s,
        model_name=analysis.model_name,
        meta=meta,
    )


__all__ = ["ImageAnalysis", "analyze_frame", "analyze_image"]
