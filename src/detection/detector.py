"""Person detection with YOLOv8n (CPU by default).

Counting is a DETECTOR ESTIMATE unless compared against ground truth — this
project ships no counting datasets, so no MAE/RMSE is ever reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.config import PROJECT_ROOT

PERSON_CLASS_ID = 0
DEFAULT_WEIGHTS = PROJECT_ROOT / "yolov8n.pt"


class ModelNotFoundError(RuntimeError):
    """Raised when detection weights are missing."""


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls_id: int

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass(frozen=True)
class DetectionResult:
    detections: tuple[Detection, ...]
    inference_time_s: float
    model_name: str
    image_shape: tuple[int, ...]  # (h, w[, c])


@lru_cache(maxsize=2)
def _load_model(weights_path: str, device: str):
    from ultralytics import YOLO

    path = Path(weights_path)
    if not path.is_file():
        raise ModelNotFoundError(
            f"Detection weights not found: {path}. Expected yolov8n.pt in the project root."
        )
    return YOLO(str(path)), device


class PersonDetector:
    """Thin wrapper over ultralytics YOLOv8n filtered to the person class."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        conf: float = 0.05,  # calibrated 2026-10-22 (configs/experiments.yaml)
        device: str = "cpu",
        imgsz: int = 640,
    ):
        self.model_path = Path(model_path) if model_path is not None else DEFAULT_WEIGHTS
        self.conf = conf
        self.device = device
        self.imgsz = imgsz

    def detect(self, image: np.ndarray) -> DetectionResult:
        if image is None or image.size == 0:
            raise ValueError("detect() requires a non-empty image array")
        model, device = _load_model(str(self.model_path.resolve()), self.device)
        results = model.predict(image, conf=self.conf, device=device,
                                imgsz=self.imgsz, verbose=False)[0]
        detections: list[Detection] = []
        if results.boxes is not None and len(results.boxes) > 0:
            xyxy = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                if int(k) != PERSON_CLASS_ID:
                    continue  # person-only
                detections.append(
                    Detection(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                              conf=float(c), cls_id=int(k))
                )
        speed_ms = getattr(results, "speed", {}) or {}
        total_s = sum(v for v in speed_ms.values() if isinstance(v, (int, float))) / 1000.0
        return DetectionResult(
            detections=tuple(detections),
            inference_time_s=total_s,
            model_name=self.model_path.name,
            image_shape=tuple(image.shape),
        )


def draw_detections(image: np.ndarray, detections: tuple[Detection, ...]) -> np.ndarray:
    """Return a copy of ``image`` with green boxes and confidence labels."""
    import cv2

    out = image.copy()
    for d in detections:
        p1 = (int(round(d.x1)), int(round(d.y1)))
        p2 = (int(round(d.x2)), int(round(d.y2)))
        cv2.rectangle(out, p1, p2, (0, 255, 0), 2)
        label = f"{d.conf:.2f}"
        cv2.putText(out, label, (p1[0], max(12, p1[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    return out
