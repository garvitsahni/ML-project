"""Pipeline adapters used by the FastAPI layer.

The service is deliberately thin: it decodes uploads, delegates to the same
``src`` pipeline the project already tests, and shapes JSON for the React
frontend. It never fabricates counts, ground truth, or metrics.
"""

from __future__ import annotations

import base64
import math
import tempfile
import time
from pathlib import Path

import numpy as np

from src.inference.presentation import (
    find_model_path,
    interpret_prediction,
    model_caveat,
)

MAX_SIDE = 1280
MAX_VIDEO_BYTES = 200 * 1024 * 1024
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class InvalidUploadError(ValueError):
    """Raised for undecodable images/videos or unsupported uploads."""


def _decode_image(data: bytes, max_side: int = MAX_SIDE) -> np.ndarray:
    """Decode uploaded bytes to a BGR frame, resized so the long side <= max_side."""
    import cv2

    if not data:
        raise InvalidUploadError("Empty upload")
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidUploadError("Could not decode image — upload a valid JPG/PNG.")
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))))
    return image


def _encode_data_url(image_bgr: np.ndarray, quality: int = 85) -> str:
    """Encode a BGR frame as a JPEG data URL (kept in memory; nothing written)."""
    import cv2

    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise InvalidUploadError("Could not encode annotated image")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _feature_specs() -> dict:
    from src.features.schema import FeatureSchema

    return {spec.name: spec for spec in FeatureSchema.default().specs}


def _static_feature_rows(features: dict[str, float]) -> list[dict]:
    specs = _feature_specs()
    rows = []
    for name, value in features.items():
        spec = specs.get(name)
        rows.append({
            "name": name,
            "value": float(value) if value is not None else None,
            "unit": spec.unit if spec else "",
            "limitation": spec.limitation if spec else "",
            "group": spec.group if spec else "static",
        })
    return rows


def _safe_video_path(directory: Path, filename: str) -> Path:
    name = Path(filename or "upload").name
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise InvalidUploadError(
            f"Unsupported video type '{suffix or 'unknown'}'. "
            f"Use one of: {', '.join(sorted(ALLOWED_VIDEO_SUFFIXES))}."
        )
    return directory / f"upload{suffix}"


def model_info() -> dict:
    """Describe the newest trained bundle without raising on missing/corrupt files."""
    from src.inference.service import InferenceError, InferenceService

    path = find_model_path()
    if path is None:
        return {
            "available": False,
            "caveat": model_caveat(),
            "note": (
                "No trained bundle found under reports/models. "
                "Run `python scripts/train_ml.py` to enable anomaly scores."
            ),
        }
    try:
        svc = InferenceService(path)
    except (InferenceError, ValueError) as exc:
        return {
            "available": False,
            "caveat": model_caveat(),
            "note": str(exc),
        }
    bundle = svc.bundle
    return {
        "available": True,
        "algorithm": bundle.algorithm,
        "trained_at": bundle.trained_at,
        "threshold": float(bundle.threshold),
        "feature_names": list(bundle.feature_names),
        "caveat": model_caveat(),
    }


def _load_service():
    """Return ``(service, note)``; ``service`` is None when unusable."""
    from src.inference.service import InferenceError, InferenceService

    path = find_model_path()
    if path is None:
        return None, "No trained bundle — detector-only output."
    try:
        return InferenceService(path), None
    except (InferenceError, ValueError) as exc:
        return None, f"Model not usable: {exc}"


def detect_image(data: bytes, conf: float, imgsz: int, grid_size: int = 4) -> dict:
    """Detect persons in one photo and return static features + annotation."""
    from src.detection.analyze import analyze_frame
    from src.detection.detector import PersonDetector
    from src.features.static import static_features

    image = _decode_image(data)
    detector = PersonDetector(conf=conf, imgsz=imgsz)
    analysis = analyze_frame(image, detector=detector, grid_size=grid_size)

    detections = analysis.detection_result.detections
    centers = np.array([d.center for d in detections], dtype=np.float64).reshape(-1, 2)
    feats = static_features(
        analysis.count, analysis.grid_counts, centers, image.shape[:2], grid_size=grid_size,
    )

    h, w = image.shape[:2]
    return {
        "count": int(analysis.count),
        "boxes": [
            {
                "x1": float(d.x1), "y1": float(d.y1),
                "x2": float(d.x2), "y2": float(d.y2),
                "confidence": float(d.conf),
            }
            for d in detections
        ],
        "grid_counts": analysis.grid_counts.astype(int).tolist(),
        "grid_size": grid_size,
        "static_features": _static_feature_rows(feats),
        "image": {"width": int(w), "height": int(h)},
        "annotated_image": _encode_data_url(analysis.annotated),
        "metadata": {
            "model_name": analysis.model_name,
            "inference_time_s": float(analysis.inference_time_s),
            "confidence": float(conf),
            "imgsz": int(imgsz),
            "caveat": "Detector count is an estimate — no counting ground truth exists.",
        },
        "model_note": (
            "A single photo has no tracks or optical flow, so the trained "
            "anomaly model (which consumes movement features) is not applied. "
            "Use Video Analysis for anomaly scores."
        ),
    }


def detect_frame(data: bytes, conf: float, imgsz: int) -> dict:
    """Detect persons in one live-camera frame (no temporal features)."""
    from src.detection.detector import PersonDetector

    image = _decode_image(data, max_side=960)
    detector = PersonDetector(conf=conf, imgsz=imgsz)
    result = detector.detect(image)
    h, w = image.shape[:2]
    return {
        "count": len(result.detections),
        "boxes": [
            {
                "x1": float(d.x1), "y1": float(d.y1),
                "x2": float(d.x2), "y2": float(d.y2),
                "confidence": float(d.conf),
            }
            for d in result.detections
        ],
        "image": {"width": int(w), "height": int(h)},
        "metadata": {
            "model_name": result.model_name,
            "inference_time_s": float(result.inference_time_s),
            "confidence": float(conf),
            "imgsz": int(imgsz),
            "caveat": "Detector count is an estimate — no counting ground truth exists.",
        },
    }


def _probe_video(path: Path, max_frames: int, stride: int) -> tuple[int, int, int]:
    """Return (frames_to_consume, width, height); raise on unreadable video."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise InvalidUploadError("Could not open video — upload a valid MP4/AVI/MOV/MKV.")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    if total <= 0:
        raise InvalidUploadError("No decodable frames in the uploaded video.")
    return min(math.ceil(total / stride), max_frames), width, height


def analyze_video(
    data: bytes,
    filename: str,
    *,
    max_frames: int = 150,
    frame_stride: int = 1,
    conf: float = 0.05,
    imgsz: int = 640,
    window: int = 30,
    window_stride: int = 5,
    grid_size: int = 4,
) -> dict:
    """Detect -> track -> flow -> sliding-window features -> optional scoring."""
    import cv2

    from src.detection.detector import PersonDetector
    from src.features.extract import (
        FrameSignals,
        extract_window_features,
        optical_flow_stats,
    )
    from src.features.pipeline import WindowConfig, _centers_to_grid, _maybe_resize
    from src.features.schema import FeatureSchema
    from src.preprocessing.video import iter_video_frames
    from src.tracking.tracker import CrowdTracker
    from src.tracking.trajectories import TrajectoryStore

    if len(data) > MAX_VIDEO_BYTES:
        raise InvalidUploadError(
            f"Video exceeds {MAX_VIDEO_BYTES // (1024 * 1024)} MB limit."
        )
    tmp_dir = Path(tempfile.mkdtemp(prefix="crowdsense-"))
    started = time.perf_counter()
    try:
        path = _safe_video_path(tmp_dir, filename)
        path.write_bytes(data)
        n_frames, src_w, src_h = _probe_video(path, max_frames, frame_stride)

        cfg = WindowConfig(window=window, stride=window_stride,
                           grid_size=grid_size, max_dim=MAX_SIDE)
        detector = PersonDetector(conf=conf, imgsz=imgsz)
        tracker: CrowdTracker | None = None
        signals: list[FrameSignals] = []
        persons_by_frame: dict[int, list] = {}
        prev_gray = None
        read = 0
        h = w = 0
        for _, frame in iter_video_frames(path):
            if read % frame_stride != 0:
                read += 1
                continue
            read += 1
            if len(signals) >= n_frames:
                break
            frame = _maybe_resize(frame, cfg.max_dim)
            h, w = frame.shape[:2]
            if tracker is None:
                tracker = CrowdTracker(frame_width=w, frame_height=h)
            result = detector.detect(frame)
            persons = tracker.update(result.detections)
            persons_by_frame[len(signals)] = persons
            centers = np.array([p.center for p in persons], float).reshape(-1, 2)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fm, fs, fdi = (
                (0.0, 0.0, 0.0) if prev_gray is None
                else optical_flow_stats(prev_gray, gray)
            )
            prev_gray = gray
            signals.append(FrameSignals(
                frame_index=len(signals),
                count=len(result.detections),
                grid_counts=_centers_to_grid(centers, (h, w), cfg.grid_size),
                centers=centers,
                flow_mean=fm,
                flow_std=fs,
                flow_dir_inconsistency=fdi,
            ))

        rows: list[dict] = []
        for start in range(0, len(signals) - cfg.window + 1, cfg.stride):
            end = start + cfg.window
            store = TrajectoryStore()
            for fi in range(start, end):
                store.add_frame(fi, persons_by_frame.get(fi, []))
            feats = extract_window_features(
                signals[start:end], store,
                schema=FeatureSchema.default(), frame_shape=(h, w),
            )
            rows.append({"window_start": start, "window_end": end, **feats})

        svc, note = _load_service()
        windows = []
        anomaly_windows: int | None = None
        if svc is not None:
            anomaly_windows = 0
            bundle = svc.bundle
            model = {
                "available": True,
                "algorithm": bundle.algorithm,
                "trained_at": bundle.trained_at,
                "threshold": float(bundle.threshold),
                "feature_names": list(bundle.feature_names),
                "caveat": model_caveat(),
                "note": None,
            }
        else:
            model = {
                "available": False,
                "algorithm": None,
                "trained_at": None,
                "threshold": None,
                "feature_names": [],
                "caveat": model_caveat(),
                "note": note,
            }
        for row in rows:
            prediction = None
            if svc is not None:
                result = svc.predict_row(row)
                headline, why = interpret_prediction(result.label, result.score)
                prediction = {
                    "label": int(result.label),
                    "score": float(result.score),
                    "headline": headline,
                    "note": why,
                }
                anomaly_windows = int(anomaly_windows) + int(result.label == 1)
            windows.append({
                "window_start": int(row["window_start"]),
                "window_end": int(row["window_end"]),
                "features": {
                    k: (float(v) if v is not None else None)
                    for k, v in row.items()
                    if k not in {"window_start", "window_end"}
                },
                "prediction": prediction,
            })

        total_windows = len(windows)
        return {
            "stats": {
                "frames_processed": len(signals),
                "total_windows": total_windows,
                "anomaly_windows": anomaly_windows,
                "anomaly_rate": (
                    (anomaly_windows / total_windows)
                    if anomaly_windows is not None and total_windows else None
                ),
                "processing_time_s": float(time.perf_counter() - started),
                "window": window,
                "window_stride": window_stride,
            },
            "windows": windows,
            "model": model,
            "video": {
                "name": Path(filename or "upload").name,
                "width": int(w or src_w),
                "height": int(h or src_h),
            },
        }
    finally:
        # best-effort cleanup; a failed unlink must not mask the response
        try:
            for child in tmp_dir.glob("*"):
                child.unlink(missing_ok=True)
            tmp_dir.rmdir()
        except OSError:
            pass
