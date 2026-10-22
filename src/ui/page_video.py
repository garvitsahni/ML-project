"""Video Analysis: upload -> detect+track+flow -> windows -> bundle predictions."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from src.detection.detector import PersonDetector
from src.features.pipeline import WindowConfig, _centers_to_grid, _maybe_resize
from src.features.schema import FeatureSchema
from src.inference.service import InferenceError, InferenceService
from src.preprocessing.video import InvalidVideoError, iter_video_frames
from src.ui.common import find_model_path, interpret_prediction, model_caveat
from src.ui.theme import page_header

VIDEO_TYPES = ["mp4", "avi", "mov", "mkv"]


@dataclass(frozen=True)
class _VideoMeta:
    path: Path
    stem: str
    n_frames: int  # frames to consume (after stride)
    frame_stride: int


def _count_frames(video_path: Path, cap_frames: int, stride: int) -> _VideoMeta:
    """Probe frame count; raise InvalidVideoError when unreadable."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise InvalidVideoError(f"Could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if total <= 0:
        raise InvalidVideoError(f"No decodable frames in: {video_path}")
    n = min((total + stride - 1) // stride, cap_frames)
    return _VideoMeta(path=video_path, stem=video_path.stem, n_frames=n, frame_stride=stride)


def _process_video(meta: _VideoMeta, cfg: WindowConfig, progress) -> list[dict]:
    import cv2
    import numpy as np

    from src.features.extract import FrameSignals, extract_window_features, optical_flow_stats
    from src.tracking.tracker import CrowdTracker
    from src.tracking.trajectories import TrajectoryStore

    detector = PersonDetector()
    tracker: CrowdTracker | None = None
    signals: list[FrameSignals] = []
    persons_by_frame: dict[int, list] = {}
    prev_gray = None
    h = w = 0
    read = 0
    for _, frame in iter_video_frames(meta.path):
        if read % meta.frame_stride != 0:
            read += 1
            continue
        read += 1
        if len(signals) >= meta.n_frames:
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
        fm, fs, fdi = (0.0, 0.0, 0.0) if prev_gray is None else optical_flow_stats(prev_gray, gray)
        prev_gray = gray
        signals.append(FrameSignals(
            frame_index=len(signals), count=len(result.detections),
            grid_counts=_centers_to_grid(centers, (h, w), cfg.grid_size),
            centers=centers, flow_mean=fm, flow_std=fs, flow_dir_inconsistency=fdi,
        ))
        progress.progress(min(1.0, len(signals) / max(1, meta.n_frames)))

    rows: list[dict] = []
    for start in range(0, len(signals) - cfg.window + 1, cfg.stride):
        end = start + cfg.window
        ws = TrajectoryStore()
        for fi in range(start, end):
            ws.add_frame(fi, persons_by_frame.get(fi, []))
        feats = extract_window_features(signals[start:end], ws,
                                        schema=FeatureSchema.default(), frame_shape=(h, w))
        rows.append({"window_start": start, "window_end": end, **feats})
    return rows


def render() -> None:
    page_header(
        "Video Analysis", "🎬",
        "Sliding-window (30 frames, stride 5) density + movement features; "
        "model output only if a compatible trained bundle exists.",
    )
    uploaded = st.file_uploader("Video", type=VIDEO_TYPES)
    if uploaded is None:
        return
    max_frames = st.slider("Max frames to process (CPU: ~20-50 ms/frame)", 30, 600, 150, 10)
    stride = st.slider("Frame stride (1 = every frame)", 1, 5, 1)
    tmp = Path(tempfile.mkdtemp()) / uploaded.name
    tmp.write_bytes(uploaded.getvalue())

    cfg = WindowConfig(window=30, stride=5, grid_size=4)
    try:
        meta = _count_frames(tmp, max_frames, stride)
    except InvalidVideoError as exc:
        st.error(f"Could not read video: {exc}. Upload a valid MP4/AVI/MOV/MKV.")
        return

    progress = st.progress(0.0)
    with st.spinner(f"Processing up to {meta.n_frames} frames (detect → track → flow → features)..."):
        try:
            rows = _process_video(meta, cfg, progress)
        except Exception as exc:
            st.error(f"Processing failed: {exc}")
            return
    if not rows:
        st.warning(
            f"Video too short for one 30-frame window (got {meta.n_frames} frames after "
            "stride). Upload a longer clip or raise 'max frames'."
        )
        return

    import pandas as pd

    df = pd.DataFrame(rows)
    st.subheader(f"{len(df)} windows from {meta.n_frames} frames")
    left, right = st.columns(2)
    with left:
        st.caption("Person count per window (detector estimate)")
        st.line_chart(df.set_index("window_start")["person_count"])
    with right:
        st.caption("Mean speed (px/frame — never m/s)")
        st.line_chart(df.set_index("window_start")["mean_speed_pxf"])

    mp = find_model_path()
    if mp is None:
        st.info(
            "No trained model — run Training & Evaluation "
            "(`python scripts/train_ml.py`) to get anomaly predictions here."
        )
    else:
        try:
            svc = InferenceService(mp)
            preds = [svc.predict_row(r) for r in df.to_dict("records")]
            interp = [interpret_prediction(p.label, p.score) for p in preds]
            out = pd.DataFrame({
                "window": [f"{a}:{b}" for a, b in zip(df.window_start, df.window_end)],
                "what it means": [h for h, _ in interp],
                "label (pred)": [p.label for p in preds],
                "anomaly score": [round(p.score, 3) for p in preds],
                "note": [n for _, n in interp],
            })
            st.subheader("Model predictions (schema-strict)")
            st.caption(f"model: {svc.bundle.algorithm} · trained {svc.bundle.trained_at}")
            st.dataframe(out, width="stretch")
            st.line_chart(out.set_index("window")["anomaly score"])
            st.caption(model_caveat())
        except (InferenceError, ValueError) as exc:
            st.warning(f"Model not usable on this data: {exc}")

    with st.expander("Feature table"):
        st.dataframe(df, width="stretch")
        st.download_button(
            "Download windows CSV", df.to_csv(index=False), "crowd windows.csv", "text/csv"
        )
