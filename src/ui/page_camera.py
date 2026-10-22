"""Live Camera: device index capture, rolling-window features, model if bundle."""

from __future__ import annotations

import streamlit as st

from src.config import load_paths
from src.preprocessing.camera import CameraUnavailableError
from src.ui.theme import page_header


def render() -> None:
    page_header(
        "Live Camera", "📹",
        "Rolling 30-frame window of density + movement features from a webcam. "
        "Requires a physical camera; graceful status when unavailable.",
    )
    device = st.number_input("Camera device index", min_value=0, max_value=10, value=0)
    stride = st.slider("Frame stride", 1, 5, 2)
    max_seconds = st.slider("Capture cap (seconds, CPU budget)", 5, 60, 15)
    start = st.button("Start capture", type="primary")

    if start:
        _run_capture(int(device), stride, max_seconds)


def _run_capture(device: int, stride: int, cap_seconds: int) -> None:
    import time

    import cv2
    import numpy as np

    from src.detection.detector import PersonDetector
    from src.features.extract import FrameSignals, extract_window_features, optical_flow_stats
    from src.features.pipeline import _centers_to_grid
    from src.features.schema import FeatureSchema
    from src.preprocessing.camera import Camera, CameraUnavailableError
    from src.tracking.tracker import CrowdTracker
    from src.tracking.trajectories import TrajectoryStore
    from src.ui.common import find_model_path, interpret_prediction, model_caveat
    from src.inference.service import InferenceError, InferenceService

    window = 30
    status = st.empty()
    frame_slot = st.empty()
    metrics_slot = st.empty()
    cap = None
    try:
        cap = Camera(device).open()
        detector = PersonDetector()
        tracker = None
        signals: list[FrameSignals] = []
        persons_by_frame: dict[int, list] = {}
        prev_gray = None
        deadline = time.time() + cap_seconds
        read = 0
        h = w = 0
        svc = None
        mp = find_model_path()
        if mp:
            try:
                svc = InferenceService(mp)
            except InferenceError:
                pass
        while time.time() < deadline:
            ok, frame = cap.read()
            if not ok:
                status.error("Camera stream ended unexpectedly.")
                break
            if read % stride != 0:
                read += 1
                continue
            read += 1
            frame = cv2.resize(frame, (640, 360))
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
                grid_counts=_centers_to_grid(centers, (h, w), 4),
                centers=centers, flow_mean=fm, flow_std=fs, flow_dir_inconsistency=fdi,
            ))
            frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")
            if len(signals) >= window:
                seg = signals[-window:]
                store = TrajectoryStore()
                for fi in range(len(seg)):
                    store.add_frame(fi, persons_by_frame[len(signals) - window + fi])
                feats = extract_window_features(seg, store, schema=FeatureSchema.default(),
                                                frame_shape=(h, w))
                cols = metrics_slot.columns(4)
                cols[0].metric("count", round(feats["person_count"], 1))
                cols[1].metric("speed px/f", round(feats["mean_speed_pxf"], 2))
                cols[2].metric("dir. consistency", round(feats["directional_consistency"], 2))
                if svc:
                    res = svc.predict_row(feats)
                    cols[3].metric("anomaly score", round(res.score, 3))
                    headline, note = interpret_prediction(res.label, res.score)
                    st.caption(f"{headline} — {note}. {model_caveat()}")
            status.info(f"frames: {len(signals)} · stride {stride} · ends in "
                        f"{max(0, int(deadline - time.time()))}s")
        else:
            status.success("Capture window complete.")
    except CameraUnavailableError as exc:
        st.error(f"Camera unavailable: {exc}. Check the device index / permissions.")
    except Exception as exc:  # keep dashboard alive; never crash
        st.error(f"Capture failed: {exc}")
    finally:
        if cap is not None:
            cap.release()
        status.info("Camera released.")
