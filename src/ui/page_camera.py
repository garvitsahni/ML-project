"""Live Camera workspace: batched capture, rolling windows, measured timings.

Capture runs in ~15-frame batches inside an auto-rerunning fragment, so the
Stop button takes effect at a batch boundary and the script never blocks. All
live numbers (FPS, latency, windows, scores) are measured from the real
pipeline; when no capture is running the panels show empty states instead of
fabricated metrics.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import streamlit as st

from src.inference.service import InferenceError, InferenceService
from src.preprocessing.camera import CameraUnavailableError
from src.ui import components as ui
from src.ui import state
from src.ui.analysis_loop import FrameAnalyzer, draw_persons
from src.ui.charts import timeline_series
from src.ui.common import find_model_path, load_model_algorithm

_WINDOW = 30           # rolling window length in processed frames
_WINDOW_STRIDE = 5     # window evaluation step (pipeline default)
_BATCH_FRAMES = 15     # frames per fragment rerun (Stop lands between batches)
_CAMERA_MAX_DIM = 640  # live processing cap (CPU latency budget)
_TIMELINE_POINTS = 120  # observation history kept for the timeline chart


# --------------------------------------------------------------------------
# pure-ish helpers (unit-tested with injected fakes)
# --------------------------------------------------------------------------

def step_capture(n: int, *, store: Any = None) -> dict:
    """Process up to n frames from the injected capture (batch stepper).

    Reads `camera_cap`, `camera_analyzer` and `camera_cfg` (stride/max_dim)
    from `store` (defaults to st.session_state), so tests inject fakes.
    Returns measured step metrics — FPS and latency are real timings.
    """
    from src.features.pipeline import _maybe_resize

    store = store if store is not None else st.session_state
    cap = store["camera_cap"]
    analyzer: FrameAnalyzer = store["camera_analyzer"]
    cfg = store.get("camera_cfg") or {}
    stride = int(cfg.get("stride", 1))
    max_dim = cfg.get("max_dim")

    t0 = time.perf_counter()
    read = processed = 0
    last = None
    last_frame = None
    stream_ended = False
    while read < n:
        ok, frame = cap.read()
        read += 1
        if not ok:
            stream_ended = True
            break
        if (read - 1) % stride != 0:
            continue
        frame = _maybe_resize(frame, max_dim)
        last = analyzer.add_frame(frame)
        last_frame = frame
        processed += 1
    elapsed = time.perf_counter() - t0
    return {
        "read": read,
        "processed": processed,
        "stream_ended": stream_ended,
        "last": last,
        "frame": last_frame,
        "window": analyzer.latest_window_features(),
        "fps": (processed / elapsed) if elapsed > 0 else 0.0,
        "latency_ms": (elapsed / processed * 1000.0) if processed else 0.0,
        "elapsed_s": elapsed,
    }


def _overlay_grid(frame, grid, *, color: tuple[int, int, int] = (191, 212, 45)):
    """Teal per-cell density overlay; intensity scales with the cell count.

    Zero-count cells are left untouched so the overlay never invents density
    where the tracker saw nobody.
    """
    import cv2
    import numpy as np

    h, w = frame.shape[:2]
    g = grid.shape[0]
    out = frame.copy()
    peak = max(1.0, float(grid.max()))
    ch, cw = h / g, w / g
    tint = np.zeros_like(frame)
    for r in range(g):
        for c in range(g):
            v = float(grid[r, c])
            if v <= 0:
                continue
            y0 = int(r * ch)
            y1 = int((r + 1) * ch) if r < g - 1 else h
            x0 = int(c * cw)
            x1 = int((c + 1) * cw) if c < g - 1 else w
            inten = 0.35 + 0.65 * (v / peak)
            tint[y0:y1, x0:x1] = np.array(color, dtype=np.float32) * inten
    blended = cv2.addWeighted(frame, 0.6, tint.astype(np.uint8), 0.4, 0)
    mask = tint.sum(axis=2) > 0
    out[mask] = blended[mask]
    return out


def _probe_devices(max_index: int = 9, *,
                   opener: Callable[[int], Any] | None = None,
                   store: Any = None) -> list[int]:
    """Indexes that open as camera devices (cached per session).

    `opener` is injectable for tests; the default opens cv2.VideoCapture.
    """
    store = store if store is not None else st.session_state
    cached = store.get("camera_devices")
    if cached is not None:
        return list(cached)
    if opener is None:
        import cv2

        def open_fn(i: int):
            return cv2.VideoCapture(i)
    else:
        open_fn = opener
    found = []
    for i in range(max_index + 1):
        cap = open_fn(i)
        try:
            if cap.isOpened():
                found.append(i)
        finally:
            cap.release()
    store["camera_devices"] = found
    return list(found)


# --------------------------------------------------------------------------
# session lifecycle
# --------------------------------------------------------------------------

def _get_service() -> InferenceService | None:
    """Cached inference service for live anomaly scores (None when absent)."""
    store = st.session_state
    if "camera_svc" not in store:
        store["camera_svc"] = None
        model_path = find_model_path()
        if model_path:
            try:
                store["camera_svc"] = InferenceService(model_path)
            except InferenceError:
                store["camera_svc"] = None
    return store["camera_svc"]


def _record_observation(run: dict, window: dict, score: float | None) -> None:
    t = time.time() - run["started"]
    run["observations"].append({
        "t": round(t, 1),
        "person_count": float(window["person_count"]),
        "speed": float(window["mean_speed_pxf"]),
        "score": float(score) if score is not None else None,
    })
    if len(run["observations"]) > _TIMELINE_POINTS:
        run["observations"] = run["observations"][-_TIMELINE_POINTS:]


def _record_history(run: dict) -> None:
    model_path = find_model_path()
    algo = load_model_algorithm(model_path) if model_path else "detection only"
    state.record_analysis("camera", {
        "device": run["device"],
        "frames": run["frames"],
        "seconds": round(run.get("duration", 0.0), 1),
        "model": algo,
    })


def _restart_camera() -> None:
    """Drop the run state and release the device (keeps the model cache)."""
    store = st.session_state
    store.pop("camera_run", None)
    store.pop("camera_analyzer", None)
    store.pop("camera_cfg", None)
    state.release_camera(store)


def _start_capture(device: int, stride: int, cap_seconds: int) -> None:
    """Open the device and initialise the incremental analyzer + run state."""
    from src.preprocessing.camera import Camera

    store = st.session_state
    _restart_camera()  # safety: never leak a previous capture handle
    try:
        cap = Camera(device).open()
    except CameraUnavailableError as exc:
        store["camera_run"] = {
            "phase": "error",
            "error": f"{exc}. Check the device index / permissions.",
            "device": device, "frames": 0, "duration": 0.0,
            "last_window": None, "last_step": None, "observations": [],
        }
        return
    store["camera_cap"] = cap
    store["camera_running"] = True
    store["camera_analyzer"] = FrameAnalyzer(
        max_dim=_CAMERA_MAX_DIM, window=_WINDOW, window_stride=_WINDOW_STRIDE)
    store["camera_cfg"] = {"stride": stride, "max_dim": _CAMERA_MAX_DIM}
    store["camera_run"] = {
        "phase": "processing", "device": device, "stride": stride,
        "max_seconds": cap_seconds, "started": time.time(),
        "frames": 0, "steps": 0, "last_step": None, "last_window": None,
        "score": None, "frame_rgb": None, "observations": [],
        "error": None, "reason": None, "duration": 0.0,
    }


def _finish_capture(run: dict, *, phase: str, reason: str) -> None:
    """Release the device and move to a terminal phase (batch boundary)."""
    state.release_camera()
    run["duration"] = max(0.0, time.time() - run["started"])
    run["phase"] = phase
    run["reason"] = reason
    if phase in ("stopped", "ended"):
        _record_history(run)


# --------------------------------------------------------------------------
# processing fragment
# --------------------------------------------------------------------------

@st.fragment(run_every="0.4s")
def _capture_fragment() -> None:
    """One capture batch per rerun; terminal phases end via a full rerun."""
    run = st.session_state.get("camera_run")
    if not run or run.get("phase") != "processing":
        return
    if st.button("Stop capture", key="camera_stop"):
        _finish_capture(run, phase="stopped", reason="stopped by user")
        st.rerun()
        return
    try:
        step = step_capture(_BATCH_FRAMES)
    except Exception as exc:  # keep the dashboard alive; release the device
        run["error"] = f"{type(exc).__name__}: {exc}"
        _finish_capture(run, phase="error", reason="capture failed")
        st.rerun()
        return
    run["last_step"] = step
    run["steps"] += 1
    run["frames"] += step["processed"]
    if step["stream_ended"]:
        _finish_capture(run, phase="ended",
                        reason="the camera stream ended unexpectedly")
        st.rerun()
        return
    last = step["last"]
    window = step["window"]
    if window is not None:
        run["last_window"] = window
        score = None
        service = _get_service()
        if service is not None:
            feats = {k: v for k, v in window.items()
                     if k not in ("window_start", "window_end")}
            try:
                score = float(service.predict_row(feats).score)
            except InferenceError:
                score = None
        run["score"] = score
        _record_observation(run, window, score)
    if last is not None and step.get("frame") is not None:
        import cv2

        vis = step["frame"]
        if state.settings().get("show_density", True):
            vis = _overlay_grid(vis, last["grid_counts"])
        run["frame_rgb"] = cv2.cvtColor(
            draw_persons(vis, last["persons"]), cv2.COLOR_BGR2RGB)
    _render_live(run)


# --------------------------------------------------------------------------
# render pieces
# --------------------------------------------------------------------------

def _render_model_status() -> None:
    model_path = find_model_path()
    if model_path:
        algo = load_model_algorithm(model_path)
        st.info(f"Trained {algo} bundle available — anomaly scores appear "
                "once a full rolling window is collected.")
    else:
        st.info("No trained ML bundle — live metrics are "
                "detection/tracking/flow only; anomaly scores are omitted, "
                "not zeroed.")


def _render_config() -> bool:
    """Settings panel + Start button; True when a capture was started."""
    s = state.settings()
    devices = _probe_devices()
    with ui.card("Camera",
                 hint="Detected capture devices — pick the webcam to "
                      "analyse."):
        if not devices:
            st.caption("No camera device was detected — you can still try "
                       "index 0 manually.")
        device = st.selectbox(
            "Camera device", devices or [0], key="camera_device_sel",
            format_func=lambda i: f"Device {i}")
        sc1, sc2 = st.columns(2)
        with sc1:
            stride = st.slider(
                "Frame stride (process every Nth frame)", 1, 5,
                min(5, max(1, int(s["frame_stride"]))), key="camera_stride",
                help="Higher = faster, coarser timeline.")
        with sc2:
            cap_seconds = st.slider(
                "Capture cap (seconds)", 5, 60, 20, key="camera_seconds",
                help="CPU budget: capture stops automatically at the cap.")
        st.caption(
            f"Live capture processes at ≤{_CAMERA_MAX_DIM} px for CPU "
            f"latency; rolling window = {_WINDOW} frames. The density "
            "overlay follows the show-density setting.")
        start = st.button("Start capture", type="primary", key="camera_start")
    _render_model_status()
    if start:
        _start_capture(int(device), stride, int(cap_seconds))
        return True
    return False


def _render_live(run: dict) -> None:
    import pandas as pd

    step = run.get("last_step") or {}
    elapsed = max(0.0, time.time() - run["started"])
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Measured FPS", f"{step.get('fps', 0.0):.1f}",
              help="Processed frames per second — measured over the last "
                   "batch, not estimated.")
    f2.metric("Frame latency", f"{step.get('latency_ms', 0.0):.0f} ms",
              help="Mean per-frame processing time in the last batch "
                   "(real timing).")
    f3.metric("Frames processed", run["frames"])
    f4.metric("Elapsed", f"{elapsed:.0f} / {run['max_seconds']} s")

    if run.get("frame_rgb") is not None:
        last = (run.get("last_step") or {}).get("last")
        caption = f"last processed frame · {last['count']} boxes this frame"
        if state.settings().get("show_density", True):
            caption += " · density overlay on"
        st.image(run["frame_rgb"], caption=caption, width="stretch")

    st.markdown(f"#### Rolling window ({_WINDOW} frames)")
    window = run.get("last_window")
    if window is None:
        st.caption(f"Collecting frames — the first window completes at "
                   f"{_WINDOW} processed frames "
                   f"({run['frames']}/{_WINDOW}).")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Persons (window)", f"{window['person_count']:.1f}",
                  help="Mean detector boxes per frame over the window.")
        m2.metric("Movement", f"{window['mean_speed_pxf']:.2f} px/f",
                  help="Mean tracked-center speed in pixels per frame.")
        m3.metric("Directional consistency",
                  f"{window['directional_consistency']:.2f}",
                  help="1 = all one direction, 0 = canceling flows.")
        score = run.get("score")
        if score is None:
            m4.metric("Anomaly score", "—",
                      help="No trained bundle available or scoring failed — "
                           "the value is omitted rather than zeroed.")
        else:
            m4.metric("Anomaly score", f"{score:.3f}",
                      help="Model output on the rolling window's features — "
                           "not a verified incident label.")

    observations = run.get("observations", [])
    if len(observations) >= 2:
        df = pd.DataFrame(observations)
        with ui.card("Observation timeline",
                     hint=f"Last {len(observations)} window evaluations "
                          f"(real values; window = {_WINDOW} frames)."):
            c1, c2 = st.columns(2)
            with c1:
                st.altair_chart(timeline_series(
                    df, x="t", y="person_count",
                    title="Persons (rolling window)", y_label="persons",
                    unit="persons", x_label="s since start",
                ), width="stretch")
            with c2:
                st.altair_chart(timeline_series(
                    df, x="t", y="speed",
                    title="Movement (rolling window)", y_label="speed",
                    unit="px/frame", x_label="s since start",
                ), width="stretch")
            if df["score"].notna().any():
                st.altair_chart(timeline_series(
                    df, x="t", y="score",
                    title="Anomaly score (rolling window)", y_label="score",
                    unit="score (0–1)", x_label="s since start",
                    y_domain=[0.0, 1.0],
                ), width="stretch")
    else:
        ui.data_note("The observation timeline appears after the second "
                     "window evaluation.")

    frac = min(1.0, elapsed / max(1, run["max_seconds"]))
    st.progress(
        frac,
        text=f"Capture {frac * 100:.0f}% of the {run['max_seconds']} s "
             "budget · Stop takes effect at the next batch boundary.")


def _render_final(run: dict) -> None:
    if run["phase"] == "error":
        ui.error_state("Camera unavailable", detail=run.get("error"),
                       retry_label="Try again", on_click=_restart_camera,
                       key="camera_error")
        return
    ui.empty_state(
        "Capture stopped",
        f"{run.get('reason', 'capture ended')}. Live panels are cleared — "
        "the values below are the last real observation; start again for "
        "live numbers.")
    with ui.card("Final observation",
                 hint="Real values from the last completed rolling window — "
                      "kept, never fabricated."):
        window = run.get("last_window")
        if window is None:
            st.caption(f"Capture ended before a full window completed "
                       f"({run['frames']} frames) — no window metrics to "
                       "show.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Persons (window)", f"{window['person_count']:.1f}")
            m2.metric("Movement", f"{window['mean_speed_pxf']:.2f} px/f")
            m3.metric("Directional consistency",
                      f"{window['directional_consistency']:.2f}")
            score = run.get("score")
            m4.metric("Anomaly score", f"{score:.3f}" if score is not None
                      else "—")
        if run.get("frame_rgb") is not None:
            st.image(run["frame_rgb"], caption="last processed frame",
                     width="stretch")
    s1, s2, s3 = st.columns(3)
    s1.metric("Frames processed", run["frames"])
    s2.metric("Capture duration", f"{run.get('duration', 0.0):.0f} s")
    s3.metric("Average throughput",
              f"{run['frames'] / max(0.001, run.get('duration', 0.0)):.1f} "
              "fps")
    _render_model_status()
    st.button("Start again", type="primary", on_click=_restart_camera,
              key="camera_restart")


# --------------------------------------------------------------------------
# page entrypoint
# --------------------------------------------------------------------------

def render() -> None:
    ui.page_header(
        eyebrow="Workspace · webcam",
        title="Live Camera",
        description=(
            "Rolling 30-frame window of density and movement features from "
            "a webcam. Capture runs in small batches so Stop works between "
            "batches; FPS and latency are measured, not estimated."),
    )

    run = st.session_state.get("camera_run")
    phase = run.get("phase") if run else None

    if phase is None:
        if _render_config():
            run = st.session_state.get("camera_run")
            phase = run.get("phase") if run else None

    if phase == "processing":
        _capture_fragment()
        run = st.session_state["camera_run"]
        phase = run["phase"]
        if phase == "processing":
            return  # timer-driven fragment reruns continue the batches

    if phase in ("stopped", "ended", "error"):
        _render_final(st.session_state["camera_run"])
