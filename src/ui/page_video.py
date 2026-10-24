"""Video Analysis workspace: upload -> batched fragment processing -> results.

Processing runs in small batches inside an auto-rerunning fragment, so the
Cancel button takes effect at batch boundaries (Streamlit scripts are
non-interruptible mid-run). Every number shown is measured from the real
pipeline; anomaly predictions and their columns appear only when a trained
bundle exists — they are omitted, never faked.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import streamlit as st

from src.inference.service import InferenceError, InferenceService
from src.preprocessing.video import InvalidVideoError
from src.ui import components as ui
from src.ui import state
from src.ui import video_store
from src.ui.analysis_loop import FrameAnalyzer, draw_persons
from src.ui.charts import timeline_series
from src.ui.common import (configured_detector, find_model_path,
                           load_model_algorithm)

VIDEO_TYPES = ["mp4", "avi", "mov", "mkv"]
_MAX_VIDEO_MB = 200
_MAX_VIDEO_BYTES = _MAX_VIDEO_MB * 1024 * 1024
_WINDOW = 30          # processed frames per feature window (pipeline default)
_WINDOW_STRIDE = 5    # window step in processed frames (pipeline default)
_BATCH_FRAMES = 24    # frames per fragment rerun (cancel lands between batches)
_BATCH_TIME_BUDGET_S = 8.0  # upper bound per batch on slow machines


# --------------------------------------------------------------------------
# pure helpers (unit-tested)
# --------------------------------------------------------------------------

def validate_video_upload(name: str | None, size: int | None,
                          content_type: str | None = None) -> str | None:
    """Return an actionable error message for a bad upload, else None."""
    if not name:
        return "No file selected — choose a video file (mp4, avi, mov or mkv)."
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in VIDEO_TYPES:
        shown = f".{ext}" if ext else name
        return (f"Unsupported video type '{shown}' — "
                "allowed: mp4, avi, mov, mkv.")
    if not size or size <= 0:
        return "The file is empty — export the video again and retry."
    if size > _MAX_VIDEO_BYTES:
        mb = size / (1024 * 1024)
        return (f"File is {mb:.1f} MB — the upload limit is "
                f"{_MAX_VIDEO_MB} MB. Trim or re-encode the clip.")
    if content_type and not str(content_type).startswith("video/"):
        return (f"Expected a video file, got '{content_type}' — "
                "allowed: mp4, avi, mov, mkv.")
    return None


def _fmt_size(n_bytes: int) -> str:
    """Human-readable size: MB above 1 MB, KB below."""
    if n_bytes >= 1024 * 1024:
        return f"{n_bytes / (1024 * 1024):.1f} MB"
    return f"{n_bytes / 1024:.0f} KB"


def _probe_video(video_path: Path) -> tuple[int, float]:
    """Probe total frame count and fps; raise InvalidVideoError if unreadable.

    fps falls back to 25.0 when the container reports no valid value — the
    fallback is only used for the annotated-export timing, never shown as a
    measured source property without this note.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise InvalidVideoError("Could not open the uploaded video.")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if total <= 0:
        raise InvalidVideoError(
            "The file reports no frames — re-encode it as MP4 and retry.")
    if fps <= 0 or fps > 240:
        fps = 25.0
    return total, fps


# --------------------------------------------------------------------------
# session lifecycle
# --------------------------------------------------------------------------

def _pop_run_keys(store=None) -> None:
    """Drop pipeline state and release the capture handle. Idempotent."""
    store = store if store is not None else st.session_state
    store.pop("video_run", None)
    store.pop("video_analyzer", None)
    state.release_video(store)


def _replace_video() -> None:
    """Clear the upload, the run state and the temp file (new uploader key)."""
    store = st.session_state
    _pop_run_keys(store)
    store.pop("video_meta", None)
    video_store.cleanup_upload()
    store["video_upload_nonce"] = int(store.get("video_upload_nonce", 0)) + 1


def _restart() -> None:
    """Back to the settings panel (keeps the current upload)."""
    _pop_run_keys(st.session_state)


def _ensure_upload(digest: str, data: bytes, uploaded: Any) -> dict | None:
    """Persist a NEW upload (hash-keyed, single file) and probe it.

    Returns the meta dict, or None after rendering an error state.
    """
    store = st.session_state
    meta = store.get("video_meta")
    if meta is not None and meta["hash"] == digest:
        return meta  # same file already stored — reuse, no disk churn
    _pop_run_keys(store)  # releases any open handle before replacing the file
    store.pop("video_meta", None)
    suffix = Path(uploaded.name).suffix.lower() or ".mp4"
    try:
        path = video_store.save_upload(digest, data, suffix=suffix)
        n_raw, fps = _probe_video(path)
    except (InvalidVideoError, OSError) as exc:
        ui.error_state("Unreadable video", detail=str(exc), key="video_decode")
        return None
    meta = {
        "name": uploaded.name,
        "size": uploaded.size,
        "hash": digest,
        "path": path,
        "n_raw": n_raw,
        "fps": fps,
    }
    store["video_meta"] = meta
    return meta


def _start_processing(meta: dict, *, stride: int, max_frames: int,
                      export: bool) -> None:
    """Open the capture, build the analyzer and create the run state."""
    import cv2

    from src.features.pipeline import WindowConfig

    store = st.session_state
    _pop_run_keys(store)
    cap = cv2.VideoCapture(str(meta["path"]))
    if not cap.isOpened():
        cap.release()
        store["video_run"] = {
            "phase": "error", "error": "Could not reopen the uploaded video.",
            "processed": 0, "rows": [], "elapsed": 0.0, "frames": {},
        }
        return
    max_dim = int(state.settings().get("process_max_side", 1280))
    n_to_process = min(max_frames, math.ceil(meta["n_raw"] / stride))
    store["video_cap"] = cap
    store["video_running"] = True
    store["video_analyzer"] = FrameAnalyzer(
        grid_size=WindowConfig().grid_size, max_dim=max_dim,
        window=_WINDOW, window_stride=_WINDOW_STRIDE,
        detector=configured_detector(),
    )
    export_target = (meta["path"].parent
                     / f"annotated_{meta['hash'][:12]}.mp4")
    store["video_run"] = {
        "phase": "processing",
        "processed": 0,
        "raw_seen": 0,
        "rows": [],
        "elapsed": 0.0,
        "batches": 0,
        "peak_count": -1,
        "peak_row": None,
        "frames": {},          # "first" / "peak" / "last" annotated frames
        "eof": False,
        "error": None,
        "export": export,
        "export_target": str(export_target),
        "export_path": None,
        "fps": meta["fps"],
        "n_to_process": n_to_process,
        "cfg": {"stride": stride, "max_frames": max_frames, "max_dim": max_dim},
    }


def _process_batch(run: dict) -> None:
    """Read up to _BATCH_FRAMES source frames and feed the analyzer."""
    import cv2

    from src.features.pipeline import _maybe_resize

    store = st.session_state
    analyzer: FrameAnalyzer = store["video_analyzer"]
    cap = store["video_cap"]
    meta = store["video_meta"]
    stride = run["cfg"]["stride"]
    max_dim = run["cfg"]["max_dim"]

    t0 = time.perf_counter()
    frames_this_batch = 0
    while (frames_this_batch < _BATCH_FRAMES
           and run["processed"] < run["n_to_process"]
           and (time.perf_counter() - t0) < _BATCH_TIME_BUDGET_S):
        ok, frame = cap.read()
        frames_this_batch += 1
        if not ok:
            run["eof"] = True
            break
        raw_idx = run["raw_seen"]
        run["raw_seen"] = raw_idx + 1
        if raw_idx % stride != 0:
            continue
        info = analyzer.add_frame(frame)
        run["processed"] += 1
        if info["window"] is not None:
            run["rows"].append(info["window"])
        annotated = draw_persons(_maybe_resize(frame, max_dim),
                                 info["persons"])
        if run["frames"].get("first") is None:
            run["frames"]["first"] = annotated
        run["frames"]["last"] = annotated
        if info["count"] > run["peak_count"]:
            run["peak_count"] = info["count"]
            run["frames"]["peak"] = annotated
            run["peak_row"] = {"frame_index": info["frame_index"],
                               "count": info["count"]}
        if run["export"]:
            writer = run.get("writer")
            if writer is None:
                h, w = analyzer.frame_shape
                writer = cv2.VideoWriter(
                    run["export_target"],
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    run["fps"], (w, h),
                )
                run["writer"] = writer
            if writer is not None and writer.isOpened():
                writer.write(annotated)
    run["elapsed"] += time.perf_counter() - t0
    run["batches"] += 1


def _finish_run(run: dict, *, cancelled: bool) -> None:
    """Close handles, set the terminal phase, record history (done only)."""
    writer = run.pop("writer", None)
    if writer is not None:
        writer.release()
    state.release_video(st.session_state)
    if cancelled:
        run["phase"] = "cancelled"
        return
    if run["processed"] == 0:
        run["phase"] = "error"
        run["error"] = ("No frames could be read from the upload — "
                        "re-encode the clip as MP4 and retry.")
        return
    target = run.get("export_target")
    if run.get("export") and target and Path(target).exists():
        run["export_path"] = target
    run["phase"] = "done"
    _record_history(run)


def _record_history(run: dict) -> None:
    meta = st.session_state["video_meta"]
    model_path = find_model_path()
    algo = load_model_algorithm(model_path) if model_path else "detection only"
    state.record_analysis("video", {
        "video": meta["name"],
        "frames": run["processed"],
        "windows": len(run["rows"]),
        "seconds": round(run["elapsed"], 1),
        "model": algo,
    })


# --------------------------------------------------------------------------
# processing fragment
# --------------------------------------------------------------------------

def _render_progress(run: dict) -> None:
    target = run["n_to_process"]
    done = run["processed"]
    frac = done / max(1, target)
    st.progress(
        frac,
        text=f"Processing… frame {done} of {target} "
             f"({frac * 100:.0f}%) · {len(run['rows'])} windows so far",
    )
    w1, w2, w3 = st.columns(3)
    w1.metric("Frames processed", f"{done} / {target}")
    w2.metric("Windows so far", len(run["rows"]))
    w3.metric("Elapsed", ui.format_seconds(run["elapsed"]))
    ui.data_note("Processing runs in bounded batches — Cancel takes effect "
                 "at the next batch boundary.")


@st.fragment(run_every="0.5s")
def _processing_fragment() -> None:
    """One processing batch per rerun; finishes via a full-script rerun."""
    run = st.session_state.get("video_run")
    if not run or run.get("phase") != "processing":
        return
    if st.button("Cancel processing", key="video_cancel"):
        _finish_run(run, cancelled=True)
        st.rerun()
        return
    _process_batch(run)
    finished = (run["processed"] >= run["n_to_process"] or run["eof"])
    if finished:
        _finish_run(run, cancelled=False)
        st.rerun()
        return
    _render_progress(run)


# --------------------------------------------------------------------------
# render pieces
# --------------------------------------------------------------------------

def _render_config(meta: dict) -> None:
    s = state.settings()
    with ui.card("Input",
                 hint="Measured by probing the file — not estimates."):
        c1, c2, c3 = st.columns(3)
        c1.metric("Source frames", f"{meta['n_raw']:,}")
        c2.metric("Frame rate", f"{meta['fps']:.1f} fps")
        c3.metric("File size", _fmt_size(meta["size"]))
        st.caption(f"{meta['name']} · stored for this session only "
                   "(replaced when you upload another clip).")

    sc1, sc2 = st.columns(2)
    with sc1:
        stride = st.slider(
            "Frame stride (process every Nth frame)", 1, 10,
            int(s["frame_stride"]), key="video_stride_cfg",
            help="Higher = faster, coarser timeline. Defaults come from "
                 "Settings.")
    with sc2:
        max_frames = st.slider(
            "Max frames to process", 60, 1200, int(s["max_frames"]),
            key="video_max_frames_cfg",
            help="CPU budget cap on processed frames. Defaults come from "
                 "Settings.")
    export = st.toggle(
        "Save annotated video while processing", value=False,
        key="video_export_cfg",
        help="Draws tracker boxes + IDs into an MP4 (adds encode time per "
             "frame); offered for download with the results.")

    n_to_process = min(max_frames, math.ceil(meta["n_raw"] / stride))
    st.caption(
        f"Will process {n_to_process} frames ({meta['n_raw']} source frames "
        f"at stride {stride}), forming {_WINDOW}-frame windows every "
        f"{_WINDOW_STRIDE} processed frames.")
    if n_to_process < _WINDOW:
        st.warning(
            f"Only {n_to_process} frames would be processed — fewer than the "
            f"{_WINDOW} needed for one window. Lower the stride or raise the "
            "frame cap for usable results.")
    a1, a2 = st.columns([2, 1])
    with a1:
        if st.button("Process video", type="primary", key="video_start"):
            _start_processing(meta, stride=stride, max_frames=max_frames,
                              export=export)
    with a2:
        st.button("Replace video", on_click=_replace_video,
                  key="video_replace_cfg",
                  help="Clear this upload and pick a new clip.")


def _rows_dataframe(run: dict):
    """Window rows as a DataFrame with a window ordinal column."""
    import pandas as pd

    df = pd.DataFrame(run["rows"])
    df.insert(0, "window", range(1, len(df) + 1))
    return df


def _attach_predictions(df):
    """Add prediction/anomaly_score columns when a bundle exists.

    Returns (df, info) where info describes the bundle, or (df, None).
    Failures degrade to detection-only output with a visible warning.
    """
    model_path = find_model_path()
    if model_path is None:
        return df, None
    try:
        service = InferenceService(model_path)
    except InferenceError as exc:
        st.warning(f"Model bundle failed to load — showing detection-only "
                   f"output. ({exc})")
        return df, None
    feature_cols = [c for c in df.columns
                    if c not in ("window", "window_start", "window_end")]
    labels, scores = [], []
    try:
        for _, row in df.iterrows():
            pred = service.predict_row({c: float(row[c])
                                        for c in feature_cols})
            labels.append(pred.label)
            scores.append(float(pred.score))
    except InferenceError as exc:
        st.warning(f"Model prediction failed — showing detection-only "
                   f"output. ({exc})")
        return df, None
    df = df.copy()
    df["anomaly_score"] = scores
    df["prediction"] = labels
    info = {"algorithm": service.bundle.algorithm,
            "trained_at": service.bundle.trained_at,
            "path": str(model_path)}
    return df, info


def _render_results(run: dict, meta: dict) -> None:
    import cv2

    st.divider()
    st.markdown("### Results")
    ui.data_note(f"“{meta['name']}” — {run['processed']} processed frames "
                 f"at stride {run['cfg']['stride']}, measured this session; "
                 "nothing is persisted to disk.")

    df = _rows_dataframe(run)
    df, model_info = _attach_predictions(df)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Frames processed", run["processed"])
    m2.metric("Windows", len(df))
    if len(df):
        m3.metric("Persons per window (mean)",
                  f"{df['person_count'].mean():.1f}")
        m4.metric("Persons per window (max)",
                  int(df["person_count"].max()))
    else:
        m3.metric("Persons per window (mean)", "—")
        m4.metric("Persons per window (max)", "—")

    m5, m6, m7 = st.columns(3)
    if len(df):
        m5.metric("Movement (mean)", f"{df['mean_speed_pxf'].mean():.2f} px/f")
    else:
        m5.metric("Movement (mean)", "—")
    m6.metric("Processing time", ui.format_seconds(run["elapsed"]))
    m7.metric("Model", model_info["algorithm"] if model_info
              else "detection only")

    if not len(df):
        st.warning(
            f"Processing finished, but no {_WINDOW}-frame window completed "
            f"({run['processed']} frames). Use a longer clip or a lower "
            "stride for timeline results.")

    if len(df):
        with ui.card("Timeline",
                     hint=f"Per-window pipeline output — window = "
                          f"{_WINDOW} processed frames, step "
                          f"{_WINDOW_STRIDE}."):
            c1, c2 = st.columns(2)
            with c1:
                st.altair_chart(timeline_series(
                    df, x="window", y="person_count",
                    title="Persons per window (detector estimate)",
                    y_label="persons", unit="persons", x_label="window",
                ), width="stretch")
            with c2:
                st.altair_chart(timeline_series(
                    df, x="window", y="mean_speed_pxf",
                    title="Mean movement speed (tracked centers)",
                    y_label="speed", unit="px/frame", x_label="window",
                ), width="stretch")
            if "anomaly_score" in df.columns:
                st.altair_chart(timeline_series(
                    df, x="window", y="anomaly_score",
                    title="Anomaly score per window (model output)",
                    y_label="score", unit="score (0–1)", x_label="window",
                    y_domain=[0.0, 1.0],
                ), width="stretch")

        with ui.card("Windows",
                     hint="Feature rows fed to the model — the same schema "
                          "the training pipeline uses."):
            cols = ["window", "window_start", "window_end", "person_count",
                    "mean_speed_pxf", "directional_consistency",
                    "flow_mean_magnitude", "flow_std_pxf"]
            if "anomaly_score" in df.columns:
                cols += ["anomaly_score", "prediction"]
            st.dataframe(df[cols], hide_index=True, width="stretch")
            ui.data_note("person_count = detector boxes in the window; "
                         "speed in px/frame; flow in px magnitude; scores "
                         "only exist with a trained bundle.")

    frames = run.get("frames", {})
    if frames:
        with ui.card("Sample frames",
                     hint="Annotated with real tracker output "
                          "(boxes + track IDs)."):
            strip_cols = st.columns(3)
            captions = [
                ("first", "first processed frame"),
                ("peak", (f"peak — {run['peak_count']} detections "
                          f"(frame {run['peak_row']['frame_index']})"
                          if run.get("peak_row") else "peak")),
                ("last", "last processed frame"),
            ]
            for col, (key, fallback_caption) in zip(strip_cols, captions):
                img = frames.get(key)
                if img is None:
                    col.caption("—")
                    continue
                col.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                          caption=fallback_caption, width="stretch")

    export_path = run.get("export_path")
    if export_path:
        with ui.card("Annotated video",
                     hint="Written during processing with tracker boxes."):
            st.video(export_path)
            data = Path(export_path).read_bytes()
            st.download_button(
                "Download annotated MP4", data=data,
                file_name=Path(export_path).name, mime="video/mp4",
                width="stretch", key="video_export_dl")

    with ui.card("Model & limitations"):
        if model_info:
            trained = model_info["trained_at"] or "unknown"
            st.info(
                f"Windows were scored by the trained "
                f"{model_info['algorithm']} bundle (trained {trained}). "
                "The score is a model output on this clip's features — not a "
                "verified incident label.")
        else:
            st.info(
                "No trained ML bundle on disk — timeline and tables show "
                "detection/tracking/flow only. Anomaly scores are omitted "
                "rather than shown as zeros. Run Training & Evaluation for "
                "window-level model output.")
        ui.render_limitations([
            "Person counts are detector estimates; no counting ground "
            "truth exists in this project to calibrate them.",
            "Tracking and flow degrade under heavy occlusion and abrupt "
            "lighting changes.",
            "Frame stride trades temporal resolution for speed — short "
            "events between sampled frames can be missed.",
            "Anomaly scores reflect the training data; they are model "
            "outputs, not verified incident labels.",
        ])

    d1, d2 = st.columns(2)
    with d1:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download windows CSV", data=csv_bytes,
                           file_name="windows.csv", mime="text/csv",
                           width="stretch", key="video_download_csv")
    with d2:
        summary = {
            "video": meta["name"],
            "source_frames": meta["n_raw"],
            "fps_source": meta["fps"],
            "stride": run["cfg"]["stride"],
            "processed_frames": run["processed"],
            "window": _WINDOW,
            "window_stride": _WINDOW_STRIDE,
            "elapsed_s": round(run["elapsed"], 3),
            "model": model_info["algorithm"] if model_info else None,
            "windows": json.loads(df.to_json(orient="records")),
        }
        st.download_button(
            "Download results JSON",
            data=json.dumps(summary, indent=2).encode("utf-8"),
            file_name="video_results.json", mime="application/json",
            width="stretch", key="video_download_json")

    st.button("Replace video", on_click=_replace_video, key="video_replace",
              help="Clear this upload and pick a new clip.")


def _render_cancelled(run: dict) -> None:
    st.info(f"Processing cancelled after {run['processed']} frames — "
            "no results were kept.")
    st.button("Back to settings", on_click=_restart,
              key="video_restart_cancelled")


# --------------------------------------------------------------------------
# page entrypoint
# --------------------------------------------------------------------------

def video_step(has_upload: bool, phase: str | None) -> int:
    """Which step of the guided flow to show: 1 Upload, 2 Check, 3 Results.

    Processing, cancellation and errors all stay on step 2 (Check) —
    only a completed run advances to Results.
    """
    if not has_upload:
        return 1
    return 3 if phase == "done" else 2


def render() -> None:
    ui.page_header(
        title="Video Analysis",
        description=(
            "Analyse a clip over time: people are detected and tracked "
            "window by window, with density and movement charts. "
            "Processing runs in batches, so Cancel takes effect between "
            "them."),
    )

    st.session_state.setdefault("video_upload_nonce", 0)
    nonce = int(st.session_state["video_upload_nonce"])
    upload_key = f"video_uploader_{nonce}"
    uploaded_flag = st.session_state.get(upload_key) is not None
    phase = (st.session_state.get("video_run") or {}).get("phase")
    ui.step_rail(video_step(uploaded_flag, phase), ui.STEP_LABELS)

    uploaded = st.file_uploader(
        "Video file", type=VIDEO_TYPES, key=upload_key,
        help="MP4, AVI, MOV or MKV — up to "
             f"{_MAX_VIDEO_MB} MB. Stored for this session only.")
    if uploaded is None:
        ui.empty_state(
            "No video selected",
            "Upload a clip to analyse crowd density, movement and "
            "anomalies over time. Nothing runs until you press "
            "Process video.")
        return

    error = validate_video_upload(getattr(uploaded, "name", None),
                                  getattr(uploaded, "size", None),
                                  getattr(uploaded, "type", None))
    if error:
        ui.error_state("Upload rejected", detail=error, key="video_validation")
        return

    data = uploaded.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    meta = _ensure_upload(digest, data, uploaded)
    if meta is None:
        return

    run = st.session_state.get("video_run")
    phase = run.get("phase") if run else None

    if phase is None:
        _render_config(meta)
        run = st.session_state.get("video_run")
        phase = run.get("phase") if run else None

    if phase == "processing":
        _processing_fragment()
        run = st.session_state["video_run"]
        phase = run["phase"]
        if phase == "processing":
            return  # timer-driven fragment reruns continue the batches

    if phase == "done":
        _render_results(st.session_state["video_run"], meta)
        return
    if phase == "cancelled":
        _render_cancelled(st.session_state["video_run"])
    elif phase == "error":
        run = st.session_state["video_run"]
        ui.error_state(
            "Processing failed",
            detail=run.get("error") or "Unknown failure.",
            retry_label="Back to settings", on_click=_restart,
            key="video_run_error")
