"""Photo Analysis workspace: upload -> validate -> explicit analyse -> results.

Static analysis only: detection, spatial density and static features.
Temporal/movement outputs are intentionally absent — they need a video
window. Analysis runs only when the user presses "Analyse image"; results
are stored in session state and labelled with their provenance.
"""

from __future__ import annotations

import time

import streamlit as st

from src.detection.analyze import analyze_frame
from src.ui import components as ui
from src.ui import state
from src.ui.common import (configured_detector, find_model_path,
                           load_model_algorithm)

_MAX_SIDE = 1280
_ALLOWED_EXTS = {"jpg", "jpeg", "png", "tif", "bmp"}
_MAX_UPLOAD_MB = 20
_MAX_UPLOAD_BYTES = _MAX_UPLOAD_MB * 1024 * 1024


# --------------------------------------------------------------------------
# pure helpers (unit-tested)
# --------------------------------------------------------------------------

def validate_upload(name: str | None, size: int | None,
                    content_type: str | None = None) -> str | None:
    """Return an actionable error message for a bad upload, else None."""
    if not name:
        return ("No file selected — choose an image file "
                "(JPG, JPEG, PNG, TIF or BMP).")
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in _ALLOWED_EXTS:
        shown = f".{ext}" if ext else name
        return (f"Unsupported file type '{shown}' — "
                "allowed: JPG, JPEG, PNG, TIF, BMP.")
    if not size or size <= 0:
        return "The file is empty — export the image again and retry."
    if size > _MAX_UPLOAD_BYTES:
        mb = size / (1024 * 1024)
        return (f"File is {mb:.1f} MB — the upload limit is "
                f"{_MAX_UPLOAD_MB} MB. Resize or re-compress the image.")
    if content_type and not str(content_type).startswith("image/"):
        return (f"Expected an image file, got '{content_type}' — "
                "allowed: JPG, JPEG, PNG, TIF, BMP.")
    return None


def _decode_upload(data: bytes):
    """Decode uploaded image bytes to a BGR frame (resized to _MAX_SIDE)."""
    import cv2
    import numpy as np

    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")
    h, w = image.shape[:2]
    scale = min(1.0, _MAX_SIDE / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    return image


def analyze_upload(data: bytes, detector=None, grid_size: int = 4) -> tuple:
    """Decode uploaded image bytes and analyze the in-memory frame.

    Returns ``(image, analysis)``. Uploads never touch disk, so this uses
    analyze_frame — analyze_image is the disk-path variant.
    """
    image = _decode_upload(data)
    return image, analyze_frame(image, detector=detector, grid_size=grid_size)


# --------------------------------------------------------------------------
# session callbacks
# --------------------------------------------------------------------------

def _reset_analysis() -> None:
    """Clear the stored analysis (keeps the current upload)."""
    st.session_state.pop("photo_result", None)


def _replace_image() -> None:
    """Clear the analysis AND the uploader (a new widget key empties it)."""
    st.session_state.pop("photo_result", None)
    st.session_state["photo_upload_nonce"] = (
        int(st.session_state.get("photo_upload_nonce", 0)) + 1)


# --------------------------------------------------------------------------
# render pieces
# --------------------------------------------------------------------------

def _run_analysis(data: bytes, fingerprint: tuple, name: str) -> dict | None:
    """Analyse with spinner + real timing; store and record the result."""
    with st.spinner("Detecting persons (YOLOv8n, CPU)..."):
        t0 = time.perf_counter()
        try:
            image, analysis = analyze_upload(data,
                                             detector=configured_detector())
        except Exception as exc:
            ui.error_state(
                "Analysis failed",
                detail=f"{type(exc).__name__}: {exc} — check that the YOLO "
                       "weights are present under models/ and retry.",
                key="photo_run_error")
            return None
    elapsed = time.perf_counter() - t0
    result = {
        "fingerprint": fingerprint,
        "image": image,
        "analysis": analysis,
        "elapsed": elapsed,
        "name": name,
    }
    st.session_state["photo_result"] = result
    state.record_analysis("photo", {
        "persons": analysis.count,
        "model": analysis.model_name,
        "seconds": round(elapsed, 2),
    })
    return result


def _render_metric_row(analysis, elapsed: float) -> None:
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        ui.metric_tile(
            "Persons detected", analysis.count,
            help="Detector estimate from YOLO boxes — not calibrated against "
                 "counting ground truth (no counting dataset in this project).")
    with m2:
        ui.metric_tile(
            "Detector inference", ui.format_seconds(analysis.inference_time_s),
            help="Measured wall-clock time of the detection pass on this image.")
    with m3:
        ui.metric_tile(
            "Total processing", ui.format_seconds(elapsed),
            help="Decode + detection + feature extraction for this upload "
                 "(measured, not a model estimate).")
    with m4:
        ui.metric_tile(
            "Detector model", analysis.model_name,
            help="Detection weights used for this run.")


def _render_detections(analysis) -> None:
    dets = analysis.detection_result.detections
    with ui.card("Detections",
                 hint="Per-box detector output — confidence gated by the "
                      "detector's confidence threshold."):
        if not dets:
            ui.empty_state(
                "No persons detected",
                "The detector found no boxes above its confidence "
                "threshold in this image.")
            return
        confs = [d.conf for d in dets]
        c1, c2, c3 = st.columns(3)
        c1.metric("Boxes", len(dets))
        c2.metric("Mean confidence", ui.format_pct(sum(confs) / len(confs)))
        c3.metric("Min / max confidence",
                  f"{ui.format_pct(min(confs))} / {ui.format_pct(max(confs))}")
        import pandas as pd

        rows = [
            [i + 1, ui.format_pct(d.conf),
             f"({int(d.x1)}, {int(d.y1)}) – ({int(d.x2)}, {int(d.y2)})",
             f"({d.center[0]:.0f}, {d.center[1]:.0f})"]
            for i, d in enumerate(dets)
        ]
        st.dataframe(
            pd.DataFrame(rows, columns=[
                "#", "Confidence", "Box (x1, y1) – (x2, y2)", "Center (px)"]),
            hide_index=True, width="stretch")
        ui.data_note(f"Coordinates are in resized-image space "
                     f"(longest side capped at {_MAX_SIDE} px); confidence is "
                     "a model output, not a calibrated probability.")


def _render_density_grid(analysis) -> None:
    """Heatmap of persons per grid cell (only when show_density is on)."""
    import altair as alt
    import pandas as pd

    from src.ui.theme import TOKENS

    grid = analysis.grid_counts
    size = grid.shape[0]
    rows = [
        {"row": str(r), "col": str(c), "persons": int(grid[r, c])}
        for r in range(size) for c in range(size)
    ]
    df = pd.DataFrame(rows)
    peak = max(1, int(df["persons"].max()))
    chart = alt.Chart(df).mark_rect().encode(
        x=alt.X("col:O", title="grid column (left → right)",
                axis=alt.Axis(labelAngle=0)),
        y=alt.Y("row:O", title="grid row (top → bottom)", sort="descending"),
        color=alt.Color(
            "persons:Q", title="persons / cell",
            scale=alt.Scale(domain=[0, peak],
                            range=[TOKENS["surface_2"], TOKENS["accent"]]),
            legend=alt.Legend(direction="horizontal", gradientLength=140)),
        tooltip=[alt.Tooltip("row:O", title="row"),
                 alt.Tooltip("col:O", title="col"),
                 alt.Tooltip("persons:Q", title="persons")],
    ).properties(height=260, title="Density grid — detector estimate per cell")
    st.altair_chart(chart, width="stretch")
    ui.data_note("Cells count box centers; brighter = denser. Cells are "
                 "relative within this image — no absolute density "
                 "calibration exists (no counting ground truth).")


def _render_static_features(image, analysis) -> None:
    import numpy as np
    import pandas as pd

    from src.features.static import static_features

    dets = analysis.detection_result.detections
    centers = np.array([d.center for d in dets],
                       dtype=np.float64).reshape(-1, 2)
    feats = static_features(
        analysis.count, analysis.grid_counts, centers, image.shape[:2],
        grid_size=analysis.grid_counts.shape[0],
    )
    rows = [[name, f"{value:.4f}"] for name, value in feats.items()]
    st.dataframe(pd.DataFrame(rows, columns=["Feature", "Value"]),
                 hide_index=True, width="stretch")
    ui.data_note("Units: density_mean = mean count per grid cell, "
                 "density_peak = max count per cell, extent/aspect = "
                 "normalised image area. Movement features (flow, speed, "
                 "direction) are undefined for a single frame.")


def _render_model_and_limitations() -> None:
    model_path = find_model_path()
    if model_path:
        algo = load_model_algorithm(model_path)
        st.info(
            f"A trained {algo} bundle exists, but it consumes all 15 "
            "features including movement — a single photo cannot supply "
            "them, so NO model prediction is shown for this input. Use "
            "Video Analysis for window-level model output.")
    else:
        st.info(
            "No trained ML bundle on disk — results above are detection "
            "only. Run Training & Evaluation to enable window-level "
            "model output in Video/Camera analysis.")
    ui.render_limitations([
        "The person count is a detector estimate; no counting ground "
        "truth exists in this project to calibrate it.",
        "A single frame has no temporal context: speed, direction, "
        "tracking and anomaly scores are undefined here.",
        "Perspective and occlusion can bias the estimate; density cells "
        "are relative within this image only.",
        "Confidence values are detector outputs, not calibrated "
        "probabilities.",
    ])


def _render_results(result: dict) -> None:
    import cv2

    image = result["image"]
    analysis = result["analysis"]
    elapsed = result["elapsed"]

    st.divider()
    st.markdown("### Results")
    ui.data_note(f"Analysis of “{result['name']}” — measured on this "
                 "session; nothing below is persisted to disk.")
    _render_metric_row(analysis, elapsed)

    left, right = st.columns(2)
    with left:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                 caption=f"original — {result['name']}",
                 width="stretch")
    with right:
        st.image(cv2.cvtColor(analysis.annotated, cv2.COLOR_BGR2RGB),
                 caption=f"annotated — {analysis.count} persons",
                 width="stretch")

    _render_detections(analysis)

    if state.settings().get("show_density", True):
        with ui.card("Density grid"):
            _render_density_grid(analysis)
    else:
        ui.data_note("Density grid hidden by the density display setting.")

    with ui.card("Static features",
                 hint="Spatial features from this frame — the static part "
                      "of the 15-feature schema."):
        _render_static_features(image, analysis)

    with ui.card("Model & limitations"):
        _render_model_and_limitations()

    st.divider()
    a1, a2, a3 = st.columns(3)
    with a1:
        ok, png = cv2.imencode(".png", analysis.annotated)
        if ok:
            from pathlib import Path

            stem = Path(result["name"]).stem or "photo"
            st.download_button(
                "Download annotated image (PNG)", data=png.tobytes(),
                file_name=f"{stem}_annotated.png", mime="image/png",
                width="stretch", key="photo_export_png")
    with a2:
        st.button("Reset analysis", on_click=_reset_analysis,
                  width="stretch", key="photo_reset",
                  help="Clear results, keep the current image.")
    with a3:
        st.button("Replace image", on_click=_replace_image,
                  width="stretch", key="photo_replace",
                  help="Clear results and the upload — pick a new file.")


# --------------------------------------------------------------------------
# page entrypoint
# --------------------------------------------------------------------------

def render() -> None:
    ui.page_header(
        title="Photo Analysis",
        description=(
            "Static analysis of one image: person detection, spatial "
            "density and static features. Movement, speed, tracking and "
            "anomaly scores require Video Analysis — a single frame has "
            "no temporal context."),
    )

    nonce = int(st.session_state.get("photo_upload_nonce", 0))
    uploaded = st.file_uploader(
        "Image file",
        type=sorted(_ALLOWED_EXTS),
        key=f"photo_uploader_{nonce}",
        help="Drag & drop or browse — JPG, JPEG, PNG, TIF, BMP, up to "
             f"{_MAX_UPLOAD_MB} MB.",
    )
    if uploaded is None:
        ui.empty_state(
            "No image selected",
            "Drag and drop a photograph (JPG, PNG, TIF, BMP) to begin. "
            "Nothing is analysed until you press Analyse image.")
        return

    error = validate_upload(getattr(uploaded, "name", None),
                            getattr(uploaded, "size", None),
                            getattr(uploaded, "type", None))
    if error:
        ui.error_state("Upload rejected", detail=error,
                       key="photo_validation")
        return

    data = uploaded.getvalue()
    try:
        image = _decode_upload(data)
    except ValueError:
        ui.error_state(
            "Could not decode image",
            detail="The file is not a readable image — re-export it as "
                   "JPG or PNG and retry.",
            key="photo_decode")
        return

    fingerprint = (uploaded.name, uploaded.size, hash(data))
    result = st.session_state.get("photo_result")
    if result is not None and result.get("fingerprint") != fingerprint:
        st.session_state.pop("photo_result", None)  # input changed: stale
        result = None

    if result is None:
        with ui.card("Input",
                     hint="Detection runs only when you press "
                          "“Analyse image”."):
            st.image(_rgb(image),
                     caption=f"{uploaded.name} · {len(data) / 1024:.0f} KB · "
                             f"{image.shape[1]}×{image.shape[0]} px",
                     width="stretch")
            if st.button("Analyse image", type="primary", key="photo_run"):
                result = _run_analysis(data, fingerprint, uploaded.name)
        if result is None:
            return

    _render_results(result)


def _rgb(image):
    import cv2

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
