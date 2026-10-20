"""Photo Analysis: single-image detection, density grid, static features only.

Temporal/movement outputs are intentionally absent here — they need a video window.
"""

from __future__ import annotations

import streamlit as st

from src.detection.analyze import analyze_frame
from src.features.schema import FeatureSchema
from src.features.static import static_features
from src.ui.common import find_model_path, load_model_algorithm

_MAX_SIDE = 1280


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


def render() -> None:
    st.header("Photo Analysis")
    st.caption(
        "Static analysis of one frame. Movement/temporal outputs require "
        "Video Analysis — a single image has no tracks or flow."
    )
    uploaded = st.file_uploader("Image", type=["jpg", "jpeg", "png", "tif", "bmp"])
    if uploaded is None:
        return
    import cv2

    try:
        image = _decode_upload(uploaded.getvalue())
    except ValueError:
        st.error("Could not decode image — upload a valid JPG/PNG.")
        return

    with st.spinner("Detecting persons (YOLOv8n, CPU)..."):
        analysis = analyze_frame(image)
    left, right = st.columns(2)
    with left:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="original", width="content")
    with right:
        st.image(
            cv2.cvtColor(analysis.annotated, cv2.COLOR_BGR2RGB),
            caption=f"annotated — {analysis.count} persons", width="content",
        )
    st.metric("Detected persons (detector estimate — no counting GT exists)", analysis.count)
    st.subheader("Density grid")
    st.dataframe(analysis.grid_counts, use_container_width=True)
    st.subheader("Static features")
    import numpy as np

    dets = analysis.detection_result.detections
    centers = np.array([d.center for d in dets], dtype=np.float64).reshape(-1, 2)
    feats = static_features(
        analysis.count, analysis.grid_counts, centers, image.shape[:2], grid_size=4,
    )
    st.table({"feature": list(feats), "value": [round(v, 4) for v in feats.values()]})
    mp = find_model_path()
    if mp:
        algo = load_model_algorithm(mp)
        st.info(
            f"A trained {algo} bundle exists, but it consumes all 15 features "
            "including movement — a single photo cannot supply them. "
            "Use Video Analysis for model output."
        )
