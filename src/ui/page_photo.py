"""Photo Analysis: single-image detection, density grid, static features only.

Temporal/movement outputs are intentionally absent here — they need a video window.
"""

from __future__ import annotations

import streamlit as st

from src.detection.analyze import analyze_image
from src.features.schema import FeatureSchema
from src.features.static import static_features
from src.ui.common import find_model_path, load_model_algorithm

_MAX_SIDE = 1280


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
    import numpy as np

    data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        st.error("Could not decode image — upload a valid JPG/PNG.")
        return
    h, w = image.shape[:2]
    scale = min(1.0, _MAX_SIDE / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    with st.spinner("Detecting persons (YOLOv8n, CPU)..."):
        analysis = analyze_image(image)
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
    feats = static_features(analysis.detections, image.shape, grid_size=4)
    st.table({"feature": list(feats), "value": [round(v, 4) for v in feats.values()]})
    mp = find_model_path()
    if mp:
        algo = load_model_algorithm(mp)
        st.info(
            f"A trained {algo} bundle exists, but it consumes all 15 features "
            "including movement — a single photo cannot supply them. "
            "Use Video Analysis for model output."
        )
