"""Overview: research question, pipeline, dataset status, honest limitations."""

from __future__ import annotations

import streamlit as st

from src.ui.common import RESULTS_DIR, find_model_path, limitations
from src.ui.theme import page_header


def render() -> None:
    page_header(
        "Overview", "🧭",
        "Research question, pipeline, and what this project does not claim",
    )
    st.markdown(
        """
**Research question:** do temporal movement features (speed, direction, optical flow,
track persistence) improve anomaly prediction over static density features alone,
on the UCSD Ped1/Ped2 anomaly benchmark?

**Approach:** YOLOv8n person detection → ByteTrack tracking → per-frame signals
(counts, grid density, Farneback optical flow) → sliding windows (30 frames,
stride 5) → schema-ordered feature rows → grouped-split ML experiments (A: static
only, B: movement only, C: combined, D: algorithm comparison).
"""
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Pipeline")
        st.code(
            "frames → detect → track → flow\n"
            "       → window features\n"
            "       → grouped-split ML\n"
            "       → reports/results/*.json",
            language=None,
        )
    with c2:
        from src.datasets.ucsd_anomaly import UcsdAdapter
        from src.config import load_paths

        try:
            adapter = UcsdAdapter(root=load_paths().ucsd_dir)
            seqs = list(adapter.iter_sequences())
            n_frames = sum(s.n_frames for s in seqs)
            n_anom = sum(int(s.frame_labels.sum()) for s in seqs if s.frame_labels is not None)
            st.subheader("Dataset")
            st.success(
                f"UCSD v1p2 ready — {len(seqs)} sequences, "
                f"{n_frames:,} frames, {n_anom:,} anomalous-frame labels."
            )
        except Exception as exc:
            st.warning(f"UCSD not found — see Dataset Explorer. ({exc})")
    with c3:
        st.subheader("Model")
        mp = find_model_path()
        if mp:
            st.success(f"Bundle: {mp.name}")
        else:
            st.info("No trained bundle yet — run Training & Evaluation.")
    st.subheader("Research outputs")
    files = sorted(RESULTS_DIR.glob("*.json")) if RESULTS_DIR.is_dir() else []
    if files:
        for f in files:
            st.write("- `", f.relative_to(RESULTS_DIR.parent.parent), "`")
    else:
        st.caption("No results yet — run `python scripts/train_ml.py` after prepare_data.")
    st.subheader("Limitations (read before citing anything)")
    for lim in limitations():
        st.markdown(f"- {lim}")
