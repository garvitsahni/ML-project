"""Methodology page: honest limitations, split/label policies, Experiment E gate."""

from __future__ import annotations

import json

import streamlit as st

from src.config import PROJECT_ROOT
from src.ui.common import RESULTS_DIR, limitations


def render() -> None:
    st.header("Methodology & Limitations")
    st.markdown(
        """
**Features (15 total, canonical schema in `src/features/schema.py`):**
- *Static (6):* person_count, density_per_mpx, grid_concentration, spread_x, spread_y, grid_entropy.
- *Movement (9):* density_delta, mean_speed_pxf, speed_std_pxf, direction_entropy,
  directional_consistency, flow_mean_magnitude, flow_std_pxf, track_count, track_persistence.

**Split policy:** grouped by sequence id (never random frames). A sequence appears in
exactly one of train/val/test; `assert_group_disjoint` verifies at runtime.

**Window label policy:** a window is anomalous if ANY frame inside it is anomalous
(UCSD frame-level ground truth). Partial trailing windows are dropped.

**Units:** movement features are px/frame at processed resolution — never m/s
(scene scale is unknown). Detector counts are YOLO estimates, not ground truth.
"""
    )
    st.subheader("Honest limitations")
    for lim in limitations():
        st.markdown(f"- {lim}")
    st.subheader("Experiment E status (gated)")
    status_file = RESULTS_DIR / "exp_e_status.json"
    if status_file.is_file():
        payload = json.loads(status_file.read_text(encoding="utf-8"))
        st.code(json.dumps(payload, indent=2), language="json")
        if payload.get("status") != "not_supported":
            st.error("Unexpected: E ran without credible labels — investigate before trusting it.")
    else:
        st.info("No experiment run yet. Run `python scripts/evaluate.py` to generate status files.")
    st.subheader("Data provenance")
    st.markdown(
        """
- Dataset: **UCSD Anomaly Detection Dataset v1p2** (Mahadevan et al., CVPR 2010) —
  placed manually under `data/raw/ucsd_anomaly/` (never auto-downloaded).
- Citation: V. Mahadevan, W. Li, N. Boccali, A. K. Roy-Chowdhury,
  "Anomaly Detection in Crowded Scenes," CVPR 2010.
- UCF-QNRF and ShanghaiTech were **dropped from scope** (anomaly experiments only);
  counting MAE/RMSE is therefore not reported anywhere in this project.
"""
    )
