"""Overview: product intro, primary actions, live system status, activity.

Every number on this page is read from disk at render time (models/,
data/, reports/). Nothing is hardcoded; absent artifacts render as
informative empty states. Session history (CSV export + clear) lives
here so operators never need a separate reports page.
"""

from __future__ import annotations

import time
from datetime import datetime

import streamlit as st

from src.config import load_paths
from src.ui import components as ui
from src.ui import state
from src.ui.common import dataset_probe, limitations, list_model_files, \
    load_evaluation

_PROBE_TTL_SECONDS = 120


@st.cache_data(ttl=_PROBE_TTL_SECONDS, show_spinner=False)
def _probe_cached(root: str) -> dict:
    """Cached dataset probe (adapter scan is too slow to repeat per rerun)."""
    return dataset_probe(root)


def _goto(page_obj) -> None:
    """CTA helper: jump to another registered page."""
    st.switch_page(page_obj)


def _render_models_status() -> None:
    st.markdown("**Models**")
    files = list_model_files()
    if not files:
        ui.empty_state(
            "No trained model available",
            "Train bundles on the analysis machine: "
            "`python scripts/train_ml.py`.",
            key="models_empty",
        )
        return
    ui.status_pill(
        f"{len(files)} trained bundle{'s' if len(files) != 1 else ''}", "ok")
    newest, mtime = files[0]
    st.write(f"Newest: **{newest.name}**")
    st.caption("trained " + time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime)))


def _render_dataset_status() -> None:
    st.markdown("**Dataset**")
    probe = _probe_cached(str(load_paths().ucsd_dir))
    if probe["available"]:
        ui.status_pill("UCSD v1p2 ready", "ok")
        st.write(f"**{probe['sequences']} sequences** · "
                 f"{ui.format_count(probe['frames'])} frames")
        st.caption(f"{ui.format_count(probe['anomalous'])} anomalous-frame labels")
    else:
        ui.empty_state(
            "Dataset not found",
            "Datasets are never auto-downloaded — place UCSD v1p2 under "
            "`data/raw/ucsd_anomaly/UCSD_Anomaly_Dataset.v1p2` "
            "(see configs/paths.yaml).",
            key="dataset_empty",
        )


def _render_evaluation_status() -> None:
    st.markdown("**Latest evaluation**")
    evaluation = load_evaluation()
    bundles = evaluation.get("bundles") or {}
    if not bundles:
        ui.empty_state(
            "No evaluation yet",
            "Run training, then evaluation, to produce "
            "`reports/results/evaluation.json`.",
            key="eval_empty",
        )
        return
    ui.status_pill(f"evaluation.json · {len(bundles)} bundles", "ok")
    st.write(f"**{evaluation.get('n_test_windows', 0)} held-out test windows**")
    st.caption(f"{len(evaluation.get('test_sequences', []))} test sequences "
               "(grouped split — no sequence spans train/test)")


def _render_recent_activity() -> None:
    import pandas as pd

    entries = state.get_history()
    if not entries:
        ui.empty_state(
            "No analyses yet",
            "Analyse a photo or video, or start the camera — results from "
            "this session will be listed here.")
        return
    rows = ui.history_rows(entries)
    st.dataframe(pd.DataFrame(rows, columns=["When", "Input", "Key results"]),
                 hide_index=True, width="stretch")
    ui.data_note("Session-only history — not persisted to disk; download the "
                 "CSV to keep a copy.")
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    ui.download_csv("Download history CSV", ui.export_csv(entries),
                    f"crowdsense_history_{stamp}.csv",
                    help="Full session history with raw result keys.")
    confirmed = st.checkbox("I understand this clears the session history",
                            key="activity_clear_confirm")
    if st.button("Clear history", key="activity_clear_go",
                 disabled=not confirmed) and confirmed:
        state.clear_history()
        st.rerun()


def render() -> None:
    from src.ui import nav

    ui.page_header(
        title="CrowdSense AI",
        description=(
            "Analyse crowd photographs, videos and live camera streams: "
            "detect and track people, estimate density, and extract "
            "movement features. Every value shown is computed from your "
            "inputs or read from files under reports/."),
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Analyse photo", type="primary", width="stretch", key="cta_photo"):
            _goto(nav.PHOTO)
    with c2:
        if st.button("Analyse video", type="primary", width="stretch", key="cta_video"):
            _goto(nav.VIDEO)
    with c3:
        if st.button("Start camera", type="primary", width="stretch", key="cta_camera"):
            _goto(nav.CAMERA)

    st.divider()

    with ui.card("System status",
                 hint="Read live from models/, data/ and reports/results/."):
        s1, s2, s3 = st.columns(3)
        with s1:
            _render_models_status()
        with s2:
            _render_dataset_status()
        with s3:
            _render_evaluation_status()

    with ui.card("Recent activity",
                 hint="Analyses from this browser session."):
        _render_recent_activity()

    with ui.card("About the numbers"):
        ui.render_limitations(limitations())
