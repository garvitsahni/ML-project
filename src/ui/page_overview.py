"""Overview: product intro, primary actions, live system status, activity.

Every number on this page is read from disk at render time (models/,
data/, reports/). Nothing is hardcoded; absent artifacts render as
informative empty states.
"""

from __future__ import annotations

import time

import streamlit as st

from src.config import load_paths
from src.ui import components as ui
from src.ui import state
from src.ui.common import (FIGURES_DIR, RESULTS_DIR, dataset_probe,
                           limitations, list_model_files, list_result_files,
                           load_evaluation)

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
        if ui.empty_state(
            "No trained model available",
            "Run the training pipeline to produce the five candidate bundles.",
            action_label="Open Training & Evaluation",
            key="models_empty",
        ):
            from src.ui import nav
            _goto(nav.TRAINING)
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
        if ui.empty_state(
            "Dataset not found",
            "Datasets are never auto-downloaded — the Dataset Explorer shows "
            "exact placement instructions.",
            action_label="Open Dataset Explorer",
            key="dataset_empty",
        ):
            from src.ui import nav
            _goto(nav.DATASETS)


def _render_evaluation_status() -> None:
    st.markdown("**Latest evaluation**")
    evaluation = load_evaluation()
    bundles = evaluation.get("bundles") or {}
    if not bundles:
        if ui.empty_state(
            "No evaluation yet",
            "Run training, then evaluation, to produce "
            "`reports/results/evaluation.json`.",
            action_label="Open Training & Evaluation",
            key="eval_empty",
        ):
            from src.ui import nav
            _goto(nav.TRAINING)
        return
    ui.status_pill(f"evaluation.json · {len(bundles)} bundles", "ok")
    st.write(f"**{evaluation.get('n_test_windows', 0)} held-out test windows**")
    st.caption(f"{len(evaluation.get('test_sequences', []))} test sequences "
               "(grouped split — no sequence spans train/test)")
    if st.button("Open Reports", key="eval_open_results"):
        from src.ui import nav
        _goto(nav.REPORTS)


def _render_available_models() -> None:
    import pandas as pd

    files = list_model_files()
    if not files:
        ui.empty_state("No model artifacts on disk")
        return
    rows = [
        [path.name, f"{path.stat().st_size / 1024:.1f}",
         time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))]
        for path, mtime in files
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Bundle", "Size (KB)", "Modified"]),
                 hide_index=True, width="stretch")
    ui.data_note("Source: models/ — schema-versioned joblib bundles used for "
                 "video/camera window inference.")


def _render_research_outputs() -> None:
    result_files = list_result_files(RESULTS_DIR)
    fig_count = len(list(FIGURES_DIR.glob("*.png"))) if FIGURES_DIR.is_dir() else 0
    if not result_files and not fig_count:
        if ui.empty_state(
            "No research results yet",
            "Run `python scripts/train_ml.py` (experiments A–D) and "
            "`python scripts/evaluate.py` to populate reports/.",
            action_label="Open Training & Evaluation",
            key="outputs_empty",
        ):
            from src.ui import nav
            _goto(nav.TRAINING)
        return
    ui.status_pill(f"{len(result_files)} result files · {fig_count} figures", "ok")
    st.write(" · ".join(f"`{p.name}`" for p in result_files) or "—")
    if st.button("Open Reports", key="outputs_open_results"):
        from src.ui import nav
        _goto(nav.REPORTS)


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
    ui.data_note("Session-only history — not persisted to disk; cleared when "
                 "the app server restarts.")
    if st.button("Open Reports", key="activity_open_reports"):
        from src.ui import nav
        _goto(nav.REPORTS)


def render() -> None:
    from src.ui import nav

    ui.page_header(
        eyebrow="Crowd analytics research",
        title="CrowdSense AI",
        description=(
            "Analyse crowd photographs, videos and live camera streams: "
            "detect and track people, estimate density, extract movement "
            "features, and evaluate trained ML models. Every value shown is "
            "computed from your inputs or read from files under reports/."),
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

    m1, m2 = st.columns(2)
    with m1:
        with ui.card("Available models",
                     hint="Trained bundles used for video/camera inference."):
            _render_available_models()
    with m2:
        with ui.card("Recent activity",
                     hint="Analyses from this browser session."):
            _render_recent_activity()

    with ui.card("Research outputs",
                 hint="The dashboard renders only files that exist under reports/."):
        _render_research_outputs()

    with ui.card("Limitations (read before citing anything)"):
        ui.render_limitations(limitations())
        if st.button("Full methodology", key="limitations_methodology"):
            _goto(nav.METHODOLOGY)
