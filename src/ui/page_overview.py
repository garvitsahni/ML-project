"""Overview: task choices, quiet live-status strip, session history.

Every number on this page is read from disk at render time (models/,
data/, reports/). Nothing is hardcoded; absent artifacts render as
warn/idle wording in the status strip. Session history (CSV export +
clear-with-confirmation) lives here so operators never need a separate
reports page.
"""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from src.config import load_paths
from src.ui import components as ui
from src.ui import state
from src.ui.common import dataset_probe, limitations, list_model_files, \
    load_evaluation

_PROBE_TTL_SECONDS = 120

# task tiles — inline SVG icons from the approved mockup (no emoji)
_ICONS = {
    "photo": '<svg viewBox="0 0 24 24" width="19" height="19" '
             'fill="none" stroke="#0F766E" stroke-width="1.6">'
             '<rect x="3" y="4" width="18" height="16" rx="2"/>'
             '<circle cx="8.5" cy="9.5" r="1.6"/>'
             '<path d="M4 17l5-5 4 4 3-3 4 4"/></svg>',
    "video": '<svg viewBox="0 0 24 24" width="19" height="19" '
             'fill="none" stroke="#0F766E" stroke-width="1.6">'
             '<rect x="3" y="5" width="18" height="14" rx="2"/>'
             '<path d="M10 9.5l5 2.5-5 2.5z"/></svg>',
    "camera": '<svg viewBox="0 0 24 24" width="19" height="19" '
              'fill="none" stroke="#0F766E" stroke-width="1.6">'
              '<path d="M4 8h3l2-2h6l2 2h3v10H4z"/>'
              '<circle cx="12" cy="13" r="3.2"/></svg>',
}


@st.cache_data(ttl=_PROBE_TTL_SECONDS, show_spinner=False)
def _probe_cached(root: str) -> dict:
    """Cached dataset probe (adapter scan is too slow to repeat per rerun)."""
    return dataset_probe(root)


def status_items(model_count: int, probe: dict, evaluation: dict,
                 runs: int) -> list[tuple[str, str]]:
    """Quiet status-strip entries: (kind, text) pairs for `ui.pill`.

    Every value is a real read passed in by the caller; absent artifacts
    render warn/idle wording, never green.
    """
    if model_count:
        noun = "model bundle" if model_count == 1 else "model bundles"
        models = ("ok", f"{model_count} {noun} ready")
    else:
        models = ("warn", "No models yet")

    if probe.get("available"):
        sequences = probe["sequences"]
        seq_noun = "sequence" if sequences == 1 else "sequences"
        dataset = ("ok", f"Dataset ready · {sequences} {seq_noun}")
    else:
        dataset = ("warn", "Dataset not found")

    if evaluation.get("bundles"):
        windows = evaluation.get("n_test_windows", 0)
        win_noun = "evaluation window" if windows == 1 else "evaluation windows"
        evaluation_item = ("idle", f"{windows} {win_noun}")
    else:
        evaluation_item = ("idle", "No evaluation yet")

    if runs:
        noun = "run" if runs == 1 else "runs"
        runs_item = ("idle", f"{runs} {noun} this session")
    else:
        runs_item = ("idle", "No runs yet")

    return [models, dataset, evaluation_item, runs_item]


def render() -> None:
    from src.ui import nav

    ui.page_header(
        title="What do you want to analyse?",
        description="Pick a task — system status and this session's "
                    "runs are below.",
    )

    tiles = [
        ("Photo", "One image, quick count", "photo", "tile_photo", nav.PHOTO),
        ("Video", "File, processed in steps", "video", "tile_video", nav.VIDEO),
        ("Camera", "Live from a device", "camera", "tile_camera", nav.CAMERA),
    ]
    cols = st.columns(3)
    for col, (name, desc, icon, key, page) in zip(cols, tiles):
        with col:
            with st.container(border=True, key=f"{key}-box"):
                st.markdown(
                    f'<div class="tile">{_ICONS[icon]} '
                    f'<strong>{name}</strong>'
                    f'<div class="page-desc" style="margin-top:.2rem">'
                    f'{desc}</div></div>',
                    unsafe_allow_html=True,
                )
                if st.button("Start →", key=key, width="stretch"):
                    st.switch_page(page)

    entries = state.get_history()
    items = status_items(
        len(list_model_files()),
        _probe_cached(str(load_paths().ucsd_dir)),
        load_evaluation(),
        len(entries),
    )
    strip = "".join(ui.pill(text, kind) for kind, text in items)
    st.markdown(f'<div class="status-strip">{strip}</div>',
                unsafe_allow_html=True)

    if entries:
        import pandas as pd

        with st.expander(f"This session: {len(entries)} runs"):
            rows = ui.history_rows(entries)
            st.dataframe(
                pd.DataFrame(rows, columns=["When", "Input", "Key results"]),
                hide_index=True, width="stretch")

        link, clear, settings_col = st.columns([1, 1, 2])
        with link:
            stamp = datetime.now().strftime("%Y%m%d-%H%M")
            ui.download_csv("Download history CSV", ui.export_csv(entries),
                            f"crowdsense_history_{stamp}.csv",
                            help="Full session history with raw result keys.")
        with clear:
            confirmed = st.checkbox("I understand this clears the session "
                                    "history", key="activity_clear_confirm")
            if st.button("Clear history", key="activity_clear_go",
                         disabled=not confirmed) and confirmed:
                state.clear_history()
                st.rerun()
        with settings_col:
            s = state.settings()
            st.caption(f"Settings: confidence ≥ {s['conf_threshold']:.2f} · "
                       f"stride {s['frame_stride']}")
        ui.data_note("Session-only — download the CSV to keep a copy.")

    st.markdown("### About the numbers")
    ui.render_limitations(limitations())
