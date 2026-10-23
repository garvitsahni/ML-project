"""Reports: session analysis history (session-only) + research results on disk.

History rows are built from real session entries; research sections render
only files that exist under reports/results/. Missing outputs show the exact
command that produces them. Exports are CSV of real data only — no PDF is
offered because none is implemented.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import streamlit as st

from src.ui import components as ui
from src.ui import state
from src.ui.common import FIGURES_DIR, RESULTS_DIR, list_result_files
from src.ui.page_research import experiment_table

_MISSING = "—"
_KIND_LABELS = {"photo": "Photo", "video": "Video", "camera": "Camera"}

#: Per-file payload -> wrapper key `experiment_table` expects.
_TABLE_KEY = {
    "experiment_algorithms.json": "algorithms",
    "experiment_ablation.json": "ablation",
}

#: Expected result files and the exact command that produces each.
COMMAND_HINTS = {
    "evaluation.json": "python scripts/evaluate.py",
    "experiment_algorithms.json": "python scripts/train_ml.py",
    "experiment_ablation.json": "python scripts/train_ml.py",
    "exp_e_status.json": "python scripts/evaluate.py",
    "feature_table_meta.json": "python scripts/prepare_data.py",
}


# --------------------------------------------------------------------------
# pure helpers (unit-tested)
# --------------------------------------------------------------------------

def _entry_input(entry: dict) -> str:
    kind = entry.get("kind")
    if kind == "photo":
        return "uploaded photo"
    if kind == "video":
        return str(entry.get("video") or _MISSING)
    if kind == "camera":
        return f"camera {entry.get('device')}"
    return _MISSING


def _entry_results(entry: dict) -> str:
    kind = entry.get("kind")
    if kind == "photo" and entry.get("persons") is not None:
        return f"{entry['persons']} persons"
    if kind == "video" and entry.get("windows") is not None:
        return f"{entry['windows']} windows · {entry.get('frames', 0)} frames"
    if kind == "camera" and entry.get("frames") is not None:
        return f"{entry['frames']} frames"
    return _MISSING


def history_frame(entries: list[dict]):
    """Flat, display-ready table of session history entries (in order)."""
    import pandas as pd

    columns = ["time", "input", "kind", "model", "results", "duration"]
    rows = [{
        "time": str(e.get("ts", "")).replace("T", " "),
        "input": _entry_input(e),
        "kind": _KIND_LABELS.get(e.get("kind"), str(e.get("kind", _MISSING))),
        "model": str(e.get("model") or _MISSING),
        "results": _entry_results(e),
        "duration": (f"{e['seconds']} s" if e.get("seconds") is not None
                     else _MISSING),
    } for e in entries]
    return pd.DataFrame(rows, columns=columns)


def export_csv(entries: list[dict]) -> str:
    """History table + raw summary keys — machine-usable, properly quoted."""
    import pandas as pd

    table = history_frame(entries).reset_index(drop=True)
    raw = pd.DataFrame([{k: str(v) for k, v in e.items()}
                        for e in entries]).reset_index(drop=True)
    return pd.concat([table, raw], axis=1).to_csv(index=False)


# --------------------------------------------------------------------------
# session history
# --------------------------------------------------------------------------

def _render_history() -> None:
    entries = state.get_history()
    with ui.card("Analysis history (this session)",
                 hint="Recorded by Photo, Video and Camera runs in this "
                      "browser session."):
        st.caption("**Session-only — not persisted.** History lives in this "
                   "browser session and disappears when the tab closes; "
                   "download the CSV to keep a copy.")
        if not entries:
            ui.empty_state(
                "No analyses in this session yet",
                "Run a photo, video or camera analysis — it will appear "
                "here with its key results.")
            return
        st.dataframe(history_frame(entries), hide_index=True,
                     width="stretch")
        for entry in entries:
            label = (f"{_KIND_LABELS.get(entry.get('kind'), 'Run')} — "
                     f"{str(entry.get('ts', '')).replace('T', ' ')}")
            with st.expander(label):
                ui.kv_table([(str(k), str(v)) for k, v in entry.items()])
        stamp = datetime.now().strftime("%Y%m%d-%H%M")
        st.download_button(
            "Download history CSV", export_csv(entries),
            file_name=f"crowdsense_history_{stamp}.csv",
            mime="text/csv", key="reports_history_csv")
        _render_clear()


def _render_clear() -> None:
    confirmed = st.checkbox("I understand this clears the session history",
                            key="reports_clear_confirm")
    if st.button("Clear history", key="reports_clear_go",
                 disabled=not confirmed) and confirmed:
        state.clear_history()
        st.rerun()


# --------------------------------------------------------------------------
# research results on disk
# --------------------------------------------------------------------------

def _render_research() -> None:
    st.markdown("### Research results on disk")
    st.caption("Rendered from reports/results/*.json and reports/figures — "
               "only files that exist are shown; no value is invented.")
    json_files = list_result_files(RESULTS_DIR, suffix=".json")
    if not json_files:
        ui.empty_state("No result files yet",
                       "reports/results/ contains no JSON outputs.")
    for path in json_files:
        _render_result_file(path)
    _render_missing()
    _render_figures()


def _render_result_file(path: Path) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        ui.error_state(f"Could not read {path.name}",
                       detail=f"{type(exc).__name__}: {exc}",
                       key=f"reports_read_{path.stem}")
        return
    st.markdown(f"#### `{path.name}`")
    if path.name == "exp_e_status.json":
        st.code(json.dumps(payload, indent=2), language="json")
        return
    if path.name == "evaluation.json":
        _render_evaluation(payload)
        return
    key = _TABLE_KEY.get(path.name)
    table = experiment_table({key: payload} if key else payload)
    if table is not None and len(table):
        st.dataframe(table, hide_index=True, width="stretch")
        st.download_button(
            f"Download {path.stem}.csv", table.to_csv(index=False),
            file_name=f"{path.stem}.csv", mime="text/csv",
            key=f"reports_csv_{path.stem}")
        return
    with st.expander("raw JSON"):
        st.code(json.dumps(payload, indent=2)[:8000], language="json")


def _render_evaluation(payload: dict) -> None:
    import pandas as pd

    bits = []
    if isinstance(payload.get("n_test_windows"), (int, float)):
        bits.append(f"{payload['n_test_windows']} test windows")
    seqs = payload.get("test_sequences")
    if isinstance(seqs, list) and seqs:
        bits.append(f"{len(seqs)} held-out sequences")
    if bits:
        st.caption(" · ".join(bits) + " (sequence-grouped split)")
    bundles = payload.get("bundles") or {}
    if not bundles:
        st.caption("No bundle metrics in this file.")
        return
    rows = []
    for algo, bundle in sorted(bundles.items()):
        metrics = (bundle or {}).get("test_metrics") or {}
        rows.append({
            "algorithm": algo,
            "f1": metrics.get("f1"),
            "roc_auc": metrics.get("roc_auc"),
            "n_train": (bundle or {}).get("n_train"),
            "trained_at": (bundle or {}).get("trained_at"),
        })
    frame = pd.DataFrame(rows)
    st.dataframe(frame, hide_index=True, width="stretch")
    st.download_button(
        "Download evaluation.csv", frame.to_csv(index=False),
        file_name="evaluation.csv", mime="text/csv",
        key="reports_csv_evaluation")


def _render_missing() -> None:
    missing = [(name, cmd) for name, cmd in COMMAND_HINTS.items()
               if not (RESULTS_DIR / name).is_file()]
    if not missing:
        return
    with ui.card("Not generated yet",
                 hint="The exact command that produces each missing file."):
        for name, cmd in missing:
            st.caption(f"`reports/results/{name}`")
            st.code(cmd, language="bash")


def _render_figures() -> None:
    with ui.card("Figures",
                 hint="PNGs written by the training/evaluation scripts "
                      "(confusion matrix, ROC, feature importance)."):
        names = (sorted(p.name for p in FIGURES_DIR.glob("*.png"))
                 if FIGURES_DIR.is_dir() else [])
        if not names:
            st.caption("No figures on disk.")
            st.code("python scripts/train_ml.py\npython scripts/evaluate.py",
                    language="bash")
            return
        cols = st.columns(3)
        for index, name in enumerate(names):
            with cols[index % 3]:
                st.image(str(FIGURES_DIR / name), caption=name,
                         width="stretch")


# --------------------------------------------------------------------------
# page entrypoint
# --------------------------------------------------------------------------

def render() -> None:
    ui.page_header(
        eyebrow="Session & results",
        title="Reports",
        description=("What this session analysed, plus the research results "
                     "that exist on disk — every number from a real run or "
                     "a real file. Missing outputs show the exact command "
                     "that produces them."),
    )
    _render_history()
    _render_research()
