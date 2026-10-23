"""Dataset Explorer: live UCSD probe, built-table stats, provenance — no downloads.

Every number here is probed from files on disk (the UCSD adapter or the built
window table's meta JSON). When the dataset is not placed, the page shows the
placement hint and the exact inspection command — it never downloads and never
fabricates counts.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.config import load_paths
from src.ui import components as ui
from src.ui.charts import bar_h
from src.ui.common import ucSD_placement_hint
from src.ui.page_research import load_window_meta


# --------------------------------------------------------------------------
# pure helpers (unit-tested)
# --------------------------------------------------------------------------

def dataset_summary(adapter) -> dict:
    """Count sequences/frames/labels from an adapter's iter_sequences().

    Frames whose labels are absent (None) contribute 0 anomalous frames —
    never a guess.
    """
    by_split: dict[str, int] = {}
    by_part: dict[str, int] = {}
    frames_by_split: dict[str, int] = {}
    rows: list[dict] = []
    sequences = frames = anomalous = masks = 0
    for seq in adapter.iter_sequences():
        sequences += 1
        n_frames = int(seq.n_frames)
        frames += n_frames
        labels = seq.frame_labels
        n_anom = int(labels.sum()) if labels is not None else 0
        anomalous += n_anom
        by_split[seq.split] = by_split.get(seq.split, 0) + 1
        by_part[seq.part] = by_part.get(seq.part, 0) + 1
        frames_by_split[seq.split] = (frames_by_split.get(seq.split, 0)
                                      + n_frames)
        if seq.has_masks:
            masks += 1
        rows.append({
            "sequence_id": seq.sequence_id,
            "part": seq.part,
            "split": seq.split,
            "n_frames": n_frames,
            "anomalous": n_anom,
            "anomaly_rate": (n_anom / n_frames) if n_frames else 0.0,
            "has_masks": bool(seq.has_masks),
        })
    return {
        "sequences": sequences,
        "frames": frames,
        "anomalous_frames": anomalous,
        "by_split": by_split,
        "by_part": by_part,
        "frames_by_split": frames_by_split,
        "with_masks": masks,
        "rows": rows,
    }


def _dataset_fingerprint(root: Path) -> str:
    """Cheap cache key: 'absent' when missing, else an mtime-derived stamp."""
    root = Path(root)
    if not root.is_dir():
        return "absent"
    stamp = root.stat().st_mtime_ns
    for part in sorted(root.glob("UCSDped*")):
        if part.is_dir():
            stamp ^= part.stat().st_mtime_ns
    return f"{stamp:x}"


@st.cache_data(show_spinner=False, max_entries=4)
def _probe_cached(root_str: str, fingerprint: str) -> dict:
    """Probe the dataset behind st.cache_data, keyed by the mtime stamp.

    Dataset problems come back as {"error": ...} so the page renders an
    actionable block instead of a traceback.
    """
    from src.datasets.ucsd_anomaly import UcsdAdapter

    try:
        adapter = UcsdAdapter(root=Path(root_str))
        return dataset_summary(adapter)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------
# render pieces
# --------------------------------------------------------------------------

def _render_summary(summary: dict, root: Path) -> None:
    with ui.card("UCSD Anomaly Detection Dataset v1p2",
                 hint=f"Probed live from {root} — frame-level anomaly "
                      "labels (Ped1 + Ped2)."):
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Sequences", summary["sequences"])
        s2.metric("Frames", f"{summary['frames']:,}")
        s3.metric("Anomalous frames", f"{summary['anomalous_frames']:,}")
        rate = ((summary["anomalous_frames"] / summary["frames"])
                if summary["frames"] else 0.0)
        s4.metric("Anomaly rate", f"{rate * 100:.1f}%")
        splits = summary["by_split"]
        parts = summary["by_part"]
        st.caption(
            f"Train/test sequences: {splits.get('train', 0)}/"
            f"{splits.get('test', 0)} (grouped split — no sequence spans "
            "both) · parts: "
            + (", ".join(f"{k} ({v})" for k, v in sorted(parts.items()))
               or "—")
            + f" · sequences with pixel GT masks: {summary['with_masks']}.")
    _render_sequence_table(summary)


def _render_sequence_table(summary: dict) -> None:
    rows = summary.get("rows") or []
    if not rows:
        return
    import pandas as pd

    with st.expander(f"Per-sequence inspection ({len(rows)} sequences)"):
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        ui.data_note("anomalous = frame-level anomaly labels summed by the "
                     "adapter; anomaly_rate is that count over n_frames.")


def _render_windows_section() -> None:
    import pandas as pd

    meta = load_window_meta()
    with ui.card("Built feature table",
                 hint="data/processed/ucsd_windows.meta.json — produced by "
                      "python scripts/prepare_data.py."):
        if not meta:
            st.caption("No built table on disk.")
            st.code("python scripts/prepare_data.py", language="bash")
            return
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Windows", f"{meta.get('rows', 0):,}")
        s2.metric("Sequences", meta.get("sequences", 0))
        s3.metric("Window / stride",
                  f"{meta.get('window')} / {meta.get('stride')} frames")
        s4.metric("Grid", f"{meta.get('grid')}×{meta.get('grid')}")
        labels = meta.get("label_counts") or {}
        splits = meta.get("split_sequence_counts") or {}
        c1, c2 = st.columns(2)
        with c1:
            if labels:
                st.altair_chart(bar_h(
                    pd.DataFrame({
                        "label": ["normal (0)", "anomalous (1)"],
                        "windows": [labels.get("0", 0), labels.get("1", 0)],
                    }),
                    x="label", y="windows",
                    title="Window class distribution",
                    y_label="windows", unit="windows"), width="stretch")
            else:
                st.caption("No label counts in the meta file.")
        with c2:
            if splits:
                st.altair_chart(bar_h(
                    pd.DataFrame({
                        "split": sorted(splits),
                        "sequences": [splits[k] for k in sorted(splits)],
                    }),
                    x="split", y="sequences",
                    title="Sequences per split (grouped)",
                    y_label="sequences", unit="sequences"), width="stretch")
            else:
                st.caption("No split counts in the meta file.")
        ui.data_note("Window label = 1 when ANY frame in the window is "
                     "anomalous (UCSD frame GT); partial trailing windows "
                     "are dropped at build time.")


def _render_kinds() -> None:
    with ui.card("Dataset kinds in this project",
                 hint="What exists on disk vs what is out of scope — no "
                      "dataset is ever fetched automatically."):
        ui.kv_table([
            ("Anomaly (frame-level)",
             "present — UCSD Ped1/Ped2 v1p2; window labels inherit frame GT"),
            ("Counting (MAE/RMSE)",
             "absent / out of scope — UCF-QNRF and ShanghaiTech were "
             "dropped; no counting ground truth exists here"),
            ("Congestion-labelled (future events)",
             "absent / gated — Experiment E stays disabled until validated "
             "congestion labels exist"),
        ])


def _render_provenance() -> None:
    with ui.card("Source & annotation",
                 hint="Facts from the project README — not inferred."):
        st.markdown(
            "- **Source:** UCSD Anomaly Detection Dataset v1p2 (Ped1 + "
            "Ped2), Mahadevan et al., *Anomaly Detection in Crowded "
            "Scenes*, CVPR 2010.\n"
            "- **Annotation:** frame-level anomaly labels (Train/Test .txt) "
            "plus optional pixel GT masks (*_gt folders) — the adapter "
            "reads both.\n"
            "- **Placement:** manually extracted under "
            "data/raw/ucsd_anomaly/ (see the README for the exact tree). "
            "This app never downloads datasets and grants no licence — "
            "use the original dataset's terms.\n"
            "- **Split policy:** grouped by sequence, never random frames.")


# --------------------------------------------------------------------------
# page entrypoint
# --------------------------------------------------------------------------

def render() -> None:
    ui.page_header(
        eyebrow="Research · datasets",
        title="Dataset Explorer",
        description=(
            "Live inspection of the placed UCSD anomaly dataset and the "
            "built feature table — every number is probed from files on "
            "disk. Nothing is ever downloaded."),
    )
    root = load_paths().ucsd_dir
    fingerprint = _dataset_fingerprint(root)

    if fingerprint == "absent":
        ui.empty_state(
            "UCSD dataset not placed",
            "The raw dataset is not under the configured path. This "
            "project never downloads data — place it manually once:")
        st.code(ucSD_placement_hint(), language="text")
        st.code("python scripts/inspect_dataset.py   # probe after placing",
                language="bash")
    else:
        summary = _probe_cached(str(root), fingerprint)
        if summary.get("error"):
            ui.error_state("Dataset probe failed", detail=summary["error"],
                           key="datasets_probe_error")
        else:
            _render_summary(summary, root)

    _render_windows_section()
    _render_kinds()
    _render_provenance()
