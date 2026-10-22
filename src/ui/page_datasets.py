"""Dataset Explorer: live UCSD adapter inspection cards; placement hints; no downloads."""

from __future__ import annotations

import streamlit as st

from src.config import load_paths
from src.datasets.ucsd_anomaly import UcsdAdapter
from src.ui.common import ucSD_placement_hint
from src.ui.theme import page_header


def render() -> None:
    page_header(
        "Dataset Explorer", "📂",
        "Live inspection via `UcsdAdapter` — nothing is downloaded or fabricated.",
    )
    try:
        adapter = UcsdAdapter(root=load_paths().ucsd_dir)
        seqs = list(adapter.iter_sequences())
        n_frames = sum(s.n_frames for s in seqs)
        n_anom = sum(int(s.frame_labels.sum()) for s in seqs if s.frame_labels is not None)
    except Exception as exc:
        st.error(f"UCSD dataset unavailable: {exc}")
        st.code(ucSD_placement_hint(), language=None)
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Sequences", len(seqs))
    c2.metric("Frames", f"{n_frames:,}")
    c3.metric("Anomalous-frame labels", f"{n_anom:,}")

    for seq in adapter.iter_sequences():
        badge = "normal" if seq.frame_labels.sum() == 0 else f"anomaly @ {int(seq.frame_labels.sum())} frames"
        with st.expander(f"{seq.sequence_id} — {seq.n_frames} frames — {badge}"):
            cols = st.columns(3)
            cols[0].write(f"part: {seq.part}")
            cols[1].write(f"split: {seq.split}")
            cols[2].write(f"pixel masks: {'yes' if seq.has_masks else 'no'}")
