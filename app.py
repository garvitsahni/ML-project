"""CrowdSense AI — Streamlit dashboard shell."""

from __future__ import annotations

import streamlit as st

from src.ui import (
    page_camera,
    page_datasets,
    page_methodology,
    page_overview,
    page_photo,
    page_results,
    page_training,
    page_video,
)
from src.ui.common import find_model_path, model_status_badge, nav_group_labels
from src.ui.theme import footer, inject_css

PAGES = {
    "Overview": page_overview,
    "Methodology": page_methodology,
    "Dataset Explorer": page_datasets,
    "Photo Analysis": page_photo,
    "Video Analysis": page_video,
    "Live Camera": page_camera,
    "Training & Evaluation": page_training,
    "Research Results": page_results,
}

NAV_GROUPS = [
    ("Learn", ["Overview", "Methodology", "Dataset Explorer"]),
    ("Analyze", ["Photo Analysis", "Video Analysis", "Live Camera"]),
    ("Train & Review", ["Training & Evaluation", "Research Results"]),
]


def main() -> None:
    st.set_page_config(page_title="CrowdSense AI", page_icon="🧍", layout="wide")
    inject_css()
    with st.sidebar:
        st.markdown("## CrowdSense AI")
        st.caption("Crowd anomaly research on UCSD Ped1/Ped2")
        choice = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
        for line in nav_group_labels(NAV_GROUPS):
            st.caption(line)
        text, emoji = model_status_badge(find_model_path())
        st.caption(f"{emoji} {text}")
    PAGES[choice].render()
    footer()


if __name__ == "__main__":
    main()
