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
from src.ui.common import find_model_path, model_status_badge

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


def main() -> None:
    st.set_page_config(page_title="CrowdSense AI", page_icon="🧍", layout="wide")
    with st.sidebar:
        st.title("CrowdSense AI")
        choice = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
        text, emoji = model_status_badge(find_model_path())
        st.caption(f"{emoji} {text}")
        st.caption(
            "Density · tracking · movement features\nfor crowd anomaly research (UCSD)"
        )
    PAGES[choice].render()


if __name__ == "__main__":
    main()
