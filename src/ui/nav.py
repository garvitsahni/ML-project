"""Central navigation registry: page objects shared by the shell and CTAs.

The app shell builds `st.navigation` from `GROUPS`. Individual pages are
imported as modules; CTAs inside pages import this module lazily (inside
`render()`) to avoid import cycles.

Nav map (spec §4): destinations are only listed once implemented — legacy
Research pages (Training & Evaluation, Research Results, Methodology) are
replaced in later tasks by Research Lab / Reports.
"""

from __future__ import annotations

import streamlit as st

from src.ui import (page_camera, page_datasets, page_methodology,
                    page_overview, page_photo, page_results, page_training,
                    page_video)

OVERVIEW = st.Page(page_overview.render, title="Overview", icon=":material/home:",
                   url_path="overview", default=True)
PHOTO = st.Page(page_photo.render, title="Photo Analysis", icon=":material/image:",
                url_path="photo-analysis")
VIDEO = st.Page(page_video.render, title="Video Analysis", icon=":material/video_library:",
                url_path="video-analysis")
CAMERA = st.Page(page_camera.render, title="Live Camera", icon=":material/videocam:",
                 url_path="camera")
DATASETS = st.Page(page_datasets.render, title="Dataset Explorer",
                   icon=":material/storage:", url_path="datasets")
TRAINING = st.Page(page_training.render, title="Training & Evaluation",
                   icon=":material/school:", url_path="training")
RESULTS = st.Page(page_results.render, title="Research Results",
                  icon=":material/assessment:", url_path="results")
METHODOLOGY = st.Page(page_methodology.render, title="Methodology",
                      icon=":material/menu_book:", url_path="methodology")

GROUPS: dict[str, list] = {
    "Workspace": [OVERVIEW, PHOTO, VIDEO, CAMERA],
    "Research": [DATASETS, TRAINING, RESULTS, METHODOLOGY],
}
