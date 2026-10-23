"""Central navigation registry: page objects shared by the shell and CTAs.

The app shell builds `st.navigation` from `GROUPS`. Individual pages are
imported as modules; CTAs inside pages import this module lazily (inside
`render()`) to avoid import cycles.

Nav map (spec §4): destinations are only listed once implemented — legacy
Training & Evaluation and Methodology pages are folded into Research Lab;
Research Results is replaced by Reports in a later task.
"""

from __future__ import annotations

import streamlit as st

from src.ui import (page_camera, page_datasets, page_models, page_overview,
                    page_photo, page_research, page_results, page_video)

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
RESEARCH = st.Page(page_research.render, title="Research Lab",
                   icon=":material/science:", url_path="research")
MODELS = st.Page(page_models.render, title="Model Registry",
                 icon=":material/model_training:", url_path="models")
RESULTS = st.Page(page_results.render, title="Research Results",
                  icon=":material/assessment:", url_path="results")

GROUPS: dict[str, list] = {
    "Workspace": [OVERVIEW, PHOTO, VIDEO, CAMERA],
    "Research": [DATASETS, RESEARCH, MODELS, RESULTS],
}
