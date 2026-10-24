"""Central navigation registry: page objects shared by the shell and CTAs.

The app shell builds `st.navigation` from `GROUPS`. Individual pages are
imported as modules; CTAs inside pages import this module lazily (inside
`render()`) to avoid import cycles.

Nav map: two groups only — Workspace (the four operator surfaces) and
System (Settings). Research tooling lives in the CLI scripts under
scripts/, not in this dashboard.
"""

from __future__ import annotations

import streamlit as st

from src.ui import (page_camera, page_overview, page_photo, page_settings,
                    page_video)

OVERVIEW = st.Page(page_overview.render, title="Overview", icon=":material/home:",
                   url_path="overview", default=True)
PHOTO = st.Page(page_photo.render, title="Photo Analysis", icon=":material/image:",
                url_path="photo-analysis")
VIDEO = st.Page(page_video.render, title="Video Analysis", icon=":material/video_library:",
                url_path="video-analysis")
CAMERA = st.Page(page_camera.render, title="Live Camera", icon=":material/videocam:",
                 url_path="camera")
SETTINGS = st.Page(page_settings.render, title="Settings",
                   icon=":material/tune:", url_path="settings")

GROUPS: dict[str, list] = {
    "Workspace": [OVERVIEW, PHOTO, VIDEO, CAMERA],
    "System": [SETTINGS],
}
