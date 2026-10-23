"""CrowdSense AI — Streamlit application shell (sectioned navigation).

Layout: branded sidebar → sectioned page navigation (st.navigation) →
live status → the selected page. `cleanup_resources()` runs after every
dispatch so camera/temp resources are released when the user navigates
away from the page that owns them.
"""

from __future__ import annotations

import streamlit as st

from src.ui import components, nav, state, theme
from src.ui.common import find_model_path, model_status_badge

_BRAND_HTML = """
<div class="crowdsense-brand">
  <span class="crowdsense-mark">C</span>
  <div>
    <div class="crowdsense-name">CrowdSense AI</div>
    <div class="crowdsense-sub">Crowd analytics research</div>
  </div>
</div>
"""


def _render_sidebar_status() -> None:
    """Live model status + active settings (every value is a real read)."""
    with st.sidebar:
        model_path = find_model_path()
        text, _emoji = model_status_badge(model_path)
        components.status_pill(text, "ok" if model_path else "warn")
        settings = state.settings()
        selected = settings.get("selected_model") or "newest bundle"
        st.caption(
            f"confidence ≥ {settings['conf_threshold']:.2f} · "
            f"stride {settings['frame_stride']} · inference: {selected}"
        )
        st.caption("Density · tracking · movement features "
                   "for crowd anomaly research (UCSD)")


def main() -> None:
    st.set_page_config(
        page_title="CrowdSense AI",
        page_icon=":material/monitoring:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    state.init_session()
    theme.inject_theme()

    with st.sidebar:
        st.markdown(_BRAND_HTML, unsafe_allow_html=True)

    current = st.navigation(nav.GROUPS, position="sidebar")
    st.session_state["active_page"] = current.url_path
    _render_sidebar_status()

    try:
        current.run()
    finally:
        state.cleanup_resources()


if __name__ == "__main__":
    main()
