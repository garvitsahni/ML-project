"""AppTest flow tests: widget gating that unit tests cannot reach."""

from streamlit.testing.v1 import AppTest

# Seed history once per AppTest session (re-running the script must not
# re-seed, or "clear" would never take effect).
_OVERVIEW = """
import streamlit as st
from src.ui import state
if not st.session_state.get("seeded"):
    st.session_state["seeded"] = True
    state.record_analysis("photo", {"persons": 5, "model": "yolov8n.pt",
                                    "seconds": 0.8})
from src.ui.page_overview import render
render()
"""


def test_overview_clear_requires_confirmation():
    at = AppTest.from_string(_OVERVIEW, default_timeout=90).run()
    assert not at.exception
    assert any(getattr(w, "label", "") == "Download history CSV"
               for w in at.get("download_button"))
    clear = [b for b in at.button if b.key == "activity_clear_go"]
    assert len(clear) == 1 and clear[0].disabled
    at.checkbox[0].set_value(True).run()
    clear = [b for b in at.button if b.key == "activity_clear_go"]
    assert not clear[0].disabled
    clear[0].click().run()
    assert not at.exception
    assert len(at.dataframe) == 0          # history gone after confirmed clear
