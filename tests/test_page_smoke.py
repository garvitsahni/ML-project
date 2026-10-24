"""Every page must render exception-free under AppTest (light UI regression net)."""

import pytest
from streamlit.testing.v1 import AppTest

PAGES = ["page_overview", "page_photo", "page_video", "page_camera",
         "page_settings"]


@pytest.mark.parametrize("name", PAGES)
def test_page_renders_exception_free(name):
    at = AppTest.from_string(f"from src.ui import {name}\n{name}.render()\n",
                             default_timeout=90)
    at.run()
    assert not at.exception, at.exception
    assert len(at.error) == 0, [e.value for e in at.error]
