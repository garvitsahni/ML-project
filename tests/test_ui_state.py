"""Session-state store tests: settings defaults/validation, history, cleanup.

`src/ui/state.py` accepts an injectable mapping so these tests run without
a Streamlit runtime; production callers pass nothing (defaults to
`st.session_state`).
"""

from datetime import datetime

import pytest

from src.ui import state


def test_init_session_populates_defaults_and_is_idempotent():
    store: dict = {}
    state.init_session(store)
    s = state.settings(store)
    assert s["conf_threshold"] == 0.25
    assert s["frame_stride"] == 1
    assert s["max_frames"] == 300
    assert s["process_max_side"] == 1280
    assert s["selected_model"] is None
    assert s["show_density"] is True
    # second call must not clobber user changes
    s["conf_threshold"] = 0.5
    state.init_session(store)
    assert state.settings(store)["conf_threshold"] == 0.5


def test_settings_auto_initializes_from_bare_store():
    store: dict = {}
    assert state.settings(store)["frame_stride"] == 1


def test_update_settings_accepts_valid_values():
    store: dict = {}
    clean, errors = state.update_settings(
        {"conf_threshold": 0.4, "frame_stride": 3, "max_frames": 600,
         "process_max_side": 1920, "show_density": False},
        store,
    )
    assert errors == []
    assert clean["conf_threshold"] == 0.4
    assert state.settings(store)["frame_stride"] == 3
    assert state.settings(store)["show_density"] is False


def test_validate_settings_rejects_out_of_range_uses_defaults():
    clean, errors = state.validate_settings(
        {"conf_threshold": 1.5, "frame_stride": 0, "max_frames": 99999,
         "process_max_side": 777, "selected_model": None, "show_density": True},
    )
    assert clean["conf_threshold"] == 0.25
    assert clean["frame_stride"] == 1
    assert clean["max_frames"] == 300
    assert clean["process_max_side"] == 1280
    assert len(errors) == 4
    joined = " ".join(errors).lower()
    assert "confidence" in joined
    assert "stride" in joined
    assert "frames" in joined
    assert "resolution" in joined


def test_validate_settings_ignores_unknown_keys():
    clean, errors = state.validate_settings({"bogus": 1})
    assert errors == []
    assert "bogus" not in clean


def test_record_analysis_caps_history_and_orders_newest_first():
    store: dict = {}
    state.init_session(store)
    t0 = datetime(2026, 10, 23, 12, 0, 0)
    for i in range(55):
        state.record_analysis("photo", {"count": i}, store,
                              now=datetime(2026, 10, 23, 12, i % 60, i // 60))
    history = state.get_history(store)
    assert len(history) == 50
    assert history[0]["count"] == 54  # newest first
    assert history[-1]["count"] == 5  # oldest kept after cap
    assert history[0]["kind"] == "photo"
    assert history[0]["ts"]  # ISO timestamp present


def test_clear_history():
    store: dict = {}
    state.init_session(store)
    state.record_analysis("video", {}, store)
    assert state.get_history(store)
    state.clear_history(store)
    assert state.get_history(store) == []


def test_release_camera_releases_cap_and_clears_flags():
    class FakeCap:
        released = False

        def release(self):
            self.released = True

    cap = FakeCap()
    store = {"camera_cap": cap, "camera_running": True}
    state.release_camera(store)
    assert cap.released
    assert "camera_cap" not in store
    assert store["camera_running"] is False
    # no camera present: no-op, no error
    state.release_camera(store)


def test_cleanup_resources_releases_camera_only_off_the_camera_page():
    class FakeCap:
        def __init__(self):
            self.released = False

        def release(self):
            self.released = True

    away = FakeCap()
    store = {"camera_cap": away, "camera_running": True,
             "active_page": "photo-analysis"}
    state.cleanup_resources(store)
    assert away.released
    assert "camera_cap" not in store

    on_page = FakeCap()
    store2 = {"camera_cap": on_page, "active_page": state.CAMERA_URL}
    state.cleanup_resources(store2)
    assert not on_page.released  # page still owns the camera


def test_cleanup_resources_runs_registered_hook():
    calls = []
    store: dict = {"_cleanup_hook": lambda: calls.append("hook"),
                   "active_page": "overview"}
    state.cleanup_resources(store)
    assert calls == ["hook"]


def test_cleanup_resources_tolerates_empty_store():
    state.cleanup_resources({})  # must not raise
