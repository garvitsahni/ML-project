"""Session-state store: settings, analysis history, camera/resource cleanup.

Every function takes an optional injectable `store` mapping so the logic is
unit-testable without a Streamlit runtime; production callers omit it and
operate on `st.session_state`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, MutableMapping

import streamlit as st

#: `url_path` of the Live Camera page — resources owned by that page are
#: only released while a different page is active.
CAMERA_URL = "camera"

#: `url_path` of the Video Analysis page — same ownership rule for the
#: video-file handle held during batched processing.
VIDEO_URL = "video-analysis"

HISTORY_LIMIT = 50

DEFAULT_SETTINGS: dict[str, Any] = {
    "conf_threshold": 0.25,   # detector confidence gate (0.05–0.95)
    "frame_stride": 1,        # process every Nth video frame (1–10)
    "max_frames": 300,        # video processing cap (60–1200)
    "process_max_side": 1280, # processing resolution: long side px
    "selected_model": None,   # inference bundle; None = newest available
    "show_density": True,     # density heatmap overlays
}

_RANGES: dict[str, tuple[float, float]] = {
    "conf_threshold": (0.05, 0.95),
    "frame_stride": (1, 10),
    "max_frames": (60, 1200),
}
_CHOICES: dict[str, tuple[int, ...]] = {
    "process_max_side": (640, 1280, 1920),
}
_LABELS: dict[str, str] = {
    "conf_threshold": "Detection confidence",
    "frame_stride": "Frame stride",
    "max_frames": "Max frames",
    "process_max_side": "Processing resolution",
}


def _resolve(store: MutableMapping[str, Any] | None) -> MutableMapping[str, Any]:
    if store is not None:
        return store
    return st.session_state  # type: ignore[return-value]


def init_session(store: MutableMapping[str, Any] | None = None) -> None:
    """Create default settings/history if absent. Never clobbers changes."""
    store = _resolve(store)
    if "settings" not in store:
        store["settings"] = dict(DEFAULT_SETTINGS)
    if "history" not in store:
        store["history"] = []


def settings(store: MutableMapping[str, Any] | None = None) -> dict[str, Any]:
    """Live settings dict (auto-initialized from defaults)."""
    store = _resolve(store)
    if "settings" not in store:
        store["settings"] = dict(DEFAULT_SETTINGS)
    return store["settings"]


def validate_settings(raw: Any) -> tuple[dict[str, Any], list[str]]:
    """Validate user input against the settings schema.

    Returns (clean settings, human-readable errors). Invalid fields fall
    back to their defaults; unknown keys are ignored.
    """
    clean = dict(DEFAULT_SETTINGS)
    errors: list[str] = []
    if not isinstance(raw, dict):
        return clean, ["Invalid settings input; defaults restored."]

    for key, default in DEFAULT_SETTINGS.items():
        if key not in raw:
            continue
        value = raw[key]
        if key in _RANGES:
            lo, hi = _RANGES[key]
            try:
                num = float(value)
            except (TypeError, ValueError):
                errors.append(
                    f"{_LABELS[key]} must be a number between {lo} and {hi}; "
                    f"using default {default}.")
                continue
            if not (lo <= num <= hi):
                errors.append(
                    f"{_LABELS[key]} must be between {lo} and {hi} "
                    f"(got {value}); using default {default}.")
                continue
            clean[key] = num if key == "conf_threshold" else int(num)
        elif key in _CHOICES:
            choices = _CHOICES[key]
            try:
                num = int(value)
            except (TypeError, ValueError):
                errors.append(
                    f"{_LABELS[key]} must be one of {list(choices)}; "
                    f"using default {default}.")
                continue
            if num not in choices:
                errors.append(
                    f"{_LABELS[key]} must be one of {list(choices)} "
                    f"(got {value}); using default {default}.")
                continue
            clean[key] = num
        elif key == "selected_model":
            # None or artifact path string; existence is checked against the
            # model registry at inference time, not here.
            clean[key] = value if isinstance(value, str) or value is None else None
        elif key == "show_density":
            clean[key] = bool(value)
    return clean, errors


def update_settings(raw: Any,
                    store: MutableMapping[str, Any] | None = None
                    ) -> tuple[dict[str, Any], list[str]]:
    """Validate and persist settings; returns (clean, errors)."""
    store = _resolve(store)
    clean, errors = validate_settings(raw)
    store["settings"] = clean
    return clean, errors


def record_analysis(kind: str, summary: dict[str, Any],
                    store: MutableMapping[str, Any] | None = None,
                    now: datetime | None = None) -> None:
    """Append one session-history entry (capped at HISTORY_LIMIT)."""
    store = _resolve(store)
    history: list = store.setdefault("history", [])
    ts = (now or datetime.now()).isoformat(timespec="seconds")
    entry: dict[str, Any] = {"ts": ts, "kind": kind}
    entry.update(summary)
    history.append(entry)
    if len(history) > HISTORY_LIMIT:
        del history[:len(history) - HISTORY_LIMIT]


def get_history(store: MutableMapping[str, Any] | None = None) -> list[dict]:
    """History entries, newest first."""
    store = _resolve(store)
    return list(reversed(store.get("history", [])))


def clear_history(store: MutableMapping[str, Any] | None = None) -> None:
    store = _resolve(store)
    store["history"] = []


def release_camera(store: MutableMapping[str, Any] | None = None) -> None:
    """Release the cv2 capture (if any) and clear camera flags. Idempotent."""
    store = _resolve(store)
    cap = store.pop("camera_cap", None)
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    store["camera_running"] = False


def release_video(store: MutableMapping[str, Any] | None = None) -> None:
    """Release the video-file capture (if any) and clear flags. Idempotent."""
    store = _resolve(store)
    cap = store.pop("video_cap", None)
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    store["video_running"] = False


def cleanup_resources(store: MutableMapping[str, Any] | None = None) -> None:
    """Release resources owned by pages that are no longer active.

    The shell calls this after dispatch, having set `active_page` to the
    page that just ran: still-open camera/video handles are released only
    when a different page is active. Registered cleanup hooks always run.
    """
    store = _resolve(store)
    if store.get("camera_cap") is not None and store.get("active_page") != CAMERA_URL:
        release_camera(store)
    if store.get("video_cap") is not None and store.get("active_page") != VIDEO_URL:
        release_video(store)
    hook = store.get("_cleanup_hook")
    if callable(hook):
        hook()


def register_cleanup(fn: Any,
                     store: MutableMapping[str, Any] | None = None) -> None:
    """Register a single cleanup hook (e.g. video temp-file removal)."""
    store = _resolve(store)
    store["_cleanup_hook"] = fn
