"""Settings: validated session preferences + an honest diagnostics panel.

The form writes through `state.update_settings` (range-checked, defaults on
error); visualization toggles feed the existing overlay reads. Diagnostics
expose only whitelisted facts — versions, device, and file presence — never
paths, environment values or secrets.
"""

from __future__ import annotations

import platform

import streamlit as st

from src.config import load_paths
from src.ui import components as ui
from src.ui import state
from src.ui.common import RESULTS_DIR, find_model_path

_HELP = {
    "conf_threshold": ("Minimum YOLO confidence for a detection to count as "
                       "a person. Higher = fewer, more certain detections. "
                       "Affects Photo, Video and Camera."),
    "frame_stride": ("Process every Nth video frame (Video Analysis). "
                     "1 = every frame; higher = faster with coarser "
                     "timelines."),
    "max_frames": ("Hard cap on frames processed per video run (Video "
                   "Analysis)."),
    "process_max_side": ("Long side of the processing resolution (Video and "
                         "Camera). Lower = faster on CPU."),
    "show_density": ("Draw density heatmap overlays where available (Photo "
                     "Analysis, Camera grid)."),
}

_SIDES = [640, 1280, 1920]

_DIAG_LABELS = {
    "python": "Python",
    "streamlit": "Streamlit",
    "torch_device": "Torch device",
    "dataset_present": "UCSD dataset",
    "model_present": "Model bundle",
    "results_present": "Evaluation results",
}


# --------------------------------------------------------------------------
# pure helpers (unit-tested)
# --------------------------------------------------------------------------

def diagnostics() -> dict:
    """Whitelisted environment facts — versions, device, file presence.

    Deliberately excludes paths, environment variables and anything else
    that could leak machine detail.
    """
    try:
        import torch
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        torch_device = "unavailable"
    return {
        "python": platform.python_version(),
        "streamlit": st.__version__,
        "torch_device": torch_device,
        "dataset_present": load_paths().ucsd_dir.is_dir(),
        "model_present": find_model_path() is not None,
        "results_present": (RESULTS_DIR / "evaluation.json").is_file(),
    }


def _format_diag(value) -> str:
    if isinstance(value, bool):
        return "present" if value else "absent"
    return str(value)


# --------------------------------------------------------------------------
# render pieces
# --------------------------------------------------------------------------

def _render_form() -> None:
    current = state.settings()
    side = int(current["process_max_side"])
    with ui.card("Analysis preferences",
                 hint="Validated on save; applied to the next analysis. "
                      "Session-only — not persisted across browser "
                      "sessions."):
        with st.form("settings_form"):
            conf = st.slider("Detection confidence", 0.05, 0.95,
                             float(current["conf_threshold"]), step=0.05,
                             help=_HELP["conf_threshold"])
            stride = st.slider("Frame stride", 1, 10,
                               int(current["frame_stride"]), step=1,
                               help=_HELP["frame_stride"])
            max_frames = st.slider("Max frames per video", 60, 1200,
                                   int(current["max_frames"]), step=10,
                                   help=_HELP["max_frames"])
            st.selectbox("Processing resolution (long side px)", _SIDES,
                         index=_SIDES.index(side) if side in _SIDES else 1,
                         help=_HELP["process_max_side"],
                         key="settings_side")
            st.markdown("**Visualization**")
            st.checkbox("Show density overlays",
                        value=bool(current["show_density"]),
                        help=_HELP["show_density"],
                        key="settings_density")
            saved = st.form_submit_button("Save settings")
        if saved:
            clean, errors = state.update_settings({
                "conf_threshold": conf,
                "frame_stride": stride,
                "max_frames": max_frames,
                "process_max_side": st.session_state.get(
                    "settings_side", side),
                "show_density": bool(st.session_state.get(
                    "settings_density", current["show_density"])),
            })
            if errors:
                for message in errors:
                    st.error(message)
            else:
                st.success("Settings saved — they apply to the next "
                           "analysis.")


def _render_reset() -> None:
    if st.button("Reset to defaults", key="settings_reset"):
        state.update_settings(dict(state.DEFAULT_SETTINGS))
        st.rerun()


def _render_diagnostics() -> None:
    with ui.card("Diagnostics",
                 hint="Whitelisted facts only — versions, device, and "
                      "whether key files exist. No paths, no secrets, no "
                      "environment values."):
        diag = diagnostics()
        ui.kv_table([(label, _format_diag(diag[key]))
                     for key, label in _DIAG_LABELS.items()])


# --------------------------------------------------------------------------
# page entrypoint
# --------------------------------------------------------------------------

def render() -> None:
    ui.page_header(
        title="Settings",
        description=("Session preferences for detection and processing — "
                     "validated on save, applied to the next analysis."),
    )
    _render_form()
    _render_reset()
    _render_diagnostics()
