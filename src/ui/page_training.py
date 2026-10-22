"""Training & Evaluation: run real scripts as subprocesses with live log tails."""

from __future__ import annotations

import subprocess
import sys
import time

import streamlit as st

from src.config import PROJECT_ROOT
from src.ui.theme import page_header

STEPS = [
    ("Inspect dataset", [sys.executable, "scripts/inspect_dataset.py"]),
    ("Prepare feature table", [sys.executable, "scripts/prepare_data.py"]),
    ("Train models (A-D experiments)", [sys.executable, "scripts/train_ml.py"]),
    ("Evaluate", [sys.executable, "scripts/evaluate.py"]),
]


def render() -> None:
    page_header(
        "Training & Evaluation", "⚙️",
        "Each step runs the project's real CLI script in a subprocess and tails "
        "its stdout. Nothing here is simulated.",
    )
    for label, cmd in STEPS:
        with st.expander(label):
            st.code(" ".join(c if " " not in c else f'"{c}"' for c in cmd), language="bash")
            if st.button(f"Run: {label}", key=f"run_{label}"):
                _run_and_tail(cmd)


def _run_and_tail(cmd: list[str]) -> None:
    log = st.empty()
    lines: list[str] = []
    start = time.time()
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line.rstrip())
            log.code("\n".join(lines[-30:]), language="log")
        code = proc.wait(timeout=10)
    except FileNotFoundError as exc:
        st.error(f"Script missing: {exc}")
        return
    except subprocess.TimeoutExpired:
        proc.kill()
        st.error("Step timed out after kill — see log above.")
        return
    elapsed = time.time() - start
    if code == 0:
        st.success(f"Finished in {elapsed:.0f}s — results under reports/. See Research Results.")
    else:
        st.error(f"Exited with code {code} after {elapsed:.0f}s — fix and rerun.")
