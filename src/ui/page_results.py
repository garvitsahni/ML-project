"""Research Results: renders ONLY files that exist under reports/ (never fabricated)."""

from __future__ import annotations

import json

import streamlit as st

from src.ui.common import FIGURES_DIR, RESULTS_DIR, list_result_files

COMMAND_HINTS = {
    "experiment_ablation.json": "python scripts/train_ml.py",
    "experiment_algorithms.json": "python scripts/train_ml.py",
    "exp_e_status.json": "python scripts/evaluate.py",
    "feature_table_meta.json": "python scripts/prepare_data.py",
}


def render() -> None:
    st.header("Research Results")
    st.caption(
        "Every value shown here was computed by this project's scripts on the UCSD "
        "dataset. Missing outputs show the exact command that produces them."
    )
    json_files = list_result_files(RESULTS_DIR, suffix=".json")
    if not json_files:
        st.warning("No results yet.")
        _show_all_hints()
        return

    for f in json_files:
        st.subheader(f.name)
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            st.error(f"Could not read {f.name}: {exc}")
            continue
        _render_payload(f.name, payload)

    figures = sorted(FIGURES_DIR.glob("*.png")) if FIGURES_DIR.is_dir() else []
    if figures:
        st.subheader("Figures (confusion / ROC / importance)")
        st.caption("All figures computed on the held-out TEST split.")
        for fig in figures:
            st.image(str(fig), caption=fig.name, width="content")


def _show_all_hints() -> None:
    for name, cmd in COMMAND_HINTS.items():
        st.code(f"# produces reports/results/{name}\n{cmd}", language="bash")


def _render_payload(name: str, payload: dict) -> None:
    if name == "exp_e_status.json":
        st.code(json.dumps(payload, indent=2), language="json")
        return
    if name.startswith("experiment_"):
        _render_experiment(name, payload)
        return
    with st.expander("raw JSON"):
        st.code(json.dumps(payload, indent=2)[:8000], language="json")


def _render_experiment(name: str, payload: dict) -> None:
    import pandas as pd

    if name == "experiment_ablation.json":
        st.caption("Ablation: does movement beat static-only? (TEST split, grouped by sequence)")
        rows = []
        for cond, algos in payload.get("per_condition", {}).items():
            for algo, splits in algos.items():
                m = splits.get("test", {})
                rows.append({"condition": cond, "algorithm": algo,
                             "f1": m.get("f1"), "roc_auc": m.get("roc_auc"),
                             "n_test": m.get("n_samples")})
    else:
        st.caption("Algorithm comparison on static+movement features (TEST split)")
        rows = []
        for algo, splits in payload.get("per_algorithm", {}).items():
            m = splits.get("test", {})
            rows.append({"algorithm": algo, "f1": m.get("f1"), "roc_auc": m.get("roc_auc"),
                         "n_test": m.get("n_samples")})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    seqs = payload.get("train_sequences") or payload.get("test_sequences")
    if seqs:
        with st.expander("split membership"):
            st.write("train:", payload.get("train_sequences"))
            st.write("test:", payload.get("test_sequences"))
