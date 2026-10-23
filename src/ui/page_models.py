"""Model Registry: real bundle artifacts, evaluation metrics, default-model pick.

Every field shown is read from the bundle artifact (joblib) or
reports/results/evaluation.json. Unreadable artifacts are listed as
unavailable with their reason — never dropped silently, never fabricated.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import streamlit as st

from src.ui import components as ui
from src.ui import state
from src.ui.common import (MODELS_DIR, RESULTS_DIR, format_bytes,
                           limitations)

_TASK_NOTE = ("anomaly-window classification — 15-feature, 30-frame window "
              "schema trained on UCSD Ped1 windows")


# --------------------------------------------------------------------------
# pure helpers (unit-tested)
# --------------------------------------------------------------------------

def list_bundles(models_dir: Path) -> list[dict]:
    """Registry entries for every *.joblib under models_dir (name-sorted).

    Missing optional fields inside a bundle are tolerated and surfaced as
    None/0/{} — values are never invented.
    """
    from src.models.bundle import ModelBundleError, load_bundle

    models_dir = Path(models_dir)
    entries: list[dict] = []
    if not models_dir.is_dir():
        return entries
    for path in sorted(models_dir.glob("*.joblib")):
        stat = path.stat()
        entry = {
            "name": path.stem,
            "path": str(path),
            "available": False,
            "algorithm": None,
            "trained_at": None,
            "n_train": 0,
            "metrics": {},
            "feature_names": [],
            "size_bytes": stat.st_size,
            "mtime": stat.st_mtime,
            "error": None,
        }
        try:
            bundle = load_bundle(path)
        except ModelBundleError as exc:
            entry["error"] = str(exc)
            entries.append(entry)
            continue
        entry.update({
            "available": True,
            "algorithm": bundle.algorithm,
            "trained_at": bundle.trained_at or None,
            "n_train": int(bundle.n_train),
            "metrics": dict(bundle.metrics or {}),
            "feature_names": list(bundle.feature_names),
        })
        entries.append(entry)
    return entries


def metrics_for(algo: str, evaluation_json: dict) -> dict | None:
    """test_metrics mapping for one algorithm; None when absent."""
    bundle = ((evaluation_json or {}).get("bundles") or {}).get(algo) or {}
    metrics = bundle.get("test_metrics")
    return dict(metrics) if isinstance(metrics, dict) and metrics else None


def resolve_selected_model(
        entries: list[dict], store=None) -> tuple[str | None, str | None]:
    """Validate settings['selected_model'] against the registry.

    Returns (selected_name, warning). A missing/invalid/unavailable pick
    falls back to the newest available bundle; nothing is trusted silently.
    """
    available = [e for e in entries if e.get("available")]
    newest = (max(available, key=lambda e: e["mtime"])["name"]
              if available else None)
    chosen = state.settings(store).get("selected_model")
    if chosen and any(e["name"] == chosen and e.get("available")
                      for e in entries):
        return chosen, None
    if not chosen:
        return newest, None
    if newest:
        return newest, (f"Selected model '{chosen}' is not an available "
                        f"bundle — defaulting to '{newest}'.")
    return None, (f"Selected model '{chosen}' is not an available bundle "
                  "and no trained bundle exists — inference disabled.")


def _load_evaluation() -> dict:
    path = RESULTS_DIR / "evaluation.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


# --------------------------------------------------------------------------
# render pieces
# --------------------------------------------------------------------------

def _render_selector(entries: list[dict], selected: str | None,
                     warning: str | None) -> None:
    available = [e["name"] for e in entries if e["available"]]
    with ui.card("Default inference model",
                 hint="Used by Video Analysis and Live Camera; a stale or "
                      "missing pick falls back to the newest bundle."):
        if not available:
            st.caption("No trained bundle on disk — inference stays off "
                       "until one exists.")
            st.code("python scripts/train_ml.py", language="bash")
            return
        if warning:
            st.warning(warning)

        def _persist() -> None:
            state.settings()["selected_model"] = st.session_state.get(
                "models_pick")

        idx = available.index(selected) if selected in available else 0

        def _label(name: str) -> str:
            algo = next((e["algorithm"] for e in entries if e["name"] == name),
                        None)
            return name if algo in (None, name) else f"{name} — {algo}"

        st.selectbox("Bundle", available, index=idx, key="models_pick",
                     on_change=_persist, format_func=_label)
        current = state.settings().get("selected_model")
        st.caption(
            f"Inference default right now: {selected}."
            + (" Saved in settings." if current == selected else
               " (settings has no valid saved pick — newest used.)"))


def _render_bundle_cards(entries: list[dict]) -> None:
    st.markdown("### Bundles on disk")
    if not entries:
        ui.empty_state("No bundle artifacts",
                       "models/ contains no *.joblib files.")
        st.code("python scripts/train_ml.py", language="bash")
        return
    for entry in entries:
        with ui.card(f"{entry['name']}.joblib", hint=_TASK_NOTE):
            if not entry["available"]:
                st.error(f"Unavailable — {entry['error']}")
                st.caption(
                    f"Artifact present ({format_bytes(entry['size_bytes'])}, "
                    "modified "
                    f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(entry['mtime']))}) "
                    "but unreadable or schema-incompatible; retrain to "
                    "replace it.")
                continue
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Algorithm", entry["algorithm"])
            m2.metric("Trained at", entry["trained_at"] or "unknown")
            m3.metric("Training windows", f"{entry['n_train']:,}")
            m4.metric("Artifact",
                      format_bytes(entry["size_bytes"]))
            st.caption(
                "modified "
                f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(entry['mtime']))}"
                " · training dataset: UCSD Ped1 sequence-grouped windows")
            val = (entry["metrics"] or {}).get("val") or {}
            if val:
                parts = [f"{label} {ui.format_pct(val[key])}"
                         for label, key in (("accuracy", "accuracy"),
                                            ("precision", "precision"),
                                            ("recall", "recall"),
                                            ("f1", "f1"))
                         if val.get(key) is not None]
                if isinstance(val.get("roc_auc"), (int, float)):
                    parts.append(f"ROC AUC {val['roc_auc']:.3f}")
                st.caption("Validation: " + (", ".join(parts) or "—"))
            else:
                st.caption("No validation metrics stored inside the bundle "
                           "— see evaluation.json below.")
            names = entry["feature_names"]
            st.caption(
                f"Feature schema: {len(names)} features — "
                + ", ".join(names[:6]) + ("…" if len(names) > 6 else ""))


def _render_evaluation(entries: list[dict]) -> None:
    import pandas as pd

    evaluation = _load_evaluation()
    bundles = evaluation.get("bundles") or {}
    with ui.card("Held-out test metrics (evaluation.json)",
                 hint="Reported by python scripts/evaluate.py on the "
                      "sequence-held-out test split."):
        if not bundles:
            st.caption("No evaluation.json on disk.")
            st.code("python scripts/evaluate.py", language="bash")
            return
        rows = []
        for algo, bundle in sorted(bundles.items()):
            metrics = metrics_for(algo, evaluation) or {}
            bundle = bundle or {}
            rows.append({
                "algorithm": algo,
                "accuracy": (ui.format_pct(metrics["accuracy"])
                             if metrics.get("accuracy") is not None
                             else "—"),
                "precision": (ui.format_pct(metrics["precision"])
                              if metrics.get("precision") is not None
                              else "—"),
                "recall": (ui.format_pct(metrics["recall"])
                           if metrics.get("recall") is not None else "—"),
                "f1": (ui.format_pct(metrics["f1"])
                       if metrics.get("f1") is not None else "—"),
                "ROC AUC": (f"{metrics['roc_auc']:.3f}"
                            if isinstance(metrics.get("roc_auc"),
                                          (int, float)) else "—"),
                "n_train": bundle.get("n_train", "—"),
                "trained_at": bundle.get("trained_at", "—"),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        ui.data_note("Scores are window-level classification metrics on "
                     "held-out sequences — model outputs, not verified "
                     "incident labels.")


def _render_compat() -> None:
    with ui.card("Where these bundles are used",
                 hint="Schema compatibility — honest about what cannot "
                      "use them."):
        st.markdown(
            "- **Video Analysis & Live Camera:** yes — both build "
            "30-frame windows with the same 15-feature schema the bundles "
            "were trained on.\n"
            "- **Photo Analysis:** no — a single image has no temporal "
            "window, so it uses density heuristics only (no bundle "
            "inference).\n"
            "- Bundles reject feature-width mismatches at load time "
            "(schema guard), so an incompatible row is refused, not "
            "scored.")
    with ui.card("Known limitations"):
        ui.render_limitations([
            "Bundles are trained on one dataset (UCSD Ped1) — scores on "
            "other cameras may mean little.",
            "Metrics are window-level classification; per-incident "
            "precision is not established.",
            "Class imbalance (anomalous windows are the minority) makes "
            "accuracy less informative than ROC AUC.",
            "The default-model pick is a UI preference, not a guarantee "
            "of better results — compare the metrics above.",
        ])


# --------------------------------------------------------------------------
# page entrypoint
# --------------------------------------------------------------------------

def render() -> None:
    ui.page_header(
        eyebrow="Research · models",
        title="Model Registry",
        description=(
            "Trained bundle artifacts on disk — what each one is, when it "
            "was trained, and which one the dashboards use by default. "
            "Every field is read from the artifact or evaluation.json."),
    )
    entries = list_bundles(MODELS_DIR)
    selected, warning = resolve_selected_model(entries)
    _render_selector(entries, selected, warning)
    _render_bundle_cards(entries)
    _render_evaluation(entries)
    _render_compat()
