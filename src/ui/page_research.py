"""Research Lab: run the real experiment pipeline, read real result files.

Folds the legacy Methodology + Training pages into one research surface:
reproducible subprocess runners with live log tails and Cancel, ablation and
algorithm charts built strictly from reports/results/*.json, the gated
Experiment E status, figure gallery, and the methodology/limitations text as
a single source. Nothing here fabricates numbers — absent files render empty
states with the exact command that would produce them.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st

from src.config import PROJECT_ROOT
from src.ui import components as ui
from src.ui.charts import grouped_bars
from src.ui.common import FIGURES_DIR, RESULTS_DIR, limitations

# (key, label, command, requires confirmation before running)
_RUNNERS: list[tuple[str, str, list[str], bool]] = [
    ("inspect", "Inspect dataset",
     [sys.executable, "scripts/inspect_dataset.py"], False),
    ("prepare", "Prepare feature table",
     [sys.executable, "scripts/prepare_data.py"], True),
    ("train", "Train models (experiments A–D)",
     [sys.executable, "scripts/train_ml.py"], True),
    ("evaluate", "Evaluate bundles",
     [sys.executable, "scripts/evaluate.py"], False),
]

_RESULT_FILES: dict[str, str] = {
    "ablation": "experiment_ablation.json",
    "algorithms": "experiment_algorithms.json",
    "evaluation": "evaluation.json",
    "exp_e": "exp_e_status.json",
}

_METHODOLOGY_MD = """
**Features (15 total, canonical schema in `src/features/schema.py`):**
- *Static (6):* person_count, density_per_mpx, grid_concentration, spread_x, spread_y, grid_entropy.
- *Movement (9):* density_delta, mean_speed_pxf, speed_std_pxf, direction_entropy,
directional_consistency, flow_mean_magnitude, flow_std_pxf, track_count, track_persistence.

**Split policy:** grouped by sequence id (never random frames). A sequence appears in
exactly one of train/val/test; `assert_group_disjoint` verifies at runtime.

**Window label policy:** a window is anomalous if ANY frame inside it is anomalous
(UCSD frame-level ground truth). Partial trailing windows are dropped.

**Units:** movement features are px/frame at processed resolution — never m/s
(scene scale is unknown). Detector counts are YOLO estimates, not ground truth.
"""

_PROVENANCE_MD = """
- Dataset: **UCSD Anomaly Detection Dataset v1p2** (Mahadevan et al., CVPR 2010) —
placed manually under `data/raw/ucsd_anomaly/` (never auto-downloaded).
- Citation: V. Mahadevan, W. Li, N. Boccali, A. K. Roy-Chowdhury,
"Anomaly Detection in Crowded Scenes," CVPR 2010.
- UCF-QNRF and ShanghaiTech were **dropped from scope** (anomaly experiments only);
counting MAE/RMSE is therefore not reported anywhere in this project.
"""


# --------------------------------------------------------------------------
# pure helpers (unit-tested)
# --------------------------------------------------------------------------

def load_experiment_results(results_dir: Path) -> dict:
    """Parsed experiment JSONs; only files that exist are included.

    Returns {} when nothing exists; a corrupt file is reported under
    "errors" instead of raising — its data is never fabricated.
    """
    results_dir = Path(results_dir)
    out: dict[str, Any] = {}
    errors: list[str] = []
    for key, filename in _RESULT_FILES.items():
        path = results_dir / filename
        if not path.is_file():
            continue
        try:
            out[key] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append(f"{filename}: {exc}")
    if errors:
        out["errors"] = errors
    return out


def build_findings(results: dict) -> list[str]:
    """Findings generated ONLY from numbers actually present in `results`.

    Missing or non-numeric metrics contribute nothing; returns [] when there
    is not enough data to state anything.
    """
    findings: list[str] = []

    per_algorithm = ((results.get("algorithms") or {})
                     .get("per_algorithm") or {})
    auc_by_algo: dict[str, float] = {}
    for name, entry in per_algorithm.items():
        auc = ((entry or {}).get("test") or {}).get("roc_auc")
        if isinstance(auc, (int, float)):
            auc_by_algo[name] = float(auc)
    if auc_by_algo:
        best = max(auc_by_algo, key=lambda k: auc_by_algo[k])
        findings.append(f"Best algorithm by test ROC AUC: {best} "
                        f"({auc_by_algo[best]:.3f}).")

    per_condition = ((results.get("ablation") or {})
                     .get("per_condition") or {})
    a_rows = per_condition.get("A_static_only") or {}
    c_rows = per_condition.get("C_static_plus_movement") or {}
    deltas: list[tuple[str, float]] = []
    for algo, c_entry in c_rows.items():
        c_auc = ((c_entry or {}).get("test") or {}).get("roc_auc")
        a_auc = ((a_rows.get(algo) or {}).get("test") or {}).get("roc_auc")
        if isinstance(c_auc, (int, float)) and isinstance(a_auc, (int, float)):
            deltas.append((algo, float(c_auc) - float(a_auc)))
    if deltas:
        mean_delta = sum(d for _, d in deltas) / len(deltas)
        best_algo, best_delta = max(deltas, key=lambda t: t[1])
        findings.append(
            f"Movement features change test ROC AUC vs static-only by "
            f"{mean_delta:+.3f} on average across {len(deltas)} algorithms "
            f"(largest: {best_algo} {best_delta:+.3f}).")

    evaluation = results.get("evaluation") or {}
    n_test = evaluation.get("n_test_windows")
    sequences = evaluation.get("test_sequences")
    if isinstance(n_test, (int, float)) and isinstance(sequences, list) \
            and sequences:
        findings.append(f"Evaluation covers {n_test} test windows across "
                        f"{len(sequences)} held-out sequences.")
    return findings


_TABLE_COLUMNS = ["experiment", "condition", "algorithm", "split",
                  "accuracy", "precision", "recall", "f1", "roc_auc",
                  "n_samples"]


def experiment_table(results: dict):
    """Long-format DataFrame of every experiment/split metric present."""
    import pandas as pd

    rows: list[dict] = []

    def add(experiment: str, condition: str, algorithm: str, split: str,
            metrics: Any) -> None:
        if not isinstance(metrics, dict):
            return
        rows.append({
            "experiment": experiment, "condition": condition,
            "algorithm": algorithm, "split": split,
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"), "f1": metrics.get("f1"),
            "roc_auc": metrics.get("roc_auc"),
            "n_samples": metrics.get("n_samples"),
        })

    per_algorithm = ((results.get("algorithms") or {})
                     .get("per_algorithm") or {})
    for algo, entry in per_algorithm.items():
        entry = entry or {}
        for split in ("val", "test"):
            add("D_algorithms", "", algo, split, entry.get(split))
    per_condition = ((results.get("ablation") or {})
                     .get("per_condition") or {})
    for condition, condition_rows in per_condition.items():
        for algo, entry in (condition_rows or {}).items():
            entry = entry or {}
            for split in ("val", "test"):
                add("ablation", condition, algo, split, entry.get(split))
    bundles = ((results.get("evaluation") or {}).get("bundles") or {})
    for algo, bundle in bundles.items():
        add("evaluation", "bundles", algo, "test",
            (bundle or {}).get("test_metrics"))
    return pd.DataFrame(rows, columns=_TABLE_COLUMNS)


def load_experiment_config(path: Path | None = None) -> dict:
    """configs/experiments.yaml as a dict; {} when missing/corrupt."""
    path = path or PROJECT_ROOT / "configs" / "experiments.yaml"
    if not Path(path).is_file():
        return {}
    try:
        import yaml

        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def load_window_meta(path: Path | None = None) -> dict:
    """data/processed/ucsd_windows.meta.json as a dict; {} when missing."""
    path = path or PROJECT_ROOT / "data" / "processed" / "ucsd_windows.meta.json"
    if not Path(path).is_file():
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _log_tail(path: Path, n: int = 30) -> list[str]:
    """Last n lines of a log file (reads at most the final 64 KB)."""
    try:
        with Path(path).open("r", encoding="utf-8", errors="replace") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 65536))
            chunk = fh.read()
    except OSError:
        return []
    lines = chunk.splitlines()
    if size > 65536 and lines:
        lines = lines[1:]  # drop the possibly-partial first line
    return lines[-n:]


# --------------------------------------------------------------------------
# subprocess runners
# --------------------------------------------------------------------------

def _start_run(key: str, label: str, cmd: list[str]) -> None:
    """Launch one CLI step; stdout+stderr go to a log file (no PIPE stalls).

    Refuses to start while another run is active — one run at a time.
    """
    existing = st.session_state.get("research_run")
    if existing and existing.get("phase") == "running":
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / f"run_{key}.log"
    handle = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd, cwd=str(PROJECT_ROOT), stdout=handle,
        stderr=subprocess.STDOUT)
    st.session_state["research_run"] = {
        "key": key, "label": label, "cmd": list(cmd), "proc": proc,
        "started": time.time(), "log": log_path, "handle": handle,
        "phase": "running", "code": None, "duration": None,
    }


def _finalize_run(run: dict, code: int | None) -> None:
    handle = run.pop("handle", None)
    if handle is not None:
        try:
            handle.close()
        except Exception:
            pass
    run["code"] = code
    run["duration"] = time.time() - run["started"]
    if code is None:
        run["phase"] = "cancelled"
    elif code == 0:
        run["phase"] = "done"
    else:
        run["phase"] = "failed"


def _cancel_run() -> None:
    """Terminate the active run (real SIGTERM-equivalent), then finalize."""
    run = st.session_state.get("research_run")
    if not run or run.get("phase") != "running":
        return
    proc = run["proc"]
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    _finalize_run(run, proc.returncode)
    run["phase"] = "cancelled"  # the user's action defines the phase, not
    # the OS exit code the termination produced


@st.fragment(run_every="1s")
def _runner_panel() -> None:
    """Live status of the active run: elapsed, log tail, Cancel."""
    run = st.session_state.get("research_run")
    if not run:
        return
    if run["phase"] == "running":
        code = run["proc"].poll()
        if code is not None:
            _finalize_run(run, code)
    phase = run["phase"]
    elapsed = (run["duration"] if run["duration"] is not None
               else time.time() - run["started"])
    with ui.card(f"Run: {run['label']}",
                 hint="Executed exactly as shown; log tail is live."):
        st.code(" ".join(run["cmd"]), language="bash")
        status_col, time_col, action_col = st.columns([3, 1, 1])
        with status_col:
            if phase == "running":
                st.caption("● running…")
            elif phase == "done":
                st.success(f"Finished in {elapsed:.0f} s (exit 0).")
            elif phase == "cancelled":
                st.warning(f"Cancelled after {elapsed:.0f} s "
                           "(process terminated).")
            else:
                st.error(f"Failed with exit code {run['code']} after "
                         f"{elapsed:.0f} s — see the log tail below.")
        with time_col:
            st.metric("Elapsed", f"{elapsed:.0f} s")
        with action_col:
            if phase == "running":
                if st.button("Cancel run", key="research_cancel"):
                    _cancel_run()
                    st.rerun()
            else:
                if st.button("Clear panel", key="research_clear"):
                    st.session_state.pop("research_run", None)
                    st.rerun()
        tail = _log_tail(run["log"])
        if tail:
            st.code("\n".join(tail), language="log")
        else:
            st.caption("No log output yet.")


def _runner_card(key: str, label: str, cmd: list[str], confirm: bool,
                 *, busy: bool) -> None:
    hint = ("Rewrites result files under reports/results/."
            if confirm else "Read-only inspection/evaluation step.")
    with ui.card(label, hint=hint):
        st.code(" ".join(cmd), language="bash")
        confirmed = True
        if confirm:
            confirmed = st.checkbox(
                "Confirm — overwrite result files",
                key=f"research_confirm_{key}")
        if st.button(f"Run: {label}", key=f"research_run_{key}",
                     disabled=busy or not confirmed, width="stretch"):
            _start_run(key, label, cmd)


# --------------------------------------------------------------------------
# render pieces
# --------------------------------------------------------------------------

def _render_question() -> None:
    with ui.card("Research question",
                 hint="What experiments A–D actually test."):
        st.markdown(
            "Can classical ML on detector-derived crowd features (static "
            "density + optical-flow movement) discriminate anomalous vs "
            "normal video windows on the UCSD Ped1 anomaly dataset — and "
            "which feature family carries the signal?\n\n"
            "Scope: window-level binary anomaly classification "
            "(experiments A–D). Experiment E (future-congestion "
            "prediction) is gated — no validated labels exist.")


def _render_configuration() -> None:
    cfg = load_experiment_config()
    meta = load_window_meta()
    with ui.card("Experiment configuration",
                 hint="From configs/experiments.yaml — the reproducibility "
                      "contract for every run."):
        if not cfg:
            ui.empty_state(
                "No configuration found",
                "configs/experiments.yaml is missing — restore it to see "
                "the experiment contract.")
        else:
            feats = cfg.get("features") or {}
            split = cfg.get("split") or {}
            exp = cfg.get("experiments") or {}
            ui.kv_table([
                ("seed", cfg.get("seed")),
                ("window / stride",
                 f"{feats.get('window_frames')} / "
                 f"{feats.get('window_stride')} frames"),
                ("spatial grid",
                 f"{feats.get('grid_size')}×{feats.get('grid_size')}"),
                ("algorithms", ", ".join(exp.get("algorithms") or [])),
                ("split (val / test)",
                 f"{split.get('val_frac')} / {split.get('test_frac')} "
                 "of sequences (grouped)"),
            ])
        if meta:
            labels = meta.get("label_counts") or {}
            splits = meta.get("split_sequence_counts") or {}
            st.caption(
                f"Built table: {meta.get('rows')} windows from "
                f"{meta.get('sequences')} sequences · labels "
                f"{labels.get('0', 0)} normal / {labels.get('1', 0)} "
                f"anomalous · sequences train/test "
                f"{splits.get('train', '?')}/{splits.get('test', '?')} "
                "(no sequence spans two splits).")
        else:
            st.caption("No built feature table yet — run Prepare to create "
                       "data/processed/ucsd_windows.parquet.")


def _render_run_controls() -> None:
    st.markdown("### Run experiments")
    ui.data_note(
        "Each step executes the project's real CLI script in a subprocess "
        "(log: reports/results/run_<step>.log). One run at a time.")
    running = ((st.session_state.get("research_run") or {})
               .get("phase") == "running")
    grid = st.columns(2)
    for i, (key, label, cmd, confirm) in enumerate(_RUNNERS):
        with grid[i % 2]:
            _runner_card(key, label, cmd, confirm, busy=running)


def _render_metric_charts(df, *, x: str, group: str, title_prefix: str,
                          columns) -> None:
    with columns[0]:
        st.altair_chart(grouped_bars(
            df, x=x, y="roc_auc", group=group,
            title=f"{title_prefix} — test ROC AUC", y_label="ROC AUC",
            unit="score (0–1)", y_domain=[0.0, 1.0]), width="stretch")
    with columns[1]:
        st.altair_chart(grouped_bars(
            df, x=x, y="f1", group=group,
            title=f"{title_prefix} — test F1", y_label="F1",
            unit="score (0–1)", y_domain=[0.0, 1.0]), width="stretch")


def _render_ablation(results: dict, table) -> None:
    per_condition = ((results.get("ablation") or {})
                     .get("per_condition") or {})
    with ui.card("Experiments A/B/C — feature ablation",
                 hint="Test metrics per feature set (A: static only, "
                      "B: movement only, C: static + movement) — source: "
                      "reports/results/experiment_ablation.json."):
        if not per_condition:
            st.caption("No ablation results on disk.")
            st.code("python scripts/train_ml.py", language="bash")
            return
        df = table[(table["experiment"] == "ablation")
                   & (table["split"] == "test")]
        _render_metric_charts(df, x="algorithm", group="condition",
                              title_prefix="Ablation",
                              columns=st.columns(2))


def _render_algorithms(results: dict, table) -> None:
    per_algorithm = ((results.get("algorithms") or {})
                     .get("per_algorithm") or {})
    with ui.card("Experiment D — algorithm comparison",
                 hint="Val vs test metrics per algorithm (grouped split) — "
                      "source: reports/results/experiment_algorithms.json."):
        if not per_algorithm:
            st.caption("No algorithm results on disk.")
            st.code("python scripts/train_ml.py", language="bash")
            return
        df = table[table["experiment"] == "D_algorithms"]
        _render_metric_charts(df, x="algorithm", group="split",
                              title_prefix="Algorithms",
                              columns=st.columns(2))


def _render_evaluation_table(results: dict) -> None:
    import pandas as pd

    evaluation = results.get("evaluation") or {}
    bundles = evaluation.get("bundles") or {}
    with ui.card("Trained bundles — evaluation.json",
                 hint="Test metrics of the saved bundles the dashboards "
                      "load from models/."):
        if not bundles:
            st.caption("No evaluation results on disk.")
            st.code("python scripts/evaluate.py", language="bash")
            return
        rows = []
        for algo, bundle in sorted(bundles.items()):
            m = (bundle or {}).get("test_metrics") or {}
            rows.append([
                algo,
                ui.format_pct(m.get("accuracy")) if m.get("accuracy")
                is not None else "—",
                ui.format_pct(m.get("precision")) if m.get("precision")
                is not None else "—",
                ui.format_pct(m.get("recall")) if m.get("recall")
                is not None else "—",
                ui.format_pct(m.get("f1")) if m.get("f1") is not None
                else "—",
                f"{m['roc_auc']:.3f}" if isinstance(m.get("roc_auc"),
                                                    (int, float)) else "—",
                bundle.get("n_train", "—"),
                bundle.get("trained_at", "—"),
            ])
        st.dataframe(pd.DataFrame(rows, columns=[
            "algorithm", "accuracy", "precision", "recall", "f1",
            "ROC AUC", "n_train", "trained_at"]),
            hide_index=True, width="stretch")
        n_test = evaluation.get("n_test_windows")
        sequences = evaluation.get("test_sequences") or []
        if n_test is not None:
            ui.data_note(f"Scored on {n_test} test windows across "
                         f"{len(sequences)} held-out sequences "
                         "(file: reports/results/evaluation.json).")


def _render_exp_e(results: dict) -> None:
    exp_e = results.get("exp_e")
    with ui.card("Experiment E — prediction horizons",
                 hint="Gated experiment status — source: "
                      "reports/results/exp_e_status.json."):
        if not exp_e:
            st.caption("No status file on disk.")
            st.code("python scripts/evaluate.py", language="bash")
        elif exp_e.get("status") == "not_supported":
            st.info("Gated — no congestion labels: "
                    f"{exp_e.get('reason')}")
        else:
            st.error(f"Unexpected status '{exp_e.get('status')}' — "
                     "investigate before trusting any E result.")
            st.code(json.dumps(exp_e, indent=2), language="json")


_FIGURE_ALGOS = ["logreg", "dt", "rf", "svm", "xgb"]
_FIGURE_KINDS = ["confusion", "roc", "importance"]


def _parse_figure_name(name: str, prefix: str) -> tuple[str, str, str] | None:
    """'ablation_C_xgb_roc.png' -> ('C', 'xgb', 'roc'); None if unknown."""
    body = name[len(prefix):-len(".png")]
    for algo in _FIGURE_ALGOS:
        for kind in _FIGURE_KINDS:
            suffix = f"{algo}_{kind}"
            if body == suffix:
                return ("—", algo, kind)
            if body.endswith(f"_{suffix}"):
                return (body[:-(len(suffix) + 1)], algo, kind)
    return None


def _render_figures() -> None:
    import pandas as pd

    figures_dir = FIGURES_DIR
    names = (sorted(p.name for p in figures_dir.glob("*.png"))
             if figures_dir.is_dir() else [])
    with ui.card("Figure gallery",
                 hint="PNGs written by the training/eval scripts — the "
                      "file name is shown as provenance."):
        if not names:
            st.caption("No figures on disk.")
            st.code("python scripts/train_ml.py\npython scripts/evaluate.py",
                    language="bash")
            return
        ablation_names = [n for n in names if n.startswith("ablation_")]
        algo_names = [n for n in names if n.startswith("algorithms_")]
        families = ([("A/B/C ablation", "ablation_", ablation_names)]
                    if ablation_names else [])
        if algo_names:
            families.append(("D algorithms", "algorithms_", algo_names))
        family_label, prefix, family_names = st.selectbox(
            "Figure family", families, format_func=lambda t: t[0],
            key="research_fig_family")
        entries = []
        for name in family_names:
            parsed = _parse_figure_name(name, prefix)
            if parsed:
                entries.append({"condition": parsed[0],
                                "algorithm": parsed[1], "kind": parsed[2],
                                "file": name})
        df = pd.DataFrame(entries)
        if df.empty:
            st.caption("Figures present but none matched the known "
                       "naming scheme — rerun the training scripts.")
            return
        sel_cols = st.columns(3 if prefix.startswith("ablation") else 2)
        conditions = sorted(df["condition"].unique())
        idx = 0
        picked = {}
        if prefix.startswith("ablation"):
            picked["condition"] = sel_cols[idx].selectbox(
                "Condition", conditions, key="research_fig_cond")
            idx += 1
        picked["algorithm"] = sel_cols[idx].selectbox(
            "Algorithm", sorted(df["algorithm"].unique()),
            key="research_fig_algo")
        picked["kind"] = sel_cols[idx + 1].selectbox(
            "Type", sorted(df["kind"].unique()), key="research_fig_kind")
        match = df[
            (df["algorithm"] == picked["algorithm"])
            & (df["kind"] == picked["kind"])]
        if "condition" in picked:
            match = match[match["condition"] == picked["condition"]]
        if len(match):
            file_name = match.iloc[0]["file"]
            st.image(str(figures_dir / file_name), width="stretch",
                     caption=file_name)
        else:
            st.caption("That combination has no figure on disk — run "
                       "python scripts/train_ml.py to generate it.")


def _render_findings(results: dict) -> None:
    findings = build_findings(results)
    with ui.card("Findings",
                 hint="Auto-generated strictly from the numbers present in "
                      "reports/results/*.json — regenerate after each run."):
        if findings:
            for finding in findings:
                st.markdown(f"- {finding}")
        else:
            st.caption("Not enough data on disk to state any finding — run "
                       "the Train and Evaluate steps above.")


def _render_methodology() -> None:
    st.markdown("### Methodology & limitations")
    with ui.card("Methodology",
                 hint="How every number above was produced — the same "
                      "policies the pipeline enforces at runtime."):
        st.markdown(_METHODOLOGY_MD)
    with ui.card("Honest limitations"):
        ui.render_limitations(limitations())
    with ui.card("Data provenance"):
        st.markdown(_PROVENANCE_MD)


def _render_results() -> None:
    st.markdown("### Results")
    results = load_experiment_results(RESULTS_DIR)
    provenance = [f"reports/results/{name}" for key, name
                  in _RESULT_FILES.items() if key in results]
    if not results:
        ui.empty_state(
            "No experiment results yet",
            "The charts below appear once result files exist. Run the "
            "pipeline to produce them:")
        st.code("python scripts/train_ml.py    # experiments A–D\n"
                "python scripts/evaluate.py    # bundle metrics + status",
                language="bash")
        return
    ui.data_note("Loaded from: " + (", ".join(provenance)
                                    if provenance else "—"))
    table = experiment_table(results)
    _render_ablation(results, table)
    _render_algorithms(results, table)
    _render_evaluation_table(results)
    _render_exp_e(results)
    _render_figures()
    _render_findings(results)
    if results.get("errors"):
        st.warning("Some result files could not be read — "
                   + "; ".join(results["errors"]))


# --------------------------------------------------------------------------
# page entrypoint
# --------------------------------------------------------------------------

def render() -> None:
    ui.page_header(
        eyebrow="Research",
        title="Research Lab",
        description=(
            "Reproducible experiments on the UCSD window table: run the "
            "real CLI pipeline with live logs and Cancel, compare feature "
            "ablations and algorithms, and read the methodology with its "
            "honest limitations."),
    )
    _render_question()
    _render_configuration()
    _render_run_controls()
    if st.session_state.get("research_run"):
        _runner_panel()
    _render_results()
    _render_methodology()
