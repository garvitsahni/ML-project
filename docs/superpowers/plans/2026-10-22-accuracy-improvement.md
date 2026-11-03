# Accuracy Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise best-model held-out test AUC from 0.675 to ≥0.75 (F1 improved alongside) via detector calibration, richer movement features, grouped-CV tuning + threshold selection, and a soft-vote ensemble — plus a modest Streamlit UX pass.

**Architecture:** Schema v2 = full candidate feature superset computed in a single ~1–2h rebuild with a calibrated detector config; bundles record the feature subset they were trained on; all selection decisions (features, hyperparameters, threshold, ensemble membership) happen on train-side GroupKFold CV and the val split; the test split is evaluated exactly four times, pre-declared and provenance-logged.

**Tech Stack:** Python 3.13, Windows, CPU-only; sklearn (GridSearchCV/GroupKFold), xgboost, ultralytics YOLOv8n, OpenCV Farneback flow, Streamlit, pytest, joblib, pandas.

**Spec:** `docs/superpowers/specs/2026-10-21-accuracy-improvement-design.md` — the plan argues from the spec; executors read both.

## Global Constraints

- Python 3.13, Windows, CPU-only. `n_jobs=1` for any parallel-capable estimator (loky pools crash intermittently on Windows).
- Full test suite via `powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1` (pins native thread pools; plain pytest can flake on OpenMP).
- Honesty rules (project-wide, from README/spec): px/frame units only, never m/s; undefined metrics are `null`/`None`, never invented; no counting-GT claims (detector estimates stay labeled as such); grouped splits by `sequence_id` — frames from one sequence never span train/val/test; seed 42 everywhere.
- `FeatureSchema.names` remains the single ordering source for training, bundles, and inference.
- No new dependencies (sklearn GridSearchCV/GroupKFold already available via scikit-learn).
- All four pre-declared test-split evaluations (spec Goal 3) live in Task 11 only; no other task may evaluate on test.
- Commit after every task. Message convention: `feat: ...` / `test: ...` / `docs: ...`.

## Review Focus

Failure modes the spec implies but no single task's happy path exercises; each is pinned to the task that owns the code.

1. **Old schema-v1 bundles after the v2 rebuild** — loading must fail with a friendly "retrain" message, not a stack trace or silent misprediction. Pinned in Task 6 (InferenceService rejection test) and Task 3 (bundle threshold round-trip test).
2. **Degenerate windows** (0 detections, 1 track, all-zero flow) — every new feature must return 0.0, never NaN, never raise. Pinned in Task 1 (edge-case tests per feature).
3. **Group leakage inside tuning CV** — GroupKFold must receive `sequence_id` groups; a test asserts group-disjoint folds. Pinned in Task 5.
4. **Threshold overfitting / degenerate val** — empty val or single-class val must fall back to 0.5, and the selected threshold is clamped to [0.05, 0.95]. Pinned in Task 5.
5. **Tables missing `label_frac`** (old parquet) — the robustness experiment must raise a clear "rebuild required" error, not compute wrong numbers. Pinned in Task 10.

---

### Task 1: New movement features (schema v2 core)

**Files:**
- Modify: `src/features/movement.py`
- Modify: `src/features/schema.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: existing `TrajectoryStore` API (`track_ids()`, `trajectory(tid)` → (n,3) `[frame, cx, cy]`, `speeds(tid)` → 1-D px/frame, `total_frames`).
- Produces (all in `src/features/movement.py`, all take `(store: TrajectoryStore, track_ids: Sequence[int])` unless noted, all return `float` in [0,1] or px/frame as documented):
  - `nearest_neighbor_stats(store, track_ids, centers_by_frame: dict[int, np.ndarray], image_diagonal: float) -> tuple[float, float]` — `(nn_distance_mean, nn_distance_std)`: per frame with ≥2 centers, mean NN distance per center; window value = mean over frames; normalized by `image_diagonal` (0 when <2 centers in all frames).
  - `abrupt_stop_fraction(store, track_ids, median_window_speed: float) -> float` — track counts as abrupt stop iff it has ≥3 speed samples AND `max(speeds) >= 2.0 * max(median_window_speed, 1e-6)` AND `speeds[-1] <= 0.25 * speeds.max()`. Fraction over tracks with ≥3 samples (0.0 when none).
  - `direction_reversal_rate(store, track_ids) -> float` — count consecutive unit-step pairs with dot product < -0.36 (~110°); rate = total reversals / total consecutive-step pairs (0.0 when no pairs).
  - `fast_mover_fraction(store, track_ids, median_window_speed: float) -> float` — track is fast iff `mean_speed > 3.0 * max(median_window_speed, 1e-6)`; fraction over tracks with ≥1 speed sample.
  - `speed_p90(store, track_ids) -> float` — 90th percentile of pooled speeds across tracks (0.0 when empty).
  - `track_turnover_rate(store, track_ids, window_start: int, window_end: int) -> float` — (births + deaths) / window length; birth = track whose first observation frame > `window_start`, death = last observation frame < `window_end - 1`. Use `trajectory(tid)[0,0]` / `trajectory(tid)[-1,0]`.
- Module constants (exact values, at top of `movement.py`): `FAST_MOVER_K = 3.0`, `STOP_SPEED_FACTOR = 2.0`, `STOP_REL_SPEED = 0.25`, `REVERSAL_DOT = -0.36`, `MIN_SPEED_SAMPLES_FOR_STOP = 3`.

Schema v2 additions to `_MOVEMENT` in `src/features/schema.py` (append after existing nine, with limitation strings in existing style):

```python
("nn_distance_mean", "normalized 0-1", "mean nearest-neighbor distance / image diagonal; 0 with <2 centers"),
("nn_distance_std", "normalized 0-1", "frame-to-frame variability of mean NN distance"),
("abrupt_stop_fraction", "fraction 0-1", "tracks decelerating hard within window; 0 with <3 speed samples"),
("direction_reversal_rate", "fraction 0-1", "consecutive-step direction flips >110 deg per step pair"),
("fast_mover_fraction", "fraction 0-1", "tracks >3x scene median speed (intruder proxy)"),
("speed_p90", "px/frame", "90th percentile track speed; NEVER m/s"),
("track_turnover_rate", "tracks/frame", "track births+deaths per frame in window"),
```

- [ ] **Step 1: Write failing tests** in `tests/test_features.py` (follow existing synthetic-trajectory fixture style in that file): one test per function covering (a) the nominal case with hand-computed expected value, (b) the degenerate case returning 0.0 without raising (empty store; single track; <3 samples for stops; no centers for NN).
- [ ] **Step 2: Run to verify failure**
  Run: `python -m pytest tests/test_features.py -q`
  Expected: FAIL with `ImportError`/`AttributeError` for the new names.
- [ ] **Step 3: Implement** the six functions and constants in `src/features/movement.py`; append the seven specs to `_MOVEMENT` in `src/features/schema.py`.
- [ ] **Step 4: Run to verify pass**
  Run: `python -m pytest tests/test_features.py -q`
  Expected: PASS (all).
- [ ] **Step 5: Run the full suite** (schema change touches everything downstream)
  Run: `powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1`
  Expected: some failures in tests that hard-code the old 15-feature width — fix those tests to derive widths from `FeatureSchema.default().names` instead of literals (this is the intended adaptation; do not change production code to fit old literals).
- [ ] **Step 6: Commit**
  `git add src/features/movement.py src/features/schema.py tests/test_features.py tests/ && git commit -m "feat: schema v2 movement features (NN crowding, stops, reversals, fast movers, turnover)"`

---

### Task 2: Flow-direction inconsistency + FrameSignals extension

**Files:**
- Modify: `src/features/extract.py` (`FrameSignals`, `optical_flow_stats`, `extract_window_features`)
- Modify: `src/features/pipeline.py` (flow call site)
- Modify: `src/ui/page_video.py` (flow call site)
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: Task 1 functions; existing `FrameSignals` dataclass.
- Produces:
  - `FrameSignals` gains field `flow_dir_inconsistency: float = 0.0` (per-frame scalar, 0–1).
  - `optical_flow_stats(prev_gray, gray) -> tuple[float, float, float]` — now returns `(mean_magnitude, std_magnitude, dir_inconsistency)`. `dir_inconsistency` = 1 − |mean unit direction vector| over all flow pixels with magnitude > 0.5 px/frame (0.0 when no qualifying pixels; 0 when called with degenerate input).
  - `extract_window_features` writes `feats["flow_direction_inconsistency"]` = mean of the window's per-frame `flow_dir_inconsistency` values.
- Both existing call sites (`pipeline.py:101`, `page_video.py:76`) unpack the 3-tuple.

- [ ] **Step 1: Write failing tests** in `tests/test_features.py`: (a) `optical_flow_stats` on two synthetic frames with uniform translation → dir_inconsistency near 0; on random noise → meaningfully > 0; degenerate blank frames → 0.0; (b) `extract_window_features` includes `flow_direction_inconsistency` equal to the mean of provided `FrameSignals.flow_dir_inconsistency`.
- [ ] **Step 2: Run to verify failure**
  Run: `python -m pytest tests/test_features.py -q -k flow_dir`
  Expected: FAIL (field/return do not exist yet).
- [ ] **Step 3: Implement** the `FrameSignals` field, the 3-tuple return with unit-vector mean logic in `optical_flow_stats`, the window aggregation in `extract_window_features`, and update both call sites to unpack 3 values.
- [ ] **Step 4: Run to verify pass + no regressions**
  Run: `python -m pytest tests/test_features.py tests/test_pipeline.py tests/test_dashboard.py -q`
  Expected: PASS (all).
- [ ] **Step 5: Commit**
  `git add src/features/extract.py src/features/pipeline.py src/ui/page_video.py tests/test_features.py && git commit -m "feat: flow-direction inconsistency feature (per-frame + window aggregation)"`

---

### Task 3: `label_frac` column + bundle threshold field

**Files:**
- Modify: `src/features/pipeline.py` (`META_COLUMNS`, `process_sequence` row dict)
- Modify: `src/models/bundle.py` (`ModelBundle`, `save_bundle`, `load_bundle`)
- Modify: `src/inference/service.py` (`predict_row` threshold use)
- Test: `tests/test_pipeline.py`, `tests/test_inference.py`

**Interfaces:**
- Consumes: existing `labels[start:end]` numpy slice in `process_sequence`.
- Produces:
  - `META_COLUMNS = ["sequence_id", "part", "split", "window_start", "window_end", "label", "label_frac"]`; each row gains `"label_frac": float(labels[start:end].mean())`.
  - `ModelBundle` gains field `threshold: float = 0.5`; `save_bundle`/`load_bundle` persist it (missing key in an old file → 0.5).
  - `InferenceService.predict_row` labels `1` iff `score >= self.bundle.threshold` (instead of `model.predict`), keeping the same `InferenceResult` shape. `predict_proba`-less fallback keeps `model.predict` behavior.

- [ ] **Step 1: Write failing tests**: pipeline test asserts rows contain `label_frac` and that a window with 3/30 anomalous frames yields exactly 0.1; inference tests assert (a) a bundle with `threshold=0.9` labels a 0.85-score row as 0, (b) `threshold` defaults to 0.5 when loading a bundle dict without the key (simulate via saved file round-trip after removing key), (c) score exactly equal to threshold → label 1.
- [ ] **Step 2: Run to verify failure**
  Run: `python -m pytest tests/test_pipeline.py tests/test_inference.py -q`
  Expected: FAIL (new key/field absent).
- [ ] **Step 3: Implement** the row key, the dataclass field + persistence, and the threshold-based labeling in `InferenceService.predict_row`.
- [ ] **Step 4: Run to verify pass**
  Run: `python -m pytest tests/test_pipeline.py tests/test_inference.py -q`
  Expected: PASS (all).
- [ ] **Step 5: Commit**
  `git add src/features/pipeline.py src/models/bundle.py src/inference/service.py tests/test_pipeline.py tests/test_inference.py && git commit -m "feat: label_frac column + bundle decision threshold (default 0.5)"`

---

### Task 4: Detector imgsz support + calibration module + script

**Files:**
- Modify: `src/detection/detector.py` (`PersonDetector.__init__`, `detect`)
- Create: `src/detection/calibration.py`
- Create: `scripts/calibrate_detection.py`
- Test: `tests/test_detection.py`

**Interfaces:**
- Consumes: existing `PersonDetector`, `UcsdAdapter.iter_sequences(split="train")`, `AnomalySequence.frame_files`.
- Produces:
  - `PersonDetector.__init__(..., imgsz: int = 640)`; `detect()` passes `imgsz=self.imgsz` to `model.predict`.
  - `src/detection/calibration.py` (pure, testable):
    - `sample_frame_files(adapter, n_frames: int = 40, seed: int = 42) -> list[Path]` — evenly spread across train-split sequences, deterministic.
    - `project_rebuild_hours(mean_seconds_per_frame: float, n_frames: int = 18560) -> float`.
    - `select_config(results: list[dict], max_rebuild_hours: float = 2.0) -> dict` — among configs with `projected_rebuild_hours <= max_rebuild_hours`, pick highest `mean_detections_per_frame`; ties → faster `mean_seconds_per_frame`; raises `ValueError` if no config fits the budget. Each result dict: `{"conf": float, "imgsz": int, "mean_detections_per_frame": float, "mean_seconds_per_frame": float, "projected_rebuild_hours": float, "overlay_files": list[str]}`.
  - `scripts/calibrate_detection.py` — argparse `--conf-list 0.05,0.10,0.15,0.25 --imgsz-list 320,640,1280 --max-frames 40 --max-hours 2.0 --out reports/detection_calibration/`; runs the grid on sampled train frames, writes overlay images per config, writes `calibration.json` with all results + `"selected": <select_config result>` + the selection rule text, prints the selected config.

- [ ] **Step 1: Write failing tests** in `tests/test_detection.py`: (a) `select_config` picks highest recall within budget; (b) excludes over-budget configs; (c) tie-break toward faster config; (d) all over budget → `ValueError`; (e) `project_rebuild_hours` arithmetic; (f) `sample_frame_files` deterministic (same seed → same list, length == n_frames, all from train split). Use synthetic result dicts — no YOLO needed.
- [ ] **Step 2: Run to verify failure**
  Run: `python -m pytest tests/test_detection.py -q`
  Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `src/detection/calibration.py`, the `imgsz` param in `PersonDetector`, and the thin script.
- [ ] **Step 4: Run to verify pass**
  Run: `python -m pytest tests/test_detection.py -q`
  Expected: PASS (all).
- [ ] **Step 5: Smoke the script on 4 frames/config** (seconds, not minutes):
  Run: `python scripts/calibrate_detection.py --max-frames 4 --imgsz-list 640 --conf-list 0.25`
  Expected: `reports/detection_calibration/calibration.json` exists with a `selected` entry.
- [ ] **Step 6: Commit**
  `git add src/detection/detector.py src/detection/calibration.py scripts/calibrate_detection.py tests/test_detection.py && git commit -m "feat: detector calibration grid + pre-declared automatic config selection"`

---

### Task 5: Tuning, threshold selection, soft-vote ensemble (`src/models/tuning.py`)

**Files:**
- Create: `src/models/tuning.py`
- Test: `tests/test_tuning.py`

**Interfaces:**
- Consumes: sklearn `GridSearchCV`, `GroupKFold`; `compute_binary_metrics` pattern from `src/experiments/metrics.py`.
- Produces (all in `src/models/tuning.py`):
  - `GRIDS: dict[str, dict]` — pre-declared small grids (exact): `logreg`: `{"clf__C": [0.01, 0.1, 1.0, 10.0]}`; `rf`: `{"clf__n_estimators": [200, 500], "clf__max_depth": [4, 8, None]}`; `xgb`: `{"clf__max_depth": [3, 4, 6], "clf__learning_rate": [0.05, 0.1], "clf__n_estimators": [200, 400]}`; `svm`: `{"clf__C": [0.1, 1.0, 10.0]}`; `dt`: `{"clf__max_depth": [3, 5, None]}`. `dt`/`svm` included but bounded (they grid fast at this data size).
  - `tune_model(name: str, X, y, groups: np.ndarray, seed: int = 42) -> tuple[Pipeline, dict, float]` — builds via `build_model(name, seed)`, `GridSearchCV(cv=GroupKFold(n_splits=4), scoring="roc_auc", n_jobs=1, refit=True)`, fits with `groups=groups`; returns `(best_pipeline, best_params, best_cv_auc)`. Raises `ValueError` for unknown name (delegates to `build_model`).
  - `best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float` — argmax F1 over `np.arange(0.05, 0.96, 0.05)`; `score >= t` defines positive; empty input raises `ValueError`; single-class `y_true` returns 0.5.
  - `soft_vote(score_matrix: np.ndarray) -> np.ndarray` — row-wise mean of per-model probability columns; validates equal column count.
- [ ] **Step 1: Write failing tests** in `tests/test_tuning.py`: (a) `tune_model("logreg", ...)` on a small separable synthetic table with a `groups` array returns a fitted pipeline whose `best_params_` came from the grid, and CV AUC in [0,1]; (b) group leakage guard: construct data where a held-out group is separable only via its own rows — assert the returned cv score is below a memorization-only threshold (i.e., grouped CV is genuinely holding groups out; keep the assertion loose, e.g. `auc < 1.0`); (c) `best_threshold` on hand-made scores picks the F1-maximizing bucket; single-class → 0.5; empty → raises; (d) `soft_vote` averages correctly on a known 2-column matrix; mismatched columns raise.
- [ ] **Step 2: Run to verify failure**
  Run: `python -m pytest tests/test_tuning.py -q`
  Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `src/models/tuning.py` per the Interfaces block.
- [ ] **Step 4: Run to verify pass**
  Run: `python -m pytest tests/test_tuning.py -q`
  Expected: PASS (all).
- [ ] **Step 5: Commit**
  `git add src/models/tuning.py tests/test_tuning.py && git commit -m "feat: grouped-CV tuning, F1-optimal threshold, soft-vote ensemble helpers"`

---

### Task 6: Feature redundancy screen

**Files:**
- Create: `src/features/screening.py`
- Create: `scripts/screen_features.py`
- Test: `tests/test_features.py` (append) or new `tests/test_screening.py`

**Interfaces:**
- Consumes: rebuilt parquet table columns.
- Produces:
  - `redundant_features(df: pd.DataFrame, candidates: list[str], existing: list[str], threshold: float = 0.9) -> dict[str, str]` — Spearman |ρ| > threshold against any *existing* feature (or earlier candidate is not considered; candidates only compared to `existing`); returns `{candidate: "matched:<existing_feature> (|rho|=<value:.2f>)"}` for flagged ones.
  - `scripts/screen_features.py --table <parquet> --out reports/results/feature_screen.json` — flags only the seven new v2 features against the original 15; writes per-feature max-|ρ| + flagged dict; prints flagged features with the instruction that `train_ml.py` excludes them via `--exclude-features`.
- [ ] **Step 1: Write failing tests**: synthetic DataFrame where candidate is a copy of an existing column (flagged) and an independent column (not flagged); threshold boundary respected.
- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/test_screening.py -q` → FAIL (module missing).
- [ ] **Step 3: Implement** module + script.
- [ ] **Step 4: Run to verify pass** — `python -m pytest tests/test_screening.py -q` → PASS.
- [ ] **Step 5: Commit** — `git add src/features/screening.py scripts/screen_features.py tests/test_screening.py && git commit -m "feat: Spearman redundancy screen for schema-v2 candidates"`

---

### Task 7: Attribution runner (the four pre-declared test evaluations)

**Files:**
- Create: `src/experiments/attribution.py`
- Create: `scripts/attribution.py`
- Test: `tests/test_experiments.py` (append)

**Interfaces:**
- Consumes: `grouped_split`, `assert_group_disjoint`, `train_model`, `tune_model`, `best_threshold`, `soft_vote` (Tasks 1–5), `FeatureSchema.default()`.
- Produces:
  - `ORIGINAL_15: list[str]` — the pre-v2 schema names (hard-coded list; verify against git history of `schema.py`).
  - `run_attribution(df: pd.DataFrame, seed: int = 42, results_dir: Path) -> dict` — builds the grouped split ONCE (same `grouped_split` call signature as `runner._split_frame`), then runs exactly four configs, evaluating each on test once:
    1. `baseline_old15_new_detector`: `ORIGINAL_15` columns, untuned `train_model` per algorithm, threshold 0.5.
    2. `schema_v2_untuned`: full schema names, untuned, threshold 0.5.
    3. `tuned_threshold`: full schema minus `--exclude-features`, per-algorithm `tune_model` on train (+groups), threshold from `best_threshold` on val.
    4. `ensemble_final`: soft-vote of top-3 algorithms by val AUC from config 3; ensemble threshold from val.
    All four share split membership; each config's payload records `{"columns": [...], "params": {...}, "threshold": t, "test_metrics": compute_binary_metrics(...), "val_metrics": ...}` plus the split's sequence-id lists once at top level. Writes `reports/results/experiment_attribution.json`. The `test` evaluations happen ONLY inside this function — the only sanctioned test-touching code (spec Goal 3).
  - `scripts/attribution.py --table <parquet> --exclude-features ""` — thin CLI, prints per-config test AUC/F1 summary.
- [ ] **Step 1: Write failing tests** in `tests/test_experiments.py`: on a tiny synthetic grouped table (~8 sequences, separable signal), `run_attribution` (a) returns dict with exactly the four config keys; (b) every config's payload has identical `test` split membership; (c) records `threshold` values in (0,1); (d) ensemble payload lists 3 member algorithms; (e) no config's columns include `label` or meta columns.
- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/test_experiments.py -q` → FAIL.
- [ ] **Step 3: Implement** module + script.
- [ ] **Step 4: Run to verify pass** — `python -m pytest tests/test_experiments.py -q` → PASS.
- [ ] **Step 5: Commit** — `git add src/experiments/attribution.py scripts/attribution.py tests/test_experiments.py && git commit -m "feat: pre-declared 4-run attribution experiment with provenance logging"`

---

### Task 8: Label-robustness experiment (val-only)

**Files:**
- Modify: `src/experiments/runner.py` (new function)
- Test: `tests/test_experiments.py` (append)

**Interfaces:**
- Consumes: `label_frac` column (Task 3), `_split_frame`, `_fit_score` patterns.
- Produces: `label_robustness_experiment(df: pd.DataFrame, config: ExperimentConfig) -> dict` — raises `ValueError("label_frac missing; rebuild the feature table with prepare_data.py")` when the column is absent. On the VAL split only (test is never touched here — spec Goal 3): trains untuned `logreg` on full schema columns and reports AUC/F1 under (a) any-frame label and (b) strict label `label_frac >= 0.2`; writes `reports/results/experiment_label_robustness.json` with both metrics, the val sequence ids, and a `"note": "val split only; different label definition — not comparable to test headline"`.
- [ ] **Step 1: Write failing tests**: (a) missing `label_frac` → the exact `ValueError`; (b) synthetic table → JSON-serializable dict with both label variants present, each with `roc_auc`/`f1` keys, and `"val_sequence_ids"` recorded; (c) assert the function never reads `split.test` rows (construct df where test rows would flip the metric — result must be identical).
- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/test_experiments.py -q -k robustness` → FAIL.
- [ ] **Step 3: Implement** the function in `src/experiments/runner.py`.
- [ ] **Step 4: Run to verify pass** — `python -m pytest tests/test_experiments.py -q` → PASS.
- [ ] **Step 5: Commit** — `git add src/experiments/runner.py tests/test_experiments.py && git commit -m "feat: label-robustness experiment (val-only, stricter window label)"`

---

### Task 9: Full feature-table rebuild with chosen detector config

**Files:**
- Modify: `scripts/prepare_data.py` (add `--conf`, `--imgsz` args defaulting from `configs/experiments.yaml` `detection:` section via the existing config loader; pass through to `PersonDetector`)
- Modify: `configs/experiments.yaml` (update `detection.conf` / add `detection.imgsz` to the calibration-selected values)
- No new tests beyond a CLI-defaults unit check appended to `tests/test_scripts.py`.

**Interfaces:**
- Consumes: `select_config` output from Task 4's real calibration run; `build_feature_table`.
- Produces: `data/processed/ucsd_windows.parquet` (schema v2, `label_frac`, new detector config) + `.meta.json` recording `detection_conf`, `detection_imgsz`, schema version, and row/label counts.

- [ ] **Step 1: Write failing test** (CLI defaults): add a small `resolve_detection_defaults(yaml_dict) -> tuple[float, int]` helper in `scripts/prepare_data.py` returning `(conf, imgsz)` from the `detection:` section; test asserts it reads both values from a synthetic dict and raises `KeyError` when `imgsz` is absent.
- [ ] **Step 2: Implement** the args + passthrough; update `configs/experiments.yaml` detection values from the calibration report.
- [ ] **Step 3: Smoke rebuild** (seconds):
  Run: `python scripts/prepare_data.py --max-sequences 2 --max-frames 60 --window 30 --stride 10 --out data/processed/smoke_v2.parquet`
  Expected: parquet exists; columns include all v2 names + `label_frac`.
- [ ] **Step 4: Full rebuild** (long-running; run in background, ~1–2h; verify projection matched Task 4's estimate):
  Run: `python scripts/prepare_data.py`
  Expected: `data/processed/ucsd_windows.parquet` with ~2k rows, `label_counts` populated, meta records new detector config.
- [ ] **Step 5: Feature screen on the real table**
  Run: `python scripts/screen_features.py --table data/processed/ucsd_windows.parquet`
  Expected: `reports/results/feature_screen.json`; note flagged features for Task 10's `--exclude-features`.
- [ ] **Step 6: Commit** — `git add scripts/prepare_data.py configs/experiments.yaml tests/test_scripts.py && git commit -m "feat: prepare_data detector config passthrough; calibrated detection defaults"`

---

### Task 10: Train-ML integration (tuning + ensemble flags) + attribution run

**Files:**
- Modify: `scripts/train_ml.py` (add `--tune`, `--exclude-features`, `--no-ensemble` flags; bundles save tuned pipelines with selected `threshold`; writes `experiment_tuning.json` + `experiment_ensemble.json`)
- Test: `tests/test_ml.py` (append)

**Interfaces:**
- Consumes: `tune_model`, `best_threshold`, `soft_vote`, `run_attribution` (Tasks 5, 7), `ModelBundle.threshold` (Task 3).
- Produces:
  - With `--tune`: per-algorithm tuned bundle saved with `threshold` from val; `reports/results/experiment_tuning.json` = `{algo: {"best_params", "cv_auc", "val_metrics", "test_metrics"}}` (test metrics here come from the sanctioned config-3 run — the script delegates the test-touching to `run_attribution`; in the no-`--tune` path nothing new touches test beyond the existing behavior).
  - Ensemble bundle: `models/ensemble.joblib` saved via `save_bundle` with `algorithm="ensemble"`, `feature_names` = the shared column list, `model` = a small `src/models/ensemble_model.py`-style wrapper — implement as a plain class `SoftVoteModel(member_pipelines: list)` with `predict_proba(X)` (mean) and `predict(X)` (`proba[:,1] >= self.threshold`), defined in `src/models/tuning.py` and picklable via joblib.
  - `reports/results/experiment_ensemble.json` = member algorithms + val AUCs + test metrics from the sanctioned run.
- [ ] **Step 1: Write failing tests** in `tests/test_ml.py`: (a) `SoftVoteModel` predict/predict_proba on two stub models with known probabilities (mean asserted exactly; predict respects threshold); (b) picklable round-trip via joblib to a tmp path; (c) `run_attribution` end-to-end on the synthetic table writes all four configs to a tmp results dir (re-uses Task 7 fixture).
- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/test_ml.py -q` → FAIL.
- [ ] **Step 3: Implement** `SoftVoteModel` + script flags.
- [ ] **Step 4: Run to verify pass** — `python -m pytest tests/test_ml.py -q` → PASS.
- [ ] **Step 5: Real run on the rebuilt table** (train/val work only; the four sanctioned test evaluations happen inside `run_attribution`):
  Run: `python scripts/train_ml.py --tune --exclude-features "<flagged from Task 9 Step 5>"`
  Expected: bundles in `models/`, `experiment_tuning.json`, `experiment_ensemble.json`, `experiment_attribution.json` written; log prints per-algorithm AUC/F1.
- [ ] **Step 6: Label robustness**
  Run: `python -c "from src.experiments.runner import *"` equivalent — or a `--robustness` flag: run the label-robustness experiment on the real table.
  Expected: `reports/results/experiment_label_robustness.json`.
- [ ] **Step 7: Commit** — `git add scripts/train_ml.py src/models/tuning.py tests/test_ml.py && git commit -m "feat: tuned + thresholded bundles, soft-vote ensemble bundle, tuning/ensemble result JSONs"`

---

### Task 11: Frontend UX pass

**Files:**
- Modify: `app.py` (sidebar restructure)
- Modify: `src/ui/common.py` (new pure helper)
- Modify: `src/ui/page_video.py`, `src/ui/page_camera.py` (prediction interpretation)
- Test: `tests/test_dashboard.py` (append)

**Interfaces:**
- Consumes: existing `InferenceResult`, `model_status_badge`.
- Produces:
  - `src/ui/common.py`: `interpret_prediction(label: int, score: float) -> tuple[str, str]` → `(headline, note)`; e.g. label 1 & score ≥0.8 → `("Anomaly likely", "model confidence 84%")`; label 1 & <0.8 → `("Possible anomaly", "low model confidence — treat as a hint")`; label 0 & score ≥0.3 → `("Probably normal", "score near the decision boundary")`; else `("Normal", ...)`. Plus `model_caveat() -> str` = "Models are sequence-held-out trained and modest (see Methodology). Detector counts are estimates."
  - `app.py`: sidebar groups pages under captions ("Learn": Overview, Methodology, Dataset Explorer; "Analyze": Photo, Video, Live Camera; "Train & Review": Training, Evaluation, Research Results) — `st.radio` with `horizontal=False` kept, but wrapped per-group with `st.caption`.
  - `page_video.py` / `page_camera.py`: prediction tables gain an interpretation column from `interpret_prediction`, and a `st.caption(model_caveat())` under every predictions block.
- [ ] **Step 1: Write failing tests** in `tests/test_dashboard.py` (pure-function style, matching that file's existing pattern): all four `interpret_prediction` branches return non-empty strings; `model_caveat` mentions Methodology; scores at boundaries (0.8, 0.3) classified as specified.
- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/test_dashboard.py -q` → FAIL.
- [ ] **Step 3: Implement** helper + page/sidebar changes.
- [ ] **Step 4: Run to verify pass** — `python -m pytest tests/test_dashboard.py -q` → PASS.
- [ ] **Step 5: Manual smoke**: `python -m streamlit run app.py` — sidebar shows grouped nav; Video Analysis with no model shows the friendly message; nothing errors.
- [ ] **Step 6: Commit** — `git add app.py src/ui/common.py src/ui/page_video.py src/ui/page_camera.py tests/test_dashboard.py && git commit -m "feat: grouped sidebar nav + plain-language prediction interpretation"`

---

### Task 12: README + docs close-out

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-10-21-accuracy-improvement-design.md` (status → completed)

**Interfaces:**
- Consumes: `reports/results/experiment_attribution.json`, `experiment_tuning.json`, `experiment_ensemble.json`, `experiment_label_robustness.json`, `feature_screen.json`, `calibration.json`.

- [ ] **Step 1: Update README headline table** with the four-run before/after (run 1 vs run 4) exactly as recorded in `experiment_attribution.json`, provenance sentence naming seed/split/detector config; update Limitations if the calibration changed detection defaults (state the new conf/imgsz and the recall-over-size framing); add the new scripts to Quick start and the artifacts map; keep or revise the "modest AUC" honesty sentence to match the real outcome (if <0.75 achieved, say so plainly — the spec's fallback clause).
- [ ] **Step 2: Full suite green**
  Run: `powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1`
  Expected: all pass.
- [ ] **Step 3: Commit** — `git add README.md docs/superpowers/specs/2026-10-21-accuracy-improvement-design.md && git commit -m "docs: README before/after headline table + calibration provenance"`
