# CrowdSense AI — Crowd Density, Tracking, and Movement-Based Anomaly Research

**Research question:** *Do temporal crowd-movement features improve anomaly
prediction over static crowd-density features alone?*

Everything in this repository is computed from real data — the dashboard only
renders files that exist under `reports/`, and no metric is ever fabricated,
rounded up, or reported without its provenance.

> Environment: Python 3.13 (Windows), CPU-only (torch 2.9.0+cpu — no CUDA).

## Quick start (PowerShell)

```powershell
# 1. Dependencies (verified environment; requirements.txt lists them)
python -m pip install -r requirements.txt

# 2. Tests (TDD suite: ~179 tests)
#    Prefer the wrapper — it pins native thread pools to 1 and avoids
#    intermittent OpenMP access-violation crashes on Windows:
powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1
#    (plain `python -m pytest tests/ -q` usually works too, but can flake)

# 3. Inspect the dataset (graceful when absent)
python scripts/inspect_dataset.py

# 4. Detector calibration (40 sampled frames; pre-declared rule: most
#    detections under a 2h rebuild budget -> conf=0.05, imgsz=640)
python scripts/calibrate_detection.py

# 5. Build the sliding-window feature table  (CPU-heavy: ~1-2 ms/frame overhead
#    on top of YOLO; full UCSD takes on the order of an hour)
python scripts/prepare_data.py

# 6. Train the five models + run experiments A-D
python scripts/train_ml.py
#    Accuracy pass — grouped-CV tuning + val thresholds + soft-vote ensemble +
#    label robustness; the four pre-declared test runs land in
#    experiment_attribution.json (test touched once per config):
python scripts/train_ml.py --tune --robustness
#    Redundancy screen of the schema-v2 features (Spearman vs original 15):
python scripts/screen_features.py

# 7. Evaluate bundles on the held-out test split (+ Experiment E gate status)
python scripts/evaluate.py

# 8. Dashboard
python -m streamlit run app.py
```

Smoke variants (seconds instead of an hour):

```powershell
python scripts/prepare_data.py --max-sequences 4 --max-frames 100 --window 30 --stride 10
python scripts/train_ml.py --algorithms logreg,dt
```

## Datasets (placed manually — never auto-downloaded)

### UCSD Anomaly Detection Dataset v1p2  (required)

Download from the UCSD CVPR 2010 page and extract so that this exists:

```
data/raw/ucsd_anomaly/UCSD_Anomaly_Dataset.v1p2/
├── UCSDped1/
│   ├── Train/Train001..020/*.tif
│   └── Test/Test001..036/*.tif + Test00X_gt/*.bmp + UCSDped1.m
└── UCSDped2/
    ├── Train/Train001..016/*.tif
    └── Test/Test001..012/*.tif + Test00X_gt/*.bmp + UCSDped2.m
```

- Citation: V. Mahadevan, W. Li, N. Boccali, A. K. Roy-Chowdhury,
  *Anomaly Detection in Crowded Scenes*, CVPR 2010.
- Verified against the real data: **98 sequences, 18,560 frames,
  5,693 anomalous-frame labels** (frame GT parsed from the `.m` files).

### Congestion annotations (optional, for gated Experiment E)

Experiment E (congestion classification) stays disabled until credible labels
exist. If you create them, use the format in
`data/metadata/annotation_format.md`; `exp_e_status.json` will only change
status when labeled data is present at the configured path.

### Dropped from scope (by decision)

UCF-QNRF and ShanghaiTech are **not** used. Counting has no ground truth here,
so this project reports **no MAE/RMSE counting metrics anywhere** — detector
counts are estimates, labeled as such in every output.

## Experiments

| ID | Question | Status |
|----|----------|--------|
| A | static density features only | run by `train_ml.py` |
| B | movement features only | run by `train_ml.py` |
| C | static + movement (does C beat A?) | run by `train_ml.py` |
| D | algorithm comparison (logreg, dt, rf, svm, xgb) | run by `train_ml.py` |
| E | congestion classification | **gated** — `exp_e_status.json = not_supported` |

Method guarantees:

- **Grouped splits by sequence id** — frames from one video never span
  train/val/test (`assert_group_disjoint` enforces it; membership is recorded
  in the result JSON files).
- **Window label** = 1 if *any* frame in the window is anomalous (UCSD frame GT).
- **Units** are px/frame at processed resolution — never m/s (no scale).
- Metrics are computed on the held-out **test** split; undefined metrics are
  `null`, never invented (e.g. ROC AUC with one class present).
- **Capped test evaluation (accuracy effort)** — exactly four pre-declared
  configurations touch test, each logged with provenance in
  `experiment_attribution.json`; feature screening, thresholds, and ensemble
  membership all use train-side GroupKFold CV or the val split only.

## Limitations (read before citing anything)

1. **No density CNN** — CPU-only machine; detection counting baseline (YOLOv8n)
   only. `density_per_mpx` is a detector-derived grid estimate, not a regressor.
2. **No counting ground truth** — QNRF/ShanghaiTech dropped; no MAE/RMSE.
3. **No congestion labels** — Experiment E is hard-gated off.
4. **Speeds are px/frame** — no scene-scale calibration, so no m/s conversion.
5. **Anomaly labels are frame-level** — window labels inherit "any frame" logic.
6. **Detector defaults are a calibrated choice, not a truth** — a 40-frame
   grid (`calibrate_detection.py`) selected **conf=0.05 / imgsz=640** by the
   pre-declared rule "most detections under a 2h rebuild" (16.40 vs 10.25
   boxes/frame at the old conf=0.25): recall over detector precision.
   Counts remain estimates (±) — fine for relative features, not absolute
   occupancy claims.

## Artifacts map

```
data/processed/
└── ucsd_windows.parquet + .meta.json   # feature table (gitignored, regenerable)
models/
└── {logreg,dt,rf,svm,xgb,ensemble}.joblib  # tuned bundles + val thresholds (gitignored)
reports/
├── detection_calibration/
│   └── calibration.json + overlays     # conf/imgsz grid, selection rule, audit frames
├── results/
│   ├── experiment_attribution.json     # FOUR pre-declared runs -> headline table
│   ├── experiment_tuning.json          # grouped-CV params + val thresholds per algo
│   ├── experiment_ensemble.json        # soft-vote members (by val AUC) + val/test
│   ├── experiment_label_robustness.json# val-only stricter labels (never test)
│   ├── experiment_ablation.json        # A/B/C per algorithm (val+test)
│   ├── experiment_algorithms.json      # D comparison
│   ├── feature_screen.json             # Spearman redundancy screen (exclusions)
│   ├── evaluation.json                 # bundle test metrics (evaluate.py)
│   └── exp_e_status.json               # E gate status
└── figures/*.png                       # confusion/ROC/importance (test split)
```

## Headline results — four pre-declared runs (full UCSD, seed 42)

Provenance: 98 sequences → 3222 windows (30 frames, stride 5) from the
calibrated detector (YOLOv8n conf=0.05, imgsz=640 → 16.40 boxes/frame);
grouped split by sequence id — **68 train / 15 val / 15 test, disjoint**
(seed 42) — test = 495 windows (179 anomalous). Test is evaluated exactly
once per config, inside `src/experiments/attribution.py`; every threshold,
hyperparameter, and ensemble-membership decision used train-side GroupKFold
CV or the val split only. Source: `reports/results/experiment_attribution.json`.

| # | config (selection rule) | features | threshold | test AUC | test F1 |
|---|-------------------------|----------|-----------|----------|---------|
| 1 | baseline — original 15 features, logreg, untuned | 15 | 0.50 | 0.6677 | 0.5068 |
| 2 | + schema-v2 movement features, logreg, untuned | 23 | 0.50 | 0.6475 | 0.4892 |
| 3 | + grouped-CV tuning + val threshold (rf; best by **val** AUC) | 23 | 0.25 | **0.6714** | **0.5311** |
| 4 | + soft-vote ensemble rf+svm+xgb (members by **val** AUC) | 23 | 0.20 | 0.6705 | 0.5090 |

Before → after against the previous headline (same grouped splits, older
conf=0.25 detector table): **AUC 0.675 → 0.6714, F1 0.517 → 0.5311.**

**Honest outcome: the ≥ 0.75 AUC goal was not met — the ceiling held at
~0.67.** What the four runs show:

- static density alone is near-chance (AUC 0.43–0.56 across algorithms);
  movement features carry the signal (ablation below);
- the calibrated detector (16.40 vs 10.25 boxes/frame) did **not** lift the
  old-15 baseline by itself (0.6677 vs 0.675);
- the 8 new schema-v2 features *hurt* untuned (run 2 < run 1) and only pay
  off with grouped-CV tuning + a val-chosen threshold (run 3 = best);
- the ensemble did not beat the tuned single model (run 4 < run 3).

Sequence-held-out window-level anomaly detection on UCSD with a CPU-only
detector thus plateaus near AUC 0.67 on these splits; this effort's gain is
in F1 (0.517 → 0.5311, val-tuned threshold). Reproduce with the Quick start
commands; every number above is stored in `reports/results/*.json` with its
split membership.

### Ablation — experiments A–C (calibrated-detector table, threshold 0.5)

| algorithm | A static only (6 feat) | B movement only (17 feat) | C static+movement (23 feat) | C − A (AUC) |
|-----------|------------------------|---------------------------|-----------------------------|-------------|
| logreg    | 0.505 / 0.159 | 0.653 / 0.490 | 0.648 / 0.489 | +0.142 |
| svm       | 0.427 / 0.162 | 0.665 / 0.386 | **0.682 / 0.429** | **+0.255** |
| xgb       | 0.499 / 0.256 | 0.629 / 0.420 | 0.614 / 0.351 | +0.115 |
| rf        | 0.470 / 0.267 | 0.635 / 0.385 | 0.671 / 0.376 | +0.202 |
| dt        | 0.563 / 0.426 | 0.552 / 0.409 | 0.593 / 0.446 | +0.029 |

Interpretation: static density alone is near-chance for frame-window
anomaly prediction; temporal movement features improve AUC on **every**
algorithm (C − A > 0 in all five rows) — the research question's answer is
*yes, movement features help* — while the absolute ceiling stays modest
(~0.67; sequence-held-out anomaly detection on UCSD is hard). Stored in
`reports/results/experiment_ablation.json`.

## Project structure

```
app.py                     # Streamlit shell (sidebar navigation)
configs/
  paths.yaml               # dataset + report paths
  experiments.yaml         # seed, window/stride/grid, algorithms
scripts/
  inspect_dataset.py       # dataset summary (graceful if absent)
  calibrate_detection.py   # conf/imgsz grid on sampled frames (pre-declared rule)
  prepare_data.py          # frames -> window feature table (parquet)
  screen_features.py       # Spearman redundancy screen for schema-v2 candidates
  train_ml.py              # bundles + experiments A-D; --tune --robustness pass
  evaluate.py              # test-split bundle metrics + E gate
src/
  config.py                # typed config loaders
  datasets/                # UCSD adapter (frame GT parsing, inspection)
  preprocessing/           # image/video/camera IO, validation
  detection/               # YOLOv8n person detector + analyze helpers
  tracking/                # ByteTrack wrapper + trajectory store
  features/                # schema, static, movement, window pipeline
  models/                  # grouped splits, 5 trainers, joblib bundles
  experiments/             # metrics + A-D runner (E gated)
  inference/               # schema-strict InferenceService
  visualization/           # Agg-backend PNG plots
  ui/                      # 8 dashboard pages + shared helpers
tests/                     # TDD suite (~179 tests) + synthetic fixtures
docs/superpowers/          # approved spec + plan
```

## Dashboard pages

Overview · Methodology (honest limitations) · Dataset Explorer ·
Photo Analysis (static features only — no temporal claims from one frame) ·
Video Analysis (window features + model predictions if a bundle exists) ·
Live Camera (rolling window; friendly failure without a device) ·
Training & Evaluation (real subprocess runs with log tails) ·
Research Results (renders only existing files; missing → exact command shown).
