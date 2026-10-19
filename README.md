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

# 2. Tests (TDD suite: ~124 tests)
python -m pytest tests/ -q

# 3. Inspect the dataset (graceful when absent)
python scripts/inspect_dataset.py

# 4. Build the sliding-window feature table  (CPU-heavy: ~1-2 ms/frame overhead
#    on top of YOLO; full UCSD takes on the order of an hour)
python scripts/prepare_data.py

# 5. Train the five models + run experiments A-D
python scripts/train_ml.py

# 6. Evaluate bundles on the held-out test split (+ Experiment E gate status)
python scripts/evaluate.py

# 7. Dashboard
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

## Limitations (read before citing anything)

1. **No density CNN** — CPU-only machine; detection counting baseline (YOLOv8n)
   only. `density_per_mpx` is a detector-derived grid estimate, not a regressor.
2. **No counting ground truth** — QNRF/ShanghaiTech dropped; no MAE/RMSE.
3. **No congestion labels** — Experiment E is hard-gated off.
4. **Speeds are px/frame** — no scene-scale calibration, so no m/s conversion.
5. **Anomaly labels are frame-level** — window labels inherit "any frame" logic.
6. **YOLOv8n confidence is not calibrated** — counts are estimates (±),
   fine for relative features, not absolute occupancy claims.

## Artifacts map

```
reports/
├── processed/ucsd_windows.parquet      # feature table + .meta.json
├── models/{logreg,dt,rf,svm,xgb}.joblib # schema-versioned bundles
├── results/
│   ├── experiment_ablation.json        # A/B/C per algorithm (val+test)
│   ├── experiment_algorithms.json      # D comparison
│   ├── evaluation.json                 # bundle test metrics
│   ├── exp_e_status.json               # E gate status
│   └── prepare_full.log                # latest prepare run log
└── figures/*.png                       # confusion/ROC/importance (test split)
```

## Project structure

```
app.py                     # Streamlit shell (sidebar navigation)
configs/
  paths.yaml               # dataset + report paths
  experiments.yaml         # seed, window/stride/grid, algorithms
scripts/
  inspect_dataset.py       # dataset summary (graceful if absent)
  prepare_data.py          # frames -> window feature table (parquet)
  train_ml.py              # bundles + experiments A-D
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
tests/                     # TDD suite (~124 tests) + synthetic fixtures
docs/superpowers/          # approved spec + plan
```

## Dashboard pages

Overview · Methodology (honest limitations) · Dataset Explorer ·
Photo Analysis (static features only — no temporal claims from one frame) ·
Video Analysis (window features + model predictions if a bundle exists) ·
Live Camera (rolling window; friendly failure without a device) ·
Training & Evaluation (real subprocess runs with log tails) ·
Research Results (renders only existing files; missing → exact command shown).
