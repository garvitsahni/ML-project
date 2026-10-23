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

# 2. Tests (TDD suite: ~201 tests)
#    Prefer the wrapper — it pins native thread pools to 1 and avoids
#    intermittent OpenMP access-violation crashes on Windows:
powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1
#    (plain `python -m pytest tests/ -q` usually works too, but can flake)

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
data/processed/
└── ucsd_windows.parquet + .meta.json   # feature table (gitignored, regenerable)
models/
└── {logreg,dt,rf,svm,xgb}.joblib       # schema-versioned bundles (gitignored)
reports/
├── results/
│   ├── experiment_ablation.json        # A/B/C per algorithm (val+test)
│   ├── experiment_algorithms.json      # D comparison
│   ├── evaluation.json                 # bundle test metrics
│   └── exp_e_status.json               # E gate status
└── figures/*.png                       # confusion/ROC/importance (test split)
```

## Headline results (full UCSD, seed 42, 68 train / 15 test sequences, 495 test windows)

Honest test-split numbers — movement features help on every algorithm:

| algorithm | A static only (AUC/F1) | B movement only | C static+movement | C − A (AUC) |
|-----------|------------------------|-----------------|-------------------|-------------|
| logreg    | 0.460 / 0.165 | 0.634 / 0.487 | **0.675 / 0.517** | **+0.214** |
| svm       | 0.407 / 0.192 | 0.643 / 0.419 | 0.651 / 0.424 | **+0.245** |
| xgb       | 0.436 / 0.309 | 0.623 / 0.424 | 0.602 / 0.373 | +0.166 |
| rf        | 0.423 / 0.260 | 0.627 / 0.438 | 0.582 / 0.360 | +0.159 |
| dt        | 0.406 / 0.259 | 0.558 / 0.414 | 0.519 / 0.348 | +0.113 |

Interpretation: static density alone is near-chance for frame-window anomaly
prediction; temporal movement features carry the signal (best absolute AUC is
still a modest 0.675 — sequence-held-out anomaly detection on UCSD is hard).
Reproduce with the Quick start commands; every number above is stored in
`reports/results/*.json` with its split membership.

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
  ui/                      # 9 dashboard pages + shared design-system helpers
tests/                     # TDD suite (~201 tests) + synthetic fixtures
docs/superpowers/          # approved spec + plan
```

## Dashboard (redesigned interface)

`python -m streamlit run app.py` starts the shell: branded sidebar →
sectioned navigation → the selected page. Destinations only appear in the
sidebar once implemented. Every number rendered by the dashboard is read
from real files or computed live — nothing is fabricated, and no download
is ever offered.

**Workspace**

- **Overview** — live status of dataset / models / results (read from
  disk), recent session activity, and next-step CTAs.
- **Photo Analysis** — upload → validation → explicit analyse: detection,
  spatial density grid, static features. Temporal outputs are intentionally
  absent (one frame has no window).
- **Video Analysis** — upload → batched, cancellable processing (24-frame
  batches; Cancel/Stop takes effect at batch boundaries). Timeline charts,
  window-level model predictions, annotated-frame strip, optional annotated
  video export, CSV/JSON downloads.
- **Live Camera** — rolling window over a camera device with measured
  FPS/latency; the device is only opened on "Start capture" and always
  released on stop/navigation. Without a device the page fails gracefully
  with the real error message.

**Research**

- **Dataset Explorer** — live probe of the placed UCSD dataset (sequence
  and frame counts, per-sequence table, built-table stats, provenance and
  annotation facts). When the dataset is absent it shows the exact placement
  tree and inspection command; it never downloads anything.
- **Research Lab** — runs the real pipeline scripts as subprocesses (run
  controls below), A/B/C + D result charts, evaluation table, Experiment
  E-gate status, figure gallery, and the methodology/provenance write-up.
- **Model Registry** — every `models/*.joblib` bundle: algorithm, trained
  timestamp, training rows, artifact size, validation metrics, feature
  schema; held-out test metrics from `reports/results/evaluation.json`; the
  default-inference-model selector (invalid or stale picks fall back to the
  newest available bundle with a visible warning).
- **Reports** — this session's analysis history (table, per-run detail
  expanders, CSV export, clear-with-confirmation) plus the research results
  that exist on disk (tables with CSV export where data exists, figure
  gallery, and the exact command for every missing output).

**System**

- **Settings** — session preferences (validated on save, applied to the
  next analysis) and a whitelisted diagnostics panel (Python/Streamlit
  versions, torch device, dataset/model/results presence — no paths, no
  environment values, no secrets).

### Session-only history

Analysis history (Photo/Video/Camera runs) lives in the browser session and
**is not persisted** — it disappears when the tab closes. Use Reports →
"Download history CSV" to keep a copy. Settings are session-only too.

### Settings (System → Settings)

| Setting | Range / choices | Default | Effect |
|---------|-----------------|---------|--------|
| Detection confidence | 0.05–0.95 (step 0.05) | 0.25 | YOLO gate for counting a person (Photo/Video/Camera) |
| Frame stride | 1–10 | 1 | Process every Nth video frame (Video) |
| Max frames per video | 60–1200 | 300 | Hard cap per video run |
| Processing resolution | 640 / 1280 / 1920 (long side px) | 1280 | Speed/quality on CPU (Video/Camera) |
| Show density overlays | on / off | on | Density heatmap overlays (Photo/Camera) |
| Default model | any registered bundle | newest bundle | Inference bundle for Video/Camera, chosen on Model Registry |

### Research Lab run controls

Steps run as real subprocesses (`prepare_data`, `train_ml`, `evaluate`),
**one run at a time**: a live elapsed timer, a tail of the real log
(`reports/results/run_<step>.log`), and **Cancel** that terminates the
process. `prepare` and `train` additionally require an explicit confirmation
checkbox because they rewrite result files. The E-gate card always reports
Experiment E as `not_supported` with its reason — the UI never enables it
without validated congestion labels.
