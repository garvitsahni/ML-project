# CrowdSense AI — Design Specification

**Date:** 2026-10-18 · **Status:** approved by user · **Plan:** `docs/superpowers/plans/2026-10-18-crowdsense-ai.md`

## 1. Goal

A runnable, local Windows ML research project with three input modes (photo upload,
video upload, live webcam) performing crowd counting, spatial density analysis,
pedestrian tracking, movement analysis, and ML-based prediction — built primarily as
**research**, not a pretrained-YOLO demo.

Research question: *Does incorporating temporal crowd-movement features improve
congestion or anomaly prediction compared with using static crowd-density features alone?*

Honesty constraints (non-negotiable):
- No fabricated data, labels, metrics, or citations; the dashboard renders only files
  that actually exist under `reports/results/`.
- No stampede/danger prediction claims. No congestion classifier without credible labels.
- Movement in px/frame; density as an image-area estimate; counting explicitly a
  detector estimate when no ground truth is available.

## 2. Environment (verified Phase 0)

- Python 3.13.2 system install; preinstalled & verified: ultralytics 8.3.225,
  torch 2.9.0+cpu (no CUDA), cv2 5.0.0, sklearn 1.8.0, xgboost 3.2.0, streamlit 1.51.0,
  pandas 3.0.0, numpy 2.4.2, scipy 1.17.0, matplotlib 3.10.7, pytest 8.3.3, joblib 1.5.3,
  lap 0.5.13; seaborn 0.13.2 added. `yolov8n.pt` present.
- CPU-only torch ⇒ detection-based counting baseline only; density-CNN training (CSRNet)
  documented as blocked.

## 3. Scope decisions (user)

- Datasets provided manually into `data/raw/` (ask before any download).
- **UCSD Pedestrian Anomaly Dataset only** — UCF-QNRF and ShanghaiTech excluded by
  decision; no counting-GT experiments (MAE/RMSE) are possible ⇒ documented limitation.
- Counting = YOLO detection estimate (count + density visualization), no GT comparison.

## 4. Architecture

Library-first monolith: `src/` packages with dataclass interfaces; `scripts/` thin CLIs;
`app.py` Streamlit consumes the same inference APIs.

```
app.py + src/ui/*            8-page Streamlit (sidebar radio nav, render() per page)
scripts/                     inspect_dataset.py, prepare_data.py, train_ml.py, evaluate.py
src/datasets/                base types + UCF-free UCSD adapter (frame sequences + .m GT)
src/preprocessing/           image validation, frame-sequence/video iteration, Camera
src/detection/               PersonDetector (yolov8n, person-only, cpu, cached), drawing, image analysis
src/tracking/                ByteTrack wrapper (ultralytics.trackers.BYTETracker) + TrajectoryStore
src/features/                FeatureSchema (order/units/limitations), static + movement extraction
src/models/                  grouped splitting, 5 trainers (logreg/dt/rf/svm/xgb), joblib bundles
src/evaluation/              metrics, experiments A–D (E gated), counting eval (unused w/o GT)
src/visualization/           heatmap, time series, confusion matrix, importance, ROC (Agg)
```

Data flow: frames → detections → trajectories → windowed features (schema-ordered) →
bundled pipeline → labels/scores; every experiment writes CSV/JSON to `reports/results/`
and PNGs to `reports/figures/`.

## 5. Datasets — UCSD Pedestrian Anomaly (v1p2)

Layout (verified in workspace): `UCSDped1/{Train,Test}`, `UCSDped2/{Train,Test}` —
sequences are folders of `.tif` frames (README: ped1 34 train/36 test, ped2 16 train/12 test);
frame-level abnormal-event GT in `Test/UCSDped{1,2}.m` (`gt_frame` ranges, parsed by regex);
pixel masks `.bmp` in `*_gt` folders for 10 ped1 + 12 ped2 test clips. Train = normal only;
every test clip contains some anomalous frames (per README). Citation: Mahadevan et al.,
"Anomaly Detection in Crowded Scenes", CVPR 2010.

Adapter reports actual counts/missing files; never assumes schemas.

## 6. Features

Static (per window): person_count, density_per_mpx, grid_concentration, spread_x/y,
grid_entropy. Temporal: density_delta, mean_speed_pxf, speed_std_pxf, direction_entropy,
directional_consistency, flow_mean_magnitude/flow_std (Farneback optical flow), track_count,
track_persistence. Every feature carries `{name, dtype, unit, limitation}` in the schema;
`FeatureSchema.names` is the single ordering source consumed by training, bundles, inference.

## 7. ML & experiments

- Grouped splits by sequence id (never random frames); val for selection, test held out;
  scalers inside sklearn `Pipeline` fit on train only.
- Algorithms: LogisticRegression, DecisionTree, RandomForest, SVM, XGBoost.
- Exp A: algorithm comparison (identical splits/metrics). Exp B: static vs
  static+movement (**the research question**). Exp C: feature-group ablation.
  Exp D: cross-video generalization (ped1 ↔ ped2). Exp E: horizons only if
  future-event labels exist — otherwise writes `exp_e_status.json = not_supported`.
- Metrics: precision/recall/F1, ROC-AUC/PR-AUC, confusion matrices, inference time,
  feature importance, dataset size/class distribution — all recorded from real runs.

## 8. UI (8 pages)

Overview · Photo · Video · Live Camera · Dataset Explorer · Training & Evaluation ·
Research Results · Methodology. Model outputs only when a compatible bundle exists in
`models/`; Photo page never shows temporal/future-risk outputs; Results page reads only
existing files and names the dataset+model for every figure.

## 9. Testing

pytest, synthetic fixtures (cv2-drawn images, `cv2.VideoWriter` clips, savemat-style
fixtures as needed) for unit tests only: annotation parsing, feature math, invalid
image/video rejection, missing datasets, split integrity, bundle round-trip + schema
mismatch, missing model files, webcam failure paths, dashboard import/startup.

## 10. Known limitations (documented, not hidden)

CPU-only torch (no density-CNN) · no counting datasets ⇒ no MAE/RMSE · no congestion
labels ⇒ no risk classifier · Experiment E unsupported by available labels · px/frame units.

## 11. Risks / blockers

- UCSD `.m` parsing must tolerate comma-separated ranges (`[5:90, 140:200]`).
- Ultralytics API drift (BYTETracker vs ByteTrack) — verified present.
- Streamlit webcam on Windows: always release in `finally`; handle missing devices.
