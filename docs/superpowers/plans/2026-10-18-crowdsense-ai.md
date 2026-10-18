# CrowdSense AI — Implementation Plan (approved 2026-10-18)

**Goal:** A runnable local ML research project that counts crowds, tracks movement, and empirically tests whether temporal movement features improve anomaly prediction over static density features — with honest, non-fabricated results.

**Architecture:** Library-first monolith. `src/` packages with dataclass interfaces; `scripts/` thin CLIs; `app.py` Streamlit consumes the same inference APIs; experiment outputs land in `reports/results|figures/`; dashboard only renders files that exist.

**Tech Stack:** Python 3.13 (existing env), OpenCV, Ultralytics YOLO (yolov8n.pt, CPU), ByteTrack (ultralytics.trackers.BYTETracker + lap), NumPy/Pandas, scikit-learn, XGBoost, matplotlib/seaborn, Streamlit, joblib, pytest, scipy.io.

## Phase 0 findings (verified)
- Workspace empty; git initialized 2026-10-18; Python 3.13.2 (+3.11 bare).
- Installed & verified: ultralytics 8.3.225, torch 2.9.0+cpu, cv2 5.0.0, sklearn 1.8.0, xgboost 3.2.0, streamlit 1.51.0, pandas 3.0.0, numpy 2.4.2, scipy 1.17.0, matplotlib 3.10.7, pytest 8.3.3, joblib 1.5.3, lap 0.5.13, seaborn 0.13.2 (installed this session).
- yolov8n.pt downloaded & loads; correct ByteTrack import verified.
- No CUDA (CPU torch); 515 GB free; network slow (~90 KB/s).

## Global Constraints
- System Python 3.13; only new dependency seaborn.
- Never fabricate metrics/data/citations; dashboard shows only files under reports/results/.
- No congestion/danger classifier unless credible labels provided; document limitation.
- Movement features in px/frame (never m/s); density = image-area estimate.
- Splits grouped by video/sequence; scalers inside sklearn Pipeline fit on train only.
- Windows/PowerShell; datasets in data/raw/{ucf_qnrf,shanghaitech,ucsd_anomaly}/ via configs/paths.yaml.
- .gitignore excludes data/raw/, data/processed/, models/*.joblib, *.pt.
- Blockers documented: CPU-only torch (no density-CNN), no congestion labels, Experiment E gated.

## User decisions
- Datasets provided manually into data/raw/ (ask before any download).
- Existing Python 3.13 environment.
- Counting = detection baseline only.

---

### Task 1: Spec, skeleton, env verification
Create: docs/superpowers/specs/2026-10-18-crowdsense-ai-design.md, .gitignore, requirements.txt, configs/paths.yaml, configs/experiments.yaml, src/*/__init__.py, tests/conftest.py, README.md stub, dirs data/{raw,processed,metadata}, models/, reports/{figures,results}, notebooks/, scripts/, tests/.
Verify: python -c "import cv2, ultralytics, sklearn, xgboost, streamlit, seaborn, scipy; from ultralytics.trackers import BYTETracker; print('env ok')" then git commit.

### Task 2: Config loader
Create: src/config.py, tests/test_config.py.
Interfaces: load_paths() -> PathsConfig (raw_dir, processed_dir, metadata_dir, models_dir, results_dir, figures_dir, ucf_qnrf_dir, shanghaitech_dir, ucsd_dir); load_experiment_config() -> dict; ConfigError with actionable message.
TDD: failing tests (yaml read; missing file raises) -> yaml.safe_load implementation -> pass -> commit.

### Task 3: Dataset base + UCF-QNRF adapter
Create: src/datasets/base.py, src/datasets/ucf_qnrf.py, tests/test_ucf_qnrf.py, tests/fixtures.py.
Interfaces: InspectionReport(dataset, root, image_count, annotation_count, images_missing, annotations_missing, labels_available, notes); CountSample(image_path, gt_count, width, height, source); DatasetAdapter protocol (name, inspect(root), iter_count_samples(root)); read_mat_points(path) -> np.ndarray (N,2) via scipy.io.loadmat tolerating annote/gt keys; MissingDatasetError.
TDD: synthetic fixture (tiny PNGs + scipy.io.savemat in UCF layout Img/, GT_*.mat) -> implement -> pass -> commit.

### Task 4: ShanghaiTech adapter
Create: src/datasets/shanghaitech.py, tests/test_shanghaitech.py.
Interfaces: DatasetAdapter over Part_A|Part_B/{train,test}/images + ground-truth/GT_IMG_n.mat; source = shanghaitech_partA_train etc. (group id).
TDD: Part_A/Part_B fixture -> implement -> pass -> commit.

### Task 5: UCSD anomaly adapter
Create: src/datasets/ucsd_anomaly.py, tests/test_ucsd_anomaly.py.
Interfaces: AnomalySequence(video_path, sequence_id, n_frames, frame_labels); UcsdAdapter.inspect() + iter_sequences(root); parses videos/{Ped1,Ped2}/{train,test}/*.avi + gt labels (frame images or list files; absent -> notes).
TDD: 2 tiny .avi via cv2.VideoWriter + per-frame labels fixture -> implement -> pass -> commit.

### Task 6: Preprocessing (image/video/webcam)
Create: src/preprocessing/images.py, videos.py, camera.py, tests/test_preprocessing.py.
Interfaces: load_image(path)->np.ndarray raises InvalidImageError; iter_video_frames(path, max_dim=1280, stride=1)->Iterator[(int,float,np.ndarray)] raises InvalidVideoError; Camera open/read/release + context manager; CameraError.
TDD: garbage bytes/wrong ext/nonexistent video/Camera(999) raises/release idempotent -> implement -> pass -> commit.

### Task 7: Detection (YOLO person detector + image analysis)
Create: src/detection/detector.py, draw.py, image_analysis.py, tests/test_detection.py.
Interfaces: Detection(xyxy, conf, cls); PersonDetector.detect(image_bgr)->list[Detection] (yolov8n, classes=[0], conf .25, cpu, cached); draw_detections(image, dets); ImageAnalysis(count, detections, annotated, static_features); analyze_image(path_or_array, detector=None).
TDD: draw math + monkeypatched-detector analyze (count==3, features exact) -> implement -> pass -> commit.

### Task 8: Tracking (ByteTrack + trajectories)
Create: src/tracking/tracker.py, trajectories.py, tests/test_tracking.py.
Interfaces: CrowdTracker.update(dets, frame_idx, t)->list[TrackState] via BYTETracker; TrajectoryStore.add(...) -> to_dataframe() cols [track_id, frame_idx, t, x1,y1,x2,y2,cx,cy]; persisted_tracks(min_frames).
TDD: synthetic sequential dets (stable ids, schema, persistence) -> implement -> pass -> commit.

### Task 9: Feature engineering + schema
Create: src/features/schema.py, static.py, temporal.py, docs.py, tests/test_features.py.
Interfaces: FeatureSpec(name, dtype, unit, limitation); STATIC_FEATURES (person_count, density_per_mpx, grid_concentration, spread_x, spread_y, grid_entropy...); MOVEMENT_FEATURES (density_delta, mean_speed_pxf, speed_std_pxf, direction_entropy, directional_consistency, flow_mean_magnitude, flow_std, track_count, track_persistence...); FeatureSchema.names + to_array (KeyError on missing); extract_static(dets, frame_shape, image_megapixels=None); extract_movement(traj_df, window_frames, frame_shape, fps, prev_density=None); feature_groups().
TDD: exact geometry math tests, schema validation, unit+limitation present -> implement -> pass -> commit.

### Task 10: Visualization helpers
Create: src/visualization/plots.py, tests/test_visualization.py.
Interfaces: density_heatmap(image, dets); plot_count_over_time(times, counts, path); plot_confusion_matrix(y, yhat, labels, path); plot_feature_importance(names, importances, path); plot_roc(y, scores, path); Agg; returns Path.
TDD: files created non-empty -> implement -> pass -> commit.

### Task 11: Grouped splitting + ML trainers + bundles
Create: src/models/splitting.py, trainers.py, bundle.py, tests/test_ml.py.
Interfaces: group_train_val_test(groups, y, val_frac=.15, test_frac=.15, seed=42)->Split (raises on group leakage or <3 groups); build_estimator(name) for logreg|dt|rf|svm|xgb with scaler inside pipeline; train_one(name, X_train, y_train, X_val, y_val)->TrainedModel(name, pipeline, metrics); save_bundle/load_bundle joblib dict {pipeline, feature_names, task, metrics, created_at, config} raising ModelNotFoundError/SchemaMismatchError.
TDD: leakage test, 5 estimators fit, round-trip, tampered names, missing file -> implement -> pass -> commit.

### Task 12: Counting baseline
Create: scripts/train_counting.py, src/evaluation/counting.py, tests/test_counting_eval.py.
Interfaces: evaluate_counts(gt, pred)->dict(mae, rmse, n); CLI: per counting dataset analyze_image over annotated images -> reports/results/counting_<dataset>.csv + counting_summary.json (dataset, n_images, mae, rmse, detector, timestamp); missing dataset -> clean message.
TDD: exact MAE/RMSE on known arrays -> implement -> fixture run -> commit.

### Task 13: Feature-table builder (prepare_data)
Create: scripts/prepare_data.py, src/features/table_builder.py, tests/test_table_builder.py.
Interfaces: build_anomaly_table(root, out_csv, window_frames, stride) -> sliding-window rows: static+movement features + label ONLY from UCSD annotations (window label = any-anomalous; raises if gt absent) + group=sequence_id -> data/processed/anomaly_features.csv.
TDD: fixture rows/groups/label-source tests -> implement -> pass -> commit.

### Task 14: Experiment runner (A-D, E gated)
Create: scripts/train_ml.py, src/evaluation/experiments.py, tests/test_experiments.py.
Interfaces: run_experiment_a (5 algos, identical grouped splits -> exp_a_metrics.csv + confusion PNGs); run_experiment_b (['static'] vs ['static','movement'] -> exp_b_metrics.csv + ROC); run_experiment_c (ablation via feature_groups() -> exp_c_ablation.csv + importance); run_experiment_d (cross-sequence/dataset transfer -> exp_d_metrics.csv); run_experiment_e (only if enabled+labels else exp_e_status.json not_supported). Fields per result: dataset, model, n_samples, class_distribution, split_groups, inference_ms.
TDD: synthetic 2-group table tests -> implement -> pass -> commit.

### Task 15: Inference service
Create: src/models/inference.py, tests/test_inference.py.
Interfaces: discover_models(models_dir)->list[ModelInfo]; predict(features, bundle)->Prediction(label, score, valid_features); RollingWindowFeatureExtractor(window_frames, fps).push(frame_idx, t, dets, traj_df, frame_shape)->dict|None.
TDD: no-models empty list, wrong feature set -> clear error, window None until full -> implement -> pass -> commit.

### Task 16: Streamlit shell (Overview + Methodology)
Create: app.py, src/ui/common.py, page_overview.py, page_methodology.py, tests/test_dashboard.py.
Interfaces: page_header, model_status_badge, require_file, list_result_files; pages expose render(); app.py = set_page_config + sidebar st.radio nav + dispatch; Methodology = real limitations.
TDD: imports + `python -m streamlit run app.py --server.headless true` 8s no traceback -> implement -> pass -> commit.

### Task 17: Photo Analysis page
Create: src/ui/page_photo.py. JPG/JPEG/PNG validation -> spinner -> original+annotated, count, density viz, static features; model output only if compatible bundle; never temporal/risk outputs.
TDD: rejects .gif/bad bytes; render callable -> implement -> pass -> commit.

### Task 18: Video Analysis page
Create: src/ui/page_video.py. mp4/avi/mov/mkv -> stride+downscale -> annotated output video + count/movement plots + model outputs if bundle; progress bar; corrupt video clean error.
TDD: fixture avi accepted, .txt rejected -> implement -> pass -> commit.

### Task 19: Live Camera page
Create: src/ui/page_camera.py. Device index, Start/Stop, stride+downscale, rolling features, CameraError -> friendly status, finally release; model output only if bundle.
TDD: monkeypatched Camera failure path -> implement -> pass -> commit.

### Task 20: Dataset Explorer page
Create: src/ui/page_datasets.py. Live adapter.inspect() cards; missing root -> placement hint; no downloads.

### Task 21: Training & Evaluation + Research Results pages
Create: src/ui/page_training.py, src/ui/page_results.py. Training: subprocess runs with log tail. Results: ONLY existing reports/results files; missing -> exact command; charts labeled dataset+model.

### Task 22: Full test pass + end-to-end verification
python -m pytest tests/ -v -> fix all; streamlit headless start/stop; scripts/inspect_dataset.py graceful with absent datasets; record actual results.

### Task 23: README + runbook
Overview, research question, honest limitations, dataset placement guides, optional congestion annotation format (data/metadata/annotation_format.md), PowerShell commands (install, inspect, prepare, train, evaluate, pytest, `python -m streamlit run app.py`), artifact map, structure tree.

## Scope change (2026-10-18, user decision)
- Datasets: UCSD anomaly ONLY. Tasks 3/4 (QNRF/ShanghaiTech adapters) and Task 12 (counting-GT baseline) are DROPPED. 
- UCSD adapter targets frame-sequence folders (.tif) + .m frame GT (not .avi as originally assumed).
- Counting = detection estimate without GT; documented as limitation. The spec (docs/superpowers/specs/) reflects this.
