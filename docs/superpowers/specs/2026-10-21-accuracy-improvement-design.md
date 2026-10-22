# Design: Meaningful Accuracy Improvement for CrowdSense AI

Date: 2026-10-21
Status: draft, awaiting human-partner approval
Scope: ML accuracy (primary) + targeted frontend UX fixes (secondary)

## Intent and why

The project answers one research question honestly: *do temporal
crowd-movement features improve anomaly prediction over static density
features alone?* They do — every algorithm gains +0.11 to +0.25 AUC.
But the best absolute test AUC is **0.675** (logreg, static+movement),
which the owner finds disappointing. The owner wants the headline numbers
themselves to get meaningfully better, while keeping the project's
guarantees: real data only, grouped splits, no fabricated metrics, no
counting ground truth claims.

Separately, the owner finds the Streamlit dashboard awkward and not
user-friendly. That is a secondary workstream here (a fuller redesign is
out of scope).

## Ground truth from recon

- UCSD frames are tiny (ped1 238×158, ped2 240×360), people ~10–30 px tall.
- `PersonDetector` uses YOLOv8n at `conf=0.25`, default `imgsz`, no
  multi-scale/TTA. Detection recall on such frames is unmeasured and
  presumed poor; every downstream feature inherits that noise.
- The feature table has 15 features (6 static, 9 movement). None encode
  the interaction/flow-violation signals that define UCSD anomalies
  (wrong-way movers, sudden stops, fast intruders).
- Window label = 1 if ANY frame in the window is anomalous. Noisy for
  windows with one anomalous frame out of thirty.
- Hyperparameters are fixed defaults; no CV search, no ensemble,
  prediction threshold hard-coded at 0.5 via `predict()`.
- Full UCSD feature-table rebuild takes on the order of an hour (CPU-only,
  Windows, thread-pinned).

## Goals

1. Raise best-model held-out **test AUC from 0.675 to ≥ 0.75**, same seed,
   same grouped splits, same UCSD data — no cherry-picking, test split
   touched once per final config.
2. Improve F1 alongside AUC (no AUC-at-all-costs; report both).
3. Every gain attributable and documented: before/after table in README
   with provenance, new result JSON files alongside the old ones.
   **Owner decision (2026-10-22): capped, pre-declared test evaluations.**
   Exactly these test-split evaluations run, each logged with provenance
   in its results JSON, nothing else ever touches test:
   1. baseline reproduction — current 15 features, chosen detector config
   2. + new movement features (schema v2)
   3. + tuned hyperparameters + threshold selection
   4. + ensemble (final headline config)
   All intermediate feature/threshold/ensemble decisions use train-side
   GroupKFold CV and the val split only. The README before/after table is
   built from these four runs.
4. Keep all existing guarantees: grouped sequence splits, px/frame units,
   `null` for undefined metrics, honesty labels on detector estimates.
5. Modest frontend UX pass: clearer navigation, progress feedback during
   long operations, and result presentation that a first-time user can
   follow without reading the methodology page.

## Non-goals / anti-goals

- No density CNN or counting regressor (CPU-only constraint stands).
- No new datasets (UCF-QNRF, ShanghaiTech stay dropped).
- No m/s conversion, no congestion-label invention, Experiment E stays gated.
- No claiming the result is "good" in absolute terms — 0.75–0.85 is the
  realistic ceiling for sequence-held-out UCSD; the README must keep saying so.
- No full frontend redesign, no framework change (Streamlit stays).
- No silent test-split reuse: tuning happens on train/val only.

## Constraints

- Python 3.13, Windows, CPU-only (torch CPU, no CUDA), ~1-hour full rebuilds.
- OpenMP flakiness on Windows: tests via `scripts/run_tests.ps1`,
  `n_jobs=1` for parallel-capable estimators.
- Feature schema is versioned and frozen at load time; adding features
  bumps the schema version and invalidates old bundles (accepted: retrain).
- Existing TDD suite (~124 tests) must stay green; new behavior gets tests
  first, matching the project's style.

## The how — workstreams

### W1: Perception calibration (biggest lever, empirical)

Goal: make the detector actually see the UCSD crowd.

- Add a small calibration script `scripts/calibrate_detection.py` that runs
  YOLOv8n over a fixed sample of UCSD frames (train split only) across a
  small grid of `conf` (0.05/0.10/0.15/0.25) and `imgsz` (320/640/1280),
  and reports per-config detection counts, timing, and saves overlay
  images under `reports/detection_calibration/` for visual inspection.
- **Owner decision (2026-10-22): automatic selection, no blocking review.**
  Pre-declared selection rule: among configs whose projected full-rebuild
  time is ≤ ~2 hours (scaled from measured per-frame timing) and whose
  dashboard live-analysis path stays usable, pick the one with the highest
  mean detections per frame on the calibration sample. Ties break toward
  the faster config. The chosen config becomes the new documented default
  in `configs/experiments.yaml` with a README note (detector estimate, no
  counting GT — framed honestly); overlays stay published for anyone to
  audit the pick.
- No counting-GT claims anywhere; the calibration is explicitly a
  recall-for-features exercise, documented as such.

### W2: Richer movement features (the signal the README already proven)

Goal: encode flow-violation structure, not just flow averages. Candidate
features (all CPU-cheap, computed from existing `TrajectoryStore` +
Farneback flow; final set decided by val-set behavior + redundancy check):

- `nn_distance_mean` / `nn_distance_std` — nearest-neighbor distance
  between track centers (crowding texture).
- `abrupt_stop_fraction` — tracks whose speed drops below a scene-relative
  threshold inside the window.
- `direction_reversal_rate` — consecutive-step direction flips > ~120°.
- `fast_mover_fraction` — tracks exceeding k× the scene median speed
  (the classic UCSD intruder: bike, vehicle, wrong-way runner).
- `flow_direction_inconsistency` — spread of local optical-flow directions
  around the window mean (crowd moving against itself).
- `track_turnover_rate` — track births+deaths per frame (appearance/
  disappearance churn).
- Optionally `speed_p90` — high percentile of track speeds.

Process: TDD, schema v2 bump in `src/features/schema.py` with per-feature
units/limitations strings matching existing style, tests for edge cases
(empty window, <2 tracks, zero flow) mirroring existing feature tests.
Redundancy screen: drop any new feature with |Spearman| > ~0.9 against an
existing one before finalizing the schema.

**Design decision (author's call, from builder check):** schema v2 is the
full candidate superset; the parquet table always carries all schema-v2
columns; each bundle records the subset of feature names it was trained on
(`ModelBundle.feature_names` already exists for exactly this), and the
InferenceService computes the schema-v2 superset then selects the bundle's
subset. One full rebuild, no mid-project re-freeze, and the redundancy
screen runs on the rebuilt table without invalidating it.

**Design decision (author's call, from builder check):** the W2 feature
survival gate runs inside train-side GroupKFold CV (feature ablation on
train CV AUC), NOT on the val split. Val is reserved for exactly two
selection jobs: threshold selection and ensemble membership. This limits
overfit-by-selection on the ~15-sequence val split.

### W3: Label robustness (cheap, honest variant)

- `prepare_data.py` additionally writes `label_frac` (fraction of
  anomalous frames in the window) next to the existing binary `label`.
- Experiments keep the original any-frame `label` as the primary
  comparable definition; one new experiment reports AUC under a stricter
  window label (≥20% anomalous frames) as a labeled robustness variant,
  clearly marked as a different label definition in results + README.

### W4: Tuning + ensemble + threshold (medium lever)

- `src/models/tuning.py`: GroupKFold (groups = sequence_id) search on the
  train split only, small pre-declared grids per algorithm (e.g. logreg C,
  rf n_estimators/depth, xgb depth/eta/estimators). Seed fixed.
- Best-params retrained on full train split; val used for threshold
  selection (maximize F1), test evaluated once.
- Soft-vote ensemble: probability average of the tuned top-3 algorithms by
  val AUC; same single test evaluation.
- New results files (`experiment_tuning.json`, `experiment_ensemble.json`)
  so the original A/B/C/D outputs remain intact for comparison.
- All of it lands in `scripts/train_ml.py` behind flags with honest
  defaults, plus tests with tiny synthetic tables (matching existing
  experiment-test style).

### W5: Frontend UX pass (secondary, targeted)

- Navigation: sidebar radio → grouped sidebar with short descriptions or
  `st.navigation`-style structure; make the primary flow (Video Analysis →
  results) obvious.
- Progress feedback: `st.spinner`/progress for detection+inference paths
  with a friendly "this can take a while on CPU" note.
- Results clarity: present prediction with plain-language interpretation
  ("Anomaly likely — 82% model confidence; models are modest, see
  Methodology"), link to Methodology's limitations from every prediction.
- Keep Streamlit, keep the 8-page structure; no visual redesign, no new
  dependencies.

## Sequencing

1. W2 + W3 first (no hour-long rebuilds to decide anything; feature-table
   rebuild happens once afterward).
2. W1 calibration in parallel where possible; chosen detection config
   triggers the one full rebuild together with W2 features.
3. W4 on the rebuilt table (train/val only, test once at the end).
4. W5 anytime — independent of ML work.
5. README before/after table + honesty review as the closer.

## Success criteria

- Best-model test AUC ≥ 0.75 with F1 improved vs 0.517 baseline, same
  seed/splits (or an honest write-up of why the ceiling held, with the
  achieved numbers).
- All ~124 existing tests green; new features/tuning covered by new tests.
- README headline table updated with before/after + provenance; limitations
  section still true.

## Risks

- Detection calibration may show YOLOv8n can't do better at any setting on
  these frames → fallback: W2+W4 alone (expected +few points, may not hit
  0.75; documented honestly if so).
- Multiple hour-long rebuilds if detection config iterates → keep the
  calibration grid on a frame sample, rebuild full table once.
- New features could be noise on val → redundancy screen + val gate before
  schema freeze.
- Windows/OpenMP flakiness → existing ps1 wrapper and n_jobs=1 conventions.

## What's left to the builder (builder's call)

- Exact grids, search budgets, and code structure inside `src/models/tuning.py`.
- Test file layout and fixture details for new features/tuning.
- Exact sidebar structure and wording in W5.
- Choice of top-3 ensemble members (decided by val AUC at run time, not hard-coded).
