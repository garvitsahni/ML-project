# CrowdSense AI — Production-Grade Frontend Redesign (Streamlit)

**Spec (binding authority):** the user's master prompt
"OpenCode Master Prompt — CrowdSense AI Production-Grade Frontend Redesign"
(conversation of 2026-10-23), plus `README.md` honesty rules (no fabricated
metrics, units policy, limitations). Conflicts resolve in favor of the spec.

**Architecture decision (spec §2, confirmed by audit):** the app is
Streamlit 1.51 / Python 3.13. We keep Streamlit and rebuild the UI within it.
No React rewrite, no new runtime dependencies (Altair ships with Streamlit).

**Branch:** `frontend-redesign` (worktree isolation rejected — see ledger
ruling: gitignored artifacts `models/`, `data/raw/`, `reports/` are required
for verification and do not exist in a fresh worktree).

## Global constraints

1. Never hardcode metrics, predictions, file paths or dataset availability;
   every number rendered must be read from a real file or real computation.
2. Every page module exposes a callable `render()` (tests assert this).
3. Keep `src.ui.common.model_status_badge(path) -> (text, emoji)` and
   `src.ui.page_photo.analyze_upload(bytes, detector=...) -> (ndarray, result)`
   signatures working (regression tests depend on them).
4. No new dependencies in `requirements.txt`.
5. Full suite must stay green after every task:
   `powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1`
   (125 tests baseline, ~110s).
6. TDD: failing test first for every new production function. Pure helpers
   get unit tests; page wiring gets smoke assertions (import/render/headless).
   Theme/CSS/config tasks are exempt (configuration) but still smoke-tested.
7. Distinguish in UI copy: measured observation / model prediction / anomaly
   score / gated label. Photo page must never show temporal claims.
8. Streamlit runs are non-interruptible mid-script: any long loop must be
   batched across reruns (st.fragment(run_every=...)) so Cancel/Stop buttons
   work at batch boundaries.

## Shared interfaces (pre-flight)

- **Task 2 → Tasks 3–9:** `src/ui/theme.py` (`TOKENS`, `CSS`, `inject_theme()`)
  and `src/ui/components.py` (`page_header`, `card`, `pill`, `status_pill`,
  `metric_tile`, `empty_state`, `error_state`, `kv_rows`, `format_*`,
  `render_limitations`, `data_note`) and `src/ui/charts.py`
  (`timeline_series`, `grouped_bars`, `bar_h`, `style_chart`).
  Task 2 defines these exact names; consumers must not rename them.
- **Task 3 → Tasks 4–9:** `src/ui/state.py` (`init_session()`, `settings()`,
  `validate_settings()`, `record_analysis()`, `get_history()`,
  `clear_history()`, `release_camera()`, `cleanup_resources()`) and the nav
  registry `PAGES` in `app.py` (used by `st.switch_page`).
  Task 3 also owns the shell contract: `cleanup_resources()` is called on
  every app run before page dispatch.
- **Task 5 → Task 6:** shared frame-loop helper
  `src/ui/analysis_loop.py` (`FrameAnalyzer`) produced by Task 5, consumed
  by Task 6 (video + camera currently duplicate detect→track→flow logic).
- **All page tasks:** when adding/removing a page module, update the
  `PAGES` list in `tests/test_dashboard.py` in the same task.
- **Task 7/9 remove legacy modules** `page_methodology.py`,
  `page_training.py`, `page_results.py` (content folded into Research Lab /
  Reports — ledgered ruling). Task 7 removes methodology+training; Task 9
  removes results. Tests updated in the same tasks.

## Task 1: Audit and baseline (Phase 1)

Status: **complete** (evidence: audit report in conversation; baseline test
run `scripts/run_tests.ps1` → 125 passed in 111.30s on `frontend-redesign`).

## Task 2: Design system (Phase 2)

Create the visual foundation everything else consumes.

Steps:
1. RED: create `tests/test_ui_components.py` with failing tests for pure
   helpers: `theme.TOKENS` contains required keys (bg, surface, border, text,
   accent, success, warn, danger, radii); `theme.CSS` contains the accent
   hex and a `:root` block; `components.pill("ok")` returns html containing
   the label and an "ok" class; `components.kv_rows([("a", 1)]) ==
   [["a", "1"]]`; `components.format_count`, `format_pct`, `format_seconds`
   round/format per spec; `components.page_header` module import + callable;
   `charts.timeline_series` returns an alt.Chart with y-axis title containing
   the unit.
   Expected: FAIL (module theme.py/components.py/charts.py do not exist).
2. GREEN: implement:
   - `.streamlit/config.toml` — theme: primaryColor `#2DD4BF`,
     backgroundColor `#0B1220`, secondaryBackgroundColor `#111A2B`,
     textColor `#E6EDF3`, font `sans`, baseRadius `10`.
   - `src/ui/theme.py` — `TOKENS` dict (deep-navy charcoal palette, teal
     accent, spacing scale, radius scale) and `CSS` string (sidebar styling,
     card surfaces, header typography, focus-visible rings, status colors,
     table/density tweaks, reduced-motion guard). `inject_theme()` renders
     once per session (session_state flag, `st.markdown(..., unsafe_allowHTML=True)`).
   - `src/ui/components.py` — rendering helpers listed in Shared interfaces
     (all built on native widgets + `st.container(border=True)` for cards;
     emoji only as fallback, prefer text status with icon-shape+color not
     color-alone).
   - `src/ui/charts.py` — Altair builders with axis titles incl. units,
     tooltips, dark-friendly scheme, height defaults.
   Expected: new tests pass.
3. Run full suite. Expected: 125 + new pass.
4. Commit: `feat(ui): design system — theme tokens, shared components, chart builders`.

Test command: `powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1`

## Task 3: Application shell, navigation, state, Overview (Phase 3)

Steps:
1. RED: `tests/test_ui_state.py` — failing tests:
   `validate_settings` rejects out-of-range confidence/stride and returns
   defaults + errors; `settings()` returns defaults after `init_session()`;
   `record_analysis` appends and caps at 50 entries with timestamp;
   `get_history` sorted newest-first; `clear_history` empties.
   Expected: FAIL (`state.py` missing).
2. GREEN: `src/ui/state.py` per Shared-interfaces contract. Settings keys:
   `conf_threshold` (0.05–0.95, default 0.25), `frame_stride` (1–10, default 1),
   `max_frames` (60–1200, default 300), `process_max_side` (640/1280/1920,
   default 1280), `selected_model` (None = newest bundle), `show_density` (True).
3. Rewrite `app.py`: `st.set_page_config` (no emoji icon; title "CrowdSense AI"),
   `st.navigation` with sections — **Workspace**: Overview (icon `home`,
   default); **Analyse**: Photo Analysis (`image`), Video Analysis
   (`video_library`), Live Camera (`videocam`); **Research**: Research Lab
   (`science`), Dataset Explorer (`storage`), Model Registry (`hub`),
   Reports (`assessment`); **System**: Settings (`tune`). Sidebar renders
   identity block (product name + "Crowd Analytics Research" caption) via
   `st.sidebar`, model status badge, and settings summary. Every run:
   `state.init_session()`, `state.inject/…`, `state.cleanup_resources()`
   before dispatch (camera release on nav-away).
   Verify installed Streamlit accepts `st.Page(callable)` +
   `st.switch_page(Page)` before committing to it (inspect site-packages);
   if switch_page rejects Page objects, fall back to url-path string pages.
   Expected: `st.navigation` shell starts headless, all 9 destinations listed.
4. Rewrite `src/ui/page_overview.py`: product intro; three primary action
   buttons → `st.switch_page` to photo/video/camera; **System status** row
   (models available: count from `common.find_model_files()`; dataset ready:
   `UcsdAdapter` inspection with cached probe, graceful missing state;
   latest experiment: `reports/results/evaluation.json` if present else
   empty state "No trained model evaluation yet — run Research Lab");
   **Available models** list (5 bundles, algorithm + mtime); **Recent
   activity** from session history with "session-only" caption or
   "No analyses yet" empty state; shortcuts to research results.
   Expected: real counts only; no decorative metric cards.
5. Update `tests/test_dashboard.py` PAGES list if module set changed (not
   yet in this task); add headless smoke already covers `app.py`.
6. Run full suite + start app headless manually to eyeball shell.
   Expected: green.
Commit: `feat(ui): app shell — sectioned st.navigation, session state store, overview redesign`.

Test command: `powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1`

## Task 4: Photo Analysis workspace (Phase 4)

Steps:
1. RED: extend `tests/test_dashboard.py`/new tests: `analyze_upload` keeps
   working (existing regression test must pass unchanged); new pure helper
   `page_photo.validate_upload(name, size, content_type) -> error|None`
   (rejects >20MB, wrong extension/mime, empty file) — write failing tests.
   Expected: FAIL (validate_upload missing).
2. GREEN rewrite `src/ui/page_photo.py`:
   - uploader (jpg/jpeg/png/tif/bmp, ≤20 MB) with validation errors as
     actionable `error_state`; drag-drop note (native uploader supports it).
   - Preview; explicit **Analyse image** button (results computed only on
     click, stored in `state`), spinner during run; **Replace image** and
     **Reset analysis** actions.
   - Results: original vs annotated side-by-side (`st.image`), person count
     as `metric_tile` labeled "detector estimate (±, no counting GT)",
     detection-confidence summary (mean/min conf, per-detection table with
     boxes), density grid heatmap (Altair or styled dataframe) when
     `show_density`, static feature table, model/limitation panel (why no
     temporal claims from one frame).
   - Processing time measured with `time.perf_counter`, real.
   - Export: annotated PNG download (`cv2.imencode` of drawn frame).
   - No-detections → informative empty state, not an error.
   - Record analysis into `state.record_analysis("photo", ...)`.
   Expected: existing regression + new tests green.
3. Full suite. Expected: green.
Commit: `feat(ui): photo analysis workspace — gated analysis, validation, annotated export`.

Test command: `powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1`

## Task 5: Video Analysis workspace + shared analysis loop (Phase 5)

Steps:
1. RED: tests for `src/ui/analysis_loop.py` `FrameAnalyzer` (extracted from
   duplicated video/camera logic): processes a synthetic frame list with a
   fake detector → returns per-frame counts/speeds and window aggregates;
   test temp-file hygiene helper `video_store.save_upload(hash, bytes) ->
   path` reuses one path per hash and replaces old files.
   Expected: FAIL (module missing).
2. GREEN: build `FrameAnalyzer` (detect → ByteTrack → Farnebäck flow →
   `FrameSignals` → window features, callback per batch) +
   `src/ui/video_store.py` (one temp file per session, keyed by content
   hash; replaced on new upload; `cleanup()` removes dir on session cleanup
   via `state.cleanup_resources`).
3. Rewrite `src/ui/page_video.py`:
   - Drag-drop uploader (mp4/avi/mov/mkv, ≤200 MB) + format note + validation.
   - Preview (`st.video`), sampling controls (max frames, stride, confidence
     from settings), explicit **Process video** button — processing runs in
     `st.fragment(run_every="0.5s")` batches of ≤40 frames so the UI never
     blocks; progress bar + **Cancel** that works at batch boundary; results
     accumulate in `state` (`video_result`).
   - Results: annotated preview (first/last/peak frames drawn with boxes +
     track IDs as image strip; optional annotated video export via
     `cv2.VideoWriter` at sampled frames → download link), timeline charts
     (people count over time, mean speed px/frame, anomaly score per window —
     Altair with units on axes), stats panel (count mean/max, speed mean/p95,
     directional consistency, anomaly summary only if a bundle exists),
     prediction-vs-measurement labeling, occlusion/perspective/tracking
     limitation notes.
   - Export: features CSV (existing download kept) + results JSON summary.
   - Record into session history.
   Expected: green.
4. Full suite. Expected: green.
Commit: `feat(ui): video workspace — batched non-blocking processing, timeline charts, export; extract FrameAnalyzer`.

Test command: `powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1`

## Task 6: Live Camera workspace (Phase 6)

Steps:
1. RED: tests: `state.release_camera()` releases a fake cap and clears keys;
   camera page exposes `step_capture(n)` pure-ish stepper that processes n
   frames from an injected capture and returns metrics dict (fake cap).
   Expected: FAIL.
2. GREEN rewrite `src/ui/page_camera.py`:
   - Device selection (index list via `cv2` probe), permission/error states,
     Start/Stop buttons; capture loop runs in `st.fragment(run_every="0.4s")`
     batches (~15 frames per invocation) so Stop works at batch boundary and
     the script never blocks >1s; rolling-window metrics (count, speed
     px/frame, directional consistency, anomaly score iff bundle exists),
     measured FPS + per-frame latency (real timings), density overlay when
     `show_density`, observation timeline (last N windows) as Altair chart.
   - Cap stored in `st.session_state`; released on Stop, on error, and by
     `state.cleanup_resources()` when navigating away (shell calls it every
     run). `finally: cap.release()` inside batch only when stopping.
   - Model status shown; friendly failure when no camera device
     (`CameraUnavailableError` message preserved).
   - Never fabricate metrics when camera is stopped: panels show empty state.
   Expected: green.
3. Full suite. Expected: green.
Commit: `feat(ui): live camera — batched capture, stop control, measured FPS/latency, resource release`.

Test command: `powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1`

## Task 7: Research Lab (Phase 7a)

Folds in legacy `page_methodology.py` + `page_training.py` (ledgered ruling:
destination renamed per spec §9; features preserved).

Steps:
1. RED: tests for pure helpers: `page_research.load_experiment_results(dir)
   -> dict` merges ablation/algorithms/evaluation JSONs (returns only files
   that exist, empty dict otherwise); `page_research.build_findings(results)
   -> list[str]` generates findings ONLY from present numbers (e.g. best
   algorithm by test AUC, C−A delta) and returns [] when data missing;
   `page_research.experiment_table(results) -> DataFrame` shape test.
   Expected: FAIL (`page_research.py` missing).
2. GREEN `src/ui/page_research.py`:
   - Research question header (from README) + methodology/limitations section
     (content from `common.limitations()` + methodology page, single source).
   - Experiment configuration rendered from `configs/experiments.yaml`
     (seed, window/stride, algorithms) + split info from
     `data/processed/ucsd_windows.meta.json` when present.
   - **Run controls**: four real subprocess runners (inspect, prepare, train,
     evaluate) in `st.fragment(run_every="1s")` status panel — Popen to log
     file, elapsed timer, tail of last 30 lines, **Cancel** → `terminate()`
     (real), one active run at a time, confirmation before train/prepare.
     Commands shown exactly as executed.
   - **Results**: A/B/C ablation grouped bars (test AUC/F1 per algorithm,
     Altair), algorithm comparison D chart, evaluation.json bundle metrics
     table, experiment status incl. Experiment E gate (labelled "gated — no
     congestion labels"), figure gallery (confusion/ROC/importance PNGs)
     with algorithm+type selectors, findings box generated by
     `build_findings` from actual JSONs, provenance notes (file names).
   - Empty states with exact commands when result files absent.
3. Remove `src/ui/page_methodology.py`, `src/ui/page_training.py`; update
   `tests/test_dashboard.py` PAGES (drop them, add `page_research`) and any
   tests importing them (grep first).
   Expected: green suite (125 baseline − moved tests + new).
Commit: `feat(ui): research lab — experiment runs with cancel, real result charts, methodology folded in`.

Test command: `powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1`

## Task 8: Dataset Explorer + Model Registry (Phase 7b)

Steps:
1. RED: tests: `page_datasets.dataset_summary(adapter) -> dict` (counts from
   injected fake adapter), `page_models.list_bundles(models_dir) -> list[dict]`
   reads real/mocked joblib bundles returning name/algorithm/trained_at/
   metrics fields with missing-field tolerance; `page_models.metrics_for_
   (algo, evaluation_json)` mapping from evaluation.json with None on miss.
   Expected: FAIL.
2. GREEN `src/ui/page_datasets.py` rewrite: cached `UcsdAdapter` probe
   (`st.cache_data` keyed by dataset mtime/absence), summary cards
   (sequences/frames/anomalous labels), split counts + class distribution
   charts from `ucsd_windows.meta.json`, per-sequence table in expander,
   annotation/licence/source block (from README facts), dataset-kind explainer
   (counting vs anomaly vs congestion-labelled — congestion shown as
   "absent / gated"), placement hint + exact inspection command when absent;
   never claims download.
3. GREEN `src/ui/page_models.py` (new): bundle cards from `models/*.joblib`
   (name, algorithm, family, task: anomaly-window classification, training
   dataset, feature schema from bundle.feature_names, trained_at, n_train,
   validation metrics when in bundle, artifact size/mtime, availability);
   evaluation.json test metrics table per algorithm; compatibility note
   (bundles require the 30-window feature schema → inference on video/camera
   windows only, not photos); **default inference model selector** wired to
   `settings()["selected_model"]` — validates the pick is a known bundle
   before use (else falls back to newest + warning); known-limitations block.
4. Full suite. Expected: green.
Commit: `feat(ui): dataset explorer + model registry from real artifacts`.

Test command: `powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1`

## Task 9: Reports & history + Settings (Phase 8)

Steps:
1. RED: tests: `page_reports.history_frame(entries) -> DataFrame` columns/
   formatting; `page_reports.export_csv(entries) -> str` round-trips;
   settings page validation errors surface via `state.validate_settings`
   (already tested) + `page_settings.diagnostics() -> dict` returns only
   whitelisted keys (python/streamlit versions, torch device, dataset/model
   booleans — assert no absolute paths, no env vars).
   Expected: FAIL.
2. GREEN `src/ui/page_reports.py` (replaces `page_results.py`): session
   history table (time, input type, model, key results, duration, view
   details via expanders) with prominent "session-only — not persisted"
   caption, CSV export of history, clear-history with confirmation; plus
   **Research results** section: existing JSON result files (tables),
   figures gallery, export to CSV where data exists (no PDF — not
   implemented, so not offered). Empty states with exact commands.
3. GREEN `src/ui/page_settings.py` (new): settings form bound to
   `state.settings()` with validation + sensible defaults + "Reset to
   defaults"; per-setting help tooltips (units, effect); visualization
   toggles; diagnostics panel (whitelisted keys only); explicit note that
   no secrets/paths are exposed. Wire settings into photo/video/camera pages
   (confidence threshold actually passed to detector where supported —
   verify `src/detection/detector.py` API; if unsupported, the setting is
   not offered).
4. Remove `src/ui/page_results.py`; update `tests/test_dashboard.py` PAGES.
   Full suite. Expected: green.
Commit: `feat(ui): reports/history + settings & diagnostics`.

Test command: `powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1`

## Task 10: Responsiveness, accessibility, interaction QA (Phase 9)

Steps:
1. Audit every page for: horizontal overflow risks (fixed widths), labelled
   controls, focus visibility (theme CSS), status-not-color-alone (pill
   includes text), tooltips on technical metrics, all buttons functional
   (grep for buttons without handlers), skeleton/empty/loading states.
   Add regression tests where logic was added (e.g. `pill` includes text
   label, `empty_state` used in camera when stopped — pure parts only).
2. Manual responsive pass by launching the app and checking 1440/1280/768/390
   (Streamlit sidebar collapse behaviour; tables via `use_container_width`);
   fix issues found (CSS max-widths for content, dataframe wrapping).
3. Full suite + headless smoke. Expected: green.
Commit: `fix(ui): responsive/accessibility pass — focus states, labels, overflow`.

Test command: `powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1`

## Task 11: Run the app, fix errors, README (Phase 10)

Steps:
1. Launch `python -m streamlit run app.py --server.headless true`, drive
   key flows (navigate all 9 pages, photo analysis with a fixture image,
   video processing with a short synthetic clip, camera graceful failure),
   fix any runtime errors (tracebacks must not appear).
2. Update `README.md`: new dashboard-pages section, screenshots not required,
   documented start command, IA description, session-only history note,
   settings list, Research Lab run controls.
3. Full suite. Expected: green; headless smoke green.
Commit: `docs: README — redesigned interface, IA, settings, runbook`.

Test command: `powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1`

## Review focus (final whole-branch review)

Input classes the tests do NOT exercise: actual visual quality vs spec §3,
real Streamlit runtime behaviour (fragments, switch_page), camera resource
release under navigation, video temp-file lifecycle, honest copy (no
fabricated telemetry), settings actually affecting pipelines.
