# Operator Console Light UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-skin the CrowdSense AI Streamlit app from the dark "AI dashboard" look into the approved light, calm, guided operator console — presentation only, zero behavior change.

**Architecture:** `src/ui/theme.py` stays the single source of truth (TOKENS + CSS), flipped to the approved light palette; `.streamlit/config.toml` mirrors the core colors so native widgets match (charts inherit automatically — `charts.py` reads `theme.TOKENS`). `components.py` loses the eyebrow header and gains a step-rail helper. Pages are re-organized for presentation: Overview becomes the approved v2 layout; Photo/Video gain Upload → Check → Results rails; Camera/Settings get plain-language headers. Every status value keeps its existing real read.

**Tech Stack:** Streamlit + stdlib only (CSS via `st.markdown(unsafe_allow_html=True)`; Altair charts unchanged).

**Spec:** `docs/superpowers/specs/2026-10-24-operator-console-design.md` (approved mockups: `docs/superpowers/designs/2026-10-24/*.html` — read both before starting).

## Global Constraints

- No new dependencies: Streamlit + stdlib only — custom CSS fine; no component libraries, no web fonts, no JS frameworks.
- No fabricated metrics: every number from real files/computation; no tracebacks rendered; no emoji as primary icons (inline SVG only).
- No dark-mode toggle; no behavior changes — this is presentation only.
- Palette (exact): bg `#FFFFFF`, panel `#FAFBFC`, tiles `#FBFCFD`, nav `#F6F8FA`, border `#E4E9EF`, heading `#0F172A`, body `#1B2430`, muted `#5B6675`/`#6B7686`, accent `#0F766E`, accent-soft `#E6F5F3`, success `#16A34A`, warn `#B45309`, danger `#DC2626`.
- No eyebrow labels anywhere; status = colored dot + word, never a bordered/neon pill; one accent family used sparingly.
- Session-only history with CSV export + clear-with-confirmation stays on Overview (keys `activity_clear_confirm` / `activity_clear_go` preserved).
- Full suite green after every task: `powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1`. TDD for all new production logic. Commit per task on `main`.
- Video per-run controls (2 sliders + export toggle) and camera device lifecycle remain untouched.
- App served live on port 8501 when the last task finishes.

## Review Focus

1. **Clear-history gating after the layout rework** — a click with no confirmation must never clear, and the disabled button must enable only after the checkbox (owner: Task 3, AppTest test).
2. **Step rails must not lie** — no "Upload done" before a file exists, no "Results" before a result exists (owner: Tasks 4–5, `photo_step`/`video_step` unit tests).
3. **Status strip honesty** — absent models/dataset/evaluation render warn/idle wording with real counts, never green (owner: Task 3, `status_items` tests).
4. **Video controls survive** — stride/max-frames sliders + annotated-export toggle still exist after the step rail lands (owner: Task 5, source-inspection test).
5. **Dark leftovers** — no old dark hexes or eyebrow rules in the shipped CSS/config (owner: Task 1, token/config tests).

---

### Task 1: Light theme foundation

**Files:**
- Modify: `src/ui/theme.py` (TOKENS values, `_RULES`)
- Modify: `.streamlit/config.toml`
- Test: `tests/test_ui_components.py`
- Commit: also stage `docs/superpowers/specs/2026-10-24-operator-console-design.md`, `docs/superpowers/designs/2026-10-24/`, `docs/superpowers/plans/2026-10-24-operator-console-light-ui.md`, `.gitignore` (first commit of the effort)

**Interfaces:**
- Produces: `theme.TOKENS` (same required keys: `bg, surface, border, text, accent, success, warn, danger, radii` + `info, text_muted, heading, sidebar_bg, space, max_content, font_sans`), `theme.CSS`, `theme.inject_theme()` — consumed by all pages, `charts.py`, `app.py` unchanged call sites.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ui_components.py`:

```python
def test_theme_tokens_light_direction():
    from src.ui import theme
    assert theme.TOKENS["bg"] == "#FFFFFF"
    assert theme.TOKENS["accent"] == "#0F766E"
    assert theme.TOKENS["success"] == "#16A34A"
    assert theme.TOKENS["border"] == "#E4E9EF"


def test_theme_css_drops_eyebrow_and_dark_hexes():
    from src.ui import theme
    assert ".page-eyebrow" not in theme.CSS
    for dark in ("#0B1220", "#111A2B", "#2DD4BF", "#0D1422"):
        assert dark not in theme.CSS


def test_streamlit_config_is_light():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert 'base = "light"' in text
    assert 'primaryColor = "#0F766E"' in text
    assert 'backgroundColor = "#FFFFFF"' in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ui_components.py -q -k "light or drops_eyebrow or config_is_light"`
Expected: FAIL — TOKENS still dark, `.page-eyebrow` still in CSS, config still `base = "dark"`.

- [ ] **Step 3: Flip `theme.py` to the light palette**

Rewrite `TOKENS` with the Global Constraints palette (keep every existing key; add `heading: "#0F172A"`; radii become `{"sm": "6px", "md": "8px", "lg": "10px"}`; keep `max_content`, `space`, `font_sans`). In `_RULES`: delete the `.page-eyebrow` rule; `.page-header h1 { color: var(--heading) }`; restyle `.pill` to quiet inline text (no border/background/padding — dot keeps `currentColor`, per-kind colors become `var(--success)`/`var(--warn)`/`var(--danger)`/`var(--text-muted)`); `.state-block`/metric tiles/`stMetric` to light surfaces with `var(--border)`; `.crowdsense-sub` loses `text-transform: uppercase` and letter-spacing; `.crowdsense-mark` gradient uses `var(--accent)` with `#FFFFFF` text. **Preserve verbatim** (tests pin them): `":root"`, the accent value appearing in CSS, `:focus-visible`, `outline: 2px solid var(--accent)`, `max-width: var(--max-content)`. Add new rules: `.status-strip` (flex row, `gap: 1rem`, `background: var(--surface)`, `border: 1px solid var(--border)`, `border-radius: var(--radius-sm)`, `padding: .55rem .9rem`, `.pill { margin-right: .9rem }` inside it) and `.steps` (flex; `.step` muted; `.step__num` 20px circle; `.step.is-done .step__num { background: var(--success); color: #fff }`; `.step.is-now .step__num { background: var(--accent); color: #fff; box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 25%, transparent) }`; `.step.is-now { color: var(--heading); font-weight: 650 }`).

- [ ] **Step 4: Flip `.streamlit/config.toml`**

`base = "light"`, `primaryColor = "#0F766E"`, `backgroundColor = "#FFFFFF"`, `secondaryBackgroundColor = "#FAFBFC"`, `textColor = "#1B2430"`, `borderColor = "#E4E9EF"`, `dataframeBorderColor = "#E4E9EF"`, `dataframeHeaderBackgroundColor = "#F6F8FA"`, `codeBackgroundColor = "#F1F5F9"`, sidebar `backgroundColor = "#F6F8FA"`, `chartCategoricalColors = ["#0F766E", "#60A5FA", "#FBBF24", "#DC2626", "#A78BFA", "#16A34A"]`; update the file's comment to light. Keep `font`, `baseRadius`, `buttonRadius`, `showSidebarBorder`.

- [ ] **Step 5: Run theme tests to verify they pass**

Run: `python -m pytest tests/test_ui_components.py -q`
Expected: PASS (existing token/focus/CSS tests + the three new ones).

- [ ] **Step 6: Run the full suite**

Run: `powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1`
Expected: all green (charts tests assert structure, not hexes).

- [ ] **Step 7: Commit**

```bash
git add src/ui/theme.py .streamlit/config.toml tests/test_ui_components.py docs/superpowers/specs/2026-10-24-operator-console-design.md docs/superpowers/designs/2026-10-24 docs/superpowers/plans/2026-10-24-operator-console-light-ui.md .gitignore
git commit -m "feat(theme): light console palette - tokens, CSS, config"
```

### Task 2: Eyebrow removal + step rail component

**Files:**
- Modify: `src/ui/components.py`
- Modify: `src/ui/page_overview.py:120-126`, `src/ui/page_photo.py:335-343`, `src/ui/page_video.py:610-618`, `src/ui/page_camera.py:474-481`, `src/ui/page_settings.py:148-153` (header call sites only)
- Test: `tests/test_ui_components.py`

**Interfaces:**
- Produces: `page_header(title: str, description: str | None = None) -> None` (eyebrow param removed); `STEP_LABELS = ("Upload", "Check", "Results")`; `steps_html(current: int, labels: Sequence[str]) -> str`; `step_rail(current: int, labels: Sequence[str]) -> None` — consumed by Tasks 4–5.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ui_components.py`:

```python
def test_page_header_has_no_eyebrow_parameter():
    import inspect
    from src.ui import components
    assert "eyebrow" not in inspect.signature(components.page_header).parameters


def test_steps_html_marks_done_now_and_pending():
    from src.ui import components
    labels = components.STEP_LABELS
    first = components.steps_html(1, labels)
    assert 'class="step is-now"' in first
    assert "is-done" not in first
    second = components.steps_html(2, labels)
    assert second.index("is-done") < second.index("is-now")
    assert "Upload" in second and "✓" in second
    clamped = components.steps_html(99, labels)
    assert clamped.count("is-now") == 1 and clamped.endswith("</ol>")
    escaped = components.steps_html(1, ("<b>",))
    assert "<b>" not in escaped and "&lt;b&gt;" in escaped
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ui_components.py -q -k "eyebrow_parameter or steps_html"`
Expected: FAIL — `page_header` still has `eyebrow`; `steps_html` missing.

- [ ] **Step 3: Implement in `src/ui/components.py`**

- `page_header`: drop the `eyebrow` parameter and its HTML branch (keep title/description markup and `.page-header` classes).
- Add `STEP_LABELS = ("Upload", "Check", "Results")`.
- Add `steps_html(current: int, labels: Sequence[str]) -> str` returning
  `<ol class="steps">…</ol>` where label *i* (1-based) renders as `<li class="step is-done"><span class="step__num">✓</span>label</li>` for `i < current`, `… is-now … <span class="step__num">i</span> …` for `i == current`, plain `… class="step" …` otherwise; `current` clamped to `1..len(labels)`; labels passed through `html.escape`.
- Add `step_rail(current, labels)` → `st.markdown(steps_html(current, labels), unsafe_allow_html=True)`.

- [ ] **Step 4: Remove `eyebrow=` from all five page header calls**

Delete only the `eyebrow=...` line in each of the five files listed above.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_ui_components.py -q`
Expected: PASS.

- [ ] **Step 6: Run the full suite**

Run: `powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1`
Expected: all green (the only `page_header` callers are the five pages).

- [ ] **Step 7: Commit**

```bash
git add src/ui/components.py src/ui/page_overview.py src/ui/page_photo.py src/ui/page_video.py src/ui/page_camera.py src/ui/page_settings.py tests/test_ui_components.py
git commit -m "feat(ui): drop eyebrow headers, add step-rail component"
```

### Task 3: Overview v2 layout

**Files:**
- Modify: `src/ui/page_overview.py` (full rewrite of `render()` + helpers)
- Test: `tests/test_dashboard.py`, create `tests/test_ui_flows.py`

**Interfaces:**
- Consumes: `components.pill(label, kind)`, `components.download_csv`, `components.history_rows`, `components.export_csv`, `state.get_history/clear_history/settings`, `common.{list_model_files, dataset_probe, load_evaluation, limitations}` (all unchanged).
- Produces: `page_overview.status_items(model_count: int, probe: dict, evaluation: dict, runs: int) -> list[tuple[str, str]]` returning `(kind, text)` pairs with kind in `("ok", "warn", "idle")`; widget keys `tile_photo`, `tile_video`, `tile_camera` (replacing `cta_*`), `activity_clear_confirm`, `activity_clear_go` (preserved).

- [ ] **Step 1: Write the failing test for `status_items`**

Append to `tests/test_dashboard.py`:

```python
def test_status_items_reports_real_state():
    from src.ui.page_overview import status_items

    ok_probe = {"available": True, "sequences": 200}
    ev = {"bundles": {"xgb": {}}, "n_test_windows": 48}
    assert status_items(3, ok_probe, ev, 2) == [
        ("ok", "3 model bundles ready"),
        ("ok", "Dataset ready · 200 sequences"),
        ("idle", "48 evaluation windows"),
        ("idle", "2 runs this session"),
    ]
    assert status_items(0, {"available": False}, {}, 0) == [
        ("warn", "No models yet"),
        ("warn", "Dataset not found"),
        ("idle", "No evaluation yet"),
        ("idle", "No runs yet"),
    ]
    assert status_items(1, ok_probe, ev, 1)[0] == ("ok", "1 model bundle ready")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_dashboard.py -q -k status_items`
Expected: FAIL — `ImportError: cannot import name 'status_items'`.

- [ ] **Step 3: Implement `status_items` in `src/ui/page_overview.py`**

Signature above. Rules: models — `model_count` bundles → `(model_count, "model bundle(s) ready")` with singular `bundle` when 1; else `("warn", "No models yet")`. Dataset — `probe.get("available")` → `("ok", f"Dataset ready · {probe['sequences']} sequences")`, else `("warn", "Dataset not found")`. Evaluation — `evaluation.get("bundles")` non-empty → `("idle", f"{evaluation.get('n_test_windows', 0)} evaluation windows")`, else `("idle", "No evaluation yet")`. Runs — positive → `(f"{runs} run(s) this session")` singular `run` when 1, else `("idle", "No runs yet")`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_dashboard.py -q -k status_items`
Expected: PASS.

- [ ] **Step 5: Rewrite `render()` and helpers to the approved v2 layout**

- Module constants: `_ICONS = {"photo": "...", "video": "...", "camera": "..."}` — the three inline SVG strings from `docs/superpowers/designs/2026-10-24/overview-v2.html` (stroke-only, 24×24, no emoji); `_TILES = [("Photo", "One image, quick count", "photo", "tile_photo", nav.PHOTO), ("Video", "File, processed in steps", ...), ("Camera", "Live from a device", ...)]` — nav objects imported lazily inside `render()`.
- `ui.page_header(title="What do you want to analyse?", description="Pick a task — system status and this session's runs are below.")`.
- Three bordered containers (`st.container(border=True)`): inside each `st.markdown` the SVG + bold name + muted one-line description (wrap in `<div class="tile">…</div>`), then `if st.button("Start →", key=tile_key): st.switch_page(page_obj)`.
- Status strip: one `st.markdown(f'<div class="status-strip">{"".join(ui.pill(text, kind) for kind, text in items)}</div>', unsafe_allow_html=True)` where `items = status_items(len(list_model_files()), _probe_cached(str(load_paths().ucsd_dir)), load_evaluation(), len(entries))` and `entries = state.get_history()` — all real reads, no caching changes.
- History block (only when `entries` non-empty): `st.expander(f"This session: {len(entries)} runs")` containing `st.dataframe(pd.DataFrame(ui.history_rows(entries), columns=["When", "Input", "Key results"]), hide_index=True, width="stretch")`; below it the links row in `st.columns([1, 1, 2])` — col1 `ui.download_csv("Download history CSV", ui.export_csv(entries), f"crowdsense_history_{datetime.now():%Y%m%d-%H%M}.csv")`; col2 the existing confirm checkbox + `st.button("Clear history", key="activity_clear_go", disabled=not confirmed) and confirmed:` → `state.clear_history(); st.rerun()` (exact existing pattern, keys preserved); col3 `st.caption(f"Settings: confidence ≥ {s['conf_threshold']:.2f} · stride {s['frame_stride']}")` from `state.settings()`. After the columns: `ui.data_note("Session-only — download the CSV to keep a copy.")`.
- When `entries` is empty: no expander, no links row (the strip already says "No runs yet").
- Close with `st.markdown("### About the numbers")` + `ui.render_limitations(limitations())`.
- Delete now-unused `_render_models_status`, `_render_dataset_status`, `_render_evaluation_status`, `_render_recent_activity`, `_goto` (tiles call `st.switch_page` directly). Trim imports to what the new code uses (`datetime`, `load_paths`, `ui`, `state`, `dataset_probe`, `limitations`, `list_model_files`, `load_evaluation`; pandas imported locally as today).

- [ ] **Step 6: Write the AppTest gating test**

Create `tests/test_ui_flows.py`:

```python
"""AppTest flow tests: widget gating that unit tests cannot reach."""
from streamlit.testing.v1 import AppTest

_OVERVIEW = """
import streamlit as st
from src.ui import state
if not st.session_state.get("seeded"):
    st.session_state["seeded"] = True
    state.record_analysis("photo", {"persons": 5, "model": "yolov8n.pt", "seconds": 0.8})
from src.ui.page_overview import render
render()
"""


def test_overview_clear_requires_confirmation():
    at = AppTest.from_string(_OVERVIEW, default_timeout=90).run()
    assert not at.exception
    assert any(getattr(w, "label", "") == "Download history CSV"
               for w in at.get("download_button"))
    clear = [b for b in at.button if b.key == "activity_clear_go"]
    assert len(clear) == 1 and clear[0].disabled
    at.checkbox[0].set_value(True).run()
    clear = [b for b in at.button if b.key == "activity_clear_go"]
    assert not clear[0].disabled
    clear[0].click().run()
    assert not at.exception
    assert len(at.dataframe) == 0          # history gone after confirmed clear
```

- [ ] **Step 7: Run the new flow test**

Run: `python -m pytest tests/test_ui_flows.py -q`
Expected: PASS (if `at.dataframe` is non-empty, the clear path did not run — fix `render()`, not the test).

- [ ] **Step 8: Run the full suite**

Run: `powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1`
Expected: all green.

- [ ] **Step 9: Commit**

```bash
git add src/ui/page_overview.py tests/test_dashboard.py tests/test_ui_flows.py
git commit -m "feat(ui): overview v2 - task tiles, quiet status strip, history expander"
```

### Task 4: Photo guided steps

**Files:**
- Modify: `src/ui/page_photo.py`
- Test: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: `components.step_rail`, `components.STEP_LABELS` (Task 2).
- Produces: `page_photo.photo_step(has_upload: bool, has_result: bool) -> int`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dashboard.py`:

```python
def test_photo_step_reflects_upload_and_result():
    from src.ui.page_photo import photo_step
    assert photo_step(False, False) == 1
    assert photo_step(True, False) == 2
    assert photo_step(True, True) == 3
    assert photo_step(False, True) == 3   # result wins over stale upload flag
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_dashboard.py -q -k photo_step`
Expected: FAIL — `ImportError: cannot import name 'photo_step'`.

- [ ] **Step 3: Implement `photo_step` and wire the rail**

- `photo_step(has_upload: bool, has_result: bool) -> int`: `3` if `has_result`, else `2` if `has_upload`, else `1`.
- In `render()`, immediately after `ui.page_header(...)`: read `nonce` (move the existing line up), `uploaded_flag = st.session_state.get(f"photo_uploader_{nonce}") is not None`, `result_flag = st.session_state.get("photo_result") is not None and the stored fingerprint's first two items (name, size) match the uploader-state object's `.name`/`.size` when present`, then `ui.step_rail(photo_step(uploaded_flag, result_flag), ui.STEP_LABELS)`. The uploader and everything below it stay byte-for-byte unchanged.

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_dashboard.py -q -k photo_step`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1`
Expected: all green (photo analysis tests untouched — they call helpers, not `render()`).

- [ ] **Step 6: Commit**

```bash
git add src/ui/page_photo.py tests/test_dashboard.py
git commit -m "feat(ui): photo page - Upload/Check/Results step rail"
```

### Task 5: Video guided steps

**Files:**
- Modify: `src/ui/page_video.py`
- Test: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: `components.step_rail`, `components.STEP_LABELS`.
- Produces: `page_video.video_step(has_upload: bool, phase: str | None) -> int`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dashboard.py`:

```python
def test_video_step_reflects_upload_and_phase():
    from src.ui.page_video import video_step
    assert video_step(False, None) == 1
    assert video_step(True, None) == 2
    assert video_step(True, "processing") == 2
    assert video_step(True, "cancelled") == 2
    assert video_step(True, "error") == 2
    assert video_step(True, "done") == 3


def test_video_config_keeps_editable_controls():
    import inspect
    from src.ui import page_video
    src = inspect.getsource(page_video._render_config)
    assert src.count("st.slider(") >= 2       # stride + max frames stay editable
    assert "st.toggle(" in src                # annotated-export toggle stays
    assert "Process video" in src
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_dashboard.py -q -k "video_step or video_config"`
Expected: FAIL — `ImportError: cannot import name 'video_step'`.

- [ ] **Step 3: Implement `video_step` and wire the rail**

- `video_step(has_upload: bool, phase: str | None) -> int`: `1` if not `has_upload`; `3` if `phase == "done"`; else `2` (covers `None`, `processing`, `cancelled`, `error`).
- In `render()`, immediately after `ui.page_header(...)`: `uploaded_flag = st.session_state.get(f"video_uploader_{int(st.session_state.get('video_upload_nonce', 0))}") is not None`, `phase = (st.session_state.get("video_run") or {}).get("phase")`, `ui.step_rail(video_step(uploaded_flag, phase), ui.STEP_LABELS)`. Everything below stays unchanged — `_render_config` untouched (its sliders/toggle are pinned by the test above).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_dashboard.py -q -k "video_step or video_config"`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/ui/page_video.py tests/test_dashboard.py
git commit -m "feat(ui): video page - step rail, config controls untouched"
```

### Task 6: Plain-language page copy

**Files:**
- Modify: `src/ui/page_photo.py`, `src/ui/page_video.py`, `src/ui/page_camera.py` (header `description` strings only)

**Interfaces:** none (copy only; `page_header(title, description)` signature from Task 2).

- [ ] **Step 1: Replace the three header descriptions with the approved plain-language copy**

- Photo: `"Detect people in one photo and see where the crowd is densest. A single frame has no timeline — movement and speed live in Video Analysis."`
- Video: `"Analyse a clip over time: people are detected and tracked window by window, with density and movement charts. Processing runs in batches, so Cancel takes effect between them."`
- Camera: `"Watch a webcam in rolling windows of density and movement features. FPS and latency are measured from the run, not estimated."`

(Settings' description is already plain — leave it.)

- [ ] **Step 2: Run the full suite**

Run: `powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1`
Expected: all green (no test pins these strings).

- [ ] **Step 3: Commit**

```bash
git add src/ui/page_photo.py src/ui/page_video.py src/ui/page_camera.py
git commit -m "docs(ui): plain-language page descriptions"
```

### Task 7: README, E2E sweep, live verification

**Files:**
- Modify: `README.md`
- Create: `tests/test_page_smoke.py`
- Test: full suite + live app

- [ ] **Step 1: Update `README.md` Dashboard section to match the shipped UI**

- Overview bullet → task-first: "three task choices (photo/video/camera), one quiet live-status line (models, dataset, evaluation, runs), session history in a collapsed *This session* expander with CSV download and confirmed clear, and the quiet *About the numbers* list."
- Photo bullet → prefix "Three steps — upload, check, results:" before the existing behavior text; same one-line prefix for Video ("…batched, cancellable processing…" keeps its detail).
- Camera bullet → mention plain start/stop (no steps) only if it reads wrong otherwise; otherwise leave.
- Leave the rest (settings table, session-history section, structure lines) as-is — they are still accurate.

- [ ] **Step 2: Write the committed page-smoke test**

Create `tests/test_page_smoke.py`:

```python
"""Every page must render exception-free under AppTest (light UI regression net)."""
import pytest
from streamlit.testing.v1 import AppTest

PAGES = ["page_overview", "page_photo", "page_video", "page_camera", "page_settings"]


@pytest.mark.parametrize("name", PAGES)
def test_page_renders_exception_free(name):
    at = AppTest.from_string(f"from src.ui import {name}\n{name}.render()\n",
                             default_timeout=90)
    at.run()
    assert not at.exception, at.exception
    assert len(at.error) == 0, [e.value for e in at.error]
```

- [ ] **Step 3: Run the full suite**

Run: `powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1`
Expected: all green, including the 5 new smoke cases and `test_ui_flows`.

- [ ] **Step 4: Restart the app and verify it serves**

Stop any running instance on port 8501, then background-run:
`python -m streamlit run app.py --server.headless true --server.port 8501 --browser.gatherUsageStats false`
Verify: `Invoke-WebRequest http://localhost:8501 -UseBasicParsing -Proxy $null` → HTTP 200.

- [ ] **Step 5: Commit**

```bash
git add README.md tests/test_page_smoke.py
git commit -m "test(ui): page smoke sweep + README for light guided console"
```
