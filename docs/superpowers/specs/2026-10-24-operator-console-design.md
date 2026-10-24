# Operator Console — Light Guided UI Design

**Date:** 2026-10-24
**Status:** design complete (mockups + builder check answered); awaiting owner approval to start planning
**Supersedes visual parts of:** `docs/superpowers/plans/2026-10-23-frontend-redesign.md`

## Intent

The current dashboard reads as AI-generated: dark navy shell, single neon-teal
accent (`#2DD4BF`), small-caps eyebrows, pill badges, equal-weight rounded
cards everywhere, marketing-style captions. The owner wants a **user-friendly
operator console that looks hand-crafted** — calm, plain-language, obvious
what to do next.

The direction was chosen through side-by-side mockups in the visual
companion:

1. Three directions were shown: (A) action-first dark home, (B) light dense
   console, (C) guided numbered step flows. Owner: *"I like b and c"*.
2. The merge was shown: light console skin + guided steps. Owner: *"second
   one is better"* — the step-page treatment wins over the dense console one.
3. Two Overviews were shown: v1 (status as labeled rows) vs v2 (quiet status
   line, links instead of buttons). Owner: *"v2"*.

**Resulting direction: light skin (B) + guided step flows (C) + the calm v2
Overview.**

Reference mockups (copies committed with this doc):
`docs/superpowers/designs/2026-10-24/{ui-directions,merged-light-guided,overview-v2}.html`.

## Goals

- Light, quiet, plain-language UI across the whole shell and all five pages.
- One accent family; no neon pills, no eyebrows, no card soup.
- Overview follows the **v2** mockup: task choices lead, status collapses to
  one quiet line, history actions are small links.
- Photo and Video pages wear a **three-step guided flow** (Upload → Check →
  Results). Camera stays a simple start/stop page (live feeds aren't a
  three-step path — builder may adjust presentation).
- All existing behavior and data rules unchanged: every number read/computed
  from real files or inputs, no fabricated metrics, no tracebacks rendered,
  session history with CSV export + clear-with-confirmation on Overview,
  settings validation and diagnostics whitelist intact, video cancel at
  batch boundaries, camera device lifecycle untouched.
- Full test suite green via `powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1`;
  AppTest E2E sweep (all pages exception-free + clear-gating + settings flow)
  green; headless boot green; app served live at the end.

## Non-goals / anti-goals

- No new pages, features, or dependencies (Streamlit + stdlib only; custom
  CSS is fine — no component libraries, no web fonts, no JS frameworks).
- No dark mode or theme toggle.
- No behavior changes — this is presentation only.
- No emoji as primary icons (inline SVG icons as in the mockups).
- No fabricated demo/sample data anywhere.
- Not reintroducing anything research-related (Research section stays gone).

## Design decisions

### Theme tokens (my call, taken from the mockups)

| Token | Value |
|---|---|
| Page background | `#FFFFFF` |
| Panel / soft background | `#FAFBFC` (tiles `#FBFCFD`, nav `#F6F8FA`) |
| Border | `#E4E9EF` (1px) |
| Heading text | `#0F172A` |
| Body text | `#1B2430` |
| Muted text | `#5B6675` / `#6B7686` |
| Accent (buttons, links, step dots) | dark teal `#0F766E`, hover darker |
| Accent-soft (active nav bg) | `#E6F5F3` |
| Ready/success | green `#16A34A` (small dot + word, never a bordered pill) |
| Warning | amber `#B45309`; Error | red `#DC2626` (message text unchanged) |
| Radius | 6–10px; shadows minimal or none |
| Type | system font stack; page title ≈ 20px; **no eyebrow labels at all** |

### Shared components (`src/ui/components.py`)

- `page_header`: drop the eyebrow argument/usage entirely — title + one plain
  sentence.
- `status_pill` → quiet inline status: colored dot + word ("ready",
  "not found"). No borders, no neon, no background chip.
- `card` → flat bordered panel with a plain section label (`.label`-style
  heading) or replaced by plain section headings; `hint` becomes normal
  muted caption text.
- `empty_state`: plain title + one sentence; no primary button unless it
  navigates to an implemented page (current empty states have no CTAs).
- Primary actions: solid `#0F766E` button. Secondary actions (history CSV,
  clear, per-run downloads where secondary): text-link style (teal, bold).
- Tables/dataframes: light rows, light borders — restyled via CSS, values
  unchanged.
- `data_note` → small muted caption.

### Page specs

**Overview (exactly the v2 mockup)**
- Title: "What do you want to analyse?" + one-line sub.
- Three task tiles (inline SVG icon, name, one-line plain description,
  "Start →" link) → `st.switch_page` to Photo / Video / Camera.
- One quiet status strip: models / dataset / evaluation / runs-this-session —
  same live reads as today (`list_model_files`, dataset probe,
  `load_evaluation`, history length).
- Links row: **Download history CSV** · **Clear history** (keeps the
  checkbox-confirmation + disabled-button + inline-guard pattern) · current
  settings values as muted text.
- The session run list stays visible but quiet: a collapsed expander
  **"This session: N runs"** under the links row opens the full run rows
  (same data as today's table); with no runs, the expander and history
  links hide and the strip reads "no runs yet". *Builder-round decision —
  owner delegated ("whichever is the best").*
- "About the numbers" (the real `limitations()` content) stays at the bottom
  as a quiet muted section — honesty requirement, just no longer a shouting
  card.

**Photo Analysis — steps 1·2·3**
- Step rail: 1 Upload (done ✓ when file present) → 2 Check → 3 Results.
- Step 1: drop-zone styled uploader (JPG/PNG, existing validation and error
  messages unchanged).
- Step 2: read-only summary of applicable settings (confidence, overlays)
  + "Change in Settings" link; the Analyse button is the step's forward
  action.
- Step 3: existing results presentation (count, density grid, static
  features, downloads) in the light skin.
- Behavior of analysis (in-memory decode, explicit analyse) unchanged.

**Video Analysis — steps 1·2·3**
- Same rail framing the existing flow: 1 Upload → 2 Check (stride / max
  frames / resolution + annotated-video export toggle, **editable exactly as
  today** — seeded from Settings; the Settings link only changes the
  defaults, never removes or locks a control; start processing from here)
  → 3 Results. *Builder-round decision — owner delegated ("whichever is the
  best"); no feature is removed or made read-only.*
- Between 2 and 3 the existing progress presentation (batched processing,
  Cancel/Stop taking effect at batch boundaries) is preserved exactly —
  steps are visual framing, not a rewrite of the processing model.
- Results keep timeline charts, window predictions, annotated-frame strip,
  annotated-video export, CSV/JSON downloads.

**Live Camera — not steps**
- Plain header + device selector + primary Start capture / Stop; rolling
  window results, FPS/latency readouts; graceful real-error state when no
  device. Lifecycle (open only on start, always release on stop/navigation)
  unchanged.

**Settings**
- Same form, validated save, reset-to-defaults, whitelisted diagnostics —
  only re-skinned (plain section labels, light inputs, quiet captions).

**Shell / sidebar**
- `app.py` emits the shared CSS once; sidebar becomes light (`#F6F8FA`),
  same five pages in Workspace + System, same status line (newest-bundle
  caption) restyled quietly.

## Constraints

- TDD for new/moved production logic; presentation verified through AppTest
  (exception-free renders, clear-gating flow, settings save/reset) plus the
  headless boot smoke — the suite stays green throughout.
- Session-only history rule, no-tracebacks rule, real-data-only rule all
  stand; README IA description updated if wording it describes changes.
- Windows environment; app restarted on port 8501 at the end; commit on
  `main` when green.

## What's left to the builder

- Concrete CSS implementation (Streamlit style overrides, specificity
  battles, which elements need `st.markdown` with `unsafe_allow_html`),
  step-state management for the rails, exact sequencing of the refactor so
  the suite stays green, and any layout details the mockups imply but don't
  pin down.
