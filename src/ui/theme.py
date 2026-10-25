"""Design tokens and global CSS for the CrowdSense AI light console.

`TOKENS` is the single source of truth for palette/spacing/radii; the
`.streamlit/config.toml` theme mirrors the core colors so native widgets
match, and `CSS` styles everything custom (headers, status dots, steps,
tables).

Rendering note: `inject_theme()` must be called on every Streamlit run —
Streamlit removes elements that are not re-rendered, so a "once per
session" flag would silently drop the stylesheet on the next interaction.
Repeated `<style>` injection at the same position is idempotent.
"""

from __future__ import annotations

import streamlit as st

TOKENS: dict = {
    # surfaces — white page, soft panels
    "bg": "#FFFFFF",
    "surface": "#FAFBFC",
    "surface_2": "#FBFCFD",
    "sidebar_bg": "#F6F8FA",
    "border": "#E4E9EF",
    # text
    "heading": "#0F172A",
    "text": "#1B2430",
    "text_muted": "#5B6675",
    # accent — dark teal, legible on white
    "accent": "#0F766E",
    "accent_soft": "#E6F5F3",
    # semantic status (always paired with a text label, never color alone)
    "success": "#16A34A",
    "warn": "#B45309",
    "danger": "#DC2626",
    "info": "#60A5FA",
    # scales
    "radii": {"sm": "8px", "md": "10px", "lg": "14px"},
    "space": {"xs": "4px", "sm": "8px", "md": "16px", "lg": "24px", "xl": "32px"},
    "max_content": "1280px",
    "font_sans": '"Segoe UI", system-ui, -apple-system, sans-serif',
    # soft depth — one quiet shadow instead of hard outlines
    "shadow": ("0 1px 2px rgba(15, 23, 42, 0.04), "
               "0 8px 24px rgba(15, 23, 42, 0.05)"),
}


def _root_block() -> str:
    lines: list[str] = []
    for key, value in TOKENS.items():
        if isinstance(value, dict):
            for sub, subval in value.items():
                lines.append(f"  --{key.replace('_', '-')}-{sub}: {subval};")
                # `_RULES` references the radii scale as --radius-*; emit the
                # alias here so no rule silently loses its declaration.
                if key == "radii":
                    lines.append(f"  --radius-{sub}: {subval};")
        else:
            lines.append(f"  --{key.replace('_', '-')}: {value};")
    return ":root {\n" + "\n".join(lines) + "\n}"


_RULES = """
/* ---------- page header ---------- */
.page-header { margin: 0 0 1.4rem; }
.page-header h1 {
  font-size: 1.4rem;
  font-weight: 600;
  letter-spacing: -0.01em;
  line-height: 1.3;
  margin: 0;
  color: var(--heading);
}
.page-desc {
  margin: 0.4rem 0 0;
  max-width: 78ch;
  font-size: 0.92rem;
  line-height: 1.6;
  color: var(--text-muted);
}

/* ---------- cards & sections ---------- */
.card-title {
  margin: 0 0 0.35rem;
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--heading);
}
.card-hint { margin: -0.2rem 0 0.6rem; font-size: 0.8rem; color: var(--text-muted); }
.data-note { margin: 0.35rem 0 0; font-size: 0.78rem; color: var(--text-muted); }

/* ---------- status: quiet dot + word (never a bordered pill) ---------- */
.pill {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0;
  border: none;
  font-size: 0.82rem;
  font-weight: 500;
  line-height: 1.4;
  white-space: nowrap;
  color: var(--text-muted);
}
.pill__dot {
  width: 0.5rem;
  height: 0.5rem;
  border-radius: 50%;
  background: currentColor;
  flex: none;
}
.pill--ok    { color: var(--success); }
.pill--warn  { color: var(--warn); }
.pill--error { color: var(--danger); }
.pill--info  { color: var(--info); }
.pill--idle  { color: var(--text-muted); }

/* ---------- status strip (overview one-liner) ---------- */
.status-strip {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.35rem 1rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow);
  padding: 0.7rem 1.1rem;
}

/* ---------- step rail (guided flows) ---------- */
.steps {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.4rem 2rem;
  list-style: none;
  margin: 0 0 1.3rem;
  padding: 0;
}
.step {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  color: var(--text-muted);
  font-size: 0.9rem;
  transition: color .15s ease;
}
.step__num {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  border: 1px solid var(--border);
  background: var(--surface);
  font-size: 0.72rem;
  font-weight: 650;
  flex: none;
}
.step.is-done .step__num {
  background: var(--success);
  border-color: var(--success);
  color: #FFFFFF;
}
.step.is-done { color: var(--text); }
.step.is-now { color: var(--heading); font-weight: 650; }
.step.is-now .step__num {
  background: var(--accent);
  border-color: var(--accent);
  color: #FFFFFF;
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 25%, transparent);
}

/* ---------- empty / error states ---------- */
.state-block {
  padding: 2.5rem 1.5rem;
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow);
  background: var(--surface);
  text-align: center;
}
.state-block__title { margin: 0.3rem 0 0.25rem; font-size: 1rem; font-weight: 650; color: var(--heading); }
.state-block__msg { margin: 0 auto; max-width: 62ch; font-size: 0.88rem; line-height: 1.55; color: var(--text-muted); }
.state-block--error { border-style: solid; border-color: color-mix(in srgb, var(--danger) 40%, transparent); }
.state-block--error .state-block__title { color: var(--danger); }

/* ---------- metric tiles ---------- */
[data-testid="stMetric"] {
  padding: 0.9rem 1rem;
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow);
  background: var(--surface-2);
}
[data-testid="stMetricLabel"] { font-size: 0.78rem; color: var(--text-muted); }
[data-testid="stMetricValue"] { font-size: 1.45rem; font-weight: 650; color: var(--heading); }

/* ---------- tables & dataframes ---------- */
[data-testid="dataframe"] { border-radius: var(--radius-sm); }
[data-testid="stDataFrame"] { border-radius: var(--radius-sm); }

/* ---------- content width cap (readability on wide screens) ---------- */
[data-testid="stMainBlockContainer"] {
  max-width: var(--max-content);
  margin-inline: auto;
}

/* ---------- focus visibility ---------- */
:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 2px;
}

/* ---------- sidebar identity ---------- */
.crowdsense-brand { display: flex; align-items: center; gap: 0.55rem; margin-bottom: 0.1rem; }
.crowdsense-mark {
  display: inline-flex; align-items: center; justify-content: center;
  width: 1.7rem; height: 1.7rem; border-radius: var(--radius-md);
  background: var(--accent);
  color: #FFFFFF; font-weight: 800; font-size: 0.95rem; flex: none;
}
.crowdsense-name { font-size: 1.02rem; font-weight: 700; letter-spacing: -0.01em; color: var(--heading); }
.crowdsense-sub { font-size: 0.74rem; color: var(--text-muted); }

/* ---------- soft surfaces (smoothing pass) ---------- */
/* Keyed cards + tiles: quiet depth instead of a hard outline. The key lands
   as `st-key-card-*` / `st-key-tile-*` on the bordered block itself. */
[data-testid="stVerticalBlock"][class*="st-key-card"],
[data-testid="stVerticalBlock"][class*="st-key-tile"] {
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow);
}
[data-testid="stExpander"] {
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow);
}
/* Upload dropzone: a soft, welcoming target. */
[data-testid="stFileUploaderDropzone"] {
  border-radius: var(--radius-lg);
  background: var(--surface);
  transition: border-color .15s ease, background-color .15s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
  background: var(--surface-2);
}
/* Buttons: gentle motion, no layout jump. */
[data-testid^="stBaseButton"] {
  border-radius: var(--radius-md);
  transition: background-color .15s ease, border-color .15s ease,
              box-shadow .15s ease, transform .1s ease;
}
[data-testid^="stBaseButton"]:hover {
  box-shadow: var(--shadow);
}
[data-testid^="stBaseButton"]:active {
  transform: translateY(1px);
}
/* Sidebar nav: rounded pills with a soft active state. */
[data-testid="stSidebarNavLink"] {
  border-radius: var(--radius-md);
  transition: background-color .15s ease, color .15s ease;
}
[data-testid="stSidebarNavLink"][aria-current="page"] {
  background: var(--accent-soft);
  color: var(--accent);
}

/* ---------- misc restraint ---------- */
h1, h2, h3 { letter-spacing: -0.01em; color: var(--heading); }
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { transition: none !important; animation: none !important; }
}
"""

CSS: str = _root_block() + "\n" + _RULES


def inject_theme() -> None:
    """Render the global stylesheet. Call unconditionally on every run."""
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
