"""Design tokens and global CSS for the CrowdSense AI dark shell.

`TOKENS` is the single source of truth for palette/spacing/radii; the
`.streamlit/config.toml` theme mirrors the core colors so native widgets
match, and `CSS` styles everything custom (headers, pills, cards, tables).

Rendering note: `inject_theme()` must be called on every Streamlit run —
Streamlit removes elements that are not re-rendered, so a "once per
session" flag would silently drop the stylesheet on the next interaction.
Repeated `<style>` injection at the same position is idempotent.
"""

from __future__ import annotations

import streamlit as st

TOKENS: dict = {
    # surfaces — deep navy-charcoal shell
    "bg": "#0B1220",
    "surface": "#111A2B",
    "surface_2": "#162136",
    "sidebar_bg": "#0D1422",
    "border": "#1F2A3E",
    # text
    "text": "#E6EDF3",
    "text_muted": "#8B98AC",
    # accent — restrained teal
    "accent": "#2DD4BF",
    "accent_dim": "#14B8A6",
    # semantic status (always paired with a text label, never color alone)
    "success": "#34D399",
    "warn": "#FBBF24",
    "danger": "#F87171",
    "info": "#60A5FA",
    # scales
    "radii": {"sm": "6px", "md": "10px", "lg": "14px"},
    "space": {"xs": "4px", "sm": "8px", "md": "16px", "lg": "24px", "xl": "32px"},
    "max_content": "1280px",
    "font_sans": '"Segoe UI", system-ui, -apple-system, sans-serif',
}


def _root_block() -> str:
    lines: list[str] = []
    for key, value in TOKENS.items():
        if isinstance(value, dict):
            for sub, subval in value.items():
                lines.append(f"  --{key.replace('_', '-')}-{sub}: {subval};")
        else:
            lines.append(f"  --{key.replace('_', '-')}: {value};")
    return ":root {\n" + "\n".join(lines) + "\n}"


_RULES = """
/* ---------- page header ---------- */
.page-header { margin: 0 0 1.1rem; }
.page-header h1 {
  font-size: 1.5rem;
  font-weight: 650;
  letter-spacing: -0.01em;
  line-height: 1.25;
  margin: 0.15rem 0 0;
  color: var(--text);
}
.page-eyebrow {
  margin: 0;
  font-size: 0.72rem;
  font-weight: 650;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  color: var(--accent);
}
.page-desc {
  margin: 0.4rem 0 0;
  max-width: 78ch;
  font-size: 0.92rem;
  line-height: 1.5;
  color: var(--text-muted);
}

/* ---------- cards & sections ---------- */
.card-title {
  margin: 0 0 0.35rem;
  font-size: 0.95rem;
  font-weight: 650;
  color: var(--text);
}
.card-hint { margin: -0.2rem 0 0.6rem; font-size: 0.8rem; color: var(--text-muted); }
.data-note { margin: 0.35rem 0 0; font-size: 0.78rem; color: var(--text-muted); }

/* ---------- status pills ---------- */
.pill {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.16rem 0.6rem;
  border: 1px solid;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 650;
  line-height: 1.4;
  white-space: nowrap;
}
.pill__dot {
  width: 0.5rem;
  height: 0.5rem;
  border-radius: 50%;
  background: currentColor;
  flex: none;
}
.pill--ok    { color: var(--success); border-color: color-mix(in srgb, var(--success) 40%, transparent); background: color-mix(in srgb, var(--success) 10%, transparent); }
.pill--warn  { color: var(--warn);    border-color: color-mix(in srgb, var(--warn) 40%, transparent);    background: color-mix(in srgb, var(--warn) 10%, transparent); }
.pill--error { color: var(--danger);  border-color: color-mix(in srgb, var(--danger) 40%, transparent);  background: color-mix(in srgb, var(--danger) 10%, transparent); }
.pill--info  { color: var(--info);    border-color: color-mix(in srgb, var(--info) 40%, transparent);    background: color-mix(in srgb, var(--info) 10%, transparent); }
.pill--idle  { color: var(--text-muted); border-color: var(--border); background: transparent; }

/* ---------- empty / error states ---------- */
.state-block {
  padding: 2rem 1.25rem;
  border: 1px dashed var(--border);
  border-radius: var(--radius-md);
  background: color-mix(in srgb, var(--surface) 55%, transparent);
  text-align: center;
}
.state-block__title { margin: 0.3rem 0 0.25rem; font-size: 1rem; font-weight: 650; color: var(--text); }
.state-block__msg { margin: 0 auto; max-width: 62ch; font-size: 0.88rem; line-height: 1.55; color: var(--text-muted); }
.state-block--error { border-style: solid; border-color: color-mix(in srgb, var(--danger) 45%, transparent); }
.state-block--error .state-block__title { color: var(--danger); }

/* ---------- metric tiles ---------- */
[data-testid="stMetric"] {
  padding: 0.75rem 0.9rem;
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  background: var(--surface);
}
[data-testid="stMetricLabel"] { font-size: 0.78rem; color: var(--text-muted); }
[data-testid="stMetricValue"] { font-size: 1.45rem; font-weight: 650; color: var(--text); }

/* ---------- tables & dataframes ---------- */
[data-testid="dataframe"] { border-radius: var(--radius-sm); }
[data-testid="stDataFrame"] { border-radius: var(--radius-sm); }

/* ---------- focus visibility ---------- */
:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 2px;
}

/* ---------- sidebar identity ---------- */
.crowdsense-brand { display: flex; align-items: center; gap: 0.55rem; margin-bottom: 0.1rem; }
.crowdsense-mark {
  display: inline-flex; align-items: center; justify-content: center;
  width: 1.7rem; height: 1.7rem; border-radius: 7px;
  background: linear-gradient(135deg, var(--accent) 0%, var(--info) 100%);
  color: #06202B; font-weight: 800; font-size: 0.95rem; flex: none;
}
.crowdsense-name { font-size: 1.02rem; font-weight: 700; letter-spacing: -0.01em; color: var(--text); }
.crowdsense-sub { font-size: 0.7rem; letter-spacing: 0.07em; text-transform: uppercase; color: var(--text-muted); }

/* ---------- misc restraint ---------- */
h1, h2, h3 { letter-spacing: -0.01em; }
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { transition: none !important; animation: none !important; }
}
"""

CSS: str = _root_block() + "\n" + _RULES


def inject_theme() -> None:
    """Render the global stylesheet. Call unconditionally on every run."""
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
