"""Dashboard visual skin: CSS + shared page furniture (no extra deps).

Applied once per Streamlit run from app.py. Colors match
``.streamlit/config.toml`` — edit both together.
"""

from __future__ import annotations

import streamlit as st

_CSS = """
<style>
  /* ---------- page header ---------- */
  .cs-page-header {
    font-size: 1.85rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #1C1917;
    border-bottom: 3px solid #0F766E;
    display: inline-block;
    padding-bottom: 0.3rem;
    margin: 0.1rem 0 0.9rem;
  }

  /* ---------- metric cards ---------- */
  [data-testid="stMetric"] {
    border: 1px solid rgba(28, 25, 23, 0.10);
    border-radius: 0.6rem;
    background: #ffffff;
    padding: 0.75rem 1rem;
    box-shadow: 0 1px 2px rgba(28, 25, 23, 0.05);
  }
  [data-testid="stMetricValue"] {
    font-variant-numeric: tabular-nums;
    font-weight: 700;
  }
  [data-testid="stMetricLabel"] {
    color: rgba(28, 25, 23, 0.62);
    font-size: 0.8rem;
  }

  /* ---------- sidebar ---------- */
  [data-testid="stSidebar"] {
    border-right: 1px solid rgba(28, 25, 23, 0.08);
  }
  [data-testid="stSidebar"] .block-container {
    padding-top: 1.1rem;
  }
  [data-testid="stSidebar"] [role="radiogroup"] label {
    font-size: 0.92rem;
  }

  /* ---------- tables / dataframes ---------- */
  [data-testid="stDataFrame"] {
    border: 1px solid rgba(28, 25, 23, 0.10);
    border-radius: 0.5rem;
    overflow: hidden;
  }

  /* ---------- footer ---------- */
  .cs-footer {
    margin-top: 2.8rem;
    padding-top: 0.8rem;
    border-top: 1px solid rgba(28, 25, 23, 0.10);
    color: rgba(28, 25, 23, 0.48);
    font-size: 0.78rem;
  }
</style>
"""


def inject_css() -> None:
    """Apply the dashboard skin; call once at the top of app.main()."""
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(title: str, icon: str = "", subtitle: str | None = None) -> None:
    """Consistent page heading: icon + title with an accent underline."""
    label = f"{icon} {title}" if icon else title
    st.markdown(f"<div class='cs-page-header'>{label}</div>", unsafe_allow_html=True)
    if subtitle:
        st.caption(subtitle)


def footer() -> None:
    """Honesty footer; call once at the end of app.main()."""
    st.markdown(
        "<div class='cs-footer'>CrowdSense AI — research dashboard · detector "
        "counts are estimates, never ground truth · speeds in px/frame · "
        "grouped splits, seed 42</div>",
        unsafe_allow_html=True,
    )
