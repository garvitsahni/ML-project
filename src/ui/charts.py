"""Altair chart builders with consistent dark styling and unit-aware axes.

Every builder returns a fully configured `alt.Chart`: axes carry units in
their titles, tooltips show raw values, and colors come from `theme.TOKENS`
so charts match the shell. Charts must be fed real backend outputs only.
"""

from __future__ import annotations

from typing import Sequence

import altair as alt
import pandas as pd

from src.ui.theme import TOKENS

DEFAULT_HEIGHT = 280
_SERIES_COLORS = [TOKENS["accent"], TOKENS["info"], TOKENS["warn"],
                  TOKENS["danger"], TOKENS["success"], "#A78BFA"]


def _axis_title(label: str, unit: str) -> str:
    label = label.replace("_", " ").strip()
    if unit:
        return f"{label} ({unit})"
    return label


def _finalize(chart: alt.Chart, height: int, title: str = "") -> alt.Chart:
    props: dict = {"height": height}
    if title:
        props["title"] = title
    return (
        chart.properties(**props)
        .configure_title(
            color=TOKENS["text"],
            fontSize=13,
            fontWeight=600,
            anchor="start",
            offset=6,
        )
        .configure_view(stroke=None)
        .configure_axis(
            labelColor=TOKENS["text_muted"],
            titleColor=TOKENS["text"],
            gridColor=TOKENS["border"],
            domainColor=TOKENS["border"],
            tickColor=TOKENS["border"],
            labelFontSize=11,
            titleFontSize=12,
        )
        .configure_legend(
            labelColor=TOKENS["text_muted"],
            titleColor=TOKENS["text"],
            labelFontSize=11,
            titleFontSize=12,
        )
        .configure(
            background="transparent",
            padding={"left": 8, "right": 8, "top": 8, "bottom": 4},
        )
    )


def timeline_series(df: pd.DataFrame, x: str, y: str, title: str,
                    y_label: str, unit: str, *, color: str | None = None,
                    x_label: str | None = None, height: int = DEFAULT_HEIGHT,
                    y_domain: Sequence[float] | None = None) -> alt.Chart:
    """Line chart for a metric over time (windows/frames).

    `unit` is appended to the y-axis title so the chart never shows a bare
    number without its unit.
    """
    y_enc = alt.Y(f"{y}:Q", title=_axis_title(y_label, unit))
    if y_domain is not None:
        y_enc = y_enc.scale(domain=list(y_domain))

    encoding: dict = {
        "x": alt.X(f"{x}:Q", title=_axis_title(x_label or x, "")),
        "y": y_enc,
        "tooltip": [alt.Tooltip(f"{x}:Q"), alt.Tooltip(f"{y}:Q", format=",.3f")],
    }
    chart = alt.Chart(df)
    if color is not None:
        encoding["color"] = alt.Color(
            f"{color}:N",
            scale=alt.Scale(range=_SERIES_COLORS),
            legend=alt.Legend(title=color.replace("_", " ").title()),
        )
        encoding["detail"] = alt.Detail(f"{color}:N")
        chart = chart.mark_line()
    else:
        chart = chart.mark_line(color=TOKENS["accent"], point=True)
    return _finalize(chart.encode(**encoding), height, title)


def grouped_bars(df: pd.DataFrame, x: str, y: str, group: str, title: str,
                 y_label: str, unit: str, *, x_label: str | None = None,
                 height: int = DEFAULT_HEIGHT,
                 y_domain: Sequence[float] | None = None) -> alt.Chart:
    """Grouped bar chart (e.g. feature-set comparison per algorithm)."""
    y_enc = alt.Y(f"{y}:Q", title=_axis_title(y_label, unit))
    if y_domain is not None:
        y_enc = y_enc.scale(domain=list(y_domain))

    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X(f"{x}:N", title=_axis_title(x_label or x, ""),
                axis=alt.Axis(labelAngle=0)),
        y=y_enc,
        color=alt.Color(
            f"{group}:N",
            scale=alt.Scale(range=_SERIES_COLORS),
            legend=alt.Legend(title=group.replace("_", " ").title()),
        ),
        xOffset=alt.XOffset(f"{group}:N"),
        tooltip=[
            alt.Tooltip(f"{x}:N"),
            alt.Tooltip(f"{group}:N"),
            alt.Tooltip(f"{y}:Q", format=",.3f"),
        ],
    )
    return _finalize(chart, height, title)


def bar_h(df: pd.DataFrame, x: str, y: str, title: str, y_label: str,
          unit: str, *, height: int = DEFAULT_HEIGHT) -> alt.Chart:
    """Horizontal bars sorted by value (feature importance, rankings)."""
    chart = alt.Chart(df).mark_bar(color=TOKENS["accent"]).encode(
        y=alt.Y(f"{x}:N", title=None,
                sort=alt.Sort(field=y, order="descending")),
        x=alt.X(f"{y}:Q", title=_axis_title(y_label, unit)),
        tooltip=[
            alt.Tooltip(f"{x}:N"),
            alt.Tooltip(f"{y}:Q", format=",.4f"),
        ],
    )
    return _finalize(chart, height, title)
