"""Shared UI components for the CrowdSense AI dashboard.

Pure helpers (`pill`, `kv_rows`, `format_*`) carry no `st` side effects and
are unit-tested; render helpers wrap native Streamlit widgets so every page
composes the same visual language. Status is always conveyed by a text
label, never by color alone.
"""

from __future__ import annotations

import html
from contextlib import contextmanager
from typing import Callable, Iterator, Sequence

import streamlit as st

_PILL_KINDS = ("ok", "warn", "error", "info", "idle")
_DASH = "—"


# --------------------------------------------------------------------------
# pure helpers
# --------------------------------------------------------------------------

def pill(label: str, kind: str) -> str:
    """Return the HTML for an inline status pill.

    The text label is always rendered inside the pill so the status remains
    readable without color perception.
    """
    if kind not in _PILL_KINDS:
        raise ValueError(f"unknown pill kind: {kind!r} (expected one of {list(_PILL_KINDS)})")
    safe = html.escape(str(label))
    return (
        f'<span class="pill pill--{kind}">'
        f'<span class="pill__dot" aria-hidden="true"></span>{safe}</span>'
    )


def kv_rows(pairs: Sequence[tuple[str, object]]) -> list[list[str]]:
    """(label, value) pairs -> rows for st.dataframe/st.table, None -> dash."""
    return [[str(k), _DASH if v is None else str(v)] for k, v in pairs]


def format_count(n: object) -> str:
    if n is None:
        return _DASH
    return f"{int(n):,}"


def format_pct(x: object, digits: int = 1) -> str:
    """Format a fraction (0–1) as a percentage."""
    if x is None:
        return _DASH
    return f"{float(x) * 100:.{digits}f}%"


def format_seconds(s: object) -> str:
    """Format seconds with an appropriate unit (ms below 1 s)."""
    if s is None:
        return _DASH
    value = float(s)
    if value < 1:
        return f"{value * 1000:.0f} ms"
    return f"{value:.2f} s"


# --------------------------------------------------------------------------
# render helpers
# --------------------------------------------------------------------------

def page_header(title: str, description: str | None = None,
                eyebrow: str | None = None) -> None:
    """Consistent page-level heading: eyebrow, H1, one-line description."""
    eye = f'<p class="page-eyebrow">{html.escape(eyebrow)}</p>' if eyebrow else ""
    desc = f'<p class="page-desc">{html.escape(description)}</p>' if description else ""
    st.markdown(
        f'<header class="page-header">{eye}<h1>{html.escape(title)}</h1>{desc}</header>',
        unsafe_allow_html=True,
    )


@contextmanager
def card(title: str | None = None, hint: str | None = None) -> Iterator[None]:
    """A bordered surface for one logical section of a page."""
    with st.container(border=True):
        if title is not None:
            st.markdown(f'<p class="card-title">{html.escape(title)}</p>',
                        unsafe_allow_html=True)
        if hint is not None:
            st.markdown(f'<p class="card-hint">{html.escape(hint)}</p>',
                        unsafe_allow_html=True)
        yield


def status_pill(label: str, kind: str) -> None:
    """Render a status pill in place."""
    st.markdown(pill(label, kind), unsafe_allow_html=True)


def metric_tile(label: str, value: object, delta: str | None = None,
                help: str | None = None, delta_color: str = "normal",
                format_fn: Callable[[object], str] | None = None) -> None:
    """A consistently-styled metric tile (styled globally via theme CSS)."""
    if format_fn is not None:
        value = format_fn(value)
    st.metric(label=str(label), value=value, delta=delta,
              delta_color=delta_color, help=help)


def empty_state(title: str, message: str | None = None,
                action_label: str | None = None,
                on_click: Callable[[], None] | None = None,
                key: str | None = None) -> bool:
    """Helpful empty state with an optional primary action.

    Returns True when the action button was pressed. Pass a unique `key`
    when a page may render several empty states with the same label.
    """
    msg = f'<p class="state-block__msg">{html.escape(message)}</p>' if message else ""
    st.markdown(
        f'<div class="state-block"><p class="state-block__title">'
        f'{html.escape(title)}</p>{msg}</div>',
        unsafe_allow_html=True,
    )
    if action_label:
        return st.button(action_label, type="primary", on_click=on_click, key=key)
    return False


def error_state(message: str, detail: str | None = None,
                retry_label: str | None = None,
                on_click: Callable[[], None] | None = None,
                key: str | None = None) -> bool:
    """Actionable error block with an optional retry button.

    Returns True when the retry button was pressed.
    """
    det = f'<p class="state-block__msg">{html.escape(detail)}</p>' if detail else ""
    st.markdown(
        f'<div class="state-block state-block--error">'
        f'<p class="state-block__title">{html.escape(message)}</p>{det}</div>',
        unsafe_allow_html=True,
    )
    if retry_label:
        return st.button(retry_label, on_click=on_click, key=key)
    return False


def kv_table(pairs: Sequence[tuple[str, object]]) -> None:
    """Render (label, value) pairs as a compact label/value table."""
    st.dataframe(kv_rows(pairs), hide_index=True, width="stretch",
                 column_config={"0": "Property", "1": "Value"})


def data_note(text: str) -> None:
    """Small provenance/units note under a section."""
    st.markdown(f'<p class="data-note">{html.escape(text)}</p>',
                unsafe_allow_html=True)


def render_limitations(items: Sequence[str]) -> None:
    """Shared renderer for the honest-limitations list."""
    for item in items:
        st.markdown(f"- {item}")


_KIND_LABELS = {"photo": "Photo", "video": "Video", "camera": "Camera",
                "experiment": "Experiment"}


def history_rows(entries: Sequence[dict]) -> list[list[str]]:
    """Session-history entries -> [when, input, key results] rows."""
    rows: list[list[str]] = []
    for entry in entries:
        ts = str(entry.get("ts", "")).replace("T", " ")[:16]
        kind = str(entry.get("kind", ""))
        label = _KIND_LABELS.get(kind, kind.replace("_", " ").title())
        details = " · ".join(
            f"{str(k).replace('_', ' ')} {v}"
            for k, v in entry.items() if k not in ("ts", "kind")
        )
        rows.append([ts, label, details or "—"])
    return rows


def download_csv(label: str, data: str | bytes, file_name: str,
                 help: str | None = None) -> None:
    """Consistent CSV export button."""
    st.download_button(label, data=data, file_name=file_name,
                       mime="text/csv", help=help)
