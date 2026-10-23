"""Design-system tests: theme tokens, shared component helpers, chart builders.

These cover the pure helpers of the redesign's foundation layer
(`src/ui/theme.py`, `src/ui/components.py`, `src/ui/charts.py`).
"""

import pytest


def test_theme_tokens_contain_required_keys():
    from src.ui import theme

    for key in ("bg", "surface", "border", "text", "accent",
                "success", "warn", "danger", "radii"):
        assert key in theme.TOKENS, f"missing token: {key}"
    # accent must be a hex color (teal direction from the spec)
    assert theme.TOKENS["accent"].startswith("#")
    assert len(theme.TOKENS["accent"]) in (4, 7)
    # radii scale with at least small/large steps
    assert "sm" in theme.TOKENS["radii"]
    assert "lg" in theme.TOKENS["radii"]


def test_theme_css_has_root_block_and_uses_accent():
    from src.ui import theme

    assert ":root" in theme.CSS
    assert theme.TOKENS["accent"] in theme.CSS


def test_inject_theme_is_callable():
    from src.ui import theme

    assert callable(theme.inject_theme)


def test_pill_contains_label_and_kind_class():
    from src.ui import components

    html = components.pill("Ready", "ok")
    assert "Ready" in html
    assert "pill--ok" in html
    # status must not be conveyed by color alone: label text is present
    assert "<span" in html


def test_pill_rejects_unknown_kind():
    from src.ui import components

    with pytest.raises(ValueError):
        components.pill("x", "chartreuse")


def test_kv_rows_stringifies_and_handles_missing():
    from src.ui import components

    assert components.kv_rows([("a", 1)]) == [["a", "1"]]
    assert components.kv_rows([("n", None)]) == [["n", "—"]]


def test_format_count():
    from src.ui import components

    assert components.format_count(1234) == "1,234"
    assert components.format_count(None) == "—"


def test_format_pct_expects_fraction():
    from src.ui import components

    assert components.format_pct(0.6754) == "67.5%"
    assert components.format_pct(None) == "—"


def test_format_seconds_switches_units():
    from src.ui import components

    assert components.format_seconds(1.234) == "1.23 s"
    assert components.format_seconds(0.05) == "50 ms"
    assert components.format_seconds(None) == "—"


def test_page_header_is_callable():
    from src.ui import components

    assert callable(components.page_header)


def test_empty_state_and_error_state_are_callable():
    from src.ui import components

    assert callable(components.empty_state)
    assert callable(components.error_state)


def test_timeline_series_axis_title_contains_unit():
    import altair as alt
    import pandas as pd

    from src.ui import charts

    df = pd.DataFrame({"window": [0, 1, 2], "people": [3, 5, 4]})
    chart = charts.timeline_series(
        df, x="window", y="people", title="People over time",
        y_label="People", unit="people",
    )
    assert isinstance(chart, alt.Chart)
    y_title = str(chart.to_dict()["encoding"]["y"]["title"])
    assert "people" in y_title


def test_grouped_bars_returns_chart_with_legend():
    import altair as alt
    import pandas as pd

    from src.ui import charts

    df = pd.DataFrame({
        "feature_set": ["A", "A", "C", "C"],
        "algorithm": ["logreg", "rf", "logreg", "rf"],
        "auc": [0.46, 0.42, 0.67, 0.58],
    })
    chart = charts.grouped_bars(
        df, x="algorithm", y="auc", group="feature_set",
        title="Ablation", y_label="ROC AUC", unit="ROC AUC",
    )
    assert isinstance(chart, alt.Chart)
    spec = chart.to_dict()
    assert spec["encoding"].get("color", {}).get("legend") is not None
