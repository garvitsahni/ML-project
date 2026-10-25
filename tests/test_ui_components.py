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


def test_css_has_focus_and_width_cap():
    """Task 10 a11y pass: focus outline + capped content width."""
    from src.ui import theme

    assert ":focus-visible" in theme.CSS
    assert "outline: 2px solid var(--accent)" in theme.CSS
    assert "max-width: var(--max-content)" in theme.CSS


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


def test_history_rows_formats_entries():
    from src.ui import components

    entries = [
        {"ts": "2026-10-23T10:00:00", "kind": "photo", "count": 12},
        {"ts": "2026-10-23T11:00:00", "kind": "video",
         "windows": 40, "anomalies": 3},
    ]
    rows = components.history_rows(entries)
    assert rows[0][0] == "2026-10-23 10:00"
    assert rows[0][1] == "Photo"
    assert "12" in rows[0][2]
    assert rows[1][1] == "Video"
    assert "40" in rows[1][2] and "3" in rows[1][2]


def test_history_rows_empty_input():
    from src.ui import components

    assert components.history_rows([]) == []


def test_history_frame_flattens_entries():
    from src.ui import components

    entries = [
        {"ts": "2026-10-23T10:00:00", "kind": "photo", "persons": 5,
         "model": "yolov8n.pt", "seconds": 0.8},
        {"ts": "2026-10-23T10:01:00", "kind": "video", "video": "clip.mp4",
         "frames": 40, "windows": 3, "seconds": 12.3, "model": "xgb"},
        {"ts": "2026-10-23T10:02:00", "kind": "camera", "device": 0,
         "frames": 15, "seconds": 4.4, "model": "xgb"},
    ]
    frame = components.history_frame(entries)
    assert list(frame.columns) == ["time", "input", "kind", "model",
                                   "results", "duration"]
    assert len(frame) == 3
    photo, video, camera = frame.iloc[0], frame.iloc[1], frame.iloc[2]
    assert photo["time"] == "2026-10-23 10:00:00"
    assert photo["input"] == "uploaded photo"
    assert photo["kind"] == "Photo"
    assert photo["results"] == "5 persons"
    assert photo["duration"] == "0.8 s"
    assert video["input"] == "clip.mp4"
    assert video["results"] == "3 windows · 40 frames"
    assert camera["input"] == "camera 0"
    assert camera["results"] == "15 frames"
    # missing fields stay empty markers — never fabricated
    sparse = components.history_frame([{"ts": "2026-10-23T11:00:00",
                                        "kind": "photo"}])
    assert sparse.iloc[0]["results"] == "—"
    assert sparse.iloc[0]["duration"] == "—"
    assert sparse.iloc[0]["model"] == "—"


def test_export_csv_round_trips():
    import csv
    import io

    from src.ui import components

    entries = [
        {"ts": "2026-10-23T10:00:00", "kind": "photo", "persons": 5,
         "model": "yolov8n.pt", "seconds": 0.8},
        {"ts": "2026-10-23T10:01:00", "kind": "video", "video": "a,b.mp4",
         "frames": 40, "windows": 3, "seconds": 12.3, "model": "xgb"},
    ]
    text = components.export_csv(entries)
    rows = list(csv.DictReader(io.StringIO(text)))
    assert len(rows) == 2
    assert rows[0]["input"] == "uploaded photo"
    assert rows[0]["results"] == "5 persons"
    assert rows[1]["input"] == "a,b.mp4"   # comma survives quoting
    # raw summary keys preserved for machine use
    assert rows[0]["persons"] == "5"
    assert rows[1]["windows"] == "3"


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

    text = (Path(__file__).resolve().parents[1] / ".streamlit"
            / "config.toml").read_text(encoding="utf-8")
    assert 'base = "light"' in text
    assert 'primaryColor = "#0F766E"' in text
    assert 'backgroundColor = "#FFFFFF"' in text


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


def test_every_css_var_has_a_definition():
    """A var() with no --x: definition silently drops the whole declaration."""
    import re

    from src.ui import theme

    used = set(re.findall(r"var\(--([a-z0-9-]+)", theme.CSS))
    defined = set(re.findall(r"--([a-z0-9-]+):\s", theme.CSS))
    missing = used - defined
    assert not missing, f"undefined CSS variables: {sorted(missing)}"


def test_theme_css_drops_all_legacy_dark_hexes():
    from src.ui import theme

    legacy_dark = (
        "#0B1220", "#111A2B", "#162136", "#0D1422", "#1F2A3E",
        "#2DD4BF", "#14B8A6", "#E6EDF3", "#8B98AC", "#34D399",
        "#F87171", "#06202B", "#0E1626",
    )
    for hexval in legacy_dark:
        assert hexval not in theme.CSS, f"dark leftover in CSS: {hexval}"


def test_theme_geometry_is_soft():
    """Softer radii + a real shadow token (smoother, friendlier surfaces)."""
    from src.ui import theme

    assert theme.TOKENS["radii"] == {"sm": "8px", "md": "10px", "lg": "14px"}
    assert "shadow" in theme.TOKENS
    assert "--shadow:" in theme.CSS
    assert "var(--shadow)" in theme.CSS


def test_css_targets_soft_surfaces():
    """The key surfaces the smoothing pass must reach must be selectable."""
    from src.ui import theme

    for sel in ("st-key-card", "st-key-tile", "stSidebarNavLink",
                "stBaseButton", "stFileUploaderDropzone", "stExpander"):
        assert sel in theme.CSS, f"missing soft-surface selector: {sel}"


def test_config_rounds_softly():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / ".streamlit"
            / "config.toml").read_text(encoding="utf-8")
    assert 'baseRadius = "14"' in text
    assert 'buttonRadius = "10"' in text


def test_card_gives_each_container_a_key():
    """Keys are what emit the st-key-* CSS class on the bordered box."""
    import inspect

    from src.ui import components

    assert "key=" in inspect.getsource(components.card)


def test_card_renders_multiple_instances_without_key_collision():
    from streamlit.testing.v1 import AppTest

    src = ("import streamlit as st\n"
           "from src.ui import components\n"
           'with components.card("One"):\n    st.write("a")\n'
           'with components.card("Two"):\n    st.write("b")\n')
    at = AppTest.from_string(src, default_timeout=60).run()
    assert not at.exception
    assert len(at.error) == 0


def test_theme_fits_to_frame():
    """Nothing may spill past the frame; the page uses its full width."""
    from src.ui import theme

    assert theme.TOKENS["max_content"] == "1440px"
    assert "overflow-x" in theme.CSS
    assert "clamp(" in theme.CSS
    # media embeds stay inside their column
    assert "max-width: 100%" in theme.CSS


def test_theme_uses_new_font():
    """Plus Jakarta Sans, with a local fallback stack if the CDN is offline."""
    from src.ui import theme

    stack = theme.TOKENS["font_sans"]
    assert stack.startswith('"Plus Jakarta Sans"')
    assert "Segoe UI" in stack  # offline fallback
    assert "font-family: var(--font-sans)" in theme.CSS


def test_config_loads_the_font_source():
    """Streamlit 1.51 loads `name:source_url` webfonts natively from config.toml."""
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / ".streamlit"
            / "config.toml").read_text(encoding="utf-8")
    line = next(l for l in text.splitlines() if l.startswith("font ="))
    name, url = line.split("=", 1)[1].strip().strip('"').split(":", 1)
    assert name == "Plus Jakarta Sans"
    assert url.startswith("https://fonts.googleapis.com/css2")
    assert url.count("family=") == 1  # one font per source URL (Streamlit rule)
