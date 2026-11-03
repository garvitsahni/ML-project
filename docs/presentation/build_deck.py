"""Build the 10-12 minute class deck for CrowdSense AI.

Every number in the deck is copied from files under ``reports/results`` (via
README tables) or ``src/features/schema.py``; nothing is invented. Figures are
the project's own test-split PNGs.

Run from the repository root::

    python docs/presentation/build_deck.py

Output: docs/presentation/crowdsense-ai-class.pptx
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "crowdsense-ai-class.pptx"
FIGURES = ROOT / "reports" / "figures"

# ---------------------------------------------------------------- design tokens
INK = RGBColor(0x0F, 0x17, 0x2A)
MUTED = RGBColor(0x47, 0x55, 0x69)
FAINT = RGBColor(0x64, 0x74, 0x8B)
ACCENT = RGBColor(0x1D, 0x4E, 0xD8)
ACCENT_SOFT = RGBColor(0xEF, 0xF6, 0xFF)
ACCENT_MID = RGBColor(0xBF, 0xDB, 0xFE)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BORDER = RGBColor(0xCB, 0xD5, 0xE1)
SOFT = RGBColor(0xF8, 0xFA, 0xFC)
GREEN = RGBColor(0x04, 0x78, 0x57)
GREEN_SOFT = RGBColor(0xEC, 0xFD, 0xF5)
AMBER = RGBColor(0xB4, 0x53, 0x09)
AMBER_SOFT = RGBColor(0xFF, 0xFB, 0xEB)
RED = RGBColor(0xB9, 0x1C, 0x1C)
NAVY = RGBColor(0x0B, 0x12, 0x20)

FONT = "Segoe UI"
W, H = 13.333, 7.5
MARGIN = 0.6
CONTENT_W = W - 2 * MARGIN


# --------------------------------------------------------------------- helpers
def _rgb(color: RGBColor) -> RGBColor:
    return color


def add_rect(
    slide,
    left,
    top,
    width,
    height,
    fill=None,
    line=None,
    line_w=1.0,
    shape=MSO_SHAPE.RECTANGLE,
    radius=None,
):
    shp = slide.shapes.add_shape(shape, Inches(left), Inches(top), Inches(width), Inches(height))
    shp.shadow.inherit = False
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(line_w)
    if radius is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        shp.adjustments[0] = radius
    shp.text_frame.word_wrap = True
    return shp


def add_text(
    slide,
    left,
    top,
    width,
    height,
    text,
    *,
    size=16,
    bold=False,
    color=INK,
    align=PP_ALIGN.LEFT,
    anchor=MSO_ANCHOR.TOP,
    font=FONT,
    spacing=None,
    italic=False,
):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = align
        if spacing:
            p.line_spacing = spacing
        for run in p.runs:
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.italic = italic
            run.font.color.rgb = color
            run.font.name = font
    return box


def set_bullet(paragraph, char="•", indent_in=0.24, color=ACCENT):
    pPr = paragraph._p.get_or_add_pPr()
    pPr.set("marL", str(Emu(Inches(indent_in))))
    pPr.set("indent", str(-Emu(Inches(indent_in))))
    bu_clr = pPr.makeelement(qn("a:buClr"), {})
    srgb = bu_clr.makeelement(qn("a:srgbClr"), {"val": str(color)})
    bu_clr.append(srgb)
    pPr.append(bu_clr)
    bu_font = pPr.makeelement(qn("a:buFont"), {"typeface": "Arial"})
    pPr.append(bu_font)
    bu_char = pPr.makeelement(qn("a:buChar"), {"char": char})
    pPr.append(bu_char)


def add_bullets(
    slide,
    left,
    top,
    width,
    height,
    items,
    *,
    size=16,
    color=INK,
    gap=8,
    bullet_color=ACCENT,
    line_spacing=1.05,
):
    """items: str or (str, level). Level 1 is a sub-bullet."""
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    first = True
    for item in items:
        text, level = (item, 0) if isinstance(item, str) else item
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.text = text
        p.line_spacing = line_spacing
        p.space_after = Pt(gap if level == 0 else max(2, gap - 3))
        set_bullet(p, "–" if level else "•", 0.34 if level else 0.22, bullet_color if level == 0 else FAINT)
        for run in p.runs:
            run.font.size = Pt(size - (1 if level else 0))
            run.font.color.rgb = color if level == 0 else MUTED
            run.font.name = FONT
    return box


def add_header(slide, kicker, title, subtitle=None):
    add_text(slide, MARGIN, 0.34, CONTENT_W, 0.28, kicker.upper(), size=11, bold=True, color=ACCENT)
    add_text(slide, MARGIN, 0.62, CONTENT_W, 0.5, title, size=27, bold=True, color=INK)
    add_rect(slide, MARGIN, 1.2, 1.1, 0.055, fill=ACCENT)
    if subtitle:
        add_text(slide, MARGIN, 1.33, CONTENT_W, 0.34, subtitle, size=14, color=MUTED)


def add_footer(slide, index, total=12):
    add_rect(slide, MARGIN, 7.0, CONTENT_W, 0.012, fill=BORDER)
    add_text(
        slide, MARGIN, 7.08, 9.0, 0.24,
        "CrowdSense AI  ·  UCSD Anomaly Detection v1p2  ·  grouped splits, seed 42",
        size=9.5, color=FAINT,
    )
    add_text(
        slide, W - MARGIN - 1.2, 7.08, 1.2, 0.24,
        f"{index} / {total}", size=9.5, color=FAINT, align=PP_ALIGN.RIGHT,
    )


def add_stat(slide, left, top, width, height, value, label, *, accent=ACCENT, fill=SOFT):
    add_rect(slide, left, top, width, height, fill=fill, line=BORDER, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
    add_text(slide, left + 0.18, top + 0.14, width - 0.36, 0.5, value, size=28, bold=True, color=accent)
    add_text(slide, left + 0.18, top + height - 0.5, width - 0.36, 0.4, label, size=11, color=MUTED)


def add_card(slide, left, top, width, height, title, body, *, accent=ACCENT, fill=WHITE, title_size=17, body_size=14):
    add_rect(slide, left, top, width, height, fill=fill, line=BORDER, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
    add_rect(slide, left + 0.02, top + 0.16, 0.06, 0.42, fill=accent)
    add_text(slide, left + 0.24, top + 0.18, width - 0.48, 0.4, title, size=title_size, bold=True, color=INK)
    add_text(slide, left + 0.24, top + 0.68, width - 0.48, height - 0.84, body, size=body_size, color=MUTED, spacing=1.08)


def add_table(slide, left, top, width, rows, col_widths, *, header_size=12, body_size=13, row_h=0.42, highlight=None):
    n_rows = len(rows)
    n_cols = len(rows[0])
    shape = slide.shapes.add_table(n_rows, n_cols, Inches(left), Inches(top), Inches(width), Inches(row_h * n_rows))
    table = shape.table
    table.first_row = False
    table.horz_banding = False
    table.vert_banding = False
    for i, w in enumerate(col_widths):
        table.columns[i].width = Inches(w)
    for r in range(n_rows):
        table.rows[r].height = Inches(row_h if r else row_h + 0.04)
        for c in range(n_cols):
            cell = table.cell(r, c)
            cell.margin_left = Inches(0.09)
            cell.margin_right = Inches(0.09)
            cell.margin_top = Inches(0.04)
            cell.margin_bottom = Inches(0.04)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if r == 0:
                cell.fill.fore_color.rgb = RGBColor(0x1E, 0x29, 0x3B)
            elif highlight is not None and r == highlight:
                cell.fill.fore_color.rgb = ACCENT_SOFT
            else:
                cell.fill.fore_color.rgb = WHITE if r % 2 else SOFT
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = str(rows[r][c])
            p.alignment = PP_ALIGN.LEFT if c <= 1 else PP_ALIGN.CENTER
            for run in p.runs:
                run.font.name = FONT
                run.font.size = Pt(header_size if r == 0 else body_size)
                run.font.bold = r == 0 or (highlight is not None and r == highlight)
                run.font.color.rgb = WHITE if r == 0 else INK
    return table


def add_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


# ----------------------------------------------------------------------- slides
def slide_title(prs):
    s = blank(prs)
    add_rect(s, 0, 0, W, H, fill=NAVY)
    add_rect(s, 0, 0, 0.18, H, fill=ACCENT)
    add_text(s, 1.0, 1.55, 11.4, 0.4, "MACHINE LEARNING PROJECT  ·  CLASS PRESENTATION", size=13, bold=True, color=ACCENT_MID)
    add_text(s, 1.0, 2.0, 11.4, 1.3, "CrowdSense AI", size=54, bold=True, color=WHITE)
    add_text(
        s, 1.0, 3.15, 10.6, 1.0,
        "Do temporal crowd-movement features improve anomaly\nprediction over static crowd-density features alone?",
        size=21, color=RGBColor(0xE2, 0xE8, 0xF0), spacing=1.12,
    )
    add_rect(s, 1.0, 4.55, 2.2, 0.05, fill=ACCENT)
    add_text(
        s, 1.0, 4.8, 11.0, 0.9,
        "Detection · tracking · optical flow · window features · sequence-held-out evaluation",
        size=14, color=RGBColor(0x94, 0xA3, 0xB8),
    )
    add_text(s, 1.0, 6.35, 11.0, 0.35, "UCSD Anomaly Detection Dataset v1p2  ·  CPU-only  ·  seed 42", size=12, color=RGBColor(0x64, 0x74, 0x8B))
    add_notes(
        s,
        "Opening (about 20 seconds). This is CrowdSense AI. The question is simple: "
        "when we look at a crowd, does how people move tell us more about anomalies than "
        "how many people there are and how they are arranged? I'll show the data, the "
        "pipeline, the four pre-declared test runs, and the honest result — including the "
        "target we did not hit.",
    )


def slide_question(prs):
    s = blank(prs)
    add_header(s, "01 · Motivation", "Two ways to read a crowd")
    add_card(
        s, MARGIN, 1.75, 5.9, 2.55,
        "Static density",
        "How many people are there, and how are they arranged?\n\n"
        "Person count, density per megapixel, grid concentration, spread in x and y, "
        "grid entropy — six features computed from one frame.",
        accent=RGBColor(0x0E, 0xA5, 0xE9),
        body_size=14.5,
    )
    add_card(
        s, MARGIN + 6.23, 1.75, 5.9, 2.55,
        "Temporal movement",
        "How is the crowd moving?\n\n"
        "Speed, acceleration, direction changes, flow, abrupt stops, reversals, track "
        "persistence — seventeen features computed over a 30-frame window.",
        accent=ACCENT,
        body_size=14.5,
    )
    add_rect(s, MARGIN, 4.62, CONTENT_W, 1.1, fill=ACCENT_SOFT, line=ACCENT_MID, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
    add_text(
        s, MARGIN + 0.3, 4.76, CONTENT_W - 0.6, 0.85,
        "Research question: do movement features improve anomaly prediction over static density alone?",
        size=18, bold=True, color=ACCENT, spacing=1.05, anchor=MSO_ANCHOR.MIDDLE,
    )
    add_text(
        s, MARGIN, 6.05, CONTENT_W, 0.4,
        "Pre-declared success: held-out test AUC ≥ 0.75 with better F1.  The honest answer is on slide 7.",
        size=12.5, color=MUTED,
    )
    add_notes(
        s,
        "About 45 seconds. A single frame gives density: count and layout. A video adds "
        "motion: speed, turns, stops, flow. The research question asks whether the motion "
        "signal adds predictive value beyond density. We committed to a success bar of "
        "0.75 test AUC before running the final experiments, so we cannot move the goalposts "
        "afterwards.",
    )


def slide_data(prs):
    s = blank(prs)
    add_header(s, "02 · Data", "UCSD Anomaly Detection v1p2, split by sequence")
    add_stat(s, MARGIN, 1.7, 2.85, 1.35, "98", "sequences (videos)")
    add_stat(s, MARGIN + 3.05, 1.7, 2.85, 1.35, "18,560", "frames")
    add_stat(s, MARGIN + 6.1, 1.7, 2.85, 1.35, "5,693", "anomalous-frame labels")
    add_stat(s, MARGIN + 9.15, 1.7, 2.98, 1.35, "3,222", "30-frame windows", accent=GREEN, fill=GREEN_SOFT)

    add_text(s, MARGIN, 3.35, 7.4, 0.35, "Grouped split by sequence id — no video spans two splits", size=15, bold=True, color=INK)
    bar_left, bar_top, bar_w, bar_h = MARGIN, 3.82, 7.4, 0.62
    segs = [("68 train", 68 / 98, ACCENT), ("15 val", 15 / 98, RGBColor(0x38, 0xBD, 0xF8)), ("15 test", 15 / 98, RGBColor(0xF5, 0x9E, 0x0B))]
    x = bar_left
    for label, frac, color in segs:
        seg_w = bar_w * frac
        add_rect(s, x, bar_top, seg_w, bar_h, fill=color)
        add_text(s, x, bar_top + 0.16, seg_w, 0.3, label, size=12.5, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        x += seg_w
    add_text(s, bar_left, 4.52, bar_w, 0.3, "Test set = 495 windows, of which 179 are anomalous.", size=12, color=MUTED)

    add_bullets(
        s, 8.45, 3.32, 4.3, 2.6,
        [
            "Splits enforced in code: assert_group_disjoint",
            "Window label = 1 if any frame is anomalous (frame-level ground truth)",
            "No counting dataset — counts are detector estimates, never MAE/RMSE",
        ],
        size=13, gap=9,
    )

    add_rect(s, MARGIN, 5.2, 7.4, 1.5, fill=SOFT, line=BORDER, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
    add_text(s, MARGIN + 0.24, 5.33, 6.9, 0.3, "Why grouped splits?", size=14, bold=True, color=INK)
    add_text(
        s, MARGIN + 0.24, 5.68, 6.95, 0.9,
        "Frames from one video are near-duplicates. A random frame split would leak them into test and inflate the score; grouping by sequence keeps test videos unseen.",
        size=11.5, color=MUTED, spacing=1.05,
    )

    add_rect(s, 8.35, 5.2, 4.38, 1.5, fill=WHITE, line=BORDER, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
    add_text(s, 8.55, 5.33, 3.95, 0.3, "How a window gets its label", size=14, bold=True, color=INK)
    add_text(s, 8.55, 5.66, 3.95, 0.28, "30 frames, schematic", size=10.5, color=FAINT)
    sq, gap_sq = 0.17, 0.055
    for i in range(30):
        row, col = divmod(i, 15)
        left_sq = 8.55 + col * (sq + gap_sq)
        top_sq = 5.98 + row * (sq + gap_sq)
        fill = RED if i in (4, 5, 6) else BORDER
        add_rect(s, left_sq, top_sq, sq, sq, fill=fill)
    add_text(s, 11.92, 5.98, 0.7, 0.3, "= 1", size=12, bold=True, color=RED)
    add_text(s, 11.92, 6.22, 0.7, 0.3, "= 0", size=12, bold=True, color=FAINT)
    add_notes(
        s,
        "About 45 seconds. UCSD v1p2 has 98 sequences and 18,560 frames, with 5,693 "
        "anomalous frames parsed from the dataset's own ground-truth files. We build "
        "3,222 sliding windows of 30 frames with stride 5. The split is by sequence, "
        "68 train, 15 validation, 15 test — so the model never sees frames from a test "
        "video during training. The test split is 495 windows, 179 anomalous. We did not "
        "use any counting dataset, so we never report counting error.",
    )


def slide_pipeline(prs):
    s = blank(prs)
    add_header(s, "03 · Method", "From video to one anomaly score per window")
    steps = [
        ("Video", "MP4 / AVI / MOV\nor a photo", RGBColor(0x0E, 0xA5, 0xE9)),
        ("Detect", "YOLOv8n persons\nconf 0.05 · 640 px", ACCENT),
        ("Track", "ByteTrack IDs\nacross frames", RGBColor(0x7C, 0x3A, 0xED)),
        ("Flow", "Farneback optical\nflow", RGBColor(0xDB, 0x27, 0x77)),
        ("Window", "30 frames\nstride 5", RGBColor(0xF5, 0x9E, 0x0B)),
        ("Score", "23 features →\nmodel → P(anomaly)", GREEN),
    ]
    box_w, gap = 1.78, 0.24
    left = MARGIN
    top = 2.15
    for i, (name, desc, color) in enumerate(steps):
        add_rect(s, left, top, box_w, 1.55, fill=WHITE, line=BORDER, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
        add_rect(s, left, top, box_w, 0.09, fill=color)
        add_text(s, left, top + 0.26, box_w, 0.35, name, size=16, bold=True, color=INK, align=PP_ALIGN.CENTER)
        add_text(s, left, top + 0.68, box_w, 0.7, desc, size=11.5, color=MUTED, align=PP_ALIGN.CENTER, spacing=1.05)
        if i < len(steps) - 1:
            add_rect(s, left + box_w + 0.03, top + 0.66, gap - 0.06, 0.22, fill=BORDER, shape=MSO_SHAPE.RIGHT_ARROW)
        left += box_w + gap

    add_rect(s, MARGIN, 4.15, 7.3, 1.5, fill=SOFT, line=BORDER, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
    add_text(s, MARGIN + 0.25, 4.3, 6.8, 0.3, "Two paths through the same pipeline", size=14, bold=True, color=INK)
    add_bullets(
        s, MARGIN + 0.25, 4.68, 6.85, 0.9,
        [
            "Photo: one frame → detection + six static features (no anomaly score)",
            "Video: window → all 23 features → per-window anomaly score",
        ],
        size=12.5, gap=4,
    )
    add_rect(s, 8.2, 4.15, 4.53, 1.5, fill=AMBER_SOFT, line=RGBColor(0xFD, 0xE6, 0x8A), shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
    add_text(s, 8.42, 4.3, 4.1, 0.3, "Why a window?", size=14, bold=True, color=AMBER)
    add_text(
        s, 8.42, 4.68, 4.1, 0.85,
        "Motion needs time. One frame has no speed, direction, or flow — so no anomaly score is shown for photos.",
        size=12, color=MUTED, spacing=1.05,
    )
    add_text(s, MARGIN, 5.95, CONTENT_W, 0.6,
             "Detector calibration: a 40-frame grid chose conf 0.05 / 640 px by a pre-declared rule — most detections under a 2-hour rebuild budget (16.40 vs 10.25 boxes/frame).",
             size=11.5, color=FAINT, italic=True)
    add_notes(
        s,
        "About 60 seconds. The pipeline is: read frames, detect people with YOLOv8n, link "
        "them across frames with ByteTrack, compute Farneback optical flow, and aggregate "
        "everything into 30-frame windows with stride 5. Each window becomes a 23-feature "
        "row and the trained model returns P(anomaly). A photo stops after detection and "
        "the six static features, because one frame has no motion. The detector threshold "
        "came from a pre-declared calibration rule, not from tuning on test.",
    )


def slide_features(prs):
    s = blank(prs)
    add_header(s, "04 · Features", "6 static + 17 movement = 23 features")
    static = [
        "person_count — detector estimate",
        "density_per_mpx — persons / megapixel",
        "grid_concentration — fraction 0–1",
        "spread_x — normalized",
        "spread_y — normalized",
        "grid_entropy — normalized 0–1",
    ]
    movement = [
        "density_delta", "mean_speed_pxf", "speed_std_pxf",
        "direction_entropy", "directional_consistency", "flow_mean_magnitude",
        "flow_std_pxf", "track_count", "track_persistence",
        "nn_distance_mean", "nn_distance_std", "abrupt_stop_fraction",
        "direction_reversal_rate", "fast_mover_fraction", "speed_p90",
        "track_turnover_rate", "flow_direction_inconsistency",
    ]
    add_rect(s, MARGIN, 1.72, 4.35, 4.5, fill=WHITE, line=BORDER, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
    add_text(s, MARGIN + 0.25, 1.9, 3.9, 0.3, "Static — one frame", size=16, bold=True, color=RGBColor(0x0E, 0xA5, 0xE9))
    add_text(s, MARGIN + 0.25, 2.28, 3.9, 0.28, "6 features", size=11.5, color=FAINT)
    add_bullets(s, MARGIN + 0.25, 2.7, 3.9, 3.3, static, size=12, gap=7, bullet_color=RGBColor(0x0E, 0xA5, 0xE9))

    add_rect(s, MARGIN + 4.6, 1.72, 7.53, 4.5, fill=WHITE, line=BORDER, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
    add_text(s, MARGIN + 4.85, 1.9, 7.0, 0.3, "Movement — 30-frame window", size=16, bold=True, color=ACCENT)
    add_text(s, MARGIN + 4.85, 2.28, 7.0, 0.28, "17 features", size=11.5, color=FAINT)
    half = 9
    add_bullets(s, MARGIN + 4.85, 2.72, 3.4, 3.2, movement[:half], size=11.5, gap=7)
    add_bullets(s, MARGIN + 8.35, 2.72, 3.4, 3.2, movement[half:], size=11.5, gap=7)

    add_rect(s, MARGIN, 6.4, CONTENT_W, 0.42, fill=SOFT, line=BORDER, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.2)
    add_text(s, MARGIN + 0.2, 6.47, CONTENT_W - 0.4, 0.3,
             "All speeds are px/frame at processed resolution — never m/s (no scene-scale calibration).",
             size=11.5, color=MUTED)
    add_notes(
        s,
        "About 45 seconds. Six static features describe count and layout. Seventeen "
        "movement features describe speed, speed spread, direction entropy, flow, track "
        "persistence, neighbour distance, stops, reversals, fast movers, and turnover. "
        "Every speed is in pixels per frame, because we have no way to convert pixels to "
        "metres in these scenes. That unit rule is enforced throughout the code and the UI.",
    )


def slide_method(prs):
    s = blank(prs)
    add_header(s, "05 · Evaluation design", "Decisions on train/val only; test touched once per config")
    steps = [
        ("1", "Tune", "GroupKFold on train. Five algorithms: logreg, decision tree, random forest, SVM, XGBoost.", ACCENT),
        ("2", "Threshold", "Pick the decision threshold on validation F1, never on test.", RGBColor(0x0E, 0xA5, 0xE9)),
        ("3", "Ensemble", "Soft-vote RF + SVM + XGB; members chosen by validation AUC.", RGBColor(0x7C, 0x3A, 0xED)),
        ("4", "Test", "Four pre-declared configurations, each evaluated once on the held-out split.", GREEN),
    ]
    top = 1.75
    for i, (num, name, desc, color) in enumerate(steps):
        y = top + i * 1.13
        add_rect(s, MARGIN, y, 0.62, 0.62, fill=color, shape=MSO_SHAPE.OVAL)
        add_text(s, MARGIN, y + 0.12, 0.62, 0.35, num, size=17, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        add_text(s, MARGIN + 0.85, y + 0.02, 2.0, 0.35, name, size=17, bold=True, color=INK)
        add_text(s, MARGIN + 2.7, y + 0.05, 9.4, 0.55, desc, size=13, color=MUTED, spacing=1.05)

    add_rect(s, MARGIN, 6.05, CONTENT_W, 0.75, fill=ACCENT_SOFT, line=ACCENT_MID, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
    add_text(
        s, MARGIN + 0.25, 6.17, CONTENT_W - 0.5, 0.5,
        "Why the cap matters: every extra look at test inflates the score. The screen, tuning, and threshold all avoid test.",
        size=12.5, color=ACCENT,
    )
    add_notes(
        s,
        "About 60 seconds. The evaluation design is the part I would defend hardest. "
        "Hyperparameters come from GroupKFold cross-validation on train. The decision "
        "threshold comes from validation F1. Ensemble membership comes from validation "
        "AUC. Only then do we score test, and only for four configurations that were "
        "declared before the accuracy pass. The pre-existing A-to-D experiment suite also "
        "reports test metrics, but those numbers drive no selection decision.",
    )


def slide_results(prs):
    s = blank(prs)
    add_header(s, "06 · Results", "Four pre-declared runs on the held-out test split")
    rows = [
        ["#", "Configuration", "Feat.", "Thr.", "Test AUC", "Test F1"],
        ["1", "Baseline — original 15 features, logreg, untuned", "15", "0.50", "0.6677", "0.5068"],
        ["2", "+ schema-v2 movement features, logreg, untuned", "23", "0.50", "0.6475", "0.4892"],
        ["3", "+ grouped-CV tuning + val threshold (RF)", "23", "0.25", "0.6714", "0.5311"],
        ["4", "+ soft-vote ensemble RF + SVM + XGB", "23", "0.20", "0.6705", "0.5090"],
    ]
    add_table(
        s, MARGIN, 1.72, 12.1, rows,
        [0.5, 6.9, 0.95, 0.9, 1.45, 1.4],
        header_size=12.5, body_size=13.5, row_h=0.56, highlight=3,
    )
    add_text(s, MARGIN, 4.72, CONTENT_W, 0.35,
             "Source: reports/results/experiment_attribution.json  ·  best row highlighted  ·  threshold chosen on validation",
             size=10.5, color=FAINT, italic=True)

    add_stat(s, MARGIN, 5.25, 3.55, 1.2, "0.6714", "best test AUC (run 3)", accent=ACCENT)
    add_stat(s, MARGIN + 3.8, 5.25, 3.55, 1.2, "0.5311", "best test F1 (run 3)", accent=GREEN, fill=GREEN_SOFT)
    add_stat(s, MARGIN + 7.6, 5.25, 4.53, 1.2, "−0.004", "vs. previous AUC 0.675", accent=AMBER, fill=AMBER_SOFT)
    add_notes(
        s,
        "About 60 seconds. Here are the four declared runs. The baseline with the original "
        "15 features scores 0.6677 AUC. Adding the eight new movement features without "
        "tuning actually hurts: 0.6475. Grouped-CV tuning plus a validation-chosen "
        "threshold gives the best result, 0.6714 AUC and 0.5311 F1. The ensemble does not "
        "beat the tuned single model. Against the previous headline of 0.675 AUC, we are "
        "slightly down on AUC but up on F1, 0.517 to 0.5311.",
    )


def slide_outcome(prs):
    s = blank(prs)
    add_header(s, "07 · The honest result", "The 0.75 AUC goal was not met — the ceiling held near 0.67")
    add_bullets(
        s, MARGIN, 1.72, 6.3, 3.3,
        [
            "Static density alone is near chance: AUC 0.43–0.56 across five algorithms.",
            "Movement carries the signal, but absolute performance plateaus near 0.67 on sequence-held-out UCSD.",
            "The calibrated detector raised detections from 10.25 to 16.40 boxes/frame and did not lift the old baseline.",
            "The eight new features hurt untuned; they only pay off with grouped-CV tuning and a validation threshold.",
            "The ensemble did not beat the tuned single model.",
        ],
        size=13.5, gap=10,
    )
    add_rect(s, 7.2, 1.72, 5.53, 1.15, fill=GREEN_SOFT, line=RGBColor(0xA7, 0xF3, 0xD0), shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
    add_text(s, 7.45, 1.86, 5.0, 0.3, "What did improve", size=13, bold=True, color=GREEN)
    add_text(s, 7.45, 2.18, 5.0, 0.6, "Test F1: 0.517 → 0.5311 with the validation-tuned threshold.", size=12.5, color=MUTED)

    add_rect(s, 7.2, 3.02, 5.53, 1.6, fill=AMBER_SOFT, line=RGBColor(0xFD, 0xE6, 0x8A), shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
    add_text(s, 7.45, 3.16, 5.0, 0.3, "Why the ceiling is real", size=13, bold=True, color=AMBER)
    add_text(s, 7.45, 3.48, 5.0, 1.05,
             "Window labels inherit frame-level ground truth; normal crowds still contain unusual motion. With 15 test sequences, a 0.08 AUC gain is hard to establish.",
             size=11.5, color=MUTED, spacing=1.06)

    # Goal vs achieved, drawn to scale on 0-1.
    add_text(s, MARGIN, 4.72, 6.3, 0.3, "AUC: achieved vs pre-declared goal", size=12.5, bold=True, color=INK)
    track_left, track_w = MARGIN, 6.3
    add_rect(s, track_left, 5.12, track_w, 0.26, fill=RGBColor(0xE2, 0xE8, 0xF0))
    add_rect(s, track_left, 5.12, track_w * 0.6714, 0.26, fill=ACCENT)
    goal_x = track_left + track_w * 0.75
    add_rect(s, goal_x - 0.015, 4.98, 0.03, 0.54, fill=AMBER)
    add_text(s, track_left, 5.42, 1.2, 0.26, "0", size=10, color=FAINT)
    add_text(s, track_left + track_w - 0.4, 5.42, 0.4, 0.26, "1.0", size=10, color=FAINT, align=PP_ALIGN.RIGHT)
    add_text(s, track_left + track_w * 0.6714 - 1.1, 5.42, 2.2, 0.26, "achieved 0.6714", size=11, bold=True, color=ACCENT)
    add_text(s, goal_x - 0.55, 4.68, 1.4, 0.26, "goal 0.75", size=11, bold=True, color=AMBER, align=PP_ALIGN.CENTER)

    add_rect(s, MARGIN, 6.05, CONTENT_W, 0.7, fill=SOFT, line=BORDER, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
    add_text(s, MARGIN + 0.25, 6.16, CONTENT_W - 0.5, 0.45,
             "Reported as measured: the gain is in F1, not AUC. The earlier 0.675 baseline is not re-labelled, and no metric is dropped or rounded up.",
             size=12, color=INK)
    add_notes(
        s,
        "About 60 seconds. This is the slide I want to be judged on. We set 0.75 AUC and "
        "we got 0.6714. I am not going to present that as a win. Static density is close "
        "to chance; movement clearly carries the signal, but the absolute ceiling on "
        "sequence-held-out UCSD is around 0.67. Calibration increased detections without "
        "lifting AUC. The new features need tuning to help. What did improve is F1, from "
        "0.517 to 0.5311, because the threshold is chosen on validation rather than 0.5.",
    )


def slide_ablation(prs):
    s = blank(prs)
    add_header(s, "08 · Ablation", "Movement features beat static density for every algorithm")
    rows = [
        ["Algorithm", "A static (6)", "B movement (17)", "C both (23)", "C − A"],
        ["logreg", "0.505 / 0.159", "0.653 / 0.490", "0.648 / 0.489", "+0.142"],
        ["SVM", "0.427 / 0.162", "0.665 / 0.386", "0.682 / 0.429", "+0.255"],
        ["XGBoost", "0.499 / 0.256", "0.629 / 0.420", "0.614 / 0.351", "+0.115"],
        ["Random forest", "0.470 / 0.267", "0.635 / 0.385", "0.671 / 0.376", "+0.202"],
        ["Decision tree", "0.563 / 0.426", "0.552 / 0.409", "0.593 / 0.446", "+0.029"],
    ]
    add_table(
        s, MARGIN, 1.72, 7.25, rows,
        [1.75, 1.45, 1.55, 1.45, 1.05],
        header_size=11.5, body_size=12.5, row_h=0.5,
    )
    add_text(s, MARGIN, 4.85, 7.55, 0.3, "Cells are test AUC / test F1 at threshold 0.5.  C − A is the AUC gain from adding movement.",
             size=10.5, color=FAINT, italic=True)
    add_bullets(
        s, MARGIN, 5.3, 7.55, 1.3,
        [
            "C − A is positive for all five algorithms — the answer to the research question is yes.",
            "Static-only is near chance; movement-only already recovers most of the signal.",
        ],
        size=12.5, gap=6,
    )

    roc = FIGURES / "ablation_C_static_plus_movement_rf_roc.png"
    if roc.is_file():
        pic_w = 3.75
        add_picture = s.shapes.add_picture(str(roc), Inches(8.45), Inches(1.72), width=Inches(pic_w))
        add_text(s, 8.45, 1.72 + pic_w * 472 / 505 + 0.08, 4.3, 0.55,
                 "Random forest, config C (static + movement), test ROC — AUC 0.671.",
                 size=10.5, color=MUTED, spacing=1.04)
    add_notes(
        s,
        "About 60 seconds. The ablation isolates the two feature groups. Column A is static "
        "only: AUC 0.43 to 0.56, barely better than chance. Column B is movement only, and "
        "it recovers most of the signal. Column C combines them. The last column is C minus "
        "A: positive for all five algorithms, from plus 0.029 for a decision tree to plus "
        "0.255 for an SVM. That is the direct answer to the research question: movement "
        "features help. The ROC curve on the right is the random forest on config C.",
    )


def slide_limits(prs):
    s = blank(prs)
    add_header(s, "09 · Limitations", "What this work does not claim")
    items = [
        ("No density CNN", "CPU-only machine. Counting is a YOLOv8n detection baseline, not a density regressor."),
        ("No counting ground truth", "QNRF and ShanghaiTech were dropped from scope, so no MAE or RMSE is reported anywhere."),
        ("No congestion labels", "Experiment E (congestion classification) is hard-gated off as not supported."),
        ("Speeds are px/frame", "No scene-scale calibration, so no m/s conversion is possible."),
        ("Frame-level labels", "A window is anomalous if any frame in it is; the label is coarser than the score."),
        ("Calibration is a choice", "conf 0.05 / 640 px maximises detections under a 2-hour budget — recall over precision."),
    ]
    for i, (title, body) in enumerate(items):
        col = i % 2
        row = i // 2
        left = MARGIN + col * 6.22
        top = 1.72 + row * 1.55
        add_rect(s, left, top, 5.9, 1.32, fill=WHITE, line=BORDER, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
        add_text(s, left + 0.22, top + 0.16, 5.4, 0.3, title, size=14, bold=True, color=INK)
        add_text(s, left + 0.22, top + 0.5, 5.45, 0.72, body, size=11.5, color=MUTED, spacing=1.05)

    add_rect(s, MARGIN, 6.32, CONTENT_W, 0.5, fill=SOFT, line=BORDER, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.12)
    add_text(s, MARGIN + 0.22, 6.43, CONTENT_W - 0.44, 0.3,
             "Any claim about counting accuracy, m/s speeds, or congestion classes would be unsupported by this data.",
             size=11.5, color=MUTED)
    add_notes(
        s,
        "About 45 seconds. These are the limits I would put on any citation. There is no "
        "density CNN because the machine is CPU-only. There is no counting ground truth, "
        "so no counting error is reported. Congestion classification is switched off "
        "because credible labels do not exist. Speeds are pixels per frame, not metres per "
        "second. Window labels are coarser than the model output. And the detector "
        "threshold is a documented budget choice, not a truth.",
    )


def slide_product(prs):
    s = blank(prs)
    add_header(s, "10 · The product", "The same pipeline behind a three-tool web app")
    tools = [
        ("Photo Analysis", "Detect, annotate, 4×4 density grid, six static features.", RGBColor(0x0E, 0xA5, 0xE9)),
        ("Video Analysis", "Windows, per-window anomaly score, top-scoring windows.", ACCENT),
        ("Live Camera", "Browser camera → single-frame detection, no anomaly score.", RGBColor(0x7C, 0x3A, 0xED)),
    ]
    for i, (name, desc, color) in enumerate(tools):
        left = MARGIN + i * 4.14
        add_rect(s, left, 1.72, 3.85, 1.28, fill=WHITE, line=BORDER, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
        add_rect(s, left, 1.72, 0.08, 1.28, fill=color)
        add_text(s, left + 0.24, 1.86, 3.45, 0.3, name, size=14.5, bold=True, color=INK)
        add_text(s, left + 0.24, 2.2, 3.45, 0.7, desc, size=11.5, color=MUTED, spacing=1.04)

    arch = [
        ("Browser", "React + TypeScript\n(Vite SPA)"),
        ("API", "FastAPI\n/api/*"),
        ("Pipeline", "detect → track →\nflow → features"),
        ("Model", "bundle +\nP(anomaly)"),
    ]
    box_w, gap = 2.6, 0.5
    left = MARGIN
    for i, (name, desc) in enumerate(arch):
        add_rect(s, left, 3.45, box_w, 1.15, fill=SOFT, line=BORDER, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.07)
        add_text(s, left, 3.58, box_w, 0.3, name, size=14, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)
        add_text(s, left, 3.94, box_w, 0.55, desc, size=11, color=MUTED, align=PP_ALIGN.CENTER, spacing=1.03)
        if i < len(arch) - 1:
            add_rect(s, left + box_w + 0.08, 3.92, gap - 0.16, 0.2, fill=BORDER, shape=MSO_SHAPE.RIGHT_ARROW)
        left += box_w + gap

    add_rect(s, MARGIN, 5.0, CONTENT_W, 1.05, fill=ACCENT_SOFT, line=ACCENT_MID, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
    add_text(s, MARGIN + 0.25, 5.12, CONTENT_W - 0.5, 0.3, "Demo", size=13, bold=True, color=ACCENT)
    add_text(s, MARGIN + 0.25, 5.44, CONTENT_W - 0.5, 0.5,
             "python -m uvicorn backend.app:app  →  http://127.0.0.1:8000   ·   missing metrics render as —, counts are labelled estimates, speeds stay px/frame",
             size=11.5, color=MUTED, spacing=1.05)
    add_notes(
        s,
        "About 45 seconds. The pipeline is not just a notebook. The same code sits behind "
        "a React and FastAPI app with three tools: photo, video, and live camera. The UI "
        "keeps the honesty rules: missing values render as a dash, counts are labelled "
        "detector estimates, and speeds stay in pixels per frame. If you want a live demo, "
        "the server is already running on port 8000.",
    )


def slide_takeaways(prs):
    s = blank(prs)
    add_header(s, "11 · Takeaways", "Three things to remember")
    items = [
        ("Movement matters", "Temporal features beat static density for every algorithm tested; static-only is near chance."),
        ("The ceiling is real", "Sequence-held-out UCSD plateaus near 0.67 AUC. The defensible gain is F1, not AUC."),
        ("Honesty is a design choice", "Grouped splits, four pre-declared runs, validation-only selection, and no invented metrics."),
    ]
    for i, (title, body) in enumerate(items):
        top = 1.72 + i * 1.12
        add_rect(s, MARGIN, top, 0.5, 0.5, fill=ACCENT, shape=MSO_SHAPE.OVAL)
        add_text(s, MARGIN, top + 0.08, 0.5, 0.3, str(i + 1), size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        add_text(s, MARGIN + 0.75, top, 4.3, 0.35, title, size=16, bold=True, color=INK)
        add_text(s, MARGIN + 0.75, top + 0.38, 10.8, 0.55, body, size=12.5, color=MUTED, spacing=1.04)

    add_rect(s, MARGIN, 5.3, CONTENT_W, 0.9, fill=SOFT, line=BORDER, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
    add_text(s, MARGIN + 0.25, 5.5, CONTENT_W - 0.5, 0.5,
             "Next: more sequences for tighter intervals, a GPU density CNN, and richer anomaly labels.   Questions?",
             size=14, bold=True, color=INK)
    add_notes(
        s,
        "About 30 seconds plus questions. Three takeaways: movement features matter, the "
        "ceiling on this benchmark is real, and the honesty rules are part of the design, "
        "not an afterthought. Next steps would be more sequences, a GPU density model, and "
        "better labels. Thank you — happy to take questions.",
    )


def build():
    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)
    slide_title(prs)
    slide_question(prs)
    slide_data(prs)
    slide_pipeline(prs)
    slide_features(prs)
    slide_method(prs)
    slide_results(prs)
    slide_outcome(prs)
    slide_ablation(prs)
    slide_limits(prs)
    slide_product(prs)
    slide_takeaways(prs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f"wrote {OUT} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    build()
