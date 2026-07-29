"""Pure-Python SVG plotting + tabular output for the CTIM reference implementation.

No matplotlib, no numpy — standard library only (Python 3.9 compatible).

Public API (see API.md, section `ctim/plotting.py`):

    line_chart(path, series, xlabel, ylabel, title,
               xticks=None, logy=False, width=560, height=380,
               colors=None, ymin=None, ymax=None) -> None
    write_csv(path, header, rows) -> None
    ascii_table(header, rows) -> str

`line_chart` emits a standalone, self-contained SVG 1.1 document: it carries a
`viewBox`, an explicit `xmlns`, and references no external file, font, script or
stylesheet.  Log-scale y (`logy=True`) is required by the paper's Figs. 2b and 3b
(running time in seconds, log axis) and renders decade gridlines with `10^k`
tick labels.

This module implements no equation of the paper, so it carries no `# Eq (N)`
comments; it only draws the results.  It uses no randomness.
"""

from __future__ import annotations

import csv
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "line_chart",
    "write_csv",
    "ascii_table",
    "DEFAULT_COLORS",
    "PALETTE",
    "MARKERS",
]

# --------------------------------------------------------------------------
# Style constants
# --------------------------------------------------------------------------

# Distinct default colours, keyed by a normalised method label, chosen to match
# the paper's figures where sensible (SPEC.md section 7 / Figs 2-5).
DEFAULT_COLORS: Dict[str, str] = {
    "ctim": "#1f5fd0",      # blue
    "ctimcga": "#e8830c",   # orange
    "aircga": "#2e9e4f",    # green
    "cinema": "#8a8f98",    # grey
    "greedy": "#e0b013",    # yellow / gold
    # a few convenience aliases used by the experiment driver
    "ctimpaper": "#1f5fd0",
    "paper": "#9aa0a6",
    "ours": "#1f5fd0",
    "spread": "#1f5fd0",
    "time": "#e8830c",
    "seconds": "#e8830c",
}

# Fallback cycle for labels that are not one of the five known methods.
PALETTE: List[str] = [
    "#1f5fd0", "#e8830c", "#2e9e4f", "#8a8f98", "#e0b013",
    "#b5359c", "#159aa8", "#c0392b", "#6b4fbb", "#4d7c0f",
]

# Marker shapes are cycled per series so the chart survives greyscale printing.
MARKERS: List[str] = ["circle", "square", "diamond", "triangle", "cross", "tri-down"]

_FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, "
         "sans-serif")

_AXIS_COLOR = "#5f6368"
_GRID_COLOR = "#dfe1e5"
_MINOR_GRID_COLOR = "#eef0f2"
_TEXT_COLOR = "#202124"
_MUTED_COLOR = "#70757a"
_BG_COLOR = "#ffffff"

_FS_TITLE = 14.0
_FS_SUBTITLE = 10.5
_FS_AXIS = 11.5
_FS_TICK = 10.0
_FS_LEGEND = 10.5

# Rough advance width of one character as a fraction of the font size.  Good
# enough to size the legend box and the left margin without a font engine.
_CHAR_W = 0.60

# Per-character-class refinements: all-caps labels such as "CTIM" or "CINEMA"
# are noticeably wider than _CHAR_W suggests, which would clip the legend box.
_NARROW_CHARS = "ijlt.,;:'`|!()[]{}/\\ "
_WIDE_CHARS = "mwMW@%"


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _esc(text: Any) -> str:
    """Escape a value for use as SVG character data / attribute value."""
    s = str(text)
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return s.replace('"', "&quot;")


def _norm_label(label: Any) -> str:
    """Normalise a series label for colour lookup: 'AIR+CGA' -> 'aircga'."""
    return "".join(ch for ch in str(label).lower() if ch.isalnum())


def _text_width(text: str, font_size: float) -> float:
    """Estimate rendered text width without a font engine (deliberately
    slightly generous, so boxes sized with it never clip their contents)."""
    total = 0.0
    for ch in str(text):
        if ch in _NARROW_CHARS:
            total += 0.30
        elif ch in _WIDE_CHARS:
            total += 0.88
        elif ch.isupper() or ch.isdigit():
            total += 0.68
        else:
            total += _CHAR_W
    return total * font_size


def _ellipsize(text: str, font_size: float, max_width: float) -> str:
    """Shorten `text` with a trailing ellipsis so it fits within `max_width`."""
    text = str(text)
    if max_width <= 0 or _text_width(text, font_size) <= max_width:
        return text
    ell = "…"
    ell_w = _text_width(ell, font_size)
    if ell_w > max_width:
        return ""
    kept = ""
    for ch in text:
        if _text_width(kept + ch, font_size) + ell_w > max_width:
            break
        kept += ch
    return kept.rstrip() + ell


def _is_finite(v: Any) -> bool:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return not (math.isnan(f) or math.isinf(f))


def _fmt_num(v: float) -> str:
    """Format a tick value compactly and without spurious precision."""
    if v == 0:
        return "0"
    av = abs(v)
    if av >= 1e6 or av < 1e-4:
        # compact scientific, e.g. 2.5e+05 -> 2.5e5
        s = "%.2e" % v
        mant, exp = s.split("e")
        mant = mant.rstrip("0").rstrip(".")
        return "%se%d" % (mant, int(exp))
    if abs(v - round(v)) < 1e-9 * max(1.0, av):
        iv = int(round(v))
        return "{:,}".format(iv) if abs(iv) >= 10000 else str(iv)
    # choose enough decimals to be informative but not noisy
    for nd in (1, 2, 3, 4):
        s = "%.*f" % (nd, v)
        if abs(float(s) - v) < 1e-9 * max(1.0, av):
            break
    s = s.rstrip("0").rstrip(".")
    return s


def _num(v: float) -> str:
    """Format a coordinate for the SVG path/attribute stream."""
    return ("%.2f" % v).rstrip("0").rstrip(".") if v != int(v) else str(int(v))


def _nice_ticks(lo: float, hi: float, target: int = 6) -> List[float]:
    """Human-friendly linear tick positions covering [lo, hi]."""
    if not (_is_finite(lo) and _is_finite(hi)) or hi <= lo:
        return [lo]
    raw = (hi - lo) / float(max(1, target))
    if raw <= 0:
        return [lo]
    mag = 10.0 ** math.floor(math.log10(raw))
    step = 10.0 * mag
    for mult in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= mult * mag:
            step = mult * mag
            break
    ticks: List[float] = []
    start = math.ceil(lo / step - 1e-9) * step
    v = start
    # guard against pathological loops
    for _ in range(1000):
        if v > hi + step * 1e-9:
            break
        ticks.append(round(v, 12))
        v += step
    return ticks or [lo]


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _resolve_colors(labels: Sequence[str], colors: Any) -> List[str]:
    """Map series labels to colours.

    `colors` may be None, a list/tuple parallel to `series`, or a dict keyed by
    the (raw or normalised) series label.
    """
    out: List[str] = []
    for idx, label in enumerate(labels):
        chosen: Optional[str] = None
        if isinstance(colors, dict):
            if label in colors:
                chosen = colors[label]
            elif _norm_label(label) in colors:
                chosen = colors[_norm_label(label)]
        elif isinstance(colors, (list, tuple)) and idx < len(colors) and colors[idx]:
            chosen = colors[idx]
        if chosen is None:
            chosen = DEFAULT_COLORS.get(_norm_label(label))
        if chosen is None:
            chosen = PALETTE[idx % len(PALETTE)]
        out.append(str(chosen))
    return out


def _marker_svg(shape: str, cx: float, cy: float, color: str, r: float = 3.4) -> str:
    """One marker glyph, drawn with a white halo so overlaps stay readable."""
    x, y = _num(cx), _num(cy)
    common = 'fill="%s" stroke="%s" stroke-width="1"' % (color, _BG_COLOR)
    if shape == "square":
        s = r * 1.8
        return '<rect x="%s" y="%s" width="%s" height="%s" %s/>' % (
            _num(cx - s / 2), _num(cy - s / 2), _num(s), _num(s), common)
    if shape == "diamond":
        d = r * 1.35
        pts = "%s,%s %s,%s %s,%s %s,%s" % (
            _num(cx), _num(cy - d), _num(cx + d), _num(cy),
            _num(cx), _num(cy + d), _num(cx - d), _num(cy))
        return '<polygon points="%s" %s/>' % (pts, common)
    if shape == "triangle":
        d = r * 1.35
        pts = "%s,%s %s,%s %s,%s" % (
            _num(cx), _num(cy - d), _num(cx + d), _num(cy + d * 0.85),
            _num(cx - d), _num(cy + d * 0.85))
        return '<polygon points="%s" %s/>' % (pts, common)
    if shape == "tri-down":
        d = r * 1.35
        pts = "%s,%s %s,%s %s,%s" % (
            _num(cx), _num(cy + d), _num(cx + d), _num(cy - d * 0.85),
            _num(cx - d), _num(cy - d * 0.85))
        return '<polygon points="%s" %s/>' % (pts, common)
    if shape == "cross":
        d = r * 1.15
        return ('<path d="M%s %s L%s %s M%s %s L%s %s" stroke="%s" '
                'stroke-width="2.1" stroke-linecap="round" fill="none"/>' % (
                    _num(cx - d), _num(cy - d), _num(cx + d), _num(cy + d),
                    _num(cx - d), _num(cy + d), _num(cx + d), _num(cy - d), color))
    return '<circle cx="%s" cy="%s" r="%s" %s/>' % (x, y, _num(r), common)


# --------------------------------------------------------------------------
# line_chart
# --------------------------------------------------------------------------

def line_chart(path: str,
               series: Sequence[Tuple[Any, Sequence[Tuple[float, float]]]],
               xlabel: str,
               ylabel: str,
               title: str,
               xticks: Optional[Sequence[Any]] = None,
               logy: bool = False,
               width: int = 560,
               height: int = 380,
               colors: Any = None,
               ymin: Optional[float] = None,
               ymax: Optional[float] = None) -> None:
    """Write a standalone SVG line chart to `path`.

    Parameters
    ----------
    path : str
        Output ``.svg`` file. Parent directories are created if missing.
    series : list of (label, points)
        ``points`` is a sequence of ``(x, y)`` pairs.  Points are sorted by x
        before drawing.  Non-finite points are dropped.
    xlabel, ylabel, title : str
        Axis captions and chart title (any may be empty).
    xticks : optional
        Either a list of x positions, or a list of ``(x, label)`` pairs.  When
        omitted, ticks are chosen from the data (the x values themselves if
        there are at most 12 distinct ones, otherwise a "nice" linear scale).
    logy : bool
        Log10 y axis with decade gridlines and ``10^k`` tick labels
        (SPEC.md Figs 2b / 3b).  Non-positive y values cannot be shown on a log
        axis: they are dropped and the count is reported in the subtitle.
    width, height : int
        Canvas size in user units; the SVG scales to any viewport via viewBox.
    colors : optional
        ``None`` (use `DEFAULT_COLORS` / `PALETTE`), a list parallel to
        `series`, or a dict mapping label -> colour.
    ymin, ymax : optional
        Explicit y limits.  On a log axis a non-positive `ymin` is ignored.

    Degenerate cases handled: no series at all, empty series, a single data
    point, all-equal y values, all-equal x values, and (with `logy`) series
    that lose some or all of their points to the positivity filter.
    """
    width = int(width)
    height = int(height)
    notes: List[str] = []

    # ---- 1. clean the data ------------------------------------------------
    labels: List[str] = []
    cleaned: List[List[Tuple[float, float]]] = []
    n_dropped_nonpos = 0
    n_dropped_bad = 0
    for entry in (series or []):
        label, pts = entry[0], entry[1]
        good: List[Tuple[float, float]] = []
        for pt in (pts or []):
            x, y = pt[0], pt[1]
            if not (_is_finite(x) and _is_finite(y)):
                n_dropped_bad += 1
                continue
            x, y = float(x), float(y)
            if logy and y <= 0.0:
                n_dropped_nonpos += 1
                continue
            good.append((x, y))
        good.sort(key=lambda p: p[0])
        labels.append(str(label))
        cleaned.append(good)

    if n_dropped_nonpos:
        notes.append("%d non-positive value%s omitted (log scale)"
                     % (n_dropped_nonpos, "" if n_dropped_nonpos == 1 else "s"))
    if n_dropped_bad:
        notes.append("%d non-finite point%s omitted"
                     % (n_dropped_bad, "" if n_dropped_bad == 1 else "s"))

    all_pts = [p for pts in cleaned for p in pts]
    has_data = bool(all_pts)
    if not has_data:
        notes.append("no data to plot")

    series_colors = _resolve_colors(labels, colors)

    # ---- 2. data ranges ---------------------------------------------------
    if has_data:
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        dx_lo, dx_hi = min(xs), max(xs)
        dy_lo, dy_hi = min(ys), max(ys)
    else:
        dx_lo, dx_hi = 0.0, 1.0
        dy_lo, dy_hi = (1.0, 10.0) if logy else (0.0, 1.0)

    # x range: a single distinct x (or one point) gets a symmetric pad.
    if dx_hi <= dx_lo:
        pad = abs(dx_lo) * 0.5 if dx_lo != 0 else 1.0
        x_lo, x_hi = dx_lo - pad, dx_hi + pad
    else:
        pad = (dx_hi - dx_lo) * 0.04
        x_lo, x_hi = dx_lo - pad, dx_hi + pad

    # y range
    if logy:
        if ymin is not None and float(ymin) > 0:
            y_lo = float(ymin)
        else:
            if ymin is not None:
                notes.append("ymin<=0 ignored on log scale")
            y_lo = dy_lo
        y_hi = float(ymax) if (ymax is not None and float(ymax) > 0) else dy_hi
        if y_hi <= y_lo:
            y_hi = y_lo * 10.0
        lmin = math.floor(math.log10(y_lo) - 1e-12)
        lmax = math.ceil(math.log10(y_hi) + 1e-12)
        if lmax <= lmin:
            lmax = lmin + 1
        v_lo, v_hi = 10.0 ** lmin, 10.0 ** lmax
    else:
        y_lo = float(ymin) if ymin is not None else dy_lo
        y_hi = float(ymax) if ymax is not None else dy_hi
        if y_hi <= y_lo:
            # all-equal y values (or a single point): open up a readable band
            span = abs(y_lo) * 0.5 if y_lo != 0 else 1.0
            y_lo, y_hi = y_lo - span, y_hi + span
        if ymin is None and ymax is None:
            head = (y_hi - y_lo) * 0.08
            y_hi += head
            # anchor at zero when the data is positive and close to the origin
            if 0.0 < y_lo <= 0.35 * y_hi:
                y_lo = 0.0
            elif y_lo < 0 < y_hi:
                pass
            else:
                y_lo -= (y_hi - y_lo) * 0.05
        v_lo, v_hi = y_lo, y_hi
        if v_hi <= v_lo:
            v_hi = v_lo + 1.0

    # ---- 3. ticks ---------------------------------------------------------
    # y ticks
    y_ticks: List[Tuple[float, str, bool]] = []   # (value, label, is_major)
    if logy:
        decades = list(range(int(lmin), int(lmax) + 1))
        label_every = 1
        while len(decades) // label_every > 9:
            label_every += 1
        for k in decades:
            y_ticks.append((10.0 ** k, "10^%d" % k,
                            (k - decades[0]) % label_every == 0))
    else:
        for tv in _nice_ticks(v_lo, v_hi, target=6):
            if v_lo - 1e-12 <= tv <= v_hi + 1e-12:
                y_ticks.append((tv, _fmt_num(tv), True))
        if not y_ticks:
            y_ticks = [(v_lo, _fmt_num(v_lo), True), (v_hi, _fmt_num(v_hi), True)]

    # x ticks
    x_ticks: List[Tuple[float, str]] = []
    if xticks is not None:
        for t in xticks:
            if isinstance(t, (tuple, list)) and len(t) >= 2:
                if _is_finite(t[0]):
                    x_ticks.append((float(t[0]), str(t[1])))
            elif _is_finite(t):
                x_ticks.append((float(t), _fmt_num(float(t))))
    else:
        distinct = sorted({p[0] for p in all_pts})
        if 0 < len(distinct) <= 12:
            x_ticks = [(v, _fmt_num(v)) for v in distinct]
        else:
            x_ticks = [(v, _fmt_num(v)) for v in _nice_ticks(x_lo, x_hi, target=6)]
        if not x_ticks:
            x_ticks = [(x_lo, _fmt_num(x_lo)), (x_hi, _fmt_num(x_hi))]

    # ---- 4. layout --------------------------------------------------------
    subtitle = "; ".join(notes)
    m_top = 12.0
    if title:
        m_top += _FS_TITLE + 6.0
    if subtitle:
        m_top += _FS_SUBTITLE + 4.0

    y_tick_w = 0.0
    for _v, lab, major in y_ticks:
        if not major:
            continue
        w = _text_width(lab.split("^")[0], _FS_TICK)
        if "^" in lab:
            w += _text_width(lab.split("^")[1], _FS_TICK * 0.78)
        y_tick_w = max(y_tick_w, w)
    m_left = 12.0 + y_tick_w + 8.0 + (_FS_AXIS + 4.0 if ylabel else 0.0)
    m_right = 16.0
    m_bottom = 10.0 + _FS_TICK + 8.0 + (_FS_AXIS + 4.0 if xlabel else 0.0)

    px0, px1 = m_left, float(width) - m_right
    py0, py1 = m_top, float(height) - m_bottom

    # A pathologically small canvas cannot hold the chrome (labels, title).
    # Sacrifice the chrome rather than letting the plot spill off the canvas.
    if px1 - px0 < 40.0:
        deficit = 40.0 - (px1 - px0)
        px0 = max(1.0, px0 - deficit * 0.6)
        px1 = min(float(width) - 1.0, px1 + deficit * 0.4)
        if px1 - px0 < 4.0:
            px0, px1 = 1.0, max(5.0, float(width) - 1.0)
    if py1 - py0 < 40.0:
        deficit = 40.0 - (py1 - py0)
        py0 = max(1.0, py0 - deficit * 0.4)
        py1 = min(float(height) - 1.0, py1 + deficit * 0.6)
        if py1 - py0 < 4.0:
            py0, py1 = 1.0, max(5.0, float(height) - 1.0)

    def sx(x: float) -> float:
        if x_hi == x_lo:
            return (px0 + px1) / 2.0
        return px0 + (x - x_lo) / (x_hi - x_lo) * (px1 - px0)

    if logy:
        log_lo, log_hi = math.log10(v_lo), math.log10(v_hi)

        def sy(y: float) -> float:
            if y <= 0:
                return py1
            t = (math.log10(y) - log_lo) / (log_hi - log_lo)
            return py1 - t * (py1 - py0)
    else:
        def sy(y: float) -> float:
            t = (y - v_lo) / (v_hi - v_lo)
            return py1 - t * (py1 - py0)

    # ---- 5. emit ----------------------------------------------------------
    o: List[str] = []
    o.append('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
    o.append('<svg xmlns="http://www.w3.org/2000/svg" '
             'xmlns:xlink="http://www.w3.org/1999/xlink" '
             'width="%d" height="%d" viewBox="0 0 %d %d" '
             'font-family="%s" role="img">' % (width, height, width, height, _FONT))
    o.append('<title>%s</title>' % _esc(title or "chart"))
    o.append('<defs><clipPath id="plotclip"><rect x="%s" y="%s" width="%s" '
             'height="%s"/></clipPath></defs>'
             % (_num(px0), _num(py0), _num(px1 - px0), _num(py1 - py0)))
    o.append('<rect x="0" y="0" width="%d" height="%d" fill="%s"/>'
             % (width, height, _BG_COLOR))

    # title / subtitle
    ty = 12.0 + _FS_TITLE
    if title:
        o.append('<text x="%s" y="%s" font-size="%s" font-weight="600" '
                 'fill="%s">%s</text>'
                 % (_num(px0), _num(ty - 4), _num(_FS_TITLE), _TEXT_COLOR,
                    _esc(title)))
    if subtitle:
        sy_txt = (ty + _FS_SUBTITLE + 1.0) if title else (12.0 + _FS_SUBTITLE)
        o.append('<text x="%s" y="%s" font-size="%s" fill="%s">%s</text>'
                 % (_num(px0), _num(sy_txt - 4), _num(_FS_SUBTITLE), _MUTED_COLOR,
                    _esc(subtitle)))

    # plot background
    o.append('<rect x="%s" y="%s" width="%s" height="%s" fill="%s" '
             'stroke="%s" stroke-width="1"/>'
             % (_num(px0), _num(py0), _num(px1 - px0), _num(py1 - py0),
                _BG_COLOR, _GRID_COLOR))

    # minor gridlines: intra-decade 2..9 lines when the log range is compact
    if logy and (lmax - lmin) <= 6:
        minor: List[str] = []
        for k in range(int(lmin), int(lmax)):
            for mmul in range(2, 10):
                yv = mmul * (10.0 ** k)
                if v_lo <= yv <= v_hi:
                    yy = _num(sy(yv))
                    minor.append('M%s %s H%s' % (_num(px0), yy, _num(px1)))
        if minor:
            o.append('<path d="%s" stroke="%s" stroke-width="1" fill="none"/>'
                     % (" ".join(minor), _MINOR_GRID_COLOR))

    # major y gridlines + labels
    for value, lab, major in y_ticks:
        yy = sy(value)
        if not (py0 - 0.5 <= yy <= py1 + 0.5):
            continue
        o.append('<path d="M%s %s H%s" stroke="%s" stroke-width="1" fill="none"/>'
                 % (_num(px0), _num(yy), _num(px1), _GRID_COLOR))
        if not major:
            continue
        o.append('<path d="M%s %s H%s" stroke="%s" stroke-width="1"/>'
                 % (_num(px0 - 4), _num(yy), _num(px0), _AXIS_COLOR))
        if "^" in lab:
            base, expo = lab.split("^", 1)
            o.append('<text x="%s" y="%s" font-size="%s" fill="%s" '
                     'text-anchor="end">%s<tspan font-size="%s" dy="%s">%s</tspan>'
                     '</text>'
                     % (_num(px0 - 7), _num(yy + _FS_TICK * 0.35), _num(_FS_TICK),
                        _MUTED_COLOR, _esc(base), _num(_FS_TICK * 0.78),
                        _num(-_FS_TICK * 0.42), _esc(expo)))
        else:
            o.append('<text x="%s" y="%s" font-size="%s" fill="%s" '
                     'text-anchor="end">%s</text>'
                     % (_num(px0 - 7), _num(yy + _FS_TICK * 0.35), _num(_FS_TICK),
                        _MUTED_COLOR, _esc(lab)))

    # x gridlines + labels
    for value, lab in x_ticks:
        xx = sx(value)
        if not (px0 - 0.5 <= xx <= px1 + 0.5):
            continue
        o.append('<path d="M%s %s V%s" stroke="%s" stroke-width="1" fill="none"/>'
                 % (_num(xx), _num(py0), _num(py1), _GRID_COLOR))
        o.append('<path d="M%s %s V%s" stroke="%s" stroke-width="1"/>'
                 % (_num(xx), _num(py1), _num(py1 + 4), _AXIS_COLOR))
        o.append('<text x="%s" y="%s" font-size="%s" fill="%s" '
                 'text-anchor="middle">%s</text>'
                 % (_num(xx), _num(py1 + 6 + _FS_TICK), _num(_FS_TICK),
                    _MUTED_COLOR, _esc(lab)))

    # axis lines
    o.append('<path d="M%s %s V%s H%s" stroke="%s" stroke-width="1.2" fill="none"/>'
             % (_num(px0), _num(py0), _num(py1), _num(px1), _AXIS_COLOR))

    # axis captions
    if xlabel:
        o.append('<text x="%s" y="%s" font-size="%s" fill="%s" '
                 'text-anchor="middle">%s</text>'
                 % (_num((px0 + px1) / 2.0), _num(height - 8), _num(_FS_AXIS),
                    _TEXT_COLOR, _esc(xlabel)))
    if ylabel:
        cy = (py0 + py1) / 2.0
        o.append('<text x="%s" y="%s" font-size="%s" fill="%s" '
                 'text-anchor="middle" transform="rotate(-90 %s %s)">%s</text>'
                 % (_num(12.0 + _FS_AXIS * 0.8), _num(cy), _num(_FS_AXIS),
                    _TEXT_COLOR, _num(12.0 + _FS_AXIS * 0.8), _num(cy),
                    _esc(ylabel)))

    # ---- 6. the series ----------------------------------------------------
    o.append('<g clip-path="url(#plotclip)">')
    screen_pts: List[Tuple[float, float]] = []
    for idx, pts in enumerate(cleaned):
        color = series_colors[idx]
        if not pts:
            continue
        coords = [(sx(x), sy(y)) for (x, y) in pts]
        screen_pts.extend(coords)
        if len(coords) > 1:
            d = "M" + " L".join("%s %s" % (_num(cx), _num(cy)) for cx, cy in coords)
            o.append('<path d="%s" fill="none" stroke="%s" stroke-width="2" '
                     'stroke-linejoin="round" stroke-linecap="round"/>' % (d, color))
        shape = MARKERS[idx % len(MARKERS)]
        for cx, cy in coords:
            o.append(_marker_svg(shape, cx, cy, color))
    o.append('</g>')

    if not has_data:
        o.append('<text x="%s" y="%s" font-size="%s" fill="%s" '
                 'text-anchor="middle">no data to plot</text>'
                 % (_num((px0 + px1) / 2.0), _num((py0 + py1) / 2.0),
                    _num(_FS_AXIS), _MUTED_COLOR))

    # ---- 7. legend --------------------------------------------------------
    if labels:
        row_h = _FS_LEGEND + 5.0
        sw_len = 20.0
        pad = 7.0
        chrome = pad + sw_len + 5.0 + pad     # everything except the label text

        entries = [(labels[i] if cleaned[i] else labels[i] + " (no data)")
                   for i in range(len(labels))]

        # Fit horizontally: ellipsize labels rather than clipping the box.
        avail_text = (px1 - px0) - 8.0 - chrome
        if avail_text > 0.0:
            entries = [_ellipsize(e, _FS_LEGEND, avail_text) for e in entries]

        # Fit vertically: keep as many rows as fit, summarise the remainder.
        max_rows = int(((py1 - py0) - 8.0 - 2 * pad + 3.0) // row_h)
        shown = list(range(len(entries)))
        if max_rows >= 1 and len(shown) > max_rows:
            keep = max(1, max_rows - 1)
            hidden = len(shown) - keep
            shown = shown[:keep]
            overflow = "+%d more" % hidden
        else:
            overflow = None

        n_rows = len(shown) + (1 if overflow else 0)
        text_w = max([_text_width(entries[i], _FS_LEGEND) for i in shown]
                     + ([_text_width(overflow, _FS_LEGEND)] if overflow else []))
        lw = chrome + text_w
        lh = pad + row_h * n_rows + pad - 3.0

        inset = 8.0
        # If even the trimmed legend cannot fit the plot area, omit it rather
        # than letting it spill off the canvas.
        draw_legend = (lw <= (px1 - px0) - 4.0) and (lh <= (py1 - py0) - 4.0)
    else:
        draw_legend = False

    if labels and draw_legend:
        candidates = [
            (px1 - lw - inset, py0 + inset),   # top-right (preferred)
            (px1 - lw - inset, py1 - lh - inset),
            (px0 + inset, py0 + inset),
            (px0 + inset, py1 - lh - inset),
        ]
        best, best_score = candidates[0], None
        for cx0, cy0 in candidates:
            score = sum(1 for (mx, my) in screen_pts
                        if cx0 - 4 <= mx <= cx0 + lw + 4
                        and cy0 - 4 <= my <= cy0 + lh + 4)
            if best_score is None or score < best_score:
                best, best_score = (cx0, cy0), score
        lx, ly = best
        # belt and braces: keep the box strictly inside the plot area
        lx = max(px0 + 2.0, min(lx, px1 - lw - 2.0))
        ly = max(py0 + 2.0, min(ly, py1 - lh - 2.0))

        o.append('<g>')
        o.append('<rect x="%s" y="%s" width="%s" height="%s" rx="4" '
                 'fill="%s" fill-opacity="0.92" stroke="%s" stroke-width="1"/>'
                 % (_num(lx), _num(ly), _num(lw), _num(lh), _BG_COLOR, _GRID_COLOR))
        for row, idx in enumerate(shown):
            cy = ly + pad + row_h * row + row_h / 2.0 - 1.5
            color = series_colors[idx]
            o.append('<path d="M%s %s H%s" stroke="%s" stroke-width="2" '
                     'stroke-linecap="round"/>'
                     % (_num(lx + pad), _num(cy), _num(lx + pad + sw_len), color))
            o.append(_marker_svg(MARKERS[idx % len(MARKERS)],
                                 lx + pad + sw_len / 2.0, cy, color, r=3.0))
            o.append('<text x="%s" y="%s" font-size="%s" fill="%s">%s</text>'
                     % (_num(lx + pad + sw_len + 5.0), _num(cy + _FS_LEGEND * 0.35),
                        _num(_FS_LEGEND), _TEXT_COLOR, _esc(entries[idx])))
        if overflow:
            cy = ly + pad + row_h * len(shown) + row_h / 2.0 - 1.5
            o.append('<text x="%s" y="%s" font-size="%s" fill="%s">%s</text>'
                     % (_num(lx + pad + sw_len + 5.0), _num(cy + _FS_LEGEND * 0.35),
                        _num(_FS_LEGEND), _MUTED_COLOR, _esc(overflow)))
        o.append('</g>')

    o.append('</svg>')

    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(o))
        fh.write("\n")


# --------------------------------------------------------------------------
# write_csv
# --------------------------------------------------------------------------

def write_csv(path: str, header: Sequence[Any], rows: Sequence[Sequence[Any]]) -> None:
    """Write `header` + `rows` as a UTF-8 CSV file (parent dirs auto-created)."""
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        if header:
            writer.writerow(list(header))
        for row in (rows or []):
            writer.writerow(["" if c is None else c for c in row])


# --------------------------------------------------------------------------
# ascii_table
# --------------------------------------------------------------------------

def _cell_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return _fmt_num(value)
    return str(value)


def _is_numeric(text: str) -> bool:
    t = text.strip().replace(",", "")
    if not t:
        return False
    try:
        float(t)
        return True
    except ValueError:
        return False


def ascii_table(header: Sequence[Any], rows: Sequence[Sequence[Any]]) -> str:
    """Render a fixed-width, markdown-ish table (readable in a terminal).

    Numeric columns are right-aligned, everything else left-aligned; the
    alignment is also encoded in the markdown separator row, so the output
    renders correctly when pasted into `report.md`.
    """
    head = [_cell_str(h) for h in (header or [])]
    body = [[_cell_str(c) for c in row] for row in (rows or [])]

    n_cols = max([len(head)] + [len(r) for r in body]) if (head or body) else 0
    if n_cols == 0:
        return ""
    head += [""] * (n_cols - len(head))
    for row in body:
        row += [""] * (n_cols - len(row))

    # a column is numeric when every non-empty body cell parses as a number
    numeric: List[bool] = []
    for j in range(n_cols):
        vals = [r[j] for r in body if r[j].strip()]
        numeric.append(bool(vals) and all(_is_numeric(v) for v in vals))

    widths = [max(3, len(head[j]), *( [len(r[j]) for r in body] or [0] ))
              for j in range(n_cols)]

    def fmt_row(cells: Sequence[str], align_numeric: bool) -> str:
        out = []
        for j, cell in enumerate(cells):
            if align_numeric and numeric[j]:
                out.append(cell.rjust(widths[j]))
            else:
                out.append(cell.ljust(widths[j]))
        return "| " + " | ".join(out) + " |"

    lines = [fmt_row(head, align_numeric=True)]
    sep = []
    for j in range(n_cols):
        sep.append(("-" * (widths[j] + 1) + ":") if numeric[j]
                   else (":" + "-" * (widths[j] + 1)))
    lines.append("|" + "|".join(sep) + "|")
    for row in body:
        lines.append(fmt_row(row, align_numeric=True))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def _self_test() -> int:
    import tempfile

    out_dir = tempfile.mkdtemp(prefix="ctim_plotting_")
    written: List[str] = []

    # --- 1. linear chart: Fig. 2a of the paper (SPEC.md section 9) ---------
    ks = [1, 11, 21, 31, 41, 51]
    fig2a = [
        ("CTIM", list(zip(ks, [900, 3000, 4350, 4700, 4950, 5050]))),
        ("CTIM_CGA", list(zip(ks, [900, 3000, 4300, 4700, 4900, 5000]))),
        ("AIR+CGA", list(zip(ks, [700, 2700, 3900, 4400, 4700, 4700]))),
        ("CINEMA", list(zip(ks, [500, 2400, 3400, 3900, 4300, 4600]))),
        ("Greedy", list(zip(ks, [300, 2000, 2900, 3300, 3700, 3950]))),
    ]
    p1 = os.path.join(out_dir, "fig2a_linear.svg")
    line_chart(p1, fig2a, "seed set size K", "influence spread I(S)",
               "Fig. 2a - Yelp: influence spread vs K (Z=8, C=100)", xticks=ks)
    written.append(p1)

    # --- 2. log chart: Fig. 2b, with a deliberate 0 and a negative value ---
    fig2b = [
        ("CTIM", [(1, 2.0e2), (51, 7.0e2)]),
        ("CTIM_CGA", [(1, 3.0e3), (51, 3.0e4)]),
        ("CINEMA", [(1, 1.0e4), (51, 1.4e5)]),
        ("AIR+CGA", [(1, 2.0e4), (51, 2.0e5)]),
        ("Greedy", [(1, 3.0e3), (21, 0.0), (31, -5.0), (51, 2.5e4)]),
    ]
    p2 = os.path.join(out_dir, "fig2b_log.svg")
    line_chart(p2, fig2b, "seed set size K", "running time (s)",
               "Fig. 2b - Yelp: running time vs K (log scale)",
               xticks=ks, logy=True)
    written.append(p2)

    # --- 3. degenerate: a single point, and an all-equal-y series ----------
    p3 = os.path.join(out_dir, "single_point.svg")
    line_chart(p3, [("CTIM", [(20, 4320.0)]), ("flat", [(10, 7.0), (30, 7.0)])],
               "C", "influence spread", "degenerate: single point + flat series")
    written.append(p3)

    # --- assertions --------------------------------------------------------
    for p in written:
        assert os.path.exists(p), "missing output: %s" % p
        with open(p, "r", encoding="utf-8") as fh:
            text = fh.read()
        assert len(text) > 0, "empty file: %s" % p
        first = text.lstrip()
        if first.startswith("<?xml"):
            first = first.split("?>", 1)[1].lstrip()
        assert first.startswith("<svg"), "not an SVG root: %s" % p
        assert 'viewBox="' in text, "no viewBox: %s" % p
        assert text.rstrip().endswith("</svg>"), "unterminated SVG: %s" % p
        assert "http://" not in text.replace('xmlns="http://www.w3.org/2000/svg"', "") \
            .replace('xmlns:xlink="http://www.w3.org/1999/xlink"', ""), \
            "external reference in %s" % p
        # well-formedness
        import xml.dom.minidom
        xml.dom.minidom.parseString(text)
        print("ok  %-22s %6d bytes" % (os.path.basename(p), len(text)))

    # log-scale specifics
    with open(written[1], "r", encoding="utf-8") as fh:
        log_svg = fh.read()
    assert "<tspan" in log_svg, "no 10^k superscript tick labels"
    assert "non-positive" in log_svg, "dropped-point note missing from subtitle"

    # --- csv + ascii_table -------------------------------------------------
    csv_path = os.path.join(out_dir, "fig2a.csv")
    header = ["K", "CTIM", "CTIM_CGA", "AIR+CGA", "CINEMA", "Greedy"]
    rows = [[k] + [s[1][i][1] for s in fig2a] for i, k in enumerate(ks)]
    write_csv(csv_path, header, rows)
    with open(csv_path, "r", encoding="utf-8") as fh:
        csv_text = fh.read()
    assert csv_text.startswith("K,CTIM,"), csv_text[:40]
    assert len(csv_text.strip().splitlines()) == len(ks) + 1
    print("ok  %-22s %6d bytes" % ("fig2a.csv", len(csv_text)))

    table = ascii_table(header, rows)
    lines = table.splitlines()
    assert len(lines) == len(ks) + 2, "unexpected table height"
    assert len(set(len(l) for l in lines)) == 1, "table is not fixed-width"
    print()
    print(table)
    print()
    print("all plotting self-tests passed; output in %s" % out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
