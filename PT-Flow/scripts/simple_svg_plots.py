"""Small dependency-free SVG charts for cluster evaluation scripts.

The Palmetto ``ptflow`` environment intentionally contains the model stack but
not matplotlib.  These helpers keep evaluation jobs self-contained: scripts
always emit vector figures alongside their CSV files without installing a
package into a shared environment.  SVG opens directly in a browser and can be
converted to PDF/PNG later for the paper.
"""

from __future__ import annotations

import math
from pathlib import Path
from xml.sax.saxutils import escape


_PALETTE = ("#a33b3b", "#c47b17", "#2774a6", "#2d8a4b", "#7554a6")


def _number(value: float) -> str:
    return f"{value:.3g}"


def _finite(values):
    return [float(v) for v in values if math.isfinite(float(v))]


def _scale(values, errors=()):
    all_values = _finite(values)
    for value, error in errors:
        if math.isfinite(float(value)) and math.isfinite(float(error)):
            all_values.extend((float(value) - float(error), float(value) + float(error)))
    if not all_values:
        return 0.0, 1.0
    low, high = min(all_values), max(all_values)
    if low == high:
        pad = max(abs(low) * 0.12, 0.05)
    else:
        pad = (high - low) * 0.12
    return low - pad, high + pad


def _svg_text(x, y, text, *, size=13, anchor="start", fill="#151515", extra=""):
    return (f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, sans-serif" '
            f'font-size="{size}" text-anchor="{anchor}" fill="{fill}" {extra}>'
            f'{escape(str(text))}</text>')


def render_panels_svg(path: Path, *, title: str, panels: list[dict], columns: int = 2) -> None:
    """Render basic bar/line panels using only SVG and the standard library.

    Each panel is ``{"kind": "line"|"bar", "series": [...], "xlabel": ...,
    "ylabel": ...}``.  A series has ``label``, ``x``, ``y``, optional ``err``,
    and optional ``color``.  ``x_log`` is supported for positive numeric x.
    """
    columns = max(1, min(int(columns), len(panels)))
    rows = int(math.ceil(len(panels) / columns))
    width, panel_h = 1080, 400
    height = 72 + rows * panel_h
    panel_w = width / columns
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             _svg_text(width / 2, 32, title, size=19, anchor="middle", extra='font-weight="bold"')]

    for index, panel in enumerate(panels):
        col, row = index % columns, index // columns
        left = col * panel_w + 72
        right = (col + 1) * panel_w - 28
        top = 72 + row * panel_h + 36
        bottom = 72 + (row + 1) * panel_h - 70
        pw, ph = right - left, bottom - top
        series = panel["series"]
        y_values = [v for item in series for v in item["y"]]
        errors = [(v, e) for item in series for v, e in zip(item["y"], item.get("err", [0.0] * len(item["y"]))) ]
        y0, y1 = _scale(y_values, errors)
        if panel.get("clamp_y_zero", False):
            y0 = min(0.0, y0)

        xs = [float(v) for item in series for v in item["x"]]
        x_log = bool(panel.get("x_log", False))
        tx = [math.log10(max(v, 1e-12)) for v in xs] if x_log else xs
        x0, x1 = _scale(tx)
        if x0 == x1:
            x0 -= 1.0; x1 += 1.0
        def px(x):
            value = math.log10(max(float(x), 1e-12)) if x_log else float(x)
            return left + (value - x0) / (x1 - x0) * pw
        def py(y):
            return bottom - (float(y) - y0) / (y1 - y0) * ph

        parts.extend((_svg_text((left + right) / 2, top - 15, panel.get("title", ""), size=14, anchor="middle", extra='font-weight="bold"'),
                      f'<line x1="{left:.2f}" y1="{bottom:.2f}" x2="{right:.2f}" y2="{bottom:.2f}" stroke="#333"/>',
                      f'<line x1="{left:.2f}" y1="{top:.2f}" x2="{left:.2f}" y2="{bottom:.2f}" stroke="#333"/>'))
        for fraction in range(6):
            y = top + ph * fraction / 5
            value = y1 - (y1 - y0) * fraction / 5
            parts.append(f'<line x1="{left:.2f}" y1="{y:.2f}" x2="{right:.2f}" y2="{y:.2f}" stroke="#ddd"/>')
            parts.append(_svg_text(left - 7, y + 4, _number(value), size=10, anchor="end", fill="#555"))

        if panel.get("kind") == "bar":
            labels = panel.get("xlabels", [str(v) for v in series[0]["x"]])
            n = len(labels)
            group_w = pw / max(n, 1)
            bar_w = group_w / (len(series) + 1)
            for s_idx, item in enumerate(series):
                color = item.get("color", _PALETTE[s_idx % len(_PALETTE)])
                for j, (value, error) in enumerate(zip(item["y"], item.get("err", [0.0] * n))):
                    x = left + group_w * j + bar_w * (s_idx + 0.5)
                    zero = py(0.0)
                    y = py(value)
                    parts.append(f'<rect x="{x:.2f}" y="{min(y, zero):.2f}" width="{bar_w * .82:.2f}" '
                                 f'height="{abs(zero-y):.2f}" fill="{color}"/>')
                    if math.isfinite(float(error)):
                        e0, e1 = py(float(value) - float(error)), py(float(value) + float(error))
                        cx = x + bar_w * .41
                        parts.extend((f'<line x1="{cx:.2f}" y1="{e0:.2f}" x2="{cx:.2f}" y2="{e1:.2f}" stroke="#222"/>',
                                      f'<line x1="{cx-3:.2f}" y1="{e0:.2f}" x2="{cx+3:.2f}" y2="{e0:.2f}" stroke="#222"/>',
                                      f'<line x1="{cx-3:.2f}" y1="{e1:.2f}" x2="{cx+3:.2f}" y2="{e1:.2f}" stroke="#222"/>'))
            for j, label in enumerate(labels):
                parts.append(_svg_text(left + group_w * (j + .5), bottom + 17, label, size=10, anchor="middle"))
        else:
            ticks = sorted(set(xs))
            for x in ticks:
                parts.append(f'<line x1="{px(x):.2f}" y1="{bottom:.2f}" x2="{px(x):.2f}" y2="{bottom+4:.2f}" stroke="#333"/>')
                parts.append(_svg_text(px(x), bottom + 17, _number(x), size=10, anchor="middle", fill="#555"))
            for s_idx, item in enumerate(series):
                color = item.get("color", _PALETTE[s_idx % len(_PALETTE)])
                coords = " ".join(f"{px(x):.2f},{py(y):.2f}" for x, y in zip(item["x"], item["y"]))
                parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.4"/>')
                for x, value, error in zip(item["x"], item["y"], item.get("err", [0.0] * len(item["y"]))):
                    xpix, ypix = px(x), py(value)
                    if math.isfinite(float(error)):
                        e0, e1 = py(float(value) - float(error)), py(float(value) + float(error))
                        parts.append(f'<line x1="{xpix:.2f}" y1="{e0:.2f}" x2="{xpix:.2f}" y2="{e1:.2f}" stroke="{color}"/>')
                    parts.append(f'<circle cx="{xpix:.2f}" cy="{ypix:.2f}" r="3.3" fill="{color}"/>')

        parts.extend((_svg_text((left + right) / 2, bottom + 47, panel.get("xlabel", ""), size=12, anchor="middle"),
                      _svg_text(left - 51, (top + bottom) / 2, panel.get("ylabel", ""), size=12, anchor="middle",
                                extra=f'transform="rotate(-90 {left-51:.2f} {(top+bottom)/2:.2f})"')))
        legend_x, legend_y = right - 8, top + 14
        for s_idx, item in enumerate(series):
            color = item.get("color", _PALETTE[s_idx % len(_PALETTE)])
            text_width = max(54, len(str(item["label"])) * 6.2)
            x = legend_x - text_width
            parts.append(f'<line x1="{x-17:.2f}" y1="{legend_y:.2f}" x2="{x-3:.2f}" y2="{legend_y:.2f}" stroke="{color}" stroke-width="3"/>')
            parts.append(_svg_text(x, legend_y + 4, item["label"], size=10, anchor="start"))
            legend_y += 15
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
