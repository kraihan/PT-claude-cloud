"""Render the fixed-checkpoint bridge/prox/network table as a paper figure."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.simple_svg_plots import render_panels_svg


def _numbers(rows, name):
    return [float(row[name]) for row in rows]


def _ci(rows, name):
    values = _numbers(rows, name)
    return [float(row[f"{name.rsplit('_mean', 1)[0]}_mc_ci95"]) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="summary.csv written by eval_prox_table.py")
    parser.add_argument("--out", required=True, help="SVG output path")
    args = parser.parse_args()
    with Path(args.summary).open(newline="", encoding="utf-8") as f:
        rows = sorted(list(csv.DictReader(f)), key=lambda r: float(r["epsilon"]))
    if not rows:
        raise ValueError("summary CSV contains no rows")
    eps = _numbers(rows, "epsilon")
    gaps = []
    for key, label, color in (
        ("bridge_to_prox_rms_mean", "T_hat - y_ref", "#2774a6"),
        ("bridge_to_network_rms_mean", "T_hat - m_eta", "#c47b17"),
        ("prox_to_network_rms_mean", "y_ref - m_eta", "#2d8a4b"),
    ):
        gaps.append({"label": label, "x": eps, "y": _numbers(rows, key), "err": _ci(rows, key), "color": color})
    render_panels_svg(Path(args.out), title="Fixed-checkpoint bridge / prox / network diagnostics", panels=[
        {"kind": "line", "x_log": True, "series": gaps, "xlabel": "epsilon", "ylabel": "per-coordinate RMS"},
        {"kind": "line", "x_log": True, "series": [{"label": "relative prox residual", "x": eps,
         "y": _numbers(rows, "relative_prox_residual_mean"), "err": _ci(rows, "relative_prox_residual_mean"), "color": "#a33b3b"}],
         "xlabel": "epsilon", "ylabel": "mean ||r_eta|| / ||m_eta-x0||"},
        {"kind": "line", "x_log": True, "series": [{"label": "ESS / K", "x": eps,
         "y": _numbers(rows, "mean_ess_over_K_mean"), "err": _ci(rows, "mean_ess_over_K_mean"), "color": "#6d50a3"}],
         "xlabel": "epsilon", "ylabel": "ESS / K"},
    ], columns=3)


if __name__ == "__main__":
    main()
