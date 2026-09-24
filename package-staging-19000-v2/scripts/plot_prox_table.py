"""Render the fixed-checkpoint bridge/prox/network table as a paper figure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _numbers(rows, name):
    return np.asarray([float(row[name]) for row in rows])


def _ci(rows, name):
    values = _numbers(rows, name)
    return np.asarray([float(row[f"{name.rsplit('_mean', 1)[0]}_mc_ci95"]) if np.isfinite(float(row[f"{name.rsplit('_mean', 1)[0]}_mc_ci95"])) else 0.0 for row in rows])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="summary.csv written by eval_prox_table.py")
    parser.add_argument("--out", required=True, help="PNG output path")
    args = parser.parse_args()
    with Path(args.summary).open(newline="", encoding="utf-8") as f:
        rows = sorted(list(csv.DictReader(f)), key=lambda r: float(r["epsilon"]))
    if not rows:
        raise ValueError("summary CSV contains no rows")
    eps = _numbers(rows, "epsilon")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1), constrained_layout=True)
    for key, label, color in (
        ("bridge_to_prox_rms_mean", r"$\\hat T_\\epsilon - y_{ref}$", "#2774a6"),
        ("bridge_to_network_rms_mean", r"$\\hat T_\\epsilon - m_\\eta$", "#c47b17"),
        ("prox_to_network_rms_mean", r"$y_{ref} - m_\\eta$", "#2d8a4b"),
    ):
        axes[0].errorbar(eps, _numbers(rows, key), yerr=_ci(rows, key), marker="o", capsize=2, label=label, color=color)
    axes[0].set_xscale("log"); axes[0].set_xlabel("epsilon")
    axes[0].set_ylabel("per-coordinate RMS")
    axes[0].set_title("Bridge / prox / network gaps")
    axes[0].grid(alpha=0.25); axes[0].legend(frameon=False)

    axes[1].errorbar(eps, _numbers(rows, "relative_prox_residual_mean"),
                     yerr=_ci(rows, "relative_prox_residual_mean"), marker="o", capsize=2, color="#a33b3b")
    axes[1].set_xscale("log"); axes[1].set_xlabel("epsilon")
    axes[1].set_ylabel(r"mean $||r_\\eta|| / ||m_\\eta-x_0||$")
    axes[1].set_title("Learned-prox residual")
    axes[1].grid(alpha=0.25)

    axes[2].errorbar(eps, _numbers(rows, "mean_ess_over_K_mean"),
                     yerr=_ci(rows, "mean_ess_over_K_mean"), marker="o", capsize=2, color="#6d50a3")
    axes[2].axhline(0.05, color="#555555", linewidth=1, linestyle="--", label="reportability gate")
    axes[2].set_xscale("log"); axes[2].set_xlabel("epsilon")
    axes[2].set_ylabel("ESS / K")
    axes[2].set_title(r"$\\hat T_\\epsilon$ importance-estimator health")
    axes[2].grid(alpha=0.25); axes[2].legend(frameon=False)
    fig.savefig(Path(args.out), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
