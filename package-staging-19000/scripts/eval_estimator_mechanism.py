"""Mechanism figures for a frozen ImageNet-latent PT-Flow checkpoint.

The program measures the proposal used by the *actual* tilted estimator on
fixed Gaussian x0 inputs and fixed, broadly spaced ImageNet class labels.  It
does not generate FID samples, train, or alter a checkpoint.  It writes:

  raw.csv / summary.csv
  proposal_components.png
  epsilon_sweep.png
  proposal_mismatch.png

The four proposal labels have exact code-level meanings:

  naive              q = N(x0, 2 eps I)
  recenter_only      q = N(m_eta(x0), 2 eps I)
  learned_diagonal   q = N(m_eta(x0), 2 eps diag(exp(s_eta)))
  full_defensive     (1-alpha) learned_diagonal + alpha naive

All component comparisons reuse the same fixed x0/labels and the same random
stream within each Monte-Carlo seed.  ``chi2_hat`` is an empirical weight-moment
diagnostic, not an exact divergence at finite K.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference import _load_model  # noqa: E402
from ptflow.estimator import log_kernel_over_proposal, sample_proposal  # noqa: E402
from ptflow.sampling import _generator_output  # noqa: E402
from ptflow.schedule import build_schedule  # noqa: E402
from utils.misc import load_config  # noqa: E402


PROPOSALS = ("naive", "recenter_only", "learned_diagonal", "full_defensive")
PROPOSAL_LABELS = {
    "naive": "naive",
    "recenter_only": "recenter only",
    "learned_diagonal": "learned diagonal",
    "full_defensive": "full defensive",
}
COLORS = {
    "naive": "#a33b3b",
    "recenter_only": "#c47b17",
    "learned_diagonal": "#2774a6",
    "full_defensive": "#2d8a4b",
}


def _guided_phi(potential, x: torch.Tensor, labels: torch.Tensor, pt_w: float) -> torch.Tensor:
    if float(pt_w) == 0.0:
        return potential.phi(x, labels)
    return ((1.0 + float(pt_w)) * potential.phi(x, labels)
            - float(pt_w) * potential.phi(x, potential.null_labels(labels)))


@torch.no_grad()
def _proposal_metrics(
    potential,
    x0: torch.Tensor,
    labels: torch.Tensor,
    center: torch.Tensor,
    log_scale: torch.Tensor | None,
    eps: float,
    pt_w: float,
    alpha_def: float,
    K: int,
    rng: torch.Generator,
) -> dict[str, float]:
    """Weight-moment metrics for one proposal, without optimization."""
    y = sample_proposal(x0, center, log_scale, eps, K, alpha_def, generator=rng, antithetic=True)
    log_ratio = log_kernel_over_proposal(y, x0, center, log_scale, eps, alpha_def)
    b, k = y.shape[:2]
    phi = _guided_phi(
        potential, y.reshape(b * k, *y.shape[2:]), labels.repeat_interleave(k), pt_w
    ).reshape(b, k).double()
    # A per-x0 constant does not change normalized weights or their moments.
    log_w = log_ratio - (phi - phi.mean(dim=1, keepdim=True)) / (2.0 * float(eps))
    finite = torch.isfinite(log_w).all(dim=1)
    lse1 = torch.logsumexp(log_w, dim=1)
    lse2 = torch.logsumexp(2.0 * log_w, dim=1)
    ess = torch.exp(2.0 * lse1 - lse2 - math.log(k)).clamp(0.0, 1.0)
    log1p_chi2 = (math.log(k) + lse2 - 2.0 * lse1).clamp_min(0.0)
    chi2 = torch.expm1(log1p_chi2)
    spread2 = log_w.var(dim=1, correction=0)
    max_share = torch.exp(log_w.max(dim=1).values - lse1).clamp(0.0, 1.0)
    return {
        "ess_over_k": float(ess.mean().item()),
        "chi2_hat": float(chi2.mean().item()),
        "log1p_chi2": float(log1p_chi2.mean().item()),
        "logw_variance": float(spread2.mean().item()),
        "max_weight_share": float(max_share.mean().item()),
        "finite_rate": float(finite.float().mean().item()),
    }


def _proposal_parameters(
    proposal: str,
    x0: torch.Tensor,
    m: torch.Tensor,
    s: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor | None, float]:
    if proposal == "naive":
        return x0, None, 1.0
    if proposal == "recenter_only":
        return m, None, 0.0
    if proposal == "learned_diagonal":
        return m, s, 0.0
    if proposal == "full_defensive":
        return m, s, float(alpha)
    raise ValueError(proposal)


def _mean_ci(values: list[float]) -> tuple[float, float]:
    average = mean(values)
    if len(values) < 2:
        return average, float("nan")
    return average, 1.96 * stdev(values) / math.sqrt(len(values))


def _write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    group_fields = ("study", "proposal", "epsilon", "mismatch_axis", "mismatch_level")
    metrics = ("ess_over_k", "chi2_hat", "log1p_chi2", "logw_variance", "max_weight_share", "finite_rate")
    for row in rows:
        groups[tuple(row.get(key, "") for key in group_fields)].append(row)
    output = []
    for values in groups.values():
        first = values[0]
        item = {key: first.get(key, "") for key in group_fields}
        item["mc_seeds"] = len(values)
        for metric in metrics:
            avg, ci = _mean_ci([float(v[metric]) for v in values])
            item[f"{metric}_mean"] = avg
            item[f"{metric}_ci95"] = ci
        output.append(item)
    return output


def _summary_lookup(rows: list[dict], **where) -> list[dict]:
    return [row for row in rows if all(str(row.get(k)) == str(v) for k, v in where.items())]


def _plot_components(summary: list[dict], out: Path, eps: float) -> None:
    rows = [_summary_lookup(summary, study="components", epsilon=eps, proposal=p)[0] for p in PROPOSALS]
    x = np.arange(len(rows))
    labels = [PROPOSAL_LABELS[row["proposal"]] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, metric, ylabel, transform in (
        (axes[0], "ess_over_k", "ESS / K (higher is better)", lambda v: v),
        (axes[1], "log1p_chi2", r"log(1 + $\\hat\\chi^2$) (lower is better)", lambda v: v),
    ):
        values = [transform(float(row[f"{metric}_mean"])) for row in rows]
        errors = [transform(float(row[f"{metric}_ci95"])) if np.isfinite(float(row[f"{metric}_ci95"])) else 0.0 for row in rows]
        ax.bar(x, values, yerr=errors, capsize=3, color=[COLORS[row["proposal"]] for row in rows])
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"Proposal components at epsilon={eps:g}; fixed ImageNet-latent x0/labels")
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_epsilon(summary: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for proposal in PROPOSALS:
        rows = sorted(_summary_lookup(summary, study="epsilon", proposal=proposal), key=lambda r: float(r["epsilon"]))
        eps = np.asarray([float(r["epsilon"]) for r in rows])
        for ax, metric, ylabel in (
            (axes[0], "ess_over_k", "ESS / K (higher is better)"),
            (axes[1], "log1p_chi2", r"log(1 + $\\hat\\chi^2$) (lower is better)"),
        ):
            values = np.asarray([float(r[f"{metric}_mean"]) for r in rows])
            errors = np.asarray([float(r[f"{metric}_ci95"]) if np.isfinite(float(r[f"{metric}_ci95"])) else 0.0 for r in rows])
            ax.errorbar(eps, values, yerr=errors, marker="o", capsize=2,
                        label=PROPOSAL_LABELS[proposal], color=COLORS[proposal])
            ax.set_xscale("log")
            ax.set_xlabel("epsilon")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Estimator health versus epsilon; same fixed x0/labels across the sweep")
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_mismatch(summary: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    for column, (axis_name, xlab) in enumerate((("center", r"center shift beta"), ("scale", "std. multiplier"))):
        for proposal in ("learned_diagonal", "full_defensive"):
            rows = sorted(_summary_lookup(summary, study="mismatch", mismatch_axis=axis_name, proposal=proposal),
                          key=lambda r: float(r["mismatch_level"]))
            xs = np.asarray([float(r["mismatch_level"]) for r in rows])
            for row_index, metric in enumerate(("ess_over_k", "log1p_chi2")):
                ys = np.asarray([float(r[f"{metric}_mean"]) for r in rows])
                es = np.asarray([float(r[f"{metric}_ci95"]) if np.isfinite(float(r[f"{metric}_ci95"])) else 0.0 for r in rows])
                axes[row_index, column].errorbar(xs, ys, yerr=es, marker="o", capsize=2,
                                                 color=COLORS[proposal], label=PROPOSAL_LABELS[proposal])
        axes[0, column].set_title("center perturbation" if axis_name == "center" else "scale perturbation")
        axes[1, column].set_xlabel(xlab)
        axes[0, column].set_ylabel("ESS / K")
        axes[1, column].set_ylabel(r"log(1 + $\\hat\\chi^2$)")
        for row_index in (0, 1):
            axes[row_index, column].grid(alpha=0.25)
            axes[row_index, column].legend(frameon=False)
    fig.suptitle("Proposal-mismatch stress test at the checkpoint epsilon")
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _read_checkpoint_alpha(config_path: Path, checkpoint: Path) -> tuple[float, float]:
    config = load_config(str(config_path))
    schedule = build_schedule(dict((config.get("pt") or {}).get("schedule", {})))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    schedule.load_state_dict(state.get("pt_schedule", {}))
    eps = float(schedule.eps())
    alpha = float(schedule.alpha_def())
    del state
    return eps, alpha


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, help="New or empty output directory")
    parser.add_argument("--cfg-scale", type=float, default=1.2)
    parser.add_argument("--pt-w", type=float, default=None)
    parser.add_argument("--points", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--K", type=int, default=512)
    parser.add_argument("--mc-seeds", type=int, default=5)
    parser.add_argument("--eps", default="0.20,0.10,0.05,0.02,0.01")
    parser.add_argument("--component-eps", type=float, default=None,
                        help="Defaults to the epsilon restored from this checkpoint")
    parser.add_argument("--alpha-def", type=float, default=None,
                        help="Defaults to the defensive weight restored from this checkpoint")
    parser.add_argument("--center-levels", default="0,0.25,0.5,1.0")
    parser.add_argument("--scale-levels", default="0.5,0.75,1.0,1.25,2.0")
    parser.add_argument("--seed", type=int, default=19000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ckpt = Path(args.ckpt).expanduser().resolve()
    config = Path(args.config).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if not ckpt.is_file() or not config.is_file():
        raise FileNotFoundError("Both --ckpt and --config must be existing files")
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Refusing to mix runs in nonempty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    eps_values = [float(x) for x in args.eps.split(",") if x.strip()]
    center_levels = [float(x) for x in args.center_levels.split(",") if x.strip()]
    scale_levels = [float(x) for x in args.scale_levels.split(",") if x.strip()]
    if any(x <= 0 for x in eps_values) or any(x <= 0 for x in scale_levels):
        raise ValueError("epsilons and scale levels must be positive")
    pt_w = float(args.cfg_scale - 1.0 if args.pt_w is None else args.pt_w)
    checkpoint_eps, checkpoint_alpha = _read_checkpoint_alpha(config, ckpt)
    component_eps = float(checkpoint_eps if args.component_eps is None else args.component_eps)
    alpha_def = float(checkpoint_alpha if args.alpha_def is None else args.alpha_def)
    if not 0.0 < alpha_def < 1.0:
        raise ValueError("full defensive proposal needs 0 < alpha_def < 1")

    model, _, step, device, potential, scale_net, _ = _load_model(str(ckpt), str(config), want_potential=True)
    if scale_net is None:
        raise RuntimeError("This checkpoint has no learned scale net; it cannot support the requested curvature-scaled comparison.")
    model.eval(); potential.eval(); scale_net.eval()
    manifest = {
        "checkpoint": str(ckpt), "config": str(config), "step": int(step), "device": str(device),
        "points": int(args.points), "K": int(args.K), "mc_seeds": int(args.mc_seeds),
        "cfg_scale": float(args.cfg_scale), "pt_w": pt_w,
        "checkpoint_epsilon": checkpoint_eps, "component_epsilon": component_eps,
        "checkpoint_alpha_def": checkpoint_alpha, "alpha_def": alpha_def,
        "epsilon_sweep": eps_values, "center_levels": center_levels, "scale_levels": scale_levels,
        "input_distribution": "fixed x0 ~ N(0,I), broadly spaced ImageNet-1k class labels",
        "note": "Finite-K chi2_hat/ESS are empirical proposal diagnostics, not exact divergences.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    accum: dict[tuple, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    x_rng = torch.Generator(device=device).manual_seed(int(args.seed))
    model_rng = torch.Generator(device=device).manual_seed(int(args.seed) + 1)

    def add(study: str, proposal: str, eps: float, seed: int, metrics: dict[str, float], *, mismatch_axis="", mismatch_level=""):
        key = (study, proposal, eps, seed, mismatch_axis, mismatch_level)
        bucket = accum[key]
        bucket["batches"] += 1.0
        for name, value in metrics.items():
            bucket[name] += float(value)

    for start in range(0, int(args.points), int(args.batch_size)):
        count = min(int(args.batch_size), int(args.points) - start)
        x0 = torch.randn((count, model.input_size, model.input_size, model.in_channels),
                         generator=x_rng, device=device, dtype=torch.float32)
        labels = (torch.arange(start, start + count, device=device) * int(model.num_classes) // int(args.points)).long()
        with torch.no_grad():
            m, _, s = _generator_output(model, labels, float(args.cfg_scale), x0=x0,
                                         rng=model_rng, scale_net=scale_net)
            m, s = m.float(), s.float()

        for seed in range(int(args.mc_seeds)):
            # Resetting this generator for every proposal makes z/u common random
            # numbers across the component comparison; only q's transformation changes.
            base_seed = int(args.seed) + 1_000_003 * (seed + 1) + start
            for proposal in PROPOSALS:
                rng = torch.Generator(device=device).manual_seed(base_seed)
                center, log_scale, alpha = _proposal_parameters(proposal, x0, m, s, alpha_def)
                add("components", proposal, component_eps, seed,
                    _proposal_metrics(potential, x0, labels, center, log_scale, component_eps,
                                      pt_w, alpha, int(args.K), rng))

            for eps_index, eps in enumerate(eps_values):
                for proposal in PROPOSALS:
                    rng = torch.Generator(device=device).manual_seed(base_seed + 100_000 * (eps_index + 1))
                    center, log_scale, alpha = _proposal_parameters(proposal, x0, m, s, alpha_def)
                    add("epsilon", proposal, eps, seed,
                        _proposal_metrics(potential, x0, labels, center, log_scale, eps,
                                          pt_w, alpha, int(args.K), rng))

            for proposal in ("learned_diagonal", "full_defensive"):
                _, _, alpha = _proposal_parameters(proposal, x0, m, s, alpha_def)
                for level_index, beta in enumerate(center_levels):
                    rng = torch.Generator(device=device).manual_seed(base_seed + 2_000_000 + level_index)
                    shifted = m + float(beta) * (m - x0)
                    add("mismatch", proposal, component_eps, seed,
                        _proposal_metrics(potential, x0, labels, shifted, s, component_eps,
                                          pt_w, alpha, int(args.K), rng),
                        mismatch_axis="center", mismatch_level=beta)
                for level_index, multiplier in enumerate(scale_levels):
                    rng = torch.Generator(device=device).manual_seed(base_seed + 3_000_000 + level_index)
                    shifted_s = s + 2.0 * math.log(float(multiplier))
                    add("mismatch", proposal, component_eps, seed,
                        _proposal_metrics(potential, x0, labels, m, shifted_s, component_eps,
                                          pt_w, alpha, int(args.K), rng),
                        mismatch_axis="scale", mismatch_level=multiplier)
        print(f"processed fixed x0/label points {start + 1}-{start + count}/{args.points}", flush=True)

    raw_rows = []
    for (study, proposal, eps, seed, axis, level), values in sorted(accum.items()):
        n = values.pop("batches")
        row = {"study": study, "proposal": proposal, "epsilon": eps, "mc_seed": seed,
               "mismatch_axis": axis, "mismatch_level": level}
        row.update({name: value / n for name, value in values.items()})
        raw_rows.append(row)
    summary_rows = _aggregate(raw_rows)
    _write_csv(out / "raw.csv", raw_rows)
    _write_csv(out / "summary.csv", summary_rows)
    _plot_components(summary_rows, out / "proposal_components.png", component_eps)
    _plot_epsilon(summary_rows, out / "epsilon_sweep.png")
    _plot_mismatch(summary_rows, out / "proposal_mismatch.png")
    print(f"Wrote mechanism figures and CSVs to {out}", flush=True)


if __name__ == "__main__":
    main()
