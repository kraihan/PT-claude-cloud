"""Nested-Monte-Carlo latent-NLL grid on fixed ImageNet validation latents.

This evaluates the likelihood formula implemented by PT-Flow at a frozen
checkpoint.  It intentionally reads ``val_moments.npy`` and ``val_targets.npy``
directly, without the dataset loader's random flip, so every (K_outer, K_inner,
seed) cell uses exactly the same held-out ImageNet latent examples.

The reported number is **latent-space nested-MC NLL**, not pixel-space bpd and
not an IWAE lower bound.  Both Monte-Carlo budgets are swept and repeated seeds
give the displayed confidence intervals.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference import _load_model  # noqa: E402
from ptflow.sampling import log_likelihood  # noqa: E402
from ptflow.schedule import build_schedule  # noqa: E402
from utils.env import IMAGENET_CACHE_PATH  # noqa: E402
from utils.misc import load_config  # noqa: E402
from scripts.simple_svg_plots import render_panels_svg  # noqa: E402


def _parse_ints(value: str) -> list[int]:
    values = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not values or any(x < 1 for x in values):
        raise ValueError("budget lists must contain positive integers")
    return sorted(set(values))


def _mean_ci(values: list[float]) -> tuple[float, float]:
    average = mean(values)
    return (average, float("nan")) if len(values) < 2 else (average, 1.96 * stdev(values) / math.sqrt(len(values)))


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _checkpoint_alpha(config_path: Path, checkpoint: Path) -> tuple[float, float]:
    config = load_config(str(config_path))
    schedule = build_schedule(dict((config.get("pt") or {}).get("schedule", {})))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    schedule.load_state_dict(state.get("pt_schedule", {}))
    eps, alpha = float(schedule.eps()), float(schedule.alpha_def())
    del state
    return eps, alpha


def _plot(summary: list[dict], output: Path) -> None:
    inner_budgets = sorted({int(r["K_inner"]) for r in summary})
    series = []
    for inner in inner_budgets:
        rows = sorted((r for r in summary if int(r["K_inner"]) == inner), key=lambda r: int(r["K_outer"]))
        series.append({"label": f"K_inner={inner}", "x": [int(r["K_outer"]) for r in rows],
                       "y": [float(r["nll_per_dim_nats_mean"]) for r in rows],
                       "err": [float(r["nll_per_dim_nats_ci95"]) for r in rows]})
    render_panels_svg(output, title="Fixed ImageNet-1k validation latents; 95% MC intervals",
                      panels=[{"kind": "line", "x_log": True, "series": series,
                               "xlabel": "outer Monte-Carlo budget K_outer",
                               "ylabel": "latent NLL / dimension (nats; lower is not certified tighter)"}],
                      columns=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, help="New or empty output directory")
    parser.add_argument("--latent-cache", default=IMAGENET_CACHE_PATH,
                        help="Directory containing val_moments.npy and val_targets.npy")
    parser.add_argument("--points", type=int, default=32, help="Fixed held-out validation latents")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Keep at 1 for the L checkpoint unless memory measurements support more")
    parser.add_argument("--k-outer", default="8,16,32")
    parser.add_argument("--k-inner", default="8,16,32")
    parser.add_argument("--mc-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=19000)
    parser.add_argument("--alpha-def", type=float, default=None,
                        help="Default: defensive-mixture weight restored from checkpoint schedule")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ckpt, config, out = Path(args.ckpt).expanduser().resolve(), Path(args.config).expanduser().resolve(), Path(args.out).expanduser().resolve()
    cache = Path(args.latent_cache).expanduser().resolve()
    moments_path, targets_path = cache / "val_moments.npy", cache / "val_targets.npy"
    for path in (ckpt, config, moments_path, targets_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Refusing to mix results into nonempty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    k_outer, k_inner = _parse_ints(args.k_outer), _parse_ints(args.k_inner)
    if args.points < 1 or args.batch_size < 1 or args.mc_seeds < 1:
        raise ValueError("points, batch-size, and mc-seeds must be positive")

    moments = np.load(moments_path, mmap_mode="r")
    targets = np.load(targets_path, mmap_mode="r")
    if len(moments) != len(targets) or len(moments) < args.points:
        raise ValueError("Validation cache is missing data or contains fewer rows than --points")
    # The cache is class-sorted by ImageFolder. Even spacing makes this a stable,
    # broadly class-covered held-out subset without inspecting samples by label.
    indices = (np.arange(int(args.points), dtype=np.int64) * len(moments) // int(args.points)).astype(np.int64)
    selected_latents = np.asarray(moments[indices], dtype=np.float32)
    selected_labels = np.asarray(targets[indices], dtype=np.int64)
    np.save(out / "val_indices.npy", indices)
    index_sha256 = hashlib.sha256(indices.tobytes()).hexdigest()

    checkpoint_eps, checkpoint_alpha = _checkpoint_alpha(config, ckpt)
    alpha_def = float(checkpoint_alpha if args.alpha_def is None else args.alpha_def)
    model, _, step, device, potential, scale_net, _ = _load_model(str(ckpt), str(config), want_potential=True)
    model.eval(); potential.eval()
    if scale_net is not None:
        scale_net.eval()
    if tuple(selected_latents.shape[1:]) != (model.input_size, model.input_size, model.in_channels):
        raise ValueError(f"Latent cache shape {selected_latents.shape[1:]} does not match model input "
                         f"{(model.input_size, model.input_size, model.in_channels)}")

    manifest = {
        "checkpoint": str(ckpt), "config": str(config), "step": int(step), "device": str(device),
        "latent_cache": str(cache), "points": int(args.points), "val_indices_sha256": index_sha256,
        "k_outer": k_outer, "k_inner": k_inner, "mc_seeds": int(args.mc_seeds),
        "checkpoint_epsilon": checkpoint_eps, "alpha_def": alpha_def,
        "quantity": "fixed-validation latent-space nested-MC NLL; not pixel likelihood and not a certified bound",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    raw_rows: list[dict] = []
    for inner in k_inner:
        for outer in k_outer:
            for mc_seed in range(int(args.mc_seeds)):
                rng = torch.Generator(device=device).manual_seed(int(args.seed) + 1_000_003 * (mc_seed + 1))
                nll_sum = bpd_sum = ess_sum = 0.0
                count = 0
                for start in range(0, int(args.points), int(args.batch_size)):
                    stop = min(start + int(args.batch_size), int(args.points))
                    x1 = torch.from_numpy(selected_latents[start:stop]).to(device=device, dtype=torch.float32)
                    labels = torch.from_numpy(selected_labels[start:stop]).to(device=device, dtype=torch.long)
                    _, info = log_likelihood(
                        model, potential, x1, labels, checkpoint_eps,
                        K_outer=outer, K_inner=inner, alpha_def=alpha_def,
                        scale_net=scale_net, rng=rng,
                    )
                    n = stop - start
                    nll_sum += float(info["nll_per_dim"]) * n
                    bpd_sum += float(info["bits_per_dim_latent"]) * n
                    ess_sum += float(info["inner_ess"]) * n
                    count += n
                raw_rows.append({
                    "K_outer": outer, "K_inner": inner, "mc_seed": mc_seed, "points": count,
                    "nll_per_dim_nats": nll_sum / count,
                    "bits_per_dim_latent": bpd_sum / count,
                    "inner_ess_over_K": ess_sum / count,
                    "eps": checkpoint_eps, "alpha_def": alpha_def,
                })
                print(f"K_outer={outer:>3} K_inner={inner:>3} seed={mc_seed}: "
                      f"NLL/dim={nll_sum / count:.6f} nats, inner ESS/K={ess_sum / count:.3f}", flush=True)

    summary: list[dict] = []
    for inner in k_inner:
        for outer in k_outer:
            subset = [row for row in raw_rows if row["K_outer"] == outer and row["K_inner"] == inner]
            item = {"K_outer": outer, "K_inner": inner, "mc_seeds": len(subset), "points": int(args.points)}
            for metric in ("nll_per_dim_nats", "bits_per_dim_latent", "inner_ess_over_K"):
                avg, ci = _mean_ci([float(row[metric]) for row in subset])
                item[f"{metric}_mean"] = avg
                item[f"{metric}_ci95"] = ci
            summary.append(item)
    _write_csv(out / "raw.csv", raw_rows)
    _write_csv(out / "summary.csv", summary)
    _plot(summary, out / "nll_budget_grid.svg")
    print(f"Wrote latent-NLL CSVs and figure to {out}", flush=True)


if __name__ == "__main__":
    main()
