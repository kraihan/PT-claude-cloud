"""Fixed-checkpoint bridge-mean / prox / network diagnostic.

This program produces the experimental table needed to distinguish three
objects in PT-Flow at a *fixed* checkpoint:

    T_hat_eps(x)  : high-budget self-normalized importance estimate of the
                    bridge conditional mean;
    y_ref(x)      : numerically converged minimizer of
                    phi^w(y) + ||y - x||^2 / 2;
    m_eta(x)      : the deployed one-pass generator output.

The output deliberately calls the first quantity ``T_hat``.  A finite-K
importance estimate is not an exact bridge mean.  The CSV and Markdown table
include ESS/K, repeated-Monte-Carlo uncertainty, and prox-solver convergence so
that unsuitable rows are visible rather than silently reported.

Run from the repository root, for example:

  python scripts/eval_prox_table.py --ckpt /path/state_00019000.pt \
      --config /path/config.json --out /path/prox_table

This is an evaluation program only: it never writes a checkpoint and never
modifies model parameters.
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
from typing import Iterable

import torch


# ``python scripts/eval_prox_table.py`` otherwise places scripts/, rather than
# the repository root, first on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference import _load_model  # noqa: E402
from ptflow.estimator import log_kernel_over_proposal, sample_proposal  # noqa: E402
from ptflow.potential import prox_residual  # noqa: E402


def _guided_phi(potential, y: torch.Tensor, labels: torch.Tensor, pt_w: float) -> torch.Tensor:
    """phi^w(y,c) in the same guidance convention as Modes B and C."""
    if float(pt_w) == 0.0:
        return potential.phi(y, labels)
    conditional = potential.phi(y, labels)
    unconditional = potential.phi(y, potential.null_labels(labels))
    return (1.0 + float(pt_w)) * conditional - float(pt_w) * unconditional


def _rms(x: torch.Tensor) -> float:
    return float(x.float().square().mean().sqrt().item())


def _per_sample_rms(x: torch.Tensor) -> torch.Tensor:
    return x.float().flatten(1).square().mean(dim=1).sqrt()


def _energy_and_residual(
    potential, y: torch.Tensor, x0: torch.Tensor, labels: torch.Tensor, pt_w: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate F_x^w and its first-order residual at y (no parameter grads)."""
    yin = y.detach().float().requires_grad_(True)
    with torch.enable_grad():
        phi = _guided_phi(potential, yin, labels, pt_w)
        (grad,) = torch.autograd.grad(phi.sum(), yin, create_graph=False)
    residual = grad + yin - x0.float()
    energy = phi.detach() + 0.5 * (yin.detach() - x0.float()).flatten(1).square().sum(dim=1)
    return energy.detach(), residual.detach()


def _energy(
    potential, y: torch.Tensor, x0: torch.Tensor, labels: torch.Tensor, pt_w: float
) -> torch.Tensor:
    with torch.no_grad():
        return _guided_phi(potential, y.float(), labels, pt_w) + 0.5 * (
            y.float() - x0.float()
        ).flatten(1).square().sum(dim=1)


def solve_reference_prox(
    potential,
    x0: torch.Tensor,
    labels: torch.Tensor,
    initial: torch.Tensor,
    pt_w: float,
    *,
    max_steps: int,
    initial_step: float,
    tolerance: float,
    max_backtracks: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Backtracking first-order solve of y=prox_{phi^w}(x0).

    The learned output is only an initializer.  The solver checks the actual
    first-order condition and does not identify a partially refined point with
    the prox oracle.
    """
    y = initial.detach().float().clone()
    x_rms = _per_sample_rms(x0).clamp_min(1.0)
    converged = torch.zeros(len(y), dtype=torch.bool, device=y.device)
    accepted_any = torch.zeros(len(y), dtype=torch.bool, device=y.device)
    steps_used = 0

    for step in range(int(max_steps)):
        before, residual = _energy_and_residual(potential, y, x0, labels, pt_w)
        rel_residual = _per_sample_rms(residual) / x_rms
        converged |= rel_residual <= float(tolerance)
        if bool(converged.all()):
            steps_used = step
            break

        active = ~converged
        step_size = torch.full((len(y),), float(initial_step), device=y.device)
        accepted = torch.zeros_like(active)
        candidate_out = y
        residual_sq = residual.flatten(1).square().sum(dim=1)

        for _ in range(int(max_backtracks)):
            candidate = y - step_size.view(-1, *([1] * (y.ndim - 1))) * residual
            after = _energy(potential, candidate, x0, labels, pt_w)
            good = active & torch.isfinite(after) & (
                after <= before - 1e-4 * step_size * residual_sq
            )
            take = good & ~accepted
            candidate_out = torch.where(
                take.view(-1, *([1] * (y.ndim - 1))), candidate, candidate_out
            )
            accepted |= good
            if bool((accepted | ~active).all()):
                break
            step_size = torch.where(accepted | ~active, step_size, step_size * 0.5)

        y = candidate_out.detach()
        accepted_any |= accepted
        steps_used = step + 1

    _, final_residual = _energy_and_residual(potential, y, x0, labels, pt_w)
    final_relative = _per_sample_rms(final_residual) / x_rms
    converged = final_relative <= float(tolerance)
    return y, {
        "converged": converged.detach().cpu(),
        "relative_residual": final_relative.detach().cpu(),
        "accepted_any": accepted_any.detach().cpu(),
        "steps_used": int(steps_used),
    }


@torch.no_grad()
def estimate_bridge_mean(
    potential,
    x0: torch.Tensor,
    labels: torch.Tensor,
    prox_reference: torch.Tensor,
    scale_net,
    eps: float,
    pt_w: float,
    *,
    K: int,
    proposal_chunk: int,
    alpha_def: float,
    reference_scale: str,
    rng: torch.Generator,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Stream a high-budget SNIS conditional-mean estimate.

    The calculation is streamed over proposal points, so K can be increased
    without retaining [batch, K, 32, 32, 4] latent tensors in GPU memory.  The
    log weights use a common potential anchor per x0; this is algebraically the
    same target density while retaining the cancellation in float64.
    """
    if K < 2:
        raise ValueError("K must be at least 2 for a reference estimate")
    if proposal_chunk < 1:
        raise ValueError("proposal_chunk must be positive")

    if reference_scale == "learned" and scale_net is not None:
        log_scale = scale_net(prox_reference, labels).detach()
    elif reference_scale == "identity":
        log_scale = None
    else:
        raise ValueError(
            "reference_scale='learned' needs a scale net in this checkpoint; "
            "use --reference-scale identity instead."
        )

    B = x0.shape[0]
    tail = x0.shape[1:]
    phi_anchor = _guided_phi(potential, prox_reference, labels, pt_w).double()
    log_max = torch.full((B,), -torch.inf, device=x0.device, dtype=torch.float64)
    weight_sum = torch.zeros((B,), device=x0.device, dtype=torch.float64)
    squared_weight_sum = torch.zeros((B,), device=x0.device, dtype=torch.float64)
    weighted_sum = torch.zeros((B, int(math.prod(tail))), device=x0.device, dtype=torch.float64)

    remaining = int(K)
    while remaining:
        current = min(int(proposal_chunk), remaining)
        y = sample_proposal(
            x0, prox_reference, log_scale, float(eps), current, float(alpha_def),
            generator=rng, antithetic=True,
        )
        log_ratio = log_kernel_over_proposal(
            y, x0, prox_reference, log_scale, float(eps), float(alpha_def)
        )
        y_flat = y.reshape(B * current, *tail)
        labels_flat = labels.repeat_interleave(current, dim=0)
        phi_y = _guided_phi(potential, y_flat, labels_flat, pt_w).view(B, current).double()
        log_weight = log_ratio - (phi_y - phi_anchor[:, None]) / (2.0 * float(eps))

        chunk_max = log_weight.max(dim=1).values
        new_max = torch.maximum(log_max, chunk_max)
        old_factor = torch.where(torch.isfinite(log_max), torch.exp(log_max - new_max), torch.zeros_like(new_max))
        scaled = torch.exp(log_weight - new_max[:, None])
        weight_sum = weight_sum * old_factor + scaled.sum(dim=1)
        squared_weight_sum = squared_weight_sum * old_factor.square() + scaled.square().sum(dim=1)
        weighted_sum = weighted_sum * old_factor[:, None] + (
            scaled[:, :, None] * y.reshape(B, current, -1).double()
        ).sum(dim=1)
        log_max = new_max
        remaining -= current

    bridge_mean = (weighted_sum / weight_sum[:, None]).reshape(B, *tail).float()
    ess_over_k = (weight_sum.square() / (float(K) * squared_weight_sum)).clamp(0.0, 1.0)
    max_weight_share = (1.0 / weight_sum).clamp(0.0, 1.0)
    return bridge_mean, {
        "ess_over_k": ess_over_k.float(),
        "max_weight_share": max_weight_share.float(),
    }


def _mean_ci(values: Iterable[float]) -> tuple[float, float]:
    values = list(values)
    avg = mean(values)
    if len(values) < 2:
        return avg, float("nan")
    return avg, 1.96 * stdev(values) / math.sqrt(len(values))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _format_pm(avg: float, half_width: float, digits: int = 5) -> str:
    if math.isnan(half_width):
        return f"{avg:.{digits}g}"
    return f"{avg:.{digits}g} +/- {half_width:.2g}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, help="PT-Flow state_*.pt checkpoint")
    p.add_argument("--config", required=True, help="Matching training config.json")
    p.add_argument("--out", required=True, help="New or empty output directory")
    p.add_argument("--eps", default="0.20,0.10,0.05,0.02,0.01", help="Comma-separated evaluation epsilons")
    p.add_argument("--cfg-scale", type=float, default=1.2)
    p.add_argument("--pt-w", type=float, default=None, help="Default: cfg-scale - 1")
    p.add_argument("--points", type=int, default=16, help="Fixed x0/label pairs")
    p.add_argument("--batch-size", type=int, default=4, help="Points solved at one time")
    p.add_argument("--K", type=int, default=8192, help="Proposal draws per x0 for each MC seed")
    p.add_argument("--proposal-chunk", type=int, default=64, help="Potential-evaluation chunk size")
    p.add_argument("--mc-seeds", type=int, default=5, help="Independent IS seeds on the same x0 pairs")
    p.add_argument("--seed", type=int, default=19000, help="Seed for fixed x0 and generator noise labels")
    p.add_argument("--alpha-def", type=float, default=0.05)
    p.add_argument("--reference-scale", choices=["learned", "identity"], default="learned")
    p.add_argument("--prox-max-steps", type=int, default=500)
    p.add_argument("--prox-initial-step", type=float, default=0.5)
    p.add_argument("--prox-retry-max-steps", type=int, default=2500,
                   help="Extra iterations for only initially unconverged reference points")
    p.add_argument("--prox-retry-initial-step", type=float, default=0.1,
                   help="Conservative restart step for initially unconverged reference points")
    p.add_argument("--prox-tolerance", type=float, default=1e-3,
                   help="Reference prox relative RMS residual tolerance")
    p.add_argument("--prox-backtracks", type=int, default=12)
    p.add_argument("--allow-unconverged-prox", action="store_true",
                   help="Write diagnostic rows even if y_ref did not converge. Never report those rows as a prox oracle.")
    p.add_argument("--min-mean-ess", type=float, default=0.05,
                   help="Flag rows whose mean ESS/K is below this threshold")
    return p


def main() -> None:
    args = build_parser().parse_args()
    ckpt = Path(args.ckpt).expanduser().resolve()
    config = Path(args.config).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if not config.is_file():
        raise FileNotFoundError(f"Config not found: {config}")
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Refusing to mix results into a nonempty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    if args.points < 1 or args.batch_size < 1 or args.mc_seeds < 1:
        raise ValueError("points, batch-size, and mc-seeds must be positive")

    eps_values = [float(x.strip()) for x in args.eps.split(",") if x.strip()]
    if not eps_values or any(eps <= 0 for eps in eps_values):
        raise ValueError("--eps must contain positive values")
    pt_w = float(args.cfg_scale - 1.0 if args.pt_w is None else args.pt_w)

    model, _, step, device, potential, scale_net, checkpoint_eps = _load_model(
        str(ckpt), str(config), want_potential=True
    )
    model.eval()
    potential.eval()
    if scale_net is not None:
        scale_net.eval()

    metadata = {
        "checkpoint": str(ckpt),
        "config": str(config),
        "checkpoint_step": int(step),
        "checkpoint_schedule_eps": float(checkpoint_eps),
        "device": str(device),
        "points": int(args.points),
        "batch_size": int(args.batch_size),
        "K": int(args.K),
        "proposal_chunk": int(args.proposal_chunk),
        "mc_seeds": int(args.mc_seeds),
        "x0_seed": int(args.seed),
        "eps": eps_values,
        "cfg_scale": float(args.cfg_scale),
        "pt_w": pt_w,
        "alpha_def": float(args.alpha_def),
        "reference_scale": args.reference_scale,
        "prox_max_steps": int(args.prox_max_steps),
        "prox_retry_max_steps": int(args.prox_retry_max_steps),
        "prox_retry_initial_step": float(args.prox_retry_initial_step),
        "prox_tolerance": float(args.prox_tolerance),
    }
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    # Per epsilon / MC seed accumulators.  The x0 and labels are fixed across
    # eps and across MC seeds; only importance randomness changes.
    accum: dict[tuple[float, int], dict[str, object]] = {}
    for eps in eps_values:
        for mc_seed in range(int(args.mc_seeds)):
            accum[(eps, mc_seed)] = defaultdict(float)
            accum[(eps, mc_seed)]["n_coordinates"] = 0.0
            accum[(eps, mc_seed)]["n_points"] = 0.0
            accum[(eps, mc_seed)]["prox_converged"] = 0.0

    x_rng = torch.Generator(device=device)
    x_rng.manual_seed(int(args.seed))
    model_rng = torch.Generator(device=device)
    model_rng.manual_seed(int(args.seed) + 1)

    for start in range(0, int(args.points), int(args.batch_size)):
        count = min(int(args.batch_size), int(args.points) - start)
        x0 = torch.randn(
            (count, model.input_size, model.input_size, model.in_channels),
            generator=x_rng, device=device, dtype=torch.float32,
        )
        labels = torch.arange(start, start + count, device=device, dtype=torch.long) % int(model.num_classes)
        with torch.no_grad():
            generated = model(c=labels, cfg_scale=float(args.cfg_scale), deterministic=True,
                              train=False, rng=model_rng, x0=x0)["samples"].float()

        prox_reference, prox_info = solve_reference_prox(
            potential, x0, labels, generated, pt_w,
            max_steps=args.prox_max_steps,
            initial_step=args.prox_initial_step,
            tolerance=args.prox_tolerance,
            max_backtracks=args.prox_backtracks,
        )
        converged = prox_info["converged"].to(device=device, dtype=torch.bool)
        retry_count = 0
        # Never promote a partial iterate to y_ref.  Retrying only the failed
        # points from their last accepted iterate keeps the fixed evaluation
        # set intact while making the numerical oracle more robust.
        if not bool(converged.all()):
            bad_device = torch.nonzero(~converged, as_tuple=False).flatten()
            retry_count = int(len(bad_device))
            retry_reference, retry_info = solve_reference_prox(
                potential, x0[bad_device], labels[bad_device], prox_reference[bad_device], pt_w,
                max_steps=args.prox_retry_max_steps,
                initial_step=args.prox_retry_initial_step,
                tolerance=args.prox_tolerance,
                max_backtracks=args.prox_backtracks,
            )
            prox_reference[bad_device] = retry_reference
            bad_cpu = bad_device.detach().cpu()
            for field in ("converged", "relative_residual", "accepted_any"):
                prox_info[field][bad_cpu] = retry_info[field]
            prox_info["steps_used"] = int(prox_info["steps_used"]) + int(retry_info["steps_used"])
            converged = prox_info["converged"].to(device=device, dtype=torch.bool)
        if not bool(converged.all()) and not args.allow_unconverged_prox:
            bad = torch.nonzero(~converged, as_tuple=False).flatten().tolist()
            residuals = prox_info["relative_residual"][torch.tensor(bad)].tolist()
            raise RuntimeError(
                f"Reference prox did not converge for fixed points {[(start + i) for i in bad]}. "
                f"Final relative residuals: {[float(x) for x in residuals]}. Do not use a partially "
                "refined point as y_ref; inspect this checkpoint potential or tighten the reference solver."
            )

        network_residual = prox_residual(potential, generated, x0, labels, pt_w, create_graph=False)
        displacement = (generated - x0).flatten(1).norm(dim=1).clamp_min(1e-12)
        relative_network_residual = network_residual.flatten(1).norm(dim=1) / displacement
        reference_relative = prox_info["relative_residual"].to(device=device, dtype=torch.float32)
        prox_to_network_sse = float((prox_reference - generated).double().square().sum().item())
        network_residual_sse = float(network_residual.double().square().sum().item())
        coordinate_count = float(generated.numel())

        for eps_index, eps in enumerate(eps_values):
            for mc_seed in range(int(args.mc_seeds)):
                is_rng = torch.Generator(device=device)
                is_rng.manual_seed(int(args.seed) + 10_000_000 * (eps_index + 1) + 10_000 * (mc_seed + 1) + start)
                bridge_mean, is_info = estimate_bridge_mean(
                    potential, x0, labels, prox_reference, scale_net, eps, pt_w,
                    K=args.K, proposal_chunk=args.proposal_chunk,
                    alpha_def=args.alpha_def, reference_scale=args.reference_scale, rng=is_rng,
                )
                bucket = accum[(eps, mc_seed)]
                bucket["bridge_to_prox_sse"] += float((bridge_mean - prox_reference).double().square().sum().item())
                bucket["bridge_to_network_sse"] += float((bridge_mean - generated).double().square().sum().item())
                bucket["prox_to_network_sse"] += prox_to_network_sse
                bucket["network_residual_sse"] += network_residual_sse
                bucket["relative_network_residual_sum"] += float(relative_network_residual.sum().item())
                bucket["reference_relative_residual_sum"] += float(reference_relative.sum().item())
                bucket["ess_sum"] += float(is_info["ess_over_k"].sum().item())
                bucket["max_weight_share_sum"] += float(is_info["max_weight_share"].sum().item())
                bucket["n_coordinates"] += coordinate_count
                bucket["n_points"] += float(count)
                bucket["prox_converged"] += float(converged.sum().item())
                bucket["prox_retry_count"] += float(retry_count)
                bucket["prox_steps_sum"] += float(prox_info["steps_used"]) * count

        print(
            f"processed fixed points {start + 1}-{start + count}/{args.points}; "
            f"reference prox converged {int(converged.sum())}/{count}",
            flush=True,
        )

    rows: list[dict[str, object]] = []
    for eps in eps_values:
        for mc_seed in range(int(args.mc_seeds)):
            b = accum[(eps, mc_seed)]
            coordinates = float(b["n_coordinates"])
            points = float(b["n_points"])
            rows.append({
                "epsilon": eps,
                "mc_seed": mc_seed,
                "points": int(points),
                "K": int(args.K),
                "bridge_to_prox_rms": math.sqrt(float(b["bridge_to_prox_sse"]) / coordinates),
                "prox_to_network_rms": math.sqrt(float(b["prox_to_network_sse"]) / coordinates),
                "bridge_to_network_rms": math.sqrt(float(b["bridge_to_network_sse"]) / coordinates),
                "network_residual_rms": math.sqrt(float(b["network_residual_sse"]) / coordinates),
                "relative_prox_residual_mean": float(b["relative_network_residual_sum"]) / points,
                "reference_relative_residual_mean": float(b["reference_relative_residual_sum"]) / points,
                "mean_ess_over_K": float(b["ess_sum"]) / points,
                "mean_max_weight_share": float(b["max_weight_share_sum"]) / points,
                "prox_converged_rate": float(b["prox_converged"]) / points,
                "prox_steps_mean": float(b["prox_steps_sum"]) / points,
                "prox_retry_count": float(b["prox_retry_count"]),
            })
    _write_csv(out / "per_seed.csv", rows)

    summary_rows: list[dict[str, object]] = []
    for eps in eps_values:
        subset = [r for r in rows if float(r["epsilon"]) == float(eps)]
        summary: dict[str, object] = {"epsilon": eps, "mc_seeds": len(subset)}
        for key in (
            "bridge_to_prox_rms", "prox_to_network_rms", "bridge_to_network_rms",
            "network_residual_rms", "relative_prox_residual_mean",
            "reference_relative_residual_mean", "mean_ess_over_K", "mean_max_weight_share",
            "prox_converged_rate", "prox_steps_mean",
        ):
            avg, ci95 = _mean_ci(float(r[key]) for r in subset)
            summary[f"{key}_mean"] = avg
            summary[f"{key}_mc_ci95"] = ci95
        summary["reference_adequate"] = bool(
            float(summary["prox_converged_rate_mean"]) == 1.0
            and float(summary["mean_ess_over_K_mean"]) >= float(args.min_mean_ess)
        )
        summary_rows.append(summary)
    _write_csv(out / "summary.csv", summary_rows)

    table = [
        "# Fixed-checkpoint bridge/prox/network diagnostic",
        "",
        "All errors are per-coordinate RMS over the same fixed latent inputs. `T_hat` is a finite-K SNIS conditional-mean estimate, not an exact bridge mean. The +/- values are 95% Monte-Carlo intervals across repeated importance seeds on those fixed inputs.",
        "",
        "| epsilon | ESS/K | T_hat -> prox | prox -> network | T_hat -> network | relative prox residual | y_ref converged | reportable |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in summary_rows:
        table.append(
            "| {epsilon:.3g} | {ess} | {bp} | {pn} | {bn} | {rr} | {conv:.3f} | {ok} |".format(
                epsilon=float(r["epsilon"]),
                ess=_format_pm(float(r["mean_ess_over_K_mean"]), float(r["mean_ess_over_K_mc_ci95"])),
                bp=_format_pm(float(r["bridge_to_prox_rms_mean"]), float(r["bridge_to_prox_rms_mc_ci95"])),
                pn=_format_pm(float(r["prox_to_network_rms_mean"]), float(r["prox_to_network_rms_mc_ci95"])),
                bn=_format_pm(float(r["bridge_to_network_rms_mean"]), float(r["bridge_to_network_rms_mc_ci95"])),
                rr=_format_pm(float(r["relative_prox_residual_mean_mean"]), float(r["relative_prox_residual_mean_mc_ci95"])),
                conv=float(r["prox_converged_rate_mean"]),
                ok="yes" if bool(r["reference_adequate"]) else "no",
            )
        )
    table.extend([
        "",
        "A row is marked reportable only when every numerical prox solve reached the requested residual tolerance and mean ESS/K met the configured minimum. Low ESS/K or wide seed intervals mean that K must be increased or the row must be reported as an unstable estimate, not as an empirical bridge mean.",
        "",
    ])
    (out / "table.md").write_text("\n".join(table), encoding="utf-8")
    print("\n".join(table), flush=True)
    print(f"\nWrote {out / 'table.md'}", flush=True)


if __name__ == "__main__":
    main()
