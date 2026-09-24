"""Inference + evaluation with the repaired Section 2.5 ("new s^2").

This is a thin wrapper around the project's own ``inference.py``.  Nothing about
the trained checkpoint changes: sampling (Modes A/B/C) and FID are delegated to
``inference.py`` verbatim, because the *map* and the *samples* are unchanged by
the theory repair.  What this script adds is the new-s^2 characterization of the
importance estimator (``ptflow`` was fine; only the reported variance/budget
moved from a log-normal e^{s^2} to the chi^2 read straight off the weights):

    python -m s_res_var_log.infer variance   --ckpt ... --config ...   # NEW
    python -m s_res_var_log.infer likelihood --ckpt ... --config ...   # NLL + chi^2 budget
    python -m s_res_var_log.infer evaluate   --ckpt ... --config ...   # FID (delegated)
    python -m s_res_var_log.infer sample     --ckpt ... --config ...   # grid (delegated)

Run from the PT-Flow repo root.  Requires an *enabled* pt: block in the config
(a PT-Flow checkpoint) for `variance` and `likelihood`; `evaluate`/`sample` in
Mode A work on a baseline checkpoint too.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Allow both `python -m s_res_var_log.infer` and `python s_res_var_log/infer.py`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from inference import (
    _load_model,
    _print0,
    make_sampler,
    run_eval,
    run_eval_streaming,
    run_sample,
)
from utils.dist_util import barrier, init_distributed, process_count, process_index
from utils.env import CIFAR10_FID_NPZ, IMAGENET_FID_NPZ
from utils.misc import load_config

from s_res_var_log.s_res import (
    variance_report, eps_sweep, guided_log_likelihood, _GuidedPotential,
)


# ---------------------------------------------------------------------------
# variance -- the new-s^2 report on noise samples
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_variance(
    model, potential, scale_net, eps, *,
    num_batches: int, bsz: int, K: int, alpha_def: float,
    cfg_scale: float, pt_w, deltas, n_curvature_probes: int,
    eps_sweep_list, seed: int, device: torch.device,
    csv_out: str = "", json_out: str = "",
) -> dict | None:
    """Report chi^2 / K*(delta) / s_res^2 / (A6) term for the trained model.

    All quantities are inference-time functions of the same importance weights
    the estimator already produces -- the model is not retrained or altered.
    """
    if process_index() != 0:
        barrier()
        return None
    if potential is None:
        raise ValueError(
            "`variance` needs a PT-Flow potential (an enabled pt: block). "
            "It characterizes the tilted estimator, which a baseline run has none of."
        )

    num_classes = int(getattr(model, "num_classes", 1000))
    rng = torch.Generator(device=device).manual_seed(int(seed))

    reports, sweep_rows = [], []
    for b in range(int(num_batches)):
        labels = torch.arange(int(bsz), device=device).remainder(num_classes).long()
        # x0 = None: the generator draws its own noise, which is what Mode A does.
        rep = variance_report(
            model, potential, scale_net, x0=None, c=labels, eps=eps,
            K=int(K), alpha_def=float(alpha_def), cfg_scale=float(cfg_scale),
            pt_w=pt_w, deltas=deltas, n_curvature_probes=int(n_curvature_probes),
            measure_pure_tilt=(alpha_def > 0.0), rng=rng,
        )
        reports.append(rep.as_dict())
        _print0(
            f"  batch {b:2d}  chi2={rep.chi2:9.4f}  ESS/K={rep.ess_frac:.3f}  "
            f"s_res^2={rep.s_res2:.4e}  rel.rmse(K={K})={rep.rel_rmse:.4f}  "
            + "  ".join(f"K*({k})={v:.0f}" for k, v in rep.k_budget.items())
        )

    # optional eps-sweep (the variance-reversal curve, new chi^2 y-axis)
    if eps_sweep_list:
        labels = torch.arange(int(bsz), device=device).remainder(num_classes).long()
        _print0("  eps-sweep (variance reversal, chi^2 budget):")
        for rep in eps_sweep(
            model, potential, scale_net, x0=None, c=labels,
            eps_list=eps_sweep_list, K=int(K), alpha_def=0.0,
            cfg_scale=float(cfg_scale), deltas=tuple(deltas),
            n_curvature_probes=int(n_curvature_probes), rng=rng,
        ):
            sweep_rows.append(rep.as_dict())
            _print0(
                f"    eps={rep.eps:<8g} chi2={rep.chi2:10.4f}  "
                f"s_res^2={rep.s_res2:.4e}  legacy_lognormal_relvar={rep.legacy_lognormal_rel_var:.3e}"
            )

    def _mean(key):
        return float(np.mean([r[key] for r in reports]))

    summary = {
        "eps": float(eps),
        "K": int(K),
        "alpha_def": float(alpha_def),
        "guidance_pt_w": (float(pt_w) if pt_w is not None else float(cfg_scale) - 1.0),
        "num_batches": int(num_batches),
        "bsz": int(bsz),
        "chi2_mean": _mean("chi2"),
        "rel_var_mean": _mean("rel_var"),
        "rel_rmse_mean": _mean("rel_rmse"),
        "ess_frac_mean": _mean("ess_frac"),
        "s_res2_mean": _mean("s_res2"),
        "estimator": "chi2_from_weight_moments_repaired_section_2p5",
        "note": "budget = chi2/K (actual weight moments); NOT the log-normal e^{s^2}.",
        "per_batch": reports,
        "eps_sweep": sweep_rows,
    }

    # Per-batch table as CSV (the tabular output), summary as JSON (repo convention).
    if csv_out:
        out = Path(csv_out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        flat = []
        for i, r in enumerate(reports + sweep_rows):
            row = {k: v for k, v in r.items() if not isinstance(v, dict)}
            row["kind"] = "batch" if i < len(reports) else "eps_sweep"
            for k, v in r.get("k_budget", {}).items():
                row[f"K_budget[{k}]"] = v
            flat.append(row)
        cols = sorted({k for row in flat for k in row})
        with out.open("w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=cols)
            wr.writeheader()
            wr.writerows(flat)
        _print0(f"  wrote per-batch table -> {out}")

    if json_out:
        out = Path(json_out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        _print0(f"  wrote summary -> {out}")

    barrier()
    return summary


# ---------------------------------------------------------------------------
# likelihood -- NLL ladder (unchanged value) + new chi^2 budget on the inner MC
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_likelihood(
    model, potential, scale_net, eps, config_path, ckpt_path, ckpt_step, *,
    num_batches: int, bsz: int, k_inner: int, k_ladder, alpha_def: float,
    seed: int, device: torch.device, json_out: str = "",
) -> dict | None:
    """NLL along a K_outer ladder (identical estimator to inference.py), with the
    inner tilted-estimator variance now read as a chi^2 budget instead of a spread.

    The NLL numbers are unchanged -- the estimator is unbiased and untouched.
    Only the health/budget columns change: inner_chi2 = 1/inner_ess - 1, and the
    K_inner needed for a target inner relative RMSE.
    """
    if process_index() != 0:
        barrier()
        return None
    if potential is None:
        raise ValueError("`likelihood` requires a PT-Flow potential (enabled pt: block).")

    from pipelines import build_pipeline
    from ptflow.sampling import log_likelihood

    config = load_config(config_path)
    loader, preprocess_fn, _ = build_pipeline(config).build_split(
        batch_size=int(bsz), split="val"
    )
    rng = torch.Generator(device=device).manual_seed(int(seed))

    rows = []
    for K_eval in sorted(k_ladder):
        it = iter(loader)
        rng.manual_seed(int(seed))
        nlls, esss = [], []
        for _ in range(int(num_batches)):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            proc = preprocess_fn(batch)
            x1 = proc["images"].to(device).float()
            c = proc["labels"].to(device).long()
            _, info = log_likelihood(
                model, potential, x1, c, eps,
                K_outer=int(K_eval), K_inner=int(k_inner),
                alpha_def=float(alpha_def), scale_net=scale_net, rng=rng,
            )
            nlls.append(info["nll_per_dim"])
            esss.append(info["inner_ess"])

        ess = float(np.mean(esss))
        chi2 = max(1.0 / max(ess, 1e-8) - 1.0, 0.0)          # inner chi^2 from ESS
        row = {
            "K_eval": int(K_eval),
            "nll_per_dim_nats": float(np.mean(nlls)),
            "bits_per_dim_latent": float(np.mean(nlls)) / float(np.log(2.0)),
            "inner_ess": ess,
            "inner_chi2": chi2,
            "inner_rel_var": chi2 / float(k_inner),
            "inner_rel_rmse": (chi2 / float(k_inner)) ** 0.5,
            "K_inner_for_rmse_0p1": max(1, int(np.ceil(chi2 / 0.1 ** 2))),
        }
        rows.append(row)
        _print0(
            f"  K_eval={K_eval:<5d} NLL/dim={row['nll_per_dim_nats']:+.5f} nats  "
            f"inner ESS={ess:.3f}  chi2={chi2:.3f}  "
            f"inner rel.rmse(K_inner={k_inner})={row['inner_rel_rmse']:.4f}"
        )

    monotone = all(b["nll_per_dim_nats"] <= a["nll_per_dim_nats"] + 1e-6
                   for a, b in zip(rows, rows[1:]))
    result = {
        "ckpt": ckpt_path, "step": ckpt_step, "eps": float(eps),
        "guidance_w": 0.0, "k_inner": int(k_inner), "alpha_def": float(alpha_def),
        "ladder": rows, "monotone_tightening": bool(monotone),
        "estimator": "nested_monte_carlo_not_a_certified_bound",
        "variance_reporting": "chi2_from_ess_repaired_section_2p5",
    }
    if json_out:
        out = Path(json_out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        _print0(f"  wrote -> {out}")
    barrier()
    return result


# ---------------------------------------------------------------------------
# varll -- guided variance + guided per-sample likelihood at ONE cfg
# (one array task per cfg; merge_varll.py collates them into a master CSV)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_varll(
    model, potential, scale_net, eps, config_path, ckpt_path, ckpt_step, *,
    cfg_scale: float, num_batches: int, bsz: int, k_inner: int, k_outer: int,
    alpha_def: float, var_K: int, curvature_probes: int, deltas,
    seed: int, device: torch.device,
    json_out: str = "", csv_out: str = "", persample_csv: str = "",
) -> dict | None:
    """Guided variance + normalized guided per-sample likelihood at cfg = 1 + w.

    NLL is the exactly-normalized density of the guided marginal rho_1^w
    (Prop 2.10), evaluated by the repaired-2.5 nested estimator.  Variance is the
    guided estimator's chi^2 / s_res^2 / K*(delta) / (A6) term.
    """
    if process_index() != 0:
        barrier()
        return None
    if potential is None:
        raise ValueError("`varll` requires a PT-Flow potential (enabled pt: block).")

    from pipelines import build_pipeline

    w = float(cfg_scale) - 1.0
    num_classes = int(getattr(model, "num_classes", 1000))
    rng = torch.Generator(device=device).manual_seed(int(seed))

    # --- guided variance (proposal + weights + curvature all at phi^w) --------
    gp = _GuidedPotential(potential, w)
    labels = torch.arange(int(bsz), device=device).remainder(num_classes).long()
    vrep = variance_report(
        model, gp, scale_net, x0=None, c=labels, eps=eps,
        K=int(var_K), alpha_def=float(alpha_def), cfg_scale=float(cfg_scale),
        pt_w=0.0, deltas=deltas, n_curvature_probes=int(curvature_probes),
        measure_pure_tilt=(alpha_def > 0.0), rng=rng,
    ).as_dict()

    # --- guided likelihood over held-out data ---------------------------------
    config = load_config(config_path)
    loader, preprocess_fn, _ = build_pipeline(config).build_split(batch_size=int(bsz), split="val")
    it = iter(loader)
    rng.manual_seed(int(seed))
    nlls, esss, chi2s, persample = [], [], [], []
    for _ in range(int(num_batches)):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        proc = preprocess_fn(batch)
        x1 = proc["images"].to(device).float()
        c = proc["labels"].to(device).long()
        log_p, info = guided_log_likelihood(
            model, potential, x1, c, eps, w=w,
            K_outer=int(k_outer), K_inner=int(k_inner),
            alpha_def=float(alpha_def), scale_net=scale_net, rng=rng,
        )
        nlls.append(info["nll_per_dim_nats"])
        esss.append(info["inner_ess"])
        chi2s.append(info["inner_chi2"])
        if persample_csv:
            d = float(x1[0].numel())
            for lp in log_p.tolist():
                persample.append({"cfg_scale": cfg_scale, "w": w,
                                  "logp_nats": lp, "nll_per_dim_nats": -lp / d})

    nll = float(np.mean(nlls))
    row = {
        "cfg_scale": float(cfg_scale),
        "w": float(w),
        "eps": float(eps),
        "step": ckpt_step,
        "ckpt": ckpt_path,
        # normalized per-sample likelihood (the headline number)
        "nll_per_dim_nats": nll,
        "bits_per_dim_latent": nll / float(np.log(2.0)),
        # likelihood-estimator health (new-s^2, chi^2 from ESS)
        "ll_inner_ess": float(np.mean(esss)),
        "ll_inner_chi2": float(np.mean(chi2s)),
        "K_outer": int(k_outer),
        "K_inner": int(k_inner),
        # guided variance report (new-s^2)
        "var_chi2": vrep["chi2"],
        "var_rel_rmse": vrep["rel_rmse"],
        "var_ess_frac": vrep["ess_frac"],
        "s_res2": vrep["s_res2"],
        "offdiag_half_fro": vrep.get("offdiag_half_fro"),
        "var_K": int(var_K),
        "alpha_def": float(alpha_def),
        "num_ll_samples": int(num_batches) * int(bsz),
        "estimator": "guided_normalized_likelihood_repaired_section_2p5",
    }
    for k, v in (vrep.get("k_budget") or {}).items():
        row[f"var_K_budget[{k}]"] = v

    _print0(
        f"  cfg={cfg_scale:<4g} (w={w:+.2f})  NLL/dim={nll:+.5f} nats "
        f"({row['bits_per_dim_latent']:+.5f} bits)  ll_ESS={row['ll_inner_ess']:.3f}  "
        f"var_chi2={row['var_chi2']:.3f}  s_res2={row['s_res2']:.4e}"
    )

    if json_out:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    if csv_out:
        Path(csv_out).parent.mkdir(parents=True, exist_ok=True)
        cols = list(row.keys())
        with Path(csv_out).open("w", newline="", encoding="utf-8") as f:
            wtr = csv.DictWriter(f, fieldnames=cols); wtr.writeheader(); wtr.writerow(row)
    if persample_csv and persample:
        Path(persample_csv).parent.mkdir(parents=True, exist_ok=True)
        pcols = list(persample[0].keys())
        with Path(persample_csv).open("w", newline="", encoding="utf-8") as f:
            wtr = csv.DictWriter(f, fieldnames=pcols); wtr.writeheader(); wtr.writerows(persample)
    barrier()
    return row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _floats(s):
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def _ints(s):
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PT-Flow inference/eval with the repaired Section 2.5 (new s^2)."
    )
    sub = p.add_subparsers(dest="mode", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--ckpt", required=True, help="Path to state_*.pt checkpoint.")
    shared.add_argument("--config", required=True, help="Training config YAML.")
    shared.add_argument("--cfg-scale", type=float, default=1.0)
    shared.add_argument("--pt-w", type=float, default=None,
                        help="PT-Flow potential guidance weight; defaults to cfg_scale - 1.")
    shared.add_argument("--seed", type=int, default=0)
    shared.add_argument("--ema-decay", type=float, default=None)
    shared.add_argument("--workdir", default="runs/s_res_var_log")

    # variance ---------------------------------------------------------------
    vp = sub.add_parser("variance", parents=[shared],
                        help="New-s^2 report: chi^2, K*(delta), s_res^2, (A6) term.")
    vp.add_argument("--num-batches", type=int, default=8)
    vp.add_argument("--bsz", type=int, default=16)
    vp.add_argument("--K", type=int, default=64, help="Proposal draws used to estimate chi^2/s_res^2.")
    vp.add_argument("--alpha-def", type=float, default=0.05)
    vp.add_argument("--deltas", type=str, default="0.3,0.1,0.05",
                    help="Target relative RMSEs for the K*(delta) budget.")
    vp.add_argument("--curvature-probes", type=int, default=0,
                    help="Hutchinson HVP probes for the (A6) 1/2||B_x||_F^2 term; 0 = skip.")
    vp.add_argument("--eps-sweep", type=str, default="",
                    help="Comma-separated eps ladder for the variance-reversal sweep, e.g. 0.2,0.1,0.05,0.02,0.01.")
    vp.add_argument("--csv-out", type=str, default="")
    vp.add_argument("--json-out", type=str, default="")

    # likelihood -------------------------------------------------------------
    lp = sub.add_parser("likelihood", parents=[shared],
                        help="NLL ladder (unchanged) + chi^2-based inner budget.")
    lp.add_argument("--num-batches", type=int, default=8)
    lp.add_argument("--bsz", type=int, default=8)
    lp.add_argument("--k-inner", type=int, default=16)
    lp.add_argument("--k-ladder", type=str, default="16,32,64,128")
    lp.add_argument("--alpha-def", type=float, default=0.05)
    lp.add_argument("--json-out", type=str, default="")

    # varll: guided variance + normalized guided likelihood at ONE cfg -------
    xp = sub.add_parser("varll", parents=[shared],
                        help="Guided variance + normalized per-sample likelihood at one --cfg-scale.")
    xp.add_argument("--num-batches", type=int, default=8)
    xp.add_argument("--bsz", type=int, default=8)
    xp.add_argument("--k-inner", type=int, default=16)
    xp.add_argument("--k-outer", type=int, default=16)
    xp.add_argument("--var-K", type=int, default=64, help="Proposal draws for the variance chi^2/s_res^2.")
    xp.add_argument("--alpha-def", type=float, default=0.05)
    xp.add_argument("--deltas", type=str, default="0.3,0.1,0.05")
    xp.add_argument("--curvature-probes", type=int, default=0)
    xp.add_argument("--json-out", type=str, default="")
    xp.add_argument("--csv-out", type=str, default="")
    xp.add_argument("--persample-csv", type=str, default="")

    # evaluate / sample: delegated to inference.py (samples are unchanged) ----
    for name, helptext in (("evaluate", "Generate and compute FID (delegated to inference.py)."),
                           ("sample", "Preview grid (delegated to inference.py).")):
        dp = sub.add_parser(name, parents=[shared], help=helptext)
        dp.add_argument("--sampler", choices=["A", "B", "C"], default="A")
        dp.add_argument("--refine-steps", type=int, default=4)
        dp.add_argument("--refine-gamma", type=float, default=0.5)
        dp.add_argument("--snis-k", type=int, default=32)
        dp.add_argument("--alpha-def", type=float, default=0.05)
        if name == "evaluate":
            dp.add_argument("--num-samples", type=int, default=50000)
            dp.add_argument("--gen-bsz", type=int, default=64)
            dp.add_argument("--fid-ref", type=str, default="")
            dp.add_argument("--json-out", type=str, default="")
            dp.add_argument("--keep-samples", action="store_true")
            dp.add_argument("--eval-backend", choices=["streaming", "png"], default="streaming")
        else:
            dp.add_argument("--class-ids", type=str, default="207,360,387,974,88,979,417,279")
            dp.add_argument("--num-rows", type=int, default=2)
            dp.add_argument("--save-path", type=str, default="")
    return p


def main() -> None:
    init_distributed()
    args = build_parser().parse_args()
    if process_index() == 0:
        os.makedirs(args.workdir, exist_ok=True)
    barrier()

    needs_potential = args.mode in ("variance", "likelihood", "varll") or getattr(args, "sampler", "A") != "A"
    model, postprocess_fn, ckpt_step, device, potential, scale_net, eps = _load_model(
        args.ckpt, args.config, want_potential=needs_potential, ema_decay=args.ema_decay
    )

    if args.mode == "variance":
        run_variance(
            model, potential, scale_net, eps,
            num_batches=args.num_batches, bsz=args.bsz, K=args.K,
            alpha_def=args.alpha_def, cfg_scale=args.cfg_scale, pt_w=args.pt_w,
            deltas=_floats(args.deltas), n_curvature_probes=args.curvature_probes,
            eps_sweep_list=_floats(args.eps_sweep) if args.eps_sweep else None,
            seed=args.seed, device=device,
            csv_out=args.csv_out, json_out=args.json_out,
        )

    elif args.mode == "varll":
        run_varll(
            model, potential, scale_net, eps, args.config, args.ckpt, ckpt_step,
            cfg_scale=args.cfg_scale, num_batches=args.num_batches, bsz=args.bsz,
            k_inner=args.k_inner, k_outer=args.k_outer, alpha_def=args.alpha_def,
            var_K=args.var_K, curvature_probes=args.curvature_probes,
            deltas=_floats(args.deltas), seed=args.seed, device=device,
            json_out=args.json_out, csv_out=args.csv_out, persample_csv=args.persample_csv,
        )

    elif args.mode == "likelihood":
        result = run_likelihood(
            model, potential, scale_net, eps, args.config, args.ckpt, ckpt_step,
            num_batches=args.num_batches, bsz=args.bsz, k_inner=args.k_inner,
            k_ladder=_ints(args.k_ladder), alpha_def=args.alpha_def,
            seed=args.seed, device=device, json_out=args.json_out,
        )
        if result is not None:
            print(json.dumps(result, indent=2))

    elif args.mode in ("evaluate", "sample"):
        sampler = make_sampler(
            model, potential, eps, scale_net=scale_net,
            mode=args.sampler, cfg_scale=args.cfg_scale, pt_w=args.pt_w,
            n_steps=args.refine_steps, gamma=args.refine_gamma,
            snis_K=args.snis_k, alpha_def=args.alpha_def,
        )
        if args.mode == "sample":
            if process_index() == 0:
                class_ids = [int(x) for x in args.class_ids.split(",") if x.strip()]
                save_path = args.save_path or os.path.join(args.workdir, "sample_grid.png")
                run_sample(
                    model, postprocess_fn, class_ids=class_ids, cfg_scale=args.cfg_scale,
                    seed=args.seed, num_rows=args.num_rows, save_path=save_path,
                    device=device, sampler=sampler,
                )
            barrier()
        else:
            if not args.fid_ref:
                args.fid_ref = (CIFAR10_FID_NPZ if model.num_classes == 10 and model.in_channels == 3
                                else IMAGENET_FID_NPZ)
            common = dict(num_samples=args.num_samples, cfg_scale=args.cfg_scale,
                          gen_bsz=args.gen_bsz, fid_ref=args.fid_ref, seed=args.seed,
                          device=device, sampler=sampler, mode=args.sampler)
            if args.eval_backend == "png" or args.keep_samples:
                result = run_eval(model, postprocess_fn, args.ckpt, ckpt_step, args.workdir,
                                  keep_samples=args.keep_samples, **common)
            else:
                result = run_eval_streaming(model, postprocess_fn, args.ckpt, ckpt_step, **common)
            if result is not None:
                result.update(eps=eps, pt_w=args.pt_w)
                print(json.dumps(result, indent=2))
                if args.json_out:
                    out = Path(args.json_out).resolve()
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
