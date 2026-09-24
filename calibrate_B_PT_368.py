#!/usr/bin/env python3
"""Isolated proposal-scale calibration at checkpoint 368; generator/potential frozen."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

NAME = "B_PT_scale368"
KS = (16, 64, 128)
MULTIPLIERS = (.25, .5, 1., 2.)  # variance, not standard deviation
ALPHAS = (.1, .75)
FIT_SEED, TEST_SEED = 173683, 273683
WORKER = r'''#!/bin/bash
#SBATCH --job-name=B_PT_scale368
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -Eeuo pipefail
source "$HOME/ptflow_env.sh"
CAL_HOME=$1
FULL_HOME=$2
source "$FULL_HOME/settings.sh"
export PYTHONPATH="$FULL_HOME/PT-Flow${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export DRIFT_COMPILE=0 TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$CAL_HOME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
srun --ntasks=1 python "$CAL_HOME/calibrate_B_PT_368.py" run \
    --full-home "$FULL_HOME" --output "$CAL_HOME" --helper "$CAL_HOME/audit_B_PT_368.py"
'''


def get_helper(path):
    spec = importlib.util.spec_from_file_location("audit368_helper", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    for name in ("validate_source", "load_raw_models", "prefix_statistics", "write_csv", "write_json"):
        if not callable(getattr(helper, name, None)):
            raise ValueError(f"Incompatible audit helper: missing {name}")
    return helper


def make_pairs(gen, pot, scale, count, seed, p_uncond, device, refine_steps, lr):
    import torch
    from ptflow.potential import phi_grad
    from ptflow.recovery import refine_proposal
    rng = torch.Generator().manual_seed(seed)
    labels = torch.randint(gen.num_classes, (count,), generator=rng)
    x = torch.randn((count, gen.input_size, gen.input_size, gen.in_channels), generator=rng)
    is_null = torch.zeros(count, dtype=torch.bool)
    # Prespecified split gives both cohorts representation; record exact counts.
    n_null = max(1, min(count - 1, round(count * p_uncond)))
    is_null[torch.randperm(count, generator=rng)[:n_null]] = True
    cond = labels.clone()
    cond[is_null] = pot.uncond_index
    result = dict(x=x.to(device), labels=labels.to(device), cond=cond.to(device), is_null=is_null)
    means, generated, residuals, distances = [], [], [], []
    grng = torch.Generator(device=device).manual_seed(seed + 1)
    for i in range(0, count, 2):
        xb, cb = result["x"][i:i+2], result["cond"][i:i+2]
        with torch.no_grad():
            m = gen(c=result["labels"][i:i+2], cfg_scale=1., train=False, deterministic=True,
                    x0=xb, rng=grng)["samples"].float()
        mean, _ = refine_proposal(pot, xb, m, cb, steps=refine_steps, lr=lr)
        g, _ = phi_grad(pot, mean, cb, create_graph=False)
        if not torch.isfinite(mean).all() or not torch.isfinite(g).all():
            raise FloatingPointError("Nonfinite refined mean/gradient")
        generated.append(m.detach())
        means.append(mean.detach())
        residuals.append((g + mean - xb).flatten(1).square().mean(1).sqrt().detach())
        distances.append((mean - m).flatten(1).square().mean(1).sqrt().detach())
        if (i+2) % 16 == 0 or i+2 >= count:
            print(f"prepare seed={seed} pairs={min(i+2,count)}/{count}", flush=True)
    result.update(mean=torch.cat(means), generator_mean=torch.cat(generated),
                  residual_rms=torch.cat(residuals), distance_rms=torch.cat(distances))
    return result


def reverse_kl_loss(pot, x, mean, cond, log_scale, eps, *, K=8, rng=None, chunk=8):
    """KL(q || target), up to constants, per dimension; paired reparameterization.

    q=N(mean,2*eps*diag(exp(log_scale))). Potential/mean are fixed. Only
    log_scale receives gradients. No importance weights or clipping in this fit.
    """
    import torch
    if K < 2 or K % 2 or eps <= 0:
        raise ValueError("Positive epsilon and an even K>=2 are required")
    if any(p.requires_grad for p in pot.parameters()):
        raise ValueError("Potential must be frozen")
    mean, x = mean.detach(), x.detach()
    b, shape = len(x), x.shape[1:]
    z = torch.randn((b, K//2, *shape), generator=rng, device=x.device, dtype=x.dtype)
    z = torch.cat((z, -z), dim=1)
    y = mean[:, None] + math.sqrt(2*eps) * torch.exp(.5*log_scale)[:, None] * z
    flat, labels = y.reshape(b*K, *shape), cond.repeat_interleave(K)
    phi = torch.cat([pot.phi(flat[i:i+chunk], labels[i:i+chunk]) for i in range(0, b*K, chunk)]).view(b, K)
    quadratic = .5 * (y-x[:, None]).flatten(2).square().sum(2)
    # Normalization by 2*eps*d rescales the training code's objective by a
    # positive constant. The scale has its own optimizer and no lambda_scale coefficient.
    d = x[0].numel()
    return (((quadratic + phi).mean(1)/(2*eps)) - .5*log_scale.flatten(1).sum(1)).mean()/d


def fit_scale(pot, scale, pairs, eps, *, steps=500, lr=3e-4, batch=4, K=8, logger=None):
    import torch
    device = pairs["x"].device
    scale.train().requires_grad_(True)
    optimizer = torch.optim.AdamW(scale.parameters(), lr=lr, weight_decay=0.)
    pick_rng = torch.Generator().manual_seed(373683)
    rng = torch.Generator(device=device).manual_seed(473683)
    trace, started = [], time.perf_counter()
    for step in range(1, steps+1):
        idx = torch.randint(len(pairs["x"]), (batch,), generator=pick_rng).to(device)
        x, mean, c = (pairs[k][idx] for k in ("x", "mean", "cond"))
        optimizer.zero_grad(set_to_none=True)
        s = scale(mean, c)
        loss = reverse_kl_loss(pot, x, mean, c, s, eps, K=K, rng=rng)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite calibration loss at {step}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(scale.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == steps:
            row = dict(step=step, fit_reverse_kl_up_to_constant_per_dim=float(loss.detach()),
                       grad_norm=float(norm), log_scale_mean=float(s.detach().mean()),
                       log_scale_std=float(s.detach().std()), elapsed_s=time.perf_counter()-started)
            trace.append(row)
            print("SCALE_FIT " + json.dumps(row, allow_nan=False), flush=True)
            if logger:
                logger(trace)
    optimizer.zero_grad(set_to_none=True)
    scale.eval().requires_grad_(False)
    return trace


def aggregate(rows):
    groups = {}
    for r in rows:
        key = (r["stage"], r["variance_multiplier"], r["alpha_def"], r["K"])
        for cohort in ("all", "null" if r["is_null"] else "conditional"):
            groups.setdefault((*key, cohort), []).append(r)
    result = []
    for (stage, multiplier, alpha, k, cohort), group in sorted(groups.items()):
        pair_values = {}
        for r in group:
            pair_values.setdefault(r["pair"], []).append(r["control_ess"])
        # SE across independent input pairs, not across repeats treated as pairs.
        per_pair = [statistics.mean(v) for v in pair_values.values()]
        se = statistics.stdev(per_pair)/math.sqrt(len(per_pair)) if len(per_pair) > 1 else None
        result.append(dict(stage=stage, variance_multiplier=multiplier, alpha_def=alpha, K=k,
            cohort=cohort, pairs=len(pair_values), proposal_evaluations=len(group),
            control_ess_mean=statistics.mean(per_pair), control_ess_pair_se=se,
            control_ess_median=statistics.median(r["control_ess"] for r in group),
            ess_draws_mean=statistics.mean(r["ess_draws"] for r in group),
            near_one_draw_fraction=statistics.mean(r["ess_draws"] < 1.1 for r in group),
            max_weight_mean=statistics.mean(r["max_weight"] for r in group),
            logw_std_mean=statistics.mean(r["logw_std"] for r in group),
            log_scale_mean=statistics.mean(r["log_scale_mean"] for r in group),
            scale_bound_fraction=statistics.mean(r["scale_bound_fraction"] for r in group)))
    return result


def evaluate(pot, scale, pairs, eps, stage, helper, rows, out, repeats=3):
    import torch
    from ptflow.estimator import tilted_phi0
    device = pairs["x"].device
    for i in range(0, len(pairs["x"]), 2):
        x, mean, c = (pairs[k][i:i+2] for k in ("x", "mean", "cond"))
        with torch.no_grad():
            learned = scale(mean, c).detach().float()
            bound = (learned.abs() >= .98*scale.scale_max).flatten(1).float().mean(1)
            for multiplier in MULTIPLIERS:
                s = learned + math.log(multiplier)
                for alpha in ALPHAS:
                    for repeat in range(repeats):
                        rng = torch.Generator(device=device).manual_seed(573683 + i*101 + repeat)
                        estimate = tilted_phi0(pot, x, c, mean, s, eps, K=max(KS), alpha_def=alpha,
                                               generator=rng, antithetic=True, chunk=8, logw_clip=0.)
                        stats = helper.prefix_statistics(estimate, eps, ks=KS)
                        for k, stat in stats.items():
                            for j in range(len(x)):
                                row = dict(stage=stage, variance_multiplier=multiplier, alpha_def=alpha,
                                           K=k, pair=i+j, repeat=repeat, is_null=bool(pairs["is_null"][i+j]))
                                row.update({name: float(v[j]) for name,v in stat.items() if name != "phi0"})
                                row.update(log_scale_mean=float(s[j].mean()), scale_bound_fraction=float(bound[j]))
                                rows.append(row)
                        del estimate
        if (i+2) % 16 == 0 or i+2 >= len(pairs["x"]):
            helper.write_csv(out/"diagnostics.csv", aggregate(rows))
            print(f"EVAL {stage}: {min(i+2,len(pairs['x']))}/{len(pairs['x'])} pairs", flush=True)
    helper.write_csv(out/"per_sample.csv", rows)


def decision(summary, healthy=.3):
    # Fixed endpoint and unit multiplier are prespecified. Other multipliers
    # are exploratory; selecting their best result must not imply validation.
    checks = [r for r in summary if r["stage"] == "after" and r["variance_multiplier"] == 1.
              and r["alpha_def"] == .1 and r["K"] in (min(KS), max(KS))
              and r["cohort"] in ("conditional", "null")]
    passed = len(checks) == 4 and all(r["control_ess_mean"] >= healthy and
                                    r["near_one_draw_fraction"] <= .1 for r in checks)
    return dict(candidate_for_extended_pilot=passed, primary_checks=checks,
        next_action=("Review a bounded training pilot beyond step 368; no automatic restart."
                     if passed else "Do not restart 30K. Scale-only calibration did not meet the prespecified screen."))


def run_calibration(full_home, out, helper, *, device_name="cuda", fit_pairs=128,
                    test_pairs=64, fit_steps=500, refine_steps=64, repeats=3):
    import torch
    import yaml
    if (out/"manifest.json").exists():
        raise FileExistsError("Refusing to overwrite an existing calibration; submit a fresh job")
    paths = helper.validate_source(full_home)
    sys.path.insert(0, str(paths["repo"]))
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Run on the allocated GPU, not a login node")
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(paths["config"].read_text())
    before_stat = paths["checkpoint"].stat()
    gen, pot, scale, sched, saved_schedule = helper.load_raw_models(cfg, paths["checkpoint"], device)
    versions = [[p._version for p in module.parameters()] for module in (gen, pot)]
    scale_versions = [p._version for p in scale.parameters()]
    eps = sched.eps()
    started = time.perf_counter()
    manifest = dict(checkpoint=str(paths["checkpoint"]), step=368, eps=eps, cfg_scale=1.,
        fit_pairs=fit_pairs, test_pairs=test_pairs, fit_steps=fit_steps, fit_K=8, fit_batch=4,
        scale_lr=3e-4, scale_weight_decay=0., refine_steps=refine_steps, repeats=repeats,
        fit_seed=FIT_SEED, test_seed=TEST_SEED, K=list(KS), variance_multipliers=list(MULTIPLIERS),
        alpha_def=list(ALPHAS), schedule_state=saved_schedule,
        config_sha256=hashlib.sha256(paths["config"].read_bytes()).hexdigest(),
        torch_version=str(torch.__version__),
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU test",
        primary_protocol="Compare before/after at alpha=.1, variance multiplier=1, K16 and K128; "
                         "same held-out inputs and proposal random streams. No test-set tuning or early stopping.",
        limitations="Scale-only diagnostic. Generator/potential and epsilon frozen. Refined mean uses 64 steps "
                    "by default; this is not one-step generation. ESS is not proof of coverage, likelihood "
                    "accuracy, FID improvement, or long-run stability. Variance multiplier grid is exploratory.")
    helper.write_json(out/"manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    fit = make_pairs(gen, pot, scale, fit_pairs, FIT_SEED, float(cfg["pt"].get("p_uncond",.1)),
                     device, refine_steps, sched.proposal_refine_lr)
    test = make_pairs(gen, pot, scale, test_pairs, TEST_SEED, float(cfg["pt"].get("p_uncond",.1)),
                      device, refine_steps, sched.proposal_refine_lr)
    torch.save({name: {k:v.cpu() for k,v in pairs.items()} for name,pairs in (("fit",fit),("test",test))}, out/"inputs.pt")
    rows = []
    evaluate(pot, scale, test, eps, "before", helper, rows, out, repeats)
    fit_scale(pot, scale, fit, eps, steps=fit_steps,
              logger=lambda trace: helper.write_csv(out/"fit_trace.csv",trace))
    # Save only the experimental scale: this is deliberately not a resumable
    # PT-Flow checkpoint and must not overwrite the trained checkpoint.
    torch.save(dict(proposal_scale_state_dict={k:v.detach().cpu() for k,v in scale.state_dict().items()},
                    source_checkpoint=str(paths["checkpoint"]), source_step=368, protocol=manifest), out/"scale_candidate.pt")
    evaluate(pot, scale, test, eps, "after", helper, rows, out, repeats)
    stat = paths["checkpoint"].stat()
    if (before_stat.st_size,before_stat.st_mtime_ns) != (stat.st_size,stat.st_mtime_ns):
        raise RuntimeError("Source checkpoint metadata changed during calibration")
    if versions != [[p._version for p in module.parameters()] for module in (gen,pot)]:
        raise RuntimeError("Generator/potential changed during scale-only calibration")
    if any(p.grad is not None for module in (gen,pot) for p in module.parameters()):
        raise RuntimeError("Unexpected generator/potential parameter gradient")
    if sched.state_dict() != saved_schedule:
        raise RuntimeError("Schedule changed during calibration")
    changed = scale_versions != [p._version for p in scale.parameters()]
    if not changed:
        raise RuntimeError("Scale optimizer did not update any parameters")
    summary = aggregate(rows)
    primary = [r for r in summary if r["variance_multiplier"] == 1. and r["alpha_def"] == .1
               and r["cohort"] == "all"]
    result = dict(completed=True, elapsed_s=time.perf_counter()-started, generator_unchanged=True,
        potential_unchanged=True, source_checkpoint_unchanged=True, schedule_unchanged=True,
        scale_updated=True, fit_null_pairs=int(fit["is_null"].sum()), test_null_pairs=int(test["is_null"].sum()),
        heldout_pairs=test_pairs, primary_before_after=primary, **decision(summary),
        exploratory_top_after=sorted([r for r in summary if r["stage"]=="after" and r["cohort"]=="all"],
                                     key=lambda r:r["control_ess_mean"],reverse=True)[:5],
        caveat="Diagnostic only. No automatic training submission; good ESS alone does not prove coverage.")
    helper.write_json(out/"summary.json", result)
    (out/"COMPLETED").write_text("Scale-only calibration completed. No full training restart.\n")
    print(json.dumps(result,indent=2),flush=True)
    print(f"Report: {out/'diagnostics.csv'}",flush=True)


def main():
    user = os.environ.get("USER","mdraihk")
    work = Path("/scratch")/user/"ptflow"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("submit","run"))
    parser.add_argument("--full-home",type=Path,default=work/"active_pt/B_PT_Full_20260924_030649_u9mui_x5/job")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--helper",type=Path)
    args=parser.parse_args()
    if args.helper is None:
        pointer=work/"diagnostics/latest_B_PT_audit368.txt"
        args.helper=Path(pointer.read_text().strip())/"audit_B_PT_368.py"
    helper=get_helper(args.helper)
    helper.validate_source(args.full_home)
    if args.command=="run":
        if args.output is None:
            parser.error("run requires --output")
        run_calibration(args.full_home,args.output,helper)
        return 0
    existing=subprocess.check_output(["squeue","-h","-u",user,"-n",NAME,"-o","%i"],text=True).strip()
    if existing:
        raise RuntimeError(f"Calibration already queued/running: {existing}; refusing duplicate")
    parent=work/"diagnostics"
    parent.mkdir(parents=True,exist_ok=True)
    stamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out=Path(tempfile.mkdtemp(prefix=f"{NAME}_{stamp}_",dir=parent))
    (out/"logs").mkdir()
    shutil.copy2(Path(__file__),out/"calibrate_B_PT_368.py")
    shutil.copy2(args.helper,out/"audit_B_PT_368.py")
    (out/"worker.sbatch").write_text(WORKER,encoding="utf-8",newline="\n")
    helper.write_json(out/"submission.json",dict(full_home=str(args.full_home),helper_source=str(args.helper),
        helper_sha256=hashlib.sha256(args.helper.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    subprocess.run(["bash","-n",str(out/"worker.sbatch")],check=True)
    response=subprocess.check_output(["sbatch","--parsable",f"--chdir={out}",
        f"--output={out/'logs/scale_%j.out'}",f"--error={out/'logs/scale_%j.err'}",
        str(out/"worker.sbatch"),str(out),str(args.full_home)],text=True).strip()
    job=response.split(";",1)[0]
    if not re.fullmatch(r"\d+",job):
        raise RuntimeError(f"Unexpected sbatch response: {response}; check queue before retrying")
    (out/"job_id.txt").write_text(job+"\n")
    (parent/"latest_B_PT_scale368.txt").write_text(str(out)+"\n")
    print(f"Submitted {job}: one H200; 500 scale-only updates; generator/potential frozen.\nFolder: {out}")
    print(f"tail -F {out}/logs/scale_{job}.out {out}/logs/scale_{job}.err")
    print(f"cat {out}/summary.json")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
