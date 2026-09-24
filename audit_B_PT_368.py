#!/usr/bin/env python3
"""Fixed-weight proposal diagnosis for B_PT_Full checkpoint 368; no training."""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
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

NAME = "B_PT_audit368"
KS = (16, 64, 128)
REFINEMENTS = (0, 8, 32, 64)
ALPHAS = (.10, .50, .75)
COVARIANCES = ("learned", "identity")

WORKER = r'''#!/bin/bash
#SBATCH --job-name=B_PT_audit368
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -Eeuo pipefail
source "$HOME/ptflow_env.sh"
AUDIT_HOME=$1
FULL_HOME=$2
source "$FULL_HOME/settings.sh"
REPO="$FULL_HOME/PT-Flow"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export DRIFT_COMPILE=0 TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$AUDIT_HOME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
srun --ntasks=1 python "$AUDIT_HOME/audit_B_PT_368.py" run \
    --full-home "$FULL_HOME" --output "$AUDIT_HOME" --pairs "$3" --repeats "$4"
'''


def write_json(path, obj):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def input_paths(full_home):
    return dict(config=full_home / "full.yaml", repo=full_home / "PT-Flow",
                checkpoint=full_home / "B_PT_Full/checkpoints/state_00000368.pt")


def validate_source(full_home):
    paths = input_paths(full_home)
    required = [paths["config"], paths["checkpoint"], full_home / "settings.sh",
                paths["repo"] / "ptflow/recovery.py", paths["repo"] / "ptflow/estimator.py"]
    for p in required:
        if not p.is_file():
            raise FileNotFoundError(p)
    return paths


def load_raw_models(config, checkpoint, device):
    import torch
    from models.generator import DitGen
    from ptflow.potential import PotentialNet, ScaleNet
    from ptflow.schedule import build_schedule
    from utils.ckpt_util import canonical_state_dict, check_model_behavior

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload["step"]) != 368:
        raise ValueError("Expected step 368 checkpoint")
    for key in ("model", "pt_model", "pt_scale_model", "pt_schedule"):
        if payload.get(key) is None:
            raise ValueError(f"Missing raw training state: {key}")
    mc = dict(config["model"])
    mc.setdefault("num_classes", int(config["dataset"].get("num_classes", 1000)))
    generator = DitGen(**mc)
    check_model_behavior(generator, payload)
    generator.load_state_dict(canonical_state_dict(payload["model"]), strict=True)
    common = dict(num_classes=generator.num_classes, input_size=generator.input_size,
                  in_channels=generator.in_channels, cond_dim=generator.cond_dim)
    pc, sc = dict(config["pt"]["model"]), dict(config["pt"]["scale_model"])
    for k, v in common.items():
        pc.setdefault(k, v)
        sc.setdefault(k, v)
    potential, scale = PotentialNet(**pc), ScaleNet(**sc)
    potential.load_state_dict(payload["pt_model"], strict=True)
    scale.load_state_dict(payload["pt_scale_model"], strict=True)
    sched = build_schedule(config["pt"]["schedule"])
    sched.load_state_dict(payload["pt_schedule"])
    if sched.step != 368:
        raise ValueError("Checkpoint and PT schedule steps disagree")
    saved_schedule = dict(payload["pt_schedule"])
    del payload  # discard optimizer/EMA tensors on CPU; never instantiate optimizers
    for module in (generator, potential, scale):
        module.to(device).eval().requires_grad_(False)
    return generator, potential, scale, sched, saved_schedule


def prefix_statistics(estimate, eps, ks=KS):
    """Use nested antithetic pairs; identical target and samples across K.

    sample_proposal orders points [z_1..z_H, -z_1..-z_H]. Reordering them into
    pairs before taking prefixes preserves antithetic sampling for every even K.
    phi0 prefixes undo the centering used for the original full-budget estimate.
    """
    import torch
    maximum = estimate.log_w.shape[1]
    if maximum % 2 or any(k % 2 or k > maximum for k in ks):
        raise ValueError("Nested antithetic budgets must be even and <= maximum")
    half = maximum // 2
    order = torch.stack((torch.arange(half, device=estimate.log_w.device),
                         torch.arange(half, maximum, device=estimate.log_w.device)), dim=1).flatten()
    phi_ref = estimate.phi_y.detach().mean(dim=1).double()
    result = {}
    for k in ks:
        w = estimate.log_w[:, order[:k]].double()
        if not torch.isfinite(w).all():
            raise FloatingPointError("Nonfinite log weights")
        lse = torch.logsumexp(w, dim=1)
        ess = torch.exp(2 * lse - torch.logsumexp(2 * w, dim=1)).clamp(1, k)
        result[k] = dict(ess_draws=ess, ess_fraction=ess / k,
            control_ess=((ess - 1) / (k - 1)).clamp(0, 1),
            max_weight=torch.softmax(w, dim=1).max(dim=1).values,
            logw_std=w.std(dim=1, correction=0),
            phi0=phi_ref - 2 * eps * (lse - math.log(k)))
    return result


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row["refine_steps"], row["covariance"], row["alpha_def"], row["K"])
        # Preserve conditioning split, in addition to the overall aggregate.
        for cohort in ("all", "null" if row["is_null"] else "conditional"):
            groups.setdefault((*key, cohort), []).append(row)
    output = []
    for (refine, cov, alpha, k, cohort), group in sorted(groups.items()):
        def vals(name):
            return [r[name] for r in group]
        means_by_pair = {}
        for r in group:
            means_by_pair.setdefault(r["pair"], []).append(r["phi0_per_dim"])
        output.append(dict(refine_steps=refine, covariance=cov, alpha_def=alpha, K=k,
            cohort=cohort, pairs=len(means_by_pair), proposal_evaluations=len(group),
            ess_draws_mean=statistics.mean(vals("ess_draws")),
            ess_fraction_mean=statistics.mean(vals("ess_fraction")),
            control_ess_mean=statistics.mean(vals("control_ess")),
            control_ess_median=statistics.median(vals("control_ess")),
            near_one_draw_fraction=sum(v < 1.1 for v in vals("ess_draws")) / len(group),
            max_weight_mean=statistics.mean(vals("max_weight")),
            logw_std_mean=statistics.mean(vals("logw_std")),
            phi0_per_dim_mean=statistics.mean(vals("phi0_per_dim")),
            phi0_repeat_std_per_dim=statistics.mean(statistics.pstdev(v) for v in means_by_pair.values()),
            residual_rms_mean=statistics.mean(vals("residual_rms")),
            residual_rel_mean=statistics.mean(vals("residual_rel")),
            distance_from_generator_rms_mean=statistics.mean(vals("distance_from_generator_rms")),
            energy_per_dim_mean=statistics.mean(vals("energy_per_dim")),
            log_scale_mean=statistics.mean(vals("log_scale_mean"))))
    return output


def write_csv(path, rows):
    if not rows:
        return
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def run_audit(full_home, out, pairs=32, repeats=3, device_name="cuda", micro=2):
    import torch
    import yaml
    paths = validate_source(full_home)
    sys.path.insert(0, str(paths["repo"]))
    from ptflow.estimator import tilted_phi0
    from ptflow.potential import phi_grad
    from ptflow.recovery import refine_proposal, energy
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Submit the audit to a GPU node; do not run it on the login node")
    out.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(paths["config"].read_text())
    initial_stat = paths["checkpoint"].stat()
    models = load_raw_models(config, paths["checkpoint"], device)
    gen, pot, scale, sched, schedule_state = models
    eps = sched.eps()
    shape = (gen.input_size, gen.input_size, gen.in_channels)
    d = math.prod(shape)
    versions = [[p._version for p in m.parameters()] for m in (gen, pot, scale)]
    rng = torch.Generator(device="cpu").manual_seed(846368)
    labels = torch.randint(0, gen.num_classes, (pairs,), generator=rng)
    x = torch.randn((pairs, *shape), generator=rng).to(device)
    is_null = torch.rand((pairs,), generator=rng) < float(config["pt"].get("p_uncond", .1))
    c = labels.clone()
    c[is_null] = pot.uncond_index
    labels, c = labels.to(device), c.to(device)
    generation_rng = torch.Generator(device=device).manual_seed(846369)
    with torch.no_grad():
        m = torch.cat([gen(c=labels[i:i+micro], cfg_scale=1., train=False,
            deterministic=True, x0=x[i:i+micro], rng=generation_rng)["samples"].float()
            for i in range(0, pairs, micro)])
    torch.save(dict(x0=x.cpu(), generator_mean=m.cpu(), labels=labels.cpu(),
                    potential_labels=c.cpu(), is_null=is_null), out / "probe_inputs.pt")
    manifest = dict(checkpoint=str(paths["checkpoint"]), checkpoint_step=368,
        checkpoint_bytes=initial_stat.st_size, checkpoint_mtime_ns=initial_stat.st_mtime_ns,
        weight_keys=["model", "pt_model", "pt_scale_model"], schedule_state=schedule_state,
        eps=eps, training_alpha_def=sched.alpha_def(), cfg_scale=1., potential_w=0.,
        pairs=pairs, repeats=repeats, null_pairs=int(is_null.sum()),
        K=list(KS), refinement_steps=list(REFINEMENTS), alpha_def=list(ALPHAS),
        covariance=list(COVARIANCES), seed=846368, code_source=str(paths["repo"]),
        config_sha256=hashlib.sha256(paths["config"].read_bytes()).hexdigest(),
        torch_version=torch.__version__, device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU test",
        residual_rel_definition="norm(grad_phi(mean)+mean-x0) / norm(generator_mean-x0); "
                                "the original generator displacement is the denominator for every refinement depth",
        limitations="Fixed raw step-368 weights; fresh held-out noise, not exact training-batch replay. "
                     "Screening proposal diagnostics only: not FID, a likelihood table, or proof of coverage. "
                     "Repeated draws on the same input pairs are not independent training seeds. "
                     "EMA weights and optimizer state are not used. Epsilon and model weights never change.")
    write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    rows = []
    started = time.perf_counter()
    case_count = 0
    for steps in REFINEMENTS:
        for i in range(0, pairs, micro):
            xb, mb, cb = x[i:i+micro], m[i:i+micro], c[i:i+micro]
            if steps:
                mean, _ = refine_proposal(pot, xb, mb, cb, steps=steps,
                                         lr=float(sched.proposal_refine_lr))
            else:
                mean = mb
            g, _ = phi_grad(pot, mean, cb, create_graph=False)
            residual = g + mean - xb
            rms = residual.flatten(1).square().mean(dim=1).sqrt()
            relative = residual.flatten(1).norm(dim=1) / (mb - xb).flatten(1).norm(dim=1).clamp_min(1e-6)
            distance = (mean - mb).flatten(1).square().mean(dim=1).sqrt()
            with torch.no_grad():
                en = energy(pot, mean, xb, cb) / d
                learned = scale(mean, cb).detach().float()
            if not all(torch.isfinite(v).all() for v in (mean, g, en, learned)):
                raise FloatingPointError(f"Nonfinite mean/gradient/scale at refinement {steps}")
            for cov in COVARIANCES:
                s = learned if cov == "learned" else None
                log_scale = learned.flatten(1).mean(dim=1) if s is not None else torch.zeros(len(xb), device=device)
                for alpha in ALPHAS:
                    for repeat in range(repeats):
                        # Same random stream across mean/covariance/alpha cases;
                        # nested antithetic prefixes are shared across budgets K.
                        proposal_rng = torch.Generator(device=device).manual_seed(946368 + i * 101 + repeat)
                        with torch.no_grad():
                            est = tilted_phi0(pot, xb, cb, mean, s, eps,
                                K=max(KS), alpha_def=alpha, generator=proposal_rng,
                                logw_clip=0., antithetic=True, chunk=8)
                            statistics_by_k = prefix_statistics(est, eps)
                        for k, stats in statistics_by_k.items():
                            for j in range(len(xb)):
                                row = dict(refine_steps=steps, covariance=cov, alpha_def=alpha,
                                    K=k, pair=i+j, repeat=repeat, is_null=bool(is_null[i+j]),
                                    eps=eps, label=int(labels[i+j]))
                                row.update({name: float(value[j].cpu()) for name, value in stats.items()})
                                row["phi0_per_dim"] = row.pop("phi0") / d
                                row.update(residual_rms=float(rms[j].detach().cpu()),
                                    residual_rel=float(relative[j].detach().cpu()),
                                    distance_from_generator_rms=float(distance[j].cpu()),
                                    energy_per_dim=float(en[j].cpu()), log_scale_mean=float(log_scale[j].cpu()))
                                rows.append(row)
                        del est
            case_count += 1
            print(f"refinement={steps} input_pairs={i+len(xb)}/{pairs}; "
                  f"elapsed={time.perf_counter()-started:.1f}s", flush=True)
        write_csv(out / "per_sample.csv", rows)
        write_csv(out / "diagnostics.csv", summarize(rows))
    final_stat = paths["checkpoint"].stat()
    if (initial_stat.st_size, initial_stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
        raise RuntimeError("Checkpoint metadata changed during audit; review provenance")
    if versions != [[p._version for p in model.parameters()] for model in (gen, pot, scale)]:
        raise RuntimeError("Model parameters changed during fixed-weight audit")
    if sched.state_dict() != schedule_state:
        raise RuntimeError("Schedule state changed during audit")
    summary = summarize(rows)
    all_rows = [r for r in summary if r["cohort"] == "all"]
    baseline = next(r for r in all_rows if r["refine_steps"] == 8 and r["covariance"] == "learned"
                    and r["alpha_def"] == .75 and r["K"] == 16)
    write_json(out / "summary.json", dict(completed=True, checkpoint_unchanged=True,
        model_parameters_unchanged=True, schedule_unchanged=True,
        cases=len(all_rows), per_sample_rows=len(rows), elapsed_s=time.perf_counter()-started,
        training_like_baseline=baseline,
        screening_top_rows=sorted(all_rows, key=lambda r: r["control_ess_mean"], reverse=True)[:8],
        decision="DIAGNOSTIC_ONLY: no automatic training restart; high ESS does not prove target coverage."))
    (out / "COMPLETED").write_text("Diagnostic finished; no training performed.\n")
    print(f"Diagnostic complete: {out / 'diagnostics.csv'}", flush=True)


def main():
    user = os.environ.get("USER", "mdraihk")
    work = Path("/scratch") / user / "ptflow"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["submit", "run"])
    parser.add_argument("--full-home", type=Path,
        default=work / "active_pt/B_PT_Full_20260924_030649_u9mui_x5/job")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.pairs < 1 or args.repeats < 1:
        parser.error("pairs and repeats must be positive")
    if args.command == "run":
        if args.output is None:
            parser.error("run requires --output")
        run_audit(args.full_home, args.output, args.pairs, args.repeats)
        return 0
    validate_source(args.full_home)
    existing = subprocess.check_output(["squeue", "-h", "-u", user, "-n", NAME, "-o", "%i"], text=True).strip()
    if existing:
        raise RuntimeError(f"Audit already queued/running: {existing}; refusing duplicate")
    parent = work / "diagnostics"
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(tempfile.mkdtemp(prefix=f"{NAME}_{stamp}_", dir=parent))
    (out / "logs").mkdir()
    shutil.copy2(Path(__file__), out / "audit_B_PT_368.py")
    (out / "worker.sbatch").write_text(WORKER, encoding="utf-8", newline="\n")
    subprocess.run(["bash", "-n", str(out / "worker.sbatch")], check=True)
    response = subprocess.check_output(["sbatch", "--parsable", f"--chdir={out}",
        f"--output={out / 'logs/audit_%j.out'}", f"--error={out / 'logs/audit_%j.err'}",
        str(out / "worker.sbatch"), str(out), str(args.full_home),
        str(args.pairs), str(args.repeats)], text=True).strip()
    job = response.split(";", 1)[0]
    if not re.fullmatch(r"\d+", job):
        raise RuntimeError(f"Unexpected sbatch response: {response}; check queue before retrying")
    (out / "job_id.txt").write_text(job + "\n")
    (parent / "latest_B_PT_audit368.txt").write_text(str(out) + "\n")
    print(f"Submitted diagnostic {job}: one H200, two-hour limit, no training\nFolder: {out}")
    print(f"tail -F {out}/logs/audit_{job}.out {out}/logs/audit_{job}.err")
    print(f"Reports: {out}/diagnostics.csv and {out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
