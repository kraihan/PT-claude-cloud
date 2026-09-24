#!/usr/bin/env python3
"""Create/submit an isolated 30K extension of the completed cosine pilot.

Run on Palmetto after sourcing ~/ptflow_env.sh. No production files are edited.
Training starts at zero from W-Flow EMA weights. Requeues resume only this run.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import yaml

NAME = "B_PT_Full"
STEPS = 30000
PILOT_JOB = "16199241"
IGNORE = shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "*.pyc",
    "assets", "data", "runs", "checkpoints", "params_ema", "wandb", "log", "logs")

WORKER = r'''#!/bin/bash
#SBATCH --job-name=B_PT_Full
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --signal=B:USR1@1200
#SBATCH --requeue
#SBATCH --open-mode=append

set -Eeuo pipefail
source "$HOME/ptflow_env.sh"
FULL_HOME=$1
source "$FULL_HOME/settings.sh"
REPO="$FULL_HOME/PT-Flow"
RUN="$FULL_HOME/B_PT_Full"
CFG="$FULL_HOME/full.yaml"
HELPER="$FULL_HOME/submit_B_PT_Full.py"
FINAL="$RUN/checkpoints/state_00030000.pt"
export PT_ACTIVE_REPO="$REPO" PT_ACTIVE_RUN="$RUN" PT_ACTIVE_CFG="$CFG"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export DRIFT_COMPILE=0 TORCHDYNAMO_DISABLE=1 WANDB_RESUME=allow
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$REPO"
mkdir -p "$RUN" "$FULL_HOME/report"
test ! -f "$RUN/PT_FAILED.txt" || { cat "$RUN/PT_FAILED.txt"; exit 1; }
rm -f "$RUN/requeue.request" "$RUN/requeue.ready"
REQUESTED=0
request_checkpoint() {
    REQUESTED=1
    echo "Wall-time warning: save at next optimizer boundary, then requeue"
    touch "$RUN/requeue.request"
}
trap request_checkpoint USR1
trap 'CODE=$?; python "$HELPER" report --home "$FULL_HOME" || true; exit "$CODE"' EXIT

# Background srun lets Bash process the advance wall-time signal immediately.
run_command() {
    "$@" &
    CHILD=$!
    while true; do
        if wait "$CHILD"; then CODE=0; else CODE=$?; fi
        if kill -0 "$CHILD" 2>/dev/null; then continue; fi
        return "$CODE"
    done
}
requeue_if_requested() {
    if (( REQUESTED )) || [[ -s "$RUN/requeue.ready" ]]; then
        test ! -f "$RUN/PT_FAILED.txt" || { cat "$RUN/PT_FAILED.txt"; exit 1; }
        echo "Requeueing $SLURM_JOB_ID; checkpoints and finished FID results retained"
        scontrol requeue "$SLURM_JOB_ID"
        exit 0
    fi
}
evaluate() {
    CKPT=$1
    TAG=$2
    EXPECTED_STEP=$3
    RESULT="$FULL_HOME/report/$TAG.json"
    if python "$HELPER" valid-result --path "$RESULT" --step "$EXPECTED_STEP"; then
        echo "Reuse completed evaluation: $TAG"
    else
        run_command srun --ntasks=1 torchrun --standalone --nproc_per_node=2 inference.py evaluate \
            --config "$CFG" --ckpt "$CKPT" --sampler A --cfg-scale 1.2 \
            --num-samples 50000 --seed 42 --gen-bsz 32 --eval-backend streaming \
            --workdir "$FULL_HOME/eval_$TAG" --json-out "$RESULT"
        python "$HELPER" valid-result --path "$RESULT" --step "$EXPECTED_STEP"
    fi
    python "$HELPER" report --home "$FULL_HOME"
    requeue_if_requested
}

echo "Job=$SLURM_JOB_ID Name=B_PT_Full Home=$FULL_HOME"
echo "30K fresh updates; global batch=1024; saves every 1000; 2 H200 GPUs"
echo "Full prox max=0.02 after warmup20/ramp80; cosine eps 0.1->0.05 after warmup20/anneal20000"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
INIT=$(cat "$FULL_HOME/initializer.txt")
evaluate "$INIT" initializer_ema 200000

if [[ ! -s "$FINAL" ]]; then
    # Trainer detects local checkpoints and restores all model/optimizer/PT/EMA
    # state. Without a local checkpoint it uses initializer weights at step 0.
    run_command srun --ntasks=1 torchrun --standalone --nproc_per_node=2 train.py \
        --config "$CFG" --workdir "$RUN"
    requeue_if_requested
fi
test -s "$FINAL" || { echo "Missing final checkpoint; refusing completion"; exit 1; }
evaluate "$FINAL" final_ema 30000
run_command srun --ntasks=1 python "$HELPER" raw-export --path "$FINAL" \
    --output "$FULL_HOME/report/raw_generator_for_eval.pt"
requeue_if_requested
evaluate "$FULL_HOME/report/raw_generator_for_eval.pt" final_raw 30000
python "$HELPER" report --home "$FULL_HOME" --require-complete
touch "$FULL_HOME/COMPLETED"
echo "B_PT_Full finished. FID-50K: $FULL_HOME/report/fid.csv"
'''


def dump_json(path: Path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def configure(base):
    cfg = copy.deepcopy(base)
    assert cfg["pipeline"] == "imagenet_latent", "Expected ImageNet latent pilot"
    assert cfg["pt"]["schedule"]["policy"] == "active_recovery_v1"
    train, pt = cfg["train"], cfg["pt"]
    assert train["total_steps"] == 200, "Expected the completed 200-step pilot config"
    assert train["train_batch_size"] == 16 and train["grad_accum_steps"] == 4
    assert train["forward_dict"]["gen_per_label"] == 64
    assert pt["K"] == 16 and pt["enabled"] and pt["prox_mode"] == "full"
    assert pt["schedule"]["eps_schedule"] == "cosine"
    assert pt["schedule"]["eps_min"] == .05
    for key in ("resume_from", "init_from", "init_generator_from"):
        train.pop(key, None)
    assert Path(train["init_ema_from"]).name == "state_00200000.pt"
    train.update(total_steps=STEPS, save_per_step=1000, keep_every=1000, keep_last=2,
                 eval_per_step=0, eval_at_start=False, benchmark_steps=0,
                 profile_every=100, push_at_resume=20)
    # Preserve pilot learning rates and warmups; extend their recorded horizon.
    cfg["optimizer"]["lr_schedule"]["total_steps"] = STEPS
    pt["optimizer"]["lr_schedule"]["total_steps"] = STEPS
    pt["schedule"].update(eps_max=.1, eps_min=.05, eps_schedule="cosine",
        eps_warmup=20, eps_anneal_steps=20000,
        prox_warmup=20, prox_ramp=80, lambda_prox_max=.02,
        alignment_weight=.1, alignment_hold=1000, alignment_end=5000)
    # Retain the existing failure thresholds; never relabel broken ESS as healthy.
    cfg.setdefault("logging", {}).update(name=NAME, project="ptflow-active-diagnostic",
                                        use_wandb=True, log_every_k=20, allow_resume=True)
    return cfg


def locate_pilot(work):
    matches = [p.parent for p in (work / "active_pt").glob("B_epscos200_*/job_id.txt")
               if p.read_text().strip() == PILOT_JOB]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one pilot directory for {PILOT_JOB}; found {matches}")
    return matches[0]


def prepare(pilot: Path, home: Path):
    source = Path((pilot / "source.txt").read_text().strip())
    repo = source / "PT-Flow"
    base = yaml.safe_load((pilot / "pilot.yaml").read_text())
    cfg = configure(base)
    initializer = Path(cfg["train"]["init_ema_from"])
    required = [initializer, source / "settings.sh", pilot / "run/checkpoints/state_00000200.pt",
                repo / "train.py", repo / "inference.py", repo / "ptflow/run_guard.py"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    assert "active_recovery_v1" in (repo / "ptflow/schedule.py").read_text()
    assert "Existing checkpoints: resuming full state" in (repo / "train.py").read_text()
    assert "--eval-backend" in (repo / "inference.py").read_text()
    home.mkdir(exist_ok=False)
    # Copy only the code snapshot; all dataset/model assets retain existing paths.
    shutil.copytree(repo, home / "PT-Flow", ignore=IGNORE)
    shutil.copy2(source / "settings.sh", home / "settings.sh")
    shutil.copy2(Path(__file__), home / "submit_B_PT_Full.py")
    for folder in ("logs", "report", NAME):
        (home / folder).mkdir()
    (home / "full.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (home / "initializer.txt").write_text(str(initializer) + "\n")
    (home / "worker.sbatch").write_text(WORKER, encoding="utf-8", newline="\n")
    files = sorted(p for p in (home / "PT-Flow").rglob("*") if p.is_file())
    hashes = {str(p.relative_to(home / "PT-Flow")): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in files}
    manifest = dict(name=NAME, source_pilot_job=PILOT_JOB, source_pilot=str(pilot),
        code_source=str(repo), code_hashes=hashes, initializer=str(initializer),
        init_kind="W-Flow EMA generator weights only; new step=0; fresh optimizer/PT state",
        run=str(home / NAME), config=str(home / "full.yaml"), steps=STEPS,
        save_every=1000, preserve_every=1000, global_generated_batch=1024, gpus="h200:2",
        eps_schedule=cfg["pt"]["schedule"], generator_lr=cfg["optimizer"]["lr_schedule"],
        potential_lr=cfg["pt"]["optimizer"]["lr_schedule"], fid_samples=50000,
        fid_cfg=1.2, fid_seed=42, walltime="72:00:00",
        note="User-authorized experimental full run; pilots did not establish FID gain. "
             "Full-state requeue resumes this run, not the W-Flow source checkpoint. "
             "Existing nonfinite/refined-estimator failure stops retained. "
             "Generator proposal ESS is separately logged; refined health does not certify it.")
    dump_json(home / "manifest.json", manifest)
    return manifest


def result_valid(path, step):
    try:
        r = json.loads(Path(path).read_text())
        return (r["step"] == step and r["num_samples"] == 50000 and r["mode"] == "A"
                and r["cfg_scale"] == 1.2 and r["seed"] == 42 and r["world_size"] == 2
                and r["gen_bsz"] == 32 and r["backend"] == "streaming"
                and math.isfinite(float(r["fid"])))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def report(home, require_complete=False):
    folder = home / "report"
    folder.mkdir(exist_ok=True)
    rows, signature = [], None
    for tag, step in (("initializer_ema", 200000), ("final_ema", STEPS), ("final_raw", STEPS)):
        path = folder / f"{tag}.json"
        if not path.exists():
            continue
        if not result_valid(path, step):
            if require_complete:
                raise ValueError(f"Invalid or partial evaluation: {path}")
            continue
        result = json.loads(path.read_text())
        sig = (result["fid_ref"], result["feature_extractor"])
        if signature is not None and sig != signature:
            raise ValueError("FID reference or extractor differs across evaluations")
        signature = sig
        rows.append(dict(model=tag, new_steps=0 if tag == "initializer_ema" else STEPS,
                         fid_50k=result["fid"], cfg_scale=1.2, seed=42))
    tmp = folder / "fid.csv.tmp"
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "new_steps", "fid_50k", "cfg_scale", "seed"])
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(folder / "fid.csv")
    checkpoint = home / NAME / "checkpoints/state_00030000.pt"
    status = home / NAME / "pt_status.json"
    failure = home / NAME / "PT_FAILED.txt"
    summary = dict(run=NAME, full_checkpoint=checkpoint.is_file(), fid_rows=len(rows),
                   completed=checkpoint.is_file() and len(rows) == 3 and not failure.exists(),
                   pt_failure=failure.read_text() if failure.exists() else None)
    if status.exists():
        summary["pt_status"] = json.loads(status.read_text())
    dump_json(folder / "summary.json", summary)
    if require_complete and not summary["completed"]:
        raise RuntimeError("Incomplete full run; refusing successful completion")
    print(f"Report: {folder / 'fid.csv'} ({len(rows)}/3 FID results)")
    return summary


def raw_export(path, output):
    import torch
    src = torch.load(path, map_location="cpu", weights_only=False)
    if int(src["step"]) != STEPS:
        raise ValueError("Expected final 30K checkpoint")
    tmp = output.with_suffix(".tmp")
    torch.save(dict(step=src["step"], ema_model=src["model"],
                    model_behavior=src.get("model_behavior", {}),
                    weights_kind="RAW_GENERATOR_EVALUATION_ONLY"), tmp)
    tmp.replace(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["submit", "prepare", "report", "valid-result", "raw-export"])
    p.add_argument("--work", type=Path, default=Path("/scratch") / os.environ.get("USER", "mdraihk") / "ptflow")
    p.add_argument("--home", type=Path)
    p.add_argument("--path", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--step", type=int)
    p.add_argument("--require-complete", action="store_true")
    a = p.parse_args()
    if a.command == "valid-result":
        return 0 if result_valid(a.path, a.step) else 1
    if a.command == "raw-export":
        raw_export(a.path, a.output)
        return 0
    if a.command == "report":
        report(a.home, a.require_complete)
        return 0
    if a.command == "submit":
        jobs = subprocess.check_output(["squeue", "-h", "-u", os.environ["USER"], "-n", NAME, "-o", "%i"], text=True).strip()
        if jobs:
            raise RuntimeError(f"{NAME} already queued/running: {jobs}; refusing duplicate")
    pilot = locate_pilot(a.work)
    import datetime
    parent = a.work / "active_pt"
    # Reserve a unique container; installer requires a not-yet-existing child.
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    container = Path(tempfile.mkdtemp(prefix=f"B_PT_Full_{stamp}_", dir=parent))
    home = container / "job"
    prepare(pilot, home)
    subprocess.run(["bash", "-n", str(home / "worker.sbatch")], check=True)
    print(f"Prepared: {home}", flush=True)
    if a.command == "prepare":
        return 0
    response = subprocess.check_output(["sbatch", "--parsable", f"--chdir={home / 'PT-Flow'}",
        f"--output={home / 'logs/B_PT_Full_%j.out'}", f"--error={home / 'logs/B_PT_Full_%j.err'}",
        str(home / "worker.sbatch"), str(home)], text=True).strip()
    job = response.split(";", 1)[0]
    if not re.fullmatch(r"\d+", job):
        raise RuntimeError(f"Unexpected sbatch response: {response!r}; inspect queue before retry")
    (home / "job_id.txt").write_text(job + "\n")
    (parent / "latest_B_PT_Full.txt").write_text(str(home) + "\n")
    print(f"Submitted B_PT_Full job {job}\nFolder: {home}\nRun: {home / NAME}")
    print(f"tail -F {home}/logs/B_PT_Full_{job}.out {home}/logs/B_PT_Full_{job}.err")
    print(f"Reports: {home}/report/fid.csv and summary.json")
    subprocess.run(["squeue", "-j", job, "-o", "%.12i %.28j %.3t %.12M %R"], check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
