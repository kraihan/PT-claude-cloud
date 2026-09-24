"""CIFAR-10 20-row OFAT study: 10K training and final FID-50K.

The scheduler submits twenty one-GPU tasks together.  Thirteen tasks train a
unique 10K-step configuration; seven sampling-only tasks reuse the exact
central checkpoint so CFG and repeated reference rows do not confound training.
The final ablation.csv contains only: ablation,value,fid.
"""
from __future__ import annotations

import argparse
import copy
import csv
import getpass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

try:
    from . import run_parallel_ablation as parallel
except ImportError:
    import run_parallel_ablation as parallel


six = parallel.six
STEPS = 10_000
FID_SAMPLES = 50_000
OUTPUT_NAME = "cifar10_20way_10k_fid50k"
WANDB_GROUP = OUTPUT_NAME


def tag(value) -> str:
    return str(value).replace(".", "p").replace("->", "to")


def common_arm(template, name):
    arm = copy.deepcopy(template)
    arm["name"] = name
    arm["settings"].update(
        proposal="defensive_diagonal", eps=0.05, K=8,
        lambda_prox_max=0.10, n=0, cfg_scale=1.2,
    )
    cfg = arm["config"]
    cfg["train"].update(
        total_steps=STEPS, save_per_step=STEPS, keep_every=STEPS,
        keep_last=2, eval_per_step=0, eval_at_start=False,
    )
    cfg["optimizer"]["lr_schedule"].update(total_steps=STEPS, warmup_steps=500)
    cfg["pt"]["optimizer"]["lr_schedule"].update(total_steps=STEPS, warmup_steps=500)
    cfg["pt"].update(K=8, scale_mode="learned")
    cfg["pt"]["schedule"].update(
        eps_max=0.20, eps_min=0.05, eps_warmup=500,
        eps_anneal_steps=7500, eps_schedule="cosine",
        alpha_def_start=0.10, alpha_def_end=0.01,
        alpha_def_degraded=0.10,
        prox_warmup=1250, prox_ramp=2500,
        lambda_prox_max=0.10,
    )
    logging = cfg.setdefault("logging", {})
    logging.update(group=WANDB_GROUP, name=f"train/{name}")
    return arm


def entry(factor, value, arm, result, train, cfg_scale=1.2):
    return dict(factor=factor, value=value, arm=arm, result=result,
                train=train, n=0, cfg_scale=float(cfg_scale))


def build_plan(manifest: Path):
    plan = copy.deepcopy(six.build_plan(manifest))
    repo = Path(plan["repo"])
    template = next(arm for arm in plan["arms"] if arm["name"] == "reference")
    plan.update(
        root=str((repo.parent / "runs" / OUTPUT_NAME).resolve()),
        steps=STEPS,
    )
    plan["slurm"].update(
        gpu="a100", constraint="", cpus=8, memory="64G", time="36:00:00",
    )
    plan["evaluation"].update(num_samples=FID_SAMPLES, batch_size=64, seed=1234)

    shared = Path("/scratch") / getpass.getuser() / "ptflow"
    assets, data = shared / "assets", shared / "data"
    plan["environment"].update(
        PTFLOW_ASSETS=str(assets), PTFLOW_DATA=str(data),
        CIFAR10_PATH=str(data / "cifar10"),
        CIFAR10_FID_NPZ=str(assets / "fid_stats/cifar10_train_fid_stats.npz"),
        TORCH_HUB_DIR=str(assets / "torch_hub"),
    )
    if plan.get("wandb", {}).get("enabled"):
        plan["wandb"]["group"] = WANDB_GROUP

    reference = common_arm(template, "reference")
    arms = [reference]

    def add_train(name, mutate):
        arm = common_arm(template, name)
        mutate(arm)
        arms.append(arm)
        return name

    eps_entries = []
    for value in (0.20, 0.10, 0.05, 0.02):
        if value == 0.05:
            name, train = "reference", True
        else:
            name, train = f"epsmin_{tag(value)}", True
            add_train(name, lambda arm, v=value: (
                arm["settings"].update(eps=v),
                arm["config"]["pt"]["schedule"].update(eps_min=v),
            ))
        eps_entries.append(entry("eps_min", value, name, name, train))

    k_entries = []
    for value in (4, 8, 16, 32):
        if value == 8:
            name, train = "reference", False
            result = "K_8"
        else:
            name, train, result = f"K_{value}", True, f"K_{value}"
            add_train(name, lambda arm, v=value: (
                arm["settings"].update(K=v), arm["config"]["pt"].update(K=v),
            ))
        k_entries.append(entry("K", value, name, result, train))

    alpha_entries = []
    for label, start, end in (
        ("0.0", 0.0, 0.0), ("0.01", 0.01, 0.01),
        ("0.1", 0.1, 0.1), ("0.1->0.01", 0.1, 0.01),
    ):
        if label == "0.1->0.01":
            name, result, train = "reference", "alpha_schedule_ours", False
        else:
            name, result, train = f"alpha_def_{tag(label)}", f"alpha_def_{tag(label)}", True
            add_train(name, lambda arm, a=start, b=end: arm["config"]["pt"]["schedule"].update(
                alpha_def_start=a, alpha_def_end=b, alpha_def_degraded=a,
            ))
        alpha_entries.append(entry("alpha_def", label, name, result, train))

    w_entries = []
    for value in (1.0, 1.2, 1.5, 2.0):
        name = f"w_{tag(value)}" + ("_ours" if value == 1.2 else "")
        w_entries.append(entry("w", value, "reference", name, False, value))

    schedule_entries = []
    for value in ("constant", "linear", "cosine", "exp"):
        if value == "cosine":
            name, result, train = "reference", "eps_schedule_cosine", False
        else:
            name, result, train = f"eps_schedule_{value}", f"eps_schedule_{value}", True
            add_train(name, lambda arm, mode=value: arm["config"]["pt"]["schedule"].update(
                eps_schedule=mode,
            ))
        schedule_entries.append(entry("eps_schedule", value, name, result, train))

    plan["arms"] = arms
    plan["stages"] = [
        dict(factor="eps_min", entries=eps_entries),
        dict(factor="K", entries=k_entries),
        dict(factor="alpha_def", entries=alpha_entries),
        dict(factor="w", entries=w_entries),
        dict(factor="eps_schedule", entries=schedule_entries),
    ]
    return plan


def slim_report(plan):
    root = Path(plan["root"])
    rows = parallel.report(plan, sync=True)
    full = root / "report/full_results.csv"
    six.write_csv(full, rows)
    missing = [row["result_id"] for row in rows
               if row.get("status") != "complete" or row.get("fid") is None]
    if missing:
        raise RuntimeError("Cannot write final slim CSV; missing FID: " + ", ".join(missing))
    final = root / "report/ablation.csv"
    temporary = final.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("ablation", "value", "fid"))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(ablation=row["factor"], value=row["value"], fid=row["fid"]))
    temporary.replace(final)
    print(f"Final 20-row CSV: {final}")
    return final


def submit_slim_report(plan):
    root = Path(plan["root"])
    history = json.loads((root / "parallel/submissions.json").read_text())
    reports = [str(row["job_id"]) for row in history if row.get("task") == "pt6p-report"]
    if not reports:
        raise RuntimeError("Parallel report job was not recorded")
    dependency = reports[-1]
    script = root / "parallel/pt20-csv.sh"
    command = [sys.executable, str(Path(__file__).resolve()), "slim-report", str(root / "plan.json")]
    script.write_text("#!/bin/bash\nset -euo pipefail\nexec srun --ntasks=1 " + shlex.join(command) + "\n",
                      encoding="utf-8", newline="\n")
    logs = root / "parallel/logs"
    slurm = plan["slurm"]
    args = [
        "sbatch", "--parsable", "--nodes=1", "--ntasks=1", "--export=ALL",
        f"--partition={slurm['partition']}", "--job-name=pt20-csv",
        "--cpus-per-task=2", "--mem=4G", "--time=00:30:00",
        f"--dependency=afterok:{dependency}",
        f"--output={logs / '%j.out'}", f"--error={logs / '%j.err'}",
    ]
    if slurm.get("account"):
        args.append(f"--account={slurm['account']}")
    environment = {key: value for key, value in os.environ.items() if not key.startswith("SBATCH_")}
    response = subprocess.run(args + [str(script)], env=environment,
                              capture_output=True, text=True, check=True)
    ids = [line.split(";", 1)[0] for line in response.stdout.splitlines()
           if line.split(";", 1)[0].isdigit()]
    if len(ids) != 1:
        raise RuntimeError(f"Cannot parse slim-report submission: {response.stdout}")
    six.dump(root / "slim_report_submission.json",
             dict(job_id=ids[0], dependency=dependency, final_csv=str(root / "report/ablation.csv")))
    print(f"Submitted pt20-csv: {ids[0]} (afterok:{dependency})")


def submit(manifest):
    plan = build_plan(manifest)
    six.preflight(plan)
    root = Path(plan["root"])
    six.materialize(plan, resume=root.exists())
    print("CIFAR-10: 20 GPU tasks, 13 unique 10K training runs, 7 shared-checkpoint evaluations")
    print("Final evaluation: FID-50K; GPU: A100")
    parallel.submit(plan, wait_seconds=0)
    submit_slim_report(plan)


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "submit", "slim-report"))
    parser.add_argument("file", nargs="?", default=str(repo / "configs/ablation/six_h200_4k_wandb.yaml"))
    args = parser.parse_args()
    if args.action == "slim-report":
        plan = json.loads(Path(args.file).read_text())
        slim_report(plan)
        return 0
    plan = build_plan(Path(args.file))
    if args.action == "plan":
        tasks = parallel.tasks(plan)
        print(json.dumps(dict(
            root=plan["root"], dataset=plan["arms"][0]["config"]["pipeline"],
            steps=plan["steps"], fid_samples=plan["evaluation"]["num_samples"],
            rows=sum(len(stage["entries"]) for stage in plan["stages"]),
            gpu_tasks=len(tasks), training_tasks=sum(bool(x["train"]) for x in tasks),
            evaluation_only_tasks=sum(not x["train"] for x in tasks),
            gpu=plan["slurm"]["gpu"],
            csv_columns=["ablation", "value", "fid"],
            final_csv=str(Path(plan["root"]) / "report/ablation.csv"),
        ), indent=2))
    else:
        submit(Path(args.file))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
