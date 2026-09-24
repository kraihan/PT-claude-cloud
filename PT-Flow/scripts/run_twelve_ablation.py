"""Twelve-task ImageNet screening: epsilon schedule, CFG w, and alpha_def.

Seven tasks train real 1K-step models; five tasks evaluate the shared "ours"
checkpoint.  Every task performs one final 50K-image FID evaluation.
"""
from __future__ import annotations

import argparse
import copy
import getpass
import json
from pathlib import Path

try:
    from . import run_parallel_ablation as parallel
except ImportError:
    import run_parallel_ablation as parallel


six = parallel.six
EPS_SCHEDULES = ("constant", "linear", "cosine", "exp")
W_VALUES = (1.0, 1.2, 1.5, 2.0)
ALPHA_VARIANTS = (
    ("0.0", 0.0, 0.0),
    ("0.01", 0.01, 0.01),
    ("0.1", 0.1, 0.1),
    ("0.1->0.01", 0.1, 0.01),
)
MINI_STEPS = 1000
OUTPUT_NAME = "ablation_imagenet_B_subset50_schedule_w_alpha_1k_fid50k"
WANDB_GROUP = "imagenet_B_subset50_schedule_w_alpha_1k_fid50k"


def tag(value) -> str:
    return str(value).replace(".", "p").replace("->", "to")


def configure_common(arm, name):
    arm["name"] = name
    schedule = arm["config"]["pt"]["schedule"]
    schedule.update(
        eps_max=0.20, eps_min=0.05, eps_warmup=50,
        eps_anneal_steps=750, eps_schedule="cosine",
        alpha_def_start=0.10, alpha_def_end=0.01,
        alpha_def_degraded=0.10, prox_warmup=125, prox_ramp=250,
    )
    train = arm["config"]["train"]
    train.update(total_steps=MINI_STEPS, save_per_step=MINI_STEPS,
                 keep_every=MINI_STEPS, keep_last=2)
    arm["config"]["optimizer"]["lr_schedule"].update(
        total_steps=MINI_STEPS, warmup_steps=50,
    )
    arm["config"]["pt"]["optimizer"]["lr_schedule"].update(
        total_steps=MINI_STEPS, warmup_steps=50,
    )
    logging = arm["config"].setdefault("logging", {})
    logging["group"] = WANDB_GROUP
    logging["name"] = f"train/{name}"
    return arm


def build_plan(manifest: Path):
    plan = copy.deepcopy(six.build_plan(manifest))
    repo = Path(plan["repo"])
    template = next(arm for arm in plan["arms"] if arm["name"] == "reference")
    plan["steps"] = MINI_STEPS
    plan["root"] = str((repo.parent / "runs" / OUTPUT_NAME).resolve())
    plan["slurm"].update(
        gpu="a100", constraint="", cpus=8, memory="64G", time="24:00:00",
    )
    plan["evaluation"].update(num_samples=50000, batch_size=8)
    if plan.get("wandb", {}).get("enabled"):
        plan["wandb"]["group"] = WANDB_GROUP

    # The shared reference is the claimed method: cosine epsilon, CFG w=1.2,
    # alpha_def 0.1 -> 0.01.  Evaluation-only arms depend on this checkpoint.
    reference = configure_common(copy.deepcopy(template), "reference")
    reference["settings"].update(eps=0.05, cfg_scale=1.2)
    arms = [reference]

    eps_entries = []
    for mode in EPS_SCHEDULES:
        if mode == "cosine":
            name, arm, train = "reference", reference, True
        else:
            name, train = f"eps_schedule_{mode}", True
            arm = configure_common(copy.deepcopy(template), name)
            arm["settings"]["eps"] = 0.05
            arm["config"]["pt"]["schedule"]["eps_schedule"] = mode
            arms.append(arm)
        eps_entries.append(dict(
            factor="eps_schedule", value=mode, arm=name, result=name,
            train=train, n=0, cfg_scale=1.2,
        ))

    w_entries = []
    for value in W_VALUES:
        name = f"w_{tag(value)}" + ("_ours" if value == 1.2 else "")
        w_entries.append(dict(
            factor="w", value=value, arm="reference", result=name,
            train=False, n=0, cfg_scale=value,
        ))

    alpha_entries = []
    for label, start, end in ALPHA_VARIANTS:
        if label == "0.1->0.01":
            name, arm, train = "alpha_schedule_ours", reference, False
        else:
            name, train = f"alpha_def_{tag(label)}", True
            arm = configure_common(copy.deepcopy(template), name)
            arm["config"]["pt"]["schedule"].update(
                alpha_def_start=start, alpha_def_end=end,
                alpha_def_degraded=start,
            )
            arms.append(arm)
        alpha_entries.append(dict(
            factor="alpha_def", value=label, arm=arm["name"], result=name,
            train=train, n=0, cfg_scale=1.2,
        ))

    plan["arms"] = arms
    plan["stages"] = [
        dict(factor="eps_schedule", entries=eps_entries),
        dict(factor="w", entries=w_entries),
        dict(factor="alpha_def", entries=alpha_entries),
    ]
    return plan


def default_manifest(repo: Path) -> Path:
    sibling = repo.parent.with_name("ptflow-ablation")
    return sibling / "runs/ablation_imagenet_B_subset50_a100_4k_setup/manifest.yaml"


def submit(manifest: Path):
    plan = build_plan(manifest)
    six.preflight(plan)
    root = Path(plan["root"])
    six.materialize(plan, resume=root.exists())
    print("12 final FID-50K tasks: 7 train+eval, 5 evaluation-only")
    print(f"Training: 7 x {MINI_STEPS} real train.py steps; GPU: A100")
    print(f"Final CSV: {root / 'report/ablation.csv'}")
    parallel.submit(plan, wait_seconds=0)


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "submit", "report"))
    parser.add_argument("--manifest", default=str(default_manifest(repo)))
    args = parser.parse_args()
    plan = build_plan(Path(args.manifest))
    if args.action == "plan":
        tasks = parallel.tasks(plan)
        print(json.dumps({
            "root": plan["root"], "steps": plan["steps"], "gpu": "a100",
            "fid_samples": plan["evaluation"]["num_samples"],
            "tasks": len(tasks), "training_tasks": sum(bool(x["train"]) for x in tasks),
            "evaluation_only_tasks": sum(not x["train"] for x in tasks),
            "eps_schedule": list(EPS_SCHEDULES), "w": list(W_VALUES),
            "alpha_def": [x[0] for x in ALPHA_VARIANTS],
            "final_csv": str(Path(plan["root"]) / "report/ablation.csv"),
        }, indent=2))
    elif args.action == "submit":
        submit(Path(args.manifest))
    else:
        parallel.report(plan, sync=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
