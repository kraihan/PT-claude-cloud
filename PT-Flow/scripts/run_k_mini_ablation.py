"""Run four real ImageNet PT-Flow jobs with K in {2, 4, 8, 16}.

This is intentionally a small wrapper around the existing training/evaluation
launchers.  It does not run a smoke test and it does not modify train.py.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

try:
    from . import run_parallel_ablation as parallel
except ImportError:
    import run_parallel_ablation as parallel


six = parallel.six
K_VALUES = (2, 4, 8, 16)
OUTPUT_NAME = "ablation_imagenet_B_subset50_K_4k"
WANDB_GROUP = "imagenet_B_subset50_K_4k"


def build_plan(manifest: Path, gpu: str = "a100"):
    if gpu not in {"a100", "h100", "h200"}:
        raise ValueError("--gpu must be a100, h100, or h200")
    plan = copy.deepcopy(six.build_plan(manifest))
    repo = Path(plan["repo"])
    reference = next(arm for arm in plan["arms"] if arm["name"] == "reference")
    plan["root"] = str((repo.parent / "runs" / OUTPUT_NAME).resolve())
    plan["slurm"].update(
        gpu=gpu,
        constraint="",
        cpus=8,
        memory="64G",
        time="48:00:00",
    )
    if plan.get("wandb", {}).get("enabled"):
        plan["wandb"]["group"] = WANDB_GROUP

    arms, entries = [], []
    for value in K_VALUES:
        arm = copy.deepcopy(reference)
        arm["name"] = f"K_{value}"
        arm["settings"]["K"] = value
        arm["config"]["pt"]["K"] = value
        logging = arm["config"].setdefault("logging", {})
        if plan.get("wandb", {}).get("enabled"):
            logging["group"] = WANDB_GROUP
            logging["name"] = f"train/K_{value}"
        arms.append(arm)
        entries.append(dict(
            factor="K", value=value, arm=arm["name"], result=arm["name"],
            train=True, n=int(plan["reference"]["n"]),
            cfg_scale=float(plan["reference"]["cfg_scale"]),
        ))
    plan["arms"] = arms
    plan["stages"] = [dict(factor="K", entries=entries)]
    return plan


def submit(manifest: Path, gpu: str):
    plan = build_plan(manifest, gpu)
    six.preflight(plan)
    root = Path(plan["root"])
    six.materialize(plan, resume=root.exists())
    print(f"Real train.py runs: {len(K_VALUES)} x {plan['steps']} steps")
    print("K values: " + ", ".join(map(str, K_VALUES)))
    print(f"Final FID CSV: {root / 'report/ablation.csv'}")
    parallel.submit(plan, wait_seconds=0)


def main():
    repo = Path(__file__).resolve().parents[1]
    default_manifest = repo.parent / "runs/ablation_imagenet_B_subset50_a100_4k_setup/manifest.yaml"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "submit", "report"))
    parser.add_argument("--manifest", default=str(default_manifest))
    parser.add_argument("--gpu", choices=("a100", "h100", "h200"), default="a100",
                        help="Use one modern GPU type for all four comparable runs")
    args = parser.parse_args()
    plan = build_plan(Path(args.manifest), args.gpu)
    if args.action == "plan":
        print(json.dumps({
            "root": plan["root"], "steps": plan["steps"], "K": list(K_VALUES),
            "jobs": len(parallel.tasks(plan)), "gpu": plan["slurm"]["gpu"],
            "final_csv": str(Path(plan["root"]) / "report/ablation.csv"),
        }, indent=2))
    elif args.action == "submit":
        submit(Path(args.manifest), args.gpu)
    else:
        parallel.report(plan, sync=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
