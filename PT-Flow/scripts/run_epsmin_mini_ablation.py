"""Four real ImageNet epsilon-minimum runs with final 50K FID.

The prepared ImageNet B/2 4K manifest supplies every setting except the
epsilon endpoint.  No smoke run and no core-code modification are performed.
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
EPS_MIN_VALUES = (0.20, 0.10, 0.05, 0.02)
OUTPUT_NAME = "ablation_imagenet_B_subset50_epsmin_4k_fid50k"
WANDB_GROUP = "imagenet_B_subset50_epsmin_4k_fid50k"
EPS_MAX = 0.20
EPS_WARMUP = 200
EPS_ANNEAL_STEPS = 3000


def result_name(value: float) -> str:
    tag = format(value, ".2f").replace(".", "p")
    return f"epsmin_{tag}" + ("_ours" if value == 0.05 else "")


def build_plan(manifest: Path):
    plan = copy.deepcopy(six.build_plan(manifest))
    repo = Path(plan["repo"])
    reference = next(arm for arm in plan["arms"] if arm["name"] == "reference")
    plan["root"] = str((repo.parent / "runs" / OUTPUT_NAME).resolve())
    plan["slurm"].update(
        gpu="a100", constraint="", cpus=8, memory="64G", time="48:00:00",
    )
    plan["evaluation"].update(num_samples=50000, batch_size=8)
    if plan.get("wandb", {}).get("enabled"):
        plan["wandb"]["group"] = WANDB_GROUP

    arms, entries = [], []
    for value in EPS_MIN_VALUES:
        name = result_name(value)
        arm = copy.deepcopy(reference)
        arm["name"] = name
        # The legacy report field is called eps; factor/value state explicitly
        # that this experiment changes eps_min only.
        arm["settings"]["eps"] = value
        schedule = arm["config"]["pt"]["schedule"]
        schedule.update(
            eps_max=EPS_MAX,
            eps_min=value,
            eps_warmup=EPS_WARMUP,
            eps_anneal_steps=EPS_ANNEAL_STEPS,
        )
        logging = arm["config"].setdefault("logging", {})
        if plan.get("wandb", {}).get("enabled"):
            logging["group"] = WANDB_GROUP
            logging["name"] = f"train/{name}"
        arms.append(arm)
        entries.append(dict(
            factor="eps_min", value=value, arm=name, result=name, train=True,
            n=int(plan["reference"]["n"]),
            cfg_scale=float(plan["reference"]["cfg_scale"]),
        ))
    plan["arms"] = arms
    plan["stages"] = [dict(factor="eps_min", entries=entries)]
    return plan


def submit(manifest: Path):
    plan = build_plan(manifest)
    six.preflight(plan)
    root = Path(plan["root"])
    six.materialize(plan, resume=root.exists())
    print(f"Real train.py runs: {len(EPS_MIN_VALUES)} x {plan['steps']} steps")
    print("eps_min values: " + ", ".join(map(str, EPS_MIN_VALUES)))
    print(f"Final FID samples per run: {plan['evaluation']['num_samples']}")
    print(f"Final CSV: {root / 'report/ablation.csv'}")
    parallel.submit(plan, wait_seconds=0)


def main():
    repo = Path(__file__).resolve().parents[1]
    default_manifest = repo.parent / "runs/ablation_imagenet_B_subset50_a100_4k_setup/manifest.yaml"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "submit", "report"))
    parser.add_argument("--manifest", default=str(default_manifest))
    args = parser.parse_args()
    plan = build_plan(Path(args.manifest))
    if args.action == "plan":
        print(json.dumps({
            "root": plan["root"], "steps": plan["steps"],
            "eps_min": list(EPS_MIN_VALUES), "eps_max_fixed": EPS_MAX,
            "eps_warmup": EPS_WARMUP, "eps_anneal_steps": EPS_ANNEAL_STEPS,
            "jobs": len(parallel.tasks(plan)), "gpu": plan["slurm"]["gpu"],
            "fid_samples": plan["evaluation"]["num_samples"],
            "final_csv": str(Path(plan["root"]) / "report/ablation.csv"),
        }, indent=2))
    elif args.action == "submit":
        submit(Path(args.manifest))
    else:
        parallel.report(plan, sync=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
