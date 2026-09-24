"""CPU subset preparation and a separate GPU smoke test for ImageNet ablations."""
import argparse
import copy
import json
import os
from pathlib import Path
import time

import yaml

try:
    from . import run_six_ablation as six
    from . import run_parallel_ablation as parallel
    from .make_latent_subset import create_subset
except ImportError:
    import run_six_ablation as six
    import run_parallel_ablation as parallel
    from make_latent_subset import create_subset


def prepare(manifest, resolved):
    repo = Path(__file__).resolve().parents[1]
    spec = yaml.safe_load(Path(manifest).read_text())
    defaults = six.build_plan(manifest)["environment"]
    # Honor the existing ImageNet run's asset paths from ~/ptflow_env.sh.
    assets = Path(os.environ.get("PTFLOW_ASSETS", defaults["PTFLOW_ASSETS"])).resolve()
    data = Path(os.environ.get("PTFLOW_DATA", defaults["PTFLOW_DATA"])).resolve()
    source = Path(os.environ.get("IMAGENET_SOURCE_CACHE", os.environ.get("IMAGENET_CACHE_PATH", data / "latents"))).resolve()
    subset = Path(os.environ.get("PTFLOW_SUBSET_CACHE", data / "latents_subset50_seed42")).resolve()
    environment = dict(PTFLOW_ASSETS=str(assets), PTFLOW_DATA=str(data), IMAGENET_CACHE_PATH=str(subset))
    for key, default in dict(IMAGENET_FID_NPZ=assets / "fid_stats/jit_in256_stats.npz",
                             VAE_HF_PATH=assets / "sdvae", HF_ROOT=assets / "mae",
                             TORCH_HUB_DIR=assets / "torch_hub").items():
        environment[key] = str(Path(os.environ.get(key, default)).resolve())
    spec["environment"] = environment
    settings = spec["overrides"]["dataset"]
    metadata = create_subset(source, subset, settings["subset_per_class"], settings["subset_seed"])
    # Freeze provenance into every training config and therefore every CSV row.
    settings["subset_indices_sha256"] = metadata["indices_sha256"]
    settings["subset_manifest"] = str(subset / "subset.json")
    resolved = Path(resolved).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    candidate = resolved.with_suffix(".candidate.yaml")
    candidate.write_text(yaml.safe_dump(spec, sort_keys=False))
    plan = six.build_plan(candidate, repo)
    six.preflight(plan)
    if resolved.exists() and yaml.safe_load(resolved.read_text()) != spec:
        raise ValueError(f"Resolved manifest changed: use a new setup/output folder, not {resolved}")
    candidate.replace(resolved)
    print(f"Prepared manifest: {resolved}", flush=True)
    six.describe(plan)


def smoke_plan(manifest, root):
    plan = six.build_plan(manifest)
    plan = copy.deepcopy(plan)
    plan.update(root=str(Path(root).resolve()), steps=20, init_ema_checkpoint="", wandb={"enabled": False})
    reference = plan["arms"][0]
    reference["settings"].update(K=16, lambda_prox_max=0.2, cfg_scale=2.0)
    cfg = reference["config"]
    cfg["logging"].update(use_wandb=False, log_every_k=1)
    cfg["train"].update(total_steps=20, save_per_step=20, keep_every=20)
    cfg["optimizer"]["lr_schedule"].update(total_steps=20, warmup_steps=0)
    cfg["pt"]["optimizer"]["lr_schedule"].update(total_steps=20, warmup_steps=0)
    cfg["pt"]["K"] = 16
    cfg["pt"]["schedule"].update(prox_warmup=0, prox_ramp=1, lambda_prox_max=0.2)
    plan["arms"] = [reference]
    first = dict(factor="proposal", value="defensive_diagonal", arm="reference", result="reference",
                 train=True, n=0, cfg_scale=2.0)
    second = dict(factor="n", value=4, arm="reference", result="smoke_n4", train=False, n=4, cfg_scale=2.0)
    plan["stages"] = [dict(factor="smoke", entries=[first, second])]
    plan["evaluation"]["num_samples"] = 32  # Pipeline check, never a scientific FID estimate.
    return plan


def smoke(manifest, setup):
    info = parallel.hardware()
    if info["gpu_memory_gib"] < 38 or info["compute_capability"][0] < 8:
        raise RuntimeError(f"Expected an Ampere-or-newer GPU with at least 38 GiB: {info}")
    tag = os.environ.get("SLURM_JOB_ID", str(time.time_ns()))
    plan = smoke_plan(manifest, Path(setup) / f"smoke_{tag}")
    six.preflight(plan)
    six.materialize(plan)
    for index in range(2):
        if parallel.worker(plan, index):
            raise RuntimeError(f"GPU smoke check failed. Inspect {plan['root']}/logs; full sweep was NOT submitted.")
    six.dump(Path(setup) / "smoke_passed.json", dict(hardware=info, smoke_root=plan["root"],
                                                   description="20-step K=16 + Mode A/B(4) check; 32-sample FIDs are NOT ablation results"))
    print("GPU smoke test passed; full 4k ablation submission may proceed.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "smoke"))
    parser.add_argument("manifest")
    parser.add_argument("--setup-dir", required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.manifest, Path(args.setup_dir) / "manifest.yaml")
    else:
        smoke(args.manifest, args.setup_dir)


if __name__ == "__main__":
    main()
