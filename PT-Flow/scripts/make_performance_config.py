"""Generate measured-tradeoff candidates from the ACTUAL saved run config.

The generator architecture and residual convention are never replaced. Recipes
other than preserve deliberately change the objective/particle budget. They are
experimental candidates, not claims of measured speed or FID improvement.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import yaml


def make_config(base, recipe="balanced", world_size=8, steps=None, benchmark=0,
                finetune=False, compile_generator=False, keep_pt=False):
    if recipe not in ("preserve", "quality", "balanced", "speed"):
        raise ValueError(recipe)
    if world_size < 1 or benchmark < 0 or (steps is not None and steps < 1):
        raise ValueError("world_size/steps must be positive; benchmark must be nonnegative")
    cfg = copy.deepcopy(base)
    ds, train, model = cfg["dataset"], cfg["train"], cfg["model"]
    cifar = cfg.get("pipeline") == "cifar10_pixel" or ds.get("name") == "cifar10"
    for key in ("resume_from", "init_ema_from", "init_from", "benchmark_steps"):
        train.pop(key, None)
    cfg.setdefault("logging", {}).update(use_wandb=False, log_every_k=20)
    cfg["optimizer"]["fused"] = "auto"
    train["compile_generator"] = bool(compile_generator)
    train["profile_every"] = 100
    train.setdefault("ot_kwargs", {})["reuse_costs"] = True
    cfg.setdefault("feature", {}).update(channels_last=True)
    # preserve keeps particle counts, precision, learning rates and objectives.
    if recipe != "preserve":
        model.update(precision="auto", use_bf16=True, attn_fp32=False)
        cfg.setdefault("pt", {})["enabled"] = bool(keep_pt)
        # Offline evaluation prevents frequent multi-CFG 50k sweeps from
        # blocking the training allocation.
        train.update(eval_per_step=0, save_per_step=5000, keep_every=10000,
                     keep_last=2, push_at_resume=20, diverse_noise=True)
        train["ot_mode"] = "debiased"
        ot = train["ot_kwargs"]
        ot.update(use_new_cfg=True, use_quadratic_cost=True, reuse_costs=True)
        if recipe in ("balanced", "speed"):
            labels = 16 if cifar else 64
            labels = max(world_size, math.ceil(labels / world_size) * world_size)
            particles = 32 if recipe == "balanced" else 16
            train.update(train_batch_size=labels, pos_per_sample=32 if recipe == "speed" else 64,
                         neg_per_sample=16 if recipe == "speed" else 32)
            train["forward_dict"]["gen_per_label"] = particles
            # Limit generator microbatch to 64 without changing global samples.
            train["grad_accum_steps"] = max(1, math.ceil(labels // world_size * particles / 64))
            model["use_remat"] = False
            ot.update(resample_neg=recipe == "balanced", disable_diag_mask=recipe == "balanced",
                      resample_gen_per_label=particles)
        if recipe == "quality":
            # Preserve all ImageNet feature losses and its two-batch estimator.
            ot.update(resample_neg=True, disable_diag_mask=True)
            ot.pop("resample_gen_per_label", None)
            if cifar:
                labels = max(world_size, math.ceil(16 / world_size) * world_size)
                train.update(train_batch_size=labels, pos_per_sample=64, neg_per_sample=32)
                train["forward_dict"]["gen_per_label"] = 64
                train["grad_accum_steps"] = max(1, math.ceil(labels // world_size * 64 / 64))
        train["feature_chunk_size"] = 64 if cifar else 128
        cfg["feature"].update(chunk_size=64 if cifar else 128, checkpoint=False)
        if cifar:
            # Keep the trained extractor family; test higher resolution in the
            # quality recipe, alongside extra particles and slower EMA.
            size = 224 if recipe == "quality" else 128
            train.setdefault("activation_kwargs", {}).setdefault("convnext_kwargs", {})["image_size"] = size
            train["feature_cache_gib"] = 2.0
            train["extra_ema_decays"] = [0.999, 0.9999]
            ot["sinkhorn_num_iter"] = 10 if recipe == "quality" else 3
        else:
            train["feature_cache_gib"] = 0.0  # Full MAE spatial features are too large.
            if recipe in ("balanced", "speed"):
                train["activation_kwargs"]["every_k_block"] = float("inf")
            if recipe == "speed":
                train["activation_kwargs"].update(patch_mean_size=[], patch_std_size=[])
            ot["sinkhorn_num_iter"] = 1
        ds["kwargs"]["num_workers"] = min(4, int(ds["kwargs"].get("num_workers", 4)))
    if cfg.get("pt", {}).get("enabled"):
        cfg["pt"].setdefault("optimizer", {})["fused"] = "auto"
    # Fix loader divisibility for the requested allocation only.
    for key in ("batch_size", "eval_batch_size"):
        ds[key] = max(world_size, math.ceil(ds[key] / world_size) * world_size)
    if train["train_batch_size"] % world_size:
        raise ValueError("Saved global label batch is not divisible by world_size")
    if steps is not None:
        train["total_steps"] = int(steps)
    if finetune:
        if recipe == "preserve":
            raise ValueError("Use a separate candidate for finetuning; preserve is a matched control")
        if steps is None:
            raise ValueError("Finetuning requires an explicit --steps budget")
        cfg["optimizer"]["lr_schedule"].update(learning_rate=5e-5, warmup_steps=200,
                                               total_steps=train["total_steps"], lr_schedule="cosine")
    if benchmark:
        train.update(benchmark_steps=int(benchmark), eval_per_step=0)
    cfg["experiment"] = dict(recipe=recipe, finetune=bool(finetune), world_size=world_size,
                              status="unvalidated_candidate", pt_enabled=cfg.get("pt", {}).get("enabled", False))
    return cfg


def budget(cfg, world_size):
    t = cfg["train"]
    ot = t.get("ot_kwargs", {})
    labels, particles = t["train_batch_size"], t["forward_dict"]["gen_per_label"]
    return dict(global_generated=labels * particles,
                global_negative_generated=labels * ot.get("resample_gen_per_label", particles) if ot.get("resample_neg") else 0,
                global_real_feature_inputs=labels * (t["pos_per_sample"] + t["neg_per_sample"]),
                generated_microbatch_per_gpu=math.ceil(labels // world_size / t.get("grad_accum_steps", 1)) * particles,
                pt_enabled=cfg.get("pt", {}).get("enabled", False),
                generator_rematerialization=cfg["model"].get("use_remat", False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True, help="Saved config.json is preferable to a template")
    p.add_argument("--recipe", choices=["preserve", "quality", "balanced", "speed"], default="balanced")
    p.add_argument("--world-size", type=int, default=8)
    p.add_argument("--steps", type=int)
    p.add_argument("--benchmark", type=int, default=0)
    p.add_argument("--finetune", action="store_true", help="New EMA-initialized run, 5e-5 cosine LR; use train.py --init-ema")
    p.add_argument("--compile", action="store_true", help="Opt-in; measure after compilation warmup")
    p.add_argument("--keep-pt", action="store_true", help="Retain full auxiliary PT objective in a candidate")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    base = yaml.safe_load(Path(args.base).read_text(encoding="utf-8"))
    cfg = make_config(base, args.recipe, args.world_size, args.steps, args.benchmark,
                      args.finetune, args.compile, args.keep_pt)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(json.dumps(budget(cfg, args.world_size), indent=2))
    print(f"Wrote {out}. No speed/FID result is implied by this recipe.")


if __name__ == "__main__":
    main()
