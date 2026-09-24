"""Install a private, versioned training snapshot and resolve the 30k recipe."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import yaml


def prepare(source_config, destination, initializer):
    source_config, destination, initializer = map(Path, (source_config, destination, initializer))
    if destination.exists():
        raise FileExistsError(f"Refusing to reuse destination: {destination}")
    if not initializer.is_file():
        raise FileNotFoundError(initializer)
    with source_config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    bundled = Path(__file__).resolve().parent / "repo"
    cfg["pipeline"] = "imagenet_latent"
    ds = cfg.setdefault("dataset", {})
    if int(ds.get("num_classes", 1000)) != 1000:
        raise ValueError("This recipe requires ImageNet-1k")
    ds.update(num_classes=1000, resolution=256, use_cache=True, use_latent=True,
              use_aug=False, batch_size=512, eval_batch_size=128)
    ds.setdefault("kwargs", {}).update(num_workers=4, pin_memory=True)
    model = cfg.setdefault("model", {})
    model.pop("num_classes", None)  # dataset owns this argument
    model.update(use_remat=True, residual=False)
    cfg.pop("init_generator_from", None)
    train = cfg.setdefault("train", {})
    for key in ("init_generator_from", "resume_from", "init_from", "init_ema_from"):
        train.pop(key, None)
    pt = cfg.setdefault("pt", {})
    pt.pop("init_generator_from", None)
    train.update(total_steps=30000, save_per_step=1000, keep_every=1000, keep_last=2,
                 train_batch_size=128, grad_accum_steps=32, init_ema_from=str(initializer.resolve()),
                 eval_per_step=10000, eval_at_start=False, eval_samples=50000, cfg_list=[1.2],
                 push_at_resume=20, compile_generator=False,
                 feature_chunk_size=16, feature_cache_gib=0.)
    train.setdefault("forward_dict", {}).update(gen_per_label=64)
    # Streaming the real batch and checkpointing feature forwards address
    # different allocations. Forward chunking alone retains all real features.
    cfg.setdefault("feature", {}).update(chunk_size=16, checkpoint=True)
    # Fail explicitly on unsupported inherited trainer fields instead of
    # launching jobs that cannot start or silently dropping behavior.
    tree = ast.parse((bundled / "train.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train_gen")
    allowed = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
    unknown = sorted(set(train) - allowed)
    if unknown:
        raise ValueError(f"Unsupported train keys in base recipe: {unknown}")
    cfg.setdefault("optimizer", {}).setdefault("lr_schedule", {}).update(
        learning_rate=2e-5, warmup_steps=1000, total_steps=30000, lr_schedule="cosine")
    pt.update(enabled=True, K=16, noise_bsz=32, data_bsz=32, potential_chunk=16,
              prox_max_batch=4, prox_period=1, prox_mode="full", prox_norm="none", logw_clip=0.,
              scale_K=4, scale_mode="learned", max_grad_norm=1., lambda_gauge=.1,
              curv_probe=False)
    pt.setdefault("model", {}).update(use_bf16=False, use_remat=True, output_mode="phi")
    pt.setdefault("scale_model", {}).update(use_bf16=False)
    pt.setdefault("optimizer", {}).setdefault("lr_schedule", {}).update(
        learning_rate=2e-5, warmup_steps=500, total_steps=30000, lr_schedule="cosine")
    pt["schedule"] = dict(
        policy="active_recovery_v1", eps_schedule="cosine", eps_max=.1, eps_min=.005,
        eps_warmup=1000, eps_anneal_steps=20000,
        prox_warmup=1000, prox_ramp=5000, lambda_prox_max=.1,
        alpha_def_start=.5, alpha_def_end=.05, alpha_def_degraded=.75,
        theta_period=1, theta_period_degraded=1, health_check_period=1,
        lambda_scale=.1, lambda_curv=0., ess_healthy=.3, ess_broken=.05, ess_ema_decay=.98,
        proposal_refine_steps=8, proposal_refine_lr=.5,
        alignment_weight=.1, alignment_hold=1000, alignment_end=5000, alignment_batch=4,
        recovery_check_after=1500, recovery_bad_patience=500,
    )
    cfg.setdefault("logging", {}).update(use_wandb=True, log_every_k=20, name=destination.name)
    cfg["logging"].setdefault("project", "ptflow-active-recovery")
    destination.mkdir(parents=True)
    repo = destination / "PT-Flow"
    shutil.copytree(bundled, repo, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "assets"))
    shutil.copy2(source_config, destination / "original_base.yaml")
    config_file = destination / "active_B_30k.yaml"
    config_file.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    hashes = {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(repo.rglob("*")) if p.is_file()}
    report = dict(policy="active_recovery_v1", initializer=str(initializer.resolve()),
                  source_config=str(source_config.resolve()), config=str(config_file),
                  source_config_sha256=hashlib.sha256(source_config.read_bytes()).hexdigest(),
                  code_sha256=hashes, train_steps=30000, global_generated_batch=8192,
                  ranks=2, per_rank_labels=64, chunks_per_rank=32,
                  generated_microbatch_per_rank=128,
                  objective="W-Flow + full PT prox; refined IS; temporary potential gradient calibration",
                  note="Iteration-driven schedules are not ESS certification; K=16 and quality are unvalidated on ImageNet")
    (destination / "manifest.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(f"Isolated repository: {repo}")
    print(f"Configuration: {config_file}")
    print("30,000 updates; save every 1,000; global generated batch 8192; full PT prox derivative")
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--destination", required=True)
    ap.add_argument("--initializer", required=True)
    a = ap.parse_args()
    prepare(a.base, a.destination, a.initializer)
