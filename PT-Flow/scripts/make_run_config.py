"""Fresh-run profiles; smoke and benchmark runs retain the training schedules."""
import argparse
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def make_config(profile, *, baseline=False, smoke=0, benchmark=0):
    cifar = profile == "cifar10_t4"
    path = ROOT / "configs/gen" / ("ptflow_cifar10.yaml" if cifar else "ptflow_scratch.yaml")
    cfg = yaml.safe_load(path.read_text())
    cfg["logging"] = {"use_wandb": False, "log_every_k": 20}
    model, train, pt = cfg["model"], cfg["train"], cfg["pt"]
    model.update(precision="auto", use_bf16=True, attn_fp32=False)
    train.update(total_steps=200000, save_per_step=2000, keep_every=20000,
                 eval_per_step=10000, eval_samples=50000, eval_at_start=False,
                 feature_chunk_size=64, push_at_resume=20, diverse_noise=True)
    train["cfg_list"] = [1.0, 1.2, 1.5, 2.0]
    cfg["optimizer"]["lr_schedule"]["total_steps"] = train["total_steps"]
    pt.update(enabled=not baseline, logw_clip=0.0, prox_norm="bounded_rms",
              data_bsz=16, noise_bsz=16, prox_max_batch=32, potential_chunk=32)
    pt["model"]["use_bf16"] = False
    pt["scale_model"]["use_bf16"] = False
    pt["optimizer"]["lr_schedule"]["total_steps"] = train["total_steps"]
    pt["schedule"].update(eps_warmup=5000, eps_anneal_steps=60000,
                         prox_warmup=5000, prox_ramp=15000, lambda_prox_max=0.1,
                         theta_period=2, theta_period_degraded=4, health_check_period=20)
    cfg["feature"].update(chunk_size=64, checkpoint=False)
    if cifar:
        model.update(depth=8, noise_classes=0, noise_coords=1)
        cfg["dataset"].update(batch_size=256, eval_batch_size=128)
        cfg["dataset"]["kwargs"]["num_workers"] = 2
        cfg["feature"].update(convnext_model="tiny", convnext_bf16=True)
        # Explicit speed/feature-resolution ablation; use 224 for full resolution.
        train["activation_kwargs"] = {"convnext_kwargs": {"image_size": 128}}
        train.update(train_batch_size=16, grad_accum_steps=2, ema_decay=0.999)
        pt["model"].update(cond_dim=128, hidden_size=128, depth=3, num_heads=4, patch_size=4)
        pt["scale_model"].update(cond_dim=64, hidden_size=64, depth=2, num_heads=4, patch_size=4)
        pt["ema_decay"] = 0.999
    else:
        # Match upstream B/2 generator, features, global OT particle counts,
        # optimizer and zero-output initialization. Accumulate for 8 large GPUs.
        model.update(residual=False)
        cfg["dataset"].update(batch_size=4096, eval_batch_size=256)
        train.update(train_batch_size=128, pos_per_sample=64, neg_per_sample=32,
                     push_per_step=128, grad_accum_steps=4, ema_decay=0.999)
        train["forward_dict"].update(gen_per_label=64, neg_cfg_pw=5.0)
        cfg["feature"]["mae_path"] = "hf://mae_latent_640"
        cfg["optimizer"].update(weight_decay=0.0)
        cfg["optimizer"]["lr_schedule"].update(learning_rate=0.0004, warmup_steps=10000)
    if smoke:
        train.update(total_steps=smoke, save_per_step=smoke, eval_per_step=0)
        cfg["logging"]["log_every_k"] = 1
    if benchmark:
        train.update(benchmark_steps=benchmark, eval_per_step=0)
        cfg["logging"]["log_every_k"] = 1
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=["cifar10_t4", "imagenet_b"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--smoke", type=int, default=0)
    p.add_argument("--benchmark", type=int, default=0)
    a = p.parse_args()
    cfg = make_config(a.profile, baseline=a.baseline, smoke=a.smoke, benchmark=a.benchmark)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"Wrote {out}. Use a NEW workdir for this fresh-run profile.")


if __name__ == "__main__":
    main()
