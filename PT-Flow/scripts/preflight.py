"""Check a run before spending GPU time. No training or downloads by default."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from models.generator import DitGen
from pipelines import build_pipeline
from utils import env
from utils.misc import load_config
from utils.precision import amp_dtype


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--world-size", type=int, default=1)
    ap.add_argument("--check-assets", action="store_true")
    ap.add_argument("--check-features", action="store_true", help="Load/download pretrained features and verify input gradients.")
    a = ap.parse_args()
    cfg = load_config(a.config)
    pipeline = build_pipeline(cfg)
    pipeline.validate(cfg.model)
    for key, size in (("loader", cfg.dataset.batch_size), ("eval", cfg.dataset.eval_batch_size), ("labels", cfg.train.train_batch_size)):
        if size < a.world_size or size % a.world_size:
            raise ValueError(f"{key} batch {size} is not divisible by world size {a.world_size}")
    with torch.device("meta"):
        model = DitGen(num_classes=cfg.dataset.num_classes, **cfg.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = amp_dtype(device, cfg.model.get("precision", "auto"), cfg.model.get("use_bf16", False))
    pt = cfg.get("pt", {})
    report = dict(pipeline=pipeline.name, device=str(device), precision=str(dtype or torch.float32),
                  parameters=sum(p.numel() for p in model.parameters()),
                  global_generated_per_step=cfg.train.train_batch_size * cfg.train.forward_dict.gen_per_label,
                  labels_per_rank=cfg.train.train_batch_size // a.world_size,
                  pt_enabled=pt.get("enabled", False), objective="W-Flow feature drift + auxiliary PT prox" if pt.get("enabled") else "W-Flow feature drift")
    if a.check_assets:
        if pipeline.name == "cifar10_pixel":
            paths = [Path(env.CIFAR10_PATH) / "cifar-10-batches-py" / "data_batch_1"]
            ref = env.CIFAR10_FID_NPZ
        else:
            paths = [Path(env.IMAGENET_CACHE_PATH) / f"{split}_{part}.npy"
                     for split in ("train", "val") for part in ("moments", "moments_flip", "targets")]
            paths += [Path(env.VAE_HF_PATH) / "config.json"]
            ref = env.IMAGENET_FID_NPZ
        missing = [str(p) for p in paths if not p.exists()]
        if cfg.train.eval_per_step > 0:
            missing += [] if Path(ref).is_file() else [ref]
            if Path(ref).is_file():
                with np.load(ref) as stats:
                    mu = stats["ref_mu"] if "ref_mu" in stats else stats["mu"]
                    sigma = stats["ref_sigma"] if "ref_sigma" in stats else stats["sigma"]
                    if mu.shape != (2048,) or sigma.shape != (2048, 2048) or not np.isfinite(mu).all() or not np.isfinite(sigma).all():
                        raise ValueError("Invalid FID reference dimensions or nonfinite values")
        if missing:
            raise FileNotFoundError("Missing assets:\n" + "\n".join(missing))
        report["asset_paths"] = "present (feature checkpoint checked separately with --check-features)"
    if a.check_features:
        feature, params = pipeline.build_features()
        x = torch.randn(2, pipeline.input_size, pipeline.input_size, pipeline.in_channels, device=device, requires_grad=True)
        values = feature(params, x, **cfg.train.get("activation_kwargs", {}))
        loss = sum(v.float().square().mean() for v in values.values())
        loss.backward()
        if not torch.isfinite(loss) or not torch.isfinite(x.grad).all() or x.grad.abs().sum() == 0:
            raise RuntimeError("Feature extractor failed the finite, nonzero input-gradient check")
        report["feature_groups"] = list(values)
        report["feature_gradient_rms"] = x.grad.square().mean().sqrt().item()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
