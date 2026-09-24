from __future__ import annotations
import os
from pathlib import Path

A = os.environ.get("PTFLOW_ASSETS", str(Path(__file__).resolve().parents[1] / "assets"))
DATA_ROOT = os.environ.get("PTFLOW_DATA", str(Path(__file__).resolve().parents[1] / "data"))

# REQUIRED: MAE feature extractor -- both drift losses are computed on its
# features, and train.py asserts on it when feature.use_mae is true.
HF_REPO_ID = "Goodeat/drifting"
HF_ROOT = os.environ.get("HF_ROOT", f"{A}/mae")

# REQUIRED: SD-VAE -- builds the latent cache, decodes samples.
VAE_HF_PATH = os.environ.get("VAE_HF_PATH", f"{A}/sdvae")

# FID only. Leave empty and set train.eval_per_step high to defer.
TORCH_HUB_DIR = os.environ.get("TORCH_HUB_DIR", f"{A}/torch_hub")

# Resume only. Empty is correct when training from scratch.
BASELINE_HF_REPO_ID = ""
BASELINE_HF_ROOT = ""
BASELINE_CKPTS = {}

# --- CIFAR-10 (pixel space, 32x32x3; no VAE, no latent cache) ---------------
# CIFAR10_PATH is the directory CONTAINING cifar-10-batches-py/, which is what
# torchvision.datasets.CIFAR10 expects as its root.
CIFAR10_PATH = os.environ.get("CIFAR10_PATH", f"{DATA_ROOT}/cifar10")
CIFAR10_FID_NPZ = os.environ.get("CIFAR10_FID_NPZ", f"{A}/fid_stats/cifar10_train_fid_stats.npz")

# data
IMAGENET_PATH = os.environ.get("IMAGENET_PATH", f"{DATA_ROOT}/imagenet")
IMAGENET_CACHE_PATH = os.environ.get("IMAGENET_CACHE_PATH", f"{DATA_ROOT}/latents")
IMAGENET_FID_NPZ = os.environ.get("IMAGENET_FID_NPZ", f"{A}/fid_stats/jit_in256_stats.npz")
IMAGENET_PR_NPZ = ""
