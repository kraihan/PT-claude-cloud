"""Fetch the pretrained assets named in utils/env.py.

Every section is guarded on its own configuration, so an unset entry is skipped
rather than failing the whole script.  That is what makes a from-scratch run
possible without the reference-checkpoint settings: leave BASELINE_HF_REPO_ID
and BASELINE_HF_ROOT empty and this simply reports them as skipped.

Required for training from scratch:  VAE_HF_PATH, HF_REPO_ID/HF_ROOT.
Required for FID:                    TORCH_HUB_DIR (and IMAGENET_FID_NPZ).
Required only to resume:             BASELINE_HF_REPO_ID/BASELINE_HF_ROOT.
"""

import os
import sys

# Run either as `python -m misc.download_pretrained` or as
# `python misc/download_pretrained.py`.  The second form puts THIS directory on
# sys.path[0] rather than the repo root, so `import utils` would fail; prepend
# the repo root explicitly.  (Note: the top-level `misc/` directory and the
# module `utils/misc.py` are unrelated despite the shared name.)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from huggingface_hub import snapshot_download

from utils.env import (
    BASELINE_HF_REPO_ID,
    BASELINE_HF_ROOT,
    HF_REPO_ID,
    HF_ROOT,
    TORCH_HUB_DIR,
    VAE_HF_PATH,
)

skipped, done = [], []


def fetch(repo_id, local_dir, what, needed_for):
    if not repo_id or not local_dir:
        skipped.append(f"{what} (needed for {needed_for})")
        return False
    snapshot_download(repo_id=repo_id, local_dir=local_dir)
    done.append(f"{what}: {repo_id} -> {local_dir}")
    return True


# --- SD-VAE: encodes the latent cache, decodes samples ----------------------
fetch("stabilityai/sd-vae-ft-mse", VAE_HF_PATH, "SD-VAE", "training and sampling")

# --- MAE feature extractor: the drift loss is computed on its features ------
fetch(HF_REPO_ID, HF_ROOT, "MAE feature extractor", "training")

# --- reference checkpoints: only to resume an existing run ------------------
fetch(BASELINE_HF_REPO_ID, BASELINE_HF_ROOT,
      "reference OT-drift checkpoints", "resuming, not for training from scratch")

# --- torch-fidelity Inception: FID / IS ------------------------------------
if TORCH_HUB_DIR:
    os.makedirs(TORCH_HUB_DIR, exist_ok=True)
    torch.hub.set_dir(TORCH_HUB_DIR)

    from torch_fidelity.utils import create_feature_extractor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Must match utils/fid_util.py and utils/fidelity_wrapper.py: the
    # inception-v3-compat extractor, 2048-d features, logits_unbiased for IS.
    # Mixing extractors produces FID numbers that compare to nothing.
    fe = create_feature_extractor(
        "inception-v3-compat", ["2048", "logits_unbiased"],
        cuda=(device.type == "cuda"),
    ).eval()

    # inception-v3-compat expects NCHW uint8 in [0, 255].
    with torch.no_grad():
        feats, logits = fe(torch.zeros(1, 3, 256, 256, dtype=torch.uint8, device=device))
    done.append(
        f"torch-fidelity Inception -> {torch.hub.get_dir()} "
        f"(features {tuple(feats.shape)}, logits {tuple(logits.shape)})"
    )
else:
    skipped.append("torch-fidelity Inception (needed for FID / IS)")

print("\n--- downloaded ---")
for line in done:
    print("  " + line)
if skipped:
    print("\n--- skipped (not configured in utils/env.py) ---")
    for line in skipped:
        print("  " + line)
print()
