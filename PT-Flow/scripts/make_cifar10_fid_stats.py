"""Build the CIFAR-10 FID reference statistics.

    python -m scripts.make_cifar10_fid_stats

Embeds the full 50,000-image CIFAR-10 train split with the SAME feature
extractor `utils/fid_util.py` uses at evaluation time -- torch-fidelity's
`inception-v3-compat`, 2048-d pool features -- and writes `mu` / `sigma` to
CIFAR10_FID_NPZ.

Using a different Inception (torchvision's, say) would produce statistics that
silently disagree with the ones evaluation computes, and every FID number would
be wrong without ever looking wrong.
"""

import argparse
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.env import CIFAR10_FID_NPZ, CIFAR10_PATH, TORCH_HUB_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=CIFAR10_FID_NPZ)
    ap.add_argument("--batch-size", type=int, default=250)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    args = ap.parse_args()

    if not args.out:
        raise SystemExit("Set CIFAR10_FID_NPZ in utils/env.py, or pass --out.")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    if TORCH_HUB_DIR:
        os.makedirs(TORCH_HUB_DIR, exist_ok=True)
        torch.hub.set_dir(TORCH_HUB_DIR)

    from torch_fidelity.utils import create_feature_extractor
    from torchvision.datasets import CIFAR10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fe = create_feature_extractor(
        "inception-v3-compat", ["2048"], cuda=(device.type == "cuda")
    ).eval()

    ds = CIFAR10(root=CIFAR10_PATH, train=(args.split == "train"), download=False)
    # ds.data is (N, 32, 32, 3) uint8 HWC.  inception-v3-compat wants NCHW uint8
    # in [0, 255] and resizes internally, so no normalization here.
    data = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()
    print(f"{args.split}: {tuple(data.shape)} uint8")

    feats = []
    with torch.no_grad():
        for i in range(0, data.shape[0], args.batch_size):
            batch = data[i : i + args.batch_size].to(device)
            feats.append(fe(batch)[0].double().cpu())
            if (i // args.batch_size) % 20 == 0:
                print(f"  {i + batch.shape[0]:>6} / {data.shape[0]}", flush=True)
    f = torch.cat(feats).numpy()
    print("features:", f.shape)

    mu = f.mean(axis=0)
    sigma = np.cov(f, rowvar=False)
    np.savez(args.out, mu=mu, sigma=sigma)
    print(f"wrote {args.out}  (mu {mu.shape}, sigma {sigma.shape})")
    print("Point CIFAR10_FID_NPZ at this file if you passed a custom --out.")


if __name__ == "__main__":
    main()
