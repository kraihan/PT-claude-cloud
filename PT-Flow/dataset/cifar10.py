"""CIFAR-10 in raw pixel space, 32x32x3.

Kept separate from dataset/dataset.py rather than added as a branch inside it:
the two datasets do not share a data space (SD-VAE latents vs RGB pixels), a
feature extractor, or FID statistics, and pretending otherwise is what makes a
codebase quietly ImageNet-shaped.

The tensor contract is the same as the ImageNet pixel path, and the rest of the
training loop depends on it:

    preprocess_fn(batch)  -> {"images": [B, H, W, C] float in [-1, 1],
                              "labels": [B] long}
    postprocess_fn(images) -> [B, C, H, W] float in [0, 1]
"""

from __future__ import annotations

import os
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import CIFAR10

from dataset.dataset import worker_init_fn
from utils.dist_util import process_count, process_index
from utils.env import CIFAR10_PATH
from utils.logging import log_for_0

IMAGE_SIZE = 32
IN_CHANNELS = 3
NUM_CLASSES = 10

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def build_transforms(use_aug: bool, split: str):
    """CIFAR-10 is already 32x32, so no crop or resize is wanted.

    Output is CHW in [-1, 1]; create_cifar10_split permutes to BHWC to match the
    convention the trainer and the feature extractor expect.
    """
    ops = []
    if split == "train" and use_aug:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
    return transforms.Compose(ops)


def build_dataset(*, use_aug: bool, split: str):
    root = CIFAR10_PATH
    if not root:
        raise ValueError("Set CIFAR10_PATH in utils/env.py.")
    batches = os.path.join(root, "cifar-10-batches-py")
    if not os.path.exists(os.path.join(batches, "data_batch_1")):
        raise FileNotFoundError(
            f"CIFAR-10 not found at {batches}. Run:  python -m misc.download_cifar10"
        )
    # download=False deliberately: a download racing across DistributedSampler
    # workers corrupts the archive.  Fetch once, up front, via the script above.
    return CIFAR10(
        root=root,
        train=(split == "train"),
        transform=build_transforms(use_aug, split),
        download=False,
    )


def create_cifar10_split(
    *,
    batch_size: int,
    split: str,
    use_aug: bool = False,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
    **_ignored,
):
    """Return (loader, preprocess_fn, postprocess_fn) for one CIFAR-10 split.

    ``split`` follows the ImageNet convention: "train" or "val" (CIFAR's test
    split stands in for val).
    """
    ds = build_dataset(use_aug=use_aug, split=("train" if split == "train" else "test"))
    log_for_0(ds)

    world, rank = process_count(), process_index()
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        drop_last=(split == "train"),
        worker_init_fn=partial(worker_init_fn, rank=rank),
        sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=(prefetch_factor if num_workers > 0 else None),
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    def preprocess_fn(batch, rng=0):
        del rng
        image, label = batch
        image = torch.as_tensor(image, dtype=torch.float32)
        image = image.permute(0, 2, 3, 1).contiguous()      # BCHW -> BHWC
        return {"images": image, "labels": torch.as_tensor(label, dtype=torch.long)}

    def postprocess_fn(images, has_clip: bool = True):
        out = (torch.as_tensor(images) + 1.0) / 2.0
        if has_clip:
            out = torch.clamp(out, 0.0, 1.0)
        return out.permute(0, 3, 1, 2).contiguous()          # BHWC -> BCHW

    return loader, preprocess_fn, postprocess_fn


def get_postprocess_fn(has_clip: bool = True):
    """Standalone postprocess, for callers that have no loader (e.g. features)."""

    def postprocess(images):
        out = (torch.as_tensor(images) + 1.0) / 2.0
        if has_clip:
            out = torch.clamp(out, 0.0, 1.0)
        return out.permute(0, 3, 1, 2).contiguous()

    return postprocess
