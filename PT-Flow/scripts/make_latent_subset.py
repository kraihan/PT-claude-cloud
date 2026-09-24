"""Create an immutable, balanced ImageNet latent subset without re-encoding.

Only training rows are copied. Validation arrays are linked read-only by the
dataset loader to the original cache. No source files are modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np


def select_indices(labels, per_class=50, seed=42, num_classes=1000):
    if labels.ndim != 1 or labels.dtype.kind not in "iu" or per_class < 1:
        raise ValueError("Expected integer labels and positive per_class")
    if len(labels) == 0 or labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError("Labels outside the requested class range")
    rng = np.random.default_rng(seed)
    selected = []
    # One stable sort avoids 1000 scans of all ImageNet labels.
    order = np.argsort(labels, kind="stable")
    counts = np.bincount(labels.astype(np.int64), minlength=num_classes)
    offsets = np.r_[0, np.cumsum(counts)]
    for label, count in enumerate(counts):
        if count < per_class:
            raise ValueError(f"Class {label} has {count} images; need {per_class}")
        selected.append(rng.choice(order[offsets[label]:offsets[label + 1]], per_class, replace=False))
    # Sorted source indices reduce random disk reads; the train loader shuffles.
    return np.sort(np.concatenate(selected)).astype(np.int64)


def create_subset(source, target, per_class=50, seed=42, num_classes=1000):
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target:
        raise ValueError("Source and subset target must differ")
    arrays, files = {}, {}
    for split in ("train", "val"):
        for part in ("moments", "moments_flip", "targets"):
            name = f"{split}_{part}.npy"
            path = source / name
            arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
            stat = path.stat()
            files[name] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        labels = arrays[f"{split}_targets.npy"]
        if labels.ndim != 1 or labels.dtype.kind not in "iu":
            raise ValueError(f"Invalid {split} labels")
        for part in ("moments", "moments_flip"):
            array = arrays[f"{split}_{part}.npy"]
            if array.shape != (len(labels), 32, 32, 4) or array.dtype.kind != "f":
                raise ValueError(f"Expected {split} BHWC 32x32x4 floating latents")
    indices = select_indices(arrays["train_targets.npy"], per_class, seed, num_classes)
    metadata = dict(version=1, source=str(source), source_files=files, per_class=per_class,
                    seed=seed, num_classes=num_classes, num_train=len(indices),
                    indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
                    labels_sha256=hashlib.sha256(np.asarray(arrays["train_targets.npy"]).tobytes()).hexdigest())
    manifest = target / "subset.json"
    if target.exists():
        if not manifest.is_file() or json.loads(manifest.read_text()) != metadata:
            raise ValueError(f"Existing subset differs or is incomplete: {target}. Use a new target.")
        for name in files:
            array = np.load(target / name, mmap_mode="r", allow_pickle=False)
            expected = len(indices) if name.startswith("train_") else len(arrays[name])
            if len(array) != expected:
                raise ValueError(f"Truncated subset file: {name}")
        print(f"Reuse verified subset: {target}", flush=True)
        return metadata
    staging = target.with_name(target.name + ".building-" + uuid.uuid4().hex)
    staging.mkdir(parents=True)
    # An interrupted build remains a separate .building-* directory and never
    # looks like a completed subset. Never overwrite/delete the source cache.
    for part in ("moments", "moments_flip", "targets"):
        name = f"train_{part}.npy"
        array = arrays[name]
        output = np.lib.format.open_memmap(staging / name, mode="w+", dtype=array.dtype,
                                          shape=(len(indices), *array.shape[1:]))
        for begin in range(0, len(indices), 256):
            batch = array[indices[begin:begin + 256]]
            if not np.isfinite(batch).all():
                raise ValueError(f"Nonfinite selected latents in {name}")
            output[begin:begin + len(batch)] = batch
        output.flush()
        del output
    np.save(staging / "source_train_indices.npy", indices, allow_pickle=False)
    for part in ("moments", "moments_flip", "targets"):
        name = f"val_{part}.npy"
        (staging / name).symlink_to(source / name)
    (staging / "subset.json").write_text(json.dumps(metadata, indent=2) + "\n")
    staging.rename(target)
    print(f"Created {len(indices)} training examples: {per_class} x {num_classes} classes at {target}", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--per-class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_subset(args.source, args.target, args.per_class, args.seed)


if __name__ == "__main__":
    main()
