"""Dataset integrity, ImageNet planning and smoke-gate tests; no real GPU/Slurm."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import make_latent_subset as subset
from scripts import prepare_imagenet_ablation as setup
from scripts import run_parallel_ablation as parallel

six = parallel.six
REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "configs/ablation/six_imagenet_a100_4k.yaml"


def tiny_cache(path, labels):
    path.mkdir()
    for split, targets in (("train", labels), ("val", np.array([0], dtype=np.int64))):
        np.save(path / f"{split}_targets.npy", targets)
        for part in ("moments", "moments_flip"):
            array = np.lib.format.open_memmap(path / f"{split}_{part}.npy", mode="w+",
                                             dtype=np.float32, shape=(len(targets), 32, 32, 4))
            array[:] = np.arange(len(targets))[:, None, None, None]
            array.flush()
            del array


def test_subset_balanced_deterministic_all_classes():
    labels = np.repeat(np.arange(1000, dtype=np.int64), 60)
    first = subset.select_indices(labels)
    second = subset.select_indices(labels)
    assert np.array_equal(first, second)
    assert len(first) == 50000 and len(np.unique(first)) == 50000
    assert np.all(np.bincount(labels[first], minlength=1000) == 50)
    assert not np.array_equal(first, subset.select_indices(labels, seed=43))


def test_subset_insufficient_class_rejected():
    with pytest.raises(ValueError, match="Class 1"):
        subset.select_indices(np.array([0, 0, 1]), per_class=2, num_classes=2)


def test_subset_copies_correct_rows_and_reuses_without_changing_source(tmp_path, monkeypatch):
    source, target = tmp_path / "source", tmp_path / "subset"
    tiny_cache(source, np.repeat(np.arange(3, dtype=np.int64), 4))
    before = {p.name: p.stat().st_mtime_ns for p in source.iterdir()}
    # Cluster creates symlinks. Hard links exercise the same no-copy semantics
    # on Windows machines where symbolic-link privileges may be unavailable.
    monkeypatch.setattr(Path, "symlink_to", lambda self, dest: self.hardlink_to(dest))
    metadata = subset.create_subset(source, target, per_class=2, num_classes=3)
    indices = np.load(target / "source_train_indices.npy")
    assert metadata["num_train"] == 6
    for part in ("moments", "moments_flip", "targets"):
        assert np.array_equal(np.load(target / f"train_{part}.npy"), np.load(source / f"train_{part}.npy")[indices])
        assert (target / f"val_{part}.npy").samefile(source / f"val_{part}.npy")
    assert subset.create_subset(source, target, per_class=2, num_classes=3) == metadata
    assert before == {p.name: p.stat().st_mtime_ns for p in source.iterdir()}
    with pytest.raises(ValueError, match="differs"):
        subset.create_subset(source, target, per_class=3, num_classes=3)


def test_imagenet_plan_and_a100_resources():
    plan = six.build_plan(MANIFEST)
    assert plan["steps"] == 4000 and len(plan["arms"]) == 11
    assert len(parallel.tasks(plan)) == 17
    cfg = plan["arms"][0]["config"]
    assert cfg["pipeline"] == "imagenet_latent"
    assert cfg["dataset"]["num_classes"] == 1000 and cfg["dataset"]["subset_per_class"] == 50
    assert (cfg["model"]["hidden_size"], cfg["model"]["depth"], cfg["model"]["in_channels"]) == (768, 12, 4)
    assert cfg["train"]["train_batch_size"] // cfg["train"]["grad_accum_steps"] * cfg["train"]["forward_dict"]["gen_per_label"] == 16
    assert cfg["feature"]["use_mae"] and cfg["feature"]["checkpoint"]
    command = parallel.sbatch_args(plan, "fixture")
    assert "--gpus=a100:1" in command and "--constraint=gpu_a100_40gb" in command
    assert not any("constraint" in p or "gpus" in p for p in parallel.sbatch_args(plan, "report", gpu=False))
    row = six.result_row(plan, plan["stages"][0]["entries"][0])
    assert row["dataset"] == "imagenet_latent" and row["subset_per_class"] == 50
    assert "jit_in256_stats.npz" in row["fid_reference"]


def test_imagenet_preflight_checks_labels_shapes_and_assets(tmp_path):
    plan = six.build_plan(MANIFEST)
    cfg = copy.deepcopy(plan["arms"][0]["config"])
    cfg["dataset"]["subset_per_class"] = 2
    cache = tmp_path / "cache"
    tiny_cache(cache, np.repeat(np.arange(1000, dtype=np.int64), 2))
    assets = tmp_path / "assets"
    vae = assets / "sdvae"
    vae.mkdir(parents=True)
    (vae / "config.json").write_text("{}")
    (vae / "diffusion_pytorch_model.safetensors").write_text("fixture; no model is loaded")
    mae = assets / "mae/models/mae/jax/mae_latent_640"
    mae.mkdir(parents=True)
    (mae / "metadata.json").write_text("{}")
    (mae / "ema_params.pt").write_text("fixture")
    stats = assets / "fid_stats/jit_in256_stats.npz"
    stats.parent.mkdir()
    np.savez_compressed(stats, mu=np.zeros(2048), sigma=np.eye(2048))
    env = dict(PTFLOW_ASSETS=str(assets), IMAGENET_CACHE_PATH=str(cache))
    six.preflight_imagenet(cfg, env, REPO)
    labels = np.load(cache / "train_targets.npy")
    labels[0] = 1
    np.save(cache / "train_targets.npy", labels)
    with pytest.raises(ValueError, match="exactly 2"):
        six.preflight_imagenet(cfg, env, REPO)


def test_smoke_uses_worst_k_and_both_samplers_without_polluting_study(tmp_path):
    original = six.build_plan(MANIFEST)
    plan = setup.smoke_plan(MANIFEST, tmp_path / "smoke")
    cfg = plan["arms"][0]["config"]
    assert plan["steps"] == 20 and cfg["pt"]["K"] == 16
    assert cfg["pt"]["schedule"]["prox_warmup"] == 0
    assert not plan["wandb"]["enabled"] and not cfg["logging"]["use_wandb"]
    assert [e["n"] for e in parallel.tasks(plan)] == [0, 4]
    assert plan["evaluation"]["num_samples"] == 32
    assert plan["root"] != original["root"] and original["steps"] == 4000


def test_smoke_failure_never_writes_success(tmp_path, monkeypatch):
    monkeypatch.setattr(parallel, "hardware", lambda: dict(gpu_memory_gib=40, compute_capability=[8, 0]))
    monkeypatch.setattr(six, "preflight", lambda p: None)
    monkeypatch.setattr(six, "materialize", lambda p: None)
    monkeypatch.setattr(parallel, "worker", lambda p, i: 1)
    with pytest.raises(RuntimeError, match="NOT submitted"):
        setup.smoke(MANIFEST, tmp_path)
    assert not (tmp_path / "smoke_passed.json").exists()
