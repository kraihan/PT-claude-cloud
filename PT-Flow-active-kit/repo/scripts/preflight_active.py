"""Read-only preflight in the allocation before any training starts."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from models.generator import DitGen
from utils.misc import load_config
from utils.ckpt_util import canonical_state_dict, check_model_behavior
from utils import env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    a = p.parse_args()
    cfg = load_config(a.config)
    assert cfg.pt.schedule.policy == "active_recovery_v1"
    assert torch.cuda.device_count() == 2, "This launch expects two allocated GPUs"
    print("GPUs:", [torch.cuda.get_device_name(i) for i in range(2)])
    assert all(torch.cuda.get_device_capability(i)[0] >= 9 for i in range(2)), "Hopper GPUs expected"
    ckpt = torch.load(cfg.train.init_ema_from, map_location="cpu", weights_only=False, mmap=True)
    assert ckpt.get("ema_model") is not None, "Initializer must contain EMA generator weights"
    assert int(ckpt.get("step", -1)) == 200000, "Expected the W-Flow 200k checkpoint"
    with torch.device("meta"):
        model = DitGen(num_classes=cfg.dataset.num_classes, **cfg.model)
    check_model_behavior(model, ckpt)
    want = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    have = {k: tuple(v.shape) for k, v in canonical_state_dict(ckpt["ema_model"]).items()}
    assert want == have, "Generator configuration does not exactly match W-Flow EMA tensors"
    root = Path(env.IMAGENET_CACHE_PATH)
    count = {}
    for split in ("train", "val"):
        arrays = [np.load(root / f"{split}_{name}.npy", mmap_mode="r")
                  for name in ("moments", "moments_flip", "targets")]
        assert len({len(a) for a in arrays}) == 1
        assert arrays[0].shape[1:] == (32, 32, 4)
        assert arrays[1].shape == arrays[0].shape
        assert set(np.unique(arrays[2]).tolist()) == set(range(1000)), f"{split} lacks all 1000 class labels"
        count[split] = len(arrays[2])
    assert count["train"] >= 1000000, "Full ImageNet training requested, but cache looks like a subset"
    assert count["val"] >= 1000
    assert (Path(env.VAE_HF_PATH) / "config.json").is_file(), "Missing SD-VAE config"
    assert Path(env.IMAGENET_FID_NPZ).is_file(), "Missing ImageNet FID reference"
    print(json.dumps(dict(initializer_step=ckpt["step"], new_training_step=0,
                          generator_parameters=sum(v.numel() for v in model.parameters()),
                          full_cache=str(root), counts=count,
                          global_generated_per_update=cfg.train.train_batch_size*cfg.train.forward_dict.gen_per_label), indent=2))


if __name__ == "__main__":
    main()
