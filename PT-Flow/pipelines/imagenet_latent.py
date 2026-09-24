"""ImageNet-1k in SD-VAE latent space, 32x32x4.

The SOTA configuration: class-conditional ImageNet 256x256, encoded once into a
latent cache, with the drift loss computed against a latent-space MAE.

Requires, in utils/env.py:
    IMAGENET_PATH / IMAGENET_CACHE_PATH   the images and the cache
    VAE_HF_PATH                           SD-VAE, to build the cache and decode
    HF_REPO_ID / HF_ROOT                  the MAE feature extractor
    IMAGENET_FID_NPZ                      reference statistics
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

from dataset.dataset import create_imagenet_split, get_postprocess_fn
from models.mae_model import build_activation_function
from pipelines.base import Pipeline


class ImageNetLatentPipeline(Pipeline):
    name = "imagenet_latent"
    fid_dataset_name = "imagenet256"
    input_size = 32
    in_channels = 4          # SD-VAE latent channels, not RGB
    num_classes = 1000
    description = "ImageNet-1k 256x256 in SD-VAE latent space, MAE features"

    def __init__(self, config):
        super().__init__(config)
        ds = config.dataset
        self.resolution = int(ds.get("resolution", 256))
        self.num_classes = int(ds.get("num_classes", 1000))
        self.use_aug = bool(ds.get("use_aug", False))
        self.use_latent = bool(ds.get("use_latent", True))
        self.use_cache = bool(ds.get("use_cache", True))
        self.kwargs = dict(ds.get("kwargs", {}))
        self.fid_dataset_name = f"imagenet{self.resolution}"

        if not (self.use_latent or self.use_cache):
            raise ValueError(
                "imagenet_latent expects dataset.use_latent or dataset.use_cache "
                "to be true; for pixel-space training use a pixel pipeline."
            )

    def build_split(self, *, split: str, batch_size: int) -> Tuple[object, Callable, Callable]:
        return create_imagenet_split(
            resolution=self.resolution,
            use_aug=self.use_aug,
            use_latent=self.use_latent,
            use_cache=self.use_cache,
            batch_size=batch_size,   # caller already divided by world size
            split=split,
            **self.kwargs,
        )

    def build_features(self) -> Tuple[Callable, Dict]:
        cfg = dict(self.config.get("feature", {}))
        use_mae = bool(cfg.get("use_mae", True))

        mae_path = str(cfg.get("mae_path", "")).strip()
        if not mae_path and use_mae:
            load_dict = cfg.get("load_dict", {})
            if str(load_dict.get("source", "hf")).strip().lower() == "local":
                mae_path = str(load_dict.get("path", "")).strip()
            else:
                model_name = str(load_dict.get("hf_model_name", "")).strip()
                if model_name:
                    mae_path = f"hf://{model_name}"
        if use_mae and not mae_path:
            raise ValueError(
                "feature.mae_path (or feature.load_dict.hf_model_name / .path) is "
                "required when feature.use_mae is true."
            )

        # The ConvNeXt branch normalizes with postprocess_fn first, so it needs
        # the unclipped version to avoid saturating before normalization.
        postprocess_noclip = get_postprocess_fn(
            use_aug=self.use_aug,
            use_latent=self.use_latent,
            use_cache=self.use_cache,
            has_clip=False,
        )
        return build_activation_function(
            mae_path=mae_path,
            use_convnext=bool(cfg.get("use_convnext", False)),
            convnext_bf16=bool(cfg.get("convnext_bf16", False)),
            use_mae=use_mae,
            postprocess_fn=postprocess_noclip,
            feature_chunk_size=int(cfg.get("chunk_size", 64)),
            checkpoint_features=bool(cfg.get("checkpoint", False)),
            channels_last=bool(cfg.get("channels_last", False)),
        )
