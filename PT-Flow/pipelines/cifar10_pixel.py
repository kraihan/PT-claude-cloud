"""CIFAR-10 in raw pixel space, 32x32x3.

The first-experiment configuration, and the standard one for a one-step
generative model: small, fast to iterate, and the dataset the field reports
first.  No VAE, no latent cache -- the model operates directly on pixels in
[-1, 1].

The feature extractor is the one thing that cannot carry over from the latent
pipeline.  The drift loss is computed *entirely* in feature space, and the MAE
extractors are trained on SD-VAE latents of ImageNet; handed RGB pixels they
produce features with no meaning, so the drift arm -- the stability anchor --
would be anchoring to noise.  ConvNeXt-V2 takes RGB, upsamples 32 -> 224 and
ImageNet-normalizes, which is why it is the extractor that applies here.

Requires, in utils/env.py:
    CIFAR10_PATH        directory containing cifar-10-batches-py/
    CIFAR10_FID_NPZ     reference statistics (scripts/make_cifar10_fid_stats.py)
    TORCH_HUB_DIR       Inception cache, for FID
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

from dataset.cifar10 import IMAGE_SIZE, IN_CHANNELS, NUM_CLASSES, create_cifar10_split
from dataset.cifar10 import get_postprocess_fn as cifar_postprocess
from models.mae_model import build_activation_function
from pipelines.base import Pipeline


class Cifar10PixelPipeline(Pipeline):
    name = "cifar10_pixel"
    fid_dataset_name = "cifar10"
    input_size = IMAGE_SIZE
    in_channels = IN_CHANNELS
    num_classes = NUM_CLASSES
    description = "CIFAR-10 32x32 RGB, pixel space, ConvNeXt-V2 features"

    def __init__(self, config):
        super().__init__(config)
        ds = config.dataset
        self.use_aug = bool(ds.get("use_aug", True))
        self.kwargs = dict(ds.get("kwargs", {}))

        if bool(ds.get("use_latent", False)) or bool(ds.get("use_cache", False)):
            raise ValueError(
                "cifar10_pixel trains on raw pixels: set dataset.use_latent and "
                "dataset.use_cache to false. There is no SD-VAE for 32x32x3."
            )
        n = int(ds.get("num_classes", NUM_CLASSES))
        if n != NUM_CLASSES:
            raise ValueError(f"CIFAR-10 has {NUM_CLASSES} classes, config says {n}.")

    def build_split(self, *, split: str, batch_size: int) -> Tuple[object, Callable, Callable]:
        return create_cifar10_split(
            batch_size=batch_size,   # caller already divided by world size
            split=split,
            use_aug=self.use_aug,
            **self.kwargs,
        )

    def build_features(self) -> Tuple[Callable, Dict]:
        cfg = dict(self.config.get("feature", {}))

        if bool(cfg.get("use_mae", False)):
            raise ValueError(
                "feature.use_mae must be false for cifar10_pixel: the MAE "
                "extractors are trained on SD-VAE latents of ImageNet and yield "
                "meaningless features from RGB pixels. Use use_convnext: true."
            )
        if not bool(cfg.get("use_convnext", True)):
            raise ValueError(
                "cifar10_pixel needs a feature extractor for the drift loss; "
                "set feature.use_convnext: true."
            )

        # ConvNeXt normalizes after postprocess, so it wants the unclipped map
        # back to [0, 1] -- clipping first would saturate before normalization.
        return build_activation_function(
            mae_path="",
            use_convnext=True,
            convnext_bf16=bool(cfg.get("convnext_bf16", True)),
            use_mae=False,
            postprocess_fn=cifar_postprocess(has_clip=False),
            convnext_model_name=str(cfg.get("convnext_model", "tiny")),
            feature_chunk_size=int(cfg.get("chunk_size", 64)),
            checkpoint_features=bool(cfg.get("checkpoint", False)),
            channels_last=bool(cfg.get("channels_last", False)),
        )
