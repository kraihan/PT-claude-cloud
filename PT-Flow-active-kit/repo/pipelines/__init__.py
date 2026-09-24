"""Dataset pipeline registry.

    pipeline = build_pipeline(config)

Selected by the top-level `pipeline:` key in a config:

    pipeline: cifar10_pixel      # 32x32x3 RGB, ConvNeXt features
    pipeline: imagenet_latent    # 32x32x4 SD-VAE latents, MAE features

The method core (`ptflow/`) and the backbones (`models/`) are shared unchanged
across both; only what is genuinely dataset-specific lives behind the Pipeline
interface.  See pipelines/base.py for why the data space, the feature extractor
and the FID statistics are bundled rather than configured independently.
"""

from __future__ import annotations

from pipelines.base import Pipeline
from pipelines.cifar10_pixel import Cifar10PixelPipeline
from pipelines.imagenet_latent import ImageNetLatentPipeline

PIPELINES = {
    Cifar10PixelPipeline.name: Cifar10PixelPipeline,
    ImageNetLatentPipeline.name: ImageNetLatentPipeline,
}


def _infer_name(config) -> str:
    """Pick a pipeline for configs written before `pipeline:` existed."""
    ds = config.get("dataset", {})
    name = str(ds.get("name", "")).lower()
    if name.startswith("cifar"):
        return Cifar10PixelPipeline.name
    if bool(ds.get("use_latent", False)) or bool(ds.get("use_cache", False)):
        return ImageNetLatentPipeline.name
    raise ValueError(
        "Config has no `pipeline:` key and none could be inferred. Add one of: "
        + ", ".join(sorted(PIPELINES))
    )


def build_pipeline(config) -> Pipeline:
    name = str(config.get("pipeline", "") or "").strip().lower()
    if not name:
        name = _infer_name(config)
    if name not in PIPELINES:
        raise ValueError(
            f"unknown pipeline {name!r}; available: {', '.join(sorted(PIPELINES))}"
        )
    return PIPELINES[name](config)


__all__ = ["Pipeline", "PIPELINES", "build_pipeline",
           "Cifar10PixelPipeline", "ImageNetLatentPipeline"]
