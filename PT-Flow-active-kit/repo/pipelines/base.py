"""The dataset pipeline contract.

PT-Flow's method core (`ptflow/`) and its backbones (`models/`) are
dataset-agnostic: the potential, the tilted estimator, the schedule and the DiT
are all parameterized by `(input_size, in_channels)` and know nothing about
where the data came from.  Everything that *is* dataset-specific lives behind
this interface:

    * how the data is loaded, and what space it lives in
      (SD-VAE latents at 32x32x4, or raw RGB at 32x32x3)
    * which feature extractor the drift loss is computed against
      (a latent-space MAE, or a pixel-space ConvNeXt)
    * which FID reference statistics apply

Those three travel together and cannot be mixed: a latent-space MAE produces
meaningless features from RGB pixels, and ImageNet FID statistics say nothing
about CIFAR-10 samples.  Bundling them in one object is what stops a
half-matched combination from being expressible.

Adding a dataset means adding one module here and registering it -- not
threading a `name` flag through the trainer.
"""

from __future__ import annotations

import abc
from typing import Any, Callable, Dict, Tuple


class Pipeline(abc.ABC):
    """One dataset, one data space, one feature extractor, one FID reference."""

    #: registry key, matched against `pipeline:` in the config
    name: str = ""
    #: what utils/fid_util._canonical_dataset_name() resolves
    fid_dataset_name: str = ""
    #: spatial size and channel count of the space the model actually operates in
    input_size: int = 0
    in_channels: int = 0
    num_classes: int = 0
    #: human-readable one-liner, printed at startup
    description: str = ""

    def __init__(self, config):
        self.config = config

    # -- data ---------------------------------------------------------------

    @abc.abstractmethod
    def build_split(self, *, split: str, batch_size: int) -> Tuple[Any, Callable, Callable]:
        """Return ``(loader, preprocess_fn, postprocess_fn)`` for one split.

        The tensor contract is fixed across pipelines, because the trainer
        depends on it:

            preprocess_fn(batch)   -> {"images": [B, H, W, C], "labels": [B]}
            postprocess_fn(images) -> [B, C, H, W] in [0, 1]

        For a latent pipeline "images" are latents and postprocess decodes them;
        for a pixel pipeline they are the pixels themselves.
        """

    # -- features for the drift loss ----------------------------------------

    @abc.abstractmethod
    def build_features(self) -> Tuple[Callable, Dict]:
        """Return ``(activation_fn, feature_params)`` for the drift loss.

        The drift loss is computed *entirely* in this feature space -- it never
        touches raw samples -- so this choice is not cosmetic.
        """

    # -- consistency ---------------------------------------------------------

    def validate(self, model_cfg) -> None:
        """Fail at startup if the model config disagrees with the data space.

        Catching this here turns a silent 200-step-later shape error, or worse a
        run that trains against nonsense, into one line at launch.
        """
        problems = []
        got_size = int(model_cfg.get("input_size", -1))
        got_ch = int(model_cfg.get("in_channels", -1))
        got_out = int(model_cfg.get("out_channels", got_ch))
        if got_size != self.input_size:
            problems.append(f"model.input_size={got_size}, pipeline expects {self.input_size}")
        if got_ch != self.in_channels:
            problems.append(f"model.in_channels={got_ch}, pipeline expects {self.in_channels}")
        if got_out != self.in_channels:
            problems.append(f"model.out_channels={got_out}, pipeline expects {self.in_channels}")

        # LightningDiTBlock discards its cond_dim and builds adaLN as
        # Linear(hidden_size, 6*hidden_size), so the two must agree.
        cond, hidden = int(model_cfg.get("cond_dim", -1)), int(model_cfg.get("hidden_size", -1))
        if cond != hidden:
            problems.append(
                f"model.cond_dim={cond} must equal model.hidden_size={hidden} "
                "(LightningDiTBlock builds adaLN from hidden_size)"
            )

        if problems:
            raise ValueError(
                f"config does not match the '{self.name}' pipeline:\n  - "
                + "\n  - ".join(problems)
            )

    def summary(self) -> str:
        d = self.input_size * self.input_size * self.in_channels
        return (
            f"pipeline '{self.name}': {self.description}\n"
            f"  data space   {self.input_size}x{self.input_size}x{self.in_channels}  (d = {d})\n"
            f"  classes      {self.num_classes}\n"
            f"  FID stats    {self.fid_dataset_name}"
        )
