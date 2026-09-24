from __future__ import annotations

import math
from pathlib import Path

import torch

from pipelines import build_pipeline
from utils.dist_util import process_count
from utils.logging import WandbLogger
from utils.misc import EasyDict
from utils.performance import adamw


def create_learning_rate_fn(
    learning_rate,
    warmup_steps,
    total_steps,
    lr_schedule="const",
):
    learning_rate = float(learning_rate)
    warmup_steps = int(warmup_steps)
    total_steps = int(total_steps)

    def warmup(step: int) -> float:
        if warmup_steps <= 0:
            return learning_rate
        t = min(max(step, 0), warmup_steps)
        return 1e-6 + (learning_rate - 1e-6) * (t / max(1, warmup_steps))

    def main(step: int) -> float:
        if lr_schedule in ["cosine", "cos"]:
            cosine_steps = max(total_steps - warmup_steps, 1)
            t = min(max(step - warmup_steps, 0), cosine_steps)
            alpha = 1e-6
            cosine = 0.5 * (1 + math.cos(math.pi * t / cosine_steps))
            return learning_rate * ((1 - alpha) * cosine + alpha)
        if lr_schedule == "const":
            return learning_rate
        raise NotImplementedError(lr_schedule)

    def schedule_fn(step: int) -> float:
        if step < warmup_steps:
            return warmup(step)
        return main(step)

    return schedule_fn


def build_model_dict(config, model_class, *, workdir: str = "runs", pipeline=None):
    """Assemble model, data, optimizer and logger for one run.

    The dataset lives behind a Pipeline (see pipelines/base.py), so this
    function is the same for CIFAR-10 pixels and ImageNet latents: it asks the
    pipeline for splits and never learns which one it got.
    """
    pipeline = pipeline or build_pipeline(config)
    print(pipeline.summary())

    # Fail now if the backbone config disagrees with the data space, rather than
    # on a shape error hundreds of steps in.
    pipeline.validate(config.model)

    print("Building model...")
    model = model_class(
        num_classes=config.dataset.num_classes,
        **config.model,
    )

    print("Building dataset...")
    world = max(1, process_count())
    for name, size in (("dataset.batch_size", config.dataset.batch_size),
                       ("dataset.eval_batch_size", config.dataset.eval_batch_size),
                       ("train.train_batch_size", config.train.train_batch_size)):
        if size < world or size % world:
            raise ValueError(f"{name}={size} must be positive and divisible by world_size={world}.")
    train_loader, preprocess_fn, postprocess_fn = pipeline.build_split(
        split="train", batch_size=config.dataset.batch_size // world,
    )
    eval_loader, _, _ = pipeline.build_split(
        split="val", batch_size=config.dataset.eval_batch_size // world,
    )

    learning_rate_fn = create_learning_rate_fn(**config.optimizer.lr_schedule)

    def optimizer_builder(params):
        return adamw(
            params,
            fused=config.optimizer.get("fused", False),
            lr=learning_rate_fn(0),
            weight_decay=float(config.optimizer.get("weight_decay", 0.0)),
            betas=(float(config.optimizer.adam_b1), float(config.optimizer.adam_b2)),
        )

    logger = WandbLogger()
    w_cfg = EasyDict(dict(config.get("logging", {})))
    use_wandb = bool(w_cfg.get("use_wandb", config.get("use_wandb", True)))
    if "use_wandb" in w_cfg:
        del w_cfg["use_wandb"]
    output_root = Path(workdir).resolve()
    logger.set_logging(
        config=config,
        use_wandb=use_wandb,
        workdir=str(output_root),
        **w_cfg,
    )

    return EasyDict(
        model=model,
        optimizer=optimizer_builder,
        logger=logger,
        eval_loader=eval_loader,
        train_loader=train_loader,
        dataset_name=pipeline.fid_dataset_name,
        preprocess_fn=preprocess_fn,
        postprocess_fn=postprocess_fn,
        train=config.train,
        learning_rate_fn=learning_rate_fn,
        pipeline=pipeline,
        feature=config.get("feature", {}),
    )
