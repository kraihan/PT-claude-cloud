from __future__ import annotations

import argparse
import copy
import gc
import os
import time
import statistics
import json
import numpy as np
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from einops import rearrange, repeat
from tqdm import tqdm

from dataset.dataset import infinite_sampler
from ptflow.drift import drift_loss
from ptflow.ot_drift import ot_drift_loss
from ptflow.memory_bank import ArrayMemoryBank
from pipelines import build_pipeline
from ptflow.potential import PotentialNet, ScaleNet
from ptflow.schedule import PTSchedule, build_schedule
from ptflow.train_steps import global_mean, pt_generator_terms, pt_potential_step
from utils.ckpt_util import restore_checkpoint, save_checkpoint, save_params_ema_artifact
from utils.dist_util import barrier, broadcast_module, maybe_ddp_model, process_count, process_index, unwrap_ddp
from utils.precision import amp_dtype
from utils.performance import StepTimer, adamw
from utils.feature_cache import FrozenFeatureCache
from utils.env import HF_ROOT
from utils.fid_util import evaluate_fid
from utils.init_util import maybe_init_state_params
from utils.logging import is_rank_zero, log_for_0
from utils.misc import load_config, profile_func, run_init, seed_everything
from utils.model_builder import build_model_dict, create_learning_rate_fn


run_init()


@dataclass
class PTBundle:
    """Everything the PT-Flow half of the run owns.

    Kept in one container hanging off TrainState so that the baseline training
    loop reads unchanged when it is absent (config has no ``pt:`` block, or
    ``pt.enabled: false``).
    """

    potential: torch.nn.Module
    ema_potential: torch.nn.Module
    optimizer: torch.optim.Optimizer
    sched: PTSchedule
    lr_fn: Any
    cfg: dict
    # The diagonal log-scale head lives here, on the theta side, and NOT as a
    # second head on the generator -- that would change the generator's
    # state_dict and break resuming from a baseline checkpoint.
    scale: Optional[torch.nn.Module] = None
    ema_scale: Optional[torch.nn.Module] = None
    ema_decay: float = 0.999


@dataclass
class TrainState:
    step: int
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    ema_model: torch.nn.Module
    ema_decay: float
    pt: Optional[PTBundle] = None
    scaler: Any = None
    extra_emas: dict = field(default_factory=dict)


def _generator_model_config(model) -> dict:
    model = unwrap_ddp(model)
    return {
        name: value
        for name, value in vars(model).items()
        if name not in {"parent", "name"} and not name.startswith("_")
    }


def _set_lr(optimizer: torch.optim.Optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg["lr"] = float(lr)


@torch.no_grad()
def _update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, ema_decay: float):
    model = unwrap_ddp(model)
    ema_params, params = list(ema_model.parameters()), list(model.parameters())
    torch._foreach_mul_(ema_params, ema_decay)
    torch._foreach_add_(ema_params, params, alpha=1.0 - ema_decay)
    for b_ema, b in zip(ema_model.buffers(), model.buffers()):
        b_ema.copy_(b)


def _to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_device(v, device) for v in x)
    return x


def train_step(
    state: TrainState,
    labels,
    samples,
    negative_samples,
    feature_params,
    feature_apply,
    learning_rate_fn: Any = None,
    cfg_min=1.0,
    cfg_max=4.0,
    neg_cfg_pw=1.0,
    no_cfg_frac=0.0,
    gen_per_label=8,
    activation_kwargs=dict(),
    loss_kwargs=dict(R_list=[0.02, 0.05, 0.2]),
    max_grad_norm=2.0,
    grad_accum_steps=1,
    device: torch.device = torch.device("cpu"),
    ot_mode: str = "none",
    ot_kwargs: dict | None = None,
    diverse_noise: bool = False,
    feature_chunk_size: int = 0,
    profile: bool = False,
    feature_cache=None,
    positive_ids=None,
    negative_ids=None,
):
    timer = StepTimer(device, profile)
    # ---- PT-Flow -------------------------------------------------------------
    # `state.pt` is None for a the plain baseline run and every PT branch below is a
    # no-op, so this function is unchanged in that case.
    pt = state.pt
    pt_cfg = dict(pt.cfg) if pt is not None else {}
    pt_eps = pt.sched.eps() if pt is not None else 0.0
    pt_metrics_acc: dict = {}

    labels = torch.as_tensor(labels, device=device, dtype=torch.long)

    _inline_feat = feature_chunk_size > 0 or int(os.environ.get("DRIFT_FEAT_CHUNK", "0")) > 0
    samples = torch.as_tensor(samples, device=device)
    negative_samples = torch.as_tensor(negative_samples, device=device)

    bsz = labels.shape[0]
    rng = torch.Generator(device=device)
    if diverse_noise:
        rng.manual_seed(int(state.step) * process_count() + process_index() + 1)
    else:
        rng.manual_seed(int(state.step) + 1)

    frac = torch.rand((bsz,), generator=rng, device=device)
    pw = 1.0 - float(neg_cfg_pw)
    if abs(pw) < 1e-6:
        cfg = torch.exp(torch.log(torch.tensor(float(cfg_min), device=device)) + frac * (torch.log(torch.tensor(float(cfg_max), device=device)) - torch.log(torch.tensor(float(cfg_min), device=device))))
    else:
        cfg = (cfg_min**pw + frac * (cfg_max**pw - cfg_min**pw)) ** (1.0 / pw)

    frac2 = torch.rand((bsz,), generator=rng, device=device)
    cfg = torch.where(frac2 < float(no_cfg_frac), torch.ones_like(cfg), cfg)

    n_pos = samples.shape[1]
    n_gen = int(gen_per_label)
    n_uncond = negative_samples.shape[1]

    def real_features(images, begin, end):
        if feature_cache is None:
            return feature_apply(feature_params, images, **activation_kwargs)
        if positive_ids is None or negative_ids is None:
            raise ValueError("Frozen feature caching requires memory-bank sample identities")
        ids = np.concatenate([positive_ids[begin:end], negative_ids[begin:end]], axis=1).reshape(-1)
        return feature_cache.get(feature_params, images, ids, feature_apply, **activation_kwargs)

    uncond_w = (cfg - 1.0) * (n_gen - 1) / max(1, n_uncond)

    # --- PT-Flow data term: capture real latents before the OT-drift baseline frees them ---
    pt_x1_pool = None
    pt_x1_labels = None
    if pt is not None:
        n_pool = int(pt_cfg.get("data_bsz", 32))
        flat = samples.reshape(-1, *samples.shape[2:])
        flat_labels = repeat(labels, "b -> (b x)", x=n_pos)
        sel = torch.randperm(flat.shape[0], device=device)[:n_pool]
        pt_x1_pool = flat[sel].detach().clone()
        pt_x1_labels = flat_labels[sel].clone()
        del flat, flat_labels

    # --- real features (full batch, no grad) ---
    timer.mark("inputs")
    sg_features = None
    if not _inline_feat:
        neg_samples_input = torch.cat([samples, negative_samples], dim=1)
        neg_samples_input = rearrange(neg_samples_input, "b x h w c -> (b x) h w c")
        with torch.no_grad():
            sg_features = real_features(neg_samples_input, 0, bsz)
            del neg_samples_input, samples, negative_samples
            sg_features = {k: rearrange(v, "(b x) f d -> b x f d", b=bsz, x=n_pos + n_uncond) for k, v in sg_features.items()}
    timer.mark("real_features")

    # --- learning rate ---
    lr = float(learning_rate_fn(state.step)) if learning_rate_fn is not None else state.optimizer.param_groups[0]["lr"]
    _set_lr(state.optimizer, lr)

    state.model.train()
    state.optimizer.zero_grad(set_to_none=True)

    # --- gradient accumulation loop ---
    grad_accum_steps = max(1, int(grad_accum_steps))
    chunk_size = (bsz + grad_accum_steps - 1) // grad_accum_steps
    actual_accum = (bsz + chunk_size - 1) // chunk_size

    total_loss_accum = torch.zeros((), device=device)
    total_info = {}

    for accum_idx in range(actual_accum):
        s = accum_idx * chunk_size
        e = min(s + chunk_size, bsz)
        if s >= bsz:
            break

        chunk_labels = labels[s:e]
        chunk_cfg = cfg[s:e]
        chunk_uncond_w = uncond_w[s:e]
        chunk_bsz = e - s

        if _inline_feat:
            _ci = torch.cat([samples[s:e], negative_samples[s:e]], dim=1)
            _ci = rearrange(_ci, "b x h w c -> (b x) h w c")
            with torch.no_grad():
                chunk_sg = real_features(_ci, s, e)
                del _ci
                chunk_sg = {k: rearrange(v, "(b x) f d -> b x f d", b=chunk_bsz, x=n_pos + n_uncond) for k, v in chunk_sg.items()}
        else:
            chunk_sg = {k: v[s:e] for k, v in sg_features.items()}
        timer.mark("real_features")

        input_labels = repeat(chunk_labels, "b -> (b g)", g=n_gen)
        input_cfg = repeat(chunk_cfg, "b -> (b g)", g=n_gen)

        use_no_sync = hasattr(state.model, "no_sync") and accum_idx < actual_accum - 1
        sync_ctx = state.model.no_sync() if use_no_sync else nullcontext()

        _use_ot = ot_mode == "debiased"
        _ot_kw = ot_kwargs or {}
        _use_new_cfg = _ot_kw.get("use_new_cfg", False)
        _resample_neg = _ot_kw.get("resample_neg", False) and _use_ot
        # A smaller INDEPENDENT negative batch retains two-batch transport.
        n_resample = int(_ot_kw.get("resample_gen_per_label", n_gen))
        if n_resample < 1:
            raise ValueError("resample_gen_per_label must be positive")

        resamp_features = None
        if _resample_neg:
            with torch.no_grad():
                neg_samples = unwrap_ddp(state.model)(
                    c=repeat(chunk_labels, "b -> (b g)", g=n_resample),
                    cfg_scale=repeat(chunk_cfg, "b -> (b g)", g=n_resample),
                    deterministic=False,
                    train=False,
                    rng=rng,
                )["samples"]
                timer.mark("negative_generation")
                resamp_features = feature_apply(feature_params, neg_samples, **activation_kwargs)
                del neg_samples
                resamp_features = {k: rearrange(v, "(b g) f d -> b g f d", b=chunk_bsz, g=n_resample) for k, v in resamp_features.items()}
                timer.mark("negative_features")

        with sync_ctx:
            gen_out = state.model(
                c=input_labels,
                cfg_scale=input_cfg,
                deterministic=False,
                train=True,
                rng=rng,
            )
            gen_samples = gen_out["samples"]
            timer.mark("generation")

            gen_features = feature_apply(feature_params, gen_samples, **activation_kwargs)
            gen_features = {k: rearrange(v, "(b g) f d -> b g f d", b=chunk_bsz, g=n_gen) for k, v in gen_features.items()}
            timer.mark("generated_features")

            chunk_loss = torch.zeros((), device=device)

            for k in chunk_sg.keys():
                feature_pos = chunk_sg[k][:, :n_pos]
                feature_uncond = chunk_sg[k][:, n_pos:]
                feature_gen = gen_features[k]

                feature_pos = rearrange(feature_pos, "b x f d -> (b f) x d")
                feature_gen = rearrange(feature_gen, "b x f d -> (b f) x d")
                feature_uncond = rearrange(feature_uncond, "b x f d -> (b f) x d")

                if _resample_neg:
                    feature_neg_detached = rearrange(resamp_features[k], "b x f d -> (b f) x d")
                else:
                    feature_neg_detached = feature_gen.detach()

                Bf = feature_gen.shape[0]
                weight_neg = repeat(chunk_uncond_w, "b -> (b f) k", f=Bf // max(1, chunk_uncond_w.shape[0]), k=n_uncond)

                if _use_ot:
                    ot_loss_kwargs = dict(loss_kwargs)
                    ot_loss_kwargs["sinkhorn_num_iter"] = _ot_kw.get("sinkhorn_num_iter", 20)
                    ot_loss_kwargs["sinkhorn_stop_thr"] = _ot_kw.get("sinkhorn_stop_thr", 1e-4)
                    ot_loss_kwargs["disable_diag_mask"] = _ot_kw.get("disable_diag_mask", False)
                    ot_loss_kwargs["batch_sinkhorn"] = _ot_kw.get("batch_sinkhorn", False)
                    ot_loss_kwargs["use_quadratic_cost"] = _ot_kw.get("use_quadratic_cost", False)
                    ot_loss_kwargs["reuse_costs"] = _ot_kw.get("reuse_costs", False)

                    if _use_new_cfg:
                        cfg_w = repeat(chunk_cfg - 1.0, "b -> (b f)", f=Bf // max(1, chunk_cfg.shape[0]))
                        loss_feat, info = ot_drift_loss(
                            gen=feature_gen,
                            fixed_pos=feature_pos,
                            fixed_neg=feature_neg_detached,
                            weight_gen=torch.ones_like(feature_gen[:, :, 0]),
                            weight_pos=torch.ones_like(feature_pos[:, :, 0]),
                            weight_neg=torch.ones_like(feature_neg_detached[:, :, 0]),
                            use_new_cfg=True,
                            fixed_uncond=feature_uncond,
                            weight_uncond=repeat(cfg_w, "b -> b 1"),
                            **ot_loss_kwargs,
                        )
                    else:
                        ot_neg = torch.cat([feature_neg_detached, feature_uncond], dim=1)
                        ot_neg_w = torch.cat([
                            torch.ones(Bf, feature_neg_detached.shape[1], device=device),
                            weight_neg,
                        ], dim=1)
                        loss_feat, info = ot_drift_loss(
                            gen=feature_gen,
                            fixed_pos=feature_pos,
                            fixed_neg=ot_neg,
                            weight_gen=torch.ones_like(feature_gen[:, :, 0]),
                            weight_pos=torch.ones_like(feature_pos[:, :, 0]),
                            weight_neg=ot_neg_w,
                            **ot_loss_kwargs,
                        )
                else:
                    loss_feat, info = drift_loss(
                        gen=feature_gen,
                        fixed_pos=feature_pos,
                        fixed_neg=feature_uncond,
                        weight_gen=torch.ones_like(feature_gen[:, :, 0]),
                        weight_pos=torch.ones_like(feature_pos[:, :, 0]),
                        weight_neg=weight_neg,
                        **loss_kwargs,
                    )

                chunk_loss = chunk_loss + loss_feat.mean()
                for k2, v2 in info.items():
                    key = f"{k2}/{k}"
                    if key not in total_info:
                        total_info[key] = v2.detach() if torch.is_tensor(v2) else torch.tensor(float(v2), device=device)
                    else:
                        total_info[key] = total_info[key] + (v2.detach() if torch.is_tensor(v2) else torch.tensor(float(v2), device=device))

            chunk_loss = chunk_loss / max(1, len(chunk_sg))
            timer.mark("ot_loss")

            # --- PT-Flow (A): prox-residual + scale-head terms ---------------
            # Additive on top of the OT drift.  lambda_prox is ramped in by
            # PTSchedule only after warmup and only while ESS certifies the
            # estimator, so this term cannot take the wheel before the potential
            # it depends on is trustworthy.  At lambda_prox = 0 it is exactly 0.
            if pt is not None and state.step % max(1, int(pt_cfg.get("prox_period", 1))) == 0:
                pt_extra, pt_m = pt_generator_terms(
                    unwrap_ddp(pt.potential),
                    gen_samples=gen_samples,
                    x0=gen_out["noise"]["x"],
                    labels=input_labels,
                    cfg=input_cfg,
                    sched=pt.sched,
                    eps=pt_eps,
                    max_batch=int(pt_cfg.get("prox_max_batch", 128)),
                    prox_mode=str(pt_cfg.get("prox_mode", "detach")),
                    prox_norm=str(pt_cfg.get("prox_norm", "none")),
                    generator=rng,
                )
                chunk_loss = chunk_loss + pt_extra
                for k2, v2 in pt_m.items():
                    pt_metrics_acc[k2] = pt_metrics_acc.get(k2, 0.0) + v2.detach()
            timer.mark("prox")

            chunk_weight = chunk_bsz / bsz
            scaled_loss = chunk_loss * chunk_weight
            if state.scaler is not None:
                state.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            total_loss_accum = total_loss_accum + chunk_loss.detach() * chunk_weight
            timer.mark("backward")

    if state.scaler is not None:
        state.scaler.unscale_(state.optimizer)
    g_norm = torch.nn.utils.clip_grad_norm_(state.model.parameters(), max_grad_norm, error_if_nonfinite=state.scaler is None)
    did_step = True
    if state.scaler is not None:
        old_scale = state.scaler.get_scale()
        state.scaler.step(state.optimizer)
        state.scaler.update()
        did_step = state.scaler.get_scale() >= old_scale
    else:
        state.optimizer.step()
    if did_step:
        _update_ema(state.ema_model, state.model, state.ema_decay)
        for decay, ema in state.extra_emas.items():
            _update_ema(ema, state.model, float(decay))
    timer.mark("optimizer_ema")

    metrics = {}
    for k, v in total_info.items():
        val = v / actual_accum
        metrics[k] = val.mean().detach() if torch.is_tensor(val) else torch.tensor(float(val), device=device)
    metrics["loss"] = total_loss_accum
    metrics["optimizer_step_skipped"] = float(not did_step)
    metrics["g_norm"] = torch.as_tensor(g_norm, device=device)
    metrics["lr"] = torch.tensor(lr, device=device)

    for k, v in pt_metrics_acc.items():
        metrics[k] = (v / actual_accum) if torch.is_tensor(v) else torch.as_tensor(v, device=device)
    if pt is not None and "pt/lambda_prox" not in metrics:
        metrics["pt/lambda_prox"] = 0.0

    # ---- PT-Flow (B) potential step, then (C) monitor and anneal -------------
    if pt is not None:
        batch_ess = None
        update_theta = pt.sched.update_potential()
        check_health = pt.sched.step % max(1, pt.sched.health_check_period) == 0
        if update_theta or check_health:
            pt_lr = float(pt.lr_fn(pt.sched.step)) if pt.lr_fn is not None else None
            with nullcontext() if update_theta else torch.no_grad():
                pt_m, batch_ess = pt_potential_step(
                pt.potential,
                pt.scale,
                pt.optimizer,
                state.model,
                x1=pt_x1_pool,
                labels_data=pt_x1_labels,
                labels_noise=labels[: int(pt_cfg.get("noise_bsz", 32))],
                sched=pt.sched,
                eps=pt_eps,
                K=int(pt_cfg.get("K", 8)),
                scale_K=int(pt_cfg.get("scale_K", 4)),
                p_uncond=float(pt_cfg.get("p_uncond", 0.1)),
                lambda_gauge=float(pt_cfg.get("lambda_gauge", 0.1)),
                lambda_mag=float(pt_cfg.get("lambda_mag", 0.0)),
                logw_clip=float(pt_cfg.get("logw_clip", 0.0)),
                max_grad_norm=float(pt_cfg.get("max_grad_norm", 1.0)),
                lr=pt_lr,
                chunk=int(pt_cfg.get("potential_chunk", 0)),
                curv_probe=bool(pt_cfg.get("curv_probe", False)),
                curv_allow=float(pt_cfg.get("curv_allow", 0.5)),
                rng=rng,
                device=device,
                update=update_theta,
            )
            if update_theta:
                _update_ema(pt.ema_potential, pt.potential, pt.ema_decay)
                if pt.scale is not None and pt.ema_scale is not None:
                    _update_ema(pt.ema_scale, pt.scale, pt.ema_decay)
            metrics.update(pt_m)

        # The health decision must be identical on every rank: it gates whether
        # the potential step runs at all and whether lambda_prox is nonzero, and
        # a disagreement would hang DDP on a mismatched backward.
        pt.sched.observe(global_mean(batch_ess, device) if batch_ess is not None else None)
        metrics.update(pt.sched.metrics(device=device))

    timer.mark("potential")
    metrics.update(timer.metrics())
    if feature_cache is not None:
        metrics.update(feature_cache.metrics())
    state.step += 1
    return state, metrics


@torch.no_grad()
def generate_step(batch, model, rng, postprocess_fn, cfg_scale=1.0, device: Optional[torch.device] = None):
    _, labels = batch
    labels = torch.as_tensor(labels, dtype=torch.long)
    if device is None:
        device = next(model.parameters()).device
    labels = labels.to(device)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(rng) + 1)

    latent_samples = model(
        c=labels,
        cfg_scale=cfg_scale,
        deterministic=True,
        train=False,
        rng=gen,
    )["samples"]
    return postprocess_fn(latent_samples)


def train_gen(
    model,
    optimizer,
    logger,
    eval_loader,
    train_loader,
    learning_rate_fn,
    preprocess_fn,
    postprocess_fn,
    dataset_name="imagenet256",
    train_batch_size=0,
    total_steps=100000,
    save_per_step=10000,
    eval_per_step=5000,
    eval_samples=50000,
    activation_fn=None,
    feature_params=None,
    ema_decay=0.999,
    seed=42,
    pos_per_sample=32,
    neg_per_sample=16,
    forward_dict=dict(
        gen_per_label=16,
        cfg_min=1.0,
        cfg_max=4.0,
        neg_cfg_pw=1.0,
        no_cfg_frac=0.0,
    ),
    positive_bank_size=64,
    negative_bank_size=512,
    cfg_list=(1.0,),
    activation_kwargs=dict(
        patch_mean_size=[2, 4],
        patch_std_size=[2, 4],
        use_std=True,
        use_mean=True,
        every_k_block=2,
    ),
    max_grad_norm=2.0,
    loss_kwargs=dict(R_list=(0.02, 0.05, 0.2)),
    keep_every=500000,
    keep_last=2,
    init_from="",
    push_per_step=0,
    push_at_resume=20,
    grad_accum_steps=1,
    workdir="runs",
    ot_mode="none",
    ot_kwargs=None,
    diverse_noise=False,
    pt_config=None,
    feature_chunk_size=0,
    eval_at_start=False,
    benchmark_steps=0,
    profile_every=0,
    resume_from="",
    init_ema_from="",
    feature_cache_gib=0.0,
    compile_generator=None,
    extra_ema_decays=(),
):
    if isinstance(ema_decay, (list, tuple)):
        if len(ema_decay) != 1:
            raise ValueError(f"Expected a single ema_decay value, got {ema_decay}")
        ema_decay = float(ema_decay[0])
    else:
        ema_decay = float(ema_decay)

    if cfg_list is None:
        cfg_list = [1.0]
    elif isinstance(cfg_list, (list, tuple)):
        cfg_list = [float(cfg) for cfg in cfg_list]
    else:
        cfg_list = [float(cfg_list)]

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    seed_everything(int(seed) + process_index())

    model = model.to(device)
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    # ---- PT-Flow bundle -----------------------------------------------------
    pt_bundle = None
    _pt_cfg = dict(pt_config) if pt_config else {}
    if _pt_cfg and bool(_pt_cfg.pop("enabled", False)):
        _pt_model_cfg = dict(_pt_cfg.pop("model", {}))
        _pt_scale_cfg = dict(_pt_cfg.pop("scale_model", {}))
        _pt_sched_cfg = dict(_pt_cfg.pop("schedule", {}))
        _pt_opt_cfg = dict(_pt_cfg.pop("optimizer", {}))
        _pt_init_gen = str(_pt_cfg.pop("init_generator_from", "") or "")

        _pt_model_cfg.setdefault("num_classes", int(getattr(model, "num_classes", 1000)))
        _pt_model_cfg.setdefault("input_size", int(getattr(model, "input_size", 32)))
        _pt_model_cfg.setdefault("in_channels", int(getattr(model, "in_channels", 4)))
        _pt_model_cfg.setdefault("cond_dim", int(getattr(model, "cond_dim", 768)))

        if _pt_init_gen:
            from ptflow.convert import load_baseline_into_generator

            load_baseline_into_generator(model, _pt_init_gen)
            ema_model.load_state_dict(model.state_dict())

        potential = PotentialNet(**_pt_model_cfg).to(device)
        broadcast_module(potential)
        ema_potential = copy.deepcopy(potential).to(device)
        ema_potential.eval()
        for p in ema_potential.parameters():
            p.requires_grad_(False)

        scale_net = ema_scale = None
        if str(_pt_cfg.get("scale_mode", "learned")).lower() != "none":
            _sc = dict(_pt_scale_cfg)
            for k in ("num_classes", "input_size", "in_channels", "cond_dim"):
                _sc.setdefault(k, _pt_model_cfg[k])
            scale_net = ScaleNet(**_sc).to(device)
            broadcast_module(scale_net)
            ema_scale = copy.deepcopy(scale_net).to(device)
            ema_scale.eval()
            for p in ema_scale.parameters():
                p.requires_grad_(False)

        _pt_lr_sched = dict(_pt_opt_cfg.pop("lr_schedule", {"learning_rate": 1e-4, "warmup_steps": 2000, "total_steps": int(total_steps), "lr_schedule": "const"}))
        pt_lr_fn = create_learning_rate_fn(**_pt_lr_sched)
        _theta_params = list(potential.parameters())
        if scale_net is not None:
            _theta_params += list(scale_net.parameters())
        pt_opt = adamw(
            _theta_params,
            fused=_pt_opt_cfg.get("fused", False),
            lr=pt_lr_fn(0),
            weight_decay=float(_pt_opt_cfg.get("weight_decay", 0.01)),
            betas=(float(_pt_opt_cfg.get("adam_b1", 0.9)), float(_pt_opt_cfg.get("adam_b2", 0.95))),
        )

        pt_bundle = PTBundle(
            potential=potential,
            ema_potential=ema_potential,
            optimizer=pt_opt,
            sched=build_schedule(_pt_sched_cfg),
            lr_fn=pt_lr_fn,
            cfg=_pt_cfg,
            scale=scale_net,
            ema_scale=ema_scale,
            ema_decay=float(_pt_cfg.get("ema_decay", ema_decay)),
        )
        log_for_0(
            "PT-Flow enabled: potential=%d params, scale=%d params, "
            "generator UNCHANGED (%d params, baseline-shaped); "
            "eps %.4g -> %.4g, lambda_prox_max=%.3g",
            sum(p.numel() for p in potential.parameters()),
            sum(p.numel() for p in scale_net.parameters()) if scale_net is not None else 0,
            sum(p.numel() for p in model.parameters()),
            pt_bundle.sched.eps_max,
            pt_bundle.sched.eps_min,
            pt_bundle.sched.lambda_prox_max,
        )

    _compile = device.type == "cuda" and (bool(compile_generator) if compile_generator is not None
                                         else os.environ.get("DRIFT_COMPILE", "0") != "0")
    if _compile and hasattr(model, "model"):
        if getattr(model.model, "use_remat", False):
            import torch._dynamo.config as _dynamo_config
            _dynamo_config.optimize_ddp = False
            log_for_0("Disabled DDPOptimizer (use_remat + torch.compile)")
        log_for_0("Compiling inner generator (LightningDiT) with torch.compile ...")
        model.model.compile(dynamic=True)

    model = maybe_ddp_model(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    opt = optimizer(model.parameters())

    # The potential is deliberately NOT wrapped in DDP.  DistributedDataParallel
    # only installs its reducer bookkeeping when its own ``forward`` is called,
    # and every PT-Flow call site invokes ``potential.phi(...)`` directly -- for
    # input-gradients, for the K proposal points, and for the data term, i.e.
    # several forwards before one backward, which is the pattern DDP handles
    # worst.  pt_potential_step all-reduces the gradients explicitly instead,
    # which is what DDP would have done, minus the fragility.

    state = TrainState(
        step=0, model=model, optimizer=opt, ema_model=ema_model,
        ema_decay=ema_decay, pt=pt_bundle,
    )
    for decay in extra_ema_decays:
        if not 0 < float(decay) < 1:
            raise ValueError("EMA decays must be in (0, 1)")
        if float(decay) != ema_decay:
            state.extra_emas[f"{float(decay):g}"] = copy.deepcopy(ema_model)
    raw_model = unwrap_ddp(model)
    if amp_dtype(device, raw_model.precision, raw_model.use_bf16) == torch.float16:
        state.scaler = torch.amp.GradScaler("cuda")
    if resume_from and init_ema_from:
        raise ValueError("Choose resume_from OR init_ema_from")
    local_ckpts = list((Path(workdir) / "checkpoints").glob("state_*.pt"))
    if (resume_from or init_ema_from) and local_ckpts:
        raise ValueError("Explicit resume/init needs a new workdir to protect existing checkpoints")
    state = restore_checkpoint(state=state, workdir=workdir, checkpoint=resume_from)
    if init_ema_from:
        from utils.ckpt_util import canonical_state_dict, check_model_behavior
        payload = torch.load(init_ema_from, map_location="cpu", weights_only=False)
        check_model_behavior(state.model, payload)
        if payload.get("ema_model") is None:
            raise ValueError("init_ema_from requires EMA weights in the source checkpoint")
        weights = canonical_state_dict(payload["ema_model"])
        unwrap_ddp(state.model).load_state_dict(weights, strict=True)
        state.ema_model.load_state_dict(weights, strict=True)
        for ema in state.extra_emas.values():
            ema.load_state_dict(weights, strict=True)
        log_for_0("Initialized from EMA at source step %s; new optimizer and step zero", payload.get("step"))
        del payload, weights
    if int(state.step) == 0 and init_from and not init_ema_from:
        log_for_0("Initializing generator params from init_from=%s", init_from)
        state = maybe_init_state_params(
            state,
            model_type="generator",
            init_from=init_from,
            hf_cache_dir=HF_ROOT,
        )
        for ema in state.extra_emas.values():
            ema.load_state_dict(state.ema_model.state_dict(), strict=True)

    assert feature_params is not None, "feature_params must be provided for feature extraction"

    log_for_0("Starting training loop (world_size=%d, grad_accum=%d)...", process_count(), grad_accum_steps)
    step = int(state.step)
    initial_step = step
    if benchmark_steps:
        total_steps = initial_step + int(benchmark_steps)
    pbar = tqdm(range(step, total_steps), initial=step, total=total_steps) if is_rank_zero() else range(step, total_steps)
    # The bank preallocates num_classes * max_size slots, so a hardcoded 1000
    # would waste ~1.5 GB of empty rows on a 10-class dataset.  Take the count
    # from the model that was actually built.
    _n_classes = int(getattr(unwrap_ddp(model), "num_classes", 1000))
    memory_bank_positive = ArrayMemoryBank(num_classes=_n_classes, max_size=positive_bank_size)
    memory_bank_negative = ArrayMemoryBank(num_classes=1, max_size=negative_bank_size)
    real_feature_cache = FrozenFeatureCache(float(feature_cache_gib) * 2**30) if feature_cache_gib > 0 else None
    train_iter = infinite_sampler(train_loader, step)
    _ot_kw = dict(ot_kwargs) if ot_kwargs else {}

    benchmark_times = []
    benchmark_phase_rows = []
    for step in pbar:
        if benchmark_steps and torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        n_push = 0
        logger.set_step(step)

        goal = push_per_step
        if initial_step > 0 and step == initial_step:
            goal = push_at_resume * push_per_step
            log_for_0("pushing at resume: %d", goal)

        while True:
            batch = next(train_iter)
            processed_batch = preprocess_fn(batch)
            images = processed_batch["images"]
            labels = processed_batch["labels"]
            ids = memory_bank_positive.add(images, labels)
            memory_bank_negative.add(images, labels * 0, ids=ids)
            n_push += int(images.shape[0])
            if n_push >= goal:
                break

        bsz_per_host = train_batch_size // max(1, process_count())
        if labels.shape[0] < bsz_per_host:
            raise ValueError(f"Labels shape {labels.shape[0]} < bsz_per_host {bsz_per_host}")

        perm = torch.randperm(labels.shape[0])[:bsz_per_host]
        labels_sel = labels[perm]

        positive_samples, positive_ids = memory_bank_positive.sample(labels_sel, n_samples=pos_per_sample, return_ids=True)
        negative_samples, negative_ids = memory_bank_negative.sample(labels_sel * 0, n_samples=neg_per_sample, return_ids=True)

        process_time = time.time() - start_time

        profile_metrics = {}
        if step == initial_step:
            profile_metrics = profile_func(
                lambda s, l, p, n, fp: train_step(
                    s, l, p, n, fp, activation_fn,
                    learning_rate_fn=learning_rate_fn,
                    activation_kwargs=activation_kwargs,
                    loss_kwargs=loss_kwargs,
                    max_grad_norm=max_grad_norm,
                    grad_accum_steps=grad_accum_steps,
                    device=device,
                    ot_mode=ot_mode,
                    ot_kwargs=_ot_kw,
                    diverse_noise=diverse_noise,
                    **forward_dict,
                ),
                (state, labels_sel, positive_samples, negative_samples, feature_params),
                name="train_step",
            )

        state, metrics = train_step(
            state,
            labels_sel,
            positive_samples,
            negative_samples,
            feature_params,
            activation_fn,
            learning_rate_fn=learning_rate_fn,
            activation_kwargs=activation_kwargs,
            loss_kwargs=loss_kwargs,
            max_grad_norm=max_grad_norm,
            grad_accum_steps=grad_accum_steps,
            device=device,
            ot_mode=ot_mode,
            ot_kwargs=_ot_kw,
            diverse_noise=diverse_noise,
            feature_chunk_size=feature_chunk_size,
            profile=bool(benchmark_steps or (profile_every and step % profile_every == 0)),
            feature_cache=real_feature_cache,
            positive_ids=positive_ids,
            negative_ids=negative_ids,
            **forward_dict,
        )

        if benchmark_steps and torch.cuda.is_available():
            torch.cuda.synchronize()
        total_time = time.time() - start_time
        if benchmark_steps:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                elapsed = torch.tensor(total_time, device=device, dtype=torch.float64)
                torch.distributed.all_reduce(elapsed, op=torch.distributed.ReduceOp.MAX)
                total_time = elapsed.item()
            benchmark_times.append(total_time)
            benchmark_phase_rows.append({k: float(v) for k, v in metrics.items() if k.startswith("profile/")})
        metrics["total_time"] = total_time
        metrics["process_time"] = process_time
        metrics["kimg"] = (step + 1) * positive_samples.shape[0] / 1000.0
        metrics["forward_kimg"] = (step + 1) * positive_samples.shape[0] / 1000.0 * forward_dict["gen_per_label"]
        metrics["global_generated_per_step"] = train_batch_size * forward_dict["gen_per_label"]
        metrics["global_independent_negatives_per_step"] = (train_batch_size * int(_ot_kw.get("resample_gen_per_label", forward_dict["gen_per_label"]))
                                                            if _ot_kw.get("resample_neg", False) and ot_mode == "debiased" else 0)
        metrics.update(profile_metrics)

        logger.log_dict(metrics)
        step += 1
        if benchmark_steps and step - initial_step >= benchmark_steps:
            measured = benchmark_times[min(5, len(benchmark_times) - 1):]
            summary = {"benchmark/median_step_s": statistics.median(measured),
                       "benchmark/mean_step_s": statistics.mean(measured),
                       "benchmark/measured_steps": len(measured)}
            summary["benchmark/generated_per_second"] = train_batch_size * forward_dict["gen_per_label"] / statistics.mean(measured)
            rows = benchmark_phase_rows[min(5, len(benchmark_phase_rows) - 1):]
            for key in rows[0]:
                summary[key] = statistics.mean(row.get(key, 0.0) for row in rows)
            summary.update(world_size=process_count(), torch_version=torch.__version__,
                           device=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                           initial_step=initial_step, global_generated_per_step=train_batch_size * forward_dict["gen_per_label"],
                           warmup_steps_discarded=min(5, len(benchmark_times) - 1),
                           first_step_s=benchmark_times[0], pt_enabled=state.pt is not None,
                           compile_generator=_compile)
            if torch.cuda.is_available():
                peak = torch.tensor(torch.cuda.max_memory_allocated() / 2**30, device=device)
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
                summary["benchmark/peak_allocated_gib"] = peak.item()
            logger.log_dict(summary)
            if is_rank_zero():
                print("Benchmark:", summary)
                Path(workdir).mkdir(parents=True, exist_ok=True)
                (Path(workdir) / "benchmark.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            break

        if not benchmark_steps and ((save_per_step > 0 and step % save_per_step == 0) or step == total_steps):
            save_checkpoint(state, keep=keep_last, keep_every=keep_every, workdir=workdir)
            if is_rank_zero():
                save_params_ema_artifact(
                    state,
                    workdir=workdir,
                    kind="gen",
                    model_config=_generator_model_config(state.model),
                )

        if eval_per_step > 0 and ((step % eval_per_step == 0) or (eval_at_start and step == 1) or (step == total_steps)):
            torch.cuda.empty_cache()
            is_sanity = step == 1
            n_samples = 500 if is_sanity else eval_samples
            folder_prefix = "sanity" if is_sanity else "CFG"
            round_best_fid = float("inf")
            round_best_cfg = cfg_list[0]
            eval_cfg_list = cfg_list if not is_sanity else [cfg_list[0]]

            for eval_cfg in eval_cfg_list:
                result = evaluate_fid(
                    dataset_name=dataset_name,
                    gen_func=generate_step,
                    gen_params={"model": state.ema_model, "cfg_scale": eval_cfg, "postprocess_fn": postprocess_fn, "device": device},
                    eval_loader=eval_loader,
                    logger=logger,
                    num_samples=n_samples,
                    log_folder=f"{folder_prefix}{eval_cfg}",
                    log_prefix=f"EMA_{state.ema_decay:g}",
                    rng_eval=0,
                )
                fid_val = result.get("fid", float("inf"))
                if fid_val < round_best_fid:
                    round_best_fid = fid_val
                    round_best_cfg = eval_cfg
            if not is_sanity:
                log_for_0("best_fid=%.4f best_cfg=%.1f (step=%d)", round_best_fid, round_best_cfg, step)
                if is_rank_zero():
                    logger.log_dict({"best_fid": round_best_fid, "best_cfg": round_best_cfg})

    logger.finish()
    del model, eval_loader, train_loader, state
    gc.collect()


def main_gen(config, output_dir="runs"):
    if "logging" not in config:
        config.logging = {}
    config.logging.name = Path(output_dir).resolve().name
    if is_rank_zero():
        root = Path(output_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        config_path = root / "config.json"
        if config_path.exists():
            config_path = root / f"config_launch_{time.time_ns()}.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    from models.generator import DitGen

    # Fix seed here
    train_seed = int(config.train.get("seed", 42))
    seed_everything(train_seed)

    # One pipeline object owns the data space, the feature extractor and the
    # FID reference; the trainer below is identical for every dataset.
    pipeline = build_pipeline(config)
    model_dict = build_model_dict(config, DitGen, workdir=output_dir, pipeline=pipeline)
    activation_fn, variables = pipeline.build_features()

    train_gen(
        model=model_dict.model,
        optimizer=model_dict.optimizer,
        logger=model_dict.logger,
        eval_loader=model_dict.eval_loader,
        train_loader=model_dict.train_loader,
        learning_rate_fn=model_dict.learning_rate_fn,
        preprocess_fn=model_dict.preprocess_fn,
        postprocess_fn=model_dict.postprocess_fn,
        dataset_name=model_dict.dataset_name,
        activation_fn=activation_fn,
        feature_params=variables,
        workdir=output_dir,
        pt_config=config.get("pt", None),
        **config.train,
    )


def main(args):
    run_init()
    config = load_config(args.config)
    if args.resume:
        config.train.resume_from = args.resume
    if args.init_ema:
        config.train.init_ema_from = args.init_ema
    main_gen(config, output_dir=args.workdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gen/baseline_ablation.yaml", help="Path to configuration file.")
    parser.add_argument("--workdir", type=str, default="runs", help="Local workdir root for checkpoints/logs.")
    parser.add_argument("--resume", default="", help="Full checkpoint resume into a new workdir.")
    parser.add_argument("--init-ema", default="", help="EMA-only initialization into a new workdir; optimizer/step reset.")
    args = parser.parse_args()
    args.output_dir = args.workdir

    main(args)
