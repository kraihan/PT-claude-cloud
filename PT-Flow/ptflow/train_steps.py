"""The two PT-Flow training steps, packaged so train.py stays readable.

The the OT-drift baseline training loop is left structurally intact -- that is the "arm".  Both
functions here are additive:

    pt_generator_terms   returns an extra scalar to add into the drift arm's existing
                         per-chunk loss.  At lambda_prox = 0 it returns exactly
                         zero and costs nothing, so the run is bit-for-bit
                         the OT-drift baseline.

    pt_potential_step    a self-contained forward/backward/step on the
                         potential's own optimizer.  It never touches the
                         generator's parameters or optimizer.

Ordering follows Algorithm 1: generator step (A) at frozen theta, then potential
step (B) at frozen eta, then monitor and anneal (C).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from ptflow.losses import curvature_hinge, potential_loss, prox_loss, scale_loss
from ptflow.schedule import PTSchedule
from utils.dist_util import dist_is_initialized, unwrap_ddp


def _subsample(n_total: int, n_keep: int, device, generator=None) -> torch.Tensor:
    if n_keep <= 0 or n_keep >= n_total:
        return torch.arange(n_total, device=device)
    return torch.randperm(n_total, generator=generator, device=device)[:n_keep]


def allreduce_grads_(module: torch.nn.Module) -> None:
    """Average parameter gradients across ranks, in place.

    Stands in for DDP, which is not usable here: it installs its reducer only
    when its own ``forward`` runs, whereas every PT-Flow call site goes through
    ``potential.phi(...)``.  Without this the potential would train on
    rank-local gradients and the ranks would silently diverge.
    """
    if not dist_is_initialized():
        return
    import torch.distributed as dist

    world = float(dist.get_world_size())
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    if not grads:
        return
    flat = torch._utils._flatten_dense_tensors(grads)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(world)
    for g, synced in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
        g.copy_(synced)


# ---------------------------------------------------------------------------
# (A) extra generator terms
# ---------------------------------------------------------------------------

def pt_generator_terms(
    potential,
    *,
    gen_samples: torch.Tensor,
    x0: torch.Tensor,
    labels: torch.Tensor,
    cfg: torch.Tensor,
    sched: PTSchedule,
    eps: float,
    max_batch: int = 128,
    prox_mode: str = "detach",
    prox_norm: str = "none",
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """The prox-residual term to add to the baseline generator loss.

    This is the ONLY thing PT-Flow adds to the generator's objective.  The
    scale head moved to the potential step, because it now lives on the theta
    side (ptflow.potential.ScaleNet) -- which is what keeps the generator's
    state_dict baseline-shaped and therefore checkpoint-compatible.

    Args:
        gen_samples: [N, H, W, C] the generator's output m_eta(x0, c, w).
        x0: [N, ...] the noise those samples came from.
        labels: [N] class labels.
        cfg: [N] the OT-drift baseline cfg scale (= w + 1), converted to the paper's w here.
        max_batch: cap on how many samples the prox term is evaluated on.  The
            prox term needs an input-gradient pass through the potential per
            sample, and the OT-drift baseline generates gen_per_label * batch samples per step
            (thousands).  Subsampling only adds variance to a regression target.

    Returns ``(extra_loss, metrics)``.  ``extra_loss`` is already weighted.
    """
    device = gen_samples.device
    lam = sched.lambda_prox()
    # Always report the weight that was *actually applied* this step.  The
    # schedule's own metrics are read after observe() has advanced the ramp, so
    # they are one step ahead; logging both makes the ramp legible instead of
    # looking off-by-one in the dashboard.
    used = {"pt/lambda_prox": torch.as_tensor(lam, device=device)}
    if lam <= 0.0:
        return torch.zeros((), device=device), used

    idx = _subsample(gen_samples.shape[0], int(max_batch), device, generator)
    m = gen_samples[idx]
    x0_s = x0[idx].detach()
    c_s = labels[idx]
    # the "cfg scale" is (w + 1); the paper's guidance weight is w >= 0.
    w_s = (cfg[idx].float() - 1.0).clamp_min(0.0)

    total = torch.zeros((), device=device)
    metrics: Dict[str, torch.Tensor] = dict(used)

    if lam > 0.0:
        # prox_mode="full" builds a graph through the potential's parameters, so
        # the generator's backward leaves stray gradients in potential.grad.
        # They are harmless: pt_potential_step zero_grads before its own
        # backward, and it always runs after this.  Mentioned because it looks
        # like a leak in a profiler.
        lp, mp = prox_loss(potential, m, x0_s, c_s, w_s, mode=prox_mode, norm=prox_norm)
        total = total + lam * lp
        metrics.update(mp)

    return total, metrics


# ---------------------------------------------------------------------------
# (B) potential step
# ---------------------------------------------------------------------------

def pt_potential_step(
    potential,
    scale_net,
    potential_opt: torch.optim.Optimizer,
    generator_model,
    *,
    x1: torch.Tensor,
    labels_data: torch.Tensor,
    labels_noise: torch.Tensor,
    sched: PTSchedule,
    eps: float,
    K: int = 8,
    scale_K: int = 4,
    p_uncond: float = 0.1,
    lambda_gauge: float = 0.1,
    lambda_mag: float = 0.0,
    logw_clip: float = 0.0,
    max_grad_norm: float = 1.0,
    lr: Optional[float] = None,
    chunk: int = 0,
    curv_probe: bool = False,
    curv_allow: float = 0.5,
    rng: Optional[torch.Generator] = None,
    device: torch.device = torch.device("cpu"),
    update: bool = True,
) -> Tuple[Dict[str, torch.Tensor], float]:
    """One theta update on the flat objective.  Returns (metrics, batch ESS).

    The generator is frozen and evaluated at guidance weight zero -- the rule of
    Appendix F.1.  Maximum likelihood is valid only against real data, so
    guidance exists solely in the generator's conditioning and at sampling time.

    Conditioning dropout is applied to the *potential's* label only, not to the
    generator's.  the baseline generator has no null-class token (its CFG lives in
    the cfg_scale conditioning), so the unconditional potential's proposal comes
    from the class-conditional generator.  Importance sampling is unbiased for
    any covering proposal, so this costs variance, not correctness -- and the
    ESS reports exactly how much.
    """
    B = int(labels_noise.shape[0])
    gen = unwrap_ddp(generator_model)

    # -- proposal from the frozen generator, at w = 0 ----------------------
    with torch.no_grad():
        gen.eval()
        out = gen(
            c=labels_noise, cfg_scale=1.0, deterministic=True, train=False,
            rng=rng, x0=None,
        )
        m0 = out["samples"].detach().float()
        x0 = out["noise"]["x"].detach().float()
        gen.train()

    # -- conditioning dropout (potential side only) ------------------------
    pot = unwrap_ddp(potential)
    c0 = pot.drop_cond(labels_noise, p_uncond, generator=rng)
    c1 = pot.drop_cond(labels_data, p_uncond, generator=rng)

    # The proposal's diagonal scale, evaluated at the prox point m0 -- the point
    # where S^-1 = I + grad^2 phi(y*) is defined.  Detached for the estimator
    # (the tilt must not be a function the potential can game) and re-evaluated
    # with grad below for its own reverse-KL fit.
    s0 = None
    if scale_net is not None:
        with torch.no_grad():
            s0 = unwrap_ddp(scale_net)(m0, c0).detach().float()

    if update and lr is not None:
        for pg in potential_opt.param_groups:
            pg["lr"] = float(lr)

    potential.train()
    if update:
        potential_opt.zero_grad(set_to_none=True)

    loss, est, metrics = potential_loss(
        pot, x0, c0, x1.float(), c1, m0, s0, eps,
        K=int(K), alpha_def=sched.alpha_def(),
        lambda_gauge=float(lambda_gauge), lambda_mag=float(lambda_mag),
        logw_clip=float(logw_clip), generator=rng, chunk=int(chunk),
    )

    # Raw normalized ESS has floor 1/K: with K=8 a collapsed estimate is
    # 0.125, so the old 0.05 'broken' threshold was unreachable.
    control_ess = ((est.ess - 1.0 / K) / (1.0 - 1.0 / K)).clamp(0, 1) if K > 1 else torch.zeros_like(est.ess)
    metrics["pt/control_ess"] = control_ess.mean().detach()
    if not update:
        return metrics, float(control_ess.mean().item())

    # -- scale head: reverse KL at the same prox points ---------------------
    if scale_net is not None and sched.lambda_scale > 0.0:
        s_grad = unwrap_ddp(scale_net)(m0, c0)
        ls, ms = scale_loss(
            pot, m0, s_grad, x0, c0, eps, K=int(scale_K), generator=rng,
        )
        loss = loss + float(sched.lambda_scale) * ls
        metrics.update(ms)

    if curv_probe and sched.lambda_curv > 0.0:
        pen, mc = curvature_hinge(
            pot, m0, c0, lambda_allow=float(curv_allow), generator=rng
        )
        loss = loss + float(sched.lambda_curv) * pen
        metrics.update(mc)

    loss.backward()
    allreduce_grads_(pot)
    theta_params = list(pot.parameters())
    if scale_net is not None:
        sn = unwrap_ddp(scale_net)
        allreduce_grads_(sn)
        theta_params += list(sn.parameters())
    gnorm = torch.nn.utils.clip_grad_norm_(theta_params, float(max_grad_norm), error_if_nonfinite=True)
    potential_opt.step()

    metrics["pt/g_norm_potential"] = torch.as_tensor(gnorm, device=device)
    if lr is not None:
        metrics["pt/lr_potential"] = torch.as_tensor(float(lr), device=device)

    return metrics, float(control_ess.mean().item())


# ---------------------------------------------------------------------------
# (C) monitor
# ---------------------------------------------------------------------------

def global_mean(value: float, device: torch.device) -> float:
    """All-reduce a scalar so every rank makes the same scheduling decision.

    This is not cosmetic.  ``PTSchedule.update_potential()`` can skip the
    potential step, and lambda_prox changes which parameters receive gradients.
    If ranks disagreed, DDP would hang on the first mismatched backward.
    """
    if not dist_is_initialized():
        return float(value)
    import torch.distributed as dist

    t = torch.as_tensor([float(value)], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item()) / float(dist.get_world_size())
