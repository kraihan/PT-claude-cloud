"""PT-Flow inference: Modes A / B / C, and the exactly-normalized likelihood.

    Mode A (default, strict 1-NFE)
        x_hat = m_eta(x0, c, w).  One forward pass, no autograd, no potential.
        Within W2 = O(sqrt(eps d / (1 + lambda_bar))) of the model law by
        Proposition 2.4.  This path is unchanged from the baseline, which is the
        point: the fast sampler stays exactly as fast and exactly as good.

    Mode B (n-NFE refinement)
        Gradient descent on the prox residual.  A monotone quality/compute dial
        that requires no retraining -- the potential is already trained.

    Mode C (SNIS exact, K-NFE)
        Draw from the defensive proposal, self-normalize, resample one.  As
        K -> infinity this samples the exact bridge kernel and hence exactly
        rho_hat_1, the distribution the likelihood scores.  This is what makes
        the sampler/likelihood gap auditable rather than rhetorical.

Guidance bookkeeping.  Two different weights are in play and conflating them is
the easiest mistake to make here:

    cfg_scale   the baseline generator conditioning.  The codebase's convention is
                cfg_scale = w + 1, so cfg_scale = 1.0 means unguided.
    pt_w        the PT-Flow potential-arithmetic weight of eq. 2.20, used by
                Modes B and C:  phi^w = (1+w) phi(.,c) - w phi(.,null).

They default to being consistent (pt_w = cfg_scale - 1) but are exposed
separately, because Mode B/C guidance is a genuinely different mechanism -- an
exact bridge onto a w-sharpened marginal, with zero discretization error at any
w -- and is worth sweeping on its own.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from ptflow.estimator import (
    log_kernel_over_proposal,
    sample_proposal,
    snis_resample,
    tilted_phi0,
)
from ptflow.potential import guided_phi_grad


# ---------------------------------------------------------------------------

@torch.no_grad()
def _generator_output(
    generator,
    c: torch.Tensor,
    cfg_scale,
    x0: Optional[torch.Tensor] = None,
    rng: Optional[torch.Generator] = None,
    scale_net=None,
):
    """Run the generator, plus the separate scale net if one is supplied.

    The generator is the stock baseline and knows nothing about the log-scale field;
    that lives in ptflow.potential.ScaleNet on the theta side, which is what keeps
    the generator's state_dict checkpoint-compatible.  scale_net=None means
    S = I -- a valid, if higher-variance, proposal.
    """
    out = generator(
        c=c, cfg_scale=cfg_scale, deterministic=True, train=False,
        rng=rng, x0=x0,
    )
    m = out["samples"]
    x0_used = out["noise"]["x"]
    s = scale_net(m, c).detach() if scale_net is not None else None
    return m, x0_used, s


# ---------------------------------------------------------------------------
# Mode A
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_mode_a(
    generator,
    c: torch.Tensor,
    *,
    cfg_scale: float = 1.0,
    x0: Optional[torch.Tensor] = None,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Strict 1-NFE sampling.  Identical to the baseline sampler."""
    m, _, _ = _generator_output(generator, c, cfg_scale, x0=x0, rng=rng)
    return m


# ---------------------------------------------------------------------------
# Mode B
# ---------------------------------------------------------------------------

def sample_mode_b(
    generator,
    potential,
    c: torch.Tensor,
    *,
    cfg_scale: float = 1.0,
    pt_w: Optional[float] = None,
    n_steps: int = 4,
    gamma: float = 0.5,
    x0: Optional[torch.Tensor] = None,
    rng: Optional[torch.Generator] = None,
    return_trace: bool = False,
    backtracking: bool = True,
):
    """n-step refinement:  y <- y - gamma [grad phi^w(y) + y - x0].

    Weak convexity alone does not bound the largest Hessian eigenvalue, so a
    fixed gamma=0.5 need not converge. By default use per-sample backtracking
    on phi^w(y) + ||y-x0||^2/2. This controls the prox objective, not FID.
    """
    if pt_w is None:
        pt_w = float(cfg_scale) - 1.0
    if gamma <= 0 or n_steps < 0:
        raise ValueError("Require gamma > 0 and n_steps >= 0.")

    m, x0_used, _ = _generator_output(generator, c, cfg_scale, x0=x0, rng=rng)
    y = m.detach().clone()
    trace = []

    @torch.no_grad()
    def energy(point):
        value = potential.phi(point, c)
        if float(pt_w) != 0:
            value = (1 + float(pt_w)) * value - float(pt_w) * potential.phi(point, potential.null_labels(c))
        return value + 0.5 * (point - x0_used).flatten(1).square().sum(1)

    for _ in range(int(n_steps)):
        g = guided_phi_grad(potential, y, c, float(pt_w), create_graph=False)
        resid = g + y - x0_used
        if return_trace:
            trace.append(resid.detach().flatten(1).norm(dim=1).mean().item())
        if not torch.isfinite(resid).all():
            raise ValueError("Nonfinite prox residual during refinement.")
        if backtracking:
            before = energy(y)
            step_size = torch.full((len(y),), float(gamma), device=y.device)
            norm2 = resid.flatten(1).square().sum(1)
            accepted = torch.zeros(len(y), dtype=torch.bool, device=y.device)
            next_y = y.clone()
            for _ in range(12):
                candidate = y - step_size.view(-1, *([1] * (y.ndim - 1))) * resid
                after = energy(candidate)
                good = torch.isfinite(after) & (after <= before - 1e-4 * step_size * norm2)
                take = good & ~accepted
                next_y = torch.where(take.view(-1, *([1] * (y.ndim - 1))), candidate, next_y)
                accepted |= good
                if bool(accepted.all()):
                    break
                step_size = torch.where(accepted, step_size, step_size * 0.5)
            y = next_y
        else:
            y = y - float(gamma) * resid

    if return_trace:
        g = guided_phi_grad(potential, y, c, float(pt_w), create_graph=False)
        trace.append((g + y - x0_used).detach().flatten(1).norm(dim=1).mean().item())
        return y.detach(), trace
    return y.detach()


# ---------------------------------------------------------------------------
# Mode C
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_mode_c(
    generator,
    potential,
    c: torch.Tensor,
    eps: float,
    *,
    cfg_scale: float = 1.0,
    pt_w: Optional[float] = None,
    K: int = 32,
    alpha_def: float = 0.05,
    x0: Optional[torch.Tensor] = None,
    rng: Optional[torch.Generator] = None,
    scale_net=None,
    return_ess: bool = False,
):
    """SNIS exact sampler.  K potential evaluations, one generator evaluation.

    Note the guidance route: the weights use phi^w, so the resampled point is a
    draw from the *guided* bridge kernel.  Proposition 2.10 makes that exact --
    phi^w is an admissible data-side potential, so every result of the method
    applies verbatim with phi -> phi^w.
    """
    if pt_w is None:
        pt_w = float(cfg_scale) - 1.0

    m, x0_used, s = _generator_output(
        generator, c, cfg_scale, x0=x0, rng=rng, scale_net=scale_net
    )

    y = sample_proposal(x0_used, m, s, eps, int(K), float(alpha_def), generator=rng)
    log_ratio = log_kernel_over_proposal(y, x0_used, m, s, eps, float(alpha_def))

    B, Kk = y.shape[0], y.shape[1]
    tail = y.shape[2:]
    y_flat = y.reshape(B * Kk, *tail)
    c_rep = c.repeat_interleave(Kk, dim=0)

    if float(pt_w) == 0.0:
        phi_y = potential.phi(y_flat, c_rep).view(B, Kk)
    else:
        phi_c = potential.phi(y_flat, c_rep)
        phi_u = potential.phi(y_flat, potential.null_labels(c_rep))
        phi_y = ((1.0 + float(pt_w)) * phi_c - float(pt_w) * phi_u).view(B, Kk)

    phi_ref = phi_y.mean(dim=1, keepdim=True)
    log_w = log_ratio - (phi_y - phi_ref) / (2.0 * float(eps))

    out = snis_resample(y, log_w, generator=rng)
    if return_ess:
        lse1 = torch.logsumexp(log_w, dim=1)
        lse2 = torch.logsumexp(2.0 * log_w, dim=1)
        ess = torch.exp(2.0 * lse1 - lse2 - math.log(Kk)).clamp(0.0, 1.0)
        return out, ess
    return out


# ---------------------------------------------------------------------------
# Exactly-normalized likelihood
# ---------------------------------------------------------------------------

@torch.no_grad()
def log_likelihood(
    generator,
    potential,
    x1: torch.Tensor,
    c: torch.Tensor,
    eps: float,
    *,
    K_outer: int = 16,
    K_inner: int = 16,
    alpha_def: float = 0.05,
    scale_net=None,
    rng: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, dict]:
    """log rho_hat_1(x) = log psi_hat_1(x) - u_theta(x), Theorem 2.5.

        psi_hat_1(x) = E_z[ (rho_0 / psi_0)(x + sqrt(2 eps) z) ]

    with every inner psi_0 evaluated by the tilted estimator.  Two Monte-Carlo
    layers make this a biased nested Monte Carlo estimate. The reciprocal of
    the estimated inner psi0 is biased, so an IWAE lower-bound or monotone
    tightening guarantee does not follow from the outer logarithm. Check
    convergence in BOTH K_outer and K_inner, at w = 0.

    Two caveats worth stating in any table produced from this:
      * The value is a density in the training space. A lossy stochastic VAE
        needs an observation model to define a pixel likelihood; there is no
        invertible change-of-variables Jacobian for this decoder.
      * Poor proposals can bias the finite nested estimate in either direction.
    """
    B = x1.shape[0]
    tail = x1.shape[1:]
    d = float(x1[0].numel())
    sqrt2e = math.sqrt(2.0 * float(eps))
    J = int(K_outer)

    z = torch.randn((B, J, *tail), generator=rng, device=x1.device, dtype=x1.dtype)
    v = x1.unsqueeze(1) + sqrt2e * z                      # [B,J,...]
    v_flat = v.reshape(B * J, *tail)
    c_rep = c.repeat_interleave(J, dim=0)

    # inner: phi_0(v) via the tilted estimator, proposal centred on m_eta(v)
    m, _, s = _generator_output(
        generator, c_rep, 1.0, x0=v_flat, rng=rng, scale_net=scale_net
    )
    est = tilted_phi0(
        potential, v_flat, c_rep, m, s, eps,
        K=int(K_inner), alpha_def=float(alpha_def), generator=rng, logw_clip=0.0,
    )
    phi0_v = est.phi0.view(B, J)

    # log(rho_0 / psi_0)(v) = log rho_0(v) + phi_0(v) / (2 eps)
    log_rho0 = -0.5 * v.reshape(B, J, -1).pow(2).sum(-1) - 0.5 * d * math.log(2.0 * math.pi)
    inner = log_rho0 + phi0_v / (2.0 * float(eps))

    log_psi1_hat = torch.logsumexp(inner, dim=1) - math.log(J)
    log_p = log_psi1_hat - potential.phi(x1, c) / (2.0 * float(eps))

    info = {
        "estimator": "nested_monte_carlo_not_a_certified_bound",
        "nll_nats": (-log_p).mean().item(),
        "nll_per_dim": (-log_p).mean().item() / d,
        "bits_per_dim_latent": (-log_p).mean().item() / (d * math.log(2.0)),
        "inner_ess": est.ess.mean().item(),
        "K_outer": J,
        "K_inner": int(K_inner),
        "eps": float(eps),
    }
    return log_p, info
