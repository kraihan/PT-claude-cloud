"""The prox-tilted importance estimator -- Section 2.5 of the PT-Flow paper.

What this module computes
-------------------------
The single Gaussian expectation that propagates the potential across all of
time (eq. 2.4),

    psi_0(x0) = E_{z ~ N(0,I)} [ exp(-u_theta(x0 + sqrt(2 eps) z)) ],

evaluated by importance sampling against the Laplace-matched proposal

    q_x   = N( m(x0), 2 eps diag(exp(s(x0))) ),   m ~= prox_phi(x0)
    q_def = (1 - alpha) q_x + alpha N(x0, 2 eps I).

Three implementation choices deserve their own note, because each fixes a
failure mode that a literal transcription of the paper walks straight into.

1.  We return  phi0_hat = -2 eps log psi0_hat  (the estimated Moreau envelope)
    rather than log psi0_hat, and the training objective is taken in phi-units:

        2 eps * L(theta)  =  E_{x1}[ phi(x1) ]  -  E_{x0}[ phi_0(x0) ].

    This is a positive rescaling, so the minimizer is unchanged, but the loss
    value and its gradient become O(d) and O(1) respectively instead of O(d/eps)
    and O(1/eps) -- and, crucially, they stop moving by 200x as eps anneals.

2.  The k-independent part of phi is factored out of the logsumexp before the
    division by 2 eps, using the identity (exact for any constant c)

        logsumexp_k(A_k - phi_k/2eps) = logsumexp_k(A_k - (phi_k - c)/2eps) - c/2eps

    This keeps the reported log-weights bounded and interpretable.  It is worth
    being precise about what it does *not* do: it cannot recover precision that
    was already lost when phi was stored as fp32.  Measured at d = 4096,
    eps = 5e-3, with a k-spread of 0.01 in phi (the O(2 eps) spread a healthy
    estimator has), the fp32 error in u is set by phi's OFFSET, not by the
    centering:

        phi offset       0  ->  error 0.0000 in u   (u spread ~1)
        phi offset     300  ->  error 0.0037
        phi offset    4096  ->  error 0.0368

    The thing that actually keeps the offset small is the gauge pin in
    ptflow.losses.potential_loss, which fixes the objective's one flat direction.
    That is the precision guard; the centering here is bookkeeping.

3.  Because N(y; x0, 2 eps I) *is* G_{2eps}(y - x0), the Gaussian kernel in the
    numerator and the defensive component in the denominator are the same
    density.  The whole log-ratio collapses to

        log G - log q_def = -logaddexp( log(1-alpha) + r,  log alpha ),
        r = log q_tilt(y) - log G(y),

    which needs no (d/2) log(4 pi eps) normalizer at all -- it cancels
    identically.  Only r is computed, in float64, from two squared norms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class TiltedEstimate:
    """Output of :func:`tilted_phi0`."""

    phi0: torch.Tensor        # [B]   estimated Moreau envelope, phi-units
    log_w: torch.Tensor       # [B,K] un-normalized log importance weights
    ess: torch.Tensor         # [B]   normalized effective sample size in (0,1]
    y: torch.Tensor           # [B,K,...] proposal points
    phi_y: torch.Tensor       # [B,K] potential at the proposal points
    logw_spread: torch.Tensor # [B]   std of log_w across k -- the s_res of Thm 2.9
    clip_frac: torch.Tensor   # []    fraction of weights hit by the safety clip

    def log_psi0(self, eps: float) -> torch.Tensor:
        return -self.phi0 / (2.0 * float(eps))


# ---------------------------------------------------------------------------
# Proposal
# ---------------------------------------------------------------------------

def sample_proposal(
    x0: torch.Tensor,
    m0: torch.Tensor,
    s0: Optional[torch.Tensor],
    eps: float,
    K: int,
    alpha_def: float,
    *,
    generator: Optional[torch.Generator] = None,
    antithetic: bool = True,
) -> torch.Tensor:
    """Draw K points from the defensive mixture proposal.

    Args:
        x0, m0: [B, ...] noise sample and prox guess (the generator output).
        s0: [B, ...] diagonal log-scale, or None for S = I.
        eps: viscosity.
        K: number of proposal points.  Rounded up to even when antithetic.
        alpha_def: defensive mixture weight.  Never annealed to zero
            (Appendix E.4): it upper-bounds the weights against proposal
            misplacement and degrades gracefully to the naive estimator when
            m_eta is uninformative.
        antithetic: use +/- z pairs.  Free variance reduction (Appendix H).

    Returns:
        y: [B, K, ...]
    """
    B = x0.shape[0]
    tail = x0.shape[1:]
    sqrt2e = math.sqrt(2.0 * float(eps))

    if antithetic:
        half = (int(K) + 1) // 2
        z_half = torch.randn(
            (B, half, *tail), generator=generator, device=x0.device, dtype=x0.dtype
        )
        z = torch.cat([z_half, -z_half], dim=1)[:, : int(K)]
        # A +/- pair must stay inside one mixture component to remain antithetic.
        u_half = torch.rand((B, half), generator=generator, device=x0.device)
        from_def = (torch.cat([u_half, u_half], dim=1)[:, : int(K)] < float(alpha_def))
    else:
        z = torch.randn(
            (B, int(K), *tail), generator=generator, device=x0.device, dtype=x0.dtype
        )
        from_def = (
            torch.rand((B, int(K)), generator=generator, device=x0.device) < float(alpha_def)
        )

    scale = torch.ones_like(x0) if s0 is None else torch.exp(0.5 * s0)
    y_tilt = m0.unsqueeze(1) + sqrt2e * scale.unsqueeze(1) * z
    y_naive = x0.unsqueeze(1) + sqrt2e * z

    mask = from_def.view(B, int(K), *([1] * len(tail)))
    return torch.where(mask, y_naive, y_tilt)


def log_kernel_over_proposal(
    y: torch.Tensor,
    x0: torch.Tensor,
    m0: torch.Tensor,
    s0: Optional[torch.Tensor],
    eps: float,
    alpha_def: float,
) -> torch.Tensor:
    """log G_{2 eps}(y - x0) - log q_def(y | x0), computed in float64.

    The (d/2) log(4 pi eps) normalizers cancel identically because
    N(y; x0, 2 eps I) and G_{2 eps}(y - x0) are the same density.  What remains
    is r = log q_tilt - log G, a difference of two squared norms:

        r = |y - x0|^2 / (4 eps)  -  |(y - m0) exp(-s0/2)|^2 / (4 eps)
            - (1/2) sum_i s0_i

    and then  log G - log q_def = -logaddexp(log(1-alpha) + r, log alpha).

    Returned in float64, deliberately.  At d = 4096, eps = 5e-3 these values run
    to ~1.5e5 with a k-to-k variation of ~2e4, and the variation is cancelled
    almost exactly by the u-term (Appendix E.1: the Gaussian proposal's own
    log-density *is* the quadratic model of the integrand's exponent, so only
    the Taylor remainder survives).  Casting to float32 first would inject an
    ulp of ~1e-2 into a difference whose true size is O(1).

    Returns: [B, K] in float64.
    """
    B, K = y.shape[0], y.shape[1]
    four_eps = 4.0 * float(eps)

    yf = y.reshape(B, K, -1).double()
    x0f = x0.reshape(B, 1, -1).double()
    m0f = m0.reshape(B, 1, -1).double()

    qa = (yf - x0f).pow(2).sum(-1) / four_eps                       # [B,K]
    if s0 is None:
        qb = (yf - m0f).pow(2).sum(-1) / four_eps
        half_logdet = torch.zeros((B, 1), device=y.device, dtype=torch.float64)
    else:
        s0f = s0.reshape(B, 1, -1).double()
        qb = ((yf - m0f) * torch.exp(-0.5 * s0f)).pow(2).sum(-1) / four_eps
        half_logdet = 0.5 * s0f.sum(-1)                              # [B,1]

    r = qa - qb - half_logdet                                        # [B,K]

    a = float(alpha_def)
    if a <= 0.0:
        return -r
    if a >= 1.0:
        # Pure naive proposal: q_def == G_{2eps}(. - x0) exactly, ratio is 1.
        return torch.zeros_like(r)

    log_1ma = math.log(1.0 - a)
    log_a = math.log(a)
    log_q_over_g = torch.logaddexp(r + log_1ma, torch.full_like(r, log_a))
    return -log_q_over_g


# ---------------------------------------------------------------------------
# The estimator
# ---------------------------------------------------------------------------

def tilted_phi0(
    potential,
    x0: torch.Tensor,
    c: torch.Tensor,
    m0: torch.Tensor,
    s0: Optional[torch.Tensor],
    eps: float,
    *,
    K: int = 8,
    alpha_def: float = 0.1,
    generator: Optional[torch.Generator] = None,
    antithetic: bool = True,
    logw_clip: float = 0.0,
    y: Optional[torch.Tensor] = None,
    chunk: int = 0,
) -> TiltedEstimate:
    """Estimate the smoothed potential phi_0(x0) = -2 eps log psi_0(x0).

    Gradients flow to ``potential`` only.  ``m0`` and ``s0`` must already be
    detached by the caller (the potential step runs at frozen eta).

    Args:
        potential: a :class:`ptflow.potential.PotentialNet`.
        x0: [B, ...] noise samples.
        c: [B] labels (already conditioning-dropped by the caller).
        m0, s0: detached proposal mean and diagonal log-scale.
        eps: viscosity.
        K: proposal points per x0.
        alpha_def: defensive mixture weight.
        logw_clip: safety clip.  A log-weight is not allowed to exceed the
            per-sample median by more than this.  Inactive at healthy ESS (the
            spread is then O(1)); it only fires in pathology, where an
            unclipped single weight would otherwise take the whole batch.
        y: optionally reuse proposal points already drawn (e.g. by the scale
            head fit) instead of drawing new ones.
        chunk: if > 0, evaluate the potential in chunks of this many rows to
            bound activation memory.
    """
    B = x0.shape[0]
    tail = x0.shape[1:]
    K = int(K)
    if K < 1 or eps <= 0 or not 0 <= alpha_def <= 1:
        raise ValueError("Require K >= 1, eps > 0 and alpha_def in [0, 1].")

    if y is None:
        with torch.no_grad():
            y = sample_proposal(
                x0, m0, s0, eps, K, alpha_def,
                generator=generator, antithetic=antithetic,
            )
    y = y.detach()

    with torch.no_grad():
        log_ratio = log_kernel_over_proposal(y, x0, m0, s0, eps, alpha_def)  # [B,K]

    # -- potential at the proposal points (this is where theta's gradient lives)
    y_flat = y.reshape(B * K, *tail)
    c_rep = c.repeat_interleave(K, dim=0)
    if chunk and chunk > 0:
        parts = [
            potential.phi(y_flat[i : i + chunk], c_rep[i : i + chunk])
            for i in range(0, B * K, int(chunk))
        ]
        phi_y = torch.cat(parts, dim=0).view(B, K)
    else:
        phi_y = potential.phi(y_flat, c_rep).view(B, K)

    # -- centering: exact, and the only thing that keeps fp32 alive at cold eps
    phi_ref = phi_y.detach().mean(dim=1, keepdim=True)               # [B,1]
    u_centered = (phi_y.double() - phi_ref.double()) / (2.0 * float(eps))

    # fp64 for the combination: log_ratio and the u-term individually run to
    # ~1e5 and cancel to O(1) (the exact cancellation of Appendix E.1), so the
    # subtraction is where precision is won or lost.  Autograd passes through
    # the cast, and [B,K] is far too small for the cost to matter.
    log_w = log_ratio - u_centered.double()                          # [B,K] fp64
    raw_log_w = log_w

    # -- safety clip (bias-for-variance; logged so it stays visible)
    if logw_clip is not None and logw_clip > 0:
        ref = log_w.detach().median(dim=1, keepdim=True).values
        ceiling = ref + float(logw_clip)
        clipped_values = torch.minimum(log_w.detach(), ceiling)
        # Optional biased robust estimator: preserve each phi derivative.
        # A hard min with a detached ceiling zeroed the dominant gradients.
        clipped = log_w + (clipped_values - log_w.detach())
        clip_frac = (log_w.detach() > ceiling).float().mean()
        log_w = clipped
    else:
        clip_frac = torch.zeros((), device=x0.device)

    lse = torch.logsumexp(log_w, dim=1) - math.log(K)                # [B] fp64

    # phi0 = phi_ref - 2 eps * lse   (undo the centering, in phi-units)
    phi0 = (phi_ref.squeeze(1).double() - 2.0 * float(eps) * lse).float()

    with torch.no_grad():
        # Health must describe the actual importance weights, never the clip.
        lw = raw_log_w.detach()
        lse1 = torch.logsumexp(lw, dim=1)
        lse2 = torch.logsumexp(2.0 * lw, dim=1)
        ess = torch.exp(2.0 * lse1 - lse2 - math.log(K)).clamp(0.0, 1.0)
        spread = lw.std(dim=1, correction=0)

    return TiltedEstimate(
        phi0=phi0,
        log_w=log_w,
        ess=ess,
        y=y,
        phi_y=phi_y,
        logw_spread=spread,
        clip_frac=clip_frac,
    )


# ---------------------------------------------------------------------------
# Self-normalized importance resampling -- Mode C
# ---------------------------------------------------------------------------

@torch.no_grad()
def snis_resample(
    y: torch.Tensor,
    log_w: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Resample one point per row with probability proportional to the weights.

    As K -> infinity this samples the exact bridge kernel p(x1 | x0), hence
    exactly the model marginal rho_hat_1 -- the distribution the likelihood
    scores.  This is what makes Proposition 2.4's sampler/likelihood gap
    measurable rather than rhetorical.
    """
    B, K = log_w.shape
    probs = torch.softmax(log_w.float(), dim=1)
    idx = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(1)
    gather_idx = idx.view(B, 1, *([1] * (y.ndim - 2))).expand(B, 1, *y.shape[2:])
    return y.gather(1, gather_idx).squeeze(1)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

@torch.no_grad()
def naive_phi0(
    potential,
    x0: torch.Tensor,
    c: torch.Tensor,
    eps: float,
    *,
    K: int = 8,
    generator: Optional[torch.Generator] = None,
) -> TiltedEstimate:
    """The naive estimator of eq. 2.9, for the variance-reversal sweep.

    Identical to :func:`tilted_phi0` with m0 = x0, S = I, alpha = 1.  Kept as a
    separate entry point so the R0 experiment reads as a comparison rather than
    a configuration change.
    """
    return tilted_phi0(
        potential, x0, c, m0=x0, s0=None, eps=eps,
        K=K, alpha_def=1.0, generator=generator, logw_clip=0.0,
    )
