"""The PT-Flow training losses.

Three objectives, all expressed in phi-units (physical transport units) and all
normalized per coordinate, so that every number logged during training is
comparable across model sizes and invariant under the eps anneal.

    potential_loss   the flat objective of eq. 2.6, which by Theorem 2.6(i) has
                     the remaining Schrodinger boundary condition as its exact
                     stationarity condition.  In phi-units it reads

                         2 eps L(theta) = E_{x1}[phi(x1)] - E_{x0}[phi_0(x0)]

                     -- push the potential down on data, up on model samples.
                     This is a contrastive-divergence structure, with the
                     generator's own output supplying the negatives.

    prox_loss        the generator's regression onto the prox first-order
                     condition, eq. 2.16.  Default mode detaches the target,
                     making it structurally identical to the drift arm's own
                     regression onto a detached OT ``goal``.

    scale_loss       reverse-KL fit of the diagonal log-scale head, evaluated
                     at proposal points that are already being computed.  Its
                     stationary point is exactly S^-1 = I + grad^2 phi (eq.
                     2.11); the derivation is in the docstring.

Plus two guards that the paper leaves implicit and that a DiT-scale run needs:
a gauge pin and a weak-convexity hinge.
"""

from __future__ import annotations

import contextlib
import math
from typing import Dict, Optional, Tuple

import torch

from ptflow.estimator import TiltedEstimate, tilted_phi0
from ptflow.potential import guided_phi_grad, phi_grad


@contextlib.contextmanager
def _frozen(module, active: bool = True):
    """Temporarily disable requires_grad on a module's parameters.

    Unlike torch.no_grad this still builds the graph, so gradients keep flowing
    to the *inputs* -- which is what a reparameterized loss needs.
    """
    if not active:
        yield
        return
    flags = [(p, p.requires_grad) for p in module.parameters()]
    try:
        for p, _ in flags:
            p.requires_grad_(False)
        yield
    finally:
        for p, was in flags:
            p.requires_grad_(was)


# ---------------------------------------------------------------------------
# (B) Potential step
# ---------------------------------------------------------------------------

def potential_loss(
    potential,
    x0: torch.Tensor,
    c0: torch.Tensor,
    x1: torch.Tensor,
    c1: torch.Tensor,
    m0: torch.Tensor,
    s0: Optional[torch.Tensor],
    eps: float,
    *,
    K: int = 8,
    alpha_def: float = 0.1,
    lambda_gauge: float = 0.1,
    lambda_mag: float = 0.0,
    logw_clip: float = 0.0,
    generator: Optional[torch.Generator] = None,
    chunk: int = 0,
) -> Tuple[torch.Tensor, TiltedEstimate, Dict[str, torch.Tensor]]:
    """The flat objective, in phi-units and normalized per coordinate.

    ``m0`` and ``s0`` come from the *frozen* generator at guidance weight zero.
    One rule from Appendix F.1 is load-bearing and enforced by the caller: the
    potential is trained at w = 0 always.  Feeding a w-tilted potential here
    while the data term still uses real x1 would ask the network to make a
    w-indexed family of wrong bridges fit one dataset simultaneously.

    Returns ``(loss, estimate, metrics)``.
    """
    d = float(x0[0].numel())

    est = tilted_phi0(
        potential, x0, c0, m0, s0, eps,
        K=K, alpha_def=alpha_def, generator=generator,
        logw_clip=logw_clip, chunk=chunk,
    )

    phi_data = potential.phi(x1, c1)              # [B1]
    p1 = phi_data / d                             # per-coordinate, O(1)
    p0 = est.phi0 / d

    loss = p1.mean() - p0.mean()

    # -- gauge pin ---------------------------------------------------------
    # phi -> phi + k leaves the loss exactly invariant (Theorem 2.6(ii): the
    # scalar gauge is the objective's only flat direction).  In function space
    # that is harmless; for a network it is a free direction that drifts and
    # wrecks conditioning.  Pin the gauge coordinate itself -- the *sum*, which
    # shifts by 2k -- and not the difference, which is the objective.
    gauge = 0.5 * (p1.mean() + p0.mean())
    if lambda_gauge > 0.0:
        loss = loss + float(lambda_gauge) * gauge.pow(2)

    # -- soft magnitude bound (A1) ----------------------------------------
    if lambda_mag > 0.0:
        loss = loss + float(lambda_mag) * (p1.pow(2).mean() + p0.pow(2).mean())

    metrics = {
        "pt/loss_potential": loss.detach(),
        "pt/phi_data": p1.mean().detach(),
        "pt/phi0_noise": p0.mean().detach(),
        "pt/gauge": gauge.detach(),
        "pt/ess": est.ess.mean().detach(),
        "pt/ess_min": est.ess.min().detach(),
        "pt/logw_spread": est.logw_spread.mean().detach(),
        "pt/clip_frac": est.clip_frac.detach(),
        "pt/eps": torch.as_tensor(float(eps), device=x0.device),
        "pt/alpha_def": torch.as_tensor(float(alpha_def), device=x0.device),
    }
    return loss, est, metrics


# ---------------------------------------------------------------------------
# (A) Generator step -- prox residual
# ---------------------------------------------------------------------------

def prox_loss(
    potential,
    m: torch.Tensor,
    x0: torch.Tensor,
    c: torch.Tensor,
    w: torch.Tensor | float = 0.0,
    *,
    mode: str = "detach",
    norm: str = "none",
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Regress the generator onto the prox first-order condition.

        grad phi^w(y*) + y* - x0 = 0        =>        y* = x0 - grad phi^w(y*)

    mode="detach" (default, stable):
        Compute ``target = (x0 - grad phi^w(m)).detach()`` with a single
        input-gradient pass through the potential, then regress m onto it.
        No double backward.  This is *structurally the same object* as the baseline's
        loss, which regresses generated features onto a detached OT-displaced
        ``goal`` -- which is why the two terms compose without fighting.

    mode="full" (paper-faithful):
        Differentiate the residual through m, requiring a Hessian-vector
        product per step.  More faithful, materially slower, and noticeably
        less forgiving early in training when phi is still meaningless.

    norm="none" (default, paper-literal):
        loss = ||m - (x0 - grad phi(m))||^2 / d, the residual at its natural
        magnitude.

    norm="rms" (the drift arm's own convention):
        Normalize the demanded displacement to unit RMS before regressing onto
        it, exactly as ot_drift_loss.py:501 does for its OT velocity:

            f_norm      = (V_raw ** 2).mean()
            force_scale = sqrt(clamp(f_norm, min=1e-8))
            V_agg       = V_raw / force_scale

        WHY THIS MATTERS.  The two losses have the same *shape* -- both regress
        onto a detached displaced target -- which is the stated reason they
        compose additively rather than fighting.  But the baseline's displacement is
        normalized to unit RMS by construction, so its loss sits at 1.0 no
        matter how wrong the model is, while the prox displacement is a *raw*
        gradient of an unconverged potential.  Measured at d=3072: the baseline
        term held 1.0000 (range [0.9929, 1.0007]) while the prox term ranged
        over [2.3, 367580] -- a 163000x dynamic range.  lambda_prox is a
        constant, and a constant cannot balance that.

        The theory is unaffected: at the fixed point grad phi equals the
        transport displacement, both terms are O(1), and the normalizer is
        ~1.  Off the fixed point it discards a magnitude that carries no
        information while phi is untrained, and keeps the direction, which
        does.  The normalizer is logged as pt/prox_disp_scale -- never hide a
        normalizer, or the loss curve stops meaning anything.

    Returns per-coordinate MSE and metrics.
    """
    d = float(x0[0].numel())

    if mode == "full":
        with _frozen(potential):
            g = guided_phi_grad(potential, m, c, w, create_graph=True)
        resid = g + m - x0
    elif mode == "detach":
        g = guided_phi_grad(potential, m.detach(), c, w, create_graph=False)
        target = (x0 - g).detach()
        resid = m - target
    else:
        raise ValueError("prox_loss mode must be 'detach' or 'full', got %r" % (mode,))

    if norm == "none":
        disp_scale = torch.ones((), device=resid.device, dtype=resid.dtype)
        loss = resid.pow(2).flatten(1).sum(1).mean() / d

    elif norm in ("rms", "bounded_rms"):
        # Normalize the TARGET DISPLACEMENT, not the residual.  the OT-drift baseline builds
        #     goal = old_gen + V_raw / rms(V_raw)
        # and regresses onto that, so its gradient stays O(1).  Normalizing the
        # residual instead would give d(loss)/dm ~ resid / rms(resid)^2, which
        # DIVERGES as the run converges -- the opposite of what is wanted.
        if mode != "detach":
            raise ValueError('prox_loss norm="rms" requires mode="detach"; '
                             'mode="full" differentiates through the residual, '
                             'which a detached target would silently discard.')
        disp = (-resid).detach()                    # x0 - grad phi(m) - m
        disp_scale = disp.pow(2).mean().sqrt().clamp_min(1e-8)
        if norm == "bounded_rms":
            # Cap large targets without amplifying a small residual.
            disp_scale = disp_scale.clamp_min(1.0)
        tgt = (m.detach() + disp / disp_scale).detach()
        loss = (m - tgt).pow(2).flatten(1).sum(1).mean() / d

    elif norm == "rel":
        # Scale-free AND vanishing at the fixed point: the residual measured in
        # units of the transport displacement the generator actually made.  This
        # is exactly mean(pt/prox_resid_rel^2) -- the diagnostic README_PTFLOW.md
        # says to watch (healthy < 0.3) -- so the run optimizes the number it is
        # judged by.  Unlike "rms" it is 0 at the prox and works with mode="full".
        # The divisor must be floored against the residual, not against a fixed
        # epsilon.  At the residual=True warm start m == x0 exactly, so the
        # transport displacement is 0 while grad phi need not be -- a fixed
        # epsilon floor then divides by 1e-6 and NaNs the run on step 0.
        # Flooring at RATIO_CAP^-1 * ||resid|| caps the per-sample ratio instead.
        RATIO_CAP = 20.0
        rn = resid.detach().flatten(1).norm(dim=1)                          # [B]
        dt = (m.detach() - x0).flatten(1).norm(dim=1)                       # [B]
        dt = torch.maximum(dt, rn / RATIO_CAP).clamp_min(1e-8)
        disp_scale = dt.mean()
        loss = (resid.flatten(1).norm(dim=1) / dt).pow(2).mean()

    else:
        raise ValueError("prox_loss norm must be 'none', 'rms' or 'rel', got %r" % (norm,))

    with torch.no_grad():
        rnorm = resid.detach().flatten(1).norm(dim=1)
        disp = (m.detach() - x0).flatten(1).norm(dim=1)
        metrics = {
            "pt/prox_resid": rnorm.mean(),
            # A multi-modal blow-up of the residual is the (A2)-violation alarm
            # of Appendix H: the spread matters as much as the mean.
            "pt/prox_resid_max": rnorm.max(),
            "pt/prox_resid_rel": (rnorm / disp.clamp_min(1e-6)).mean(),
            "pt/displacement": disp.mean(),
            "pt/loss_prox": loss.detach(),
            # 1.0 for norm="none"; the RMS actually divided out otherwise.
            "pt/prox_disp_scale": disp_scale.detach(),
        }
    return loss, metrics


# ---------------------------------------------------------------------------
# (A) Generator step -- diagonal scale head
# ---------------------------------------------------------------------------

def scale_loss(
    potential,
    m: torch.Tensor,
    s: torch.Tensor,
    x0: torch.Tensor,
    c: torch.Tensor,
    eps: float,
    *,
    K: int = 4,
    generator: Optional[torch.Generator] = None,
    detach_potential: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Reverse-KL fit of the diagonal log-scale head.

    Target: q = N(m, 2 eps diag(e^s)) should match the Laplace shape of
    p(y) ~ G_{2eps}(y - x0) e^{-u(y)}.  Dropping the s-independent log Z and
    multiplying by 2 eps to stay in phi-units:

        L_scale = -eps * sum_i s_i + E_q[ |y - x0|^2 / 2 + phi(y) ]

    with y = m + sqrt(2 eps) e^{s/2} z reparameterized so the gradient reaches
    s.  Its stationary point is the paper's eq. 2.11: expanding phi to second
    order at m gives  -eps sum_i log sigma_i^2 + (1/2) sum_i (1 + H_ii)
    sigma_i^2, whose minimizer is sigma_i^2 = 2 eps / (1 + H_ii), i.e.
    S^-1 = I + grad^2 phi.

    ``m`` is detached: its gradient is the prox loss's business, not this one's.
    ``detach_potential`` (default True) likewise stops the scale fit from
    reaching back into phi.  The scale head and the potential share an optimizer
    step here, and letting a variance-reduction fit reshape the potential it is
    supposed to be approximating is a feedback loop with no upside.
    """
    d = float(x0[0].numel())
    B = x0.shape[0]
    tail = x0.shape[1:]
    K = int(K)
    sqrt2e = math.sqrt(2.0 * float(eps))

    m_d = m.detach()
    z = torch.randn((B, K, *tail), generator=generator, device=x0.device, dtype=x0.dtype)
    y = m_d.unsqueeze(1) + sqrt2e * torch.exp(0.5 * s).unsqueeze(1) * z   # grad -> s

    # Cut theta, keep the pathwise gradient.  torch.no_grad() would kill both --
    # and the pathwise term d/ds E_q[phi(y(s))] IS the loss, since y = m +
    # sqrt(2 eps) e^{s/2} z is where s enters.  Freezing the parameters instead
    # leaves the graph through y intact while giving phi's weights no gradient.
    with _frozen(potential, detach_potential):
        phi_y = potential.phi(
            y.reshape(B * K, *tail), c.repeat_interleave(K, dim=0)
        ).view(B, K)

    quad = 0.5 * (y - x0.unsqueeze(1)).pow(2).flatten(2).sum(-1)          # [B,K]
    entropy_term = float(eps) * s.flatten(1).sum(-1)                      # [B]

    loss = ((quad + phi_y).mean(dim=1) - entropy_term).mean() / d

    with torch.no_grad():
        metrics = {
            "pt/loss_scale": loss.detach(),
            "pt/log_scale_mean": s.detach().mean(),
            "pt/log_scale_std": s.detach().std(),
        }
    return loss, metrics


# ---------------------------------------------------------------------------
# (A2) weak-convexity monitor / soft enforcement
# ---------------------------------------------------------------------------

def curvature_hinge(
    potential,
    x: torch.Tensor,
    c: torch.Tensor,
    *,
    g_at_x: Optional[torch.Tensor] = None,
    h: float = 1e-2,
    lambda_allow: float = 0.5,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Hutchinson probe of directional curvature, with a hinge on violations.

    (A2) asks grad^2 phi >= -lambda I with lambda < 1, which is what makes
    F_x = phi + |x - .|^2 / 2 strongly convex and the prox unique.  A single
    Rademacher probe v gives

        v^T grad^2 phi v  ~=  (grad phi(x + h v) - grad phi(x)) . v / h

    normalized by |v|^2 = d.  We penalize relu(-curv - lambda_allow), i.e. only
    curvature more negative than the allowance.  One extra input-gradient pass
    (or zero extra, if ``g_at_x`` is supplied from the prox step).
    """
    d = float(x[0].numel())
    v = torch.randint(
        0, 2, x.shape, generator=generator, device=x.device, dtype=torch.int8
    ).to(x.dtype).mul_(2).sub_(1)

    if g_at_x is None:
        g_at_x, _ = phi_grad(potential, x, c, create_graph=True)
    g_pert, _ = phi_grad(potential, x + float(h) * v, c, create_graph=True)

    curv = ((g_pert - g_at_x) * v).flatten(1).sum(1) / (float(h) * d)
    pen = torch.relu(-curv - float(lambda_allow)).mean()

    with torch.no_grad():
        metrics = {
            "pt/curv_mean": curv.detach().mean(),
            "pt/curv_min": curv.detach().min(),
            "pt/curv_viol_frac": (curv.detach() < -float(lambda_allow)).float().mean(),
            "pt/loss_curv": pen.detach(),
        }
    return pen, metrics
