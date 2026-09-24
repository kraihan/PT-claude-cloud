"""New s^2 diagnostics for PT-Flow -- the repaired Section 2.5.

WHY THIS FILE EXISTS
--------------------
The manuscript's repair of Section 2.5 changes *how the estimator's variance and
sample budget are characterized*.  It does NOT change the estimator, the
potential, the generator, the scale head, or the training objective.  All of
those are unbiased for any proposal at any eps and are untouched here.  A model
trained under the old write-up is therefore valid under the new one -- there is
nothing to retrain.  Only the *reported* variance/budget quantities move, and
they are inference-time functions of the same importance weights.

WHAT CHANGED  (old Theorem 2.9  ->  repaired Section 2.5)
---------------------------------------------------------
OLD reporting:
    s_res      := std_k(log w)                          (logw_spread)
    s_res^2    ~  (5/6) d M_3^2 eps                      (a claimed rate)
    rel. var.  ~  (e^{s_res^2} - 1) / K                  (LOG-NORMAL approx)
    K_tilted   >~ exp(c d M_3^2 eps)                     (log-normal budget)

NEW reporting (this file):
    s_res^2    := Var_k(log w)   -- still measured, now bounded  <= O(eps)
                  via  log w = c(x) - 1/2 z^T B_x z - R_x/(2 eps).
    rel. var.  =  chi2( p_{eps,x} || q_x ) / K           (ACTUAL weight moments)
    chi2_hat   =  mean_k(w^2)/mean_k(w)^2 - 1  =  1/(ESS/K) - 1
    K*(delta)  =  max(1, ceil( chi2_hat / delta^2 ))     (eq. Ktilted)
    defensive  :  chi2(p||q_def) <= (chi2(p||q) + alpha)/(1 - alpha)
    (A6) term  :  1/2 ||B_x||_F^2 = 1/2 tr(Q^2),  Q = S^{1/2} H S^{1/2} - I,
                  H = I + grad^2 phi(m),  measured by Hutchinson HVP probes.

The one conceptual point: the new budget is read straight off the empirical
importance weights (a chi^2 divergence, i.e. 1/ESS - 1), with no log-normal
step.  ``chi2_hat`` is exactly ``1/(ESS/K) - 1`` -- the ESS the codebase already
logs -- so the "new s2 code" is, at heart, "stop exponentiating the log-weight
variance; use the weights themselves."

Everything here is import-light: it reuses ptflow.estimator (the weights) and
ptflow.potential (the HVP for the (A6) term) unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

import torch

from ptflow.estimator import tilted_phi0
from ptflow.potential import guided_phi_grad


# ---------------------------------------------------------------------------
# 1. The weight-moment quantities: chi^2, relative variance, sample budget.
#    These are the *primary* new-s2 outputs.  All operate on [B, K] log-weights
#    and never touch a log-normal approximation.
# ---------------------------------------------------------------------------

def ess_fraction(log_w: torch.Tensor) -> torch.Tensor:
    """Normalized effective sample size ESS/K in (0, 1], per row.

    ESS/K = (sum_k w_k)^2 / (K sum_k w_k^2), computed in the log domain.
    """
    K = log_w.shape[1]
    lse1 = torch.logsumexp(log_w, dim=1)
    lse2 = torch.logsumexp(2.0 * log_w, dim=1)
    return torch.exp(2.0 * lse1 - lse2 - math.log(K)).clamp(0.0, 1.0)


def chi2_from_log_w(log_w: torch.Tensor) -> torch.Tensor:
    """chi^2( p || q ) estimated from the importance weights, per row.

        chi^2_hat = mean_k(w^2) / mean_k(w)^2 - 1 = 1/(ESS/K) - 1  >= 0.

    Computed as expm1(log K + lse2 - 2 lse1) so it is exact and non-negative
    even when the weights are near-uniform (chi^2 -> 0).  This is the quantity
    the repaired Section 2.5 uses -- the relative variance of the K-sample
    estimator is chi^2 / K, from actual moments, not (e^{s^2}-1)/K.
    """
    K = log_w.shape[1]
    lse1 = torch.logsumexp(log_w, dim=1)
    lse2 = torch.logsumexp(2.0 * log_w, dim=1)
    return torch.expm1(math.log(K) + lse2 - 2.0 * lse1).clamp_min(0.0)


def logw_variance(log_w: torch.Tensor) -> torch.Tensor:
    """s_res^2 := Var_k(log w), per row.

    Invariant to the per-row constant that the estimator's centering adds to
    log_w, so it measures exactly the s_res of the repaired Section 2.5.
    """
    return log_w.var(dim=1, unbiased=False)


def sample_budget(chi2: torch.Tensor, delta: float) -> torch.Tensor:
    """K*(delta) = max(1, ceil( chi^2 / delta^2 )) -- eq. (Ktilted), per row.

    The number of proposal draws needed for the estimator's relative
    root-mean-square error to reach ``delta``.  Reads straight off chi^2, with
    no exponentiation.
    """
    if delta <= 0:
        raise ValueError("delta must be > 0")
    k = torch.ceil(chi2 / (float(delta) ** 2))
    return torch.clamp(k, min=1.0)


def chi2_defensive_upper_bound(chi2_pure: torch.Tensor, alpha: float) -> torch.Tensor:
    """Upper bound of eq. (defensive-variance):

        chi^2(p || q_def) <= (chi^2(p || q) + alpha) / (1 - alpha).

    ``chi2_pure`` is the chi^2 of the *pure* tilted proposal (alpha = 0).
    Choosing alpha = O(eps) preserves the O(eps) relative-variance bound; a
    fixed alpha does not.  Returned for comparison against the directly measured
    mixture chi^2.
    """
    a = float(alpha)
    if not 0.0 <= a < 1.0:
        raise ValueError("alpha must be in [0, 1)")
    return (chi2_pure + a) / (1.0 - a)


# ---------------------------------------------------------------------------
# 2. The (A6) off-diagonal curvature term  1/2 ||B_x||_F^2.
#    This is the piece a *diagonal* proposal cannot cancel; the repaired theory
#    makes it the thing s_res^2 depends on.  Measured for the ACTUAL proposal
#    (mean m, diagonal scale s), so Q = S^{1/2} H S^{1/2} - I rather than the
#    exactly-matched B_x -- they coincide when S^{-1} = diag(H).
# ---------------------------------------------------------------------------

@torch.enable_grad()
def _hvp(potential, m: torch.Tensor, c: torch.Tensor, vec: torch.Tensor,
         w: float) -> torch.Tensor:
    """Hessian-vector product grad^2 phi^w(m) @ vec, one double-backward pass."""
    m = m.detach().requires_grad_(True)
    g = guided_phi_grad(potential, m, c, float(w), create_graph=True)
    (hv,) = torch.autograd.grad(g, m, grad_outputs=vec, retain_graph=False)
    return hv.detach()


@torch.no_grad()
def offdiagonal_curvature_term(
    potential,
    m: torch.Tensor,
    s: Optional[torch.Tensor],
    c: torch.Tensor,
    *,
    w: float = 0.0,
    n_probes: int = 8,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """Estimate 1/2 ||Q||_F^2 = 1/2 tr(Q^2), Q = S^{1/2} H S^{1/2} - I, per row.

    H = I + grad^2 phi^w(m) is the Laplace Hessian at the prox point (m ~ prox),
    S = diag(exp(s)) is the proposal covariance shape actually used (S = I when
    s is None).  Since Q is symmetric and E[v v^T] = I for Rademacher v,

        tr(Q^2) = E_v[ ||Q v||^2 ],   Q v = S^{1/2}( H (S^{1/2} v) ) - v.

    Returns per-row tensors:
        offdiag_half_fro   1/2 ||Q||_F^2  -- the quadratic-only s_res^2 floor,
                                             O(eps) under (A6), zero if S^{-1}=H.
        fro_Q2             ||Q||_F^2  = tr(Q^2).
    One HVP per probe; needs a potential that supports double backward (the same
    path ptflow.losses.curvature_hinge and prox_loss(mode='full') use).
    """
    B = m.shape[0]
    tail = m.shape[1:]
    sqrt_S = torch.ones_like(m) if s is None else torch.exp(0.5 * s)

    acc = torch.zeros(B, device=m.device, dtype=torch.float64)
    for _ in range(int(n_probes)):
        v = torch.randint(0, 2, m.shape, generator=generator,
                          device=m.device, dtype=torch.int8).to(m.dtype).mul_(2).sub_(1)
        u = sqrt_S * v                                    # S^{1/2} v
        hu = _hvp(potential, m, c, u, w)                  # grad^2 phi @ u
        Hu = u + hu                                        # H u = (I + grad^2 phi) u
        Qv = sqrt_S * Hu - v                               # S^{1/2} H S^{1/2} v - v
        acc = acc + Qv.reshape(B, -1).double().pow(2).sum(-1)

    fro_Q2 = acc / float(n_probes)                         # tr(Q^2) estimate
    return {
        "offdiag_half_fro": (0.5 * fro_Q2).to(m.dtype),
        "fro_Q2": fro_Q2.to(m.dtype),
    }


# ---------------------------------------------------------------------------
# 3. A batch-level report that ties the pieces together on a trained model.
# ---------------------------------------------------------------------------

@dataclass
class VarianceReport:
    """New-s2 diagnostics for one batch, at one eps.  All means over the batch."""

    eps: float
    K: int
    alpha_def: float
    # weight-moment quantities (the primary new-s2 outputs)
    chi2: float                    # chi^2(p || q_def), from actual weights
    rel_var: float                 # chi^2 / K   (relative variance of psi0_hat)
    rel_rmse: float                # sqrt(rel_var)
    ess_frac: float                # ESS/K = 1/(1 + chi^2)
    s_res2: float                  # Var_k(log w) -- the repaired-2.5 s_res^2
    s_res: float                   # sqrt(s_res2)
    # sample budgets K*(delta)  (eq. Ktilted), median over the batch
    k_budget: Dict[str, float]
    # (A6) curvature term, if measured
    offdiag_half_fro: Optional[float] = None
    s_res2_over_offdiag: Optional[float] = None   # remainder share: s_res2 / (1/2||B||_F^2)
    # defensive-mixture cross-check (measured pure-tilt chi^2 and its bound)
    chi2_pure: Optional[float] = None
    chi2_def_bound: Optional[float] = None
    # legacy log-normal number, reported ONLY for contrast with the old write-up
    legacy_lognormal_rel_var: Optional[float] = None

    def as_dict(self) -> Dict:
        return asdict(self)


@torch.no_grad()
def variance_report(
    generator,
    potential,
    scale_net,
    x0: torch.Tensor,
    c: torch.Tensor,
    eps: float,
    *,
    K: int = 64,
    alpha_def: float = 0.05,
    cfg_scale: float = 1.0,
    pt_w: Optional[float] = None,
    deltas: Sequence[float] = (0.3, 0.1, 0.05),
    n_curvature_probes: int = 0,
    measure_pure_tilt: bool = True,
    rng: Optional[torch.Generator] = None,
) -> VarianceReport:
    """Full new-s2 report for a trained model on one noise batch.

    Runs the frozen generator to get the proposal (m, s), draws K tilted
    proposal points, and reads chi^2 / K*(delta) / s_res^2 off the weights.  The
    NLL/FID numbers a model reports are unchanged; this only re-characterizes the
    estimator that produced them.
    """
    from ptflow.sampling import _generator_output

    if pt_w is None:
        pt_w = float(cfg_scale) - 1.0

    m, x0_used, s = _generator_output(
        generator, c, cfg_scale, x0=x0, rng=rng, scale_net=scale_net
    )

    # Defensive-mixture weights (the ones the estimator/likelihood actually use).
    est = tilted_phi0(
        potential, x0_used, c, m, s, eps,
        K=int(K), alpha_def=float(alpha_def), generator=rng, logw_clip=0.0,
    )
    log_w = est.log_w
    chi2 = chi2_from_log_w(log_w)
    s_res2 = logw_variance(log_w)
    rel_var = chi2 / float(K)

    budgets = {f"delta={d:g}": float(sample_budget(chi2, d).median().item())
               for d in deltas}

    report = VarianceReport(
        eps=float(eps),
        K=int(K),
        alpha_def=float(alpha_def),
        chi2=float(chi2.mean().item()),
        rel_var=float(rel_var.mean().item()),
        rel_rmse=float(rel_var.clamp_min(0).sqrt().mean().item()),
        ess_frac=float(est.ess.mean().item()),
        s_res2=float(s_res2.mean().item()),
        s_res=float(s_res2.clamp_min(0).sqrt().mean().item()),
        k_budget=budgets,
        legacy_lognormal_rel_var=float((torch.expm1(s_res2) / float(K)).mean().item()),
    )

    # Optional pure-tilt chi^2 for the defensive-mixture bound (eq. defensive-variance).
    if measure_pure_tilt and alpha_def > 0.0:
        est0 = tilted_phi0(
            potential, x0_used, c, m, s, eps,
            K=int(K), alpha_def=0.0, generator=rng, logw_clip=0.0, y=None,
        )
        chi2_pure = chi2_from_log_w(est0.log_w)
        report.chi2_pure = float(chi2_pure.mean().item())
        report.chi2_def_bound = float(
            chi2_defensive_upper_bound(chi2_pure, alpha_def).mean().item()
        )

    # Optional (A6) off-diagonal curvature term (needs HVP double-backward).
    if n_curvature_probes and n_curvature_probes > 0:
        cur = offdiagonal_curvature_term(
            potential, m, s, c, w=float(pt_w),
            n_probes=int(n_curvature_probes), generator=rng,
        )
        half_fro = cur["offdiag_half_fro"]
        report.offdiag_half_fro = float(half_fro.mean().item())
        denom = half_fro.clamp_min(1e-12)
        report.s_res2_over_offdiag = float((s_res2 / denom).median().item())

    return report


# ---------------------------------------------------------------------------
# 4. Guided (per-cfg) likelihood -- the repaired-2.5 estimator carried onto the
#    guided marginal rho_1^w.  Prop 2.10: phi^w = (1+w) phi(.,c) - w phi(.,null)
#    is an admissible data-side potential, so Theorem 2.5 normalization holds
#    verbatim with phi -> phi^w and log rho_hat_1^w(x) = log psi_hat_1^w(x) - u^w(x).
#    Still nested Monte Carlo, still not a certified bound; the value is a density
#    in the training (latent) space, comparable across cfg for one checkpoint.
# ---------------------------------------------------------------------------

class _GuidedPotential:
    """Presents phi^w(x,c) = (1+w) phi(x,c) - w phi(x,null) as a potential.

    Duck-types :class:`ptflow.potential.PotentialNet` for the two things the
    estimator and the HVP use -- ``phi`` and ``null_labels`` -- so it drops into
    ``tilted_phi0``/``variance_report`` unchanged.  Guiding here (rather than via
    a pt_w argument threaded everywhere) keeps the guided psi_0 weights, the
    guided curvature, and the guided u^w(x1) term all consistent.
    """

    def __init__(self, potential, w: float):
        self.potential = potential
        self.w = float(w)

    def phi(self, x, c):
        base = self.potential.phi(x, c)
        if self.w == 0.0:
            return base
        null = self.potential.phi(x, self.potential.null_labels(c))
        return (1.0 + self.w) * base - self.w * null

    def null_labels(self, c):
        return self.potential.null_labels(c)


@torch.no_grad()
def guided_log_likelihood(
    generator,
    potential,
    x1: torch.Tensor,
    c: torch.Tensor,
    eps: float,
    *,
    w: float = 0.0,
    K_outer: int = 16,
    K_inner: int = 16,
    alpha_def: float = 0.05,
    scale_net=None,
    rng: Optional[torch.Generator] = None,
):
    """log rho_hat_1^w(x) for the guided marginal, w = cfg_scale - 1.

    Same nested estimator as ptflow.sampling.log_likelihood, but every potential
    evaluation is phi^w and the inner proposal is centred at the guided prox
    (generator run at cfg_scale = 1 + w) so the importance weights stay healthy.
    Returns (log_p [B], info).
    """
    from ptflow.sampling import _generator_output

    gp = _GuidedPotential(potential, float(w))
    B = x1.shape[0]
    tail = x1.shape[1:]
    d = float(x1[0].numel())
    sqrt2e = math.sqrt(2.0 * float(eps))
    J = int(K_outer)
    cfg_scale = 1.0 + float(w)

    z = torch.randn((B, J, *tail), generator=rng, device=x1.device, dtype=x1.dtype)
    v = x1.unsqueeze(1) + sqrt2e * z
    v_flat = v.reshape(B * J, *tail)
    c_rep = c.repeat_interleave(J, dim=0)

    m, _, s = _generator_output(
        generator, c_rep, cfg_scale, x0=v_flat, rng=rng, scale_net=scale_net
    )
    est = tilted_phi0(
        gp, v_flat, c_rep, m, s, eps,
        K=int(K_inner), alpha_def=float(alpha_def), generator=rng, logw_clip=0.0,
    )
    phi0_v = est.phi0.view(B, J)

    log_rho0 = -0.5 * v.reshape(B, J, -1).pow(2).sum(-1) - 0.5 * d * math.log(2.0 * math.pi)
    inner = log_rho0 + phi0_v / (2.0 * float(eps))
    log_psi1_hat = torch.logsumexp(inner, dim=1) - math.log(J)
    log_p = log_psi1_hat - gp.phi(x1, c) / (2.0 * float(eps))

    ess = float(est.ess.mean().item())
    info = {
        "w": float(w),
        "cfg_scale": cfg_scale,
        "nll_per_dim_nats": float((-log_p).mean().item()) / d,
        "bits_per_dim_latent": float((-log_p).mean().item()) / (d * math.log(2.0)),
        "logp_per_sample_nats_mean": float(log_p.mean().item()),
        "inner_ess": ess,
        "inner_chi2": max(1.0 / max(ess, 1e-8) - 1.0, 0.0),
        "K_outer": J,
        "K_inner": int(K_inner),
        "eps": float(eps),
        "estimator": "guided_nested_monte_carlo_not_a_certified_bound",
    }
    return log_p, info


@torch.no_grad()
def eps_sweep(
    generator,
    potential,
    scale_net,
    x0: torch.Tensor,
    c: torch.Tensor,
    eps_list: Sequence[float],
    *,
    K: int = 64,
    alpha_def: float = 0.0,
    cfg_scale: float = 1.0,
    deltas: Sequence[float] = (0.1,),
    n_curvature_probes: int = 0,
    rng: Optional[torch.Generator] = None,
) -> List[VarianceReport]:
    """Re-evaluate the estimator across a ladder of eps (the variance-reversal
    sweep), reporting the NEW chi^2/K budget instead of the log-normal e^{s^2}.

    Evaluating at eps != eps_train is a legal diagnostic: it changes only the
    kernel/proposal widths inside the estimator, not the trained networks.  The
    point of the sweep is unchanged -- s_res^2 and chi^2 shrink as eps cools --
    but the y-axis is now an actual relative variance, not an exponentiated one.
    """
    return [
        variance_report(
            generator, potential, scale_net, x0, c, float(e),
            K=K, alpha_def=alpha_def, cfg_scale=cfg_scale, deltas=deltas,
            n_curvature_probes=n_curvature_probes,
            measure_pure_tilt=False, rng=rng,
        )
        for e in eps_list
    ]
