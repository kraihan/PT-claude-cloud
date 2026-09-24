"""Schedules and the ESS health state machine.

The paper's Appendix F.4 names one scalar as the certificate for the whole
scheme: the normalized effective sample size ESS/K of the importance weights,
which is the empirical stand-in for assumption (A4) (the generator tracking the
prox).  Its thresholds are

    healthy    ESS/K > 0.30     proceed; allow the eps anneal to advance
    degrading  0.05 - 0.30      slow the anneal, raise alpha_def, add eta steps
    broken     ESS/K < 0.05     freeze theta, retrain eta, resume

This module makes that certificate the single controller for everything that can
destabilize the run: the viscosity eps, the defensive mixture weight alpha_def,
and -- the addition that makes the baseline marriage work -- the weight lambda_prox
on the PT-Flow term in the generator's loss.

    lambda_prox = 0  reproduces the OT-drift baseline exactly.  The generator is driven only by
    the Sinkhorn-OT drift, and the potential is a passive observer learning to
    match a generator it does not influence.  lambda_prox is ramped in only
    after a warmup *and* only while the estimator is certified healthy, so the
    PT-Flow objective can never take the wheel while its own diagnostics say it
    is not ready.  If ESS collapses mid-run, lambda_prox decays back toward zero
    and the run falls back onto the baseline arm rather than diverging.

Note on eps_min.  The paper anneals to 1e-3.  We default to 5e-3 and say why:
the estimator's signal is the spread of u = phi/(2 eps) across proposal points,
which is O(1) on an offset of O(d / 2 eps).  ptflow.estimator removes that offset
analytically, but the residual fp32 relative precision still bounds how cold eps
can usefully get at d = 4096.  Push eps_min lower only while watching
``pt/logw_spread``: if it stops shrinking as eps falls, the estimator has become
precision-limited rather than variance-limited and further cooling buys nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict

import torch


HEALTHY, DEGRADING, BROKEN = "healthy", "degrading", "broken"


@dataclass
class EstimatorHealth:
    """EMA-smoothed ESS with hysteresis, so the controller does not flap."""

    ess_healthy: float = 0.30
    ess_broken: float = 0.05
    ema_decay: float = 0.98
    value: float = 1.0
    state: str = HEALTHY
    broken_steps: int = 0

    def update(self, ess: float) -> str:
        ess = float(ess)
        if not math.isfinite(ess):
            ess = 0.0
        self.value = self.ema_decay * self.value + (1.0 - self.ema_decay) * ess
        if self.value < self.ess_broken:
            self.state = BROKEN
            self.broken_steps += 1
        elif self.value < self.ess_healthy:
            self.state = DEGRADING
            self.broken_steps = 0
        else:
            self.state = HEALTHY
            self.broken_steps = 0
        return self.state

    @property
    def is_healthy(self) -> bool:
        return self.state == HEALTHY

    @property
    def is_broken(self) -> bool:
        return self.state == BROKEN


@dataclass
class PTSchedule:
    """All PT-Flow schedules, gated on estimator health.

    Args:
        eps_max, eps_min: cosine anneal endpoints for the viscosity.
        eps_anneal_steps: length of the anneal in *healthy* steps.  Progress is
            frozen whenever the estimator is not healthy, so a run that never
            certifies never cools -- which is the desired failure mode.
        alpha_def_start, alpha_def_end: defensive mixture weight.  Never
            annealed to zero (Appendix E.4).
        prox_warmup: steps of the pure baseline before the PT term can appear at all.
        prox_ramp: steps over which lambda_prox rises to its maximum.
        lambda_prox_max: final weight on the prox-residual term.
        prox_decay_on_degrade: multiplicative decay applied to the ramp progress
            on every non-healthy step, so a mid-run ESS collapse walks the run
            back to the baseline instead of letting a bad potential steer.
        theta_steps_per_gen / gen_steps_per_theta: the two-timescale ratio.  The
            paper's "one to two eta steps per theta step, doubled whenever ESS
            degrades" is expressed here as an update *period* for each side.
    """

    eps_max: float = 0.2
    eps_min: float = 5e-3
    eps_anneal_steps: int = 60000
    eps_warmup: int = 2000
    eps_schedule: str = "cosine"

    alpha_def_start: float = 0.1
    alpha_def_end: float = 0.01
    alpha_def_degraded: float = 0.25

    prox_warmup: int = 5000
    prox_ramp: int = 15000
    lambda_prox_max: float = 0.0
    prox_decay_on_degrade: float = 0.999

    lambda_scale: float = 0.1
    lambda_curv: float = 0.0

    theta_period: int = 1          # potential update every N steps
    theta_period_degraded: int = 2
    health_check_period: int = 20

    health: EstimatorHealth = field(default_factory=EstimatorHealth)

    # -- mutable progress --------------------------------------------------
    step: int = 0
    eps_progress: int = 0
    prox_progress: float = 0.0

    # -- state -------------------------------------------------------------

    def state_dict(self) -> Dict[str, float]:
        return {
            "step": self.step,
            "eps_progress": self.eps_progress,
            "prox_progress": self.prox_progress,
            "health_value": self.health.value,
            "health_state": self.health.state,
        }

    def load_state_dict(self, d: Dict) -> None:
        if not d:
            return
        self.step = int(d.get("step", 0))
        self.eps_progress = int(d.get("eps_progress", 0))
        self.prox_progress = float(d.get("prox_progress", 0.0))
        self.health.value = float(d.get("health_value", 1.0))
        self.health.state = str(d.get("health_state", HEALTHY))

    # -- schedules ---------------------------------------------------------

    def eps(self) -> float:
        """Configured anneal, advanced only on healthy steps."""
        mode = str(self.eps_schedule).lower()
        if mode not in {"constant", "linear", "cosine", "exp"}:
            raise ValueError(f"Unknown eps_schedule: {self.eps_schedule}")
        if mode == "constant":
            return float(self.eps_min)
        if self.eps_progress <= self.eps_warmup:
            return float(self.eps_max)
        t = min(
            max(self.eps_progress - self.eps_warmup, 0),
            max(self.eps_anneal_steps, 1),
        ) / max(self.eps_anneal_steps, 1)
        if mode == "linear":
            return float(self.eps_max + (self.eps_min - self.eps_max) * t)
        if mode == "exp":
            if self.eps_min <= 0 or self.eps_max <= 0:
                raise ValueError("Exponential epsilon schedule requires positive endpoints")
            return float(math.exp(math.log(self.eps_max) +
                                  (math.log(self.eps_min) - math.log(self.eps_max)) * t))
        cos = 0.5 * (1.0 + math.cos(math.pi * t))
        return float(self.eps_min + (self.eps_max - self.eps_min) * cos)

    def alpha_def(self) -> float:
        if not self.health.is_healthy:
            return float(self.alpha_def_degraded)
        t = min(max(self.eps_progress, 0), max(self.eps_anneal_steps, 1)) / max(
            self.eps_anneal_steps, 1
        )
        return float(
            self.alpha_def_start + (self.alpha_def_end - self.alpha_def_start) * t
        )

    def lambda_prox(self) -> float:
        if self.lambda_prox_max <= 0.0:
            return 0.0
        return float(self.lambda_prox_max * self.prox_progress)

    def update_potential(self) -> bool:
        """Whether to run the potential (theta) step this iteration."""
        if self.health.is_broken:
            # "broken: freeze theta, retrain eta, resume" -- Appendix F.4.
            return False
        period = self.theta_period_degraded if not self.health.is_healthy else self.theta_period
        return (self.step % max(1, int(period))) == 0

    # -- the controller ----------------------------------------------------

    def observe(self, ess: float | None) -> str:
        """Feed one batch's ESS/K in and advance every gated schedule."""
        state = self.health.update(ess) if ess is not None else self.health.state

        if state == HEALTHY and ess is not None:
            self.eps_progress += 1
            if self.step >= self.prox_warmup:
                inc = 1.0 / max(1, int(self.prox_ramp))
                self.prox_progress = min(1.0, self.prox_progress + inc)
        elif state != HEALTHY:
            # Do not advance the anneal; walk the PT term back toward the OT-drift baseline.
            self.prox_progress *= float(self.prox_decay_on_degrade)

        self.step += 1
        return state

    def metrics(self, device=None) -> Dict[str, torch.Tensor]:
        vals = {
            "pt/sched_eps": self.eps(),
            "pt/sched_alpha_def": self.alpha_def(),
            "pt/sched_lambda_prox": self.lambda_prox(),
            "pt/sched_eps_progress": float(self.eps_progress),
            "pt/ess_ema": float(self.health.value),
            "pt/health": {HEALTHY: 2.0, DEGRADING: 1.0, BROKEN: 0.0}[self.health.state],
        }
        return {k: torch.as_tensor(v, device=device) for k, v in vals.items()}


def build_schedule(cfg: Dict | None) -> PTSchedule:
    """Construct a :class:`PTSchedule` from a plain config dict.

    ``ess_ema_decay`` configures the health monitor's smoothing.  It is spelled
    out rather than called ``ema_decay`` because the ``pt:`` block already has an
    ``ema_decay`` meaning the potential network's weight EMA, and the two are
    unrelated.  ``ema_decay`` is still accepted here for convenience.
    """
    cfg = dict(cfg or {})
    alias = {
        "ess_healthy": "ess_healthy",
        "ess_broken": "ess_broken",
        "ess_ema_decay": "ema_decay",
        "ema_decay": "ema_decay",
    }
    health_cfg = {alias[k]: float(cfg.pop(k)) for k in list(cfg) if k in alias}
    known = {f.name for f in PTSchedule.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs = {k: v for k, v in cfg.items() if k in known and k != "health"}
    sched = PTSchedule(**kwargs)
    if health_cfg:
        sched.health = EstimatorHealth(**health_cfg)
    return sched
