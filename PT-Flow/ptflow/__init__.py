"""PT-Flow: one-step Schrodinger-bridge generation by variance-reversed proximal tilting.

This package implements the PT-Flow method on top of the OT-drift codebase.  The
design principle is stated once here and referenced throughout:

    PT-Flow's central identity is  m_eta = prox_{phi_theta} = T = the generator.
    Its one structural weakness is the bootstrap: at step 0 the generator is the
    identity, so the tilted proposal degenerates to the naive N(x0, 2 eps I) --
    exactly the regime Proposition 2.7 proves infeasible.  the baseline's Sinkhorn-OT
    drift solves that from the other end: it supplies a well-conditioned
    generator target from step 0 with no dependence on u_theta.

    So the bootstrap is inverted.  The potential learns from the baseline generator
    (whose output is already a good prox guess, hence a well-placed Laplace
    proposal, hence healthy ESS and an actually-engaged variance reversal), and
    only once ESS certifies the estimator does the generator start learning back
    from the potential through a ramped prox-residual term.

    At lambda_prox = 0 the run is exactly the baseline.  That is the "arm".

Module map
----------
    ptflow.potential   scalar log-potential network u_theta / phi_theta and its
                   input-gradients (guided and unguided)
    ptflow.estimator   the prox-tilted importance estimator: proposal, log-weights,
                   ESS, log psi_0, SNIS resampling
    ptflow.losses      the two PT-Flow losses (flat potential objective, prox
                   residual) plus the scale head and (A2) curvature hinge
    ptflow.schedule    eps annealing, defensive-mixture annealing, lambda_prox ramp,
                   and the ESS health state machine that gates all three
    ptflow.sampling    inference Modes A / B / C and the normalized likelihood
    ptflow.convert     warm-start a PT-Flow generator from a released the OT-drift baseline ckpt
"""

from ptflow.potential import PotentialNet, phi_grad, guided_phi_grad  # noqa: F401
from ptflow.schedule import PTSchedule, EstimatorHealth  # noqa: F401

__all__ = [
    "PotentialNet",
    "phi_grad",
    "guided_phi_grad",
    "PTSchedule",
    "EstimatorHealth",
]
