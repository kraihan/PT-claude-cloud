"""Experimental warm-start recovery, used only by active_recovery_v1.

The pretrained generator is not generally the prox of a fresh zero potential.
Choose the lower-energy starting point and refine the proposal mean using the
current potential. This does NOT establish global optimality or coverage.
The actual refined mean is passed to both proposal sampling and its density.
"""
from __future__ import annotations

import torch
from ptflow.potential import phi_grad


def energy(potential, y, x, labels):
    return potential.phi(y, labels).double() + 0.5 * (y.double() - x.double()).flatten(1).square().sum(1)


def refine_proposal(potential, x, generator_mean, labels, *, steps=8, lr=0.5):
    """Backtracking descent, with frozen targets and per-row acceptance."""
    x, generator_mean = x.detach().float(), generator_mean.detach().float()
    shape = (-1,) + (1,) * (x.ndim - 1)
    with torch.no_grad():
        ex = energy(potential, x, x, labels)
        em = energy(potential, generator_mean, x, labels)
        if not (torch.isfinite(ex).all() and torch.isfinite(em).all()):
            raise FloatingPointError("Nonfinite initial proposal energy")
        from_gen = em < ex
        y = torch.where(from_gen.view(shape), generator_mean, x)
        initial = torch.minimum(em, ex)
        current = initial
    accepted_total = torch.zeros((), device=x.device)
    for _ in range(int(steps)):
        g, _ = phi_grad(potential, y, labels, create_graph=False)
        residual = (g + y - x).detach()
        if not torch.isfinite(residual).all():
            raise FloatingPointError("Nonfinite proposal refinement gradient")
        with torch.no_grad():
            norm2 = residual.double().flatten(1).square().sum(1)
            rates = torch.full((len(x),), float(lr), device=x.device)
            accepted = torch.zeros(len(x), dtype=torch.bool, device=x.device)
            for _ in range(6):
                trial = y - rates.view(shape) * residual
                et = energy(potential, trial, x, labels)
                good = (~accepted) & torch.isfinite(et) & (et <= current - 1e-4 * rates * norm2)
                y = torch.where(good.view(shape), trial, y)
                current = torch.where(good, et, current)
                accepted |= good
                if bool(accepted.all()):
                    break
                rates = torch.where(accepted, rates, rates * 0.5)
            accepted_total += accepted.float().mean()
    g, _ = phi_grad(potential, y, labels, create_graph=False)
    with torch.no_grad():
        info = {
            "pt/refine_energy_drop_per_dim": ((initial - current) / x[0].numel()).mean().float(),
            "pt/refine_residual_rms": (g + y - x).square().mean().sqrt(),
            "pt/refine_accept": accepted_total / max(1, int(steps)),
            "pt/refine_started_from_generator": from_gen.float().mean(),
            "pt/refine_distance_from_generator": (y - generator_mean).square().mean().sqrt(),
        }
    return y.detach(), info


def alignment_loss(potential, x, generator_mean, labels):
    """Temporary potential calibration on frozen W-Flow transport pairs.

Fit grad phi(m(x)) ~= x-m(x). A generic generator need not admit an exact
scalar-potential representation; this auxiliary is an empirical bootstrap,
not the original likelihood objective. It is annealed to zero.
"""
    m = generator_mean.detach().float()
    g, _ = phi_grad(potential, m, labels, create_graph=True)
    return (g + m - x.detach()).square().mean()
