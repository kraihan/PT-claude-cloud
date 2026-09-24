"""CPU checks for the PT-Flow math.  No GPU, no data, no checkpoints.

    python -m tests.test_math

The core test uses a quadratic potential, for which every object in the paper is
available in closed form, so the estimator can be checked against an exact
answer rather than against itself.  For phi(y) = (tau/2)|y - a|^2:

    psi_0(x)   = (1+tau)^{-d/2} exp( -tau |x-a|^2 / (4 eps (1+tau)) )
    phi_0(x)   = eps d log(1+tau)  +  tau |x-a|^2 / (2(1+tau))
    prox(x)    = (x + tau a) / (1 + tau)
    S^-1       = I + grad^2 phi = (1+tau) I     =>   s* = -log(1+tau)

Note that phi_0 reproduces the Hopf-Lax expansion of eq. 2.5 exactly: the Moreau
envelope tau|x-a|^2 / (2(1+tau)) plus eps log det grad^2 F = eps d log(1+tau),
with no higher correction because F is exactly quadratic.  That also means the
Laplace-matched proposal is *exact* here -- the Taylor remainder R is
identically zero -- so a correct implementation must return ESS = 1 and the
exact phi_0 from as few as K = 2 samples.
"""

from __future__ import annotations

import math
import sys

import torch

from ptflow.estimator import naive_phi0, sample_proposal, tilted_phi0
from ptflow.losses import prox_loss, scale_loss
from ptflow.potential import phi_grad

torch.manual_seed(0)

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "ok  " if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"   {detail}" if detail else ""))
    assert cond, f"{name}: {detail}"


# ---------------------------------------------------------------------------
# Analytic stand-in for PotentialNet
# ---------------------------------------------------------------------------

class QuadraticPotential(torch.nn.Module):
    """phi(y) = (tau/2) |y - a|^2, with a class-dependent centre."""

    def __init__(self, tau: float, a: torch.Tensor):
        super().__init__()
        self.tau = float(tau)
        self.register_buffer("a", a)
        self.scale = torch.nn.Parameter(torch.ones(()))

    def phi(self, x, c):
        del c
        return 0.5 * self.tau * self.scale * (x - self.a).flatten(1).pow(2).sum(1)

    def null_labels(self, c):
        return torch.zeros_like(c)

    # closed forms
    def phi0_exact(self, x, eps):
        d = x[0].numel()
        return eps * d * math.log(1.0 + self.tau) + self.tau / (
            2.0 * (1.0 + self.tau)
        ) * (x - self.a).flatten(1).pow(2).sum(1)

    def prox_exact(self, x):
        return (x + self.tau * self.a) / (1.0 + self.tau)

    def s_exact(self, like):
        return torch.full_like(like, -math.log(1.0 + self.tau))


class MildlyCubicPotential(torch.nn.Module):
    """phi(y) = (tau/2)|y|^2 + (kappa/6) sum_i y_i^3 -- a nonzero M_3."""

    def __init__(self, tau: float, kappa: float):
        super().__init__()
        self.tau, self.kappa = float(tau), float(kappa)
        self.scale = torch.nn.Parameter(torch.ones(()))

    def phi(self, x, c):
        del c
        xf = x.flatten(1) * self.scale
        return 0.5 * self.tau * xf.pow(2).sum(1) + (self.kappa / 6.0) * xf.pow(3).sum(1)

    def null_labels(self, c):
        return torch.zeros_like(c)


# ---------------------------------------------------------------------------

def test_exact_quadratic():
    print("\n1. Exact recovery of phi_0 on a quadratic potential")
    B, d, tau = 16, 32, 0.7
    shape = (B, 4, 4, 2)
    assert shape[1] * shape[2] * shape[3] == d

    a = torch.randn(1, 4, 4, 2) * 0.5
    pot = QuadraticPotential(tau, a)
    c = torch.zeros(B, dtype=torch.long)

    for eps in (0.2, 0.02, 0.005):
        x0 = torch.randn(shape)
        m0 = pot.prox_exact(x0)
        s0 = pot.s_exact(x0)

        est = tilted_phi0(pot, x0, c, m0, s0, eps, K=4, alpha_def=0.0, logw_clip=0.0)
        exact = pot.phi0_exact(x0, eps)
        rel = ((est.phi0 - exact).abs() / exact.abs().clamp_min(1e-8)).max().item()

        check(
            f"eps={eps:<6g} phi0 matches closed form",
            rel < 1e-5,
            f"max rel err {rel:.3e}",
        )
        # Tolerance is 1e-4, not 0: the residual is fp32 roundoff in the
        # quadratic forms, and it grows as 1/eps because that is how the log
        # weights are scaled.  It is exactly the precision limit discussed in
        # ptflow/potential.py, and is the reason eps_min defaults to 5e-3.
        check(
            f"eps={eps:<6g} ESS/K == 1 (proposal is exact, R == 0)",
            est.ess.min().item() > 1.0 - 1e-4,
            f"min ESS {est.ess.min().item():.6f}",
        )
        check(
            f"eps={eps:<6g} log-weight spread == 0",
            est.logw_spread.max().item() < 1e-4,
            f"max spread {est.logw_spread.max().item():.3e}",
        )


def test_defensive_mixture_unbiased():
    print("\n2. Defensive mixture leaves the estimator unbiased")
    B, tau, eps = 8, 0.7, 0.05
    a = torch.zeros(1, 4, 4, 2)
    pot = QuadraticPotential(tau, a)
    c = torch.zeros(B, dtype=torch.long)
    x0 = torch.randn(B, 4, 4, 2)
    m0, s0 = pot.prox_exact(x0), pot.s_exact(x0)
    exact = pot.phi0_exact(x0, eps)

    # With alpha > 0 some draws come from the naive component, so individual
    # weights differ; the *estimate* must still concentrate on the truth.
    for alpha in (0.0, 0.1, 0.5):
        est = tilted_phi0(
            pot, x0, c, m0, s0, eps, K=4096, alpha_def=alpha, logw_clip=0.0
        )
        rel = ((est.phi0 - exact).abs() / exact.abs().clamp_min(1e-8)).mean().item()
        check(
            f"alpha_def={alpha:<4g} estimate concentrates on phi_0",
            rel < 5e-3,
            f"mean rel err {rel:.3e}, ESS {est.ess.mean().item():.3f}",
        )


def test_variance_reversal():
    print("\n3. Variance reversal (the R0 sweep of Theorem 2.9)")
    # Naive:  s^2 ~ |grad phi|^2 / (2 eps)   -- blows up as eps -> 0
    # Tilted: s^2 ~ (5/6) d M_3^2 eps        -- vanishes as eps -> 0
    B, tau, kappa = 64, 0.5, 0.4
    pot = MildlyCubicPotential(tau, kappa)
    c = torch.zeros(B, dtype=torch.long)
    x0 = torch.randn(B, 4, 4, 2) * 0.5

    epss = [0.2, 0.1, 0.05, 0.02, 0.01]
    naive_s, tilt_s = [], []
    print("      eps      naive spread    tilted spread")
    for eps in epss:
        # first-order prox for this potential, good enough to place the proposal
        m0 = x0.clone()
        for _ in range(30):
            g, _ = phi_grad(pot, m0, c)
            m0 = x0 - g
        g, _ = phi_grad(pot, m0, c)
        # S^-1 = I + diag(grad^2 phi) = I + tau + kappa * m
        s0 = -torch.log((1.0 + tau + kappa * m0).clamp_min(1e-3))

        en = naive_phi0(pot, x0, c, eps, K=64)
        et = tilted_phi0(pot, x0, c, m0, s0, eps, K=64, alpha_def=0.0, logw_clip=0.0)
        naive_s.append(en.logw_spread.mean().item())
        tilt_s.append(et.logw_spread.mean().item())
        print(f"      {eps:<8g} {naive_s[-1]:<15.4f} {tilt_s[-1]:.6f}")

    check(
        "naive log-weight spread grows as eps cools",
        naive_s[-1] > naive_s[0] * 1.5,
        f"{naive_s[0]:.3f} -> {naive_s[-1]:.3f}",
    )
    check(
        "tilted log-weight spread shrinks as eps cools",
        tilt_s[-1] < tilt_s[0] * 0.5,
        f"{tilt_s[0]:.5f} -> {tilt_s[-1]:.5f}",
    )
    check(
        "the two curves cross (tilted wins everywhere in this range)",
        all(t < n for t, n in zip(tilt_s, naive_s)),
    )
    # Theorem 2.9 predicts s_res^2 = (5/6) d M_3^2 eps, i.e. s_res ~ sqrt(eps).
    # Over a 20x range in eps that is a 4.47x change in spread.
    ratio = tilt_s[0] / tilt_s[-1]
    predicted = math.sqrt(epss[0] / epss[-1])
    check(
        "tilted spread follows the predicted sqrt(eps) law",
        abs(ratio / predicted - 1.0) < 0.25,
        f"observed {ratio:.2f}x vs predicted {predicted:.2f}x",
    )


def test_scale_fixed_point():
    print("\n4. Scale-head loss has S^-1 = I + grad^2 phi as its minimizer")
    B, tau, eps = 64, 1.3, 0.05
    a = torch.zeros(1, 4, 4, 2)
    pot = QuadraticPotential(tau, a)
    c = torch.zeros(B, dtype=torch.long)
    x0 = torch.randn(B, 4, 4, 2)
    m = pot.prox_exact(x0)

    s = torch.zeros(B, 4, 4, 2, requires_grad=True)
    opt = torch.optim.Adam([s], lr=0.05)
    for _ in range(600):
        opt.zero_grad()
        loss, _ = scale_loss(pot, m, s, x0, c, eps, K=32)
        loss.backward()
        opt.step()

    target = -math.log(1.0 + tau)
    err = (s.detach().mean().item() - target)
    check(
        "learned log-scale converges to -log(1+tau)",
        abs(err) < 0.05,
        f"learned {s.detach().mean().item():.4f} vs exact {target:.4f}",
    )


def test_prox_loss_zero_at_prox():
    print("\n5. Prox loss vanishes exactly at the true prox")
    B, tau = 16, 0.7
    a = torch.randn(1, 4, 4, 2)
    pot = QuadraticPotential(tau, a)
    c = torch.zeros(B, dtype=torch.long)
    x0 = torch.randn(B, 4, 4, 2)

    m_true = pot.prox_exact(x0)
    l_true, mt = prox_loss(pot, m_true, x0, c, 0.0, mode="detach")
    l_id, _ = prox_loss(pot, x0.clone(), x0, c, 0.0, mode="detach")

    check("loss == 0 at the true prox", l_true.item() < 1e-10, f"{l_true.item():.3e}")
    check("loss > 0 at the identity map", l_id.item() > 1e-3, f"{l_id.item():.3e}")

    m_grad = m_true.clone().requires_grad_(True)
    l_full, _ = prox_loss(pot, m_grad, x0, c, 0.0, mode="full")
    (g,) = torch.autograd.grad(l_full, m_grad)
    check(
        "mode='full' double-backward runs and is stationary at the prox",
        g.abs().max().item() < 1e-6,
        f"max |grad| {g.abs().max().item():.3e}",
    )


def test_proposal_moments():
    print("\n6. Proposal draws have the right mean and covariance")
    B, eps, K = 4, 0.03, 20000
    x0 = torch.randn(B, 4, 4, 2)
    m0 = torch.randn(B, 4, 4, 2)
    s0 = torch.randn(B, 4, 4, 2) * 0.3

    y = sample_proposal(x0, m0, s0, eps, K, alpha_def=0.0)
    mean_err = (y.mean(dim=1) - m0).abs().max().item()
    var_emp = y.var(dim=1)
    var_exp = 2.0 * eps * torch.exp(s0)
    var_err = ((var_emp - var_exp).abs() / var_exp).max().item()

    check("proposal mean == m0", mean_err < 5e-3, f"max err {mean_err:.3e}")
    check("proposal var == 2 eps exp(s0)", var_err < 0.06, f"max rel err {var_err:.3e}")

    y_anti = sample_proposal(x0, m0, s0, eps, 8, alpha_def=0.0, antithetic=True)
    pair_sum = (y_anti[:, :4] + y_anti[:, 4:] - 2 * m0.unsqueeze(1)).abs().max().item()
    check("antithetic pairs cancel about m0", pair_sum < 1e-5, f"{pair_sum:.3e}")


def test_network_plumbing():
    print("\n7. Real networks: shapes, identity init, ESS == 1 at step 0")
    from models.generator import DitGen
    from ptflow.potential import PotentialNet, ScaleNet

    B = 4
    gen = DitGen(
        cond_dim=64, num_classes=10, input_size=8, in_channels=2, patch_size=2,
        hidden_size=64, depth=2, num_heads=4, out_channels=2,
        n_cls_tokens=0, noise_classes=0, use_bf16=False,
        residual=True,
    )
    pot = PotentialNet(
        cond_dim=64, num_classes=10, input_size=8, in_channels=2, patch_size=2,
        hidden_size=64, depth=2, num_heads=4,
    )
    # The log-scale head is a separate theta-side module, NOT a head on the
    # generator -- that is what keeps the generator checkpoint-compatible with
    # the reference release (see tests/test_ckpt_compat.py).
    scale = ScaleNet(
        cond_dim=64, num_classes=10, input_size=8, in_channels=2, patch_size=2,
        hidden_size=32, depth=2, num_heads=4,
    )
    c = torch.randint(0, 10, (B,))

    out = gen(c=c, cfg_scale=1.0)
    m, x0 = out["samples"], out["noise"]["x"]
    s = scale(m, c)
    check("generator returns x0; scale net matches its shape",
          m.shape == x0.shape == s.shape)
    check(
        "residual + zero-init => generator is the identity at step 0",
        torch.allclose(m, x0, atol=1e-6),
        f"max dev {(m - x0).abs().max().item():.3e}",
    )
    check(
        "zero-init potential => phi == 0 at step 0",
        pot.phi(x0, c).abs().max().item() < 1e-6,
    )
    check("log_scale == 0 at step 0 (S = I)", s.abs().max().item() < 1e-6)

    est = tilted_phi0(pot, x0, c, m.detach(), s.detach(), 0.1, K=8, alpha_def=0.1)
    check(
        "joint init is inside the feasible region: ESS/K == 1",
        est.ess.min().item() > 1.0 - 1e-5,
        f"min ESS {est.ess.min().item():.6f}",
    )

    # guided gradient, both label branches
    from ptflow.potential import guided_phi_grad

    g = guided_phi_grad(pot, x0, c, 1.5)
    check("guided potential gradient has the input shape", g.shape == x0.shape)


def test_schedule_controller():
    print("\n8. Schedule: ESS gates the anneal and the prox ramp")
    from ptflow.schedule import build_schedule

    sched = build_schedule(
        dict(
            eps_max=0.2, eps_min=0.01, eps_anneal_steps=100, eps_warmup=0,
            prox_warmup=10, prox_ramp=50, lambda_prox_max=1.0, ema_decay=0.0,
        )
    )
    for _ in range(200):
        sched.observe(0.9)
    healthy_eps, healthy_lam = sched.eps(), sched.lambda_prox()
    check("healthy run cools eps to the floor", abs(healthy_eps - 0.01) < 1e-6, f"eps={healthy_eps:.4g}")
    check("healthy run ramps lambda_prox to max", abs(healthy_lam - 1.0) < 1e-6, f"lam={healthy_lam:.4g}")

    sched2 = build_schedule(
        dict(
            eps_max=0.2, eps_min=0.01, eps_anneal_steps=100, eps_warmup=0,
            prox_warmup=10, prox_ramp=50, lambda_prox_max=1.0, ema_decay=0.0,
        )
    )
    for _ in range(200):
        sched2.observe(0.01)  # broken
    check("broken run never cools eps", abs(sched2.eps() - 0.2) < 1e-6, f"eps={sched2.eps():.4g}")
    check("broken run keeps lambda_prox at 0", sched2.lambda_prox() < 1e-9)
    check("broken run freezes the potential step", not sched2.update_potential())
    check("broken run raises alpha_def", sched2.alpha_def() > 0.2, f"alpha={sched2.alpha_def():.3g}")

    # mid-run collapse walks the PT term back toward the OT-drift baseline
    sched3 = build_schedule(
        dict(eps_max=0.2, eps_min=0.01, eps_anneal_steps=100, eps_warmup=0,
             prox_warmup=0, prox_ramp=10, lambda_prox_max=1.0, ema_decay=0.0,
             prox_decay_on_degrade=0.9)
    )
    for _ in range(50):
        sched3.observe(0.9)
    before = sched3.lambda_prox()
    for _ in range(50):
        sched3.observe(0.1)
    check(
        "mid-run ESS collapse decays lambda_prox back toward the OT-drift baseline",
        sched3.lambda_prox() < before * 0.1,
        f"{before:.3f} -> {sched3.lambda_prox():.4f}",
    )


def test_generator_adds_no_parameters():
    print("\n9. The PT-Flow generator flag adds no parameters")
    # The checkpoint contract in one assertion: `residual` is a behaviour flag,
    # so toggling it must not change a single tensor in the state_dict.  The
    # full per-size contract against upstream lives in tests/test_ckpt_compat.py.
    from models.generator import DitGen

    kw = dict(cond_dim=64, num_classes=10, input_size=8, in_channels=2,
              patch_size=2, hidden_size=64, depth=2, num_heads=4, out_channels=2,
              n_cls_tokens=0, noise_classes=0, use_bf16=False)
    a = DitGen(residual=False, **kw)
    b = DitGen(residual=True, **kw)
    sa = {k: tuple(v.shape) for k, v in a.state_dict().items()}
    sb = {k: tuple(v.shape) for k, v in b.state_dict().items()}
    check("residual=True/False give identical state_dicts", sa == sb,
          f"{len(sa)} tensors, {sum(p.numel() for p in a.parameters()):,} params")

    # ...and it does change the map, which is why it must stay false on resume.
    c = torch.zeros(2, dtype=torch.long)
    x0 = torch.randn(2, 8, 8, 2)
    ya = a(c=c, cfg_scale=1.0, x0=x0)["samples"]
    yb = b(c=c, cfg_scale=1.0, x0=x0)["samples"]
    check("residual=True does change the map (so it must be false on resume)",
          not torch.allclose(ya, yb),
          f"||diff|| = {(ya - yb).norm().item():.4f}")


if __name__ == "__main__":
    print("PT-Flow math checks (CPU)")
    test_exact_quadratic()
    test_defensive_mixture_unbiased()
    test_variance_reversal()
    test_scale_fixed_point()
    test_prox_loss_zero_at_prox()
    test_proposal_moments()
    test_network_plumbing()
    test_schedule_controller()
    test_generator_adds_no_parameters()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed:", ", ".join(FAIL))
    sys.exit(1 if FAIL else 0)
