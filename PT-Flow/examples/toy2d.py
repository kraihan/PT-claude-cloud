"""PT-Flow on the OT-drift baseline, end to end, in 2-D.  Runs on a laptop CPU in ~2 minutes.

    python -m examples.toy2d --steps 3000 --out runs/toy2d

This is the whole method in miniature, using the *real* modules -- the baseline's
``ot_drift_loss`` for the arm, and ``ptflow.estimator`` / ``ptflow.losses`` /
``ptflow.schedule`` for the payload.  Only the two networks are replaced by MLPs, so
what is exercised is the graft itself rather than the DiT plumbing (which
tests/test_train_smoke.py covers).

It answers the question the ImageNet configs cannot answer without a cluster:
does adding the PT-Flow term to a working the OT-drift baseline run help, hurt, or destabilize?
Here you can watch ESS, the eps anneal, the lambda_prox ramp and sample quality
on one screen, and you can turn the PT half off with --lambda-prox 0 to see the
the OT-drift baseline it is being measured against.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("DRIFT_COMPILE", "0")

import torch
import torch.nn as nn

from ptflow.ot_drift import ot_drift_loss
from ptflow.estimator import tilted_phi0
from ptflow.losses import potential_loss, prox_loss, scale_loss
from ptflow.potential import guided_phi_grad, prox_residual
from ptflow.sampling import snis_resample  # noqa: F401  (re-exported for notebooks)
from ptflow.schedule import build_schedule


# ---------------------------------------------------------------------------
# Data: eight Gaussians on a ring, unit-ish scale (assumption A5)
# ---------------------------------------------------------------------------

def sample_data(n: int, device, std: float = 0.12) -> torch.Tensor:
    k = torch.randint(0, 8, (n,), device=device)
    ang = k.float() * (2 * math.pi / 8)
    centres = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1) * 1.6
    return centres + std * torch.randn(n, 2, device=device)


# ---------------------------------------------------------------------------
# Networks.  Both satisfy the duck-typed contracts the pt.* modules expect.
# ---------------------------------------------------------------------------

def mlp(din: int, dout: int, width: int, depth: int, zero_last: bool) -> nn.Sequential:
    layers, d = [], din
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.SiLU()]
        d = width
    last = nn.Linear(d, dout)
    if zero_last:
        # Mirrors the baseline's zero-init FinalLayer: the generator starts at the
        # identity and the potential starts at zero, which is what makes the
        # joint initialization feasible (ESS/K = 1).
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


class ToyGenerator(nn.Module):
    """m_eta(x0) = x0 + net(x0), plus a diagonal log-scale head."""

    def __init__(self, width=256, depth=3, scale_max=3.0):
        super().__init__()
        self.net = mlp(2, 4, width, depth, zero_last=True)
        self.scale_max = float(scale_max)

    def forward(self, x0):
        out = self.net(x0)
        delta, raw_s = out[:, :2], out[:, 2:]
        m = x0 + delta
        s = self.scale_max * torch.tanh(raw_s / self.scale_max)
        return m, s


class ToyPotential(nn.Module):
    """phi_theta(x).  Unconditional, so null_labels is the identity."""

    def __init__(self, width=256, depth=3, phi_scale=2.0):
        super().__init__()
        self.net = mlp(2, 1, width, depth, zero_last=True)
        self.phi_scale = float(phi_scale)

    def phi(self, x, c=None):
        del c
        return self.phi_scale * self.net(x.float()).squeeze(-1)

    def null_labels(self, c):
        return c


# ---------------------------------------------------------------------------

def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    gen = ToyGenerator().to(device)
    pot = ToyPotential().to(device)
    opt_g = torch.optim.AdamW(gen.parameters(), lr=args.lr_gen)
    opt_p = torch.optim.AdamW(pot.parameters(), lr=args.lr_pot)

    sched = build_schedule(dict(
        eps_max=args.eps_max, eps_min=args.eps_min,
        eps_anneal_steps=int(args.steps * 0.6), eps_warmup=int(args.steps * 0.05),
        alpha_def_start=0.1, alpha_def_end=0.01, alpha_def_degraded=0.25,
        prox_warmup=int(args.steps * args.prox_warmup_frac),
        prox_ramp=int(args.steps * 0.2),
        lambda_prox_max=args.lambda_prox,
        lambda_scale=0.1,
        ess_healthy=0.3, ess_broken=0.05, ess_ema_decay=0.95,
    ))

    zeros = torch.zeros(args.batch, dtype=torch.long, device=device)
    hist = {k: [] for k in
            ("step", "loss_ot", "loss_prox", "loss_pot", "ess", "eps",
             "lambda_prox", "logw_spread", "prox_resid")}

    for step in range(args.steps):
        x1 = sample_data(args.batch, device)
        x0 = torch.randn(args.batch, 2, device=device)

        # ---- (A) generator step: the OT-drift baseline OT arm + ramped PT-Flow prox term ----
        opt_g.zero_grad(set_to_none=True)
        m, s = gen(x0)

        # the baseline's real debiased OT loss, on [B=1, N, D].
        loss_ot, _ = ot_drift_loss(
            gen=m[None], fixed_pos=x1[None],
            fixed_neg=m.detach()[None],
            R_list=(0.05,), sinkhorn_num_iter=20,
            disable_diag_mask=True, use_quadratic_cost=True,
        )
        loss_ot = loss_ot.mean()

        lam = sched.lambda_prox()
        loss_g, l_prox = loss_ot, torch.zeros((), device=device)
        if lam > 0.0:
            l_prox, mprox = prox_loss(pot, m, x0, zeros, 0.0, mode="detach")
            loss_g = loss_g + lam * l_prox
        if step >= sched.prox_warmup and not sched.health.is_broken:
            l_scale, _ = scale_loss(pot, m, s, x0, zeros, sched.eps(), K=4)
            loss_g = loss_g + sched.lambda_scale * l_scale

        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 2.0)
        opt_g.step()

        # ---- (B) potential step: the flat objective, at frozen eta ----------
        eps = sched.eps()
        ess = float(sched.health.value)
        loss_p = torch.zeros((), device=device)
        spread = 0.0
        if sched.update_potential():
            with torch.no_grad():
                m0, s0 = gen(x0)
            opt_p.zero_grad(set_to_none=True)
            loss_p, est, mp = potential_loss(
                pot, x0, zeros, x1, zeros, m0.detach(), s0.detach(), eps,
                K=args.K, alpha_def=sched.alpha_def(), lambda_gauge=0.1,
            )
            loss_p.backward()
            torch.nn.utils.clip_grad_norm_(pot.parameters(), 1.0)
            opt_p.step()
            ess = float(est.ess.mean())
            spread = float(est.logw_spread.mean())

        # ---- (C) monitor and anneal -----------------------------------------
        sched.observe(ess)

        if step % args.log_every == 0 or step == args.steps - 1:
            hist["step"].append(step)
            hist["loss_ot"].append(float(loss_ot))
            hist["loss_prox"].append(float(l_prox))
            hist["loss_pot"].append(float(loss_p))
            hist["ess"].append(ess)
            hist["eps"].append(eps)
            hist["lambda_prox"].append(lam)
            hist["logw_spread"].append(spread)
            # The prox residual, relative to transport displacement.  This is
            # the number that says whether the identity m = prox_phi is actually
            # holding, which is what the whole method rests on.
            r = prox_residual(pot, m.detach(), x0, zeros, 0.0)
            disp = (m.detach() - x0).norm(dim=1).clamp_min(1e-6)
            hist["prox_resid"].append(float((r.norm(dim=1) / disp).mean()))
            if step % (args.log_every * 10) == 0:
                print(f"  step {step:6d}  ot={float(loss_ot):.4f}  "
                      f"pot={float(loss_p):+.4f}  ESS={ess:.3f}  "
                      f"eps={eps:.4f}  lam={lam:.3f}  spread={spread:.3f}  "
                      f"prox_rel={hist['prox_resid'][-1]:.3f}")

    return gen, pot, sched, hist


# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(gen, pot, sched, device, n=4096, seed=0):
    """Energy distance to the data, for Modes A and C; Mode B needs gradients."""
    torch.manual_seed(seed)
    eps = sched.eps()
    x0 = torch.randn(n, 2, device=device)
    real = sample_data(n, device)
    zeros = torch.zeros(n, dtype=torch.long, device=device)

    m, s = gen(x0)
    out = {"A": m}

    # Mode C: SNIS against the tilted proposal
    est = tilted_phi0(pot, x0, zeros, m, s, eps, K=32, alpha_def=0.05, logw_clip=0.0)
    out["C"] = snis_resample(est.y, est.log_w)
    return out, real, est


def energy_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """2 E|a-b| - E|a-a'| - E|b-b'|.  Zero iff the distributions match."""
    def m(u, v):
        return torch.cdist(u, v).mean()
    return float(2 * m(a, b) - m(a, a) - m(b, b))


@torch.enable_grad()
def mode_b(gen, pot, x0, n_steps=8, gamma=0.5):
    with torch.no_grad():
        y, _ = gen(x0)
    zeros = torch.zeros(x0.shape[0], dtype=torch.long, device=x0.device)
    for _ in range(n_steps):
        g = guided_phi_grad(pot, y, zeros, 0.0)
        y = y - gamma * (g + y - x0)
    return y.detach()


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--lr-gen", type=float, default=2e-4)
    p.add_argument("--lr-pot", type=float, default=1e-4)
    p.add_argument("--eps-max", type=float, default=0.2)
    p.add_argument("--eps-min", type=float, default=0.01)
    p.add_argument("--lambda-prox", type=float, default=0.3,
                   help="0.0 reproduces the plain baseline.")
    p.add_argument("--prox-warmup-frac", type=float, default=0.2)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="runs/toy2d")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    print(f"PT-Flow 2-D demo   device={args.device}  lambda_prox_max={args.lambda_prox}")
    gen, pot, sched, hist = train(args)

    device = torch.device(args.device)
    out, real, est = evaluate(gen, pot, sched, device)
    torch.manual_seed(0)
    x0 = torch.randn(4096, 2, device=device)
    out["B"] = mode_b(gen, pot, x0, n_steps=8, gamma=0.5)

    print("\nEnergy distance to data (lower is better):")
    results = {}
    for k in ("A", "B", "C"):
        ed = energy_distance(out[k], real)
        results[f"mode_{k}"] = ed
        label = {"A": "Mode A (1-NFE)", "B": "Mode B (8-NFE refine)",
                 "C": "Mode C (SNIS K=32)"}[k]
        print(f"  {label:<24s} {ed:.5f}")
    results["final_ess"] = hist["ess"][-1]
    results["final_eps"] = hist["eps"][-1]
    results["lambda_prox_max"] = args.lambda_prox

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "results.json").write_text(json.dumps(results, indent=2))
    (outdir / "history.json").write_text(json.dumps(hist))
    print(f"\nWrote {outdir/'results.json'}")

    if args.plot:
        make_plots(hist, out, real, outdir)
        print(f"Wrote {outdir/'toy2d.png'}")


def make_plots(hist, out, real, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 4, figsize=(18, 8))
    r = real.cpu()
    for j, k in enumerate(("A", "B", "C")):
        a = ax[0, j]
        a.scatter(r[:, 0].cpu(), r[:, 1].cpu(), s=2, alpha=0.15, c="0.6", label="data")
        s = out[k].cpu()
        a.scatter(s[:, 0], s[:, 1], s=2, alpha=0.35, c="C0", label=f"Mode {k}")
        a.set_title(f"Mode {k}")
        a.set_aspect("equal"); a.set_xlim(-2.5, 2.5); a.set_ylim(-2.5, 2.5)
        a.legend(markerscale=6, fontsize=8)

    ax[0, 3].plot(hist["step"], hist["ess"], lw=1)
    ax[0, 3].axhline(0.3, ls="--", c="C2", lw=1, label="healthy")
    ax[0, 3].axhline(0.05, ls="--", c="C3", lw=1, label="broken")
    ax[0, 3].set_title("ESS / K"); ax[0, 3].set_ylim(0, 1.05); ax[0, 3].legend(fontsize=8)

    for a, key, title in (
        (ax[1, 0], "loss_ot", "the OT-drift baseline OT loss (the arm)"),
        (ax[1, 1], "loss_pot", "PT-Flow potential loss"),
        (ax[1, 2], "logw_spread", "log-weight spread  (s_res)"),
    ):
        a.plot(hist["step"], hist[key], lw=1); a.set_title(title)

    a = ax[1, 3]
    a.plot(hist["step"], hist["eps"], lw=1, label="eps")
    a.plot(hist["step"], hist["lambda_prox"], lw=1, label="lambda_prox")
    a.set_title("schedules (both ESS-gated)"); a.legend(fontsize=8)

    for a in ax.ravel():
        a.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "toy2d.png", dpi=110)


if __name__ == "__main__":
    main()
