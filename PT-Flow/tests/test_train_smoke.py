"""End-to-end smoke test of the PT-Flow training step, on CPU with fake data.

    python -m tests.test_train_smoke

No dataset, no MAE weights, no GPU.  A stub feature extractor stands in for the
real one so that the baseline half of ``train_step`` runs unchanged and the PT-Flow
half is exercised on the code path it will actually take.

The load-bearing assertion is the last one: with lambda_prox = 0 the generator's
parameters after one step must be **bit-identical** to a the plain baseline step.
That is the whole "keep the arm on the OT-drift baseline" contract -- PT-Flow can only ever add
to the run once its own ESS diagnostic says it is ready, and until then it must
be provably inert.
"""

from __future__ import annotations

import copy
import os
import sys

os.environ.setdefault("DRIFT_COMPILE", "0")

import torch

from models.generator import DitGen
from ptflow.potential import PotentialNet, ScaleNet
from ptflow.schedule import build_schedule
from train import PTBundle, TrainState, train_step

torch.manual_seed(0)

PASS, FAIL = [], []
DEV = torch.device("cpu")

BSZ, N_POS, N_UNCOND, GEN_PER_LABEL = 4, 4, 2, 3
HW, CH, NUM_CLASSES = 8, 2, 10


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    assert cond, f"{name}: {detail}"
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def fake_feature_apply(params, x, **kwargs):
    """Stand-in for the MAE encoder: [N,H,W,C] -> {"f": [N, tokens, dim]}."""
    del params, kwargs
    n = x.shape[0]
    return {"f": x.reshape(n, 4, -1).float()}


def make_generator(**kw):
    return DitGen(
        cond_dim=64, num_classes=NUM_CLASSES, input_size=HW, in_channels=CH,
        patch_size=2, hidden_size=64, depth=2, num_heads=4, out_channels=CH,
        n_cls_tokens=0, noise_classes=0, use_bf16=False, attn_fp32=True, **kw
    )


def make_state(pt_enabled: bool, *, lambda_prox_max=0.0, prox_warmup=0,
               with_scale=False, residual=False):
    torch.manual_seed(1234)
    model = make_generator(residual=residual)
    ema = copy.deepcopy(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    pt = None
    if pt_enabled:
        torch.manual_seed(4321)
        potential = PotentialNet(
            cond_dim=64, num_classes=NUM_CLASSES, input_size=HW, in_channels=CH,
            patch_size=2, hidden_size=64, depth=2, num_heads=4,
        )
        scale = None
        if with_scale:
            scale = ScaleNet(
                cond_dim=64, num_classes=NUM_CLASSES, input_size=HW,
                in_channels=CH, patch_size=2, hidden_size=32, depth=2, num_heads=4,
            )
        theta_params = list(potential.parameters())
        if scale is not None:
            theta_params += list(scale.parameters())
        pt = PTBundle(
            potential=potential,
            ema_potential=copy.deepcopy(potential),
            scale=scale,
            ema_scale=copy.deepcopy(scale) if scale is not None else None,
            optimizer=torch.optim.AdamW(theta_params, lr=1e-4),
            sched=build_schedule(dict(
                eps_max=0.1, eps_min=0.01, eps_anneal_steps=100, eps_warmup=0,
                prox_warmup=prox_warmup, prox_ramp=10,
                lambda_prox_max=lambda_prox_max, lambda_scale=0.1,
            )),
            lr_fn=lambda s: 1e-4,
            cfg=dict(K=4, noise_bsz=BSZ, data_bsz=4, prox_max_batch=6, scale_K=2),
        )
    return TrainState(step=0, model=model, optimizer=opt, ema_model=ema, ema_decay=0.99, pt=pt)


def run_step(state, ot_mode="debiased"):
    torch.manual_seed(7)
    labels = torch.randint(0, NUM_CLASSES, (BSZ,))
    samples = torch.randn(BSZ, N_POS, HW, HW, CH)
    negatives = torch.randn(BSZ, N_UNCOND, HW, HW, CH)

    return train_step(
        state, labels, samples, negatives, {}, fake_feature_apply,
        learning_rate_fn=lambda s: 1e-4,
        cfg_min=1.0, cfg_max=2.0, neg_cfg_pw=1.0, no_cfg_frac=0.0,
        gen_per_label=GEN_PER_LABEL,
        activation_kwargs={},
        loss_kwargs=dict(R_list=[0.05]),
        max_grad_norm=2.0, grad_accum_steps=2, device=DEV,
        ot_mode=ot_mode,
        ot_kwargs=dict(sinkhorn_num_iter=3, use_new_cfg=True,
                       disable_diag_mask=True, use_quadratic_cost=True),
    )


# ---------------------------------------------------------------------------

def test_baseline_baseline_still_runs():
    print("\n1. the OT-drift baseline path is untouched (pt=None)")
    for mode in ("debiased", "none"):
        state = make_state(pt_enabled=False)
        state, metrics = run_step(state, ot_mode=mode)
        check(f"ot_mode={mode!r} step completes", torch.isfinite(metrics["loss"]).item(),
              f"loss={metrics['loss'].item():.4f}")
        check(f"ot_mode={mode!r} no PT metrics leak in",
              not any(k.startswith("pt/") for k in metrics))


def test_pt_enabled_runs():
    print("\n2. PT-Flow enabled: potential trains, diagnostics appear")
    # residual=True so the generator warm-starts at the identity -- see test 7
    # for what happens without it.
    state = make_state(pt_enabled=True, lambda_prox_max=1.0, prox_warmup=0,
                       with_scale=True, residual=True)
    before = [p.detach().clone() for p in state.pt.potential.parameters()]

    # Two steps: lambda_prox is exactly 0 on the first one by construction (the
    # ramp starts at zero and only advances once ESS has been observed), so the
    # prox term cannot be asserted active until step 2.
    state, _ = run_step(state)
    state, metrics = run_step(state)

    check("step completes with finite loss", torch.isfinite(metrics["loss"]).item(),
          f"loss={metrics['loss'].item():.4f}")
    for key in ("pt/ess", "pt/sched_eps", "pt/sched_lambda_prox", "pt/loss_potential",
                "pt/phi_data", "pt/phi0_noise", "pt/gauge", "pt/logw_spread",
                "pt/g_norm_potential", "pt/health"):
        check(f"metric {key} logged", key in metrics and torch.isfinite(metrics[key]).all().item(),
              f"{metrics[key].item():.4g}" if key in metrics else "MISSING")

    after = [p.detach() for p in state.pt.potential.parameters()]
    changed = sum(1 for a, b in zip(before, after) if not torch.equal(a, b))
    check("potential parameters were updated", changed > 0, f"{changed}/{len(before)} tensors")

    check("ESS starts near 1 (joint init is feasible)",
          metrics["pt/ess"].item() > 0.9, f"ESS={metrics['pt/ess'].item():.5f}")
    check("prox term active and finite",
          "pt/loss_prox" in metrics and torch.isfinite(metrics["pt/loss_prox"]).item(),
          f"{metrics.get('pt/loss_prox', float('nan'))}")
    check("lambda_prox actually applied is logged separately from the schedule",
          "pt/lambda_prox" in metrics and metrics["pt/lambda_prox"].item() > 0.0,
          f"applied={metrics['pt/lambda_prox'].item():.4g} "
          f"next={metrics['pt/sched_lambda_prox'].item():.4g}")
    check("scale term active and finite (now on the theta side)",
          "pt/loss_scale" in metrics and torch.isfinite(metrics["pt/loss_scale"]).item())
    check("scale net parameters were updated too",
          any(p.grad is not None for p in state.pt.scale.parameters()))


def test_lambda_prox_zero_is_bit_identical():
    print("\n3. lambda_prox = 0 is bit-identical to the plain baseline")
    s_off = make_state(pt_enabled=False)
    s_off, m_off = run_step(s_off)

    # PT enabled, but the ramp has not started: lambda_prox must be exactly 0.
    s_on = make_state(pt_enabled=True, lambda_prox_max=1.0, prox_warmup=10_000)
    s_on, m_on = run_step(s_on)

    check("lambda_prox is exactly zero before warmup",
          m_on["pt/sched_lambda_prox"].item() == 0.0)
    check("the OT-drift baseline loss is identical",
          torch.equal(m_off["loss"], m_on["loss"]),
          f"{m_off['loss'].item():.10f} vs {m_on['loss'].item():.10f}")

    same = all(
        torch.equal(a.detach(), b.detach())
        for a, b in zip(s_off.model.parameters(), s_on.model.parameters())
    )
    check("generator parameters after one step are bit-identical", same)


def test_broken_ess_freezes_potential():
    print("\n4. A broken estimator freezes theta rather than diverging")
    state = make_state(pt_enabled=True, lambda_prox_max=1.0, prox_warmup=0)
    state.pt.sched.health.value = 0.0
    state.pt.sched.health.state = "broken"
    before = [p.detach().clone() for p in state.pt.potential.parameters()]

    state, metrics = run_step(state)

    unchanged = all(
        torch.equal(a, b)
        for a, b in zip(before, [p.detach() for p in state.pt.potential.parameters()])
    )
    check("potential step was skipped", unchanged)
    check("lambda_prox held at zero", metrics["pt/sched_lambda_prox"].item() == 0.0)
    # Compared with a tolerance, not for equality: the metric is a float32
    # tensor and 0.1 does not survive the round trip exactly.
    check("eps did not cool",
          abs(metrics["pt/sched_eps"].item() - state.pt.sched.eps_max) < 1e-6,
          f"eps={metrics['pt/sched_eps'].item():.6g}")


def test_multi_step_stability():
    print("\n5. Ten consecutive steps stay finite")
    state = make_state(pt_enabled=True, lambda_prox_max=1.0, prox_warmup=2,
                       with_scale=True, residual=True)
    losses, esss = [], []
    for _ in range(10):
        state, m = run_step(state)
        losses.append(m["loss"].item())
        esss.append(m["pt/ess"].item())
    check("all losses finite", all(l == l and abs(l) < 1e6 for l in losses),
          f"last={losses[-1]:.4f}")
    check("all ESS values in (0, 1]", all(0.0 < e <= 1.0 + 1e-6 for e in esss),
          f"min={min(esss):.4f} max={max(esss):.4f}")
    check("schedule advanced", state.pt.sched.step == 10, f"step={state.pt.sched.step}")


def test_inference_modes():
    print("\n6. Inference Modes A / B / C and the likelihood run")
    from ptflow.sampling import log_likelihood, sample_mode_a, sample_mode_b, sample_mode_c

    state = make_state(pt_enabled=True, lambda_prox_max=1.0, prox_warmup=0,
                       with_scale=True, residual=True)
    for _ in range(3):
        state, _ = run_step(state)

    gen, pot = state.model.eval(), state.pt.potential.eval()
    scl = state.pt.scale.eval() if state.pt.scale is not None else None
    c = torch.randint(0, NUM_CLASSES, (4,))
    eps = state.pt.sched.eps()

    a = sample_mode_a(gen, c, cfg_scale=1.5)
    check("Mode A shape and finiteness", a.shape == (4, HW, HW, CH) and torch.isfinite(a).all())

    b, trace = sample_mode_b(gen, pot, c, cfg_scale=1.5, n_steps=4, gamma=0.5,
                            return_trace=True)
    check("Mode B shape and finiteness", b.shape == a.shape and torch.isfinite(b).all())
    check("Mode B residual decreases monotonically",
          all(t2 <= t1 + 1e-8 for t1, t2 in zip(trace, trace[1:])),
          " -> ".join(f"{t:.4f}" for t in trace))

    cc, ess = sample_mode_c(gen, pot, c, eps, cfg_scale=1.5, K=8,
                            scale_net=scl, return_ess=True)
    check("Mode C shape and finiteness", cc.shape == a.shape and torch.isfinite(cc).all())
    check("Mode C ESS in (0, 1]", bool(((ess > 0) & (ess <= 1 + 1e-6)).all()),
          f"mean={ess.mean().item():.4f}")

    x1 = torch.randn(2, HW, HW, CH)
    lp, info = log_likelihood(gen, pot, x1, c[:2], eps, K_outer=4, K_inner=4,
                              scale_net=scl)
    check("likelihood is finite", torch.isfinite(lp).all(),
          f"nll/dim={info['nll_per_dim']:.4f} nats")


def test_residual_flag_governs_the_init():
    """Why `residual` is not a free choice when training from scratch.

    With residual=True the generator is the identity at step 0, which is exactly
    prox of the zero potential, so the proposal is perfectly placed and
    ESS/K = 1.  With residual=False the zero-init final layer makes the
    generator output *zero* rather than x0, so the proposal sits at the origin
    while x0 ~ N(0, I) -- a misplaced proposal, and the ESS says so immediately.

    Hence the config rule: from scratch use residual=true; warm-starting from a
    baseline checkpoint use residual=false, where m is already good for a
    different reason.
    """
    print("\n7. residual flag governs whether the joint init is feasible")
    from ptflow.estimator import tilted_phi0

    torch.manual_seed(11)
    c = torch.randint(0, NUM_CLASSES, (8,))
    pot = PotentialNet(cond_dim=64, num_classes=NUM_CLASSES, input_size=HW,
                       in_channels=CH, patch_size=2, hidden_size=64, depth=2,
                       num_heads=4)
    scl = ScaleNet(cond_dim=64, num_classes=NUM_CLASSES, input_size=HW,
                   in_channels=CH, patch_size=2, hidden_size=32, depth=2,
                   num_heads=4)

    for residual, want_high in ((True, True), (False, False)):
        gen = make_generator(residual=residual)
        out = gen(c=c, cfg_scale=1.0)
        m = out["samples"].detach()
        est = tilted_phi0(pot, out["noise"]["x"], c, m,
                          scl(m, c).detach(), 0.1, K=8, alpha_def=0.1)
        ess = est.ess.mean().item()
        check(
            f"residual={residual}: ESS at init is {'high' if want_high else 'low'}",
            (ess > 0.99) if want_high else (ess < 0.9),
            f"ESS={ess:.4f}, log-weight spread={est.logw_spread.mean().item():.3f}",
        )


if __name__ == "__main__":
    print("PT-Flow training smoke test (CPU, fake data)")
    test_baseline_baseline_still_runs()
    test_pt_enabled_runs()
    test_lambda_prox_zero_is_bit_identical()
    test_broken_ess_freezes_potential()
    test_multi_step_stability()
    test_inference_modes()
    test_residual_flag_governs_the_init()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed:", ", ".join(FAIL))
    sys.exit(1 if FAIL else 0)
