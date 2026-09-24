import copy
import torch
import pytest
from ptflow.schedule import build_schedule
from ptflow.recovery import refine_proposal, alignment_loss, energy
from ptflow.estimator import tilted_phi0
from ptflow.losses import prox_loss


def test_streamed_real_features_bound_batch_and_preserve_pt_update():
    from tests.test_train_smoke import make_state, fake_feature_apply, HW, CH
    from train import train_step
    initial = make_state(True, with_scale=True)
    initial.pt.sched = build_schedule(dict(policy="active_recovery_v1", prox_warmup=0,
        prox_ramp=1, lambda_prox_max=.1, proposal_refine_steps=2, alignment_batch=2))
    initial.pt.sched.prox_progress = 1.
    initial.pt.cfg.update(prox_mode="full", prox_norm="none", prox_max_batch=2)
    eager, streamed = copy.deepcopy(initial), copy.deepcopy(initial)
    x = torch.randn(4, 4, HW, HW, CH)
    neg = torch.randn(4, 2, HW, HW, CH)
    sizes = []
    def spy(params, images, **kwargs):
        if not torch.is_grad_enabled():
            sizes.append(len(images))
        return fake_feature_apply(params, images, **kwargs)
    common = dict(gen_per_label=3, grad_accum_steps=2, ot_mode="debiased",
        ot_kwargs=dict(resample_neg=True, sinkhorn_num_iter=3,
            use_new_cfg=True, use_quadratic_cost=True, disable_diag_mask=True))
    torch.manual_seed(921)
    _, old_metrics = train_step(eager, torch.arange(4), x, neg, {}, spy,
                               feature_chunk_size=0, **common)
    assert max(sizes) == 24
    sizes.clear()
    torch.manual_seed(921)
    _, new_metrics = train_step(streamed, torch.arange(4), x, neg, {}, spy,
                               feature_chunk_size=16, **common)
    assert max(sizes) == 12
    torch.testing.assert_close(old_metrics["loss"], new_metrics["loss"])
    assert new_metrics["pt/lambda_prox"] > 0
    assert new_metrics["pt/potential_updated"] == 1
    assert new_metrics["pt/scale_updated"] == 1
    for a, b in ((eager.model, streamed.model), (eager.pt.potential, streamed.pt.potential),
                 (eager.pt.scale, streamed.pt.scale)):
        for pa, pb in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(pa, pb)


def test_chunked_checkpointed_mae_preserves_features_and_input_gradient(monkeypatch):
    from models import mae_model
    model = mae_model.MAEResNetJAX(base_channels=8, in_channels=4, layers=(1, 1, 1, 1))
    model.eval().requires_grad_(False)
    monkeypatch.setattr(mae_model, "build_feature_model_and_params", lambda **kw: (model, {}))
    plain, _ = mae_model.build_activation_function(mae_path="unused", feature_chunk_size=0)
    bounded, _ = mae_model.build_activation_function(mae_path="unused", feature_chunk_size=2,
                                                    checkpoint_features=True)
    x = torch.randn(4, 16, 16, 4, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    options = dict(patch_mean_size=[2], patch_std_size=[2], every_k_block=1)
    a, b = plain({}, x, **options), bounded({}, y, **options)
    assert set(a) == set(b)
    for k in a:
        torch.testing.assert_close(a[k], b[k], rtol=1e-4, atol=1e-5)
    sum(v.square().mean() for v in a.values()).backward()
    sum(v.square().mean() for v in b.values()).backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=2e-4, atol=1e-5)
    assert all(p.grad is None for p in model.parameters())


def test_requeue_saves_private_run_state_before_exit(tmp_path):
    from tests.test_train_smoke import make_state
    from ptflow.run_guard import inspect_boundary
    s = make_state(True, with_scale=True)
    s.pt.sched = build_schedule(dict(policy="active_recovery_v1"))
    s.pt.sched.observe(.9)
    s.step = 1
    (tmp_path / "requeue.request").touch()
    with pytest.raises(SystemExit) as exit_info:
        inspect_boundary(s, {}, tmp_path)
    assert exit_info.value.code == 0
    assert (tmp_path / "requeue.ready").read_text() == "1"
    assert (tmp_path / "checkpoints/state_00000001.pt").is_file()
    assert (tmp_path / "pt_status.json").is_file()
    assert not (tmp_path / "PT_FAILED.txt").exists()


def test_remat_matches_eager_for_full_recovery_update():
    from tests.test_train_smoke import make_state, run_step
    eager = make_state(True, with_scale=True, lambda_prox_max=.1)
    eager.pt.sched = build_schedule(dict(policy="active_recovery_v1", prox_warmup=0,
                                        prox_ramp=1, lambda_prox_max=.1,
                                        proposal_refine_steps=2, alignment_batch=2))
    eager.pt.sched.prox_progress = 1.
    eager.pt.cfg.update(prox_mode="full", prox_norm="none", prox_max_batch=2)
    with torch.no_grad():
        for module in (eager.model, eager.pt.potential, eager.pt.scale):
            for p in module.parameters():
                p.add_(torch.randn_like(p) * .003)
    remat = copy.deepcopy(eager)
    remat.model.model.use_remat = True
    remat.pt.potential.trunk.use_remat = True
    _, a = run_step(eager)
    _, b = run_step(remat)
    torch.testing.assert_close(a["loss"], b["loss"])
    for m, n in ((eager.model, remat.model), (eager.pt.potential, remat.pt.potential),
                 (eager.pt.scale, remat.pt.scale)):
        for p, q in zip(m.parameters(), n.parameters()):
            torch.testing.assert_close(p, q, rtol=1e-4, atol=1e-6)
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=3e-4, atol=3e-6)


class Quadratic(torch.nn.Module):
    def __init__(self, a=0.):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(float(a)))

    def phi(self, x, c):
        return .5 * self.a * x.flatten(1).square().sum(1)


def test_zero_potential_refinement_repairs_warmstart_mismatch():
    p = Quadratic()
    x = torch.randn(32, 1, 1, 64)
    m = x + 12
    c = torch.zeros(32, dtype=torch.long)
    fixed, info = refine_proposal(p, x, m, c, steps=4)
    assert torch.equal(fixed, x)
    est = tilted_phi0(p, x, c, fixed, None, .1, K=16, alpha_def=.5)
    assert torch.allclose(est.ess, torch.ones_like(est.ess), atol=1e-6)
    assert info["pt/refine_started_from_generator"] == 0
    assert info["pt/refine_distance_from_generator"] == 12


def test_refinement_decreases_energy_and_finds_quadratic_prox():
    p = Quadratic(a=1.)
    x = torch.randn(4, 2, 2, 3)
    m = x * 4
    c = torch.zeros(4, dtype=torch.long)
    fixed, info = refine_proposal(p, x, m, c, steps=4, lr=.5)
    assert torch.allclose(fixed, x / 2, atol=1e-6)
    assert (energy(p, fixed, x, c) <= energy(p, x, x, c)).all()
    assert info["pt/refine_residual_rms"] < 1e-6


def test_broken_state_does_not_stop_updates_but_watchdog_fails():
    cfg = dict(policy="active_recovery_v1", eps_warmup=2, eps_anneal_steps=10,
               prox_warmup=2, prox_ramp=4, lambda_prox_max=.1,
               recovery_check_after=4, recovery_bad_patience=3, ess_ema_decay=0.)
    s = build_schedule(cfg)
    for _ in range(6):
        assert s.update_potential()
        s.observe(0.)
    assert s.health.is_broken
    assert s.eps() < s.eps_max
    assert s.lambda_prox() == .1
    assert s.measured_healthy_steps == 0  # scheduled motion never fakes ESS
    assert s.recovery_failure()
    clone = build_schedule(cfg)
    clone.load_state_dict(s.state_dict())
    assert clone.state_dict() == s.state_dict()
    assert clone.eps() == s.eps()
    with pytest.raises(ValueError, match="policy mismatch"):
        s.load_state_dict({"step": 1000})


def test_full_prox_uses_hessian_and_does_not_train_potential():
    p = Quadratic(a=2.)
    m = torch.randn(3, 1, 1, 4, requires_grad=True)
    x = torch.randn_like(m)
    c = torch.zeros(3, dtype=torch.long)
    loss, _ = prox_loss(p, m, x, c, 0., mode="full", norm="none")
    loss.backward()
    expected = 2 * (3 * m.detach() - x) * 3 / m.numel()
    assert torch.allclose(m.grad, expected, atol=1e-6)
    assert p.a.grad is None


def test_alignment_changes_potential_without_training_generator():
    p = Quadratic(a=0.)
    x = torch.randn(3, 1, 1, 4)
    m = (x / 2).detach().requires_grad_(True)
    c = torch.zeros(3, dtype=torch.long)
    loss = alignment_loss(p, x, m, c)
    loss.backward()
    assert p.a.grad < 0  # optimum a=1 gives prox=x/2
    assert m.grad is None


def test_ema_initializer_resets_nothing_and_imports_no_training_state(tmp_path):
    from tests.test_train_smoke import make_state
    from ptflow.initialize import initialize_ema_only
    source = make_state(False)
    target = make_state(True, with_scale=True)
    with torch.no_grad():
        for p in source.ema_model.parameters():
            p.add_(.03)
    path = tmp_path / "state_00200000.pt"
    torch.save(dict(step=200000, ema_model=source.ema_model.state_dict(),
                    optimizer={"unwanted": True}, pt_schedule={"step": 200000}), path)
    original_pt = {k: v.clone() for k, v in target.pt.potential.state_dict().items()}
    initialize_ema_only(target, path)
    assert target.step == 0
    assert not target.optimizer.state
    assert not target.pt.optimizer.state
    assert target.pt.sched.step == 0
    for k, v in target.model.state_dict().items():
        assert torch.equal(v, source.ema_model.state_dict()[k])
    for k, v in target.pt.potential.state_dict().items():
        assert torch.equal(v, original_pt[k])


def test_active_training_updates_potential_scale_and_full_prox_from_broken_state():
    from tests.test_train_smoke import make_state, run_step
    state = make_state(True, with_scale=True, lambda_prox_max=.1)
    state.pt.sched = build_schedule(dict(policy="active_recovery_v1", eps_warmup=1,
        eps_anneal_steps=10, prox_warmup=0, prox_ramp=1, lambda_prox_max=.1,
        proposal_refine_steps=2, alignment_batch=2))
    state.pt.sched.health.value = 0.
    state.pt.sched.health.state = "broken"
    state.pt.cfg.update(prox_mode="full", prox_norm="none", prox_max_batch=2)
    old = {k: v.detach().clone() for k, v in state.pt.potential.state_dict().items()}
    state, first = run_step(state)
    state, second = run_step(state)
    assert second["pt/lambda_prox"] > 0
    assert second["pt/loss_prox"] > 0
    assert second["pt/potential_updated"] == 1
    assert second["pt/scale_updated"] == 1
    assert "pt/loss_scale" in second
    assert state.pt.sched.eps() < state.pt.sched.eps_max
    assert any(not torch.equal(old[k], v) for k, v in state.pt.potential.state_dict().items())
    steps = [v["step"].item() for v in state.pt.optimizer.state.values() if "step" in v]
    assert max(steps) == 2
    assert min(steps) == 2
