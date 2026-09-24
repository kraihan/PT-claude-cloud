"""Regressions for failures missed by the original smoke tests."""
import copy
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch
from models.generator import DitGen
from ptflow.estimator import tilted_phi0
from ptflow.losses import curvature_hinge, prox_loss
from ptflow.schedule import build_schedule, BROKEN
from utils.precision import amp_dtype


class Quadratic(torch.nn.Module):
    def __init__(self, tau=0.5):
        super().__init__()
        self.tau = torch.nn.Parameter(torch.tensor(tau))
    def phi(self, x, c):
        return 0.5 * self.tau * x.flatten(1).square().sum(1)
    def null_labels(self, c):
        return c


def tiny_generator(**kwargs):
    return DitGen(cond_dim=32, hidden_size=32, num_heads=4, depth=3,
                  input_size=4, patch_size=2, in_channels=2, out_channels=2,
                  num_classes=2, noise_classes=0, **kwargs)


def test_checkpointed_gradients_match_eager():
    torch.manual_seed(41)
    eager = tiny_generator()
    # Zero initialization hides the bug: exercise nonzero trained blocks.
    with torch.no_grad():
        for p in eager.parameters():
            p.add_(torch.randn_like(p) * 0.03)
    remat = copy.deepcopy(eager)
    remat.model.use_remat = True
    x, c = torch.randn(3, 4, 4, 2), torch.tensor([0, 1, 0])
    for model in (eager, remat):
        model(c, x0=x)["samples"].square().mean().backward()
    for a, b in zip(eager.parameters(), remat.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=2e-5)


def test_compilation_preserves_state_dict_names():
    model = tiny_generator()
    before = set(model.state_dict())
    model.model.compile(backend="eager")
    model(torch.tensor([0]))["samples"].sum().backward()
    assert before == set(model.state_dict())
    tiny_generator().load_state_dict(model.state_dict(), strict=True)


def test_clipping_preserves_gradient_and_reports_raw_ess():
    pot = Quadratic(-1.0)
    x, c = torch.zeros(1, 1), torch.zeros(1, dtype=torch.long)
    y = torch.tensor([[[0.0], [1.0], [2.0], [20.0]]])
    est = tilted_phi0(pot, x, c, x, None, 0.1, K=4, alpha_def=1, y=y, logw_clip=2)
    est.phi0.sum().backward()
    assert pot.tau.grad.abs() > 1 and est.clip_frac > 0
    torch.testing.assert_close(est.ess, torch.tensor([0.25], dtype=est.ess.dtype))


def test_unclipped_estimator_gradient_matches_logsumexp():
    pot = Quadratic(0.7)
    x, c = torch.zeros(1, 1), torch.zeros(1, dtype=torch.long)
    y = torch.tensor([[[0.0], [1.0], [2.0], [3.0]]])
    est = tilted_phi0(pot, x, c, x, None, 0.1, K=4, alpha_def=1, y=y)
    actual = torch.autograd.grad(est.phi0.sum(), pot.tau)[0]
    expected = (torch.softmax(-0.5 * 0.7 * y.flatten().square() / 0.2, 0)
                * 0.5 * y.flatten().square()).sum()
    torch.testing.assert_close(actual, expected)
    assert est.clip_frac == 0


def test_single_proposal_diagnostics():
    est = tilted_phi0(Quadratic(), torch.zeros(2, 1), torch.zeros(2, dtype=torch.long),
                      torch.zeros(2, 1), None, 0.1, K=1)
    assert torch.isfinite(est.logw_spread).all()


def test_full_prox_freezes_theta():
    pot, m = Quadratic(), torch.ones(3, 2, requires_grad=True)
    loss, _ = prox_loss(pot, m, torch.zeros_like(m), torch.zeros(3, dtype=torch.long), mode="full")
    loss.backward()
    assert pot.tau.grad is None
    torch.testing.assert_close(m.grad, torch.full_like(m, 0.75))


def test_curvature_penalty_has_gradient():
    pot = Quadratic(-0.9)
    loss, _ = curvature_hinge(pot, torch.randn(4, 3), torch.zeros(4, dtype=torch.long), lambda_allow=0.5)
    loss.backward()
    assert pot.tau.grad < -0.9


def test_recovery_probe_does_not_update_potential():
    from tests.test_train_smoke import make_state, run_step
    state = make_state(True, residual=True)
    state.pt.sched.health.value = 0.0
    state.pt.sched.health.state = BROKEN
    state.pt.sched.health.ema_decay = 0.0
    before = copy.deepcopy(state.pt.potential.state_dict())
    state, metrics = run_step(state)
    assert state.pt.sched.health.is_healthy
    assert "pt/control_ess" in metrics and "pt/g_norm_potential" not in metrics
    for k, v in state.pt.potential.state_dict().items():
        torch.testing.assert_close(v, before[k])


def test_schedule_does_not_reobserve_stale_health():
    s = build_schedule({})
    before = s.eps_progress
    s.observe(None)
    assert s.step == 1 and s.eps_progress == before


def test_precision_selection():
    with patch("torch.cuda.get_device_capability", return_value=(7, 5)):
        assert amp_dtype("cuda", "auto") == torch.float16
        with pytest.raises(ValueError, match="native"):
            amp_dtype("cuda", "bf16")
    with patch("torch.cuda.get_device_capability", return_value=(8, 0)):
        assert amp_dtype("cuda", "auto") == torch.bfloat16
    assert amp_dtype("cpu", "auto") is None


def test_fid_requested_count_balanced_classes(monkeypatch):
    from utils import fid_util
    labels_seen, seeds = [], []
    def generate(batch, rng):
        labels_seen.extend(batch[1].tolist())
        seeds.append(rng)
        return torch.zeros(len(batch[1]), 3, 4, 4)
    monkeypatch.setattr(fid_util, "_extract_inception_features", lambda images, **_: (np.zeros((len(images), 2)), None))
    monkeypatch.setattr(fid_util, "_load_ref_stats", lambda _: {"mu": np.zeros(2), "sigma": np.eye(2)})
    logger = SimpleNamespace(log_dict=lambda _: None, log_image=lambda *_: None)
    result = fid_util.evaluate_fid("cifar10", generate, {}, SimpleNamespace(batch_size=7), logger,
                                  num_samples=50, eval_isc=False)
    assert result["num_samples"] == 50
    assert np.bincount(labels_seen).tolist() == [5] * 10
    assert len(seeds) == len(set(seeds))


def test_fid_rejects_nan():
    from utils.fid_util import _to_uint8
    with pytest.raises(ValueError, match="Nonfinite"):
        _to_uint8(np.array([np.nan]))


def test_residual_semantics_and_legacy_compile_keys():
    from utils.ckpt_util import check_model_behavior, canonical_state_dict
    model = tiny_generator(residual=True)
    with pytest.raises(ValueError, match="residual"):
        check_model_behavior(model, {"model_behavior": {"residual": False}})
    sd = {k.replace("model.", "model._orig_mod.", 1): v for k, v in model.state_dict().items()}
    model.load_state_dict(canonical_state_dict(sd), strict=True)


def test_real_training_loop_save_and_resume(tmp_path):
    from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
    from train import train_gen
    ds = TensorDataset(torch.randn(16, 4, 4, 2), torch.arange(16) % 2)
    loader = DataLoader(ds, batch_size=8, sampler=DistributedSampler(ds, num_replicas=1, rank=0))
    rows = []
    logger = SimpleNamespace(set_step=lambda _: None, log_dict=rows.append, finish=lambda: None)
    pt = dict(enabled=True, model=dict(cond_dim=32, hidden_size=32, depth=1, num_heads=4, patch_size=2),
              scale_mode="none", K=4, data_bsz=4, noise_bsz=2,
              schedule=dict(prox_warmup=0, prox_ramp=2, lambda_prox_max=0.1))
    common = dict(optimizer=lambda p: torch.optim.AdamW(p, lr=1e-4), logger=logger,
                  eval_loader=loader, train_loader=loader, learning_rate_fn=lambda _: 1e-4,
                  preprocess_fn=lambda b: dict(images=b[0], labels=b[1]), postprocess_fn=lambda x: x,
                  train_batch_size=2, save_per_step=2, eval_per_step=0, pos_per_sample=3,
                  neg_per_sample=2, positive_bank_size=8, negative_bank_size=8, push_at_resume=1,
                  forward_dict=dict(gen_per_label=3, cfg_min=1., cfg_max=2.), activation_kwargs={},
                  loss_kwargs=dict(R_list=[0.05]), activation_fn=lambda p, x, **kw: {"x": x.reshape(len(x), 1, -1)},
                  feature_params={}, workdir=str(tmp_path), pt_config=pt, ot_mode="debiased",
                  ot_kwargs=dict(use_new_cfg=True, resample_neg=True, disable_diag_mask=True, use_quadratic_cost=True))
    train_gen(model=tiny_generator(residual=True), total_steps=2, **common)
    train_gen(model=tiny_generator(residual=True), total_steps=3, **common)
    ck = torch.load(tmp_path / "checkpoints/state_00000003.pt", weights_only=False)
    assert ck["step"] == 3 and ck["pt_schedule"]["step"] == 3
    assert len(rows) == 3
    assert (tmp_path / "params_ema/metadata.json").exists()


def test_refinement_backtracks_for_large_positive_curvature():
    from ptflow.sampling import sample_mode_b
    gen, pot = tiny_generator(residual=True), Quadratic(10.0)
    c, x = torch.tensor([0, 1]), torch.ones(2, 4, 4, 2)
    y = sample_mode_b(gen, pot, c, x0=x, n_steps=3, gamma=0.5)
    def energy(point):
        return pot.phi(point, c) + 0.5 * (point-x).flatten(1).square().sum(1)
    assert torch.all(energy(y) < energy(x))


def test_bounded_prox_target_vanishes_at_solution():
    pot = Quadratic(0.5)
    x = torch.ones(2, 3)
    loss, _ = prox_loss(pot, x / 1.5, x, torch.zeros(2, dtype=torch.long), norm="bounded_rms")
    assert loss < 1e-12


def test_uneven_accumulation_weights_samples(monkeypatch):
    import train
    class ScalarGenerator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.tensor(1.0))
        def forward(self, c, **kwargs):
            samples = (self.p * c.float()).reshape(-1, 1, 1, 1)
            return {"samples": samples, "noise": {"x": torch.zeros_like(samples)}}
    model = ScalarGenerator()
    state = train.TrainState(0, model, torch.optim.SGD(model.parameters(), lr=0.1), copy.deepcopy(model), 0.9)
    monkeypatch.setattr(train, "ot_drift_loss", lambda gen, **kw: (gen.mean(dim=(1, 2)), {}))
    state, metrics = train.train_step(state, torch.arange(1, 6), torch.zeros(5, 1, 1, 1, 1),
        torch.zeros(5, 1, 1, 1, 1), {}, lambda p, x, **kw: {"x": x.reshape(len(x), 1, 1)},
        gen_per_label=2, grad_accum_steps=3, max_grad_norm=100, ot_mode="debiased")
    torch.testing.assert_close(model.p, torch.tensor(0.7))
    torch.testing.assert_close(metrics["loss"], torch.tensor(3.0))


def test_foreach_ema_matches_reference():
    from train import _update_ema
    model, ema = tiny_generator(), tiny_generator()
    expected = copy.deepcopy(ema)
    with torch.no_grad():
        for a, b in zip(expected.parameters(), model.parameters()):
            a.mul_(0.99).add_(b, alpha=0.01)
    _update_ema(ema, model, 0.99)
    for a, b in zip(ema.parameters(), expected.parameters()):
        torch.testing.assert_close(a, b)


def test_vae_decoder_preserves_latent_gradients_with_frozen_weights(monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path
    class FakeVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))
        def decode(self, z):
            return SimpleNamespace(sample=z * self.weight)
    vae = FakeVAE()
    monkeypatch.setitem(sys.modules, "diffusers.models", SimpleNamespace(
        AutoencoderKL=SimpleNamespace(from_pretrained=lambda _: vae)))
    spec = importlib.util.spec_from_file_location("vae_gradient_test", Path(__file__).parents[1] / "dataset/vae.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, decode = module.vae_enc_decode()
    x = torch.ones(2, 4, 4, 4, requires_grad=True)
    decode(x).sum().backward()
    assert x.grad.abs().sum() > 0 and vae.weight.grad is None
    assert not vae.weight.requires_grad
    with torch.no_grad():
        assert not decode(x).requires_grad
