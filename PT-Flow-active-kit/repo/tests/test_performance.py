import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from ptflow.ot_drift import ot_drift_loss
from utils.feature_cache import FrozenFeatureCache
from utils.feature_moments import FeatureMoments


@pytest.mark.parametrize("quadratic", [False, True])
@pytest.mark.parametrize("cfg", [False, True])
@pytest.mark.parametrize("iterations", [1, 5])
def test_reused_costs_match_reference_gradients(quadratic, cfg, iterations):
    torch.manual_seed(9)
    x = torch.randn(3, 7, 17)
    pos, neg, uncond = torch.randn(3, 11, 17), torch.randn(3, 5, 17), torch.randn(3, 4, 17)
    values, gradients = [], []
    for optimized in (False, True):
        gen = x.clone().requires_grad_()
        loss, metrics = ot_drift_loss(gen, pos, neg, weight_neg=torch.rand(3, 5).fill_(1),
            R_list=[0.05, 0.2], use_new_cfg=cfg, fixed_uncond=uncond,
            weight_uncond=torch.tensor([[0.0], [0.2], [1.5]]),
            sinkhorn_num_iter=iterations, use_quadratic_cost=quadratic,
            disable_diag_mask=False, reuse_costs=optimized)
        loss.mean().backward()
        values.append(loss)
        gradients.append(gen.grad)
    torch.testing.assert_close(values[0], values[1], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(gradients[0], gradients[1], rtol=2e-4, atol=2e-6)


def test_reused_costs_zero_weight_support_is_finite():
    x = torch.randn(2, 4, 6, requires_grad=True)
    loss, _ = ot_drift_loss(x, torch.randn(2, 5, 6), torch.randn(2, 4, 6),
                           weight_neg=torch.tensor([[1., 0, 1, 0], [0, 1., 0, 1.]]),
                           reuse_costs=True, use_quadratic_cost=True)
    loss.sum().backward()
    assert torch.isfinite(loss).all() and torch.isfinite(x.grad).all()


def test_feature_cache_identity_eviction_and_gradient_exclusion():
    calls = []
    def extract(params, x, **kwargs):
        calls.append(len(x))
        return {"f": x.reshape(len(x), 2, 3).square()}
    cache = FrozenFeatureCache(2 * 2 * 3 * 4)
    x = torch.randn(3, 2, 3, requires_grad=True)
    ids = [101, 102, 101]
    x = torch.stack([x[0], x[1], x[0]])
    result = cache.get({}, x, ids, extract)
    torch.testing.assert_close(result["f"], extract({}, x)["f"])
    assert not result["f"].requires_grad and cache.bytes <= cache.max_bytes
    before = len(calls)
    cache.get({}, x, ids, extract)
    assert len(calls) == before
    cache.get({}, torch.ones(1, 2, 3), [103], extract)
    assert len(cache.entries) == 2 and 103 in cache.entries


def test_bank_ids_change_on_overwrite_and_share_across_banks():
    from ptflow.memory_bank import ArrayMemoryBank
    pos, neg = ArrayMemoryBank(1, 1), ArrayMemoryBank(1, 1)
    old = pos.add(torch.zeros(1, 2), [0])
    new = pos.add(torch.ones(1, 2), [0])
    neg.add(torch.ones(1, 2), [0], ids=new)
    a, keys = pos.sample([0], 1, return_ids=True)
    b, other = neg.sample([0], 1, return_ids=True)
    assert new[0] != old[0] and keys[0, 0] == other[0, 0] == new[0]
    torch.testing.assert_close(a, b)


def test_fid_moments_equal_full_numpy_covariance():
    rng = np.random.default_rng(45)
    x = rng.normal(size=(103, 9)) + 10
    moments = FeatureMoments(9)
    for chunk in np.array_split(x, 7):
        moments.update(chunk)
    mean, cov = moments.statistics()
    np.testing.assert_allclose(mean, x.mean(0), atol=1e-12)
    np.testing.assert_allclose(cov, np.cov(x, rowvar=False), atol=1e-12)


def test_streaming_fid_balanced_exact_count(tmp_path, monkeypatch):
    import inference
    from utils import fid_util
    labels_seen = []
    def sampler(labels, rng):
        labels_seen.extend(labels.tolist())
        return torch.zeros(len(labels), 3, 4, 4)
    monkeypatch.setattr(fid_util, "_extract_inception_features",
                        lambda x, **kw: (np.tile(np.arange(3), (len(x), 1)), None))
    ref = tmp_path / "ref.npz"
    np.savez(ref, mu=np.arange(3), sigma=np.zeros((3, 3)))
    result = inference.run_eval_streaming(SimpleNamespace(num_classes=10), lambda x: x,
        "checkpoint.pt", 180000, num_samples=50, cfg_scale=1.2, gen_bsz=7,
        fid_ref=str(ref), seed=42, device=torch.device("cpu"), sampler=sampler)
    assert result["num_samples"] == 50 and abs(result["fid"]) < 1e-10
    assert np.bincount(labels_seen).tolist() == [5] * 10


def test_channels_last_convnext_preserves_input_gradients():
    from models.convnext import ConvNextV2
    torch.manual_seed(17)
    original = ConvNextV2(depths=[1, 1, 1, 1], dims=[8, 16, 32, 64])
    modified = copy.deepcopy(original).to(memory_format=torch.channels_last)
    for module in modified.modules():
        module.channels_last = True
    gradients, outputs = [], []
    data = torch.randn(2, 32, 32, 3)
    for model in (original, modified):
        x = data.clone().requires_grad_()
        features = model.get_activations(x, image_size=32)
        loss = sum(f.square().mean() for f in features.values())
        loss.backward()
        outputs.append(loss)
        gradients.append(x.grad)
    torch.testing.assert_close(outputs[0], outputs[1], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(gradients[0], gradients[1], atol=2e-5, rtol=2e-4)


def test_optimized_training_and_extra_ema():
    from tests.test_train_smoke import make_state, fake_feature_apply, HW, CH, BSZ
    from train import train_step
    state = make_state(False)
    state.extra_emas = {"0.9999": copy.deepcopy(state.ema_model)}
    original = copy.deepcopy(state.model.state_dict())
    _, metrics = train_step(state, torch.arange(BSZ), torch.randn(BSZ, 4, HW, HW, CH),
                           torch.randn(BSZ, 2, HW, HW, CH), {}, fake_feature_apply,
                           gen_per_label=3, feature_chunk_size=4, grad_accum_steps=2,
                           ot_mode="debiased", profile=True,
                           ot_kwargs=dict(resample_neg=True, resample_gen_per_label=2, use_new_cfg=True,
                                          use_quadratic_cost=True, reuse_costs=True, disable_diag_mask=True))
    assert state.step == 1 and metrics["profile/backward_s"] > 0
    assert metrics["profile/negative_generation_s"] > 0
    for k, p in state.extra_emas["0.9999"].state_dict().items():
        if k in dict(state.model.named_parameters()):
            torch.testing.assert_close(p, original[k] * 0.9999 + state.model.state_dict()[k] * 0.0001)


def test_profiles_preserve_architecture_and_show_actual_budgets():
    from scripts.make_performance_config import make_config, budget
    root = Path(__file__).resolve().parents[1]
    base = yaml.safe_load((root / "configs/gen/ptflow_L.yaml").read_text())
    preserved = make_config(base, "preserve")
    assert preserved["model"] == base["model"]
    assert preserved["pt"]["enabled"]
    assert budget(preserved, 8)["global_generated"] == 8192
    balanced = make_config(base, "balanced")
    fast = make_config(base, "speed")
    assert not balanced["pt"]["enabled"]
    assert budget(balanced, 8)["global_generated"] == 2048
    assert budget(fast, 8)["global_generated"] == 1024
    assert budget(fast, 8)["global_negative_generated"] == 0
    assert balanced["model"]["hidden_size"] == base["model"]["hidden_size"]
    assert not balanced["model"]["residual"]


def test_buffered_logging_detaches_and_averages(tmp_path):
    from utils.logging import WandbLogger
    logger = WandbLogger()
    logger.set_logging(use_wandb=False, workdir=str(tmp_path), log_every_k=3)
    for step in (1, 2, 3):
        logger.set_step(step)
        logger.log_dict({"value": torch.tensor(float(step), requires_grad=True)})
        assert all(not v.requires_grad for v in logger._buffer.values() if torch.is_tensor(v))
    rows = (tmp_path / "log/metrics.jsonl").read_text().splitlines()
    assert len(rows) == 1 and json.loads(rows[0])["value"] == 2


def test_explicit_resume_restores_extra_ema(tmp_path):
    from tests.test_train_smoke import make_state, run_step
    from utils.ckpt_util import save_checkpoint, restore_checkpoint
    state = make_state(False)
    state.extra_emas = {"0.9999": copy.deepcopy(state.ema_model)}
    state, _ = run_step(state)
    save_checkpoint(state, workdir=str(tmp_path / "source"))
    restored = make_state(False)
    restored.extra_emas = {"0.9999": copy.deepcopy(restored.ema_model)}
    restore_checkpoint(state=restored, workdir=str(tmp_path / "new"),
                       checkpoint=tmp_path / "source/checkpoints/state_00000001.pt")
    assert restored.step == 1
    for k, p in state.extra_emas["0.9999"].state_dict().items():
        torch.testing.assert_close(restored.extra_emas["0.9999"].state_dict()[k], p)


def test_cached_loop_ema_finetune_and_finished_run_benchmark(tmp_path):
    from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
    from tests.test_regressions import tiny_generator
    from train import train_gen
    ds = TensorDataset(torch.randn(16, 4, 4, 2), torch.arange(16) % 2)
    loader = DataLoader(ds, batch_size=8, sampler=DistributedSampler(ds, num_replicas=1, rank=0))
    rows = []
    common = dict(optimizer=lambda params: torch.optim.AdamW(params, lr=1e-4),
                  logger=SimpleNamespace(set_step=lambda step: None, log_dict=rows.append, finish=lambda: None),
                  eval_loader=loader, train_loader=loader, learning_rate_fn=lambda step: 1e-4,
                  preprocess_fn=lambda b: dict(images=b[0], labels=b[1]), postprocess_fn=lambda x: x,
                  train_batch_size=2, save_per_step=2, eval_per_step=0, pos_per_sample=3, neg_per_sample=2,
                  positive_bank_size=8, negative_bank_size=8, push_at_resume=1,
                  forward_dict=dict(gen_per_label=3, cfg_min=1., cfg_max=2.), activation_kwargs={},
                  loss_kwargs=dict(R_list=[0.05]), activation_fn=lambda p, x, **kw: {"x": x.reshape(len(x), 1, -1)},
                  feature_params={}, feature_cache_gib=0.001, extra_ema_decays=[0.9999],
                  feature_chunk_size=2, ot_mode="debiased",
                  ot_kwargs=dict(use_new_cfg=True, resample_neg=True, resample_gen_per_label=2,
                                 disable_diag_mask=True, use_quadratic_cost=True, reuse_costs=True))
    source = tmp_path / "source"
    train_gen(model=tiny_generator(residual=True), total_steps=2, workdir=str(source), **common)
    ckpt = source / "checkpoints/state_00000002.pt"
    weights = torch.load(ckpt, weights_only=False)["ema_model"]
    fork = tmp_path / "fork"
    train_gen(model=tiny_generator(residual=True), total_steps=1, init_ema_from=str(ckpt), workdir=str(fork), **common)
    new = torch.load(fork / "checkpoints/state_00000001.pt", weights_only=False)
    assert new["step"] == 1
    for key in dict(tiny_generator().named_parameters()):
        expected = weights[key] * 0.9999 + new["model"][key] * 0.0001
        torch.testing.assert_close(new["extra_emas"]["0.9999"][key], expected)
    assert any("feature_cache/gib" in row for row in rows)
    bench = tmp_path / "bench"
    train_gen(model=tiny_generator(residual=True), total_steps=2, benchmark_steps=2,
              resume_from=str(ckpt), workdir=str(bench), **common)
    result = json.loads((bench / "benchmark.json").read_text())
    assert result["initial_step"] == 2 and result["benchmark/measured_steps"] == 1
    assert not list((bench / "checkpoints").glob("*.pt"))
