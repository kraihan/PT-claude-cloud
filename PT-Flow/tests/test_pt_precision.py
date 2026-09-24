"""BF16 generator -> FP32 potential regressions from the ImageNet smoke run."""
import copy

import pytest
import torch

from ptflow.losses import prox_loss
from ptflow.potential import PotentialNet, ScaleNet, guided_phi_grad


def small_head(kind=PotentialNet):
    torch.manual_seed(817)
    model = kind(cond_dim=16, hidden_size=32, depth=1, num_heads=4,
                 input_size=4, in_channels=2, patch_size=2, num_classes=2,
                 use_bf16=False)
    # Zero readout initialization would hide loss/gradient discrepancies.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.02)
    return model


@pytest.mark.parametrize("kind", [PotentialNet, ScaleNet])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("outer_amp", [False, True])
def test_low_precision_input_matches_fp32_head(kind, dtype, outer_amp):
    model = small_head(kind)
    reference = copy.deepcopy(model)
    x = torch.randn(2, 4, 4, 2).to(dtype).requires_grad_(True)
    x_ref = x.detach().float().requires_grad_(True)
    labels = torch.tensor([0, 1])
    expected = reference(x_ref, labels)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=outer_amp):
        actual = model(x, labels)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    actual.square().sum().backward()
    expected.square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    torch.testing.assert_close(x.grad, x_ref.grad.to(dtype), rtol=0, atol=0)
    for actual_p, expected_p in zip(model.parameters(), reference.parameters()):
        if expected_p.grad is not None:
            torch.testing.assert_close(actual_p.grad, expected_p.grad, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("full", [False, True])
def test_guided_gradient_is_fp32_and_preserves_higher_order_path(full):
    model = small_head()
    labels = torch.tensor([0, 1])
    x = torch.randn(2, 4, 4, 2).bfloat16().requires_grad_(True)
    x_ref = x.detach().float().requires_grad_(True)
    actual = guided_phi_grad(model, x, labels, 0.2, create_graph=full)
    expected = guided_phi_grad(model, x_ref, labels, 0.2, create_graph=full)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    if full:
        grad = torch.autograd.grad(actual.square().sum(), x)[0]
        grad_ref = torch.autograd.grad(expected.square().sum(), x_ref)[0]
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
        torch.testing.assert_close(grad, grad_ref.bfloat16(), rtol=0, atol=0)
    else:
        assert not actual.requires_grad


@pytest.mark.parametrize("mode,norm", [("detach", "bounded_rms"), ("full", "none")])
def test_bf16_generator_receives_prox_gradient(mode, norm):
    model = small_head()
    generator_parameter = torch.nn.Parameter(torch.randn(2, 4, 4, 2))
    samples = generator_parameter.to(torch.bfloat16)
    loss, _ = prox_loss(model, samples, torch.randn_like(generator_parameter),
                        torch.tensor([0, 1]), 0.2, mode=mode, norm=norm)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(generator_parameter.grad).all() and generator_parameter.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())
