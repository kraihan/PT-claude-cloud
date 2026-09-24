"""Small CPU-only regression check for the ImageNet BF16/FP32 boundary.

Run from the repository: python -m scripts.check_pt_precision
Does not load datasets/checkpoints or require a GPU/pytest.
"""
import os
os.environ.setdefault("DRIFT_COMPILE", "0")

import torch
from ptflow.potential import PotentialNet, ScaleNet, guided_phi_grad
from ptflow.losses import prox_loss
from ptflow.sampling import sample_mode_b


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    kwargs = dict(cond_dim=16, hidden_size=32, depth=1, num_heads=4,
                  input_size=4, patch_size=2, in_channels=2, num_classes=2,
                  use_bf16=False)
    potential, scale = PotentialNet(**kwargs), ScaleNet(**kwargs)
    with torch.no_grad():
        for module in (potential, scale):
            for p in module.parameters():
                p.add_(torch.randn_like(p) * 0.02)
    labels = torch.tensor([0, 1])
    for dtype in (torch.bfloat16, torch.float16):
        parameter = torch.nn.Parameter(torch.randn(2, 4, 4, 2))
        m = parameter.to(dtype)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            phi, logscale = potential.phi(m, labels), scale(m, labels)
        assert phi.dtype == logscale.dtype == torch.float32
        assert torch.isfinite(phi).all() and torch.isfinite(logscale).all()
        g = guided_phi_grad(potential, m, labels, 0.2)
        assert g.dtype == torch.float32 and torch.isfinite(g).all()
        for mode, norm in (("detach", "bounded_rms"), ("full", "none")):
            parameter.grad = None
            loss, _ = prox_loss(potential, parameter.to(dtype), torch.randn_like(parameter),
                                labels, 0.2, mode=mode, norm=norm)
            loss.backward()
            assert torch.isfinite(loss) and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
        print(f"PASS: {dtype} generator output -> FP32 potential, guided gradient, scale and prox backward")
    class BFloatGenerator(torch.nn.Module):
        def forward(self, c, x0=None, **kwargs):
            return {"samples": (0.5 * x0).bfloat16(), "noise": {"x": x0}}
    refined = sample_mode_b(BFloatGenerator(), potential, labels, cfg_scale=1.2,
                            x0=torch.randn(2, 4, 4, 2), n_steps=4)
    assert refined.dtype == torch.float32 and torch.isfinite(refined).all()
    print("PASS: four-step Mode B refinement from BF16 generator output")
    print("Precision regression passed on CPU. Re-run the A100 smoke test to verify the full pipeline.")


if __name__ == "__main__":
    main()
