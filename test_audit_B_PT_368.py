import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "PT-Flow-active-kit/repo"
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("audit368", ROOT / "audit_B_PT_368.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
torch.set_num_threads(1)


def test_nested_estimator_matches_known_quadratic():
    from ptflow.estimator import tilted_phi0
    class Quadratic(torch.nn.Module):
        def phi(self, x, c):
            return .3 * x.flatten(1).square().sum(1) / 2
    x = torch.randn(3, 2, 2, 2)
    c = torch.zeros(3, dtype=torch.long)
    m = x / 1.3
    s = torch.full_like(x, -__import__('math').log(1.3))
    eps = .1
    est = tilted_phi0(Quadratic(), x, c, m, s, eps, K=128, alpha_def=0,
                     generator=torch.Generator().manual_seed(42), chunk=8)
    stats = audit.prefix_statistics(est, eps)
    expected = .3 / (2 * 1.3) * x.flatten(1).square().sum(1) + eps * 8 * __import__('math').log(1.3)
    for k, row in stats.items():
        torch.testing.assert_close(row["ess_draws"], torch.full((3,), k, dtype=torch.float64))
        torch.testing.assert_close(row["phi0"], expected.double(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(stats[128]["phi0"], est.phi0.double(), rtol=1e-5, atol=1e-5)


def test_antithetic_prefix_reorders_both_halves():
    from types import SimpleNamespace
    # Stored pairs are [1,2,3,4,-1,-2,-3,-4], not adjacent.
    lw = torch.tensor([[1., 2., 3., 4., -1., -2., -3., -4.]], dtype=torch.float64)
    estimate = SimpleNamespace(log_w=lw, phi_y=torch.zeros_like(lw))
    result = audit.prefix_statistics(estimate, .1, ks=(2, 4, 8))
    selected = lw[:, [0, 4]]
    expected = (2 * torch.logsumexp(selected, 1) - torch.logsumexp(2 * selected, 1)).exp()
    torch.testing.assert_close(result[2]["ess_draws"], expected)
    with pytest.raises(ValueError):
        audit.prefix_statistics(estimate, .1, ks=(3,))


def test_raw_loading_and_readonly_end_to_end(tmp_path, monkeypatch):
    from models.generator import DitGen
    from ptflow.potential import PotentialNet, ScaleNet
    from ptflow.schedule import build_schedule
    home = tmp_path / "job"
    checkpoint = home / "B_PT_Full/checkpoints/state_00000368.pt"
    checkpoint.parent.mkdir(parents=True)
    cfg = dict(model=dict(cond_dim=8, num_classes=3, input_size=4, in_channels=1,
                          out_channels=1, hidden_size=8, depth=1, num_heads=2,
                          patch_size=2, use_bf16=False, use_rope=False),
        dataset=dict(num_classes=3),
        pt=dict(p_uncond=.25,
            model=dict(cond_dim=8, num_classes=3, input_size=4, in_channels=1,
                       hidden_size=8, depth=1, num_heads=2, use_rope=False, quad_anchor=1.),
            scale_model=dict(cond_dim=8, num_classes=3, input_size=4, in_channels=1,
                             hidden_size=8, depth=1, num_heads=2, use_rope=False),
            schedule=dict(policy="active_recovery_v1", eps_max=.1, eps_min=.05,
                          eps_warmup=20, eps_anneal_steps=20000)))
    (home / "full.yaml").write_text(yaml.safe_dump(cfg))
    (home / "settings.sh").write_text("# not used in CPU test\n")
    gen, pot, scale = DitGen(**cfg["model"]), PotentialNet(**cfg["pt"]["model"]), ScaleNet(**cfg["pt"]["scale_model"])
    sched = build_schedule(cfg["pt"]["schedule"])
    for _ in range(368):
        sched.observe(.01)
    # EMA dictionaries are deliberately incompatible: audit must use raw keys.
    torch.save(dict(step=368, model=gen.state_dict(), pt_model=pot.state_dict(),
                    pt_scale_model=scale.state_dict(), pt_schedule=sched.state_dict(),
                    ema_model={"wrong": torch.ones(1)}, pt_ema_model={"wrong": torch.ones(1)},
                    pt_ema_scale_model={"wrong": torch.ones(1)}, optimizer={"not_used": True}), checkpoint)
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    # Use the real source modules while keeping the fixture checkpoint separate.
    monkeypatch.setattr(audit, "validate_source", lambda h: dict(config=home / "full.yaml", repo=REPO, checkpoint=checkpoint))
    monkeypatch.setattr(audit, "KS", (16, 32))
    monkeypatch.setattr(audit, "REFINEMENTS", (0, 8))
    monkeypatch.setattr(audit, "ALPHAS", (.1, .75))
    original_prefix = audit.prefix_statistics
    monkeypatch.setattr(audit, "prefix_statistics", lambda est, eps: original_prefix(est, eps, ks=(16, 32)))
    out = tmp_path / "audit"
    audit.run_audit(home, out, pairs=4, repeats=2, device_name="cpu", micro=2)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before
    summary = json.loads((out / "summary.json").read_text())
    assert summary["completed"] and summary["model_parameters_unchanged"]
    assert summary["cases"] == 16
    assert summary["per_sample_rows"] == 4 * 2 * 16
    assert (out / "diagnostics.csv").is_file()
    assert (out / "per_sample.csv").is_file()
    assert not (out / "checkpoints").exists()
    assert sorted(p.name for p in checkpoint.parent.iterdir()) == [checkpoint.name]


def test_worker_requests_one_gpu_and_no_training(tmp_path):
    file = tmp_path / "audit.sbatch"
    file.write_text(audit.WORKER, newline="\n")
    subprocess.run(["C:/Program Files/Git/bin/bash.exe", "-n", str(file)], check=True)
    assert "--gpus=h200:1" in audit.WORKER
    assert "train.py" not in audit.WORKER
    assert "scancel" not in audit.WORKER
