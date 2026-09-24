import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("run_k_mini_ablation", ROOT / "scripts/run_k_mini_ablation.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def fake_plan(tmp_path):
    cfg = {
        "pt": {"K": 8},
        "logging": {"use_wandb": True, "group": "old"},
    }
    settings = dict(proposal="defensive_diagonal", eps=.1, K=8,
                    lambda_prox_max=.1, n=0, cfg_scale=1.2)
    return {
        "repo": str(tmp_path / "PT-Flow"),
        "root": str(tmp_path / "old"),
        "steps": 4000,
        "reference": copy.deepcopy(settings),
        "slurm": {"gpu": "a100", "constraint": "gpu_a100_40gb"},
        "wandb": {"enabled": True, "group": "old"},
        "arms": [{"name": "reference", "settings": settings, "config": cfg}],
        "stages": [],
    }


def test_build_plan_has_only_four_real_k_runs(monkeypatch, tmp_path):
    monkeypatch.setattr(module.six, "build_plan", lambda manifest: fake_plan(tmp_path))
    plan = module.build_plan(tmp_path / "manifest.yaml")
    assert [arm["config"]["pt"]["K"] for arm in plan["arms"]] == [2, 4, 8, 16]
    assert [entry["value"] for entry in plan["stages"][0]["entries"]] == [2, 4, 8, 16]
    assert all(entry["train"] for entry in plan["stages"][0]["entries"])
    assert len(module.parallel.tasks(plan)) == 4
    assert plan["slurm"]["gpu"] == "a100"
    assert plan["slurm"]["constraint"] == ""


def test_submit_uses_real_materialized_parallel_plan(monkeypatch, tmp_path):
    source = fake_plan(tmp_path)
    monkeypatch.setattr(module.six, "build_plan", lambda manifest: copy.deepcopy(source))
    called = []
    monkeypatch.setattr(module.six, "preflight", lambda plan: called.append("preflight"))
    monkeypatch.setattr(module.six, "materialize", lambda plan, resume: called.append(("materialize", resume)))
    monkeypatch.setattr(module.parallel, "submit", lambda plan, wait_seconds: called.append(("submit", wait_seconds, len(module.parallel.tasks(plan)))))
    module.submit(tmp_path / "manifest.yaml", "h100")
    assert called == ["preflight", ("materialize", False), ("submit", 0, 4)]
