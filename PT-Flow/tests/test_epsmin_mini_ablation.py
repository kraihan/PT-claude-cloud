import copy
import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "run_epsmin_mini_ablation", ROOT / "scripts/run_epsmin_mini_ablation.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def fake_plan(tmp_path):
    cfg = {
        "pt": {"K": 8, "schedule": {
            "eps_max": .1, "eps_min": .1,
            "eps_warmup": 5000, "eps_anneal_steps": 60000,
        }},
        "logging": {"use_wandb": True, "group": "old"},
    }
    settings = dict(proposal="defensive_diagonal", eps=.1, K=8,
                    lambda_prox_max=.1, n=0, cfg_scale=1.2)
    return {
        "repo": str(tmp_path / "PT-Flow"), "root": str(tmp_path / "old"),
        "steps": 4000, "reference": copy.deepcopy(settings),
        "slurm": {"gpu": "a100", "constraint": "gpu_a100_40gb"},
        "evaluation": {"num_samples": 10000, "batch_size": 8},
        "wandb": {"enabled": True, "group": "old"},
        "arms": [{"name": "reference", "settings": settings, "config": cfg}],
        "stages": [],
    }


def test_plan_is_four_eps_min_runs_with_fid50k(monkeypatch, tmp_path):
    monkeypatch.setattr(module.six, "build_plan", lambda manifest: fake_plan(tmp_path))
    plan = module.build_plan(tmp_path / "manifest.yaml")
    schedules = [arm["config"]["pt"]["schedule"] for arm in plan["arms"]]
    assert [schedule["eps_min"] for schedule in schedules] == [.20, .10, .05, .02]
    assert {schedule["eps_max"] for schedule in schedules} == {.20}
    assert {schedule["eps_warmup"] for schedule in schedules} == {200}
    assert {schedule["eps_anneal_steps"] for schedule in schedules} == {3000}
    assert [entry["value"] for entry in plan["stages"][0]["entries"]] == [.20, .10, .05, .02]
    assert all(entry["train"] for entry in plan["stages"][0]["entries"])
    assert len(module.parallel.tasks(plan)) == 4
    assert plan["evaluation"]["num_samples"] == 50000
    assert plan["slurm"]["gpu"] == "a100"
    assert plan["slurm"]["constraint"] == ""
    assert module.result_name(.05).endswith("_ours")


def test_submit_materializes_and_uses_parallel_runner(monkeypatch, tmp_path):
    source = fake_plan(tmp_path)
    monkeypatch.setattr(module.six, "build_plan", lambda manifest: copy.deepcopy(source))
    called = []
    monkeypatch.setattr(module.six, "preflight", lambda plan: called.append("preflight"))
    monkeypatch.setattr(module.six, "materialize", lambda plan, resume: called.append(("materialize", resume)))
    monkeypatch.setattr(module.parallel, "submit", lambda plan, wait_seconds: called.append(
        ("submit", wait_seconds, len(module.parallel.tasks(plan)), plan["evaluation"]["num_samples"])
    ))
    module.submit(tmp_path / "manifest.yaml")
    assert called == ["preflight", ("materialize", False), ("submit", 0, 4, 50000)]
