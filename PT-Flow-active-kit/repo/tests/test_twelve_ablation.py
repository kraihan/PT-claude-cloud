import copy
import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("run_twelve_ablation", ROOT / "scripts/run_twelve_ablation.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def fake_plan(tmp_path):
    cfg = {
        "pt": {"schedule": {}, "optimizer": {"lr_schedule": {}}},
        "train": {}, "optimizer": {"lr_schedule": {}}, "logging": {},
    }
    settings = dict(proposal="defensive_diagonal", eps=.1, K=8,
                    lambda_prox_max=.1, n=0, cfg_scale=1.2)
    return {
        "repo": str(tmp_path / "ptflow-ablation-12/PT-Flow"),
        "root": str(tmp_path / "old"), "steps": 4000,
        "slurm": {}, "evaluation": {}, "wandb": {"enabled": True},
        "reference": copy.deepcopy(settings),
        "arms": [{"name": "reference", "settings": settings, "config": cfg}],
        "stages": [],
    }


def test_exact_twelve_task_design(monkeypatch, tmp_path):
    monkeypatch.setattr(module.six, "build_plan", lambda manifest: fake_plan(tmp_path))
    plan = module.build_plan(tmp_path / "manifest.yaml")
    tasks = module.parallel.tasks(plan)
    assert len(tasks) == 12
    assert sum(bool(x["train"]) for x in tasks) == 7
    assert sum(not x["train"] for x in tasks) == 5
    assert [x["value"] for x in plan["stages"][0]["entries"]] == ["constant", "linear", "cosine", "exp"]
    assert [x["value"] for x in plan["stages"][1]["entries"]] == [1.0, 1.2, 1.5, 2.0]
    assert [x["value"] for x in plan["stages"][2]["entries"]] == ["0.0", "0.01", "0.1", "0.1->0.01"]
    assert plan["evaluation"]["num_samples"] == 50000
    assert plan["steps"] == 1000
    assert plan["slurm"]["gpu"] == "a100"
    assert plan["slurm"]["constraint"] == ""
    assert {arm["config"]["pt"]["schedule"]["eps_min"] for arm in plan["arms"]} == {.05}


def test_reference_is_cosine_and_scheduled_alpha(monkeypatch, tmp_path):
    monkeypatch.setattr(module.six, "build_plan", lambda manifest: fake_plan(tmp_path))
    plan = module.build_plan(tmp_path / "manifest.yaml")
    ref = next(a for a in plan["arms"] if a["name"] == "reference")
    schedule = ref["config"]["pt"]["schedule"]
    assert schedule["eps_schedule"] == "cosine"
    assert (schedule["alpha_def_start"], schedule["alpha_def_end"]) == (.1, .01)
    assert (schedule["eps_warmup"], schedule["eps_anneal_steps"]) == (50, 750)
    assert (schedule["prox_warmup"], schedule["prox_ramp"]) == (125, 250)
    assert ref["config"]["train"]["total_steps"] == 1000
    assert all(entry["arm"] == "reference" and not entry["train"]
               for entry in plan["stages"][1]["entries"])


def test_epsilon_schedule_curves_reach_expected_values():
    from ptflow.schedule import build_schedule
    expected_mid = {
        "constant": .05,
        "linear": .125,
        "cosine": .125,
        "exp": .1,
    }
    for mode, middle in expected_mid.items():
        sched = build_schedule(dict(eps_max=.2, eps_min=.05, eps_warmup=0,
                                    eps_anneal_steps=100, eps_schedule=mode))
        sched.eps_progress = 50
        assert abs(sched.eps() - middle) < 1e-9
        sched.eps_progress = 100
        assert abs(sched.eps() - .05) < 1e-9
