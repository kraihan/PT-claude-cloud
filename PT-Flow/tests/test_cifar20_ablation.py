import copy
import csv
import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("run_cifar20_ablation", ROOT / "scripts/run_cifar20_ablation.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def fake_plan(tmp_path):
    cfg = {
        "pipeline": "cifar10_pixel", "dataset": {"num_classes": 10},
        "logging": {}, "optimizer": {"lr_schedule": {}},
        "train": {},
        "pt": {"optimizer": {"lr_schedule": {}}, "schedule": {}},
    }
    settings = dict(proposal="defensive_diagonal", eps=.1, K=8,
                    lambda_prox_max=.1, n=0, cfg_scale=1.2)
    return {
        "repo": str(tmp_path / "ptflow-cifar20/PT-Flow"),
        "root": str(tmp_path / "old"), "steps": 4000,
        "slurm": {"partition": "work1"}, "evaluation": {}, "environment": {},
        "wandb": {"enabled": True}, "reference": copy.deepcopy(settings),
        "arms": [{"name": "reference", "settings": settings, "config": cfg}],
        "stages": [],
    }


def test_exact_twenty_row_plan(monkeypatch, tmp_path):
    monkeypatch.setattr(module.six, "build_plan", lambda manifest: fake_plan(tmp_path))
    plan = module.build_plan(tmp_path / "manifest.yaml")
    tasks = module.parallel.tasks(plan)
    assert sum(len(stage["entries"]) for stage in plan["stages"]) == 20
    assert len(tasks) == 20
    assert sum(bool(task["train"]) for task in tasks) == 13
    assert sum(not task["train"] for task in tasks) == 7
    assert plan["steps"] == 10_000
    assert plan["evaluation"]["num_samples"] == 50_000
    assert plan["slurm"]["gpu"] == "a100"
    assert [stage["factor"] for stage in plan["stages"]] == [
        "eps_min", "K", "alpha_def", "w", "eps_schedule",
    ]
    assert [entry["value"] for entry in plan["stages"][0]["entries"]] == [.2, .1, .05, .02]
    assert [entry["value"] for entry in plan["stages"][1]["entries"]] == [4, 8, 16, 32]


def test_slim_report_has_only_requested_columns(monkeypatch, tmp_path):
    plan = fake_plan(tmp_path)
    plan["root"] = str(tmp_path / "run")
    rows = []
    for factor, values in (
        ("eps_min", [.2, .1, .05, .02]), ("K", [4, 8, 16, 32]),
        ("alpha_def", ["0.0", "0.01", "0.1", "0.1->0.01"]),
        ("w", [1., 1.2, 1.5, 2.]),
        ("eps_schedule", ["constant", "linear", "cosine", "exp"]),
    ):
        rows.extend(dict(factor=factor, value=value, fid=1.23,
                         status="complete", result_id=f"{factor}_{value}") for value in values)
    monkeypatch.setattr(module.parallel, "report", lambda plan, sync: rows)
    module.slim_report(plan)
    with (Path(plan["root"]) / "report/ablation.csv").open() as handle:
        reader = csv.DictReader(handle)
        saved = list(reader)
        assert reader.fieldnames == ["ablation", "value", "fid"]
    assert len(saved) == 20
