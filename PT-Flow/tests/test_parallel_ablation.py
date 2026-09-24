"""CPU-only DAG/report tests; scheduler, GPU and W&B actions are mocked."""
import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_parallel_ablation as parallel

six = parallel.six
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def plan(tmp_path):
    result = six.build_plan(REPO / "configs/ablation/six_h200_4k_wandb.yaml")
    result["root"] = str(tmp_path / "study with spaces")
    result["wandb"]["enabled"] = False
    six.materialize(result)
    result["slurm"]["gpu"] = "any"
    return result


def fake_scheduler(monkeypatch):
    calls = []

    def run(command, **kwargs):
        if command[0] == "squeue":
            return SimpleNamespace(stdout="")
        calls.append(command)
        return SimpleNamespace(stdout=f"{1000 + len(calls)}\n")

    monkeypatch.setattr(parallel.subprocess, "run", run)
    monkeypatch.setattr(six, "preflight", lambda plan: None)
    return calls


def test_parallel_dag_has_only_reference_dependencies(plan, monkeypatch):
    calls = fake_scheduler(monkeypatch)
    parallel.submit(plan, wait_seconds=0)
    assert len(calls) == 18  # 17 GPU tasks + one CPU collector.
    assert sum(entry["train"] for entry in parallel.tasks(plan)) == 11
    for index, command in enumerate(calls[:17]):
        assert "--gpus=1" in command
        dependencies = [arg for arg in command if arg.startswith("--dependency")]
        assert dependencies == ([] if index < 11 else ["--dependency=afterany:1001"])
        assert Path(command[-1]).is_file()
    assert not any(arg.startswith("--gpus") for arg in calls[-1])
    assert "--dependency=afterany:" + ":".join(str(i) for i in range(1001, 1018)) in calls[-1]


def test_existing_reference_checkpoint_releases_inference_jobs_immediately(plan, monkeypatch):
    checkpoint = six.paths(plan, "reference")[2]
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("fixture")
    calls = fake_scheduler(monkeypatch)
    parallel.submit(plan, wait_seconds=0)
    assert len(calls) == 18
    assert all(not any(a.startswith("--dependency") for a in command) for command in calls[:17])


def test_active_existing_chain_is_not_double_submitted(plan, monkeypatch):
    six.dump(Path(plan["root"]) / "submissions.json", [dict(job_id="16114668")])
    calls = []

    def active(command, **kwargs):
        calls.append(command)
        assert command[0] == "squeue"
        return SimpleNamespace(stdout="16114668\n15983725\n")

    monkeypatch.setattr(parallel.subprocess, "run", active)
    monkeypatch.setattr(six, "preflight", lambda p: None)
    with pytest.raises(RuntimeError, match="16114668"):
        parallel.submit(plan, wait_seconds=0)
    assert len(calls) == 1


def test_worker_owns_only_its_result_and_csv(plan, monkeypatch):
    root = Path(plan["root"])
    monkeypatch.setattr(parallel, "hardware", lambda: dict(gpu_name="Mock A100", gpu_memory_gib=80))
    monkeypatch.setattr(six, "collect", lambda p: pytest.fail("Parallel workers must not write aggregate reports"))
    commands = []

    def execute(command, p, tag):
        commands.append(tag)
        if tag.startswith("train_"):
            ckpt = six.paths(p, tag.removeprefix("train_"))[2]
            ckpt.parent.mkdir(parents=True)
            ckpt.write_text("fixture")
        else:
            six.dump(command[command.index("--json-out") + 1], dict(fid=99, step=4000))

    monkeypatch.setattr(six, "execute", execute)
    assert parallel.worker(plan, 0) == 0
    assert commands == ["train_reference", "eval_reference"]
    assert not (root / "report/ablation.csv").exists()
    with (root / "report/rows/reference.csv").open() as handle:
        row = next(csv.DictReader(handle))
    assert row["worker_gpu"] == "Mock A100"
    assert float(row["fid"]) == 99
    assert row["eps"] == "0.1" and row["K"] == "8"
    assert json.loads(row["config_json"])["pt"]["K"] == 8


def test_failed_reference_is_not_retrained_by_six_evaluation_workers(plan, monkeypatch):
    monkeypatch.setattr(parallel, "hardware", lambda: dict(gpu_name="Mock GPU", gpu_memory_gib=40))
    monkeypatch.setattr(six, "execute", lambda *a: pytest.fail("Missing reference must not trigger a fresh training run"))
    assert parallel.worker(plan, 11) == 1
    path = Path(plan["root"]) / "report/rows/n_1.csv"
    with path.open() as handle:
        row = next(csv.DictReader(handle))
    assert row["status"] == "failed"
    assert row["fid"] == "" if "fid" in row else True


def test_final_csv_contains_every_fid_and_full_configuration(plan):
    for entry in parallel.tasks(plan):
        six.dump(Path(plan["root"]) / "results" / f"{entry['result']}.json", dict(fid=99, step=4000))
    parallel.report(plan)
    report = Path(plan["root"]) / "report"
    with (report / "results.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 17
    assert len({r["result_id"] for r in rows}) == 17
    assert all(r["status"] == "complete" for r in rows)
    required = {"proposal", "eps", "K", "lambda_prox_max", "n", "cfg_scale", "gamma", "ema_decay",
                "train_seed", "eval_seed", "steps_requested", "num_samples_requested", "backend",
                "config_path", "config_sha256", "config_json", "checkpoint", "fid"}
    assert required <= set(rows[0])
    assert all(json.loads(r["config_json"])["train"]["total_steps"] == 4000 for r in rows)
    with (report / "ablation.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 22


def test_all_completed_work_needs_only_cpu_report(plan, monkeypatch):
    for entry in parallel.tasks(plan):
        checkpoint = six.paths(plan, entry["arm"])[2]
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text("fixture")
        six.dump(Path(plan["root"]) / "results" / f"{entry['result']}.json", dict(fid=99))
    calls = fake_scheduler(monkeypatch)
    parallel.submit(plan, wait_seconds=0)
    assert len(calls) == 1
    assert not any(arg.startswith("--gpus") or arg.startswith("--dependency") for arg in calls[0])
