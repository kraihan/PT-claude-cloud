"""Dual local/W&B logging and report uploads; W&B is mocked, never contacted."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

from scripts import run_six_ablation as ab
from utils.logging import WandbLogger


def test_online_training_preserves_local_metrics_for_csv(tmp_path, monkeypatch):
    calls = []
    initializations = []
    fake = SimpleNamespace(init=lambda **kw: initializations.append(kw),
                           log=lambda values, step: calls.append((dict(values), step)),
                           finish=lambda: None)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    logger = WandbLogger()
    workdir = tmp_path / "arm"
    logger.set_logging(use_wandb=True, workdir=str(workdir), project="test", group="study", log_every_k=3)
    logger.set_step(1)
    logger.log_dict({"pt/ess": torch.tensor(0.5), "loss": 2.0})
    logger.set_step(3)
    logger.log_dict({"pt/ess": torch.tensor(1.0), "loss": 4.0})
    logger.finish()
    record = json.loads((workdir / "log/metrics.jsonl").read_text())
    assert record == dict(step=3, **{"pt/ess": 0.75, "loss": 3.0})
    assert calls == [({"pt/ess": 0.75, "loss": 3.0}, 3)]
    assert initializations[0]["dir"] == str(workdir)
    assert initializations[0]["group"] == "study"
    # The same workdir resumes the same W&B training run.
    second = WandbLogger()
    second.set_logging(use_wandb=True, workdir=str(workdir), project="test", group="study")
    assert initializations[0]["id"] == initializations[1]["id"]


def report_plan(tmp_path):
    return dict(root=str(tmp_path), steps=4000, seed=42, evaluation={"num_samples": 10000},
                reference={"cfg_scale": 1.2},
                wandb=dict(enabled=True, project="test", entity=None, group="study", mode="online"))


def test_report_uploads_table_csv_and_measured_fid_without_duplicate_counts(tmp_path, monkeypatch):
    (tmp_path / "report").mkdir()
    for name in ("ablation.csv", "REPORT.md"):
        (tmp_path / "report" / name).write_text("test fixture")
    files, logs, artifacts, starts = [], [], [], []
    run = SimpleNamespace(summary={}, url="https://wandb.ai/test/test/runs/mock",
                          log=logs.append, log_artifact=artifacts.append, finish=lambda: None)

    def init(**kwargs):
        starts.append(kwargs)
        return run

    fake = SimpleNamespace(init=init, Table=lambda **kwargs: kwargs,
                           Artifact=lambda *args, **kwargs: SimpleNamespace(add_file=lambda path, name: files.append(name)))
    monkeypatch.setitem(sys.modules, "wandb", fake)
    plan = report_plan(tmp_path)
    base = dict(factor="proposal", value="defensive_diagonal", arm="reference", status="complete",
                fid=99.0, result_file="reference.json", num_samples=10000)
    ab.publish_report(plan, [base, dict(base, factor="eps", value=0.1),
                             dict(base, factor="K", value=4, status="pending_or_failed", result_file="K_4.json")])
    assert run.summary["completed_evaluations"] == 1
    assert run.summary["expected_evaluations"] == 2
    assert run.summary["fid/eps/0.1"] == 99.0
    assert "fid/K/4" not in run.summary
    assert set(files) == {"ablation.csv", "REPORT.md"}
    assert len(logs) == 1 and len(artifacts) == 1
    assert starts[0]["group"] == "study"
    assert starts[0]["resume"] == "allow"
    assert json.loads((tmp_path / "report/wandb.json").read_text())["status"] == "logged"


def test_failed_remote_upload_preserves_csv_and_records_failure(tmp_path, monkeypatch):
    (tmp_path / "report").mkdir()
    csv = tmp_path / "report/ablation.csv"
    csv.write_text("factor,fid\neps,99\n")

    def unavailable(**kwargs):
        raise ConnectionError("Mock service unavailable")

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=unavailable))
    ab.publish_report(report_plan(tmp_path), [])
    assert csv.read_text() == "factor,fid\neps,99\n"
    assert json.loads((tmp_path / "report/wandb.json").read_text())["status"] == "upload_failed"
