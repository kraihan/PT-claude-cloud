"""Test background setup dependencies without contacting a Slurm cluster."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import submit_imagenet_slurm as launch

MANIFEST = Path(__file__).resolve().parents[1] / "configs/ablation/six_imagenet_a100_4k.yaml"


def scheduler(monkeypatch, active="", accounting=""):
    calls = []
    monkeypatch.setattr(launch.parallel, "active_jobs", lambda p: set())

    def run(command, **kwargs):
        if command[0] == "squeue":
            return SimpleNamespace(stdout=active)
        if command[0] == "sacct":
            return SimpleNamespace(stdout=accounting)
        assert command[0] == "sbatch"
        calls.append(command)
        assert not any(key.startswith("SBATCH_") for key in kwargs["env"])
        return SimpleNamespace(stdout=f"Submit checks complete!\n{9000 + len(calls)};palmetto\n")

    monkeypatch.setattr(launch.subprocess, "run", run)
    return calls


def test_background_workflow_uses_afterok_without_wait(tmp_path, monkeypatch):
    calls = scheduler(monkeypatch)
    launch.submit(MANIFEST, tmp_path)
    assert len(calls) == 3
    assert all("--wait" not in command for command in calls)
    assert not any("--dependency" in arg for arg in calls[0])
    for i in (1, 2):
        assert f"--dependency=afterok:{9000 + i}" in calls[i]
        assert "--kill-on-invalid-dep=yes" in calls[i]
    assert "--gpus=a100:1" in calls[1]
    assert "--constraint=gpu_a100_40gb" in calls[1]
    assert all(not any(arg.startswith("--gpus") for arg in calls[i]) for i in (0, 2))
    for command in calls:
        assert "exec srun --ntasks=1" in Path(command[-1]).read_text()
    assert "run_parallel_ablation.py" in Path(calls[2][-1]).read_text()


def test_adopt_running_preparation_submits_only_remaining_stages(tmp_path, monkeypatch):
    calls = scheduler(monkeypatch, active="16115552|ptin-prepare\n15983725|pt-Lw\n")
    launch.submit(MANIFEST, tmp_path, prepare_job="16115552")
    assert len(calls) == 2
    assert "--dependency=afterok:16115552" in calls[0]
    assert "--dependency=afterok:9001" in calls[1]


def test_adopt_completed_preparation_avoids_expired_dependency(tmp_path, monkeypatch):
    (tmp_path / "manifest.yaml").write_text("fixture")
    user = launch.getpass.getuser()
    calls = scheduler(monkeypatch, accounting=f"16115552|ptin-prepare|COMPLETED|0:0|{user}|\n")
    launch.submit(MANIFEST, tmp_path, prepare_job="16115552")
    assert len(calls) == 2
    assert not any(arg.startswith("--dependency") for arg in calls[0])


@pytest.mark.parametrize("active,existing", [("16115552|pt-Lw\n", "16115552"),
                                             ("16115552|ptin-prepare\n", None),
                                             ("16115553|ptin-smoke\n", None)])
def test_no_adoption_of_unrelated_job_or_duplicate_workflow(tmp_path, monkeypatch, active, existing):
    calls = scheduler(monkeypatch, active=active)
    with pytest.raises((ValueError, RuntimeError)):
        launch.submit(MANIFEST, tmp_path, prepare_job=existing)
    assert not calls


def test_failed_preparation_cannot_be_adopted(tmp_path, monkeypatch):
    user = launch.getpass.getuser()
    calls = scheduler(monkeypatch, accounting=f"16115552|ptin-prepare|FAILED|1:0|{user}|\n")
    with pytest.raises(ValueError, match="not active or successfully completed"):
        launch.submit(MANIFEST, tmp_path, prepare_job="16115552")
    assert not calls


def test_second_submission_does_not_duplicate_registered_jobs(tmp_path, monkeypatch):
    (tmp_path / "slurm_workflow.json").write_text(json.dumps([dict(job_id="9001")]))
    calls = scheduler(monkeypatch, active="9001|ptin-prepare\n")
    with pytest.raises(RuntimeError, match="already active"):
        launch.submit(MANIFEST, tmp_path)
    assert not calls
