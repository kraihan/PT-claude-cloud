"""Exercise sweep isolation, serial Slurm submission and restart behavior on CPU.

Subprocess/GPU work is explicitly mocked: these tests do not submit jobs or
produce scientific measurements.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest
import torch
import yaml

from scripts import run_six_ablation as ab

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "configs/ablation/six_h200.yaml"


@pytest.fixture
def plan(tmp_path):
    result = ab.build_plan(MANIFEST)
    result["root"] = str(tmp_path / "experiment with spaces")
    return result


def differences(a, b, prefix=""):
    found = set()
    for key in a.keys() | b.keys():
        name = prefix + key
        if isinstance(a.get(key), dict) and isinstance(b.get(key), dict):
            found |= differences(a[key], b[key], name + ".")
        elif a.get(key) != b.get(key):
            found.add(name)
    return found


def test_design_is_ofat_with_reused_reference(plan):
    assert len(plan["arms"]) == 11
    entries = [e for stage in plan["stages"] for e in stage["entries"]]
    assert len({e["result"] for e in entries}) == 17
    assert len(entries) == 22
    assert sum(e["train"] for e in entries) == 11
    reference = plan["arms"][0]["config"]
    allowed = {
        "proposal": {"pt.scale_mode", "pt.schedule.alpha_def_start", "pt.schedule.alpha_def_end", "pt.schedule.alpha_def_degraded"},
        "eps": {"pt.schedule.eps_min", "pt.schedule.eps_max"},
        "K": {"pt.K"}, "lambda_prox_max": {"pt.schedule.lambda_prox_max"},
    }
    for arm in plan["arms"][1:]:
        factor = next(e["factor"] for e in entries if e["arm"] == arm["name"])
        changed = differences(reference, arm["config"])
        assert changed and changed <= allowed[factor]
    for arm in plan["arms"]:
        cfg = arm["config"]
        assert cfg["pt"]["enabled"]
        assert cfg["pt"]["schedule"]["eps_min"] == cfg["pt"]["schedule"]["eps_max"]
        assert cfg["pt"]["scale_K"] == 4
        assert cfg["train"]["total_steps"] == 5000
        assert cfg["train"]["eval_per_step"] == 0
        assert cfg["pt"]["schedule"]["prox_warmup"] == 500
    for entry in entries:
        command = ab.eval_command(plan, entry)
        assert command[command.index("--sampler") + 1] == ("B" if entry["n"] else "A")


def test_resume_rejects_changed_settings_and_keeps_local_checkpoint(plan):
    frozen = ab.materialize(plan)
    assert ab.materialize(plan, resume=True) == frozen
    with pytest.raises(FileExistsError):
        ab.materialize(plan)
    changed = copy.deepcopy(plan)
    changed["steps"] = 4000
    with pytest.raises(ValueError, match="changed"):
        ab.materialize(changed, resume=True)
    plan["init_ema_checkpoint"] = "source.pt"
    assert "--init-ema" in ab.train_command(plan, "reference")
    _, work, _ = ab.paths(plan, "reference")
    (work / "checkpoints").mkdir(parents=True)
    (work / "checkpoints/state_00001000.pt").write_text("fixture")
    assert "--init-ema" not in ab.train_command(plan, "reference")


def test_environment_references_do_not_depend_on_yaml_key_order(tmp_path):
    spec = yaml.safe_load(MANIFEST.read_text())
    source = tmp_path / "reordered.yaml"
    source.write_text(yaml.safe_dump(spec, sort_keys=True))
    assert ab.build_plan(source, REPO) == ab.build_plan(MANIFEST)


def test_resume_submission_waits_for_existing_chain(plan, monkeypatch):
    frozen = ab.materialize(plan)
    ab.dump(Path(plan["root"]) / "submissions.json", [dict(job_id="999", stage=5)])
    calls = []

    def fake_run(command, **kwargs):
        if command[0] == "squeue":
            return SimpleNamespace(stdout="999\n")
        calls.append(command)
        return SimpleNamespace(stdout=f"{200 + len(calls)}\n")

    monkeypatch.setattr(ab.subprocess, "run", fake_run)
    ab.submit(plan, frozen, resume=True)
    assert "--dependency=afterany:999" in calls[0]
    assert len(calls) == 6


def test_six_slurm_jobs_are_serial_and_use_one_gpu(plan, monkeypatch):
    frozen = ab.materialize(plan)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        assert command[0] == "sbatch"
        return SimpleNamespace(stdout=f"Submit checks complete!\n{100 + len(calls)};palmetto\n")

    monkeypatch.setattr(ab.subprocess, "run", fake_run)
    ab.submit(plan, frozen)
    assert len(calls) == 6
    assert not any(a.startswith("--dependency") for a in calls[0])
    for index, command in enumerate(calls):
        assert "--gpus=h200:1" in command
        if index:
            assert f"--dependency=afterany:{100 + index}" in command
        script = Path(command[-1]).read_text()
        assert "exec srun --ntasks=1" in script
        assert f"--stage {index}" in script


def test_workers_train_once_evaluate_once_and_reuse_results(plan, monkeypatch):
    plan["wandb"]["enabled"] = False  # No network access in simulated GPU workers.
    ab.materialize(plan)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    calls = []

    def fake_execute(command, current, tag):
        calls.append(tag)
        if tag.startswith("train_"):
            _, work, checkpoint = ab.paths(current, tag.removeprefix("train_"))
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text("test fixture, not a checkpoint")
            (work / "log").mkdir()
            (work / "log/metrics.jsonl").write_text(json.dumps({
                "step": 4999, "pt/ess": 0.7, "pt/control_ess": 0.6,
                "pt/lambda_prox": 0.1, "pt/sched_lambda_prox": 0.1, "total_time": 0.01,
            }) + "\n")
        else:
            # Fixture values are confined to pytest's temporary directory.
            ab.dump(command[command.index("--json-out") + 1], dict(fid=99.0, num_samples=10000, seed=1234))

    monkeypatch.setattr(ab, "execute", fake_execute)
    for index in range(6):
        assert ab.run_stage(plan, index) == 0
    assert sum(c.startswith("train_") for c in calls) == 11
    assert sum(c.startswith("eval_") for c in calls) == 17
    rows = ab.collect(plan)
    assert len(rows) == 22 and all(r["status"] == "complete" for r in rows)
    assert all(not r["warning"] for r in rows)
    calls.clear()
    for index in range(6):
        assert ab.run_stage(plan, index) == 0
    assert calls == []


def test_failures_are_visible_and_independent_arms_continue(plan, monkeypatch):
    plan["wandb"]["enabled"] = False
    ab.materialize(plan)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    attempted = []

    def failed(command, current, tag):
        attempted.append(tag)
        raise RuntimeError("Intentional test failure")

    monkeypatch.setattr(ab, "execute", failed)
    assert ab.run_stage(plan, 0) == 1
    assert len(attempted) == 4  # All four proposal arms attempted.
    assert not any(r["status"] == "complete" for r in ab.collect(plan))
    assert json.loads((Path(plan["root"]) / "status/stage_0.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("change", [
    {"steps": 100},
    {"sweeps": {"K": [0, 8, 16]}},
    {"reference": {"n": 1}},
    {"overrides": {"pt": {"enabled": False}}},
])
def test_invalid_manifest_rejected(tmp_path, change):
    spec = yaml.safe_load(MANIFEST.read_text())
    ab.merge(spec, change)
    source = tmp_path / "bad.yaml"
    source.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError):
        ab.build_plan(source, REPO)


def test_training_configs_share_wandb_group(plan):
    assert plan["wandb"]["enabled"]
    for arm in plan["arms"]:
        logging = arm["config"]["logging"]
        assert logging["use_wandb"]
        assert logging["project"] == "ptflow-ablation"
        assert logging["group"] == plan["wandb"]["group"]
        assert logging["job_type"] == "train"


@pytest.fixture
def older_engine_plan(tmp_path):
    archive = REPO.parent / "PT-Flow-before-speed-quality.zip"
    if not archive.exists():
        pytest.skip("Archived pre-performance engine is not installed")
    legacy = tmp_path / "old_engine"
    with zipfile.ZipFile(archive) as source:
        for relative in ("train.py", "inference.py", "ptflow/ot_drift.py", "configs/gen/ptflow_cifar10_t4.yaml"):
            target = legacy / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read("PT-Flow/" + relative))
    # No engine modules are imported and no models/data are loaded by planning.
    plan = ab.build_plan(MANIFEST, repo=legacy)
    plan["root"] = str(tmp_path / "results")
    data = Path(plan["environment"]["CIFAR10_PATH"]) / "cifar-10-batches-py"
    data.mkdir(parents=True)
    stats = Path(plan["environment"]["CIFAR10_FID_NPZ"])
    stats.parent.mkdir(parents=True)
    stats.write_text("Preflight-only fixture; not real statistics")
    return plan


def test_actual_older_engine_falls_back_without_changing_ablation(older_engine_plan):
    plan = older_engine_plan
    assert plan["engine"]["evaluation_backend"] == "png"
    assert set(plan["engine"]["removed_optimizations"]) == {
        "train.profile_every", "train.feature_cache_gib", "train.compile_generator",
        "train.extra_ema_decays", "train.ot_kwargs.reuse_costs"}
    ab.preflight(plan)  # Fresh runs must not require --init-ema.
    assert len(plan["arms"]) == 11
    assert len({e["result"] for s in plan["stages"] for e in s["entries"]}) == 17
    for arm in plan["arms"]:
        cfg = arm["config"]
        assert not (set(cfg["train"]) - set(plan["engine"]["train_args"]))
        assert "reuse_costs" not in cfg["train"]["ot_kwargs"]
        assert cfg["pt"]["enabled"]
    for stage in plan["stages"]:
        for entry in stage["entries"]:
            command = ab.eval_command(plan, entry)
            assert "--eval-backend" not in command
            assert set(arg for arg in command if arg.startswith("--")) <= set(plan["engine"]["inference_flags"])


def test_old_engine_rejects_requested_ema_initialization(older_engine_plan):
    older_engine_plan["init_ema_checkpoint"] = "existing_checkpoint.pt"
    with pytest.raises(ValueError, match="cannot EMA-initialize"):
        ab.preflight(older_engine_plan)


def test_current_engine_keeps_streaming_and_performance_settings(plan):
    assert plan["engine"]["evaluation_backend"] == "streaming"
    assert plan["engine"]["removed_optimizations"] == []
    assert "--eval-backend" in ab.eval_command(plan, plan["stages"][0]["entries"][0])


def test_unknown_training_option_is_not_silently_dropped(plan):
    cfg = copy.deepcopy(plan["arms"][0]["config"])
    cfg["train"]["misspelled_important_setting"] = 123
    with pytest.raises(ValueError, match="misspelled_important_setting"):
        ab.adapt_config(cfg, plan["engine"])
