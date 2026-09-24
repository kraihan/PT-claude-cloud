import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("full_launcher", ROOT / "submit_B_PT_Full.py")
full = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full)
BASH = "C:/Program Files/Git/bin/bash.exe"


@pytest.fixture
def pilot(tmp_path, monkeypatch):
    # Generate the pilot using its actual shipped configuration block.
    source, out = tmp_path / "source", tmp_path / "pilot"
    source.mkdir()
    out.mkdir()
    initializer = tmp_path / "state_00200000.pt"
    initializer.touch()
    base = yaml.safe_load((ROOT / "PT-Flow-active-kit/repo/configs/gen/ptflow_B.yaml").read_text())
    base["train"]["init_ema_from"] = str(initializer)
    base["pt"]["schedule"]["policy"] = "active_recovery_v1"
    (source / "active_B_30k.yaml").write_text(yaml.safe_dump(base))
    text = (ROOT / "submit_pt_epscos200.sh").read_text()
    code = text.split('python - "$SOURCE" "$PILOT" <<\'PY\'\n', 1)[1].split('\nPY\n', 1)[0]
    monkeypatch.setattr(sys, "argv", ["-", str(source), str(out)])
    exec(compile(code, "pilot_config", "exec"), {})
    return out


def test_long_run_preserves_pilot_batch_and_initialization(pilot):
    original = (pilot / "pilot.yaml").read_text()
    base = yaml.safe_load(original)
    cfg = full.configure(base)
    assert (pilot / "pilot.yaml").read_text() == original
    assert base == yaml.safe_load(original)
    assert cfg["model"] == base["model"]
    assert cfg["dataset"] == base["dataset"]
    assert cfg["feature"] == base["feature"]
    train = cfg["train"]
    assert train["train_batch_size"] * train["forward_dict"]["gen_per_label"] == 1024
    assert train["train_batch_size"] // 2 // train["grad_accum_steps"] == 2
    assert train["total_steps"] == 30000
    assert train["save_per_step"] == train["keep_every"] == 1000
    assert train["init_ema_from"] == base["train"]["init_ema_from"]
    assert not train.get("resume_from")
    assert cfg["pt"]["prox_mode"] == "full" and cfg["pt"]["enabled"]
    assert cfg["pt"]["schedule"]["lambda_scale"] > 0
    for block in (cfg, cfg["pt"]):
        assert block["optimizer"]["lr_schedule"]["total_steps"] == 30000
    assert cfg["logging"]["name"] == "B_PT_Full"


def test_active_schedule_advances_without_faking_health(pilot):
    cfg = full.configure(yaml.safe_load((pilot / "pilot.yaml").read_text()))
    sys.path.insert(0, str(ROOT / "PT-Flow-active-kit/repo"))
    from ptflow.schedule import build_schedule
    sched = build_schedule(cfg["pt"]["schedule"])
    expected = {0: .1, 20: .1, 10020: .075, 20020: .05, 30000: .05}
    for step in range(30001):
        assert sched.update_potential()
        if step in expected:
            assert sched.eps() == pytest.approx(expected[step])
        if step == 20:
            assert sched.lambda_prox() == 0
        if step >= 100:
            assert sched.lambda_prox() == pytest.approx(.02)
        if step == 1000:
            assert sched.alignment_lambda() == pytest.approx(.1)
        if step >= 5000:
            assert sched.alignment_lambda() == 0
        if step < 30000:
            sched.observe(0)
    assert sched.health.is_broken
    assert sched.recovery_failure()  # failure must remain visible, not suppressed


def test_installer_copies_without_editing_source(pilot, tmp_path):
    source = Path((pilot / "source.txt").read_text().strip())
    repo = source / "PT-Flow"
    (repo / "ptflow").mkdir(parents=True)
    (repo / "train.py").write_text('# Existing checkpoints: resuming full state\n')
    (repo / "inference.py").write_text('# --eval-backend\n')
    (repo / "ptflow/schedule.py").write_text('# active_recovery_v1\n')
    (repo / "ptflow/run_guard.py").write_text('# original\n')
    (source / "settings.sh").write_text('export PTFLOW_DATA=/existing/cache\n')
    (pilot / "run/checkpoints").mkdir(parents=True)
    (pilot / "run/checkpoints/state_00000200.pt").touch()
    before = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
    home = tmp_path / "full"
    manifest = full.prepare(pilot, home)
    assert all(p.read_bytes() == content for p, content in before.items())
    assert (home / "PT-Flow/train.py").read_bytes() == (repo / "train.py").read_bytes()
    assert not (home / "B_PT_Full/checkpoints").exists()
    assert manifest["init_kind"].startswith("W-Flow EMA generator weights only")
    subprocess.run([BASH, "-n", str(home / "worker.sbatch")], check=True)
    with pytest.raises(FileExistsError):
        full.prepare(pilot, home)


def result(step, **changes):
    return dict(step=step, num_samples=50000, mode="A", cfg_scale=1.2, seed=42,
                world_size=2, gen_bsz=32, backend="streaming", fid=4.1,
                fid_ref="same.npz", feature_extractor="same", **changes)


def test_report_partial_failure_and_complete(tmp_path):
    report = tmp_path / "report"
    report.mkdir()
    assert not full.report(tmp_path)["completed"]
    with pytest.raises(RuntimeError, match="Incomplete"):
        full.report(tmp_path, True)
    checkpoint = tmp_path / "B_PT_Full/checkpoints/state_00030000.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    for name, step in (("initializer_ema", 200000), ("final_ema", 30000), ("final_raw", 30000)):
        (report / f"{name}.json").write_text(json.dumps(result(step)))
    assert full.report(tmp_path, True)["completed"]
    failure = tmp_path / "B_PT_Full/PT_FAILED.txt"
    failure.write_text("Estimator broken")
    with pytest.raises(RuntimeError, match="Incomplete"):
        full.report(tmp_path, True)


def test_rejects_different_protocol_or_reference(tmp_path):
    (tmp_path / "report").mkdir()
    original = result(200000)
    path = tmp_path / "report/initializer_ema.json"
    path.write_text(json.dumps(original))
    assert full.result_valid(path, 200000)
    original["num_samples"] = 10000
    path.write_text(json.dumps(original))
    assert not full.result_valid(path, 200000)
    path.write_text(json.dumps(result(200000)))
    changed = result(30000)
    changed["fid_ref"] = "different.npz"
    (tmp_path / "report/final_ema.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="differs"):
        full.report(tmp_path)


def test_raw_export_and_atomic_checkpoint_policy(tmp_path):
    import torch
    src, dest = tmp_path / "final.pt", tmp_path / "raw.pt"
    torch.save(dict(step=30000, model={"w": torch.ones(1)}, ema_model={"w": torch.zeros(1)},
                    optimizer={"must_not_export": 1}), src)
    full.raw_export(src, dest)
    payload = torch.load(dest, weights_only=False)
    assert payload["ema_model"]["w"].item() == 1
    assert "optimizer" not in payload
    assert payload["weights_kind"] == "RAW_GENERATOR_EVALUATION_ONLY"


@pytest.mark.parametrize("exit_code", [0, 7])
def test_worker_wait_preserves_child_exit_code(exit_code):
    function = full.WORKER.split('run_command() {', 1)[1].split('\nrequeue_if_requested()', 1)[0]
    code = 'set -e\nrun_command() {' + function + f'\nrun_command "$BASH" -c "exit {exit_code}"\n'
    completed = subprocess.run([BASH, "-c", code], capture_output=True, text=True)
    assert completed.returncode == exit_code, completed.stderr
