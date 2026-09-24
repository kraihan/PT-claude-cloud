import csv
import json
from pathlib import Path
import sys
import yaml
import pytest

ROOT = Path(__file__).resolve().parent
TEXT = (ROOT / "PTFLOW_PILOT_200_COMMANDS.txt").read_text()
CONFIG = TEXT.split('python - "$SOURCE" "$PILOT" <<\'PY\'\n', 1)[1].split('\nPY\n', 1)[0]
REPORT = TEXT.split('cat > "$PILOT/report.py" <<\'PY\'\n', 1)[1].split('\nPY\n', 1)[0]
RAW = TEXT.split('python - "$FINAL" "$PILOT/report/raw_generator_for_eval.pt" <<\'PY\'\n', 1)[1].split('\nPY\n', 1)[0]


def run(code, args, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["-", *map(str, args)])
    exec(compile(code, "pilot_embedded_python", "exec"), {})


def test_pilot_config_isolated_and_bounded(tmp_path, monkeypatch):
    source, out = tmp_path / "source", tmp_path / "out"
    source.mkdir()
    out.mkdir()
    init = tmp_path / "state_00200000.pt"
    init.touch()
    cfg = yaml.safe_load((ROOT / "PT-Flow-active-kit/repo/configs/gen/ptflow_B.yaml").read_text())
    cfg["train"]["init_ema_from"] = str(init)
    cfg["pt"]["schedule"]["policy"] = "active_recovery_v1"
    base = source / "active_B_30k.yaml"
    original = yaml.safe_dump(cfg)
    base.write_text(original)
    run(CONFIG, [source, out], monkeypatch)
    new = yaml.safe_load((out / "pilot.yaml").read_text())
    assert base.read_text() == original
    assert new["model"] == cfg["model"]
    assert new["train"]["init_ema_from"] == str(init)
    assert not new["train"].get("resume_from")
    assert new["train"]["total_steps"] == 200
    assert new["train"]["train_batch_size"] * new["train"]["forward_dict"]["gen_per_label"] == 1024
    assert new["train"]["train_batch_size"] // 2 // new["train"]["grad_accum_steps"] == 2
    assert new["train"]["save_per_step"] == new["train"]["keep_every"] == 50
    assert new["train"]["feature_chunk_size"] == 16
    assert new["feature"]["checkpoint"] is True
    assert new["train"]["eval_per_step"] == 0  # explicit same-protocol evaluations instead
    assert new["pt"]["prox_mode"] == "full"
    sys.path.insert(0, str(ROOT / "PT-Flow-active-kit/repo"))
    from ptflow.schedule import build_schedule
    sched = build_schedule(new["pt"]["schedule"])
    used = []
    for _ in range(200):
        used.append(sched.lambda_prox())
        assert sched.eps() == .1
        sched.observe(.5)
    assert used[20] == 0 and used[21] > 0
    assert used[100] == used[-1] == .02
    assert sched.alignment_lambda() == .1


@pytest.mark.parametrize("complete", [False, True])
def test_report_does_not_mistake_refined_health_for_generator_health(tmp_path, monkeypatch, complete):
    report = tmp_path / "report"
    report.mkdir()
    result = dict(step=200000, fid=5., num_samples=10000, seed=42, cfg_scale=1.2,
        mode="A", world_size=2, gen_bsz=32, fid_ref="/same/ref.npz",
        backend="streaming", feature_extractor="inception-v3-compat")
    (report / "initializer_ema.json").write_text(json.dumps(result))
    if complete:
        for name in ("pilot_raw", "pilot_ema"):
            (report / f"{name}.json").write_text(json.dumps({**result, "step": 200, "fid": 6.}))
    (tmp_path / "run").mkdir()
    with (tmp_path / "run/pt_status.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "pt/lambda_prox", "pt/generator_proposal_control_ess",
                                             "pt/prox_resid_rel", "pt/loss_alignment"])
        writer.writeheader()
        for step in (40, 60, 80):
            writer.writerow(dict(step=step, **{"pt/lambda_prox": .02,
                "pt/generator_proposal_control_ess": .01, "pt/prox_resid_rel": .8, "pt/loss_alignment": 1.}))
    run(REPORT, [tmp_path], monkeypatch)
    summary = json.loads((report / "diagnostics.json").read_text())
    assert summary["completed_steps"] == 80
    assert not summary["full_run_checkpoint"]
    assert summary["proposal_near_floor_below_0p05"]
    assert summary["complete_fid_comparison"] == complete
    assert summary["decision"].startswith("REVIEW_REQUIRED")
    rows = list(csv.DictReader((report / "fid.csv").open()))
    assert rows[0]["new_steps"] == "0"
    assert len(rows) == (3 if complete else 1)


def test_report_rejects_mismatched_fid_protocol(tmp_path, monkeypatch):
    report = tmp_path / "report"
    report.mkdir()
    for tag, seed in (("initializer_ema", 42), ("pilot_ema", 99)):
        (report / f"{tag}.json").write_text(json.dumps(dict(step=200, fid=3, seed=seed, num_samples=10000, cfg_scale=1.2)))
    with pytest.raises(RuntimeError, match="protocols differ"):
        run(REPORT, [tmp_path], monkeypatch)


def test_raw_evaluation_adapter_contains_no_optimizer(tmp_path, monkeypatch):
    import torch
    src, dest = tmp_path / "checkpoint.pt", tmp_path / "raw.pt"
    torch.save(dict(step=200, model={"weight": torch.ones(2)}, ema_model={"weight": torch.zeros(2)},
                    optimizer={"unwanted": True}), src)
    run(RAW, [src, dest], monkeypatch)
    out = torch.load(dest, weights_only=False)
    assert torch.equal(out["ema_model"]["weight"], torch.ones(2))
    assert out["weights_kind"] == "RAW_GENERATOR_EVALUATION_ONLY"
    assert "optimizer" not in out
