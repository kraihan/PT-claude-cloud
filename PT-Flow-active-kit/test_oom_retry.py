import copy
from pathlib import Path
import sys
import yaml


def test_retry_changes_only_memory_settings_and_keeps_backup(tmp_path, monkeypatch):
    kit = Path(__file__).resolve().parent
    cfg = yaml.safe_load((kit / "repo/configs/gen/ptflow_B.yaml").read_text())
    cfg["train"].update(total_steps=30000, save_per_step=1000, train_batch_size=128,
                        grad_accum_steps=32, init_ema_from="/source/state_00200000.pt")
    cfg["pt"]["schedule"]["policy"] = "active_recovery_v1"
    path = tmp_path / "active_B_30k.yaml"
    before = yaml.safe_dump(cfg, sort_keys=False)
    path.write_text(before)
    text = (kit / "retry_feature_oom.sh").read_text()
    script = text.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    monkeypatch.setattr(sys, "argv", ["-", str(tmp_path)])
    exec(compile(script, "retry_feature_oom.sh:python", "exec"), {})
    expected = copy.deepcopy(cfg)
    expected["train"].update(feature_chunk_size=16, feature_cache_gib=0.)
    expected["feature"].update(chunk_size=16, checkpoint=True)
    assert yaml.safe_load(path.read_text()) == expected
    backups = list(tmp_path.glob("active_B_30k.before_feature_oom_*.yaml"))
    assert len(backups) == 1 and backups[0].read_text() == before
    assert len(list(tmp_path.glob("feature_oom_fix_*.json"))) == 1
