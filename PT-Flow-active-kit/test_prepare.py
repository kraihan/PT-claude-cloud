import importlib.util
import hashlib
from pathlib import Path
import yaml
import pytest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("active_prepare", ROOT / "prepare.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_isolated_recipe_and_fresh_state(tmp_path):
    base = ROOT / "repo/configs/gen/ptflow_B.yaml"
    before = hashlib.sha256(base.read_bytes()).hexdigest()
    ckpt = tmp_path / "state_00200000.pt"
    ckpt.write_bytes(b"installer only checks existence; GPU preflight checks the payload")
    dest = tmp_path / "isolated"
    cfg = module.prepare(base, dest, ckpt)
    assert hashlib.sha256(base.read_bytes()).hexdigest() == before
    assert cfg["train"]["init_ema_from"] == str(ckpt.resolve())
    assert "resume_from" not in cfg["train"]
    assert cfg["train"]["total_steps"] == 30000
    assert cfg["train"]["save_per_step"] == cfg["train"]["keep_every"] == 1000
    assert cfg["train"]["train_batch_size"] * cfg["train"]["forward_dict"]["gen_per_label"] == 8192
    assert cfg["train"]["train_batch_size"] // 2 // cfg["train"]["grad_accum_steps"] == 2
    assert cfg["pt"]["schedule"]["policy"] == "active_recovery_v1"
    assert cfg["pt"]["prox_mode"] == "full"
    assert cfg["train"]["feature_chunk_size"] > 0
    assert cfg["train"]["feature_cache_gib"] == 0
    assert cfg["feature"]["chunk_size"] == 16
    assert cfg["feature"]["checkpoint"] is True
    assert cfg["model"]["residual"] is False
    assert cfg["logging"]["name"] == dest.name
    assert (dest / "PT-Flow/train.py").is_file()
    assert not (dest / "run/checkpoints").exists()
    assert yaml.safe_load((dest / "active_B_30k.yaml").read_text()) == cfg
    with pytest.raises(FileExistsError):
        module.prepare(base, dest, ckpt)
