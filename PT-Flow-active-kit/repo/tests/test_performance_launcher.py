"""No Slurm jobs are submitted: sbatch is a local stub for these tests."""
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


def bash_path(path):
    value = str(Path(path).resolve()).replace("\\", "/")
    return "/" + value[0].lower() + value[2:] if os.name == "nt" else value


@pytest.mark.parametrize("mode,rows", [("benchmark", 5), ("eval", 9), ("finetune", 3)])
def test_mock_slurm_submission(tmp_path, mode, rows):
    bash = r"C:\Program Files\Git\bin\bash.exe" if os.name == "nt" else shutil.which("bash")
    if not bash or not Path(bash).exists():
        pytest.skip("Bash not installed")
    repo = Path(__file__).resolve().parents[1]
    env_root = tmp_path / "env"
    bin_dir = env_root / "bin"
    bin_dir.mkdir(parents=True)
    for name, text in (("python", "#!/usr/bin/env bash\nexit 0\n"), ("sbatch", "#!/usr/bin/env bash\necho 12345\n")):
        script = bin_dir / name
        script.write_text(text, encoding="utf-8", newline="\n")
        script.chmod(0o755)
    base = repo / "configs/gen/ptflow_L.yaml"
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_text("submission-only fixture; no model load occurs")
    output = tmp_path / "run"
    env = dict(os.environ, CONDA_PREFIX=bash_path(env_root),
               PTFLOW_ASSETS=bash_path(tmp_path / "assets"), PTFLOW_DATA=bash_path(tmp_path / "data"),
               PATH=str(bin_dir) + os.pathsep + os.environ["PATH"])
    # Remove optional user sweep settings for reproducible fixture counts.
    for key in ("CFG_VALUES", "EMA_DECAYS", "SEEDS"):
        env.pop(key, None)
    result = subprocess.run([bash, bash_path(repo / "scripts/performance_palmetto.sh"), mode,
                             bash_path(base), bash_path(checkpoint), bash_path(output)],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len((output / "manifest.tsv").read_text().splitlines()) == rows
    for filename in ("worker.sh", "report.sh", "settings.sh"):
        subprocess.run([bash, "-n", bash_path(output / filename)], check=True, timeout=15)
    # Validate Python embedded in the original-control worker.
    worker = (output / "worker.sh").read_text()
    ast.parse(worker.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0])
    from scripts.collect_performance import collect
    collect(output)
    report = (output / "report/REPORT.md").read_text()
    assert "row 0" in report  # Submitted but unexecuted jobs are not called successes.


def test_collector_aggregates_seeds_not_best_seed(tmp_path):
    from scripts.collect_performance import collect
    (tmp_path / "results").mkdir()
    for seed, fid in ((42, 3.0), (43, 5.0)):
        row = dict(ckpt="runs/test/checkpoints/state.pt", cfg_scale=1.2, mode="A", num_samples=50000,
                   fid_ref="ref.npz", backend="streaming", seed=seed, fid=fid, world_size=1, gen_bsz=64)
        (tmp_path / "results" / f"{seed}.json").write_text(json.dumps(row))
    collect(tmp_path)
    import csv
    with (tmp_path / "report/fid_aggregate.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and float(rows[0]["fid_mean"]) == 4.0
