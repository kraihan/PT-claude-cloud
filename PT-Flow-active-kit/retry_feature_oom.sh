#!/usr/bin/env bash
# Config-only repair for failed job 16179458. No production code edits.
set -Eeuo pipefail
source "$HOME/ptflow_env.sh"
ACTIVE_HOME="/scratch/$USER/ptflow/active_pt/B_H200x2_30k_20260923_141250"
test -s "$ACTIVE_HOME/active_B_30k.yaml"
test -s "$ACTIVE_HOME/worker.sbatch"

# Prevent another submission while a job for this exact run is still queued.
CURRENT_JOB=$(cat "$ACTIVE_HOME/job_id.txt")
QUEUED=$(squeue -h -u "$USER" -o '%i')
if grep -Fxq "$CURRENT_JOB" <<< "$QUEUED"; then
    echo "Job $CURRENT_JOB is still queued/running for this directory; not submitting a duplicate."
    exit 1
fi

python - "$ACTIVE_HOME" <<'PY'
from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import sys
import yaml

root = Path(sys.argv[1])
path = root / "active_B_30k.yaml"
if list((root / "run/checkpoints").glob("state_*.pt")):
    raise SystemExit("A checkpoint now exists; this repair command is for the step-zero failed run.")
cfg = yaml.safe_load(path.read_text())
train = cfg["train"]
assert cfg["pt"]["schedule"]["policy"] == "active_recovery_v1"
assert train["total_steps"] == 30000 and train["save_per_step"] == 1000
assert train["train_batch_size"] == 128 and train["grad_accum_steps"] == 32
assert train["forward_dict"]["gen_per_label"] == 64
assert train.get("init_ema_from") and not train.get("resume_from")
stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
shutil.copy2(path, root / f"active_B_30k.before_feature_oom_{stamp}.yaml")
changes = {
    "train.feature_chunk_size": [train.get("feature_chunk_size", 0), 16],
    "train.feature_cache_gib": [train.get("feature_cache_gib", 0), 0.0],
    "feature.chunk_size": [cfg.get("feature", {}).get("chunk_size", 64), 16],
    "feature.checkpoint": [cfg.get("feature", {}).get("checkpoint", False), True],
}
# These controls bound different allocations; both are required.
train["feature_chunk_size"] = 16
train["feature_cache_gib"] = 0.0
cfg.setdefault("feature", {}).update(chunk_size=16, checkpoint=True)
tmp = path.with_suffix(".tmp")
tmp.write_text(yaml.safe_dump(cfg, sort_keys=False))
tmp.replace(path)
(root / f"feature_oom_fix_{stamp}.json").write_text(json.dumps(changes, indent=2) + "\n")
print("Updated:", path)
print("Real features stream per 2-label microbatch; MAE forwards use at most 16 images.")
print("Preserved: global batch 8192, K=16, full PT, 30k steps, 1k checkpoints, weights-only init.")
PY

RESPONSE=$(sbatch --parsable \
    --chdir="$ACTIVE_HOME/PT-Flow" \
    --output="$ACTIVE_HOME/logs/train_%j.out" \
    --error="$ACTIVE_HOME/logs/train_%j.err" \
    "$ACTIVE_HOME/worker.sbatch" "$ACTIVE_HOME")
JOB=${RESPONSE%%;*}
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $RESPONSE"; exit 1; }
printf '%s\n' "$CURRENT_JOB" >> "$ACTIVE_HOME/previous_job_ids.txt"
printf '%s\n' "$JOB" > "$ACTIVE_HOME/job_id.txt"
printf '%s\n' "$ACTIVE_HOME" > "/scratch/$USER/ptflow/active_pt/latest_B.txt"
echo "Submitted corrected H200x2 job: $JOB"
echo "tail -F $ACTIVE_HOME/logs/train_${JOB}.out $ACTIVE_HOME/logs/train_${JOB}.err"
squeue -j "$JOB" -o '%.16i %.30j %.3t %.12M %R'
