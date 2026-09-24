#!/usr/bin/env bash
# Submit the fixed-checkpoint bridge/prox/network diagnostic on Palmetto.
#
# This is deliberately a single, typed-GPU evaluation job -- not an array and
# not a training/resume job.  It refuses an existing output directory, so a
# rerun cannot silently mix two estimates.
set -Eeuo pipefail

source ~/ptflow_env.sh
export WORK="${WORK:-/scratch/$USER/ptflow}"
export REPO="${REPO:-$WORK/PT-Flow}"

CKPT="${CKPT:-$WORK/runs/in256_L_warmstart/checkpoints/state_00019000.pt}"
CONFIG="${CONFIG:-$WORK/runs/in256_L_warmstart/config.json}"
OUT="${OUT:-$WORK/runs/in256_L_warmstart/prox_table_step19000}"
PARTITION="${PALMETTO_PARTITION:-work1}"
GPU_TYPE="${GPU_TYPE:-h200}"
POINTS="${POINTS:-16}"
POINT_BATCH="${POINT_BATCH:-4}"
K_REF="${K_REF:-8192}"
PROPOSAL_CHUNK="${PROPOSAL_CHUNK:-64}"
MC_SEEDS="${MC_SEEDS:-5}"
EPS_LIST="${EPS_LIST:-0.20,0.10,0.05,0.02,0.01}"
CFG_SCALE="${CFG_SCALE:-1.2}"
ALPHA_DEF="${ALPHA_DEF:-0.05}"
WALLTIME="${WALLTIME:-24:00:00}"
MEMORY="${MEMORY:-240G}"
CPUS="${CPUS:-8}"

case "$GPU_TYPE" in
  a40|l40|l40s|a100|h100|h200) ;;
  *) echo "Refusing unsupported GPU_TYPE=$GPU_TYPE; use a40, l40, l40s, a100, h100, or h200." >&2; exit 2 ;;
esac

[[ -s "$CKPT" ]] || { echo "Missing checkpoint: $CKPT" >&2; exit 1; }
[[ -s "$CONFIG" ]] || { echo "Missing matching config: $CONFIG" >&2; exit 1; }
[[ -s "$REPO/scripts/eval_prox_table.py" ]] || {
  echo "Missing evaluator: $REPO/scripts/eval_prox_table.py" >&2
  echo "Install the evaluated repository change before submitting." >&2
  exit 1
}
[[ ! -e "$OUT" ]] || { echo "Refusing existing output directory: $OUT" >&2; exit 1; }

python - "$CONFIG" "$CKPT" "$OUT" "$GPU_TYPE" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
ckpt = Path(sys.argv[2])
out = Path(sys.argv[3])
gpu = sys.argv[4]
cfg = json.loads(config_path.read_text(encoding="utf-8"))
pt = cfg.get("pt") or {}
model = cfg.get("model") or {}
train = cfg.get("train") or {}
if not pt.get("enabled", False):
    raise SystemExit("config.pt.enabled is false; this checkpoint has no PT potential to evaluate")
print(json.dumps({
    "job_kind": "fixed-checkpoint evaluation; no training or checkpoint writes",
    "gpu_request": f"{gpu}:1",
    "checkpoint": str(ckpt),
    "matching_config": str(config_path),
    "new_output": str(out),
    "generator": {k: model.get(k) for k in ("hidden_size", "depth", "num_heads", "patch_size", "cond_dim", "use_remat")},
    "pt_potential": {k: (pt.get("model") or {}).get(k) for k in ("hidden_size", "depth", "num_heads", "patch_size")},
    "provenance_total_steps": train.get("total_steps"),
    "provenance_save_per_step": train.get("save_per_step"),
    "provenance_init_generator_from": train.get("init_generator_from", cfg.get("init_generator_from")),
    "warmstart_or_raw": "not applicable: this evaluates the saved state_00019000 checkpoint in place",
}, indent=2))
PY

mkdir -p "$OUT/logs" "$OUT/results"
cat > "$OUT/worker.sh" <<'WORKER'
#!/usr/bin/env bash
set -Eeuo pipefail
source ~/ptflow_env.sh
export WORK="${WORK:-/scratch/$USER/ptflow}"
export REPO="${REPO:-$WORK/PT-Flow}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export DRIFT_COMPILE=0

CKPT="$WORK/runs/in256_L_warmstart/checkpoints/state_00019000.pt"
CONFIG="$WORK/runs/in256_L_warmstart/config.json"
cd "$REPO"
test -s "$CKPT"
test -s "$CONFIG"
test -s scripts/eval_prox_table.py
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

python scripts/eval_prox_table.py \
  --ckpt "$CKPT" \
  --config "$CONFIG" \
  --out "$PTFLOW_PROX_RESULTS" \
  --eps "$PTFLOW_PROX_EPS" \
  --cfg-scale "$PTFLOW_PROX_CFG" \
  --points "$PTFLOW_PROX_POINTS" \
  --batch-size "$PTFLOW_PROX_POINT_BATCH" \
  --K "$PTFLOW_PROX_K" \
  --proposal-chunk "$PTFLOW_PROX_CHUNK" \
  --mc-seeds "$PTFLOW_PROX_MC_SEEDS" \
  --seed 19000 \
  --alpha-def "$PTFLOW_PROX_ALPHA" \
  --reference-scale learned \
  --prox-max-steps 500 \
  --prox-initial-step 0.5 \
  --prox-tolerance 1e-3 \
  --min-mean-ess 0.05
WORKER
chmod 700 "$OUT/worker.sh"

SBATCH_ARGS=(
  --parsable
  --partition="$PARTITION"
  --nodes=1
  --ntasks=1
  --cpus-per-task="$CPUS"
  --mem="$MEMORY"
  --time="$WALLTIME"
  --gpus="$GPU_TYPE:1"
  --job-name=ptf-prox-table-19k
  --output="$OUT/logs/%x_%j.out"
  --error="$OUT/logs/%x_%j.err"
  --export=ALL,PTFLOW_PROX_RESULTS="$OUT/results",PTFLOW_PROX_EPS="$EPS_LIST",PTFLOW_PROX_CFG="$CFG_SCALE",PTFLOW_PROX_POINTS="$POINTS",PTFLOW_PROX_POINT_BATCH="$POINT_BATCH",PTFLOW_PROX_K="$K_REF",PTFLOW_PROX_CHUNK="$PROPOSAL_CHUNK",PTFLOW_PROX_MC_SEEDS="$MC_SEEDS",PTFLOW_PROX_ALPHA="$ALPHA_DEF"
)
[[ -z ${PALMETTO_ACCOUNT:-} ]] || SBATCH_ARGS+=(--account="$PALMETTO_ACCOUNT")

echo "Current active jobs (informational):"
squeue --me
RAW_JOB=$(sbatch "${SBATCH_ARGS[@]}" "$OUT/worker.sh")
JOB_ID=${RAW_JOB%%;*}
[[ $JOB_ID =~ ^[0-9]+$ ]] || { echo "Could not parse sbatch output: $RAW_JOB" >&2; exit 1; }
printf '%s\n' "$JOB_ID" > "$OUT/job_id.txt"
printf 'Submitted job %s\nLog command after it starts:\n' "$JOB_ID"
printf 'J=%q\nLOG=$(scontrol show job -o "$J" | sed -n '\''s/.*StdOut=\\([^ ]*\\).*/\\1/p'\'')\necho "Log: $LOG"\ntail -F "$LOG"\n' "$JOB_ID"
