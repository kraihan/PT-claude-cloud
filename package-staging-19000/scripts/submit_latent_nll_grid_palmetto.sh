#!/usr/bin/env bash
# One typed-GPU, fixed-validation-latent NLL grid. No training and no array.
set -Eeuo pipefail

source ~/ptflow_env.sh
export WORK="${WORK:-/scratch/$USER/ptflow}"
export REPO="${REPO:-$WORK/PT-Flow}"
CKPT="${CKPT:-$WORK/runs/in256_L_warmstart/checkpoints/state_00019000.pt}"
CONFIG="${CONFIG:-$WORK/runs/in256_L_warmstart/config.json}"
OUT="${OUT:-$WORK/runs/in256_L_warmstart/latent_nll_step19000}"
GPU_TYPE="${GPU_TYPE:-h200}"
PARTITION="${PALMETTO_PARTITION:-work1}"
POINTS="${POINTS:-32}"
MC_SEEDS="${MC_SEEDS:-3}"
WALLTIME="${WALLTIME:-24:00:00}"
MEMORY="${MEMORY:-240G}"
CPUS="${CPUS:-8}"
CACHE="${IMAGENET_CACHE_PATH:-$WORK/data/latents}"

case "$GPU_TYPE" in a40|l40|l40s|a100|h100|h200) ;; *) echo "Unsupported GPU_TYPE=$GPU_TYPE" >&2; exit 2;; esac
for f in "$CKPT" "$CONFIG" "$CACHE/val_moments.npy" "$CACHE/val_targets.npy" "$REPO/scripts/eval_imagenet_latent_nll_grid.py"; do
  [[ -s "$f" ]] || { echo "Missing required file: $f" >&2; exit 1; }
done
[[ ! -e "$OUT" ]] || { echo "Refusing existing output directory: $OUT" >&2; exit 1; }
python - "$CONFIG" "$CKPT" "$OUT" "$GPU_TYPE" "$CACHE" <<'PY'
import json, sys
from pathlib import Path
c = json.loads(Path(sys.argv[1]).read_text())
if not (c.get("pt") or {}).get("enabled", False): raise SystemExit("PT potential is disabled in matching config")
print(json.dumps({
  "job_kind": "frozen-checkpoint fixed-validation-latent nested-MC NLL; no training/checkpoint writes",
  "gpu_request": sys.argv[4] + ":1", "checkpoint": sys.argv[2], "config": sys.argv[1], "output": sys.argv[3], "latent_cache": sys.argv[5],
  "provenance_total_steps": (c.get("train") or {}).get("total_steps"),
  "provenance_save_per_step": (c.get("train") or {}).get("save_per_step"),
  "warmstart_or_raw": "not applicable: evaluates existing state_00019000 only",
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
cd "$REPO"
CKPT="$WORK/runs/in256_L_warmstart/checkpoints/state_00019000.pt"
CONFIG="$WORK/runs/in256_L_warmstart/config.json"
test -s "$CKPT"; test -s "$CONFIG"; test -s scripts/eval_imagenet_latent_nll_grid.py
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
python scripts/eval_imagenet_latent_nll_grid.py --ckpt "$CKPT" --config "$CONFIG" \
  --out "$PTFLOW_NLL_RESULTS" --latent-cache "$PTFLOW_NLL_CACHE" --points "$PTFLOW_NLL_POINTS" \
  --batch-size 1 --k-outer 8,16,32 --k-inner 8,16,32 --mc-seeds "$PTFLOW_NLL_SEEDS" --seed 19000
WORKER
chmod 700 "$OUT/worker.sh"
ARGS=(--parsable --partition="$PARTITION" --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" --mem="$MEMORY" --time="$WALLTIME" --gpus="$GPU_TYPE:1" --job-name=ptf-latent-nll-19k --output="$OUT/logs/%x_%j.out" --error="$OUT/logs/%x_%j.err" --export=ALL,PTFLOW_NLL_RESULTS="$OUT/results",PTFLOW_NLL_CACHE="$CACHE",PTFLOW_NLL_POINTS="$POINTS",PTFLOW_NLL_SEEDS="$MC_SEEDS")
[[ -z ${PALMETTO_ACCOUNT:-} ]] || ARGS+=(--account="$PALMETTO_ACCOUNT")
squeue --me
RAW=$(sbatch "${ARGS[@]}" "$OUT/worker.sh")
JOB=${RAW%%;*}; [[ $JOB =~ ^[0-9]+$ ]] || { echo "Could not parse sbatch output: $RAW" >&2; exit 1; }
printf '%s\n' "$JOB" > "$OUT/job_id.txt"
echo "Submitted $JOB. Results will be in $OUT/results/."
