#!/usr/bin/env bash
# Self-contained dense-CFG 50k FID sweep on the in256_L checkpoint.
#
#   bash fid_sweep/submit_cfg_sweep.sh
#
# Submits a 12-row SLURM array (one cfg each, MAX_JOBS at a time), each row
# running THIS repo's `inference.py evaluate` (Mode A, streaming FID).  After the
# array finishes, a dependent job folds every result JSON into a cumulative
# master CSV and writes a per-run report CSV.  Does NOT depend on
# performance_palmetto.sh -- it needs only inference.py, which is in this repo.
#
# Everything is env-overridable; see fid_sweep/README.md for the smoke test.
set -Eeuo pipefail

source ~/ptflow_env.sh

# Self-locate: this script and the appender live in fid_sweep/; the repo is its
# parent.  NOTE: ~/ptflow_env.sh exports its own REPO and RUN, so we do NOT read
# them here -- REPO is forced to wherever THIS script lives, and the run/ckpt use
# dedicated override names (RUN_DIR / CONFIG_PATH / CKPT_PATH) the env can't clobber.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APPEND="$HERE/append_fid_master.py"
REPO="$(cd "$HERE/.." && pwd)"
test -s "$APPEND"          || { echo "MISSING appender: $APPEND" >&2; exit 1; }
test -s "$REPO/inference.py" || { echo "MISSING inference.py in repo: $REPO" >&2; exit 1; }

export WORK="${WORK:-/scratch/$USER/ptflow}"
RUN_DIR="${RUN_DIR:-$WORK/runs/in256_L_warmstart}"

# Asset/data env the eval needs (same names performance_palmetto.sh exports).
export PTFLOW_ASSETS="${PTFLOW_ASSETS:-$WORK/assets}"
export PTFLOW_DATA="${PTFLOW_DATA:-$WORK/data}"
export TORCH_HUB_DIR="${TORCH_HUB_DIR:-$PTFLOW_ASSETS/torch_hub}"
export HF_ROOT="${HF_ROOT:-$PTFLOW_ASSETS/mae}"
export VAE_HF_PATH="${VAE_HF_PATH:-$PTFLOW_ASSETS/sdvae}"
export IMAGENET_CACHE_PATH="${IMAGENET_CACHE_PATH:-$PTFLOW_DATA/latents}"
export CIFAR10_FID_NPZ="${CIFAR10_FID_NPZ:-$PTFLOW_ASSETS/fid_stats/cifar10_train_fid_stats.npz}"
export IMAGENET_FID_NPZ="${IMAGENET_FID_NPZ:-$PTFLOW_ASSETS/fid_stats/jit_in256_stats.npz}"

# --- knobs (match the "regular scheduling command", GPU auto-pickable) --------
# Throughput ranks h200 > h100 > a100; if the h200 queue is long, set GPU_TYPE=a100.
# GPU_TYPE=any (or empty) requests an untyped GPU -- SLURM grabs whatever is free
# (handy for a small smoke where the card doesn't matter).
GPU_TYPE="${GPU_TYPE:-h200}"
FID_SAMPLES="${FID_SAMPLES:-50000}"          # 50k only; must be a multiple of num_classes (1000)
CFG_VALUES="${CFG_VALUES:-0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 2.2}"   # 12 values
MAX_JOBS="${MAX_JOBS:-12}"                   # 12 infer schedules at a time
SEEDS="${SEEDS:-42}"
EMA_DECAYS="${EMA_DECAYS:-primary}"
GEN_BSZ="${GEN_BSZ:-16}"
WALLTIME="${WALLTIME:-12:00:00}"
CONFIG="${CONFIG_PATH:-$RUN_DIR/config.json}"
CKPT="${CKPT_PATH:-$RUN_DIR/checkpoints/state_00014000.pt}"
MASTER="${MASTER:-$WORK/results/fid_master.csv}"   # cumulative FID master CSV
PARTITION="${PALMETTO_PARTITION:-work1}"
PT_ENV="${CONDA_PREFIX:-$(python -c 'import sys; print(sys.prefix)')}"

echo "Repo    : $REPO"
echo "Run dir : $RUN_DIR"
echo "Config  : $CONFIG"
echo "Ckpt    : $CKPT"
echo "FID ref : $IMAGENET_FID_NPZ"
for f in "$CONFIG" "$CKPT" "$IMAGENET_FID_NPZ"; do
  [[ -s "$f" ]] || { echo "MISSING required file: $f  (override with RUN_DIR=... or CONFIG_PATH=.../CKPT_PATH=...)" >&2; exit 1; }
done

# GPU request. A typed request (--gpus=<type>:1) is what work1 accepts; the
# untyped generic form uses --gres=gpu:1 (more widely schedulable than --gpus=1,
# which failed on work1 with "node configuration not available").
if [[ -z "$GPU_TYPE" || "$GPU_TYPE" == "any" ]]; then
  GPU_LABEL="any"; GRES=(--gres=gpu:1)
else
  GPU_LABEL="$GPU_TYPE"; GRES=(--gpus="$GPU_TYPE:1")
fi

STAMP=$(date +%Y%m%d_%H%M%S)
OUT="$WORK/results/in256_L_step14k_cfg_dense_${FID_SAMPLES}_${GPU_LABEL}_${STAMP}"
mkdir -p "$OUT"/logs "$OUT"/results "$OUT"/eval

# Paths/settings are shell-quoted into a file the worker sources -- never
# interpolated into executable code.
for key in REPO PT_ENV CKPT CONFIG FID_SAMPLES GEN_BSZ IMAGENET_FID_NPZ OUT \
           PTFLOW_ASSETS PTFLOW_DATA TORCH_HUB_DIR HF_ROOT VAE_HF_PATH \
           IMAGENET_CACHE_PATH CIFAR10_FID_NPZ; do
  printf '%s=%q\n' "$key" "${!key}"
done > "$OUT/settings.sh"

# 12-row manifest: cfg x seed x ema.
for seed in $SEEDS; do for ema in $EMA_DECAYS; do for cfg in $CFG_VALUES; do
  printf '%s\t%s\t%s\n' "$cfg" "$seed" "$ema"
done; done; done > "$OUT/manifest.tsv"

cat > "$OUT/worker.sh" <<'WORKER'
#!/usr/bin/env bash
set -Eeuo pipefail
source "$1/settings.sh"
export PATH="$PT_ENV/bin:$PATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
cd "$REPO"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1/manifest.tsv")
[[ -n "$LINE" ]] || { echo "missing manifest row $SLURM_ARRAY_TASK_ID" >&2; exit 1; }
IFS=$'\t' read -r CFG SEED EMA <<< "$LINE"
TAG="cfg${CFG}_seed${SEED}_ema${EMA}"
EXTRA=()
[[ "$EMA" == primary ]] || EXTRA+=(--ema-decay "$EMA")
srun --ntasks=1 python inference.py evaluate --ckpt "$CKPT" --config "$CONFIG" \
  --sampler A --cfg-scale "$CFG" --seed "$SEED" --num-samples "$FID_SAMPLES" \
  --gen-bsz "$GEN_BSZ" --fid-ref "$IMAGENET_FID_NPZ" \
  --workdir "$1/eval/$TAG" --json-out "$1/results/$TAG.json" \
  --eval-backend streaming "${EXTRA[@]}"
touch "$1/results/done_${SLURM_ARRAY_TASK_ID}"
WORKER

COUNT=$(wc -l < "$OUT/manifest.tsv")
COMMON=(--parsable --partition="$PARTITION" --nodes=1 --ntasks=1 --export=ALL)
[[ -z "${PALMETTO_ACCOUNT:-}" ]] || COMMON+=(--account="$PALMETTO_ACCOUNT")

# --- 1. the sweep array -------------------------------------------------------
ARRAY=$(sbatch "${COMMON[@]}" --job-name=pt-fid-eval "${GRES[@]}" \
  --cpus-per-task="${CPUS:-8}" --mem="${MEMORY:-128G}" --time="$WALLTIME" \
  --array="0-$((COUNT - 1))%$MAX_JOBS" \
  --output="$OUT/logs/%A_%a.out" --error="$OUT/logs/%A_%a.err" \
  "$OUT/worker.sh" "$OUT")
echo "$ARRAY" > "$OUT/job_id.txt"

# --- 2. master-CSV + per-run report, after ALL rows finish --------------------
MASTER_JOB=$(sbatch "${COMMON[@]}" --job-name=fid-master --dependency="afterany:$ARRAY" \
  --cpus-per-task=2 --mem=4G --time=00:15:00 \
  --output="$OUT/logs/fid_master_%j.out" \
  --wrap "'$PT_ENV/bin/python' '$APPEND' --root '$OUT' --master '$MASTER' \
          --gpu-type '$GPU_LABEL' --stamp '$STAMP' --run-csv '$OUT/fid_summary.csv'")

echo "Repo        : $REPO"
echo "Sweep array : $ARRAY  ($COUNT cfg x seed, $FID_SAMPLES samples, GPU=$GPU_LABEL x1, $MAX_JOBS at a time)"
echo "Master job  : $MASTER_JOB  (afterany:$ARRAY)"
echo "Per-run dir : $OUT"
echo "Per-run CSV : $OUT/fid_summary.csv   (this sweep's 12 rows)"
echo "Master CSV  : $MASTER                (cumulative, updated after the array)"
squeue --me
