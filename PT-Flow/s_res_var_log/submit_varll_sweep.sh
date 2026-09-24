#!/usr/bin/env bash
# Guided variance + normalized per-sample likelihood across 16 cfg values,
# one SLURM array task per cfg (16 a100 at a time, %16), merged into one CSV.
#
#   bash s_res_var_log/submit_varll_sweep.sh
#
# Each task runs `python -m s_res_var_log.infer varll --cfg-scale <v>` on the
# CURRENT repo (the one that holds this script), writing results/varll_cfg<v>.json.
# A dependent job merges the 16 into a per-run CSV and a cumulative master CSV.
#
# NOTE: like the FID wrapper, this ignores ~/ptflow_env.sh's exported REPO/RUN --
# REPO is forced to this script's repo, run/ckpt use RUN_DIR/CONFIG_PATH/CKPT_PATH.
set -Eeuo pipefail

source ~/ptflow_env.sh

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../s_res_var_log
MERGE="$HERE/merge_varll.py"
REPO="$(cd "$HERE/.." && pwd)"
test -s "$MERGE"             || { echo "MISSING $MERGE" >&2; exit 1; }
test -s "$REPO/inference.py" || { echo "MISSING inference.py in repo: $REPO" >&2; exit 1; }
grep -q eval-backend "$REPO/inference.py" || {
  echo "This repo copy is the OLD one (no streaming/current API): $REPO" >&2
  echo "Run from speed_quality_v1/PT-Flow instead." >&2; exit 1; }

export WORK="${WORK:-/scratch/$USER/ptflow}"
RUN_DIR="${RUN_DIR:-$WORK/runs/in256_L_warmstart}"
export PTFLOW_ASSETS="${PTFLOW_ASSETS:-$WORK/assets}"
export PTFLOW_DATA="${PTFLOW_DATA:-$WORK/data}"
export TORCH_HUB_DIR="${TORCH_HUB_DIR:-$PTFLOW_ASSETS/torch_hub}"
export HF_ROOT="${HF_ROOT:-$PTFLOW_ASSETS/mae}"
export VAE_HF_PATH="${VAE_HF_PATH:-$PTFLOW_ASSETS/sdvae}"
export IMAGENET_CACHE_PATH="${IMAGENET_CACHE_PATH:-$PTFLOW_DATA/latents}"
export IMAGENET_FID_NPZ="${IMAGENET_FID_NPZ:-$PTFLOW_ASSETS/fid_stats/jit_in256_stats.npz}"

# --- knobs --------------------------------------------------------------------
GPU_TYPE="${GPU_TYPE:-a100}"
# 16 cfg = the dense 12 (0.0..2.2) plus 1.1, 1.15, 1.25, 1.3.
CFG_VALUES="${CFG_VALUES:-0.0 0.2 0.4 0.6 0.8 1.0 1.1 1.15 1.2 1.25 1.3 1.4 1.6 1.8 2.0 2.2}"
MAX_JOBS="${MAX_JOBS:-16}"                # 16 at a time (%16)
SEED="${SEED:-42}"
LL_BATCHES="${LL_BATCHES:-8}"             # held-out batches for the likelihood
LL_BSZ="${LL_BSZ:-8}"
K_INNER="${K_INNER:-16}"
K_OUTER="${K_OUTER:-16}"
VAR_K="${VAR_K:-64}"                      # proposal draws for the variance chi^2
CURV_PROBES="${CURV_PROBES:-8}"          # Hutchinson HVP probes for the (A6) term
ALPHA_DEF="${ALPHA_DEF:-0.05}"
PERSAMPLE="${PERSAMPLE:-0}"              # 1 = also dump per-sample log-likelihoods
WALLTIME="${WALLTIME:-04:00:00}"
CONFIG="${CONFIG_PATH:-$RUN_DIR/config.json}"
CKPT="${CKPT_PATH:-$RUN_DIR/checkpoints/state_00014000.pt}"
MASTER="${MASTER:-$WORK/results/varll_master.csv}"
PARTITION="${PALMETTO_PARTITION:-work1}"
PT_ENV="${CONDA_PREFIX:-$(python -c 'import sys; print(sys.prefix)')}"

echo "Repo    : $REPO"
echo "Config  : $CONFIG"
echo "Ckpt    : $CKPT"
for f in "$CONFIG" "$CKPT"; do
  [[ -s "$f" ]] || { echo "MISSING: $f  (override RUN_DIR=... or CONFIG_PATH=.../CKPT_PATH=...)" >&2; exit 1; }
done

if [[ -z "$GPU_TYPE" || "$GPU_TYPE" == "any" ]]; then GPU_LABEL=any; GRES=(--gres=gpu:1)
else GPU_LABEL="$GPU_TYPE"; GRES=(--gpus="$GPU_TYPE:1"); fi

STAMP=$(date +%Y%m%d_%H%M%S)
OUT="$WORK/results/in256_L_step14k_varll_16cfg_${GPU_LABEL}_${STAMP}"
mkdir -p "$OUT"/logs "$OUT"/results

for key in REPO PT_ENV CKPT CONFIG SEED LL_BATCHES LL_BSZ K_INNER K_OUTER VAR_K \
           CURV_PROBES ALPHA_DEF PERSAMPLE OUT PTFLOW_ASSETS PTFLOW_DATA \
           TORCH_HUB_DIR HF_ROOT VAE_HF_PATH IMAGENET_CACHE_PATH IMAGENET_FID_NPZ; do
  printf '%s=%q\n' "$key" "${!key}"
done > "$OUT/settings.sh"

printf '%s\n' $CFG_VALUES > "$OUT/manifest.tsv"   # one cfg per line

cat > "$OUT/worker.sh" <<'WORKER'
#!/usr/bin/env bash
set -Eeuo pipefail
source "$1/settings.sh"
export PATH="$PT_ENV/bin:$PATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
cd "$REPO"
CFG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1/manifest.tsv")
[[ -n "$CFG" ]] || { echo "missing manifest row $SLURM_ARRAY_TASK_ID" >&2; exit 1; }
PS=()
[[ "$PERSAMPLE" == 1 ]] && PS=(--persample-csv "$1/results/persample_cfg${CFG}.csv")
srun --ntasks=1 python -m s_res_var_log.infer varll \
  --ckpt "$CKPT" --config "$CONFIG" --cfg-scale "$CFG" --seed "$SEED" \
  --num-batches "$LL_BATCHES" --bsz "$LL_BSZ" \
  --k-inner "$K_INNER" --k-outer "$K_OUTER" --var-K "$VAR_K" \
  --curvature-probes "$CURV_PROBES" --alpha-def "$ALPHA_DEF" \
  --json-out "$1/results/varll_cfg${CFG}.json" \
  --csv-out  "$1/results/varll_cfg${CFG}.csv" "${PS[@]}"
touch "$1/results/done_${SLURM_ARRAY_TASK_ID}"
WORKER

COUNT=$(wc -l < "$OUT/manifest.tsv")
COMMON=(--parsable --partition="$PARTITION" --nodes=1 --ntasks=1 --export=ALL)
[[ -z "${PALMETTO_ACCOUNT:-}" ]] || COMMON+=(--account="$PALMETTO_ACCOUNT")

ARRAY=$(sbatch "${COMMON[@]}" --job-name=pt-varll "${GRES[@]}" \
  --cpus-per-task="${CPUS:-8}" --mem="${MEMORY:-128G}" --time="$WALLTIME" \
  --array="0-$((COUNT - 1))%$MAX_JOBS" \
  --output="$OUT/logs/%A_%a.out" --error="$OUT/logs/%A_%a.err" \
  "$OUT/worker.sh" "$OUT")
echo "$ARRAY" > "$OUT/job_id.txt"

MERGE_JOB=$(sbatch "${COMMON[@]}" --job-name=varll-merge --dependency="afterany:$ARRAY" \
  --cpus-per-task=2 --mem=4G --time=00:15:00 \
  --output="$OUT/logs/varll_merge_%j.out" \
  --wrap "'$PT_ENV/bin/python' '$MERGE' --root '$OUT' --master '$MASTER' --run-csv '$OUT/varll_summary.csv'")

echo "Repo        : $REPO"
echo "varll array : $ARRAY  ($COUNT cfg, GPU=$GPU_LABEL x1, $MAX_JOBS at a time)"
echo "Merge job   : $MERGE_JOB  (afterany:$ARRAY)"
echo "Per-run dir : $OUT"
echo "Per-run CSV : $OUT/varll_summary.csv   (16 rows: cfg -> NLL/dim + variance)"
echo "Master CSV  : $MASTER"
squeue --me
