#!/usr/bin/env bash
# One fresh isolated 30k training run. Never edit the production checkout.
set -Eeuo pipefail
source "$HOME/ptflow_env.sh"
KIT=$(cd "$(dirname "$0")" && pwd)
WORK="/scratch/$USER/ptflow"
BASE="$WORK/PT-Flow/configs/gen/ptflow_B_warmstart_180k.yaml"
PRETRAINED="$WORK/baseline_wflow/checkpoints/latent_sota_B_ot/state_00200000.pt"
test -s "$BASE"
test -s "$PRETRAINED"
STAMP=$(date +%Y%m%d_%H%M%S)
ACTIVE_HOME="$WORK/active_pt/B_H200x2_30k_${STAMP}"
python "$KIT/prepare.py" --base "$BASE" --initializer "$PRETRAINED" --destination "$ACTIVE_HOME"
PT_ACTIVE_REPO="$ACTIVE_HOME/PT-Flow"
PT_ACTIVE_CFG="$ACTIVE_HOME/active_B_30k.yaml"
PT_ACTIVE_RUN="$ACTIVE_HOME/run"
mkdir -p "$ACTIVE_HOME/logs" "$PT_ACTIVE_RUN"
# Preserve explicit user environment paths; default to existing main assets/cache.
export PTFLOW_ASSETS=${PTFLOW_ASSETS:-$WORK/assets}
export PTFLOW_DATA=${PTFLOW_DATA:-$WORK/data}
export IMAGENET_CACHE_PATH=${IMAGENET_CACHE_PATH:-$PTFLOW_DATA/latents}
export VAE_HF_PATH=${VAE_HF_PATH:-$PTFLOW_ASSETS/sdvae}
export HF_ROOT=${HF_ROOT:-$PTFLOW_ASSETS/mae}
export TORCH_HUB_DIR=${TORCH_HUB_DIR:-$PTFLOW_ASSETS/torch_hub}
export IMAGENET_FID_NPZ=${IMAGENET_FID_NPZ:-$PTFLOW_ASSETS/fid_stats/jit_in256_stats.npz}
for key in PT_ACTIVE_REPO PT_ACTIVE_CFG PT_ACTIVE_RUN PTFLOW_ASSETS PTFLOW_DATA IMAGENET_CACHE_PATH VAE_HF_PATH HF_ROOT TORCH_HUB_DIR IMAGENET_FID_NPZ; do
    printf 'export %s=%q\n' "$key" "${!key}"
done > "$ACTIVE_HOME/settings.sh"
cp "$KIT/worker.sbatch" "$ACTIVE_HOME/worker.sbatch"
RESPONSE=$(sbatch --parsable --chdir="$PT_ACTIVE_REPO" \
    --output="$ACTIVE_HOME/logs/train_%j.out" \
    --error="$ACTIVE_HOME/logs/train_%j.err" \
    "$ACTIVE_HOME/worker.sbatch" "$ACTIVE_HOME")
JOB_ID=${RESPONSE%%;*}
[[ "$JOB_ID" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $RESPONSE" >&2; exit 1; }
printf '%s\n' "$JOB_ID" > "$ACTIVE_HOME/job_id.txt"
mkdir -p "$WORK/active_pt"
printf '%s\n' "$ACTIVE_HOME" > "$WORK/active_pt/latest_B.txt"
echo "Submitted $JOB_ID"
echo "Weights-only initializer: $PRETRAINED"
echo "NEW step zero -> 30,000; potential/scale/optimizers fresh"
echo "Run folder: $PT_ACTIVE_RUN"
echo "Log: tail -F $ACTIVE_HOME/logs/train_${JOB_ID}.out"
echo "Status: cat $PT_ACTIVE_RUN/pt_status.json"
echo "Checkpoints: $PT_ACTIVE_RUN/checkpoints/state_XXXXXXXX.pt"
squeue -j "$JOB_ID" -o '%.16i %.30j %.3t %.12M %R'
