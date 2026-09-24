#!/bin/bash
#SBATCH --job-name=pt-B-active-30k
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=72:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue

set -Eeuo pipefail
source "$HOME/ptflow_env.sh"
ACTIVE_HOME=$1
source "$ACTIVE_HOME/settings.sh"
cd "$PT_ACTIVE_REPO"
export PYTHONPATH="$PT_ACTIVE_REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export DRIFT_COMPILE=0
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
echo "Job: $SLURM_JOB_ID; host: $(hostname)"
echo "Isolated code: $PT_ACTIVE_REPO"
echo "Run: $PT_ACTIVE_RUN"
echo "Global batch: 128 label groups x64 samples =8192; 2 ranks; accumulation 32"
echo "Policy: active_recovery_v1, full prox derivative, refined importance proposal"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Marker files are private to this fresh run. No checkpoint is removed.
rm -f "$PT_ACTIVE_RUN/requeue.request" "$PT_ACTIVE_RUN/requeue.ready"
request_checkpoint() {
    echo "Wall-time signal: requesting checkpoint at next optimizer boundary"
    touch "$PT_ACTIVE_RUN/requeue.request"
}
trap request_checkpoint USR1

if [[ ! -f "$ACTIVE_HOME/preflight.ok" ]]; then
    srun --ntasks=1 python scripts/preflight_active.py --config "$PT_ACTIVE_CFG"
    srun --ntasks=1 python -m pytest tests/test_active_recovery.py -q
    srun --ntasks=1 torchrun --standalone --nproc_per_node=2 scripts/smoke_active_distributed.py
    touch "$ACTIVE_HOME/preflight.ok"
fi

srun --ntasks=1 torchrun --standalone --nproc_per_node=2 train.py \
    --config "$PT_ACTIVE_CFG" --workdir "$PT_ACTIVE_RUN" &
LAUNCH_PID=$!
CODE=0
while true; do
    if wait "$LAUNCH_PID"; then CODE=0; else CODE=$?; fi
    # A caught USR1 can interrupt wait while srun is still alive.
    if kill -0 "$LAUNCH_PID" 2>/dev/null; then continue; fi
    break
done

if [[ -s "$PT_ACTIVE_RUN/requeue.ready" && ! -f "$PT_ACTIVE_RUN/PT_FAILED.txt" ]]; then
    echo "Checkpoint saved at step $(cat "$PT_ACTIVE_RUN/requeue.ready"); requeueing $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi
if (( CODE != 0 )); then
    echo "Training failed; inspect PT_FAILED.txt and stderr. No automatic retry of a failed estimator."
    exit "$CODE"
fi
test -s "$PT_ACTIVE_RUN/checkpoints/state_00030000.pt"
echo "Completed 30000 steps. Status CSV: $PT_ACTIVE_RUN/pt_status.csv"
