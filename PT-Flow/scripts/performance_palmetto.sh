#!/usr/bin/env bash
# Submit measurements/candidates from an EXISTING environment and saved run.
# bash scripts/performance_palmetto.sh benchmark CONFIG CKPT OUTDIR
# bash scripts/performance_palmetto.sh eval      CONFIG CKPT OUTDIR
# bash scripts/performance_palmetto.sh finetune  CONFIG CKPT OUTDIR
set -Eeuo pipefail

if [[ $# != 4 || ! "$1" =~ ^(benchmark|eval|finetune)$ ]]; then
  echo "Usage: bash $0 {benchmark|eval|finetune} CONFIG CKPT NEW_OUTDIR" >&2
  exit 2
fi
MODE=$1
BASE=$(realpath "$2")
CKPT=$(realpath "$3")
OUT=$(realpath -m "$4")
REPO=$(cd "$(dirname "$0")/.." && pwd)
PT_ENV=${CONDA_PREFIX:-$(python -c 'import sys; print(sys.prefix)')}
[[ -s "$BASE" && -s "$CKPT" && -x "$PT_ENV/bin/python" ]] || { echo "Missing config/checkpoint or active Python environment" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Use a new output directory: $OUT already exists" >&2; exit 1; }
mkdir -p "$OUT"/logs "$OUT"/results "$OUT"/configs
cp "$BASE" "$OUT/base.yaml"
BASE="$OUT/base.yaml"

export PTFLOW_ASSETS=${PTFLOW_ASSETS:-/scratch/$USER/ptflow/assets}
export PTFLOW_DATA=${PTFLOW_DATA:-/scratch/$USER/ptflow/data}
export TORCH_HUB_DIR=${TORCH_HUB_DIR:-$PTFLOW_ASSETS/torch_hub}
export HF_ROOT=${HF_ROOT:-$PTFLOW_ASSETS/mae}
export VAE_HF_PATH=${VAE_HF_PATH:-$PTFLOW_ASSETS/sdvae}
export IMAGENET_CACHE_PATH=${IMAGENET_CACHE_PATH:-$PTFLOW_DATA/latents}
export CIFAR10_FID_NPZ=${CIFAR10_FID_NPZ:-$PTFLOW_ASSETS/fid_stats/cifar10_train_fid_stats.npz}
export IMAGENET_FID_NPZ=${IMAGENET_FID_NPZ:-$PTFLOW_ASSETS/fid_stats/jit_in256_stats.npz}

NPROC=${NPROC:-8}
PARTITION=${PALMETTO_PARTITION:-work1}
GPU_TYPE=${GPU_TYPE:-h100}
STEPS=${STEPS:-5000}
BENCHMARK_STEPS=${BENCHMARK_STEPS:-40}
FID_SAMPLES=${FID_SAMPLES:-10000}
GEN_BSZ=${GEN_BSZ:-64}
COMPILE=${COMPILE:-0}
KEEP_PT=${KEEP_PT:-0}
EMA_DECAYS=${EMA_DECAYS:-primary}
CFG_VALUES=${CFG_VALUES:-"1.0 1.05 1.1 1.15 1.2 1.3 1.5 1.8 2.0"}
SEEDS=${SEEDS:-42}

# Paths and settings are shell-quoted, not interpolated into executable code.
for key in MODE BASE CKPT OUT REPO PT_ENV NPROC STEPS BENCHMARK_STEPS FID_SAMPLES GEN_BSZ COMPILE KEEP_PT; do
  printf '%s=%q\n' "$key" "${!key}"
done > "$OUT/settings.sh"

if [[ $MODE == benchmark ]]; then
  # Each row requires its own NPROC-GPU allocation. The default cap avoids
  # asking for several 8-H100 nodes just to obtain short measurements.
  printf '%s\n' original preserve quality balanced speed > "$OUT/manifest.tsv"
  JOB_GPUS=$NPROC
  MAX_JOBS=${MAX_JOBS:-1}
  WALLTIME=${WALLTIME:-02:00:00}
elif [[ $MODE == finetune ]]; then
  printf '%s\n' quality balanced speed > "$OUT/manifest.tsv"
  JOB_GPUS=$NPROC
  MAX_JOBS=${MAX_JOBS:-2}
  WALLTIME=${WALLTIME:-24:00:00}
else
  for seed in $SEEDS; do
    for ema in $EMA_DECAYS; do
      for cfg in $CFG_VALUES; do
        printf '%s\t%s\t%s\n' "$cfg" "$seed" "$ema"
      done
    done
  done > "$OUT/manifest.tsv"
  JOB_GPUS=1
  MAX_JOBS=${MAX_JOBS:-8}
  WALLTIME=${WALLTIME:-04:00:00}
fi

cat > "$OUT/worker.sh" <<'WORKER'
#!/usr/bin/env bash
set -Eeuo pipefail
source "$1/settings.sh"
export PATH="$PT_ENV/bin:$PATH"
export PYTHONUNBUFFERED=1
# Limit CPU oversubscription when all GPUs run a process.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export DRIFT_COMPILE=0
cd "$REPO"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$OUT/manifest.tsv")
[[ -n "$LINE" ]] || { echo "Missing manifest row" >&2; exit 1; }
if [[ $MODE == eval ]]; then
  IFS=$'\t' read -r CFG SEED EMA <<< "$LINE"
  TAG="cfg${CFG}_seed${SEED}_ema${EMA}"
  EXTRA=()
  [[ $EMA == primary ]] || EXTRA+=(--ema-decay "$EMA")
  srun --ntasks=1 python inference.py evaluate --ckpt "$CKPT" --config "$BASE" \
    --sampler A --cfg-scale "$CFG" --seed "$SEED" --num-samples "$FID_SAMPLES" \
    --gen-bsz "$GEN_BSZ" --workdir "$OUT/eval/$TAG" --json-out "$OUT/results/$TAG.json" \
    --eval-backend streaming "${EXTRA[@]}"
else
  RECIPE=$LINE
  CONFIG="$OUT/configs/$RECIPE.yaml"
  EXTRA=()
  [[ $COMPILE == 0 ]] || EXTRA+=(--compile)
  [[ $KEEP_PT == 0 ]] || EXTRA+=(--keep-pt)
  if [[ $MODE == benchmark ]]; then
    if [[ $RECIPE == original ]]; then
      # Original numerical/training choices on the updated engine, same ckpt.
      python - "$BASE" "$CONFIG" "$BENCHMARK_STEPS" <<'PY'
import sys, yaml
from pathlib import Path
c = yaml.safe_load(Path(sys.argv[1]).read_text())
for k in ('resume_from', 'init_ema_from', 'init_from'):
    c['train'].pop(k, None)
c['train'].update(benchmark_steps=int(sys.argv[3]), eval_per_step=0, compile_generator=False)
c['train'].setdefault('ot_kwargs', {})['reuse_costs'] = False
c['optimizer']['fused'] = False
c.setdefault('logging', {}).update(use_wandb=False, log_every_k=20)
Path(sys.argv[2]).write_text(yaml.safe_dump(c, sort_keys=False))
PY
    else
      python -m scripts.make_performance_config --base "$BASE" --recipe "$RECIPE" \
        --world-size "$NPROC" --benchmark "$BENCHMARK_STEPS" --output "$CONFIG" "${EXTRA[@]}"
    fi
    srun --ntasks=1 torchrun --standalone --nproc_per_node="$NPROC" train.py \
      --config "$CONFIG" --resume "$CKPT" --workdir "$OUT/train/$RECIPE"
  else
    python -m scripts.make_performance_config --base "$BASE" --recipe "$RECIPE" \
      --world-size "$NPROC" --steps "$STEPS" --finetune --output "$CONFIG" "${EXTRA[@]}"
    srun --ntasks=1 torchrun --standalone --nproc_per_node="$NPROC" train.py \
      --config "$CONFIG" --init-ema "$CKPT" --workdir "$OUT/train/$RECIPE"
    FINAL="$OUT/train/$RECIPE/checkpoints/$(printf 'state_%08d.pt' "$STEPS")"
    test -s "$FINAL"
    # Matched quick screen; run the parallel eval command on finalists later.
    for CFG in 1.0 1.2 1.5; do
      srun --ntasks=1 torchrun --standalone --nproc_per_node="$NPROC" inference.py evaluate \
        --ckpt "$FINAL" --config "$CONFIG" --sampler A --cfg-scale "$CFG" --seed 42 \
        --num-samples "$FID_SAMPLES" --gen-bsz "$GEN_BSZ" \
        --workdir "$OUT/eval/${RECIPE}_cfg$CFG" --json-out "$OUT/results/${RECIPE}_cfg$CFG.json"
    done
  fi
fi
touch "$OUT/results/done_${SLURM_ARRAY_TASK_ID}"
WORKER

COUNT=$(wc -l < "$OUT/manifest.tsv")
COMMON=(--parsable --partition="$PARTITION" --nodes=1 --ntasks=1 --export=ALL)
[[ -z ${PALMETTO_ACCOUNT:-} ]] || COMMON+=(--account="$PALMETTO_ACCOUNT")
JOB_RAW=$(sbatch "${COMMON[@]}" --job-name="pt-$MODE" --gpus="$GPU_TYPE:$JOB_GPUS" \
  --cpus-per-task="${CPUS:-$((JOB_GPUS * 8))}" --mem="${MEMORY:-128G}" --time="$WALLTIME" \
  --array="0-$((COUNT - 1))%$MAX_JOBS" --output="$OUT/logs/%A_%a.out" --error="$OUT/logs/%A_%a.err" \
  "$OUT/worker.sh" "$OUT")
JOB=$(printf '%s\n' "$JOB_RAW" | tail -n1 | cut -d';' -f1)
[[ $JOB =~ ^[0-9]+$ ]] || { echo "Cannot parse sbatch output: $JOB_RAW" >&2; exit 1; }
printf '%s\n' "$JOB" > "$OUT/job_id.txt"

cat > "$OUT/report.sh" <<'REPORT'
#!/usr/bin/env bash
set -Eeuo pipefail
source "$1/settings.sh"
export PATH="$PT_ENV/bin:$PATH"
cd "$REPO"
srun --ntasks=1 python -m scripts.collect_performance --root "$OUT"
REPORT
sbatch "${COMMON[@]}" --job-name=pt-report --dependency="afterany:$JOB" \
  --cpus-per-task=2 --mem=4G --time=00:10:00 --output="$OUT/logs/report_%j.out" \
  "$OUT/report.sh" "$OUT"
echo "Submitted array $JOB ($COUNT experiments, maximum $MAX_JOBS concurrent allocations)."
echo "Each allocation: $JOB_GPUS $GPU_TYPE GPUs. Report: $OUT/report/REPORT.md"
