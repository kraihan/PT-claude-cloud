#!/bin/bash
# Resume a released baseline checkpoint under PT-Flow and keep training.
#
#   SIZE=XL bash scripts/train/resume_baseline.sh
#   SIZE=L  bash scripts/train/resume_baseline.sh
#   SIZE=B  bash scripts/train/resume_baseline.sh
#
# What this does: copies the baseline state_*.pt into a fresh workdir's
# checkpoints/ directory and starts train.py.  restore_checkpoint picks it up as
# an ordinary resume -- generator, EMA, AdamW moments and step count all come
# across, because DitGen is shape-identical to the release (enforced by
# tests/test_ckpt_compat.py).  The PT-Flow potential and scale net start fresh
# at that step.
#
# The run therefore continues as *exactly the baseline* for the first
# ptflow.schedule.prox_warmup steps (lambda_prox = 0), while the potential learns to
# match the generator it inherited.  Only after that does the PT-Flow term begin
# to influence the generator, and only while ESS certifies the estimator.
#
# So: FID at the resume point should equal the checkpoint's published FID.  If
# it does not, stop -- something is wrong with the handoff, not with PT-Flow.

set -euo pipefail

SIZE=${SIZE:-XL}
NGPU=${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}
NGPU=${NGPU:-1}
MASTER_PORT=${MASTER_PORT:-6667}

BASELINE_HF_ROOT=${BASELINE_HF_ROOT:-/path/to/baseline_hf_root}

# Checkpoint directory + step come from utils/env.py:BASELINE_CKPTS, so the
# upstream identifiers live in exactly one place.
read -r CKPT_DIR STEP <<< "$(python -c "
import sys
from utils.env import BASELINE_CKPTS
d = BASELINE_CKPTS.get(sys.argv[1])
if d is None:
    sys.exit('SIZE must be one of ' + ', '.join(BASELINE_CKPTS))
print(d[0], d[1])
" "$SIZE")" || exit 1

CONFIG=${CONFIG:-configs/gen/ptflow_${SIZE}.yaml}
SRC_CKPT=${SRC_CKPT:-$BASELINE_HF_ROOT/checkpoints/$CKPT_DIR/state_$STEP.pt}
EXP_NAME=ptflow_from_baseline_${SIZE}_$STEP
WORKDIR=${WORKDIR:-/path/to/workdir/$EXP_NAME}
WANDB_PROJECT=${WANDB_PROJECT:-YOUR_WANDB_PROJECT}

if [ ! -f "$SRC_CKPT" ]; then
    echo "baseline checkpoint not found: $SRC_CKPT" >&2
    echo "Set BASELINE_HF_ROOT (or SRC_CKPT) and run: python -m misc.download_pretrained" >&2
    exit 1
fi

mkdir -p "$WORKDIR/checkpoints"

# Preflight: verify shape compatibility BEFORE claiming any GPUs.  A mismatch
# caught here costs seconds; caught at step 0 of a multi-node job it does not.
python - "$CONFIG" "$SRC_CKPT" <<'PY'
import sys
from models.generator import DitGen
from ptflow.convert import check_resume_compatible, inspect_checkpoint
from utils.misc import load_config

cfg_path, ckpt = sys.argv[1:3]
cfg = load_config(cfg_path)
mcfg = dict(cfg.model)
if mcfg.pop("residual", False):
    raise SystemExit(
        "model.residual must be false when resuming the OT-drift baseline weights: the release "
        "was trained as m(x) = net(x), not x + net(x)."
    )
gen = DitGen(num_classes=int(cfg.dataset.num_classes), **mcfg)
info = inspect_checkpoint(ckpt)
print(f"  checkpoint: {info['kind']} @ step {info['step']}, "
      f"{info['generator_tensors']} tensors, {info['generator_params']:,} params")
check_resume_compatible(gen, ckpt, strict=True)
print("  preflight OK -- safe to resume")
PY

# Only copy if the workdir is empty, so a restarted job resumes its OWN latest
# checkpoint rather than rewinding to the baseline one.
if ! ls "$WORKDIR"/checkpoints/state_*.pt >/dev/null 2>&1; then
    echo "Seeding $WORKDIR/checkpoints/ with $(basename "$SRC_CKPT")"
    cp "$SRC_CKPT" "$WORKDIR/checkpoints/"
else
    echo "Workdir already has checkpoints; resuming the latest one:"
    ls -1 "$WORKDIR"/checkpoints/state_*.pt | tail -1
fi

DRIFT_COMPILE=1 \
NCCL_DEBUG=WARN \
WANDB_PROJECT=$WANDB_PROJECT \
WANDB_NAME=$EXP_NAME \
torchrun \
    --nproc_per_node="$NGPU" \
    --master_port="$MASTER_PORT" \
    train.py \
    --config "$CONFIG" \
    --workdir "$WORKDIR"

echo "finished!"
