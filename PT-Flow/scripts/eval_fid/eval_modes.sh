#!/bin/bash
# FID across the three PT-Flow sampling modes, from one checkpoint.
#
# This is the experiment the method is for.  Mode A is the baseline 1-NFE sampler
# and should reproduce its FID.  Modes B and C spend extra NFEs to close the
# O(sqrt(eps d)) gap between the deterministic map and the model marginal
# (Proposition 2.4).  If B and C do not beat A, the potential has not learned a
# useful prox and the run's pt/ess history is the first thing to look at.

set -euo pipefail

NGPU=${NGPU:-8}
CKPT=${CKPT:?set CKPT to a PT-Flow state_*.pt}
CONFIG=${CONFIG:-configs/gen/ptflow_XL.yaml}
CFG=${CFG:-1.2}
NUM_SAMPLES=${NUM_SAMPLES:-50000}
EXP_NAME=$(basename "$(dirname "$(dirname "$CKPT")")")
STEPNUM=$(basename "$CKPT" .pt | sed 's/state_//')

OUTDIR=results/$EXP_NAME/ckpt_$STEPNUM
mkdir -p "$OUTDIR"

run_mode () {
    local tag=$1; shift
    local workdir="runs/$EXP_NAME/ckpt_$STEPNUM/$tag"
    mkdir -p "$workdir"
    echo ""
    echo "=== $tag ==="
    NCCL_DEBUG=WARN torchrun --nproc_per_node="$NGPU" --master_port=6667 \
        inference.py evaluate \
        --ckpt "$CKPT" --config "$CONFIG" --cfg-scale "$CFG" \
        --num-samples "$NUM_SAMPLES" --gen-bsz 64 \
        --workdir "$workdir" --json-out "$OUTDIR/$tag.json" "$@"
    cat "$OUTDIR/$tag.json"
}

run_mode "modeA_1nfe"        --sampler A
run_mode "modeB_4nfe"        --sampler B --refine-steps 4  --refine-gamma 0.5
run_mode "modeB_8nfe"        --sampler B --refine-steps 8  --refine-gamma 0.5
run_mode "modeC_snis_k32"    --sampler C --snis-k 32

echo ""
echo "All modes written to $OUTDIR"
