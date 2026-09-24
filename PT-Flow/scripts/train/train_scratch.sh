#!/bin/bash
# PT-Flow from scratch, single node.  See configs/gen/ptflow_scratch.yaml.
#
# Watch pt/ess in the logs from step 0.  It should sit near 1.0 while the
# generator is still close to the identity, then settle somewhere above 0.3 as
# the OT drift pulls it away.  If it falls below 0.05 the schedule freezes the
# potential and holds lambda_prox at 0 -- the run degrades to the baseline rather than
# diverging, which is the intended failure mode, but it is worth investigating.

set -euo pipefail

NGPU=${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}
NGPU=${NGPU:-1}

MASTER_PORT=${MASTER_PORT:-6667}

CONFIG=configs/gen/ptflow_scratch.yaml
EXP_NAME=ptflow_ablation_1node

WORKDIR=/path/to/workdir/$EXP_NAME
WANDB_PROJECT=YOUR_WANDB_PROJECT
WANDB_NAME=$EXP_NAME

DRIFT_COMPILE=1 \
NCCL_DEBUG=WARN \
WANDB_PROJECT=$WANDB_PROJECT \
WANDB_NAME=$WANDB_NAME \
torchrun \
    --nproc_per_node="$NGPU" \
    --master_port="$MASTER_PORT" \
    train.py \
    --config "$CONFIG" \
    --workdir "$WORKDIR"

echo "finished!"
