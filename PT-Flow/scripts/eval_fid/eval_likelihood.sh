#!/bin/bash
# Exactly-normalized per-sample likelihood (Theorem 2.5), reported as a ladder.
#
# Single GPU: the two nested Monte-Carlo layers make this expensive per sample,
# and it is a diagnostic, not a headline number.  Read the ladder, not any one
# entry: each is an IWAE-style lower bound, so only the monotone tightening is
# interpretable.  Reported at w = 0 only, and in SD-VAE latent space -- not
# comparable to pixel bits/dim.

set -euo pipefail

CKPT=${CKPT:?set CKPT to a PT-Flow state_*.pt}
CONFIG=${CONFIG:-configs/gen/ptflow_XL.yaml}
OUT=${OUT:-results/ptflow_likelihood.json}

mkdir -p "$(dirname "$OUT")"

python inference.py likelihood \
    --ckpt "$CKPT" \
    --config "$CONFIG" \
    --num-batches 8 \
    --bsz 8 \
    --k-inner 16 \
    --k-ladder "16,32,64,128,256" \
    --json-out "$OUT"

echo ""
cat "$OUT"
