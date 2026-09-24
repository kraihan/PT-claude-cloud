#!/bin/bash
# ============================================================================
# PT-Flow ablation — RUN ALL SWEEPS (submit as SLURM jobs)
# Run AFTER 00_setup.sh printed "ABL SETUP OK".
#
#   bash 01_run_ablations.sh
#
# Every sweep trains/evaluates on CIFAR-10 at total_steps=5000 (fast) and writes
# one JSON per configuration into $WORK/results/. Then run 02_collect_report.py
# to turn them into a table.
#
# Sweeps (the mandatory set):
#   lambda   pt.schedule.lambda_prox_max ∈ {0, 0.05, 0.1, 0.2, 0.5}   (λ=0 == PT-OFF baseline)
#   K        pt.K (training)            ∈ {8, 16, 32, 64}
#   modeB    inference --refine-steps n × --refine-gamma γ  (Prop 2.4 / C3)   [inference-only]
#   cfg      inference --cfg-scale w    ∈ {0,0.5,1,1.2,1.5,2,3}              [inference-only]
#   Klad     inference likelihood K-ladder                                    [inference-only]
#   estim    examples/toy2d.py: ESS vs eps, PT-on/off  (C1/C2 direction, 2-D)
# ============================================================================
set -euo pipefail
source ~/ptflow_abl_env.sh
cd "$REPO"

GPUS=${GPUS:-a100:2}          # CIFAR is small; 2 GPUs is plenty
NPROC=2
NUM_SAMPLES=${NUM_SAMPLES:-10000}   # screening FID; set 50000 for final numbers
BASECFG=configs/gen/abl_cifar_base.yaml
RES=$WORK/results; mkdir -p "$RES" configs/gen/abl
LOG=$WORK/logs

# ---- generic TRAIN+EVAL worker (one CIFAR config -> FID json) --------------
cat > "$REPO/abl_train_eval.slurm" <<EOF
#!/bin/bash
#SBATCH --job-name abl-te
#SBATCH --gpus $GPUS
#SBATCH --cpus-per-task 16
#SBATCH --mem 96gb
#SBATCH --time 06:00:00
#SBATCH --output $LOG/%x_%j.out
set -e
source ~/ptflow_abl_env.sh
cd \$REPO
: "\${ACFG:?}"; : "\${WD:?}"; : "\${OUT:?}"
srun torchrun --standalone --nproc_per_node=$NPROC train.py --config "\$ACFG" --workdir "\$WD"
CK=\$(ls -t "\$WD"/checkpoints/state_*.pt | head -1)
echo "eval ckpt: \$CK"
torchrun --standalone --nproc_per_node=$NPROC inference.py evaluate \
  --ckpt "\$CK" --config "\$ACFG" --sampler A --cfg-scale 1.2 \
  --num-samples $NUM_SAMPLES --gen-bsz 64 --workdir "\$WD/eval" --json-out "\$OUT"
cat "\$OUT"
EOF

# ---- generic INFERENCE-only worker (one base ckpt -> FID json) -------------
cat > "$REPO/abl_infer.slurm" <<EOF
#!/bin/bash
#SBATCH --job-name abl-inf
#SBATCH --gpus $GPUS
#SBATCH --cpus-per-task 16
#SBATCH --mem 96gb
#SBATCH --time 03:00:00
#SBATCH --output $LOG/%x_%j.out
set -e
source ~/ptflow_abl_env.sh
cd \$REPO
: "\${CK:?}"; : "\${OUT:?}"; : "\${ARGS:?}"
torchrun --standalone --nproc_per_node=$NPROC inference.py evaluate \
  --ckpt "\$CK" --config "$BASECFG" \
  --num-samples $NUM_SAMPLES --gen-bsz 64 --workdir "\$WD_INF" --json-out "\$OUT" \$ARGS
cat "\$OUT"
EOF

mkcfg () {  # mkcfg <out.yaml> <python-edits-on-object-c>
  local out=$1; shift
  python - "$out" "$*" <<'PY'
import sys,yaml
out,edit=sys.argv[1],sys.argv[2]
c=yaml.safe_load(open('configs/gen/abl_cifar_base.yaml'))
exec(edit)
yaml.safe_dump(c,open(out,'w'),sort_keys=False)
print('wrote',out)
PY
}

echo "=== 0) baseline (default) train — its checkpoint feeds the inference sweeps ==="
BASEWD=$WORK/runs/abl_base
rm -rf "$BASEWD"
JB=$(sbatch --parsable --job-name abl-base \
     --export=ALL,ACFG=$BASECFG,WD=$BASEWD,OUT=$RES/base.json $REPO/abl_train_eval.slurm)
echo "baseline job=$JB"

echo "=== 1) lambda_prox_max sweep (λ=0 is the PT-OFF headline control) ==="
for L in 0 0.05 0.1 0.2 0.5; do
  CFG=configs/gen/abl/lam_$L.yaml
  mkcfg "$CFG" "c['pt']['schedule']['lambda_prox_max']=$L"
  sbatch --job-name abl-lam$L \
    --export=ALL,ACFG=$CFG,WD=$WORK/runs/abl_lam_$L,OUT=$RES/train_lambda__lam$L.json \
    $REPO/abl_train_eval.slurm
done

echo "=== 2) training K sweep (pt.K) ==="
for K in 8 16 32 64; do
  CFG=configs/gen/abl/K_$K.yaml
  mkcfg "$CFG" "c['pt']['K']=$K"
  sbatch --job-name abl-K$K \
    --export=ALL,ACFG=$CFG,WD=$WORK/runs/abl_K_$K,OUT=$RES/train_K__K$K.json \
    $REPO/abl_train_eval.slurm
done

echo "=== 3) Mode-B refinement (inference on baseline ckpt) — waits for baseline ==="
BASECK="$BASEWD/checkpoints/state_00005000.pt"
for N in 0 1 2 4 8; do for G in 0.25 0.5 0.75; do
  sbatch --job-name abl-B_n${N}_g${G} --dependency=afterok:$JB \
    --export=ALL,CK=$BASECK,WD_INF=$WORK/runs/abl_B_n${N}_g${G},OUT=$RES/modeB__n${N}_g${G}.json,ARGS="--sampler B --refine-steps $N --refine-gamma $G" \
    $REPO/abl_infer.slurm
done; done

echo "=== 4) guidance cfg_scale sweep (inference on baseline ckpt) ==="
for W in 0 0.5 1 1.2 1.5 2 3; do
  sbatch --job-name abl-cfg$W --dependency=afterok:$JB \
    --export=ALL,CK=$BASECK,WD_INF=$WORK/runs/abl_cfg_$W,OUT=$RES/cfg__w$W.json,ARGS="--sampler A --cfg-scale $W" \
    $REPO/abl_infer.slurm
done

echo "=== 5) likelihood K-ladder (C4 diagnostic, single GPU) ==="
cat > "$REPO/abl_like.slurm" <<EOF
#!/bin/bash
#SBATCH --job-name abl-like
#SBATCH --gpus a100:1
#SBATCH --cpus-per-task 8
#SBATCH --mem 64gb
#SBATCH --time 03:00:00
#SBATCH --output $LOG/%x_%j.out
#SBATCH --dependency=afterok:$JB
set -e
source ~/ptflow_abl_env.sh
cd \$REPO
python inference.py likelihood --ckpt "$BASECK" --config "$BASECFG" \
  --num-batches 8 --bsz 8 --k-inner 16 --k-ladder "16,32,64,128,256" \
  --json-out "$RES/likelihood__ladder.json"
cat "$RES/likelihood__ladder.json"
EOF
sbatch "$REPO/abl_like.slurm"

echo "=== 6) estimator ESS-vs-eps + PT on/off (toy2d, 2-D, fast; login-node OK) ==="
mkdir -p "$RES/estimator"
for EM in 0.1 0.05 0.02 0.01 0.005; do for LP in 0.3 0; do
  python -m examples.toy2d --steps 3000 --eps-min $EM --lambda-prox $LP \
    --out "$RES/estimator/epsmin${EM}_lam${LP}" >/dev/null 2>&1 \
    && echo "  toy2d eps_min=$EM lambda=$LP -> done" || echo "  toy2d eps_min=$EM lambda=$LP -> FAILED"
done; done

echo ""
echo ">>> All ablation jobs submitted. Watch: squeue -u \$USER"
echo ">>> When they finish: python 02_collect_report.py --results $RES"
