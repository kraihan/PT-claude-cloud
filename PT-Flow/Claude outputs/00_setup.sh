#!/bin/bash
# ============================================================================
# PT-Flow ablation — ONE-TIME SETUP on Palmetto 2 (Clemson)
# Run this once on a login node. It creates the env, clones the repo, and
# downloads the CIFAR-10 data + FID stats + ConvNeXt features the ablations need.
#
#   bash 00_setup.sh
#
# Ablations run on CIFAR-10 (fast: ~0.55 s/it, and its FID reference stats can be
# built locally). ImageNet is NOT used for the sweeps: at 4-5k iters it would be
# ~18h/run AND has no public FID-stats file. Confirm the winning params on
# ImageNet separately afterwards.
# ============================================================================
set -euo pipefail

# ---- EDIT IF NEEDED --------------------------------------------------------
REPO_URL="https://github.com/kraihan/PT-Flow.git"
WORK="/scratch/$USER/ptflow_abl"          # your scratch workspace
# ---------------------------------------------------------------------------

# 1) env file every job will source
cat > ~/ptflow_abl_env.sh <<EOF
module load anaconda3
module load cuda
source activate ptflow_abl
export WORK=$WORK
export REPO=\$WORK/PT-Flow
export PTFLOW_ASSETS=\$WORK/assets
export PTFLOW_DATA=\$WORK/data
export CIFAR10_PATH=\$WORK/data/cifar10
export CIFAR10_FID_NPZ=\$WORK/assets/fid_stats/cifar10_train_fid_stats.npz
export DRIFT_COMPILE=0
export WANDB_MODE=offline          # no wandb login needed; FID comes from JSON
export OMP_NUM_THREADS=8
EOF
source ~/ptflow_abl_env.sh

# 2) conda env + repo + deps
mkdir -p "$WORK" && cd "$WORK"
module load anaconda3
conda create -y -n ptflow_abl python=3.11 || true
source activate ptflow_abl
[ -d "$REPO" ] || git clone "$REPO_URL" PT-Flow
cd "$REPO"
pip install -r requirements.txt
pip install gdown                  # CIFAR downloader dependency
python -m pytest -q || echo "WARN: some CPU tests failed — inspect before long runs"

# 3) CIFAR data + FID stats + ConvNeXt, as a short GPU job
mkdir -p "$WORK/logs"
cat > "$REPO/abl_setup.slurm" <<'EOF'
#!/bin/bash
#SBATCH --job-name abl-setup
#SBATCH --gpus a100:1
#SBATCH --cpus-per-task 8
#SBATCH --mem 64gb
#SBATCH --time 01:00:00
#SBATCH --output ABLLOGS/%x_%j.out
set -e
source ~/ptflow_abl_env.sh
cd $REPO
python -m misc.download_cifar10
python -m scripts.make_cifar10_fid_stats --batch-size 64
# base ablation config (compressed schedule so PT actually activates within ~5k)
python -m scripts.make_run_config --profile cifar10_t4 --output configs/gen/abl_cifar_base.yaml
python - <<'PY'
import yaml
p='configs/gen/abl_cifar_base.yaml'; c=yaml.safe_load(open(p))
c['logging']['use_wandb']=False
c['train'].update(total_steps=5000, save_per_step=5000, keep_every=5000, eval_per_step=0)
c['optimizer']['lr_schedule']['total_steps']=5000
c['pt']['optimizer']['lr_schedule']['total_steps']=5000
c['pt']['schedule'].update(eps_warmup=200, eps_anneal_steps=3000,
                           prox_warmup=500, prox_ramp=1500)   # PT full by ~2k of 5k
yaml.safe_dump(c,open(p,'w'),sort_keys=False)
print('wrote', p)
PY
python -m scripts.preflight --config configs/gen/abl_cifar_base.yaml \
  --world-size 2 --check-assets --check-features
echo "ABL SETUP OK"
EOF
sed -i "s#ABLLOGS#$WORK/logs#" "$REPO/abl_setup.slurm"

echo ""
echo ">>> Submitting setup job. When it prints 'ABL SETUP OK', run 01_run_ablations.sh"
sbatch "$REPO/abl_setup.slurm"
squeue -u "$USER"
