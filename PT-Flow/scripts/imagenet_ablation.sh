#!/usr/bin/env bash
# Run from an activated ptflow environment. The terminal waits for two setup
# jobs, then submits the parallel sweep. Use tmux if disconnecting from SSH.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repo="$PWD"
python_bin="$(command -v python)"
manifest="$repo/configs/ablation/six_imagenet_a100_4k.yaml"
setup="$repo/../runs/ablation_imagenet_B_subset50_a100_4k_setup"
mkdir -p "$setup"

# Use the same partition/account as the manifest for both setup jobs.
partition="$("$python_bin" -c 'import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))["slurm"]["partition"])' "$manifest")"
account="$("$python_bin" -c 'import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))["slurm"].get("account", ""))' "$manifest")"
common=(--partition="$partition" --nodes=1 --ntasks=1 --export=ALL --chdir="$repo")
if [[ -n "$account" ]]; then common+=(--account="$account"); fi

printf -v prep_command '%q ' srun --ntasks=1 "$python_bin" "$repo/scripts/prepare_imagenet_ablation.py" prepare "$manifest" --setup-dir "$setup"
printf 'Preparing 50 cached training images per class on a CPU node. Logs: %s/prep_JOBID.log\n' "$setup"
if ! sbatch --wait "${common[@]}" --cpus-per-task=2 --mem=16G --time=02:00:00 \
  --job-name=ptin-prepare --output="$setup/prep_%j.log" --error="$setup/prep_%j.log" --wrap="$prep_command"; then
  printf 'Subset/assets preparation failed. Inspect %s/prep_*.log; no GPU sweep was submitted.\n' "$setup" >&2
  exit 1
fi

printf -v smoke_command '%q ' srun --ntasks=1 "$python_bin" "$repo/scripts/prepare_imagenet_ablation.py" smoke "$setup/manifest.yaml" --setup-dir "$setup"
printf 'Checking 20 training steps, K=16, and both samplers on one A100 40 GB.\n'
if ! sbatch --wait "${common[@]}" --gpus=a100:1 --constraint=gpu_a100_40gb \
  --cpus-per-task=8 --mem=64G --time=01:00:00 --job-name=ptin-smoke \
  --output="$setup/smoke_%j.log" --error="$setup/smoke_%j.log" --wrap="$smoke_command"; then
  printf 'GPU smoke check failed. Inspect %s/smoke_*.log and smoke_*/logs. Full sweep was NOT submitted.\n' "$setup" >&2
  exit 1
fi

"$python_bin" scripts/run_parallel_ablation.py submit "$setup/manifest.yaml" --gpu a100
printf '\nReports: %s/../runs/ablation_imagenet_B_subset50_a100_4k/report/\n' "$repo"
squeue -u "$USER" -o '%.12i %.32j %.3t %.12M %R'
