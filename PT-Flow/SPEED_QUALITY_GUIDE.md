# Speed and quality experiments — 18 September 2026

This update targets the reported 13.8 s/update on 8 H100s for ImageNet L/256
and CIFAR FID 4.92 at CFG 1.2 / EMA 0.999. GPU throughput and new FID have
NOT been measured here. Sub-2 CIFAR FID and 2 s/ImageNet iteration are targets,
not results. Do not commit another full 180k run before the measurements below.

## What changed

- Optional OT cost reuse: compute pairwise costs once across temperatures;
  quadratic normalization uses moments instead of another distance matrix.
  The new transport path batches plans without replicating wide feature tensors.
- Frozen real-feature cache with a byte budget and insertion identities. Ring
  overwrites and fresh augmentations get new identities. Generated features
  always retain their gradient path. Enabled for CIFAR candidates only.
- Channels-last convolution option, fused CUDA AdamW, no checkpoint bookkeeping
  in no-gradient generator passes, and metric transfers only at log intervals.
- Independent negative-particle count, explicit generator compilation, per-phase
  CUDA timing, and benchmark JSON with the slowest-rank step time and peak memory.
- Separate full-checkpoint resume and EMA-only finetuning, both into NEW folders.
  Generator tensor shapes and the residual convention are copied from your config.
- Optional additional EMA models. CIFAR candidates track 0.999 and 0.9999;
  `inference.py --ema-decay` selects an EMA actually stored in a checkpoint.
- Distributed streaming FID: all GPUs extract Inception features, reduce float64
  statistics, and avoid 50,000 PNG writes/reads. Default evaluation returns FID.
  `--eval-backend png` retains the old FID + IS path; `--keep-samples` selects it too.
- Slurm arrays for benchmarks, guidance sweeps and short finetuning candidates;
  automatic CSV/Markdown reports include failed or missing experiments.

These optimizations are opt-in through new configs where numerical behavior or
memory use can change. Existing configs continue to load. Floating-point results
can differ with convolution layout, precision, fusion, or another sampling seed.

## Why the existing recipe is expensive

The checked-in L recipe has 128 label groups and 64 generated particles per
group: 8,192 generated images per update. Its two-batch estimator generates
another 8,192 independent negatives. It also extracts features for 12,288 real
images, supervises intermediate MAE blocks and their spatial statistics, and
trains the PT potential/scale networks. Rematerialization recomputes generator
blocks during backward. Frequent evaluation sweeps add time outside updates.

Your running config was not supplied. The launcher reads its saved config.json;
the checked-in recipe alone does not establish your exact compute bottleneck.
At 13.8 seconds, 180k updates take 28.75 days before evaluation/checkpoint time.
Seconds/update must be accompanied by particles/update, particles/second and FID
at equal GPU-hours. Reducing particles changes optimization and can worsen FID.

## Install this update alongside the existing checkout

Upload `PT-Flow-speed-quality.zip` to `/scratch/$USER/ptflow/`. The archive contains
the updated `PT-Flow/` directory. These changes are local; they are not published
to GitHub. Keep the current training job/checkouts and checkpoints intact.

```bash
conda activate ptflow
export WORK=/scratch/$USER/ptflow
mkdir "$WORK/speed_quality_v1"
unzip "$WORK/PT-Flow-speed-quality.zip" -d "$WORK/speed_quality_v1"
cd "$WORK/speed_quality_v1/PT-Flow"

export PTFLOW_ASSETS="$WORK/assets"
export PTFLOW_DATA="$WORK/data"
export HF_ROOT="$PTFLOW_ASSETS/mae"
export TORCH_HUB_DIR="$PTFLOW_ASSETS/torch_hub"
export VAE_HF_PATH="$PTFLOW_ASSETS/sdvae"
export IMAGENET_CACHE_PATH="$PTFLOW_DATA/latents"
export CIFAR10_FID_NPZ="$PTFLOW_ASSETS/fid_stats/cifar10_train_fid_stats.npz"
export IMAGENET_FID_NPZ="$PTFLOW_ASSETS/fid_stats/jit_in256_stats.npz"
```

Use the paths from your working training environment if any differ. Reuse your
existing environment, ImageNet latent cache, CIFAR data, MAE and FID statistics.
No new package or dataset download is needed. If your cluster requires an account:

```bash
export PALMETTO_ACCOUNT=YOUR_REAL_ACCOUNT
```

The commands below are submitted from the login node. Training/evaluation happen
only in GPU jobs. `work1`/`h100` are defaults; use your existing partition and GPU
request if they differ. The launch script never cancels existing training.

## 1. Measure ImageNet on 8 H100s

Choose a fully written checkpoint, not one the training process is still saving.
The listing below chooses the highest numbered `.pt`, excluding `.pt.tmp` files.

```bash
RUN="$WORK/runs/in256_L_180k"
CKPT=$(printf '%s\n' "$RUN"/checkpoints/state_*.pt | sort | tail -n 1)
test -s "$CKPT"

NPROC=8 GPU_TYPE=h100 BENCHMARK_STEPS=40 MAX_JOBS=1 \
  bash scripts/performance_palmetto.sh benchmark \
  "$RUN/config.json" "$CKPT" "$WORK/results/h100_benchmark_v1"
```

Five array jobs compare the original recipe on this engine, preserve, quality,
balanced and speed. Each uses a separate 8-GPU allocation; `MAX_JOBS=2` allows
two 8-GPU allocations at once if your allocation permits 16 GPUs. Do not divide
one 8-GPU node between competing benchmarks when comparing with 13.8 s/update.

| Candidate | Generated/update for checked-in L | Extra generated negatives | PT | Other changes |
|---|---:|---:|---|---|
| original | 8192 | 8192 | saved setting | old OT path; original training choices |
| preserve | 8192 | 8192 | saved setting | cost reuse, channels-last, fused optimizer |
| quality | saved budget, normally 8192 | same as generated | off | retain full ImageNet MAE supervision |
| balanced | 2048 | 2048 | off | stage features; microbatch <=64; no rematerialization |
| speed | 1024 | 0 | off | reuse detached generated negatives with diagonal exclusion; fewer feature losses |

Quality/balanced/speed deliberately change the method. They are W-Flow-derived
experimental controls, not the original PT objective. `KEEP_PT=1` preserves the
auxiliary PT branch in these candidates, at additional cost. Disabling a term is
a hypothesis to test, not evidence that it caused the observed FID.

Inspect:

```bash
cat "$WORK/results/h100_benchmark_v1/report/REPORT.md"
cat "$WORK/results/h100_benchmark_v1/train/preserve/benchmark.json"
squeue -u "$USER"
```

The report is submitted with `afterany`; failures are recorded even when one
candidate runs out of memory. Phase timings include real/generated/negative
features, generator forwards, OT, backward, optimizer/EMA and potential work.
Initial data refill and compilation are included in the first-step time; five
warmup steps are excluded from the median. Benchmarks write no training weights.

To test compilation separately, repeat to a NEW output folder with `COMPILE=1`.
Compilation can add minutes and memory; compare settled throughput before keeping
it. If a candidate runs out of memory, enable `model.use_remat: true` or increase
`train.grad_accum_steps` in its generated config and rerun a benchmark. Reducing
only `feature.chunk_size` does not discard the generated feature graphs retained
until backward, so it is not a complete memory solution.

## 2. Sweep the existing CIFAR checkpoint before retraining

The reported 4.92 was one CFG/EMA choice. This sweep tests the SAME checkpoint at
CFG 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5, 1.8 and 2.0. Each task uses one H100;
up to eight run simultaneously. No new training is involved.

```bash
RUN="$WORK/runs/cifar_180k"
CKPT=$(printf '%s\n' "$RUN"/checkpoints/state_*.pt | sort | tail -n 1)
test -s "$CKPT"

GPU_TYPE=h100 MAX_JOBS=8 FID_SAMPLES=10000 \
  bash scripts/performance_palmetto.sh eval \
  "$RUN/config.json" "$CKPT" "$WORK/results/cifar_cfg_screen_v1"
```

The report lives at `.../cifar_cfg_screen_v1/report/REPORT.md`; raw scores are in
`results/` and CSV files in `report/`. Select promising CFG values, then rerun them
with 50k samples and independent evaluation seeds. Example values below are NOT
claimed winners; replace them with the shortlist from your report:

```bash
CFG_VALUES="1.1 1.2 1.3" SEEDS="101 102 103" FID_SAMPLES=50000 MAX_JOBS=8 \
  bash scripts/performance_palmetto.sh eval \
  "$RUN/config.json" "$CKPT" "$WORK/results/cifar_cfg_confirm_v1"
```

The streaming and old PNG backends use different deterministic sample ordering;
the same integer seed does not give identical images. Rerun the original CFG1.2
checkpoint with the SAME new protocol as candidates. Do not call a change from
the old 4.92 alone a model improvement. For paired comparisons keep backend,
sample count, reference, seed, GPU count and generation batch size fixed.

## 3. Short CIFAR finetuning candidates in parallel

Reuse the current 180k EMA weights; do not discard that training. These are new
5k-step runs with fresh optimizers, 200-step LR warmup and a 5e-5 cosine LR.
They copy the generator architecture and residual convention from config.json.

```bash
NPROC=2 GPU_TYPE=h100 MAX_JOBS=3 STEPS=5000 FID_SAMPLES=10000 \
  bash scripts/performance_palmetto.sh finetune \
  "$RUN/config.json" "$CKPT" "$WORK/results/cifar_finetune_v1"
```

This requests three independent 2-GPU jobs, up to six GPUs total. Each trains one
candidate, saves its checkpoint, then evaluates CFG 1.0/1.2/1.5 on its allocation.
The candidates are:

| Candidate | Generated/class | Positive/class | Negative real/class | Generated negative batch | ConvNeXt size | Sinkhorn iterations |
|---|---:|---:|---:|---|---:|---:|
| quality | 64 | 64 | 32 | independent, 64/class | 224 | 10 |
| balanced | 32 | 64 | 32 | independent, 32/class | 128 | 3 |
| speed | 16 | 32 | 16 | reuse generated features | 128 | 3 |

Each uses a 2 GiB real-feature cache per GPU and tracks EMA 0.999/0.9999. The
larger quality candidate can be slower than the old CIFAR recipe; retain it only
if FID per GPU-hour justifies that cost. These three are coarse recipe comparisons,
not single-factor scientific ablations. Follow up on individual changes if claiming
which component improved FID.

Sweep both EMAs on a selected candidate (example: balanced):

```bash
RUN="$WORK/results/cifar_finetune_v1/train/balanced"
CKPT="$RUN/checkpoints/state_00005000.pt"

EMA_DECAYS="0.999 0.9999" CFG_VALUES="1.0 1.1 1.2 1.3 1.5" \
FID_SAMPLES=50000 MAX_JOBS=8 \
  bash scripts/performance_palmetto.sh eval \
  "$RUN/config.json" "$CKPT" "$WORK/results/cifar_ema_confirm_v1"
```

Old checkpoints do not retroactively gain EMA 0.9999. New extra EMAs start from
the source primary EMA and only accumulate their distinct history after the fork.
Evaluation rejects a requested EMA that is absent instead of silently substituting.

## 4. Continue ImageNet only after the benchmark and short quality check

Use the saved ImageNet config and checkpoint again. For a short quality screen,
the same `finetune` command works with `NPROC=8`; each concurrent candidate then
requires a whole 8-H100 allocation. Start with `MAX_JOBS=1` if only eight GPUs are
available. The screen changes particle counts; compare both equal GPU-hours and
equal generated-particle budgets before choosing a long run.

For a full resume (optimizer, step and primary EMA restored), generate a chosen
candidate and launch within an existing GPU allocation, for example:

```bash
RUN="$WORK/runs/in256_L_180k"
CKPT=$(printf '%s\n' "$RUN"/checkpoints/state_*.pt | sort | tail -n 1)
python -m scripts.make_performance_config --base "$RUN/config.json" \
  --recipe balanced --world-size 8 --steps 200000 \
  --output "$WORK/results/imagenet_balanced_continue.yaml"

# Run this torchrun command INSIDE an 8-GPU Slurm job, not on the login node.
torchrun --standalone --nproc_per_node=8 train.py \
  --config "$WORK/results/imagenet_balanced_continue.yaml" \
  --resume "$CKPT" --workdir "$WORK/runs/in256_L_balanced_continue_v1"
```

`--steps` here is the absolute final step; it must exceed the source checkpoint's
step. In contrast, `--init-ema` starts step zero with a new optimizer. Use a new
workdir for either explicit option. For subsequent automatic resumes of that new
workdir, omit `--resume`/`--init-ema` and clear `train.resume_from/init_ema_from`
if launching directly from its saved config. Saved launch configs and original
checkpoints are retained for provenance. PT-disabled runs cannot use samplers B/C.

## Validation and interpretation

Local validation uses CPU PyTorch 2.4.1: unit/integration tests compare optimized
OT outputs and gradients to the original path; exercise cache eviction, memory-bank
identities, EMA checkpoints, moment-based FID, logging, and mocked Slurm submission.
A two-rank Gloo smoke test checks generator and potential gradient agreement and
distributed FID counts. No cluster login, H100 benchmark, pretrained feature GPU
test, or new FID run has been performed by this local update.

Run after deployment on a compute node:

```bash
python -m pytest tests -q
python -m tests.distributed_performance_smoke --output tmp/ddp-check
python -m scripts.preflight --config "$WORK/runs/cifar_180k/config.json" \
  --world-size 2 --check-assets --check-features
```

For a paper result: full-budget runs, matched reference statistics and particle/
compute budgets, 50k-sample FID, multiple independent training seeds, and complete
reported configuration selection are still required. Theory should describe the
algorithm that was actually trained; removing PT makes this a different experiment.

Sources for baseline/cluster conventions:
- https://github.com/hanjq17/W-Flow
- https://github.com/zlab-princeton/SoFlow
- https://docs.rcd.clemson.edu/palmetto/job_management/submit/
- https://docs.rcd.clemson.edu/palmetto/job_management/arrays/
