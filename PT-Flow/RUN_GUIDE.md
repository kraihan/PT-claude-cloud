# Running the repaired implementation

Use this guide in place of the old README/notebook commands. These are launch
profiles for GPU validation, not measured FID results. No trained weights or
datasets are included. The current trainer is **W-Flow feature drift plus an
auxiliary PT prox loss**; see AUDIT.md for differences from Algorithm 1.

## Your two-T4 CIFAR run

The supplied run used raw flattened pixels (`USE_CONVNEXT=False`), a 150-step
prox warmup, a 600-step anneal, and checkpoint writes every 100 steps. Those are
not the feature-training or schedule settings used by the new profile. A T4
does not have native BF16. `precision: auto` now selects FP16 with GradScaler
for its generator. The potential and scale head use FP32.

Start a **new workdir**. `ptflow_cifar10_t4.yaml` has an 8-block generator,
smaller PT networks, and no extra categorical noise. It cannot resume the
25,000-step checkpoint from your supplied settings unchanged. Keep that
checkpoint and its exact original config for diagnosis. The repo does not
contain that checkpoint or the notebook version that produced it.

## Setup (Linux/Kaggle)

Keep a working CUDA PyTorch/torchvision installation. Do not install flash-attn
on T4. The requirements file no longer requires it.

```bash
cd /path/to/PT-Flow
pip install -r requirements.txt
export DRIFT_COMPILE=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PTFLOW_ASSETS=/kaggle/working/ptflow-assets
export PTFLOW_DATA=/kaggle/working/ptflow-data
python -m pytest -q
```

All asset paths can be overridden through environment variables (utils/env.py).
For existing Kaggle data, set `CIFAR10_PATH` to the folder containing
`cifar-10-batches-py`; that avoids downloading another copy.

```bash
python -m misc.download_cifar10
python -m scripts.make_cifar10_fid_stats --batch-size 64
python -m scripts.preflight --config configs/gen/ptflow_cifar10_t4.yaml \
  --world-size 2 --check-assets --check-features
```

`--check-features` downloads/loads ConvNeXtV2-Tiny and verifies finite, nonzero
gradients to its input. The CIFAR profile uses its features at 128x128 as an
explicit speed/quality tradeoff. Compare with 224x224 before reporting a best
FID; the resize setting is `train.activation_kwargs.convnext_kwargs.image_size`.
It is **not** a pretrained CIFAR feature model or an established FID recipe.

## Smoke and timing

Smoke mode changes the number of steps, checkpoint interval, and evaluation
only. It does not compress the potential or viscosity schedules.

```bash
python -m scripts.make_run_config --profile cifar10_t4 \
  --smoke 20 --output configs/gen/c10_smoke.yaml
torchrun --standalone --nproc_per_node=2 train.py \
  --config configs/gen/c10_smoke.yaml --workdir runs/c10_smoke_fixed

python -m scripts.make_run_config --profile cifar10_t4 \
  --benchmark 30 --output configs/gen/c10_benchmark.yaml
torchrun --standalone --nproc_per_node=2 train.py \
  --config configs/gen/c10_benchmark.yaml --workdir runs/c10_benchmark_fixed
```

Timing synchronizes CUDA, excludes the first five iterations from its summary,
and reports median/mean seconds per step and peak allocated GPU memory.
Benchmark mode excludes checkpoint and FID work. It measures the early schedule
only; repeat with a matching checkpoint in a separate benchmark workdir to
measure an active prox term. No T4 timing has been measured in this workspace.

## CIFAR training and evaluation

```bash
torchrun --standalone --nproc_per_node=2 train.py \
  --config configs/gen/ptflow_cifar10_t4.yaml --workdir runs/c10_fixed

torchrun --standalone --nproc_per_node=2 inference.py evaluate \
  --ckpt runs/c10_fixed/checkpoints/state_00020000.pt \
  --config configs/gen/ptflow_cifar10_t4.yaml --sampler A \
  --cfg-scale 1.2 --num-samples 50000 --gen-bsz 64 \
  --workdir runs/c10_eval --json-out runs/c10_eval/result.json
```

Replace the checkpoint filename with one actually saved. Training saves every
2,000 steps. FID defaults to the dataset's reference statistics and uses
`inception-v3-compat`. The training evaluator now generates the requested count,
uses balanced classes and rank-distinct seeds, and streams images through
Inception rather than accumulating 50k ImageNet images in RAM. A 500-sample
sanity FID is not a benchmark. Set `train.eval_per_step: 0` to disable evaluation
completely, including the last step. Leave it enabled for long-run monitoring.

Monitor `runs/c10_fixed/log/metrics.jsonl`: finite gradient norms, raw
`pt/ess`, `pt/control_ess`, `pt/clip_frac`, `pt/lambda_prox`, `pt/sched_eps`,
`pt/prox_resid_rel`, and `optimizer_step_skipped`. Repeated skipped FP16 updates
need investigation. `pt/control_ess` removes the finite-K floor from ESS/K;
with K=8, ordinary ESS/K cannot be less than 0.125. Persistent frozen cooling or
zero PT weight means the PT estimator is not tracking adequately, even if the
feature-driven generator improves. Do not conceal that in a paper result.

The normalized OT regression loss can remain close to 1 by construction.
Its failure to decrease is not itself proof of failed training. Track generated
images and FID. Evaluate CFG around 1.0, 1.1, 1.2, 1.5 and 2.0 on the same
checkpoint, using the same sample count/reference protocol.

## Matched control

```bash
python -m scripts.make_run_config --profile cifar10_t4 --baseline \
  --output configs/gen/c10_feature_control.yaml
torchrun --standalone --nproc_per_node=2 train.py \
  --config configs/gen/c10_feature_control.yaml --workdir runs/c10_feature_control
```

This uses the same generator/features/batches with PT disabled. Comparing it
with the PT profile is how to determine whether the PT term helps. Both are
fresh runs. Do not compare against the old raw-pixel notebook as if only PT
changed. Use more than one seed for a final claim.

## ImageNet-256 B/2

ImageNet requires the real 1,000-class dataset, the SD-VAE, latent cache, MAE
feature weights, and matching FID statistics. A 10-class subset only tests the
pipeline; it cannot establish an ImageNet-1k FID.

```bash
export PTFLOW_ASSETS=/workspace/assets
export PTFLOW_DATA=/workspace/data
export IMAGENET_PATH=/workspace/data/imagenet
export IMAGENET_CACHE_PATH=/workspace/data/latents
export IMAGENET_FID_NPZ=/workspace/assets/fid_stats/jit_in256_stats.npz
python -m misc.download_pretrained
python -m dataset.latent --data-path "$IMAGENET_PATH" \
  --target-path "$IMAGENET_CACHE_PATH" --local-batch-size 64 \
  --num-workers 8 --pin-memory
python -m scripts.preflight --config configs/gen/ptflow_imagenet_B_fresh.yaml \
  --world-size 8 --check-assets --check-features
torchrun --standalone --nproc_per_node=8 train.py \
  --config configs/gen/ptflow_imagenet_B_fresh.yaml --workdir runs/imagenet_B_fixed
```

This is a substantial-GPU-memory profile, not a two-T4 ImageNet promise. It
retains upstream B/2 generator shape, zero-output initialization, MAE-640,
8,192 global generated particles/step, LR 4e-4, and 200k steps. Accumulation
and added PT computation mean it is not a bit-identical reproduction of the
upstream 8-node run. More accumulation reduces activation memory but not the
global number of particles. Verify fit with a 20-step smoke profile first.
Generate a matched ImageNet control with `--profile imagenet_b --baseline`.
Existing L/XL checkpoint-compatible profiles remain available; they have not
been benchmarked on hardware here.

## Resume, compilation, and sampling

- Re-running the same command resumes the latest complete `state_*.pt` in that
  workdir, including optimizer, EMA, PT schedule, and FP16 scaler.
- Old `_orig_mod` generator checkpoint keys are accepted. New in-place
  compilation preserves canonical checkpoint keys.
- Start with `DRIFT_COMPILE=0`. Benchmark `DRIFT_COMPILE=1` separately on the
  actual Linux/CUDA environment; compilation is opt-in and its warmup is costly.
- Keep tensor dimensions and `residual` unchanged across resume. New checkpoints
  store and check residual semantics; old ones cannot prove this flag for you.
- Mode A is one generator call. Mode B now backtracks on the prox energy; this
  does not guarantee improving FID and its evaluation count is variable.
- Mode C is finite-budget SNIS, asymptotically exact as K grows under the
  assumptions. Finite K=32 is not an exact sample from the model marginal.
- Likelihood output is a nested Monte Carlo estimate, not a certified finite-K
  likelihood bound. Sweep both inner/outer budgets; latent likelihood is not
  pixel bits/dim for a lossy VAE.

## Validation status

CPU math, gradient, full training/save/resume and configuration checks have
passed locally (38 distinct tests). A separate two-process Gloo test verifies identical manually
synchronized parameters after updates. CUDA kernels, real pretrained feature
downloads, ImageNet data, T4 throughput, and 50k-image FID require GPU validation.
See AUDIT.md for the complete findings and exact test result.
