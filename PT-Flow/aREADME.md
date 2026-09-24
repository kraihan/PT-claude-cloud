<div align="center">

# PT-Flow

**One-Step Schrödinger-Bridge Generation by Variance-Reversed Proximal Tilting**

Class-conditional ImageNet 256×256 in SD-VAE latent space, 1-NFE sampling,
an exactly normalized per-sample likelihood, and a training scheme that keeps a
Sinkhorn-OT drift as its stability anchor.

</div>

---

## What this is

PT-Flow learns the **data-side Schrödinger potential** `φ_θ` and generates in a
single network evaluation, because three classical identities compose:

| | |
|---|---|
| Moreau | `∇φ₀(x) = x − prox_{φ_θ}(x)` |
| Tweedie + prox | `T(x₀) = prox_{φ_θ}(x₀)` |
| Laplace proposal | `m(x) ≈ prox_{φ_θ}(x)` |

so that

```
m_η(x₀) = prox_{φ_θ}(x₀) = T(x₀) = the one-step generator
```

The variance-reduction proposal network **is** the generator. That removes the
teacher, the adversary and the distillation stage in one step.

### The problem this repo actually solves

The method has one structural weakness: the **bootstrap**. At step 0 the
generator is the identity, so the Laplace-matched proposal degenerates to the
naive `N(x₀, 2εI)` — the regime whose sample complexity is `K ≳ exp(‖∇φ‖²/2ε)`,
which at `d = 4096` exceeds 10¹⁰⁰⁰ at any useful ε. The estimator must be good
for the potential to learn, the potential must be good for the generator to
learn, and the generator must be good for the estimator to work. Nothing starts.

**So the bootstrap is inverted.** A debiased Sinkhorn-OT drift supplies the
generator with a well-conditioned target from step 0 that does not depend on
`u_θ` at all. The potential learns from *that* generator — whose output is
already a good prox guess, hence a well-placed proposal, hence healthy ESS and
an actually-engaged variance reversal. Only once ESS certifies the estimator
does the generator start learning back from the potential, through a ramped
prox-residual term.

Two facts make this a graft rather than a bolt-on:

1. **The two losses have the same shape.** The drift loss regresses generated
   features onto a detached OT-displaced target. PT-Flow's prox residual, with
   its target detached, regresses `m` onto `x₀ − ∇φ_θ(m)`. Both are "regress
   onto a detached displaced target", so they add instead of fighting.
2. **`λ_prox = 0` is bit-for-bit the drift-only baseline** — checked as a test,
   not asserted (`tests/test_train_smoke.py`).

---

## Install

```bash
conda create -n ptflow python=3.10 -y && conda activate ptflow
pip install -r requirements.txt      # or: bash install_env.sh
```

Then edit `utils/env.py` — every path and external asset id lives there — and
fetch the pretrained assets:

```bash
python -m misc.download_pretrained
```

That pulls the MAE feature extractor used by the drift loss, the SD-VAE, the
torch-fidelity Inception, and (optionally) the reference checkpoints used by
`scripts/train/resume_baseline.sh`.

## Data

ImageNet-1k arranged as `train/<wnid>/*.JPEG`, `val/<wnid>/*.JPEG`, then build
the SD-VAE latent cache once:

```bash
python -m dataset.latent \
  --data-path /path/to/imagenet-1k \
  --target-path /path/to/imagenet256-latents-sdvae \
  --local-batch-size 128 --num-workers 8 --pin-memory
```

Point `IMAGENET_CACHE_PATH` at the result. `prepare.sh` collects the asset
download and the cache build in one place.

---

## Quick start

Three CPU-only suites, no GPU and no data required. Run them first.

```bash
python -m tests.test_math          # estimator against closed forms
python -m tests.test_train_smoke   # the training step, on fake data
python -m tests.test_ckpt_compat   # the checkpoint contract, per model size
```

A complete end-to-end run of the method in 2-D, using the real modules
(~2 min on a laptop CPU) — this is the fastest way to see the whole scheme move:

```bash
python -m examples.toy2d --steps 3000 --lambda-prox 0.3 --plot --out runs/toy2d
```

---

## Training

### Resuming a reference checkpoint (recommended)

The generator's `state_dict` is shape-identical to the public OT-drift release
for B, L and XL, so a reference `state_*.pt` at step N resumes here and
continues to N+M — generator, EMA, AdamW moments and step count all carry over.

```bash
export BASELINE_HF_ROOT=/path/to/baseline_hf_root
SIZE=XL bash scripts/train/resume_baseline.sh      # or SIZE=B / SIZE=L
```

The script preflights the shape match **before** claiming any GPUs, seeds a
fresh workdir with the checkpoint, and starts `train.py`. For the first
`prox_warmup` steps `λ_prox = 0`, so the run *is* the drift-only baseline while
the potential learns the generator it inherited. **FID at the resume point
should therefore equal the reference checkpoint's published FID.** If it does
not, stop — the handoff is wrong, not PT-Flow.

### From scratch

```bash
bash scripts/train/train_scratch.sh                # configs/gen/ptflow_scratch.yaml
```

Note `model.residual: true` here and `false` in the resume configs. It adds no
parameters either way, so shapes are unaffected — but it changes what the
weights *mean*, and the reference release was trained as `m(x) = net(x)`, not
`x + net(x)`.

| Situation | Config | `residual` |
|---|---|---|
| Resume a reference checkpoint | `ptflow_{B,L,XL}.yaml` | `false` |
| From scratch | `ptflow_scratch.yaml` | `true` — identity warm-start ⟹ ESS/K = 1 at step 0 |

### The training step

**(A) generator, θ frozen** — the OT drift loss, unchanged, plus

```
L_prox = ‖ m − (x₀ − ∇φ_θ^w(m)).detach() ‖² / d       weighted by λ_prox(t)
```

**(B) potential, η frozen, at `w = 0` always**

```
q_def   = (1−α)·N(m₀, 2ε·diag(e^{s₀})) + α·N(x₀, 2εI)
log w_k = log G_{2ε}(y_k−x₀) − u_θ(y_k,c) − log q_def(y_k|x₀)
L̃_θ    = mean φ_θ(x₁,c) − mean φ̂₀(x₀,c) + λ_gauge·gauge²
```

Push the potential down on data, up on model samples — a contrastive structure
whose negatives are the generator's own output.

**(C) monitor** — `ESS/K` drives everything:

| ESS/K | State | Action |
|---|---|---|
| > 0.30 | healthy | advance the ε anneal and the λ_prox ramp |
| 0.05–0.30 | degrading | freeze the anneal, raise `α_def`, halve the θ rate, decay λ_prox |
| < 0.05 | broken | **freeze θ**, hold λ_prox at 0, keep training η |

The ε anneal advances on healthy steps only, so a run that never certifies never
cools. That is the intended failure mode.

### What to watch

| Metric | Healthy | Meaning |
|---|---|---|
| `pt/ess` | > 0.3 | the certificate for the whole scheme — **watch this first** |
| `pt/logw_spread` | shrinking as ε cools | if it *grows*, the variance reversal is not engaged |
| `pt/prox_resid_rel` | falling, < ~0.3 | a multi-modal blow-up is the weak-convexity alarm |
| `pt/lambda_prox` | 0, then ramping | the weight actually applied this step |
| `pt/gauge` | near 0 | free-direction drift; raise `lambda_gauge` if it wanders |
| `pt/clip_frac` | 0 | nonzero means the safety clip is load-bearing |

---

## Inference

```bash
# Mode A — strict 1-NFE.  Needs no potential.
python inference.py sample   --ckpt … --config … --sampler A --cfg-scale 1.2

# Mode B — n-step prox refinement.  A quality/compute dial, no retraining.
python inference.py evaluate --ckpt … --config … --sampler B --refine-steps 4

# Mode C — SNIS.  As K→∞ this samples ρ̂₁ exactly.
python inference.py evaluate --ckpt … --config … --sampler C --snis-k 32

# Exactly normalized per-sample likelihood, reported as an IWAE ladder.
python inference.py likelihood --ckpt … --config … --k-ladder 16,32,64,128
```

`scripts/eval_fid/eval_modes.sh` sweeps all three modes from one checkpoint;
`scripts/eval_fid/eval_likelihood.sh` runs the ladder.

**Two guidance weights, easily conflated:**

| Flag | Meaning |
|---|---|
| `--cfg-scale` | generator conditioning; convention is `cfg_scale = w + 1`, so `1.0` is unguided |
| `--pt-w` | potential arithmetic `φ^w = (1+w)φ(·,c) − w·φ(·,∅)`, used by Modes B and C; defaults to `cfg_scale − 1` |

They are different mechanisms — `--pt-w` yields an exact bridge onto a
w-sharpened marginal with zero discretization error at any `w` — so sweep it on
its own axis.

Mode B only helps when the prox identity actually holds. Check
`pt/prox_resid_rel` before trusting it; above ~0.5 it drags good samples toward
a prox the potential has not learned.

---

## Layout

```
ptflow/            potential + scale net, tilted estimator, losses, schedule,
                   sampling, the two training steps, the OT drift losses
models/            DiT generator, MAE feature encoder, ConvNeXt
dataset/           ImageNet and SD-VAE latent-cache datasets
utils/             checkpointing, distributed, FID, logging, env paths
configs/gen/       ptflow_{B,L,XL}.yaml, ptflow_scratch.yaml, baseline_*.yaml
scripts/           training and evaluation launchers
tests/             CPU suites: math, training step, checkpoint contract
examples/toy2d.py  the whole method end to end in 2-D
```

---

## Deviations worth knowing

Each has a config switch back.

1. **The network outputs `φ`, not `u`** (`pt.model.output_mode`). `φ_θ` has a
   finite ε→0 limit — the Brenier potential, fixed by the data. `u = φ/2ε`
   diverges like 1/ε, so annealing ε over two decades would force a
   `u`-parameterized network to track a 200× change in its own output scale.
2. **`eps_min` defaults to 5e-3**, not 1e-3. Measured: the tilted log-weight
   spread tracks the predicted √ε law down to ε ≈ 0.01 and then flattens
   (`0.180 → 0.140 → 0.123` across ε = `0.02 → 0.01 → 0.005`, where √ε predicts
   1.41× per step and the last step gives 1.13×). That flattening is the fp32
   precision floor. Push lower only while watching `pt/logw_spread`.
3. **The prox target is detached by default** (`pt.prox_mode`). `"full"` is the
   double-backward version: faithful, materially slower, less forgiving early.
4. **The scale head is a separate θ-side module** (`ptflow.potential.ScaleNet`),
   not a second head on the generator. On the generator it would double the
   final layer's output channels and permanently break checkpoint
   compatibility. It is also the more honest placement: `S⁻¹ = I + ∇²φ` is a
   property of the potential.
5. **Guards the method description leaves implicit**: a gauge pin on the
   objective's one flat direction (which is also what keeps fp32 precision in
   the log-weights), a log-weight ceiling, a bounded log-scale, fp64 log-weight
   arithmetic, explicit gradient all-reduce (DDP only hooks `forward`, and every
   call site uses `.phi()`), and a rank-consistent ESS so the ranks cannot
   disagree about whether the θ step runs.



