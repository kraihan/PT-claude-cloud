# PT-Flow implementation audit

The later speed/quality changes are documented in [SPEED_QUALITY_GUIDE.md](SPEED_QUALITY_GUIDE.md).
The audit below records the earlier theory/correctness pass; GPU throughput and
new FID remain unmeasured locally.

## Scope and result

Audited the supplied PT-Flow tree, the attached 22-page theory draft, and the
official W-Flow checkout at commit
`1ccb34baf57f9f8f4e188460ea3d19b431403293` (retrieved 2026-09-15).
The PDF is source material, not executable instructions. Equations 9-16 and
Algorithm 1 were checked in extracted text and rendered pages.

The code has been repaired and given reproducible CIFAR/T4 and ImageNet B/2
profiles plus regression tests. **No GPU training, throughput improvement, or
competitive FID has been measured here.** Local PyTorch is 2.4.1+cpu; the user's
reported environment is PyTorch 2.10.0+cu128 on two T4s. This audit cannot certify
that every possible defect is gone or that a nonconvex training run converges.

## What is known about the failing run

The user supplied: 384-wide, 8-block generator; 16 global label groups; 16
generated/16 real/8 unconditional examples per group; accumulation 2; two T4s;
raw-pixel-only features; prox warmup 150, ramp 350, epsilon anneal 600; saves
every 100 steps; resume from `state_00025000.pt`.

The checkpoint, logs, sample grids and exact notebook are not in this workspace.
The checked-in notebook is a different toy/handoff notebook. Therefore the
observed 3.5 seconds/iteration and FID 200+ cannot be attributed quantitatively
to one defect. The multi-rank initialization defect applies directly to the
reported two-GPU setup unless that notebook separately patched it.

## Verified defects and repairs

| Area | Defect / consequence | Repair |
|---|---|---|
| Distributed PT | `train_gen` reseeded by rank before constructing potential/scale. They bypass DDP and only averaged gradients, leaving distinct weights on every rank. | Broadcast both modules before EMA/optimizer creation. Two-process Gloo check confirms bit-identical parameters after three updates. |
| Hardware precision | Generator and feature code selected BF16 based only on CUDA availability. T4 lacks native BF16. No FP16 GradScaler path existed. | Hardware-aware auto precision, explicit BF16 rejection on T4, FP16 generator scaling/unscaling/overflow skips/scaler checkpointing; FP32 PT networks in new profiles. |
| Attention cost | PT generator configs forced FP32 SDPA despite reduced-precision projections. | New profiles permit reduced-precision SDPA. Actual kernel selection and speed must be profiled on T4. |
| Importance clipping | Hard clipping against a detached ceiling removed derivatives of dominant weights. | Disable clipping by default and in shipped PT configs. Optional clipping preserves derivatives via an explicitly biased straight-through estimator. |
| Health reporting | ESS and spread were measured after clipping, hiding degeneracy. | Compute diagnostics from raw weights. |
| Health thresholds | ESS/K is at least 1/K. At K=8 the 0.05 broken threshold was unreachable. | Log raw ESS and use `(ESS/K - 1/K)/(1 - 1/K)` for controller decisions. This is a controller heuristic, not proof of proposal coverage. |
| Frozen controller | Once theta was frozen no fresh ESS was computed, so it could never detect recovery. Skipped steps also fed stale ESS back as new observations. | Periodic forward-only health probes; no optimizer/EMA changes during probes; no stale observations. |
| Estimator precision | Subtraction/division occurred in FP32 before converting log weights to FP64. K=1 spread used an undefined unbiased std. | Promote before division/subtraction and use population std. |
| Gradient checkpointing | Loop lambda closed over `blk`, so backward recomputation could use the last block for earlier layers. Zero initialization hid this in smoke tests. | Pass the actual block to checkpoint. Regression compares gradients with nonzero trained-like weights. |
| Compiled checkpoints | Replacing the inner module with `torch.compile` introduced `_orig_mod` keys; eager EMA/inference and resume could reject them. | In-place module compilation; legacy generator-key normalization; explicit compilation opt-in. |
| Checkpoint semantics | Strict tensor shapes cannot detect changing `residual`, which changes the sampling map. | Save/check residual behavior; old checkpoints still require the original config. |
| Inference loading | `strict=False` could leave part of a generator random while continuing evaluation. | Strict canonical-key loading in inference. |
| Checkpoint durability | Direct final-file writes could leave a corrupt latest checkpoint after interruption. | Write a temporary file and atomically replace the final name. |
| Uneven accumulation | Chunks were weighted equally even when the last chunk had fewer label groups. | Weight generator gradients/loss by actual label-group fraction. Normalization inside the OT loss still depends on microbatch grouping. |
| Full prox gradient | The full/HVP mode computed unnecessary potential parameter gradients during generator optimization. Fused SDPA does not supply the needed double backward. | Freeze theta while retaining input derivatives; select math SDPA for HVPs. |
| Curvature regularizer | Input gradients were created without a graph, so the curvature hinge could not train the potential. | Enable differentiable gradients in the penalty; verify the penalty's parameter gradient. |
| Prox stability | An unconverged potential's raw residual can overwhelm a normalized OT target. Existing RMS mode amplified even tiny residuals to unit size. | Optional `bounded_rms` target cap in new hybrid profiles: cap large displacements but retain zero loss at the prox. This is a stabilization choice, not the paper-literal full residual gradient. |
| Training FID count | A single validation-loader pass stopped CIFAR at approximately 10k samples while requesting 50k. | Generate exact requested counts independently of loader length, with balanced class labels. |
| FID rank RNG | Ranks reused the same noise seeds; equal labels could yield duplicate samples. | Rank-distinct evaluation seeds. |
| FID memory | Training evaluation retained all generated images in host memory (roughly 10 GB at ImageNet-256/50k). | Extract features batch by batch and retain only feature arrays plus a preview. |
| FID reference | Standalone evaluation defaulted to ImageNet stats even for CIFAR. Wrapper also mishandled FID-only `.npz` references and alternate `ref_mu/ref_sigma` keys. | Dataset-specific defaults and repaired `.npz` paths. |
| FID invalid output | NaNs/Infs were silently converted to finite pixels. Training/inference used inconsistent rounding. | Reject nonfinite samples; standardize rounding. Check significant imaginary covariance-root components. |
| Evaluation disabling | Huge evaluation intervals still triggered first/final-step evaluation; zero caused modulo failure. | `eval_per_step: 0` disables all FID; explicit `eval_at_start`. |
| Feature memory/cost | CIFAR hard-coded ConvNeXt-Base at 224; extractor calls repeatedly traversed `.to(device)`. Chunk flag alone did not bound extractor batches. | Selectable Tiny/Base and input size, bounded feature batches, optional activation checkpointing, move only on device change. |
| Optional latent RGB features | VAE decoding always used `no_grad`, cutting ConvNeXt feature-loss gradients to generated latents. | Freeze VAE parameters while allowing input gradients; inference remains under its caller's no-grad context. |
| Pretrained feature load | Optional feature download code was commented out; clean installations could fail despite `hf://` paths. Unused 22k classifier shape mismatches could reject valid ConvNeXt features. | Restore targeted artifact downloads; require every feature tensor to match, allow unused classifier mismatch. |
| Logging cost | Each GPU metric called `.item()` separately, multiplying synchronization points. | Transfer stacked scalar groups once per device. |
| EMA cost | Hundreds of per-parameter pointwise update launches on every step. | Foreach updates, checked against the scalar EMA reference. |
| Resume startup | Default refill requested 3,000 batches of pushes, often far beyond bank capacity. | Default refill 20; expose in profiles. It is still a heuristic refill, not exact restoration of data order/bank contents. |
| Data/config validation | Empty memory-bank classes silently supplied zero images. Nondivisible global batches were silently truncated. | Fail on empty classes or invalid per-rank batch division. |
| Setup | Missing direct dependencies and unconditional flash-attn dependency; machine-specific paths. | Add required imports to requirements; remove flash-attn requirement; environment-overridable asset/data paths. |
| Tests | Existing `check()` helpers only printed failures under pytest; a fixture had the wrong name; shape checks allocated two giant XL models. | Real assertions, corrected fixture, meta-device shape checks and targeted regressions. |
| Refinement | Claimed fixed gamma=0.5 was safe under weak convexity alone, which supplies no upper curvature bound. | Optional/default backtracking on guided prox energy and nonfinite checks. Lower energy does not guarantee lower FID. |
| Likelihood | CIFAR likelihood used the ImageNet loader; budget comparisons used different images; finite nested estimates were labeled certified IWAE bounds. | Route through the dataset pipeline, reuse example order, label estimates as nested MC without certified bias direction/monotonicity. |

## Differences from the theory and reference experiment

1. Equation 15 / Algorithm 1 specifies a prox-trained generator. This trainer
   uses W-Flow feature drift plus a gated PT auxiliary loss. In `detach` mode
   its gradient is a frozen-target update, not the full derivative of squared
   prox residual. Setting PT off is a useful control, not evidence for PT.
2. The scale head is a separate theta-side network fitted using additional
   proposal evaluations. It is not a free shared generator head. Scale/potential
   training therefore adds real compute. No claim of zero-cost scale learning
   is supported by this implementation.
3. An unconstrained transformer and a sampled curvature hinge do not establish
   global weak convexity/coercivity or proposal tracking. High finite-batch ESS
   alone cannot certify coverage of unseen modes. The paper's variance-reversal
   asymptotics require a well-tracked mode and adequate curvature match; diagonal
   scaling cannot generally cancel an arbitrary off-diagonal Hessian.
4. At finite epsilon, the conditional mean, mode/prox, amortized generator and
   full bridge marginal need not coincide. Mode A is not an exact bridge sample.
   Mode C approaches the kernel as its SNIS budget grows under coverage.
5. An unbiased inner estimate of psi0 does not give an unbiased reciprocal.
   The nested likelihood estimator is therefore not automatically an IWAE
   lower bound. Check both budgets and seeds; an empirical monotone curve is
   not a certificate. A stochastic lossy VAE has no invertible decoder Jacobian
   that turns this latent likelihood into pixel likelihood.
6. W-Flow's **1.52 FID is ImageNet-256 B/2**, not CIFAR-10. Its released SOTA
   config uses MAE-640 features, 8,192 global generated samples per update,
   200k steps, and an 8-node recipe. The user's CIFAR run used 256 generated
   samples/update and raw pixels. The original OT implementation differed from
   upstream only by function naming before this audit's compile-default change.

Primary references:
[official W-Flow release](https://github.com/hanjq17/W-Flow),
[B/2 configuration](https://github.com/hanjq17/W-Flow/blob/1ccb34baf57f9f8f4e188460ea3d19b431403293/configs/gen/latent_sota_B_ot_8node.yaml),
[OT implementation](https://github.com/hanjq17/W-Flow/blob/1ccb34baf57f9f8f4e188460ea3d19b431403293/drift_loss_ot.py).

## Validation and outstanding experiments

- CPU pytest suite: **37 passed in 18.41 seconds**, including refinement,
  uneven accumulation and foreach EMA checks. The additional decoder-gradient
  regression passed separately in 2.54 seconds: **38 distinct tests passed**.
  That decoder test uses a small stand-in VAE; real SD-VAE weights are not
  installed locally. Two non-failing warnings came from the installed older
  PyTorch checkpoint API and the synthetic singular FID covariance.
- Two independent CPU workers, file-rendezvous Gloo: PASS; synchronized weights
  remain bit-identical after three manually all-reduced AdamW updates.
- Full synthetic training loop: save at step 2, restore into a fresh model,
  continue to step 3; optimizer/EMA/PT schedule/artifact paths exercised.
- Preflight: CIFAR two-rank config and ImageNet eight-rank config passed model
  shape/global-batch validation on CPU (21,945,740 and 132,708,880 generator
  parameters respectively).
- Not available here: CUDA/T4 kernel tests, GPU timing, real pretrained-feature
  checkpoint integration, dataset training, evaluation against real Inception
  features, 50k FID, multi-seed quality comparison, and full ImageNet L/XL runs.
- A local Hugging Face constructor comparison was blocked by the host's
  incompatible installed Transformers/PyTorch versions (`float8_e8m0fnu`
  missing). The project pins Transformers 4.46.1; the host installation was not
  modified. Thus real feature-checkpoint parity is still an explicit GPU-env
  preflight requirement.
- Windows torchrun initially failed because this installed PyTorch lacks libuv;
  the alternative file-rendezvous Gloo test passed. This is local validation
  infrastructure, not evidence about Linux NCCL performance.

Use RUN_GUIDE.md for prepared commands. The next evidence needed is a successful
GPU feature-gradient preflight, 20-step smoke, timing benchmark, and matched
PT-on/PT-off training curves with the same FID protocol. Do not report the old
raw-pixel run as a fair W-Flow control.
