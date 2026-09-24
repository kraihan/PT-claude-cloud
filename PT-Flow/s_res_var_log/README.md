# s_res_var_log — new-s² inference (repaired Section 2.5)

Inference/evaluation code for a model **you already trained**. No retraining.

## Why this exists

The manuscript's repair of §2.5 changes only *how the tilted estimator's
variance and sample budget are characterized*. It does **not** change the
estimator, the potential `u_θ`, the generator `m_η`, the scale head `s_η`, or
the training objective. The importance estimator (`ptflow/estimator.py`) is
unbiased for any proposal at any `ε`, so a checkpoint trained under the old
write-up is fully valid under the new one. This is purely an inference-time
re-reporting of quantities computed from the same importance weights.

## What actually changed

| | old (Theorem 2.9) | new (repaired §2.5, this folder) |
|---|---|---|
| `s_res` | `std_k(log w)` | `Var_k(log w)` — same measurement, now bounded `O(ε)` |
| relative variance | `(e^{s_res²} − 1)/K` (log-normal) | `χ²(p‖q)/K` from **actual weight moments** |
| `χ̂²` | — | `mean_k(w²)/mean_k(w)² − 1 = 1/(ESS/K) − 1` |
| sample budget | `K ≳ exp(c·d·M₃²·ε)` | `K*(δ) = ⌈χ̂²/δ²⌉` (eq. Ktilted) |
| defensive mix | — | `χ²(p‖q_def) ≤ (χ²(p‖q)+α)/(1−α)` |
| (A6) floor | — | `½‖B_x‖_F² = ½·tr(Q²)`, `Q = S^{1/2}HS^{1/2} − I` |

The one-line summary: **stop exponentiating the log-weight variance; read the
χ² divergence off the weights directly** (it is exactly `1/ESS − 1`, the ESS the
codebase already logs).

`ptflow/*` is untouched. `s_res.py` reuses `ptflow.estimator.tilted_phi0` (the
weights) and `ptflow.potential.guided_phi_grad` (the HVP for the (A6) term).

## Files

- `s_res.py` — the new-s² library: `chi2_from_log_w`, `sample_budget`,
  `logw_variance`, `chi2_defensive_upper_bound`, `offdiagonal_curvature_term`,
  `variance_report`, `eps_sweep`.
- `infer.py` — CLI. `variance` and `likelihood` are new/augmented; `evaluate`
  (FID) and `sample` are delegated verbatim to the repo's `inference.py` since
  the samples themselves do not change.

## Usage (run from the PT-Flow repo root)

New-s² report on the trained model (χ², K*(δ), s_res², optional (A6) term):

```bash
python -m s_res_var_log.infer variance \
    --ckpt runs/.../state_XXXXXX.pt --config configs/....yaml \
    --num-batches 8 --bsz 16 --K 64 --alpha-def 0.05 \
    --deltas 0.3,0.1,0.05 --curvature-probes 8 \
    --csv-out out/variance.csv --json-out out/variance.json
```

Variance-reversal sweep (the paper's crossing figure, now on a real χ²/K axis):

```bash
python -m s_res_var_log.infer variance \
    --ckpt ... --config ... --K 64 \
    --eps-sweep 0.2,0.1,0.05,0.02,0.01 --csv-out out/reversal.csv
```

Likelihood ladder — NLL values are identical to `inference.py` (unbiased,
untouched); the health columns are now `inner_chi2` / `inner_rel_rmse`:

```bash
python -m s_res_var_log.infer likelihood \
    --ckpt ... --config ... --k-inner 16 --k-ladder 16,32,64,128 \
    --json-out out/nll.json
```

FID / preview (unchanged samples, delegated to `inference.py`):

```bash
python -m s_res_var_log.infer evaluate --ckpt ... --config ... --sampler C --snis-k 32
python -m s_res_var_log.infer sample   --ckpt ... --config ... --sampler A
```

## Reading the output

- **`chi2` / `rel_var` / `rel_rmse`** — the estimator's true relative error at
  the `K` used. `rel_var = χ²/K`. This replaces `(e^{s²}−1)/K`.
- **`K*(δ)`** — draws needed for relative RMSE `δ`. Small and shrinking with `ε`
  is the "variance reversal" the repair actually claims.
- **`s_res2`** — `Var_k(log w)`; should be `O(ε)` and `≥ ½‖B_x‖_F²`.
- **`offdiag_half_fro`** — the (A6) floor a diagonal proposal cannot cancel;
  `s_res2_over_offdiag ≈ 1` means the quadratic term dominates (diagonal
  matching is the binding constraint), `≫ 1` means the Taylor remainder does.
- **`chi2_pure` vs `chi2_def_bound`** — measured pure-tilt χ² and its
  defensive-mixture upper bound `(χ²+α)/(1−α)`. Keep `α = O(ε)` to preserve the
  `O(ε)` relative-variance bound.
- **`legacy_lognormal_rel_var`** — the old `(e^{s²}−1)/K`, printed only for
  contrast; do not report it as the budget.
