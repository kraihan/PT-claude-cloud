# PT-Flow pre-submission revision package

Source reviewed: `C:\Users\kraih\Downloads\PT_flow_ICLR_2027 (5).pdf` (29 pages), together with the implementation and audit material in `PT-Flow/`.

## Bottom line

The central idea is worth preserving, but the paper is not submission-ready in its current form. The main problem is not a missing lemma. It is that four different objects are repeatedly collapsed into one:

1. the exact bridge conditional mean `T_{epsilon,theta}`;
2. the mode/proximal point `y*_theta` used in a small-epsilon approximation;
3. the learned network output `m_eta`;
4. a finite-budget Monte Carlo approximation to training and likelihood quantities.

The revision should make those layers explicit everywhere. The paper can still claim one generator evaluation at inference for Mode A, an analytically normalized model density, and a useful conditional variance result. It should not claim that Mode A is an exact bridge sample, that the finite-budget likelihood value is exact or an IWAE bound, or that the implemented generator objective is literally Equation 16 unless the implementation is changed.

There is also a provenance problem. The workspace contains no completed ImageNet runs, checkpoints, raw FID outputs, NLL outputs, or H100 benchmark artifacts supporting the paper's headline FID and throughput numbers. The repository itself says these quantities are unmeasured. Do not retain any headline number unless its checkpoint, frozen config, code hash, sample/reference protocol, raw result, and hardware benchmark log are archived.

## 1. Vocabulary to enforce throughout

| Layer | Definition | What is exact | What is not exact |
|---|---|---|---|
| Bridge identity | `T_{epsilon,theta}(x) = E[X_1 | X_0=x] = x - grad phi_{0,theta}(x)` | The conditional-mean identity for the bridge induced by `phi_theta`, assuming exact heat-semigroup expectations | It is not a draw from the bridge kernel and need not push `rho_0` exactly to the bridge marginal |
| Prox approximation | `y*_theta(x) = prox_{phi_theta}(x)` | It is the exact minimizer of `phi_theta(y) + ||y-x||^2/2` under the stated uniqueness assumptions | `T_{epsilon,theta}=y*_theta` only in a controlled small-epsilon/regularity regime |
| Learned sampler | `m_eta(x,c,w)` | One forward pass is one generator NFE | It is an amortized approximation to the prox; its accuracy must be measured or bounded by its residual |
| Finite training estimator | `hat psi_0` from `K` importance samples | The arithmetic mean is unbiased for `psi_0` when the proposal covers the target and weights are evaluated exactly | `log hat psi_0` is biased; the optimized finite-`K` objective is not the exact function-space objective |
| Model density | `hat rho_{1,theta}=hat psi_{1,theta} psi_{1,theta}` | The mathematical density is normalized by the heat-semigroup identity | A numerical nested-MC evaluation of `log hat rho_1(x)` is not exact, unbiased, or generally a certified lower bound |
| Mode C | Finite-`K` self-normalized importance resampling | Consistent under coverage as `K -> infinity` | A finite-`K` resampled point is not an exact bridge sample |

Recommended global terminology:

- Replace "exact one-step generation" with "one-pass amortized approximation to the bridge conditional mean" or simply "one-pass generation."
- Reserve "exact" for identities evaluated with exact expectations, not for learned networks or finite Monte Carlo.
- Replace "exact per-sample likelihood" with "an analytically normalized latent-space model density, evaluated with nested Monte Carlo."
- Replace "formal infeasibility" and "empty feasible set" with "an asymptotic variance barrier for the naive proposal under the stated local and log-normal approximations."
- Replace "ESS certifies" with "ESS diagnoses on-sample weight degeneracy." Finite-batch ESS does not establish proposal coverage or Assumption A6.

## 2. A guarantee that actually matches Mode A

This statement should replace the informal equality chain on page 6 and the incomplete Mode-A guarantee on page 7.

Let

`F_{x,theta}(y) = phi_theta(y) + ||y-x||^2/2`,

and assume `F_{x,theta}` is `mu`-strongly convex, uniformly in `x`, with `mu=1-lambda>0`. Define

- `y*_theta(x) = argmin_y F_{x,theta}(y)`;
- `T_{epsilon,theta}(x) = E[X_1 | X_0=x]` under the bridge induced by `phi_theta`;
- `r_eta(x) = grad phi_theta(m_eta(x)) + m_eta(x) - x`;
- `hat rho_{1,theta}` as the induced bridge marginal;
- `e_fit = W_2(hat rho_{1,theta}, rho_data)`.

Strong monotonicity gives a directly auditable prox-network bound:

`||m_eta(x)-y*_theta(x)|| <= ||r_eta(x)||/mu`.

For the conditional law proportional to `exp(-F_{x,theta}(y)/(2 epsilon))`, integration by parts and strong convexity give the conservative non-asymptotic mean-mode bound

`||T_{epsilon,theta}(x)-y*_theta(x)|| <= sqrt(2 epsilon d / mu)`.

Therefore

`(E ||m_eta(X_0)-T_{epsilon,theta}(X_0)||^2)^(1/2)`

`<= (E ||r_eta(X_0)||^2)^(1/2)/mu + sqrt(2 epsilon d / mu)`.

Coupling the conditional mean to the actual bridge endpoint and then adding model-fit error yields

`W_2((m_eta)#rho_0, rho_data)`

`<= e_prox + e_mean-mode + e_kernel + e_fit`,

where

- `e_prox = (E ||r_eta(X_0)||^2)^(1/2)/mu`;
- `e_mean-mode = (E ||T_{epsilon,theta}(X_0)-y*_theta(X_0)||^2)^(1/2)`;
- `e_kernel = (E tr Cov(X_1|X_0))^(1/2)`;
- `e_fit = W_2(hat rho_{1,theta},rho_data)`.

Under uniform `mu`-strong convexity, the first two stochastic terms admit the conservative bounds

`e_mean-mode <= sqrt(2 epsilon d/mu)` and `e_kernel <= sqrt(2 epsilon d/mu)`.

Hence a simple headline corollary is

`W_2((m_eta)#rho_0,rho_data)`

`<= (E ||r_eta||^2)^(1/2)/mu + 2 sqrt(2 epsilon d/mu) + e_fit`.

This decomposition has the right interpretation:

- the residual measures learned-prox error;
- the mean-mode term measures the finite-epsilon Laplace/prox approximation;
- the kernel term measures replacing a stochastic endpoint by its mean;
- the fit term measures imperfect potential training, including optimization and finite-estimator effects.

Do not collapse `e_fit` into zero merely because the infinite-dimensional objective has the correct stationary point. The implemented neural optimizer and finite-`K` estimator do not establish that stationarity.

If the authors prefer a sharper `O(epsilon)` mean-mode statement, they must give the needed uniform derivative assumptions and their dimension dependence. The present manuscript itself notes an `O(epsilon d)` Laplace correction, so an unqualified `O(epsilon)` equality is internally inconsistent.

## 3. Paste-ready replacement text

### Revised abstract

One-step generative models usually approximate a many-step sampling process. PT-Flow instead starts from a Schrödinger bridge whose potential evolution can be written as a Gaussian heat-semigroup expectation. This identity gives an exact expression for the bridge conditional mean, `T_epsilon(x)=x-grad phi_0(x)`. In the small-viscosity regime this mean is close to the proximal point of the learned data-side potential. We amortize that proximal computation with a network `m_eta`, which is also used to center an importance proposal for estimating the heat-semigroup expectation. Under explicit mode-tracking, curvature-matching, and remainder assumptions, the resulting proposal has log-weight variance `O(epsilon)`; we test those assumptions empirically through weight moments, effective sample size, and curvature diagnostics. Mode-A sampling uses one generator evaluation, but at finite viscosity it is an approximation to the bridge conditional mean rather than an exact bridge sample. The learned potential defines an analytically normalized latent-space density; numerical likelihood values are obtained with a finite-budget nested Monte Carlo estimator. On ImageNet 256x256, [insert only independently verified FID, compute, and uncertainty results, with an archived evaluation manifest].

### Revised contribution bullets

- We derive an exact heat-semigroup representation and conditional-mean identity for a Schrödinger bridge, and isolate the finite-epsilon error incurred when replacing that mean by a proximal point.
- We introduce a prox-centered, curvature-scaled importance proposal. Its `O(epsilon)` variance conclusion is conditional on explicit mode-tracking, off-diagonal curvature, and Taylor-remainder assumptions, which we evaluate empirically.
- We train an amortized prox network that supports one-pass generation and provide a final sampler bound that separates prox residual, mean-mode, stochastic-kernel, and model-fit errors.
- We define an analytically normalized latent-space density and evaluate it using a budget-controlled nested Monte Carlo procedure, reporting convergence diagnostics rather than calling the finite estimate exact.

### Replacement for the page-6 equality chain

The three objects in this construction are related but not identical. The bridge conditional mean is `T_{epsilon,theta}(x)=x-grad phi_{0,theta}(x)`. The proximal point `y*_theta(x)=prox_{phi_theta}(x)` approximates this mean in a small-viscosity regime. The deployed sampler `m_eta(x)` is an amortized approximation to `y*_theta(x)`. Thus

`m_eta - T_{epsilon,theta} = (m_eta-y*_theta) + (y*_theta-T_{epsilon,theta})`.

Under `mu`-strong convexity, the first term is bounded by the measurable residual `||grad phi_theta(m_eta)+m_eta-x||/mu`; the second is a finite-viscosity mean-mode error. We report both rather than identifying the learned sampler with the exact bridge mean.

### Replacement for the naive-Monte-Carlo conclusion on page 5

Under the local Taylor and log-normal approximations of Proposition 3.7, the naive proposal can require a sample count exponential in a transport-to-viscosity ratio. Together with the finite-viscosity dispersion of the deterministic mean map, this predicts a severe practical tension in high dimension. These calculations do not prove that the feasible set is empty for every admissible potential or estimator; they motivate changing the proposal and are tested empirically in Section 4.

### Replacement qualification for Theorem 3.9

The variance reversal is conditional. A diagonal proposal removes the diagonal quadratic term, but an arbitrary Transformer Hessian can retain an off-diagonal contribution `0.5 ||B_x||_F^2`. The `O(epsilon)` rate therefore requires `||B_x||_F^2=O(epsilon)` together with the stated remainder control. We do not assume that architecture alone enforces this. We report the measured log-weight variance, chi-squared weight moment, ESS, and an HVP-based estimate of the off-diagonal term across epsilon.

### Replacement implementation paragraph for generator optimization

The released implementation uses a stop-gradient fixed-point update by default. It evaluates `g=grad phi_theta(m_eta)` at a detached generator output, forms the detached target `x-g`, and regresses `m_eta` toward that target. This avoids differentiating through `grad phi_theta` and therefore avoids the Hessian-vector product required by the full derivative of the squared residual. We include the full second-order variant as an ablation and report its wall-clock and memory cost. The scale model is a separate network trained by a reparameterized reverse-KL objective using additional potential evaluations; it is not a free Hessian head sharing the generator pass.

### Replacement sampling and likelihood paragraph

Mode A returns `m_eta(x_0,c,w)` in one generator evaluation. It is an amortized prox sampler and is not an exact draw from the finite-epsilon bridge. Mode B iteratively reduces the prox objective and is likewise not guaranteed to improve FID. Mode C uses finite-budget self-normalized importance resampling and converges to the conditional bridge kernel as its budget grows under coverage; finite `K` is approximate. The model marginal `hat rho_{1,theta}` is normalized analytically, but evaluating `log hat rho_{1,theta}(x)` requires nested expectations. Replacing the inner expectation by an unbiased estimate and then taking its reciprocal, outer average, and logarithm does not yield an unbiased estimate or a general IWAE lower bound. We therefore report nested-Monte-Carlo estimates over a two-dimensional budget grid with repeated seeds and label them latent-space NLL estimates.

### Revised limitations paragraph

PT-Flow's theory is local and conditional: the small-viscosity approximation requires dimension-aware control, while the variance reversal additionally requires accurate mode tracking and small off-diagonal curvature mismatch. The current implementation uses unconstrained Transformer potentials, a detached prox update, and a separately trained scale model, so these assumptions are monitored rather than guaranteed. One-pass samples approximate the bridge conditional mean and do not exactly follow the bridge marginal at finite viscosity. Likelihood evaluation is expensive nested Monte Carlo in the VAE latent space, not pixel-space bits per dimension. Finally, the strongest ImageNet comparisons require matched, independently reproducible training and evaluation artifacts.

## 4. Epsilon regime: the current values do not satisfy the stated asymptotic window

The ImageNet latent dimension is `d=32*32*4=4096`. The manuscript says the Hopf-Lax regime requires `epsilon d -> 0` and that `epsilon` should be at most on the order of `1/d`.

| Source | Reported epsilon | `epsilon*d` at `d=4096` |
|---|---:|---:|
| Table 1 / Appendix M default | 0.10 | 409.6 |
| Table 1 sweep floor | 0.02 | 81.92 |
| Released L/XL config floor | 0.005 | 20.48 |
| Algorithm floor cap | 0.001 | 4.096 |
| `1/d` | 0.000244 | 1.0 |

None of the reported ImageNet settings is in a regime where `epsilon d` is small. The paper has two defensible options:

1. Present the asymptotic results as qualitative motivation and explicitly state that ImageNet operates outside their proven regime; or
2. Reformulate the theory with a dimension-normalized geometry that matches the code, then re-derive every heat kernel, prox objective, covariance, and variance rate under that normalization.

The first option is much safer for this submission. Do not claim that Algorithm 1 "keeps epsilon inside the feasible window at every dimension" unless the actual run uses that rule and the resulting value is logged.

The final paper should have one epsilon table containing: `epsilon_max`, `epsilon_min`, schedule, achieved final epsilon, per-coordinate proposal variance `2 epsilon`, latent dimension, `epsilon*d`, and whether the run is theoretical/toy or empirical/high-dimensional.

## 5. Experiments that directly test the mechanism

The repository contains launch code for much of this work, but no completed measurements. The paper should distinguish ready-to-run infrastructure from reported evidence.

### A. Proposal-component study

Use the same checkpoint, same `x_0`, same labels, and common random numbers to compare:

1. naive: center `x_0`, identity scale;
2. recentered only: center `m_eta`, identity scale;
3. diagonal: center `m_eta`, learned diagonal scale, no defensive component;
4. defensive diagonal: the full proposal.

Report, not just FID:

- normalized ESS and control ESS;
- `Var(log w)`;
- `chi2_hat = mean(w^2)/mean(w)^2 - 1`;
- estimated relative variance `chi2_hat/K`;
- the predicted budget `ceil(chi2_hat/delta^2)` for several `delta` values;
- failure/nonfinite rate;
- training time and peak memory.

Run this first as a fixed-checkpoint estimator study to isolate the proposal mechanism, then as an end-to-end training ablation to show downstream FID effects. Retraining every proposal arm alone does not isolate estimator variance.

### B. Variance versus epsilon

At fixed checkpoints, sweep at least `epsilon in {0.2,0.1,0.05,0.02,0.01,0.005}` and plot the metrics above for all four proposals. Add an HVP/Hutchinson estimate of

`0.5 ||S^(1/2) (I+Hessian(phi)) S^(1/2) - I||_F^2`.

The theory predicts an `O(epsilon)` trend only if this off-diagonal/mismatch floor shrinks accordingly. Fit slopes with uncertainty rather than drawing the theoretical line without testing the assumption.

### C. Prox quality

Report the distribution, not only the mean, of

- absolute residual `||r_eta||`;
- relative residual `||r_eta||/||m_eta-x_0||`;
- objective decrease under Mode-B refinement;
- distance from Mode A to the refined solution;
- toy low-dimensional error to quadrature/Newton ground truth for both `T_epsilon` and `y*`.

The last item directly separates bridge-mean-to-prox error from prox-to-network error.

### D. Proposal mismatch stress test

Perturb the proposal center and log-scale at inference without retraining. Plot ESS/chi-squared variance against controlled center bias and scale multipliers. This tests whether the defensive component provides the claimed robustness when the learned proposal is wrong.

### E. Likelihood/NLL

If likelihood stays in the abstract or contribution list, report a real held-out experiment with:

- both inner and outer budgets varied, not one `K` ladder alone;
- at least three Monte Carlo seeds on the identical examples;
- per-example uncertainty or confidence intervals;
- toy quadrature calibration where possible;
- an explicit label: "latent-space nested-MC NLL estimate";
- no claim of monotone tightening or lower-bound direction unless separately proved for this nested reciprocal estimator.

Do not compare this number to pixel-space bits/dim. The SD-VAE decoder is lossy and does not supply an invertible change-of-variables likelihood.

## 6. Paper/implementation mismatches that must be resolved

| Topic | Manuscript | Released implementation | Required action |
|---|---|---|---|
| Generator loss | Full squared prox residual, implying differentiation through `grad phi(m_eta)` | Default `prox_mode: detach`; frozen fixed-point target; no HVP | Describe the detached update and ablate `full` mode, or change the implementation |
| Overall trainer | PT-Flow alternating scheme presented as the complete objective | W-Flow feature drift plus a gated PT auxiliary loss | State the hybrid objective prominently and include a matched W-Flow control |
| Initialization | Main text says all models train from scratch | `ptflow_L.yaml` is explicitly a W-Flow checkpoint continuation | Use a true from-scratch config/result or relabel the experiment as fine-tuning |
| Scale fit | Same two-headed generator; Hessian/Hutchinson or reused proposal points; no extra calls | Separate theta-side `ScaleNet`; reverse-KL with `scale_K` additional potential evaluations | Rewrite method and compute accounting |
| Potential parameterization | Paper says network outputs `u_theta` | Shipped configs use `output_mode: phi`; losses are rescaled in phi units | Choose one description and make equations/code/config consistent |
| Curvature enforcement | Paper says Hutchinson monitoring, hinge penalty, and architectural fallback | Main configs set `curv_probe: false` and `lambda_curv: 0`; no demonstrated fallback | Remove the claim or enable, log, and evaluate the mechanism |
| A6 monitoring | ESS said to certify off-diagonal matching | ESS only reflects sampled weights; it cannot prove mode coverage or isolate off-diagonal Hessian error | Call ESS a diagnostic and report direct HVP estimates |
| Mode A | "Exact prox sampler of the exact bridge" | One call to learned `m_eta` | Call it amortized one-pass sampling |
| Mode C | "SNIS exact, K-NFE" | Finite self-normalized resampling | Call it asymptotically exact/consistent as `K` grows under coverage |
| Likelihood | Exact per-sample likelihood / IWAE-style bound | Nested plug-in estimator with reciprocal and outer log | Report as nested MC with no certified finite-budget bias direction |
| Computational cost | Scale matching described as free | Separate scale model and extra potential evaluations add training compute | Report measured training FLOPs/time/memory |

## 7. Numerical and textual consistency ledger

Resolve every row before submission.

| Item | Conflict |
|---|---|
| Epsilon floor | Table 1 / Appendix M: 0.10; Algorithm 1: `min(1e-3,c/(dM3^2))`; code: 0.005 |
| Epsilon maximum | Algorithm/Appendix: 0.2; L/XL code: 0.1 |
| Defensive weight | Appendix default: 0.1; L/XL code: 0.05 -> 0.01, with degraded value 0.25 |
| Batch size | Table 1 and comparison prose: 256; Table 4: 128 with accumulation 8; config also has dataset batch 512 and train batch 128 |
| Iterations | Table 4: 160k; L/XL config: 200k |
| Training origin | Main text: from scratch; L config: resume a converged W-Flow checkpoint |
| Throughput | Page 8 prose: 161.3 images/s; Table 5: 165.28 images/s |
| Parameter count | Table 2 says counts include generator+decoder, but PT-Flow is listed as 462M while Table 4 gives 462.91M trainable model parameters before clarifying decoder/potential/scale inclusion |
| Likelihood cost | Related work says likelihood at a single function evaluation; the numerical estimator uses two Monte Carlo layers and many potential evaluations |
| Variance language | Main text presents reversal categorically; theorem requires A6 and a remainder bound not enforced by the main config |
| NLL language | Main text and evaluation shell comments say IWAE/exact; `AUDIT.md`, `RUN_GUIDE.md`, and current CLI correctly say nested MC without a certified bound |

Every table should state whether batch size is per device, global per optimizer step, or effective after accumulation. Every throughput result should state exact GPU, precision, batch, inclusion of VAE decoding, warm-up, synchronization, number of repetitions, and code commit/hash.

## 8. Reference corrections

The reference list contains two entries for the same W-Flow arXiv identifier `2605.11755`. Keep only the entry whose author list matches the [official arXiv record](https://arxiv.org/abs/2605.11755): Jiaqi Han, Puheng Li, Qiushan Guo, Renyuan Xu, Stefano Ermon, and Emmanuel J. Candès, "One-Step Generative Modeling via Wasserstein Gradient Flows" (2026). Remove the conflicting Lantao Han et al. entry and use one citation key everywhere.

The VarEOT entry also has incorrect first names in the draft. The [official arXiv record](https://arxiv.org/abs/2602.02241) lists Roman Dyachenko and Kirill Sokolov, not Daniil Dyachenko and Aleksei Sokolov. Regenerate the bibliography from canonical metadata rather than editing rendered references by hand.

For all 2025-2026 baselines, freeze a comparison ledger containing paper version/date, architecture, parameter-count convention, NFE convention (especially CFG doubling), FID protocol, and source URL. Do not mix results from different revisions of a preprint.

## 9. Evidence required for the headline ImageNet result

The current workspace does not contain evidence for the reported PT-Flow FID 1.54 or H100 throughput. Before those numbers appear in a submission, archive:

- the exact checkpoint and its checksum;
- the resolved training config saved by the run;
- source hash/commit and environment lock;
- training logs showing actual epsilon, alpha, prox weight, ESS, and residual trajectories;
- raw 50k-sample FID result, sample count, seed, and reference-stat checksum;
- the generated feature moments or a reproducible evaluation artifact;
- at least three training/evaluation seeds for a new SOTA-level claim;
- the raw H100 benchmark output with warm-up and synchronization details;
- a matched W-Flow/hybrid baseline under the same code and compute budget.

Without that package, remove the numerical superiority claim and present the work as a method/theory paper with preliminary experiments.

## 10. Suggested response to the reviewer

Thank you for identifying the places where the draft conflated the exact bridge identities with their learned and finite-sample approximations. We agree that this distinction is central. In the revision we will (i) separate the exact conditional mean, the finite-viscosity proximal approximation, and the amortized network, and give a final error decomposition containing prox-residual, mean-mode, stochastic-kernel, and model-fit terms; (ii) state explicitly that the ImageNet regime is outside the paper's `epsilon d -> 0` asymptotic window unless a dimension-normalized theory is supplied; (iii) make the variance-reversal result conditional on measurable curvature-mismatch and remainder assumptions, while softening the naive-Monte-Carlo impossibility language; (iv) describe the default stop-gradient generator update and the actual scale-network cost; and (v) add proposal-component, epsilon/ESS, prox-quality, mismatch, and nested-MC likelihood studies. We will also replace "exact one-step generation" and "exact per-sample likelihood" with language that distinguishes analytic normalization from finite-budget computation, and complete a configuration, numerical, and bibliography audit before submission.

This response should be sent only after the promised experiments and consistency fixes are actually in the manuscript.

## 11. Recommended revision order

1. Freeze the actual algorithm and decide whether the paper describes the released hybrid implementation or a different clean implementation.
2. Replace the claim taxonomy, theorem chain, sampling language, and likelihood language.
3. Resolve epsilon definitions and state honestly that current ImageNet settings are outside the asymptotic window.
4. Run fixed-checkpoint mechanism diagnostics before expensive retraining.
5. Run matched end-to-end proposal and PT-on/PT-off experiments with multiple seeds.
6. Run the two-dimensional likelihood budget study or remove likelihood from the headline contributions.
7. Rebuild every table from an experiment manifest; then run the reference and number audit.

The strongest survivable paper is not "everything is exact." It is: an exact bridge identity motivates a practical amortized one-pass sampler; proximal tilting makes the required expectation estimable under explicit, testable conditions; and the gap between the mathematics and the implemented sampler is measured rather than hidden.
