# PT-Flow on Kaggle — copy each `# %%` block into its own notebook cell.
#
# Kaggle cannot train PT-Flow on ImageNet-256: that needs the ~100 GB SD-VAE
# latent cache and multi-GPU-days.  What it CAN do, and what these cells do, is
# everything that actually validates the method:
#
#   1  environment + code
#   2  the CPU math suite            (correctness against closed forms)
#   3  the training smoke suite      (the graft, incl. the baseline bit-identity)
#   4  the R0 variance-reversal sweep, with the plot        <- the paper's own
#   5  full 2-D PT-Flow training, with the ESS/eps/lambda traces
#   6  A-vs-B-vs-C ablation over lambda_prox
#   7  (needs downloads, GPU) warm-start from a released baseline checkpoint and
#      render sample grids for Modes A / B / C
#
# Cells 1-6 run on a Kaggle CPU instance.  Cell 7 needs a GPU + internet on.

# %% [1] Environment and code -------------------------------------------------
# Option A: the repo is attached as a Kaggle Dataset.  Set the path and go.
# Option B: upload the folder, or `git clone` it if internet is enabled.

import os, subprocess, sys

REPO = "/kaggle/input/ptflow-baseline/the OT-drift baseline-main"   # <-- EDIT ME
WORK = "/kaggle/working/ptflow"

if not os.path.isdir(REPO):
    raise SystemExit(f"Set REPO to the checked-out repo. Not found: {REPO}")

subprocess.run(["cp", "-r", REPO, WORK], check=True)
os.chdir(WORK)
sys.path.insert(0, WORK)
os.environ["DRIFT_COMPILE"] = "0"     # keep startup fast; compile buys nothing here

import torch
print("python", sys.version.split()[0])
print("torch ", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
print("cwd   ", os.getcwd())


# %% [2] Correctness: 38 checks against closed forms --------------------------
# The quadratic-potential case has every object in the paper available exactly:
#   psi_0, phi_0, prox, and S^-1 = (1+tau) I.  If these pass, the estimator is
#   right on this build of PyTorch.
!python -m tests.test_math


# %% [3] The graft: 37 checks on the training step ----------------------------
# The load-bearing one is "generator parameters after one step are
# bit-identical": with lambda_prox = 0 the run must BE the OT-drift baseline, not merely
# resemble it.
!python -m tests.test_train_smoke


# %% [4] R0: the variance reversal, plotted -----------------------------------
# Theorem 2.9's claim, which is the whole reason the method is feasible:
#   naive   s^2 ~ |grad phi|^2 / 2 eps    -> diverges as eps cools
#   tilted  s^2 ~ (5/6) d M_3^2 eps       -> vanishes as eps cools
import math, torch, matplotlib.pyplot as plt
from ptflow.estimator import naive_phi0, tilted_phi0
from ptflow.potential import phi_grad
from tests.test_math import MildlyCubicPotential

torch.manual_seed(0)
B, tau, kappa = 128, 0.5, 0.4
pot = MildlyCubicPotential(tau, kappa)
c = torch.zeros(B, dtype=torch.long)
x0 = torch.randn(B, 4, 4, 2) * 0.5

epss = [0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
naive_s, tilt_s = [], []
for eps in epss:
    m0 = x0.clone()
    for _ in range(40):                       # solve for the true prox
        g, _ = phi_grad(pot, m0, c)
        m0 = x0 - g
    s0 = -torch.log((1.0 + tau + kappa * m0).clamp_min(1e-3))   # S^-1 = I + H
    naive_s.append(naive_phi0(pot, x0, c, eps, K=64).logw_spread.mean().item())
    tilt_s.append(tilted_phi0(pot, x0, c, m0, s0, eps, K=64,
                              alpha_def=0.0, logw_clip=0.0).logw_spread.mean().item())
    print(f"  eps={eps:<7g} naive={naive_s[-1]:8.3f}   tilted={tilt_s[-1]:.5f}")

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.loglog(epss, naive_s, "o-", label=r"naive  $\mathcal{N}(x_0, 2\varepsilon I)$")
ax.loglog(epss, tilt_s, "s-", label="prox-tilted (Laplace-matched)")
ax.loglog(epss, [tilt_s[0] * math.sqrt(e / epss[0]) for e in epss], "k:",
          label=r"predicted $\propto\sqrt{\varepsilon}$")
ax.invert_xaxis()
ax.set_xlabel(r"$\varepsilon$  (colder $\rightarrow$)")
ax.set_ylabel("log-weight std  $s_{\\mathrm{res}}$")
ax.set_title("Variance reversal: the colder the bridge, the cheaper the estimator")
ax.legend(); ax.grid(alpha=0.3, which="both")
plt.tight_layout(); plt.show()


# %% [5] Full 2-D PT-Flow training --------------------------------------------
# The real modules: the baseline's ot_drift_loss for the arm, pt.* for the payload.
# ~2 min on CPU.  Watch ESS stay above 0.3 and eps reach its floor.
!python -m examples.toy2d --steps 3000 --batch 256 --lambda-prox 0.3 \
        --plot --out /kaggle/working/toy2d_pt

from IPython.display import Image, display
display(Image("/kaggle/working/toy2d_pt/toy2d.png"))


# %% [6] Does the PT term help, hurt, or destabilize? -------------------------
# THE experiment.  lambda_prox = 0 is the baseline; everything else is
# measured as a delta against it.  Energy distance, lower is better.
import json, subprocess

rows = []
for lam in (0.0, 0.1, 0.3, 1.0):
    out = f"/kaggle/working/toy2d_lam{lam}"
    subprocess.run([sys.executable, "-m", "examples.toy2d",
                    "--steps", "3000", "--batch", "256",
                    "--lambda-prox", str(lam), "--out", out], check=True)
    r = json.load(open(f"{out}/results.json")); r["lambda_prox"] = lam
    rows.append(r)

print(f"{'lambda_prox':>12} {'Mode A':>10} {'Mode B':>10} {'Mode C':>10} {'ESS':>8}")
for r in rows:
    print(f"{r['lambda_prox']:>12.2f} {r['mode_A']:>10.5f} {r['mode_B']:>10.5f} "
          f"{r['mode_C']:>10.5f} {r['final_ess']:>8.3f}")

print("\nRead it this way:")
print("  * Mode A must not get WORSE as lambda_prox rises.  If it does, the PT")
print("    term is fighting the OT arm -- lower lambda_prox or lengthen the ramp.")
print("  * Mode B only helps when the prox identity m = prox_phi actually holds.")
print("    Check pt/prox_resid_rel in the history; if it is above ~0.5, Mode B")
print("    will drag good samples toward a prox the potential has not learned.")
print("  * Mode C is the safe extra-NFE mode: it reweights rather than moves.")


# %% [7] GPU: verify the baseline -> PT-Flow handoff ---------------------------
# Needs internet ON.  ~1 GB download.  No training -- this proves the resume
# path: a released baseline checkpoint loads into the PT-Flow generator with
# strict=True, because DitGen is shape-identical by construction.
from huggingface_hub import hf_hub_download

# The upstream repo id and checkpoint names live only in utils/env.py.
from utils.env import BASELINE_CKPTS, BASELINE_HF_REPO_ID

_dir, _step = BASELINE_CKPTS["B"]
CKPT = hf_hub_download(BASELINE_HF_REPO_ID,
                       f"checkpoints/{_dir}/state_{_step}.pt",
                       local_dir="/kaggle/working/baseline_hf")

import torch
from models.generator import DitGen
from ptflow.convert import check_resume_compatible, inspect_checkpoint
from utils.misc import load_config

print(inspect_checkpoint(CKPT))

cfg = load_config("configs/gen/ptflow_B.yaml")
mcfg = dict(cfg.model); mcfg.pop("residual", None)
gen = DitGen(num_classes=cfg.dataset.num_classes, **mcfg)

ok, msg = check_resume_compatible(gen, CKPT, strict=False)
print(("OK  " if ok else "FAIL") + " " + msg)

sd = torch.load(CKPT, map_location="cpu", weights_only=False)["ema_model"]
gen.load_state_dict(sd, strict=True)      # strict=True: no surgery, no slack
print("loaded with strict=True -- the resume path is sound")

# %% [8] The full contract, as a test ----------------------------------------
# Shape identity for B/L/XL against the pristine upstream checkout, plus a
# simulated resume: the OT-drift baseline saves at step 100000, PT-Flow picks it up, continues,
# and its checkpoint loads back into the stock baseline.
!python -m tests.test_ckpt_compat

# %% [9] Train, resuming a real baseline checkpoint -----------------------------
# Needs the ImageNet latent cache, so this is a server cell, not a Kaggle one.
# Included here so the command lives next to the checks that justify it.
#
#   export BASELINE_HF_ROOT=/path/to/baseline_hf_root
#   SIZE=XL bash scripts/train/resume_baseline.sh
#
# Watch, in order:
#   1. the resume log line naming the step it picked up
#   2. FID at that step == the checkpoint's published FID (else stop: the
#      handoff is wrong, not PT-Flow)
#   3. pt/ess climbing as the potential learns the inherited generator
#   4. pt/lambda_prox lifting off zero only after ptflow.schedule.prox_warmup
print(open("scripts/train/resume_baseline.sh").read()[:1200])
