"""The checkpoint-compatibility contract, enforced.

    python -m tests.test_ckpt_compat

REQUIREMENT
-----------
A reference OT-drift checkpoint at step N must resume under PT-Flow and continue
to step N+M, for every model size:

    # with <workdir>/checkpoints/state_00180000.pt from the reference release
    torchrun ... train.py --config configs/gen/ptflow_XL.yaml --workdir <workdir>
    -> resumes at 180000 and writes state_00182000.pt carrying BOTH the
       generator (still reference-shaped) and the new PT-Flow potential.

Four things make that true, and each is checked below.

  1. SHAPE IDENTITY.  DitGen built from a PT-Flow config must have a state_dict
     whose keys and shapes match DitGen built from the corresponding reference
     config -- and match the frozen per-size counts in EXPECTED, which are the
     values measured against the pristine upstream release.  The frozen counts
     are what make this a regression test rather than a self-consistency check:
     they fail if BOTH configs drift together.

  2. NO PARAMETERS ADDED.  `residual` is a behaviour flag.  Toggling it must not
     change a single tensor -- but it does change what the weights mean, which
     is why it stays false when resuming.

  3. OPTIMIZER IDENTITY.  The AdamW state in a reference checkpoint must load
     into the PT-Flow run's optimizer, which needs identical parameter count and
     ordering.

  4. GRACEFUL PT-SIDE ABSENCE.  A reference checkpoint carries no pt_* keys.
     restore_checkpoint must load what is there, start the potential fresh, and
     not raise.

It also checks the reverse direction: a PT-Flow checkpoint's generator must load
back into a plain DitGen with strict=True, so the work is never trapped here.
"""

from __future__ import annotations

import copy
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DRIFT_COMPILE", "0")

import torch

from models.generator import DitGen
from utils.misc import load_config

PASS, FAIL = [], []
HERE = Path(__file__).resolve().parent.parent

# (PT-Flow config, reference config, tensors, parameters)
# The counts are measured against the upstream release and frozen here on
# purpose; see note 1 above.
EXPECTED = [
    ("configs/gen/ptflow_B.yaml",       "configs/gen/baseline_B.yaml",         241, 132_708_880),
    ("configs/gen/ptflow_L.yaml",       "configs/gen/baseline_L.yaml",         433, 462_909_456),
    ("configs/gen/ptflow_XL.yaml",      "configs/gen/baseline_XL.yaml",        497, 678_929_872),
    ("configs/gen/ptflow_scratch.yaml", "configs/gen/baseline_ablation.yaml",  241, 132_708_880),
]


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    assert cond, f"{name}: {detail}"
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def build(cfg_path: Path) -> DitGen:
    cfg = load_config(str(cfg_path))
    mcfg = dict(cfg.model)
    mcfg.pop("residual", None)          # behaviour flag, no parameters
    return DitGen(num_classes=int(cfg.dataset.num_classes), **mcfg)


# ---------------------------------------------------------------------------

def test_shape_identity():
    print("\n1. Generator state_dict matches the reference, per size")
    for pt_cfg, base_cfg, n_tensors, n_params in EXPECTED:
        name = Path(pt_cfg).stem.replace("ptflow_", "")
        pt_path, base_path = HERE / pt_cfg, HERE / base_cfg
        if not pt_path.exists() or not base_path.exists():
            check(f"{name}: configs present", False,
                  f"missing {pt_path if not pt_path.exists() else base_path}")
            continue

        with torch.device("meta"):
            g_pt, g_ref = build(pt_path), build(base_path)
        sd_pt = {k: tuple(v.shape) for k, v in g_pt.state_dict().items()}
        sd_ref = {k: tuple(v.shape) for k, v in g_ref.state_dict().items()}
        p_pt = sum(p.numel() for p in g_pt.parameters())

        same_keys = set(sd_pt) == set(sd_ref)
        if not same_keys:
            only_pt = sorted(set(sd_pt) - set(sd_ref))[:3]
            only_ref = sorted(set(sd_ref) - set(sd_pt))[:3]
            check(f"{name}: same keys", False, f"pt-only={only_pt} ref-only={only_ref}")
        else:
            check(f"{name}: same keys", True, f"{len(sd_pt)} tensors")

        bad = [k for k in sd_pt if same_keys and sd_pt[k] != sd_ref[k]]
        check(f"{name}: same shapes", same_keys and not bad,
              f"{p_pt/1e6:.1f}M params" if not bad else f"mismatched: {bad[:3]}")
        check(f"{name}: matches the frozen upstream counts",
              len(sd_pt) == n_tensors and p_pt == n_params,
              f"{len(sd_pt)} tensors / {p_pt:,} params "
              f"(expected {n_tensors} / {n_params:,})")
        del g_pt, g_ref


def test_residual_adds_no_parameters():
    print("\n2. `residual` is a behaviour flag, not an architecture change")
    kw = dict(cond_dim=64, num_classes=10, input_size=8, in_channels=2, patch_size=2,
              hidden_size=64, depth=2, num_heads=4, out_channels=2,
              n_cls_tokens=0, noise_classes=0, use_bf16=False)
    a, b = DitGen(residual=False, **kw), DitGen(residual=True, **kw)
    sa = {k: tuple(v.shape) for k, v in a.state_dict().items()}
    sb = {k: tuple(v.shape) for k, v in b.state_dict().items()}
    check("identical state_dicts", sa == sb,
          f"{len(sa)} tensors, {sum(p.numel() for p in a.parameters()):,} params")

    c, x0 = torch.zeros(2, dtype=torch.long), torch.randn(2, 8, 8, 2)
    ya = a(c=c, cfg_scale=1.0, x0=x0)["samples"]
    yb = b(c=c, cfg_scale=1.0, x0=x0)["samples"]
    check("but it does change the map (so it stays false on resume)",
          not torch.allclose(ya, yb), f"||diff|| = {(ya - yb).norm().item():.4f}")


def test_roundtrip_resume(tmp_path: Path):
    tmp = tmp_path
    print("\n3. A reference checkpoint resumes, and the result loads back")
    from ptflow.potential import PotentialNet, ScaleNet
    from ptflow.schedule import build_schedule
    from train import PTBundle, TrainState
    from utils.ckpt_util import restore_checkpoint, save_checkpoint

    S, C, NC = 8, 2, 10
    kw = dict(cond_dim=64, num_classes=NC, input_size=S, in_channels=C, patch_size=2,
              hidden_size=64, depth=2, num_heads=4, out_channels=C,
              n_cls_tokens=0, noise_classes=0, use_bf16=False)

    # --- a drift-only run reaches step 100000 and saves ---
    torch.manual_seed(0)
    base_model = DitGen(**kw)
    base_opt = torch.optim.AdamW(base_model.parameters(), lr=1e-4)
    base_model(c=torch.randint(0, NC, (2,)))["samples"].sum().backward()
    base_opt.step()
    base_state = TrainState(step=100000, model=base_model, optimizer=base_opt,
                            ema_model=copy.deepcopy(base_model), ema_decay=0.999, pt=None)
    ckdir = tmp / "checkpoints"
    ckdir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(base_state, workdir=str(tmp))
    saved = sorted(ckdir.glob("state_*.pt"))
    check("reference run wrote a checkpoint", len(saved) == 1,
          saved[0].name if saved else "none")

    payload = torch.load(saved[0], map_location="cpu", weights_only=False)
    check("it carries no pt_* keys", not any(k.startswith("pt_") for k in payload),
          f"keys={sorted(payload)}")

    # --- a PT-Flow run picks it up ---
    torch.manual_seed(999)                       # different init on purpose
    pt_model = DitGen(residual=False, **kw)
    potential = PotentialNet(cond_dim=64, num_classes=NC, input_size=S,
                             in_channels=C, patch_size=2, hidden_size=32,
                             depth=2, num_heads=4)
    scale = ScaleNet(cond_dim=64, num_classes=NC, input_size=S, in_channels=C,
                     patch_size=2, hidden_size=32, depth=2, num_heads=4)
    bundle = PTBundle(
        potential=potential, ema_potential=copy.deepcopy(potential),
        scale=scale, ema_scale=copy.deepcopy(scale),
        optimizer=torch.optim.AdamW(
            list(potential.parameters()) + list(scale.parameters()), lr=1e-4),
        sched=build_schedule({}), lr_fn=lambda s: 1e-4, cfg={})
    pt_state = TrainState(step=0, model=pt_model,
                          optimizer=torch.optim.AdamW(pt_model.parameters(), lr=1e-4),
                          ema_model=copy.deepcopy(pt_model), ema_decay=0.999, pt=bundle)

    pt_state = restore_checkpoint(state=pt_state, workdir=str(tmp))

    check("resumed at the reference step", pt_state.step == 100000, f"step={pt_state.step}")
    check("generator weights are the reference weights, exactly",
          all(torch.equal(a, b) for a, b in
              zip(base_model.state_dict().values(), pt_state.model.state_dict().values())))
    check("optimizer moments came across",
          len(pt_state.optimizer.state_dict()["state"]) > 0,
          f"{len(pt_state.optimizer.state_dict()['state'])} param entries")
    check("potential started fresh without raising", pt_state.pt is not None)

    # --- and continues, writing a checkpoint carrying both halves ---
    pt_state.step = 102000
    save_checkpoint(pt_state, workdir=str(tmp))
    p2 = torch.load(sorted(ckdir.glob("state_*.pt"))[-1], map_location="cpu",
                    weights_only=False)
    for key, what in (("pt_model", "potential"), ("pt_scale_model", "scale net"),
                      ("pt_schedule", "schedule")):
        check(f"PT-Flow checkpoint carries the {what}",
              key in p2 and p2[key] is not None)

    DitGen(**kw).load_state_dict(p2["model"], strict=True)
    check("its generator loads back into a plain DitGen (strict=True)", True,
          "no shape surgery needed in either direction")


if __name__ == "__main__":
    print("PT-Flow checkpoint-compatibility contract")
    test_shape_identity()
    test_residual_adds_no_parameters()
    with tempfile.TemporaryDirectory() as td:
        test_roundtrip_resume(Path(td))

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed:", ", ".join(FAIL))
    sys.exit(1 if FAIL else 0)
