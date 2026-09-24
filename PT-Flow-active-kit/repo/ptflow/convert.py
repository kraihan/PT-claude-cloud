"""baseline <-> PT-Flow checkpoint interoperability.

There is deliberately almost nothing in this module, and that is the point.

PT-Flow's generator is :class:`models.generator.DitGen` with one extra
*behaviour* flag (``residual``) and no extra parameters, so its state_dict is
shape-identical to the reference release.  A the OT-drift baseline ``state_*.pt`` therefore needs
no conversion at all: drop it into ``<workdir>/checkpoints/`` and
``utils.ckpt_util.restore_checkpoint`` loads the generator, its EMA, the AdamW
moments and the step count, then starts the PT-Flow potential fresh at that
step.  Training continues from N to N+M.

An earlier revision of this port put the diagonal log-scale head on the
generator trunk, which doubled the final layer's output channels and needed
weight surgery to load a release checkpoint.  That head now lives in
:class:`ptflow.potential.ScaleNet` on the theta side.  The surgery is gone, and so
is the class of bug that came with it.

What remains is a preflight check, because the failure mode of getting this
wrong is a silently mis-initialized run rather than an exception.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch

from utils.logging import log_for_0
from utils.ckpt_util import canonical_state_dict, check_model_behavior


def inspect_checkpoint(ckpt_path: str) -> Dict[str, object]:
    """Report what a checkpoint contains, without loading it into a model."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    gen = ck.get("ema_model") or ck.get("model") or {}
    return {
        "path": str(ckpt_path),
        "step": ck.get("step", -1),
        "kind": "pt-flow" if "pt_model" in ck else "baseline",
        "keys": sorted(ck.keys()),
        "generator_tensors": len(gen),
        "generator_params": int(sum(v.numel() for v in gen.values())),
        "has_potential": "pt_model" in ck,
        "has_scale_net": "pt_scale_model" in ck,
        "has_schedule": "pt_schedule" in ck,
    }


def check_resume_compatible(model, ckpt_path: str, *, strict: bool = True) -> Tuple[bool, str]:
    """Verify that ``ckpt_path``'s generator matches ``model`` exactly.

    Call this before a long run.  Returns ``(ok, message)``; with ``strict`` it
    raises instead, because a shape mismatch discovered at step 0 of an 8-GPU
    job is cheap and one discovered later is not.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("ema_model") or ck.get("model")
    if sd is None:
        msg = f"{ckpt_path} contains neither 'model' nor 'ema_model'."
        if strict:
            raise ValueError(msg)
        return False, msg

    want = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    try:
        check_model_behavior(model, ck)
    except ValueError as exc:
        if strict:
            raise
        return False, str(exc)
    sd = canonical_state_dict(sd)
    have = {k: tuple(v.shape) for k, v in sd.items()}

    missing = sorted(set(want) - set(have))
    extra = sorted(set(have) - set(want))
    mismatch = sorted(k for k in set(want) & set(have) if want[k] != have[k])

    if missing or extra or mismatch:
        parts = []
        if missing:
            parts.append(f"missing {len(missing)} (e.g. {missing[:3]})")
        if extra:
            parts.append(f"unexpected {len(extra)} (e.g. {extra[:3]})")
        if mismatch:
            parts.append(
                "shape mismatch on "
                + ", ".join(f"{k}: ckpt{have[k]} vs model{want[k]}" for k in mismatch[:3])
            )
        msg = (
            f"{Path(ckpt_path).name} is NOT resume-compatible with this config: "
            + "; ".join(parts)
            + ".  The usual cause is a `model:` block that does not match the "
            "the OT-drift baseline config the checkpoint came from -- compare against "
            "configs/gen/baseline_{B,L,XL}.yaml."
        )
        if strict:
            raise ValueError(msg)
        return False, msg

    n = sum(v.numel() for v in model.state_dict().values())
    msg = (
        f"{Path(ckpt_path).name} (step {ck.get('step', '?')}, "
        f"{'PT-Flow' if 'pt_model' in ck else 'the OT-drift baseline'}) is resume-compatible: "
        f"{len(want)} tensors, {n:,} params."
    )
    log_for_0("%s", msg)
    return True, msg


def load_baseline_into_generator(generator, ckpt_path: str, *, prefer_ema: bool = True) -> None:
    """Load a baseline generator checkpoint into a PT-Flow generator.

    Only needed to *initialize* a fresh workdir from a release checkpoint.  For
    an ordinary resume, do nothing special: put the file in
    ``<workdir>/checkpoints/`` and let restore_checkpoint handle it, which also
    carries over the optimizer moments and the step count.
    """
    if getattr(generator, "residual", False):
        raise ValueError(
            "Refusing to load the OT-drift baseline weights into a residual generator: the "
            "release was trained as m(x) = net(x), not x + net(x).  Set "
            "model.residual: false (as configs/gen/ptflow_*.yaml do)."
        )
    check_resume_compatible(generator, ckpt_path, strict=True)

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = (ck.get("ema_model") if prefer_ema else None) or ck.get("model", ck)
    generator.load_state_dict(canonical_state_dict(sd), strict=True)
    log_for_0("Initialized generator from %s (step %s)", ckpt_path, ck.get("step", "?"))
