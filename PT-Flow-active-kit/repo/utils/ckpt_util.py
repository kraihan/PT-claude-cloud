from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from utils.dist_util import barrier, unwrap_ddp
from utils.logging import is_rank_zero, log_for_0


def canonical_state_dict(state_dict):
    """Accept older checkpoints saved through torch.compile wrappers."""
    return {".".join(p for p in k.split(".") if p != "_orig_mod"): v for k, v in state_dict.items()}


def check_model_behavior(model, payload):
    saved = payload.get("model_behavior", {})
    for key, value in saved.items():
        if getattr(unwrap_ddp(model), key) != value:
            raise ValueError(f"Checkpoint {key}={value!r} differs from model config. Tensor shapes alone do not establish compatibility.")


def _to_python_int(x) -> int:
    if torch.is_tensor(x):
        return int(x.detach().cpu().reshape(-1)[0].item())
    return int(x)


def _output_root(workdir: Optional[str] = None) -> Path:
    if workdir:
        return Path(workdir).resolve()
    return Path("runs").resolve()


def _job_ckpt_dir(workdir: Optional[str] = None) -> Path:
    return _output_root(workdir) / "checkpoints"


def _list_ckpts(ckpt_dir: Path) -> list[Path]:
    return sorted(ckpt_dir.glob("state_*.pt"))


def _extract_step(path: Path) -> int:
    stem = path.stem
    return int(stem.split("_")[-1])


def _restore_optimizer(optimizer, saved):
    # Fused/foreach are execution choices, not optimizer moments. Set these
    # BEFORE loading so PyTorch places step tensors on the proper device too.
    groups = [{**group, "fused": optimizer.defaults.get("fused", False),
               "foreach": optimizer.defaults.get("foreach", None)} for group in saved["param_groups"]]
    optimizer.load_state_dict({**saved, "param_groups": groups})


def restore_checkpoint(step=None, state=None, workdir: Optional[str] = None, checkpoint=None):
    ckpt_dir = _job_ckpt_dir(workdir=workdir)
    if not checkpoint and not ckpt_dir.exists():
        log_for_0("No local checkpoint dir at %s", str(ckpt_dir))
        return state

    if step is not None:
        step = int(step)

    ckpts = _list_ckpts(ckpt_dir)
    if not checkpoint and not ckpts:
        return state

    if checkpoint:
        target_path = Path(checkpoint)
        if not target_path.is_file():
            raise FileNotFoundError(target_path)
    elif step is None:
        target_path = ckpts[-1]
    else:
        cand = [p for p in ckpts if _extract_step(p) == step]
        if not cand:
            return state
        target_path = cand[-1]

    payload = torch.load(target_path, map_location="cpu", weights_only=False)
    if state is None:
        return payload

    check_model_behavior(state.model, payload)
    unwrap_ddp(state.model).load_state_dict(canonical_state_dict(payload["model"]), strict=True)
    if (
        getattr(state, "ema_model", None) is not None
        and "ema_model" in payload
        and payload["ema_model"] is not None
    ):
        unwrap_ddp(state.ema_model).load_state_dict(canonical_state_dict(payload["ema_model"]), strict=True)
    _restore_optimizer(state.optimizer, payload["optimizer"])
    if getattr(state, "scaler", None) is not None and payload.get("grad_scaler"):
        state.scaler.load_state_dict(payload["grad_scaler"])
    state.step = int(payload.get("step", 0))
    state.ema_decay = float(payload.get("ema_decay", getattr(state, "ema_decay", 0.999)))
    for decay, ema in getattr(state, "extra_emas", {}).items():
        weights = payload.get("extra_emas", {}).get(decay)
        if weights is None:
            weights = state.ema_model.state_dict()
            log_for_0("EMA %s starts from the restored primary EMA at step %d", decay, state.step)
        ema.load_state_dict(canonical_state_dict(weights), strict=True)

    # -- PT-Flow: potential, scale net, their EMAs, optimizer, schedule.
    #
    # The schedule *must* be restored: eps_progress and prox_progress advance
    # only on healthy steps, so replaying them from zero on resume would reheat
    # the bridge and re-warm the prox ramp that the run had already earned.
    #
    # The `elif` branch is the baseline handoff: a released baseline state_*.pt has
    # keys {step, model, ema_model, optimizer, ema_decay} and nothing else.  The
    # generator, its EMA, the optimizer moments and the step count all load
    # above -- they have to, because DitGen is shape-identical by construction
    # (tests/test_ckpt_compat.py) -- and the PT-Flow side simply starts fresh at
    # that step.  That is the whole "resume the OT-drift baseline at N, continue to N+M" path.
    pt = getattr(state, "pt", None)
    if pt is not None and payload.get("pt_model") is not None:
        unwrap_ddp(pt.potential).load_state_dict(payload["pt_model"], strict=True)
        if payload.get("pt_ema_model") is not None:
            unwrap_ddp(pt.ema_potential).load_state_dict(payload["pt_ema_model"], strict=True)
        if getattr(pt, "scale", None) is not None and payload.get("pt_scale_model") is not None:
            unwrap_ddp(pt.scale).load_state_dict(payload["pt_scale_model"], strict=True)
            if payload.get("pt_ema_scale_model") is not None:
                unwrap_ddp(pt.ema_scale).load_state_dict(payload["pt_ema_scale_model"], strict=True)
        if payload.get("pt_optimizer") is not None:
            _restore_optimizer(pt.optimizer, payload["pt_optimizer"])
        pt.sched.load_state_dict(payload.get("pt_schedule", {}))
        log_for_0("Restored PT-Flow potential, scale net and schedule (step %d)", state.step)
    elif pt is not None:
        log_for_0(
            "Checkpoint at step %d carries no PT-Flow state -- treating it as a "
            "the OT-drift baseline handoff. Generator, EMA and optimizer moments were restored; "
            "the potential and scale net start fresh and lambda_prox stays 0 for "
            "the first ptflow.schedule.prox_warmup steps, so the run continues as "
            "the OT-drift baseline while the potential catches up.",
            state.step,
        )

    return state


def save_checkpoint(state, keep=2, keep_every=None, workdir: Optional[str] = None):
    barrier()
    if not is_rank_zero():
        barrier()
        return

    step = _to_python_int(state.step)
    ckpt_dir = _job_ckpt_dir(workdir=workdir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"state_{step:08d}.pt"

    payload = {
        "step": step,
        "model": {k: v.detach().cpu() for k, v in unwrap_ddp(state.model).state_dict().items()},
        "ema_model": {k: v.detach().cpu() for k, v in unwrap_ddp(state.ema_model).state_dict().items()} if getattr(state, "ema_model", None) is not None else None,
        "optimizer": state.optimizer.state_dict(),
        "ema_decay": float(getattr(state, "ema_decay", 0.999)),
        "model_behavior": {"residual": bool(getattr(unwrap_ddp(state.model), "residual", False))},
        "grad_scaler": state.scaler.state_dict() if getattr(state, "scaler", None) is not None else None,
        "extra_emas": {decay: {k: v.detach().cpu() for k, v in ema.state_dict().items()}
                       for decay, ema in getattr(state, "extra_emas", {}).items()},
    }

    # PT-Flow state goes in under pt_* keys only.  "model" and "ema_model" stay
    # exactly baseline-shaped, so this checkpoint also loads into the stock baseline.
    pt = getattr(state, "pt", None)
    if pt is not None:
        payload["pt_model"] = {
            k: v.detach().cpu() for k, v in unwrap_ddp(pt.potential).state_dict().items()
        }
        payload["pt_ema_model"] = {
            k: v.detach().cpu() for k, v in unwrap_ddp(pt.ema_potential).state_dict().items()
        }
        if getattr(pt, "scale", None) is not None:
            payload["pt_scale_model"] = {
                k: v.detach().cpu() for k, v in unwrap_ddp(pt.scale).state_dict().items()
            }
            payload["pt_ema_scale_model"] = {
                k: v.detach().cpu() for k, v in unwrap_ddp(pt.ema_scale).state_dict().items()
            }
        payload["pt_optimizer"] = pt.optimizer.state_dict()
        payload["pt_schedule"] = pt.sched.state_dict()

    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    log_for_0("Saving checkpoint step %d to %s", step, str(path))

    ckpts = _list_ckpts(ckpt_dir)
    if keep is not None and keep > 0 and len(ckpts) > keep:
        protected = set()
        if keep_every:
            for p in ckpts:
                s = _extract_step(p)
                if s % int(keep_every) == 0:
                    protected.add(p)
        removable = [p for p in ckpts[:-keep] if p not in protected]
        for p in removable:
            p.unlink(missing_ok=True)

    barrier()


def save_params_ema_artifact(
    state: Any,
    *,
    workdir: Optional[str] = None,
    kind: str,
    model_config: Optional[Dict[str, Any]] = None,
) -> Path:
    step = _to_python_int(state.step)
    ema_decay = float(getattr(state, "ema_decay"))

    out_dir = _output_root(workdir) / "params_ema"
    out_dir.mkdir(parents=True, exist_ok=True)

    ema_state = (
        {k: v.detach().cpu() for k, v in unwrap_ddp(state.ema_model).state_dict().items()}
        if getattr(state, "ema_model", None) is not None
        else {k: v.detach().cpu() for k, v in unwrap_ddp(state.model).state_dict().items()}
    )
    torch.save(ema_state, out_dir / "ema_params.pt")

    metadata = {
        "format": "torch.pt",
        "kind": kind,
        "backend": "torch",
        "ema_decay": ema_decay,
        "step": step,
        "path": "params_ema/ema_params.pt",
        "model_config": dict(model_config or {}),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    log_for_0("Saved EMA params artifact step %d to %s", step, str(out_dir))
    return out_dir
