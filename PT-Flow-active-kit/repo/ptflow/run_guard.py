"""Optimizer-boundary status, recovery failure, and Slurm checkpoint handoff."""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
import torch
from utils.dist_util import process_index, barrier, dist_is_initialized


def inspect_boundary(state, metrics, workdir):
    root = Path(workdir)
    sched = state.pt.sched
    step = int(state.step)
    failure = sched.recovery_failure()
    if step > sched.prox_warmup + 10 and sched.lambda_prox() <= 0:
        failure = "Active-policy prox weight is zero after warmup"
    signal = int((root / "requeue.request").exists()) if process_index() == 0 else 0
    flag = torch.tensor(signal, device=next(state.model.parameters()).device)
    if dist_is_initialized():
        torch.distributed.broadcast(flag, src=0)
    requested = bool(flag.item())

    if step % 20 == 0 or step == 1 or failure or requested:
        keys = ["pt/lambda_prox", "pt/loss_prox", "pt/prox_resid_rel", "pt/ess", "pt/control_ess",
                "pt/generator_proposal_control_ess", "pt/refine_residual_rms", "pt/loss_alignment",
                "pt/loss_scale", "pt/g_norm_potential", "g_norm"]
        values = {k: float(torch.as_tensor(metrics[k]).detach().cpu()) if k in metrics else None for k in keys}
        row = dict(step=step, eps=sched.eps(), lambda_prox=sched.lambda_prox(),
                   control_ess_ema=sched.health.value, health=sched.health.state,
                   measured_healthy_steps=sched.measured_healthy_steps,
                   consecutive_bad=sched.consecutive_bad, **values)
        if process_index() == 0:
            table = root / "pt_status.csv"
            new = not table.exists()
            with table.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    writer.writeheader()
                writer.writerow(row)
            temp = root / "pt_status.json.tmp"
            temp.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
            temp.replace(root / "pt_status.json")
            print("PT_STATUS " + json.dumps(row), flush=True)
    if failure or requested:
        from utils.ckpt_util import save_checkpoint
        save_checkpoint(state, keep=2, keep_every=1000, workdir=workdir)
        if process_index() == 0:
            if failure:
                (root / "PT_FAILED.txt").write_text(f"step={step}\n{failure}\n", encoding="utf-8")
            else:
                (root / "requeue.ready").write_text(str(step), encoding="utf-8")
        barrier()
        if failure:
            raise RuntimeError(failure)
        raise SystemExit(0)
