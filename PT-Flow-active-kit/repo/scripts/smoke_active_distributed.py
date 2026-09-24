"""Two-process training-path check with tiny synthetic models; no downloads."""
import argparse
from datetime import timedelta
import os
from pathlib import Path
import tempfile
import torch
import torch.distributed as dist


def worker(rank, rendezvous=None):
    torch.set_num_threads(2)
    use_cuda = torch.cuda.is_available() and rendezvous is None
    device = torch.device("cuda", rank) if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    if not dist.is_initialized():
        opts = dict(backend="nccl" if use_cuda else "gloo", timeout=timedelta(seconds=120))
        if rendezvous:
            opts.update(init_method=Path(rendezvous).as_uri(), rank=rank, world_size=2)
        dist.init_process_group(**opts)
    from tests.test_train_smoke import make_state, fake_feature_apply, HW, CH
    from ptflow.schedule import build_schedule
    from train import train_step
    state = make_state(True, residual=False, with_scale=True, lambda_prox_max=.1)
    for module in (state.model, state.ema_model, state.pt.potential, state.pt.ema_potential,
                   state.pt.scale, state.pt.ema_scale):
        module.to(device)
    state.optimizer = torch.optim.AdamW(state.model.parameters(), lr=1e-4)
    state.pt.optimizer = torch.optim.AdamW(list(state.pt.potential.parameters()) + list(state.pt.scale.parameters()), lr=1e-4)
    # Exercise activation checkpointing and second input derivatives together.
    state.model.model.use_remat = True
    state.pt.potential.trunk.use_remat = True
    if use_cuda:
        state.model.use_bf16 = True
    state.pt.sched = build_schedule(dict(policy="active_recovery_v1", eps_warmup=1,
        eps_anneal_steps=10, prox_warmup=0, prox_ramp=1, lambda_prox_max=.1,
        proposal_refine_steps=2, alignment_batch=2))
    state.pt.sched.health.value = 0.
    state.pt.sched.health.state = "broken"
    state.pt.cfg.update(prox_mode="full", prox_norm="none", prox_max_batch=2)
    state.model = torch.nn.parallel.DistributedDataParallel(state.model, device_ids=[rank] if use_cuda else None)
    for _ in range(3):
        torch.manual_seed(77 + rank)
        state, metrics = train_step(state, torch.tensor([0, 1, 2], device=device),
            torch.randn(3, 4, HW, HW, CH, device=device), torch.randn(3, 2, HW, HW, CH, device=device),
            {}, fake_feature_apply, gen_per_label=3, grad_accum_steps=2, device=device,
            ot_mode="debiased", diverse_noise=True,
            ot_kwargs=dict(sinkhorn_num_iter=3, use_new_cfg=True, disable_diag_mask=True,
                           use_quadratic_cost=True))
        assert torch.isfinite(metrics["loss"])
    assert metrics["pt/lambda_prox"] > 0
    assert metrics["pt/scale_updated"] == 1
    assert state.pt.sched.eps() < state.pt.sched.eps_max
    for module in (state.model.module, state.pt.potential, state.pt.scale):
        params = torch.cat([p.detach().flatten() for p in module.parameters()])
        reference = params.clone()
        dist.broadcast(reference, src=0)
        torch.testing.assert_close(params, reference, rtol=0, atol=1e-6)
    if rank == 0:
        print("ACTIVE_PT_DDP_PASS: full prox, recovery, scale, remat, and rank synchronization", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-spawn", action="store_true")
    args = parser.parse_args()
    if args.cpu_spawn:
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as td:
            mp.spawn(worker, args=(str(Path(td) / "store"),), nprocs=2, join=True)
    else:
        worker(int(os.environ["LOCAL_RANK"]))
