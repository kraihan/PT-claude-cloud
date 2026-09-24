"""Run with torchrun --standalone --nproc_per_node=2 -m scripts.check_distributed."""
import os
import sys
import tempfile
from pathlib import Path
import torch
import torch.distributed as dist
from utils.dist_util import init_distributed, broadcast_module, cleanup_distributed
from ptflow.train_steps import allreduce_grads_


def main():
    torch.set_num_threads(2)
    init_distributed()
    rank = dist.get_rank()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(100 + rank)
    model = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.SiLU(), torch.nn.Linear(8, 1)).to(device)
    broadcast_module(model)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(3):
        opt.zero_grad()
        model(torch.randn(4, 3, device=device)).square().mean().backward()
        allreduce_grads_(model)
        opt.step()
        for param in model.parameters():
            reference = param.detach().clone()
            dist.broadcast(reference, 0)
            torch.testing.assert_close(param, reference, atol=0, rtol=0)
    if rank == 0:
        print("PASS: independent rank initialization -> synchronized parameters after 3 updates")
    cleanup_distributed()


def cpu_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    main()


if __name__ == "__main__":
    if "--cpu-spawn" in sys.argv:
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = (Path(directory) / "rendezvous").as_uri()
            mp.spawn(cpu_worker, args=(rendezvous,), nprocs=2, join=True)
    else:
        main()
