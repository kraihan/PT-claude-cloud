"""Two-rank CPU/Gloo check of optimized training and FID moment reduction.

Run explicitly: python -m tests.distributed_performance_smoke --output tmp/ddp-check
No GPU, dataset, or pretrained feature downloads are required.
"""
import argparse
from datetime import timedelta
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, rendezvous, output):
    torch.set_num_threads(2)
    dist.init_process_group("gloo", init_method=Path(rendezvous).as_uri(), rank=rank, world_size=2,
                            timeout=timedelta(seconds=90))
    from tests.test_train_smoke import make_state, fake_feature_apply, HW, CH
    from train import train_step
    from utils.feature_moments import FeatureMoments
    for pt in (False, True):
        state = make_state(pt, residual=True, with_scale=pt, lambda_prox_max=0.1)
        state.model = torch.nn.parallel.DistributedDataParallel(state.model)
        for _ in range(2):
            torch.manual_seed(91 + rank)
            state, metrics = train_step(state, torch.tensor([0, 1, 2]), torch.randn(3, 4, HW, HW, CH),
                torch.randn(3, 2, HW, HW, CH), {}, fake_feature_apply, gen_per_label=3,
                grad_accum_steps=2, feature_chunk_size=4, ot_mode="debiased", diverse_noise=True,
                ot_kwargs=dict(resample_neg=True, resample_gen_per_label=2, use_new_cfg=True,
                               disable_diag_mask=True, use_quadratic_cost=True, reuse_costs=True))
            assert torch.isfinite(metrics["loss"])
        modules = [state.model.module] + ([state.pt.potential, state.pt.scale] if pt else [])
        for module in modules:
            values = torch.cat([p.detach().flatten() for p in module.parameters()])
            other = values.clone()
            dist.broadcast(other, src=0)
            torch.testing.assert_close(values, other, rtol=0, atol=1e-6)
    x = np.arange(77, dtype=np.float64).reshape(11, 7)
    moments = FeatureMoments(7)
    moments.update(x[rank::2])
    moments.reduce(torch.device("cpu"))
    mean, covariance = moments.statistics()
    assert moments.count == 11
    np.testing.assert_allclose(mean, x.mean(0))
    np.testing.assert_allclose(covariance, np.cov(x, rowvar=False))
    if rank == 0:
        Path(output).write_text(json.dumps(dict(ranks=2, backend="gloo", optimized_training="passed",
                                               potential_gradient_sync="passed", uneven_fid_reduction="passed"), indent=2))
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="tmp/ddp-check")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as temp:
        mp.spawn(worker, args=(str(Path(temp) / "store"), str(root / "result.json")), nprocs=2, join=True)
    print((root / "result.json").read_text())


if __name__ == "__main__":
    main()
