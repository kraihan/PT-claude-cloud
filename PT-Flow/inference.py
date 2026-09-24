from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from dataset.dataset import get_postprocess_fn
from models.generator import DitGen
from utils.dist_util import barrier, init_distributed, process_count, process_index
from utils.env import IMAGENET_FID_NPZ, CIFAR10_FID_NPZ
from utils.misc import load_config, run_init
from utils.ckpt_util import canonical_state_dict, check_model_behavior

run_init()


def _print0(*args, **kwargs):
    if process_index() == 0:
        print(*args, **kwargs)


def _local_device() -> torch.device:
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def _load_model(ckpt_path: str, config_path: str, *, want_potential: bool = False, ema_decay=None):
    """Load EMA model from a training checkpoint.

    Returns ``(model, postprocess_fn, step, device, potential, scale_net, eps)``.
    The potential is None for a plain baseline checkpoint or when not requested;
    Mode A never needs it, which is why it is optional rather than required.

    The generator is loaded with strict=False deliberately: a reference release and
    a PT-Flow checkpoint carry the *same* generator tensors, so this path works
    for both, and strict=False only tolerates the pt_* keys living alongside.
    """
    config = load_config(config_path)
    model_cfg = dict(config.model)
    if "num_classes" not in model_cfg and hasattr(config, "dataset"):
        model_cfg["num_classes"] = int(config.dataset.get("num_classes", 1000))

    model = DitGen(**model_cfg)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    step = ckpt.get("step", -1)

    ema_sd = ckpt.get("ema_model")
    if ema_decay is not None:
        key = f"{float(ema_decay):g}"
        if key in ckpt.get("extra_emas", {}):
            ema_sd = ckpt["extra_emas"][key]
        elif float(ckpt.get("ema_decay", -1)) != float(ema_decay):
            raise ValueError(f"EMA {key} is not stored in this checkpoint")
    if ema_sd is None:
        _print0("WARNING: no ema_model in checkpoint, falling back to model weights")
        ema_sd = ckpt.get("model", ckpt)

    check_model_behavior(model, ckpt)
    missing, unexpected = model.load_state_dict(canonical_state_dict(ema_sd), strict=True)
    if missing:
        _print0(f"WARNING: missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        _print0(f"WARNING: unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    device = _local_device()
    model = model.to(device).eval()

    postprocess_fn = get_postprocess_fn(
        use_aug=False,
        use_latent=bool(config.dataset.get("use_latent", False)),
        use_cache=bool(config.dataset.get("use_cache", False)),
    )

    _print0(f"Loaded EMA model from step {step} ({ckpt_path})")
    _print0(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # -- PT-Flow potential (optional) --------------------------------------
    potential, scale_net, eps = None, None, None
    if want_potential:
        pt_cfg = dict(config.get("pt", {}) or {})
        if not pt_cfg.get("enabled", False):
            raise ValueError(
                "This config has no enabled `pt:` block, so there is no potential "
                "to load.  Modes B and C and `likelihood` require a PT-Flow run; "
                "use `--mode sample`/`evaluate` with Mode A for a baseline model."
            )
        from ptflow.potential import PotentialNet, ScaleNet
        from ptflow.schedule import build_schedule

        pm = dict(pt_cfg.get("model", {}))
        pm.setdefault("num_classes", int(model.num_classes))
        pm.setdefault("input_size", int(model.input_size))
        pm.setdefault("in_channels", int(model.in_channels))
        pm.setdefault("cond_dim", int(model.cond_dim))
        potential = PotentialNet(**pm)

        pt_sd = ckpt.get("pt_ema_model") or ckpt.get("pt_model")
        if pt_sd is None:
            raise ValueError(
                f"Checkpoint {ckpt_path} contains no PT-Flow potential "
                "(keys 'pt_ema_model'/'pt_model' are absent).  It was written by "
                "a baseline run, or by a PT-Flow run with pt.enabled: false."
            )
        potential.load_state_dict(pt_sd, strict=True)
        potential = potential.to(device).eval()

        # The diagonal log-scale net, if the run trained one.  Absent means
        # S = I, which is a valid proposal -- Modes C and the likelihood just
        # carry more variance.
        sc_sd = ckpt.get("pt_ema_scale_model") or ckpt.get("pt_scale_model")
        if sc_sd is not None:
            sm = dict(pt_cfg.get("scale_model", {}))
            for k, v in (("num_classes", int(model.num_classes)),
                         ("input_size", int(model.input_size)),
                         ("in_channels", int(model.in_channels)),
                         ("cond_dim", int(model.cond_dim))):
                sm.setdefault(k, v)
            scale_net = ScaleNet(**sm)
            scale_net.load_state_dict(sc_sd, strict=True)
            scale_net = scale_net.to(device).eval()
        else:
            _print0("  No scale net in checkpoint; proposals will use S = I.")

        # The eps the checkpoint was last trained at.  Sampling at a different
        # eps than training is legal but changes the bridge, so it is restored
        # from the schedule rather than guessed.
        sched = build_schedule(dict(pt_cfg.get("schedule", {})))
        sched.load_state_dict(ckpt.get("pt_schedule", {}))
        eps = sched.eps()
        _print0(
            f"  Potential: {sum(p.numel() for p in potential.parameters()):,} params"
            + (f", scale net: {sum(p.numel() for p in scale_net.parameters()):,}"
               if scale_net is not None else "")
            + f", eps={eps:.5g}"
        )

    return model, postprocess_fn, step, device, potential, scale_net, eps


# ---------------------------------------------------------------------------
# Sampling mode dispatch
# ---------------------------------------------------------------------------

def make_sampler(
    model, potential, eps, *,
    scale_net=None,
    mode: str = "A",
    cfg_scale: float = 1.0,
    pt_w: float | None = None,
    n_steps: int = 4,
    gamma: float = 0.5,
    snis_K: int = 32,
    alpha_def: float = 0.05,
):
    """Return ``f(labels, rng) -> latents`` for the requested sampling mode.

    Mode A is the default and is exactly the baseline sampler: one forward pass, no
    potential, no autograd.  B and C spend extra NFEs to close the O(sqrt(eps d))
    gap between the deterministic map and the model marginal (Proposition 2.4).
    """
    mode = str(mode).upper()
    if mode != "A" and potential is None:
        raise ValueError(f"Mode {mode} requires a PT-Flow potential; pass a PT-Flow checkpoint.")
    if pt_w is None:
        pt_w = float(cfg_scale) - 1.0

    from ptflow.sampling import sample_mode_a, sample_mode_b, sample_mode_c

    if mode == "A":
        def f(labels, rng):
            return sample_mode_a(model, labels, cfg_scale=cfg_scale, rng=rng)
    elif mode == "B":
        def f(labels, rng):
            return sample_mode_b(
                model, potential, labels, cfg_scale=cfg_scale, pt_w=pt_w,
                n_steps=int(n_steps), gamma=float(gamma), rng=rng,
            )
    elif mode == "C":
        def f(labels, rng):
            return sample_mode_c(
                model, potential, labels, eps, cfg_scale=cfg_scale, pt_w=pt_w,
                K=int(snis_K), alpha_def=float(alpha_def),
                scale_net=scale_net, rng=rng,
            )
    else:
        raise ValueError(f"unknown sampling mode {mode!r}; expected A, B or C")
    return f


# ---------------------------------------------------------------------------
# Generation (multi-GPU aware)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_and_save(
    model, postprocess_fn, save_folder: str,
    *, num_samples: int, device_batch_size: int,
    cfg_scale: float, seed: int, device: torch.device,
    sampler=None,
):
    world_size = process_count()
    local_rank = process_index()

    if local_rank == 0:
        if os.path.exists(save_folder):
            shutil.rmtree(save_folder)
        os.makedirs(save_folder, exist_ok=True)
    barrier()

    num_classes = getattr(model, "num_classes", 1000)
    assert num_samples % num_classes == 0, (
        f"num_samples ({num_samples}) must be divisible by num_classes ({num_classes})"
    )

    labels_all = np.arange(num_classes).repeat(num_samples // num_classes)
    pad = world_size * device_batch_size
    labels_all = np.concatenate([labels_all, np.zeros(pad, dtype=labels_all.dtype)])

    num_steps = (num_samples + world_size * device_batch_size - 1) // (
        world_size * device_batch_size
    )

    pbar = tqdm(range(num_steps), desc="Generating", disable=(local_rank != 0))
    for step_i in pbar:
        global_start = step_i * world_size * device_batch_size
        rank_start = global_start + local_rank * device_batch_size
        rank_end = rank_start + device_batch_size

        batch_labels = torch.from_numpy(
            labels_all[rank_start:rank_end]
        ).long().to(device)

        sample_indices = rank_start + torch.arange(device_batch_size)
        rng = torch.Generator(device=device)
        rng.manual_seed(seed ^ int(sample_indices[0].item()))

        if sampler is None:
            latent_samples = model(
                c=batch_labels, cfg_scale=cfg_scale,
                deterministic=True, train=False, rng=rng,
            )["samples"]
        else:
            latent_samples = sampler(batch_labels, rng)

        pixel_images = postprocess_fn(latent_samples)
        pixel_np = pixel_images.detach().cpu().float().numpy()
        if not np.isfinite(pixel_np).all():
            raise ValueError("Nonfinite generated images; refusing FID evaluation.")
        pixel_np = np.clip(pixel_np, 0.0, 1.0)

        for b in range(device_batch_size):
            img_id = int(sample_indices[b].item())
            if img_id >= num_samples:
                break
            img_hwc = (pixel_np[b].transpose(1, 2, 0) * 255).round().astype(np.uint8)
            Image.fromarray(img_hwc).save(
                os.path.join(save_folder, f"{img_id:05d}.png")
            )

    barrier()


# ---------------------------------------------------------------------------
# Sample mode -- preview grid
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_sample(
    model, postprocess_fn, *,
    class_ids: list[int], cfg_scale: float,
    seed: int, num_rows: int, save_path: str, device: torch.device,
    sampler=None,
):
    labels = torch.tensor(class_ids, dtype=torch.long, device=device)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    if sampler is None:
        latent_samples = model(
            c=labels, cfg_scale=cfg_scale,
            deterministic=True, train=False, rng=rng,
        )["samples"]
    else:
        latent_samples = sampler(labels, rng)

    pixel_images = postprocess_fn(latent_samples)
    imgs = pixel_images.detach().cpu().float().numpy()
    if not np.isfinite(imgs).all():
        raise ValueError("Nonfinite generated preview images.")
    imgs = np.clip(imgs, 0.0, 1.0)
    imgs = (imgs.transpose(0, 2, 3, 1) * 255).round().astype(np.uint8)

    n = len(imgs)
    num_cols = (n + num_rows - 1) // num_rows
    h, w, c = imgs.shape[1], imgs.shape[2], imgs.shape[3]

    grid = np.zeros((num_rows * h, num_cols * w, c), dtype=np.uint8)
    for idx in range(n):
        r, col = divmod(idx, num_cols)
        grid[r * h : (r + 1) * h, col * w : (col + 1) * w, :] = imgs[idx]

    out = Path(save_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(out)
    print(f"Saved {num_rows}x{num_cols} grid ({n} images) to {out}")


# ---------------------------------------------------------------------------
# Evaluate mode -- FID / ISC
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_eval_streaming(model, postprocess_fn, ckpt_path, ckpt_step, *, num_samples,
                       cfg_scale, gen_bsz, fid_ref, seed, device, sampler, mode="A"):
    """Generate and extract features on EVERY rank, without writing PNG files."""
    from utils.feature_moments import FeatureMoments
    from utils.fid_util import _extract_inception_features, _compute_frechet_distance
    if num_samples < 2 or num_samples % model.num_classes:
        raise ValueError("num_samples must be >= 2 and divisible by num_classes")
    if gen_bsz < 1:
        raise ValueError("gen_bsz must be positive")
    ref = np.load(fid_ref)
    mu_ref = ref["ref_mu"] if "ref_mu" in ref else ref["mu"]
    cov_ref = ref["ref_sigma"] if "ref_sigma" in ref else ref["sigma"]
    if mu_ref.ndim != 1 or cov_ref.shape != (len(mu_ref), len(mu_ref)) or not np.isfinite(mu_ref).all() or not np.isfinite(cov_ref).all():
        raise ValueError("Invalid FID reference statistics")
    moments = FeatureMoments(len(mu_ref))
    rank, world = process_index(), process_count()
    start_time = time.perf_counter()
    for start in tqdm(range(rank, num_samples, world * gen_bsz), desc="Generate/Inception", disable=rank != 0):
        indices = torch.arange(start, min(start + world * gen_bsz, num_samples), world, device=device)
        labels = (indices % model.num_classes).long()
        rng = torch.Generator(device=device).manual_seed(int(seed) + start)
        pixels = postprocess_fn(sampler(labels, rng)).detach().float()
        if not torch.isfinite(pixels).all():
            raise ValueError("Nonfinite generated images; refusing FID evaluation")
        images = (pixels.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        features, _ = _extract_inception_features(images, batch_size=gen_bsz)
        moments.update(features)
    moments.reduce(device)
    if moments.count != num_samples:
        raise RuntimeError(f"Expected {num_samples} images, got {moments.count}")
    result = None
    if rank == 0:
        mu, cov = moments.statistics()
        fid = _compute_frechet_distance(mu_ref, cov_ref, mu, cov)
        result = dict(ckpt=ckpt_path, step=ckpt_step, cfg_scale=cfg_scale, mode=mode,
                      fid=fid, num_samples=moments.count, seed=seed, fid_ref=str(Path(fid_ref).resolve()),
                      feature_extractor="inception-v3-compat", backend="streaming",
                      world_size=world, gen_bsz=gen_bsz, elapsed_s=time.perf_counter() - start_time)
        _print0(f"FID: {fid}")
    barrier()
    return result


def run_eval(
    model, postprocess_fn, ckpt_path: str, ckpt_step: int,
    workdir: str, *, num_samples: int, cfg_scale: float,
    gen_bsz: int, fid_ref: str, seed: int,
    keep_samples: bool, device: torch.device,
    sampler=None, mode: str = "A",
) -> dict | None:
    calculate_metrics = None
    if process_index() == 0:
        from utils.fidelity_wrapper import calculate_metrics

    save_folder = os.path.join(workdir, "fid_outputs")

    t0 = time.time()
    generate_and_save(
        model, postprocess_fn, save_folder,
        num_samples=num_samples, device_batch_size=gen_bsz,
        cfg_scale=cfg_scale, seed=seed, device=device, sampler=sampler,
    )
    gen_time = time.time() - t0
    _print0(f"Generation done in {gen_time:.1f}s")

    result = None
    if process_index() == 0:
        _print0("Computing metrics via torch-fidelity (inception-v3-compat) ...")
        metrics_dict = calculate_metrics(
            input1=save_folder, input2=fid_ref,
            cuda=device.type == "cuda", isc=True, fid=True, kid=False, prc=False, verbose=True,
        )

        fid = metrics_dict.get("frechet_inception_distance")
        isc_mean = metrics_dict.get("inception_score_mean")
        isc_std = metrics_dict.get("inception_score_std")
        _print0(f"FID: {fid}")
        _print0(f"Inception Score: {isc_mean} +/- {isc_std}")

        result = {
            "ckpt": ckpt_path,
            "step": ckpt_step,
            "cfg_scale": cfg_scale,
            "mode": mode,
            "fid": fid,
            "isc_mean": isc_mean,
            "isc_std": isc_std,
            "gen_time": gen_time,
        }

        if not keep_samples:
            shutil.rmtree(save_folder)

    barrier()
    return result


# ---------------------------------------------------------------------------
# Likelihood mode -- Theorem 2.5
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_likelihood(
    model, potential, scale_net, eps, config_path: str, ckpt_path: str, ckpt_step: int,
    *, num_batches: int, bsz: int, k_inner: int, k_ladder: list[int],
    seed: int, device: torch.device,
) -> dict | None:
    """Report NLL along a ladder of evaluation budgets, at w = 0 only.

    The reporting protocol is Appendix G's, and its three rules are enforced
    here rather than left to the caller:

      * w = 0 only.  Theorem 2.5 normalizes the *untilted* model; a guided
        potential is a different (unnormalized-by-this-theorem) object.
      * A ladder over K_eval with its monotone tightening curve.  The outer
        logarithm makes each entry an IWAE-style lower bound, so a single number
        is not interpretable on its own -- only the trend is.
      * The value is a latent-space density.  Converting to pixel bits/dim needs
        the VAE's Jacobian, which this model does not represent, so the number
        is comparable across PT-Flow checkpoints and not against pixel-space
        likelihood models.
    """
    if process_index() != 0:
        barrier()
        return None

    from pipelines import build_pipeline
    from ptflow.sampling import log_likelihood

    config = load_config(config_path)
    loader, preprocess_fn, _ = build_pipeline(config).build_split(batch_size=int(bsz), split="val")

    rng = torch.Generator(device=device)
    rng.manual_seed(int(seed))

    rows = []
    for K_eval in sorted(k_ladder):
        # Evaluate each budget on the same examples and reset the MC stream.
        it = iter(loader)
        rng.manual_seed(int(seed))
        nlls, esss = [], []
        for _ in range(int(num_batches)):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            proc = preprocess_fn(batch)
            x1 = proc["images"].to(device).float()
            c = proc["labels"].to(device).long()

            _, info = log_likelihood(
                model, potential, x1, c, eps,
                K_outer=int(K_eval), K_inner=int(k_inner),
                scale_net=scale_net, rng=rng,
            )
            nlls.append(info["nll_per_dim"])
            esss.append(info["inner_ess"])

        row = {
            "K_eval": int(K_eval),
            "nll_per_dim_nats": float(np.mean(nlls)),
            "bits_per_dim_latent": float(np.mean(nlls)) / float(np.log(2.0)),
            "inner_ess": float(np.mean(esss)),
        }
        rows.append(row)
        _print0(
            f"  K_eval={K_eval:<5d} NLL/dim={row['nll_per_dim_nats']:+.5f} nats "
            f"({row['bits_per_dim_latent']:+.5f} bits, latent)   "
            f"inner ESS={row['inner_ess']:.3f}"
        )

    monotone = all(b["nll_per_dim_nats"] <= a["nll_per_dim_nats"] + 1e-6
                   for a, b in zip(rows, rows[1:]))
    if not monotone:
        _print0(
            "  The finite nested-MC estimates are not monotone. Check both "
            "inner and outer budgets across seeds; this is not a certified likelihood bound."
        )

    result = {
        "ckpt": ckpt_path,
        "step": ckpt_step,
        "eps": float(eps),
        "guidance_w": 0.0,
        "k_inner": int(k_inner),
        "ladder": rows,
        "monotone_tightening": bool(monotone),
        "estimator": "nested_monte_carlo_not_a_certified_bound",
        "space": "continuous CIFAR pixels in [-1,1]" if config.get("pipeline") == "cifar10_pixel" else "sd-vae latent (not pixel bits/dim)",
    }
    barrier()
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inference from our training checkpoints (state_*.pt)."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--ckpt", required=True,
        help="Path to state_*.pt checkpoint file.",
    )
    shared.add_argument(
        "--config", required=True,
        help="Path to training config YAML (for model architecture).",
    )
    shared.add_argument("--cfg-scale", type=float, default=1.0,
                        help="Generator conditioning. the OT-drift baseline convention: cfg_scale = w + 1.")
    shared.add_argument("--seed", type=int, default=0)
    shared.add_argument("--ema-decay", type=float, default=None, help="Select a stored EMA; omitted uses the primary EMA.")
    shared.add_argument("--workdir", default="runs/infer_ours")

    # --- PT-Flow sampling modes -------------------------------------------
    # NOTE: the flag is --sampler, not --mode; `mode` is already the subcommand.
    shared.add_argument(
        "--sampler", choices=["A", "B", "C"], default="A",
        help="A: strict 1-NFE (default, identical to the baseline).  "
             "B: n-step prox refinement.  C: SNIS exact sampler, K NFE.",
    )
    shared.add_argument(
        "--pt-w", type=float, default=None,
        help="PT-Flow potential guidance weight for Modes B/C "
             "(phi^w = (1+w) phi(.,c) - w phi(.,null)).  "
             "Defaults to cfg_scale - 1.  This is a different mechanism from "
             "--cfg-scale and is worth sweeping separately.",
    )
    shared.add_argument("--refine-steps", type=int, default=4, help="Mode B: n.")
    shared.add_argument("--refine-gamma", type=float, default=0.5, help="Mode B: step size.")
    shared.add_argument("--snis-k", type=int, default=32, help="Mode C: K.")
    shared.add_argument("--alpha-def", type=float, default=0.05,
                        help="Mode C: defensive mixture weight.")

    sp = sub.add_parser("sample", parents=[shared], help="Generate a preview grid.")
    sp.add_argument(
        "--class-ids", type=str,
        default="207,360,387,974,88,979,417,279",
    )
    sp.add_argument("--num-rows", type=int, default=2)
    sp.add_argument("--save-path", type=str, default="")

    ep = sub.add_parser("evaluate", parents=[shared], help="Generate 50k images and compute FID.")
    ep.add_argument("--num-samples", type=int, default=50000)
    ep.add_argument("--gen-bsz", type=int, default=64)
    ep.add_argument("--fid-ref", type=str, default="", help="Defaults to the config's dataset reference statistics.")
    ep.add_argument("--json-out", type=str, default="")
    ep.add_argument("--keep-samples", action="store_true")
    ep.add_argument("--eval-backend", choices=["streaming", "png"], default="streaming",
                    help="streaming: distributed FID without image files; png: legacy FID and IS. --keep-samples selects png.")

    lp = sub.add_parser(
        "likelihood", parents=[shared],
        help="Nested Monte Carlo likelihood estimate; check inner and outer budgets.",
    )
    lp.add_argument("--num-batches", type=int, default=8)
    lp.add_argument("--bsz", type=int, default=8)
    lp.add_argument("--k-inner", type=int, default=16)
    lp.add_argument(
        "--k-ladder", type=str, default="16,32,64,128",
        help="Outer evaluation budgets. Finite estimates have no monotonicity guarantee.",
    )
    lp.add_argument("--json-out", type=str, default="")

    return parser


def main() -> None:
    init_distributed()
    args = build_parser().parse_args()

    if process_index() == 0:
        os.makedirs(args.workdir, exist_ok=True)
    barrier()

    needs_potential = args.sampler != "A" or args.mode == "likelihood"
    model, postprocess_fn, ckpt_step, device, potential, scale_net, eps = _load_model(
        args.ckpt, args.config, want_potential=needs_potential, ema_decay=args.ema_decay
    )

    sampler = make_sampler(
        model, potential, eps, scale_net=scale_net,
        mode=args.sampler, cfg_scale=args.cfg_scale, pt_w=args.pt_w,
        n_steps=args.refine_steps, gamma=args.refine_gamma,
        snis_K=args.snis_k, alpha_def=args.alpha_def,
    )

    if args.mode == "sample":
        if process_index() == 0:
            class_ids = [int(x.strip()) for x in args.class_ids.split(",") if x.strip()]
            save_path = args.save_path or os.path.join(args.workdir, "sample_grid.png")
            run_sample(
                model, postprocess_fn,
                class_ids=class_ids, cfg_scale=args.cfg_scale,
                seed=args.seed, num_rows=args.num_rows,
                save_path=save_path, device=device, sampler=sampler,
            )
        barrier()

    elif args.mode == "evaluate":
        if not args.fid_ref:
            args.fid_ref = CIFAR10_FID_NPZ if model.num_classes == 10 and model.in_channels == 3 else IMAGENET_FID_NPZ
        common = dict(num_samples=args.num_samples, cfg_scale=args.cfg_scale,
                      gen_bsz=args.gen_bsz, fid_ref=args.fid_ref, seed=args.seed,
                      device=device, sampler=sampler, mode=args.sampler)
        if args.eval_backend == "png" or args.keep_samples:
            result = run_eval(model, postprocess_fn, args.ckpt, ckpt_step, args.workdir,
                              keep_samples=args.keep_samples, **common)
        else:
            result = run_eval_streaming(model, postprocess_fn, args.ckpt, ckpt_step, **common)
        if result is not None:
            result.update(num_samples=args.num_samples, seed=args.seed, config=str(Path(args.config).resolve()),
                          ema_decay=args.ema_decay,
                          refine_steps=args.refine_steps if args.sampler == "B" else 0,
                          refine_gamma=args.refine_gamma if args.sampler == "B" else None,
                          snis_k=args.snis_k if args.sampler == "C" else 0, pt_w=args.pt_w,
                          alpha_def=args.alpha_def if args.sampler == "C" else None,
                          eps=eps, gen_bsz=args.gen_bsz, world_size=process_count(),
                          fid_ref=str(Path(args.fid_ref).resolve()))
        if result is not None:
            print(json.dumps(result, indent=2))
            if args.json_out:
                out = Path(args.json_out).resolve()
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    elif args.mode == "likelihood":
        result = run_likelihood(
            model, potential, scale_net, eps, args.config, args.ckpt, ckpt_step,
            num_batches=args.num_batches, bsz=args.bsz,
            k_inner=args.k_inner,
            k_ladder=[int(x) for x in args.k_ladder.split(",") if x.strip()],
            seed=args.seed, device=device,
        )
        if result is not None:
            print(json.dumps(result, indent=2))
            if args.json_out:
                out = Path(args.json_out).resolve()
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
