"""Six OFAT sweeps, submitted as six serial, restartable single-GPU jobs.

Supports the earlier PNG-FID engine and the updated streaming-FID engine.
Planning and submission inspect Python source without importing GPU code.
"""
import argparse
import ast
import copy
import csv
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import string
import subprocess
import sys
import time

import yaml

FACTORS = ("proposal", "eps", "K", "lambda_prox_max", "n", "cfg_scale")
PROPOSALS = {"naive": ("none", 1.0), "recentered": ("none", 0.0),
             "diagonal": ("learned", 0.0), "defensive_diagonal": ("learned", 0.05)}


def inspect_engine(repo):
    """Read actual CLI/function signatures; do not guess from version strings."""
    repo = Path(repo)
    trees = {name: ast.parse((repo / name).read_text(encoding="utf-8-sig"))
             for name in ("train.py", "inference.py", "ptflow/ot_drift.py")}

    def flags(tree):
        return sorted({arg.value for node in ast.walk(tree)
                       if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                       and node.func.attr == "add_argument" for arg in node.args
                       if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                       and arg.value.startswith("--")})

    def arguments(tree, name):
        matches = [node for node in ast.walk(tree)
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
        if len(matches) != 1:
            raise ValueError(f"Cannot inspect {name}; unsupported engine layout")
        node = matches[0]
        return sorted(a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs), bool(node.args.kwarg)

    train_args, train_kwargs = arguments(trees["train.py"], "train_gen")
    ot_args, ot_kwargs = arguments(trees["ptflow/ot_drift.py"], "ot_drift_loss")
    inference_flags = flags(trees["inference.py"])
    return dict(train_flags=flags(trees["train.py"]), inference_flags=inference_flags,
                train_args=train_args, train_kwargs=train_kwargs, ot_args=ot_args, ot_kwargs=ot_kwargs,
                evaluation_backend="streaming" if "--eval-backend" in inference_flags else "png")


def adapt_config(cfg, engine):
    """Drop unsupported execution optimizations; never silently drop objective knobs."""
    removed = []
    if not engine["train_kwargs"]:
        for key in ("profile_every", "feature_cache_gib", "compile_generator", "extra_ema_decays"):
            if key in cfg["train"] and key not in engine["train_args"]:
                cfg["train"].pop(key)
                removed.append("train." + key)
        unknown = set(cfg["train"]) - set(engine["train_args"])
        if unknown:
            raise ValueError(f"train_gen does not support these settings: {', '.join(sorted(unknown))}")
    ot = cfg["train"].get("ot_kwargs", {})
    if not engine["ot_kwargs"] and "reuse_costs" not in engine["ot_args"] and "reuse_costs" in ot:
        ot.pop("reuse_costs")
        removed.append("train.ot_kwargs.reuse_costs")
    return removed


def merge(base, patch):
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def path_value(value, repo, env):
    path = Path(string.Template(str(value)).substitute(env)).expanduser()
    return str((path if path.is_absolute() else repo / path).resolve())


def build_plan(manifest, repo=None):
    repo = Path(repo or Path(__file__).resolve().parents[1]).resolve()
    spec = yaml.safe_load(Path(manifest).read_text(encoding="utf-8"))
    if spec.get("version") != 1 or set(spec["sweeps"]) != set(FACTORS):
        raise ValueError("Expected version 1 and exactly the six documented factors")
    env = dict(os.environ)
    env.setdefault("USER", getpass.getuser())
    exports = {}
    pending = dict(spec.get("environment", {}))
    while pending:
        progressed = False
        for key, value in list(pending.items()):
            dependencies = {m.group("named") or m.group("braced")
                            for m in string.Template.pattern.finditer(str(value))} - {None}
            if dependencies.intersection(pending):
                continue
            try:
                exports[key] = path_value(value, repo, env)
            except KeyError as error:
                raise ValueError(f"Undefined environment variable in {key}: {error}") from error
            env[key] = exports[key]
            del pending[key]
            progressed = True
        if not progressed:
            raise ValueError(f"Cyclic environment paths: {', '.join(pending)}")
    base_path = path_value(spec["base_config"], repo, env)
    cfg = merge(yaml.safe_load(Path(base_path).read_text(encoding="utf-8")), spec.get("overrides", {}))
    if cfg.get("pipeline") not in ("cifar10_pixel", "imagenet_latent"):
        raise ValueError("Supported pipelines: cifar10_pixel and imagenet_latent")
    if cfg["pipeline"] == "imagenet_latent":
        if not cfg["dataset"].get("use_cache") or cfg["dataset"]["num_classes"] != 1000:
            raise ValueError("ImageNet ablations require a latent cache and all 1000 classes")
        if (cfg["model"]["input_size"], cfg["model"]["in_channels"], cfg["model"]["out_channels"]) != (32, 4, 4):
            raise ValueError("ImageNet-256 requires 32x32x4 latent input/output")
    steps = int(spec["steps"])
    if steps < 1:
        raise ValueError("steps must be positive")
    train = cfg["train"]
    for key in ("resume_from", "init_ema_from", "init_from", "benchmark_steps"):
        train.pop(key, None)
    train.update(total_steps=steps, seed=int(spec["seed"]), eval_per_step=0, eval_at_start=False,
                 save_per_step=min(1000, steps), keep_every=steps, keep_last=2)
    cfg["optimizer"]["lr_schedule"]["total_steps"] = steps
    cfg["pt"]["optimizer"]["lr_schedule"]["total_steps"] = steps
    if not cfg["pt"].get("enabled"):
        raise ValueError("PT must remain enabled for these ablations")
    if int(cfg["pt"]["schedule"]["prox_warmup"]) >= steps:
        raise ValueError("prox_warmup must be shorter than this experiment")
    accumulation = int(train.get("grad_accum_steps", 1))
    if accumulation < 1 or int(train["train_batch_size"]) % accumulation:
        raise ValueError("train_batch_size must be divisible by grad_accum_steps")
    ref, sweeps = spec["reference"], spec["sweeps"]
    for factor in FACTORS:
        values = sweeps[factor]
        if not 3 <= len(values) <= 4 or len(set(values)) != len(values) or ref[factor] not in values:
            raise ValueError(f"{factor}: need 3-4 unique values including the reference")
        for value in values:
            if factor == "proposal":
                if value not in PROPOSALS:
                    raise ValueError(f"Unknown proposal {value}")
            elif not math.isfinite(float(value)):
                raise ValueError(f"Nonfinite {factor}")
            elif factor in ("K", "n") and (int(value) != value or value < (2 if factor == "K" else 0)):
                raise ValueError(f"Invalid {factor}: {value}")
            elif factor in ("eps", "cfg_scale") and value <= 0:
                raise ValueError(f"{factor} must be positive")
            elif factor == "lambda_prox_max" and value < 0:
                raise ValueError("lambda_prox_max must be nonnegative")
    if ref["n"] != 0:
        raise ValueError("The shared reference uses Mode A (n=0)")
    evaluation = spec["evaluation"]
    if int(evaluation["num_samples"]) < 2 or int(evaluation["batch_size"]) < 1 or evaluation["gamma"] <= 0:
        raise ValueError("Invalid evaluation budget or gamma")
    init = spec.get("init_ema_checkpoint", "")
    wb = dict(spec.get("wandb", {}))
    if set(wb) - {"enabled", "project", "entity", "group", "mode"}:
        raise ValueError("wandb accepts enabled/project/entity/group/mode only; authenticate with wandb login")
    if wb.get("mode", "online") not in ("online", "offline"):
        raise ValueError("wandb.mode must be online or offline")
    root = path_value(spec["output_dir"], repo, env)
    if wb.get("enabled"):
        wb.update(project=wb.get("project") or "ptflow-ablation", entity=wb.get("entity") or None,
                  group=wb.get("group") or Path(root).name, mode=wb.get("mode", "online"))
        cfg.setdefault("logging", {}).update(use_wandb=True, project=wb["project"],
                                              entity=wb["entity"], group=wb["group"],
                                              mode=wb["mode"], job_type="train")
    else:
        cfg.setdefault("logging", {})["use_wandb"] = False
    engine = inspect_engine(repo)
    engine["removed_optimizations"] = adapt_config(cfg, engine)
    plan = dict(version=1, repo=str(repo), root=root, wandb=wb,
                engine=engine,
                environment=exports, base_config=base_path,
                init_ema_checkpoint=path_value(init, repo, env) if init else "",
                steps=steps, seed=int(spec["seed"]), slurm=spec["slurm"], evaluation=evaluation,
                reference=ref, arms=[], stages=[dict(factor=f, entries=[]) for f in FACTORS])

    def add_arm(name, settings):
        arm_cfg = copy.deepcopy(cfg)
        pt = arm_cfg["pt"]
        mode, alpha = PROPOSALS[settings["proposal"]]
        pt.update(K=int(settings["K"]), scale_mode=mode)
        pt["schedule"].update(eps_min=float(settings["eps"]), eps_max=float(settings["eps"]),
                              alpha_def_start=alpha, alpha_def_end=alpha, alpha_def_degraded=alpha,
                              lambda_prox_max=float(settings["lambda_prox_max"]))
        plan["arms"].append(dict(name=name, settings=settings, config=arm_cfg))

    add_arm("reference", copy.deepcopy(ref))
    for i, factor in enumerate(FACTORS):
        values = [ref[factor]] + [v for v in sweeps[factor] if v != ref[factor]]
        for value in values:
            tag = re.sub(r"[^a-zA-Z0-9_-]", "p", str(value))
            name = "reference" if value == ref[factor] else f"{factor}_{tag}"
            arm = name if i < 4 else "reference"
            if i < 4 and name != "reference":
                settings = copy.deepcopy(ref)
                settings[factor] = value
                add_arm(name, settings)
            plan["stages"][i]["entries"].append(dict(
                factor=factor, value=value, arm=arm, result=name,
                train=(i < 4 and (i == 0 or name != "reference")),
                n=int(value if factor == "n" else ref["n"]),
                cfg_scale=float(value if factor == "cfg_scale" else ref["cfg_scale"])))
    return plan


def describe(plan):
    print(f"Output: {plan['root']}")
    print(f"Dataset: {plan['arms'][0]['config']['pipeline']}")
    print(f"{len(plan['arms'])} training runs x {plan['steps']} = {len(plan['arms']) * plan['steps']} steps")
    results = {e['result'] for s in plan['stages'] for e in s['entries']}
    print(f"{len(results)} unique FID evaluations; shared reference reused")
    for i, stage in enumerate(plan["stages"], 1):
        print(f"Job {i}: {stage['factor']}: " + ", ".join(str(e['value']) for e in stage['entries']))
    print("Six factor groups. run_six_ablation submits a chain; run_parallel_ablation submits independent arms.")
    print(f"Evaluation backend: {plan['engine']['evaluation_backend']}")
    if plan["engine"]["removed_optimizations"]:
        print("Engine compatibility: omitted " + ", ".join(plan["engine"]["removed_optimizations"]))


def preflight(plan):
    repo = Path(plan["repo"])
    engine = plan["engine"]
    required_inference = {"--ckpt", "--config", "--sampler", "--refine-steps", "--refine-gamma",
                          "--cfg-scale", "--seed", "--num-samples", "--gen-bsz", "--workdir", "--json-out"}
    missing = required_inference - set(engine["inference_flags"])
    if missing:
        raise ValueError(f"inference.py lacks required ablation flags: {', '.join(sorted(missing))}")
    missing_train = {"--config", "--workdir"} - set(engine["train_flags"])
    if missing_train:
        raise ValueError(f"train.py lacks required flags: {', '.join(sorted(missing_train))}")
    if plan["init_ema_checkpoint"] and "--init-ema" not in engine["train_flags"]:
        raise ValueError("This engine cannot EMA-initialize: choose init_ema_checkpoint: '' for fresh runs, or update train.py")
    env = dict(os.environ, **plan["environment"])
    cfg = plan["arms"][0]["config"]
    if cfg["pipeline"] == "imagenet_latent":
        preflight_imagenet(cfg, env, repo)
        fid = env.get("IMAGENET_FID_NPZ", str(repo / "assets/fid_stats/jit_in256_stats.npz"))
    else:
        data = Path(env.get("CIFAR10_PATH", str(repo / "data/cifar10"))) / "cifar-10-batches-py"
        if not data.is_dir():
            raise FileNotFoundError(f"CIFAR dataset missing: {data}; edit environment in the YAML")
        fid = env.get("CIFAR10_FID_NPZ", str(repo / "assets/fid_stats/cifar10_train_fid_stats.npz"))
    if not Path(fid).is_file():
        raise FileNotFoundError(f"FID reference missing: {fid}")
    if plan["init_ema_checkpoint"] and not Path(plan["init_ema_checkpoint"]).is_file():
        raise FileNotFoundError(plan["init_ema_checkpoint"])


def preflight_imagenet(cfg, env, repo):
    """Check cached latents/assets without importing torch or using a GPU."""
    import numpy as np
    assets = Path(env.get("PTFLOW_ASSETS", repo / "assets"))
    cache = Path(env.get("IMAGENET_CACHE_PATH", repo / "data/latents"))
    for split in ("train", "val"):
        arrays = {part: np.load(cache / f"{split}_{part}.npy", mmap_mode="r", allow_pickle=False)
                  for part in ("moments", "moments_flip", "targets")}
        labels = arrays["targets"]
        if labels.ndim != 1 or labels.dtype.kind not in "iu" or len(labels) == 0:
            raise ValueError(f"Invalid {split} labels in {cache}")
        if labels.min() < 0 or labels.max() >= 1000:
            raise ValueError("ImageNet labels must be in [0, 999]")
        for part in ("moments", "moments_flip"):
            if arrays[part].shape != (len(labels), 32, 32, 4) or arrays[part].dtype.kind != "f":
                raise ValueError(f"Invalid latent shape/dtype: {split}_{part}")
        if split == "train":
            counts = np.bincount(labels.astype(np.int64), minlength=1000)
            if (counts == 0).any():
                raise ValueError("ImageNet training subset must retain all 1000 classes")
            per_class = cfg["dataset"].get("subset_per_class")
            if per_class and not (counts == per_class).all():
                raise ValueError(f"Expected exactly {per_class} training images per class")
            if len(labels) < cfg["dataset"]["batch_size"]:
                raise ValueError("Training subset is smaller than the loader batch")
    vae = Path(env.get("VAE_HF_PATH", assets / "sdvae"))
    if not (vae / "config.json").is_file() or not any(vae.glob("*.safetensors")) and not any(vae.glob("*.bin")):
        raise FileNotFoundError(f"Local SD-VAE config/weights missing: {vae}")
    feature = cfg.get("feature", {})
    if feature.get("use_mae", True):
        mae = feature["mae_path"]
        if mae.startswith("hf://"):
            mae = Path(env.get("HF_ROOT", assets / "mae")) / "models/mae/jax" / mae[5:]
        mae = Path(mae)
        if not (mae / "metadata.json").is_file() or not any((mae / name).is_file() for name in ("ema_params.msgpack", "ema_params.pt")):
            raise FileNotFoundError(f"Local MAE artifact missing: {mae}. Reuse/download the assets from the ImageNet run first.")
    fid = Path(env.get("IMAGENET_FID_NPZ", assets / "fid_stats/jit_in256_stats.npz"))
    with np.load(fid, allow_pickle=False) as stats:
        mu = stats["ref_mu"] if "ref_mu" in stats else stats["mu"]
        sigma = stats["ref_sigma"] if "ref_sigma" in stats else stats["sigma"]
        if mu.shape != (2048,) or sigma.shape != (2048, 2048) or not np.isfinite(mu).all() or not np.isfinite(sigma).all():
            raise ValueError(f"Invalid ImageNet FID statistics: {fid}")


def materialize(plan, resume=False):
    root = Path(plan["root"])
    frozen = root / "plan.json"
    if root.exists():
        if not resume:
            raise FileExistsError(f"{root} exists: use --resume or a new output_dir")
        if not frozen.exists() or json.loads(frozen.read_text()) != plan:
            raise ValueError("Cannot resume changed manifest/base config; use a new output_dir")
        return frozen
    root.mkdir(parents=True)
    for subdir in ("configs", "logs", "results", "status", "report"):
        (root / subdir).mkdir()
    dump(frozen, plan)
    for arm in plan["arms"]:
        (root / "configs" / f"{arm['name']}.yaml").write_text(
            yaml.safe_dump(arm["config"], sort_keys=False), encoding="utf-8")
    (root / "runner.py").write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    hashes = {}
    for pattern in ("*.py", "ptflow/*.py", "models/*.py", "utils/*.py", "pipelines/*.py"):
        for path in Path(plan["repo"]).glob(pattern):
            hashes[str(path.relative_to(plan["repo"]))] = hashlib.sha256(path.read_bytes()).hexdigest()
    dump(root / "source_sha256.json", hashes)
    return frozen


def submit(plan, frozen, resume=False):
    root, slurm = Path(plan["root"]), plan["slurm"]
    history = root / "submissions.json"
    submissions = json.loads(history.read_text()) if history.exists() else []
    previous = ""
    if resume and submissions:
        previous = submissions[-1]["job_id"]
        check = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-o", "%A"],
                               capture_output=True, text=True, check=True)
        if previous not in check.stdout.split():
            previous = ""
    for index, stage in enumerate(plan["stages"]):
        command = [sys.executable, str(root / "runner.py"), "worker", str(frozen), "--stage", str(index)]
        script = "#!/bin/bash\nset -euo pipefail\nexec srun --ntasks=1 " + shlex.join(command) + "\n"
        script_path = root / f"job_{index + 1}_{stage['factor']}.sh"
        script_path.write_text(script, encoding="utf-8", newline="\n")
        args = ["sbatch", "--parsable", "--nodes=1", "--ntasks=1", "--export=ALL",
                f"--partition={slurm['partition']}", f"--gpus={slurm['gpu']}:1",
                f"--cpus-per-task={int(slurm['cpus'])}", f"--mem={slurm['memory']}",
                f"--time={slurm['time']}", f"--job-name=pt6-{index + 1}-{stage['factor']}",
                f"--output={root / 'logs' / '%j.out'}", f"--error={root / 'logs' / '%j.err'}"]
        if slurm.get("account"):
            args.append(f"--account={slurm['account']}")
        if previous:
            args.append(f"--dependency=afterany:{previous}")
        result = subprocess.run(args + [str(script_path)], capture_output=True, text=True, check=True)
        ids = [line.split(";", 1)[0] for line in result.stdout.splitlines() if line.split(";", 1)[0].isdigit()]
        if len(ids) != 1:
            raise RuntimeError(f"Cannot identify submitted job; inspect squeue before retrying: {result.stdout}")
        previous = ids[0]
        submissions.append(dict(stage=index, factor=stage["factor"], job_id=previous))
        dump(history, submissions)
        print(f"Submitted {stage['factor']}: {previous}", flush=True)
    print(f"Reports: {root / 'report'}")


def paths(plan, name):
    root = Path(plan["root"])
    work = root / "train" / name
    return root / "configs" / f"{name}.yaml", work, work / "checkpoints" / f"state_{plan['steps']:08d}.pt"


def train_command(plan, arm):
    config, work, _ = paths(plan, arm)
    command = [sys.executable, "-u", "train.py", "--config", str(config), "--workdir", str(work)]
    if plan["init_ema_checkpoint"] and not any((work / "checkpoints").glob("state_*.pt")):
        command += ["--init-ema", plan["init_ema_checkpoint"]]
    return command


def eval_command(plan, entry):
    config, _, checkpoint = paths(plan, entry["arm"])
    root, evaluation = Path(plan["root"]), plan["evaluation"]
    return [sys.executable, "-u", "inference.py", "evaluate", "--ckpt", str(checkpoint),
            "--config", str(config), "--sampler", "B" if entry["n"] else "A",
            "--refine-steps", str(entry["n"]), "--refine-gamma", str(evaluation["gamma"]),
            "--cfg-scale", str(entry["cfg_scale"]), "--seed", str(evaluation["seed"]),
            "--num-samples", str(evaluation["num_samples"]), "--gen-bsz", str(evaluation["batch_size"]),
            *(["--eval-backend", plan["engine"]["evaluation_backend"]]
              if "--eval-backend" in plan["engine"]["inference_flags"] else []),
            "--workdir", str(root / "eval" / entry["result"]),
            "--json-out", str(root / "results" / f"{entry['result']}.json")]


def execute(command, plan, tag):
    root = Path(plan["root"])
    status_path = root / "status" / f"{tag}.json"
    prior = json.loads(status_path.read_text()) if status_path.exists() else {}
    state = dict(status="running", command=command, attempts=int(prior.get("attempts", 0)) + 1,
                 elapsed_s=prior.get("elapsed_s", 0.0), log=str(root / "logs" / f"{tag}.log"),
                 slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                 gpu_name=os.environ.get("PTFLOW_WORKER_GPU_NAME"),
                 gpu_memory_gib=os.environ.get("PTFLOW_WORKER_GPU_MEMORY_GIB"))
    dump(status_path, state)
    print(shlex.join(command), flush=True)
    start = time.monotonic()
    env = dict(os.environ, **plan["environment"], PYTHONUNBUFFERED="1", DRIFT_COMPILE="0",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    with Path(state["log"]).open("a", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=plan["repo"], env=env, stdout=log, stderr=subprocess.STDOUT)
    state.update(status="complete" if result.returncode == 0 else "failed", returncode=result.returncode,
                 elapsed_s=state["elapsed_s"] + time.monotonic() - start)
    dump(status_path, state)
    if result.returncode:
        raise RuntimeError(f"{tag} failed ({result.returncode}); see {state['log']}")


def valid_result(path):
    try:
        return math.isfinite(float(json.loads(Path(path).read_text())["fid"]))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def publish_report(plan, rows):
    """Publish measured results only; local CSV survives W&B connection failures."""
    wb = plan.get("wandb", {})
    if not wb.get("enabled"):
        return
    root = Path(plan["root"])
    report_id = hashlib.sha1((str(root) + ":report").encode("utf-8")).hexdigest()[:16]
    run = None
    try:
        import wandb
        run = wandb.init(project=wb["project"], entity=wb.get("entity"),
                         group=wb["group"], name="ablation-report", job_type="report",
                         id=report_id, resume="allow", mode=wb["mode"], dir=str(root),
                         config={"steps_per_arm": plan["steps"], "training_seed": plan["seed"],
                                 "evaluation": plan["evaluation"], "reference": plan["reference"]})
        columns = ["factor", "value", "arm", "status", "fid", "n", "cfg_scale", "num_samples",
                   "pt/ess_last", "pt/control_ess_last", "pt/lambda_prox_last", "total_time_mean"]
        # A column must keep one type across the categorical and numeric factors.
        data = [[str(row.get(k)) if k == "value" else row.get(k) for k in columns] for row in rows]
        run.log({"ablation/results": wandb.Table(columns=columns, data=data)})
        unique = {row["result_file"]: row for row in rows if row["status"] == "complete"}
        run.summary["completed_evaluations"] = len(unique)
        run.summary["expected_evaluations"] = len({r["result_file"] for r in rows})
        for row in rows:
            if row["status"] == "complete":
                run.summary[f"fid/{row['factor']}/{row['value']}"] = row["fid"]
        artifact = wandb.Artifact(f"ablation-report-{report_id}", type="ablation-report")
        for name in ("ablation.csv", "REPORT.md"):
            artifact.add_file(str(root / "report" / name), name=name)
        if (root / "report/results.csv").exists():
            artifact.add_file(str(root / "report/results.csv"), name="results.csv")
        run.log_artifact(artifact)
        dump(root / "report" / "wandb.json", dict(status="logged", mode=wb["mode"],
                                                   url=getattr(run, "url", None), group=wb["group"]))
    except Exception as error:
        # Remote logging must never discard a completed local experiment.
        dump(root / "report" / "wandb.json", dict(status="upload_failed", error=str(error)))
        print(f"W&B report upload failed; CSV is saved locally: {error}", file=sys.stderr, flush=True)
    finally:
        if run is not None:
            try:
                run.finish()
            except Exception as error:
                dump(root / "report" / "wandb.json", dict(status="upload_failed", error=str(error)))
                print(f"W&B finish failed; CSV is saved locally: {error}", file=sys.stderr, flush=True)


def run_stage(plan, index):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Run inside an allocation exposing exactly one CUDA GPU")
    root, failures = Path(plan["root"]), []
    for entry in plan["stages"][index]["entries"]:
        try:
            _, _, checkpoint = paths(plan, entry["arm"])
            if entry["train"] and not checkpoint.exists():
                execute(train_command(plan, entry["arm"]), plan, f"train_{entry['arm']}")
            if not checkpoint.exists():
                raise FileNotFoundError(f"Missing completed checkpoint: {checkpoint}")
            result = root / "results" / f"{entry['result']}.json"
            if not valid_result(result):
                execute(eval_command(plan, entry), plan, f"eval_{entry['result']}")
            if not valid_result(result):
                raise ValueError(f"Missing or nonfinite FID: {result}")
        except (OSError, ValueError, RuntimeError) as error:
            failures.append(str(error))
            print(str(error), file=sys.stderr, flush=True)
        finally:
            rows = collect(plan)
            if entry["result"] != "reference" or index == 0:
                publish_report(plan, rows)
    dump(root / "status" / f"stage_{index}.json", dict(status="failed" if failures else "complete", failures=failures))
    return 1 if failures else 0


def metric_summary(plan, arm):
    _, work, _ = paths(plan, arm)
    logged = {}
    path = work / "log" / "metrics.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                logged[record["step"]] = record  # Latest record after checkpoint recovery.
            except (ValueError, KeyError):
                continue
    records = [logged[key] for key in sorted(logged)]
    result = {}
    keys = {"pt/ess", "pt/control_ess", "pt/sched_eps", "pt/sched_lambda_prox", "pt/lambda_prox", "total_time"}
    keys.update(k for row in records for k in row if k.startswith("profile/"))
    for key in sorted(keys):
        values = [float(r[key]) for r in records if key in r and math.isfinite(float(r[key]))]
        if values:
            result[key + "_last"] = values[-1]
            result[key + "_mean"] = statistics.mean(values)
    result["prox_ever_active"] = any(r.get("pt/lambda_prox", 0) > 0 for r in records)
    return result


def result_row(plan, entry, summary=None):
    """One self-describing result; used by isolated workers and the final report."""
    root = Path(plan["root"])
    arm = next(a for a in plan["arms"] if a["name"] == entry["arm"])
    cfg, settings = arm["config"], arm["settings"]
    config_path, work, checkpoint = paths(plan, entry["arm"])
    config_json = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    result_path = root / "results" / f"{entry['result']}.json"
    row = dict(factor=entry["factor"], value=entry["value"], arm=entry["arm"], result_id=entry["result"],
               shared_reference=entry["result"] == "reference", status="pending_or_failed",
               proposal=settings["proposal"], eps=settings["eps"], K=settings["K"],
               lambda_prox_max=settings["lambda_prox_max"],
               scale_mode=cfg["pt"]["scale_mode"], scale_K=cfg["pt"]["scale_K"],
               alpha_def=cfg["pt"]["schedule"]["alpha_def_start"],
               n=entry["n"], gamma=plan["evaluation"]["gamma"], cfg_scale=entry["cfg_scale"],
               sampler="B" if entry["n"] else "A", train_seed=plan["seed"],
               eval_seed=plan["evaluation"]["seed"], steps_requested=plan["steps"],
               ema_decay=cfg["train"]["ema_decay"], dataset=cfg.get("pipeline"),
               subset_per_class=cfg["dataset"].get("subset_per_class"),
               subset_seed=cfg["dataset"].get("subset_seed"),
               subset_indices_sha256=cfg["dataset"].get("subset_indices_sha256"),
               latent_cache=plan["environment"].get("IMAGENET_CACHE_PATH"),
               fid_reference=plan["environment"].get("IMAGENET_FID_NPZ" if cfg.get("pipeline") == "imagenet_latent" else "CIFAR10_FID_NPZ"),
               model_hidden_size=cfg["model"].get("hidden_size"), model_depth=cfg["model"].get("depth"),
               train_batch_size=cfg["train"]["train_batch_size"],
               gen_per_label=cfg["train"]["forward_dict"]["gen_per_label"],
               grad_accum_steps=cfg["train"].get("grad_accum_steps", 1),
               config_path=str(config_path), checkpoint=str(checkpoint), config_json=config_json,
               config_sha256=hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
               result_file=str(result_path),
               backend=plan.get("engine", {}).get("evaluation_backend", "unknown"),
               num_samples_requested=plan["evaluation"]["num_samples"],
               **(metric_summary(plan, entry["arm"]) if summary is None else summary))
    if valid_result(result_path):
        measured = json.loads(result_path.read_text())
        row.update(status="complete", fid=measured["fid"], num_samples=measured.get("num_samples"),
                   eval_elapsed_s=measured.get("elapsed_s"), checkpoint_step=measured.get("step"),
                   eval_generation_s=measured.get("gen_time"), seed=measured.get("seed", plan["evaluation"]["seed"]))
    for label, tag in (("train", f"train_{entry['arm']}"), ("eval", f"eval_{entry['result']}")):
        status = root / "status" / f"{tag}.json"
        if status.exists():
            state = json.loads(status.read_text())
            row.update({f"{label}_wall_s": state.get("elapsed_s"), f"{label}_status": state.get("status"),
                        f"{label}_gpu": state.get("gpu_name"), f"{label}_job_id": state.get("slurm_job_id"),
                        f"{label}_gpu_memory_gib": state.get("gpu_memory_gib")})
            if label == "train":
                row.update(train_elapsed_s=state.get("elapsed_s"), training_status=state.get("status"))
    row["warning"] = ""
    if settings["lambda_prox_max"] > 0 and not row["prox_ever_active"]:
        row["warning"] = "No logged nonzero prox weight; check estimator health before interpreting this arm"
    return row


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def collect(plan):
    root = Path(plan["root"])
    summaries = {arm["name"]: metric_summary(plan, arm["name"]) for arm in plan["arms"]}
    rows = []
    for stage in plan["stages"]:
        for entry in stage["entries"]:
            rows.append(result_row(plan, entry, summaries[entry["arm"]]))
    report = root / "report"
    report.mkdir(exist_ok=True)
    write_csv(report / "ablation.csv", rows)
    unique = {}
    for row in rows:
        unique.setdefault(row["result_id"], row)
    write_csv(report / "results.csv", list(unique.values()))
    lines = ["# Six-factor PT-Flow screening", "",
             f"Dataset: {plan['arms'][0]['config']['pipeline']}; training examples/class: {plan['arms'][0]['config']['dataset'].get('subset_per_class', 'full dataset')}.",
             f"Training: {len(plan['arms'])} configurations x {plan['steps']} steps; one training seed.",
             f"FID backend: {plan.get('engine', {}).get('evaluation_backend', 'unknown')}.",
             f"Initialization: {'shared generator EMA; fresh PT/optimizers/schedule' if plan['init_ema_checkpoint'] else 'from scratch, common seed'}.",
             "Reference entries repeat the same measurement, not independent runs.", "",
             "| Factor | Value | Status | FID | Logged mean step seconds |",
             "|---|---|---|---:|---:|"]
    for row in rows:
        fid = f"{row['fid']:.4f}" if "fid" in row else "--"
        seconds = row.get("total_time_mean")
        seconds = f"{seconds:.3f}" if seconds is not None else "--"
        lines.append(f"| {row['factor']} | {row['value']} | {row['status']} | {fid} | {seconds} |")
    lines += ["", "## Interpretation", "",
              "Short-run screening is not a final convergence or best-FID claim. Re-evaluate finalists with 50k samples and multiple training seeds.",
              "Step seconds are means of logged blocks, not a separate synchronized throughput benchmark. Evaluation elapsed time includes feature extraction/FID work.",
              "Training elapsed seconds sum completed launcher attempts; a killed allocation can leave incomplete timing. Logs/checkpoints retain progress.",
              "lambda=0 still trains the potential. Refinement decreases a prox objective, which does not guarantee improved FID.",
              "Afterany dependencies let independent sweeps continue after failures; missing results are not successes.", "", "## Warnings", ""]
    lines.extend(sorted({f"- {r['arm']}: {r['warning']}" for r in rows if r["warning"]}) or ["No inactive-prox warnings."])
    (report / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "submit", "worker", "report"))
    parser.add_argument("file", help="Manifest for plan/submit; frozen plan.json for worker/report")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", type=int, choices=range(6))
    parser.add_argument("--sync-wandb", action="store_true", help="For report: also upload the current CSV/table")
    args = parser.parse_args()
    if args.action in ("worker", "report"):
        plan = json.loads(Path(args.file).read_text(encoding="utf-8"))
        if args.action == "report":
            rows = collect(plan)
            if args.sync_wandb:
                publish_report(plan, rows)
            return 0
        if args.stage is None:
            parser.error("worker requires --stage 0..5")
        return run_stage(plan, args.stage)
    plan = build_plan(args.file)
    describe(plan)
    if args.action == "submit":
        preflight(plan)
        frozen = materialize(plan, args.resume)
        collect(plan)
        submit(plan, frozen, args.resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
