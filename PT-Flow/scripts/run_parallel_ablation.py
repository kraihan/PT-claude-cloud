"""Schedule independent one-GPU ablations from an existing frozen plan.

Eleven train+FID tasks and six reference-checkpoint evaluations; a final CPU
job collects results. Workers never concurrently write one CSV or W&B run.
"""
import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

try:
    from . import run_six_ablation as six
except ImportError:
    import run_six_ablation as six


@contextmanager
def lock_file(path):
    """OS-released lock, including after a killed submission/report process."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            if not path.stat().st_size:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def tasks(plan):
    unique = {}
    for stage in plan["stages"]:
        for entry in stage["entries"]:
            unique.setdefault(entry["result"], entry)
    return list(unique.values())


def active_jobs(plan):
    root = Path(plan["root"])
    ids = set()
    for path in (root / "submissions.json", root / "parallel/submissions.json"):
        if path.exists():
            ids.update(str(row["job_id"]) for row in json.loads(path.read_text()))
    if not ids:
        return set()
    response = subprocess.run(["squeue", "-h", "-u", six.getpass.getuser(), "-o", "%A"],
                              capture_output=True, text=True, check=True)
    return ids.intersection(response.stdout.split())


def wait_idle(plan, seconds=60):
    deadline = time.monotonic() + seconds
    while True:
        active = active_jobs(plan)
        if not active:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("Existing ablation jobs still active: " + ", ".join(sorted(active)) +
                               ". Cancel that ablation chain or wait before resubmitting; checkpoints are retained.")
        print("Waiting for previous ablation jobs: " + ", ".join(sorted(active)), flush=True)
        time.sleep(min(5, max(0, deadline - time.monotonic())))


def sbatch_args(plan, name, *, gpu=True, dependency=None):
    slurm = plan["slurm"]
    logs = Path(plan["root"]) / "parallel/logs"
    args = ["sbatch", "--parsable", "--nodes=1", "--ntasks=1", "--export=ALL",
            f"--partition={slurm['partition']}", f"--job-name={name}",
            f"--output={logs / '%j.out'}", f"--error={logs / '%j.err'}"]
    if slurm.get("account"):
        args.append(f"--account={slurm['account']}")
    if gpu:
        kind = slurm.get("gpu", "any")
        args += ["--gpus=1" if kind == "any" else f"--gpus={kind}:1",
                 f"--cpus-per-task={int(slurm['cpus'])}", f"--mem={slurm['memory']}", f"--time={slurm['time']}"]
        if slurm.get("constraint"):
            args.append(f"--constraint={slurm['constraint']}")
    else:
        args += ["--cpus-per-task=2", "--mem=4G", "--time=00:20:00"]
    if dependency:
        args.append("--dependency=afterany:" + ":".join(dependency))
    return args


def submit(plan, *, wait_seconds=60):
    root = Path(plan["root"])
    if not (root / "plan.json").is_file():
        raise ValueError("Pass an existing experiment's frozen plan.json")
    six.preflight(plan)
    with lock_file(root / "parallel/submission.lock"):
        wait_idle(plan, wait_seconds)
        folder = root / "parallel"
        (folder / "logs").mkdir(exist_ok=True)
        frozen = folder / "plan.json"
        if frozen.exists():
            prior = json.loads(frozen.read_text())
            # A resource choice can change on retry; experiment settings cannot.
            prior["slurm"] = plan["slurm"]
            if prior != plan:
                raise ValueError("Cannot reuse parallel results with changed experiment settings")
        six.dump(frozen, plan)
        runtime = folder / "runtime"
        runtime.mkdir(exist_ok=True)
        for name, source in (("run_parallel_ablation.py", Path(__file__)),
                             ("run_six_ablation.py", Path(six.__file__))):
            (runtime / name).write_bytes(source.read_bytes())
        history_path = folder / "submissions.json"
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        ids, reference_job = [], None
        reference_exists = six.paths(plan, "reference")[2].is_file()

        def dispatch(name, command, *, gpu=True, dependency=None):
            script = folder / f"{name}.sh"
            script.write_text("#!/bin/bash\nset -euo pipefail\nexec srun --ntasks=1 " + shlex.join(command) + "\n",
                              encoding="utf-8", newline="\n")
            response = subprocess.run(sbatch_args(plan, name, gpu=gpu, dependency=dependency) + [str(script)],
                                      capture_output=True, text=True, check=True)
            found = [line.split(";", 1)[0] for line in response.stdout.splitlines()
                     if line.split(";", 1)[0].isdigit()]
            if len(found) != 1:
                raise RuntimeError(f"Cannot parse submission result; inspect squeue: {response.stdout}")
            job_id = found[0]
            history.append(dict(job_id=job_id, task=name, gpu=gpu))
            six.dump(history_path, history)
            print(f"Submitted {name}: {job_id}", flush=True)
            return job_id

        for index, entry in enumerate(tasks(plan)):
            result = root / "results" / f"{entry['result']}.json"
            if six.valid_result(result) and six.paths(plan, entry["arm"])[2].is_file():
                print(f"Reuse completed result: {entry['result']}", flush=True)
                continue
            dependency = None
            if not entry["train"] and not reference_exists:
                if reference_job is None:
                    raise RuntimeError("Missing reference job/checkpoint")
                dependency = [reference_job]
            command = [sys.executable, str(runtime / "run_parallel_ablation.py"), "worker", str(frozen), "--task", str(index)]
            job_id = dispatch(f"pt6p-{entry['result']}", command, dependency=dependency)
            ids.append(job_id)
            if entry["result"] == "reference":
                reference_job = job_id
        report_command = [sys.executable, str(runtime / "run_parallel_ablation.py"), "report", str(frozen), "--sync-wandb"]
        dispatch("pt6p-report", report_command, gpu=False, dependency=ids or None)
        print(f"Requested {len(ids)} one-GPU jobs plus one CPU report job. Slurm determines concurrency.")
        print(f"Per-result CSVs: {root / 'report/rows'}")
        print(f"Combined CSV: {root / 'report/results.csv'}")


def hardware():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This worker requires exactly one allocated CUDA GPU")
    # A visible GPU may still be unsupported by the installed CUDA wheel (e.g.
    # Palmetto P100 with a recent cu130 wheel). Exercise real kernels now.
    probe = torch.ones((8, 8), device="cuda")
    value = (probe @ probe).sum().item()
    torch.cuda.synchronize()
    if value != 512:
        raise RuntimeError("CUDA arithmetic self-check failed")
    return dict(gpu_name=torch.cuda.get_device_name(0),
                gpu_memory_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
                compute_capability=list(torch.cuda.get_device_capability(0)),
                node=socket.gethostname(), torch_version=torch.__version__)


def publish_result(plan, entry, row, csv_path):
    """Each parallel evaluation owns a different W&B run; no shared writers."""
    wb = plan.get("wandb", {})
    if not wb.get("enabled"):
        return
    root = Path(plan["root"])
    run_id = hashlib.sha1((str(root) + ":eval:" + entry["result"]).encode()).hexdigest()[:16]
    run = None
    try:
        import wandb
        run = wandb.init(project=wb["project"], entity=wb.get("entity"), group=wb["group"],
                         name="eval/" + entry["result"], job_type="evaluation", id=run_id,
                         resume="allow", mode=wb["mode"], dir=str(root),
                         config={k: row.get(k) for k in ("proposal", "eps", "K", "lambda_prox_max", "n", "gamma",
                                                       "cfg_scale", "train_seed", "eval_seed", "backend", "ema_decay")})
        run.summary.update({k: value for k, value in row.items() if value is not None and k != "config_json"})
        artifact = wandb.Artifact("ablation-result-" + run_id, type="ablation-result")
        artifact.add_file(str(csv_path), name="result.csv")
        artifact.add_file(row["config_path"], name="config.yaml")
        run.log_artifact(artifact)
        six.dump(root / "status" / f"wandb_{entry['result']}.json", dict(status="logged", url=getattr(run, "url", None)))
    except Exception as error:
        six.dump(root / "status" / f"wandb_{entry['result']}.json", dict(status="upload_failed", error=str(error)))
        print(f"W&B upload failed; local result retained: {error}", file=sys.stderr)
    finally:
        if run is not None:
            try:
                run.finish()
            except Exception as error:
                print(f"W&B finish failed; local result retained: {error}", file=sys.stderr)


def worker(plan, index):
    entry = tasks(plan)[index]
    root = Path(plan["root"])
    error = None
    info = {}
    try:
        info = hardware()
        os.environ["PTFLOW_WORKER_GPU_NAME"] = info["gpu_name"]
        os.environ["PTFLOW_WORKER_GPU_MEMORY_GIB"] = str(info["gpu_memory_gib"])
        _, _, checkpoint = six.paths(plan, entry["arm"])
        if entry["train"] and not checkpoint.is_file():
            six.execute(six.train_command(plan, entry["arm"]), plan, "train_" + entry["arm"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Reference/training checkpoint unavailable: {checkpoint}")
        result = root / "results" / f"{entry['result']}.json"
        if not six.valid_result(result):
            six.execute(six.eval_command(plan, entry), plan, "eval_" + entry["result"])
        if not six.valid_result(result):
            raise ValueError(f"Missing/nonfinite FID: {result}")
    except Exception as failure:
        error = str(failure)
        print(error, file=sys.stderr, flush=True)
    finally:
        six.dump(root / "status" / f"task_{entry['result']}.json",
                 dict(status="failed" if error else "complete", error=error, **info))
        row = six.result_row(plan, entry)
        row.update(worker_gpu=info.get("gpu_name"), worker_gpu_memory_gib=info.get("gpu_memory_gib"), error=error)
        if error and row["status"] != "complete":
            row["status"] = "failed"
        csv_path = root / "report/rows" / f"{entry['result']}.csv"
        six.write_csv(csv_path, [row])
        publish_result(plan, entry, row, csv_path)
    return 1 if error else 0


def report(plan, sync=False):
    root = Path(plan["root"])
    with lock_file(root / "parallel/report.lock"):
        rows = six.collect(plan)
        # Preserve per-worker failure/hardware metadata in the final combined file.
        for row in rows:
            status = root / "status" / f"task_{row['result_id']}.json"
            if status.exists():
                saved = json.loads(status.read_text())
                row.update(worker_gpu=saved.get("gpu_name"), worker_gpu_memory_gib=saved.get("gpu_memory_gib"), error=saved.get("error"))
                if saved["status"] == "failed" and row["status"] != "complete":
                    row["status"] = "failed"
        unique = {}
        for row in rows:
            unique.setdefault(row["result_id"], row)
        six.write_csv(root / "report/results.csv", list(unique.values()))
        six.write_csv(root / "report/ablation.csv", rows)
        if sync:
            six.publish_report(plan, rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "submit", "worker", "report"))
    parser.add_argument("plan", help="Manifest YAML for a new study, or an existing frozen plan.json")
    parser.add_argument("--gpu", default=None, help="Default: saved GPU type. a100/h100/h200; any also permits unsupported old GPUs")
    parser.add_argument("--task", type=int)
    parser.add_argument("--wait-idle", type=int, default=60)
    parser.add_argument("--sync-wandb", action="store_true")
    args = parser.parse_args()
    manifest = Path(args.plan).suffix.lower() in (".yaml", ".yml")
    if manifest and args.action not in ("submit", "plan"):
        parser.error("worker/report require a frozen plan.json")
    plan = six.build_plan(args.plan) if manifest else json.loads(Path(args.plan).read_text())
    if args.action == "plan":
        six.describe(plan)
        return 0
    if args.action == "submit":
        plan = copy.deepcopy(plan)
        if args.gpu:
            plan["slurm"]["gpu"] = args.gpu.lower()
        if manifest:
            six.preflight(plan)
            six.materialize(plan, resume=Path(plan["root"]).exists())
        six.describe(plan)
        submit(plan, wait_seconds=args.wait_idle)
    elif args.action == "worker":
        if args.task is None or not 0 <= args.task < len(tasks(plan)):
            parser.error("worker requires a valid --task index")
        return worker(plan, args.task)
    else:
        report(plan, args.sync_wandb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
