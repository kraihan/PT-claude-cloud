"""Submit ImageNet preparation -> smoke -> parallel submission using Slurm.

Uses sbatch and srun, with afterok dependencies instead of a terminal-side
wait. Can adopt an existing ptin-prepare job without cancelling it.
"""
import argparse
import getpass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

try:
    from . import run_parallel_ablation as parallel
except ImportError:
    import run_parallel_ablation as parallel

six = parallel.six


def queue():
    result = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-o", "%A|%j"],
                            capture_output=True, text=True, check=True)
    return dict(line.strip().split("|", 1) for line in result.stdout.splitlines() if "|" in line)


def preparation_dependency(job_id, active, resolved):
    job_id = str(job_id)
    if not job_id.isdigit():
        raise ValueError("--prepare-job must be a numeric Slurm job ID")
    if job_id in active:
        if active[job_id] != "ptin-prepare":
            raise ValueError(f"Job {job_id} is not ptin-prepare; refusing to adopt an unrelated job")
        return job_id
    result = subprocess.run(["sacct", "-n", "-P", "-X", "-j", job_id,
                             "--format=JobID,JobName,State,ExitCode,User"],
                            capture_output=True, text=True, check=True)
    records = [line.strip().split("|") for line in result.stdout.splitlines()]
    found = [row for row in records if len(row) >= 5 and row[0] == job_id]
    if len(found) != 1 or found[0][1:5] != ["ptin-prepare", "COMPLETED", "0:0", getpass.getuser()]:
        raise ValueError(f"Preparation job {job_id} is not active or successfully completed. Inspect sacct/logs first.")
    if not resolved.is_file():
        raise FileNotFoundError(f"Completed preparation has no resolved manifest: {resolved}")
    # Completed jobs can age out of the controller's dependency window.
    return None


def submit(manifest, setup, prepare_job=None):
    plan = six.build_plan(manifest)
    if plan["arms"][0]["config"]["pipeline"] != "imagenet_latent":
        raise ValueError("This submission workflow is for ImageNet latent experiments")
    repo = Path(plan["repo"])
    setup = Path(setup).resolve()
    resolved = setup / "manifest.yaml"
    slurm = plan["slurm"]
    if slurm.get("gpu") != "a100" or slurm.get("constraint") != "gpu_a100_40gb":
        raise ValueError("This workflow requires gpu: a100 and constraint: gpu_a100_40gb")
    with parallel.lock_file(setup / "slurm_submission.lock"):
        active = queue()
        history_path = setup / "slurm_workflow.json"
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        owned = {row["job_id"] for row in history}
        busy = owned.intersection(active)
        busy |= {job for job, name in active.items() if name in ("ptin-smoke", "ptin-dispatch")}
        if busy:
            raise RuntimeError("An ImageNet setup workflow is already active: " + ", ".join(sorted(busy)))
        running = parallel.active_jobs(plan)
        if running:
            raise RuntimeError("ImageNet ablation jobs already active: " + ", ".join(sorted(running)))
        other_preparation = {job for job, name in active.items() if name == "ptin-prepare"} - {str(prepare_job)}
        if other_preparation:
            raise RuntimeError("Preparation already active. Reuse it with --prepare-job " + ", ".join(sorted(other_preparation)))

        def dispatch(name, command, *, gpu=False, dependency=None):
            script = setup / f"{name}.sbatch"
            script.write_text("#!/bin/bash\nset -euo pipefail\nexec srun --ntasks=1 " + shlex.join(command) + "\n",
                              encoding="utf-8", newline="\n")
            args = ["sbatch", "--parsable", "--nodes=1", "--ntasks=1", "--export=ALL",
                    f"--partition={slurm['partition']}", f"--job-name={name}", f"--chdir={repo}",
                    f"--output={setup / (name + '_%j.log')}", f"--error={setup / (name + '_%j.log')}"]
            if slurm.get("account"):
                args.append(f"--account={slurm['account']}")
            if gpu:
                args += ["--gpus=a100:1", "--constraint=gpu_a100_40gb", "--cpus-per-task=8", "--mem=64G", "--time=01:00:00"]
            elif name == "ptin-prepare":
                args += ["--cpus-per-task=2", "--mem=16G", "--time=02:00:00"]
            else:
                args += ["--cpus-per-task=1", "--mem=4G", "--time=00:20:00"]
            if dependency:
                args += [f"--dependency=afterok:{dependency}", "--kill-on-invalid-dep=yes"]
            # Prevent an exported SBATCH_WAIT from restoring terminal-side waits.
            env = {key: value for key, value in os.environ.items() if not key.startswith("SBATCH_")}
            result = subprocess.run(args + [str(script)], env=env, capture_output=True, text=True, check=True)
            ids = [line.split(";", 1)[0] for line in result.stdout.splitlines() if line.split(";", 1)[0].isdigit()]
            if len(ids) != 1:
                raise RuntimeError(f"Unrecognized sbatch result; inspect squeue before retrying: {result.stdout}")
            job_id = ids[0]
            history.append(dict(job_id=job_id, name=name, dependency=dependency, gpu=gpu))
            six.dump(history_path, history)
            print(f"{name}: {job_id}" + (f" (afterok:{dependency})" if dependency else ""), flush=True)
            return job_id

        if prepare_job:
            dependency = preparation_dependency(prepare_job, active, resolved)
            print(f"Reuse preparation job {prepare_job}", flush=True)
        else:
            dependency = dispatch("ptin-prepare", [sys.executable, str(repo / "scripts/prepare_imagenet_ablation.py"),
                                  "prepare", str(Path(manifest).resolve()), "--setup-dir", str(setup)])
        smoke = dispatch("ptin-smoke", [sys.executable, str(repo / "scripts/prepare_imagenet_ablation.py"),
                                       "smoke", str(resolved), "--setup-dir", str(setup)], gpu=True, dependency=dependency)
        dispatch("ptin-dispatch", [sys.executable, str(repo / "scripts/run_parallel_ablation.py"),
                                  "submit", str(resolved), "--gpu", "a100"], dependency=smoke)
    print("Submitted. You can disconnect. Slurm will run preparation, smoke, then submit the parallel study.")
    print(f"Setup logs and job IDs: {setup}")


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(repo / "configs/ablation/six_imagenet_a100_4k.yaml"))
    parser.add_argument("--setup-dir", default=str(repo.parent / "runs/ablation_imagenet_B_subset50_a100_4k_setup"))
    parser.add_argument("--prepare-job", help="Reuse an active/completed ptin-prepare job, e.g. 16115552")
    args = parser.parse_args()
    submit(args.manifest, args.setup_dir, args.prepare_job)


if __name__ == "__main__":
    main()
