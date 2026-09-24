# PT-Flow on Palmetto 2 — ImageNet-1k 256 B/2, 180k steps, save every 10k

All commands run on Palmetto. Fill in `<PARTITION>` and `<ACCOUNT>` from:

    sacctmgr -s show user $USER format=User,Account,Partition%20,QOS%30

Set `NPROC` to the number of A100s on the node you can get (check with
`sinfo -eO "CPUs:8,Memory:9,Gres:14,NodeAIOT:16,NodeList:50"`).
train_batch_size=128, dataset.batch_size=4096, eval_batch_size=256 must all be
divisible by NPROC — 8, 4, 2 all work.

---

## 0. One env file to rule them all:  ~/ptflow_env.sh

    cat > ~/ptflow_env.sh <<'EOF'
    module load anaconda3
    module load cuda
    source activate ptflow

    export WORK=/scratch/$USER/ptflow
    export REPO=$WORK/PT-Flow

    # assets + data (all on scratch; shared ImageNet is read-only)
    export PTFLOW_ASSETS=$WORK/assets
    export PTFLOW_DATA=$WORK/data
    export IN_ROOT=$WORK/data/imagenet_root          # prepared root: train/ + val/
    export IMAGENET_PATH=$IN_ROOT
    export IMAGENET_CACHE_PATH=$WORK/data/latents
    export IMAGENET_FID_NPZ=$PTFLOW_ASSETS/fid_stats/jit_in256_stats.npz

    export DRIFT_COMPILE=0
    export OMP_NUM_THREADS=8

    # wandb  (ROTATE this key after the run — it was shared in chat)
    export WANDB_API_KEY=PUT_YOUR_KEY_HERE
    export WANDB_PROJECT=ptflow-imagenet
    # export WANDB_ENTITY=your_team   # optional
    EOF

Source it in every shell and every SLURM job:  `source ~/ptflow_env.sh`

---

## 1. Repo + env on scratch

    source ~/ptflow_env.sh
    mkdir -p $WORK && cd $WORK
    # put the repo here (git clone <url> PT-Flow, or rsync from your laptop)
    cd $REPO
    pip install -r requirements.txt
    python -m pytest -q          # CPU tests should pass

---

## 2. Download pretrained assets (VAE + MAE + Inception)

Run as a short GPU job (needs a GPU for the Inception check):

    cat > $REPO/dl_assets.slurm <<'EOF'
    #!/bin/bash
    #SBATCH --job-name ptflow-dl
    #SBATCH --partition <PARTITION>
    #SBATCH --account <ACCOUNT>
    #SBATCH --gpus a100:1
    #SBATCH --cpus-per-task 8
    #SBATCH --mem 32gb
    #SBATCH --time 02:00:00
    #SBATCH --output %x_%j.out
    source ~/ptflow_env.sh
    cd $REPO
    python -m misc.download_pretrained
    EOF
    sbatch $REPO/dl_assets.slurm

Note: this does NOT fetch the FID reference npz. That's fine — we run with FID
off (step 6).

---

## 3. Prepare a class-foldered ImageNet root  (train symlink + val reorg)

The shared val is a flat folder; reorganize it into class folders using the CSV.

    source ~/ptflow_env.sh
    mkdir -p $IN_ROOT

    # train is already class-foldered — just link it
    ln -sfn /datasets/imagenet/ImagenetFull/ILSVRC/Data/CLS-LOC/train $IN_ROOT/train

    # val -> 1000 class folders of symlinks
    python - <<'PY'
    import csv, os
    SRC = "/datasets/imagenet/ImagenetFull/ILSVRC/Data/CLS-LOC/val"
    CSV = "/datasets/imagenet/ImagenetFull/LOC_val_solution.csv"
    DST = os.path.join(os.environ["IN_ROOT"], "val")
    os.makedirs(DST, exist_ok=True)
    n = 0
    with open(CSV) as f:
        rr = csv.reader(f); next(rr)
        for row in rr:
            img_id, wnid = row[0], row[1].split()[0]
            d = os.path.join(DST, wnid); os.makedirs(d, exist_ok=True)
            s = os.path.join(SRC, img_id + ".JPEG")
            t = os.path.join(d, img_id + ".JPEG")
            if not os.path.lexists(t): os.symlink(s, t); n += 1
    print("linked", n, "val images into", DST)
    PY

    # sanity: both should print 1000
    ls $IN_ROOT/train | wc -l
    ls $IN_ROOT/val   | wc -l

(If the CSV path differs, find it: `ls /datasets/imagenet/ImagenetFull/*.csv`)

---

## 4. Build the latent cache (train + val)  — the big one-time GPU job

    cat > $REPO/cache_latents.slurm <<'EOF'
    #!/bin/bash
    #SBATCH --job-name ptflow-cache
    #SBATCH --partition <PARTITION>
    #SBATCH --account <ACCOUNT>
    #SBATCH --gpus a100:1
    #SBATCH --cpus-per-task 16
    #SBATCH --mem 64gb
    #SBATCH --time 12:00:00
    #SBATCH --output %x_%j.out
    source ~/ptflow_env.sh
    cd $REPO
    python -m dataset.latent \
      --data-path "$IMAGENET_PATH" \
      --target-path "$IMAGENET_CACHE_PATH" \
      --local-batch-size 256 --num-workers 16 --pin-memory
    EOF
    sbatch $REPO/cache_latents.slurm

Produces 6 files in $IMAGENET_CACHE_PATH:
train_{moments,moments_flip,targets}.npy and val_{...}. ~44 GB total.
It encodes train first (~1.28M imgs), then val (50k).

---

## 5. Make the 180k config (from the B/2 fresh profile)

    source ~/ptflow_env.sh
    cd $REPO
    cp configs/gen/ptflow_imagenet_B_fresh.yaml configs/gen/ptflow_imagenet_B_180k.yaml

    C=configs/gen/ptflow_imagenet_B_180k.yaml
    sed -i 's/^  use_wandb: false/  use_wandb: true/'      $C   # turn wandb on
    sed -i 's/total_steps: 200000/total_steps: 180000/g'   $C   # all 3 schedules
    sed -i 's/save_per_step: 2000/save_per_step: 10000/'   $C   # checkpoint every 10k
    sed -i 's/keep_every: 20000/keep_every: 10000/'        $C   # KEEP every 10k ckpt
    sed -i 's/eval_per_step: 10000/eval_per_step: 0/'      $C   # FID off (no npz)

    # verify
    grep -nE "use_wandb|total_steps|save_per_step|keep_every|eval_per_step" $C

What each does:
- total_steps 180000: the training loop (train.total_steps) + both LR schedules.
- save_per_step 10000: writes a checkpoint every 10k steps.
- keep_every 10000: WITHOUT this, keep_last=2 (hardcoded) deletes old checkpoints
  and only 20k-multiples survive. With keep_every=10000, every 10k checkpoint is
  kept permanently (~18 checkpoints; fine on scratch).
- eval_per_step 0: skips FID (repo has no ImageNet FID stats). You still get all
  training/PT metrics + checkpoints. Compute FID later with inference.py evaluate.

To turn FID ON instead: put a valid jit_in256_stats.npz at $IMAGENET_FID_NPZ and
leave eval_per_step at 10000.

Do NOT change the batch sizes: train_batch_size=128 x gen_per_label=64 = 8192
global particles/step — that's the intended B/2 scale (not the 1024 latent-cache
batch, which is unrelated).

---

## 6. Preflight + wandb login + smoke test

    source ~/ptflow_env.sh; cd $REPO
    wandb login $WANDB_API_KEY

    # preflight (world-size = your GPU count)
    python -m scripts.preflight --config configs/gen/ptflow_imagenet_B_180k.yaml \
      --world-size 8 --check-assets --check-features

    # 20-step smoke to confirm it fits + runs (own workdir)
    python -m scripts.make_run_config --profile imagenet_b --smoke 20 \
      --output configs/gen/in256_smoke.yaml
    torchrun --standalone --nproc_per_node=8 train.py \
      --config configs/gen/in256_smoke.yaml --workdir $WORK/runs/in256_smoke

(Run the smoke inside a GPU allocation/job. If preflight passes and 20 steps run
without OOM, you're clear. If OOM: raise train.grad_accum_steps from 4 to 8 —
same global particle count, less activation memory.)

---

## 7. Launch the real run (auto-resumes across the 72h wall limit)

    cat > $REPO/train_in256.slurm <<'EOF'
    #!/bin/bash
    #SBATCH --job-name ptflow-in256
    #SBATCH --partition <PARTITION>
    #SBATCH --account <ACCOUNT>
    #SBATCH --nodes 1
    #SBATCH --gpus a100:8
    #SBATCH --cpus-per-task 32
    #SBATCH --mem 200gb
    #SBATCH --time 72:00:00
    #SBATCH --signal B:USR1@600
    #SBATCH --requeue
    #SBATCH --output %x_%j.out
    #SBATCH --mail-type END,FAIL
    #SBATCH --mail-user raihan@nwu.ac.bd

    source ~/ptflow_env.sh
    cd $REPO
    NPROC=8   # match --gpus above and your node's A100 count

    # requeue near the wall limit; resumed job auto-picks up latest checkpoint
    resubmit() { echo "wall limit -> requeue"; scontrol requeue $SLURM_JOB_ID; }
    trap resubmit USR1

    srun torchrun --standalone --nproc_per_node=$NPROC train.py \
      --config configs/gen/ptflow_imagenet_B_180k.yaml \
      --workdir $WORK/runs/in256_B_180k &
    wait
    EOF
    sbatch $REPO/train_in256.slurm

Resume is automatic: re-running the same command (or the requeue) restores the
generator, optimizer, EMA, PT schedule, and FP16 scaler from the latest
state_*.pt in the workdir. wandb resumes the same run (id is derived from the
workdir path).

If you get fewer than 8 GPUs: set both `--gpus a100:N` and `NPROC=N`, and if
memory is tight raise grad_accum_steps in the config (keeps the 8192 particles).

---

## 8. Monitor

    squeue -u $USER
    tail -f $WORK/runs/in256_B_180k/*_ *.out 2>/dev/null   # or the %x_%j.out file
    # wandb: project "ptflow-imagenet" in your browser

With wandb on, metrics stream there. Watch: g_norm finite, optimizer_step_skipped
(rare), pt/ess and pt/control_ess (should not collapse to ~0), pt/lambda_prox
(should ramp up after warmup), pt/sched_eps (anneals). If PT weight stays 0 or ESS
stays broken while the generator still improves, the PT estimator isn't tracking —
note it honestly rather than reporting only the feature-driven FID.

Checkpoints land in $WORK/runs/in256_B_180k/checkpoints/state_000NNNNN.pt every 10k.

---

## Order of jobs
1. dl_assets.slurm      (~1h, once)
2. step 3 val prep      (minutes, login node ok — just symlinks)
3. cache_latents.slurm  (few hours, once)  -> wait for it to finish
4. preflight + smoke    (minutes)
5. train_in256.slurm    (chains itself to 180k)
