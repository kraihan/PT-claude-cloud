# PT-Flow

**New speed/quality experiments (18 September 2026):** [SPEED_QUALITY_GUIDE.md](SPEED_QUALITY_GUIDE.md).
Includes measured phase profiling, optimized OT/features, checkpoint finetuning,
streaming FID and parallel Palmetto sweeps. Recipes are unvalidated candidates;
no H100 speedup or improved FID is claimed until they are measured.

**Current repaired launch instructions:** [RUN_GUIDE.md](RUN_GUIDE.md).
**Verified findings and limits:** [AUDIT.md](AUDIT.md).
Use `configs/gen/ptflow_cifar10_t4.yaml` for the new two-T4 CIFAR profile and
`configs/gen/ptflow_imagenet_B_fresh.yaml` for the ImageNet B/2 comparison profile.
The commands below are legacy notes, retained for reference; the run guide
supersedes their setup, smoke schedules and evaluation instructions.

# Legacy run notes

Commands only. Run in order. Use `tmux` for anything long.

---

## 0. Setup (once)

```bash
cd /workspace/code && rm -rf PT-Flow && unzip -q ~/PT-Flow.zip && cd PT-Flow
```

```bash
pip install -U pip
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install diffusers==0.29.1 transformers==4.46.1 huggingface_hub==0.24.7 \
            safetensors accelerate numpy==1.26.4 tqdm wandb absl-py einops \
            psutil pyyaml scipy torch-fidelity gdown "jax[cpu]==0.4.31" flax==0.8.5
pip uninstall -y flash-attn flash_attn flash_attn_interface
```

```bash
python -c "import torch,diffusers,safetensors.torch,accelerate;from diffusers.models import AutoencoderKL;print('env ok |',torch.cuda.get_device_name(0))"
```

Edit `utils/env.py`:

```bash
cat > utils/env.py << 'EOF'
from __future__ import annotations

A = "/workspace/assets"

HF_REPO_ID = "Goodeat/drifting"
HF_ROOT = f"{A}/mae"
VAE_HF_PATH = f"{A}/sdvae"
TORCH_HUB_DIR = f"{A}/torch_hub"

BASELINE_HF_REPO_ID = ""
BASELINE_HF_ROOT = ""
BASELINE_CKPTS = {}

CIFAR10_PATH = "/workspace/data/cifar10"
CIFAR10_FID_NPZ = f"{A}/fid_stats/cifar10_train_fid_stats.npz"

IMAGENET_PATH = "/workspace/data/imagenet"
IMAGENET_CACHE_PATH = "/workspace/data/latents"
IMAGENET_FID_NPZ = f"{A}/fid_stats/jit_in256_stats.npz"
IMAGENET_PR_NPZ = ""
EOF
```

```bash
python -m misc.download_pretrained
```

```bash
python -m tests.test_math && python -m tests.test_train_smoke
```

---

## 1. CIFAR-10

### 1.1 Data

```bash
python -m misc.download_cifar10
```

### 1.2 FID reference

```bash
python -m scripts.make_cifar10_fid_stats
```

### 1.3 Smoke (20 steps)

```bash
python - << 'EOF'
import yaml
c = yaml.safe_load(open("configs/gen/ptflow_cifar10.yaml"))
c["train"].update(total_steps=20, save_per_step=10, eval_per_step=10**9)
c["logging"]["log_every_k"] = 1
yaml.safe_dump(c, open("configs/gen/_smoke_c10.yaml", "w"), sort_keys=False)
EOF
```

```bash
torchrun --nproc_per_node=1 --master_port=29555 train.py --config configs/gen/_smoke_c10.yaml --workdir /workspace/runs/smoke_c10
```

### 1.4 Train

```bash
tmux new -s c10
```

```bash
cd /workspace/code/PT-Flow && torchrun --nproc_per_node=1 --master_port=29550 train.py --config configs/gen/ptflow_cifar10.yaml --workdir /workspace/runs/ptflow_cifar10
```

Resume after any stop: re-run the identical command.

### 1.5 Monitor

```bash
tail -f /workspace/runs/ptflow_cifar10/log/metrics.jsonl
```

### 1.6 Sample

```bash
python inference.py sample --ckpt /workspace/runs/ptflow_cifar10/checkpoints/state_00020000.pt --config configs/gen/ptflow_cifar10.yaml --sampler A --cfg-scale 1.5 --class-ids "0,1,2,3,4,5,6,7,8,9" --num-rows 4 --save-path /workspace/runs/c10_grid.png
```

### 1.7 Evaluate

```bash
CKPT=/workspace/runs/ptflow_cifar10/checkpoints/state_00020000.pt CONFIG=configs/gen/ptflow_cifar10.yaml NGPU=1 bash scripts/eval_fid/eval_modes.sh
```

---

## 2. ImageNet B

### 2.1 Pipeline check on a 10-class subset (1.5 GB, ~10 min)

```bash
mkdir -p /workspace/data && cd /workspace/data && curl -L -o imagenette2.tgz https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz && tar xzf imagenette2.tgz && mv imagenette2 imagenette && cd /workspace/code/PT-Flow
```

```bash
python -m dataset.latent --data-path /workspace/data/imagenette --target-path /workspace/data/latents_imagenette --local-batch-size 64 --num-workers 4 --pin-memory
```

```bash
ls -la /workspace/data/latents_imagenette
```

### 2.2 Data

```bash
df -h /workspace
```

```bash
hf auth login
hf download ILSVRC/imagenet-1k --repo-type dataset --local-dir /workspace/data/imagenet_raw
```

Accept the dataset terms on huggingface.co first (gated repo).

The download is archives, not `ImageFolder`. Convert to
`/workspace/data/imagenet/{train,val}/<wnid>/*.JPEG` before continuing:

```bash
ls -la /workspace/data/imagenet_raw && find /workspace/data/imagenet_raw -maxdepth 2 -type f | head -20
```

Verify before the cache build — both must print `1000`, and val must list
wnids (`n01440764`), not `.JPEG` filenames:

```bash
ls /workspace/data/imagenet/train | wc -l && ls /workspace/data/imagenet/val | wc -l && ls /workspace/data/imagenet/val | head -3
```

### 2.3 Latent cache (hours, ~85 GB)

```bash
tmux new -s cache
```

```bash
cd /workspace/code/PT-Flow && python -m dataset.latent --data-path /workspace/data/imagenet --target-path /workspace/data/latents --local-batch-size 128 --num-workers 8 --pin-memory
```

### 2.4 FID reference

Place `jit_in256_stats.npz` at `/workspace/assets/fid_stats/`, or skip FID with
`train.eval_per_step: 1000000000`.

```bash
ls -la /workspace/assets/fid_stats/jit_in256_stats.npz
```

### 2.5 Config for 1 GPU

```bash
python - << 'EOF'
import yaml
c = yaml.safe_load(open("configs/gen/ptflow_scratch.yaml"))
c["dataset"]["batch_size"] = 256
c["dataset"]["eval_batch_size"] = 500
c["train"].update(train_batch_size=16, pos_per_sample=32, neg_per_sample=8,
                  grad_accum_steps=4, push_per_step=64, eval_per_step=10000)
c["train"]["forward_dict"]["gen_per_label"] = 16
c["pt"]["prox_max_batch"] = 64
yaml.safe_dump(c, open("configs/gen/ptflow_B_1gpu.yaml", "w"), sort_keys=False)
print("wrote configs/gen/ptflow_B_1gpu.yaml")
EOF
```

### 2.6 Smoke (20 steps)

```bash
python - << 'EOF'
import yaml
c = yaml.safe_load(open("configs/gen/ptflow_B_1gpu.yaml"))
c["train"].update(total_steps=20, save_per_step=10, eval_per_step=10**9)
c["logging"]["log_every_k"] = 1
yaml.safe_dump(c, open("configs/gen/_smoke_b.yaml", "w"), sort_keys=False)
EOF
```

```bash
torchrun --nproc_per_node=1 --master_port=29556 train.py --config configs/gen/_smoke_b.yaml --workdir /workspace/runs/smoke_b
```

### 2.7 Train

```bash
tmux new -s inb
```

```bash
cd /workspace/code/PT-Flow && torchrun --nproc_per_node=1 --master_port=29551 train.py --config configs/gen/ptflow_B_1gpu.yaml --workdir /workspace/runs/ptflow_B
```

### 2.8 Monitor

```bash
tail -f /workspace/runs/ptflow_B/log/metrics.jsonl
```

### 2.9 Evaluate

```bash
CKPT=/workspace/runs/ptflow_B/checkpoints/state_00020000.pt CONFIG=configs/gen/ptflow_B_1gpu.yaml NGPU=1 bash scripts/eval_fid/eval_modes.sh
```

---

## Multi-GPU

Set `--nproc_per_node=N`. Multi-node: also export `NNODES`, `NODE_RANK`,
`MASTER_ADDR`, `MASTER_PORT`.

## L / XL

```bash
python -m tests.test_ckpt_compat
```

```bash
torchrun --nproc_per_node=8 train.py --config configs/gen/ptflow_L.yaml  --workdir /workspace/runs/ptflow_L
torchrun --nproc_per_node=8 train.py --config configs/gen/ptflow_XL.yaml --workdir /workspace/runs/ptflow_XL
```

## OOM

Raise `train.grad_accum_steps`, or lower `train.forward_dict.gen_per_label`.

## Checks

| Where | Expect |
|---|---|
| smoke banner | `32x32x3 (d = 3072)` CIFAR · `32x32x4 (d = 4096)` ImageNet |
| `ess` step 0 | ≈ 1.00 |
| `ess` steady | > 0.3 |
| `sched_eps` | reaches `eps_min` |
| `lambda_prox` | 0 until `prox_warmup`, then ramps |
