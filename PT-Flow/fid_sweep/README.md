# fid_sweep — dense-CFG 50k FID sweep + master CSV

This is **FID evaluation orchestration**, independent of the new-s² work: FID
sampling (Mode A) is bit-identical old vs new, so these numbers don't depend on
the §2.5 repair. It wraps the repo's own `scripts/performance_palmetto.sh` (which
is left untouched) to fix the dense 12-cfg / 50k / 12-concurrent knobs, let you
pick the GPU, and fold every run into one cumulative `fid_master.csv`.

## Files (upload both into the cluster repo)

Put them at `$REPO/fid_sweep/` on the cluster
(`/scratch/$USER/ptflow/speed_quality_v1/PT-Flow/fid_sweep/`):

```
$REPO/fid_sweep/submit_cfg_sweep.sh    # submitter (run this)
$REPO/fid_sweep/append_fid_master.py   # master-CSV appender (called by the chained job)
```

`submit_cfg_sweep.sh` calls `append_fid_master.py` as `fid_sweep/append_fid_master.py`
relative to `$REPO`, so the folder name and location matter.

## Smoke test first (small, ~minutes)

`FID_SAMPLES` must be a multiple of `num_classes` (1000 for ImageNet-256), so the
smallest valid run is 1000 samples. One cfg, one job:

```bash
GPU_TYPE=a100 FID_SAMPLES=1000 CFG_VALUES="1.0" MAX_JOBS=1 WALLTIME=00:30:00 \
  bash fid_sweep/submit_cfg_sweep.sh
```

Then confirm: the array + a `fid-master` job appear in `squeue --me`; when done,
`$OUT/results/cfg1.0_seed42_emaprimary.json` exists, `$OUT/report/REPORT.md` is
written, and `$WORK/results/fid_master.csv` has one row.

You can also dry-run the appender on any finished sweep dir without a GPU:

```bash
python fid_sweep/append_fid_master.py --root <sweep_dir> \
    --master $WORK/results/fid_master.csv --gpu-type a100 --stamp test
```

## The real 12-config sweep

```bash
bash fid_sweep/submit_cfg_sweep.sh
```

Defaults already encode your request: `CFG_VALUES="0.0 … 2.2"` (12), `FID_SAMPLES=50000`,
`MAX_JOBS=12`, `SEEDS=42`, `GEN_BSZ=16`, `GPU_TYPE=h200`, checkpoint
`state_00014000.pt`. Override any via env var, e.g. `GPU_TYPE=a100 bash fid_sweep/submit_cfg_sweep.sh`.

## GPU choice ("what gets fast")

Raw throughput ranks **h200 > h100 > a100**, so h200 is the default. But total
wall-clock = queue wait + compute; if `squeue`/`sinfo` shows h200 backed up, a
sooner-starting a100 can finish first. Switch with `GPU_TYPE=a100` (or `h100`).
The card used is recorded in the `gpu_type` column of the master CSV.

## What lands where

- `$OUT/results/*.json` — one FID row per cfg (from `inference.py evaluate`).
- `$OUT/report/REPORT.md`, `fid.csv`, `fid_aggregate.csv` — per-run report
  (performance_palmetto.sh's own dependent job).
- `$WORK/results/fid_master.csv` — **cumulative** master, updated by the chained
  `fid-master` job. De-duplicated on (ckpt, step, cfg, seed, ema, mode,
  num_samples, fid_ref): re-running a setting **updates its row in place** rather
  than adding a duplicate. Columns: submit_stamp, gpu_type, ckpt_short, ckpt,
  step, cfg_scale, seed, ema_decay, mode, num_samples, fid, isc_mean, gen_bsz,
  world_size, backend, fid_ref, eps, pt_w, elapsed_s, run_out, result_file.
