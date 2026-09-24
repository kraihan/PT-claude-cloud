"""Append a run's FID results into a cumulative master CSV.

Reads every ``$ROOT/results/*.json`` written by ``inference.py evaluate`` (the
per-cfg FID rows a performance_palmetto.sh eval sweep produces) and merges them
into one growing master CSV, de-duplicating on the evaluation identity so a
re-run updates its row instead of appending a duplicate.

    python fid_sweep/append_fid_master.py \
        --root  /scratch/$USER/ptflow/results/<sweep_dir> \
        --master /scratch/$USER/ptflow/results/fid_master.csv \
        --gpu-type h200 --stamp 20260921_120000

Nothing here touches the model or the FID numbers; it only collates the JSON the
eval jobs already wrote.  Output is CSV only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

# Stable column order; any extra keys found in the JSON are appended after these.
PREFERRED = [
    "submit_stamp", "gpu_type", "ckpt_short", "ckpt", "step",
    "cfg_scale", "seed", "ema_decay", "mode", "num_samples",
    "fid", "isc_mean", "gen_bsz", "world_size", "backend",
    "fid_ref", "eps", "pt_w", "elapsed_s", "run_out", "result_file",
]

# The evaluation identity: same key => same experiment => newest row wins.
KEY_FIELDS = ("ckpt", "step", "cfg_scale", "seed", "ema_decay",
              "mode", "num_samples", "fid_ref")


def _norm(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _key(row: dict) -> str:
    return "|".join(_norm(row.get(k)) for k in KEY_FIELDS)


def _short_ckpt(ckpt: str) -> str:
    p = Path(str(ckpt))
    try:
        return f"{p.parent.parent.name}/{p.name}"
    except Exception:
        return p.name


def load_master(master: Path) -> dict:
    rows = {}
    if master.exists():
        with master.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows[_key(row)] = row
    return rows


def collect_run(root: Path, *, gpu_type: str, stamp: str) -> list:
    results = sorted((root / "results").glob("*.json"))
    out = []
    for path in results:
        try:
            r = json.loads(path.read_text())
        except (ValueError, OSError) as e:
            print(f"  skip {path.name}: {e}")
            continue
        if "fid" not in r or not math.isfinite(float(r["fid"])):
            print(f"  skip {path.name}: missing/nonfinite FID")
            continue
        row = {
            "submit_stamp": stamp,
            "gpu_type": gpu_type,
            "ckpt": r.get("ckpt", ""),
            "ckpt_short": _short_ckpt(r.get("ckpt", "")),
            "step": r.get("step", ""),
            "cfg_scale": r.get("cfg_scale", ""),
            "seed": r.get("seed", ""),
            "ema_decay": r.get("ema_decay") or "primary",
            "mode": r.get("mode", ""),
            "num_samples": r.get("num_samples", ""),
            "fid": r.get("fid", ""),
            "isc_mean": r.get("isc_mean", ""),
            "gen_bsz": r.get("gen_bsz", ""),
            "world_size": r.get("world_size", ""),
            "backend": r.get("backend", ""),
            "fid_ref": r.get("fid_ref", ""),
            "eps": r.get("eps", ""),
            "pt_w": r.get("pt_w", ""),
            "elapsed_s": r.get("elapsed_s", ""),
            "run_out": str(root),
            "result_file": str(path),
        }
        out.append(row)
    return out


def _cfg_of(r):
    try:
        return float(r.get("cfg_scale", "nan"))
    except (ValueError, TypeError):
        return float("nan")


def _columns_for(rows_iter) -> list:
    ordered = list(PREFERRED)
    for row in rows_iter:
        for k in row:
            if k not in ordered:
                ordered.append(k)
    return ordered


def _write_rows(path: Path, rows: list, ordered: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ordered)
        w.writeheader()
        for r in rows:
            w.writerow({k: _norm(r.get(k)) for k in ordered})
    tmp.replace(path)  # atomic on the same filesystem


def write_master(master: Path, rows: dict) -> None:
    ordered = _columns_for(rows.values())
    # sort for readability: checkpoint, then sample count, then cfg
    ordered_rows = sorted(
        rows.values(),
        key=lambda r: (str(r.get("ckpt_short", "")), str(r.get("num_samples", "")), _cfg_of(r)),
    )
    _write_rows(master, ordered_rows, ordered)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="Sweep output dir (has results/*.json).")
    ap.add_argument("--master", required=True, help="Master CSV path (created if absent).")
    ap.add_argument("--gpu-type", default="", help="Recorded in the gpu_type column.")
    ap.add_argument("--stamp", default="", help="Recorded in the submit_stamp column.")
    ap.add_argument("--run-csv", default="",
                    help="Also write THIS run's rows (sorted by cfg) to a per-run report CSV.")
    args = ap.parse_args()

    root, master = Path(args.root), Path(args.master)
    rows = load_master(master)
    n_before = len(rows)
    new = collect_run(root, gpu_type=args.gpu_type, stamp=args.stamp)
    updated = 0
    for row in new:
        k = _key(row)
        if k in rows:
            updated += 1
        rows[k] = row
    write_master(master, rows)
    print(f"master {master}: {n_before} -> {len(rows)} rows "
          f"({len(new)} read, {updated} updated in place)")

    if args.run_csv and new:
        run_rows = sorted(new, key=_cfg_of)
        _write_rows(Path(args.run_csv), run_rows, _columns_for(new))
        print(f"per-run report {args.run_csv}: {len(new)} rows")


if __name__ == "__main__":
    main()
