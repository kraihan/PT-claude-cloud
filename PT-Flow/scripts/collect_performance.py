"""Collect measured throughput and FID, retaining failed/missing array rows."""
import argparse
import csv
import json
import math
from pathlib import Path
from collections import defaultdict
import statistics


def write_csv(path, rows):
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def collect(root):
    root = Path(root)
    report = root / "report"
    report.mkdir(parents=True, exist_ok=True)
    bench, fid, errors = [], [], []
    for path in sorted((root / "train").glob("*/benchmark.json")):
        try:
            bench.append(dict(recipe=path.parent.name, **json.loads(path.read_text())))
        except (ValueError, OSError) as e:
            errors.append(f"{path}: {e}")
    for path in sorted((root / "results").glob("*.json")):
        try:
            row = json.loads(path.read_text())
            if not math.isfinite(float(row["fid"])):
                raise ValueError("nonfinite FID")
            fid.append(dict(result_file=str(path), **row))
        except (KeyError, ValueError, TypeError, OSError) as e:
            errors.append(f"{path}: {e}")
    manifest = root / "manifest.tsv"
    missing = []
    if manifest.exists():
        missing = [f"row {i}: {line}" for i, line in enumerate(manifest.read_text().splitlines())
                   if not (root / "results" / f"done_{i}").exists()]
    write_csv(report / "benchmark.csv", bench)
    write_csv(report / "fid.csv", fid)
    groups = defaultdict(list)
    for row in fid:
        # Do not mix sample counts, reference protocols or checkpoints.
        key = tuple(row.get(k) for k in ("ckpt", "cfg_scale", "ema_decay", "mode", "num_samples", "fid_ref", "backend", "world_size", "gen_bsz"))
        groups[key].append(row)
    aggregate = []
    for key, rows in groups.items():
        seeds = [r["seed"] for r in rows]
        if len(set(seeds)) != len(seeds):
            errors.append(f"Duplicate evaluation seeds for {key}; aggregate omitted")
            continue
        values = [float(r["fid"]) for r in rows]
        aggregate.append(dict(ckpt=key[0], cfg=key[1], ema=key[2], mode=key[3], num_samples=key[4],
                              fid_ref=key[5], backend=key[6], world_size=key[7], gen_bsz=key[8],
                              seeds=",".join(map(str, sorted(seeds))), evaluations=len(values),
                              fid_mean=statistics.mean(values), fid_std=statistics.stdev(values) if len(values) > 1 else None))
    aggregate.sort(key=lambda r: r["fid_mean"])
    write_csv(report / "fid_aggregate.csv", aggregate)
    lines = ["# PT-Flow speed/quality measurements", "",
             "These are measured results from this directory. Recipe names do not imply better FID.", "",
             "## Throughput", "",
             "| Recipe | Step seconds (median) | Generated/update | Generated/second | Peak GPU GiB |",
             "|---|---:|---:|---:|---:|"]
    for r in bench:
        lines.append(f"| {r['recipe']} | {r['benchmark/median_step_s']:.4f} | {r.get('global_generated_per_step', '')} | {r['benchmark/generated_per_second']:.1f} | {r.get('benchmark/peak_allocated_gib', '')} |")
    lines += ["", "Different particle budgets are different training workloads. Compare both FID at equal GPU-hours and at equal generated-particle counts.",
              "", "## FID", "", "| Checkpoint | CFG | EMA | Samples | Seeds | FID mean | FID std |",
              "|---|---:|---|---:|---|---:|---:|"]
    for r in aggregate:
        checkpoint = str(Path(r['ckpt']).parent.parent.name) + "/" + Path(r['ckpt']).name
        std = f"{r['fid_std']:.4f}" if r['fid_std'] is not None else "one evaluation"
        lines.append(f"| {checkpoint} | {r['cfg']} | {r['ema'] or 'primary'} | {r['num_samples']} | {r['seeds']} | {r['fid_mean']:.4f} | {std} |")
    lines += ["", "10k-sample FID is a screening measurement. Re-evaluate selected settings with 50k samples and separate seeds; report all selection settings. Evaluation seeds are not independent training seeds.",
              "", "## Missing or failed work", ""]
    lines.extend([f"- {v}" for v in missing + errors] or ["No missing completion markers or malformed results detected."])
    (report / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report / "REPORT.md")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    collect(p.parse_args().root)


if __name__ == "__main__":
    main()
