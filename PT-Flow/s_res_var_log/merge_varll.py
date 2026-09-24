"""Merge per-cfg varll results into one CSV (16 rows) + a cumulative master.

Reads every ``$ROOT/results/varll_*.json`` (one per cfg, each holding the guided
normalized per-sample NLL and the guided-variance diagnostics), writes a per-run
CSV sorted by cfg, appends into a de-duplicated master, and concatenates any
per-sample CSVs into one file.  Output is CSV only; the model is untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

KEY = ("ckpt", "step", "cfg_scale")


def _norm(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _key(r):
    return "|".join(_norm(r.get(k)) for k in KEY)


def _cfg(r):
    try:
        return float(r.get("cfg_scale", "nan"))
    except (ValueError, TypeError):
        return float("nan")


def _write(path: Path, rows: list, cols: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: _norm(r.get(k)) for k in cols})
    tmp.replace(path)


def _cols(rows):
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    return cols


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="Sweep dir holding results/varll_*.json")
    ap.add_argument("--master", required=True, help="Cumulative master CSV")
    ap.add_argument("--run-csv", default="", help="Per-run CSV (this sweep's 16 rows)")
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    for p in sorted((root / "results").glob("varll_*.json")):
        try:
            rows.append(json.loads(p.read_text()))
        except (ValueError, OSError) as e:
            print(f"  skip {p.name}: {e}")
    rows.sort(key=_cfg)
    if not rows:
        print("no varll_*.json found; nothing to merge")
        return
    cols = _cols(rows)

    if args.run_csv:
        _write(Path(args.run_csv), rows, cols)
        print(f"per-run CSV {args.run_csv}: {len(rows)} rows")

    master = Path(args.master)
    existing, mcols = {}, list(cols)
    if master.exists():
        with master.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                existing[_key(r)] = r
                for k in r:
                    if k not in mcols:
                        mcols.append(k)
    updated = sum(1 for r in rows if _key(r) in existing)
    for r in rows:
        existing[_key(r)] = r
    _write(master, sorted(existing.values(), key=_cfg), mcols)
    print(f"master {master}: {len(existing)} rows ({len(rows)} read, {updated} updated)")

    ps = sorted((root / "results").glob("persample_*.csv"))
    if ps:
        out = root / "persample_all.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            for i, p in enumerate(ps):
                lines = p.read_text().splitlines()
                if not lines:
                    continue
                f.write("\n".join(lines if i == 0 else lines[1:]) + "\n")
        print(f"per-sample concat: {out}")


if __name__ == "__main__":
    main()
