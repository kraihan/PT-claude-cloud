#!/usr/bin/env bash
# Refuse to launch unless this is the separate tree created by the installer.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
marker="$repo/.ptflow_isolated_ablation.json"
if [[ ! -f "$marker" ]]; then
  printf 'Refusing: isolated-install marker missing: %s\n' "$marker" >&2
  exit 2
fi
cd "$repo"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
python - "$marker" "$repo" <<'PY'
import json, pathlib, sys
marker = json.loads(pathlib.Path(sys.argv[1]).read_text())
repo = pathlib.Path(sys.argv[2]).resolve()
if pathlib.Path(marker["destination"]).resolve() != repo:
    raise SystemExit("Refusing: marker destination does not match this checkout")
if pathlib.Path(marker["source"]).resolve() == repo:
    raise SystemExit("Refusing: source and isolated checkout are identical")
print("Protected running source:", marker["source"])
print("Launching isolated checkout:", repo)
PY
python -m scripts.check_pt_precision
exec python scripts/submit_imagenet_slurm.py
