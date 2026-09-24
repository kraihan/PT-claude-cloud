"""Create an isolated PT-Flow ablation checkout without editing the source.

The package's overlay is applied only after the source tree has been copied to
a new destination. Existing destinations are never overwritten or removed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid


PROTECTED = ("train.py", "ptflow/potential.py", "utils/precision.py")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def inventory(root: Path, relative_paths=PROTECTED):
    result = {}
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required source file missing: {path}")
        result[relative] = digest(path)
    return result


def copy_overlay(overlay: Path, destination: Path):
    if not overlay.is_dir():
        raise FileNotFoundError(f"Package overlay missing: {overlay}")
    for source in sorted(overlay.rglob("*")):
        relative = source.relative_to(overlay)
        target = destination / relative
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif source.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(os.readlink(source))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def install(source, destination, overlay):
    source = Path(source).expanduser().resolve(strict=True)
    destination = Path(destination).expanduser().resolve(strict=False)
    overlay = Path(overlay).expanduser().resolve(strict=True)
    if not source.is_dir():
        raise NotADirectoryError(source)
    if destination.exists():
        raise FileExistsError(f"Destination already exists; choose a new path: {destination}")
    if destination == source or source in destination.parents or destination in source.parents:
        raise ValueError("Source and destination must be separate, non-nested trees")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_before = inventory(source)
    staging = destination.with_name(destination.name + ".building-" + uuid.uuid4().hex)
    print(f"Copy source read-only: {source}", flush=True)
    print(f"Build isolated tree:  {staging}", flush=True)
    try:
        shutil.copytree(source, staging, symlinks=True,
                        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"))
        copy_overlay(overlay, staging)
        source_after = inventory(source)
        if source_after != source_before:
            raise RuntimeError("Source core files changed while copying; isolated tree was not activated")
        patched = inventory(staging)
        overlay_hashes = {str(path.relative_to(overlay)): digest(path)
                          for path in overlay.rglob("*") if path.is_file()}
        marker = dict(version=1, installed_at=datetime.now(timezone.utc).isoformat(),
                      source=str(source), destination=str(destination),
                      source_core_sha256=source_before, isolated_core_sha256=patched,
                      overlay_sha256=overlay_hashes,
                      guarantee="The installer did not write to the source tree")
        (staging / ".ptflow_isolated_ablation.json").write_text(
            json.dumps(marker, indent=2) + "\n", encoding="utf-8")
        os.replace(staging, destination)
    except Exception:
        print(f"Installation stopped. Any incomplete copy is preserved for inspection: {staging}")
        raise
    if inventory(source) != source_before:
        raise RuntimeError("Post-install source integrity check failed")
    print(f"Isolated checkout ready: {destination}")
    print("Original source hashes unchanged:")
    for name, value in source_before.items():
        print(f"  {value}  {name}")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Existing running-code tree; read only")
    parser.add_argument("--destination", required=True, help="New path; must not exist")
    parser.add_argument("--overlay", default=str(Path(__file__).resolve().parent / "overlay"))
    args = parser.parse_args()
    install(args.source, args.destination, args.overlay)


if __name__ == "__main__":
    main()
