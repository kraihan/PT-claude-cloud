"""Package source and run instructions, excluding weights, data and caches."""
import argparse
import hashlib
from pathlib import Path
import zipfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="../PT-Flow-fixed.zip")
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(a.output).resolve()
    excluded = {".git", "__pycache__", ".pytest_cache", "assets", "data", "runs", "log", "tmp"}
    paths = sorted(f for f in root.rglob("*") if f.is_file() and not excluded.intersection(f.relative_to(root).parts)
                   and f.suffix in {".py", ".yaml", ".yml", ".md", ".txt", ".sh", ".ini"})
    paths += [root / "LICENSE"]
    validation = root / "SPEED_QUALITY_VALIDATION.json"
    if validation.exists():
        paths.append(validation)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for f in paths:
            archive.write(f, Path("PT-Flow") / f.relative_to(root))
    with zipfile.ZipFile(out) as archive:
        assert archive.testzip() is None
    print(f"{out}: {len(paths)} source files, {out.stat().st_size} bytes")
    print("sha256:", hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
