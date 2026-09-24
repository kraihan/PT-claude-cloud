from pathlib import Path
import hashlib
import json
import zipfile

root = Path(__file__).resolve().parent
kit = root / "PT-Flow-active-kit"
output = root / "PT-Flow-active-30k-kit.zip"
skip = {"__pycache__", ".pytest_cache", "assets", ".git"}
files = [p for p in sorted(kit.rglob("*")) if p.is_file()
         and not set(p.relative_to(kit).parts).intersection(skip)
         and p.suffix != ".pyc"]
for path in files:
    if path.suffix == ".py":
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
    for path in files:
        payload = path.read_bytes().replace(b"\r\n", b"\n")
        z.writestr(path.relative_to(kit).as_posix(), payload)
with zipfile.ZipFile(output) as z:
    assert z.testzip() is None
    assert {"submit.sh", "prepare.py", "worker.sbatch", "repo/train.py"} <= set(z.namelist())
    assert all(b"\r" not in z.read(n) for n in ("submit.sh", "worker.sbatch"))
print(json.dumps(dict(path=str(output), files=len(files), bytes=output.stat().st_size,
                      sha256=hashlib.sha256(output.read_bytes()).hexdigest()), indent=2))
