"""The isolation installer must never overwrite or mutate its source tree."""
import json
from pathlib import Path

import pytest

from tests.isolation_kit_loader import load_installer


installer = load_installer()


def tree(root, values):
    for name, value in values.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)


def test_install_changes_only_new_destination(tmp_path):
    source, overlay, destination = tmp_path / "running", tmp_path / "kit/overlay", tmp_path / "new/PT-Flow"
    original = {"train.py": "running train", "ptflow/potential.py": "old potential",
                "utils/precision.py": "old precision", "unrelated.txt": "preserve"}
    tree(source, original)
    tree(overlay, {"ptflow/potential.py": "fixed potential", "utils/precision.py": "fixed precision",
                   "scripts/new.py": "new launcher"})
    before = installer.inventory(source)
    installer.install(source, destination, overlay)
    assert installer.inventory(source) == before
    assert {name: (source / name).read_text() for name in original} == original
    assert (destination / "ptflow/potential.py").read_text() == "fixed potential"
    assert (destination / "utils/precision.py").read_text() == "fixed precision"
    assert (destination / "unrelated.txt").read_text() == "preserve"
    marker = json.loads((destination / ".ptflow_isolated_ablation.json").read_text())
    assert Path(marker["source"]) == source.resolve()
    assert Path(marker["destination"]) == destination.resolve()
    assert marker["source_core_sha256"] == before


def test_existing_destination_is_never_overwritten(tmp_path):
    source, overlay, destination = tmp_path / "running", tmp_path / "overlay", tmp_path / "existing"
    tree(source, {"train.py": "a", "ptflow/potential.py": "b", "utils/precision.py": "c"})
    tree(overlay, {"ptflow/potential.py": "fixed", "utils/precision.py": "fixed"})
    destination.mkdir()
    (destination / "keep.txt").write_text("untouched")
    with pytest.raises(FileExistsError):
        installer.install(source, destination, overlay)
    assert (destination / "keep.txt").read_text() == "untouched"


@pytest.mark.parametrize("relative", ["inside", "inside/deeper"])
def test_nested_destination_rejected(tmp_path, relative):
    source, overlay = tmp_path / "running", tmp_path / "overlay"
    tree(source, {"train.py": "a", "ptflow/potential.py": "b", "utils/precision.py": "c"})
    tree(overlay, {"ptflow/potential.py": "fixed", "utils/precision.py": "fixed"})
    with pytest.raises(ValueError, match="non-nested"):
        installer.install(source, source / relative, overlay)
