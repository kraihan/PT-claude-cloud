"""Import isolation-kit/install_isolated_ablation.py despite the hyphen."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def load_installer():
    path = Path(__file__).resolve().parents[1] / "isolation-kit/install_isolated_ablation.py"
    spec = spec_from_file_location("install_isolated_ablation", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
