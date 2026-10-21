"""Make SONIC's `gear_sonic` package importable from a GR00T-WholeBodyControl checkout (used by export_motions.py)."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path


def enter_sonic_tree(sonic_root: Path) -> None:
    """Make ``gear_sonic`` importable and stub the packages its modules import but export never calls."""
    os.chdir(sonic_root)
    sys.path.insert(0, str(sonic_root))

    class _TensorDict(dict):
        pass

    def stub(name: str, **attrs) -> None:
        if importlib.util.find_spec(name) is None:
            module = types.ModuleType(name)
            module.__dict__.update(attrs)
            sys.modules[name] = module

    stub("wandb")
    stub("open3d", io=types.SimpleNamespace(read_triangle_mesh=lambda path: None))
    stub("tensordict", TensorDict=_TensorDict)
    stub("transformers", TrainerCallback=object)
    if importlib.util.find_spec("torchvision") is None:
        sys.modules["torchvision.models"] = types.ModuleType("torchvision.models")
        stub("torchvision", models=sys.modules["torchvision.models"])
