#!/usr/bin/env python3
"""Shim to run ``tcpc_split.py`` even when VTK's OpenXR runtime is missing.

The ParaView builds on some clusters link the ``vtkRenderingOpenXR`` module
against ``libopenxr_loader.so.1`` which is not available everywhere. Importing
``vtk`` then fails before the actual TCPC logic executes, breaking the split
check in the optimisation pipelines. This wrapper detects that situation,
injects a lightweight stub module so ``vtk`` can import successfully, and then
delegates to the original ``tcpc_split.py`` entry point inside ``tnl-lbm``.
"""

from __future__ import annotations

import importlib
import runpy
import sys
from pathlib import Path
from types import ModuleType


def _ensure_vtk_openxr_stub() -> None:
    """Make ``vtkmodules.vtkRenderingOpenXR`` importable without libopenxr."""

    module_name = "vtkmodules.vtkRenderingOpenXR"
    if module_name in sys.modules:
        return

    try:
        importlib.import_module(module_name)
        return  # OpenXR module is available; nothing else to do.
    except ModuleNotFoundError:
        pass
    except ImportError as exc:
        # Only swallow the missing loader error; re-raise other issues.
        if "libopenxr_loader.so.1" not in str(exc):
            raise

    # Remove any partially imported module and install an empty stub to satisfy
    # ``import vtk``. The TCPC split script never touches the OpenXR symbols.
    sys.modules.pop(module_name, None)
    stub = ModuleType(module_name)
    stub.__all__ = []
    stub.__file__ = __file__
    sys.modules[module_name] = stub


def main() -> None:
    _ensure_vtk_openxr_stub()

    repo_root = Path(__file__).resolve().parents[1]
    split_script = repo_root / "submodules" / "tnl-lbm" / "tcpc_split.py"
    if not split_script.is_file():
        raise FileNotFoundError(f"Expected tcpc_split.py at '{split_script}'")

    # Re-run the original entry point with the current CLI arguments so behaviour
    # matches the upstream script exactly.
    sys.argv[0] = str(split_script)
    runpy.run_path(str(split_script), run_name="__main__")


if __name__ == "__main__":
    main()
