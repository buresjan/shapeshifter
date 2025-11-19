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


def _ensure_optional_vtk_module(module_name: str, missing_tokens: tuple[str, ...]) -> None:
    """Install an empty stub for ``module_name`` if its shared libs are absent."""

    if module_name in sys.modules:
        return

    try:
        importlib.import_module(module_name)
        return
    except ModuleNotFoundError:
        pass
    except ImportError as exc:
        text = str(exc)
        if not any(token in text for token in missing_tokens):
            raise

    sys.modules.pop(module_name, None)
    stub = ModuleType(module_name)
    stub.__all__ = []
    stub.__file__ = __file__
    sys.modules[module_name] = stub


def _ensure_vtk_stubs() -> None:
    """Make optional VTK runtimes importable when their libs are missing."""

    _ensure_optional_vtk_module(
        "vtkmodules.vtkRenderingOpenXR", ("libopenxr_loader.so", "libopenxr_loader.so.1")
    )
    _ensure_optional_vtk_module(
        "vtkmodules.vtkRenderingOpenVR", ("libopenvr_api.so", "libopenvr_api.so.1")
    )
    _ensure_optional_vtk_module(
        "vtkmodules.vtkRenderingAnari", ("libanari.so", "libanari.so.0")
    )
    _ensure_optional_vtk_module(
        "vtkmodules.vtkIOMySQL", ("libmariadb.so", "libmariadb.so.3")
    )


def main() -> None:
    _ensure_vtk_stubs()

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
