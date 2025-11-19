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
import os
import posixpath
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


_VTK_OPTIONAL_MODULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("vtkmodules.vtkRenderingOpenXR", ("libopenxr_loader.so", "libopenxr_loader.so.1")),
    ("vtkmodules.vtkRenderingOpenVR", ("libopenvr_api.so", "libopenvr_api.so.1")),
    ("vtkmodules.vtkRenderingAnari", ("libanari.so", "libanari.so.0")),
    ("vtkmodules.vtkIOMySQL", ("libmariadb.so", "libmariadb.so.3")),
    ("vtkmodules.vtkIOExportPDF", ("libhpdf.so", "libhpdf.so.2.4")),
    ("vtkmodules.vtkIOAlembic", ("libAlembic.so", "libAlembic.so.1.8")),
    ("vtkmodules.vtkIOODBC", ("libodbc.so", "libodbc.so.2")),
    ("vtkmodules.vtkIOExportGL2PS", ("libgl2ps.so", "libgl2ps.so.1")),
    ("vtkmodules.vtkIOExodus", ("libexodus.so", "libexodus.so.0")),
)


def _ensure_vtk_stubs() -> None:
    """Make optional VTK runtimes importable when their libs are missing."""

    for module_name, tokens in _VTK_OPTIONAL_MODULES:
        _ensure_optional_vtk_module(module_name, tokens)


def _patch_adios_bp_detection(argv: list[str]) -> None:
    """Allow tcpc_split's '--bp-file' check to accept ADIOS directory datasets."""

    bp_dirs: set[Path] = set()
    it = iter(range(len(argv)))
    for idx in it:
        token = argv[idx]
        if token == "--bp-file" and idx + 1 < len(argv):
            raw = Path(argv[idx + 1])
            try:
                resolved = raw.resolve()
            except OSError:
                resolved = raw
            if resolved.is_dir():
                bp_dirs.add(resolved)
    if not bp_dirs:
        return

    original_isfile = posixpath.isfile

    def _patched_isfile(path: str | os.PathLike[str]) -> bool:
        try:
            resolved = Path(path).resolve()
        except OSError:
            resolved = Path(path)
        if resolved in bp_dirs:
            return True
        return original_isfile(path)

    posixpath.isfile = _patched_isfile  # type: ignore[assignment]
    os.path.isfile = _patched_isfile


def main() -> None:
    _ensure_vtk_stubs()
    _patch_adios_bp_detection(sys.argv)

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
