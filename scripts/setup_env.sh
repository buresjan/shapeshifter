#!/usr/bin/env bash

# Bootstrap the Shapeshifter Python environment without pulling optional GUI tooling.
# This avoids installing Mayavi (and its VTK build step), which tends to segfault on
# headless machines, while still installing meshgen in editable mode.

set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "error: please activate your virtual environment before running this script." >&2
    exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo ">> Upgrading core packaging tooling"
python -m pip install --upgrade pip setuptools wheel

echo ">> Installing shared scientific stack"
python -m pip install "numpy>=1.24" "scipy>=1.10" "matplotlib>=3.7" gmsh trimesh tqdm

echo ">> Installing local packages without optional GUI extras"
python -m pip install --no-deps -e "${ROOT_DIR}/submodules/optilb"
python -m pip install --no-deps -e "${ROOT_DIR}/submodules/lb2dgeom"
python -m pip install --no-deps -e "${ROOT_DIR}/submodules/meshgen"

if [[ "${INSTALL_MAYAVI:-0}" == "1" ]]; then
    echo ">> INSTALL_MAYAVI=1 set – attempting to install Mayavi (may require VTK GUI stack)"
    python -m pip install mayavi
else
    echo ">> Skipping Mayavi (set INSTALL_MAYAVI=1 to attempt installing it later)."
fi

echo "Environment ready. Editable packages are linked without the optional Mayavi visualizer."
