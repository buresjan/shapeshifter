# Shapeshifter — Agent Guide

## Purpose
Script-only playground for **optimization → geometry generation → LBM simulation**.  
Use local submodules as the single source of truth.

## Submodules (use these)
- `submodules/optilb/` — optimization routines and problem wrapper.
- `submodules/lb2dgeom/` — parametric 2D geometry for the LBM domain.
- `submodules/meshgen/` — voxel-based 3D geometry builder for lattice-ready domains.
- `submodules/tnl-lbm/` — LBM solver (C++/CUDA), run as an external binary.

## Agent rules
1. Prefer **only** these libs: `optilb`, `lb2dgeom`, `meshgen`, `numpy`, `scipy`, `matplotlib`. Reach for `meshgen` only when a 3D lattice is required.
2. **Read** submodule code and APIs before proposing changes. Reuse existing functions.
3. Do **not** add new Python dependencies. Do **not** modify submodules; write helper scripts here.
4. Assume the flow:
   - Build geometry with `lb2dgeom` from a parameter vector (or `meshgen` for voxelized 3D studies).
   - Invoke `tnl-lbm` via `subprocess.run(...)`, capture a scalar objective.
   - Wrap the loop with `optilb.OptimizationProblem` (or equivalent).
5. Keep paths simple: write geometry to a local working folder; symlink/copy into `tnl-lbm` case if required.

## Quality bar
- Provide **correct, runnable** scripts.
- Explain assumptions briefly in comments.
- Validate I/O contracts: file paths exist, binary exits `0`, objective is parsed to `float`.
- Fail loud on errors; no silent fallbacks.

## Minimal environment
- Python venv with `pip install -r requirements.txt` (installs the editable submodules).
- `tnl-lbm` built separately (CMake); treat as a CLI tool.
