# Shapeshifter — Agent Guide

## Intent
Script-only playground for **optimization → geometry generation → LBM simulation**.  
Treat local submodules as the single source of truth.

## Repository map (top-level)
- `scripts/` — entry-point Python/shell scripts (TCPC pipelines, demos, helpers like `tcpc_common.py`).
- `configs/` — run configs; primary set is `configs/tcpc/` with points CSVs and resumes.
- `data/` — run outputs and logs (e.g., junction TCPC logs, merged CSVs).
- `playground/` — exploratory experiments and archives (not production pipelines).
- `commands` — saved CLI invocations for long-running jobs.
- `submodules/` — core engines:
  - `optilb/` optimization routines and problem wrappers.
  - `lb2dgeom/` parametric 2D geometry for LBM domains.
  - `meshgen/` voxel-based 3D geometry builder (only when 3D is required); includes nested `meshgen/vascular_encoding_framework/` helpers.
  - `tnl-lbm/` LBM solver (C++/CUDA), run as an external binary.
- `requirements.txt` — editable submodules + shared scientific stack.

## Default workflow (2D)
1. Choose or create a config in `configs/tcpc/`.
2. Generate geometry with `lb2dgeom` inside a `scripts/` runner.
3. Invoke `tnl-lbm` via `subprocess.run(...)`, parse a scalar objective.
4. Wrap the loop with `optilb.OptimizationProblem` (or equivalent) and persist outputs in `data/`.

## 3D workflow (only when needed)
- Use `meshgen` to build a voxelized lattice, then reuse the same `tnl-lbm` + `optilb` loop.

## Agent rules (non-negotiable)
1. Prefer **only** these libs: `optilb`, `lb2dgeom`, `meshgen`, `vascular_encoding_framework`, `numpy`, `scipy`, `matplotlib`.
2. **Read** submodule APIs before changing scripts; reuse existing helpers.
3. Do **not** add new Python dependencies without asking. Do **not** modify submodules; add helpers in this repo.
4. Keep paths simple: write geometry to a local working folder; symlink/copy into a `tnl-lbm` case if required.
5. Fail loud on errors; no silent fallbacks.

## Quality bar
- Provide **correct, runnable** scripts.
- Explain assumptions briefly in comments.
- Validate I/O contracts: file paths exist, binary exits `0`, objective is parsed to `float`.

## Minimal environment
- Python venv with `pip install -r requirements.txt` (installs editable submodules including `meshgen/vascular_encoding_framework`; keep `git submodule update --init --recursive` to pull nested submodules).
- `tnl-lbm` built separately (CMake); treat as a CLI tool.
