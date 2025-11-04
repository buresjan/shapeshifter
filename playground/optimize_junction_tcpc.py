#!/usr/bin/env python3
"""Parallel Nelder–Mead optimisation of the TCPC junction geometry.

This script minimises the scalar objective returned by the 3D TCPC solver
(`sim_tcpc_2`) while parametrising the mesh via `meshgen`'s `junction_2d`
template. The optimisation is intentionally simple and self-contained:

- Variables (in this order):
  - offset ∈ [-1.0, 1.0]
  - lower_angle ∈ [-20.0, 20.0]     [degrees]
  - upper_angle ∈ [-20.0, 20.0]     [degrees]
  - lower_flare ∈ [0.0, 0.25]       [meters]
  - upper_flare ∈ [0.0, 0.25]       [meters]

- Fixed settings:
  - Resolution: 5 (both for voxelisation and solver argument)
  - Initial point: [0.1, 4.0, -3.0, 0.1, 0.1]
  - Max evaluations: 80
  - Optimiser: Nelder–Mead (optilb), with memoisation and parallel evaluation
  - Normalisation: enabled (optimises in the unit hypercube)

How it works (per evaluation):
  1) Build a unique run directory and a per-run `sim_NSE` data root.
  2) Generate the mesh triplet (`geom_`, `dim_`, `angle_`) with meshgen.
  3) Stage the files under the run’s `sim_NSE` and launch `sim_tcpc_2` with
     `cwd=run_dir` so its relative paths resolve to the staged data.
  4) Parse the objective from `sim_NSE/tmp/val_<case>.txt`.

Run instructions (local or Slurm):
  - Local:  ensure `tnl-lbm` is built (sim_tcpc_2 exists), then:
      python playground/optimize_junction_tcpc.py
  - Slurm: submit the launcher:
      sbatch playground/optimize_junction_tcpc.sh

Notes:
- No CLI knobs — all parameters live in this script for clarity.
- The optimiser evaluates points in parallel using a thread pool to keep the
  implementation robust in diverse environments (no pickling issues). The
  heavy lifting happens in an external binary, so threads are sufficient.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from optilb import DesignSpace, OptimizationProblem

# Reuse the proven geometry+solver helpers from the runnable TCPC script.
# This file lives in the same `playground/` directory, so we can import it
# directly without package boilerplate.
from run_junction_tcpc import (
    default_paths,
    ensure_txt_suffix,
    generate_geometry,
    locate_solver_root,
    prepare_run_directory,
    stage_geometry,
    run_simulation,
    collect_scalar,
)


# ---------------------------------------------------------------------------
# Fixed configuration (edit here, no CLI)
# ---------------------------------------------------------------------------
RESOLUTION = 5
MAX_EVALS = 80

# Initial point (offset, lower_angle, upper_angle, lower_flare, upper_flare)
X0 = np.array([0.1, 4.0, -3.0, 0.1, 0.1], dtype=float)

# Bounds
LOWER = np.array([-1.0, -20.0, -20.0, 0.0, 0.0], dtype=float)
UPPER = np.array([+1.0, +20.0, +20.0, 0.25, 0.25], dtype=float)

# Parallel evaluation workers (threads)
N_WORKERS = int(os.environ.get("OPT_NM_WORKERS", "8"))

# Enforce threaded parallelism for robustness (no pickling constraints).
os.environ.setdefault("OPTILB_FORCE_THREAD_POOL", "1")


@dataclass(frozen=True)
class Paths:
    project_root: Path
    solver_binary: Path
    solver_root: Path


def _project_paths() -> Paths:
    """Resolve repository paths and the solver binary/root."""
    project_root = Path(__file__).resolve().parents[1]
    default_bin, _ = default_paths(project_root)
    solver_binary = default_bin.resolve()
    if not solver_binary.is_file():
        raise FileNotFoundError(
            f"sim_tcpc_2 binary not found at '{solver_binary}'. Build tnl-lbm first."
        )
    solver_root = locate_solver_root(solver_binary)
    return Paths(project_root=project_root, solver_binary=solver_binary, solver_root=solver_root)


def _hash_params(x: np.ndarray) -> str:
    """Stable short hash for a parameter vector (used for file/run naming)."""
    # Compact canonical string avoids float repr ambiguity; 8-char prefix is fine.
    txt = ",".join(f"{v:.8g}" for v in np.asarray(x, dtype=float))
    return hashlib.sha1(txt.encode("ascii")).hexdigest()[:12]


def _objective(x: np.ndarray) -> float:
    """Objective wrapper: generate geometry, run solver, return scalar.

    This function is intentionally stateless and creates unique per‑run
    directories to be safe under parallel evaluation.
    """
    p = _project_paths()

    # Map design vector to geometry keywords
    offset, lower_angle, upper_angle, lower_flare, upper_flare = map(float, x)

    # Unique case/run identifiers
    case_tag = _hash_params(x)
    case_name = ensure_txt_suffix(f"junction_{case_tag}.txt")

    # Run directory and per-run data root
    run_dir = prepare_run_directory(p.project_root, Path(case_name).name)
    data_root = (run_dir / "sim_NSE")
    for sub in ("geometry", "dimensions", "angle", "tmp"):
        (data_root / sub).mkdir(parents=True, exist_ok=True)

    # Meshgen workspace lives under the run directory for isolation
    workspace = run_dir / "meshgen_output"

    # 1) Generate triplet via meshgen
    files, generated_case_name = generate_geometry(
        workspace,
        case_name,
        resolution=RESOLUTION,
        lower_angle=lower_angle,
        upper_angle=upper_angle,
        upper_flare=upper_flare,
        lower_flare=lower_flare,
        offset=offset,
        num_processes=1,
    )

    # 2) Stage into per-run sim_NSE tree
    stage_geometry(files, data_root)

    # 3) Run solver and collect the latest sample
    try:
        _stdout, _stderr = run_simulation(
            p.solver_binary,
            RESOLUTION,
            Path(generated_case_name).name,
            data_root,
            run_dir,
            p.solver_root,
        )
    except Exception:
        # Preserve run_dir for debugging on failures
        raise
    finally:
        # Remove heavy meshgen workspace to conserve disk space
        try:
            shutil.rmtree(workspace, ignore_errors=True)
        except Exception:
            pass

    _t, value, _val_path = collect_scalar(data_root, Path(generated_case_name).name)
    return float(value)


def main() -> Tuple[np.ndarray, float]:
    # Design space and problem definition
    space = DesignSpace(lower=LOWER, upper=UPPER, names=(
        "offset", "lower_angle", "upper_angle", "lower_flare", "upper_flare"
    ))

    # Configure Nelder–Mead with memoisation and desired worker count
    from optilb.optimizers import NelderMeadOptimizer

    nm = NelderMeadOptimizer(
        n_workers=N_WORKERS,
        memoize=True,
        # Keep other coefficients at defaults; normalization will handle scales
    )

    problem = OptimizationProblem(
        objective=_objective,
        space=space,
        x0=X0,
        optimizer=nm,
        parallel=True,          # allow parallel evals
        normalize=True,         # operate in unit hypercube
        max_evals=MAX_EVALS,
        verbose=True,
    )

    print("[opt] Starting Nelder–Mead optimisation")
    print(f"[opt] max_evals={MAX_EVALS}, workers={N_WORKERS}, normalize=True, memoize=True")
    res = problem.run()

    best_x = res.best_x
    best_f = float(res.best_f)
    print("\n[opt] Finished.")
    print(f"[opt] Best value: {best_f}")
    print("[opt] Best parameters:")
    for name, val in zip(space.names or (), best_x):
        print(f"  - {name}: {val}")

    if problem.log is not None:
        print("[opt] Log:")
        print(f"  optimizer = {problem.log.optimizer}")
        print(f"  nfev      = {problem.log.nfev}")
        print(f"  runtime   = {problem.log.runtime:.2f} s")
        print(f"  early     = {problem.log.early_stopped}")

    return best_x, best_f


if __name__ == "__main__":
    main()

