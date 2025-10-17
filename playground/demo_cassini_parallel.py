#!/usr/bin/env python3
"""Parallel Nelder–Mead optimisation for a Cassini oval (robust pickling).

This variant is structured to work with optilb's process-based parallelism:
- All callables used by the executor are top-level and picklable.
- State is passed via simple, picklable arguments (Paths, numbers), not closures.
- Geometry filenames are unique across processes to avoid collisions.

Flow: build geometry with lb2dgeom -> submit tnl-lbm via submodules/tnl-lbm
runner -> parse scalar objective -> wrap with optilb.NelderMead (parallel=True).

Assumptions:
- submodules installed in the venv: pip install -e submodules/optilb -e submodules/lb2dgeom
- submodules/tnl-lbm built separately; treated as a CLI tool via its Python helper.
- sim_2D/sim2d_3 is the active solver binary (configured below).
"""

from __future__ import annotations

import math
import os
import time
import shutil
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

from optilb import DesignSpace, OptimizationProblem
from lb2dgeom import Grid, classify_cells, rasterize
from lb2dgeom.bouzidi import compute_bouzidi
from lb2dgeom.io import save_txt
from lb2dgeom.shapes.cassini_oval import CassiniOval


# --------------------------------------------------------------------------- #
# User-facing configuration knobs (edit these as needed).
# --------------------------------------------------------------------------- #
TNL_LBM_ROOT = Path(__file__).resolve().parents[1] / "submodules" / "tnl-lbm"
GEOMETRY_WORKDIR = Path(__file__).resolve().parent / "lbm_geometry_work"
LBM_EXECUTOR = os.environ.get("LBM_EXECUTOR", "slurm").lower()
if LBM_EXECUTOR not in {"slurm", "local"}:
    raise ValueError("LBM_EXECUTOR must be 'slurm' or 'local'.")
LBM_SIMULATION_TARGET = "sim2d_3"

INITIAL_ANGLE_DEG = 0.0  # Start with no rotation; the optimiser explores 0–180°.
INITIAL_A = 66.0         # Default semi-axis parameter a for the Cassini oval.
INITIAL_C = 62.0         # Only used to measure the target area once.

LBM_RESOLUTION = 8       # Grid resolution used by both geometry and solver.
LBM_TYPE1_BOUZIDI = "off"
LBM_RUNS_ROOT = "cassini_runs"
LBM_PARTITION = "gp"
LBM_WALLTIME = "10:00:00"
LBM_GPUS = 1
LBM_CPUS = 4
LBM_MEM = "16G"
LBM_POLL_INTERVAL = 30.0
LBM_JOB_TIMEOUT: float | None = None
LBM_RESULT_TIMEOUT: float | None = None

MAX_ITER = 25
MAX_EVALS = 60
TOL = 1e-3
SEED: int | None = None

# Set to None to let ProcessPool decide, or a small number to limit concurrency
N_WORKERS: int | None = 4

AREA_TOLERANCE = 5.0  # Acceptable difference (in grid units) when enforcing constant area.


# --------------------------------------------------------------------------- #
# Derived geometry constants.
# --------------------------------------------------------------------------- #
GRID_NX = LBM_RESOLUTION * 128  # Requirement: resolution * 128 -> 1024 for resolution 8.
GRID_NY = LBM_RESOLUTION * 32   # Requirement: resolution * 32  -> 256 for resolution 8.
GRID = Grid(nx=GRID_NX, ny=GRID_NY, dx=1.0, origin=(0.0, 0.0))


def second_quarter_center(grid: Grid) -> Tuple[float, float]:
    """Return the (x, y) location at the centre of the second quarter of the grid."""
    quarter_width = grid.nx // 4
    x_index = quarter_width + quarter_width // 2  # Middle of the second quarter.
    y_index = grid.ny // 2                        # Halfway up the channel.
    x_coord = grid.origin[0] + x_index * grid.dx
    y_coord = grid.origin[1] + y_index * grid.dx
    return float(x_coord), float(y_coord)


CASSINI_CENTER = second_quarter_center(GRID)


def rasterised_area(a_value: float, c_value: float, *, angle_rad: float = 0.0) -> float:
    """Numerically estimate the area (in grid units) of the Cassini oval."""
    shape = CassiniOval(
        x0=CASSINI_CENTER[0],
        y0=CASSINI_CENTER[1],
        a=float(a_value),
        c=float(c_value),
        theta=float(angle_rad),
    )
    _, solid = rasterize(GRID, shape)
    return float(np.count_nonzero(solid) * GRID.dx * GRID.dx)


TARGET_AREA = rasterised_area(INITIAL_A, INITIAL_C)


def solve_for_c(a_value: float, target_area: float, *, angle_rad: float = 0.0) -> float:
    """Return c that keeps the Cassini area constant for a given a and rotation.

    Note: The rasterised area depends on the orientation due to grid discretisation.
    We therefore solve for c at the actual rotation angle used for the geometry
    to minimise area drift at build time.
    """
    c_low = 1.0
    c_high = max(INITIAL_C + 20.0, a_value + 40.0)

    def area_minus_target(cand: float) -> float:
        return rasterised_area(a_value, cand, angle_rad=angle_rad) - target_area

    low_value = area_minus_target(c_low)
    while low_value < 0.0:
        c_low *= 0.5
        if c_low < 1e-3:
            raise RuntimeError("Failed to bracket Cassini area constraint on the lower side.")
        low_value = area_minus_target(c_low)

    high_value = area_minus_target(c_high)
    while high_value > 0.0:
        c_high *= 1.5
        if c_high > 10_000.0:
            raise RuntimeError("Failed to bracket Cassini area constraint on the upper side.")
        high_value = area_minus_target(c_high)

    for _ in range(60):
        c_mid = 0.5 * (c_low + c_high)
        mid_value = area_minus_target(c_mid)
        if abs(mid_value) <= AREA_TOLERANCE:
            return c_mid
        if mid_value > 0.0:
            c_low = c_mid
        else:
            c_high = c_mid

    return 0.5 * (c_low + c_high)


def _unique_geometry_name(angle_deg: float, a_value: float) -> str:
    """Return a per-evaluation unique base filename to avoid parallel clashes."""
    pid = os.getpid()
    ns = time.time_ns()
    return f"cassini_{angle_deg:07.3f}_{a_value:07.3f}_{pid}_{ns:x}.txt"


def build_geometry(angle_deg: float, a_value: float, *, tnllbm_root: Path, geometry_workdir: Path) -> tuple[str, float]:
    """Create and stage a geometry; return (basename, c_value).

    Writes to a local working folder and copies the file into the tnl-lbm root.
    """
    geometry_workdir.mkdir(parents=True, exist_ok=True)
    if not tnllbm_root.is_dir():
        raise FileNotFoundError(f"tnl-lbm root '{tnllbm_root}' is not a directory.")

    angle_rad = math.radians(angle_deg)
    c_value = solve_for_c(a_value, TARGET_AREA, angle_rad=angle_rad)
    cassini = CassiniOval(
        x0=CASSINI_CENTER[0],
        y0=CASSINI_CENTER[1],
        a=float(a_value),
        c=float(c_value),
        theta=float(angle_rad),
    )

    phi, solid = rasterize(GRID, cassini)
    bouzidi = compute_bouzidi(GRID, phi, solid)
    cell_types = classify_cells(solid)

    measured_area = float(np.count_nonzero(solid) * GRID.dx * GRID.dx)
    if abs(measured_area - TARGET_AREA) > AREA_TOLERANCE:
        raise RuntimeError(
            f"Area drift detected: target={TARGET_AREA:.3f}, measured={measured_area:.3f}",
        )

    basename = _unique_geometry_name(angle_deg, a_value)
    local_path = geometry_workdir / basename
    save_txt(local_path, GRID, cell_types, bouzidi, selection="all", include_header=False)

    staged_path = tnllbm_root / basename
    shutil.copy2(local_path, staged_path)
    return basename, c_value


def run_simulation(geometry_name: str, *, tnllbm_root: Path) -> float:
    """Submit and collect a single LBM run; return the numeric objective.

    Import happens here to ensure child processes can resolve the module after
    amending sys.path.
    """
    if str(tnllbm_root) not in sys.path:
        sys.path.insert(0, str(tnllbm_root))

    runner = tnllbm_root / "run_lbm_simulation.py"
    if not runner.is_file():
        raise FileNotFoundError(
            f"LBM runner script not found at '{runner}'. Did you initialise submodules?",
        )

    from run_lbm_simulation import submit_and_collect  # type: ignore

    result = submit_and_collect(
        geometry=geometry_name,
        resolution=int(LBM_RESOLUTION),
        partition=LBM_PARTITION,
        walltime=LBM_WALLTIME,
        gpus=LBM_GPUS,
        cpus=LBM_CPUS,
        mem=LBM_MEM,
        runs_root=LBM_RUNS_ROOT,
        type1_bouzidi=LBM_TYPE1_BOUZIDI,
        poll_interval=LBM_POLL_INTERVAL,
        timeout=LBM_JOB_TIMEOUT,
        result_timeout=LBM_RESULT_TIMEOUT,
        executor=LBM_EXECUTOR,
        simulation_target=LBM_SIMULATION_TARGET,
    )
    if result.numeric_value is None:
        raise RuntimeError("LBM run completed without a numeric objective value.")
    return float(result.numeric_value)


def lbm_objective(x: np.ndarray, *, tnllbm_root: Path, geometry_workdir: Path) -> float:
    """Picklable objective: build geometry, run solver, return negative value.

    Negative sign because Nelder–Mead minimises; we seek to maximise.
    """
    params = np.asarray(x, dtype=float)
    angle_deg = float(params[0])
    a_value = float(params[1])

    geometry_name, c_value = build_geometry(angle_deg, a_value, tnllbm_root=tnllbm_root, geometry_workdir=geometry_workdir)
    value = run_simulation(geometry_name, tnllbm_root=tnllbm_root)
    print(
        f"angle={angle_deg:7.3f} deg, a={a_value:7.3f}, c={c_value:7.3f} -> objective={value:.6f}",
        flush=True,
    )
    return -value


def build_design_space() -> DesignSpace:
    """Describe optimisation bounds: angle in [0, 180], a in [60, 72]."""
    lower = [0.0, 60.0]
    upper = [180.0, 72.0]
    names = ["angle_deg", "cassini_a"]
    return DesignSpace(lower=lower, upper=upper, names=names)


def main() -> int:
    geometry_root = GEOMETRY_WORKDIR.resolve()
    tnllbm_root = TNL_LBM_ROOT.resolve()

    # Bind context as picklable values (Paths) using a top-level function via functools.partial.
    # Note: optilb may wrap our objective with its own (picklable) partials for normalisation.
    import functools
    objective = functools.partial(lbm_objective, tnllbm_root=tnllbm_root, geometry_workdir=geometry_root)
    space = build_design_space()

    start_point = [INITIAL_ANGLE_DEG, INITIAL_A]
    optimizer_opts: dict[str, object] = {"parallel_poll_points": True, "n_workers": N_WORKERS}
    problem = OptimizationProblem(
        objective=objective,
        space=space,
        x0=start_point,
        optimizer="nelder-mead",
        normalize=True,
        max_iter=MAX_ITER,
        max_evals=MAX_EVALS,
        tol=TOL,
        seed=SEED,
        parallel=True,
        optimizer_options=optimizer_opts,
    )

    print(f"Target Cassini area A = {TARGET_AREA:.3f} grid units²")
    print("Starting PARALLEL optimisation (maximisation via negative objective)...")

    result = problem.run()

    best_angle = float(result.best_x[0])
    best_a = float(result.best_x[1])
    best_c = solve_for_c(best_a, TARGET_AREA, angle_rad=math.radians(best_angle))
    best_value = -float(result.best_f)

    print(f"Best angle (deg):  {best_angle:.6f}")
    print(f"Best Cassini a:    {best_a:.6f}")
    print(f"Derived Cassini c: {best_c:.6f}")
    print(f"Best objective:     {best_value:.6f}")
    print(f"Objective evals:    {result.nfev}")
    if problem.log:
        print(
            f"Optimizer={problem.log.optimizer} iterations={problem.log.nfev} runtime={problem.log.runtime:.2f}s",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
