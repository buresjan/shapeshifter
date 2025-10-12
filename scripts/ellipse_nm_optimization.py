#!/usr/bin/env python3
"""Straight-line driver for maximising flow around a rotated ellipse.

The script intentionally keeps every step explicit and chatty so that the
overall flow is easy to follow for a newcomer:

1. A single angle (in degrees) describes the orientation of an ellipse.
2. For every angle suggested by Nelder–Mead we build a lattice geometry file.
3. The freshly written geometry file is handed to the ``tnl-lbm`` runner.
4. The solver outputs a scalar objective, which we feed back to the optimiser.

All helper logic lives in this file so that the control flow can be read
top-to-bottom without jumping across modules or class hierarchies.
"""

from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Optional, cast

import numpy as np

from optilb import DesignSpace, OptimizationProblem
from lb2dgeom import Grid, classify_cells, rasterize
from lb2dgeom.bouzidi import compute_bouzidi
from lb2dgeom.io import save_txt
from lb2dgeom.shapes.ellipse import Ellipse


# --------------------------------------------------------------------------- #
# User-facing knobs
# --------------------------------------------------------------------------- #
# Everything that a user might want to tweak lives in the block below. Keeping
# them together makes it obvious which values shape the optimisation scenario.
# --------------------------------------------------------------------------- #
TNL_LBM_ROOT = Path(__file__).resolve().parents[1] / "submodules" / "tnl-lbm"
GEOMETRY_WORKDIR = Path(__file__).resolve().parent / "lbm_geometry_work"

ELLIPSE_SEMI_MAJOR = 80.0
ELLIPSE_SEMI_MINOR = 50.0

LBM_RESOLUTION = 8
LBM_TYPE1_BOUZIDI = "auto"
LBM_RUNS_ROOT = "optim_runs"
LBM_PARTITION = "gp"
LBM_WALLTIME = "10:00:00"
LBM_GPUS = 1
LBM_CPUS = 4
LBM_MEM = "16G"
LBM_POLL_INTERVAL = 30.0  # Polling interval (seconds) while waiting for Slurm jobs to finish.
LBM_JOB_TIMEOUT: Optional[float] = None
LBM_RESULT_TIMEOUT: Optional[float] = None

INITIAL_ANGLE = 70.0
MAX_ITER = 20
MAX_EVALS = 40
TOL = 1e-3
SEED: Optional[int] = None


def make_lbm_objective(
    *,
    geometry_config: dict[str, Any],
    geometry_workdir: Path,
    tnllbm_root: Path,
    resolution: int,
    type1_bouzidi: str,
    runs_root: str,
) -> Callable[[np.ndarray], float]:
    """Bundle the LBM evaluation logic into a simple function for Nelder–Mead.

    Returning a plain function keeps the code approachable: the returned
    callable is what goes straight into :class:`optilb.OptimizationProblem`.
    """

    # Sanity checks up front make it obvious when the solver tree is missing.
    geometry_workdir.mkdir(parents=True, exist_ok=True)
    if not tnllbm_root.is_dir():
        raise FileNotFoundError(f"tnl-lbm root '{tnllbm_root}' is not a directory")
    runner = tnllbm_root / "run_lbm_simulation.py"
    if not runner.is_file():
        raise FileNotFoundError(f"Expected runner script at '{runner}'. Did you clone submodules?")

    eval_index = 0  # Track how many geometries we wrote to create friendly filenames.

    def build_geometry(angle_deg: float) -> str:
        """Rasterise the ellipse and copy the file into ``tnl-lbm``."""
        nonlocal eval_index
        eval_index += 1
        # Extract the parameters from the plain dictionary produced by ``build_geometry_config``.
        grid = cast(Grid, geometry_config["grid"])
        center_x = float(geometry_config["center_x"])
        center_y = float(geometry_config["center_y"])
        semi_major = float(geometry_config["semi_major"])
        semi_minor = float(geometry_config["semi_minor"])

        angle_rad = math.radians(angle_deg)
        ellipse = Ellipse(
            x0=center_x,
            y0=center_y,
            a=semi_major,
            b=semi_minor,
            theta=angle_rad,
        )
        phi, solid = rasterize(grid, ellipse)
        bouzidi = compute_bouzidi(grid, phi, solid)
        cell_types = classify_cells(solid)

        basename = f"nm_geometry_{eval_index:04d}_{angle_deg:06.2f}.txt"
        local_path = geometry_workdir / basename
        save_txt(local_path, grid, cell_types, bouzidi, selection="all", include_header=False)

        staged_path = tnllbm_root / basename
        shutil.copy2(local_path, staged_path)
        return basename

    def recover_result_from_failure(submission, final_state: str) -> tuple[str, Optional[float]]:
        """Pull a value from the fallback path when Slurm marks the job as failed."""
        from run_lbm_simulation import read_result_file, update_manifest, wait_for_result_file

        recovery_path = (
            submission.run_dir / "sim_2D" / "values" / f"value_{submission.staged_geometry.name}"
        )
        try:
            wait_for_result_file(
                recovery_path,
                poll_interval=5.0,
                timeout=LBM_RESULT_TIMEOUT,
            )
        except TimeoutError as exc:  # pragma: no cover - escalate downstream
            raise RuntimeError(
                f"Job {submission.job_id} ended with state {final_state} and no recoverable value.",
            ) from exc

        raw, numeric_value = read_result_file(recovery_path)
        try:
            if not submission.result_path.exists():
                shutil.copy2(recovery_path, submission.result_path)
        except Exception as err:  # pragma: no cover - best effort copy
            print(
                f"Warning: could not stage recovered value for job {submission.job_id}: {err}",
                file=sys.stderr,
                flush=True,
            )

        print(f"Recovered value from failed job {submission.job_id} (state={final_state})", flush=True)
        update_manifest(submission.run_dir, {"recovered_from_state": final_state})
        return raw, numeric_value

    def run_simulation(geometry_name: str, angle_deg: float) -> float:
        """Fire off the LBM job and return the resulting scalar objective."""
        from datetime import UTC, datetime

        from run_lbm_simulation import (
            COMPLETED_STATES,
            SimulationResult,
            prepare_submission,
            submit_prepared,
            update_manifest,
            wait_for_job_completion,
            wait_for_result_file,
            read_result_file,
        )

        submission = prepare_submission(
            geometry=geometry_name,
            resolution=int(resolution),
            partition=LBM_PARTITION,
            walltime=LBM_WALLTIME,
            gpus=LBM_GPUS,
            cpus=LBM_CPUS,
            mem=LBM_MEM,
            runs_root=runs_root,
            job_name=None,
            type1_bouzidi=type1_bouzidi,
        )
        submission = submit_prepared(submission)

        try:  # Wait for the job to finish while honouring the optional timeout.
            final_state = wait_for_job_completion(
                submission.job_id,  # type: ignore[arg-type]
                poll_interval=LBM_POLL_INTERVAL,
                timeout=LBM_JOB_TIMEOUT,
            )
        except TimeoutError as exc:  # pragma: no cover - propagated to caller
            raise RuntimeError(
                f"Job {submission.job_id} timed out while waiting for completion.",
            ) from exc

        finished_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        update_manifest(
            submission.run_dir,
            {
                "final_state": final_state,
                "finished_at": finished_at,
            },
        )

        if final_state in COMPLETED_STATES:
            wait_for_result_file(
                submission.result_path,
                poll_interval=5.0,
                timeout=LBM_RESULT_TIMEOUT,
            )
            raw, numeric_value = read_result_file(submission.result_path)
        else:
            raw, numeric_value = recover_result_from_failure(submission, final_state)

        update_manifest(submission.run_dir, {"result_value": raw})

        result = SimulationResult(
            job_id=submission.job_id or "",
            run_id=submission.run_id,
            run_dir=submission.run_dir,
            result_path=submission.result_path,
            raw_value=raw,
            numeric_value=numeric_value,
            state=final_state,
            finished_at=finished_at,
        )
        if result.numeric_value is None:
            raise RuntimeError("LBM runner did not return a numeric objective value")
        value = float(result.numeric_value)
        print(f"angle={angle_deg:.4f} deg -> objective={value:.6f}", flush=True)
        return value

    def objective(x: np.ndarray) -> float:
        """Entry point handed to Nelder–Mead (minimisation of the negative objective)."""
        # ``optilb`` gives us the angle wrapped in an array; we unwrap and evaluate.
        angle_deg = float(np.asarray(x, dtype=float)[0])
        geometry_name = build_geometry(angle_deg)
        value = run_simulation(geometry_name, angle_deg)
        return -value

    return objective


def build_geometry_config(semi_major: float, semi_minor: float) -> dict[str, Any]:
    """Describe the fixed grid and ellipse placement used throughout this study."""
    # The grid is large enough to comfortably host the ellipse and its wake.
    grid = Grid(nx=1024, ny=256, dx=1.0, origin=(0.0, 0.0))
    # Position the ellipse approximately one quarter into the domain.
    quarter_w = grid.nx // 4
    x0_idx = quarter_w + quarter_w // 2
    y0_idx = grid.ny // 2
    x0 = grid.origin[0] + x0_idx * grid.dx
    y0 = grid.origin[1] + y0_idx * grid.dx
    return {
        "grid": grid,
        "center_x": x0,
        "center_y": y0,
        "semi_major": semi_major,
        "semi_minor": semi_minor,
    }


def main() -> int:
    """Set up the geometry, objective wrapper, and run the Nelder–Mead search."""
    # 1) Lock in the physical geometry (grid spacing, ellipse size, placement).
    geom_cfg = build_geometry_config(ELLIPSE_SEMI_MAJOR, ELLIPSE_SEMI_MINOR)
    geometry_root = GEOMETRY_WORKDIR.resolve()
    tnllbm_root = TNL_LBM_ROOT.resolve()
    # Ensure we can import ``run_lbm_simulation`` directly from the submodule.
    if str(tnllbm_root) not in sys.path:
        sys.path.insert(0, str(tnllbm_root))
    # 2) Wrap the LBM solver into a simple Python callable for the optimiser.
    objective = make_lbm_objective(
        geometry_config=geom_cfg,
        geometry_workdir=geometry_root,
        tnllbm_root=tnllbm_root,
        resolution=LBM_RESOLUTION,
        type1_bouzidi=LBM_TYPE1_BOUZIDI,
        runs_root=LBM_RUNS_ROOT,
    )

    # 3) Describe the optimisation search space: the angle ranges from 0 to 180 deg.
    space = DesignSpace(lower=[0.0], upper=[180.0], names=["angle_deg"])
    # 4) Ask ``optilb`` to run Nelder–Mead starting from ``INITIAL_ANGLE``.
    problem = OptimizationProblem(
        objective=objective,
        space=space,
        x0=[INITIAL_ANGLE],
        optimizer="nelder-mead",
        normalize=True,
        max_iter=MAX_ITER,
        max_evals=MAX_EVALS,
        tol=TOL,
        seed=SEED,
        parallel=False,
    )

    print("Starting Nelder-Mead optimisation (maximisation via negative objective)...")
    result = problem.run()

    # 5) Convert the minimisation output back into the original maximisation view.
    best_angle = float(result.best_x[0])
    best_value = -float(result.best_f)
    print(f"Best angle (deg): {best_angle:.6f}")
    print(f"Best objective value: {best_value:.6f}")
    print(f"Objective evaluations: {result.nfev}")
    if problem.log:
        print(
            f"Optimizer={problem.log.optimizer} iterations={problem.log.nfev} runtime={problem.log.runtime:.2f}s",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
