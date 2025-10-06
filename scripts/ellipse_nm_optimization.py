#!/usr/bin/env python3
"""Rotated ellipse optimisation using Nelder–Mead and the LBM runner."""

from __future__ import annotations

import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from optilb import DesignSpace, OptimizationProblem
from lb2dgeom import Grid, classify_cells, rasterize
from lb2dgeom.bouzidi import compute_bouzidi
from lb2dgeom.io import save_txt
from lb2dgeom.shapes.ellipse import Ellipse


# ------------------------- configuration knobs -------------------------
# These constants define the optimisation scenario and Slurm submission
# settings. Adjust them here rather than through a CLI to keep the script
# fully deterministic when run in batch mode.
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
LBM_POLL_INTERVAL = 30.0
LBM_JOB_TIMEOUT: Optional[float] = None
LBM_RESULT_TIMEOUT: Optional[float] = None

INITIAL_ANGLE = 60.0
MAX_ITER = 20
MAX_EVALS: Optional[int] = None
TOL = 1e-4
SEED: Optional[int] = None


@dataclass
class GeometryConfig:
    """Reusable bundle describing the lattice grid and ellipse dimensions."""

    grid: Grid
    center_x: float
    center_y: float
    semi_major: float
    semi_minor: float


class LBMSimulationObjective:
    """Objective wrapper that builds geometry and launches the LBM runner."""

    def __init__(
        self,
        *,
        geometry_config: GeometryConfig,
        geometry_workdir: Path,
        tnllbm_root: Path,
        resolution: int,
        type1_bouzidi: str,
        runs_root: str,
    ) -> None:
        self.geometry_config = geometry_config
        self.geometry_workdir = geometry_workdir
        self.tnllbm_root = tnllbm_root
        self.resolution = int(resolution)
        self.type1_bouzidi = type1_bouzidi
        self.runs_root = runs_root
        self.geometry_workdir.mkdir(parents=True, exist_ok=True)
        if not self.tnllbm_root.is_dir():
            raise FileNotFoundError(f"tnl-lbm root '{self.tnllbm_root}' is not a directory")
        runner = self.tnllbm_root / "run_lbm_simulation.py"
        if not runner.is_file():
            raise FileNotFoundError(
                f"Expected runner script at '{runner}'. Did you clone submodules?"
            )
        self._eval_index = 0

    def __call__(self, x: np.ndarray) -> float:
        """Evaluate the wrapped objective for a single candidate angle."""
        angle_deg = float(np.asarray(x, dtype=float).flat[0])
        geometry_name = self._build_geometry(angle_deg)
        value = self._run_simulation(geometry_name, angle_deg)
        # Optimizer minimises, but we want to maximise the simulation output.
        return -value

    def _build_geometry(self, angle_deg: float) -> str:
        """Rasterise the rotated ellipse and stage it for the LBM solver."""
        self._eval_index += 1
        cfg = self.geometry_config
        angle_rad = math.radians(angle_deg)
        ellipse = Ellipse(
            x0=cfg.center_x,
            y0=cfg.center_y,
            a=cfg.semi_major,
            b=cfg.semi_minor,
            theta=angle_rad,
        )
        phi, solid = rasterize(cfg.grid, ellipse)
        bouzidi = compute_bouzidi(cfg.grid, phi, solid)
        cell_types = classify_cells(solid)

        basename = f"nm_geometry_{self._eval_index:04d}_{angle_deg:06.2f}.txt"
        local_path = self.geometry_workdir / basename
        save_txt(local_path, cfg.grid, cell_types, bouzidi, selection="all", include_header=False)

        # Mirror the freshly generated geometry into the solver tree so that
        # ``run_lbm_simulation`` can pick it up without extra symlink logic.
        staged_path = self.tnllbm_root / basename
        shutil.copy2(local_path, staged_path)
        return basename

    def _run_simulation(self, geometry_name: str, angle_deg: float) -> float:
        """Submit the staged geometry to Slurm and return the scalar objective."""
        from run_lbm_simulation import submit_and_collect

        result = submit_and_collect(
            geometry=geometry_name,
            resolution=self.resolution,
            partition=LBM_PARTITION,
            walltime=LBM_WALLTIME,
            gpus=LBM_GPUS,
            cpus=LBM_CPUS,
            mem=LBM_MEM,
            runs_root=self.runs_root,
            job_name=None,
            type1_bouzidi=self.type1_bouzidi,
            poll_interval=LBM_POLL_INTERVAL,
            timeout=LBM_JOB_TIMEOUT,
            result_timeout=LBM_RESULT_TIMEOUT,
        )
        if result.numeric_value is None:
            raise RuntimeError("LBM runner did not return a numeric objective value")
        value = float(result.numeric_value)
        print(f"angle={angle_deg:.4f} deg -> objective={value:.6f}", flush=True)
        return value


def build_geometry_config(semi_major: float, semi_minor: float) -> GeometryConfig:
    """Construct the canonical grid and ellipse placement used in the study."""
    grid = Grid(nx=1024, ny=256, dx=1.0, origin=(0.0, 0.0))
    quarter_w = grid.nx // 4
    x0_idx = quarter_w + quarter_w // 2
    y0_idx = grid.ny // 2
    x0 = grid.origin[0] + x0_idx * grid.dx
    y0 = grid.origin[1] + y0_idx * grid.dx
    return GeometryConfig(
        grid=grid,
        center_x=x0,
        center_y=y0,
        semi_major=semi_major,
        semi_minor=semi_minor,
    )


def main() -> int:
    """Configure the optimisation problem and execute the Nelder–Mead run."""
    geom_cfg = build_geometry_config(ELLIPSE_SEMI_MAJOR, ELLIPSE_SEMI_MINOR)
    geometry_root = GEOMETRY_WORKDIR.resolve()
    tnllbm_root = TNL_LBM_ROOT.resolve()
    # Ensure we can import ``run_lbm_simulation`` directly from the submodule.
    if str(tnllbm_root) not in sys.path:
        sys.path.insert(0, str(tnllbm_root))
    objective = LBMSimulationObjective(
        geometry_config=geom_cfg,
        geometry_workdir=geometry_root,
        tnllbm_root=tnllbm_root,
        resolution=LBM_RESOLUTION,
        type1_bouzidi=LBM_TYPE1_BOUZIDI,
        runs_root=LBM_RUNS_ROOT,
    )

    space = DesignSpace(lower=[0.0], upper=[180.0], names=["angle_deg"])
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
