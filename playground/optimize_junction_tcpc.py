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
  1) Generate the mesh triplet (`geom_`, `dim_`, `angle_`) with meshgen under a
     temporary workspace.
  2) Stage the triplet inside the `tnl-lbm` repository under `sim_NSE`.
  3) Invoke `submodules/tnl-lbm/run_tcpc_simulation.py` so the solver is
     submitted via Slurm (`sbatch`) and monitored until completion.
  4) Return the scalar objective emitted by the solver (failing loudly if the
     job terminates without a value).

Run instructions (local or Slurm):
  - Slurm: submit the launcher:
      sbatch playground/optimize_junction_tcpc.sh

Notes:
- No CLI knobs — all parameters live in this script for clarity.
- Parallel evaluation uses a thread pool to avoid pickling issues. Each worker
  stages its own uniquely named geometry artefacts before submitting the Slurm
  job via `run_tcpc_simulation.py`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from optilb import DesignSpace, OptimizationProblem

# Reuse the proven geometry helpers from the runnable TCPC script.
from run_junction_tcpc import (
    default_paths,
    ensure_txt_suffix,
    generate_geometry,
    locate_solver_root,
    prepare_run_directory,
    stage_geometry,
)


# ---------------------------------------------------------------------------
# Fixed configuration (edit here, no CLI)
# ---------------------------------------------------------------------------
RESOLUTION = 4
MAX_EVALS = 40

# Initial point (offset, lower_angle, upper_angle, lower_flare, upper_flare)
# Geometry union is only robust when branches stay close to the stem.
X0 = np.array([0.0, 0.0, 0.0, 0.00075, 0.00075], dtype=float)

# Bounds
LOWER = np.array([-0.005, -10.0, -10.0, 0.000, 0.000], dtype=float)
UPPER = np.array([+0.005, +10.0, +10.0, 0.0015, 0.0015], dtype=float)

# Penalised objective value when geometry cannot be generated
GEOMETRY_PENALTY = 1.0e9

# Parallel evaluation workers (threads)
N_WORKERS = int(os.environ.get("OPT_NM_WORKERS", "8"))

# Slurm submission defaults (override via environment variables if desired)
def _env_optional_int(var: str, default: int | None) -> int | None:
    raw = os.environ.get(var)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {var} must be an integer, got {raw!r}") from exc


def _env_float(var: str, default: float) -> float:
    raw = os.environ.get(var)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {var} must be a float, got {raw!r}") from exc


def _env_bool(var: str, default: bool = False) -> bool:
    raw = os.environ.get(var)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


SLURM_PARTITION = os.environ.get("TCPC_SLURM_PARTITION")
SLURM_GPUS = _env_optional_int("TCPC_SLURM_GPUS", 1)
SLURM_CPUS = _env_optional_int("TCPC_SLURM_CPUS", 8)
SLURM_MEM = os.environ.get("TCPC_SLURM_MEM", "32G")
SLURM_WALLTIME = os.environ.get("TCPC_SLURM_WALLTIME", "20:00:00")
SLURM_POLL_INTERVAL = _env_float("TCPC_SLURM_POLL_INTERVAL", 60.0)
SLURM_AVG_WINDOW = _env_float("TCPC_SLURM_AVG_WINDOW", 1.0)
SLURM_VERBOSE = _env_bool("TCPC_SLURM_VERBOSE", False)


@dataclass(frozen=True)
class Paths:
    project_root: Path
    solver_binary: Path
    solver_root: Path
    run_tcpc_script: Path


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
    run_tcpc_script = solver_root / "run_tcpc_simulation.py"
    if not run_tcpc_script.is_file():
        raise FileNotFoundError(
            f"run_tcpc_simulation.py not found at '{run_tcpc_script}'. Ensure the tnl-lbm submodule is present."
        )
    return Paths(
        project_root=project_root,
        solver_binary=solver_binary,
        solver_root=solver_root,
        run_tcpc_script=run_tcpc_script,
    )


def _hash_params(x: np.ndarray) -> str:
    """Stable short hash for a parameter vector (used for file/run naming)."""
    txt = ",".join(f"{v:.8g}" for v in np.asarray(x, dtype=float))
    return hashlib.sha1(txt.encode("ascii")).hexdigest()[:12]


@lru_cache(maxsize=1)
def _load_run_tcpc_module(script_path: Path):
    """Load the run_tcpc_simulation module and patch staging for per-run data."""
    spec = importlib.util.spec_from_file_location("tnl_lbm_run_tcpc", str(script_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    if not hasattr(module, "run_tcpc_simulation"):
        raise AttributeError(
            f"'run_tcpc_simulation.py' at {script_path} does not expose run_tcpc_simulation"
        )

    if not hasattr(module, "_codex_stage_registry"):
        module._codex_stage_registry = {}

    if not getattr(module, "_codex_make_run_dir_patched", False):
        original_make_run_dir = getattr(module, "_make_run_dir")

        def _make_run_dir_with_staging(project_root: Path, filename: str) -> Path:
            run_dir = original_make_run_dir(project_root, filename)
            registry = getattr(module, "_codex_stage_registry", {})
            stage_queue = registry.get(filename)
            if stage_queue:
                stage_sources = stage_queue.pop(0)
                if not stage_queue:
                    registry.pop(filename, None)
                sim_root = run_dir / "sim_NSE"
                for subdir in ("geometry", "dimensions", "angle", "tmp"):
                    (sim_root / subdir).mkdir(parents=True, exist_ok=True)
                for label, source in stage_sources.items():
                    target_dir = "tmp" if label == "values" else label
                    destination = sim_root / target_dir / source.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
            return run_dir

        module._make_run_dir = _make_run_dir_with_staging  # type: ignore[attr-defined]
        module._codex_make_run_dir_patched = True

    return module


def _objective(x: np.ndarray) -> float:
    """Objective wrapper: generate geometry, run solver, return scalar."""
    p = _project_paths()

    offset, lower_angle, upper_angle, lower_flare, upper_flare = map(float, x)

    case_tag = _hash_params(x)
    case_name = ensure_txt_suffix(f"junction_{case_tag}.txt")
    generated_basename = Path(case_name).name

    print(
        "[obj] design",
        f"offset={offset:.6f}",
        f"lower_angle={lower_angle:.6f}",
        f"upper_angle={upper_angle:.6f}",
        f"lower_flare={lower_flare:.6f}",
        f"upper_flare={upper_flare:.6f}",
        flush=True,
    )

    run_dir = prepare_run_directory(p.project_root, generated_basename)
    workspace = run_dir / "meshgen_output"

    try:
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
    except Exception as exc:
        print(f"[obj] geometry failed: {exc}", flush=True)
        shutil.rmtree(workspace, ignore_errors=True)
        shutil.rmtree(run_dir, ignore_errors=True)
        return float(GEOMETRY_PENALTY)
    generated_basename = Path(generated_case_name).name

    staged_files: Dict[str, Path] = {}
    cleanup_geometry = False
    cleanup_run_dir = False

    try:
        to_stage = {k: v for k, v in files.items() if k in {"geometry", "dimensions", "angle"}}
        tcpc_module = _load_run_tcpc_module(p.run_tcpc_script)
        registry = getattr(tcpc_module, "_codex_stage_registry")
        registry.setdefault(generated_basename, []).append(dict(to_stage))

        data_root = p.solver_root / "sim_NSE"
        staged_files = stage_geometry(to_stage, data_root)

        run_tcpc = tcpc_module.run_tcpc_simulation  # type: ignore[attr-defined]
        value = run_tcpc(
            generated_basename,
            resolution=RESOLUTION,
            project_root=p.solver_root,
            binary_path=p.solver_binary,
            partition=SLURM_PARTITION,
            gpus=SLURM_GPUS,
            cpus=SLURM_CPUS,
            mem=SLURM_MEM,
            walltime=SLURM_WALLTIME,
            poll_interval=SLURM_POLL_INTERVAL,
            default_on_failure=float("nan"),
            avg_window=SLURM_AVG_WINDOW,
            verbose=SLURM_VERBOSE,
        )
        if math.isnan(value):
            raise RuntimeError(
                f"TCPC Slurm run for case '{generated_basename}' returned NaN. "
                "Check runs under submodules/tnl-lbm/runs_tcpc/ for diagnostics."
            )

        cleanup_geometry = True
        cleanup_run_dir = True
        return float(value)
    finally:
        if cleanup_geometry and staged_files:
            for path in staged_files.values():
                try:
                    path.unlink()
                except FileNotFoundError:
                    continue
                except OSError:
                    continue
        if cleanup_run_dir:
            shutil.rmtree(run_dir, ignore_errors=True)


def main() -> Tuple[np.ndarray, float]:
    space = DesignSpace(lower=LOWER, upper=UPPER, names=(
        "offset", "lower_angle", "upper_angle", "lower_flare", "upper_flare"
    ))

    from optilb.optimizers import NelderMeadOptimizer

    nm = NelderMeadOptimizer(
        n_workers=N_WORKERS,
        memoize=True,
        parallel_poll_points=True,
    )

    problem = OptimizationProblem(
        objective=_objective,
        space=space,
        x0=X0,
        optimizer=nm,
        parallel=True,
        normalize=True,
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
