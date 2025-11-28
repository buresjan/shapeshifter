#!/usr/bin/env python3
"""Parallel MADS optimisation of the TCPC junction geometry (resume setup).

This entry point mirrors ``optimize_junction_tcpc_mads.py`` but restarts the
search from a new initial guess and tighter bounds. The objective, geometry
generation and solver staging are imported directly from the Nelder–Mead runner
so both pipelines remain identical for apples-to-apples comparisons.

Evaluation caching is implemented locally to keep behaviour consistent with the
Nelder–Mead setup (which enables the optimiser-level memoisation flag).
"""

from __future__ import annotations

import math
import os
import threading
from typing import Dict, Tuple

import numpy as np

from optilb import DesignSpace, OptimizationProblem
from optilb.optimizers import MADSOptimizer

# Reuse the proven objective pipeline and configuration from the Nelder–Mead run.
from optimize_junction_tcpc import (  # type: ignore
    GEOMETRY_PENALTY,
    MAX_EVALS,
    PARAMETER_NAMES,
    _objective as _nm_objective,
    _current_eval_log_path,
    _log_eval,
    _next_eval_id,
)


# ---------------------------------------------------------------------------
# Resume configuration: new initial guess and bounds
# ---------------------------------------------------------------------------
X0 = np.array([-1.0e-3, 4.0, -3.0, 2.5e-3, 1.25e-3], dtype=float)

LOWER = np.array(
    [
        -1.0000e-02,  # offset      (kept original lower bound)
        -5.0000e-01,  # lower_angle
        -7.7025e+00,  # upper_angle
        1.8750e-03,   # lower_flare
        6.2500e-04,   # upper_flare
    ],
    dtype=float,
)

UPPER = np.array(
    [
        8.0000e-03,  # offset
        8.5000e+00,  # lower_angle
        -1.0875e+00, # upper_angle
        2.5000e-03,  # lower_flare (hits original upper bound)
        1.3750e-03,  # upper_flare
    ],
    dtype=float,
)


# ---------------------------------------------------------------------------
# Parallel configuration
# ---------------------------------------------------------------------------
# Default to OPT_MADS_WORKERS, but fall back to OPT_NM_WORKERS for convenience.
N_WORKERS = int(os.environ.get("OPT_MADS_WORKERS", os.environ.get("OPT_NM_WORKERS", "8")))


# ---------------------------------------------------------------------------
# Simple thread-safe memoization layer (MADS disables optimiser-level memoize)
# ---------------------------------------------------------------------------
_OBJECTIVE_CACHE: Dict[Tuple[float, ...], float] = {}
_CACHE_LOCK = threading.Lock()


def _memoized_objective(x: np.ndarray) -> float:
    """Memoized adapter around the shared TCPC objective."""

    arr = np.asarray(x, dtype=float)
    key = tuple(float(v) for v in arr)
    eval_id = _next_eval_id("mads")
    with _CACHE_LOCK:
        cached = _OBJECTIVE_CACHE.get(key)
    if cached is not None:
        log_path = _log_eval("mads", eval_id, arr, cached)
        print(
            f"[obj] eval {eval_id:04d} cached value={cached:.6g} for x={key} (log={log_path})",
            flush=True,
        )
        return float(cached)
    value = float(_nm_objective(arr, algorithm="mads", eval_id=eval_id))
    if not math.isfinite(value):
        raise RuntimeError(f"Objective returned non-finite value {value} for x={key}")
    if value >= GEOMETRY_PENALTY:
        print(
            f"[obj] constraint penalty triggered (value={value:.6g}) for x={key}",
            flush=True,
        )
    with _CACHE_LOCK:
        _OBJECTIVE_CACHE[key] = value
    return value


# ---------------------------------------------------------------------------
# Optimisation driver
# ---------------------------------------------------------------------------
def main() -> Tuple[np.ndarray, float]:
    if not MADSOptimizer.is_available():
        raise RuntimeError(
            "PyNomad (PyNomadBBO) is required for the MADS optimiser but is not installed."
        )

    space = DesignSpace(lower=LOWER, upper=UPPER, names=PARAMETER_NAMES)

    mads = MADSOptimizer(
        n_workers=N_WORKERS,
    )

    problem = OptimizationProblem(
        objective=_memoized_objective,
        space=space,
        x0=X0,
        optimizer=mads,
        parallel=True,
        normalize=True,
        max_evals=MAX_EVALS,
        verbose=True,
    )

    print("[opt] Starting MADS optimisation")
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

    evaluations = getattr(res, "evaluations", ()) or ()
    if evaluations:
        names = space.names or tuple(f"x{i}" for i in range(space.dimension))
        print(f"[opt] Evaluation log ({len(evaluations)} entries):")
        for idx, record in enumerate(evaluations, start=1):
            coords = ", ".join(
                f"{name}={float(val):.6g}" for name, val in zip(names, record.x)
            )
            print(f"  - eval {idx:03d}: f={record.value:.6g}; {coords}")
    log_path = _current_eval_log_path("mads")
    if log_path is not None:
        print(f"[opt] Evaluation CSV log saved to: {log_path}")

    return best_x, best_f


if __name__ == "__main__":
    main()
