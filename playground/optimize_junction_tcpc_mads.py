#!/usr/bin/env python3
"""Parallel MADS optimisation of the TCPC junction geometry.

This entry point mirrors ``optimize_junction_tcpc.py`` but swaps the optimiser
for NOMAD's Mesh Adaptive Direct Search (via ``optilb``). The objective,
geometry generation and solver staging are imported directly from the Nelder–
Mead runner so both pipelines remain identical for apples-to-apples comparisons.

Evaluation caching is implemented locally to keep behaviour consistent with the
Nelder–Mead setup (which enables the optimiser-level memoisation flag).
"""

from __future__ import annotations

import os
import threading
from typing import Dict, Tuple

import numpy as np

from optilb import DesignSpace, OptimizationProblem
from optilb.optimizers import MADSOptimizer

# Reuse the proven objective pipeline and configuration from the Nelder–Mead run.
from optimize_junction_tcpc import (  # type: ignore
    LOWER,
    MAX_EVALS,
    UPPER,
    X0,
    _objective as _nm_objective,
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
    with _CACHE_LOCK:
        cached = _OBJECTIVE_CACHE.get(key)
    if cached is not None:
        return float(cached)
    value = float(_nm_objective(arr))
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

    space = DesignSpace(lower=LOWER, upper=UPPER, names=(
        "offset", "lower_angle", "upper_angle", "lower_flare", "upper_flare"
    ))

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

    return best_x, best_f


if __name__ == "__main__":
    main()
