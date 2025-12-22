#!/usr/bin/env python3
"""Second resume/zoom MADS optimisation of the TCPC junction geometry (TKE objective).

This script continues the MADS-TKE search after the first resume run, focusing the
design space around a newer promising point. The objective pipeline (geometry build,
staging, solver invocation and logging) is reused from ``optimize_junction_tcpc_tke.py``
to keep outputs comparable.
"""

from __future__ import annotations

import math
import os
import threading
import multiprocessing
from typing import Dict, Tuple

import numpy as np

from optilb import DesignSpace, OptimizationProblem
from optilb.optimizers import MADSOptimizer

# Reuse the proven objective pipeline and evaluation logging from the NM-TKE runner.
from optimize_junction_tcpc_tke import (  # type: ignore
    GEOMETRY_PENALTY,
    PARAMETER_NAMES,
    _current_eval_log_path,
    _log_eval,
    _next_eval_id,
)

ALGORITHM_LABEL = "mads_tke_resume2"


# ---------------------------------------------------------------------------
# Resume configuration: supplied point + tight bounds
# ---------------------------------------------------------------------------
# Starting point for the second resume run.
X0 = np.array([-1.0e-3, -0.97, -5.04, 2.25e-3, 1.02e-3], dtype=float)

# Bounds (order: offset, lower_angle, upper_angle, lower_flare, upper_flare)
LOWER = np.array(
    [-4.0e-3, -3.5, -7.2, 1.75e-3, 6.0e-4],
    dtype=float,
)
UPPER = np.array(
    [2.0e-3, 0.8, -3.6, 2.50e-3, 1.30e-3],
    dtype=float,
)

# Cap evaluations for this refinement run (override via env var if desired).
MAX_EVALS = int(os.environ.get("TCPC_MADS_TKE_RESUME2_MAX_EVALS", "70"))


def _validate_resume_bounds() -> None:
    if LOWER.shape != UPPER.shape:
        raise ValueError("LOWER/UPPER must have the same shape")
    if np.any(~np.isfinite(LOWER)) or np.any(~np.isfinite(UPPER)):
        raise ValueError("LOWER/UPPER must be finite")
    if np.any(UPPER <= LOWER):
        raise ValueError("Each UPPER bound must be strictly greater than LOWER")
    if X0.shape != LOWER.shape:
        raise ValueError("X0 must match bound dimension")
    if np.any(X0 < LOWER) or np.any(X0 > UPPER):
        raise ValueError(f"X0={X0} must lie within [LOWER, UPPER]")


# ---------------------------------------------------------------------------
# Parallel configuration
# ---------------------------------------------------------------------------
# Default to OPT_MADS_WORKERS, fall back to OPT_NM_WORKERS, then 4 to stay within 16G.
N_WORKERS = int(os.environ.get("OPT_MADS_WORKERS", os.environ.get("OPT_NM_WORKERS", "4")))


# ---------------------------------------------------------------------------
# Simple thread-safe memoization layer (MADS disables optimiser-level memoize)
# ---------------------------------------------------------------------------
def _cache_key(x: np.ndarray) -> Tuple[float, ...]:
    """Return a stable cache key (tolerates tiny float roundoff)."""
    arr = np.asarray(x, dtype=float)
    return tuple(round(float(v), 12) for v in arr)


_OBJECTIVE_CACHE: Dict[Tuple[float, ...], float] = {}
_CACHE_LOCK = threading.Lock()


def _objective_worker(arr: np.ndarray, eid: int, queue: multiprocessing.Queue) -> None:
    """Module-level worker so it is picklable under the spawn start method."""
    try:
        import numpy as _np
        from optimize_junction_tcpc_tke import _objective as _nm_obj  # type: ignore

        val = float(
            _nm_obj(_np.asarray(arr, dtype=float), algorithm=ALGORITHM_LABEL, eval_id=eid)
        )
        queue.put(("ok", val))
    except Exception as exc:  # pragma: no cover - safety net
        queue.put(("err", str(exc)))


def _objective_subprocess(x: np.ndarray, *, eval_id: int) -> float:
    """Run the shared TCPC objective in a fresh process to avoid Gmsh thread limits."""

    # Prefer fork on POSIX to avoid signal handler issues when spawning from worker threads,
    # but keep a spawn fallback for platforms without fork.
    ctx = (
        multiprocessing.get_context("fork")
        if "fork" in multiprocessing.get_all_start_methods()
        else multiprocessing.get_context("spawn")
    )
    q: multiprocessing.Queue = ctx.Queue()

    proc = ctx.Process(target=_objective_worker, args=(np.asarray(x, dtype=float), eval_id, q))
    proc.start()
    proc.join()

    status, payload = q.get() if not q.empty() else ("err", "no-result")
    if status == "ok":
        return float(payload)
    # On error or missing result, propagate a penalty so the optimiser can continue.
    return float(GEOMETRY_PENALTY)


def _memoized_objective(x: np.ndarray) -> float:
    """Memoized adapter around the shared TCPC objective."""

    arr = np.asarray(x, dtype=float)
    key = _cache_key(arr)
    eval_id = _next_eval_id(ALGORITHM_LABEL)
    with _CACHE_LOCK:
        cached = _OBJECTIVE_CACHE.get(key)
    if cached is not None:
        log_path = _log_eval(ALGORITHM_LABEL, eval_id, arr, cached)
        print(
            f"[obj] eval {eval_id:04d} cached value={cached:.6g} for x={key} (log={log_path})",
            flush=True,
        )
        return float(cached)
    value = float(_objective_subprocess(arr, eval_id=eval_id))
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
    _validate_resume_bounds()

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

    print("[opt] Starting MADS optimisation (TKE objective, resume/zoom #2)")
    print(
        f"[opt] max_evals={MAX_EVALS}, workers={N_WORKERS}, normalize=True, memoize=True"
    )
    print("[opt] Resume start:")
    for name, val in zip(space.names or (), X0):
        print(f"  - {name}: {float(val):.6g}")
    print("[opt] Resume bounds:")
    for name, lo, hi in zip(space.names or (), LOWER, UPPER):
        print(f"  - {name}: [{float(lo):.6g}, {float(hi):.6g}]")

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
    log_path = _current_eval_log_path(ALGORITHM_LABEL)
    if log_path is not None:
        print(f"[opt] Evaluation CSV log saved to: {log_path}")

    return best_x, best_f


if __name__ == "__main__":
    main()
