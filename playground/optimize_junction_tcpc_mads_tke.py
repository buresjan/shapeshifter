#!/usr/bin/env python3
"""Parallel MADS optimisation of the TCPC junction geometry (TKE objective).

This entry point mirrors ``optimize_junction_tcpc_tke.py`` but swaps the optimiser
for NOMAD's Mesh Adaptive Direct Search (via ``optilb``). The objective,
geometry generation and solver staging are imported directly from the TKE-aware
Nelder–Mead runner so both pipelines remain identical for apples-to-apples comparisons.

Evaluation caching is implemented locally to keep behaviour consistent with the
Nelder–Mead setup (which enables the optimiser-level memoisation flag).
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

# Reuse the proven objective pipeline and configuration from the Nelder–Mead run.
from optimize_junction_tcpc_tke import (  # type: ignore
    GEOMETRY_PENALTY,
    LOWER,
    MAX_EVALS,
    PARAMETER_NAMES,
    UPPER,
    X0,
    _objective as _nm_objective,
    _current_eval_log_path,
    _log_eval,
    _next_eval_id,
)

ALGORITHM_LABEL = "mads_tke"


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
    key = tuple(float(v) for v in arr)
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

    print("[opt] Starting MADS optimisation (TKE objective)")
    print(
        f"[opt] max_evals={MAX_EVALS}, workers={N_WORKERS}, normalize=True, memoize=True"
    )
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
