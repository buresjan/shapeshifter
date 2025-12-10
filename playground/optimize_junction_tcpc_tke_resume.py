#!/usr/bin/env python3
"""Resume Nelder–Mead TKE optimisation for the TCPC junction.

This mirrors ``optimize_junction_tcpc_tke.py`` but restarts from a recorded
simplex (with objective values) around iter-011 and tight local bounds.
Differences versus the base run:
  - Resolution forced to 4.
  - Max evaluations capped at 40.
  - Initial simplex + values are provided to skip re-evaluating the seed.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from optilb import DesignSpace, OptimizationProblem

import optimize_junction_tcpc_tke as base  # noqa: E402

# Force the base module to use the lower resolution for geometry + solver.
RESOLUTION = 4
base.RESOLUTION = RESOLUTION

MAX_EVALS = 40
PARAMETER_NAMES = base.PARAMETER_NAMES
GEOMETRY_PENALTY = base.GEOMETRY_PENALTY
N_WORKERS = base.N_WORKERS
LOG_SIMPLEX = base.LOG_SIMPLEX

# Local-refinement bounds around the iter-011 simplex
LOWER = np.array(
    [
        -0.003,  # offset  (all good points in [-0.0018, 0.0004])
        4.0,  # lower_angle (good region ~ [4.0, 5.6])
        -2.5,  # upper_angle (good region ~ [-1.8, 0.4])
        0.0017,  # lower_flare (good region ~ [0.0018, 0.0021])
        0.0006,  # upper_flare (good region ~ [0.00066, 0.00093])
    ],
    dtype=float,
)

UPPER = np.array(
    [
        0.003,  # offset
        6.0,  # lower_angle
        0.6,  # upper_angle
        0.0023,  # lower_flare
        0.0011,  # upper_flare
    ],
    dtype=float,
)

# Imposed starting simplex (order matches PARAMETER_NAMES)
STARTING_SIMPLEX = [
    np.array([-0.000142, 4.071346, 0.387486, 0.002020, 0.000690], dtype=float),
    np.array([0.000312, 4.090303, -1.359927, 0.002068, 0.000681], dtype=float),
    np.array([-0.000336, 5.207173, 0.300243, 0.001956, 0.000754], dtype=float),
    np.array([-0.001054, 5.150043, -0.182016, 0.001825, 0.000713], dtype=float),
    np.array([-0.001826, 5.559794, -1.742228, 0.002034, 0.000926], dtype=float),
    np.array([0.000429, 4.035673, -1.306257, 0.001822, 0.000845], dtype=float),
]

STARTING_VALUES = [
    172.075,
    177.438,
    194.748,
    196.907,
    210.792,
    220.783,
]

# Use the best-known vertex as x0; the provided simplex overrides it internally.
X0 = STARTING_SIMPLEX[0]

# No-improvement stopping mirroring the previous resume run
NO_IMPROV_THR = 1e-3
NO_IMPROV_BREAK = 10

ALGORITHM_LABEL = "nelder_mead_tke_resume"


def _objective_resume(x: np.ndarray) -> float:
    """Objective wrapper to tag evaluations under the resume label."""
    return float(base._objective(np.asarray(x, dtype=float), algorithm=ALGORITHM_LABEL))


def main() -> Tuple[np.ndarray, float]:
    space = DesignSpace(lower=LOWER, upper=UPPER, names=PARAMETER_NAMES)
    optimize_options = {
        "initial_simplex": STARTING_SIMPLEX,
        "initial_simplex_values": STARTING_VALUES,
    }

    nm = base.LoggingNelderMeadOptimizer(
        n_workers=N_WORKERS,
        memoize=True,
        parallel_poll_points=True,
        no_improve_thr=NO_IMPROV_THR,
        no_improve_break=NO_IMPROV_BREAK,
        penalty=GEOMETRY_PENALTY,
        log_simplex=LOG_SIMPLEX,
    )

    problem = OptimizationProblem(
        objective=_objective_resume,
        space=space,
        x0=X0,
        optimizer=nm,
        parallel=True,
        normalize=True,
        max_evals=MAX_EVALS,
        tol=NO_IMPROV_THR,
        verbose=True,
        optimize_options=optimize_options,
    )

    print("[opt] Starting Nelder–Mead optimisation (TKE resume)")
    print(
        f"[opt] max_evals={MAX_EVALS}, workers={N_WORKERS}, normalize=True, "
        f"memoize=True, simplex_log={LOG_SIMPLEX}"
    )
    print(
        f"[opt] resolution={RESOLUTION}, no_improve_thr={NO_IMPROV_THR} "
        f"(break={NO_IMPROV_BREAK})"
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
    log_path = base._current_eval_log_path(ALGORITHM_LABEL)
    if log_path is not None:
        print(f"[opt] Evaluation CSV log saved to: {log_path}")

    return best_x, best_f


if __name__ == "__main__":
    main()
