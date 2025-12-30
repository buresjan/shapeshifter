#!/usr/bin/env python3
"""Resume-friendly Nelder–Mead optimisation of the TCPC junction geometry.

This entry point mirrors ``optimize_junction_tcpc.py`` but:
  - starts from a new initial guess and tightened bounds (see X0/LOWER/UPPER),
  - optionally resumes from a precomputed simplex + values (via NPZ),
  - logs simplex coordinates each iteration for easier monitoring, and
  - uses a stricter no-improvement threshold (10e-4) with break=10.

Set ``TCPC_NM_SIMPLEX_NPZ`` to an ``.npz`` containing arrays ``simplex``
((d+1) x d) and ``values`` (d+1,) to skip evaluating the starting simplex.
Disable simplex logging with ``TCPC_LOG_SIMPLEX=0`` if desired.
"""

from __future__ import annotations

import inspect
import os
from functools import partial
from pathlib import Path
from typing import Callable, Sequence, Tuple, cast

import numpy as np

import optilb.optimizers.nelder_mead as nm_impl
from optilb import DesignSpace, OptimizationProblem
from optilb.optimizers.nelder_mead import NelderMeadOptimizer

from optimize_junction_tcpc import (  # type: ignore
    MAX_EVALS,
    PARAMETER_NAMES,
    RESOLUTION,
    _current_eval_log_path,
    _objective as _nm_objective,
)


# ---------------------------------------------------------------------------
# Fixed configuration for the resumed run
# ---------------------------------------------------------------------------
X0 = np.array(
    [-0.00392024, -2.60526, 6.34250, 0.00178604, 0.00135284],
    dtype=float,
)

LOWER = np.array(
    [-0.010, -10.0, -5.0, 0.0010, 0.0005],
    dtype=float,
)
UPPER = np.array(
    [0.006, 13.0, 15.0, 0.0025, 0.0025],
    dtype=float,
)

# Penalised objective value when geometry cannot be generated (matches NM base)
GEOMETRY_PENALTY = 1.0e9

# Parallel evaluation workers (threads by default)
N_WORKERS = int(os.environ.get("OPT_NM_WORKERS", "8"))

# No-improvement stopping
NO_IMPROV_THR = 1e-3
NO_IMPROV_BREAK = 10

# Optional simplex resume file (np.savez with "simplex" and "values")
SIMPLEX_ENV_VAR = "TCPC_NM_SIMPLEX_NPZ"

# Simplex logging toggle
LOG_SIMPLEX = os.environ.get("TCPC_LOG_SIMPLEX", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}

# Label for evaluation / logging separation from the original NM run
ALGORITHM_LABEL = "nelder_mead_resume"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _objective_resume(x: np.ndarray) -> float:
    """Delegate to the shared TCPC objective with a distinct algorithm label."""
    return float(_nm_objective(np.asarray(x, dtype=float), algorithm=ALGORITHM_LABEL))


def _maybe_load_initial_simplex(dimension: int) -> tuple[list[np.ndarray], list[float], Path] | None:
    """Load a precomputed simplex + values from NPZ, if configured."""
    path_str = os.environ.get(SIMPLEX_ENV_VAR, "").strip()
    if not path_str:
        return None
    path = Path(path_str).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Initial simplex file '{path}' not found")

    data = np.load(path, allow_pickle=False)
    if "simplex" not in data or "values" not in data:
        raise ValueError(
            f"{path} must contain 'simplex' ((d+1)x d) and 'values' (d+1,) arrays"
        )

    simplex_raw = np.asarray(data["simplex"], dtype=float)
    values_raw = np.asarray(data["values"], dtype=float).reshape(-1)
    expected_vertices = dimension + 1
    if simplex_raw.shape != (expected_vertices, dimension):
        raise ValueError(
            f"simplex array in {path} must have shape {(expected_vertices, dimension)}, "
            f"got {tuple(simplex_raw.shape)}"
        )
    if values_raw.shape[0] != expected_vertices:
        raise ValueError(
            f"values array in {path} must have length {expected_vertices}, "
            f"got {values_raw.shape[0]}"
        )
    if not np.all(np.isfinite(values_raw)):
        raise ValueError(f"values array in {path} must be finite")

    simplex = [np.asarray(row, dtype=float).copy() for row in simplex_raw]
    values = [float(v) for v in values_raw]
    return simplex, values, path


class LoggingNelderMeadOptimizer(NelderMeadOptimizer):
    """Nelder–Mead variant that prints simplex coordinates each iteration."""

    def __init__(self, *, log_simplex: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.log_simplex = log_simplex

    def _log_simplex_state(
        self,
        iteration: int,
        simplex: list[np.ndarray],
        fvals: list[float],
        transform: nm_impl.SpaceTransform | None,
    ) -> None:
        if not self.log_simplex:
            return
        entries = []
        for idx, (pt, val) in enumerate(zip(simplex, fvals)):
            coords = (
                transform.from_unit(np.asarray(pt, dtype=float))
                if transform is not None
                else np.asarray(pt, dtype=float)
            )
            coords_fmt = ", ".join(f"{float(c):+.6f}" for c in coords)
            entries.append(f"v{idx}: f={float(val):.6g} [{coords_fmt}]")
        print(f"[simplex] iter={iteration:03d} " + "; ".join(entries), flush=True)

    # NOTE: This method is adapted from optilb.optimizers.nelder_mead.NelderMeadOptimizer
    # so we can emit simplex coordinates each iteration.
    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        x0: np.ndarray,
        space: DesignSpace,
        constraints: Sequence[nm_impl.Constraint] = (),
        *,
        initial_simplex: Sequence[np.ndarray] | np.ndarray | None = None,
        initial_simplex_values: Sequence[float] | None = None,
        max_iter: int = 100,
        max_evals: int | None = None,
        tol: float = 1e-6,
        seed: int | None = None,
        parallel: bool = False,
        verbose: bool = False,
        early_stopper: nm_impl.EarlyStopper | None = None,
        normalize: bool = False,
    ) -> nm_impl.OptResult:
        if seed is not None:
            np.random.default_rng(seed)

        original_space = space
        provided_simplex: list[np.ndarray] | None = None
        provided_fvals: list[float] | None = None
        if initial_simplex is not None or initial_simplex_values is not None:
            if initial_simplex is None or initial_simplex_values is None:
                raise ValueError(
                    "initial_simplex and initial_simplex_values must be provided together"
                )
            provided_simplex = [np.asarray(pt, dtype=float) for pt in initial_simplex]
            provided_fvals = [float(val) for val in initial_simplex_values]
            expected_vertices = original_space.dimension + 1
            if len(provided_simplex) != expected_vertices:
                raise ValueError("initial_simplex must contain dimension + 1 vertices")
            if len(provided_fvals) != len(provided_simplex):
                raise ValueError(
                    "initial_simplex_values length must match initial_simplex"
                )
            for vertex in provided_simplex:
                if vertex.shape != original_space.lower.shape:
                    raise ValueError("Initial simplex vertex has wrong dimension")
                if np.any(vertex < original_space.lower) or np.any(
                    vertex > original_space.upper
                ):
                    raise ValueError("Initial simplex vertex outside design bounds")

        normalize_transform: nm_impl.SpaceTransform | None = None
        self._history_transform = None
        eval_map: Callable[[np.ndarray], np.ndarray] | None = None

        if normalize:
            normalize_transform = nm_impl.SpaceTransform(space)
            lower = normalize_transform.lower
            span = normalize_transform.span
            eval_map = normalize_transform.from_unit

            objective = cast(
                Callable[[np.ndarray], float],
                partial(
                    nm_impl._objective_from_unit,
                    objective=objective,
                    lower=lower,
                    span=span,
                ),
            )
            constraints = [
                nm_impl.Constraint(
                    func=partial(
                        nm_impl._constraint_from_unit,
                        func=c.func,
                        lower=lower,
                        span=span,
                    ),
                    name=c.name,
                )
                for c in constraints
            ]

            space = DesignSpace(np.zeros(space.dimension), np.ones(space.dimension))
            x0 = normalize_transform.to_unit(x0)
            if provided_simplex is not None:
                provided_simplex = [
                    normalize_transform.to_unit(pt) for pt in provided_simplex
                ]
            self._history_transform = normalize_transform

        if provided_simplex is not None:
            x0 = np.asarray(provided_simplex[0], dtype=float)
        x0 = self._validate_x0(x0, space)

        raw_objective = objective
        counted_objective = self._wrap_objective(
            raw_objective,
            map_input=eval_map,
        )

        self.reset_history()
        self._configure_budget(max_evals)

        n = space.dimension
        use_provided_simplex = provided_simplex is not None
        simplex: list[np.ndarray] = (
            [np.asarray(pt, dtype=float) for pt in provided_simplex]
            if use_provided_simplex and provided_simplex is not None
            else [x0]
        )

        self.record(simplex[0], tag="start")

        penalised_counting = self._make_penalised(
            counted_objective, space, constraints
        )
        penalised_raw = self._make_penalised(raw_objective, space, constraints)

        if early_stopper is not None:
            early_stopper.reset()

        fvals: list[float] = []
        try:
            with nm_impl._parallel_executor(parallel, self.n_workers) as (
                executor,
                manual_count,
            ):
                evaluate = penalised_raw if manual_count else penalised_counting
                _ep_params = inspect.signature(self._eval_points).parameters
                _supports_map_kw = "map_input" in _ep_params
                if use_provided_simplex:
                    assert provided_fvals is not None
                    fvals = []
                    for vertex, value in zip(simplex, provided_fvals):
                        arr = np.asarray(vertex, dtype=float)
                        violated = bool(
                            np.any(arr < space.lower) or np.any(arr > space.upper)
                        )
                        if not violated:
                            for constraint in constraints:
                                c_val = constraint(arr)
                                if isinstance(c_val, bool):
                                    if not c_val:
                                        violated = True
                                        break
                                else:
                                    if float(c_val) > 0.0:
                                        violated = True
                                        break
                        adjusted_val = float(self.penalty if violated else value)
                        record_point = (
                            np.asarray(eval_map(arr), dtype=float)
                            if eval_map is not None
                            else arr
                        )
                        with self._state_lock:
                            if (
                                self._max_evals is not None
                                and self._nfev >= self._max_evals
                            ):
                                self._budget_exhausted = True
                                raise nm_impl.EvaluationBudgetExceeded(self._max_evals)
                            self._nfev += 1
                            self._last_eval_point = arr.copy()
                            if (
                                self._max_evals is not None
                                and self._nfev >= self._max_evals
                            ):
                                self._budget_exhausted = True
                        self._update_best(arr, adjusted_val)
                        self._record_evaluation(record_point, adjusted_val)
                        if self._cache_enabled:
                            key = self._make_cache_key(record_point)
                            self._cache[key] = float(adjusted_val)
                        fvals.append(float(adjusted_val))
                else:
                    step = np.asarray(self.step, dtype=float)
                    if step.size == 1:
                        step = np.full(n, float(step))
                    if step.shape != (n,):
                        raise ValueError(
                            "step must be scalar or of length equal to dimension"
                        )
                    for i in range(n):
                        pt = simplex[0].copy()
                        pt[i] += step[i]
                        simplex.append(pt)
                    if _supports_map_kw:
                        fvals = self._eval_points(
                            evaluate,
                            simplex,
                            executor,
                            manual_count,
                            map_input=eval_map,
                        )
                    else:
                        fvals = self._eval_points(
                            evaluate,
                            simplex,
                            executor,
                            manual_count,
                        )

                best = min(fvals)
                no_improv = 0

                for it in range(max_iter):
                    order = np.argsort(fvals)
                    simplex = [simplex[i] for i in order]
                    fvals = [fvals[i] for i in order]

                    self._log_simplex_state(it, simplex, fvals, normalize_transform)

                    current_best = fvals[0]
                    self.record(simplex[0], tag=str(it))
                    if verbose:
                        nm_impl.logger.info("%d | best %.6f", it, current_best)

                    improve_thr = self.no_improve_thr if self.no_improve_thr is not None else tol
                    if early_stopper is None:
                        if best - current_best > improve_thr:
                            best = current_best
                            no_improv = 0
                        else:
                            no_improv += 1
                        if no_improv >= self.no_improv_break:
                            break
                    else:
                        if early_stopper.update(current_best):
                            break

                    centroid = np.mean(simplex[:-1], axis=0)
                    worst = simplex[-1]

                    xr = centroid + self.alpha * (centroid - worst)
                    xe = centroid + self.gamma * (xr - centroid)
                    xoc = centroid + self.beta * (xr - centroid)
                    xic = centroid + self.delta * (worst - centroid)

                    fe: float | None
                    foc: float | None
                    fic: float | None
                    if (
                        parallel
                        and executor is not None
                        and self.parallel_poll_points
                    ):
                        if _supports_map_kw:
                            fr, fe, foc, fic = self._eval_points(
                                evaluate,
                                [xr, xe, xoc, xic],
                                executor,
                                manual_count,
                                map_input=eval_map,
                            )
                        else:
                            fr, fe, foc, fic = self._eval_points(
                                evaluate,
                                [xr, xe, xoc, xic],
                                executor,
                                manual_count,
                            )
                    else:
                        if _supports_map_kw:
                            fr = self._eval_points(
                                evaluate,
                                [xr],
                                executor,
                                manual_count,
                                map_input=eval_map,
                            )[0]
                        else:
                            fr = self._eval_points(
                                evaluate,
                                [xr],
                                executor,
                                manual_count,
                            )[0]
                        fe = foc = fic = None

                    if fvals[0] <= fr < fvals[-2]:
                        simplex[-1] = xr
                        fvals[-1] = fr
                        continue

                    if fr < fvals[0]:
                        if fe is None:
                            if _supports_map_kw:
                                fe = self._eval_points(
                                    evaluate,
                                    [xe],
                                    executor,
                                    manual_count,
                                    map_input=eval_map,
                                )[0]
                            else:
                                fe = self._eval_points(
                                    evaluate,
                                    [xe],
                                    executor,
                                    manual_count,
                                )[0]
                        if fe < fr:
                            simplex[-1] = xe
                            fvals[-1] = fe
                        else:
                            simplex[-1] = xr
                            fvals[-1] = fr
                        continue

                    if fvals[-2] <= fr < fvals[-1]:
                        if foc is None:
                            if _supports_map_kw:
                                foc = self._eval_points(
                                    evaluate,
                                    [xoc],
                                    executor,
                                    manual_count,
                                    map_input=eval_map,
                                )[0]
                            else:
                                foc = self._eval_points(
                                    evaluate,
                                    [xoc],
                                    executor,
                                    manual_count,
                                )[0]
                        if foc <= fr:
                            simplex[-1] = xoc
                            fvals[-1] = foc
                            continue

                    if fic is None:
                        if _supports_map_kw:
                            fic = self._eval_points(
                                evaluate,
                                [xic],
                                executor,
                                manual_count,
                                map_input=eval_map,
                            )[0]
                        else:
                            fic = self._eval_points(
                                evaluate,
                                [xic],
                                executor,
                                manual_count,
                            )[0]
                    if fic < fvals[-1]:
                        simplex[-1] = xic
                        fvals[-1] = fic
                        continue

                    new_points = [simplex[0]]
                    for p in simplex[1:]:
                        new_points.append(simplex[0] + self.sigma * (p - simplex[0]))
                    if _supports_map_kw:
                        new_f = self._eval_points(
                            evaluate,
                            new_points[1:],
                            executor,
                            manual_count,
                            map_input=eval_map,
                        )
                    else:
                        new_f = self._eval_points(
                            evaluate,
                            new_points[1:],
                            executor,
                            manual_count,
                        )
                    simplex = new_points
                    fvals = [fvals[0]] + list(new_f)
        except nm_impl.EvaluationBudgetExceeded:
            nm_impl.logger.info("Nelder-Mead stopped after reaching the evaluation budget")
        finally:
            self._clear_budget()

        if fvals:
            idx = int(np.argmin(fvals))
            best_x = simplex[idx]
            best_f = float(fvals[idx])
        else:
            best_eval_point, best_eval_value = self._get_best_evaluation()
            if best_eval_point is not None and best_eval_value is not None:
                best_x = best_eval_point
                best_f = float(best_eval_value)
            else:
                best_x = x0
                best_f = float("nan")

        best_x = np.asarray(best_x, dtype=float)
        if normalize and normalize_transform is not None:
            best_x = normalize_transform.from_unit(best_x)
        best_x = np.asarray(best_x, dtype=float).copy()
        self.finalize_history()
        result = nm_impl.OptResult(
            best_x=best_x,
            best_f=float(best_f),
            history=self.history,
            evaluations=self.evaluations,
            nfev=self.nfev,
        )
        return result


# ---------------------------------------------------------------------------
# Optimisation driver
# ---------------------------------------------------------------------------
def main() -> Tuple[np.ndarray, float]:
    space = DesignSpace(lower=LOWER, upper=UPPER, names=PARAMETER_NAMES)

    resume_simplex = _maybe_load_initial_simplex(space.dimension)
    optimize_options: dict[str, object] = {}
    if resume_simplex is not None:
        initial_simplex, simplex_values, path = resume_simplex
        optimize_options["initial_simplex"] = initial_simplex
        optimize_options["initial_simplex_values"] = simplex_values
        print(f"[opt] Using provided initial simplex from '{path}' (skipping re-evaluation)")

    nm = LoggingNelderMeadOptimizer(
        n_workers=N_WORKERS,
        memoize=True,
        parallel_poll_points=True,
        no_improve_thr=NO_IMPROV_THR,
        no_improv_break=NO_IMPROV_BREAK,
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

    print("[opt] Starting Nelder–Mead optimisation (resume)")
    print(
        f"[opt] max_evals={MAX_EVALS}, workers={N_WORKERS}, normalize=True, "
        f"memoize=True, simplex_log={LOG_SIMPLEX}"
    )
    print(
        f"[opt] no_improve_thr={NO_IMPROV_THR} (no_improv_break={NO_IMPROV_BREAK}), "
        f"resolution={RESOLUTION}"
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
