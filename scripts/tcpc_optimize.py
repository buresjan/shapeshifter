#!/usr/bin/env python3
"""Config-driven TCPC optimization runner."""

from __future__ import annotations

import argparse
import inspect
import math
import os
import threading
import multiprocessing
import uuid
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple, cast

import numpy as np

from optilb import DesignSpace, OptimizationProblem
from optilb.optimizers import MADSOptimizer, NelderMeadOptimizer
import optilb.optimizers.nelder_mead as nm_impl

from tcpc_config import load_config
import tcpc_objective


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a TCPC optimization from a config file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to config .py/.json")
    parser.add_argument(
        "--algorithm-label",
        help="Override algorithm label for logging/run naming.",
    )
    return parser.parse_args()


def _resolve_workers(optimizer_type: str, configured: Optional[int]) -> int:
    if configured is not None:
        return int(configured)
    if optimizer_type == "mads":
        return int(os.environ.get("OPT_MADS_WORKERS", os.environ.get("OPT_NM_WORKERS", "8")))
    return int(os.environ.get("OPT_NM_WORKERS", "8"))


def _env_bool(var: str, default: bool) -> bool:
    raw = os.environ.get(var)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_log_simplex(default: bool) -> bool:
    if "TCPC_LOG_SIMPLEX" in os.environ:
        return _env_bool("TCPC_LOG_SIMPLEX", default)
    return default


def _resolve_force_thread_pool(optimizer_cfg: dict) -> bool:
    if "OPTILB_FORCE_THREAD_POOL" in os.environ:
        return _env_bool("OPTILB_FORCE_THREAD_POOL", False)
    if "force_thread_pool" in optimizer_cfg:
        return bool(optimizer_cfg.get("force_thread_pool"))
    return True


def _apply_force_thread_pool(enabled: bool) -> None:
    if enabled:
        os.environ["OPTILB_FORCE_THREAD_POOL"] = "1"


def _normalize_config(raw: dict, *, algorithm_label_override: Optional[str]) -> dict:
    cfg = dict(raw)

    label = str(cfg.get("label") or "tcpc_run")
    algorithm_label = algorithm_label_override or cfg.get("algorithm_label") or label

    objective_kind = str(cfg.get("objective_kind") or "tcpc")
    resolution = int(cfg.get("resolution", 5))
    geometry_penalty = float(cfg.get("geometry_penalty", 1.0e9))
    max_evals = int(cfg.get("max_evals", 50))

    space_cfg = cfg.get("space") or {}
    names = tuple(
        space_cfg.get(
            "names",
            ("offset", "lower_angle", "upper_angle", "lower_flare", "upper_flare"),
        )
    )
    if "x0" not in space_cfg or "lower" not in space_cfg or "upper" not in space_cfg:
        raise ValueError("Config space must define x0, lower, and upper")

    optimizer_cfg = dict(cfg.get("optimizer") or {})
    optimizer_type = optimizer_cfg.get("type") or "nelder_mead"
    optimizer_cfg["type"] = optimizer_type
    optimizer_cfg.setdefault("normalize", True)
    optimizer_cfg.setdefault("parallel", True)
    if optimizer_type == "nelder_mead":
        optimizer_cfg.setdefault("force_thread_pool", True)
    optimizer_cfg.setdefault("verbose", True)

    solver_cfg = dict(cfg.get("solver") or {})
    split_cfg = dict(cfg.get("split") or {})
    submit_cfg = dict(cfg.get("submit") or {})
    extra_points_cfg = dict(cfg.get("extra_points") or {})

    return {
        "label": label,
        "algorithm_label": algorithm_label,
        "objective_kind": objective_kind,
        "resolution": resolution,
        "geometry_penalty": geometry_penalty,
        "max_evals": max_evals,
        "eval_log_root": cfg.get("eval_log_root"),
        "eval_log_shared": cfg.get("eval_log_shared"),
        "eval_log_shared_root": cfg.get("eval_log_shared_root"),
        "eval_log_shared_path": cfg.get("eval_log_shared_path"),
        "eval_log_shared_run": cfg.get("eval_log_shared_run"),
        "space": {
            "names": names,
            "x0": space_cfg["x0"],
            "lower": space_cfg["lower"],
            "upper": space_cfg["upper"],
        },
        "optimizer": optimizer_cfg,
        "solver": solver_cfg,
        "split": split_cfg,
        "submit": submit_cfg,
        "extra_points": extra_points_cfg,
        "run_tag": cfg.get("run_tag"),
        "mpi_accelerator": cfg.get("mpi_accelerator"),
    }


def _objective_settings_from_config(cfg: dict) -> dict:
    return {
        "label": cfg["label"],
        "algorithm_label": cfg["algorithm_label"],
        "objective_kind": cfg["objective_kind"],
        "resolution": cfg["resolution"],
        "parameter_names": cfg["space"]["names"],
        "geometry_penalty": cfg["geometry_penalty"],
        "solver": cfg["solver"],
        "split": cfg["split"],
        "run_tag": cfg.get("run_tag"),
        "eval_log_root": cfg.get("eval_log_root"),
        "eval_log_shared": cfg.get("eval_log_shared"),
        "eval_log_shared_root": cfg.get("eval_log_shared_root"),
        "eval_log_shared_path": cfg.get("eval_log_shared_path"),
        "eval_log_shared_run": cfg.get("eval_log_shared_run"),
        "mpi_accelerator": cfg.get("mpi_accelerator"),
    }


def _ensure_run_tag_env(cfg: dict) -> None:
    if cfg.get("run_tag") or os.environ.get("TCPC_RUN_TAG"):
        return
    os.environ["TCPC_RUN_TAG"] = f"run_{uuid.uuid4().hex[:8]}"


def _validate_space(lower: np.ndarray, upper: np.ndarray, x0: np.ndarray) -> None:
    if lower.shape != upper.shape:
        raise ValueError("LOWER/UPPER must have the same shape")
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise ValueError("LOWER/UPPER must be finite")
    if np.any(upper <= lower):
        raise ValueError("Each UPPER bound must be strictly greater than LOWER")
    if x0.shape != lower.shape:
        raise ValueError("X0 must match bound dimension")
    if np.any(x0 < lower) or np.any(x0 > upper):
        raise ValueError("X0 must lie within [LOWER, UPPER]")


def _maybe_load_initial_simplex(
    dimension: int,
    optimizer_cfg: dict,
) -> tuple[list[np.ndarray], list[float], Path] | None:
    if optimizer_cfg.get("initial_simplex") is not None:
        simplex_raw = optimizer_cfg["initial_simplex"]
        values_raw = optimizer_cfg.get("initial_simplex_values")
        if values_raw is None:
            raise ValueError("initial_simplex_values required with initial_simplex")
        simplex = [np.asarray(row, dtype=float) for row in simplex_raw]
        values = [float(v) for v in values_raw]
        return simplex, values, Path("inline")

    path_str = optimizer_cfg.get("simplex_npz")
    env_var = optimizer_cfg.get("simplex_npz_env")
    if not path_str and env_var:
        path_str = os.environ.get(str(env_var))
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
    """Nelder-Mead variant that prints simplex coordinates each iteration."""

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

    def optimize(
        self,
        objective,
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
        eval_map = None

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
        simplex: list[np.ndarray]
        if use_provided_simplex:
            simplex = [np.asarray(pt, dtype=float) for pt in provided_simplex]
        else:
            simplex = [x0]

        self.record(simplex[0], tag="start")

        penalised_counting = self._make_penalised(counted_objective, space, constraints)
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
                            if self._max_evals is not None and self._nfev >= self._max_evals:
                                self._budget_exhausted = True
                                raise nm_impl.EvaluationBudgetExceeded(self._max_evals)
                            self._nfev += 1
                            self._last_eval_point = arr.copy()
                            if self._max_evals is not None and self._nfev >= self._max_evals:
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
                    current_best = fvals[0]

                    if self.log_simplex:
                        self._log_simplex_state(it, simplex, fvals, normalize_transform)
                    self.record(simplex[0], tag=str(it))
                    if verbose:
                        nm_impl.logger.info("%d | best %.6f", it, current_best)

                    if early_stopper is None:
                        if best - current_best > tol:
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
                    if parallel and executor is not None and self.parallel_poll_points:
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


def _load_and_configure(config_path: Path, algorithm_label: Optional[str]) -> dict:
    raw = load_config(config_path)
    cfg = _normalize_config(raw, algorithm_label_override=algorithm_label)
    _ensure_run_tag_env(cfg)
    tcpc_objective.configure(_objective_settings_from_config(cfg))
    return cfg


def _cache_key(arr: np.ndarray, digits: Optional[int]) -> Tuple[float, ...]:
    if digits is None:
        return tuple(float(v) for v in arr)
    return tuple(round(float(v), digits) for v in arr)


def _objective_worker(
    arr: np.ndarray,
    eval_id: int,
    config_path: str,
    algorithm_label: str,
    queue: multiprocessing.Queue,
) -> None:
    try:
        cfg = _load_and_configure(Path(config_path), algorithm_label)
        value = float(
            tcpc_objective.objective(
                np.asarray(arr, dtype=float),
                algorithm=cfg["algorithm_label"],
                eval_id=eval_id,
            )
        )
        log_path = tcpc_objective.current_eval_log_path(cfg["algorithm_label"])
        queue.put(("ok", value, str(log_path) if log_path else None))
    except Exception as exc:
        queue.put(("err", str(exc), None))


def _objective_subprocess(
    x: np.ndarray,
    *,
    eval_id: int,
    config_path: Path,
    algorithm_label: str,
) -> tuple[float, Optional[str]]:
    ctx = (
        multiprocessing.get_context("fork")
        if "fork" in multiprocessing.get_all_start_methods()
        else multiprocessing.get_context("spawn")
    )
    q: multiprocessing.Queue = ctx.Queue()

    proc = ctx.Process(
        target=_objective_worker,
        args=(np.asarray(x, dtype=float), eval_id, str(config_path), algorithm_label, q),
    )
    proc.start()
    proc.join()

    if q.empty():
        return float("nan"), None
    status, payload, log_path = q.get()
    if status == "ok":
        return float(payload), log_path
    return float("nan"), None


def main() -> Tuple[np.ndarray, float]:
    args = _parse_args()
    cfg = _load_and_configure(args.config, args.algorithm_label)

    space_cfg = cfg["space"]
    names = tuple(space_cfg["names"])
    x0 = np.asarray(space_cfg["x0"], dtype=float)
    lower = np.asarray(space_cfg["lower"], dtype=float)
    upper = np.asarray(space_cfg["upper"], dtype=float)
    _validate_space(lower, upper, x0)

    space = DesignSpace(lower=lower, upper=upper, names=names)

    optimizer_cfg = cfg["optimizer"]
    optimizer_type = optimizer_cfg["type"]
    workers = _resolve_workers(optimizer_type, optimizer_cfg.get("n_workers"))
    max_evals = cfg["max_evals"]
    normalize = bool(optimizer_cfg.get("normalize", True))
    parallel = bool(optimizer_cfg.get("parallel", True))
    verbose = bool(optimizer_cfg.get("verbose", True))
    if parallel and workers < 2:
        raise ValueError(
            "parallel=True requires n_workers >= 2; set optimizer.n_workers or OPT_NM_WORKERS."
        )

    if optimizer_type == "nelder_mead":
        force_thread_pool = bool(parallel) and _resolve_force_thread_pool(optimizer_cfg)
        if parallel and not force_thread_pool:
            print(
                "[opt] Parallel tcpc objective requires thread pool; overriding to "
                "force_thread_pool=True",
                flush=True,
            )
            force_thread_pool = True
        _apply_force_thread_pool(force_thread_pool)
        log_simplex = _resolve_log_simplex(bool(optimizer_cfg.get("log_simplex", False)))
        nm_kwargs = {
            "n_workers": workers,
            "memoize": bool(optimizer_cfg.get("memoize", True)),
            "parallel_poll_points": bool(optimizer_cfg.get("parallel_poll_points", True)),
            "log_simplex": log_simplex,
        }
        if optimizer_cfg.get("no_improve_thr") is not None:
            nm_kwargs["no_improve_thr"] = float(optimizer_cfg.get("no_improve_thr"))
        if optimizer_cfg.get("no_improv_break") is not None:
            nm_kwargs["no_improv_break"] = int(optimizer_cfg.get("no_improv_break"))
        if optimizer_cfg.get("penalty") is not None:
            nm_kwargs["penalty"] = float(optimizer_cfg.get("penalty"))

        nm = LoggingNelderMeadOptimizer(**nm_kwargs)

        optimize_options: dict[str, object] = {}
        resume_simplex = _maybe_load_initial_simplex(space.dimension, optimizer_cfg)
        if resume_simplex is not None:
            initial_simplex, simplex_values, path = resume_simplex
            optimize_options["initial_simplex"] = initial_simplex
            optimize_options["initial_simplex_values"] = simplex_values
            print(f"[opt] Using provided initial simplex from '{path}'")

        def _objective_nm(arr: np.ndarray) -> float:
            return float(tcpc_objective.objective(arr, algorithm=cfg["algorithm_label"]))

        tol = optimizer_cfg.get("tol")
        if tol is None and optimizer_cfg.get("no_improve_thr") is not None:
            tol = optimizer_cfg.get("no_improve_thr")

        problem_kwargs = {
            "objective": _objective_nm,
            "space": space,
            "x0": x0,
            "optimizer": nm,
            "parallel": parallel,
            "normalize": normalize,
            "max_evals": max_evals,
            "verbose": verbose,
        }
        if tol is not None:
            problem_kwargs["tol"] = tol
        if optimize_options:
            problem_kwargs["optimize_options"] = optimize_options

        problem = OptimizationProblem(**problem_kwargs)

        print("[opt] Starting Nelder-Mead optimization")
        print(
            f"[opt] max_evals={max_evals}, workers={workers}, normalize={normalize}, "
            f"parallel={parallel}, force_threads={force_thread_pool}, "
            f"memoize={bool(optimizer_cfg.get('memoize', True))}, log_simplex={log_simplex}"
        )
        if optimizer_cfg.get("no_improve_thr") is not None:
            print(
                f"[opt] no_improve_thr={optimizer_cfg.get('no_improve_thr')} "
                f"(no_improv_break={optimizer_cfg.get('no_improv_break')})"
            )
    elif optimizer_type == "mads":
        if not MADSOptimizer.is_available():
            raise RuntimeError(
                "PyNomad (PyNomadBBO) is required for the MADS optimizer but is not installed."
            )

        cache_round = optimizer_cfg.get("cache_round")
        if cache_round is not None:
            cache_round = int(cache_round)

        cache: Dict[Tuple[float, ...], float] = {}
        cache_seed = optimizer_cfg.get("cache_seed") or []
        if cache_seed:
            for entry in cache_seed:
                key = _cache_key(np.asarray(entry["x"], dtype=float), cache_round)
                cache[key] = float(entry["value"])

        cache_lock = threading.Lock()
        use_subprocess = bool(optimizer_cfg.get("subprocess", True))

        def _memoized_objective(arr: np.ndarray) -> float:
            key = _cache_key(np.asarray(arr, dtype=float), cache_round)
            eval_id = tcpc_objective.next_eval_id(cfg["algorithm_label"])
            with cache_lock:
                cached = cache.get(key)
            if cached is not None:
                log_path = tcpc_objective.log_eval(cfg["algorithm_label"], eval_id, arr, cached)
                print(
                    f"[obj] eval {eval_id:04d} cached value={cached:.6g} for x={key} (log={log_path})",
                    flush=True,
                )
                return float(cached)

            if use_subprocess:
                value, log_path = _objective_subprocess(
                    arr,
                    eval_id=eval_id,
                    config_path=args.config,
                    algorithm_label=cfg["algorithm_label"],
                )
                if log_path:
                    tcpc_objective.set_eval_log_path(cfg["algorithm_label"], Path(log_path))
            else:
                value = float(
                    tcpc_objective.objective(
                        np.asarray(arr, dtype=float),
                        algorithm=cfg["algorithm_label"],
                        eval_id=eval_id,
                    )
                )
            if not math.isfinite(value):
                raise RuntimeError(f"Objective returned non-finite value {value} for x={key}")
            if value >= cfg["geometry_penalty"]:
                print(
                    f"[obj] constraint penalty triggered (value={value:.6g}) for x={key}",
                    flush=True,
                )
            with cache_lock:
                cache[key] = float(value)
            return float(value)

        mads = MADSOptimizer(n_workers=workers)
        problem = OptimizationProblem(
            objective=_memoized_objective,
            space=space,
            x0=x0,
            optimizer=mads,
            parallel=parallel,
            normalize=normalize,
            max_evals=max_evals,
            verbose=verbose,
        )

        print("[opt] Starting MADS optimization")
        print(
            f"[opt] max_evals={max_evals}, workers={workers}, normalize={normalize}, "
            f"parallel={parallel}, memoize=True"
        )
    else:
        raise ValueError(f"Unsupported optimizer type '{optimizer_type}'")

    if optimizer_cfg.get("print_bounds"):
        print("[opt] Start point:")
        for name, val in zip(space.names or (), x0):
            print(f"  - {name}: {float(val):.6g}")
        print("[opt] Bounds:")
        for name, lo, hi in zip(space.names or (), lower, upper):
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
            coords = ", ".join(f"{name}={float(val):.6g}" for name, val in zip(names, record.x))
            print(f"  - eval {idx:03d}: f={record.value:.6g}; {coords}")
    log_path = tcpc_objective.current_eval_log_path(cfg["algorithm_label"])
    if log_path is not None:
        print(f"[opt] Evaluation CSV log saved to: {log_path}")

    return best_x, best_f


if __name__ == "__main__":
    main()
