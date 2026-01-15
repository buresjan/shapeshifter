#!/usr/bin/env python3
"""Config-driven Fontan optimization runner."""

from __future__ import annotations

import argparse
import inspect
import math
import os
import threading
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple, cast

import numpy as np

from optilb import DesignSpace, OptimizationProblem
from optilb.optimizers import MADSOptimizer, NelderMeadOptimizer
import optilb.optimizers.nelder_mead as nm_impl

import fontan_objective
from fontan_config import load_config

_CONFIG_ENV_VAR = "FONTAN_CONFIG_PATH"
_ALGO_ENV_VAR = "FONTAN_ALGORITHM_LABEL"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Fontan optimization from a config file.",
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


def _resolve_force_thread_pool(optimizer_cfg: dict) -> bool:
    if "OPTILB_FORCE_THREAD_POOL" in os.environ:
        raw = os.environ.get("OPTILB_FORCE_THREAD_POOL", "")
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(optimizer_cfg.get("force_thread_pool", False))


def _apply_force_thread_pool(enabled: bool) -> None:
    if enabled:
        os.environ["OPTILB_FORCE_THREAD_POOL"] = "1"


def _normalize_optimizer_type(raw: str) -> str:
    norm = raw.strip().lower().replace("-", "_")
    if norm == "nelder_mead":
        return "nelder_mead"
    if norm == "mads":
        return "mads"
    raise ValueError(f"Unsupported optimizer type '{raw}'")


def _normalize_config(raw: dict, *, algorithm_label_override: Optional[str]) -> dict:
    cfg = dict(raw)

    label = str(cfg.get("label") or "fontan_run")
    algorithm_label = algorithm_label_override or cfg.get("algorithm_label") or label

    max_iter = int(cfg.get("max_iter", 20))
    max_evals_raw = cfg.get("max_evals")
    max_evals = int(max_evals_raw) if max_evals_raw is not None else None
    tol = cfg.get("tol")
    seed_raw = cfg.get("seed")
    seed = int(seed_raw) if seed_raw is not None else None
    geometry_penalty = float(cfg.get("geometry_penalty", 1.0e9))

    space_cfg = dict(cfg.get("space") or {})
    if "x0" not in space_cfg or "lower" not in space_cfg or "upper" not in space_cfg:
        raise ValueError("Config space must define x0, lower, and upper.")

    optimizer_cfg = dict(cfg.get("optimizer") or {})
    optimizer_type = _normalize_optimizer_type(str(optimizer_cfg.get("type") or "nelder_mead"))
    optimizer_cfg["type"] = optimizer_type
    optimizer_cfg.setdefault("normalize", True)
    optimizer_cfg.setdefault("parallel", False)
    optimizer_cfg.setdefault("verbose", True)

    if tol is None:
        tol = optimizer_cfg.get("tol")
    if tol is None and optimizer_cfg.get("no_improve_thr") is not None:
        tol = optimizer_cfg.get("no_improve_thr")
    if tol is None:
        tol = 1e-3

    solver_cfg = dict(cfg.get("solver") or {})
    objective_cfg = dict(cfg.get("objective") or {})
    submit_cfg = dict(cfg.get("submit") or {})

    return {
        "label": label,
        "algorithm_label": algorithm_label,
        "max_iter": max_iter,
        "max_evals": max_evals,
        "tol": float(tol),
        "seed": seed,
        "geometry_penalty": geometry_penalty,
        "space": {
            "names": tuple(space_cfg.get("names") or fontan_objective.DEFAULT_PARAM_NAMES),
            "x0": space_cfg["x0"],
            "lower": space_cfg["lower"],
            "upper": space_cfg["upper"],
        },
        "optimizer": optimizer_cfg,
        "solver": solver_cfg,
        "objective": objective_cfg,
        "submit": submit_cfg,
        "_config_path": cfg.get("_config_path"),
    }


def _load_and_configure(config_path: Path, algorithm_label: Optional[str]) -> dict:
    raw = load_config(config_path)
    cfg = _normalize_config(raw, algorithm_label_override=algorithm_label)
    fontan_objective.configure(_objective_settings_from_config(cfg))
    return cfg


def _ensure_objective_configured() -> None:
    if getattr(fontan_objective, "_CONFIGURED", False):
        return
    config_path = os.environ.get(_CONFIG_ENV_VAR)
    if not config_path:
        raise RuntimeError("Fontan objective not configured and no config env is set.")
    algorithm_label = os.environ.get(_ALGO_ENV_VAR)
    cfg = _load_and_configure(Path(config_path), algorithm_label)
    os.environ.setdefault(_ALGO_ENV_VAR, cfg["algorithm_label"])


def _objective_nm(arr: np.ndarray) -> float:
    _ensure_objective_configured()
    return float(fontan_objective.objective(np.asarray(arr, dtype=float)))


def _objective_settings_from_config(cfg: dict) -> dict:
    return {
        "label": cfg["label"],
        "algorithm_label": cfg["algorithm_label"],
        "parameter_names": cfg["space"]["names"],
        "geometry_penalty": cfg["geometry_penalty"],
        "solver": cfg["solver"],
        "objective": cfg["objective"],
    }


def _validate_space(lower: np.ndarray, upper: np.ndarray, x0: np.ndarray) -> None:
    if lower.shape != upper.shape:
        raise ValueError("LOWER/UPPER must have the same shape.")
    if x0.shape != lower.shape:
        raise ValueError("X0 must match bound dimension.")
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise ValueError("LOWER/UPPER must be finite.")
    if np.any(upper < lower):
        raise ValueError("Each UPPER bound must be >= LOWER.")
    if np.any(x0 < lower) or np.any(x0 > upper):
        raise ValueError("X0 must lie within [LOWER, UPPER].")


def _cache_key(x: np.ndarray, round_digits: int | None) -> Tuple[float, ...]:
    if round_digits is None:
        return tuple(float(value) for value in x)
    return tuple(float(value) for value in np.round(x, round_digits))


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
                    elif fr < fvals[0]:
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
                    else:
                        if fr < fvals[-1]:
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
                            else:
                                self._shrink(
                                    simplex,
                                    fvals,
                                    executor,
                                    manual_count,
                                    penalised_raw,
                                    eval_map,
                                )
                        else:
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
                            else:
                                self._shrink(
                                    simplex,
                                    fvals,
                                    executor,
                                    manual_count,
                                    penalised_raw,
                                    eval_map,
                                )
        except nm_impl.EvaluationBudgetExceeded as exc:
            best_idx = int(np.argmin(fvals)) if fvals else 0
            self._set_best_from_result(simplex[best_idx], fvals[best_idx])
            return nm_impl.OptResult(best=simplex[best_idx], best_f=fvals[best_idx], nfev=self._nfev, msg=str(exc))

        best_idx = int(np.argmin(fvals)) if fvals else 0
        self._set_best_from_result(simplex[best_idx], fvals[best_idx])
        return nm_impl.OptResult(
            best=simplex[best_idx],
            best_f=fvals[best_idx],
            nfev=self._nfev,
            msg="success",
        )


def _build_optimizer(optimizer_cfg: dict) -> tuple[object, bool, bool, bool]:
    optimizer_type = optimizer_cfg["type"]
    normalize = bool(optimizer_cfg.get("normalize", True))
    parallel = bool(optimizer_cfg.get("parallel", False))
    verbose = bool(optimizer_cfg.get("verbose", True))

    workers = _resolve_workers(optimizer_type, optimizer_cfg.get("n_workers"))
    if parallel and workers < 2:
        raise ValueError("parallel=True requires n_workers >= 2.")

    if optimizer_type == "nelder_mead":
        force_thread_pool = bool(parallel) and _resolve_force_thread_pool(optimizer_cfg)
        _apply_force_thread_pool(force_thread_pool)

        nm_kwargs: dict[str, object] = {"n_workers": workers}
        if optimizer_cfg.get("memoize") is not None:
            nm_kwargs["memoize"] = bool(optimizer_cfg.get("memoize"))
        if optimizer_cfg.get("parallel_poll_points") is not None:
            nm_kwargs["parallel_poll_points"] = bool(optimizer_cfg.get("parallel_poll_points"))
        for key in ("step", "alpha", "gamma", "beta", "delta", "sigma"):
            if optimizer_cfg.get(key) is not None:
                nm_kwargs[key] = float(optimizer_cfg.get(key))
        for key in ("no_improve_thr", "penalty"):
            if optimizer_cfg.get(key) is not None:
                nm_kwargs[key] = float(optimizer_cfg.get(key))
        if optimizer_cfg.get("no_improv_break") is not None:
            nm_kwargs["no_improv_break"] = int(optimizer_cfg.get("no_improv_break"))

        log_simplex = bool(optimizer_cfg.get("log_simplex", False))
        return (
            LoggingNelderMeadOptimizer(log_simplex=log_simplex, **nm_kwargs),
            normalize,
            parallel,
            verbose,
        )

    if optimizer_type == "mads":
        if not MADSOptimizer.is_available():
            raise RuntimeError(
                "PyNomad (PyNomadBBO) is required for the MADS optimizer but is not installed."
            )
        return MADSOptimizer(n_workers=workers), normalize, parallel, verbose

    raise ValueError(f"Unsupported optimizer type '{optimizer_type}'")


def main() -> int:
    args = _parse_args()
    cfg = _load_and_configure(args.config, args.algorithm_label)
    config_path = Path(cfg["_config_path"]) if cfg.get("_config_path") else args.config
    os.environ.setdefault(_CONFIG_ENV_VAR, str(config_path))
    os.environ.setdefault(_ALGO_ENV_VAR, cfg["algorithm_label"])

    space_cfg = cfg["space"]
    x0 = np.asarray(space_cfg["x0"], dtype=float)
    lower = np.asarray(space_cfg["lower"], dtype=float)
    upper = np.asarray(space_cfg["upper"], dtype=float)
    _validate_space(lower, upper, x0)

    optimizer, normalize, parallel, verbose = _build_optimizer(cfg["optimizer"])

    space = DesignSpace(lower=lower, upper=upper, names=space_cfg["names"])

    if cfg["optimizer"]["type"] == "mads":
        cache_round = cfg["optimizer"].get("cache_round")
        if cache_round is not None:
            cache_round = int(cache_round)

        cache: Dict[Tuple[float, ...], float] = {}
        cache_seed = cfg["optimizer"].get("cache_seed") or []
        if cache_seed:
            for entry in cache_seed:
                key = _cache_key(np.asarray(entry["x"], dtype=float), cache_round)
                cache[key] = float(entry["value"])

        cache_lock = threading.Lock()

        def _memoized_objective(arr: np.ndarray) -> float:
            key = _cache_key(np.asarray(arr, dtype=float), cache_round)
            with cache_lock:
                cached = cache.get(key)
            if cached is not None:
                print(
                    f"[obj] cached value={cached:.6g} for x={key}",
                    flush=True,
                )
                return float(cached)

            value = float(
                fontan_objective.objective(
                    np.asarray(arr, dtype=float),
                    algorithm=cfg["algorithm_label"],
                )
            )
            if not math.isfinite(value):
                raise RuntimeError(f"Objective returned non-finite value {value} for x={key}")
            with cache_lock:
                cache[key] = float(value)
            return float(value)

        objective_fn: Callable[[np.ndarray], float] = _memoized_objective
    else:
        objective_fn = _objective_nm

    problem = OptimizationProblem(
        objective=objective_fn,
        space=space,
        x0=x0,
        optimizer=optimizer,
        normalize=normalize,
        parallel=parallel,
        max_iter=cfg["max_iter"],
        max_evals=cfg["max_evals"],
        tol=cfg["tol"],
        seed=cfg["seed"],
        verbose=verbose,
    )

    print(f"[opt] config={config_path}")
    print(
        "[opt] Starting optimization",
        f"optimizer={cfg['optimizer']['type']}",
        f"max_iter={cfg['max_iter']}",
        f"max_evals={cfg['max_evals']}",
        f"tol={cfg['tol']}",
        f"parallel={parallel}",
        flush=True,
    )
    if cfg["optimizer"].get("print_bounds"):
        print("[opt] Start point:")
        for name, val in zip(space.names or (), x0):
            print(f"  - {name}: {float(val):.6g}")
        print("[opt] Bounds:")
        for name, lo, hi in zip(space.names or (), lower, upper):
            print(f"  - {name}: [{float(lo):.6g}, {float(hi):.6g}]")

    result = problem.run()

    best_x = np.asarray(result.best_x, dtype=float).reshape(-1)
    best_value = float(result.best_f) * (-1.0 if fontan_objective.MAXIMIZE else 1.0)
    print(f"best_x={best_x}")
    print(f"best_value={best_value:.6f}")
    print(f"nfev={result.nfev}")
    if problem.log:
        print(f"optimizer={problem.log.optimizer} runtime={problem.log.runtime:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
