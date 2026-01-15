#!/usr/bin/env python3
"""Fontan objective pipeline shared across optimization runs."""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from fontan_config import PROJECT_ROOT, resolve_path
from run_fontan_simulation import run_fontan_simulation
from vef_meshgen_run_fontan import (
    EXPECTED_OUTS,
    Z_EXTENT,
    ensure_txt_suffix,
    load_mesh,
    save_labeled_triplet,
    stage_geometry,
    voxelize_with_pitch,
)


DEFAULT_PARAM_NAMES: tuple[str, ...] = (
    "bump1_amp",
    "bump2_amp",
    "size_scale",
    "straighten_strength",
    "offset_x",
    "offset_y",
)
SUPPORTED_PARAM_NAMES: tuple[str, ...] = DEFAULT_PARAM_NAMES

_CONFIGURED = False
_BASE_VEF_CONFIG: dict[str, Any] | None = None

PARAMETER_NAMES: tuple[str, ...] = DEFAULT_PARAM_NAMES
GEOMETRY_PENALTY = 1.0e9
DEFAULT_ALGORITHM_LABEL = "run"

VEF_CONFIG_PATH: Path | None = None
OUTPUT_ROOT = PROJECT_ROOT / "data" / "vef_fontan_opt" / "vef_outputs"
RUNS_ROOT = PROJECT_ROOT / "data" / "vef_fontan_opt" / "runs"

Z_VOXELS = 80
SIM_RESOLUTION = 80
KEEP_TEMP_FILES = False
MAXIMIZE = False

SOLVER_BINARY = PROJECT_ROOT / "submodules" / "tnl-lbm" / "build" / "sim_NSE" / "sim_fontan_sr"

SLURM_PARTITION: str | None = None
SLURM_GPUS: int | None = 1
SLURM_CPUS: int | None = 4
SLURM_MEM: str | None = "32G"
SLURM_WALLTIME = "20:00:00"
SLURM_POLL_INTERVAL = 30.0
SLURM_AVG_WINDOW: float | None = 1.0
SLURM_VERBOSE = False

_EVAL_LOCK = threading.Lock()
_EVAL_COUNTERS: Dict[str, int] = {}
_EVAL_LOG_ROOT = PROJECT_ROOT / "data" / "vef_fontan_opt" / "logs"


def _sanitize_tag(raw: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in raw.strip())
    return cleaned or "run"


def _next_eval_id(algorithm_label: str) -> int:
    with _EVAL_LOCK:
        current = _EVAL_COUNTERS.get(algorithm_label, 0) + 1
        _EVAL_COUNTERS[algorithm_label] = current
        return current


def _append_csv_row(path: Path, header: tuple[str, ...], row: tuple[object, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(header)
        writer.writerow(row)


def _log_eval(algorithm_label: str, eval_id: int, x: np.ndarray, value: float, run_dir: Path):
    safe_label = _sanitize_tag(algorithm_label)
    log_path = _EVAL_LOG_ROOT / f"{safe_label}.csv"
    header = ("eval_id", *PARAMETER_NAMES, "objective_value", "run_dir")
    row = (eval_id, *map(float, x), float(value), str(run_dir))
    _append_csv_row(log_path, header, row)
    return log_path


def _load_vef_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"VEF config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("VEF config must be a JSON object.")
    if "run_kwargs" not in config or not isinstance(config["run_kwargs"], dict):
        raise ValueError("VEF config must define a run_kwargs mapping.")
    return config


def _resolve_solver_binary(solver_cfg: dict) -> Path:
    binary_path = solver_cfg.get("binary_path")
    binary_name = solver_cfg.get("binary_name")
    if not binary_name and not binary_path:
        binary_name = "sim_fontan_sr"
    if binary_path:
        resolved = resolve_path(binary_path)
        return resolved if resolved is not None else Path(binary_path)
    return (
        PROJECT_ROOT / "submodules" / "tnl-lbm" / "build" / "sim_NSE" / str(binary_name)
    ).resolve()


def _apply_design_to_config(
    config: dict[str, Any],
    *,
    bump1_amp: float,
    bump2_amp: float,
    size_scale: float,
    straighten_strength: float,
    offset_xy: Sequence[float],
    output_dir: Path,
    repair_pitch: float,
    keep_temp_files: bool,
) -> None:
    run_kwargs = config.get("run_kwargs")
    if not isinstance(run_kwargs, dict):
        raise ValueError("Config run_kwargs must be a mapping.")

    bumps = run_kwargs.get("bumps")
    if not isinstance(bumps, list) or len(bumps) < 2:
        raise ValueError("Expected at least two bump entries in run_kwargs['bumps'].")
    if not isinstance(bumps[0], dict) or not isinstance(bumps[1], dict):
        raise ValueError("Bump entries must be JSON objects.")
    if len(offset_xy) != 2:
        raise ValueError("offset_xy must contain exactly two values.")

    bumps[0]["amp"] = float(bump1_amp)
    bumps[1]["amp"] = float(bump2_amp)
    run_kwargs["size_scale"] = float(size_scale)
    run_kwargs["straighten_strength"] = float(straighten_strength)
    run_kwargs["offset_xy"] = [float(offset_xy[0]), float(offset_xy[1])]
    run_kwargs["repair_pitch"] = float(repair_pitch)
    run_kwargs["output_dir"] = str(output_dir)
    run_kwargs["keep_temp_files"] = bool(keep_temp_files)


def _extract_default_run_params(config: dict[str, Any]) -> dict[str, float]:
    run_kwargs = config.get("run_kwargs")
    if not isinstance(run_kwargs, dict):
        raise ValueError("Config run_kwargs must be a mapping.")

    bumps = run_kwargs.get("bumps")
    if not isinstance(bumps, list) or len(bumps) < 2:
        raise ValueError("Expected at least two bump entries in run_kwargs['bumps'].")
    if not isinstance(bumps[0], dict) or not isinstance(bumps[1], dict):
        raise ValueError("Bump entries must be JSON objects.")

    offset_xy = run_kwargs.get("offset_xy")
    if not isinstance(offset_xy, (list, tuple)) or len(offset_xy) != 2:
        raise ValueError("run_kwargs['offset_xy'] must contain exactly two values.")

    size_scale = run_kwargs.get("size_scale")
    if size_scale is None:
        raise ValueError("run_kwargs['size_scale'] is required.")
    straighten_strength = run_kwargs.get("straighten_strength")
    if straighten_strength is None:
        raise ValueError("run_kwargs['straighten_strength'] is required.")

    if "amp" not in bumps[0] or "amp" not in bumps[1]:
        raise ValueError("run_kwargs['bumps'] entries must define 'amp' values.")

    return {
        "bump1_amp": float(bumps[0]["amp"]),
        "bump2_amp": float(bumps[1]["amp"]),
        "size_scale": float(size_scale),
        "straighten_strength": float(straighten_strength),
        "offset_x": float(offset_xy[0]),
        "offset_y": float(offset_xy[1]),
    }


def _prepare_run_dir(runs_root: Path, algorithm_label: str, eval_id: int) -> Path:
    safe_label = _sanitize_tag(algorithm_label)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = f"{safe_label}_eval_{eval_id:04d}_{timestamp}"
    suffix = 0
    while True:
        candidate = runs_root / (stem if suffix == 0 else f"{stem}_{suffix:02d}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            suffix += 1


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _voxelize_to_triplet(
    stl_path: Path,
    pitch: float,
    output_dir: Path,
    case_name: str,
) -> dict[str, Path]:
    mesh = load_mesh(stl_path)
    voxels = voxelize_with_pitch(stl_path, pitch, mesh=mesh)

    from meshgen.voxels import prepare_voxel_mesh_txt

    labeled = prepare_voxel_mesh_txt(
        voxels,
        expected_in_outs=EXPECTED_OUTS,
        num_type="int",
    )
    save_labeled_triplet(labeled, output_dir, case_name)

    files = {
        "geometry": output_dir / f"geom_{case_name}",
        "dimensions": output_dir / f"dim_{case_name}",
        "values": output_dir / f"val_{case_name}",
    }
    for label, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Expected {label} file '{path}' was not created.")
    return files


def configure(settings: dict) -> None:
    """Configure global settings for the objective pipeline."""
    global _CONFIGURED
    global _BASE_VEF_CONFIG
    global PARAMETER_NAMES
    global GEOMETRY_PENALTY
    global DEFAULT_ALGORITHM_LABEL
    global VEF_CONFIG_PATH
    global OUTPUT_ROOT
    global RUNS_ROOT
    global Z_VOXELS
    global SIM_RESOLUTION
    global KEEP_TEMP_FILES
    global MAXIMIZE
    global SOLVER_BINARY
    global SLURM_PARTITION
    global SLURM_GPUS
    global SLURM_CPUS
    global SLURM_MEM
    global SLURM_WALLTIME
    global SLURM_POLL_INTERVAL
    global SLURM_AVG_WINDOW
    global SLURM_VERBOSE

    raw_names = settings.get("parameter_names") or DEFAULT_PARAM_NAMES
    PARAMETER_NAMES = tuple(str(name) for name in raw_names)
    if not PARAMETER_NAMES:
        raise ValueError("parameter_names must contain at least one entry.")
    unknown = [name for name in PARAMETER_NAMES if name not in SUPPORTED_PARAM_NAMES]
    if unknown:
        raise ValueError(
            f"Unsupported parameter_names {unknown}; allowed: {SUPPORTED_PARAM_NAMES}."
        )
    if len(set(PARAMETER_NAMES)) != len(PARAMETER_NAMES):
        raise ValueError("parameter_names must contain unique entries.")

    GEOMETRY_PENALTY = float(settings.get("geometry_penalty", GEOMETRY_PENALTY))
    DEFAULT_ALGORITHM_LABEL = str(settings.get("algorithm_label") or DEFAULT_ALGORITHM_LABEL)

    objective_cfg = settings.get("objective", {}) or {}
    solver_cfg = settings.get("solver", {}) or {}
    slurm_cfg = solver_cfg.get("slurm", {}) or {}

    vef_config_path = resolve_path(objective_cfg.get("vef_config_path"))
    if vef_config_path is None:
        raise ValueError("objective.vef_config_path is required.")
    VEF_CONFIG_PATH = vef_config_path

    output_root = resolve_path(objective_cfg.get("output_root"))
    runs_root = resolve_path(objective_cfg.get("runs_root"))
    if output_root is not None:
        OUTPUT_ROOT = output_root
    if runs_root is not None:
        RUNS_ROOT = runs_root

    Z_VOXELS = int(objective_cfg.get("z_voxels", Z_VOXELS))
    SIM_RESOLUTION = int(objective_cfg.get("sim_resolution", Z_VOXELS))
    KEEP_TEMP_FILES = bool(objective_cfg.get("keep_temp_files", KEEP_TEMP_FILES))
    MAXIMIZE = bool(objective_cfg.get("maximize", MAXIMIZE))

    solver_binary = _resolve_solver_binary(solver_cfg)
    if not solver_binary.is_file():
        raise FileNotFoundError(f"Solver binary not found: {solver_binary}")
    if not os.access(solver_binary, os.X_OK):
        raise PermissionError(f"Solver binary is not executable: {solver_binary}")
    SOLVER_BINARY = solver_binary

    SLURM_PARTITION = slurm_cfg.get("partition")
    SLURM_GPUS = slurm_cfg.get("gpus", SLURM_GPUS)
    SLURM_CPUS = slurm_cfg.get("cpus", SLURM_CPUS)
    SLURM_MEM = slurm_cfg.get("mem", SLURM_MEM)
    SLURM_WALLTIME = slurm_cfg.get("walltime", SLURM_WALLTIME)
    SLURM_POLL_INTERVAL = float(slurm_cfg.get("poll_interval", SLURM_POLL_INTERVAL))
    SLURM_AVG_WINDOW = slurm_cfg.get("avg_window", SLURM_AVG_WINDOW)
    SLURM_VERBOSE = bool(slurm_cfg.get("verbose", SLURM_VERBOSE))

    if not VEF_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"VEF config not found: {VEF_CONFIG_PATH}")

    if str(VEF_ROOT := (PROJECT_ROOT / "submodules" / "meshgen" / "vascular_encoding_framework")) not in sys.path:
        sys.path.insert(0, str(VEF_ROOT))

    _BASE_VEF_CONFIG = _load_vef_config(VEF_CONFIG_PATH)
    _CONFIGURED = True


def objective(
    x: np.ndarray, *, algorithm: Optional[str] = None, eval_id: Optional[int] = None
) -> float:
    """Objective wrapper: build geometry, run solver, return scalar objective."""
    if not _CONFIGURED:
        raise RuntimeError("fontan_objective.configure() must be called before objective().")
    if _BASE_VEF_CONFIG is None or VEF_CONFIG_PATH is None:
        raise RuntimeError("VEF config was not loaded; call configure() first.")

    algorithm_label = algorithm or DEFAULT_ALGORITHM_LABEL
    eval_id = eval_id if eval_id is not None else _next_eval_id(algorithm_label)
    params = np.asarray(x, dtype=float).reshape(-1)
    if params.shape[0] != len(PARAMETER_NAMES):
        raise ValueError(
            f"Expected {len(PARAMETER_NAMES)} parameters, got {params.shape[0]}."
        )

    params_map = dict(zip(PARAMETER_NAMES, params.tolist()))
    pitch = Z_EXTENT / float(Z_VOXELS)

    run_dir = _prepare_run_dir(RUNS_ROOT, algorithm_label, eval_id)
    vef_output_dir = OUTPUT_ROOT / run_dir.name
    vef_output_dir.mkdir(parents=True, exist_ok=True)
    geometry_workdir = run_dir / "geometry_work"
    geometry_workdir.mkdir(parents=True, exist_ok=True)

    config = copy.deepcopy(_BASE_VEF_CONFIG)
    defaults = _extract_default_run_params(config)
    bump1_amp = params_map.get("bump1_amp", defaults["bump1_amp"])
    bump2_amp = params_map.get("bump2_amp", defaults["bump2_amp"])
    size_scale = params_map.get("size_scale", defaults["size_scale"])
    straighten_strength = params_map.get("straighten_strength", defaults["straighten_strength"])
    offset_x = params_map.get("offset_x", defaults["offset_x"])
    offset_y = params_map.get("offset_y", defaults["offset_y"])
    _apply_design_to_config(
        config,
        bump1_amp=bump1_amp,
        bump2_amp=bump2_amp,
        size_scale=size_scale,
        straighten_strength=straighten_strength,
        offset_xy=(offset_x, offset_y),
        output_dir=vef_output_dir,
        repair_pitch=pitch,
        keep_temp_files=KEEP_TEMP_FILES,
    )

    _write_json(
        run_dir / "params.json",
        {
            "eval_index": eval_id,
            "parameters": dict(zip(PARAMETER_NAMES, params.tolist())),
            "pitch": pitch,
        },
    )
    _write_json(run_dir / "vef_config.json", config)

    try:
        from pipeline.vef_pipeline import run_from_config

        final_path, uid = run_from_config(config, config_path=VEF_CONFIG_PATH)
        final_path = Path(final_path)
        if not final_path.is_file():
            raise FileNotFoundError(f"VEF pipeline did not produce STL: {final_path}")

        case_name = ensure_txt_suffix(f"vef_{uid}.txt")
        files = _voxelize_to_triplet(final_path, pitch, geometry_workdir, case_name)

        data_root = run_dir / "sim_NSE"
        data_root.mkdir(parents=True, exist_ok=True)
        stage_geometry(files, data_root)

        scalar_value, time_value, val_path, slurm_state, used_fallback = run_fontan_simulation(
            case_name,
            resolution=SIM_RESOLUTION,
            run_dir=run_dir,
            data_root=data_root,
            binary_path=SOLVER_BINARY,
            partition=SLURM_PARTITION,
            gpus=SLURM_GPUS,
            cpus=SLURM_CPUS,
            mem=SLURM_MEM,
            walltime=SLURM_WALLTIME,
            poll_interval=SLURM_POLL_INTERVAL,
            default_on_failure=GEOMETRY_PENALTY,
            avg_window=SLURM_AVG_WINDOW,
            verbose=SLURM_VERBOSE,
        )
    except Exception as exc:
        result_value = float(GEOMETRY_PENALTY)
        log_path = _log_eval(algorithm_label, eval_id, params, result_value, run_dir)
        print(
            f"[obj] eval {eval_id:04d} failed ({exc}); returning penalty={GEOMETRY_PENALTY:.6g} "
            f"(log={log_path})",
            flush=True,
        )
        return result_value

    if used_fallback:
        result_value = float(GEOMETRY_PENALTY)
        log_path = _log_eval(algorithm_label, eval_id, params, result_value, run_dir)
        print(
            f"[obj] eval {eval_id:04d} slurm fallback (state={slurm_state}); "
            f"returning penalty={GEOMETRY_PENALTY:.6g} (log={log_path})",
            flush=True,
        )
        return result_value

    if not math.isfinite(scalar_value):
        result_value = float(GEOMETRY_PENALTY)
        log_path = _log_eval(algorithm_label, eval_id, params, result_value, run_dir)
        print(
            f"[obj] eval {eval_id:04d} non-finite value; returning penalty={GEOMETRY_PENALTY:.6g} "
            f"(log={log_path})",
            flush=True,
        )
        return result_value

    objective_value = -float(scalar_value) if MAXIMIZE else float(scalar_value)
    _write_json(
        run_dir / "result.json",
        {
            "uid": uid,
            "case_name": case_name,
            "time_value": time_value,
            "scalar_value": scalar_value,
            "objective_value": objective_value,
            "val_path": str(val_path),
            "vef_stl": str(final_path),
            "slurm_state": slurm_state,
        },
    )
    log_path = _log_eval(algorithm_label, eval_id, params, objective_value, run_dir)
    print(
        f"[obj] eval {eval_id:04d} value={objective_value:.6g} x={tuple(map(float, params))} "
        f"(log={log_path})",
        flush=True,
    )
    return objective_value
