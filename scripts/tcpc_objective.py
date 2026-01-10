#!/usr/bin/env python3
"""TCPC objective pipeline shared across optimization and extra-point runs."""

from __future__ import annotations

import csv
try:
    import fcntl
except ImportError:  # pragma: no cover - fcntl is Unix-only.
    fcntl = None
import hashlib
import importlib.util
import math
import os
import shutil
import subprocess
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from tcpc_common import (
    ensure_txt_suffix,
    generate_geometry,
    locate_solver_root,
    prepare_run_directory,
    stage_geometry,
)
from tcpc_config import PROJECT_ROOT


@dataclass(frozen=True)
class Paths:
    project_root: Path
    solver_binary: Path
    solver_root: Path
    run_tcpc_script: Path


_CONFIGURED = False
_PATHS: Paths | None = None

OBJECTIVE_KIND = "tcpc"
RESOLUTION = 5
PARAMETER_NAMES: tuple[str, ...] = (
    "offset",
    "lower_angle",
    "upper_angle",
    "lower_flare",
    "upper_flare",
)
GEOMETRY_PENALTY = 1.0e9
DEFAULT_ALGORITHM_LABEL = "run"
_RUN_TAG = ""

MPI_ACCELERATOR = "cuda"

SLURM_PARTITION: str | None = None
SLURM_GPUS: int | None = 1
SLURM_CPUS: int | None = 8
SLURM_MEM: str | None = "32G"
SLURM_WALLTIME = "20:00:00"
SLURM_POLL_INTERVAL = 60.0
SLURM_AVG_WINDOW = 1.0
SLURM_VERBOSE = False

TCPC_SPLIT_SCRIPT = PROJECT_ROOT / "scripts" / "tcpc_split_wrapper.py"
TCPC_SPLIT_PVPYTHON = "pvpython"
TCPC_SPLIT_TIME_INDEX = -1
TCPC_SPLIT_MIN_FRACTION = 0.25
TCPC_SPLIT_WRITE_VTP = False
TCPC_SPLIT_WRITE_DEBUG_POINTS = False

_EVAL_LOCK = threading.Lock()
_EVAL_COUNTERS: Dict[str, int] = {}
_EVAL_LOG_PATHS: Dict[str, Path] = {}
_EVAL_TIMESTAMP = ""
_EVAL_LOG_ROOT = PROJECT_ROOT / "tmp" / "junction_tcpc_logs"
_EVAL_SHARED_LOG_ENABLED = True
_EVAL_SHARED_LOG_ROOT = PROJECT_ROOT / "data" / "junction_tcpc_logs" / "shared"
_EVAL_SHARED_LOG_PATH: Path | None = None


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


def _sanitize_tag(raw: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in raw)
    return cleaned.strip("_-")


def _lock_file(handle) -> None:
    if fcntl is None:
        raise RuntimeError(
            "fcntl is required for shared eval log locking on this platform."
        )
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock_file(handle) -> None:
    if fcntl is None:
        raise RuntimeError(
            "fcntl is required for shared eval log locking on this platform."
        )
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _append_csv_row(
    path: Path,
    header: tuple[str, ...],
    row: tuple[object, ...],
    *,
    lock: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", newline="") as handle:
        if lock:
            _lock_file(handle)
        try:
            handle.seek(0, os.SEEK_END)
            new_file = handle.tell() == 0
            writer = csv.writer(handle)
            if new_file:
                writer.writerow(header)
            writer.writerow(row)
            handle.flush()
        finally:
            if lock:
                _unlock_file(handle)


def _resolve_shared_log_path(algorithm: str) -> Path:
    if _EVAL_SHARED_LOG_PATH is not None:
        return _EVAL_SHARED_LOG_PATH
    safe_label = _sanitize_tag(algorithm) or "run"
    return _EVAL_SHARED_LOG_ROOT / f"{safe_label}.csv"


def configure(settings: dict) -> None:
    """Configure global settings for the objective pipeline."""
    global _CONFIGURED
    global _PATHS
    global OBJECTIVE_KIND
    global RESOLUTION
    global PARAMETER_NAMES
    global GEOMETRY_PENALTY
    global DEFAULT_ALGORITHM_LABEL
    global _RUN_TAG
    global MPI_ACCELERATOR
    global SLURM_PARTITION
    global SLURM_GPUS
    global SLURM_CPUS
    global SLURM_MEM
    global SLURM_WALLTIME
    global SLURM_POLL_INTERVAL
    global SLURM_AVG_WINDOW
    global SLURM_VERBOSE
    global TCPC_SPLIT_SCRIPT
    global TCPC_SPLIT_PVPYTHON
    global TCPC_SPLIT_TIME_INDEX
    global TCPC_SPLIT_MIN_FRACTION
    global TCPC_SPLIT_WRITE_VTP
    global TCPC_SPLIT_WRITE_DEBUG_POINTS
    global _EVAL_COUNTERS
    global _EVAL_LOG_PATHS
    global _EVAL_TIMESTAMP
    global _EVAL_SHARED_LOG_ENABLED
    global _EVAL_SHARED_LOG_ROOT
    global _EVAL_SHARED_LOG_PATH
    global _EVAL_LOG_ROOT

    OBJECTIVE_KIND = str(settings.get("objective_kind", "tcpc"))

    solver_cfg = settings.get("solver", {}) or {}
    binary_path = solver_cfg.get("binary_path")
    binary_name = solver_cfg.get("binary_name")
    if not binary_name and not binary_path:
        binary_name = "sim_tcpc_tke" if OBJECTIVE_KIND == "tke" else "sim_tcpc_2"

    if binary_path:
        solver_binary = Path(binary_path)
    else:
        solver_binary = PROJECT_ROOT / "submodules" / "tnl-lbm" / "build" / "sim_NSE" / str(
            binary_name
        )

    if not solver_binary.is_file():
        raise FileNotFoundError(
            f"Solver binary not found at '{solver_binary}'. Build tnl-lbm first."
        )

    solver_root = locate_solver_root(solver_binary)
    run_tcpc_script = solver_root / "run_tcpc_simulation.py"
    if not run_tcpc_script.is_file():
        raise FileNotFoundError(
            f"run_tcpc_simulation.py not found at '{run_tcpc_script}'. Ensure tnl-lbm is present."
        )

    _PATHS = Paths(
        project_root=PROJECT_ROOT,
        solver_binary=solver_binary.resolve(),
        solver_root=solver_root,
        run_tcpc_script=run_tcpc_script,
    )

    RESOLUTION = int(settings.get("resolution", 5))
    PARAMETER_NAMES = tuple(
        settings.get(
            "parameter_names",
            ("offset", "lower_angle", "upper_angle", "lower_flare", "upper_flare"),
        )
    )
    GEOMETRY_PENALTY = float(settings.get("geometry_penalty", 1.0e9))
    DEFAULT_ALGORITHM_LABEL = str(
        settings.get("algorithm_label") or settings.get("label") or "run"
    )

    eval_log_root = settings.get("eval_log_root")
    if eval_log_root:
        eval_path = Path(eval_log_root)
        if not eval_path.is_absolute():
            eval_path = PROJECT_ROOT / eval_path
        _EVAL_LOG_ROOT = eval_path
    else:
        _EVAL_LOG_ROOT = PROJECT_ROOT / "tmp" / "junction_tcpc_logs"

    eval_log_shared = settings.get("eval_log_shared")
    if eval_log_shared is None:
        eval_log_shared = True
    _EVAL_SHARED_LOG_ENABLED = bool(eval_log_shared)
    if _EVAL_SHARED_LOG_ENABLED and fcntl is None:
        raise RuntimeError(
            "Shared eval logging requires fcntl; disable eval_log_shared on this platform."
        )

    shared_log_path = settings.get("eval_log_shared_path") or os.environ.get(
        "TCPC_EVAL_LOG_SHARED_PATH"
    )
    shared_log_root = settings.get("eval_log_shared_root") or os.environ.get(
        "TCPC_EVAL_LOG_SHARED_ROOT"
    )
    if shared_log_path and shared_log_root:
        raise ValueError(
            "Specify only one of eval_log_shared_path or eval_log_shared_root."
        )
    if shared_log_path:
        shared_path = Path(shared_log_path)
        if shared_path.is_dir():
            raise ValueError(
                f"eval_log_shared_path must be a file path, got directory '{shared_path}'."
            )
        if not shared_path.is_absolute():
            shared_path = PROJECT_ROOT / shared_path
        _EVAL_SHARED_LOG_PATH = shared_path
        _EVAL_SHARED_LOG_ROOT = shared_path.parent
    else:
        shared_root = (
            Path(shared_log_root)
            if shared_log_root
            else PROJECT_ROOT / "data" / "junction_tcpc_logs" / "shared"
        )
        if not shared_root.is_absolute():
            shared_root = PROJECT_ROOT / shared_root
        _EVAL_SHARED_LOG_ROOT = shared_root
        _EVAL_SHARED_LOG_PATH = None

    _EVAL_COUNTERS = {}
    _EVAL_LOG_PATHS = {}
    _EVAL_TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")

    run_tag_raw = settings.get("run_tag")
    if not run_tag_raw:
        run_tag_raw = os.environ.get("TCPC_RUN_TAG")
    if not run_tag_raw:
        run_tag_raw = f"run_{uuid.uuid4().hex[:8]}"
    run_tag = _sanitize_tag(str(run_tag_raw))
    _RUN_TAG = f"{run_tag}_" if run_tag else ""

    MPI_ACCELERATOR = (
        str(settings.get("mpi_accelerator"))
        if settings.get("mpi_accelerator")
        else os.environ.get("TCPC_MPI_ACCELERATOR")
        or os.environ.get("OMPI_MCA_accelerator")
        or "cuda"
    )
    os.environ["OMPI_MCA_accelerator"] = MPI_ACCELERATOR

    slurm_cfg = solver_cfg.get("slurm", {}) or {}
    SLURM_PARTITION = os.environ.get("TCPC_SLURM_PARTITION") or slurm_cfg.get("partition")
    SLURM_GPUS = _env_optional_int("TCPC_SLURM_GPUS", slurm_cfg.get("gpus", 1))
    SLURM_CPUS = _env_optional_int("TCPC_SLURM_CPUS", slurm_cfg.get("cpus", 8))
    slurm_mem_default = slurm_cfg.get("mem", "32G")
    if slurm_mem_default is not None and str(slurm_mem_default).strip() == "":
        slurm_mem_default = None
    SLURM_MEM = os.environ.get("TCPC_SLURM_MEM", slurm_mem_default)
    if SLURM_MEM is not None and str(SLURM_MEM).strip() == "":
        SLURM_MEM = None
    slurm_walltime_default = slurm_cfg.get("walltime") or "20:00:00"
    SLURM_WALLTIME = os.environ.get("TCPC_SLURM_WALLTIME", slurm_walltime_default)
    SLURM_POLL_INTERVAL = _env_float(
        "TCPC_SLURM_POLL_INTERVAL", slurm_cfg.get("poll_interval", 60.0)
    )
    SLURM_AVG_WINDOW = _env_float("TCPC_SLURM_AVG_WINDOW", slurm_cfg.get("avg_window", 1.0))
    SLURM_VERBOSE = _env_bool("TCPC_SLURM_VERBOSE", slurm_cfg.get("verbose", False))

    split_cfg = settings.get("split", {}) or {}
    split_default = split_cfg.get("script_path")
    if split_default is None:
        wrapper = PROJECT_ROOT / "scripts" / "tcpc_split_wrapper.py"
        split_default = wrapper if wrapper.is_file() else PROJECT_ROOT / "submodules" / "tnl-lbm" / "tcpc_split.py"
    split_env = os.environ.get("TCPC_SPLIT_SCRIPT")
    TCPC_SPLIT_SCRIPT = Path(split_env) if split_env else Path(split_default)

    default_pvpython = split_cfg.get("pvpython", "pvpython")
    TCPC_SPLIT_PVPYTHON = (
        os.environ.get("TCPC_SPLIT_PVPYTHON")
        or os.environ.get("PV_PYTHON")
        or str(default_pvpython)
    )
    split_time_default = split_cfg.get("time_index")
    if split_time_default is None:
        split_time_default = -1
    TCPC_SPLIT_TIME_INDEX = int(os.environ.get("TCPC_SPLIT_TIME_INDEX", split_time_default))

    split_fraction_default = split_cfg.get("min_fraction")
    if split_fraction_default is None:
        split_fraction_default = 0.25
    TCPC_SPLIT_MIN_FRACTION = float(
        os.environ.get("TCPC_SPLIT_MIN_FRACTION", split_fraction_default)
    )
    TCPC_SPLIT_WRITE_VTP = _env_bool(
        "TCPC_SPLIT_WRITE_VTP", split_cfg.get("write_vtp", False)
    )
    TCPC_SPLIT_WRITE_DEBUG_POINTS = _env_bool(
        "TCPC_SPLIT_WRITE_DEBUG_POINTS", split_cfg.get("write_debug_points", False)
    )

    _load_run_tcpc_module.cache_clear()
    _CONFIGURED = True


def _project_paths() -> Paths:
    if not _CONFIGURED or _PATHS is None:
        raise RuntimeError("tcpc_objective.configure() must be called before evaluating")
    return _PATHS


def _hash_params(x: np.ndarray) -> str:
    """Stable short hash for a parameter vector."""
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

    if not hasattr(module, "_codex_run_dirs_by_thread"):
        module._codex_run_dirs_by_thread = {}
    if not hasattr(module, "_codex_run_dir_lock"):
        module._codex_run_dir_lock = threading.Lock()

    if not getattr(module, "_codex_make_run_dir_patched", False):
        original_make_run_dir = getattr(module, "_make_run_dir")

        def _make_run_dir_with_staging(project_root: Path, filename: str) -> Path:
            run_dir = original_make_run_dir(project_root, filename)

            lock = getattr(module, "_codex_run_dir_lock")
            run_dir_map = getattr(module, "_codex_run_dirs_by_thread")
            with lock:
                queue = run_dir_map.setdefault(threading.get_ident(), [])
                queue.append(run_dir)

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

    if not getattr(module, "_codex_write_sbatch_patched", False):
        import shlex as _shlex

        def _write_sbatch_with_local_root(
            *,
            run_dir: Path,
            solver_binary: Path,
            resolution: int,
            filename: str,
            partition: str | None,
            gpus: int | None,
            cpus: int | None,
            mem: str | None,
            walltime: str,
            project_root: Path,
        ) -> Path:
            job_name = module._job_name(run_dir.name)
            lines = [
                "#!/bin/bash",
                f"#SBATCH --job-name={job_name}",
                "#SBATCH --output=slurm.out",
                "#SBATCH --error=slurm.err",
                f"#SBATCH --time={walltime}",
            ]
            if partition:
                lines.append(f"#SBATCH --partition={partition}")
            if gpus is not None:
                lines.append(f"#SBATCH --gpus={gpus}")
            if cpus is not None:
                lines.append(f"#SBATCH --cpus-per-task={cpus}")
            if mem:
                lines.append(f"#SBATCH --mem={mem}")

            data_root = run_dir / "sim_NSE"
            data_root.mkdir(parents=True, exist_ok=True)
            tmp_dir = data_root / "tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)

            solver_quoted = _shlex.quote(str(solver_binary.resolve()))
            filename_quoted = _shlex.quote(filename)
            data_root_quoted = _shlex.quote(str(data_root.resolve()))
            tmp_dir_rel = Path("sim_NSE") / "tmp"
            tmp_dir_quoted = _shlex.quote(str(tmp_dir_rel))
            result_basename = f"val_{Path(filename).name}"
            result_basename_quoted = _shlex.quote(result_basename)
            resolution_literal = str(resolution)

            body = (
                f"""set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

SOLVER_BINARY={solver_quoted}
INPUT_FILENAME={filename_quoted}
RESOLUTION={resolution_literal}
TMP_DIR={tmp_dir_quoted}
RESULT_BASENAME={result_basename_quoted}
DATA_ROOT={data_root_quoted}

export TNL_LBM_DATA_ROOT="$DATA_ROOT"

mkdir -p "$TMP_DIR"
rm -f "$TMP_DIR/$RESULT_BASENAME"

if [ ! -x "$SOLVER_BINARY" ]; then
    echo "Solver binary $SOLVER_BINARY is missing or not executable." >&2
    exit 3
fi

echo "sim_tcpc start $(date --iso-8601=seconds)" >&2
"$SOLVER_BINARY" "$RESOLUTION" "$INPUT_FILENAME"
echo "sim_tcpc end $(date --iso-8601=seconds)" >&2

RESULT_PATH="$TMP_DIR/$RESULT_BASENAME"
if [ ! -f "$RESULT_PATH" ]; then
    echo "Result file $RESULT_PATH not produced" >&2
    exit 2
fi
"""
            )
            script_text = "\n".join(lines + ["", body])
            sbatch_path = run_dir / "job.sbatch"
            sbatch_path.write_text(script_text, encoding="ascii")
            return sbatch_path

        module._write_sbatch = _write_sbatch_with_local_root  # type: ignore[attr-defined]
        module._codex_write_sbatch_patched = True

    return module


def _next_eval_id(algorithm: str) -> int:
    """Return a monotonically increasing evaluation id per algorithm name."""
    with _EVAL_LOCK:
        current = _EVAL_COUNTERS.get(algorithm, 0) + 1
        _EVAL_COUNTERS[algorithm] = current
        return current


def next_eval_id(algorithm: str) -> int:
    return _next_eval_id(algorithm)


def _log_eval(algorithm: str, eval_id: int, x: np.ndarray, value: float) -> Path:
    """Append an evaluation record to the per-run CSV log and return its path."""
    log_dir = _EVAL_LOG_ROOT
    log_dir.mkdir(parents=True, exist_ok=True)
    header = ("eval_id", "objective", *PARAMETER_NAMES, "timestamp_iso8601")
    row = (
        eval_id,
        float(value),
        *[float(v) for v in np.asarray(x, dtype=float)],
        datetime.now().isoformat(),
    )
    with _EVAL_LOCK:
        path = _EVAL_LOG_PATHS.get(algorithm)
        if path is None:
            run_label = f"{_RUN_TAG}{algorithm}" if _RUN_TAG else algorithm
            run_label = run_label.strip("_") or algorithm
            path = log_dir / f"{run_label}_{_EVAL_TIMESTAMP}.csv"
            _EVAL_LOG_PATHS[algorithm] = path

        _append_csv_row(path, header, row, lock=False)

    if _EVAL_SHARED_LOG_ENABLED:
        shared_path = _resolve_shared_log_path(algorithm)
        _append_csv_row(shared_path, header, row, lock=True)

    return path


def log_eval(algorithm: str, eval_id: int, x: np.ndarray, value: float) -> Path:
    return _log_eval(algorithm, eval_id, x, value)


def current_eval_log_path(algorithm: str) -> Optional[Path]:
    """Return the log path if at least one evaluation was recorded."""
    with _EVAL_LOCK:
        return _EVAL_LOG_PATHS.get(algorithm)


def set_eval_log_path(algorithm: str, path: Path) -> None:
    with _EVAL_LOCK:
        _EVAL_LOG_PATHS[algorithm] = path


def _pop_tracked_run_dir(tcpc_module) -> Optional[Path]:
    """Return the latest run directory registered for the current thread."""
    lock = getattr(tcpc_module, "_codex_run_dir_lock", None)
    run_dir_map = getattr(tcpc_module, "_codex_run_dirs_by_thread", None)
    if lock is None or run_dir_map is None:
        return None
    ident = threading.get_ident()
    with lock:
        queue = run_dir_map.get(ident)
        if not queue:
            return None
        run_dir = queue.pop(0)
        if not queue:
            run_dir_map.pop(ident, None)
        return run_dir


def _discard_failed_run_dir(tcpc_module) -> None:
    """Drop and delete the most recent run dir if a solver job failed."""
    failed_dir = _pop_tracked_run_dir(tcpc_module)
    if failed_dir is None:
        return
    try:
        shutil.rmtree(failed_dir, ignore_errors=True)
    except OSError:
        pass


def _resolve_pvpython_executable() -> str:
    """Return the pvpython executable path, raising if missing."""
    candidate = Path(TCPC_SPLIT_PVPYTHON)
    if candidate.is_file():
        return str(candidate)
    resolved = shutil.which(TCPC_SPLIT_PVPYTHON)
    if resolved:
        return resolved
    raise FileNotFoundError(
        f"pvpython executable '{TCPC_SPLIT_PVPYTHON}' not found. "
        "Set TCPC_SPLIT_PVPYTHON or PV_PYTHON to a valid binary."
    )


def _select_bp_dataset(results_dir: Path) -> Optional[Path]:
    """Choose the most suitable ADIOS2 dataset inside results_dir."""
    preferred_names = [
        "output_3D.bp",
        "output_3D",
    ]
    for name in preferred_names:
        candidate = results_dir / name
        if candidate.exists():
            return candidate

    bp_candidates = sorted(results_dir.glob("output_3D*.bp"))
    if bp_candidates:
        return bp_candidates[0]
    generic_candidates = sorted(results_dir.glob("*.bp"))
    if generic_candidates:
        return generic_candidates[0]

    return None


def _discover_bp_path(run_dir: Path, case_stem: str) -> Path:
    """Return the ADIOS2 dataset path for case_stem within run_dir."""
    expected = f"results_sim_tcpc_res{RESOLUTION:02d}_{case_stem}"
    search_roots = [run_dir, run_dir / "sim_NSE"]

    candidate_dirs: list[Path] = []
    seen: set[Path] = set()

    def _push(path: Path) -> None:
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError:
            return
        if not resolved.is_dir():
            return
        if resolved in seen:
            return
        seen.add(resolved)
        candidate_dirs.append(resolved)

    for root in search_roots:
        _push(root / expected)
        for entry in root.glob(f"{expected}*"):
            _push(entry)

    for candidate in candidate_dirs:
        bp_path = _select_bp_dataset(candidate)
        if bp_path is not None:
            return bp_path

    raise FileNotFoundError(f"Unable to locate ADIOS2 dataset for '{case_stem}' under '{run_dir}'")


def _invoke_tcpc_split(run_dir: Path, case_basename: str) -> Path:
    """Run tcpc_split and return the CSV path."""
    pvpython_bin = _resolve_pvpython_executable()
    case_stem = Path(case_basename).stem
    bp_path = _discover_bp_path(run_dir, case_stem)
    print(f"[split] Using ADIOS2 dataset '{bp_path}'", flush=True)
    output_dir = run_dir / "tcpc_split" / case_stem
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_name = f"{case_stem}_split.csv"
    cmd = [
        pvpython_bin,
        str(TCPC_SPLIT_SCRIPT),
        "--bp-file",
        str(bp_path),
        "--time-index",
        str(TCPC_SPLIT_TIME_INDEX),
        "--output-dir",
        str(output_dir),
        "--csv-basename",
        csv_name,
        "--quiet",
    ]
    cmd.append("--write-vtp" if TCPC_SPLIT_WRITE_VTP else "--no-write-vtp")
    cmd.append(
        "--write-debug-points" if TCPC_SPLIT_WRITE_DEBUG_POINTS else "--no-write-debug-points"
    )
    subprocess.run(cmd, check=True, cwd=run_dir)
    csv_path = output_dir / csv_name
    if not csv_path.is_file():
        raise FileNotFoundError(f"tcpc_split did not produce CSV '{csv_path}'")
    return csv_path


def _load_split_rows(csv_path: Path) -> list[Dict[str, float | str]]:
    """Parse the tcpc_split CSV rows into numeric dictionaries."""
    rows: list[Dict[str, float | str]] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row: Dict[str, float | str] = {}
            for key, value in raw.items():
                if key == "Source":
                    row[key] = value or ""
                    continue
                try:
                    row[key] = float(value) if value not in (None, "") else float("nan")
                except (TypeError, ValueError):
                    row[key] = float("nan")
            rows.append(row)
    if not rows:
        raise ValueError(f"TCPC split CSV '{csv_path}' contained no rows.")
    return rows


def _aggregate_split(rows: list[Dict[str, float | str]]) -> tuple[float, float]:
    """Return overall LPA/RPA fractions weighted by known streamlines."""
    total_known = 0.0
    total_lpa = 0.0
    total_rpa = 0.0
    for row in rows:
        known_raw = row.get("Known")
        lpa_raw = row.get("LPA")
        rpa_raw = row.get("RPA")
        known = known_raw if isinstance(known_raw, (int, float)) else float("nan")
        lpa = lpa_raw if isinstance(lpa_raw, (int, float)) else float("nan")
        rpa = rpa_raw if isinstance(rpa_raw, (int, float)) else float("nan")
        if math.isnan(known):
            continue
        total_known += max(0.0, known)
        if not math.isnan(lpa):
            total_lpa += max(0.0, lpa)
        if not math.isnan(rpa):
            total_rpa += max(0.0, rpa)
    if total_known <= 0.0:
        return float("nan"), float("nan")
    return total_lpa / total_known, total_rpa / total_known


def _run_tcpc_split_check(tcpc_module, case_basename: str) -> tuple[bool, float, float, Path]:
    """Execute tcpc_split and verify outlet fractions."""
    tcpc_run_dir = _pop_tracked_run_dir(tcpc_module)
    if tcpc_run_dir is None:
        raise RuntimeError("Unable to determine TCPC run directory for split evaluation.")
    csv_path = _invoke_tcpc_split(tcpc_run_dir, case_basename)
    rows = _load_split_rows(csv_path)
    for row in rows:
        source_raw = row.get("Source", "unknown")
        source = source_raw if isinstance(source_raw, str) else str(source_raw)
        frac_l = row.get("Frac_LPA")
        frac_r = row.get("Frac_RPA")
        frac_l_val = frac_l if isinstance(frac_l, (int, float)) else float("nan")
        frac_r_val = frac_r if isinstance(frac_r, (int, float)) else float("nan")
        print(
            f"[split] {source} fractions: LPA={frac_l_val:.3f} RPA={frac_r_val:.3f}",
            flush=True,
        )
    agg_lpa, agg_rpa = _aggregate_split(rows)
    print(f"[split] aggregated fractions: LPA={agg_lpa:.3f}, RPA={agg_rpa:.3f}", flush=True)
    ok = (
        not (math.isnan(agg_lpa) or math.isnan(agg_rpa))
        and agg_lpa >= TCPC_SPLIT_MIN_FRACTION
        and agg_rpa >= TCPC_SPLIT_MIN_FRACTION
    )
    return ok, agg_lpa, agg_rpa, csv_path


def objective(
    x: np.ndarray, *, algorithm: Optional[str] = None, eval_id: Optional[int] = None
) -> float:
    """Objective wrapper: generate geometry, run solver, return scalar."""
    algorithm_label = algorithm or DEFAULT_ALGORITHM_LABEL
    eval_id = eval_id if eval_id is not None else _next_eval_id(algorithm_label)
    p = _project_paths()

    offset, lower_angle, upper_angle, lower_flare, upper_flare = map(float, x)

    case_tag = _hash_params(x)
    algo_label = _sanitize_tag(algorithm_label) if algorithm_label else "run"
    case_stem = (
        f"{_RUN_TAG}{algo_label}_junction_{case_tag}"
        if _RUN_TAG
        else f"{algo_label}_junction_{case_tag}"
    )
    case_name = ensure_txt_suffix(f"{case_stem}.txt")
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

    result_value: float | None = None
    log_path: Optional[Path] = None

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
        result_value = float(GEOMETRY_PENALTY)
        log_path = _log_eval(algorithm_label, eval_id, x, result_value)
        print(
            f"[obj] returning penalty value={GEOMETRY_PENALTY:.6g} (eval {eval_id:04d}, log={log_path})",
            flush=True,
        )
        shutil.rmtree(workspace, ignore_errors=True)
        shutil.rmtree(run_dir, ignore_errors=True)
        return result_value
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
        try:
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
        except Exception as exc:
            _discard_failed_run_dir(tcpc_module)
            result_value = float(GEOMETRY_PENALTY)
            log_path = _log_eval(algorithm_label, eval_id, x, result_value)
            print(
                f"[obj] TCPC Slurm run failed for case '{generated_basename}': {exc}",
                flush=True,
            )
            print(
                f"[obj] returning penalty value={GEOMETRY_PENALTY:.6g} (eval {eval_id:04d}, log={log_path})",
                flush=True,
            )
            cleanup_geometry = True
            cleanup_run_dir = True
            return result_value
        if math.isnan(value):
            _discard_failed_run_dir(tcpc_module)
            result_value = float(GEOMETRY_PENALTY)
            log_path = _log_eval(algorithm_label, eval_id, x, result_value)
            print(
                f"[obj] TCPC Slurm run for case '{generated_basename}' returned NaN; "
                f"applying penalty (eval {eval_id:04d}, log={log_path})",
                flush=True,
            )
            cleanup_geometry = True
            cleanup_run_dir = True
            return result_value

        cleanup_run_dir = True
        try:
            split_ok, agg_lpa, agg_rpa, csv_path = _run_tcpc_split_check(
                tcpc_module, generated_basename
            )
        except Exception as exc:
            split_ok = False
            agg_lpa = float("nan")
            agg_rpa = float("nan")
            csv_path = None
            print(
                f"[obj] TCPC split evaluation failed for '{generated_basename}': {exc}",
                flush=True,
            )

        result_value = float(value)
        if not split_ok:
            print(
                f"[obj] TCPC split violation: aggregated LPA={agg_lpa:.3f}, "
                f"RPA={agg_rpa:.3f} (threshold={TCPC_SPLIT_MIN_FRACTION}, csv={csv_path})",
                flush=True,
            )
            result_value = float(GEOMETRY_PENALTY)

        print(
            f"[obj] completed case={generated_basename} raw={float(value):.6g} final={result_value:.6g}",
            flush=True,
        )
        log_path = _log_eval(algorithm_label, eval_id, x, result_value)
        print(
            f"[obj] eval {eval_id:04d} value={result_value:.6g} x={tuple(map(float, x))} (log={log_path})",
            flush=True,
        )

        cleanup_geometry = True
        cleanup_run_dir = True
        return result_value
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
