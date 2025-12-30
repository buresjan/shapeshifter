#!/usr/bin/env python3
"""Shared TCPC geometry helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

from meshgen.geometry import Geometry


def ensure_txt_suffix(name: str) -> str:
    """Force a .txt suffix."""
    if not name.endswith(".txt"):
        return f"{name}.txt"
    return name


def default_paths(project_root: Path, *, objective_kind: str = "tcpc") -> Tuple[Path, Path]:
    """Return default locations for the solver binary and data root."""
    binary_name = "sim_tcpc_tke" if objective_kind == "tke" else "sim_tcpc_2"
    binary = project_root / "submodules" / "tnl-lbm" / "build" / "sim_NSE" / binary_name
    data_root = project_root / "submodules" / "tnl-lbm" / "sim_NSE"
    return binary, data_root


def locate_solver_root(binary: Path) -> Path:
    """Return the tnl-lbm repository root for binary."""
    for parent in binary.resolve().parents:
        candidate = parent / "run_lbm_simulation.py"
        if candidate.is_file():
            return parent
    raise ValueError(
        f"Unable to locate tnl-lbm root from '{binary}'. Expected run_lbm_simulation.py nearby."
    )


def prepare_run_directory(project_root: Path, case_basename: str) -> Path:
    """Create a dedicated folder for solver artifacts."""
    runs_root = project_root / "tmp" / "junction_tcpc_runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    stem = Path(case_basename).stem or "case"
    sanitized = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    suffix = 0
    while True:
        candidate = runs_root / (
            f"{sanitized}_{timestamp}" if suffix == 0 else f"{sanitized}_{timestamp}_{suffix:02d}"
        )
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            suffix += 1
            continue


def generate_geometry(
    output_dir: Path,
    case_name: str | None,
    *,
    resolution: int,
    lower_angle: float,
    upper_angle: float,
    upper_flare: float,
    lower_flare: float,
    offset: float,
    num_processes: int,
) -> Tuple[Dict[str, Path], str]:
    """Create the junction geometry files and return their paths and base filename."""
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_faces = {"W", "E", "N", "S"}

    geom = Geometry(
        name="junction_2d",
        resolution=resolution,
        split=None,
        num_processes=num_processes,
        output_dir=str(output_dir),
        expected_in_outs=expected_faces,
        lower_angle=lower_angle,
        upper_angle=upper_angle,
        upper_flare=upper_flare,
        lower_flare=lower_flare,
        offset=offset,
    )

    final_case_name = ensure_txt_suffix(
        case_name if case_name is not None else f"{geom.name}_{geom.name_hash[:8]}.txt"
    )

    print(f"[meshgen] Generating voxel mesh (resolution={resolution})...")
    geom.generate_voxel_mesh()

    print(f"[meshgen] Exporting triplet with base '{final_case_name}' into {output_dir}...")
    geom.save_voxel_mesh_to_text(final_case_name)

    basename = Path(final_case_name).name
    files = {
        "geometry": output_dir / f"geom_{basename}",
        "dimensions": output_dir / f"dim_{basename}",
        "angle": output_dir / f"angle_{basename}",
        "values": output_dir / f"val_{basename}",
    }
    for label, path in files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Expected {label} file '{path}' was not created by meshgen"
            )
    return files, final_case_name


def stage_geometry(files: Dict[str, Path], data_root: Path) -> Dict[str, Path]:
    """Copy geometry support files into the sim_NSE data root."""
    staged_paths: Dict[str, Path] = {}
    for subdir, source in files.items():
        target_dir = "tmp" if subdir == "values" else subdir
        destination = data_root / target_dir / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"[stage] Copying {source} -> {destination}")
        shutil.copy2(source, destination)
        staged_paths[subdir] = destination
    return staged_paths


def run_simulation(
    binary: Path,
    resolution: int,
    case_name: str,
    data_root: Path,
    run_dir: Path,
) -> Tuple[Path, Path]:
    """Execute the solver binary while capturing logs on disk."""
    env = os.environ.copy()
    env["TNL_LBM_DATA_ROOT"] = str(data_root)
    command = [str(binary), str(resolution), case_name]
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    print(f"[solver] Launching: {' '.join(command)}")
    print(f"[solver] Working directory: {run_dir}")
    print(f"[solver] stdout log: {stdout_path}")
    print(f"[solver] stderr log: {stderr_path}")
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_file:
        result = subprocess.run(
            command,
            cwd=run_dir,
            env=env,
            text=True,
            stdout=stdout_file,
            stderr=stderr_file,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"{binary.name} exited with {result.returncode}. Inspect logs under '{run_dir}'."
        )
    return stdout_path, stderr_path


def collect_scalar(data_root: Path, case_name: str) -> Tuple[float, float, Path]:
    """Read the latest scalar sample from tmp/val_<case_name>."""
    val_path = data_root / "tmp" / f"val_{case_name}"
    if not val_path.exists():
        raise FileNotFoundError(f"Solver output file '{val_path}' is missing")

    last_line: str | None = None
    with val_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                last_line = line

    if last_line is None:
        raise ValueError(f"No data found in '{val_path}'")

    parts = last_line.split()
    if len(parts) < 2:
        raise ValueError(f"Unexpected data format in '{val_path}': '{last_line}'")

    try:
        time_value = float(parts[0])
        scalar_value = float(parts[1])
    except ValueError as exc:
        raise ValueError(
            f"Failed to parse numeric values from '{last_line}' in '{val_path}'"
        ) from exc

    return time_value, scalar_value, val_path
