#!/usr/bin/env python3
"""Generate a meshgen junction case, stage it for sim_tcpc, and collect the solver scalar.

This helper mirrors ``submodules/meshgen/examples/junction_2d_visualize.py`` but skips Mayavi
visualization. The script:

1. Builds a 3D lattice for the TCPC solver using the junction template in meshgen.
2. Copies the generated ``geom_``, ``dim_``, and ``angle_`` files into the
   ``sim_NSE`` data directories expected by ``sim_tcpc``.
3. Runs the pre-built ``sim_tcpc`` executable and parses the scalar objective
   written to ``sim_NSE/tmp/val_<case>.txt``.

Usage:
  python playground/run_junction_tcpc.py --binary submodules/tnl-lbm/build/sim_NSE/sim_tcpc

The executable must already exist, typically at
``submodules/tnl-lbm/build/sim_NSE/sim_tcpc``. No geometry artefacts are kept
under version control; everything is written to temporary folders.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

from meshgen.geometry import Geometry


def ensure_txt_suffix(name: str) -> str:
    """Force a ``.txt`` suffix."""
    if not name.endswith(".txt"):
        return f"{name}.txt"
    return name


def default_paths(project_root: Path) -> Tuple[Path, Path]:
    """Return default locations for the solver binary and data root."""
    binary = project_root / "submodules" / "tnl-lbm" / "build" / "sim_NSE" / "sim_tcpc"
    data_root = project_root / "submodules" / "tnl-lbm" / "sim_NSE"
    return binary, data_root


def locate_solver_root(binary: Path) -> Path:
    """Return the ``tnl-lbm`` repository root for ``binary``."""
    for parent in binary.resolve().parents:
        candidate = parent / "run_lbm_simulation.py"
        if candidate.is_file():
            return parent
    raise ValueError(
        f"Unable to locate tnl-lbm root from '{binary}'. Expected run_lbm_simulation.py nearby."
    )


def prepare_run_directory(project_root: Path, case_basename: str) -> Path:
    """Create a dedicated folder for solver artefacts, mimicking run_lbm_simulation."""
    runs_root = project_root / "tmp" / "junction_tcpc_runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    stem = Path(case_basename).stem or "case"
    sanitized = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = runs_root / f"{sanitized}_{timestamp}"

    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = runs_root / f"{sanitized}_{timestamp}_{suffix:02d}"

    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def generate_geometry(output_dir: Path, case_name: str | None, *, resolution: int, lower_angle: float,
                      upper_angle: float, upper_flare: float, lower_flare: float,
                      offset: float, num_processes: int) -> Tuple[Dict[str, Path], str]:
    """Create the junction geometry files and return their paths and base filename."""
    output_dir.mkdir(parents=True, exist_ok=True)

    geom = Geometry(
        name="junction_2d",
        resolution=resolution,
        split=None,
        num_processes=num_processes,
        output_dir=str(output_dir),
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
    """Copy geometry support files into the ``sim_NSE`` data root."""
    staged_paths: Dict[str, Path] = {}
    for subdir, source in files.items():
        # ``val_`` lives in tmp/
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
    solver_root: Path,
) -> Tuple[Path, Path]:
    """Execute the solver binary while capturing logs on disk."""
    env = os.environ.copy()
    env["TNL_LBM_DATA_ROOT"] = str(data_root)
    command = [str(binary), str(resolution), case_name]
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    print(f"[solver] Launching: {' '.join(command)}")
    print(f"[solver] Working directory: {solver_root}")
    print(f"[solver] stdout log: {stdout_path}")
    print(f"[solver] stderr log: {stderr_path}")
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_file:
        result = subprocess.run(
            command,
            cwd=solver_root,
            env=env,
            text=True,
            stdout=stdout_file,
            stderr=stderr_file,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"sim_tcpc exited with {result.returncode}. Inspect logs under '{run_dir}'."
        )
    return stdout_path, stderr_path


def collect_scalar(data_root: Path, case_name: str) -> Tuple[float, float, Path]:
    """Read the latest scalar sample from ``tmp/val_<case_name>``."""
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a meshgen junction case and run tnl-lbm sim_tcpc.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--binary",
        type=Path,
        help="Path to the pre-built sim_tcpc executable.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Root containing geometry/dimensions/angle directories (sim_NSE).",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        help="Temporary folder for meshgen outputs.",
    )
    parser.add_argument(
        "--case-name",
        help="Optional explicit case filename (with or without .txt).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=5,
        help="Voxel resolution for meshgen and lattice resolution for sim_tcpc.",
    )
    parser.add_argument(
        "--lower-angle",
        type=float,
        default=0.0,
        help="Lower branch inflow angle in degrees.",
    )
    parser.add_argument(
        "--upper-angle",
        type=float,
        default=-10.0,
        help="Upper branch inflow angle in degrees.",
    )
    parser.add_argument(
        "--lower-flare",
        type=float,
        default=0.001,
        help="Lower branch flare in meters.",
    )
    parser.add_argument(
        "--upper-flare",
        type=float,
        default=0.001,
        help="Upper branch flare in meters.",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=0.02,
        help="Branch separation offset in meters.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Number of processes for meshgen voxelization.",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Preserve the meshgen workspace directory after running.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]
    default_binary, default_data_root = default_paths(project_root)

    binary = (args.binary or default_binary).resolve()
    data_root = (args.data_root or default_data_root).resolve()
    workspace = (args.workspace or (project_root / "tmp" / "junction_tcpc")).resolve()

    if not binary.is_file():
        print(f"error: solver binary '{binary}' not found", file=sys.stderr)
        return 1

    required_subdirs = ("geometry", "dimensions", "angle", "tmp")
    for subdir in required_subdirs:
        target_dir = data_root / subdir
        target_dir.mkdir(parents=True, exist_ok=True)

    try:
        generated_files, case_name = generate_geometry(
            workspace,
            args.case_name,
            resolution=args.resolution,
            lower_angle=args.lower_angle,
            upper_angle=args.upper_angle,
            upper_flare=args.upper_flare,
            lower_flare=args.lower_flare,
            offset=args.offset,
            num_processes=args.processes,
        )
    except Exception as exc:  # mesh generation must fail loudly
        print(f"error: mesh generation failed: {exc}", file=sys.stderr)
        return 1

    staged_paths = stage_geometry(generated_files, data_root)
    case_basename = Path(case_name).name
    run_dir = prepare_run_directory(project_root, case_basename)
    print(f"[run] Solver artefacts directory: {run_dir}")

    try:
        solver_root = locate_solver_root(binary)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    try:
        stdout_path, stderr_path = run_simulation(
            binary,
            args.resolution,
            case_basename,
            data_root,
            run_dir,
            solver_root,
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(f"[solver] stdout log: {stdout_path}", file=sys.stderr)
        print(f"[solver] stderr log: {stderr_path}", file=sys.stderr)
        return 1

    try:
        time_value, scalar_value, val_path = collect_scalar(data_root, case_basename)
    except Exception as exc:
        print(f"error: failed to collect scalar output: {exc}", file=sys.stderr)
        return 1

    print(f"[result] latest sample at t={time_value}: {scalar_value}")
    print(f"[result] source file: {val_path}")

    if not args.keep_artifacts:
        try:
            shutil.rmtree(workspace)
            print(f"[cleanup] Removed workspace {workspace}")
        except OSError as exc:
            print(
                f"[cleanup] warning: failed to remove workspace {workspace}: {exc}",
                file=sys.stderr,
            )

    print("[stage] Staged files:")
    for label, path in staged_paths.items():
        print(f"  {label}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
