#!/usr/bin/env python3
"""Generate a meshgen junction case, stage it, and run the TCPC solver."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from tcpc_common import (
    collect_scalar,
    default_paths,
    generate_geometry,
    locate_solver_root,
    prepare_run_directory,
    run_simulation,
    stage_geometry,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a meshgen junction case and run the TCPC solver.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--binary",
        type=Path,
        help="Path to the pre-built solver executable.",
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
        help="Voxel resolution for meshgen and lattice resolution for the solver.",
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
        "--objective-kind",
        choices=("tcpc", "tke"),
        default="tcpc",
        help="Select which solver binary to use by default.",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Preserve the meshgen workspace directory after running.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    project_root = Path(__file__).resolve().parents[1]
    default_binary, default_data_root = default_paths(project_root, objective_kind=args.objective_kind)

    binary = (args.binary or default_binary).resolve()
    workspace = (args.workspace or (project_root / "tmp" / "junction_tcpc")).resolve()

    if not binary.is_file():
        print(f"error: solver binary '{binary}' not found", file=sys.stderr)
        return 1

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
    except Exception as exc:
        print(f"error: mesh generation failed: {exc}", file=sys.stderr)
        return 1

    case_basename = Path(case_name).name
    run_dir = prepare_run_directory(project_root, case_basename)
    print(f"[run] Solver artifacts directory: {run_dir}")

    if args.data_root is not None:
        data_root = Path(args.data_root).resolve()
        try:
            (run_dir / "sim_NSE").symlink_to(data_root, target_is_directory=True)
        except FileExistsError:
            pass
    else:
        data_root = (run_dir / "sim_NSE").resolve()

    required_subdirs = ("geometry", "dimensions", "angle", "tmp")
    for subdir in required_subdirs:
        target_dir = data_root / subdir
        target_dir.mkdir(parents=True, exist_ok=True)

    staged_paths = stage_geometry(generated_files, data_root)

    try:
        locate_solver_root(binary)
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
