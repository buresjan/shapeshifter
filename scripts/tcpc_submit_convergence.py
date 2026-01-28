#!/usr/bin/env python3
"""Submit a convergence sweep for a TCPC point via Slurm."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from tcpc_config import PROJECT_ROOT, resolve_path


DEFAULT_POINTS_CSV = Path("configs/tcpc/points/convergence_x0.csv")
DEFAULT_BINARY = Path("submodules/tnl-lbm/build/sim_NSE/sim_tcpc_combined")
DEFAULT_RESOLUTIONS = (3, 4, 5, 6, 7, 8)

# NOTE: Resolution 7 memory is assumed; adjust via --mem-by-resolution if needed.
DEFAULT_MEM_BY_RES = {
    3: "8G",
    4: "8G",
    5: "16G",
    6: "32G",
    7: "32G",
    8: "60G",
}
DEFAULT_CPUS_BY_RES = {
    3: 4,
    4: 4,
    5: 4,
    6: 8,
    7: 8,
    8: 8,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit a resolution convergence sweep using tcpc_submit_best_cases.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--points-csv",
        type=Path,
        default=DEFAULT_POINTS_CSV,
        help="CSV with case_label + 5 parameters.",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=DEFAULT_BINARY,
        help="Path to sim_tcpc_combined (or compatible) binary.",
    )
    parser.add_argument(
        "--resolutions",
        default=",".join(str(value) for value in DEFAULT_RESOLUTIONS),
        help="Comma-separated list of resolutions to run.",
    )
    parser.add_argument("--walltime", default="168:00:00", help="Slurm walltime.")
    parser.add_argument("--gpus", type=int, help="Slurm GPU count.")
    parser.add_argument(
        "--gpu-mem",
        help="Requested GPU memory (for legend only; does not affect Slurm allocation).",
    )
    parser.add_argument("--processes", type=int, default=1, help="Meshgen voxel processes.")
    parser.add_argument("--partition", help="Slurm partition.")
    parser.add_argument("--constraint", help="Slurm constraint (e.g. GPU type).")
    parser.add_argument("--gres", help="Slurm GRES override (e.g. gpu:a100:1).")
    parser.add_argument(
        "--job-name-prefix",
        default="tcpc-conv",
        help="Prefix for Slurm job names (resolution suffix is added).",
    )
    parser.add_argument(
        "--case-prefix",
        default="conv",
        help="Prefix used for generated case filenames (resolution suffix is added).",
    )
    parser.add_argument(
        "--legend",
        type=Path,
        help="Legend CSV output path (default under data/junction_tcpc_logs/convergence).",
    )
    parser.add_argument(
        "--mem-by-resolution",
        help="Override memory mapping, e.g. '3=8G,4=8G,5=16G'.",
    )
    parser.add_argument(
        "--gpu-mem-by-resolution",
        help="Override GPU memory mapping for legend, e.g. '7=12G,8=24G'.",
    )
    parser.add_argument(
        "--cpus-by-resolution",
        help="Override CPU mapping, e.g. '3=4,4=4,5=4'.",
    )
    parser.add_argument(
        "--constraint-by-resolution",
        help="Override constraint mapping, e.g. '7=rtx4090,8=a100'.",
    )
    parser.add_argument(
        "--gres-by-resolution",
        help="Override GRES mapping, e.g. '7=gpu:rtx4090:1,8=gpu:a100:1'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write sbatch scripts but do not submit.",
    )
    return parser.parse_args()


def _parse_resolutions(text: str) -> list[int]:
    resolutions: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            resolutions.append(int(part))
        except ValueError as exc:
            raise ValueError(f"Invalid resolution '{part}'") from exc
    if not resolutions:
        raise ValueError("No resolutions provided")
    return resolutions


def _parse_mapping(text: str, *, value_type: type) -> dict[int, object]:
    mapping: dict[int, object] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid mapping entry '{part}' (expected key=value)")
        key_text, value_text = part.split("=", 1)
        key_text = key_text.strip()
        value_text = value_text.strip()
        if not key_text or not value_text:
            raise ValueError(f"Invalid mapping entry '{part}'")
        try:
            key = int(key_text)
        except ValueError as exc:
            raise ValueError(f"Invalid mapping key '{key_text}'") from exc
        if value_type is int:
            try:
                value = int(value_text)
            except ValueError as exc:
                raise ValueError(f"Invalid mapping value '{value_text}'") from exc
        else:
            value = value_text
        mapping[key] = value
    if not mapping:
        raise ValueError("Mapping override is empty")
    return mapping


def main() -> int:
    args = _parse_args()

    try:
        resolutions = _parse_resolutions(args.resolutions)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    points_csv = resolve_path(args.points_csv, base=PROJECT_ROOT)
    if points_csv is None or not points_csv.is_file():
        print(f"error: points CSV '{args.points_csv}' not found", file=sys.stderr)
        return 1

    binary = resolve_path(args.binary, base=PROJECT_ROOT)
    if binary is None or not binary.is_file():
        print(f"error: solver binary '{args.binary}' not found", file=sys.stderr)
        return 1

    script_path = PROJECT_ROOT / "scripts" / "tcpc_submit_best_cases.py"
    if not script_path.is_file():
        print(f"error: missing helper script '{script_path}'", file=sys.stderr)
        return 1

    mem_by_res = dict(DEFAULT_MEM_BY_RES)
    if args.mem_by_resolution:
        try:
            mem_by_res.update(_parse_mapping(args.mem_by_resolution, value_type=str))
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    gpu_mem_by_res: dict[int, object] = {}
    if args.gpu_mem_by_resolution:
        try:
            gpu_mem_by_res = _parse_mapping(args.gpu_mem_by_resolution, value_type=str)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    cpus_by_res = dict(DEFAULT_CPUS_BY_RES)
    if args.cpus_by_resolution:
        try:
            cpus_by_res.update(_parse_mapping(args.cpus_by_resolution, value_type=int))
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    constraint_by_res: dict[int, object] = {}
    if args.constraint_by_resolution:
        try:
            constraint_by_res = _parse_mapping(args.constraint_by_resolution, value_type=str)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    gres_by_res: dict[int, object] = {}
    if args.gres_by_resolution:
        try:
            gres_by_res = _parse_mapping(args.gres_by_resolution, value_type=str)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    for res in resolutions:
        if res not in mem_by_res:
            print(f"error: missing memory mapping for resolution {res}", file=sys.stderr)
            return 1
        if res not in cpus_by_res:
            print(f"error: missing CPU mapping for resolution {res}", file=sys.stderr)
            return 1

    if args.legend:
        legend_path = resolve_path(args.legend, base=PROJECT_ROOT)
        if legend_path is None:
            print("error: unable to resolve legend path", file=sys.stderr)
            return 1
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        legend_path = (
            PROJECT_ROOT
            / "data"
            / "junction_tcpc_logs"
            / "convergence"
            / f"legend_{timestamp}.csv"
        )

    for res in resolutions:
        mem = mem_by_res[res]
        cpus = cpus_by_res[res]
        gpu_mem = gpu_mem_by_res.get(res, args.gpu_mem)
        constraint = constraint_by_res.get(res, args.constraint)
        gres = gres_by_res.get(res, args.gres)
        job_prefix = f"{args.job_name_prefix}-r{res}"
        case_prefix = f"{args.case_prefix}-r{res}"

        cmd = [
            "python3",
            "-u",
            str(script_path),
            "--points-csv",
            str(points_csv),
            "--binary",
            str(binary),
            "--resolution",
            str(res),
            "--mem",
            str(mem),
            "--cpus",
            str(cpus),
            "--walltime",
            str(args.walltime),
            "--job-name-prefix",
            job_prefix,
            "--case-prefix",
            case_prefix,
            "--legend",
            str(legend_path),
            "--processes",
            str(args.processes),
        ]
        if gpu_mem is not None:
            cmd.extend(["--gpu-mem", str(gpu_mem)])
        if args.partition:
            cmd.extend(["--partition", str(args.partition)])
        if constraint is not None:
            cmd.extend(["--constraint", str(constraint)])
        if gres is not None:
            cmd.extend(["--gres", str(gres)])
        if args.gpus is not None:
            cmd.extend(["--gpus", str(args.gpus)])
        if args.dry_run:
            cmd.append("--dry-run")

        msg = f"[convergence] resolution {res}: mem={mem} cpus={cpus}"
        if gpu_mem is not None:
            msg += f" gpu_mem={gpu_mem}"
        if constraint is not None:
            msg += f" constraint={constraint}"
        if gres is not None:
            msg += f" gres={gres}"
        print(msg, flush=True)
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
        if result.returncode != 0:
            return result.returncode

    print(f"[legend] {legend_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
