#!/usr/bin/env python3
"""Submit or run batches of extra TCPC points."""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
from pathlib import Path
from typing import Iterable

from tcpc_config import PROJECT_ROOT, load_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit or run extra-point evaluations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True, help="Config file path")
    parser.add_argument("--points-csv", type=Path, help="CSV with explicit points")
    parser.add_argument("--base", help="Comma-separated base point values")
    parser.add_argument("--step", help="Comma-separated step values")
    parser.add_argument(
        "--mode",
        choices=("sbatch", "local"),
        default="sbatch",
        help="Submit via sbatch or run locally",
    )
    parser.add_argument("--csv", type=Path, help="Output CSV path")
    parser.add_argument("--algorithm-label", help="Objective algorithm label")
    parser.add_argument("--run-tag", help="TCPC_RUN_TAG override")
    parser.add_argument("--case-suffix", default="", help="Suffix appended to case labels")
    parser.add_argument("--job-name", help="Slurm job name")
    parser.add_argument("--time", help="Slurm walltime")
    parser.add_argument("--cpus", type=int, help="Slurm CPUs per task")
    parser.add_argument("--gpus", type=int, help="Slurm GPU count")
    parser.add_argument("--mem", help="Slurm memory request")
    parser.add_argument("--partition", help="Slurm partition")
    parser.add_argument("--output", help="Slurm output path")
    parser.add_argument("--open-mode", help="Slurm open mode")
    parser.add_argument("--solver-time", help="Solver job Slurm walltime (TCPC_SLURM_WALLTIME)")
    parser.add_argument(
        "--solver-cpus", type=int, help="Solver job Slurm CPUs per task (TCPC_SLURM_CPUS)"
    )
    parser.add_argument(
        "--solver-gpus", type=int, help="Solver job Slurm GPU count (TCPC_SLURM_GPUS)"
    )
    parser.add_argument("--solver-mem", help="Solver job Slurm memory (TCPC_SLURM_MEM)")
    parser.add_argument(
        "--solver-partition",
        help="Solver job Slurm partition (TCPC_SLURM_PARTITION)",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        help="1-based start index (inclusive) for selecting points from the CSV",
    )
    parser.add_argument(
        "--stop-idx",
        type=int,
        help="1-based stop index (inclusive) for selecting points from the CSV",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    return parser.parse_args()


def _iter_points_from_csv(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def _iter_points_from_base_step(base: str, step: str) -> Iterable[dict[str, str]]:
    names = ["offset", "lower_angle", "upper_angle", "lower_flare", "upper_flare"]
    base_vals = [float(x) for x in base.split(",")]
    step_vals = [float(x) for x in step.split(",")]
    if len(base_vals) != len(names) or len(step_vals) != len(names):
        raise ValueError("BASE and STEP must each have 5 comma-separated values")

    for idx, name in enumerate(names):
        delta = step_vals[idx]
        for sign, factor in (("minus", -1.0), ("plus", 1.0)):
            vals = list(base_vals)
            vals[idx] = vals[idx] + factor * delta
            yield {
                "direction": name,
                "sign": sign,
                "offset": f"{vals[0]:.8f}",
                "lower_angle": f"{vals[1]:.8f}",
                "upper_angle": f"{vals[2]:.8f}",
                "lower_flare": f"{vals[3]:.8f}",
                "upper_flare": f"{vals[4]:.8f}",
            }


def main() -> int:
    args = _parse_args()

    if not args.points_csv and not (args.base and args.step):
        raise SystemExit("Provide --points-csv or --base/--step")

    cfg = load_config(args.config)
    extra_cfg = cfg.get("extra_points") or {}
    submit_cfg = extra_cfg.get("submit") or {}

    objective_kind = cfg.get("objective_kind", "tcpc")
    algorithm_label = (
        args.algorithm_label
        or extra_cfg.get("algorithm_label")
        or ("extra_points_tke" if objective_kind == "tke" else "extra_points")
    )

    default_csv = extra_cfg.get("csv_default") or (
        PROJECT_ROOT / "tmp" / "extra_points" / f"extra_points_{objective_kind}.csv"
    )
    csv_path = (args.csv or Path(default_csv)).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    job_name = args.job_name or submit_cfg.get("job_name") or "tcpc-extra-point"
    walltime = args.time or submit_cfg.get("time") or "24:00:00"
    cpus = args.cpus if args.cpus is not None else submit_cfg.get("cpus")
    gpus = args.gpus if args.gpus is not None else submit_cfg.get("gpus")
    mem = args.mem or submit_cfg.get("mem")
    partition = args.partition or submit_cfg.get("partition")
    output = args.output or submit_cfg.get("output")
    open_mode = args.open_mode or submit_cfg.get("open_mode")

    if args.points_csv:
        point_iter = _iter_points_from_csv(args.points_csv)
    else:
        point_iter = _iter_points_from_base_step(args.base, args.step)

    if args.start_idx is not None or args.stop_idx is not None:
        points = list(point_iter)
        total = len(points)
        start_idx = args.start_idx or 1
        stop_idx = args.stop_idx or total
        if start_idx < 1 or stop_idx < 1:
            raise ValueError("START/STOP indices must be >= 1")
        if stop_idx < start_idx:
            raise ValueError("STOP index must be >= START index")
        if start_idx > total or stop_idx > total:
            raise ValueError(f"START/STOP indices out of range (1..{total})")
        point_iter = points[start_idx - 1 : stop_idx]

    solver_env = {}
    if args.solver_time is not None:
        solver_env["TCPC_SLURM_WALLTIME"] = str(args.solver_time)
    if args.solver_cpus is not None:
        solver_env["TCPC_SLURM_CPUS"] = str(args.solver_cpus)
    if args.solver_gpus is not None:
        solver_env["TCPC_SLURM_GPUS"] = str(args.solver_gpus)
    if args.solver_mem is not None:
        solver_env["TCPC_SLURM_MEM"] = str(args.solver_mem)
    if args.solver_partition is not None:
        solver_env["TCPC_SLURM_PARTITION"] = str(args.solver_partition)

    for row in point_iter:
        direction = row.get("direction") or row.get("Direction")
        sign = row.get("sign") or row.get("Sign")
        offset = row.get("offset")
        lower_angle = row.get("lower_angle")
        upper_angle = row.get("upper_angle")
        lower_flare = row.get("lower_flare")
        upper_flare = row.get("upper_flare")
        case_label = row.get("case_label") or f"{direction}_{sign}{args.case_suffix}"

        if not all([direction, sign, offset, lower_angle, upper_angle, lower_flare, upper_flare]):
            raise ValueError(f"Missing required fields in row: {row}")

        cmd = [
            "python3",
            "-u",
            str(PROJECT_ROOT / "scripts" / "tcpc_extra_point.py"),
            "--config",
            str(args.config.resolve()),
            "--direction",
            str(direction),
            "--sign",
            str(sign),
            "--offset",
            str(offset),
            "--lower-angle",
            str(lower_angle),
            "--upper-angle",
            str(upper_angle),
            "--lower-flare",
            str(lower_flare),
            "--upper-flare",
            str(upper_flare),
            "--csv",
            str(csv_path),
            "--algorithm-label",
            str(algorithm_label),
            "--case-label",
            str(case_label),
        ]
        if args.run_tag:
            cmd.extend(["--run-tag", args.run_tag])

        if args.mode == "local":
            if args.dry_run:
                print("[local]", " ".join(shlex.quote(token) for token in cmd))
                continue
            if solver_env:
                env = os.environ.copy()
                env.update(solver_env)
            else:
                env = None
            result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False, env=env)
            if result.returncode != 0:
                raise SystemExit(result.returncode)
            continue

        env_prefix = " ".join(
            f"{key}={shlex.quote(value)}" for key, value in solver_env.items()
        )
        if env_prefix:
            wrap_cmd = "cd {} && {} {}".format(
                shlex.quote(str(PROJECT_ROOT)),
                env_prefix,
                " ".join(shlex.quote(token) for token in cmd),
            )
        else:
            wrap_cmd = "cd {} && {}".format(
                shlex.quote(str(PROJECT_ROOT)),
                " ".join(shlex.quote(token) for token in cmd),
            )

        sbatch_cmd = [
            "sbatch",
            "--parsable",
            f"--job-name={job_name}",
            f"--time={walltime}",
        ]
        if partition:
            sbatch_cmd.append(f"--partition={partition}")
        if gpus is not None:
            sbatch_cmd.append(f"--gpus={gpus}")
        if cpus is not None:
            sbatch_cmd.append(f"--cpus-per-task={cpus}")
        if mem:
            sbatch_cmd.append(f"--mem={mem}")
        if output:
            sbatch_cmd.append(f"--output={output}")
        if open_mode:
            sbatch_cmd.append(f"--open-mode={open_mode}")

        sbatch_cmd.extend(["--wrap", wrap_cmd])

        if args.dry_run:
            print("[sbatch]", " ".join(shlex.quote(token) for token in sbatch_cmd))
            continue

        result = subprocess.run(sbatch_cmd, cwd=PROJECT_ROOT, check=False, capture_output=True)
        if result.returncode != 0:
            raise SystemExit(result.returncode)
        submission_id = result.stdout.decode("utf-8").strip()
        job_id = submission_id.split(";")[0] if submission_id else ""
        print(f"[sbatch] {case_label}: job {job_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
