#!/usr/bin/env python3
"""Submit a TCPC optimization job via Slurm or run locally."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

from tcpc_config import PROJECT_ROOT, load_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit a TCPC optimization job (or run locally).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True, help="Config file path")
    parser.add_argument("--local", action="store_true", help="Run locally instead of sbatch")
    parser.add_argument("--algorithm-label", help="Override algorithm label")
    parser.add_argument("--job-name", help="Slurm job name")
    parser.add_argument("--time", help="Slurm walltime")
    parser.add_argument("--cpus", type=int, help="Slurm CPUs per task")
    parser.add_argument("--gpus", type=int, help="Slurm GPU count")
    parser.add_argument("--mem", help="Slurm memory request")
    parser.add_argument("--partition", help="Slurm partition")
    parser.add_argument("--output", help="Slurm output path")
    parser.add_argument("--open-mode", help="Slurm open mode (append/close)")
    parser.add_argument("--dry-run", action="store_true", help="Print sbatch command only")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = load_config(args.config)
    submit_cfg = cfg.get("submit") or {}

    job_name = args.job_name or submit_cfg.get("job_name") or "tcpc-opt"
    walltime = args.time or submit_cfg.get("time") or "08:00:00"
    cpus = args.cpus if args.cpus is not None else submit_cfg.get("cpus")
    gpus = args.gpus if args.gpus is not None else submit_cfg.get("gpus")
    mem = args.mem or submit_cfg.get("mem")
    partition = args.partition or submit_cfg.get("partition")
    output = args.output or submit_cfg.get("output")
    open_mode = args.open_mode or submit_cfg.get("open_mode")

    optimize_cmd = [
        "python3",
        "-u",
        str(PROJECT_ROOT / "scripts" / "tcpc_optimize.py"),
        "--config",
        str(args.config.resolve()),
    ]
    if args.algorithm_label:
        optimize_cmd.extend(["--algorithm-label", args.algorithm_label])

    if args.local:
        print("[submit] Running locally")
        print("[submit]", " ".join(shlex.quote(token) for token in optimize_cmd))
        result = subprocess.run(optimize_cmd, cwd=PROJECT_ROOT, check=False)
        return result.returncode

    wrap_cmd = "cd {} && {}".format(
        shlex.quote(str(PROJECT_ROOT)),
        " ".join(shlex.quote(token) for token in optimize_cmd),
    )

    sbatch_cmd = [
        "sbatch",
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
        print("[submit] Dry run:")
        print(" ".join(shlex.quote(token) for token in sbatch_cmd))
        return 0

    print("[submit]", " ".join(shlex.quote(token) for token in sbatch_cmd))
    result = subprocess.run(sbatch_cmd, cwd=PROJECT_ROOT, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
