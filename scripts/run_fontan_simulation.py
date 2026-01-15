#!/usr/bin/env python3
"""Submit sim_fontan_* through Slurm and return a scalar output."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

COMPLETED_STATES = {"COMPLETED"}
FAILED_STATES = {
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "BOOT_FAIL",
    "REVOKED",
    "DEADLINE",
    "STOPPED",
}

DEFAULT_WALLTIME = "20:00:00"


def run_fontan_simulation(
    filename: str,
    resolution: int,
    *,
    run_dir: Path,
    data_root: Path,
    binary_path: Path,
    partition: Optional[str] = None,
    gpus: Optional[int] = 1,
    cpus: Optional[int] = 4,
    mem: Optional[str] = "32G",
    walltime: str = DEFAULT_WALLTIME,
    poll_interval: float = 30.0,
    default_on_failure: float = 1.0e9,
    avg_window: Optional[float] = 1.0,
    verbose: bool = False,
) -> tuple[float, float | None, Path | None, Optional[str], bool]:
    """Run sim_fontan_* via Slurm and return (value, time_end, val_path, state, used_fallback)."""
    if not binary_path.is_file():
        if verbose:
            print(
                f"[fontan] solver binary not found at {binary_path}; "
                f"returning fallback {default_on_failure}",
            )
        return float(default_on_failure), None, None, None, True

    run_dir.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    (data_root / "tmp").mkdir(parents=True, exist_ok=True)

    sbatch_path = _write_sbatch(
        run_dir=run_dir,
        solver_binary=binary_path,
        resolution=resolution,
        filename=filename,
        data_root=data_root,
        partition=partition,
        gpus=gpus,
        cpus=cpus,
        mem=mem,
        walltime=walltime,
    )
    try:
        job_id = _submit_job(sbatch_path)
    except subprocess.SubprocessError as exc:
        if verbose:
            print(f"[fontan] failed to submit job: {exc}; returning fallback value")
        return float(default_on_failure), None, None, None, True

    if verbose:
        print(f"[fontan] submitted job {job_id} (run dir: {run_dir})")

    state = _wait_for_job(job_id, poll_interval=poll_interval, verbose=verbose)
    result = _read_result(
        data_root,
        filename,
        avg_window=avg_window,
        verbose=verbose,
    )
    if result is not None:
        value, time_end, val_path = result
        if verbose:
            print(f"[fontan] returning value {value} from state {state}")
        return float(value), time_end, val_path, state, False

    if verbose:
        msg = f"[fontan] result missing after job state={state}; using fallback"
        print(msg)
    return float(default_on_failure), None, None, state, True


def _write_sbatch(
    *,
    run_dir: Path,
    solver_binary: Path,
    resolution: int,
    filename: str,
    data_root: Path,
    partition: Optional[str],
    gpus: Optional[int],
    cpus: Optional[int],
    mem: Optional[str],
    walltime: str,
) -> Path:
    job_name = _job_name(run_dir.name)
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

    solver_quoted = shlex.quote(str(solver_binary.resolve()))
    filename_quoted = shlex.quote(filename)
    data_root_quoted = shlex.quote(str(data_root.resolve()))
    tmp_dir_quoted = shlex.quote(str((data_root / "tmp").resolve()))
    result_basename = f"val_{Path(filename).name}"
    result_basename_quoted = shlex.quote(result_basename)
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

echo "sim_fontan start $(date --iso-8601=seconds)" >&2
"$SOLVER_BINARY" "$RESOLUTION" "$INPUT_FILENAME"
echo "sim_fontan end $(date --iso-8601=seconds)" >&2

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


def _job_name(run_dir_name: str) -> str:
    suffix = run_dir_name.rsplit("_", 1)[-1]
    suffix = "".join(ch for ch in suffix if ch.isalnum())
    if len(suffix) > 6:
        suffix = suffix[-6:]
    return f"fontan-{suffix or 'run'}"


def _submit_job(sbatch_path: Path) -> str:
    proc = subprocess.run(
        ["sbatch", "--parsable", sbatch_path.name],
        cwd=sbatch_path.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip().split(";", 1)[0]


def _wait_for_job(job_id: str, *, poll_interval: float, verbose: bool) -> Optional[str]:
    last_state: Optional[str] = None
    missing_polls = 0
    while True:
        state = _query_job_state(job_id)
        if state:
            if state != last_state and verbose:
                print(f"[fontan] job {job_id} state: {state}")
            last_state = state
            missing_polls = 0
        else:
            missing_polls += 1
            if verbose and missing_polls == 1:
                print(f"[fontan] job {job_id} state unknown; retrying")

        if state in COMPLETED_STATES or state in FAILED_STATES:
            return state
        if missing_polls >= 10:
            if verbose:
                print(f"[fontan] job {job_id} state unresolved after retries")
            return last_state
        time.sleep(poll_interval)


def _query_job_state(job_id: str) -> Optional[str]:
    def _first_nonempty(lines: Sequence[str]) -> Optional[str]:
        for line in lines:
            stripped = line.strip()
            if stripped:
                return stripped
        return None

    squeue = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%T"],
        capture_output=True,
        text=True,
    )
    if squeue.returncode == 0:
        state = _first_nonempty(squeue.stdout.splitlines())
        if state:
            return state.upper()

    sacct = subprocess.run(
        ["sacct", "-j", job_id, "--format=State", "--parsable2", "--noheader"],
        capture_output=True,
        text=True,
    )
    if sacct.returncode == 0:
        state = _first_nonempty(sacct.stdout.splitlines())
        if state:
            return state.split("|", 1)[0].upper()
    return None


def _read_result(
    data_root: Path,
    filename: str,
    *,
    avg_window: Optional[float],
    verbose: bool,
) -> Optional[tuple[float, float | None, Path]]:
    result_path = data_root / "tmp" / f"val_{Path(filename).name}"
    if not result_path.is_file():
        if verbose:
            print(f"[fontan] result file {result_path} not found")
        return None

    try:
        lines = result_path.read_text(encoding="ascii", errors="ignore").splitlines()
    except OSError as exc:
        if verbose:
            print(f"[fontan] failed to read result file: {exc}")
        return None

    times: list[float] = []
    values: list[float] = []
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 2:
            try:
                v = float(parts[-1])
            except Exception:
                continue
            values.append(v)
            continue
        try:
            t = float(parts[0])
            v = float(parts[1])
        except Exception:
            continue
        times.append(t)
        values.append(v)

    if not values:
        if verbose:
            print(f"[fontan] could not parse any numeric value from {result_path}")
        return None

    time_end = times[-1] if times else None
    if (
        avg_window is None
        or (isinstance(avg_window, (int, float)) and avg_window <= 0)
        or not times
        or len(times) != len(values)
    ):
        return float(values[-1]), time_end, result_path

    t_end = times[-1]
    t0 = t_end - float(avg_window)
    start_idx = 0
    for i, t in enumerate(times):
        if t >= t0:
            start_idx = max(0, i - 1)
            break
    sel_t = times[start_idx:]
    sel_v = values[start_idx:]
    if len(sel_t) == 1:
        return float(sel_v[0]), time_end, result_path

    area = 0.0
    tspan = 0.0
    prev_t = max(sel_t[0], t0)
    prev_v = sel_v[0]
    for i in range(1, len(sel_t)):
        cur_t = sel_t[i]
        cur_v = sel_v[i]
        a = max(prev_t, t0)
        b = min(cur_t, t_end)
        if b > a:
            dt = b - a
            if cur_t != prev_t:
                va = prev_v + (cur_v - prev_v) * (a - prev_t) / (cur_t - prev_t)
                vb = prev_v + (cur_v - prev_v) * (b - prev_t) / (cur_t - prev_t)
            else:
                va = vb = prev_v
            area += 0.5 * (va + vb) * dt
            tspan += dt
        prev_t, prev_v = cur_t, cur_v

    if tspan <= 0.0:
        return float(values[-1]), time_end, result_path
    return float(area / tspan), time_end, result_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit sim_fontan_* through Slurm and return the scalar output."
    )
    parser.add_argument("filename", help="Data file identifier, e.g. example.txt")
    parser.add_argument(
        "--resolution",
        type=int,
        default=5,
        help="Resolution argument passed to sim_fontan_* (default: 5)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path containing geometry/dimensions/tmp subdirectories",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory used for Slurm logs and sbatch script",
    )
    parser.add_argument(
        "--binary-path",
        type=Path,
        required=True,
        help="Path to sim_fontan_* executable",
    )
    parser.add_argument("--partition", default=None, help="Slurm partition (default: none)")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs (default: 1)")
    parser.add_argument("--cpus", type=int, default=4, help="CPUs per task (default: 4)")
    parser.add_argument("--mem", default="32G", help="Memory request (default: 32G)")
    parser.add_argument(
        "--walltime",
        default=DEFAULT_WALLTIME,
        help="Slurm walltime limit (default: 20:00:00)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=30.0,
        help="Seconds between job status checks (default: 30)",
    )
    parser.add_argument(
        "--avg-window",
        type=float,
        default=1.0,
        help=(
            "Trailing time window (in physical seconds printed by the solver) "
            "over which to compute a time-weighted average. "
            "Default: 1.0 seconds. Set <=0 to return the last value."
        ),
    )
    parser.add_argument(
        "--default-on-failure",
        type=float,
        default=1.0e9,
        help="Fallback value when the job fails or no result is produced",
    )
    parser.add_argument("--verbose", action="store_true", help="Print progress information")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    value, _, _, _, _ = run_fontan_simulation(
        args.filename,
        resolution=args.resolution,
        run_dir=args.run_dir,
        data_root=args.data_root,
        binary_path=args.binary_path,
        partition=args.partition,
        gpus=args.gpus,
        cpus=args.cpus,
        mem=args.mem,
        walltime=args.walltime,
        poll_interval=args.poll_interval,
        default_on_failure=args.default_on_failure,
        avg_window=args.avg_window,
        verbose=args.verbose,
    )
    print(value)


if __name__ == "__main__":
    main()
