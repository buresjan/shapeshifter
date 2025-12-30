#!/usr/bin/env python3
"""Evaluate a single TCPC point and append the objective to a CSV."""

from __future__ import annotations

import argparse
import csv
import fcntl
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from tcpc_config import PROJECT_ROOT, load_config
import tcpc_objective


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one TCPC evaluation and append the objective to a CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True, help="Config file path")
    parser.add_argument("--direction", required=True, help="Perturbation direction label.")
    parser.add_argument("--sign", required=True, help="Perturbation sign label.")
    parser.add_argument("--offset", type=float, required=True, help="offset parameter value.")
    parser.add_argument("--lower-angle", type=float, required=True, help="lower_angle value.")
    parser.add_argument("--upper-angle", type=float, required=True, help="upper_angle value.")
    parser.add_argument("--lower-flare", type=float, required=True, help="lower_flare value.")
    parser.add_argument("--upper-flare", type=float, required=True, help="upper_flare value.")
    parser.add_argument("--csv", type=Path, help="Destination CSV that aggregates all points.")
    parser.add_argument(
        "--algorithm-label",
        help="Label forwarded to the objective logger.",
    )
    parser.add_argument(
        "--run-tag",
        help="Optional TCPC_RUN_TAG override for naming generated artifacts.",
    )
    parser.add_argument(
        "--case-label",
        help="Optional human-friendly label for this point (defaults to direction_sign).",
    )
    parser.add_argument(
        "--job-name",
        help="Optional job name stored in the CSV (defaults to SLURM_JOB_NAME).",
    )
    parser.add_argument(
        "--slurm-job-id",
        help="Optional job id stored in the CSV (defaults to SLURM_JOB_ID).",
    )
    return parser.parse_args()


def _append_row(csv_path: Path, fieldnames: list[str], row: dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0, os.SEEK_END)
        is_empty = handle.tell() == 0
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if is_empty:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _objective_settings(cfg: dict, algorithm_label: str) -> dict:
    space_cfg = cfg.get("space") or {}
    return {
        "label": cfg.get("label", "tcpc_run"),
        "algorithm_label": algorithm_label,
        "objective_kind": cfg.get("objective_kind", "tcpc"),
        "resolution": cfg.get("resolution", 5),
        "parameter_names": tuple(
            space_cfg.get(
                "names",
                ("offset", "lower_angle", "upper_angle", "lower_flare", "upper_flare"),
            )
        ),
        "geometry_penalty": cfg.get("geometry_penalty", 1.0e9),
        "solver": cfg.get("solver", {}),
        "split": cfg.get("split", {}),
        "run_tag": cfg.get("run_tag"),
        "eval_log_root": cfg.get("eval_log_root"),
        "mpi_accelerator": cfg.get("mpi_accelerator"),
    }


def main() -> int:
    args = _parse_args()

    if args.run_tag:
        os.environ["TCPC_RUN_TAG"] = args.run_tag

    cfg = load_config(args.config)
    extra_cfg = cfg.get("extra_points") or {}
    algorithm_label = (
        args.algorithm_label
        or extra_cfg.get("algorithm_label")
        or ("extra_points_tke" if cfg.get("objective_kind") == "tke" else "extra_points")
    )

    tcpc_objective.configure(_objective_settings(cfg, algorithm_label))

    x = np.array(
        [
            args.offset,
            args.lower_angle,
            args.upper_angle,
            args.lower_flare,
            args.upper_flare,
        ],
        dtype=float,
    )

    try:
        value = float(tcpc_objective.objective(x, algorithm=algorithm_label))
    except Exception as exc:
        print(f"[extra] objective evaluation failed: {exc}", file=sys.stderr, flush=True)
        raise

    geometry_penalty = float(cfg.get("geometry_penalty", 1.0e9))
    status = "ok"
    if math.isnan(value):
        status = "nan"
    elif math.isclose(value, geometry_penalty, rel_tol=0.0, abs_tol=1e-9):
        status = "penalty"

    objective_kind = cfg.get("objective_kind", "tcpc")
    default_csv = extra_cfg.get("csv_default") or (
        PROJECT_ROOT / "tmp" / "extra_points" / f"extra_points_{objective_kind}.csv"
    )
    csv_path = (args.csv or Path(default_csv)).resolve()

    job_id = args.slurm_job_id or os.environ.get("SLURM_JOB_ID", "")
    job_name = args.job_name or os.environ.get("SLURM_JOB_NAME", "")
    run_tag = os.environ.get("TCPC_RUN_TAG", "")
    case_label = args.case_label or f"{args.direction}_{args.sign}"

    fieldnames = [
        "case_label",
        "direction",
        "sign",
        "offset",
        "lower_angle",
        "upper_angle",
        "lower_flare",
        "upper_flare",
        "objective",
        "status",
        "algorithm",
        "run_tag",
        "slurm_job_id",
        "job_name",
        "timestamp_iso8601",
    ]
    row = {
        "case_label": case_label,
        "direction": args.direction,
        "sign": args.sign,
        "offset": args.offset,
        "lower_angle": args.lower_angle,
        "upper_angle": args.upper_angle,
        "lower_flare": args.lower_flare,
        "upper_flare": args.upper_flare,
        "objective": value,
        "status": status,
        "algorithm": algorithm_label,
        "run_tag": run_tag,
        "slurm_job_id": job_id,
        "job_name": job_name,
        "timestamp_iso8601": datetime.now().isoformat(),
    }

    _append_row(csv_path, fieldnames, row)

    print(
        f"[extra] completed {case_label}: objective={value:.6g}, status={status}, csv={csv_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
