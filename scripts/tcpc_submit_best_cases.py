#!/usr/bin/env python3
"""Submit best-case TCPC simulations via Slurm and write a legend CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

from tcpc_common import ensure_txt_suffix, generate_geometry, prepare_run_directory, stage_geometry
from tcpc_config import PROJECT_ROOT, resolve_path


DEFAULT_POINTS_CSV = Path("configs/tcpc/points/best_cases_combined.csv")
DEFAULT_BINARY = Path("submodules/tnl-lbm/build/sim_NSE/sim_tcpc_combined")
DEFAULT_LEGEND_DIR = Path("data/junction_tcpc_logs/best_cases")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit best-case TCPC simulations via Slurm.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--points-csv",
        type=Path,
        default=DEFAULT_POINTS_CSV,
        help="CSV with best-case points (case_label + 5 parameters).",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=DEFAULT_BINARY,
        help="Path to sim_tcpc_combined (or compatible) binary.",
    )
    parser.add_argument("--resolution", type=int, default=5, help="Solver resolution.")
    parser.add_argument("--processes", type=int, default=1, help="Meshgen voxel processes.")
    parser.add_argument("--mem", default="16G", help="Slurm memory request.")
    parser.add_argument(
        "--gpu-mem",
        help="Requested GPU memory (for legend only; does not affect Slurm allocation).",
    )
    parser.add_argument("--gpus", type=int, default=1, help="Slurm GPU count.")
    parser.add_argument("--cpus", type=int, default=4, help="Slurm CPUs per task.")
    parser.add_argument("--walltime", default="24:00:00", help="Slurm walltime.")
    parser.add_argument("--partition", help="Slurm partition.")
    parser.add_argument(
        "--constraint",
        help="Slurm constraint (e.g. GPU type).",
    )
    parser.add_argument(
        "--gres",
        help="Slurm GRES override (e.g. gpu:a100:1).",
    )
    parser.add_argument(
        "--job-name-prefix",
        default="tcpc-best",
        help="Prefix for Slurm job names.",
    )
    parser.add_argument(
        "--legend",
        type=Path,
        help="Legend CSV output path (defaults under data/junction_tcpc_logs).",
    )
    parser.add_argument(
        "--case-prefix",
        default="best",
        help="Prefix used for generated case filenames.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write geometry + sbatch scripts but do not submit.",
    )
    return parser.parse_args()


def _sanitize_label(label: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in label.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "case"


def _hash_params(values: Iterable[float]) -> str:
    text = ",".join(f"{value:.8g}" for value in values)
    return hashlib.sha1(text.encode("ascii")).hexdigest()[:10]


def _load_points(csv_path: Path) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = (row.get("case_label") or row.get("label") or "").strip()
            if not label:
                raise ValueError(f"Missing case_label in row: {row}")

            def _require(name: str) -> tuple[str, float]:
                raw = row.get(name)
                raw_txt = raw.strip() if isinstance(raw, str) else ""
                if not raw_txt:
                    raise ValueError(f"Missing {name} in row: {row}")
                return raw_txt, float(raw_txt)

            offset_raw, offset = _require("offset")
            lower_angle_raw, lower_angle = _require("lower_angle")
            upper_angle_raw, upper_angle = _require("upper_angle")
            lower_flare_raw, lower_flare = _require("lower_flare")
            upper_flare_raw, upper_flare = _require("upper_flare")

            points.append(
                {
                    "case_label": label,
                    "offset": offset,
                    "lower_angle": lower_angle,
                    "upper_angle": upper_angle,
                    "lower_flare": lower_flare,
                    "upper_flare": upper_flare,
                    "offset_raw": offset_raw,
                    "lower_angle_raw": lower_angle_raw,
                    "upper_angle_raw": upper_angle_raw,
                    "lower_flare_raw": lower_flare_raw,
                    "upper_flare_raw": upper_flare_raw,
                }
            )
    if not points:
        raise ValueError(f"No points found in {csv_path}")
    return points


def _sanitize_job_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip())
    if not cleaned:
        cleaned = "tcpc-run"
    return cleaned[:128]


def _write_sbatch(
    *,
    run_dir: Path,
    solver_binary: Path,
    resolution: int,
    filename: str,
    data_root: Path,
    partition: str | None,
    constraint: str | None,
    gres: str | None,
    gpus: int | None,
    cpus: int | None,
    mem: str | None,
    walltime: str,
    job_name: str,
) -> Path:
    resolved_job_name = _sanitize_job_name(job_name)
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={resolved_job_name}",
        "#SBATCH --output=slurm.out",
        "#SBATCH --error=slurm.err",
        f"#SBATCH --time={walltime}",
    ]
    if partition:
        lines.append(f"#SBATCH --partition={partition}")
    if constraint:
        lines.append(f"#SBATCH --constraint={constraint}")
    if gres:
        lines.append(f"#SBATCH --gres={gres}")
    if gpus is not None:
        lines.append(f"#SBATCH --gpus={gpus}")
    if cpus is not None:
        lines.append(f"#SBATCH --cpus-per-task={cpus}")
    if mem:
        lines.append(f"#SBATCH --mem={mem}")

    tmp_dir = data_root / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    solver_quoted = shlex.quote(str(solver_binary.resolve()))
    filename_quoted = shlex.quote(filename)
    data_root_quoted = shlex.quote(str(data_root.resolve()))
    tmp_dir_rel = Path("sim_NSE") / "tmp"
    tmp_dir_quoted = shlex.quote(str(tmp_dir_rel))
    result_basename = f"val_{Path(filename).name}"
    result_basename_quoted = shlex.quote(result_basename)
    resolution_literal = str(resolution)
    solver_label = shlex.quote(solver_binary.name)

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

echo "{solver_label} start $(date --iso-8601=seconds)" >&2
"$SOLVER_BINARY" "$RESOLUTION" "$INPUT_FILENAME"
echo "{solver_label} end $(date --iso-8601=seconds)" >&2

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


def _submit_job(sbatch_path: Path) -> str:
    try:
        proc = subprocess.run(
            ["sbatch", "--parsable", sbatch_path.name],
            cwd=sbatch_path.parent,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        detail = stderr or stdout or "sbatch returned a non-zero exit status"
        raise RuntimeError(detail) from exc
    job_id = proc.stdout.strip().split(";", 1)[0]
    if not job_id:
        raise RuntimeError("sbatch did not return a job id")
    return job_id


def _append_legend_row(path: Path, fieldnames: list[str], row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    args = _parse_args()

    points_csv = resolve_path(args.points_csv, base=PROJECT_ROOT)
    if points_csv is None or not points_csv.is_file():
        print(f"error: points CSV '{args.points_csv}' not found", file=sys.stderr)
        return 1

    binary = resolve_path(args.binary, base=PROJECT_ROOT)
    if binary is None or not binary.is_file():
        print(f"error: solver binary '{args.binary}' not found", file=sys.stderr)
        return 1

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    legend_path = resolve_path(
        args.legend or (DEFAULT_LEGEND_DIR / f"legend_{timestamp}.csv"),
        base=PROJECT_ROOT,
    )
    if legend_path is None:
        print("error: unable to resolve legend path", file=sys.stderr)
        return 1

    points = _load_points(points_csv)

    fieldnames = [
        "case_label",
        "case_slug",
        "offset",
        "lower_angle",
        "upper_angle",
        "lower_flare",
        "upper_flare",
        "case_name",
        "resolution",
        "binary",
        "run_dir",
        "data_root",
        "result_path",
        "sbatch_path",
        "slurm_job_id",
        "submission_status",
        "walltime",
        "cpus",
        "gpus",
        "mem",
        "gpu_mem",
        "partition",
        "constraint",
        "gres",
        "submitted_at",
    ]

    for point in points:
        label = str(point["case_label"])
        slug = _sanitize_label(label).lower()
        offset = float(point["offset"])
        lower_angle = float(point["lower_angle"])
        upper_angle = float(point["upper_angle"])
        lower_flare = float(point["lower_flare"])
        upper_flare = float(point["upper_flare"])
        case_hash = _hash_params([offset, lower_angle, upper_angle, lower_flare, upper_flare])
        case_stem = f"{args.case_prefix}_{slug}_{case_hash}"
        case_name = ensure_txt_suffix(case_stem)

        run_dir = prepare_run_directory(PROJECT_ROOT, case_name)
        workspace = run_dir / "meshgen_output"

        try:
            files, generated_case_name = generate_geometry(
                workspace,
                case_name,
                resolution=args.resolution,
                lower_angle=lower_angle,
                upper_angle=upper_angle,
                upper_flare=upper_flare,
                lower_flare=lower_flare,
                offset=offset,
                num_processes=args.processes,
            )
        except Exception as exc:
            print(f"error: mesh generation failed for {label}: {exc}", file=sys.stderr)
            return 1

        case_basename = Path(generated_case_name).name
        data_root = (run_dir / "sim_NSE").resolve()
        for subdir in ("geometry", "dimensions", "angle", "tmp"):
            (data_root / subdir).mkdir(parents=True, exist_ok=True)

        stage_geometry(files, data_root)

        job_name = f"{args.job_name_prefix}-{slug}"
        effective_gpus = None if args.gres else args.gpus
        sbatch_path = _write_sbatch(
            run_dir=run_dir,
            solver_binary=binary,
            resolution=args.resolution,
            filename=case_basename,
            data_root=data_root,
            partition=args.partition,
            constraint=args.constraint,
            gres=args.gres,
            gpus=effective_gpus,
            cpus=args.cpus,
            mem=args.mem,
            walltime=args.walltime,
            job_name=job_name,
        )

        job_id = ""
        status = "dry-run" if args.dry_run else "submitted"
        if not args.dry_run:
            try:
                job_id = _submit_job(sbatch_path)
            except Exception as exc:
                print(f"error: failed to submit {label}: {exc}", file=sys.stderr)
                status = "submit-failed"
                return 1

        result_path = data_root / "tmp" / f"val_{case_basename}"
        row = {
            "case_label": label,
            "case_slug": slug,
            "offset": point["offset_raw"],
            "lower_angle": point["lower_angle_raw"],
            "upper_angle": point["upper_angle_raw"],
            "lower_flare": point["lower_flare_raw"],
            "upper_flare": point["upper_flare_raw"],
            "case_name": case_basename,
            "resolution": args.resolution,
            "binary": str(binary.resolve()),
            "run_dir": str(run_dir.resolve()),
            "data_root": str(data_root.resolve()),
            "result_path": str(result_path.resolve()),
            "sbatch_path": str(sbatch_path.resolve()),
            "slurm_job_id": job_id,
            "submission_status": status,
            "walltime": args.walltime,
            "cpus": args.cpus,
            "gpus": effective_gpus,
            "mem": args.mem,
            "gpu_mem": args.gpu_mem or "",
            "partition": args.partition or "",
            "constraint": args.constraint or "",
            "gres": args.gres or "",
            "submitted_at": datetime.now().isoformat(timespec="seconds"),
        }
        _append_legend_row(legend_path, fieldnames, row)

        if args.dry_run:
            print(f"[dry-run] prepared {label}: {sbatch_path}")
        else:
            print(f"[slurm] submitted {label}: job {job_id} (run dir: {run_dir})")

    print(f"[legend] {legend_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
