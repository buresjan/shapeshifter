#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple


Z_EXTENT = 90.779533  # Matches XY_maxZ in the VEF sample config.
EXPECTED_OUTS = {"W", "S", "N", "F"}
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


def load_sample_config(vef_root: Path) -> dict:
    sample_path = vef_root / "vef_config.sample.json"
    if not sample_path.exists():
        raise FileNotFoundError(f"VEF sample config not found: {sample_path}")
    with sample_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("VEF sample config must be a JSON object.")
    if "run_kwargs" not in config or not isinstance(config["run_kwargs"], dict):
        raise ValueError("VEF sample config must contain a run_kwargs object.")
    return config


def update_run_kwargs(
    run_kwargs: dict,
    *,
    bump1_amp: float,
    bump2_amp: float,
    size_scale: float,
    straighten_strength: float,
    repair_pitch: float,
    output_dir: Path,
    keep_temp_files: bool,
) -> None:
    bumps = run_kwargs.get("bumps")
    if not isinstance(bumps, list) or len(bumps) < 2:
        raise ValueError("Expected at least two bump entries in run_kwargs['bumps'].")
    if not isinstance(bumps[0], dict) or not isinstance(bumps[1], dict):
        raise ValueError("Bump entries must be JSON objects.")

    bumps[0]["amp"] = float(bump1_amp)
    bumps[1]["amp"] = float(bump2_amp)
    run_kwargs["size_scale"] = float(size_scale)
    run_kwargs["straighten_strength"] = float(straighten_strength)
    run_kwargs["repair_pitch"] = float(repair_pitch)
    run_kwargs["output_dir"] = str(output_dir)
    run_kwargs["keep_temp_files"] = bool(keep_temp_files)


def voxelize_with_pitch(stl_path: Path, pitch: float):
    from meshgen import voxels as mg_voxels

    mesh = mg_voxels.trm.load(str(stl_path))
    if isinstance(mesh, mg_voxels.trm.Scene):
        mesh = mg_voxels.trm.util.concatenate(tuple(mesh.geometry.values()))
    if hasattr(mesh, "is_watertight") and not mesh.is_watertight:
        mg_voxels.trm.repair.fix_normals(mesh)
        mg_voxels.trm.repair.fix_winding(mesh)
        mg_voxels.trm.repair.fill_holes(mesh)
        if hasattr(mesh, "is_watertight") and not mesh.is_watertight:
            raise RuntimeError("STL is not watertight after repair; voxelization may be invalid.")

    return mg_voxels.voxelize_with_splitting(
        mesh,
        pitch,
        split=1,
        num_processes=1,
        target_bounds=mesh.bounds,
    )


def save_labeled_triplet(labeled_mesh, output_dir: Path, base_name: str) -> None:
    from meshgen.utilities import array_to_textfile

    geom_file = output_dir / f"geom_{base_name}"
    dim_file = output_dir / f"dim_{base_name}"
    val_file = output_dir / f"val_{base_name}"

    array_to_textfile(labeled_mesh, str(geom_file))
    with dim_file.open("w", encoding="utf-8") as handle:
        shape = labeled_mesh.shape
        handle.write(f"{shape[0]} {shape[1]} {shape[2]}\n")
    val_file.touch()

    print(f"Wrote voxel triplet: {geom_file}, {dim_file}, {val_file}")


def ensure_txt_suffix(name: str) -> str:
    if not name.endswith(".txt"):
        return f"{name}.txt"
    return name


def prepare_run_directory(project_root: Path, case_basename: str) -> Path:
    runs_root = project_root / "tmp" / "vef_fontan_runs"
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


def stage_geometry(files: Dict[str, Path], data_root: Path) -> Dict[str, Path]:
    staged_paths: Dict[str, Path] = {}
    for subdir, source in files.items():
        if not source.exists():
            raise FileNotFoundError(f"Expected geometry file '{source}' is missing.")
        target_dir = "tmp" if subdir == "values" else subdir
        destination = data_root / target_dir / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"[stage] Copying {source} -> {destination}")
        shutil.copy2(source, destination)
        staged_paths[subdir] = destination
    return staged_paths


def _job_name(run_dir_name: str) -> str:
    suffix = run_dir_name.rsplit("_", 1)[-1]
    suffix = "".join(ch for ch in suffix if ch.isalnum())
    if len(suffix) > 6:
        suffix = suffix[-6:]
    return f"fontan-{suffix or 'run'}"


def _sanitize_job_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip())
    return cleaned or "fontan-run"


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
    job_name: Optional[str],
) -> Path:
    resolved_job_name = _sanitize_job_name(job_name) if job_name else _job_name(run_dir.name)
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={resolved_job_name}",
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

    sim_root = data_root
    tmp_dir = sim_root / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    solver_quoted = shlex.quote(str(solver_binary.resolve()))
    filename_quoted = shlex.quote(filename)
    data_root_quoted = shlex.quote(str(sim_root.resolve()))
    tmp_dir_rel = Path("sim_NSE") / "tmp"
    tmp_dir_quoted = shlex.quote(str(tmp_dir_rel))
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


def _submit_job(sbatch_path: Path) -> str:
    proc = subprocess.run(
        ["sbatch", "--parsable", sbatch_path.name],
        cwd=sbatch_path.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    job_id = proc.stdout.strip().split(";", 1)[0]
    if not job_id:
        raise RuntimeError("sbatch did not return a job id")
    return job_id


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
        check=False,
    )
    if squeue.returncode == 0:
        state = _first_nonempty(squeue.stdout.splitlines())
        if state:
            return state.upper()

    sacct = subprocess.run(
        ["sacct", "-j", job_id, "--format=State", "--parsable2", "--noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if sacct.returncode == 0:
        state = _first_nonempty(sacct.stdout.splitlines())
        if state:
            return state.split("|", 1)[0].upper()
    return None


def _wait_for_job(job_id: str, *, poll_interval: float) -> Optional[str]:
    last_state: Optional[str] = None
    missing_polls = 0
    while True:
        state = _query_job_state(job_id)
        if state:
            if state != last_state:
                print(f"[slurm] job {job_id} state: {state}")
            last_state = state
            missing_polls = 0
        else:
            missing_polls += 1
            if missing_polls == 1:
                print(f"[slurm] job {job_id} state unknown; retrying")

        if state in COMPLETED_STATES or state in FAILED_STATES:
            return state
        if missing_polls >= 10:
            print(f"[slurm] job {job_id} state unresolved after retries")
            return last_state
        time.sleep(poll_interval)


def collect_scalar(data_root: Path, case_name: str) -> Tuple[float, float, Path]:
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run VEF pipeline, voxelize with meshgen, export triplet, and submit sim_fontan via Slurm."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bump1-amp", type=float, default=1.0)
    parser.add_argument("--bump2-amp", type=float, default=3.5)
    parser.add_argument("--size-scale", type=float, default=0.9)
    parser.add_argument("--straighten-strength", type=float, default=0.3)
    parser.add_argument("--z-voxels", type=int, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "vef_meshgen",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        help="Path to the pre-built sim_fontan executable.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Root containing geometry/dimensions/tmp directories (sim_NSE).",
    )
    parser.add_argument(
        "--sim-resolution",
        type=int,
        help="Resolution identifier passed to sim_fontan (used for output naming).",
    )
    parser.add_argument("--job-name", help="Slurm job name")
    parser.add_argument("--time", default=DEFAULT_WALLTIME, help="Slurm walltime")
    parser.add_argument("--cpus", type=int, help="Slurm CPUs per task")
    parser.add_argument("--gpus", type=int, help="Slurm GPU count")
    parser.add_argument("--mem", help="Slurm memory request")
    parser.add_argument("--partition", help="Slurm partition")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=30.0,
        help="Seconds between job status checks",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.z_voxels <= 0:
        print("error: --z-voxels must be > 0", file=sys.stderr)
        return 1

    project_root = Path(__file__).resolve().parents[1]
    vef_root = project_root / "submodules" / "meshgen" / "vascular_encoding_framework"
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pitch = Z_EXTENT / float(args.z_voxels)
    config = load_sample_config(vef_root)
    update_run_kwargs(
        config["run_kwargs"],
        bump1_amp=args.bump1_amp,
        bump2_amp=args.bump2_amp,
        size_scale=args.size_scale,
        straighten_strength=args.straighten_strength,
        repair_pitch=pitch,
        output_dir=output_dir,
        keep_temp_files=False,
    )

    sys.path.insert(0, str(vef_root))
    from pipeline.vef_pipeline import run_from_config

    try:
        final_path, uid = run_from_config(config, base_dir=vef_root)
    except Exception as exc:
        print(f"error: VEF pipeline failed: {exc}", file=sys.stderr)
        return 1

    final_path = Path(final_path)
    if not final_path.exists():
        print(f"error: VEF pipeline did not produce STL: {final_path}", file=sys.stderr)
        return 1

    print(f"VEF STL: {final_path} (uid={uid})")
    print(f"Voxel pitch: {pitch:.6f}")

    voxels = voxelize_with_pitch(final_path, pitch)
    print(f"Voxel grid shape: {voxels.shape}")

    from meshgen.voxels import prepare_voxel_mesh_txt

    labeled = prepare_voxel_mesh_txt(
        voxels,
        expected_in_outs=EXPECTED_OUTS,
        num_type="int",
    )

    case_name = ensure_txt_suffix(f"vef_{uid}.txt")
    save_labeled_triplet(labeled, output_dir, case_name)

    files = {
        "geometry": output_dir / f"geom_{case_name}",
        "dimensions": output_dir / f"dim_{case_name}",
        "values": output_dir / f"val_{case_name}",
    }
    for label, path in files.items():
        if not path.exists():
            print(f"error: expected {label} file '{path}' was not created", file=sys.stderr)
            return 1

    run_dir = prepare_run_directory(project_root, case_name)
    print(f"[run] Solver artifacts directory: {run_dir}")

    if args.data_root is not None:
        data_root = Path(args.data_root).resolve()
        sim_link = run_dir / "sim_NSE"
        if sim_link.exists() and sim_link.resolve() != data_root:
            print(
                f"error: {sim_link} already exists and does not point to {data_root}",
                file=sys.stderr,
            )
            return 1
        if not sim_link.exists():
            sim_link.symlink_to(data_root, target_is_directory=True)
    else:
        data_root = (run_dir / "sim_NSE").resolve()

    for subdir in ("geometry", "dimensions", "tmp"):
        (data_root / subdir).mkdir(parents=True, exist_ok=True)

    stage_geometry(files, data_root)

    default_binary = (
        project_root / "submodules" / "tnl-lbm" / "build" / "sim_NSE" / "sim_fontan"
    )
    binary = (args.binary or default_binary).resolve()
    if not binary.is_file():
        print(f"error: solver binary '{binary}' not found", file=sys.stderr)
        return 1

    sim_resolution = args.sim_resolution if args.sim_resolution is not None else args.z_voxels
    try:
        sbatch_path = _write_sbatch(
            run_dir=run_dir,
            solver_binary=binary,
            resolution=sim_resolution,
            filename=case_name,
            data_root=data_root,
            partition=args.partition,
            gpus=args.gpus,
            cpus=args.cpus,
            mem=args.mem,
            walltime=args.time,
            job_name=args.job_name,
        )
    except OSError as exc:
        print(f"error: failed to write sbatch script: {exc}", file=sys.stderr)
        return 1

    try:
        job_id = _submit_job(sbatch_path)
    except (subprocess.SubprocessError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: failed to submit Slurm job: {exc}", file=sys.stderr)
        return 1

    print(f"[slurm] submitted job {job_id} (run dir: {run_dir})")
    state = _wait_for_job(job_id, poll_interval=args.poll_interval)
    if state in FAILED_STATES:
        print(f"error: job {job_id} failed with state {state}", file=sys.stderr)
        print(f"[slurm] stdout: {run_dir / 'slurm.out'}", file=sys.stderr)
        print(f"[slurm] stderr: {run_dir / 'slurm.err'}", file=sys.stderr)
        return 1

    try:
        time_value, scalar_value, val_path = collect_scalar(data_root, case_name)
    except Exception as exc:
        print(f"error: failed to collect scalar output: {exc}", file=sys.stderr)
        return 1

    print(f"[result] latest sample at t={time_value}: {scalar_value}")
    print(f"[result] source file: {val_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
