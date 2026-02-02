#!/usr/bin/env python3
"""Submit two VEF Fontan SR points via Slurm using nm_fontan_sr preset."""

from __future__ import annotations

import argparse
import copy
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from fontan_config import PROJECT_ROOT, load_config, resolve_path
from vef_meshgen_run_fontan import (
    EXPECTED_OUTS,
    Z_EXTENT,
    ensure_txt_suffix,
    load_mesh,
    save_labeled_triplet,
    stage_geometry,
    voxelize_with_pitch,
)


DEFAULT_CONFIG = Path("configs/vef/nm_fontan_sr.py")


@dataclass(frozen=True)
class Case:
    label: str
    params: dict[str, float]


CASES: tuple[Case, ...] = (
    Case(
        label="fontan_sr_nm",
        params={
            "bump1_amp": 0.0029998,
            "bump2_amp": 0.0142211,
            "size_scale": 0.9994,
            "straighten_strength": 0.0713,
            "offset_x": -0.00812767,
        },
    ),
    Case(
        label="fontan_sr_mads",
        params={
            "bump1_amp": 0.0,
            "bump2_amp": 1.35,
            "size_scale": 1.0,
            "straighten_strength": 0.4,
            "offset_x": 0.0,
        },
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit two VEF Fontan SR points via Slurm.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Fontan config .py/.json (used for solver/objective settings).",
    )
    parser.add_argument("--binary", type=Path, help="Path to sim_fontan_sr binary.")
    parser.add_argument("--z-voxels", type=int, help="Override z_voxels.")
    parser.add_argument("--sim-resolution", type=int, help="Override sim_resolution.")
    parser.add_argument("--runs-root", type=Path, help="Override runs_root.")
    parser.add_argument("--output-root", type=Path, help="Override output_root.")
    parser.add_argument("--partition", help="Slurm partition override.")
    parser.add_argument("--gpus", type=int, help="Slurm GPU override.")
    parser.add_argument("--cpus", type=int, help="Slurm CPUs per task override.")
    parser.add_argument("--mem", help="Slurm memory request override.")
    parser.add_argument("--walltime", help="Slurm walltime override.")
    parser.add_argument(
        "--cases",
        nargs="*",
        help="Subset of case labels to run (default: all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write geometry + sbatch scripts but do not submit.",
    )
    return parser.parse_args()


def _sanitize_label(label: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label.strip())
    return cleaned or "case"


def _prepare_run_dir(runs_root: Path, label: str) -> Path:
    runs_root.mkdir(parents=True, exist_ok=True)
    safe_label = _sanitize_label(label)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = f"{safe_label}_{timestamp}"
    suffix = 0
    while True:
        candidate = runs_root / (stem if suffix == 0 else f"{stem}_{suffix:02d}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            suffix += 1


def _load_vef_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"VEF config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("VEF config must be a JSON object.")
    if "run_kwargs" not in config or not isinstance(config["run_kwargs"], dict):
        raise ValueError("VEF config must define run_kwargs.")
    return config


def _extract_offset_y(config: dict) -> float:
    run_kwargs = config.get("run_kwargs")
    if not isinstance(run_kwargs, dict):
        raise ValueError("run_kwargs must be a mapping.")
    offset_xy = run_kwargs.get("offset_xy")
    if not isinstance(offset_xy, (list, tuple)) or len(offset_xy) != 2:
        raise ValueError("run_kwargs['offset_xy'] must contain exactly two values.")
    return float(offset_xy[1])


def _apply_design(
    config: dict,
    *,
    bump1_amp: float,
    bump2_amp: float,
    size_scale: float,
    straighten_strength: float,
    offset_x: float,
    offset_y: float,
    output_dir: Path,
    repair_pitch: float,
    keep_temp_files: bool,
) -> None:
    run_kwargs = config.get("run_kwargs")
    if not isinstance(run_kwargs, dict):
        raise ValueError("run_kwargs must be a mapping.")
    bumps = run_kwargs.get("bumps")
    if not isinstance(bumps, list) or len(bumps) < 2:
        raise ValueError("run_kwargs['bumps'] must contain at least two entries.")
    if not isinstance(bumps[0], dict) or not isinstance(bumps[1], dict):
        raise ValueError("run_kwargs['bumps'] entries must be mappings.")

    bumps[0]["amp"] = float(bump1_amp)
    bumps[1]["amp"] = float(bump2_amp)
    run_kwargs["size_scale"] = float(size_scale)
    run_kwargs["straighten_strength"] = float(straighten_strength)
    run_kwargs["offset_xy"] = [float(offset_x), float(offset_y)]
    run_kwargs["repair_pitch"] = float(repair_pitch)
    run_kwargs["output_dir"] = str(output_dir)
    run_kwargs["keep_temp_files"] = bool(keep_temp_files)


def _voxelize_to_triplet(
    stl_path: Path,
    pitch: float,
    output_dir: Path,
    case_name: str,
) -> dict[str, Path]:
    mesh = load_mesh(stl_path)
    voxels = voxelize_with_pitch(stl_path, pitch, mesh=mesh)

    from meshgen.voxels import prepare_voxel_mesh_txt

    labeled = prepare_voxel_mesh_txt(
        voxels,
        expected_in_outs=EXPECTED_OUTS,
        num_type="int",
    )
    save_labeled_triplet(labeled, output_dir, case_name)

    files = {
        "geometry": output_dir / f"geom_{case_name}",
        "dimensions": output_dir / f"dim_{case_name}",
        "values": output_dir / f"val_{case_name}",
    }
    for label, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Expected {label} file '{path}' was not created.")
    return files


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
    partition: str | None,
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


def _filter_cases(cases: Iterable[Case], selected: list[str] | None) -> list[Case]:
    if not selected:
        return list(cases)
    wanted = {_sanitize_label(label).lower() for label in selected}
    filtered = [case for case in cases if _sanitize_label(case.label).lower() in wanted]
    if not filtered:
        raise ValueError(f"No cases matched {selected}. Available: {[c.label for c in cases]}")
    return filtered


def _resolve_binary(solver_cfg: dict, override: Path | None) -> Path:
    if override is not None:
        resolved = resolve_path(override, base=PROJECT_ROOT)
        if resolved is None:
            raise FileNotFoundError(f"Solver binary '{override}' could not be resolved.")
        return resolved
    if solver_cfg.get("binary_path"):
        resolved = resolve_path(solver_cfg.get("binary_path"), base=PROJECT_ROOT)
        if resolved is None:
            raise FileNotFoundError(f"Solver binary '{solver_cfg.get('binary_path')}' not found.")
        return resolved
    binary_name = solver_cfg.get("binary_name") or "sim_fontan_sr"
    return (PROJECT_ROOT / "submodules" / "tnl-lbm" / "build" / "sim_NSE" / binary_name).resolve()


def main() -> int:
    args = _parse_args()

    cfg = load_config(args.config)
    objective_cfg = cfg.get("objective", {}) or {}
    solver_cfg = cfg.get("solver", {}) or {}
    slurm_cfg = solver_cfg.get("slurm", {}) or {}

    vef_config_path = resolve_path(objective_cfg.get("vef_config_path"), base=PROJECT_ROOT)
    if vef_config_path is None or not vef_config_path.is_file():
        print("error: objective.vef_config_path is missing or invalid", file=sys.stderr)
        return 1

    output_root = resolve_path(args.output_root or objective_cfg.get("output_root"), base=PROJECT_ROOT)
    runs_root = resolve_path(args.runs_root or objective_cfg.get("runs_root"), base=PROJECT_ROOT)
    if output_root is None or runs_root is None:
        print("error: output_root or runs_root is missing", file=sys.stderr)
        return 1

    z_voxels = int(args.z_voxels) if args.z_voxels is not None else int(
        objective_cfg.get("z_voxels", 0)
    )
    if z_voxels <= 0:
        print(f"error: invalid z_voxels {z_voxels}", file=sys.stderr)
        return 1
    sim_resolution = (
        int(args.sim_resolution)
        if args.sim_resolution is not None
        else int(objective_cfg.get("sim_resolution", z_voxels))
    )
    keep_temp_files = bool(objective_cfg.get("keep_temp_files", False))

    try:
        binary = _resolve_binary(solver_cfg, args.binary)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not binary.is_file():
        print(f"error: solver binary '{binary}' not found", file=sys.stderr)
        return 1
    if not binary.stat().st_mode & 0o111:
        print(f"error: solver binary '{binary}' is not executable", file=sys.stderr)
        return 1

    partition = args.partition if args.partition is not None else slurm_cfg.get("partition")
    gpus = args.gpus if args.gpus is not None else slurm_cfg.get("gpus", 1)
    cpus = args.cpus if args.cpus is not None else slurm_cfg.get("cpus", 4)
    mem = args.mem if args.mem is not None else slurm_cfg.get("mem", "8G")
    walltime = args.walltime if args.walltime is not None else slurm_cfg.get(
        "walltime", "22:00:00"
    )

    cases = _filter_cases(CASES, args.cases)

    vef_root = PROJECT_ROOT / "submodules" / "meshgen" / "vascular_encoding_framework"
    if str(vef_root) not in sys.path:
        sys.path.insert(0, str(vef_root))
    from pipeline.vef_pipeline import run_from_config

    base_config = _load_vef_config(vef_config_path)
    offset_y = _extract_offset_y(base_config)
    pitch = Z_EXTENT / float(z_voxels)

    for case in cases:
        run_dir = _prepare_run_dir(runs_root, case.label)
        output_dir = output_root / run_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)
        geometry_workdir = run_dir / "geometry_work"
        geometry_workdir.mkdir(parents=True, exist_ok=True)

        config = copy.deepcopy(base_config)
        _apply_design(
            config,
            bump1_amp=case.params["bump1_amp"],
            bump2_amp=case.params["bump2_amp"],
            size_scale=case.params["size_scale"],
            straighten_strength=case.params["straighten_strength"],
            offset_x=case.params["offset_x"],
            offset_y=offset_y,
            output_dir=output_dir,
            repair_pitch=pitch,
            keep_temp_files=keep_temp_files,
        )

        (run_dir / "params.json").write_text(
            json.dumps(
                {
                    "label": case.label,
                    "params": case.params,
                    "offset_y": offset_y,
                    "z_voxels": z_voxels,
                    "sim_resolution": sim_resolution,
                    "pitch": pitch,
                    "vef_config_path": str(vef_config_path),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (run_dir / "vef_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        try:
            final_path, uid = run_from_config(config, config_path=vef_config_path)
        except Exception as exc:
            print(f"error: VEF pipeline failed for {case.label}: {exc}", file=sys.stderr)
            return 1
        final_path = Path(final_path)
        if not final_path.is_file():
            print(
                f"error: VEF pipeline did not produce STL for {case.label}: {final_path}",
                file=sys.stderr,
            )
            return 1

        case_slug = _sanitize_label(case.label).lower()
        case_name = ensure_txt_suffix(f"{case_slug}_{uid}.txt")
        files = _voxelize_to_triplet(final_path, pitch, geometry_workdir, case_name)

        data_root = (run_dir / "sim_NSE").resolve()
        for subdir in ("geometry", "dimensions", "tmp"):
            (data_root / subdir).mkdir(parents=True, exist_ok=True)
        stage_geometry(files, data_root)

        sbatch_path = _write_sbatch(
            run_dir=run_dir,
            solver_binary=binary,
            resolution=sim_resolution,
            filename=case_name,
            data_root=data_root,
            partition=partition,
            gpus=gpus,
            cpus=cpus,
            mem=mem,
            walltime=walltime,
            job_name=case.label,
        )

        if args.dry_run:
            print(f"[dry-run] prepared {case.label}: {sbatch_path}")
            continue

        try:
            job_id = _submit_job(sbatch_path)
        except Exception as exc:
            print(f"error: failed to submit {case.label}: {exc}", file=sys.stderr)
            return 1
        print(f"[slurm] submitted {case.label}: job {job_id} (run dir: {run_dir})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
