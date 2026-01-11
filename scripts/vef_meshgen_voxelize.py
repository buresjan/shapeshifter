#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


Z_EXTENT = 90.779533  # Matches XY_maxZ in the VEF sample config.
EXPECTED_OUTS = {"W", "S", "N", "F"}


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run VEF pipeline from sample config, then voxelize with meshgen and view in Mayavi.",
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
    args = parser.parse_args()

    if args.z_voxels <= 0:
        raise ValueError("--z-voxels must be > 0.")

    repo_root = Path(__file__).resolve().parents[1]
    vef_root = repo_root / "submodules" / "meshgen" / "vascular_encoding_framework"
    output_dir = (repo_root / args.output_dir).resolve()
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

    final_path, uid = run_from_config(config, base_dir=vef_root)
    final_path = Path(final_path)
    if not final_path.exists():
        raise FileNotFoundError(f"VEF pipeline did not produce STL: {final_path}")

    print(f"VEF STL: {final_path} (uid={uid})")
    print(f"Voxel pitch: {pitch:.6f}")

    voxels = voxelize_with_pitch(final_path, pitch)
    print(f"Voxel grid shape: {voxels.shape}")

    from meshgen.voxels import prepare_voxel_mesh_txt
    from meshgen.utilities import vis

    labeled = prepare_voxel_mesh_txt(
        voxels,
        expected_in_outs=EXPECTED_OUTS,
        num_type="int",
    )
    save_labeled_triplet(labeled, output_dir, f"vef_{uid}.txt")

    vis(voxels)


if __name__ == "__main__":
    main()
