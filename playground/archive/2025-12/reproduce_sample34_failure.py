#!/usr/bin/env python3
"""Reproduce the TCPC geometry failure observed at sample 034.

This script attempts to generate the junction geometry using the parameter
vector that triggered Gmsh's `BOPAlgo_AlertBuilderFailed` during the edge-case
sweep. It captures the exception, reports it, and leaves all temporary
artifacts under `tmp/sample34_failure/` for inspection.
"""

from __future__ import annotations

import os
from pathlib import Path

from run_junction_tcpc import ensure_txt_suffix, generate_geometry

# Sample 034 parameters (offset, lower_angle, upper_angle, lower_flare, upper_flare)
PARAMS = {
    "offset": 0.0005,
    "lower_angle": 3.0,
    "upper_angle": -2.0,
    "lower_flare": 0.001,
    "upper_flare": 0.001,
}

RESOLUTION = 5


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    workspace = project_root / "tmp" / "sample34_failure"
    if workspace.exists():
        print(f"Cleaning previous workspace: {workspace}")
        import shutil
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    case_name = ensure_txt_suffix("sample34_case")
    print("Attempting geometry generation with parameters:")
    for key, value in PARAMS.items():
        print(f"  {key:>12s} = {value:.6f}")

    prev_tmp = os.environ.get("TMPDIR")
    os.environ["TMPDIR"] = str(workspace)
    try:
        files, generated_case = generate_geometry(
            workspace,
            case_name,
            resolution=RESOLUTION,
            lower_angle=PARAMS["lower_angle"],
            upper_angle=PARAMS["upper_angle"],
            upper_flare=PARAMS["upper_flare"],
            lower_flare=PARAMS["lower_flare"],
            offset=PARAMS["offset"],
            num_processes=1,
        )
    except Exception as exc:
        print(f"\nGeometry generation FAILED: {exc}")
        filled_geo = sorted(workspace.glob("meshgen_geo_*/*.geo"))
        if filled_geo:
            print(f"Filled GEO file: {filled_geo[-1]}")
        print(f"Workspace retained at: {workspace}")
        return 1
    finally:
        if prev_tmp is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = prev_tmp

    print("\nGeometry succeeded unexpectedly. Generated files:")
    for label, path in files.items():
        print(f"  {label:>10s}: {path}")
    print(f"Case name: {generated_case}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
