#!/usr/bin/env python3
"""Sample junction geometries within safe bounds and ensure mesh generation succeeds.

This helper draws random parameter vectors inside the optimisation box used by
`optimize_junction_tcpc.py` and runs the meshgen pipeline without launching the
LBM solver. It is useful for quickly verifying that all sampled points produce
valid voxel meshes (i.e. Gmsh boolean operations succeed).

Usage
-----
    python playground/check_junction_geometries.py            # 10 random samples
    python playground/check_junction_geometries.py --samples 25 --keep-artifacts

The script prints a short report for each attempt. A failure stops the run and
leaves the workspace behind for inspection.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import List

import numpy as np

from run_junction_tcpc import ensure_txt_suffix, generate_geometry

# Safe design bounds mirrored from optimize_junction_tcpc.py
LOWER = np.array([-0.005, -10.0, -10.0, 0.0, 0.0], dtype=float)
UPPER = np.array([+0.005, +10.0, +10.0, 0.0015, 0.0015], dtype=float)
PARAM_NAMES = ("offset", "lower_angle", "upper_angle", "lower_flare", "upper_flare")
X0 = np.array([0.0, 0.0, 0.0, 0.00075, 0.00075], dtype=float)


@dataclass(frozen=True)
class SampleResult:
    index: int
    params: np.ndarray
    workspace: Path
    case_name: str


# ---------------------------------------------------------------------------
# Parameter generation helpers
# ---------------------------------------------------------------------------
def _sample_parameters(rng: random.Random) -> np.ndarray:
    """Draw a random parameter vector within [LOWER, UPPER]."""
    alpha = np.array([rng.random() for _ in range(len(PARAM_NAMES))], dtype=float)
    return LOWER + alpha * (UPPER - LOWER)


def _edge_parameter_sets() -> List[np.ndarray]:
    """Generate deterministic edge-case parameter vectors."""
    combos: List[np.ndarray] = []
    for toggles in product((0, 1), repeat=len(PARAM_NAMES)):
        mask = np.array(toggles, dtype=bool)
        vec = np.where(mask, UPPER, LOWER)
        combos.append(vec.astype(float))
    combos.append((LOWER + UPPER) * 0.5)  # midpoint
    combos.append(X0.copy())
    return combos


def _workspace_root(project_root: Path) -> Path:
    root = project_root / "tmp" / "geometry_smoke"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _run_sample(idx: int, project_root: Path, params: np.ndarray, keep: bool) -> SampleResult:
    case_name = ensure_txt_suffix(f"smoke_{idx:03d}")

    workspace = _workspace_root(project_root) / f"sample_{idx:03d}"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=False)

    offset, lower_angle, upper_angle, lower_flare, upper_flare = map(float, params)
    print(
        f"[sample {idx:03d}] offset={offset:.6f}, "
        f"lower_angle={lower_angle:.3f}, upper_angle={upper_angle:.3f}, "
        f"lower_flare={lower_flare:.6f}, upper_flare={upper_flare:.6f}"
    )

    prev_tmp = os.environ.get("TMPDIR")
    os.environ["TMPDIR"] = str(workspace)
    try:
        files, generated_case = generate_geometry(
            workspace,
            case_name,
            resolution=5,
            lower_angle=lower_angle,
            upper_angle=upper_angle,
            upper_flare=upper_flare,
            lower_flare=lower_flare,
            offset=offset,
            num_processes=1,
        )
    except Exception as exc:
        print(f"[sample {idx:03d}] FAILURE: {exc}")
        geo_candidates = sorted(workspace.glob("meshgen_geo_*/*.geo"))
        if geo_candidates:
            print(f"[sample {idx:03d}] Filled GEO preserved at {geo_candidates[-1]}")
        print(f"[sample {idx:03d}] workspace retained at {workspace}")
        raise
    finally:
        if prev_tmp is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = prev_tmp

    print(
        f"[sample {idx:03d}] SUCCESS -> {generated_case} "
        f"(geom={files['geometry'].name}, dim={files['dimensions'].name})"
    )

    if not keep:
        shutil.rmtree(workspace, ignore_errors=True)
    return SampleResult(index=idx, params=params, workspace=workspace, case_name=generated_case)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample TCPC junction geometries within safe bounds."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of random geometries to generate (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--mode",
        choices=("random", "edges", "both"),
        default="random",
        help="Geometry selection strategy: random sampling, deterministic edge sweep, or both.",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Retain meshgen workspaces under tmp/geometry_smoke/ for inspection.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    rng = random.Random(args.seed)
    project_root = Path(__file__).resolve().parents[1]

    params_to_test: List[np.ndarray] = []
    if args.mode in ("edges", "both"):
        params_to_test.extend(_edge_parameter_sets())
        print(f"[info] Added {len(params_to_test)} edge-case geometries.")
    if args.mode in ("random", "both"):
        if args.samples <= 0:
            print("[info] No random samples requested.")
        else:
            for _ in range(args.samples):
                params_to_test.append(_sample_parameters(rng))
            print(f"[info] Added {args.samples} random geometries.")

    if not params_to_test:
        print("No geometries to generate; exiting.")
        return 0

    successes: List[SampleResult] = []
    try:
        for idx, params in enumerate(params_to_test, start=1):
            result = _run_sample(idx, project_root, params, keep=args.keep_artifacts)
            successes.append(result)
    except Exception:
        return 1

    print(f"\nCompleted {len(successes)} samples without geometry failures.")
    if args.keep_artifacts:
        for res in successes:
            print(f"  sample {res.index:03d}: workspace={res.workspace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
