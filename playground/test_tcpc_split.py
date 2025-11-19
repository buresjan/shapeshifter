#!/usr/bin/env python3
"""Manually run the TCPC split wrapper for an existing solver run directory.

This helper mirrors the invocation performed inside ``optimize_junction_tcpc`` so
you can reprocess historical runs or sanity-check ParaView without triggering the
full optimisation loop. It auto-discovers the ADIOS2 dataset inside the selected
run directory and writes a fresh CSV into ``run_dir/tcpc_split/...``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from optimize_junction_tcpc import (  # type: ignore
    TCPC_SPLIT_TIME_INDEX,
    TCPC_SPLIT_WRITE_DEBUG_POINTS,
    TCPC_SPLIT_WRITE_VTP,
    _discover_bp_path,
)

_DEFAULT_RUN_DIR = (
    Path(__file__).resolve().parents[1]
    / "submodules"
    / "tnl-lbm"
    / "runs_tcpc"
    / "run_33133a2e_junction_dcde89c39477_20251117-080134"
)
_DEFAULT_CASE_STEM = "run_33133a2e_junction_dcde89c39477"


def _default_case_stem(run_dir: Path) -> str:
    """Best-effort inference of the case stem from the run directory name."""

    stem = run_dir.name
    # Strip the trailing timestamp suffix "_YYYYMMDD-HHMMSS[_NN]" if present.
    if "_20" in stem:
        prefix, _, suffix = stem.rpartition("_20")
        if len(suffix) >= 6:
            return prefix
    return stem


def _sanitize_label(raw: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in raw)
    return cleaned.strip("_-") or "manual_test"


def run_split(
    *,
    run_dir: Path,
    case_stem: str,
    pvpython: str,
    wrapper: Path,
    output_label: str,
    time_index: int,
    write_vtp: bool,
    write_debug: bool,
) -> Path:
    """Invoke the tcpc_split wrapper for ``run_dir`` and return the CSV path."""

    bp_path = _discover_bp_path(run_dir, case_stem)
    print(f"[split-test] Using ADIOS2 dataset: {bp_path}")

    output_dir = run_dir / "tcpc_split" / f"{case_stem}_{output_label}"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_basename = f"{case_stem}_split_{output_label}.csv"

    cmd = [
        pvpython,
        str(wrapper),
        "--bp-file",
        str(bp_path),
        "--time-index",
        str(time_index),
        "--output-dir",
        str(output_dir),
        "--csv-basename",
        csv_basename,
        "--quiet",
    ]
    cmd.append("--write-vtp" if write_vtp else "--no-write-vtp")
    cmd.append("--write-debug-points" if write_debug else "--no-write-debug-points")
    print(f"[split-test] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(run_dir))

    csv_path = output_dir / csv_basename
    if not csv_path.is_file():
        raise FileNotFoundError(f"tcpc_split did not produce CSV '{csv_path}'")
    return csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=_DEFAULT_RUN_DIR,
        help=(
            "Path to the solver run directory under tnl-lbm/runs_tcpc/... "
            f"(default: {_DEFAULT_RUN_DIR})"
        ),
    )
    parser.add_argument(
        "--case-stem",
        type=str,
        default=_DEFAULT_CASE_STEM,
        help=(
            "Case stem (e.g. run_xxx_junction_xxx). "
            f"Defaults to '{_DEFAULT_CASE_STEM}' or run-dir inference."
        ),
    )
    parser.add_argument(
        "--output-label",
        default="wrapper_test",
        help="Suffix appended to tcpc_split outputs (default: %(default)s).",
    )
    parser.add_argument(
        "--pvpython",
        default=os.environ.get("PV_PYTHON", "pvpython"),
        help="pvpython binary to use (default: env PV_PYTHON or 'pvpython').",
    )
    parser.add_argument(
        "--wrapper",
        type=Path,
        default=Path(__file__).resolve().parent / "tcpc_split_wrapper.py",
        help="Path to the tcpc_split wrapper script.",
    )
    parser.add_argument(
        "--time-index",
        type=int,
        default=TCPC_SPLIT_TIME_INDEX,
        help="Time index forwarded to tcpc_split (default: optimise script setting).",
    )
    parser.add_argument(
        "--write-vtp",
        action="store_true",
        default=TCPC_SPLIT_WRITE_VTP,
        help="Force enabling VTP QA outputs.",
    )
    parser.add_argument(
        "--no-write-vtp",
        dest="write_vtp",
        action="store_false",
        help="Disable VTP QA outputs.",
    )
    parser.add_argument(
        "--write-debug-points",
        action="store_true",
        default=TCPC_SPLIT_WRITE_DEBUG_POINTS,
        help="Force enabling debug point dumps.",
    )
    parser.add_argument(
        "--no-write-debug-points",
        dest="write_debug_points",
        action="store_false",
        help="Disable debug point dumps.",
    )
    args = parser.parse_args()
    if not args.case_stem:
        args.case_stem = _default_case_stem(args.run_dir.resolve())
    args.output_label = _sanitize_label(args.output_label)
    return args


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory '{run_dir}' not found")

    csv_path = run_split(
        run_dir=run_dir,
        case_stem=args.case_stem,
        pvpython=args.pvpython,
        wrapper=args.wrapper.resolve(),
        output_label=args.output_label,
        time_index=args.time_index,
        write_vtp=args.write_vtp,
        write_debug=args.write_debug_points,
    )
    print(f"[split-test] Completed. CSV stored at: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
