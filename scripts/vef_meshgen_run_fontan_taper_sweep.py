#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_SCALES = "0.5,0.6,0.7,0.8"


def _parse_scales(raw: str) -> list[float]:
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        raise ValueError("No scales provided.")
    scales: list[float] = []
    for token in tokens:
        try:
            scales.append(float(token))
        except ValueError as exc:
            raise ValueError(f"Invalid scale '{token}'. Expected a float.") from exc
    return scales


def _has_taper_override(args: list[str]) -> bool:
    for token in args:
        if token == "--taper-target-scale" or token.startswith("--taper-target-scale="):
            return True
    return False


def _parse_args(argv: list[str] | None = None) -> tuple[list[float], list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run vef_meshgen_run_fontan.py sequentially across taper_target_scale values."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scales",
        default=DEFAULT_SCALES,
        help="Comma-separated taper_target_scale values.",
    )
    args, passthrough = parser.parse_known_args(argv)

    try:
        scales = _parse_scales(args.scales)
    except ValueError as exc:
        parser.error(str(exc))

    if _has_taper_override(passthrough):
        parser.error("Do not pass --taper-target-scale here; it is set per run.")

    return scales, passthrough


def main(argv: list[str] | None = None) -> int:
    scales, passthrough = _parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    runner = repo_root / "scripts" / "vef_meshgen_run_fontan.py"
    if not runner.is_file():
        print(f"error: expected runner script at '{runner}'", file=sys.stderr)
        return 1

    for idx, scale in enumerate(scales, start=1):
        cmd = [
            sys.executable,
            str(runner),
            "--taper-target-scale",
            str(scale),
            *passthrough,
        ]
        print(f"[sweep] {idx}/{len(scales)} taper_target_scale={scale}")
        print("[sweep] running:", " ".join(shlex.quote(part) for part in cmd))
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            print(
                f"error: run failed for taper_target_scale={scale} "
                f"(exit {proc.returncode})",
                file=sys.stderr,
            )
            return proc.returncode

    print("[sweep] all runs submitted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
