#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_CASES = "22:0.9,22:1.0,19:1.0,17:1.0,15:1.0"


def _parse_cases(raw: str) -> list[tuple[float, float]]:
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        raise ValueError("No cases provided.")
    cases: list[tuple[float, float]] = []
    for token in tokens:
        if ":" not in token:
            raise ValueError(
                f"Invalid case '{token}'. Expected 'taper_length_mm:taper_target_scale'."
            )
        length_raw, scale_raw = token.split(":", 1)
        try:
            length_mm = float(length_raw)
            scale = float(scale_raw)
        except ValueError as exc:
            raise ValueError(
                f"Invalid case '{token}'. Expected numeric 'taper_length_mm:taper_target_scale'."
            ) from exc
        cases.append((length_mm, scale))
    return cases


def _has_taper_override(args: list[str]) -> bool:
    for token in args:
        if token in ("--taper-target-scale", "--taper-length-mm"):
            return True
        if token.startswith("--taper-target-scale=") or token.startswith("--taper-length-mm="):
            return True
    return False


def _parse_args(argv: list[str] | None = None) -> tuple[list[tuple[float, float]], list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run vef_meshgen_run_fontan.py sequentially across taper_length_mm and "
            "taper_target_scale case pairs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cases",
        default=DEFAULT_CASES,
        help="Comma-separated taper_length_mm:taper_target_scale pairs.",
    )
    args, passthrough = parser.parse_known_args(argv)

    try:
        cases = _parse_cases(args.cases)
    except ValueError as exc:
        parser.error(str(exc))

    if _has_taper_override(passthrough):
        parser.error(
            "Do not pass --taper-length-mm or --taper-target-scale here; they are set per run."
        )

    return cases, passthrough


def main(argv: list[str] | None = None) -> int:
    cases, passthrough = _parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    runner = repo_root / "scripts" / "vef_meshgen_run_fontan.py"
    if not runner.is_file():
        print(f"error: expected runner script at '{runner}'", file=sys.stderr)
        return 1

    for idx, (length_mm, scale) in enumerate(cases, start=1):
        cmd = [
            sys.executable,
            str(runner),
            "--taper-length-mm",
            str(length_mm),
            "--taper-target-scale",
            str(scale),
            *passthrough,
        ]
        print(
            f"[sweep] {idx}/{len(cases)} taper_length_mm={length_mm} "
            f"taper_target_scale={scale}"
        )
        print("[sweep] running:", " ".join(shlex.quote(part) for part in cmd))
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            print(
                f"error: run failed for taper_length_mm={length_mm} "
                f"taper_target_scale={scale} (exit {proc.returncode})",
                file=sys.stderr,
            )
            return proc.returncode

    print("[sweep] all runs submitted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
