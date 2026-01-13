#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import time


DEFAULT_CASES = "22:0.9,22:1.0,19:1.0,17:1.0,15:1.0"
DEFAULT_OUTPUT_DIR = Path("data") / "vef_meshgen"
_UID_RE = re.compile(r"\(uid=([^)]+)\)")
_GEOM_UID_RE = re.compile(r"geom_vef_(.+)\.txt$")
_SLURM_RE = re.compile(r"\[slurm\] submitted job (\S+)")
_RUN_DIR_RE = re.compile(r"\[slurm\] submitted job \S+ \(run dir: (.+)\)")
_RUN_DIR_FALLBACK_RE = re.compile(r"\[run\] Solver artifacts directory: (.+)")


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
        if token in ("--taper-target-scale", "--taper-length-mm", "--outlet-yz-minx"):
            return True
        if token.startswith("--taper-target-scale=") or token.startswith("--taper-length-mm="):
            return True
        if token.startswith("--outlet-yz-minx="):
            return True
    return False


def _infer_output_dir(passthrough: list[str], repo_root: Path) -> Path:
    raw_value: str | None = None
    for idx, token in enumerate(passthrough):
        if token == "--output-dir":
            if idx + 1 < len(passthrough):
                raw_value = passthrough[idx + 1]
                break
        if token.startswith("--output-dir="):
            raw_value = token.split("=", 1)[1]
            break
    resolved = repo_root / (Path(raw_value) if raw_value else DEFAULT_OUTPUT_DIR)
    return resolved.resolve()


def _parse_run_output(output: str) -> dict[str, str | None]:
    uid = None
    job_id = None
    run_dir = None
    for line in output.splitlines():
        if uid is None:
            match = _UID_RE.search(line)
            if match:
                uid = match.group(1)
        if job_id is None:
            match = _SLURM_RE.search(line)
            if match:
                job_id = match.group(1)
        if run_dir is None:
            match = _RUN_DIR_RE.search(line)
            if match:
                run_dir = match.group(1)
                continue
            match = _RUN_DIR_FALLBACK_RE.search(line)
            if match:
                run_dir = match.group(1)
    return {"uid": uid, "job_id": job_id, "run_dir": run_dir}


def _infer_uid_from_output_dir(output_dir: Path, before: set[Path]) -> str | None:
    after = set(output_dir.glob("geom_vef_*.txt"))
    new_files = after - before
    if not new_files:
        return None
    latest = max(new_files, key=lambda path: path.stat().st_mtime)
    match = _GEOM_UID_RE.match(latest.name)
    if not match:
        return None
    return match.group(1)


def _infer_run_dir(run_root: Path, before: set[Path], start_time: float) -> str | None:
    if not run_root.exists():
        return None
    after = {path for path in run_root.iterdir() if path.is_dir()}
    new_dirs = after - before
    if new_dirs:
        latest = max(new_dirs, key=lambda path: path.stat().st_mtime)
        return str(latest)
    candidates = [path for path in after if path.stat().st_mtime >= start_time - 1.0]
    if candidates:
        latest = max(candidates, key=lambda path: path.stat().st_mtime)
        return str(latest)
    return None


def _load_default_taper_config(repo_root: Path) -> tuple[float, float]:
    sample_path = (
        repo_root
        / "submodules"
        / "meshgen"
        / "vascular_encoding_framework"
        / "vef_config.sample.json"
    )
    if not sample_path.is_file():
        raise FileNotFoundError(f"VEF sample config not found: {sample_path}")
    with sample_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("VEF sample config must be a JSON object.")
    run_kwargs = config.get("run_kwargs")
    if not isinstance(run_kwargs, dict):
        raise ValueError("VEF sample config must contain a run_kwargs object.")

    taper_length_mm = run_kwargs.get("taper_length_mm")
    outlet_plane_targets = run_kwargs.get("outlet_plane_targets")
    if taper_length_mm is None:
        raise ValueError("VEF sample config is missing run_kwargs['taper_length_mm'].")
    if not isinstance(outlet_plane_targets, dict):
        raise ValueError("VEF sample config must define outlet_plane_targets as a mapping.")
    yz_minx = outlet_plane_targets.get("YZ_minX")
    if yz_minx is None:
        raise ValueError("VEF sample config is missing outlet_plane_targets['YZ_minX'].")

    return float(taper_length_mm), float(yz_minx)


def _parse_args(argv: list[str] | None = None) -> tuple[list[tuple[float, float]], list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run vef_meshgen_run_fontan.py sequentially across taper_length_mm and "
            "taper_target_scale case pairs, adjusting YZ_minX to match taper length."
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
    output_dir = _infer_output_dir(passthrough, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_root = repo_root / "tmp" / "vef_fontan_runs"
    run_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    index_path = output_dir / f"taper_length_sweep_index_{timestamp}.json"
    sweep_meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "defaults": {},
        "cases": [],
    }
    try:
        default_length_mm, default_yz_minx = _load_default_taper_config(repo_root)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    sweep_meta["defaults"] = {
        "taper_length_mm": default_length_mm,
        "outlet_YZ_minX": default_yz_minx,
    }
    runner = repo_root / "scripts" / "vef_meshgen_run_fontan.py"
    if not runner.is_file():
        print(f"error: expected runner script at '{runner}'", file=sys.stderr)
        return 1
    print(f"[sweep] writing case index to {index_path}")

    for idx, (length_mm, scale) in enumerate(cases, start=1):
        pre_geom = {path.resolve() for path in output_dir.glob("geom_vef_*.txt")}
        pre_run_dirs = {path.resolve() for path in run_root.iterdir() if path.is_dir()}
        start_time = time()
        yz_minx = default_yz_minx + (default_length_mm - length_mm)
        cmd = [
            sys.executable,
            str(runner),
            "--taper-length-mm",
            str(length_mm),
            "--taper-target-scale",
            str(scale),
            "--outlet-yz-minx",
            str(yz_minx),
            *passthrough,
        ]
        print(
            f"[sweep] {idx}/{len(cases)} taper_length_mm={length_mm} "
            f"taper_target_scale={scale} outlet_YZ_minX={yz_minx:.6f}"
        )
        print("[sweep] running:", " ".join(shlex.quote(part) for part in cmd))
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)

        combined_output = "\n".join(filter(None, [proc.stdout, proc.stderr]))
        parsed = _parse_run_output(combined_output)
        uid = parsed["uid"] or _infer_uid_from_output_dir(output_dir, pre_geom)
        job_id = parsed["job_id"]
        run_dir = parsed["run_dir"] or _infer_run_dir(run_root, pre_run_dirs, start_time)
        case_record = {
            "case_index": idx,
            "taper_length_mm": length_mm,
            "taper_target_scale": scale,
            "outlet_YZ_minX": yz_minx,
            "uid": uid,
            "case_name": f"vef_{uid}.txt" if uid else None,
            "slurm_job_id": job_id,
            "run_dir": run_dir,
            "returncode": proc.returncode,
        }
        sweep_meta["cases"].append(case_record)
        sweep_meta["updated_at"] = datetime.now().isoformat(timespec="seconds")
        index_path.write_text(json.dumps(sweep_meta, indent=2), encoding="ascii")
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
