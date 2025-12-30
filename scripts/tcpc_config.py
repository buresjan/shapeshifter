#!/usr/bin/env python3
"""Helpers for loading TCPC config files."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path | str) -> Dict[str, Any]:
    """Load a config dict from a Python or JSON file."""
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file '{cfg_path}' does not exist")

    if cfg_path.suffix == ".py":
        spec = importlib.util.spec_from_file_location("tcpc_config", str(cfg_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load config module from '{cfg_path}'")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[arg-type]
        if not hasattr(module, "CONFIG"):
            raise AttributeError(f"Config module '{cfg_path}' must define CONFIG")
        config = getattr(module, "CONFIG")
    elif cfg_path.suffix == ".json":
        config = json.loads(cfg_path.read_text(encoding="utf-8"))
    else:
        raise ValueError("Config file must be .py or .json")

    if not isinstance(config, dict):
        raise TypeError(f"Config in '{cfg_path}' must be a dict")

    config = dict(config)
    config["_config_path"] = cfg_path
    return config


def resolve_path(value: str | Path | None, *, base: Path | None = None) -> Path | None:
    """Resolve a config path value relative to base if needed."""
    if value is None:
        return None
    raw = value if isinstance(value, Path) else Path(value)
    if raw.is_absolute():
        return raw
    root = base or PROJECT_ROOT
    return (root / raw).resolve()
