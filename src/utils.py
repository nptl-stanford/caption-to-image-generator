"""Shared helpers for the generation and classification pipeline."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import yaml


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand $VAR / ${VAR} references (e.g. $SCRATCH) in string values."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    return value


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file (e.g. configs/generation.yaml)."""
    with open(path, "r", encoding="utf-8") as f:
        return _expand_env_vars(yaml.safe_load(f))


def ensure_dir(path: Path) -> Path:
    """Create a directory (and parents) if it doesn't exist, return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_manifest_row(manifest_path: Path, row: dict[str, Any]) -> None:
    """Append one row to a CSV manifest, writing the header if the file is new."""
    ensure_dir(manifest_path.parent)
    is_new = not manifest_path.exists() or manifest_path.stat().st_size == 0
    with open(manifest_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)
