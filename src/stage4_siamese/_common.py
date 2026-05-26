from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ["PIGEONVISION_DATA"]) if "PIGEONVISION_DATA" in os.environ else ROOT


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_root_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def resolve_data_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return DATA_ROOT / path


def resolve_existing(*candidates: str | Path) -> Path:
    resolved = [resolve_root_path(candidate) for candidate in candidates]
    for candidate in resolved:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("None of the candidate paths exist: " + ", ".join(str(p) for p in resolved))
