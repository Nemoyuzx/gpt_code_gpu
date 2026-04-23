from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _resolve_path(path_value: str | os.PathLike[str] | None, default_relative: str) -> Path:
    raw = Path(path_value).expanduser() if path_value else Path(default_relative)
    if not raw.is_absolute():
        raw = PROJECT_ROOT / raw
    return raw


OUTPUT_ROOT = _resolve_path(os.getenv("OUTPUT_ROOT"), "outputs")
LOG_OUTPUT_DIR = _resolve_path(os.getenv("LOG_OUTPUT_DIR"), str(OUTPUT_ROOT / "logs"))
VIS_OUTPUT_DIR = _resolve_path(os.getenv("VIS_OUTPUT_DIR"), str(OUTPUT_ROOT / "visualization"))
DATA_OUTPUT_DIR = _resolve_path(os.getenv("DATA_OUTPUT_DIR"), str(OUTPUT_ROOT / "data"))

BLE_PARSED_LOG = LOG_OUTPUT_DIR / "ble_parsed.log"
BLE_RAW_LOG = LOG_OUTPUT_DIR / "ble_raw.log"
MAP_IMAGE = VIS_OUTPUT_DIR / "map.png"
FINAL_MAP_IMAGE = VIS_OUTPUT_DIR / "final_map.png"
PATH_CSV = DATA_OUTPUT_DIR / "path.csv"
MEM_USAGE_CSV = DATA_OUTPUT_DIR / "mem_usage_log.csv"
MEM_USAGE_PLOT = VIS_OUTPUT_DIR / "mem_usage_plot.png"


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path: str | os.PathLike[str]) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
