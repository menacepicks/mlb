from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

from .io import ensure_dir


def now_utc_stamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y%m%dT%H%M%SZ")


def archive_paths(base_dir: Path, stamp: str) -> dict[str, Path]:
    year = stamp[0:4]
    month = stamp[4:6]
    day = stamp[6:8]
    raw_dir = base_dir / "raw" / year / month / day / stamp
    board_dir = base_dir / "boards" / year / month / day / stamp
    ensure_dir(raw_dir)
    ensure_dir(board_dir)
    return {
        "raw dir": raw_dir,
        "board dir": board_dir,
        "manifest": raw_dir / "manifest.json",
    }


def write_json(path: Path, payload: dict) -> Path:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path
