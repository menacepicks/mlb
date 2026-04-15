from __future__ import annotations

from pathlib import Path
import shutil

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_table(df: pd.DataFrame, path: Path) -> Path:
    ensure_dir(path.parent)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        return path
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return path
    raise ValueError(f"Unsupported output format: {path}")


def copy_file(source: Path, target: Path) -> Path:
    ensure_dir(target.parent)
    shutil.copy2(source, target)
    return target
