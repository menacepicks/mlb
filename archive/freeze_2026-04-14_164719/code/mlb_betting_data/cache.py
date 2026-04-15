from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

from .io import ensure_dir


def build_file_signature(paths: Iterable[Path]) -> dict[str, object]:
    items: list[dict[str, object]] = []
    for path in sorted((Path(p) for p in paths), key=lambda p: str(p.resolve())):
        stat = path.stat()
        items.append(
            {
                "path": str(path.resolve()),
                "name": path.name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return {"files": items}


def signature_hash(signature: dict[str, object]) -> str:
    payload = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, object]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
