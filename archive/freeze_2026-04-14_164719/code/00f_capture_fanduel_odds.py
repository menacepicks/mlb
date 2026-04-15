from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd


THIS_FILE = Path(__file__).resolve()


def _discover_project_root(start_file: Path) -> Path:
    candidates = [start_file.parent, *start_file.parents]
    for candidate in candidates:
        if (candidate / "src").exists() or (candidate / "config.py").exists():
            return candidate
    return start_file.parent


ROOT = _discover_project_root(THIS_FILE)
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _import_local_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name!r} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _import_fanduel_adapter() -> Any:
    local_candidate = ROOT / "fanduel.py"
    if local_candidate.exists():
        return _import_local_module("local_fanduel_adapter", local_candidate)

    for module_name in (
        "mlb_pipeline.book_adapters.fanduel",
        "nba_pipeline.book_adapters.fanduel",
    ):
        try:
            return importlib.import_module(module_name)
        except ImportError:
            continue

    raise ImportError(
        "Could not import a FanDuel adapter. Place fanduel.py next to this script or expose "
        "mlb_pipeline.book_adapters.fanduel / nba_pipeline.book_adapters.fanduel on PYTHONPATH."
    )


fd = _import_fanduel_adapter()


def _load_yaml_fallback(rel_path: str) -> dict[str, Any]:
    candidate = ROOT / rel_path
    if not candidate.exists():
        return {}

    try:
        import yaml  # type: ignore
    except ImportError:
        return {}

    data = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _get_load_yaml() -> Callable[[str], dict[str, Any]]:
    try:
        module = importlib.import_module("mlb_pipeline.config")
        fn = getattr(module, "load_yaml", None)
        if callable(fn):
            return fn
    except ImportError:
        pass
    return _load_yaml_fallback


load_yaml = _get_load_yaml()


def _write_parquet_fallback(df: pd.DataFrame, out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _get_write_parquet() -> Callable[[pd.DataFrame, str | Path], None]:
    for module_name in ("mlb_pipeline.io", "nba_pipeline.io"):
        try:
            module = importlib.import_module(module_name)
            fn = getattr(module, "write_parquet", None)
            if callable(fn):
                return fn
        except ImportError:
            continue
    return _write_parquet_fallback


write_parquet = _get_write_parquet()


def _pick_attr(module: Any, names: list[str]) -> Any:
    for name in names:
        obj = getattr(module, name, None)
        if callable(obj):
            return obj
    return None


def _call_with_fallbacks(fn: Any, *, sport: str, from_url: str | None, use_browser: bool) -> Any:
    attempts = [
        {"sport": sport, "from_url": from_url, "use_browser": use_browser},
        {"sport": sport, "from_url": from_url},
        {"sport": sport, "use_browser": use_browser},
        {"sport": sport},
    ]
    last_err: Exception | None = None
    for kwargs in attempts:
        try:
            sig = inspect.signature(fn)
            supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
            return fn(**supported)
        except TypeError as err:
            last_err = err
            continue
    if last_err is not None:
        raise last_err
    return fn()


def _coerce_payload_candidate(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return [value]
    return value


def _payload_candidates(payloads: Any) -> list[Any]:
    candidates: list[Any] = []
    seen: set[int] = set()

    def add(value: Any) -> None:
        if value is None:
            return
        obj = _coerce_payload_candidate(value)
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)
        candidates.append(obj)

    add(payloads)

    preferred_attrs = [
        "payloads",
        "responses",
        "raw_payloads",
        "raw_responses",
        "events",
        "data",
        "items",
        "records",
        "pages",
        "results",
    ]

    for attr in preferred_attrs:
        if hasattr(payloads, attr):
            add(getattr(payloads, attr))

    if is_dataclass(payloads):
        for field in fields(payloads):
            add(getattr(payloads, field.name, None))

    if hasattr(payloads, "__dict__"):
        for value in vars(payloads).values():
            add(value)

    return candidates


def _normalize_payloads(normalize_fn: Any, payloads: Any, sport: str) -> pd.DataFrame:
    errors: list[str] = []

    for candidate in _payload_candidates(payloads):
        if isinstance(candidate, pd.DataFrame):
            return candidate.copy()

        attempts = [
            lambda: normalize_fn(payloads=candidate, sport=sport),
            lambda: normalize_fn(payloads=candidate),
            lambda: normalize_fn(candidate, sport),
            lambda: normalize_fn(candidate),
        ]

        for attempt in attempts:
            try:
                result = attempt()
                if isinstance(result, pd.DataFrame):
                    return result
                return pd.DataFrame(result)
            except TypeError as err:
                errors.append(str(err))
                continue

    bundle_type = type(payloads).__name__
    candidate_types = [type(c).__name__ for c in _payload_candidates(payloads)]
    raise TypeError(
        "Could not normalize captured FanDuel payloads. "
        f"bundle_type={bundle_type}; candidates={candidate_types}; "
        f"recent_errors={errors[-4:]}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", required=True)
    parser.add_argument("--from-url", default=None)
    parser.add_argument("--use-browser", action="store_true")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def _resolve_out_path(args: argparse.Namespace) -> Path:
    if args.out:
        return Path(args.out)

    paths = load_yaml("config/paths.yaml")
    default_out = paths.get("fanduel_shared_odds_parquet", "artifacts/fanduel_mlb_shared_odds.parquet")
    return Path(default_out)


def main() -> None:
    args = parse_args()
    out_path = _resolve_out_path(args)

    capture_fn = _pick_attr(
        fd,
        [
            "capture_fanduel_payloads",
            "capture_fanduel_event_page_payloads",
            "capture_event_page_payloads",
            "fetch_fanduel_payloads",
            "capture_payloads",
        ],
    )
    normalize_fn = _pick_attr(
        fd,
        [
            "normalize_fanduel_payloads",
            "normalize_fanduel_payloads_to_shared",
            "normalize_payloads_to_shared",
            "normalize_fanduel_shared_odds",
        ],
    )

    if capture_fn is None:
        raise ImportError(
            "No compatible FanDuel capture function found. Expected one of: "
            "capture_fanduel_payloads, capture_fanduel_event_page_payloads, "
            "capture_event_page_payloads, fetch_fanduel_payloads, capture_payloads"
        )

    result = _call_with_fallbacks(
        capture_fn,
        sport=args.sport,
        from_url=args.from_url,
        use_browser=args.use_browser,
    )

    meta = {"source": "requests", "urls": 0, "browser": False}

    if isinstance(result, tuple) and len(result) == 2:
        payloads, maybe_meta = result
        if isinstance(maybe_meta, dict):
            meta.update(maybe_meta)
    else:
        payloads = result

    if isinstance(payloads, pd.DataFrame):
        df = payloads.copy()
    else:
        if normalize_fn is None:
            raise ImportError(
                "Capture worked but no compatible FanDuel normalize function found. Expected one of: "
                "normalize_fanduel_payloads, normalize_fanduel_payloads_to_shared, "
                "normalize_payloads_to_shared, normalize_fanduel_shared_odds"
            )
        df = _normalize_payloads(normalize_fn, payloads, args.sport)

    write_parquet(df, out_path)
    print(f"wrote {out_path} with {len(df):,} rows")
    print(
        f"source={meta.get('source', 'requests')} urls={meta.get('urls', 0)} "
        f"browser={meta.get('browser', False)}"
    )


if __name__ == "__main__":
    main()