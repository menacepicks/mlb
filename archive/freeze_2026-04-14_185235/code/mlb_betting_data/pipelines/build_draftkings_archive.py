from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from ..archive import archive_paths, now_utc_stamp, write_json
from ..config import AppConfig
from ..io import ensure_dir, save_table
from ..pipelines.build_draftkings_board import build_draftkings_mlb_board
from ..sportsbooks.draftkings import (
    capture_draftkings_payloads,
    load_draftkings_payloads_from_files,
    normalize_draftkings_payloads,
)


def _capture_bundle(capture_files, use_live_capture: bool, use_browser: bool):
    bundles = []
    if capture_files:
        bundles.append(load_draftkings_payloads_from_files(capture_files))
    if use_live_capture or not bundles:
        bundles.append(capture_draftkings_payloads(sports=["mlb"], use_browser=use_browser))

    payloads = []
    urls = []
    used_browser = False
    sources = []
    for bundle in bundles:
        payloads.extend(bundle.payloads)
        urls.extend(bundle.urls)
        used_browser = used_browser or bool(bundle.used_browser)
        if bundle.source:
            sources.append(bundle.source)

    return {
        "payloads": payloads,
        "urls": urls,
        "used_browser": used_browser,
        "sources": sources,
    }


def capture_draftkings_snapshot(
    config: AppConfig,
    capture_files: list[str | Path] | None = None,
    use_live_capture: bool = False,
    use_browser: bool = False,
) -> dict[str, Path]:
    capture_files = list(capture_files or [])
    stamp = now_utc_stamp()
    paths = archive_paths(config.draftkings_archive_dir, stamp)

    bundle = _capture_bundle(capture_files, use_live_capture=use_live_capture, use_browser=use_browser)
    payloads = bundle["payloads"]

    payload_files = []
    for i, payload in enumerate(payloads, start=1):
        payload_path = paths["raw dir"] / f"payload {i:03d}.json"
        payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        payload_files.append(payload_path)

    raw_lines = normalize_draftkings_payloads(payloads, sport_hint="mlb")
    raw_lines = raw_lines[raw_lines["sport"].astype(str).str.lower() == "mlb"].copy()
    raw_lines["captured at"] = stamp

    temp_capture_files = [str(path) for path in payload_files]
    board_outputs = build_draftkings_mlb_board(
        config=config,
        capture_files=temp_capture_files,
        use_live_capture=False,
        use_browser=False,
    )

    # Copy current board files into the timestamped archive folder too.
    board_dir = paths["board dir"]
    archived_outputs = {
        "draftkings mlb lines snapshot": save_table(pd.read_parquet(config.draftkings_mlb_lines), board_dir / "draftkings mlb lines.parquet"),
        "draftkings mlb game lines snapshot": save_table(pd.read_parquet(config.draftkings_mlb_game_lines), board_dir / "draftkings mlb game lines.parquet"),
        "draftkings mlb player props snapshot": save_table(pd.read_parquet(config.draftkings_mlb_player_props), board_dir / "draftkings mlb player props.parquet"),
        "draftkings mlb priced board snapshot": save_table(pd.read_parquet(config.draftkings_mlb_priced_board), board_dir / "draftkings mlb priced board.parquet"),
    }
    report_df = pd.read_csv(config.draftkings_mlb_betting_report)
    archived_outputs["draftkings mlb betting report snapshot"] = save_table(report_df, board_dir / "draftkings mlb betting report.csv")

    manifest = {
        "captured at": stamp,
        "source type": "live capture" if use_live_capture else "capture files",
        "capture files used": [str(x) for x in capture_files],
        "bundle urls": bundle["urls"],
        "used browser": bundle["used_browser"],
        "sources": bundle["sources"],
        "payload count": len(payload_files),
        "mlb row count": int(len(raw_lines)),
    }
    write_json(paths["manifest"], manifest)

    outputs = {
        "archive raw folder": paths["raw dir"],
        "archive board folder": paths["board dir"],
        "archive manifest": paths["manifest"],
        **archived_outputs,
    }
    return outputs


def rebuild_draftkings_history(config: AppConfig) -> dict[str, Path]:
    base = config.draftkings_archive_dir / "boards"
    if not base.exists():
        empty = pd.DataFrame()
        save_table(empty, config.draftkings_archive_history_parquet)
        save_table(empty, config.draftkings_archive_history_csv)
        return {
            "draftkings mlb archive history parquet": config.draftkings_archive_history_parquet,
            "draftkings mlb archive history csv": config.draftkings_archive_history_csv,
        }

    frames = []
    for path in sorted(base.rglob("draftkings mlb lines.parquet")):
        try:
            df = pd.read_parquet(path)
            stamp = path.parent.name
            df["captured at"] = stamp
            frames.append(df)
        except Exception:
            continue

    if frames:
        history = pd.concat(frames, ignore_index=True)
    else:
        history = pd.DataFrame()

    if not history.empty:
        rename_map = {
            "sport": "sport",
            "event_id": "event id",
            "market": "market",
            "selection": "selection",
            "line": "line",
            "price": "american odds",
            "participant": "player",
            "opponent": "opponent",
            "team_name": "team",
            "home_team": "home team",
            "away_team": "away team",
            "book": "book",
            "fetched_at": "fetched at",
            "board scope": "board scope",
            "captured at": "captured at",
        }
        history = history.rename(columns=rename_map)
        if "sport" in history.columns:
            history["sport"] = history["sport"].astype(str).str.upper()

        sort_cols = [c for c in ["captured at", "event id", "market", "selection", "player", "team"] if c in history.columns]
        if sort_cols:
            history = history.sort_values(sort_cols).reset_index(drop=True)

    save_table(history, config.draftkings_archive_history_parquet)
    save_table(history, config.draftkings_archive_history_csv)
    return {
        "draftkings mlb archive history parquet": config.draftkings_archive_history_parquet,
        "draftkings mlb archive history csv": config.draftkings_archive_history_csv,
    }
