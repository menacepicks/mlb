from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from ..board_utils import classify_scope, rename_raw_columns
from ..config import AppConfig
from ..io import save_table
from ..pricing import to_beginner_bettor_report, to_plain_english_table
from ..sportsbooks.fanduel import (
    capture_fanduel_payloads,
    load_fanduel_payloads_from_files,
    normalize_fanduel_payloads,
)


def build_fanduel_mlb_board(
    config: AppConfig,
    capture_files: Iterable[str | Path] | None = None,
    use_live_capture: bool = False,
) -> dict[str, Path]:
    capture_files = list(capture_files or [])

    bundles = []
    if capture_files:
        bundles.append(load_fanduel_payloads_from_files(capture_files))
    if use_live_capture or not bundles:
        bundles.append(capture_fanduel_payloads(sports=["mlb"]))

    payloads = []
    for bundle in bundles:
        payloads.extend(bundle.payloads)

    lines = normalize_fanduel_payloads(payloads, sport_hint="mlb")
    if lines.empty:
        lines = pd.DataFrame(columns=[
            "sport", "event_id", "market", "selection", "line", "price",
            "participant", "opponent", "team_name", "home_team", "away_team", "book", "fetched_at"
        ])

    lines = lines[lines["sport"].astype(str).str.lower() == "mlb"].copy()
    if not lines.empty:
        lines["board scope"] = classify_scope(lines)
        game_lines = lines[lines["board scope"] == "game lines"].copy()
        player_props = lines[lines["board scope"] == "player props"].copy()
    else:
        game_lines = lines.copy()
        player_props = lines.copy()

    raw_lines = rename_raw_columns(lines)
    raw_game_lines = rename_raw_columns(game_lines)
    raw_player_props = rename_raw_columns(player_props)
    priced_board = to_plain_english_table(lines)
    betting_report = to_beginner_bettor_report(lines)

    return {
        "fanduel mlb lines": save_table(raw_lines, config.fanduel_mlb_lines),
        "fanduel mlb game lines": save_table(raw_game_lines, config.fanduel_mlb_game_lines),
        "fanduel mlb player props": save_table(raw_player_props, config.fanduel_mlb_player_props),
        "fanduel mlb priced board": save_table(priced_board, config.fanduel_mlb_priced_board),
        "fanduel mlb betting report": save_table(betting_report, config.fanduel_mlb_betting_report),
    }
