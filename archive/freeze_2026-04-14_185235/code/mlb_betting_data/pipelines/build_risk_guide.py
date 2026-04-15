from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import AppConfig
from ..io import save_table
from ..risk import build_same_game_relationship_guide, build_volatility_table


def build_risk_guide(config: AppConfig) -> dict[str, Path]:
    hitter_df = pd.read_parquet(config.actual_hitter_stats_by_game)
    pitcher_df = pd.read_parquet(config.actual_pitcher_stats_by_game)
    team_df = pd.read_parquet(config.actual_team_game_results)
    game_df = pd.read_parquet(config.actual_game_results)

    volatility = build_volatility_table(hitter_df, pitcher_df, team_df)
    relationships = build_same_game_relationship_guide(hitter_df, pitcher_df, team_df, game_df)

    save_table(volatility, config.player_and_team_volatility)
    save_table(relationships, config.same_game_relationship_guide)
    return {
        "player and team volatility": config.player_and_team_volatility,
        "same game relationship guide": config.same_game_relationship_guide,
    }
