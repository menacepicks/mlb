from __future__ import annotations

import pandas as pd


PLAYER_PROP_MARKETS = {
    "hits",
    "total bases",
    "home runs",
    "runs scored",
    "runs batted in",
    "walks",
    "strikeouts",
    "stolen bases",
    "earned runs allowed",
    "pitcher strikeouts",
    "outs recorded",
    "hits allowed",
    "walks allowed",
    "home runs allowed",
    "to hit a home run",
    "to_hit_a_home_run",
    "to hit a home run yes no",
}


def classify_scope(df: pd.DataFrame) -> pd.Series:
    market = df["market"].astype(str).str.lower()
    participant = df["participant"].astype(str).str.strip()
    is_player = participant.ne("") | market.isin(PLAYER_PROP_MARKETS)
    return pd.Series(["player props" if value else "game lines" for value in is_player], index=df.index)


def rename_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
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
    }
    out = out.rename(columns=rename_map)
    if "sport" in out.columns:
        out["sport"] = out["sport"].astype(str).str.upper()
    return out
