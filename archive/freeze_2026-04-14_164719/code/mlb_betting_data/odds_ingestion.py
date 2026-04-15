from __future__ import annotations

from typing import Any

COMMON_COLUMNS = [
    "sport",
    "event_id",
    "market",
    "selection",
    "line",
    "price",
    "participant",
    "opponent",
    "team_name",
    "home_team",
    "away_team",
    "book",
    "fetched_at",
]


def normalize_market_name(sport: str, market: Any) -> str:
    text = str(market or "").strip().lower()
    text = text.replace("-", " ").replace("_", " ")
    text = " ".join(text.split())

    direct = {
        "moneyline": "moneyline",
        "run line": "spread",
        "spread": "spread",
        "game total runs": "game total runs",
        "total": "game total runs",
        "total runs": "game total runs",
        "team total": "team total",
        "hits": "hits",
        "total bases": "total bases",
        "home runs": "home runs",
        "runs scored": "runs scored",
        "runs": "runs scored",
        "runs batted in": "runs batted in",
        "rbi": "runs batted in",
        "rbis": "runs batted in",
        "walks": "walks",
        "strikeouts": "strikeouts",
        "stolen bases": "stolen bases",
        "earned runs allowed": "earned runs allowed",
        "pitcher strikeouts": "pitcher strikeouts",
        "strikeouts thrown": "pitcher strikeouts",
        "outs recorded": "outs recorded",
        "pitching outs": "outs recorded",
        "hits allowed": "hits allowed",
        "walks allowed": "walks allowed",
        "home runs allowed": "home runs allowed",
    }
    return direct.get(text, text)
