from __future__ import annotations

from typing import Any

import pandas as pd


def american_to_probability(price: Any) -> float | None:
    if price is None or price == "":
        return None
    value = float(price)
    if value == 0:
        return None
    if value > 0:
        return 100.0 / (value + 100.0)
    return abs(value) / (abs(value) + 100.0)


def american_to_decimal(price: Any) -> float | None:
    if price is None or price == "":
        return None
    value = float(price)
    if value > 0:
        return 1.0 + (value / 100.0)
    return 1.0 + (100.0 / abs(value))


def probability_to_decimal(probability: Any) -> float | None:
    if probability is None or probability == "":
        return None
    p = float(probability)
    if p <= 0 or p >= 1:
        return None
    return 1.0 / p


def probability_to_american(probability: Any) -> float | None:
    if probability is None or probability == "":
        return None
    p = float(probability)
    if p <= 0 or p >= 1:
        return None
    if p >= 0.5:
        return -100.0 * p / (1.0 - p)
    return 100.0 * (1.0 - p) / p


def format_american(price: Any) -> str:
    if price is None or price == "":
        return ""
    value = round(float(price))
    if value > 0:
        return f"+{value}"
    return f"{value}"


def format_percent(probability: Any) -> str:
    if probability is None or probability == "":
        return ""
    return f"{float(probability) * 100:.1f}%"


def format_decimal(odds: Any) -> str:
    if odds is None or odds == "":
        return ""
    return f"{float(odds):.2f}"


def format_line(line: Any) -> str:
    if line is None or line == "":
        return ""
    value = float(line)
    if value.is_integer():
        return f"{int(value)}"
    return f"{value:.1f}"


def selection_display(selection: str, line: Any) -> str:
    choice = str(selection or "").strip().lower()
    line_text = format_line(line)
    if choice == "over":
        return f"Over {line_text}" if line_text else "Over"
    if choice == "under":
        return f"Under {line_text}" if line_text else "Under"
    if choice == "yes":
        return "Yes"
    if choice == "no":
        return "No"
    if choice == "home":
        return "Home"
    if choice == "away":
        return "Away"
    return str(selection or "")


def add_pricing_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sportsbook implied probability"] = out["price"].map(american_to_probability)
    out["sportsbook decimal odds"] = out["price"].map(american_to_decimal)
    out["sportsbook american display"] = out["price"].map(format_american)
    out["selection display"] = [
        selection_display(selection, line) for selection, line in zip(out["selection"], out["line"], strict=False)
    ]
    return out


def add_no_vig_fair_prices(df: pd.DataFrame) -> pd.DataFrame:
    out = add_pricing_columns(df)

    group_keys = ["event_id", "market", "participant", "team_name", "line"]
    fair_probability = []
    for _, group in out.groupby(group_keys, dropna=False, sort=False):
        probs = pd.to_numeric(group["sportsbook implied probability"], errors="coerce")
        total = probs.sum()
        if len(group) == 2 and total > 0:
            fair = probs / total
        else:
            fair = probs
        fair_probability.extend(fair.tolist())

    out["fair probability"] = fair_probability
    out["fair decimal odds"] = out["fair probability"].map(probability_to_decimal)
    out["fair american odds"] = out["fair probability"].map(probability_to_american)
    out["fair american display"] = out["fair american odds"].map(format_american)

    numeric_cols = [
        "line",
        "price",
        "sportsbook implied probability",
        "sportsbook decimal odds",
        "fair probability",
        "fair decimal odds",
        "fair american odds",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def to_plain_english_table(df: pd.DataFrame) -> pd.DataFrame:
    out = add_no_vig_fair_prices(df)

    out["Sport"] = out["sport"].astype(str).str.upper()
    out["Event id"] = out["event_id"]
    out["Market"] = out["market"].astype(str).str.replace("_", " ", regex=False).str.title()
    out["Pick"] = out["selection display"]
    out["Line"] = out["line"].map(format_line)
    out["Sportsbook price"] = out["price"].map(format_american)
    out["Sportsbook decimal"] = out["sportsbook decimal odds"].map(format_decimal)
    out["Sportsbook chance"] = out["sportsbook implied probability"].map(format_percent)
    out["Fair chance"] = out["fair probability"].map(format_percent)
    out["Fair price"] = out["fair american display"]
    out["Fair decimal"] = out["fair decimal odds"].map(format_decimal)
    out["Player"] = out["participant"].fillna("")
    out["Team"] = out["team_name"].fillna("")
    out["Home team"] = out["home_team"].fillna("")
    out["Away team"] = out["away_team"].fillna("")
    out["Book"] = out["book"].fillna("")
    out["Fetched at"] = out["fetched_at"].fillna("")

    keep = [
        "Sport",
        "Event id",
        "Market",
        "Pick",
        "Line",
        "Sportsbook price",
        "Sportsbook decimal",
        "Sportsbook chance",
        "Fair chance",
        "Fair price",
        "Fair decimal",
        "Player",
        "Team",
        "Home team",
        "Away team",
        "Book",
        "Fetched at",
    ]
    return out[keep].copy()


def to_beginner_bettor_report(df: pd.DataFrame) -> pd.DataFrame:
    out = add_no_vig_fair_prices(df)

    out["Sport"] = out["sport"].astype(str).str.upper()
    out["Matchup"] = (
        out["away_team"].fillna("").astype(str)
        + " at "
        + out["home_team"].fillna("").astype(str)
    ).str.strip()
    out["Bet type"] = out["market"].astype(str).str.replace("_", " ", regex=False).str.title()
    out["Pick"] = out["selection display"]
    out["Line"] = out["line"].map(format_line)
    out["Sportsbook price"] = out["price"].map(format_american)
    out["Sportsbook chance"] = out["sportsbook implied probability"].map(format_percent)
    out["Fair chance"] = out["fair probability"].map(format_percent)
    out["Fair price"] = out["fair american display"]
    out["Player"] = out["participant"].fillna("")
    out["Team"] = out["team_name"].fillna("")
    out["Book"] = out["book"].fillna("DraftKings")

    keep = [
        "Sport",
        "Matchup",
        "Bet type",
        "Pick",
        "Line",
        "Sportsbook price",
        "Sportsbook chance",
        "Fair chance",
        "Fair price",
        "Player",
        "Team",
        "Book",
    ]
    return out[keep].copy()
