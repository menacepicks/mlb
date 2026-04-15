from __future__ import annotations

from pathlib import Path
import re
import unicodedata

import pandas as pd

from ..config import AppConfig
from ..io import save_table
from ..pricing import american_to_probability, format_american
from ..team_names import team_name_to_id


DISPLAY_COLUMNS = [
    "sport",
    "event id",
    "market",
    "selection",
    "line",
    "player",
    "team",
    "home team",
    "away team",
]

GROUP_COLUMNS = [
    "sport key",
    "market key",
    "selection key",
    "line key",
    "player key",
    "team key",
    "home team key",
    "away team key",
]


def _load_book(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.empty:
        return df
    for col in DISPLAY_COLUMNS + ["book", "american odds", "fetched at"]:
        if col not in df.columns:
            df[col] = pd.NA
    return df.copy()


def _norm_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    words = [word for word in text.split() if word not in {"jr", "sr", "ii", "iii", "iv"}]
    return " ".join(words)


def _line_key(value: object) -> str:
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        num = float(text)
        return f"{num:.3f}"
    except Exception:
        return text



BINARY_PLAYER_YES_MARKETS = {
    "to hit a home run",
}

def _normalize_selection_key(market: object, selection: object, player: object) -> str:
    market_key = str(market or "").strip().lower()
    selection_key = str(selection or "").strip().lower()
    if market_key in BINARY_PLAYER_YES_MARKETS:
        if selection_key in {"", "yes", "over"}:
            return "yes"
        player_key = _norm_text(player)
        if selection_key == player_key:
            return "yes"
    return selection_key

def _normalized_line_key(market: object, selection: object, line: object, player: object) -> str:
    market_key = str(market or "").strip().lower()
    selection_key = _normalize_selection_key(market, selection, player)
    if market_key in BINARY_PLAYER_YES_MARKETS and selection_key == "yes":
        return ""
    return _line_key(line)


def _prepare_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sport"] = out["sport"].astype(str).str.upper()

    out["sport key"] = out["sport"].astype(str).str.upper()
    out["market key"] = out["market"].astype(str).str.lower().str.strip()
    out["selection key"] = [
        _normalize_selection_key(market, selection, player)
        for market, selection, player in zip(out["market"], out["selection"], out["player"], strict=False)
    ]
    out["line key"] = [
        _normalized_line_key(market, selection, line, player)
        for market, selection, line, player in zip(out["market"], out["selection"], out["line"], out["player"], strict=False)
    ]
    out["player key"] = out["player"].map(_norm_text)
    out["team key"] = out["team"].map(team_name_to_id)
    out["home team key"] = out["home team"].map(team_name_to_id)
    out["away team key"] = out["away team"].map(team_name_to_id)

    # fallback for team key on home/away selections
    home_mask = out["team key"].eq("") & out["selection key"].eq("home")
    away_mask = out["team key"].eq("") & out["selection key"].eq("away")
    out.loc[home_mask, "team key"] = out.loc[home_mask, "home team key"]
    out.loc[away_mask, "team key"] = out.loc[away_mask, "away team key"]

    return out


def compare_mlb_books(config: AppConfig) -> dict[str, Path]:
    dk = _load_book(config.draftkings_mlb_lines)
    fd = _load_book(config.fanduel_mlb_lines)

    frames = [df for df in [dk, fd] if not df.empty]
    if not frames:
        empty = pd.DataFrame()
        save_table(empty, config.mlb_two_book_comparison_parquet)
        save_table(empty, config.mlb_two_book_comparison_csv)
        save_table(empty, config.mlb_best_price_by_market_parquet)
        save_table(empty, config.mlb_best_price_by_market_csv)
        return {
            "mlb two book comparison parquet": config.mlb_two_book_comparison_parquet,
            "mlb two book comparison csv": config.mlb_two_book_comparison_csv,
            "mlb best price by market parquet": config.mlb_best_price_by_market_parquet,
            "mlb best price by market csv": config.mlb_best_price_by_market_csv,
        }

    lines = pd.concat(frames, ignore_index=True)
    lines["american odds"] = pd.to_numeric(lines["american odds"], errors="coerce")
    lines["implied chance"] = lines["american odds"].map(american_to_probability)
    lines["price display"] = lines["american odds"].map(format_american)

    comparison = _prepare_keys(lines)
    keep = DISPLAY_COLUMNS + ["book", "american odds", "price display", "implied chance", "fetched at"] + GROUP_COLUMNS
    comparison = comparison[keep].sort_values(DISPLAY_COLUMNS[:-1] + ["away team", "book"]).reset_index(drop=True)

    def _best_index(group: pd.DataFrame) -> int:
        prices = pd.to_numeric(group["american odds"], errors="coerce")
        if prices.notna().any():
            return prices.idxmax()
        return group.index[0]

    idx = comparison.groupby(GROUP_COLUMNS, dropna=False).apply(_best_index).tolist()
    best_price = comparison.loc[idx].copy().reset_index(drop=True)
    best_price = best_price.rename(
        columns={
            "book": "best book",
            "american odds": "best american odds",
            "price display": "best price",
            "implied chance": "best implied chance",
            "fetched at": "best fetched at",
        }
    )

    book_count = comparison.groupby(GROUP_COLUMNS, dropna=False)["book"].nunique().reset_index(name="books found")
    best_price = best_price.merge(book_count, on=GROUP_COLUMNS, how="left")

    comparison_out = comparison.drop(columns=GROUP_COLUMNS).copy()
    best_price_out = best_price.drop(columns=GROUP_COLUMNS).copy()

    for frame in [comparison_out, best_price_out]:
        if {"market", "selection", "player"}.issubset(frame.columns):
            frame["selection"] = [
                _normalize_selection_key(market, selection, player)
                for market, selection, player in zip(frame["market"], frame["selection"], frame["player"], strict=False)
            ]

    save_table(comparison_out, config.mlb_two_book_comparison_parquet)
    save_table(comparison_out, config.mlb_two_book_comparison_csv)
    save_table(best_price_out, config.mlb_best_price_by_market_parquet)
    save_table(best_price_out, config.mlb_best_price_by_market_csv)

    return {
        "mlb two book comparison parquet": config.mlb_two_book_comparison_parquet,
        "mlb two book comparison csv": config.mlb_two_book_comparison_csv,
        "mlb best price by market parquet": config.mlb_best_price_by_market_parquet,
        "mlb best price by market csv": config.mlb_best_price_by_market_csv,
    }
