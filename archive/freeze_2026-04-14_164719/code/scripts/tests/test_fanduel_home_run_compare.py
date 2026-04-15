
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from mlb_betting_data.config import AppConfig
from mlb_betting_data.pipelines.compare_books import compare_mlb_books


def main() -> None:
    tmp = Path("artifacts test")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()

    config = AppConfig(out_dir=tmp)

    dk = pd.DataFrame([
        {
            "sport": "MLB",
            "event id": "dk1",
            "market": "to hit a home run",
            "selection": "yes",
            "line": None,
            "player": "Shohei Ohtani",
            "opponent": "",
            "team": "Los Angeles Dodgers",
            "home team": "Los Angeles Dodgers",
            "away team": "New York Mets",
            "book": "DraftKings",
            "american odds": 220,
            "fetched at": "2026-04-14T01:00:00Z",
            "board scope": "player props",
        }
    ])

    fd = pd.DataFrame([
        {
            "sport": "MLB",
            "event id": "fd1",
            "market": "to hit a home run",
            "selection": "shohei ohtani",
            "line": 0.0,
            "player": "Shohei Ohtani",
            "opponent": "",
            "team": "los angeles dodgers",
            "home team": "Los Angeles Dodgers",
            "away team": "New York Mets",
            "book": "FanDuel",
            "american odds": 245,
            "fetched at": "2026-04-14T01:00:00Z",
            "board scope": "player props",
        }
    ])

    dk.to_parquet(config.draftkings_mlb_lines, index=False)
    fd.to_parquet(config.fanduel_mlb_lines, index=False)

    compare_mlb_books(config)
    out = pd.read_csv(config.mlb_best_price_by_market_csv)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["books found"] == 2
    assert row["selection"] == "yes"
    print("ok")


if __name__ == "__main__":
    main()
