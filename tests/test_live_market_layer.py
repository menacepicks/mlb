from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlb_live_books import parse_draftkings_raw_payload, parse_fanduel_raw_payloads, from_shared_odds_table
from mlb_live_pricing import score_market_board


def _dk_payload() -> dict:
    return {
        "events": [
            {
                "id": "33984819",
                "eventId": "33984819",
                "status": "NOT_STARTED",
                "participants": [
                    {"name": "ARI Diamondbacks", "type": "Team", "venueRole": "away"},
                    {"name": "BAL Orioles", "type": "Team", "venueRole": "home"},
                ],
            }
        ],
        "markets": [
            {"id": "m1", "eventId": "33984819", "name": "Game Total Runs", "marketType": {"name": "Total Runs"}},
            {"id": "m2", "eventId": "33984819", "name": "Pete Alonso Hits O/U", "marketType": {"name": "Hits O/U"}},
            {"id": "m3", "eventId": "33984819", "name": "BAL Orioles Team Total Runs", "marketType": {"name": "Team Total Runs"}},
            {"id": "m4", "eventId": "33984819", "name": "Pete Alonso 2+ Hits", "marketType": {"name": "Hits"}},
        ],
        "selections": [
            {"id": "s1", "marketId": "m1", "label": "Over", "points": 8.5, "displayOdds": {"american": "-105"}},
            {"id": "s2", "marketId": "m1", "label": "Under", "points": 8.5, "displayOdds": {"american": "-115"}},
            {"id": "s3", "marketId": "m2", "label": "Over", "points": 1.5, "displayOdds": {"american": "+177"}, "participants": [{"name": "Pete Alonso", "type": "Player", "venueRole": "AwayPlayer"}]},
            {"id": "s4", "marketId": "m2", "label": "Under", "points": 1.5, "displayOdds": {"american": "-210"}, "participants": [{"name": "Pete Alonso", "type": "Player", "venueRole": "AwayPlayer"}]},
            {"id": "s5", "marketId": "m3", "label": "Over", "points": 4.5, "displayOdds": {"american": "-102"}},
            {"id": "s6", "marketId": "m3", "label": "Under", "points": 4.5, "displayOdds": {"american": "-118"}},
            {"id": "s7", "marketId": "m4", "label": "Yes", "points": 2.0, "displayOdds": {"american": "+230"}, "participants": [{"name": "Pete Alonso", "type": "Player", "venueRole": "AwayPlayer"}]},
            {"id": "s8", "marketId": "m4", "label": "No", "points": 2.0, "displayOdds": {"american": "-300"}, "participants": [{"name": "Pete Alonso", "type": "Player", "venueRole": "AwayPlayer"}]},
        ],
        "lastUpdatedTime": "2026-04-14T19:30:56Z",
    }


def _fd_event_pages() -> list[dict]:
    return [
        {
            "events": {
                "35487026": {
                    "eventId": 35487026,
                    "homeTeamName": "BAL Orioles",
                    "awayTeamName": "ARI Diamondbacks",
                }
            },
            "markets": {
                "734.164099949": {"marketId": "734.164099949", "eventId": 35487026, "marketName": "Game Total Runs"},
                "734.164201097": {"marketId": "734.164201097", "eventId": 35487026, "marketName": "Pete Alonso Hits O/U"},
            },
        }
    ]


def _fd_price_rows() -> list[dict]:
    return [
        {
            "marketId": "734.164099949",
            "runnerDetails": [
                {"selectionId": "fd1", "runnerName": "Over", "handicap": 8.5, "winRunnerOdds": {"americanDisplayOdds": {"americanOddsInt": -110}}},
                {"selectionId": "fd2", "runnerName": "Under", "handicap": 8.5, "winRunnerOdds": {"americanDisplayOdds": {"americanOddsInt": -110}}},
            ],
        },
        {
            "marketId": "734.164201097",
            "runnerDetails": [
                {"selectionId": "fd3", "runnerName": "Over", "handicap": 1.5, "winRunnerOdds": {"americanDisplayOdds": {"americanOddsInt": 165}}},
                {"selectionId": "fd4", "runnerName": "Under", "handicap": 1.5, "winRunnerOdds": {"americanDisplayOdds": {"americanOddsInt": -200}}},
            ],
        },
    ]


def _projections() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"scope": "player", "segment": "full_game", "market_family": "hits", "event_id": "33984819", "participant": "Pete Alonso", "team_name": "", "projection_mean": 1.7},
            {"scope": "team", "segment": "full_game", "market_family": "team_total_runs", "event_id": "33984819", "participant": "", "team_name": "BAL Orioles", "projection_mean": 5.0},
            {"scope": "game", "segment": "full_game", "market_family": "game_total_runs", "event_id": "33984819", "participant": "", "team_name": "", "projection_mean": 9.1},
        ]
    )


def test_raw_parsers_and_scoring():
    dk = parse_draftkings_raw_payload(_dk_payload())
    assert not dk.empty
    assert {"player", "team", "game"}.issubset(set(dk["scope"]))
    assert dk[dk["is_milestone"]].shape[0] >= 2

    fd = parse_fanduel_raw_payloads(_fd_event_pages(), _fd_price_rows())
    assert not fd.empty
    assert set(fd["book"]) == {"FanDuel"}

    merged = pd.concat([dk, fd], ignore_index=True)
    scored = score_market_board(merged, projections=_projections())
    assert not scored.empty
    assert (scored["scope"] == "player").sum() > 0
    assert (scored["scope"] == "team").sum() > 0
    assert (scored["scope"] == "game").sum() > 0
    assert (scored["score_status"] == "scored_monte_carlo").sum() > 0


def test_shared_table_ingest():
    shared = pd.DataFrame(
        [
            {"sport": "mlb", "event_id": "1", "market": "Pete Alonso Hits O/U", "selection": "Over", "line": 1.5, "price": 150, "participant": "Pete Alonso", "team_name": "NYM Mets", "home_team": "NYM Mets", "away_team": "ATL Braves", "book": "DraftKings", "fetched_at": "2026-04-14T19:30:56Z"},
            {"sport": "mlb", "event_id": "1", "market": "Pete Alonso Hits O/U", "selection": "Under", "line": 1.5, "price": -180, "participant": "Pete Alonso", "team_name": "NYM Mets", "home_team": "NYM Mets", "away_team": "ATL Braves", "book": "DraftKings", "fetched_at": "2026-04-14T19:30:56Z"},
        ]
    )
    parsed = from_shared_odds_table(shared, book="DraftKings")
    assert len(parsed) == 2
    assert parsed["market_family"].iloc[0] == "hits"
    assert parsed["scope"].iloc[0] == "player"

def test_cli_end_to_end(tmp_path):
    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        str(ROOT / "run_mlb_live_board.py"),
        "--dk-input", str(ROOT / "fixtures" / "dk_raw_fixture.json"),
        "--fd-event-pages", str(ROOT / "fixtures" / "fd_event_pages_fixture.json"),
        "--fd-price-rows", str(ROOT / "fixtures" / "fd_price_rows_fixture.json"),
        "--projection-file", str(ROOT / "fixtures" / "projection_fixture.csv"),
        "--out-dir", str(out_dir),
        "--iterations", "2000",
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    player = pd.read_csv(out_dir / "mlb_live_player_board.csv")
    team = pd.read_csv(out_dir / "mlb_live_team_board.csv")
    game = pd.read_csv(out_dir / "mlb_live_game_board.csv")
    assert not player.empty
    assert not team.empty
    assert not game.empty
