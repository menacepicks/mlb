from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

BASE_URL = (
    "https://sportsbook-nash.draftkings.com/sites/US-SB/api/"
    "sportscontent/controldata/league/leagueSubcategory/v1/markets"
)

DEFAULT_LEAGUE_ID = "84240"  # MLB

# IDs discovered from the user's DraftKings network traffic and screenshots.
DEFAULT_SUBCATEGORY_IDS = [
    6607,   # Total Bases O/U
    15221,  # Strikeouts Thrown O/U
    16208,  # Alternate Team Total Runs
    17319,  # discovered from leagueSubcategory trace
    17320,  # Hits Milestones
    17323,  # Strikeouts Thrown Milestones
    17406,  # Hits + Runs + RBIs O/U
    9499,
    9502,
    9505,
    9506,
    9508,
    9536,
    12150,
    17471,
    17472,
    17473,
    17474,
    17475,
    17477,
]

HEADERS = {
    "accept": "*/*",
    "origin": "https://sportsbook.draftkings.com",
    "referer": "https://sportsbook.draftkings.com/",
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
    ),
    "x-client-feature": "leagueSubcategory",
    "x-client-name": "web",
    "x-client-page": "league",
    "x-client-widget-name": "cms",
}


@dataclass
class FetchResult:
    subcategory_id: int
    ok: bool
    markets: int = 0
    selections: int = 0
    error: str = ""
    payload_path: str = ""


def safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def clean_american(value: Any) -> str:
    text = str(value or "").strip()
    return text.replace("−", "-")


def read_subcategory_ids(value: str | None) -> list[int]:
    if not value:
        return list(DEFAULT_SUBCATEGORY_IDS)
    cleaned = value.replace(",", " ").split()
    out: list[int] = []
    for item in cleaned:
        num = safe_int(item)
        if num is not None:
            out.append(num)
    return out or list(DEFAULT_SUBCATEGORY_IDS)


def build_params(league_id: str, subcategory_id: int) -> dict[str, str]:
    return {
        "isBatchable": "false",
        "templateVars": f"{league_id},{subcategory_id}",
        "eventsQuery": (
            f"$filter=leagueId eq '{league_id}' AND "
            f"clientMetadata/Subcategories/any(s: s/Id eq '{subcategory_id}')"
        ),
        "marketsQuery": (
            f"$filter=clientMetadata/subCategoryId eq '{subcategory_id}' "
            "AND tags/all(t: t ne 'SportcastBetBuilder')"
        ),
        "include": "Events",
        "entity": "events",
    }


def fetch_payload(session: requests.Session, league_id: str, subcategory_id: int) -> dict[str, Any]:
    response = session.get(
        BASE_URL,
        headers=HEADERS,
        params=build_params(league_id, subcategory_id),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def event_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for event in payload.get("events", []) or []:
        event_id = str(event.get("id", ""))
        if not event_id:
            continue
        participants = event.get("participants", []) or []
        home_team = ""
        away_team = ""
        for p in participants:
            role = str(p.get("venueRole", "")).lower()
            if role == "home":
                home_team = str(p.get("name", ""))
            elif role == "away":
                away_team = str(p.get("name", ""))
        out[event_id] = {
            "event name": str(event.get("name", "")),
            "start time": str(event.get("startEventDate", "")),
            "home team": home_team,
            "away team": away_team,
        }
    return out


YES_EQUIVALENT_MARKETS = {
    "Hits Milestones": "to record a hit",
    "Home Runs Milestones": "to hit a home run",
    "Runs Scored Milestones": "to score a run",
    "Runs Batted In Milestones": "to record an rbi",
    "Stolen Bases Milestones": "to steal a base",
}


def normalize_payload(payload: dict[str, Any], fetched_at: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = event_map(payload)
    market_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []

    for market in payload.get("markets", []) or []:
        market_rows.append(
            {
                "marketId": str(market.get("id", "")),
                "event id": str(market.get("eventId", "")),
                "subcategory id": safe_int(market.get("subcategoryId")),
                "market": str(market.get("name", "")),
                "market type": str((market.get("marketType") or {}).get("name", "")),
                "tags": ", ".join(market.get("tags", []) or []),
            }
        )

    market_df = pd.DataFrame(market_rows)

    for selection in payload.get("selections", []) or []:
        market_id = str(selection.get("marketId", ""))
        participant = (selection.get("participants") or [{}])[0]
        participant_name = str(participant.get("name", ""))
        participant_type = str(participant.get("type", ""))
        statistic_prefix = str((participant.get("statistic") or {}).get("prefix", ""))

        selection_rows.append(
            {
                "marketId": market_id,
                "selection": str(selection.get("outcomeType") or selection.get("label") or ""),
                "selection label": str(selection.get("label", "")),
                "line": selection.get("points", selection.get("milestoneValue")),
                "american odds": clean_american((selection.get("displayOdds") or {}).get("american", "")),
                "decimal odds": (selection.get("displayOdds") or {}).get("decimal", ""),
                "participant": participant_name,
                "participant type": participant_type,
                "stat prefix": statistic_prefix,
                "milestone value": selection.get("milestoneValue"),
                "tags selection": ", ".join(selection.get("tags", []) or []),
            }
        )

    selection_df = pd.DataFrame(selection_rows)
    if market_df.empty or selection_df.empty:
        return pd.DataFrame(), market_df

    joined = selection_df.merge(market_df, on="marketId", how="left")
    joined["book"] = "DraftKings"
    joined["fetched at"] = fetched_at

    joined["event name"] = joined["event id"].map(lambda x: events.get(str(x), {}).get("event name", ""))
    joined["start time"] = joined["event id"].map(lambda x: events.get(str(x), {}).get("start time", ""))
    joined["home team"] = joined["event id"].map(lambda x: events.get(str(x), {}).get("home team", ""))
    joined["away team"] = joined["event id"].map(lambda x: events.get(str(x), {}).get("away team", ""))

    joined["board scope"] = joined["participant type"].map(
        lambda x: "player props" if str(x).lower() == "player" else "game lines"
    )

    # Canonical naming for milestone 1+ markets.
    def canonical_market(row: pd.Series) -> str:
        market_type = str(row.get("market type", ""))
        milestone = row.get("milestone value")
        if pd.notna(milestone) and int(milestone) == 1 and market_type in YES_EQUIVALENT_MARKETS:
            return YES_EQUIVALENT_MARKETS[market_type]
        return str(row.get("market", ""))

    def canonical_selection(row: pd.Series) -> str:
        market_type = str(row.get("market type", ""))
        milestone = row.get("milestone value")
        if pd.notna(milestone) and int(milestone) == 1 and market_type in YES_EQUIVALENT_MARKETS:
            return "Yes"
        return str(row.get("selection", ""))

    joined["canonical market"] = joined.apply(canonical_market, axis=1)
    joined["canonical selection"] = joined.apply(canonical_selection, axis=1)

    return joined, market_df


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape all DraftKings MLB markets from league subcategory endpoints.")
    parser.add_argument("--out-dir", default="artifacts", help="Output directory")
    parser.add_argument("--league-id", default=DEFAULT_LEAGUE_ID, help="DraftKings league id (MLB=84240)")
    parser.add_argument(
        "--subcategory-ids",
        default=os.environ.get("DK_MLB_SUBCATEGORY_IDS", ""),
        help="Space or comma separated subcategory ids. Uses discovered defaults if omitted.",
    )
    parser.add_argument("--save-json", action="store_true", help="Save raw JSON payloads per subcategory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw_draftkings_subcategories"
    subcategory_ids = read_subcategory_ids(args.subcategory_ids)
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    session = requests.Session()

    all_rows: list[pd.DataFrame] = []
    summaries: list[FetchResult] = []

    for subcategory_id in subcategory_ids:
        try:
            payload = fetch_payload(session, args.league_id, subcategory_id)
            if args.save_json:
                raw_dir.mkdir(parents=True, exist_ok=True)
                payload_path = raw_dir / f"dk_mlb_subcat_{subcategory_id}.json"
                payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            else:
                payload_path = Path("")

            rows_df, market_df = normalize_payload(payload, fetched_at)
            if not rows_df.empty:
                all_rows.append(rows_df)
            summaries.append(
                FetchResult(
                    subcategory_id=subcategory_id,
                    ok=True,
                    markets=len(payload.get("markets", []) or []),
                    selections=len(payload.get("selections", []) or []),
                    payload_path=str(payload_path),
                )
            )
        except Exception as exc:  # noqa: BLE001
            summaries.append(FetchResult(subcategory_id=subcategory_id, ok=False, error=str(exc)))

    summary_df = pd.DataFrame([vars(item) for item in summaries])
    save_table(summary_df, out_dir / "draftkings mlb api subcategory summary.csv")

    if not all_rows:
        save_table(pd.DataFrame(), out_dir / "draftkings mlb api all markets.csv")
        print(out_dir / "draftkings mlb api all markets.csv")
        return 1

    df = pd.concat(all_rows, ignore_index=True)
    # Remove duplicates across repeated IDs / overlaps.
    dedupe_cols = [
        "event id",
        "subcategory id",
        "marketId",
        "selection label",
        "participant",
        "line",
        "american odds",
    ]
    dedupe_cols = [c for c in dedupe_cols if c in df.columns]
    df = df.drop_duplicates(subset=dedupe_cols).reset_index(drop=True)

    player_df = df[df["board scope"] == "player props"].copy()
    game_df = df[df["board scope"] == "game lines"].copy()

    save_table(df, out_dir / "draftkings mlb api all markets.parquet")
    save_table(df, out_dir / "draftkings mlb api all markets.csv")
    save_table(player_df, out_dir / "draftkings mlb api player props.parquet")
    save_table(player_df, out_dir / "draftkings mlb api player props.csv")
    save_table(game_df, out_dir / "draftkings mlb api game lines.parquet")
    save_table(game_df, out_dir / "draftkings mlb api game lines.csv")

    # Beginner-friendly slim report.
    report = df[
        [
            "book",
            "start time",
            "event name",
            "home team",
            "away team",
            "board scope",
            "market type",
            "canonical market",
            "canonical selection",
            "line",
            "participant",
            "participant type",
            "american odds",
            "decimal odds",
            "subcategory id",
        ]
    ].copy()
    report = report.rename(
        columns={
            "market type": "market family",
            "canonical market": "market",
            "canonical selection": "pick",
            "participant": "player or team",
        }
    )
    save_table(report, out_dir / "draftkings mlb api betting report.csv")

    print(f"all markets: {out_dir / 'draftkings mlb api all markets.parquet'}")
    print(f"player props: {out_dir / 'draftkings mlb api player props.parquet'}")
    print(f"game lines: {out_dir / 'draftkings mlb api game lines.parquet'}")
    print(f"report: {out_dir / 'draftkings mlb api betting report.csv'}")
    print(f"summary: {out_dir / 'draftkings mlb api subcategory summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
