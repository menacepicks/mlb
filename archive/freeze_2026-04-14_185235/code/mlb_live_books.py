from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from mlb_live_schema import (
    build_market_group_key,
    build_outcome_key,
    detect_market_family,
    detect_milestone,
    detect_scope,
    detect_segment,
    ensure_unified_columns,
    infer_entity_from_market_name,
    norm_text,
    parse_american_to_prob,
    parse_line,
    safe_str,
)


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table type: {path}")


def _rows_to_unified(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return ensure_unified_columns(pd.DataFrame())
    df = pd.DataFrame(rows)
    return ensure_unified_columns(df)


def from_shared_odds_table(df: pd.DataFrame, *, book: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        market_name = safe_str(_coalesce(record.get("market_name"), record.get("market")))
        selection = safe_str(record.get("selection"))
        participant = safe_str(_coalesce(record.get("participant"), record.get("player_name"), record.get("runner_name")))
        team_name = safe_str(record.get("team_name"))
        line = parse_line(record.get("line"))
        segment = detect_segment(market_name)
        family = detect_market_family(market_name)
        guessed_participant, guessed_team = infer_entity_from_market_name(market_name, family)
        participant = participant or guessed_participant
        team_name = team_name or guessed_team
        is_milestone, milestone_value = detect_milestone(market_name, selection, line)
        scope = detect_scope(market_name, participant, team_name, selection)
        if scope != "player" and norm_text(selection) in {"over", "under", "yes", "no", "home", "away"}:
            participant = participant if scope == "player" else ""
        market_group_key = build_market_group_key(
            event_id=safe_str(record.get("event_id")),
            scope=scope,
            segment=segment,
            market_family=family,
            participant=participant,
            team_name=team_name,
            line=line,
            is_milestone=is_milestone,
            milestone_value=milestone_value,
        )
        implied_prob = parse_american_to_prob(record.get("price"))
        rows.append(
            {
                "book": book,
                "sport": safe_str(_coalesce(record.get("sport"), "mlb")),
                "event_id": safe_str(record.get("event_id")),
                "market_id": safe_str(_coalesce(record.get("market_id"), market_name)),
                "selection_id": safe_str(_coalesce(record.get("selection_id"), selection)),
                "fetched_at": safe_str(record.get("fetched_at")),
                "market_name": market_name,
                "market_family": family,
                "scope": scope,
                "segment": segment,
                "is_milestone": bool(is_milestone),
                "milestone_value": milestone_value,
                "line": line,
                "selection": selection,
                "participant": participant,
                "team_name": team_name,
                "home_team": safe_str(record.get("home_team")),
                "away_team": safe_str(record.get("away_team")),
                "price_american": parse_line(record.get("price")),
                "implied_prob": implied_prob,
                "market_group_key": market_group_key,
                "outcome_key": build_outcome_key(market_group_key, selection),
            }
        )
    return _rows_to_unified(rows)


def parse_draftkings_raw_payload(payload: dict[str, Any]) -> pd.DataFrame:
    markets = payload.get("markets") or []
    selections = payload.get("selections") or []
    events = {str(e.get("eventId") or e.get("id") or ""): e for e in payload.get("events") or [] if isinstance(e, dict)}

    market_map = {str(m.get("id") or ""): m for m in markets if isinstance(m, dict)}
    rows: list[dict[str, Any]] = []

    for sel in selections:
        if not isinstance(sel, dict):
            continue
        market_id = safe_str(sel.get("marketId"))
        market = market_map.get(market_id, {})
        event_id = safe_str(_coalesce(sel.get("eventId"), market.get("eventId")))
        event = events.get(event_id, {})
        participant = ""
        participants = sel.get("participants") or []
        if participants and isinstance(participants[0], dict):
            participant = safe_str(participants[0].get("name"))
        display_odds = sel.get("displayOdds") or {}
        price = parse_line(_coalesce(display_odds.get("american"), sel.get("price"), sel.get("americanOdds"), sel.get("oddsAmerican")))
        if price is None:
            continue
        market_type = market.get("marketType") or {}
        market_name = safe_str(_coalesce(market.get("name"), market_type.get("name"), sel.get("marketName")))
        selection = safe_str(_coalesce(sel.get("label"), sel.get("name"), sel.get("outcomeType")))
        line = parse_line(_coalesce(sel.get("points"), sel.get("line"), sel.get("point"), sel.get("threshold")))
        home_team = ""
        away_team = ""
        for p in event.get("participants") or []:
            if not isinstance(p, dict):
                continue
            role = safe_str(p.get("venueRole")).lower()
            name = safe_str(p.get("name"))
            if role == "home":
                home_team = name
            elif role == "away":
                away_team = name
        team_name = ""
        if safe_str(sel.get("label")).lower() in {home_team.lower(), "home"}:
            team_name = home_team
        elif safe_str(sel.get("label")).lower() in {away_team.lower(), "away"}:
            team_name = away_team
        elif participant:
            participant_type = safe_str(participants[0].get("type")).lower() if participants else ""
            venue_role = safe_str(participants[0].get("venueRole")).lower() if participants else ""
            if participant_type == "team":
                team_name = participant
                participant = ""
            elif venue_role == "homeplayer":
                team_name = home_team
            elif venue_role == "awayplayer":
                team_name = away_team
        segment = detect_segment(market_name)
        family = detect_market_family(market_name)
        guessed_participant, guessed_team = infer_entity_from_market_name(market_name, family)
        participant = participant or guessed_participant
        team_name = team_name or guessed_team
        is_milestone, milestone_value = detect_milestone(market_name, selection, line)
        scope = detect_scope(market_name, participant, team_name, selection)
        if scope != "player" and safe_str(selection).lower() in {"over", "under", "yes", "no", "home", "away"}:
            participant = participant if scope == "player" else ""
        group_key = build_market_group_key(
            event_id=event_id,
            scope=scope,
            segment=segment,
            market_family=family,
            participant=participant,
            team_name=team_name,
            line=line,
            is_milestone=is_milestone,
            milestone_value=milestone_value,
        )
        rows.append(
            {
                "book": "DraftKings",
                "sport": "mlb",
                "event_id": event_id,
                "market_id": market_id,
                "selection_id": safe_str(sel.get("id")),
                "fetched_at": safe_str(payload.get("lastUpdatedTime")),
                "market_name": market_name,
                "market_family": family,
                "scope": scope,
                "segment": segment,
                "is_milestone": bool(is_milestone),
                "milestone_value": milestone_value,
                "line": line,
                "selection": selection,
                "participant": participant,
                "team_name": team_name,
                "home_team": home_team,
                "away_team": away_team,
                "price_american": price,
                "implied_prob": parse_american_to_prob(price),
                "market_group_key": group_key,
                "outcome_key": build_outcome_key(group_key, selection),
            }
        )
    return _rows_to_unified(rows)


def parse_fanduel_raw_payloads(event_pages: Iterable[dict[str, Any]], price_rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    market_meta: dict[str, dict[str, Any]] = {}
    event_meta: dict[str, dict[str, str]] = {}
    for page in event_pages:
        events = page.get("events") or {}
        if isinstance(events, dict):
            for event_id, event in events.items():
                event_id = safe_str(event_id)
                home_team = safe_str(_coalesce(event.get("homeTeamName"), event.get("homeTeam")))
                away_team = safe_str(_coalesce(event.get("awayTeamName"), event.get("awayTeam")))
                if not (home_team and away_team):
                    participants = event.get("participants") or []
                    if len(participants) >= 2:
                        away_team = away_team or safe_str(participants[0].get("name"))
                        home_team = home_team or safe_str(participants[1].get("name"))
                event_meta[event_id] = {"home_team": home_team, "away_team": away_team}
        markets = page.get("markets") or {}
        if isinstance(markets, dict):
            for market_id, market in markets.items():
                market_id = safe_str(market_id)
                if not market_id:
                    continue
                market_meta[market_id] = market

    rows: list[dict[str, Any]] = []
    for price_row in price_rows:
        market_id = safe_str(price_row.get("marketId"))
        market = market_meta.get(market_id, {})
        event_id = safe_str(market.get("eventId"))
        home_team = safe_str(event_meta.get(event_id, {}).get("home_team"))
        away_team = safe_str(event_meta.get(event_id, {}).get("away_team"))
        market_name = safe_str(_coalesce(market.get("marketName"), market.get("name"), market.get("marketType"), market_id))
        for runner in price_row.get("runnerDetails") or []:
            if not isinstance(runner, dict):
                continue
            participant = safe_str(_coalesce(runner.get("runnerName"), runner.get("name"), runner.get("participant")))
            selection = safe_str(_coalesce(runner.get("runnerName"), runner.get("selectionName"), runner.get("name"), participant))
            line = parse_line(_coalesce(runner.get("handicap"), runner.get("line"), runner.get("points")))
            win = runner.get("winRunnerOdds") or {}
            american = parse_line(_coalesce((win.get("americanDisplayOdds") or {}).get("americanOddsInt"), runner.get("price")))
            if american is None:
                continue
            team_name = ""
            if selection.lower() in {"home", home_team.lower()}:
                team_name = home_team
            elif selection.lower() in {"away", away_team.lower()}:
                team_name = away_team
            segment = detect_segment(market_name)
            family = detect_market_family(market_name)
            guessed_participant, guessed_team = infer_entity_from_market_name(market_name, family)
            generic_selection = selection.lower() in {"over", "under", "yes", "no", "home", "away"}
            if generic_selection and participant.lower() in {"over", "under", "yes", "no", "home", "away"}:
                participant = ""
            participant = participant or guessed_participant
            team_name = team_name or guessed_team
            is_milestone, milestone_value = detect_milestone(market_name, selection, line)
            scope = detect_scope(market_name, participant, team_name, selection)
            if scope != "player" and generic_selection:
                participant = ""
            group_key = build_market_group_key(
                event_id=event_id,
                scope=scope,
                segment=segment,
                market_family=family,
                participant=participant,
                team_name=team_name,
                line=line,
                is_milestone=is_milestone,
                milestone_value=milestone_value,
            )
            rows.append(
                {
                    "book": "FanDuel",
                    "sport": "mlb",
                    "event_id": event_id,
                    "market_id": market_id,
                    "selection_id": safe_str(runner.get("selectionId")),
                    "fetched_at": safe_str(_coalesce(price_row.get("lastUpdated"), market.get("lastUpdated"))),
                    "market_name": market_name,
                    "market_family": family,
                    "scope": scope,
                    "segment": segment,
                    "is_milestone": bool(is_milestone),
                    "milestone_value": milestone_value,
                    "line": line,
                    "selection": selection,
                    "participant": participant if scope == "player" else "",
                    "team_name": team_name,
                    "home_team": home_team,
                    "away_team": away_team,
                    "price_american": american,
                    "implied_prob": parse_american_to_prob(american),
                    "market_group_key": group_key,
                    "outcome_key": build_outcome_key(group_key, selection),
                }
            )
    return _rows_to_unified(rows)


def load_draftkings_input(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".csv", ".parquet", ".pq"}:
        table = load_table(path)
        return from_shared_odds_table(table, book="DraftKings")
    payload = _load_json(path)
    if isinstance(payload, list):
        frames = [parse_draftkings_raw_payload(item) for item in payload if isinstance(item, dict)]
        return ensure_unified_columns(pd.concat(frames, ignore_index=True)) if frames else ensure_unified_columns(pd.DataFrame())
    return parse_draftkings_raw_payload(payload)


def load_fanduel_input(event_pages_path: str | Path | None = None, price_rows_path: str | Path | None = None, shared_path: str | Path | None = None) -> pd.DataFrame:
    if shared_path:
        return from_shared_odds_table(load_table(shared_path), book="FanDuel")
    if not event_pages_path or not price_rows_path:
        raise ValueError("FanDuel load requires shared_path or both event_pages_path and price_rows_path")
    event_pages = _load_json(event_pages_path)
    price_rows = _load_json(price_rows_path)
    if isinstance(event_pages, dict):
        event_pages = [event_pages]
    if isinstance(price_rows, dict):
        for key in ("markets", "marketPrices", "result"):
            if isinstance(price_rows.get(key), list):
                price_rows = price_rows[key]
                break
        else:
            price_rows = [price_rows]
    return parse_fanduel_raw_payloads(event_pages, price_rows)
