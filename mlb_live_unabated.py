from __future__ import annotations

import json
import pathlib
import re
from typing import Any

import pandas as pd
import requests

from mlb_live_schema import ensure_unified_columns, norm_text

BETTYPE_URL = "https://data.unabated.com/bettype"

# Safe fallbacks for the most common straight MLB game bet types when bettype
# reference data is unavailable locally.
FALLBACK_BETTYPE_MAP: dict[int, dict[str, Any]] = {
    1: {"name": "Moneyline", "betOn": "event", "sides": "team", "hasPoints": False},
    2: {"name": "Run Line", "betOn": "event", "sides": "team", "hasPoints": True},
    3: {"name": "Total Runs", "betOn": "event", "sides": "ou", "hasPoints": True},
    4: {"name": "Team Total Runs", "betOn": "team", "sides": "ou", "hasPoints": True},
}

SEGMENT_BY_PERIOD_ID = {
    1: "full_game",
    11: "inning_1",
    12: "inning_2",
    13: "inning_3",
    14: "inning_4",
    15: "inning_5",
    20: "first_five_innings",
}

COUNT_FAMILY_NAME_MAP = {
    "hits": "hits",
    "hit": "hits",
    "total hits": "hits",
    "total bases": "total_bases",
    "bases": "total_bases",
    "runs": "runs",
    "runs scored": "runs",
    "rbi": "rbis",
    "rbis": "rbis",
    "runs batted in": "rbis",
    "home run": "home_runs",
    "home runs": "home_runs",
    "strikeouts": "pitcher_strikeouts",
    "strikeout": "pitcher_strikeouts",
    "pitcher strikeouts": "pitcher_strikeouts",
    "outs": "pitching_outs",
    "outs recorded": "pitching_outs",
    "pitching outs": "pitching_outs",
    "earned runs": "earned_runs",
    "earned runs allowed": "earned_runs",
    "walks allowed": "walks_allowed",
    "hits allowed": "hits_allowed",
    "stolen bases": "stolen_bases",
    "stolen base": "stolen_bases",
    "singles": "singles",
    "doubles": "doubles",
    "triples": "triples",
    "hits+runs+rbis": "hits_runs_rbis",
    "hits runs rbis": "hits_runs_rbis",
    "team total": "team_total_runs",
    "team total runs": "team_total_runs",
    "team hits": "team_hits",
    "game total": "game_total_runs",
    "total runs": "game_total_runs",
    "game total runs": "game_total_runs",
    "first inning total runs": "inning_1_runs",
    "1st inning total runs": "inning_1_runs",
}


def _to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(str(value))
    except Exception:
        return None


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        try:
            return float(str(value).replace(",", "").strip())
        except Exception:
            return None


def _american_to_implied(price: Any) -> float | None:
    p = _to_float(price)
    if p is None or p == 0:
        return None
    if p > 0:
        return 100.0 / (p + 100.0)
    return abs(p) / (abs(p) + 100.0)


def _segment_from_period(period_type_id: Any) -> str:
    pid = _to_int(period_type_id)
    if pid is None:
        return "full_game"
    return SEGMENT_BY_PERIOD_ID.get(pid, f"period_{pid}")


def _normalize_name(value: Any) -> str:
    return norm_text(value).replace(" ", "")


def _market_source_map(payload: dict[str, Any]) -> dict[int, str]:
    out: dict[int, str] = {}
    for item in payload.get("marketSources") or []:
        if not isinstance(item, dict):
            continue
        sid = _to_int(item.get("id"))
        name = str(item.get("name") or "").strip()
        if sid is not None and name:
            out[sid] = name
    return out


def _people_map(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    raw = payload.get("people") or {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            sid = _to_int(k)
            if sid is not None and isinstance(v, dict):
                out[sid] = v
    return out


def _teams_map(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    raw = payload.get("teams") or {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            sid = _to_int(k)
            if sid is not None and isinstance(v, dict):
                out[sid] = v
    return out


def _display_person(person_id: Any, people: dict[int, dict[str, Any]]) -> str:
    pid = _to_int(person_id)
    if pid is None:
        return ""
    person = people.get(pid) or {}
    preferred = str(person.get("preferredName") or "").strip()
    first = str(person.get("firstName") or "").strip()
    last = str(person.get("lastName") or "").strip()
    if preferred and last:
        return f"{preferred} {last}".strip()
    return f"{first} {last}".strip()


def _display_team(team_id: Any, teams: dict[int, dict[str, Any]]) -> str:
    tid = _to_int(team_id)
    if tid is None:
        return ""
    team = teams.get(tid) or {}
    abbr = str(team.get("abbreviation") or "").strip()
    name = str(team.get("name") or "").strip()
    return abbr or name


def _event_team_names(event_teams: dict[str, Any] | None, teams: dict[int, dict[str, Any]]) -> tuple[str, str]:
    event_teams = event_teams or {}
    away = _display_team((event_teams.get("0") or {}).get("id") or (event_teams.get("0") or {}).get("teamId"), teams)
    home = _display_team((event_teams.get("1") or {}).get("id") or (event_teams.get("1") or {}).get("teamId"), teams)
    return home, away


def load_bettypes(path: str | pathlib.Path | None = None, url: str | None = None, timeout: int = 45) -> dict[int, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if path:
        p = pathlib.Path(path)
        if p.exists():
            raw = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                records = [x for x in raw if isinstance(x, dict)]
            elif isinstance(raw, dict):
                maybe = raw.get("data") or raw.get("betTypes") or raw.get("items")
                if isinstance(maybe, list):
                    records = [x for x in maybe if isinstance(x, dict)]
    if not records and url:
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            raw = r.json()
            if isinstance(raw, list):
                records = [x for x in raw if isinstance(x, dict)]
            elif isinstance(raw, dict):
                maybe = raw.get("data") or raw.get("betTypes") or raw.get("items")
                if isinstance(maybe, list):
                    records = [x for x in maybe if isinstance(x, dict)]
        except Exception:
            records = []
    out = {int(k): dict(v) for k, v in FALLBACK_BETTYPE_MAP.items()}
    for rec in records:
        bid = _to_int(rec.get("betTypeId") or rec.get("id"))
        if bid is not None:
            out[bid] = rec
    return out


def _canonical_market_family(name: str, bet_sub_type: str | None = None) -> str:
    raw = norm_text(name)
    bet_sub = norm_text(bet_sub_type)
    if raw in {"moneyline", "money line"}:
        return "moneyline"
    if raw in {"run line", "spread", "handicap"}:
        return "spread"
    if raw in {"total", "total runs", "game total", "game total runs"}:
        return "game_total_runs"
    if raw in {"team total", "team total runs"}:
        return "team_total_runs"

    for token, family in COUNT_FAMILY_NAME_MAP.items():
        if token in raw:
            return family

    if "milestone" in bet_sub and "total" in raw:
        return "game_total_runs"
    if "milestone" in bet_sub and "team" in raw:
        return "team_total_runs"

    return raw.replace(" ", "_") if raw else ""


def _infer_scope(family: str, bet_meta: dict[str, Any], person_id: Any, team_id: Any) -> str:
    family = str(family or "")
    bet_on = norm_text(bet_meta.get("betOn"))
    if family in {"moneyline", "spread", "game_total_runs", "inning_1_runs"}:
        return "game"
    if family in {"team_total_runs", "team_hits"}:
        return "team"
    if _to_int(person_id) is not None or bet_on == "player":
        return "player"
    if _to_int(team_id) is not None or bet_on == "team":
        return "team"
    return "game"


def _selection_from_side_index(side_index: int | None, bet_meta: dict[str, Any], participant: str = "", home_team: str = "", away_team: str = "") -> str:
    sides = norm_text(bet_meta.get("sides"))
    if sides == "ou":
        return "over" if side_index == 0 else "under"
    if sides == "team":
        return "away" if side_index == 0 else "home"
    if sides == "yes no":
        return "yes" if side_index == 0 else "no"
    if sides == "player":
        return "yes" if side_index == 0 else "no"
    if participant:
        return "yes" if side_index == 0 else "no"
    return "away" if side_index == 0 else "home"


def _derive_team_name(scope: str, selection: str, team_id: Any, event_teams: dict[str, Any] | None, teams: dict[int, dict[str, Any]]) -> str:
    if scope != "team":
        return ""
    if _to_int(team_id) is not None:
        return _display_team(team_id, teams)
    event_teams = event_teams or {}
    if selection == "away":
        return _display_team((event_teams.get("0") or {}).get("id") or (event_teams.get("0") or {}).get("teamId"), teams)
    if selection == "home":
        return _display_team((event_teams.get("1") or {}).get("id") or (event_teams.get("1") or {}).get("teamId"), teams)
    return ""


def _parse_side_index(value: str) -> int | None:
    m = re.search(r"si(\d+)", str(value))
    return int(m.group(1)) if m else None


def _parse_market_source_id(value: str) -> int | None:
    m = re.search(r"ms(\d+)", str(value))
    return int(m.group(1)) if m else None


def _is_real_offer(row: dict[str, Any], source_id: int | None = None) -> bool:
    price = row.get("price")
    source_price = row.get("sourcePrice")
    source_format = row.get("sourceFormat")
    if price in (None, "") and source_price in (None, ""):
        return False
    # Filter hub/reference rows like ms49 with no source formatting and no source price.
    if source_id is not None and source_price in (None, "") and source_format in (None, ""):
        return False
    return True


def _line_value(points: Any) -> float | None:
    value = _to_float(points)
    return value


def _is_milestone(bet_sub_type: Any, line: dict[str, Any]) -> bool:
    text = norm_text(bet_sub_type)
    if "milestone" in text:
        return True
    # book ladder lines inside alternateLines are also milestone-style if they differ from base threshold
    if line.get("alternateNumber") not in (None, "", 0):
        return True
    return False


def _flatten_props(payload: dict[str, Any], bettypes: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    people = _people_map(payload)
    teams = _teams_map(payload)
    market_sources = _market_source_map(payload)
    odds = payload.get("odds") or {}
    if not isinstance(odds, dict):
        return rows

    for bucket_key, bucket in odds.items():
        if not str(bucket_key).startswith("lg5:") or not isinstance(bucket, list):
            continue
        for market in bucket:
            if not isinstance(market, dict):
                continue
            period_type_id = _to_int(market.get("periodTypeId"))
            segment = _segment_from_period(period_type_id)
            event_id = str(market.get("eventId") or "").strip()
            if not event_id:
                continue
            bet_type_id = _to_int(market.get("betTypeId")) or -1
            bet_sub_type = str(market.get("betSubType") or "").strip()
            bet_meta = bettypes.get(bet_type_id, {"name": f"bet_type_{bet_type_id}"})
            family = _canonical_market_family(str(bet_meta.get("name") or f"bet_type_{bet_type_id}"), bet_sub_type)
            participant = _display_person(market.get("personId"), people)
            home_team, away_team = _event_team_names(market.get("eventTeams"), teams)
            base_scope = _infer_scope(family, bet_meta, market.get("personId"), market.get("teamId"))
            sides = market.get("sides") or {}
            for side_key, side_block in sides.items():
                if not isinstance(side_block, dict):
                    continue
                side_index = _parse_side_index(side_key)
                selection = _selection_from_side_index(side_index, bet_meta, participant, home_team, away_team)
                for ms_key, line in side_block.items():
                    if not isinstance(line, dict):
                        continue
                    source_id = _parse_market_source_id(ms_key)
                    if not _is_real_offer(line, source_id):
                        continue
                    market_line_id = _to_int(line.get("marketLineId"))
                    scope = base_scope
                    team_name = _derive_team_name(scope, selection, market.get("teamId"), market.get("eventTeams"), teams)
                    row = {
                        "event_id": event_id,
                        "market_group_key": str(market.get("key") or f"pt{period_type_id}:bt{bet_type_id}:e{event_id}:pi{market.get('personId') or ''}:tid{market.get('teamId') or ''}"),
                        "market_line_id": market_line_id,
                        "market_family": family,
                        "scope": scope,
                        "segment": segment,
                        "participant": participant,
                        "team_name": team_name,
                        "home_team": home_team,
                        "away_team": away_team,
                        "line": _line_value(line.get("points")),
                        "milestone_value": _line_value(line.get("points")) if _is_milestone(bet_sub_type, line) else pd.NA,
                        "is_milestone": _is_milestone(bet_sub_type, line),
                        "book": market_sources.get(source_id, f"market_source_{source_id}"),
                        "selection": selection,
                        "price_american": _to_float(line.get("americanPrice") if line.get("americanPrice") is not None else line.get("price")),
                        "implied_prob": _american_to_implied(line.get("americanPrice") if line.get("americanPrice") is not None else line.get("price")),
                        "bet_type_id": bet_type_id,
                        "bet_type_name": str(bet_meta.get("name") or f"bet_type_{bet_type_id}"),
                        "bet_sub_type": bet_sub_type,
                        "unabated_key": str(market.get("key") or ""),
                    }
                    rows.append(row)

                    for alt in line.get("alternateLines") or []:
                        if not isinstance(alt, dict):
                            continue
                        alt_source_id = _to_int(alt.get("marketSourceId")) or source_id
                        if not _is_real_offer(alt, alt_source_id):
                            continue
                        alt_points = _line_value(alt.get("points"))
                        alt_is_milestone = True if alt_points is not None else _is_milestone(bet_sub_type, alt)
                        alt_row = dict(row)
                        alt_row.update(
                            {
                                "market_line_id": _to_int(alt.get("marketLineId")) or market_line_id,
                                "book": market_sources.get(alt_source_id, f"market_source_{alt_source_id}"),
                                "line": alt_points,
                                "milestone_value": alt_points if alt_is_milestone else pd.NA,
                                "is_milestone": alt_is_milestone,
                                "price_american": _to_float(alt.get("americanPrice") if alt.get("americanPrice") is not None else alt.get("price")),
                                "implied_prob": _american_to_implied(alt.get("americanPrice") if alt.get("americanPrice") is not None else alt.get("price")),
                            }
                        )
                        rows.append(alt_row)
    return rows


def _flatten_game(payload: dict[str, Any], bettypes: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    teams = _teams_map(payload)
    market_sources = _market_source_map(payload)
    game_events = payload.get("gameOddsEvents") or {}
    if not isinstance(game_events, dict):
        return rows

    for bucket_key, bucket in game_events.items():
        if not str(bucket_key).startswith("lg5:") or not isinstance(bucket, list):
            continue
        for event in bucket:
            if not isinstance(event, dict):
                continue
            event_id = str(event.get("eventId") or "").strip()
            if not event_id:
                continue
            home_team, away_team = _event_team_names(event.get("eventTeams"), teams)
            segment = _segment_from_period(event.get("periodTypeId"))
            game_lines = event.get("gameOddsMarketSourcesLines") or {}
            if not isinstance(game_lines, dict):
                continue
            for composite_key, bt_map in game_lines.items():
                if not isinstance(bt_map, dict):
                    continue
                side_index = _parse_side_index(composite_key)
                source_id = _parse_market_source_id(composite_key)
                for bt_key, line in bt_map.items():
                    if not isinstance(line, dict):
                        continue
                    if not _is_real_offer(line, source_id):
                        continue
                    bet_type_id = _to_int(bt_key.replace("bt", "")) or _to_int(line.get("betTypeId")) or -1
                    bet_meta = bettypes.get(bet_type_id, {"name": f"bet_type_{bet_type_id}"})
                    family = _canonical_market_family(str(bet_meta.get("name") or f"bet_type_{bet_type_id}"), None)
                    selection = _selection_from_side_index(side_index, bet_meta, "", home_team, away_team)
                    scope = _infer_scope(family, bet_meta, None, None)
                    # straight side markets stay game scope; team totals stay team scope
                    if family in {"moneyline", "spread", "game_total_runs", "inning_1_runs"}:
                        scope = "game"
                    team_name = _derive_team_name(scope, selection, None, event.get("eventTeams"), teams)
                    row = {
                        "event_id": event_id,
                        "market_group_key": f"eid{event_id}:bt{bet_type_id}:pt{event.get('periodTypeId')}:pregame",
                        "market_line_id": _to_int(line.get("marketLineId")),
                        "market_family": family,
                        "scope": scope,
                        "segment": segment,
                        "participant": "",
                        "team_name": team_name,
                        "home_team": home_team,
                        "away_team": away_team,
                        "line": _line_value(line.get("points")),
                        "milestone_value": pd.NA,
                        "is_milestone": False,
                        "book": market_sources.get(source_id, f"market_source_{source_id}"),
                        "selection": selection,
                        "price_american": _to_float(line.get("americanPrice") if line.get("americanPrice") is not None else line.get("price")),
                        "implied_prob": _american_to_implied(line.get("americanPrice") if line.get("americanPrice") is not None else line.get("price")),
                        "bet_type_id": bet_type_id,
                        "bet_type_name": str(bet_meta.get("name") or f"bet_type_{bet_type_id}"),
                        "bet_sub_type": "",
                        "unabated_key": str(composite_key),
                    }
                    rows.append(row)

                    for alt in line.get("alternateLines") or []:
                        if not isinstance(alt, dict):
                            continue
                        alt_source_id = _to_int(alt.get("marketSourceId")) or source_id
                        if not _is_real_offer(alt, alt_source_id):
                            continue
                        alt_row = dict(row)
                        alt_row.update(
                            {
                                "market_line_id": _to_int(alt.get("marketLineId")) or _to_int(line.get("marketLineId")),
                                "book": market_sources.get(alt_source_id, f"market_source_{alt_source_id}"),
                                "line": _line_value(alt.get("points")),
                                "price_american": _to_float(alt.get("americanPrice") if alt.get("americanPrice") is not None else alt.get("price")),
                                "implied_prob": _american_to_implied(alt.get("americanPrice") if alt.get("americanPrice") is not None else alt.get("price")),
                            }
                        )
                        rows.append(alt_row)
    return rows


def load_unabated_bundle(
    *,
    props_path: str | pathlib.Path | None = None,
    game_path: str | pathlib.Path | None = None,
    props_url: str | None = None,
    game_url: str | None = None,
    bettypes_path: str | pathlib.Path | None = None,
    bettypes_url: str | None = BETTYPE_URL,
    timeout: int = 60,
) -> pd.DataFrame:
    headers = {
        "origin": "https://unabated.com",
        "referer": "https://unabated.com/",
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/147.0.0.0 Safari/537.36"
        ),
        "accept": "*/*",
    }

    def _load_payload(path: str | pathlib.Path | None, url: str | None) -> dict[str, Any] | None:
        if path:
            p = pathlib.Path(path)
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
        if url:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        return None

    props_payload = _load_payload(props_path, props_url)
    game_payload = _load_payload(game_path, game_url)
    bettypes = load_bettypes(path=bettypes_path, url=bettypes_url)

    rows: list[dict[str, Any]] = []
    if props_payload:
        rows.extend(_flatten_props(props_payload, bettypes))
    if game_payload:
        rows.extend(_flatten_game(game_payload, bettypes))

    if not rows:
        return ensure_unified_columns(pd.DataFrame())

    df = pd.DataFrame(rows)
    if "sport" not in df.columns:
        df["sport"] = "mlb"
    df["projection_source"] = pd.NA
    df["model_prob"] = pd.NA
    df["fair_prob"] = pd.NA
    df["edge_pct_points"] = pd.NA
    df["score_status"] = pd.NA
    return ensure_unified_columns(df)
