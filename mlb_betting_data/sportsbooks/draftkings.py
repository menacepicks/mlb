"""DraftKings sportsbook adapter for the shared odds pipeline.

This module is intentionally built around **capture + normalize** instead of
hard-coding one unstable endpoint. DraftKings pages can render odds through
background JSON requests, so the adapter supports three input paths:

1. local JSON files containing captured network payloads,
2. local HTML files containing embedded JSON script tags,
3. live page capture from the user's machine via requests or Playwright.

The normalization step outputs the hub's shared odds schema so NBA/MLB/NHL/UFC
can all consume the same artifact.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    from ..odds_ingestion import COMMON_COLUMNS, normalize_market_name
except Exception:  # pragma: no cover - standalone/local fallback
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

    def normalize_market_name(sport: str, market: str) -> str:
        return str(market or "").strip()


DEFAULT_SPORT_URLS: dict[str, str] = {
    "live": "https://sportsbook.draftkings.com/live",
    "nba": "https://sportsbook.draftkings.com/leagues/basketball/nba",
    "mlb": "https://sportsbook.draftkings.com/leagues/baseball/mlb",
    "nhl": "https://sportsbook.draftkings.com/leagues/hockey/nhl",
    "ufc": "https://sportsbook.draftkings.com/leagues/mma/ufc",
}

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    )
}

DK_API_BASE = "https://sportsbook-nash.draftkings.com"
DK_PRIMARY_MARKETS_ENDPOINT = (
    f"{DK_API_BASE}/sites/US-SB/api/sportscontent/controldata/home/primaryMarkets/v1/markets"
)

SPORT_DIRECT_CONFIGS: dict[str, dict[str, Any]] = {
    "mlb": {
        "league_id": "84240",
        "page_url": DEFAULT_SPORT_URLS["mlb"],
        "event_categories": [
            "all-odds",
            "all-odds&subcategory=team-props",
            "all-odds&subcategory=player-props",
            "all-odds&subcategory=batter-props",
            "all-odds&subcategory=pitcher-props",
        ],
    },
    "nba": {
        "league_id": "42648",
        "page_url": DEFAULT_SPORT_URLS["nba"],
        "event_categories": [
            "all-odds",
            "all-odds&subcategory=player-props",
            "all-odds&subcategory=team-props",
        ],
    },

}

LEAGUE_ID_TO_SPORT: dict[str, str] = {
    "42648": "nba",
    "84240": "mlb",
}

JUNK_EVENT_RE = re.compile(
    r"(award|awards|special|specials|series|conference|rookie[- ]of[- ]the[- ]year|cy[- ]young|\bmvp\b|season|\d{4}-\d{2})",
    flags=re.IGNORECASE,
)


def _coerce_iso_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _sport_from_url(url: str) -> str:
    path = urlparse(str(url or "")).path.lower()
    if "/basketball/nba" in path or "/nba" in path:
        return "nba"
    if "/baseball/mlb" in path or "/mlb" in path:
        return "mlb"
    if "/hockey/nhl" in path or "/nhl" in path:
        return "nhl"
    if "/mma/ufc" in path or "/ufc" in path:
        return "ufc"
    return ""


def _annotate_payload(payload: Any, *, url: str = "", sport: str = "") -> Any:
    if not isinstance(payload, dict):
        return payload
    tagged = dict(payload)
    if url and "__dk_source_url" not in tagged:
        tagged["__dk_source_url"] = url
    resolved_sport = str(sport or _sport_from_url(url)).strip().lower()
    if resolved_sport and "__dk_sport" not in tagged:
        tagged["__dk_sport"] = resolved_sport
    return tagged


def _event_is_daily_candidate(event: dict[str, Any], *, sport: str = "") -> bool:
    if not isinstance(event, dict):
        return False
    status = str(event.get("status") or "").upper().strip()
    if status not in {"NOT_STARTED", "SCHEDULED"}:
        return False

    name = str(event.get("name") or "")
    seo = str(event.get("seoIdentifier") or "")
    if JUNK_EVENT_RE.search(name) or JUNK_EVENT_RE.search(seo):
        return False

    start_dt = _coerce_iso_datetime(event.get("startEventDate"))
    if start_dt is not None and start_dt > datetime.now(timezone.utc) + timedelta(days=14):
        return False

    participants = event.get("participants") or []
    participant_types = {str(p.get("type") or "") for p in participants if isinstance(p, dict)}
    resolved_sport = str(sport or _sport_from_url(str(event.get("__dk_source_url") or "")) or "").lower()

    if resolved_sport in {"mlb", "nba", "nhl"}:
        team_count = sum(1 for p in participants if isinstance(p, dict) and str(p.get("type") or "") == "Team")
        if team_count < 2:
            return False
        event_type = str(event.get("eventParticipantType") or "")
        if event_type and event_type != "TwoTeam":
            return False
    if resolved_sport == "ufc":
        if len(participants) != 2:
            return False
        if participant_types and participant_types == {"Team"}:
            return False

    return True


def _url_items(values: Iterable[Any]) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for value in values:
        if isinstance(value, tuple):
            url, sport = value
        else:
            url, sport = value, ""
        url = str(url or "").strip()
        sport = str(sport or "").strip().lower()
        if url:
            items.append((url, sport))
    return items


def _league_to_sport(value: Any) -> str:
    return LEAGUE_ID_TO_SPORT.get(str(value or "").strip(), "")


def _payload_sport_hint(payload: dict[str, Any], fallback: str = "") -> str:
    return str(payload.get("__dk_sport") or fallback or "").strip().lower()


def _keep_row_market(market: str) -> bool:
    text = _norm_text(market)
    banned = (
        "series",
        "award",
        "awards",
        "special",
        "specials",
        "conference",
        "season",
        "champion",
        "rookie of the year",
        "cy young",
        "mvp",
    )
    return not any(token in text for token in banned)


def _dk_is_real_market_payload(payload: Any) -> bool:
    if not _payload_has_market_data(payload):
        return False
    events = payload.get("events") or []
    return any(
        _event_is_daily_candidate(event, sport=_payload_sport_hint(payload))
        for event in events
        if isinstance(event, dict)
    )


def _dk_api_headers(*, feature: str, page: str) -> dict[str, str]:
    return {
        **DEFAULT_HEADERS,
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.8",
        "Content-Type": "application/json charset=utf-8",
        "Origin": "https://sportsbook.draftkings.com",
        "Referer": "https://sportsbook.draftkings.com/",
        "x-client-feature": feature,
        "x-client-name": "web",
        "x-client-page": page,
        "x-client-widget-name": "cms",
    }


def _payload_has_market_data(payload: Any) -> bool:
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("events"), list)
        and isinstance(payload.get("markets"), list)
        and isinstance(payload.get("selections"), list)
        and bool(payload.get("events"))
        and bool(payload.get("markets"))
        and bool(payload.get("selections"))
    )


def _capture_primary_markets_via_api(sport: str, *, timeout: int = 25) -> DraftKingsCaptureBundle | None:
    config = SPORT_DIRECT_CONFIGS.get(str(sport or "").strip().lower())
    if not config:
        return None
    league_id = str(config["league_id"])
    params = {
        "eventsQuery": f"$filter=leagueId eq '{league_id}'",
        "marketsQuery": "$filter=tags/any(t: t eq 'PrimaryMarket')",
        "top": "25",
        "include": "Events",
        "entity": "events",
        "isBatchable": "true",
    }
    response = requests.get(
        DK_PRIMARY_MARKETS_ENDPOINT,
        headers=_dk_api_headers(feature="primaryMarkets", page="home"),
        params=params,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = _annotate_payload(response.json(), url=response.url, sport=sport)
    if not _dk_is_real_market_payload(payload):
        return None
    return DraftKingsCaptureBundle(
        source=f"api_{sport}",
        payloads=[payload],
        urls=[response.url],
        used_browser=False,
    )


@dataclass(slots=True)
class DraftKingsCaptureBundle:
    source: str
    payloads: list[Any]
    urls: list[str]
    used_browser: bool = False


def _norm_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    for token in ["-", "_", "/", ".", ",", "(", ")"]:
        text = text.replace(token, " ")
    return " ".join(text.split())


def _coalesce(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def _iter_nodes(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _iter_nodes(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_nodes(item)


def _parse_json_text(text: str) -> Any | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _extract_json_from_html(html: str) -> list[Any]:
    soup = BeautifulSoup(html, "html.parser")
    payloads: list[Any] = []
    for script in soup.find_all("script"):
        script_text = script.string or script.get_text("", strip=False)
        payload = _parse_json_text(script_text)
        if payload is not None:
            payloads.append(payload)
    return payloads


def load_draftkings_payloads_from_files(paths: Iterable[str | Path]) -> DraftKingsCaptureBundle:
    payloads: list[Any] = []
    urls: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        urls.append(str(path))
        suffix = path.suffix.lower()
        text = path.read_text(encoding="utf-8")
        if suffix == ".json":
            payload = _parse_json_text(text)
            if payload is None:
                raise ValueError(f"Could not parse JSON from {path}")
            if isinstance(payload, list):
                payloads.extend(payload)
            else:
                payloads.append(payload)
            continue
        if suffix in {".html", ".htm"}:
            payloads.extend(_extract_json_from_html(text))
            continue
        raise ValueError(f"Unsupported DraftKings capture type: {path.suffix}")
    return DraftKingsCaptureBundle(source="files", payloads=payloads, urls=urls, used_browser=False)


def _fetch_html(url: str, timeout: int = 25) -> str:
    response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.text


def _capture_payloads_via_requests(urls: Iterable[Any], timeout: int = 25) -> DraftKingsCaptureBundle:
    payloads: list[Any] = []
    resolved_urls: list[str] = []
    for url, sport in _url_items(urls):
        resolved_urls.append(url)
        html = _fetch_html(url, timeout=timeout)
        for payload in _extract_json_from_html(html):
            payloads.append(_annotate_payload(payload, url=url, sport=sport))
    return DraftKingsCaptureBundle(source="requests", payloads=payloads, urls=resolved_urls, used_browser=False)


def _capture_payloads_via_playwright(urls: Iterable[Any], timeout_ms: int = 30000, settle_ms: int = 2500) -> DraftKingsCaptureBundle:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover - only hit when playwright missing locally
        raise RuntimeError(
            "Playwright is required for browser capture. Install it and run "
            "`python -m playwright install chromium` on the local machine."
        ) from exc

    payloads: list[Any] = []
    seen_payloads: set[str] = set()

    def remember(payload: Any, *, url: str = "", sport: str = "") -> None:
        payload = _annotate_payload(payload, url=url, sport=sport)
        try:
            key = json.dumps(payload, sort_keys=True, default=str)
        except TypeError:
            key = repr(payload)
        if key not in seen_payloads:
            seen_payloads.add(key)
            payloads.append(payload)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        try:
            page = browser.new_page(user_agent=DEFAULT_HEADERS["User-Agent"])

            current_url = ""
            current_sport = ""

            def handle_response(response: Any) -> None:
                try:
                    content_type = response.headers.get("content-type", "")
                    if "json" not in content_type.lower():
                        return
                    payload = response.json()
                    remember(payload, url=current_url or response.url, sport=current_sport)
                except Exception:
                    return

            page.on("response", handle_response)
            for url, sport in _url_items(urls):
                current_url = url
                current_sport = sport
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                except Exception:
                    page.goto(url, wait_until="load", timeout=timeout_ms)

                try:
                    page.wait_for_load_state("networkidle", timeout=5000)
                except Exception:
                    pass

                page.wait_for_timeout(3000)
                page.wait_for_timeout(settle_ms)
                for payload in _extract_json_from_html(page.content()):
                    remember(payload, url=url, sport=sport)
        finally:
            browser.close()

    return DraftKingsCaptureBundle(
        source="browser",
        payloads=payloads,
        urls=[url for url, _ in _url_items(urls)],
        used_browser=True,
    )


def _merge_capture_bundles(*bundles: DraftKingsCaptureBundle) -> DraftKingsCaptureBundle:
    payloads: list[Any] = []
    urls: list[str] = []
    seen_payloads: set[str] = set()
    seen_urls: set[str] = set()
    used_browser = False
    source_parts: list[str] = []

    for bundle in bundles:
        if bundle is None:
            continue
        used_browser = used_browser or bool(bundle.used_browser)
        if bundle.source and bundle.source not in source_parts:
            source_parts.append(bundle.source)
        for url in bundle.urls:
            if url not in seen_urls:
                seen_urls.add(url)
                urls.append(url)
        for payload in bundle.payloads:
            try:
                key = json.dumps(payload, sort_keys=True, default=str)
            except TypeError:
                key = repr(payload)
            if key not in seen_payloads:
                seen_payloads.add(key)
                payloads.append(payload)
    return DraftKingsCaptureBundle(source='+'.join(source_parts) or 'unknown', payloads=payloads, urls=urls, used_browser=used_browser)


def _extract_event_page_urls(payloads: Iterable[Any], *, sports: Iterable[str] | None = None) -> list[tuple[str, str]]:
    urls: list[tuple[str, str]] = []
    seen: set[str] = set()
    wanted = {str(s).strip().lower() for s in (sports or []) if str(s).strip()}
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        payload_sport = _payload_sport_hint(payload)
        for event in payload.get('events') or []:
            if not isinstance(event, dict):
                continue
            sport = _dk_sport_code_from_id(
                _dk_pick_first(event.get('sportId'), _league_to_sport(event.get('leagueId'))),
                sport_hint=payload_sport,
            )
            if wanted and sport not in wanted:
                continue
            if not _event_is_daily_candidate(event, sport=sport):
                continue
            event_id = str(event.get('id') or '').strip()
            seo = str(event.get('seoIdentifier') or '').strip()
            if not event_id or not seo:
                continue
            base = f"https://sportsbook.draftkings.com/event/{seo}/{event_id}"
            categories = SPORT_DIRECT_CONFIGS.get(sport, {}).get('event_categories') or ["all-odds"]
            for category in categories:
                candidate = f"{base}?category={category}"
                if candidate not in seen:
                    seen.add(candidate)
                    urls.append((candidate, sport))
    return urls


def capture_draftkings_payloads(
    *,
    sports: Iterable[str] | None = None,
    urls: Iterable[str] | None = None,
    use_browser: bool = False,
    timeout: int = 25,
) -> DraftKingsCaptureBundle:
    sport_list = [str(s).strip().lower() for s in (sports or ['nba', 'mlb', 'nhl', 'ufc']) if str(s).strip()]
    custom_urls = list(urls or [])
    bundles: list[DraftKingsCaptureBundle] = []
    fallback_urls: list[str] = list(custom_urls)

    if not custom_urls:
        for sport in sport_list:
            direct_bundle = None
            if sport in SPORT_DIRECT_CONFIGS:
                try:
                    direct_bundle = _capture_primary_markets_via_api(sport, timeout=timeout)
                except Exception:
                    direct_bundle = None
            if direct_bundle is not None and any(_dk_is_real_market_payload(p) for p in direct_bundle.payloads):
                bundles.append(direct_bundle)
            elif sport in DEFAULT_SPORT_URLS:
                fallback_urls.append(DEFAULT_SPORT_URLS[sport])

    if fallback_urls:
        fallback_bundle = (
            _capture_payloads_via_playwright(fallback_urls, timeout_ms=timeout * 1000)
            if use_browser
            else _capture_payloads_via_requests(fallback_urls, timeout=timeout)
        )
        bundles.append(fallback_bundle)

    if not bundles:
        raise ValueError('No DraftKings URLs or API payloads resolved for capture.')

    source_payloads: list[Any] = []
    for bundle in bundles:
        source_payloads.extend(bundle.payloads)

    event_urls = _extract_event_page_urls(source_payloads, sports=sport_list)
    if event_urls:
        extra_bundle = (
            _capture_payloads_via_playwright(event_urls, timeout_ms=timeout * 1000, settle_ms=1500)
            if use_browser
            else _capture_payloads_via_requests(event_urls, timeout=timeout)
        )
        bundles.append(extra_bundle)

    return _merge_capture_bundles(*bundles)


def _parse_price(value: Any) -> float | None:
    if value in (None, ""):
        return None
    text = str(value).strip().replace("−", "-")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        text = text.replace("+", "")
        try:
            return float(text)
        except ValueError:
            return None


def _parse_line(value: Any) -> float | None:
    if value in (None, ""):
        return None
    text = str(value).strip().replace("−", "-")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _infer_sport(value: Any, fallback: str = "") -> str:
    raw = _norm_text(value)
    if raw in {"nba", "mlb", "nhl", "ufc"}:
        return raw
    mapping = {
        "basketball": "nba",
        "baseball": "mlb",
        "hockey": "nhl",
        "mma": "ufc",
    }
    return mapping.get(raw, fallback)


def _event_info_from_node(node: dict[str, Any], sport_hint: str = "") -> tuple[str, dict[str, str]] | None:
    event_id = _coalesce(node, "eventId", "event_id", "eventID", "eventIdStr", "eventUuid")
    if event_id is None:
        return None

    home_team = str(_coalesce(node, "homeTeam", "home_team", "homeTeamName") or "").strip()
    away_team = str(_coalesce(node, "awayTeam", "away_team", "awayTeamName") or "").strip()
    participants = node.get("participants")
    if isinstance(participants, list):
        for participant in participants:
            if not isinstance(participant, dict):
                continue
            name = str(_coalesce(participant, "name", "participantName", "label") or "").strip()
            role = _norm_text(_coalesce(participant, "venueRole", "role", "alignment") or "")
            if not name:
                continue
            if role in {"home", "team1", "fighter1", "red"} and not home_team:
                home_team = name
            elif role in {"away", "team2", "fighter2", "blue"} and not away_team:
                away_team = name
            elif not away_team:
                away_team = name
            elif not home_team:
                home_team = name

    if (not home_team or not away_team) and isinstance(node.get("name"), str):
        name = node["name"].strip()
        if "@" in name:
            away_guess, home_guess = [part.strip() for part in name.split("@", 1)]
            away_team = away_team or away_guess
            home_team = home_team or home_guess
        elif " at " in name.lower():
            left, right = re.split(r"\s+at\s+", name, maxsplit=1, flags=re.IGNORECASE)
            away_team = away_team or left.strip()
            home_team = home_team or right.strip()

    return str(event_id).strip(), {
        "sport": _infer_sport(_coalesce(node, "sport", "sportName", "league"), fallback=sport_hint),
        "home_team": home_team,
        "away_team": away_team,
    }


def _build_event_map(payloads: Iterable[Any], sport_hint: str = "") -> dict[str, dict[str, str]]:
    event_map: dict[str, dict[str, str]] = {}
    for payload in payloads:
        for node in _iter_nodes(payload):
            info = _event_info_from_node(node, sport_hint=sport_hint)
            if info is None:
                continue
            event_id, event_payload = info
            existing = event_map.get(event_id, {})
            merged = {**existing, **{k: v for k, v in event_payload.items() if v}}
            event_map[event_id] = merged
    return event_map


def _infer_selection(label: Any, participant: str, home_team: str, away_team: str) -> str:
    text = _norm_text(label)
    participant_norm = _norm_text(participant)
    home_norm = _norm_text(home_team)
    away_norm = _norm_text(away_team)
    if text in {"over", "o"}:
        return "over"
    if text in {"under", "u"}:
        return "under"
    if text in {"yes", "y"}:
        return "yes"
    if text in {"no", "n"}:
        return "no"
    if home_norm and text == home_norm:
        return "home"
    if away_norm and text == away_norm:
        return "away"
    if participant_norm and text == participant_norm:
        return "yes"
    if text in {"home", "away"}:
        return text
    return text


def _guess_market_label(node: dict[str, Any], outcome: dict[str, Any]) -> str:
    return str(
        _coalesce(
            outcome,
            "market",
            "marketType",
            "criterionName",
            "subcategoryName",
            "offerCategoryName",
        )
        or _coalesce(
            node,
            "market",
            "marketType",
            "marketName",
            "criterionName",
            "subcategoryName",
            "offerCategoryName",
            "offerSubcategoryName",
            "label",
            "name",
        )
        or ""
    ).strip()


def _normalize_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=COMMON_COLUMNS)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=COMMON_COLUMNS)
    for col in COMMON_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[COMMON_COLUMNS].copy()
    if not df.empty:
        df["sport"] = df["sport"].map(lambda x: _infer_sport(x))
        df["market"] = [normalize_market_name(sport, market) for sport, market in zip(df["sport"], df["market"], strict=False)]
        df = df.drop_duplicates().reset_index(drop=True)
    return df


def normalize_draftkings_payloads(payloads: Iterable[Any], *, sport_hint: str = "") -> pd.DataFrame:
    payload_list = list(payloads)
    if not payload_list:
        return pd.DataFrame(columns=COMMON_COLUMNS)

    event_map = _build_event_map(payload_list, sport_hint=sport_hint)
    rows: list[dict[str, Any]] = []

    for payload in payload_list:
        for node in _iter_nodes(payload):
            outcomes = node.get("outcomes")
            if not isinstance(outcomes, list):
                outcomes = node.get("selections") if isinstance(node.get("selections"), list) else None
            if not outcomes:
                continue

            for outcome in outcomes:
                if not isinstance(outcome, dict):
                    continue
                event_id = str(
                    _coalesce(outcome, "eventId", "event_id", "eventID")
                    or _coalesce(node, "eventId", "event_id", "eventID")
                    or ""
                ).strip()
                event_info = event_map.get(event_id, {})
                participant = str(
                    _coalesce(outcome, "participant", "participantName", "playerName", "player", "fighter")
                    or _coalesce(node, "participant", "participantName", "playerName", "player", "fighter")
                    or ""
                ).strip()
                home_team = str(event_info.get("home_team") or "").strip()
                away_team = str(event_info.get("away_team") or "").strip()
                market_label = _guess_market_label(node, outcome)
                if not market_label:
                    continue
                selection_label = _coalesce(outcome, "selection", "label", "name", "outcomeName", "outcomeLabel")
                selection = _infer_selection(selection_label, participant, home_team, away_team)
                price = _parse_price(_coalesce(outcome, "oddsAmerican", "americanOdds", "price", "odds", "displayOdds"))
                if price is None:
                    continue
                line = _parse_line(_coalesce(outcome, "line", "points", "total", "value") or _coalesce(node, "line", "points", "total"))
                team_name = ""
                if selection in {"home", "away"}:
                    team_name = home_team if selection == "home" else away_team
                rows.append(
                    {
                        "sport": event_info.get("sport") or sport_hint,
                        "event_id": event_id,
                        "market": market_label,
                        "selection": selection,
                        "line": line,
                        "price": price,
                        "participant": participant,
                        "opponent": "",
                        "team_name": team_name,
                        "home_team": home_team,
                        "away_team": away_team,
                        "book": "DraftKings",
                        "fetched_at": str(
                            _coalesce(outcome, "fetched_at", "captured_at", "timestamp")
                            or _coalesce(node, "fetched_at", "captured_at", "timestamp")
                            or ""
                        ).strip(),
                    }
                )

    return _normalize_rows(rows)

# >>> DK FLAT NORMALIZER OVERRIDE START
_ORIGINAL_NORMALIZE_DRAFTKINGS_PAYLOADS = normalize_draftkings_payloads

from collections import defaultdict


def _dk_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _dk_to_float(value):
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except Exception:
        return None


def _dk_pick_first(*values):
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _dk_sport_code_from_id(raw, sport_hint=""):
    value = str(raw or "").strip().lower()
    mapping = {
        "2": "nba",
        "7": "mlb",
        "8": "nhl",
        "43": "ufc",
        "nba": "nba",
        "mlb": "mlb",
        "nhl": "nhl",
        "ufc": "ufc",
    }
    if value in mapping:
        return mapping[value]
    league_sport = _league_to_sport(raw)
    if league_sport:
        return league_sport
    hint = str(sport_hint or "").strip().lower()
    return hint


def _dk_extract_event_teams(event):
    home_team = ""
    away_team = ""
    for participant in (event.get("participants") or []):
        if not isinstance(participant, dict):
            continue
        role = str(participant.get("venueRole") or "").lower()
        name = str(participant.get("name") or "").strip()
        if role == "home" and name:
            home_team = name
        elif role == "away" and name:
            away_team = name
    return home_team, away_team


def _dk_extract_participant(selection):
    participants = selection.get("participants") or []
    if participants and isinstance(participants[0], dict):
        p = participants[0]
        return (
            str(p.get("id") or ""),
            str(p.get("name") or ""),
            str(p.get("venueRole") or ""),
            str(p.get("type") or ""),
        )
    return "", "", "", ""


def _dk_extract_line(selection):
    for key in ("points", "point", "line", "threshold", "value"):
        value = selection.get(key)
        if value not in (None, ""):
            parsed = _dk_to_float(value)
            if parsed is not None:
                return parsed
    metadata = selection.get("metadata") or {}
    for key in ("points", "point", "line", "threshold", "value"):
        value = metadata.get(key)
        if value not in (None, ""):
            parsed = _dk_to_float(value)
            if parsed is not None:
                return parsed
    selection_id = str(selection.get("id") or "")
    total_match = re.search(r"[OU](\d{3,5})(?:_|$)", selection_id)
    if total_match:
        return int(total_match.group(1)) / 100.0
    spread_match = re.search(r"([PN])(\d{3,5})(?:_|$)", selection_id)
    if spread_match:
        sign = 1.0 if spread_match.group(1) == "P" else -1.0
        return sign * (int(spread_match.group(2)) / 100.0)
    return None


def _dk_selection_to_shared(selection, participant_name, home_team, away_team):
    label = _dk_pick_first(selection.get("label"), selection.get("name"), selection.get("outcomeType"))
    return _infer_selection(label, participant_name, home_team, away_team)


def _dk_clean_market_text(value):
    text = _norm_text(value)
    for suffix in (' o u', ' over under', 'ou'):
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
    return text


def _dk_market_name_to_shared(market_name, market_type_name):
    raw = _dk_clean_market_text(_dk_pick_first(market_type_name, market_name))
    mapping = {
        'moneyline': 'moneyline',
        'run line': 'spread',
        'puck line': 'spread',
        'spread': 'spread',
        'handicap': 'spread',
        'total': 'game_total_runs',
        'game total': 'game_total_runs',
        'game total runs': 'game_total_runs',
        'total runs': 'game_total_runs',
        'team total': 'team_runs',
        'points': 'points',
        'rebounds': 'rebounds',
        'assists': 'assists',
        'three pointers made': 'threes',
        '3 pointers made': 'threes',
        'threes': 'threes',
        'points rebounds assists': 'pra',
        'points + rebounds + assists': 'pra',
        'points rebounds': 'pr',
        'points + rebounds': 'pr',
        'points assists': 'pa',
        'points + assists': 'pa',
        'rebounds assists': 'ra',
        'rebounds + assists': 'ra',
        'steals': 'steals',
        'blocks': 'blocks',
        'steals blocks': 'steals_blocks',
        'steals + blocks': 'steals_blocks',
        'turnovers': 'turnovers',
        'shots on goal': 'shots_on_goal',
        'goalie saves': 'goalie_saves',
        'goals': 'goals',
        'significant strikes': 'significant_strikes',
        'takedowns': 'takedowns',
        'total bases': 'total_bases',
        'hits': 'hits',
        'total hits': 'hits',
        'runs scored': 'runs',
        'runs': 'runs',
        'runs batted in': 'rbis',
        'rbis': 'rbis',
        'rbi': 'rbis',
        'home runs': 'home_runs',
        'hits runs rbis': 'hits_runs_rbis',
        'hits + runs + rbis': 'hits_runs_rbis',
        'earned runs allowed': 'earned_runs',
        'strikeouts thrown': 'pitcher_strikeouts',
        'pitcher strikeouts': 'pitcher_strikeouts',
        'outs recorded': 'pitching_outs',
        'pitching outs': 'pitching_outs',
        'walks allowed': 'walks_allowed',
        'hits allowed': 'hits_allowed',
        'stolen bases': 'stolen_bases',
        'singles': 'singles',
        'doubles': 'doubles',
        'triples': 'triples',
    }
    return mapping.get(raw, raw.replace(' ', '_'))


def _dk_shared_rows_from_payload(payload, sport_hint=""):
    if not isinstance(payload, dict):
        return []
    events = payload.get("events")
    markets = payload.get("markets")
    selections = payload.get("selections")
    if not isinstance(events, list) or not isinstance(markets, list) or not isinstance(selections, list):
        return []

    events_by_id = {}
    for event in events:
        if isinstance(event, dict):
            event_id = str(event.get("id") or "").strip()
            if event_id:
                events_by_id[event_id] = event

    selections_by_market_id = defaultdict(list)
    for selection in selections:
        if isinstance(selection, dict):
            market_id = str(selection.get("marketId") or "").strip()
            if market_id:
                selections_by_market_id[market_id].append(selection)

    payload_sport = _payload_sport_hint(payload, fallback=sport_hint)
    fetched_at = _dk_pick_first(payload.get("lastUpdatedTime"), _dk_now_iso())
    rows = []
    for market in markets:
        if not isinstance(market, dict):
            continue
        if bool(market.get("isSuspended")):
            continue
        market_id = str(market.get("id") or "").strip()
        event_id = str(market.get("eventId") or "").strip()
        if not market_id or not event_id:
            continue
        event = events_by_id.get(event_id)
        if not event:
            continue
        event_status = str(event.get("status") or "").upper()
        sport = _dk_sport_code_from_id(
            _dk_pick_first(
                event.get("sportId"),
                market.get("sportId"),
                event.get("sport"),
                market.get("sport"),
                event.get("leagueId"),
                market.get("leagueId"),
            ),
            sport_hint=payload_sport,
        )
        if not _event_is_daily_candidate(event, sport=sport):
            continue
        if event_status and event_status not in {"NOT_STARTED", "SCHEDULED"}:
            continue
        home_team, away_team = _dk_extract_event_teams(event)
        market_type = market.get("marketType") or {}
        market_name = _dk_market_name_to_shared(market.get("name"), market_type.get("name"))
        if not sport or not _keep_row_market(market_name):
            continue
        for selection in selections_by_market_id.get(market_id, []):
            display_odds = selection.get("displayOdds") or {}
            price = _parse_price(_dk_pick_first(display_odds.get("american"), selection.get("price"), selection.get("oddsAmerican"), selection.get("americanOdds")))
            if price is None:
                continue
            participant_id, participant_name, participant_venue_role, participant_type = _dk_extract_participant(selection)
            shared_selection = _dk_selection_to_shared(selection, participant_name, home_team, away_team)
            team_name = ""
            if shared_selection == "home":
                team_name = home_team
            elif shared_selection == "away":
                team_name = away_team
            elif participant_venue_role == "HomePlayer":
                team_name = home_team
            elif participant_venue_role == "AwayPlayer":
                team_name = away_team
            rows.append({
                "sport": sport,
                "event_id": event_id,
                "market": market_name,
                "selection": shared_selection,
                "line": _dk_extract_line(selection),
                "price": price,
                "participant": participant_name,
                "opponent": "",
                "team_name": team_name,
                "home_team": home_team,
                "away_team": away_team,
                "book": "DraftKings",
                "fetched_at": str(fetched_at or "").strip(),
            })
    return rows


def normalize_draftkings_payloads(payloads, sport_hint=""):
    if payloads is None:
        payloads = []

    shared_rows = []
    for payload in payloads:
        shared_rows.extend(_dk_shared_rows_from_payload(payload, sport_hint=sport_hint))

    if shared_rows:
        return _normalize_rows(shared_rows)

    return _ORIGINAL_NORMALIZE_DRAFTKINGS_PAYLOADS(payloads, sport_hint=sport_hint)
# <<< DK FLAT NORMALIZER OVERRIDE END


# >>> DK WRAPPER/CAPTURE COMPAT OVERRIDE START
_ORIGINAL_CAPTURE_DRAFTKINGS_PAYLOADS = capture_draftkings_payloads


def _split_csv_like_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            out.extend(_split_csv_like_values(item))
        return out
    text = str(value).replace("\r", "\n")
    parts: list[str] = []
    for chunk in text.split("\n"):
        for part in chunk.split(","):
            item = str(part).strip()
            if item:
                parts.append(item)
    return parts


def _paths_or_urls_from_inputs(*values: Any) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        for item in _split_csv_like_values(value):
            if item not in seen:
                seen.add(item)
                out.append(item)
    return out


def _looks_like_local_capture_path(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if text.startswith(("http://", "https://")):
        return False
    suffix = Path(text).suffix.lower()
    return suffix in {".json", ".html", ".htm"}


def _resolve_url_items_for_capture(
    *,
    sport: str | None = None,
    sports: Iterable[str] | None = None,
    from_url: Any = None,
    urls: Any = None,
) -> tuple[list[str], list[str]]:
    sport_list = [
        str(s).strip().lower()
        for s in _split_csv_like_values(list(sports or [])) + _split_csv_like_values(sport)
        if str(s).strip()
    ]
    if not sport_list:
        sport_list = ["nba", "mlb", "nhl", "ufc"]

    path_or_url_list = _paths_or_urls_from_inputs(from_url, urls)
    return sport_list, path_or_url_list


def _bundle_to_wrapper_result(bundle: DraftKingsCaptureBundle) -> tuple[list[Any], dict[str, Any]]:
    meta = {
        "source": bundle.source,
        "urls": len(bundle.urls),
        "browser": bool(bundle.used_browser),
    }
    return bundle.payloads, meta


def capture_draftkings_payloads(
    *,
    sport: str | None = None,
    sports: Iterable[str] | None = None,
    from_url: Any = None,
    urls: Any = None,
    use_browser: bool = False,
    timeout: int = 25,
    return_bundle: bool = False,
) -> DraftKingsCaptureBundle | tuple[list[Any], dict[str, Any]]:
    sport_list, path_or_url_list = _resolve_url_items_for_capture(
        sport=sport,
        sports=sports,
        from_url=from_url,
        urls=urls,
    )

    file_inputs = [item for item in path_or_url_list if _looks_like_local_capture_path(item)]
    url_inputs = [item for item in path_or_url_list if item not in file_inputs]

    bundles: list[DraftKingsCaptureBundle] = []

    if file_inputs:
        bundles.append(load_draftkings_payloads_from_files(file_inputs))

    if url_inputs or not file_inputs:
        live_bundle = _ORIGINAL_CAPTURE_DRAFTKINGS_PAYLOADS(
            sports=sport_list,
            urls=url_inputs or None,
            use_browser=use_browser,
            timeout=timeout,
        )
        bundles.append(live_bundle)

    bundle = _merge_capture_bundles(*bundles)
    if return_bundle:
        return bundle
    return _bundle_to_wrapper_result(bundle)


def _fallback_event_from_market(market: dict[str, Any], *, sport_hint: str = "") -> dict[str, Any]:
    return {
        "id": str(market.get("eventId") or "").strip(),
        "sportId": market.get("sportId"),
        "leagueId": market.get("leagueId"),
        "status": "NOT_STARTED",
        "participants": [],
        "__dk_sport": sport_hint,
    }


def _dk_shared_rows_from_flat_payload_no_events(payload, sport_hint=""):
    if not isinstance(payload, dict):
        return []
    markets = payload.get("markets")
    selections = payload.get("selections")
    if not isinstance(markets, list) or not isinstance(selections, list) or not markets or not selections:
        return []

    # If real events are present, the existing flat normalizer should handle it.
    events = payload.get("events")
    if isinstance(events, list) and events:
        return []

    selections_by_market_id = defaultdict(list)
    for selection in selections:
        if not isinstance(selection, dict):
            continue
        market_id = str(selection.get("marketId") or "").strip()
        if market_id:
            selections_by_market_id[market_id].append(selection)

    payload_sport = _payload_sport_hint(payload, fallback=sport_hint)
    fetched_at = _dk_pick_first(payload.get("lastUpdatedTime"), _dk_now_iso())
    rows = []

    for market in markets:
        if not isinstance(market, dict):
            continue
        if bool(market.get("isSuspended")):
            continue

        market_id = str(market.get("id") or "").strip()
        event_id = str(market.get("eventId") or "").strip()
        if not market_id or not event_id:
            continue

        sport = _dk_sport_code_from_id(
            _dk_pick_first(
                market.get("sportId"),
                market.get("sport"),
                market.get("leagueId"),
                market.get("__dk_sport"),
            ),
            sport_hint=payload_sport,
        )
        market_type = market.get("marketType") or {}
        market_name = _dk_market_name_to_shared(market.get("name"), market_type.get("name"))
        if not sport or not _keep_row_market(market_name):
            continue

        fake_event = _fallback_event_from_market(market, sport_hint=sport)
        home_team, away_team = _dk_extract_event_teams(fake_event)

        for selection in selections_by_market_id.get(market_id, []):
            display_odds = selection.get("displayOdds") or {}
            price = _parse_price(
                _dk_pick_first(
                    display_odds.get("american"),
                    selection.get("price"),
                    selection.get("oddsAmerican"),
                    selection.get("americanOdds"),
                )
            )
            if price is None:
                continue

            participant_id, participant_name, participant_venue_role, participant_type = _dk_extract_participant(selection)
            shared_selection = _dk_selection_to_shared(selection, participant_name, home_team, away_team)

            team_name = ""
            if shared_selection == "home":
                team_name = home_team
            elif shared_selection == "away":
                team_name = away_team
            elif participant_venue_role == "HomePlayer":
                team_name = home_team
            elif participant_venue_role == "AwayPlayer":
                team_name = away_team

            rows.append(
                {
                    "sport": sport,
                    "event_id": event_id,
                    "market": market_name,
                    "selection": shared_selection,
                    "line": _dk_extract_line(selection),
                    "price": price,
                    "participant": participant_name,
                    "opponent": "",
                    "team_name": team_name,
                    "home_team": home_team,
                    "away_team": away_team,
                    "book": "DraftKings",
                    "fetched_at": str(fetched_at or "").strip(),
                }
            )

    return rows


_ORIGINAL_NORMALIZE_DRAFTKINGS_PAYLOADS_FLAT = normalize_draftkings_payloads


def normalize_draftkings_payloads(payloads, sport_hint="", sport=None):
    if payloads is None:
        payloads = []

    resolved_sport_hint = str(sport or sport_hint or "").strip().lower()
    payload_list = list(payloads)
    shared_rows = []

    for payload in payload_list:
        shared_rows.extend(_dk_shared_rows_from_payload(payload, sport_hint=resolved_sport_hint))
        shared_rows.extend(_dk_shared_rows_from_flat_payload_no_events(payload, sport_hint=resolved_sport_hint))

    if shared_rows:
        return _normalize_rows(shared_rows)

    return _ORIGINAL_NORMALIZE_DRAFTKINGS_PAYLOADS_FLAT(payload_list, sport_hint=resolved_sport_hint)


def fetch_draftkings_payloads(**kwargs):
    return capture_draftkings_payloads(**kwargs)


def capture_payloads(**kwargs):
    return capture_draftkings_payloads(**kwargs)


def normalize_payloads_to_shared(payloads=None, sport=None, sport_hint=""):
    return normalize_draftkings_payloads(payloads or [], sport=sport, sport_hint=sport_hint)


def normalize_draftkings_payloads_to_shared(payloads=None, sport=None, sport_hint=""):
    return normalize_draftkings_payloads(payloads or [], sport=sport, sport_hint=sport_hint)


def normalize_draftkings_shared_odds(payloads=None, sport=None, sport_hint=""):
    return normalize_draftkings_payloads(payloads or [], sport=sport, sport_hint=sport_hint)
# <<< DK WRAPPER/CAPTURE COMPAT OVERRIDE END
