from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests


FD_EVENT_PAGE_URL = "https://api.sportsbook.fanduel.com/sbapi/event-page"
FD_PRICE_URL = "https://smp.nj.sportsbook.fanduel.com/api/sports/fixedodds/readonly/v1/getMarketPrices"
FD_APP_KEY = "FhMFpcPWXMeyZxOx"

SPORT_FALLBACK_TABS: dict[str, tuple[str, ...]] = {
    "nba": (
        "popular",
        "quick-bets",
        "team-props",
        "player-combos",
        "player-points",
        "player-rebounds",
        "player-assists",
        "player-threes",
        "player-blocks",
        "player-steals",
        "player-turnovers",
        "1st-quarter",
        "2nd-quarter",
        "3rd-quarter",
        "4th-quarter",
        "half",
        "same-game-parlay-",
    ),
    "mlb": (
        "popular",
        "quick-bets",
        "batter-props",
        "pitcher-props",
        "team-props",
        "innings",
        "same-game-parlay-",
    ),
}


def _headers() -> dict[str, str]:
    return {
        "accept": "application/json",
        "origin": "https://sportsbook.fanduel.com",
        "referer": "https://sportsbook.fanduel.com/",
        "x-application": FD_APP_KEY,
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/147.0.0.0 Safari/537.36"
        ),
    }


def _to_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(str(value))
    except (TypeError, ValueError):
        return None


def event_page_url(event_id: str | int, tab: str | int | None = None) -> str:
    base = (
        f"{FD_EVENT_PAGE_URL}?_ak={FD_APP_KEY}"
        f"&eventId={event_id}"
        "&useCombinedTouchdownsVirtualMarket=true"
        "&useQuickBets=true"
    )
    if tab is not None and str(tab) != "":
        base += f"&tab={tab}"
    return base


def fetch_event_page(
    event_id: str | int,
    tab: str | int | None = None,
    session: requests.Session | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    sess = session or requests.Session()
    resp = sess.get(event_page_url(event_id, tab), headers=_headers(), timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    if isinstance(payload, dict):
        payload.setdefault("_fd_event_id", str(event_id))
        payload.setdefault("_fd_tab", tab)
    return payload


def fetch_event_pages(
    event_ids: Iterable[str | int],
    tabs: Iterable[str | int] | None = None,
    sport: str = "nba",
    session: requests.Session | None = None,
    timeout: int = 30,
) -> list[dict[str, Any]]:
    sess = session or requests.Session()
    pages: list[dict[str, Any]] = []
    active_tabs: tuple[str | int, ...]
    if tabs is not None:
        active_tabs = tuple(tabs)
    else:
        active_tabs = SPORT_FALLBACK_TABS.get(str(sport).lower(), ("popular", "quick-bets"))

    for event_id in event_ids:
        for tab in active_tabs:
            try:
                payload = fetch_event_page(event_id, tab, session=sess, timeout=timeout)
                payload["_fd_event_id"] = str(event_id)
                payload["_fd_tab"] = tab
                payload["_fd_sport"] = str(sport).lower()
                pages.append(payload)
            except requests.HTTPError:
                continue
    return pages


def _extract_event_id_candidates(text: str) -> list[str]:
    found: list[str] = []

    parsed = urlparse(text)
    qs = parse_qs(parsed.query)
    for value in qs.get("eventId", []):
        event_id = str(value).strip()
        if event_id.isdigit():
            found.append(event_id)

    for pattern in (
        r"EVENT:(\d{5,})",
        r"eventId=(\d{5,})",
        r"(?:^|\D)(\d{7,})(?:\D|$)",
    ):
        for match in re.findall(pattern, text):
            event_id = str(match).strip()
            if event_id.isdigit():
                found.append(event_id)

    deduped: list[str] = []
    seen: set[str] = set()
    for item in found:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def parse_event_ids(from_url: str | None) -> list[str]:
    if not from_url:
        raise ValueError(
            "FanDuel event discovery needs --from-url with a FanDuel event URL, a raw event ID, "
            "a comma-separated list of URLs/IDs, or a text file containing them."
        )

    raw_items: list[str] = []
    candidate_path = Path(from_url)
    if candidate_path.exists() and candidate_path.is_file():
        raw_items.extend(line.strip() for line in candidate_path.read_text(encoding="utf-8").splitlines())
    else:
        raw_items.extend(part.strip() for part in str(from_url).split(","))

    event_ids: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        if not item:
            continue
        for event_id in _extract_event_id_candidates(item):
            if event_id not in seen:
                seen.add(event_id)
                event_ids.append(event_id)

    if not event_ids:
        raise ValueError(
            f"Could not parse any FanDuel event IDs from: {from_url!r}. "
            "Pass a valid event URL, raw event ID, comma-separated list, or file path."
        )
    return event_ids


def get_layout_tab_ids(payload: dict[str, Any]) -> list[int | str]:
    layout = payload.get("layout") or {}
    order = layout.get("tabsDisplayOrder") or []
    out: list[int | str] = []
    seen: set[str] = set()
    for value in order:
        normalized: int | str = value
        int_value = _to_int(value)
        if int_value is not None:
            normalized = int_value
        key = str(normalized)
        if key not in seen:
            seen.add(key)
            out.append(normalized)
    return out


def _extract_top_level_market_ids(payload: dict[str, Any]) -> list[str]:
    markets = payload.get("markets")
    if isinstance(markets, dict) and markets:
        ids = [str(key) for key in markets.keys() if "." in str(key)]
        if ids:
            return sorted(set(ids))
    return []


def extract_market_ids(payload: dict[str, Any]) -> list[str]:
    top_level_ids = _extract_top_level_market_ids(payload)
    if top_level_ids:
        return top_level_ids

    ids: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            market_id = node.get("marketId") or node.get("id")
            if isinstance(market_id, (str, int)) and "." in str(market_id):
                ids.append(str(market_id))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return sorted(set(ids))


def build_price_history_body(market_ids: Iterable[str]) -> dict[str, list[str]]:
    ids = [str(m) for m in market_ids if m]
    return {"marketIds": ids}


def fetch_market_prices(
    market_ids: Iterable[str],
    session: requests.Session | None = None,
    timeout: int = 30,
    price_history: bool = True,
) -> list[dict[str, Any]]:
    ids = [str(m) for m in market_ids if m]
    if not ids:
        return []

    sess = session or requests.Session()
    resp = sess.post(
        f"{FD_PRICE_URL}?priceHistory={'1' if price_history else '0'}",
        headers={**_headers(), "content-type": "application/json"},
        json=build_price_history_body(ids),
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("markets", "marketPrices", "result"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    return []


def attach_market_prices(
    discovered_rows: list[dict[str, Any]],
    price_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_market = {str(row.get("marketId")): row for row in price_rows}
    out: list[dict[str, Any]] = []
    for row in discovered_rows:
        merged = dict(row)
        price_row = by_market.get(str(row.get("market_id") or row.get("marketId")))
        if price_row is not None:
            merged["price_history"] = price_row
        out.append(merged)
    return out


def normalize_price_row(price_row: dict[str, Any]) -> list[dict[str, Any]]:
    market_id = str(price_row.get("marketId", ""))
    out: list[dict[str, Any]] = []
    for runner in price_row.get("runnerDetails", []) or []:
        win = runner.get("winRunnerOdds") or {}
        prev = runner.get("previousWinRunnerOdds") or []
        american = (((win.get("americanDisplayOdds") or {}).get("americanOddsInt")))
        previous_american = None
        if prev:
            previous_american = ((((prev[0] or {}).get("americanDisplayOdds") or {}).get("americanOddsInt")))
        out.append(
            {
                "market_id": market_id,
                "selection_id": runner.get("selectionId"),
                "selection_name": runner.get("runnerName") or runner.get("name"),
                "runner_order": runner.get("runnerOrder"),
                "line": runner.get("handicap"),
                "price": american,
                "previous_price": previous_american,
                "price_history_count": len(prev),
                "runner_status": runner.get("runnerStatus"),
                "market_status": price_row.get("marketStatus"),
                "has_sgp": price_row.get("hasSGM"),
                "bet_delay": price_row.get("betDelay"),
            }
        )
    return out


def _nested_lookup(container: dict[str, Any] | None, key: Any) -> dict[str, Any]:
    if not isinstance(container, dict):
        return {}
    if key in container and isinstance(container[key], dict):
        return container[key]
    key_str = str(key)
    if key_str in container and isinstance(container[key_str], dict):
        return container[key_str]
    key_int = _to_int(key)
    if key_int is not None and key_int in container and isinstance(container[key_int], dict):
        return container[key_int]
    return {}


def _extract_event_snapshot(pages: list[dict[str, Any]], event_id: str) -> dict[str, Any]:
    for page in pages:
        attachments = page.get("attachments") or {}
        events = attachments.get("events") or page.get("events") or {}
        event = _nested_lookup(events, event_id)
        if event:
            return event
    return {}


def _extract_competition_snapshot(pages: list[dict[str, Any]], competition_id: Any) -> dict[str, Any]:
    for page in pages:
        attachments = page.get("attachments") or {}
        competitions = attachments.get("competitions") or page.get("competitions") or {}
        competition = _nested_lookup(competitions, competition_id)
        if competition:
            return competition
    return {}


def _extract_event_type_snapshot(pages: list[dict[str, Any]], event_type_id: Any) -> dict[str, Any]:
    for page in pages:
        attachments = page.get("attachments") or {}
        event_types = attachments.get("eventTypes") or page.get("eventTypes") or {}
        event_type = _nested_lookup(event_types, event_type_id)
        if event_type:
            return event_type
    return {}


def _build_market_context(pages: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    market_meta: dict[str, dict[str, Any]] = {}
    for page in pages:
        page_tab = page.get("_fd_tab")
        layout_tabs = (page.get("layout") or {}).get("tabs") or {}
        page_tab_info = _nested_lookup(layout_tabs, page_tab)
        page_tab_title = page_tab_info.get("title") if isinstance(page_tab_info, dict) else None

        markets = page.get("markets") or {}
        if not isinstance(markets, dict):
            continue
        for market_id, market in markets.items():
            if not isinstance(market, dict):
                continue
            key = str(market_id)
            existing = market_meta.get(key, {})
            merged = dict(existing)
            merged.update(market)
            if page_tab is not None and merged.get("fd_tab") is None:
                merged["fd_tab"] = page_tab
            if page_tab_title and merged.get("fd_tab_title") is None:
                merged["fd_tab_title"] = page_tab_title
            market_meta[key] = merged
    return market_meta


def _build_rows_from_price_rows(
    price_rows: list[dict[str, Any]],
    market_context: dict[str, dict[str, Any]],
    pages: list[dict[str, Any]],
    sport: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for price_row in price_rows:
        market_id = str(price_row.get("marketId", ""))
        market_info = market_context.get(market_id, {})
        event_id = str(
            price_row.get("eventId")
            or market_info.get("eventId")
            or price_row.get("_fd_event_id")
            or ""
        )
        event_info = _extract_event_snapshot(pages, event_id) if event_id else {}
        competition_id = (
            price_row.get("competitionId")
            or market_info.get("competitionId")
            or event_info.get("competitionId")
        )
        event_type_id = (
            price_row.get("eventTypeId")
            or market_info.get("eventTypeId")
            or event_info.get("eventTypeId")
        )
        competition_info = _extract_competition_snapshot(pages, competition_id)
        event_type_info = _extract_event_type_snapshot(pages, event_type_id)

        for row in normalize_price_row(price_row):
            enriched = {
                "source_book": "fanduel",
                "sport": str(sport).lower(),
                "event_id": event_id or None,
                "event_name": event_info.get("name") or event_info.get("eventName"),
                "competition_id": competition_id,
                "competition_name": competition_info.get("name"),
                "event_type_id": event_type_id,
                "event_type_name": event_type_info.get("name"),
                "market_name": market_info.get("marketName") or market_info.get("name") or market_info.get("title"),
                "market_type": market_info.get("marketType") or market_info.get("marketTypeName"),
                "fd_tab": market_info.get("fd_tab"),
                "fd_tab_title": market_info.get("fd_tab_title"),
                **row,
            }
            rows.append(enriched)
    return rows


def capture_fanduel_payloads(
    sport: str,
    from_url: str | None = None,
    use_browser: bool = False,
    session: requests.Session | None = None,
    timeout: int = 30,
) -> tuple[dict[str, Any], dict[str, Any]]:
    del use_browser  # requests path only in this adapter

    event_ids = parse_event_ids(from_url)
    sess = session or requests.Session()
    all_pages: list[dict[str, Any]] = []
    all_market_ids: list[str] = []

    for event_id in event_ids:
        base_page = fetch_event_page(event_id, tab=None, session=sess, timeout=timeout)
        base_page["_fd_event_id"] = str(event_id)
        base_page["_fd_tab"] = None
        base_page["_fd_sport"] = str(sport).lower()
        event_pages = [base_page]

        discovered_tab_ids = get_layout_tab_ids(base_page)
        if discovered_tab_ids:
            for tab_id in discovered_tab_ids:
                try:
                    tab_page = fetch_event_page(event_id, tab=tab_id, session=sess, timeout=timeout)
                    tab_page["_fd_event_id"] = str(event_id)
                    tab_page["_fd_tab"] = tab_id
                    tab_page["_fd_sport"] = str(sport).lower()
                    event_pages.append(tab_page)
                except requests.HTTPError:
                    continue
        else:
            fallback_pages = fetch_event_pages(
                [event_id],
                tabs=None,
                sport=sport,
                session=sess,
                timeout=timeout,
            )
            event_pages.extend(fallback_pages)

        all_pages.extend(event_pages)
        for page in event_pages:
            all_market_ids.extend(extract_market_ids(page))

    market_ids = sorted(set(str(m) for m in all_market_ids if m))
    price_rows = fetch_market_prices(market_ids, session=sess, timeout=timeout, price_history=True)

    payloads = {
        "sport": str(sport).lower(),
        "event_ids": event_ids,
        "pages": all_pages,
        "market_ids": market_ids,
        "price_rows": price_rows,
    }
    meta = {
        "source": "requests",
        "urls": len(all_pages) + (1 if market_ids else 0),
        "browser": False,
        "events": len(event_ids),
        "markets": len(market_ids),
    }
    return payloads, meta


def capture_fanduel_event_page_payloads(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    return capture_fanduel_payloads(*args, **kwargs)


def capture_event_page_payloads(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    return capture_fanduel_payloads(*args, **kwargs)


def fetch_fanduel_payloads(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    return capture_fanduel_payloads(*args, **kwargs)


def capture_payloads(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    return capture_fanduel_payloads(*args, **kwargs)


def normalize_fanduel_payloads(payloads: Any, sport: str | None = None) -> pd.DataFrame:
    if isinstance(payloads, pd.DataFrame):
        return payloads.copy()

    pages: list[dict[str, Any]]
    price_rows: list[dict[str, Any]]
    normalized_sport = str(sport or "").lower()

    if isinstance(payloads, dict):
        pages = list(payloads.get("pages") or [])
        price_rows = list(payloads.get("price_rows") or [])
        normalized_sport = str(payloads.get("sport") or normalized_sport or "unknown").lower()
    elif isinstance(payloads, list):
        pages = []
        price_rows = list(payloads)
        normalized_sport = normalized_sport or "unknown"
    else:
        raise TypeError(f"Unsupported payload type for normalization: {type(payloads)!r}")

    market_context = _build_market_context(pages)
    rows = _build_rows_from_price_rows(price_rows, market_context, pages, normalized_sport)
    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(
            columns=[
                "source_book",
                "sport",
                "event_id",
                "event_name",
                "competition_id",
                "competition_name",
                "event_type_id",
                "event_type_name",
                "market_id",
                "market_name",
                "market_type",
                "fd_tab",
                "fd_tab_title",
                "selection_id",
                "selection_name",
                "runner_order",
                "line",
                "price",
                "previous_price",
                "price_history_count",
                "runner_status",
                "market_status",
                "has_sgp",
                "bet_delay",
            ]
        )

    preferred_order = [
        "source_book",
        "sport",
        "event_id",
        "event_name",
        "competition_id",
        "competition_name",
        "event_type_id",
        "event_type_name",
        "market_id",
        "market_name",
        "market_type",
        "fd_tab",
        "fd_tab_title",
        "selection_id",
        "selection_name",
        "runner_order",
        "line",
        "price",
        "previous_price",
        "price_history_count",
        "runner_status",
        "market_status",
        "has_sgp",
        "bet_delay",
    ]
    remaining = [col for col in df.columns if col not in preferred_order]
    return df[preferred_order + remaining]


def normalize_fanduel_payloads_to_shared(payloads: Any, sport: str | None = None) -> pd.DataFrame:
    return normalize_fanduel_payloads(payloads=payloads, sport=sport)


def normalize_payloads_to_shared(payloads: Any, sport: str | None = None) -> pd.DataFrame:
    return normalize_fanduel_payloads(payloads=payloads, sport=sport)


def normalize_fanduel_shared_odds(payloads: Any, sport: str | None = None) -> pd.DataFrame:
    return normalize_fanduel_payloads(payloads=payloads, sport=sport)
