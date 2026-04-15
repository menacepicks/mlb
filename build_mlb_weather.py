
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

NWS_HEADERS = {
    "User-Agent": "(mlb-weather-model, contact@example.com)",
    "Accept": "application/geo+json",
}

# Current 2026 MLB home parks. Athletics now play at Sutter Health Park in West Sacramento.
STADIUMS = {
    "diamondbacks": {"team": "ARI Diamondbacks", "stadium": "Chase Field", "lat": 33.4453, "lon": -112.0667},
    "braves": {"team": "ATL Braves", "stadium": "Truist Park", "lat": 33.8907, "lon": -84.4677},
    "orioles": {"team": "BAL Orioles", "stadium": "Oriole Park at Camden Yards", "lat": 39.2839, "lon": -76.6217},
    "red_sox": {"team": "BOS Red Sox", "stadium": "Fenway Park", "lat": 42.3467, "lon": -71.0972},
    "cubs": {"team": "CHI Cubs", "stadium": "Wrigley Field", "lat": 41.9484, "lon": -87.6553},
    "white_sox": {"team": "CHI White Sox", "stadium": "Rate Field", "lat": 41.8300, "lon": -87.6338},
    "reds": {"team": "CIN Reds", "stadium": "Great American Ball Park", "lat": 39.0979, "lon": -84.5082},
    "guardians": {"team": "CLE Guardians", "stadium": "Progressive Field", "lat": 41.4962, "lon": -81.6852},
    "rockies": {"team": "COL Rockies", "stadium": "Coors Field", "lat": 39.7559, "lon": -104.9942},
    "tigers": {"team": "DET Tigers", "stadium": "Comerica Park", "lat": 42.3390, "lon": -83.0485},
    "astros": {"team": "HOU Astros", "stadium": "Daikin Park", "lat": 29.7573, "lon": -95.3555},
    "royals": {"team": "KC Royals", "stadium": "Kauffman Stadium", "lat": 39.0517, "lon": -94.4803},
    "angels": {"team": "LAA Angels", "stadium": "Angel Stadium", "lat": 33.8003, "lon": -117.8827},
    "dodgers": {"team": "LAD Dodgers", "stadium": "Dodger Stadium", "lat": 34.0739, "lon": -118.2400},
    "marlins": {"team": "MIA Marlins", "stadium": "loanDepot park", "lat": 25.7781, "lon": -80.2197},
    "brewers": {"team": "MIL Brewers", "stadium": "American Family Field", "lat": 43.0280, "lon": -87.9712},
    "twins": {"team": "MIN Twins", "stadium": "Target Field", "lat": 44.9817, "lon": -93.2775},
    "mets": {"team": "NY Mets", "stadium": "Citi Field", "lat": 40.7571, "lon": -73.8458},
    "yankees": {"team": "NYY Yankees", "stadium": "Yankee Stadium", "lat": 40.8296, "lon": -73.9262},
    "athletics": {"team": "ATH Athletics", "stadium": "Sutter Health Park", "lat": 38.5800, "lon": -121.5136},
    "phillies": {"team": "PHI Phillies", "stadium": "Citizens Bank Park", "lat": 39.9061, "lon": -75.1665},
    "pirates": {"team": "PIT Pirates", "stadium": "PNC Park", "lat": 40.4469, "lon": -80.0057},
    "padres": {"team": "SD Padres", "stadium": "Petco Park", "lat": 32.7073, "lon": -117.1566},
    "giants": {"team": "SF Giants", "stadium": "Oracle Park", "lat": 37.7786, "lon": -122.3893},
    "mariners": {"team": "SEA Mariners", "stadium": "T-Mobile Park", "lat": 47.5914, "lon": -122.3325},
    "cardinals": {"team": "STL Cardinals", "stadium": "Busch Stadium", "lat": 38.6226, "lon": -90.1928},
    "rays": {"team": "TB Rays", "stadium": "George M. Steinbrenner Field", "lat": 27.9800, "lon": -82.5066},
    "rangers": {"team": "TEX Rangers", "stadium": "Globe Life Field", "lat": 32.7473, "lon": -97.0847},
    "blue_jays": {"team": "TOR Blue Jays", "stadium": "Rogers Centre", "lat": 43.6414, "lon": -79.3894},
    "nationals": {"team": "WAS Nationals", "stadium": "Nationals Park", "lat": 38.8730, "lon": -77.0074},
}

TEAM_ALIASES = {
    "ari": "diamondbacks", "arizona": "diamondbacks", "diamondbacks": "diamondbacks",
    "atl": "braves", "atlanta": "braves", "braves": "braves",
    "bal": "orioles", "baltimore": "orioles", "orioles": "orioles",
    "bos": "red_sox", "boston": "red_sox", "red sox": "red_sox", "redsox": "red_sox",
    "chc": "cubs", "cubs": "cubs",
    "chw": "white_sox", "cws": "white_sox", "white sox": "white_sox", "whitesox": "white_sox",
    "cin": "reds", "cincinnati": "reds", "reds": "reds",
    "cle": "guardians", "cleveland": "guardians", "guardians": "guardians",
    "col": "rockies", "colorado": "rockies", "rockies": "rockies",
    "det": "tigers", "detroit": "tigers", "tigers": "tigers",
    "hou": "astros", "houston": "astros", "astros": "astros",
    "kc": "royals", "kansas city": "royals", "royals": "royals",
    "laa": "angels", "angels": "angels",
    "lad": "dodgers", "los angeles dodgers": "dodgers", "dodgers": "dodgers",
    "mia": "marlins", "miami": "marlins", "marlins": "marlins",
    "mil": "brewers", "milwaukee": "brewers", "brewers": "brewers",
    "min": "twins", "minnesota": "twins", "twins": "twins",
    "nym": "mets", "mets": "mets",
    "nyy": "yankees", "yankees": "yankees",
    "oak": "athletics", "athletics": "athletics", "a's": "athletics", "ath": "athletics",
    "phi": "phillies", "phillies": "phillies",
    "pit": "pirates", "pirates": "pirates",
    "sd": "padres", "sdp": "padres", "padres": "padres",
    "sea": "mariners", "mariners": "mariners",
    "sf": "giants", "sfg": "giants", "giants": "giants",
    "stl": "cardinals", "cardinals": "cardinals",
    "tb": "rays", "tbr": "rays", "rays": "rays", "tampa bay": "rays",
    "tex": "rangers", "rangers": "rangers",
    "tor": "blue_jays", "blue jays": "blue_jays", "bluejays": "blue_jays",
    "was": "nationals", "wsh": "nationals", "nationals": "nationals",
}

def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())

def _normalize_team_key(value: Any) -> str:
    text = _clean_text(value).lower().replace(".", "").replace("_", " ")
    if not text:
        return ""
    parts = text.split()
    candidates = [text]
    if parts:
        candidates.append(parts[0])
        candidates.append(parts[-1])
    if len(parts) >= 2:
        candidates.append(" ".join(parts[-2:]))
    for cand in candidates:
        cand = cand.strip()
        if cand in TEAM_ALIASES:
            return TEAM_ALIASES[cand]
    return text.replace(" ", "_")

def _load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table type: {path}")

def _coalesce_columns(df: pd.DataFrame, names: list[str], default: str = "") -> pd.Series:
    cols = [n for n in names if n in df.columns]
    if not cols:
        return pd.Series([default] * len(df), index=df.index)
    out = df[cols[0]].copy()
    for col in cols[1:]:
        out = out.where(out.notna() & (out.astype(str).str.strip() != ""), df[col])
    return out

def _prep_events(board: pd.DataFrame) -> pd.DataFrame:
    cols = {str(c).strip().lower(): c for c in board.columns}
    temp = board.rename(columns={v: k for k, v in cols.items()}).copy()
    event_id = _coalesce_columns(temp, ["event id", "event_id"], "")
    home = _coalesce_columns(temp, ["home team", "home_team"], "")
    away = _coalesce_columns(temp, ["away team", "away_team"], "")
    start = _coalesce_columns(temp, ["start time", "start_time"], "")
    event_name = _coalesce_columns(temp, ["event name", "event_name"], "")

    out = pd.DataFrame({
        "event_id": event_id.astype(str),
        "home_team": home.astype(str),
        "away_team": away.astype(str),
        "start_time": start.astype(str),
        "event_name": event_name.astype(str),
    })
    out = out[out["event_id"].str.strip().ne("")]
    if out["home_team"].eq("").any() or out["away_team"].eq("").any():
        # fallback parse from "AWAY @ HOME"
        mask = out["event_name"].str.contains("@", regex=False)
        parsed = out.loc[mask, "event_name"].str.split("@", n=1, expand=True)
        if not parsed.empty:
            out.loc[mask & out["away_team"].eq(""), "away_team"] = parsed[0].str.strip()
            out.loc[mask & out["home_team"].eq(""), "home_team"] = parsed[1].str.strip()
    out = out.drop_duplicates(subset=["event_id", "home_team", "away_team", "start_time"])
    return out.reset_index(drop=True)

def _parse_wind_speed_mph(text: str) -> float | None:
    import re
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text or "")]
    if not nums:
        return None
    return round(sum(nums) / len(nums), 2)

def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

def _pick_forecast_period(periods: list[dict[str, Any]], game_start: datetime | None) -> dict[str, Any]:
    if not periods:
        raise ValueError("No hourly forecast periods returned.")
    if game_start is None:
        return periods[0]
    best = periods[0]
    best_diff = None
    for period in periods:
        start = _parse_iso(period.get("startTime"))
        if start is None:
            continue
        diff = abs((start - game_start).total_seconds())
        if best_diff is None or diff < best_diff:
            best = period
            best_diff = diff
    return best

def fetch_hourly_weather(lat: float, lon: float, game_start: datetime | None, session: Any = requests) -> dict[str, Any]:
    points_url = f"https://api.weather.gov/points/{lat},{lon}"
    r_points = session.get(points_url, headers=NWS_HEADERS, timeout=20)
    r_points.raise_for_status()
    forecast_url = r_points.json()["properties"]["forecastHourly"]

    r_forecast = session.get(forecast_url, headers=NWS_HEADERS, timeout=20)
    r_forecast.raise_for_status()
    periods = (r_forecast.json().get("properties") or {}).get("periods") or []
    current = _pick_forecast_period(periods, game_start)

    wind_speed_text = _clean_text(current.get("windSpeed"))
    return {
        "temperature": current.get("temperature"),
        "temperature_unit": current.get("temperatureUnit"),
        "wind speed": _parse_wind_speed_mph(wind_speed_text),
        "wind_speed_text": wind_speed_text,
        "wind direction": _clean_text(current.get("windDirection")),
        "forecast": _clean_text(current.get("shortForecast")),
        "precipitation_chance": ((current.get("probabilityOfPrecipitation") or {}).get("value")),
        "weather_period_start": current.get("startTime"),
        "weather_period_end": current.get("endTime"),
        "is_daytime": current.get("isDaytime"),
    }

def build_weather_table(board: pd.DataFrame, session: Any = requests) -> pd.DataFrame:
    events = _prep_events(board)
    rows: list[dict[str, Any]] = []
    cache: dict[str, dict[str, Any]] = {}
    for rec in events.to_dict(orient="records"):
        home_key = _normalize_team_key(rec["home_team"])
        stadium = STADIUMS.get(home_key)
        if not stadium:
            row = dict(rec)
            row.update({
                "stadium": "",
                "latitude": pd.NA,
                "longitude": pd.NA,
                "weather_status": "missing_stadium_mapping",
            })
            rows.append(row)
            continue
        if home_key not in cache:
            cache[home_key] = {"stadium": stadium["stadium"], "latitude": stadium["lat"], "longitude": stadium["lon"]}
        game_start = _parse_iso(rec.get("start_time"))
        try:
            wx = fetch_hourly_weather(stadium["lat"], stadium["lon"], game_start, session=session)
            row = {
                **rec,
                "stadium": stadium["stadium"],
                "latitude": stadium["lat"],
                "longitude": stadium["lon"],
                "weather_status": "ok",
                **wx,
            }
        except Exception as exc:
            row = {
                **rec,
                "stadium": stadium["stadium"],
                "latitude": stadium["lat"],
                "longitude": stadium["lon"],
                "weather_status": f"error: {exc}",
            }
        rows.append(row)
    return pd.DataFrame(rows)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a daily MLB weather table from a board/schedule file using the NWS API.")
    p.add_argument("--board-file", required=True, help="CSV/parquet with event id, home team, away team, and start time columns")
    p.add_argument("--out", default="artifacts/mlb_weather.csv")
    return p.parse_args()

def main() -> int:
    args = parse_args()
    board = _load_table(args.board_file)
    weather = build_weather_table(board)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(out_path, index=False)
    print(f"weather_rows={len(weather)}")
    print(out_path)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
