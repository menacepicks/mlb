from __future__ import annotations

from collections import defaultdict
from math import exp, factorial
from pathlib import Path
import re
import unicodedata

import difflib
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

from ..config import AppConfig
from ..io import ensure_dir, save_table
from ..pricing import (
    american_to_probability,
    format_american,
    format_decimal,
    format_percent,
    probability_to_american,
    probability_to_decimal,
)
from ..team_names import team_name_to_id


HITTER_TARGETS = {
    "hits": "hits",
    "total bases": "total bases",
    "home runs": "home runs",
    "runs scored": "runs scored",
    "runs batted in": "runs batted in",
    "walks": "walks",
    "strikeouts": "strikeouts",
    "stolen bases": "stolen bases",
}

PITCHER_TARGETS = {
    "pitcher strikeouts": "strikeouts",
    "outs recorded": "outs recorded",
    "earned runs allowed": "earned runs",
    "hits allowed": "hits allowed",
    "walks allowed": "walks allowed",
    "home runs allowed": "home runs allowed",
}

MARKET_ALIASES = {
    "hits": [("hits", "hitter")],
    "to record a hit": [("hits", "hitter")],
    "to get a hit": [("hits", "hitter")],
    "1 hit": [("hits", "hitter")],
    "1 plus hit": [("hits", "hitter")],
    "1+ hit": [("hits", "hitter")],
    "2+ hits": [("hits", "hitter")],
    "3+ hits": [("hits", "hitter")],
    "total bases": [("total bases", "hitter")],
    "tb": [("total bases", "hitter")],
    "to record 2 total bases": [("total bases", "hitter")],
    "to record two total bases": [("total bases", "hitter")],
    "2 plus total bases": [("total bases", "hitter")],
    "2+ total bases": [("total bases", "hitter")],
    "3+ total bases": [("total bases", "hitter")],
    "4+ total bases": [("total bases", "hitter")],
    "home runs": [("home runs", "hitter")],
    "home run": [("home runs", "hitter")],
    "to hit a home run": [("home runs", "hitter")],
    "to homer": [("home runs", "hitter")],
    "runs scored": [("runs scored", "hitter")],
    "runs": [("runs scored", "hitter")],
    "to score a run": [("runs scored", "hitter")],
    "1+ run": [("runs scored", "hitter")],
    "runs batted in": [("runs batted in", "hitter")],
    "run batted in": [("runs batted in", "hitter")],
    "rbi": [("runs batted in", "hitter")],
    "rbis": [("runs batted in", "hitter")],
    "to record an rbi": [("runs batted in", "hitter")],
    "to record a rbi": [("runs batted in", "hitter")],
    "1+ rbi": [("runs batted in", "hitter")],
    "walks": [("walks", "hitter")],
    "to record a walk": [("walks", "hitter")],
    "bases on balls": [("walks", "hitter")],
    "stolen bases": [("stolen bases", "hitter")],
    "stolen base": [("stolen bases", "hitter")],
    "to steal a base": [("stolen bases", "hitter")],
    "hitter strikeouts": [("strikeouts", "hitter")],
    "batter strikeouts": [("strikeouts", "hitter")],
    "batter k": [("strikeouts", "hitter")],
    "pitcher strikeouts": [("strikeouts", "pitcher")],
    "strikeouts thrown": [("strikeouts", "pitcher")],
    "pitching strikeouts": [("strikeouts", "pitcher")],
    "pitcher k": [("strikeouts", "pitcher")],
    "outs recorded": [("outs recorded", "pitcher")],
    "pitching outs": [("outs recorded", "pitcher")],
    "earned runs allowed": [("earned runs", "pitcher")],
    "earned runs": [("earned runs", "pitcher")],
    "hits allowed": [("hits allowed", "pitcher")],
    "walks allowed": [("walks allowed", "pitcher")],
    "home runs allowed": [("home runs allowed", "pitcher")],
}

NAME_CLEAN_RE = re.compile(r"[^a-z0-9]+")
SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def clean_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = text.replace("@", " ").replace("-", " ")
    text = SUFFIX_RE.sub(" ", text)
    text = NAME_CLEAN_RE.sub(" ", text)
    return "".join(text.split())


def _last_name_key(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = NAME_CLEAN_RE.sub(" ", text)
    parts = [p for p in text.split() if p and p not in {"jr", "sr", "ii", "iii", "iv", "v"}]
    return parts[-1] if parts else ""


def _first_initial(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = NAME_CLEAN_RE.sub(" ", text)
    parts = [p for p in text.split() if p and p not in {"jr", "sr", "ii", "iii", "iv", "v"}]
    return parts[0][0] if parts else ""


def _rolling_mean(values: list[float], window: int) -> float:
    if not values:
        return 0.0
    sample = values[-window:]
    return float(sum(sample) / len(sample))


def _build_hitter_training_table(
    hitter_df: pd.DataFrame,
    player_directory: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"player id", "date", "game id"}
    if hitter_df.empty or not required.issubset(hitter_df.columns):
        return pd.DataFrame(), pd.DataFrame()

    df = hitter_df.copy()
    df = df.sort_values(["player id", "date", "game id"]).reset_index(drop=True)

    name_map = (
        player_directory.sort_values(["player id", "season"])
        .drop_duplicates(subset=["player id"], keep="last")[["player id", "full name"]]
        .rename(columns={"full name": "player name"})
    )
    df = df.merge(name_map, on="player id", how="left")

    player_history: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    rows: list[dict[str, object]] = []
    latest_rows: dict[str, dict[str, object]] = {}

    for _, row in df.iterrows():
        pid = row["player id"]
        games_before_today = len(player_history[pid]["hits"])
        features: dict[str, object] = {
            "player id": pid,
            "player name": row.get("player name", ""),
            "date": row.get("date"),
            "season": row.get("season"),
            "is home": int(bool(row.get("is home"))),
            "games before today": games_before_today,
            "team id": row.get("team id", ""),
        }

        for target in HITTER_TARGETS.values():
            hist = player_history[pid][target]
            features[f"{target} recent 5"] = _rolling_mean(hist, 5)
            features[f"{target} recent 15"] = _rolling_mean(hist, 15)
            features[f"{target} season to date"] = _rolling_mean(hist, 9999)

        for target in HITTER_TARGETS.values():
            features[target] = float(row.get(target, 0) or 0)

        rows.append(features)

        for target in HITTER_TARGETS.values():
            player_history[pid][target].append(float(row.get(target, 0) or 0))
        latest_rows[pid] = {k: v for k, v in features.items() if k not in HITTER_TARGETS.values()}

    training = pd.DataFrame(rows)
    latest = pd.DataFrame(latest_rows.values())
    if not latest.empty:
        latest["player name clean"] = latest["player name"].map(clean_name)
        latest["player last name"] = latest["player name"].map(_last_name_key)
        latest["player first initial"] = latest["player name"].map(_first_initial)
    return training, latest


def _build_pitcher_training_table(
    pitcher_df: pd.DataFrame,
    player_directory: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"player id", "date", "game id"}
    if pitcher_df.empty or not required.issubset(pitcher_df.columns):
        return pd.DataFrame(), pd.DataFrame()

    df = pitcher_df.copy()
    df = df.sort_values(["player id", "date", "game id"]).reset_index(drop=True)

    name_map = (
        player_directory.sort_values(["player id", "season"])
        .drop_duplicates(subset=["player id"], keep="last")[["player id", "full name"]]
        .rename(columns={"full name": "player name"})
    )
    df = df.merge(name_map, on="player id", how="left")

    player_history: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    rows: list[dict[str, object]] = []
    latest_rows: dict[str, dict[str, object]] = {}

    for _, row in df.iterrows():
        pid = row["player id"]
        games_before_today = len(player_history[pid]["strikeouts"])
        features: dict[str, object] = {
            "player id": pid,
            "player name": row.get("player name", ""),
            "date": row.get("date"),
            "season": row.get("season"),
            "is home": int(bool(row.get("is home"))),
            "was starting pitcher": int(bool(row.get("was starting pitcher"))),
            "games before today": games_before_today,
            "team id": row.get("team id", ""),
        }

        for target in PITCHER_TARGETS.values():
            hist = player_history[pid][target]
            features[f"{target} recent 5"] = _rolling_mean(hist, 5)
            features[f"{target} recent 15"] = _rolling_mean(hist, 15)
            features[f"{target} season to date"] = _rolling_mean(hist, 9999)

        for target in PITCHER_TARGETS.values():
            features[target] = float(row.get(target, 0) or 0)

        rows.append(features)

        for target in PITCHER_TARGETS.values():
            player_history[pid][target].append(float(row.get(target, 0) or 0))
        latest_rows[pid] = {k: v for k, v in features.items() if k not in PITCHER_TARGETS.values()}

    training = pd.DataFrame(rows)
    latest = pd.DataFrame(latest_rows.values())
    if not latest.empty:
        latest["player name clean"] = latest["player name"].map(clean_name)
        latest["player last name"] = latest["player name"].map(_last_name_key)
        latest["player first initial"] = latest["player name"].map(_first_initial)
    return training, latest


def _build_team_form_table(team_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = team_df.copy()
    df = df.sort_values(["team id", "date", "game id"]).reset_index(drop=True)

    team_history: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    rows: list[dict[str, object]] = []
    latest_rows: dict[str, dict[str, object]] = {}

    for _, row in df.iterrows():
        team_id = row["team id"]
        games_before_today = len(team_history[team_id]["runs scored"])
        features: dict[str, object] = {
            "team id": team_id,
            "date": row.get("date"),
            "season": row.get("season"),
            "is home": int(bool(row.get("is home"))),
            "games before today": games_before_today,
            "team id": row.get("team id", ""),
        }

        for target in ["runs scored", "runs allowed", "won game"]:
            hist = team_history[team_id][target]
            features[f"{target} recent 5"] = _rolling_mean(hist, 5)
            features[f"{target} recent 15"] = _rolling_mean(hist, 15)
            features[f"{target} season to date"] = _rolling_mean(hist, 9999)

        features["run margin recent 5"] = float(features["runs scored recent 5"]) - float(features["runs allowed recent 5"])
        features["run margin recent 15"] = float(features["runs scored recent 15"]) - float(features["runs allowed recent 15"])
        features["run margin season to date"] = float(features["runs scored season to date"]) - float(features["runs allowed season to date"])

        features["runs scored"] = float(row.get("runs scored", 0) or 0)
        features["runs allowed"] = float(row.get("runs allowed", 0) or 0)
        features["won game"] = float(row.get("won game", 0) or 0)

        rows.append(features)

        team_history[team_id]["runs scored"].append(float(row.get("runs scored", 0) or 0))
        team_history[team_id]["runs allowed"].append(float(row.get("runs allowed", 0) or 0))
        team_history[team_id]["won game"].append(float(row.get("won game", 0) or 0))
        latest_rows[team_id] = {k: v for k, v in features.items() if k not in {"runs scored", "runs allowed", "won game"}}

    training = pd.DataFrame(rows)
    latest = pd.DataFrame(latest_rows.values())
    return training, latest


def _build_game_training_table(game_df: pd.DataFrame, team_training: pd.DataFrame) -> pd.DataFrame:
    game_df = game_df.copy()

    home_state = (
        team_training.rename(
            columns={
                "team id": "home team id",
                "games before today": "home games before today",
                "is home": "home team was home last time",
            }
        )
        .drop(columns=["runs scored", "runs allowed", "won game"], errors="ignore")
    )

    away_state = (
        team_training.rename(
            columns={
                "team id": "away team id",
                "games before today": "away games before today",
                "is home": "away team was home last time",
            }
        )
        .drop(columns=["runs scored", "runs allowed", "won game"], errors="ignore")
    )

    merged = game_df.merge(home_state, on=["home team id", "date"], how="left")
    merged = merged.merge(away_state, on=["away team id", "date"], how="left", suffixes=(" home", " away"))

    merged["home runs target"] = pd.to_numeric(merged["home runs"], errors="coerce")
    merged["away runs target"] = pd.to_numeric(merged["away runs"], errors="coerce")
    merged["total runs target"] = pd.to_numeric(merged["total runs"], errors="coerce")
    return merged


def _feature_columns(training_df: pd.DataFrame, target_name: str, extra_ignore: set[str] | None = None) -> list[str]:
    ignore = {
        "player id", "player name", "player name clean", "player last name", "player first initial",
        "date", "season", "team id", "home team id", "away team id", "game id", "winning team id",
        "day night", "ballpark id", "away starting pitcher id", "home starting pitcher id",
        "away starting lineup size", "home starting lineup size", "doubleheader number", "home team won",
        "home runs", "away runs", "total runs", "home runs target", "away runs target", "total runs target",
        *HITTER_TARGETS.values(), *PITCHER_TARGETS.values(),
    }
    if extra_ignore:
        ignore.update(extra_ignore)
    return [col for col in training_df.columns if col not in ignore and pd.api.types.is_numeric_dtype(training_df[col]) and col != target_name]


def _fit_poisson(training_df: pd.DataFrame, target_name: str, minimum_games: int = 5) -> tuple[PoissonRegressor, list[str], pd.DataFrame]:
    df = training_df.copy()

    if "games before today" in df.columns:
        df = df[df["games before today"] >= minimum_games].copy()

    if "home games before today" in df.columns and "away games before today" in df.columns:
        df = df[(df["home games before today"] >= minimum_games) & (df["away games before today"] >= minimum_games)].copy()

    if df.empty:
        df = training_df.copy()

    if df.empty:
        raise ValueError(f"No historical rows available for target: {target_name}")

    df[target_name] = pd.to_numeric(df[target_name], errors="coerce")
    df = df[df[target_name].notna()].copy()
    if df.empty:
        raise ValueError(f"No non-null historical rows available for target: {target_name}")

    train = df[df["season"] <= 2022].copy()
    valid = df[df["season"] == 2023].copy()
    test = df[df["season"] >= 2024].copy()
    if train.empty:
        train = df.copy()

    features = _feature_columns(df, target_name)
    if not features:
        raise ValueError(f"No usable features found for target: {target_name}")

    train_x = train[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    train_y = pd.to_numeric(train[target_name], errors="coerce").clip(lower=0.0)
    good_train = train_y.notna()
    train_x = train_x.loc[good_train]
    train_y = train_y.loc[good_train]
    if train_x.empty:
        raise ValueError(f"No usable training rows left after cleaning target: {target_name}")

    model = PoissonRegressor(alpha=1.0, max_iter=300)
    model.fit(train_x, train_y)

    summary_rows: list[dict[str, object]] = []
    for split_name, split_df in [("Train", train), ("Validation", valid), ("Test", test)]:
        if split_df.empty:
            continue
        split_x = split_df[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        split_y = pd.to_numeric(split_df[target_name], errors="coerce")
        good_split = split_y.notna()
        split_x = split_x.loc[good_split]
        split_y = split_y.loc[good_split]
        if split_x.empty:
            continue
        pred = model.predict(split_x)
        actual = split_y.astype(float)
        avg_miss = (pred - actual).abs().mean()
        summary_rows.append({
            "Target": target_name.title(),
            "Split": split_name,
            "Rows": int(len(split_x)),
            "Average miss": round(float(avg_miss), 3),
        })

    return model, features, pd.DataFrame(summary_rows)


def train_history_models(config: AppConfig) -> dict[str, Path]:
    hitter_df = pd.read_parquet(config.actual_hitter_stats_by_game)
    pitcher_df = pd.read_parquet(config.actual_pitcher_stats_by_game)
    team_df = pd.read_parquet(config.actual_team_game_results)
    game_df = pd.read_parquet(config.actual_game_results)
    player_directory = pd.read_parquet(config.player_directory)

    hitter_training, latest_hitter = _build_hitter_training_table(hitter_df, player_directory)
    pitcher_training, latest_pitcher = _build_pitcher_training_table(pitcher_df, player_directory)
    team_training, latest_team = _build_team_form_table(team_df)
    game_training = _build_game_training_table(game_df, team_training)

    ensure_dir(config.model_dir)
    summary_frames: list[pd.DataFrame] = []
    model_index_rows: list[dict[str, object]] = []

    def _record_skip(scope: str, market_name: str, reason: str) -> None:
        summary_frames.append(pd.DataFrame([{
            "Target": market_name.title(),
            "Split": "Skipped",
            "Rows": 0,
            "Average miss": np.nan,
            "Scope": scope,
            "Market": market_name.title(),
            "Reason": reason,
        }]))

    if not hitter_training.empty:
        for market_name, target_name in HITTER_TARGETS.items():
            if target_name not in hitter_training.columns:
                _record_skip("Hitter", market_name, "Target column missing")
                continue
            try:
                model, features, summary = _fit_poisson(hitter_training, target_name)
                model_path = config.model_dir / f"hitter {market_name}.joblib"
                joblib.dump({"model": model, "features": features, "target": target_name, "scope": "hitter"}, model_path)
                summary["Scope"] = "Hitter"
                summary["Market"] = market_name.title()
                summary_frames.append(summary)
                model_index_rows.append({"Scope": "Hitter", "Market": market_name.title(), "Model file": model_path.name})
            except Exception as exc:
                _record_skip("Hitter", market_name, str(exc))
    else:
        for market_name in HITTER_TARGETS:
            _record_skip("Hitter", market_name, "No hitter training rows")

    if not pitcher_training.empty:
        for market_name, target_name in PITCHER_TARGETS.items():
            if target_name not in pitcher_training.columns:
                _record_skip("Pitcher", market_name, "Target column missing")
                continue
            try:
                model, features, summary = _fit_poisson(pitcher_training, target_name)
                model_path = config.model_dir / f"pitcher {market_name}.joblib"
                joblib.dump({"model": model, "features": features, "target": target_name, "scope": "pitcher"}, model_path)
                summary["Scope"] = "Pitcher"
                summary["Market"] = market_name.title()
                summary_frames.append(summary)
                model_index_rows.append({"Scope": "Pitcher", "Market": market_name.title(), "Model file": model_path.name})
            except Exception as exc:
                _record_skip("Pitcher", market_name, str(exc))
    else:
        for market_name in PITCHER_TARGETS:
            _record_skip("Pitcher", market_name, "No pitcher training rows")

    for market_name, target_name in {
        "home team runs": "home runs target",
        "away team runs": "away runs target",
        "game total runs": "total runs target",
    }.items():
        if target_name not in game_training.columns:
            _record_skip("Game", market_name, "Target column missing")
            continue
        try:
            model, features, summary = _fit_poisson(game_training, target_name)
            model_path = config.model_dir / f"game {market_name}.joblib"
            joblib.dump({"model": model, "features": features, "target": target_name, "scope": "game"}, model_path)
            summary["Scope"] = "Game"
            summary["Market"] = market_name.title()
            summary_frames.append(summary)
            model_index_rows.append({"Scope": "Game", "Market": market_name.title(), "Model file": model_path.name})
        except Exception as exc:
            _record_skip("Game", market_name, str(exc))

    latest_frames = []
    if not latest_hitter.empty:
        latest_hitter["scope"] = "hitter"
        latest_frames.append(latest_hitter)
    if not latest_pitcher.empty:
        latest_pitcher["scope"] = "pitcher"
        latest_frames.append(latest_pitcher)
    latest_player_form = pd.concat(latest_frames, ignore_index=True) if latest_frames else pd.DataFrame()

    summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    model_index_df = pd.DataFrame(model_index_rows)
    if not summary_df.empty and not model_index_df.empty:
        summary_out = summary_df.merge(model_index_df, on=["Scope", "Market"], how="left")
    else:
        summary_out = summary_df

    save_table(summary_out, config.history_trained_model_summary)
    save_table(latest_player_form, config.latest_player_form)
    save_table(latest_team, config.latest_team_form)

    return {
        "history trained model summary": config.history_trained_model_summary,
        "latest player form": config.latest_player_form,
        "latest team form": config.latest_team_form,
        "history trained models folder": config.model_dir,
    }


def _poisson_pmf(k: int, mean: float) -> float:
    if mean < 0:
        return 0.0
    return exp(-mean) * (mean ** k) / factorial(k)


def _poisson_cdf(k: int, mean: float) -> float:
    if mean <= 0:
        return 1.0 if k >= 0 else 0.0
    total = 0.0
    for i in range(0, max(k, 0) + 1):
        total += _poisson_pmf(i, mean)
    return min(max(total, 0.0), 1.0)


def _win_probability(mean: float, selection: str, line_value: float | None) -> float | None:
    selection = str(selection or "").strip().lower()
    if selection not in {"over", "under", "yes", "no"}:
        return None

    if selection == "yes":
        threshold = 0.5 if line_value is None else line_value
        return _win_probability(mean, "over", threshold)
    if selection == "no":
        threshold = 0.5 if line_value is None else line_value
        return _win_probability(mean, "under", threshold)

    if line_value is None:
        return None

    if float(line_value).is_integer():
        if selection == "over":
            threshold = int(line_value)
            return 1.0 - _poisson_cdf(threshold, mean)
        threshold = int(line_value) - 1
        return _poisson_cdf(threshold, mean)

    floor_line = int(np.floor(line_value))
    if selection == "over":
        return 1.0 - _poisson_cdf(floor_line, mean)
    return _poisson_cdf(floor_line, mean)


def _load_models(config: AppConfig) -> dict[tuple[str, str], dict]:
    result: dict[tuple[str, str], dict] = {}
    for path in sorted(config.model_dir.glob("*.joblib")):
        payload = joblib.load(path)
        result[(payload["scope"], payload["target"])] = payload
    return result


def _normalize_market(market_text: str) -> str:
    text = unicodedata.normalize("NFKD", str(market_text or ""))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = " ".join(text.split())
    return text


PLUS_MARKET_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:\+|plus)")
NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)")


def _infer_line_from_market(market_text: str) -> float | None:
    market = _normalize_market(market_text)
    plus_match = PLUS_MARKET_RE.search(market)
    if plus_match:
        value = float(plus_match.group(1))
        return max(0.0, value - 0.5)
    if market in {"to hit a home run", "home run", "home runs", "to homer", "to record a hit", "to get a hit",
                  "to score a run", "to record an rbi", "to steal a base"}:
        return 0.5
    return None



def _clamp_probability(value: float | None, floor: float = 0.01, ceiling: float = 0.99) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(min(max(float(value), floor), ceiling))


def _candidate_market_targets(market_text: str) -> list[tuple[str, str]]:
    market = _normalize_market(market_text)
    candidates = []
    if market in MARKET_ALIASES:
        candidates.extend(MARKET_ALIASES[market])
    if market == "strikeouts":
        candidates.extend([("strikeouts", "hitter"), ("strikeouts", "pitcher")])
    if market in HITTER_TARGETS:
        candidates.append((HITTER_TARGETS[market], "hitter"))
    if market in PITCHER_TARGETS:
        candidates.append((PITCHER_TARGETS[market], "pitcher"))
    seen = set()
    deduped = []
    for item in candidates:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def _row_team_id(row: pd.Series) -> str:
    for field in ["team", "home team", "away team"]:
        value = row.get(field, "")
        team_id = team_name_to_id(value)
        if team_id:
            return team_id
    return ""


def _best_fuzzy_player_match(subset: pd.DataFrame, clean: str, team_id: str) -> pd.Series | None:
    if subset.empty or not clean:
        return None

    work = subset.copy()
    if team_id and "team id" in work.columns:
        same_team = work[work["team id"] == team_id]
        if not same_team.empty:
            work = same_team

    scores: list[tuple[float, int]] = []
    for idx, row in work.iterrows():
        name_clean = str(row.get("player name clean", "")).strip()
        if not name_clean:
            continue
        score = difflib.SequenceMatcher(None, clean, name_clean).ratio()
        scores.append((score, idx))

    if not scores:
        return None

    score, idx = max(scores, key=lambda x: x[0])
    if score < 0.72:
        return None
    return work.loc[idx]




def _match_player_state(row: pd.Series, player_form: pd.DataFrame, scope: str) -> pd.Series | None:
    subset = player_form[player_form["scope"] == scope].copy()
    if subset.empty:
        return None

    raw_name = str(row.get("player", "")).strip()
    clean = clean_name(raw_name)
    last = _last_name_key(raw_name)
    first_initial = _first_initial(raw_name)
    team_id = _row_team_id(row)

    if team_id and "team id" in subset.columns:
        same_team = subset[subset["team id"] == team_id]
        if not same_team.empty:
            subset = same_team

    exact = subset[subset["player name clean"] == clean]
    if not exact.empty:
        return exact.sort_values(["date", "season"]).iloc[-1]

    if last and first_initial and {"player last name", "player first initial"}.issubset(subset.columns):
        fuzzy = subset[
            (subset["player last name"] == last)
            & (subset["player first initial"] == first_initial)
        ]
        if not fuzzy.empty:
            return fuzzy.sort_values(["date", "season"]).iloc[-1]

    if last and "player last name" in subset.columns:
        fuzzy = subset[subset["player last name"] == last]
        if len(fuzzy) == 1:
            return fuzzy.sort_values(["date", "season"]).iloc[-1]

    fuzzy_best = _best_fuzzy_player_match(subset, clean, team_id)
    if fuzzy_best is not None:
        return fuzzy_best

    return None

    raw_name = str(row.get("player", "")).strip()
    clean = clean_name(raw_name)
    last = _last_name_key(raw_name)
    first_initial = _first_initial(raw_name)

    exact = subset[subset["player name clean"] == clean]
    if not exact.empty:
        return exact.sort_values(["date", "season"]).iloc[-1]

    if last and first_initial:
        fuzzy = subset[
            (subset["player last name"] == last)
            & (subset["player first initial"] == first_initial)
        ]
        if not fuzzy.empty:
            return fuzzy.sort_values(["date", "season"]).iloc[-1]

    if last:
        fuzzy = subset[subset["player last name"] == last]
        if len(fuzzy) == 1:
            return fuzzy.sort_values(["date", "season"]).iloc[-1]

    return None


def _predict_row(row: pd.Series, player_form: pd.DataFrame, models: dict[tuple[str, str], dict]) -> tuple[float | None, str | None]:
    candidates = _candidate_market_targets(row.get("market"))
    if not candidates:
        return None, None

    for target, scope in candidates:
        state = _match_player_state(row, player_form, scope)
        if state is None:
            continue
        payload = models.get((scope, target))
        if payload is None:
            continue
        features = payload["features"]
        model = payload["model"]
        X = pd.DataFrame([{feature: state.get(feature, 0.0) for feature in features}]).fillna(0.0)
        mean = float(model.predict(X)[0])
        return max(mean, 0.0), scope

    return None, None


def score_player_prop_board(config: AppConfig, board_file: Path) -> dict[str, Path]:
    if not config.latest_player_form.exists():
        raise ValueError("Latest player form file not found. Run train-history-models first.")

    board = pd.read_parquet(board_file) if board_file.suffix.lower() == ".parquet" else pd.read_csv(board_file)
    player_form = pd.read_parquet(config.latest_player_form)
    models = _load_models(config)

    rows = []
    for _, row in board.iterrows():
        board_scope = str(row.get("board scope", "")).strip().lower()
        raw_player = str(row.get("player", "")).strip()
        if board_scope and board_scope != "player props" and not raw_player:
            continue

        mean_projection, scope = _predict_row(row, player_form, models)
        if mean_projection is None:
            continue

        line_value = row.get("line")
        try:
            line_value = float(line_value) if pd.notna(line_value) else None
        except Exception:
            line_value = None
        if line_value is None:
            line_value = _infer_line_from_market(row.get("market", ""))

        fair_probability = _clamp_probability(_win_probability(mean_projection, row.get("selection"), line_value))
        book_probability = _clamp_probability(american_to_probability(row.get("american odds")))
        fair_american = probability_to_american(fair_probability) if fair_probability is not None else None
        fair_decimal = probability_to_decimal(fair_probability) if fair_probability is not None else None
        edge = (fair_probability - book_probability) if fair_probability is not None and book_probability is not None else None

        if fair_probability is None:
            continue

        rows.append(
            {
                "Sport": str(row.get("sport", "")).upper(),
                "Event id": row.get("event id"),
                "Market": str(row.get("market", "")).title(),
                "Pick": str(row.get("selection", "")).title(),
                "Line": line_value,
                "Player": row.get("player"),
                "Team": row.get("team"),
                "Home team": row.get("home team"),
                "Away team": row.get("away team"),
                "Book": row.get("book", "DraftKings"),
                "Sportsbook price": format_american(row.get("american odds")),
                "Sportsbook chance": format_percent(book_probability) if book_probability is not None else "",
                "Projection": f"{mean_projection:.2f}",
                "Fair chance": format_percent(fair_probability),
                "Fair decimal": format_decimal(fair_decimal) if fair_decimal is not None else "",
                "Fair price": format_american(fair_american) if fair_american is not None else "",
                "Edge": format_percent(edge) if edge is not None else "",
                "edge raw": edge if edge is not None else -999.0,
                "Resolved scope": scope.title(),
            }
        )

    fair_prices = pd.DataFrame(rows)
    if fair_prices.empty:
        fair_prices = pd.DataFrame(columns=[
            "Sport", "Event id", "Market", "Pick", "Line", "Player", "Team", "Home team", "Away team",
            "Book", "Sportsbook price", "Sportsbook chance", "Projection", "Fair chance",
            "Fair decimal", "Fair price", "Edge", "Resolved scope"
        ])
        save_table(fair_prices, config.player_prop_fair_prices)
        save_table(fair_prices, config.best_bets)
        return {
            "player prop fair prices": config.player_prop_fair_prices,
            "best bets": config.best_bets,
        }

    fair_prices["Line"] = fair_prices["Line"].map(lambda x: "" if pd.isna(x) else (f"{x:.1f}" if not float(x).is_integer() else f"{int(x)}"))
    save_table(fair_prices.drop(columns=["edge raw"]), config.player_prop_fair_prices)

    best_bets = fair_prices.sort_values("edge raw", ascending=False).drop(columns=["edge raw"]).head(200).copy()
    save_table(best_bets, config.best_bets)
    return {
        "player prop fair prices": config.player_prop_fair_prices,
        "best bets": config.best_bets,
    }


def _home_away_game_features(
    home_id: str,
    away_id: str,
    latest_team_form: pd.DataFrame,
    required_features: list[str] | None = None,
) -> dict[str, float] | None:
    home = latest_team_form[latest_team_form["team id"] == home_id]
    away = latest_team_form[latest_team_form["team id"] == away_id]
    if home.empty or away.empty:
        return None

    home_row = home.sort_values(["date", "season"]).iloc[-1]
    away_row = away.sort_values(["date", "season"]).iloc[-1]

    home_map = {str(k): v for k, v in home_row.items()}
    away_map = {str(k): v for k, v in away_row.items()}

    if required_features is None:
        required_features = []

    features: dict[str, float] = {}
    for feature in required_features:
        if feature == "season_x":
            features[feature] = float(home_map.get("season", 0) or 0)
            continue
        if feature == "season_y":
            features[feature] = float(away_map.get("season", 0) or 0)
            continue

        if feature.endswith(" home"):
            base = feature[:-5]
            value = home_map.get(base, 0.0)
            features[feature] = float(value) if pd.notna(value) else 0.0
            continue
        if feature.endswith(" away"):
            base = feature[:-5]
            value = away_map.get(base, 0.0)
            features[feature] = float(value) if pd.notna(value) else 0.0
            continue

        if feature in home_map:
            value = home_map.get(feature, 0.0)
            features[feature] = float(value) if pd.notna(value) else 0.0
            continue
        if feature in away_map:
            value = away_map.get(feature, 0.0)
            features[feature] = float(value) if pd.notna(value) else 0.0
            continue

        features[feature] = 0.0

    return features


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _weighted_recent_team_value(row: pd.Series, base: str) -> float:
    recent5 = _safe_float(row.get(f"{base} recent 5"), np.nan)
    recent15 = _safe_float(row.get(f"{base} recent 15"), np.nan)
    season = _safe_float(row.get(f"{base} season to date"), np.nan)

    values = []
    weights = []
    if not np.isnan(recent5):
        values.append(recent5); weights.append(0.5)
    if not np.isnan(recent15):
        values.append(recent15); weights.append(0.3)
    if not np.isnan(season):
        values.append(season); weights.append(0.2)
    if not values:
        return 0.0
    return float(np.average(values, weights=weights))


def _heuristic_game_means(home_id: str, away_id: str, latest_team_form: pd.DataFrame) -> tuple[float, float] | None:
    home = latest_team_form[latest_team_form["team id"] == home_id]
    away = latest_team_form[latest_team_form["team id"] == away_id]
    if home.empty or away.empty:
        return None

    home_row = home.sort_values(["date", "season"]).iloc[-1]
    away_row = away.sort_values(["date", "season"]).iloc[-1]

    home_off = _weighted_recent_team_value(home_row, "runs scored")
    away_off = _weighted_recent_team_value(away_row, "runs scored")
    home_def = _weighted_recent_team_value(home_row, "runs allowed")
    away_def = _weighted_recent_team_value(away_row, "runs allowed")

    # Blend offense with opponent defense, then clamp to a sane MLB range.
    home_mean = 0.58 * home_off + 0.42 * away_def + 0.15
    away_mean = 0.58 * away_off + 0.42 * home_def - 0.05

    home_mean = float(min(max(home_mean, 2.0), 8.5))
    away_mean = float(min(max(away_mean, 2.0), 8.5))
    return home_mean, away_mean




def _predict_game_means(
    home_id: str,
    away_id: str,
    latest_team_form: pd.DataFrame,
    models: dict[tuple[str, str], dict],
) -> tuple[float, float]:
    home_payload = models.get(("game", "home runs target"))
    away_payload = models.get(("game", "away runs target"))
    if home_payload is None or away_payload is None:
        raise ValueError("Game models are missing. Run train-history-models first.")

    heuristic = _heuristic_game_means(home_id, away_id, latest_team_form)

    required_features = sorted(set(home_payload["features"]) | set(away_payload["features"]))
    features = _home_away_game_features(home_id, away_id, latest_team_form, required_features=required_features)

    if features is None:
        if heuristic is not None:
            return heuristic
        raise ValueError("Missing latest team form for matchup.")

    X_home = pd.DataFrame([{feature: features.get(feature, 0.0) for feature in home_payload["features"]}]).fillna(0.0)
    X_away = pd.DataFrame([{feature: features.get(feature, 0.0) for feature in away_payload["features"]}]).fillna(0.0)

    home_mean = float(home_payload["model"].predict(X_home)[0])
    away_mean = float(away_payload["model"].predict(X_away)[0])

    if not np.isfinite(home_mean) or not np.isfinite(away_mean) or home_mean <= 0.1 or away_mean <= 0.1:
        if heuristic is not None:
            return heuristic

    return max(home_mean, 1.0), max(away_mean, 1.0)


def _independent_score_grid(home_mean: float, away_mean: float, max_runs: int = 20) -> tuple[np.ndarray, np.ndarray]:
    home_p = np.array([_poisson_pmf(i, home_mean) for i in range(max_runs + 1)])
    away_p = np.array([_poisson_pmf(i, away_mean) for i in range(max_runs + 1)])
    tail_home = max(0.0, 1.0 - float(home_p.sum()))
    tail_away = max(0.0, 1.0 - float(away_p.sum()))
    home_p[-1] += tail_home
    away_p[-1] += tail_away
    return home_p, away_p


def _moneyline_probabilities(home_mean: float, away_mean: float) -> tuple[float, float]:
    home_p, away_p = _independent_score_grid(home_mean, away_mean)
    p_home = 0.0
    p_away = 0.0
    p_tie = 0.0
    for i, hp in enumerate(home_p):
        for j, ap in enumerate(away_p):
            joint = float(hp * ap)
            if i > j:
                p_home += joint
            elif i < j:
                p_away += joint
            else:
                p_tie += joint
    not_tie = max(1e-9, 1.0 - p_tie)
    return p_home / not_tie, p_away / not_tie


def _total_probabilities(home_mean: float, away_mean: float, line: float, selection: str) -> float:
    total_mean = home_mean + away_mean
    return _win_probability(total_mean, selection, line) or 0.0


def _spread_probability(home_mean: float, away_mean: float, selection: str, line: float) -> float:
    home_p, away_p = _independent_score_grid(home_mean, away_mean)
    p = 0.0
    for h, hp in enumerate(home_p):
        for a, ap in enumerate(away_p):
            joint = float(hp * ap)
            if str(selection).lower() == "home":
                if h + line > a:
                    p += joint
            elif str(selection).lower() == "away":
                if a + line > h:
                    p += joint
    return p


def _team_total_probability(home_mean: float, away_mean: float, team_id: str, home_id: str, away_id: str, selection: str, line: float) -> float:
    if team_id == home_id:
        mean = home_mean
    elif team_id == away_id:
        mean = away_mean
    else:
        return 0.0
    return _win_probability(mean, selection, line) or 0.0


def score_game_board(config: AppConfig, board_file: Path) -> dict[str, Path]:
    if not config.latest_team_form.exists():
        raise ValueError("Latest team form file not found. Run train-history-models first.")

    board = pd.read_parquet(board_file) if board_file.suffix.lower() == ".parquet" else pd.read_csv(board_file)
    latest_team_form = pd.read_parquet(config.latest_team_form)
    models = _load_models(config)

    rows = []
    for _, row in board.iterrows():
        home_name = row.get("home team", "")
        away_name = row.get("away team", "")
        home_id = team_name_to_id(home_name)
        away_id = team_name_to_id(away_name)
        if not home_id or not away_id:
            continue

        try:
            home_mean, away_mean = _predict_game_means(home_id, away_id, latest_team_form, models)
        except Exception:
            continue

        market = _normalize_market(row.get("market", ""))
        selection = _normalize_market(row.get("selection", ""))
        if selection == _normalize_market(home_name):
            selection = "home"
        elif selection == _normalize_market(away_name):
            selection = "away"
        try:
            line = float(row.get("line")) if pd.notna(row.get("line")) else None
        except Exception:
            line = None

        fair_probability = None
        if market in {"moneyline", "to win"}:
            p_home, p_away = _moneyline_probabilities(home_mean, away_mean)
            fair_probability = p_home if selection == "home" else p_away if selection == "away" else None
        elif market in {"game total runs", "total", "total runs"} and line is not None:
            fair_probability = _total_probabilities(home_mean, away_mean, line, selection)
        elif market in {"spread", "run line"} and line is not None:
            fair_probability = _spread_probability(home_mean, away_mean, selection, line)
        elif market in {"team total", "team total runs"} and line is not None:
            team_id = team_name_to_id(row.get("team", ""))
            if team_id:
                fair_probability = _team_total_probability(home_mean, away_mean, team_id, home_id, away_id, selection, line)

        if fair_probability is None:
            continue

        fair_probability = _clamp_probability(fair_probability)
        book_probability = _clamp_probability(american_to_probability(row.get("american odds")))
        fair_american = probability_to_american(fair_probability) if fair_probability is not None else None
        fair_decimal = probability_to_decimal(fair_probability) if fair_probability is not None else None
        edge = (fair_probability - book_probability) if fair_probability is not None and book_probability is not None else None

        rows.append(
            {
                "Sport": str(row.get("sport", "")).upper(),
                "Matchup": f"{away_name} at {home_name}",
                "Bet type": str(row.get("market", "")).title(),
                "Pick": str(row.get("selection", "")).title(),
                "Line": "" if line is None else (f"{line:.1f}" if not float(line).is_integer() else f"{int(line)}"),
                "Book": row.get("book", "DraftKings"),
                "Sportsbook price": format_american(row.get("american odds")),
                "Sportsbook chance": format_percent(book_probability) if book_probability is not None else "",
                "Home team projection": f"{home_mean:.2f}",
                "Away team projection": f"{away_mean:.2f}",
                "Game total projection": f"{home_mean + away_mean:.2f}",
                "Fair chance": format_percent(fair_probability),
                "Fair decimal": format_decimal(fair_decimal) if fair_decimal is not None else "",
                "Fair price": format_american(fair_american) if fair_american is not None else "",
                "Edge": format_percent(edge) if edge is not None else "",
                "edge raw": edge if edge is not None else -999.0,
            }
        )

    fair_prices = pd.DataFrame(rows)
    if fair_prices.empty:
        fair_prices = pd.DataFrame(columns=[
            "Sport", "Matchup", "Bet type", "Pick", "Line", "Book", "Sportsbook price",
            "Sportsbook chance", "Home team projection", "Away team projection",
            "Game total projection", "Fair chance", "Fair decimal", "Fair price", "Edge"
        ])
        save_table(fair_prices, config.game_fair_prices)
        save_table(fair_prices, config.game_best_bets)
        return {
            "game fair prices": config.game_fair_prices,
            "game best bets": config.game_best_bets,
        }

    save_table(fair_prices.drop(columns=["edge raw"]), config.game_fair_prices)
    best_bets = fair_prices.sort_values("edge raw", ascending=False).drop(columns=["edge raw"]).head(200).copy()
    save_table(best_bets, config.game_best_bets)
    return {
        "game fair prices": config.game_fair_prices,
        "game best bets": config.game_best_bets,
    }
