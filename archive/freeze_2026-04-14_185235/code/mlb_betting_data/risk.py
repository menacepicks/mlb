from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np
import pandas as pd


def variance_label(std_value: float) -> str:
    if pd.isna(std_value):
        return ""
    if std_value < 0.75:
        return "Low"
    if std_value < 1.75:
        return "Medium"
    return "High"


def relationship_label(correlation: float) -> str:
    if pd.isna(correlation):
        return ""
    strength = abs(correlation)
    if strength < 0.10:
        return "Weak"
    if strength < 0.25:
        return "Light"
    if strength < 0.45:
        return "Medium"
    return "Strong"


def direction_label(correlation: float) -> str:
    if pd.isna(correlation):
        return ""
    if correlation > 0.05:
        return "Move together"
    if correlation < -0.05:
        return "Move opposite ways"
    return "Mostly unrelated"


def build_volatility_table(
    hitter_df: pd.DataFrame,
    pitcher_df: pd.DataFrame,
    team_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    hitter_targets = [
        "hits",
        "total bases",
        "home runs",
        "runs scored",
        "runs batted in",
        "walks",
        "strikeouts",
        "stolen bases",
    ]
    for column in hitter_targets:
        if column not in hitter_df.columns:
            continue
        grouped = hitter_df.groupby(["player id"], dropna=False)[column].agg(["mean", "var", "std", "count"]).reset_index()
        grouped["scope"] = "Player prop"
        grouped["market"] = column.title()
        grouped["variance"] = grouped["var"]
        grouped["standard deviation"] = grouped["std"]
        grouped["risk level"] = grouped["standard deviation"].map(variance_label)
        rows.extend(
            grouped.rename(columns={
                "player id": "id",
                "mean": "average result",
                "count": "games in sample",
            })[
                ["scope", "market", "id", "average result", "variance", "standard deviation", "games in sample", "risk level"]
            ].to_dict("records")
        )

    pitcher_targets = [
        "strikeouts",
        "outs recorded",
        "earned runs",
        "hits allowed",
        "walks allowed",
        "home runs allowed",
    ]
    for column in pitcher_targets:
        if column not in pitcher_df.columns:
            continue
        grouped = pitcher_df.groupby(["player id"], dropna=False)[column].agg(["mean", "var", "std", "count"]).reset_index()
        grouped["scope"] = "Pitcher prop"
        grouped["market"] = column.title()
        grouped["variance"] = grouped["var"]
        grouped["standard deviation"] = grouped["std"]
        grouped["risk level"] = grouped["standard deviation"].map(variance_label)
        rows.extend(
            grouped.rename(columns={
                "player id": "id",
                "mean": "average result",
                "count": "games in sample",
            })[
                ["scope", "market", "id", "average result", "variance", "standard deviation", "games in sample", "risk level"]
            ].to_dict("records")
        )

    team_targets = ["runs scored", "runs allowed"]
    for column in team_targets:
        if column not in team_df.columns:
            continue
        grouped = team_df.groupby(["team id"], dropna=False)[column].agg(["mean", "var", "std", "count"]).reset_index()
        grouped["scope"] = "Team result"
        grouped["market"] = column.title()
        grouped["variance"] = grouped["var"]
        grouped["standard deviation"] = grouped["std"]
        grouped["risk level"] = grouped["standard deviation"].map(variance_label)
        rows.extend(
            grouped.rename(columns={
                "team id": "id",
                "mean": "average result",
                "count": "games in sample",
            })[
                ["scope", "market", "id", "average result", "variance", "standard deviation", "games in sample", "risk level"]
            ].to_dict("records")
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        for col in ["average result", "variance", "standard deviation"]:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(3)
    return out


def _safe_corr(x: pd.Series, y: pd.Series) -> float | None:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 30:
        return None
    x = x[mask]
    y = y[mask]
    if x.nunique() <= 1 or y.nunique() <= 1:
        return None
    return float(x.corr(y))


def build_same_game_relationship_guide(
    hitter_df: pd.DataFrame,
    pitcher_df: pd.DataFrame,
    team_df: pd.DataFrame,
    game_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    # Team-level same-game relationships.
    if not team_df.empty and {"game id", "team id", "runs scored", "runs allowed", "is home"}.issubset(team_df.columns):
        home = team_df[team_df["is home"] == True][["game id", "team id", "runs scored", "runs allowed"]].rename(
            columns={
                "team id": "home team id",
                "runs scored": "home team runs",
                "runs allowed": "home team runs allowed",
            }
        )
        away = team_df[team_df["is home"] == False][["game id", "team id", "runs scored", "runs allowed"]].rename(
            columns={
                "team id": "away team id",
                "runs scored": "away team runs",
                "runs allowed": "away team runs allowed",
            }
        )
        merged = home.merge(away, on="game id", how="inner")
        merged["game total runs"] = merged["home team runs"] + merged["away team runs"]
        candidates = [
            ("Home team runs", "Away team runs", "home team runs", "away team runs"),
            ("Home team runs", "Game total runs", "home team runs", "game total runs"),
            ("Away team runs", "Game total runs", "away team runs", "game total runs"),
        ]
        for left_name, right_name, left_col, right_col in candidates:
            corr = _safe_corr(merged[left_col], merged[right_col])
            if corr is None:
                continue
            rows.append(
                {
                    "Area": "Game and team lines",
                    "Bet A": left_name,
                    "Bet B": right_name,
                    "Correlation": round(corr, 3),
                    "How they connect": direction_label(corr),
                    "Strength": relationship_label(corr),
                    "Sample size": int(len(merged)),
                    "Plain-English note": (
                        f"{left_name} and {right_name} {direction_label(corr).lower()}."
                    ),
                }
            )

    # Hitter vs team total relationships.
    if not hitter_df.empty and not team_df.empty and {"game id", "team id", "runs scored"}.issubset(team_df.columns):
        hitter_features = [
            "hits",
            "total bases",
            "home runs",
            "runs scored",
            "runs batted in",
            "walks",
            "strikeouts",
            "stolen bases",
        ]
        merged = hitter_df.merge(
            team_df[["game id", "team id", "runs scored"]].rename(columns={"runs scored": "team runs scored"}),
            on=["game id", "team id"],
            how="left",
        )
        for feature in hitter_features:
            if feature not in merged.columns:
                continue
            corr = _safe_corr(merged[feature], merged["team runs scored"])
            if corr is None:
                continue
            rows.append(
                {
                    "Area": "Hitter and team total",
                    "Bet A": feature.title(),
                    "Bet B": "Team runs scored",
                    "Correlation": round(corr, 3),
                    "How they connect": direction_label(corr),
                    "Strength": relationship_label(corr),
                    "Sample size": int(len(merged)),
                    "Plain-English note": (
                        f"{feature.title()} and team runs scored {direction_label(corr).lower()}."
                    ),
                }
            )

    # Pitcher vs opponent team total relationships.
    if not pitcher_df.empty and not team_df.empty and {"game id", "team id", "runs scored", "opponent team id"}.issubset(pitcher_df.columns):
        team_runs = team_df[["game id", "team id", "runs scored"]].rename(columns={"team id": "opponent team id", "runs scored": "opponent runs"})
        merged = pitcher_df.merge(team_runs, on=["game id", "opponent team id"], how="left")
        candidates = [
            ("Pitcher strikeouts", "Opponent runs", "strikeouts", "opponent runs"),
            ("Pitcher earned runs", "Opponent runs", "earned runs", "opponent runs"),
            ("Pitcher hits allowed", "Opponent runs", "hits allowed", "opponent runs"),
            ("Pitcher walks allowed", "Opponent runs", "walks allowed", "opponent runs"),
        ]
        for left_name, right_name, left_col, right_col in candidates:
            if left_col not in merged.columns or right_col not in merged.columns:
                continue
            corr = _safe_corr(merged[left_col], merged[right_col])
            if corr is None:
                continue
            rows.append(
                {
                    "Area": "Pitcher and opponent scoring",
                    "Bet A": left_name,
                    "Bet B": right_name,
                    "Correlation": round(corr, 3),
                    "How they connect": direction_label(corr),
                    "Strength": relationship_label(corr),
                    "Sample size": int(len(merged)),
                    "Plain-English note": (
                        f"{left_name} and {right_name} {direction_label(corr).lower()}."
                    ),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Area", "Strength", "Correlation"], ascending=[True, True, False]).reset_index(drop=True)
    return out
