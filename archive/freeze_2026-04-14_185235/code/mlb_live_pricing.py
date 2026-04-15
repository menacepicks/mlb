
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd

from mlb_live_schema import ensure_unified_columns, norm_text

COUNT_FAMILIES = {
    "hits",
    "total_bases",
    "runs",
    "rbis",
    "hits_runs_rbis",
    "home_runs",
    "pitcher_strikeouts",
    "pitching_outs",
    "earned_runs",
    "walks_allowed",
    "hits_allowed",
    "stolen_bases",
    "singles",
    "doubles",
    "triples",
    "team_total_runs",
    "team_hits",
    "game_total_runs",
    "inning_1_runs",
}

FAMILY_ALIASES = {
    "game_moneyline": "moneyline",
    "game_spread": "spread",
    "tb_ou": "total_bases",
    "hits_ou": "hits",
    "runs_ou": "runs",
    "rbis_ou": "rbis",
    "hrr_ou": "hits_runs_rbis",
    "hr_ou": "home_runs",
    "sb_ou": "stolen_bases",
    "k_ou": "pitcher_strikeouts",
    "so_ou": "pitcher_strikeouts",
    "outs_ou": "pitching_outs",
    "earned_runs_allowed": "earned_runs",
}

SEGMENT_ALIASES = {
    "full game": "full_game",
    "full_game": "full_game",
    "first five": "first_five_innings",
    "1st inning": "inning_1",
    "first inning": "inning_1",
}

SELECTION_ALIASES = {
    "over": "over",
    "under": "under",
    "home": "home",
    "away": "away",
    "yes": "yes",
    "no": "no",
}


@dataclass(slots=True)
class PricingConfig:
    iterations: int = 50000
    seed: int = 42


def _clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return " ".join(str(value).strip().split())


def _normalize_name(value: Any) -> str:
    return norm_text(value).replace(" ", "")


def _canonical_family(value: Any) -> str:
    text = _clean_text(value).lower()
    return FAMILY_ALIASES.get(text, text)


def _canonical_segment(value: Any) -> str:
    text = _clean_text(value).lower()
    return SEGMENT_ALIASES.get(text, text or "full_game")


def _canonical_selection(value: Any) -> str:
    text = _clean_text(value).lower()
    return SELECTION_ALIASES.get(text, text)


def apply_no_vig(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_unified_columns(df)
    if out.empty:
        return out
    for _, grp in out.groupby(["book", "market_group_key"], dropna=False):
        idx = grp.index
        implied = pd.to_numeric(grp["implied_prob"], errors="coerce")
        valid = implied.notna()
        if valid.sum() >= 2:
            denom = implied[valid].sum()
            if denom > 0:
                out.loc[idx[valid], "fair_prob"] = implied[valid] / denom
    return out


def load_projection_table(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".parquet") or path.endswith(".pq"):
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported projection file: {path}")

    aliases = {
        "event_id": "event_id",
        "event id": "event_id",
        "scope": "scope",
        "segment": "segment",
        "market_family": "market_family",
        "market family": "market_family",
        "participant": "participant",
        "team_name": "team_name",
        "team name": "team_name",
        "selection": "selection",
        "projection_mean": "projection_mean",
        "projection mean": "projection_mean",
        "mean": "projection_mean",
        "proj": "projection_mean",
        "projection_prob": "projection_prob",
        "projection prob": "projection_prob",
        "prob": "projection_prob",
        "player_name": "participant",
        "player name": "participant",
        "team": "team_name",
        "period": "segment",
    }
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})

    for col in ("scope", "segment", "market_family", "event_id", "participant", "team_name", "selection"):
        if col not in df.columns:
            df[col] = ""
    if "projection_mean" not in df.columns:
        df["projection_mean"] = pd.NA
    if "projection_prob" not in df.columns:
        df["projection_prob"] = pd.NA

    df["scope"] = df["scope"].map(lambda x: _clean_text(x).lower())
    df["segment"] = df["segment"].map(_canonical_segment)
    df["market_family"] = df["market_family"].map(_canonical_family)
    df["selection"] = df["selection"].map(_canonical_selection)
    return df


def attach_projections(df: pd.DataFrame, projections: pd.DataFrame) -> pd.DataFrame:
    out = ensure_unified_columns(df)
    if out.empty or projections.empty:
        return out

    proj = projections.copy()
    proj["_event_id"] = proj["event_id"].astype(str)
    proj["_scope"] = proj["scope"].astype(str).str.lower()
    proj["_segment"] = proj["segment"].map(_canonical_segment)
    proj["_family"] = proj["market_family"].map(_canonical_family)
    proj["_participant"] = proj["participant"].map(_normalize_name)
    proj["_team"] = proj["team_name"].map(_normalize_name)
    proj["_selection"] = proj["selection"].map(_canonical_selection)

    out["_event_id"] = out["event_id"].astype(str)
    out["_scope"] = out["scope"].astype(str).str.lower()
    out["_segment"] = out["segment"].map(_canonical_segment)
    out["_family"] = out["market_family"].map(_canonical_family)
    out["_participant"] = out["participant"].map(_normalize_name)
    out["_team"] = out["team_name"].map(_normalize_name)
    out["_selection"] = out["selection"].map(_canonical_selection)

    # Mean rows attach without selection.
    mean_proj = proj[proj["projection_mean"].notna()].copy()
    prob_proj = proj[proj["projection_prob"].notna()].copy()

    mean_player = mean_proj[mean_proj["_scope"] == "player"]
    mean_team = mean_proj[mean_proj["_scope"] == "team"]
    mean_game = mean_proj[mean_proj["_scope"] == "game"]

    prob_player = prob_proj[prob_proj["_scope"] == "player"]
    prob_team = prob_proj[prob_proj["_scope"] == "team"]
    prob_game = prob_proj[prob_proj["_scope"] == "game"]

    if not mean_player.empty:
        out = out.merge(
            mean_player[["_event_id", "_segment", "_family", "_participant", "projection_mean"]]
            .rename(columns={"projection_mean": "_player_projection_mean"})
            .drop_duplicates(),
            on=["_event_id", "_segment", "_family", "_participant"],
            how="left",
        )
    else:
        out["_player_projection_mean"] = pd.NA

    if not mean_team.empty:
        out = out.merge(
            mean_team[["_event_id", "_segment", "_family", "_team", "projection_mean"]]
            .rename(columns={"projection_mean": "_team_projection_mean"})
            .drop_duplicates(),
            on=["_event_id", "_segment", "_family", "_team"],
            how="left",
        )
    else:
        out["_team_projection_mean"] = pd.NA

    if not mean_game.empty:
        out = out.merge(
            mean_game[["_event_id", "_segment", "_family", "projection_mean"]]
            .rename(columns={"projection_mean": "_game_projection_mean"})
            .drop_duplicates(),
            on=["_event_id", "_segment", "_family"],
            how="left",
        )
    else:
        out["_game_projection_mean"] = pd.NA

    # Probability rows attach WITH selection.
    if not prob_player.empty:
        out = out.merge(
            prob_player[["_event_id", "_segment", "_family", "_participant", "_selection", "projection_prob", "projection_source"]]
            .rename(columns={"projection_prob": "_player_projection_prob", "projection_source": "_player_projection_source"})
            .drop_duplicates(),
            on=["_event_id", "_segment", "_family", "_participant", "_selection"],
            how="left",
        )
    else:
        out["_player_projection_prob"] = pd.NA
        out["_player_projection_source"] = pd.NA

    if not prob_team.empty:
        out = out.merge(
            prob_team[["_event_id", "_segment", "_family", "_team", "_selection", "projection_prob", "projection_source"]]
            .rename(columns={"projection_prob": "_team_projection_prob", "projection_source": "_team_projection_source"})
            .drop_duplicates(),
            on=["_event_id", "_segment", "_family", "_team", "_selection"],
            how="left",
        )
    else:
        out["_team_projection_prob"] = pd.NA
        out["_team_projection_source"] = pd.NA

    if not prob_game.empty:
        out = out.merge(
            prob_game[["_event_id", "_segment", "_family", "_selection", "projection_prob", "projection_source"]]
            .rename(columns={"projection_prob": "_game_projection_prob", "projection_source": "_game_projection_source"})
            .drop_duplicates(),
            on=["_event_id", "_segment", "_family", "_selection"],
            how="left",
        )
    else:
        out["_game_projection_prob"] = pd.NA
        out["_game_projection_source"] = pd.NA

    out["projection_mean"] = (
        out["_player_projection_mean"]
        .combine_first(out["_team_projection_mean"])
        .combine_first(out["_game_projection_mean"])
    )
    out["projection_prob"] = (
        out["_player_projection_prob"]
        .combine_first(out["_team_projection_prob"])
        .combine_first(out["_game_projection_prob"])
    )
    out["projection_source"] = (
        out["_player_projection_source"]
        .combine_first(out["_team_projection_source"])
        .combine_first(out["_game_projection_source"])
    )

    drop_cols = [c for c in out.columns if c.startswith("_")]
    return out.drop(columns=drop_cols)


def _simulate_count_prob(mean: float, row: pd.Series, rng: np.random.Generator, iterations: int) -> float:
    mean = max(float(mean), 0.0)
    values = rng.poisson(lam=mean, size=iterations)
    selection = _canonical_selection(row.get("selection"))
    line = row["milestone_value"] if bool(row["is_milestone"]) else row["line"]
    if line is None or (isinstance(line, float) and math.isnan(line)):
        return float("nan")
    threshold = float(line)
    if bool(row["is_milestone"]):
        if selection in {"under", "no"}:
            return float((values < threshold).mean())
        return float((values >= threshold).mean())
    if selection == "over":
        return float((values > threshold).mean())
    if selection == "under":
        return float((values < threshold).mean())
    return float("nan")


def score_market_board(df: pd.DataFrame, projections: pd.DataFrame | None = None, config: PricingConfig | None = None) -> pd.DataFrame:
    config = config or PricingConfig()
    rng = np.random.default_rng(config.seed)
    out = apply_no_vig(ensure_unified_columns(df))
    if projections is not None and not projections.empty:
        out = attach_projections(out, projections)
    if out.empty:
        return out

    out["score_status"] = out["score_status"].fillna("unscored")

    for idx, row in out.iterrows():
        if pd.notna(row.get("projection_prob")):
            out.at[idx, "model_prob"] = float(row["projection_prob"])
            out.at[idx, "score_status"] = "scored_probability"
            continue
        if pd.notna(row.get("projection_mean")) and _canonical_family(row.get("market_family")) in COUNT_FAMILIES:
            out.at[idx, "model_prob"] = _simulate_count_prob(float(row["projection_mean"]), row, rng, config.iterations)
            out.at[idx, "score_status"] = "scored_monte_carlo"
            continue
        if pd.notna(row.get("fair_prob")):
            out.at[idx, "score_status"] = "market_only"
        else:
            out.at[idx, "score_status"] = "unscored_no_projection"

    fair = pd.to_numeric(out["fair_prob"], errors="coerce")
    model = pd.to_numeric(out["model_prob"], errors="coerce")
    out["edge_pct_points"] = (model - fair) * 100.0
    return ensure_unified_columns(out)
