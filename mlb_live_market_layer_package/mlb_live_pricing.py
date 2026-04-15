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


@dataclass(slots=True)
class PricingConfig:
    iterations: int = 50000
    seed: int = 42


def apply_no_vig(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_unified_columns(df)
    if out.empty:
        return out
    for key, grp in out.groupby(["book", "market_group_key"], dropna=False):
        idx = grp.index
        implied = grp["implied_prob"].astype(float)
        valid = implied.notna()
        if valid.sum() >= 2:
            denom = implied[valid].sum()
            if denom > 0:
                out.loc[idx[valid], "fair_prob"] = implied[valid] / denom
    return out


def _normalize_name(value: Any) -> str:
    return norm_text(value).replace(" ", "")


def load_projection_table(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".parquet") or path.endswith(".pq"):
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported projection file: {path}")
    for col in ("scope", "segment", "market_family", "event_id", "participant", "team_name"):
        if col not in df.columns:
            df[col] = ""
    if "projection_mean" not in df.columns:
        df["projection_mean"] = pd.NA
    if "projection_prob" not in df.columns:
        df["projection_prob"] = pd.NA
    return df


def attach_projections(df: pd.DataFrame, projections: pd.DataFrame) -> pd.DataFrame:
    out = ensure_unified_columns(df)
    if out.empty or projections.empty:
        return out

    proj = projections.copy()
    if "projection_mean" not in proj.columns:
        proj["projection_mean"] = pd.NA
    if "projection_prob" not in proj.columns:
        proj["projection_prob"] = pd.NA
    proj["_event_id"] = proj["event_id"].astype(str)
    proj["_scope"] = proj["scope"].astype(str).str.lower()
    proj["_segment"] = proj["segment"].astype(str).str.lower().replace("", "full_game")
    proj["_family"] = proj["market_family"].astype(str).str.lower()
    proj["_participant"] = proj["participant"].map(_normalize_name)
    proj["_team"] = proj["team_name"].map(_normalize_name)

    out["_event_id"] = out["event_id"].astype(str)
    out["_scope"] = out["scope"].astype(str).str.lower()
    out["_segment"] = out["segment"].astype(str).str.lower()
    out["_family"] = out["market_family"].astype(str).str.lower()
    out["_participant"] = out["participant"].map(_normalize_name)
    out["_team"] = out["team_name"].map(_normalize_name)

    player_proj = proj[proj["_scope"] == "player"]
    team_proj = proj[proj["_scope"] == "team"]
    game_proj = proj[proj["_scope"] == "game"]

    if not player_proj.empty:
        out = out.merge(
            player_proj[["_event_id", "_segment", "_family", "_participant", "projection_mean", "projection_prob"]].rename(
                columns={"projection_mean": "_player_projection_mean", "projection_prob": "_player_projection_prob"}
            ),
            on=["_event_id", "_segment", "_family", "_participant"],
            how="left",
        )
    else:
        out["_player_projection_mean"] = pd.NA
        out["_player_projection_prob"] = pd.NA

    if not team_proj.empty:
        out = out.merge(
            team_proj[["_event_id", "_segment", "_family", "_team", "projection_mean", "projection_prob"]].rename(
                columns={"projection_mean": "_team_projection_mean", "projection_prob": "_team_projection_prob"}
            ),
            on=["_event_id", "_segment", "_family", "_team"],
            how="left",
        )
    else:
        out["_team_projection_mean"] = pd.NA
        out["_team_projection_prob"] = pd.NA

    if not game_proj.empty:
        out = out.merge(
            game_proj[["_event_id", "_segment", "_family", "projection_mean", "projection_prob"]].rename(
                columns={"projection_mean": "_game_projection_mean", "projection_prob": "_game_projection_prob"}
            ),
            on=["_event_id", "_segment", "_family"],
            how="left",
        )
    else:
        out["_game_projection_mean"] = pd.NA
        out["_game_projection_prob"] = pd.NA

    out["projection_mean"] = out["_player_projection_mean"].combine_first(out["_team_projection_mean"]).combine_first(out["_game_projection_mean"])
    out["projection_prob"] = out["_player_projection_prob"].combine_first(out["_team_projection_prob"]).combine_first(out["_game_projection_prob"])

    drop_cols = [c for c in out.columns if c.startswith("_")]
    return out.drop(columns=drop_cols)


def _simulate_count_prob(mean: float, row: pd.Series, rng: np.random.Generator, iterations: int) -> float:
    mean = max(float(mean), 0.0)
    values = rng.poisson(lam=mean, size=iterations)
    selection = str(row["selection"] or "").strip().lower()
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
        if pd.notna(row.get("projection_mean")) and str(row.get("market_family") or "") in COUNT_FAMILIES:
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
