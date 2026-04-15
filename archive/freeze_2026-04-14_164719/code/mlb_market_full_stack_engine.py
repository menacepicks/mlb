from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    from xgboost import XGBRegressor  # type: ignore
except Exception:  # pragma: no cover
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor  # type: ignore
except Exception:  # pragma: no cover
    LGBMRegressor = None

from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error


# ----------------------------- utilities -----------------------------

def norm_text(value: Any) -> str:
    s = str(value or "").strip().lower()
    s = s.replace("−", "-")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def clean_american(value: Any) -> float | None:
    if value is None:
        return None
    s = str(value).strip().replace("−", "-")
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def american_to_prob(odds: Any) -> float | None:
    a = clean_american(odds)
    if a is None:
        return None
    if a > 0:
        return 100.0 / (a + 100.0)
    if a < 0:
        return (-a) / ((-a) + 100.0)
    return None


def prob_to_american(prob: float) -> int:
    p = min(max(float(prob), 1e-9), 1 - 1e-9)
    if p >= 0.5:
        return int(round(-(100.0 * p) / (1.0 - p)))
    return int(round((100.0 * (1.0 - p)) / p))


def fair_probs_from_prices(odds: pd.Series) -> pd.Series:
    probs = odds.map(american_to_prob).astype(float)
    total = probs.sum(skipna=True)
    if total and total > 0:
        return probs / total
    count = probs.notna().sum()
    if count == 0:
        return probs
    return pd.Series([1.0 / count if pd.notna(x) else np.nan for x in probs], index=probs.index)


def poisson_cdf(k: int, lam: float) -> float:
    if k < 0:
        return 0.0
    lam = max(float(lam), 1e-9)
    term = math.exp(-lam)
    out = term
    for i in range(1, k + 1):
        term *= lam / i
        out += term
    return min(max(out, 0.0), 1.0)


def poisson_tail_ge(k: int, lam: float) -> float:
    if k <= 0:
        return 1.0
    return 1.0 - poisson_cdf(k - 1, lam)


def infer_poisson_mean_for_over(line: float, p_over: float) -> float:
    k = int(math.floor(float(line)))
    lo, hi = 1e-6, max(4.0, float(line) * 4.0 + 20.0)
    target = min(max(float(p_over), 1e-5), 1 - 1e-5)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        p = 1.0 - poisson_cdf(k, mid)
        if p < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def infer_poisson_mean_for_thresholds(threshold_probs: list[tuple[int, float]]) -> float:
    grid = np.linspace(0.01, 25.0, 800)
    best_lam, best_err = 1.0, float("inf")
    for lam in grid:
        err = 0.0
        for threshold, p in threshold_probs:
            err += (poisson_tail_ge(threshold, lam) - p) ** 2
        if err < best_err:
            best_err = err
            best_lam = float(lam)
    return best_lam


def inv_norm_cdf(p: float) -> float:
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q) + 1)
    if phigh < p:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q) + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


# ----------------------------- market mapping -----------------------------

MARKET_FAMILY = {
    "moneyline": ("game_moneyline", "game"),
    "run line": ("game_spread", "game"),
    "spread": ("game_spread", "game"),
    "total": ("game_total", "game"),
    "game total": ("game_total", "game"),
    "team total runs": ("team_total", "team"),
    "runs 1st inning": ("first_inning_runs", "game"),
    "runs 1st inning": ("first_inning_runs", "game"),
    "runs 1st inning": ("first_inning_runs", "game"),
    "runs 1st inning": ("first_inning_runs", "game"),
    "runs 1st inning": ("first_inning_runs", "game"),
    "runs 1st inning": ("first_inning_runs", "game"),
    "runs 1st inning": ("first_inning_runs", "game"),
    "runs 1st inning": ("first_inning_runs", "game"),
    "runs 1st inning": ("first_inning_runs", "game"),
    "runs 1st inning": ("first_inning_runs", "game"),
    "runs 1st inning": ("first_inning_runs", "game"),
    "runs 1st inning": ("first_inning_runs", "game"),
    "runs 1st inning": ("first_inning_runs", "game"),
    "runs 1st inning": ("first_inning_runs", "game"),
    "runs 1st inning": ("first_inning_runs", "game"),
    "runs 1st inning": ("first_inning_runs", "game"),
    "runs 1st inning": ("first_inning_runs", "game"),
    "runs 1st inning": ("first_inning_runs", "game"),
    "runs 1st inning": ("first_inning_runs", "game"),
    "runs 1st inning": ("first_inning_runs", "game"),
    "runs 1st inning": ("first_inning_runs", "game"),
    "runs 1st inning": ("first_inning_runs", "game"),
    "runs 1st inning": ("first_inning_runs", "game"),
    "runs 1st inning": ("first_inning_runs", "game"),
    "runs 1st inning": ("first_inning_runs", "game"),
    "runs - 1st inning": ("first_inning_runs", "game"),
    "total bases o u": ("tb_ou", "player"),
    "hits o u": ("hits_ou", "player"),
    "singles o u": ("singles_ou", "player"),
    "doubles o u": ("doubles_ou", "player"),
    "rbis o u": ("rbis_ou", "player"),
    "runs o u": ("runs_ou", "player"),
    "hits runs rbis o u": ("hrrbi_ou", "player"),
    "strikeouts thrown o u": ("pitcher_k_ou", "player"),
    "outs o u": ("pitcher_outs_ou", "player"),
    "outs recorded o u": ("pitcher_outs_ou", "player"),
    "hits allowed o u": ("pitcher_hits_allowed_ou", "player"),
    "earned runs allowed o u": ("pitcher_er_ou", "player"),
    "home runs milestones": ("hr_milestone", "player"),
    "hits milestones": ("hits_milestone", "player"),
    "strikeouts thrown milestones": ("pitcher_k_milestone", "player"),
    "strikeouts batter milestones": ("batter_k_milestone", "player"),
    "walks batter milestones": ("walks_milestone", "player"),
    "triples milestones": ("triples_milestone", "player"),
}

SUBCATEGORY_FALLBACK = {
    4519: ("game_bundle", "game"),
    6607: ("tb_ou", "player"),
    6719: ("hits_ou", "player"),
    8025: ("rbis_ou", "player"),
    9886: ("pitcher_hits_allowed_ou", "player"),
    11024: ("first_inning_runs", "game"),
    15221: ("pitcher_k_ou", "player"),
    16208: ("alt_team_total", "team"),
    16209: ("team_total", "team"),
    17319: ("hr_milestone", "player"),
    17320: ("hits_milestone", "player"),
    17323: ("pitcher_k_milestone", "player"),
    17406: ("hrrbi_ou", "player"),
    17407: ("runs_ou", "player"),
    17409: ("singles_ou", "player"),
    17410: ("doubles_ou", "player"),
    17412: ("pitcher_er_ou", "player"),
    17413: ("pitcher_outs_ou", "player"),
    17847: ("triples_milestone", "player"),
    17848: ("walks_milestone", "player"),
    17849: ("batter_k_milestone", "player"),
}

JOBLIB_NAME_BY_FAMILY = {
    "hits_ou": "hitter hits",
    "hits_milestone": "hitter hits",
    "tb_ou": "hitter total bases",
    "rbis_ou": "hitter runs batted in",
    "runs_ou": "hitter runs scored",
    "hr_milestone": "hitter home runs",
    "walks_milestone": "hitter walks",
    "batter_k_milestone": "hitter strikeouts",
    "pitcher_k_ou": "pitcher pitcher strikeouts",
    "pitcher_k_milestone": "pitcher pitcher strikeouts",
    "pitcher_outs_ou": "pitcher outs recorded",
    "pitcher_hits_allowed_ou": "pitcher hits allowed",
    "pitcher_er_ou": "pitcher earned runs allowed",
    "game_home_runs": "game home team runs",
    "game_away_runs": "game away team runs",
    "game_total": "game game total runs",
}

FAMILY_PRIORS = {
    "hits_ou": (0.85, 18.0),
    "singles_ou": (0.65, 14.0),
    "doubles_ou": (0.22, 8.0),
    "tb_ou": (1.45, 12.0),
    "rbis_ou": (0.42, 7.0),
    "runs_ou": (0.48, 8.0),
    "hrrbi_ou": (1.75, 8.0),
    "pitcher_k_ou": (5.7, 14.0),
    "pitcher_outs_ou": (16.2, 25.0),
    "pitcher_hits_allowed_ou": (5.3, 14.0),
    "pitcher_er_ou": (2.7, 10.0),
    "team_total": (4.2, 12.0),
    "alt_team_total": (4.2, 12.0),
    "game_total": (8.4, 18.0),
    "first_inning_runs": (0.62, 5.0),
    "hr_milestone": (0.18, 6.0),
    "hits_milestone": (0.85, 18.0),
    "pitcher_k_milestone": (5.7, 14.0),
    "batter_k_milestone": (0.95, 8.0),
    "walks_milestone": (0.38, 8.0),
    "triples_milestone": (0.03, 3.0),
}

OVERDISPERSION = {
    "hits_ou": 18.0,
    "singles_ou": 16.0,
    "doubles_ou": 8.0,
    "tb_ou": 10.0,
    "rbis_ou": 6.0,
    "runs_ou": 7.0,
    "hrrbi_ou": 7.0,
    "pitcher_k_ou": 14.0,
    "pitcher_outs_ou": 30.0,
    "pitcher_hits_allowed_ou": 14.0,
    "pitcher_er_ou": 8.0,
    "team_total": 12.0,
    "alt_team_total": 12.0,
    "game_total": 18.0,
    "first_inning_runs": 5.0,
    "hr_milestone": 4.0,
    "hits_milestone": 18.0,
    "pitcher_k_milestone": 14.0,
    "batter_k_milestone": 8.0,
    "walks_milestone": 8.0,
    "triples_milestone": 2.5,
}

PITCH_CLOCK_ERA_START = 2023
PYTH_EXPONENT = 1.83


# ----------------------------- loaders -----------------------------

def load_df(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    try:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception:
        return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {c: norm_text(c) for c in df.columns}
    out = df.rename(columns=rename_map).copy()
    # normalize some common names
    aliases = {
        "marketid": "marketid",
        "market id": "marketid",
        "event id": "event id",
        "market type": "market type",
        "american odds": "american odds",
        "participant name": "participant name",
        "participant type": "participant type",
        "participant venue role": "participant venue role",
        "subcategory id": "subcategory id",
        "milestone value": "milestone value",
        "home team": "home team",
        "away team": "away team",
        "selection": "selection",
        "line": "line",
        "market": "market",
    }
    out = out.rename(columns={k: v for k, v in aliases.items() if k in out.columns})
    return out


def load_board(path: Path) -> pd.DataFrame:
    df = normalize_columns(load_df(path) if path.exists() else pd.DataFrame())
    if df.empty:
        raise ValueError(f"Could not load board: {path}")
    required = ["market", "market type", "selection", "line", "american odds", "event id", "subcategory id", "marketid"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required board columns: {missing}")
    for c in ["line", "milestone value", "subcategory id"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "participant name" not in df.columns:
        df["participant name"] = ""
    if "participant venue role" not in df.columns:
        df["participant venue role"] = ""
    if "participant type" not in df.columns:
        df["participant type"] = ""
    df["market type norm"] = df["market type"].map(norm_text)
    df["player key"] = df["participant name"].map(norm_text)
    return df


def load_joblib_models(model_dir: Path | None) -> dict[str, dict[str, Any]]:
    if model_dir is None or not model_dir.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for file in model_dir.glob("*.joblib"):
        try:
            obj = joblib.load(file)
        except Exception:
            continue
        stem = file.stem.lower()
        if isinstance(obj, dict) and "model" in obj:
            out[stem] = obj
    return out


def choose_name_column(df: pd.DataFrame | None, candidates: list[str]) -> str | None:
    if df is None:
        return None
    cols = {norm_text(c): c for c in df.columns}
    for cand in candidates:
        if cand in cols:
            return cols[cand]
    return None


def build_player_lookup(df: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if df is None or df.empty:
        return {}
    ndf = normalize_columns(df)
    name_col = choose_name_column(ndf, ["player", "player name", "name"])
    if name_col is None:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for _, row in ndf.iterrows():
        key = norm_text(row.get(name_col))
        if key:
            out[key] = {norm_text(k): v for k, v in row.to_dict().items()}
    return out


def build_team_lookup(df: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if df is None or df.empty:
        return {}
    ndf = normalize_columns(df)
    name_col = choose_name_column(ndf, ["team", "team name", "name"])
    if name_col is None:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for _, row in ndf.iterrows():
        key = norm_text(row.get(name_col))
        if key:
            out[key] = {norm_text(k): v for k, v in row.to_dict().items()}
    return out


# ----------------------------- event context -----------------------------

def resolve_market_family(row: pd.Series) -> tuple[str, str]:
    mt = str(row.get("market type norm", "")).strip()
    if mt in MARKET_FAMILY:
        return MARKET_FAMILY[mt]
    subcat = row.get("subcategory id")
    if pd.notna(subcat):
        sc = int(subcat)
        if sc in SUBCATEGORY_FALLBACK:
            return SUBCATEGORY_FALLBACK[sc]
    market = norm_text(row.get("market"))
    if "moneyline" in market:
        return ("game_moneyline", "game")
    if "team total" in market:
        return ("team_total", "team")
    if "1st inning" in market:
        return ("first_inning_runs", "game")
    return ("unknown", "other")


def build_event_contexts(board: pd.DataFrame) -> dict[str, dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}
    for event_id, g in board.groupby("event id"):
        ctx: dict[str, Any] = {"event_id": event_id}
        sub = g[g["subcategory id"] == 4519].copy()
        if not sub.empty:
            # team names from ml/spread if available
            teams = sub[sub["participant type"].astype(str).str.lower() == "team"]
            homes = teams[teams["participant venue role"].astype(str).str.lower().isin(["home", "homeplayer"])]
            aways = teams[teams["participant venue role"].astype(str).str.lower().isin(["away", "awayplayer"])]
            if not homes.empty:
                ctx["home team"] = str(homes.iloc[0].get("participant name", ""))
            if not aways.empty:
                ctx["away team"] = str(aways.iloc[0].get("participant name", ""))
            # ML fair probs
            ml = sub[sub["market type norm"].eq("moneyline")].copy()
            if not ml.empty:
                ml["fair_prob"] = fair_probs_from_prices(ml["american odds"])
                for _, r in ml.iterrows():
                    if str(r.get("outcome type", "")).lower() == "home":
                        ctx["ml_home_prob"] = float(r["fair_prob"])
                    elif str(r.get("outcome type", "")).lower() == "away":
                        ctx["ml_away_prob"] = float(r["fair_prob"])
            # totals main line
            totals = sub[sub["selection"].isin(["Over", "Under"])].copy()
            if not totals.empty:
                main_total = totals.groupby("line").filter(lambda x: set(x["selection"]) >= {"Over", "Under"})
                if not main_total.empty:
                    main_total = main_total.sort_values("line").groupby("line", as_index=False).first()
                    ctx["game_total_line"] = float(main_total.iloc[-1]["line"])
            # spread main line
            spread = sub[(sub["participant type"].astype(str).str.lower() == "team") & (sub["line"].notna())].copy()
            if not spread.empty:
                home_spreads = spread[spread["participant venue role"].astype(str).str.lower() == "home"]
                if not home_spreads.empty:
                    main = home_spreads.iloc[0]
                    ctx["home_spread_line"] = float(main["line"])
        # main team totals from 16209
        tt = g[g["subcategory id"] == 16209].copy()
        if not tt.empty:
            for _, m in tt.groupby(["marketid", "line"]):
                if set(m["selection"]) >= {"Over", "Under"}:
                    team = str(m.iloc[0].get("participant name", ""))
                    fair = fair_probs_from_prices(m["american odds"]) 
                    p_over = float(fair[m["selection"].eq("Over")].iloc[0])
                    line = float(m.iloc[0]["line"])
                    mu = infer_poisson_mean_for_over(line, p_over)
                    if norm_text(team) == norm_text(ctx.get("home team")):
                        ctx["home_team_total_mu"] = mu
                    elif norm_text(team) == norm_text(ctx.get("away team")):
                        ctx["away_team_total_mu"] = mu
        # no double counting: use team totals first, otherwise solve from total + pyth/ML lightly
        home_mu = ctx.get("home_team_total_mu")
        away_mu = ctx.get("away_team_total_mu")
        if home_mu is not None and away_mu is not None:
            ctx["home_runs_mu"] = float(home_mu)
            ctx["away_runs_mu"] = float(away_mu)
        else:
            total_line = float(ctx.get("game_total_line", 8.5))
            home_ml = float(ctx.get("ml_home_prob", 0.5))
            z = inv_norm_cdf(min(max(home_ml, 1e-4), 1 - 1e-4))
            margin = z * 2.15
            ctx["home_runs_mu"] = max(0.5, 0.5 * (total_line + margin))
            ctx["away_runs_mu"] = max(0.5, total_line - ctx["home_runs_mu"])
        hrm = max(float(ctx["home_runs_mu"]), 0.1)
        arm = max(float(ctx["away_runs_mu"]), 0.1)
        ctx["pyth_home_prob"] = hrm ** PYTH_EXPONENT / (hrm ** PYTH_EXPONENT + arm ** PYTH_EXPONENT)
        contexts[str(event_id)] = ctx
    return contexts


# ----------------------------- model feature plumbing -----------------------------

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def find_exposure(player_row: dict[str, Any] | None, team_row: dict[str, Any] | None) -> float:
    for source in [player_row or {}, team_row or {}]:
        for key in source:
            nk = norm_text(key)
            if nk in {"plate appearances", "plate appearances season to date", "pa", "games before today", "batters faced", "innings pitched", "innings pitched season to date"}:
                val = safe_float(source[key], 0.0)
                if val > 0:
                    return val
    return 10.0


def build_feature_vector(feature_names: list[str], player_row: dict[str, Any] | None, team_row: dict[str, Any] | None, opp_team_row: dict[str, Any] | None, event_ctx: dict[str, Any]) -> dict[str, float]:
    player_row = player_row or {}
    team_row = team_row or {}
    opp_team_row = opp_team_row or {}
    merged = {}
    for src in [event_ctx, team_row, opp_team_row, player_row]:
        for k, v in src.items():
            merged[norm_text(k)] = v
    x: dict[str, float] = {}
    for feat in feature_names:
        n = norm_text(feat)
        if n == "pitch clock era flag":
            x[feat] = 1.0
            continue
        if n.startswith("season"):
            val = merged.get(n, merged.get("season", 2026))
            x[feat] = safe_float(val, 2026.0)
            continue
        if n in merged:
            x[feat] = safe_float(merged[n], 0.0)
            continue
        # flexible suffix matching
        hit = None
        for mk, mv in merged.items():
            if mk == n or mk.endswith(n) or n.endswith(mk):
                hit = mv
                break
        x[feat] = safe_float(hit, 0.0)
    return x


class ModelStack:
    def __init__(self, base_models: dict[str, dict[str, Any]], calibrator_dir: Path | None):
        self.base_models = base_models
        self.calibrators: dict[str, Any] = {}
        if calibrator_dir and calibrator_dir.exists():
            for file in calibrator_dir.glob("*.joblib"):
                try:
                    self.calibrators[file.stem.lower()] = joblib.load(file)
                except Exception:
                    pass

    def base_mean(self, family: str, row: pd.Series, player_row: dict[str, Any] | None, team_row: dict[str, Any] | None, opp_row: dict[str, Any] | None, event_ctx: dict[str, Any]) -> tuple[float | None, str]:
        # derived families from existing direct models
        if family == "hrrbi_ou":
            pieces = []
            for fam in ["hits_ou", "runs_ou", "rbis_ou"]:
                pred, _ = self.base_mean(fam, row, player_row, team_row, opp_row, event_ctx)
                if pred is not None:
                    pieces.append(pred)
            if pieces:
                return float(sum(pieces) * 0.92), "derived_components"
        if family == "singles_ou":
            hits, _ = self.base_mean("hits_ou", row, player_row, team_row, opp_row, event_ctx)
            doubles, _ = self.base_mean("doubles_ou", row, player_row, team_row, opp_row, event_ctx)
            hr, _ = self.base_mean("hr_milestone", row, player_row, team_row, opp_row, event_ctx)
            triples, _ = self.base_mean("triples_milestone", row, player_row, team_row, opp_row, event_ctx)
            if hits is not None:
                parts = [v for v in [doubles, hr, triples] if v is not None]
                return max(float(hits) - float(sum(parts)), 0.0), "derived_singles"
        model_name = JOBLIB_NAME_BY_FAMILY.get(family)
        if model_name and model_name in self.base_models:
            entry = self.base_models[model_name]
            feats = build_feature_vector(entry.get("features", []), player_row, team_row, opp_row, event_ctx)
            X = pd.DataFrame([feats], columns=entry.get("features", []))
            try:
                pred = float(entry["model"].predict(X)[0])
                return max(pred, 0.0), f"joblib:{model_name}"
            except Exception:
                return None, ""
        return None, ""

    def calibrator_mean(self, family: str, feature_row: dict[str, float]) -> tuple[float | None, dict[str, float]]:
        preds: dict[str, float] = {}
        X = pd.DataFrame([feature_row])
        for key in [f"{family}_xgb_reg", f"{family}_lgbm_reg", f"{family}_bayes_reg"]:
            model = self.calibrators.get(key)
            if model is None:
                continue
            try:
                preds[key] = float(model.predict(X)[0])
            except Exception:
                continue
        if not preds:
            return None, preds
        vals = np.array(list(preds.values()), dtype=float)
        return float(np.mean(vals)), preds


# ----------------------------- training -----------------------------

def fit_calibrators(history_file: Path, out_dir: Path) -> None:
    hist = normalize_columns(load_df(history_file) if history_file.exists() else pd.DataFrame())
    if hist.empty:
        raise ValueError(f"Could not load history file: {history_file}")
    if "actual value" not in hist.columns or "market family" not in hist.columns:
        raise ValueError("History file needs 'actual value' and 'market family' columns.")
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_cols = [c for c in hist.columns if c not in {"actual value", "market family", "selection result", "event date"}]
    report = []
    for family, g in hist.groupby("market family"):
        if len(g) < 40:
            continue
        X = g[feature_cols]
        y = pd.to_numeric(g["actual value"], errors="coerce").fillna(0.0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        # Bayesian ridge
        bayes = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", BayesianRidge())])
        bayes.fit(X_train, y_train)
        pred = bayes.predict(X_test)
        mae = float(mean_absolute_error(y_test, pred))
        joblib.dump(bayes, out_dir / f"{family}_bayes_reg.joblib")
        report.append({"family": family, "model": "bayes", "mae": mae})
        if XGBRegressor is not None:
            xgb = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                reg_lambda=2.0,
                random_state=42,
                n_jobs=4,
            ))])
            xgb.fit(X_train, y_train)
            pred = xgb.predict(X_test)
            mae = float(mean_absolute_error(y_test, pred))
            joblib.dump(xgb, out_dir / f"{family}_xgb_reg.joblib")
            report.append({"family": family, "model": "xgb", "mae": mae})
        if LGBMRegressor is not None:
            lgbm = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", LGBMRegressor(
                n_estimators=350,
                max_depth=-1,
                learning_rate=0.035,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="regression",
                random_state=42,
                verbose=-1,
            ))])
            lgbm.fit(X_train, y_train)
            pred = lgbm.predict(X_test)
            mae = float(mean_absolute_error(y_test, pred))
            joblib.dump(lgbm, out_dir / f"{family}_lgbm_reg.joblib")
            report.append({"family": family, "model": "lgbm", "mae": mae})
    pd.DataFrame(report).to_csv(out_dir / "training_report.csv", index=False)


# ----------------------------- projection logic -----------------------------

def infer_market_anchor_mean(group: pd.DataFrame, family: str) -> float:
    g = group.copy()
    g["fair_prob"] = fair_probs_from_prices(g["american odds"])
    if g["milestone value"].notna().any():
        pairs = [(int(r["milestone value"]), float(r["fair_prob"])) for _, r in g.iterrows()]
        return infer_poisson_mean_for_thresholds(pairs)
    if g["selection"].isin(["Over", "Under"]).all() and g["line"].notna().all():
        # choose main line by minimum sort order or lowest absolute line distance from median
        line_groups = []
        for (line,), gg in g.groupby(["line"]):
            if set(gg["selection"]) >= {"Over", "Under"}:
                fair = fair_probs_from_prices(gg["american odds"])
                p_over = float(fair[gg["selection"].eq("Over")].iloc[0])
                line_groups.append((abs(float(line)), float(line), p_over, int(gg.get("sort order", pd.Series([999999])).min())))
        if line_groups:
            line_groups.sort(key=lambda x: (x[3], x[0]))
            _, line, p_over, _ = line_groups[0]
            return infer_poisson_mean_for_over(line, p_over)
    return float(max(pd.to_numeric(g.get("line"), errors="coerce").median(skipna=True), 0.0))


def bayesian_shrink(mean: float, family: str, exposure: float) -> float:
    prior_mean, prior_strength = FAMILY_PRIORS.get(family, (mean, 10.0))
    n = max(float(exposure), 1.0)
    return (prior_mean * prior_strength + mean * n) / (prior_strength + n)


def build_meta_features(row: pd.Series, family: str, market_anchor: float, structural: float | None, player_row: dict[str, Any] | None, team_row: dict[str, Any] | None, event_ctx: dict[str, Any]) -> dict[str, float]:
    features: dict[str, float] = {
        "market_anchor_mean": market_anchor,
        "line": safe_float(row.get("line"), 0.0),
        "is_over": 1.0 if str(row.get("selection")) == "Over" else 0.0,
        "pitch_clock_era": 1.0,
        "pyth_home_prob": safe_float(event_ctx.get("pyth_home_prob"), 0.5),
        "home_runs_mu": safe_float(event_ctx.get("home_runs_mu"), 4.2),
        "away_runs_mu": safe_float(event_ctx.get("away_runs_mu"), 4.2),
        "family_prior_mean": FAMILY_PRIORS.get(family, (market_anchor, 10.0))[0],
    }
    if structural is not None:
        features["structural_mean"] = structural
    for src in [team_row or {}, player_row or {}, event_ctx]:
        for k, v in src.items():
            nk = norm_text(k)
            if nk in {"plate appearances", "plate appearances season to date", "games before today", "batters faced", "innings pitched", "innings pitched season to date",
                      "stuff plus", "stuff", "location plus", "location", "bullpen era recent 7", "bullpen k rate recent 7",
                      "park factor", "temperature", "wind speed", "lineup slot", "woba", "xwoba", "k rate", "bb rate"}:
                features[nk] = safe_float(v, 0.0)
    return features


def blend_means(family: str, market_anchor: float, structural: float | None, calibrator: float | None) -> tuple[float, str]:
    # no double counting: market information is collapsed into a single anchor mean,
    # not line + odds + fair_prob + derived line stats separately.
    model_parts = [x for x in [structural, calibrator] if x is not None]
    if model_parts:
        model_mean = float(np.mean(model_parts))
        # market anchor only used once as a regularizer, not another parallel feature pile.
        blended = 0.75 * model_mean + 0.25 * market_anchor
        return blended, "model_stack+anchor"
    return market_anchor, "market_anchor"


def sample_negative_binomial(mean: float, r: float, n: int, rng: np.random.Generator) -> np.ndarray:
    mean = max(float(mean), 1e-6)
    r = max(float(r), 1e-3)
    p = r / (r + mean)
    return rng.negative_binomial(r, p, size=n).astype(int)


def sample_game_runs(home_mu: float, away_mu: float, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    # shared gamma environment induces positive correlation without double-counting any side-market data.
    shape = 8.0
    env = rng.gamma(shape=shape, scale=1.0 / shape, size=n)
    home = rng.poisson(np.maximum(home_mu * env, 1e-6))
    away = rng.poisson(np.maximum(away_mu * env, 1e-6))
    return home.astype(int), away.astype(int)


def win_push_prob_from_samples(samples: np.ndarray, selection: str, line: float) -> tuple[float, float]:
    if selection == "Over":
        win = np.mean(samples > line)
        push = np.mean(samples == line)
    elif selection == "Under":
        win = np.mean(samples < line)
        push = np.mean(samples == line)
    else:
        win = np.mean(samples >= line)
        push = 0.0
    return float(win), float(push)


def moneyline_prob(home: np.ndarray, away: np.ndarray, side: str) -> float:
    if side.lower() == "home":
        return float(np.mean(home > away))
    return float(np.mean(away > home))


def spread_prob(home: np.ndarray, away: np.ndarray, side: str, line: float) -> tuple[float, float]:
    if side.lower() == "home":
        margin = home + line - away
    else:
        margin = away + line - home
    return float(np.mean(margin > 0)), float(np.mean(margin == 0))


# ----------------------------- main engine -----------------------------

def project_board(
    board: pd.DataFrame,
    event_ctx: dict[str, dict[str, Any]],
    player_lookup: dict[str, dict[str, Any]],
    team_lookup: dict[str, dict[str, Any]],
    model_stack: ModelStack,
    iterations: int,
) -> pd.DataFrame:
    out_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(42)

    # cache family means at the market-slice level to avoid repeated work and accidental duplication
    cache: dict[tuple[str, str, float | None, float | None], dict[str, Any]] = {}

    for _, row in board.iterrows():
        family, scope = resolve_market_family(row)
        event = str(row["event id"])
        group_key = (
            event,
            str(row["marketid"]),
            None if pd.isna(row.get("line")) else float(row.get("line")),
            None if pd.isna(row.get("milestone value")) else float(row.get("milestone value")),
        )
        group = board[
            (board["event id"].astype(str) == event)
            & (board["marketid"].astype(str) == str(row["marketid"]))
        ].copy()
        if pd.notna(row.get("line")):
            group = group[group["line"].fillna(-999999.0) == float(row.get("line"))]
        if pd.notna(row.get("milestone value")):
            group = group[group["milestone value"].fillna(-999999.0) == float(row.get("milestone value"))]

        ctx = event_ctx.get(event, {"home_runs_mu": 4.2, "away_runs_mu": 4.2, "pyth_home_prob": 0.5})
        player_key = norm_text(row.get("participant name"))
        player_row = player_lookup.get(player_key)

        team_name = str(row.get("participant name", "")) if str(row.get("participant type", "")).lower() == "team" else ""
        team_row = team_lookup.get(norm_text(team_name))

        # infer player's team from venue role -> map to event teams
        if player_row is None and str(row.get("participant type", "")).lower() == "player":
            pass
        if str(row.get("participant type", "")).lower() == "player":
            venue_role = str(row.get("participant venue role", "")).lower()
            if venue_role == "homeplayer":
                team_row = team_lookup.get(norm_text(ctx.get("home team")))
                opp_row = team_lookup.get(norm_text(ctx.get("away team")))
            elif venue_role == "awayplayer":
                team_row = team_lookup.get(norm_text(ctx.get("away team")))
                opp_row = team_lookup.get(norm_text(ctx.get("home team")))
            else:
                opp_row = None
        else:
            if norm_text(team_name) == norm_text(ctx.get("home team")):
                opp_row = team_lookup.get(norm_text(ctx.get("away team")))
            else:
                opp_row = team_lookup.get(norm_text(ctx.get("home team")))

        if group_key not in cache:
            market_anchor = infer_market_anchor_mean(group, family)
            structural, structural_source = model_stack.base_mean(family, row, player_row, team_row, opp_row, ctx)
            meta_features = build_meta_features(row, family, market_anchor, structural, player_row, team_row, ctx)
            calibrator, calibrator_parts = model_stack.calibrator_mean(family, meta_features)
            blended, blend_source = blend_means(family, market_anchor, structural, calibrator)
            exposure = find_exposure(player_row, team_row)
            posterior_mean = bayesian_shrink(blended, family, exposure)
            cache[group_key] = {
                "market_anchor_mean": market_anchor,
                "structural_mean": structural,
                "calibrator_mean": calibrator,
                "posterior_mean": posterior_mean,
                "source": blend_source,
                "structural_source": structural_source,
                "calibrator_parts": calibrator_parts,
            }
        cached = cache[group_key]

        # simulate per selection
        mean = float(cached["posterior_mean"])
        family_r = OVERDISPERSION.get(family, 12.0)
        win_prob = push_prob = np.nan
        sim_mean = sim_p10 = sim_p50 = sim_p90 = mean
        selection = str(row.get("selection", ""))
        line = None if pd.isna(row.get("line")) else float(row.get("line"))
        fair_american = np.nan

        if family in {"game_moneyline", "game_spread", "game_total", "team_total", "alt_team_total"}:
            home_mu = float(ctx.get("home_runs_mu", 4.2))
            away_mu = float(ctx.get("away_runs_mu", 4.2))
            home_s, away_s = sample_game_runs(home_mu, away_mu, iterations, rng)
            total_s = home_s + away_s
            sim_mean = float(np.mean(total_s))
            sim_p10, sim_p50, sim_p90 = [float(x) for x in np.quantile(total_s, [0.1, 0.5, 0.9])]
            if family == "game_moneyline":
                side = str(row.get("outcome type", selection)).lower()
                win_prob = moneyline_prob(home_s, away_s, side)
                push_prob = 0.0
                fair_american = prob_to_american(win_prob)
                mean = float(home_mu if side == "home" else away_mu)
            elif family == "game_spread":
                side = str(row.get("outcome type", selection)).lower()
                line_val = float(line or 0.0)
                win_prob, push_prob = spread_prob(home_s, away_s, side, line_val)
                fair_american = prob_to_american(win_prob / max(1.0 - push_prob, 1e-9))
                mean = float((home_mu - away_mu) if side == "home" else (away_mu - home_mu))
            elif family == "game_total":
                win_prob, push_prob = win_push_prob_from_samples(total_s, selection, float(line or 0.0))
                fair_american = prob_to_american(win_prob / max(1.0 - push_prob, 1e-9))
                mean = float(home_mu + away_mu)
            else:
                team = norm_text(row.get("participant name", ""))
                samples = home_s if team == norm_text(ctx.get("home team")) else away_s
                sim_mean = float(np.mean(samples))
                sim_p10, sim_p50, sim_p90 = [float(x) for x in np.quantile(samples, [0.1, 0.5, 0.9])]
                win_prob, push_prob = win_push_prob_from_samples(samples, selection, float(line or 0.0))
                fair_american = prob_to_american(win_prob / max(1.0 - push_prob, 1e-9))
                mean = sim_mean
        elif family == "first_inning_runs":
            samples = sample_negative_binomial(mean, OVERDISPERSION[family], iterations, rng)
            sim_mean = float(np.mean(samples))
            sim_p10, sim_p50, sim_p90 = [float(x) for x in np.quantile(samples, [0.1, 0.5, 0.9])]
            win_prob, push_prob = win_push_prob_from_samples(samples, selection, float(line or 0.0))
            fair_american = prob_to_american(win_prob / max(1.0 - push_prob, 1e-9))
        elif row.get("milestone value") == row.get("milestone value"):
            threshold = int(float(row.get("milestone value")))
            samples = sample_negative_binomial(mean, family_r, iterations, rng)
            sim_mean = float(np.mean(samples))
            sim_p10, sim_p50, sim_p90 = [float(x) for x in np.quantile(samples, [0.1, 0.5, 0.9])]
            win_prob = float(np.mean(samples >= threshold))
            push_prob = 0.0
            fair_american = prob_to_american(win_prob)
        else:
            samples = sample_negative_binomial(mean, family_r, iterations, rng)
            sim_mean = float(np.mean(samples))
            sim_p10, sim_p50, sim_p90 = [float(x) for x in np.quantile(samples, [0.1, 0.5, 0.9])]
            win_prob, push_prob = win_push_prob_from_samples(samples, selection, float(line or 0.0))
            fair_american = prob_to_american(win_prob / max(1.0 - push_prob, 1e-9))

        market_prob = american_to_prob(row.get("american odds"))
        edge = None if market_prob is None or np.isnan(win_prob) else float(win_prob - market_prob)
        fair_prob_eff = None if np.isnan(win_prob) else float(win_prob / max(1.0 - push_prob, 1e-9))
        ev = None
        a = clean_american(row.get("american odds"))
        if a is not None and not np.isnan(win_prob):
            profit = (a / 100.0) if a > 0 else (100.0 / abs(a))
            ev = float(win_prob * profit - (1.0 - win_prob - push_prob))

        out = row.to_dict()
        out.update({
            "market family": family,
            "scope": scope,
            "market anchor mean": cached["market_anchor_mean"],
            "structural mean": cached["structural_mean"],
            "calibrator mean": cached["calibrator_mean"],
            "projected mean": mean,
            "sim mean": sim_mean,
            "sim p10": sim_p10,
            "sim median": sim_p50,
            "sim p90": sim_p90,
            "win probability": win_prob,
            "push probability": push_prob,
            "fair probability": fair_prob_eff,
            "fair american": fair_american,
            "edge": edge,
            "ev": ev,
            "projection source": cached["source"],
            "structural source": cached["structural_source"],
            "pyth home probability": ctx.get("pyth_home_prob"),
            "home runs mu": ctx.get("home_runs_mu"),
            "away runs mu": ctx.get("away_runs_mu"),
        })
        out_rows.append(out)

    return pd.DataFrame(out_rows)


# ----------------------------- cli -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full-stack MLB market projector with Monte Carlo, Bayesian shrinkage, optional XGBoost/LightGBM calibrators, and Pythagorean game context.")
    p.add_argument("--board-file", type=Path)
    p.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    p.add_argument("--player-form-file", type=Path, default=None)
    p.add_argument("--team-form-file", type=Path, default=None)
    p.add_argument("--model-dir", type=Path, default=None)
    p.add_argument("--calibrator-dir", type=Path, default=None)
    p.add_argument("--iterations", type=int, default=50000)
    p.add_argument("--fit-calibrators-from-history", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.fit_calibrators_from_history is not None:
        fit_dir = args.calibrator_dir or args.out_dir / "market_calibrators"
        fit_dir.mkdir(parents=True, exist_ok=True)
        fit_calibrators(args.fit_calibrators_from_history, fit_dir)
        print(f"calibrators: {fit_dir}")
        return

    board = load_board(args.board_file)
    player_forms = load_df(args.player_form_file) if args.player_form_file else None
    team_forms = load_df(args.team_form_file) if args.team_form_file else None
    player_lookup = build_player_lookup(player_forms)
    team_lookup = build_team_lookup(team_forms)
    model_dir = args.model_dir
    if model_dir is None:
        candidate = args.out_dir / "history trained models"
        model_dir = candidate if candidate.exists() else None
    calibrator_dir = args.calibrator_dir
    if calibrator_dir is None:
        candidate = args.out_dir / "market_calibrators"
        calibrator_dir = candidate if candidate.exists() else None
    models = load_joblib_models(model_dir)
    stack = ModelStack(models, calibrator_dir)
    contexts = build_event_contexts(board)
    projected = project_board(board, contexts, player_lookup, team_lookup, stack, args.iterations)

    projected.to_csv(args.out_dir / "draftkings full stack projected all markets.csv", index=False)
    projected[projected["scope"].eq("player")].to_csv(args.out_dir / "draftkings full stack projected player props.csv", index=False)
    projected[projected["scope"].isin(["game", "team"])].to_csv(args.out_dir / "draftkings full stack projected game lines.csv", index=False)
    projected.sort_values(["edge", "ev"], ascending=False).head(500).to_csv(args.out_dir / "draftkings full stack best edges.csv", index=False)

    summary = (
        projected.groupby(["market family", "scope"], dropna=False)
        .agg(rows=("market", "size"), avg_edge=("edge", "mean"), avg_ev=("ev", "mean"))
        .reset_index()
        .sort_values(["scope", "rows"], ascending=[True, False])
    )
    summary.to_csv(args.out_dir / "draftkings full stack summary.csv", index=False)
    print(f"all markets: {args.out_dir / 'draftkings full stack projected all markets.csv'}")
    print(f"player props: {args.out_dir / 'draftkings full stack projected player props.csv'}")
    print(f"game lines: {args.out_dir / 'draftkings full stack projected game lines.csv'}")
    print(f"best edges: {args.out_dir / 'draftkings full stack best edges.csv'}")
    print(f"summary: {args.out_dir / 'draftkings full stack summary.csv'}")


if __name__ == "__main__":
    main()
