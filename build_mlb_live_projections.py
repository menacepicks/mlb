
from __future__ import annotations

import argparse
import pathlib
from typing import Any

import pandas as pd


COLUMN_ALIASES = {
    "event id": "event_id",
    "event_id": "event_id",
    "scope": "scope",
    "market family": "market_family",
    "market_family": "market_family",
    "participant": "participant",
    "participant name": "participant_name",
    "canonical selection": "selection",
    "selection": "selection",
    "projected mean": "projection_mean",
    "projection_mean": "projection_mean",
    "win probability": "projection_prob",
    "projection_prob": "projection_prob",
    "line": "line",
    "market": "market",
    "market type": "market_type",
    "market type norm": "market_type_norm",
    "stat prefix": "stat_prefix",
    "board scope": "board_scope",
    "participant type": "participant_type",
    "home team": "home_team",
    "away team": "away_team",
    "event name": "event_name",
    "projection source": "projection_source",
}


SEGMENT_ALIASES = {
    "full game": "full_game",
    "full_game": "full_game",
    "moneyline": "full_game",
    "run line": "full_game",
    "total": "full_game",
    "game total": "full_game",
    "first five": "first_five_innings",
    "1st inning": "inning_1",
    "first inning": "inning_1",
    "2nd inning": "inning_2",
    "second inning": "inning_2",
    "3rd inning": "inning_3",
    "third inning": "inning_3",
    "4th inning": "inning_4",
    "fourth inning": "inning_4",
    "5th inning": "inning_5",
    "fifth inning": "inning_5",
}


FAMILY_ALIASES = {
    "game_moneyline": "moneyline",
    "moneyline": "moneyline",
    "game_spread": "spread",
    "spread": "spread",
    "run_line": "spread",
    "game_total_runs": "game_total_runs",
    "team_total_runs": "team_total_runs",
    "team_total": "team_total_runs",
    "tb_ou": "total_bases",
    "total_bases": "total_bases",
    "hits_ou": "hits",
    "hits": "hits",
    "runs_ou": "runs",
    "runs": "runs",
    "rbis_ou": "rbis",
    "rbis": "rbis",
    "hrr_ou": "hits_runs_rbis",
    "hits_runs_rbis": "hits_runs_rbis",
    "hr_ou": "home_runs",
    "home_runs": "home_runs",
    "sb_ou": "stolen_bases",
    "stolen_bases": "stolen_bases",
    "singles_ou": "singles",
    "singles": "singles",
    "doubles_ou": "doubles",
    "doubles": "doubles",
    "triples_ou": "triples",
    "triples": "triples",
    "pitcher_strikeouts": "pitcher_strikeouts",
    "strikeouts": "pitcher_strikeouts",
    "k_ou": "pitcher_strikeouts",
    "so_ou": "pitcher_strikeouts",
    "pitching_outs": "pitching_outs",
    "outs_recorded": "pitching_outs",
    "outs_ou": "pitching_outs",
    "earned_runs": "earned_runs",
    "earned_runs_allowed": "earned_runs",
    "walks_allowed": "walks_allowed",
    "hits_allowed": "hits_allowed",
}


def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Combine duplicate column names after alias renaming.

    Example: source file has both 'selection' and 'canonical selection', both renamed
    to 'selection'. Pandas then returns a DataFrame when selecting that label.
    """
    if df.columns.is_unique:
        return df

    out = pd.DataFrame(index=df.index)
    seen: list[str] = []
    for col in df.columns:
        if col in seen:
            continue
        same = df.loc[:, df.columns == col]
        if same.shape[1] == 1:
            out[col] = same.iloc[:, 0]
        else:
            merged = same.iloc[:, 0].copy()
            for i in range(1, same.shape[1]):
                merged = merged.combine_first(same.iloc[:, i])
            out[col] = merged
        seen.append(col)
    return out


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    out = out.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in out.columns})
    out = _coalesce_duplicate_columns(out)
    return out


def _clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return " ".join(str(value).strip().split())


def _canonical_selection(value: Any) -> str:
    text = _clean_text(value).lower()
    mapping = {
        "over": "over",
        "under": "under",
        "home": "home",
        "away": "away",
        "yes": "yes",
        "no": "no",
    }
    return mapping.get(text, text)


def _infer_segment(row: pd.Series) -> str:
    text = " ".join(
        _clean_text(row.get(col))
        for col in ("market", "market_type", "market_type_norm", "board_scope", "event_name")
    ).lower()

    for key, segment in SEGMENT_ALIASES.items():
        if key in text:
            return segment

    return "full_game"


def _infer_scope(row: pd.Series) -> str:
    scope = _clean_text(row.get("scope")).lower()
    if scope in {"player", "team", "game"}:
        return scope

    participant_type = _clean_text(row.get("participant_type")).lower()
    if participant_type == "player":
        return "player"
    if participant_type == "team":
        if "team total" in _clean_text(row.get("market")).lower():
            return "team"
        return "game"

    board_scope = _clean_text(row.get("board_scope")).lower()
    if "player" in board_scope:
        return "player"
    if "team" in board_scope:
        return "team"

    market_text = _clean_text(row.get("market")).lower()
    if "team total" in market_text:
        return "team"
    if market_text in {"moneyline", "run line"} or "total" in market_text:
        return "game"

    return "player"


def _infer_family(row: pd.Series) -> str:
    raw_family = _clean_text(row.get("market_family")).lower()
    if raw_family in FAMILY_ALIASES:
        return FAMILY_ALIASES[raw_family]

    stat_prefix = _clean_text(row.get("stat_prefix")).upper()
    prefix_map = {
        "TB": "total_bases",
        "H": "hits",
        "R": "runs",
        "RBI": "rbis",
        "H+R+RBI": "hits_runs_rbis",
        "HR": "home_runs",
        "SB": "stolen_bases",
        "1B": "singles",
        "2B": "doubles",
        "3B": "triples",
        "K": "pitcher_strikeouts",
        "OUTS": "pitching_outs",
        "ER": "earned_runs",
        "BB": "walks_allowed",
        "HA": "hits_allowed",
    }
    if stat_prefix in prefix_map:
        return prefix_map[stat_prefix]

    market_text = " ".join(
        _clean_text(row.get(col)) for col in ("market", "market_type", "market_type_norm")
    ).lower()

    rules = [
        ("moneyline", "moneyline"),
        ("run line", "spread"),
        ("spread", "spread"),
        ("team total", "team_total_runs"),
        ("total bases", "total_bases"),
        ("hits+runs+rbis", "hits_runs_rbis"),
        ("hits runs rbis", "hits_runs_rbis"),
        ("home runs", "home_runs"),
        ("strikeouts", "pitcher_strikeouts"),
        ("outs recorded", "pitching_outs"),
        ("pitching outs", "pitching_outs"),
        ("earned runs", "earned_runs"),
        ("walks allowed", "walks_allowed"),
        ("hits allowed", "hits_allowed"),
        ("stolen bases", "stolen_bases"),
        ("singles", "singles"),
        ("doubles", "doubles"),
        ("triples", "triples"),
        ("rbis", "rbis"),
        ("runs", "runs"),
        ("hits", "hits"),
        ("total", "game_total_runs"),
    ]
    for key, family in rules:
        if key in market_text:
            return family

    return raw_family or ""


def _team_name(row: pd.Series, scope: str) -> str:
    participant = _clean_text(row.get("participant"))
    selection = _canonical_selection(row.get("selection"))
    if scope == "team":
        return participant
    if scope == "game" and selection == "home":
        return _clean_text(row.get("home_team"))
    if scope == "game" and selection == "away":
        return _clean_text(row.get("away_team"))
    return ""


def _participant_name(row: pd.Series, scope: str) -> str:
    if scope == "player":
        return _clean_text(row.get("participant")) or _clean_text(row.get("participant_name"))
    return ""


def _coerce_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _empty_output() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_id",
            "scope",
            "segment",
            "market_family",
            "participant",
            "team_name",
            "selection",
            "projection_mean",
            "projection_prob",
            "projection_source",
            "source_market_name",
            "line",
        ]
    )


def _build_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = _normalize_columns(df)
    if out.empty:
        return _empty_output()

    for col in (
        "event_id",
        "participant",
        "selection",
        "market_family",
        "scope",
        "market",
        "market_type",
        "market_type_norm",
        "stat_prefix",
        "board_scope",
        "participant_type",
        "home_team",
        "away_team",
        "event_name",
        "projection_source",
        "line",
        "projection_mean",
        "projection_prob",
        "participant_name",
    ):
        if col not in out.columns:
            out[col] = pd.NA

    out["scope_norm"] = out.apply(_infer_scope, axis=1)
    out["segment_norm"] = out.apply(_infer_segment, axis=1)
    out["family_norm"] = out.apply(_infer_family, axis=1)

    selection_series = out["selection"]
    if isinstance(selection_series, pd.DataFrame):
        selection_series = selection_series.iloc[:, 0]
    out["selection_norm"] = selection_series.map(_canonical_selection)

    out["participant_norm"] = out.apply(lambda r: _participant_name(r, r["scope_norm"]), axis=1)
    out["team_name_norm"] = out.apply(lambda r: _team_name(r, r["scope_norm"]), axis=1)

    out["projection_mean"] = _coerce_num(out["projection_mean"])
    out["projection_prob"] = _coerce_num(out["projection_prob"])
    out["line"] = _coerce_num(out["line"])

    keep = out["event_id"].notna() & out["family_norm"].ne("") & out["scope_norm"].ne("")
    keep &= out["projection_mean"].notna() | out["projection_prob"].notna()

    built = pd.DataFrame(
        {
            "event_id": out.loc[keep, "event_id"].astype(str),
            "scope": out.loc[keep, "scope_norm"],
            "segment": out.loc[keep, "segment_norm"],
            "market_family": out.loc[keep, "family_norm"],
            "participant": out.loc[keep, "participant_norm"],
            "team_name": out.loc[keep, "team_name_norm"],
            "selection": out.loc[keep, "selection_norm"],
            "projection_mean": out.loc[keep, "projection_mean"],
            "projection_prob": out.loc[keep, "projection_prob"],
            "projection_source": out.loc[keep, "projection_source"].fillna("full_stack"),
            "source_market_name": out.loc[keep, "market"].fillna(""),
            "line": out.loc[keep, "line"],
        }
    )

    built.loc[built["projection_mean"].notna(), "selection"] = built.loc[
        built["projection_mean"].notna(), "selection"
    ].fillna("")

    built["projection_mean"] = built["projection_mean"].round(4)
    built["projection_prob"] = built["projection_prob"].round(6)
    built["line"] = built["line"].round(3)
    return built


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a clean MLB live projection file from full-stack player props and game lines."
    )
    parser.add_argument("--player-props", required=True, help="Path to projected player props CSV")
    parser.add_argument("--game-lines", required=True, help="Path to projected game lines CSV")
    parser.add_argument("--out", default="artifacts/mlb_live_projections.csv", help="Output CSV path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    player_df = pd.read_csv(args.player_props)
    game_df = pd.read_csv(args.game_lines)

    frames = [_build_rows(player_df), _build_rows(game_df)]
    out = pd.concat(frames, ignore_index=True)

    out["_has_prob"] = out["projection_prob"].notna().astype(int)
    out["_has_mean"] = out["projection_mean"].notna().astype(int)

    mean_rows = (
        out[out["projection_mean"].notna()]
        .sort_values(by=["_has_prob", "_has_mean"], ascending=False, kind="stable")
        .drop_duplicates(
            subset=["event_id", "scope", "segment", "market_family", "participant", "team_name"],
            keep="first",
        )
    )

    prob_rows = (
        out[out["projection_prob"].notna()]
        .sort_values(by=["_has_prob", "_has_mean"], ascending=False, kind="stable")
        .drop_duplicates(
            subset=["event_id", "scope", "segment", "market_family", "participant", "team_name", "selection"],
            keep="first",
        )
    )

    merged = (
        pd.concat([mean_rows, prob_rows], ignore_index=True)
        .drop(columns=["_has_prob", "_has_mean"])
        .drop_duplicates()
        .sort_values(
            by=["event_id", "scope", "segment", "market_family", "participant", "team_name", "selection"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    print(f"projection_rows={len(merged)}")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
