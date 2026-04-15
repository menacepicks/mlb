from __future__ import annotations

import argparse
import pathlib
import subprocess
from typing import Sequence

import pandas as pd

from mlb_live_pricing import PricingConfig, load_projection_table, score_market_board
from mlb_live_schema import ensure_unified_columns
from mlb_live_unabated import BETTYPE_URL, load_unabated_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a unified MLB live market board from Unabated snapshots")
    parser.add_argument("--unabated-props-input", default=None, help="Saved Unabated props snapshot JSON")
    parser.add_argument("--unabated-game-input", default=None, help="Saved Unabated game odds snapshot JSON")
    parser.add_argument("--unabated-props-url", default=None, help="Optional Unabated props URL to fetch live")
    parser.add_argument("--unabated-game-url", default=None, help="Optional Unabated game odds URL to fetch live")
    parser.add_argument("--unabated-bettypes-input", default=None, help="Optional saved Unabated bet types JSON")
    parser.add_argument("--unabated-bettypes-url", default=BETTYPE_URL, help="Optional Unabated bet types URL")
    parser.add_argument("--projection-file", default=None, help="Optional projection CSV/parquet with scope/segment/family keys")
    parser.add_argument("--out-dir", default="artifacts", help="Output directory")
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pre-capture-cmd", default=None, help="Optional command to run before loading inputs")
    return parser.parse_args()


def _run_cmd(cmd: str | None, root: pathlib.Path) -> None:
    if not cmd:
        return
    subprocess.run(cmd, shell=True, check=True, cwd=root)


def _write_outputs(df: pd.DataFrame, out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "mlb_live_scored_board.csv", index=False)
    df[df["scope"] == "player"].to_csv(out_dir / "mlb_live_player_board.csv", index=False)
    df[df["scope"] == "team"].to_csv(out_dir / "mlb_live_team_board.csv", index=False)
    df[df["scope"] == "game"].to_csv(out_dir / "mlb_live_game_board.csv", index=False)

    hub_cols = [
        "event_id",
        "market_group_key",
        "market_family",
        "scope",
        "segment",
        "participant",
        "team_name",
        "line",
        "milestone_value",
        "book",
        "selection",
        "price_american",
        "fair_prob",
        "model_prob",
        "edge_pct_points",
        "bet_type_name",
    ]
    hub_cols = [c for c in hub_cols if c in df.columns]
    df[hub_cols].sort_values([c for c in ["event_id", "scope", "market_family", "participant", "team_name", "book"] if c in hub_cols]).to_csv(
        out_dir / "mlb_live_market_hub.csv", index=False
    )

    (
        df.groupby("score_status", dropna=False).size().reset_index(name="rows").sort_values("rows", ascending=False)
        .to_csv(out_dir / "mlb_live_score_status_summary.csv", index=False)
    )
    (
        df.groupby(["market_family", "segment", "scope", "score_status"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values("rows", ascending=False)
        .to_csv(out_dir / "mlb_live_market_family_summary.csv", index=False)
    )
    if "bet_type_name" in df.columns:
        (
            df.groupby(["bet_type_name", "market_family", "segment", "scope"], dropna=False)
            .size()
            .reset_index(name="rows")
            .sort_values("rows", ascending=False)
            .to_csv(out_dir / "mlb_live_bet_type_summary.csv", index=False)
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args()
    root = pathlib.Path.cwd()
    out_dir = pathlib.Path(args.out_dir)

    _run_cmd(args.pre_capture_cmd, root)

    merged = load_unabated_bundle(
        props_path=args.unabated_props_input,
        game_path=args.unabated_game_input,
        props_url=args.unabated_props_url,
        game_url=args.unabated_game_url,
        bettypes_path=args.unabated_bettypes_input,
        bettypes_url=args.unabated_bettypes_url,
    )
    merged = ensure_unified_columns(merged)
    projections = load_projection_table(args.projection_file) if args.projection_file else pd.DataFrame()
    scored = score_market_board(merged, projections=projections, config=PricingConfig(iterations=args.iterations, seed=args.seed))
    _write_outputs(scored, out_dir)

    print(f"merged_rows={len(merged)}")
    print(f"scored_rows={len(scored)}")
    for scope in ("player", "team", "game"):
        print(f"{scope}_rows={int((scored['scope'] == scope).sum())}")
    for _, row in (
        scored.groupby("score_status", dropna=False).size().reset_index(name="rows").sort_values("rows", ascending=False).iterrows()
    ):
        print(f"score_status[{row['score_status']}]={int(row['rows'])}")
    print(out_dir / "mlb_live_market_hub.csv")
    print(out_dir / "mlb_live_scored_board.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
