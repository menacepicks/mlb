from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
from typing import Sequence

import pandas as pd

from mlb_live_books import load_draftkings_input, load_fanduel_input
from mlb_live_pricing import PricingConfig, load_projection_table, score_market_board
from mlb_live_schema import ensure_unified_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a unified MLB live market board from DraftKings + FanDuel")
    parser.add_argument("--dk-input", default=None, help="DraftKings raw JSON or shared CSV/parquet")
    parser.add_argument("--fd-shared-input", default=None, help="FanDuel shared CSV/parquet")
    parser.add_argument("--fd-event-pages", default=None, help="FanDuel raw event pages JSON")
    parser.add_argument("--fd-price-rows", default=None, help="FanDuel raw price rows JSON")
    parser.add_argument("--projection-file", default=None, help="Optional projection CSV/parquet with scope/segment/family keys")
    parser.add_argument("--out-dir", default="artifacts", help="Output directory")
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dk-capture-cmd", default=None, help="Optional command to run before loading DK input")
    parser.add_argument("--fd-capture-cmd", default=None, help="Optional command to run before loading FD input")
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
    ]
    df[hub_cols].sort_values(["event_id", "scope", "market_family", "participant", "team_name", "book"]).to_csv(
        out_dir / "mlb_live_market_hub.csv", index=False
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args()
    root = pathlib.Path.cwd()
    out_dir = pathlib.Path(args.out_dir)

    _run_cmd(args.dk_capture_cmd, root)
    _run_cmd(args.fd_capture_cmd, root)

    frames: list[pd.DataFrame] = []
    if args.dk_input:
        frames.append(load_draftkings_input(args.dk_input))
    if args.fd_shared_input or (args.fd_event_pages and args.fd_price_rows):
        frames.append(
            load_fanduel_input(
                shared_path=args.fd_shared_input,
                event_pages_path=args.fd_event_pages,
                price_rows_path=args.fd_price_rows,
            )
        )

    if not frames:
        raise ValueError("Provide at least one of --dk-input or --fd-shared-input/--fd-event-pages+--fd-price-rows")

    merged = ensure_unified_columns(pd.concat(frames, ignore_index=True))
    projections = load_projection_table(args.projection_file) if args.projection_file else pd.DataFrame()
    scored = score_market_board(merged, projections=projections, config=PricingConfig(iterations=args.iterations, seed=args.seed))
    _write_outputs(scored, out_dir)

    print(f"merged_rows={len(merged)}")
    print(f"scored_rows={len(scored)}")
    for scope in ("player", "team", "game"):
        print(f"{scope}_rows={int((scored['scope'] == scope).sum())}")
    print(out_dir / "mlb_live_market_hub.csv")
    print(out_dir / "mlb_live_scored_board.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
