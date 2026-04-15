from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
from datetime import datetime
from typing import Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the nightly MLB Unabated workflow.")
    parser.add_argument("--player-stack-input", default=None)
    parser.add_argument("--game-stack-input", default=None)
    parser.add_argument("--player-neutral-input", default=None)
    parser.add_argument("--game-neutral-input", default=None)
    parser.add_argument("--projection-file", default=None)
    parser.add_argument("--unabated-props-input", required=True)
    parser.add_argument("--unabated-game-input", required=True)
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--run-date", default=None, help="Optional YYYY-MM-DD override. Defaults to today.")
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _dated_copy(src: pathlib.Path, stamp: str) -> pathlib.Path | None:
    if not src.exists() or not src.is_file():
        return None
    dated = src.with_name(f"{src.stem}_{stamp}{src.suffix}")
    shutil.copy2(src, dated)
    return dated


def _stamp_outputs(out_dir: pathlib.Path, stamp: str) -> None:
    names = [
        "mlb_model_player_projections.csv",
        "mlb_model_game_projections.csv",
        "mlb_live_projections.csv",
        "mlb_live_market_hub.csv",
        "mlb_live_scored_board.csv",
        "mlb_live_player_board.csv",
        "mlb_live_team_board.csv",
        "mlb_live_game_board.csv",
        "mlb_live_score_status_summary.csv",
        "mlb_live_market_family_summary.csv",
        "mlb_live_bet_type_summary.csv",
        "mlb_tonight_bet_card.csv",
        "mlb_tonight_bet_card.xlsx",
        "mlb_tonight_bet_card_best_book.csv",
        "mlb_tonight_bet_card_best_book.xlsx",
        "mlb_daily_bet_card.csv",
        "mlb_daily_bet_card.xlsx",
        "mlb_weather.csv",
    ]
    copied = []
    for name in names:
        out = _dated_copy(out_dir / name, stamp)
        if out is not None:
            copied.append(out.name)
    if copied:
        print(f"dated_copies={len(copied)}")
        for name in copied:
            print(out_dir / name)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args()
    root = pathlib.Path.cwd()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = args.run_date or datetime.now().strftime("%Y-%m-%d")

    player_neutral = pathlib.Path(args.player_neutral_input) if args.player_neutral_input else out_dir / "mlb_model_player_projections.csv"
    game_neutral = pathlib.Path(args.game_neutral_input) if args.game_neutral_input else out_dir / "mlb_model_game_projections.csv"
    proj_path = pathlib.Path(args.projection_file) if args.projection_file else out_dir / "mlb_live_projections.csv"

    if args.player_stack_input and args.game_stack_input:
        _run([
            sys.executable,
            str(root / "export_mlb_model_projections.py"),
            "--player-stack-input", args.player_stack_input,
            "--game-stack-input", args.game_stack_input,
            "--player-out", str(player_neutral),
            "--game-out", str(game_neutral),
        ])

    if player_neutral.exists() or game_neutral.exists():
        build_cmd = [
            sys.executable,
            str(root / "build_mlb_live_projections.py"),
            "--out", str(proj_path),
        ]
        if player_neutral.exists():
            build_cmd += ["--player-input", str(player_neutral)]
        if game_neutral.exists():
            build_cmd += ["--game-input", str(game_neutral)]
        _run(build_cmd)
    elif not proj_path.exists():
        raise FileNotFoundError(
            "Could not find neutral projection inputs or an existing mlb_live_projections.csv. "
            "Provide --player-stack-input/--game-stack-input, or place mlb_model_* files, or keep mlb_live_projections.csv in artifacts."
        )

    _run([
        sys.executable,
        str(root / "run_mlb_live_board.py"),
        "--unabated-props-input", args.unabated_props_input,
        "--unabated-game-input", args.unabated_game_input,
        "--projection-file", str(proj_path),
        "--out-dir", str(out_dir),
        "--iterations", str(args.iterations),
    ])

    _stamp_outputs(out_dir, stamp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
