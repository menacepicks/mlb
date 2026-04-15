from __future__ import annotations

import argparse
from pathlib import Path

from .config import AppConfig
from .models.history_trained_models import score_game_board, score_player_prop_board, train_history_models
from .pipelines.build_draftkings_archive import capture_draftkings_snapshot, rebuild_draftkings_history
from .pipelines.build_draftkings_board import build_draftkings_mlb_board
from .pipelines.build_fanduel_archive import capture_fanduel_snapshot, rebuild_fanduel_history
from .pipelines.build_fanduel_board import build_fanduel_mlb_board
from .pipelines.build_historical_betting_data import build_historical_betting_data_artifacts
from .pipelines.build_risk_guide import build_risk_guide
from .pipelines.compare_books import compare_mlb_books


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLB betting workbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    history = subparsers.add_parser("build-historical-betting-data")
    history.add_argument("--zip-dir", type=Path, default=None)
    history.add_argument("--zip-glob", default=None)
    history.add_argument("--out-dir", required=True, type=Path)
    history.add_argument("--force-rebuild", action="store_true")

    dk_board = subparsers.add_parser("build-draftkings-mlb-board")
    dk_board.add_argument("--capture-file", action="append", default=[])
    dk_board.add_argument("--use-live-capture", action="store_true")
    dk_board.add_argument("--use-browser", action="store_true")
    dk_board.add_argument("--out-dir", required=True, type=Path)

    fd_board = subparsers.add_parser("build-fanduel-mlb-board")
    fd_board.add_argument("--capture-file", action="append", default=[])
    fd_board.add_argument("--use-live-capture", action="store_true")
    fd_board.add_argument("--out-dir", required=True, type=Path)

    train = subparsers.add_parser("train-history-models")
    train.add_argument("--out-dir", required=True, type=Path)

    score = subparsers.add_parser("score-player-prop-board")
    score.add_argument("--board-file", required=True, type=Path)
    score.add_argument("--out-dir", required=True, type=Path)

    score_games = subparsers.add_parser("score-game-board")
    score_games.add_argument("--board-file", required=True, type=Path)
    score_games.add_argument("--out-dir", required=True, type=Path)

    snapshot = subparsers.add_parser("capture-draftkings-snapshot")
    snapshot.add_argument("--capture-file", action="append", default=[])
    snapshot.add_argument("--use-live-capture", action="store_true")
    snapshot.add_argument("--use-browser", action="store_true")
    snapshot.add_argument("--out-dir", required=True, type=Path)

    rebuild = subparsers.add_parser("rebuild-draftkings-history")
    rebuild.add_argument("--out-dir", required=True, type=Path)

    fd_snapshot = subparsers.add_parser("capture-fanduel-snapshot")
    fd_snapshot.add_argument("--capture-file", action="append", default=[])
    fd_snapshot.add_argument("--use-live-capture", action="store_true")
    fd_snapshot.add_argument("--out-dir", required=True, type=Path)

    fd_rebuild = subparsers.add_parser("rebuild-fanduel-history")
    fd_rebuild.add_argument("--out-dir", required=True, type=Path)

    risk = subparsers.add_parser("build-risk-guide")
    risk.add_argument("--out-dir", required=True, type=Path)

    compare = subparsers.add_parser("compare-mlb-books")
    compare.add_argument("--out-dir", required=True, type=Path)

    return parser.parse_args()


def _print_outputs(outputs: dict[str, Path]) -> None:
    for name, path in outputs.items():
        print(f"{name}: {path}")


def main() -> None:
    args = parse_args()
    config = AppConfig(out_dir=args.out_dir)

    if args.command == "build-historical-betting-data":
        _print_outputs(build_historical_betting_data_artifacts(
            zip_dir=args.zip_dir,
            zip_glob=args.zip_glob,
            config=config,
            force_rebuild=args.force_rebuild,
        ))
        return

    if args.command == "build-draftkings-mlb-board":
        _print_outputs(build_draftkings_mlb_board(
            config=config,
            capture_files=args.capture_file,
            use_live_capture=args.use_live_capture,
            use_browser=args.use_browser,
        ))
        return

    if args.command == "build-fanduel-mlb-board":
        _print_outputs(build_fanduel_mlb_board(
            config=config,
            capture_files=args.capture_file,
            use_live_capture=args.use_live_capture,
        ))
        return

    if args.command == "train-history-models":
        _print_outputs(train_history_models(config))
        return

    if args.command == "score-player-prop-board":
        _print_outputs(score_player_prop_board(config, args.board_file))
        return

    if args.command == "score-game-board":
        _print_outputs(score_game_board(config, args.board_file))
        return

    if args.command == "capture-draftkings-snapshot":
        _print_outputs(capture_draftkings_snapshot(
            config=config,
            capture_files=args.capture_file,
            use_live_capture=args.use_live_capture,
            use_browser=args.use_browser,
        ))
        return

    if args.command == "rebuild-draftkings-history":
        _print_outputs(rebuild_draftkings_history(config))
        return

    if args.command == "capture-fanduel-snapshot":
        _print_outputs(capture_fanduel_snapshot(
            config=config,
            capture_files=args.capture_file,
            use_live_capture=args.use_live_capture,
        ))
        return

    if args.command == "rebuild-fanduel-history":
        _print_outputs(rebuild_fanduel_history(config))
        return

    if args.command == "build-risk-guide":
        _print_outputs(build_risk_guide(config))
        return

    if args.command == "compare-mlb-books":
        _print_outputs(compare_mlb_books(config))
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
