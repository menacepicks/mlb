from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..cache import build_file_signature, read_json, signature_hash, write_json
from ..config import AppConfig
from ..data.retrosheet_parser import build_historical_betting_data, list_zip_paths
from ..io import ensure_dir, save_table


SLICE_NAMES = [
    "player-directory",
    "actual-game-results",
    "actual-team-game-results",
    "actual-hitter-stats-by-game",
    "actual-pitcher-stats-by-game",
]


def _safe_stem(path: Path) -> str:
    stem = path.stem
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "."} else "-" for ch in stem)
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "archive"


def _zip_cache_root(config: AppConfig, zip_path: Path, cache_key: str) -> Path:
    return config.historical_cache_dir / _safe_stem(zip_path) / cache_key


def _zip_cache_paths(cache_root: Path) -> dict[str, Path]:
    return {
        "player-directory": cache_root / "player-directory.parquet",
        "actual-game-results": cache_root / "actual-game-results.parquet",
        "actual-team-game-results": cache_root / "actual-team-game-results.parquet",
        "actual-hitter-stats-by-game": cache_root / "actual-hitter-stats-by-game.parquet",
        "actual-pitcher-stats-by-game": cache_root / "actual-pitcher-stats-by-game.parquet",
        "manifest": cache_root / "manifest.json",
    }


def _cache_ready(cache_paths: dict[str, Path], cache_key: str) -> bool:
    manifest = read_json(cache_paths["manifest"])
    if manifest is None or manifest.get("cache_key") != cache_key:
        return False
    return all(cache_paths[name].exists() for name in SLICE_NAMES)


def _write_zip_cache(zip_path: Path, config: AppConfig, force_rebuild: bool = False) -> dict[str, Path]:
    signature = build_file_signature([zip_path])
    cache_key = signature_hash(signature)
    cache_root = _zip_cache_root(config, zip_path, cache_key)
    cache_paths = _zip_cache_paths(cache_root)

    if not force_rebuild and _cache_ready(cache_paths, cache_key):
        return cache_paths

    (
        player_directory,
        actual_game_results,
        actual_team_game_results,
        actual_hitter_stats_by_game,
        actual_pitcher_stats_by_game,
    ) = build_historical_betting_data([zip_path])

    ensure_dir(cache_root)
    save_table(player_directory, cache_paths["player-directory"])
    save_table(actual_game_results, cache_paths["actual-game-results"])
    save_table(actual_team_game_results, cache_paths["actual-team-game-results"])
    save_table(actual_hitter_stats_by_game, cache_paths["actual-hitter-stats-by-game"])
    save_table(actual_pitcher_stats_by_game, cache_paths["actual-pitcher-stats-by-game"])

    write_json(
        cache_paths["manifest"],
        {
            "cache_key": cache_key,
            "zip_file": str(zip_path.resolve()),
            "source_signature": signature,
            "row_counts": {
                "player-directory": int(len(player_directory)),
                "actual-game-results": int(len(actual_game_results)),
                "actual-team-game-results": int(len(actual_team_game_results)),
                "actual-hitter-stats-by-game": int(len(actual_hitter_stats_by_game)),
                "actual-pitcher-stats-by-game": int(len(actual_pitcher_stats_by_game)),
            },
        },
    )
    return cache_paths


def _concat_frames(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in paths if path.exists()]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _dedupe_outputs(
    player_directory: pd.DataFrame,
    actual_game_results: pd.DataFrame,
    actual_team_game_results: pd.DataFrame,
    actual_hitter_stats_by_game: pd.DataFrame,
    actual_pitcher_stats_by_game: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not player_directory.empty:
        player_directory = player_directory.drop_duplicates()

    if not actual_game_results.empty and "game id" in actual_game_results.columns:
        actual_game_results = actual_game_results.drop_duplicates(subset=["game id"])

    if not actual_team_game_results.empty and {"game id", "team id"}.issubset(actual_team_game_results.columns):
        actual_team_game_results = actual_team_game_results.drop_duplicates(subset=["game id", "team id"])

    if not actual_hitter_stats_by_game.empty and {"game id", "player id"}.issubset(actual_hitter_stats_by_game.columns):
        actual_hitter_stats_by_game = actual_hitter_stats_by_game.drop_duplicates(subset=["game id", "player id"])

    if not actual_pitcher_stats_by_game.empty and {"game id", "player id", "team id"}.issubset(actual_pitcher_stats_by_game.columns):
        actual_pitcher_stats_by_game = actual_pitcher_stats_by_game.drop_duplicates(subset=["game id", "player id", "team id"])

    return (
        player_directory,
        actual_game_results,
        actual_team_game_results,
        actual_hitter_stats_by_game,
        actual_pitcher_stats_by_game,
    )


def build_historical_betting_data_artifacts(
    zip_dir: Path | None,
    zip_glob: str | None,
    config: AppConfig,
    force_rebuild: bool = False,
) -> dict[str, Path]:
    zip_paths = list_zip_paths(zip_dir, zip_glob)
    if not zip_paths:
        raise ValueError("No zip files found for parsing.")

    slice_paths = [_write_zip_cache(zip_path, config, force_rebuild=force_rebuild) for zip_path in zip_paths]

    player_directory = _concat_frames([item["player-directory"] for item in slice_paths])
    actual_game_results = _concat_frames([item["actual-game-results"] for item in slice_paths])
    actual_team_game_results = _concat_frames([item["actual-team-game-results"] for item in slice_paths])
    actual_hitter_stats_by_game = _concat_frames([item["actual-hitter-stats-by-game"] for item in slice_paths])
    actual_pitcher_stats_by_game = _concat_frames([item["actual-pitcher-stats-by-game"] for item in slice_paths])

    (
        player_directory,
        actual_game_results,
        actual_team_game_results,
        actual_hitter_stats_by_game,
        actual_pitcher_stats_by_game,
    ) = _dedupe_outputs(
        player_directory,
        actual_game_results,
        actual_team_game_results,
        actual_hitter_stats_by_game,
        actual_pitcher_stats_by_game,
    )

    outputs = {
        "player-directory": save_table(player_directory, config.player_directory),
        "actual-game-results": save_table(actual_game_results, config.actual_game_results),
        "actual-team-game-results": save_table(actual_team_game_results, config.actual_team_game_results),
        "actual-hitter-stats-by-game": save_table(actual_hitter_stats_by_game, config.actual_hitter_stats_by_game),
        "actual-pitcher-stats-by-game": save_table(actual_pitcher_stats_by_game, config.actual_pitcher_stats_by_game),
        "historical-cache": config.historical_cache_dir,
    }
    return outputs
