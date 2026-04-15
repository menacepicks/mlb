from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AppConfig:
    out_dir: Path

    @property
    def cache_dir(self) -> Path:
        return self.out_dir / "cache"

    @property
    def historical_cache_dir(self) -> Path:
        return self.cache_dir / "historical"

    @property
    def model_dir(self) -> Path:
        return self.out_dir / "history trained models"

    @property
    def archive_dir(self) -> Path:
        return self.out_dir / "archive"

    @property
    def draftkings_archive_dir(self) -> Path:
        return self.archive_dir / "draftkings"

    @property
    def fanduel_archive_dir(self) -> Path:
        return self.archive_dir / "fanduel"

    @property
    def draftkings_archive_history_parquet(self) -> Path:
        return self.out_dir / "draftkings mlb archive history.parquet"

    @property
    def draftkings_archive_history_csv(self) -> Path:
        return self.out_dir / "draftkings mlb archive history.csv"

    @property
    def fanduel_archive_history_parquet(self) -> Path:
        return self.out_dir / "fanduel mlb archive history.parquet"

    @property
    def fanduel_archive_history_csv(self) -> Path:
        return self.out_dir / "fanduel mlb archive history.csv"

    @property
    def player_directory(self) -> Path:
        return self.out_dir / "player directory.parquet"

    @property
    def actual_game_results(self) -> Path:
        return self.out_dir / "actual game results.parquet"

    @property
    def actual_team_game_results(self) -> Path:
        return self.out_dir / "actual team game results.parquet"

    @property
    def actual_hitter_stats_by_game(self) -> Path:
        return self.out_dir / "actual hitter stats by game.parquet"

    @property
    def actual_pitcher_stats_by_game(self) -> Path:
        return self.out_dir / "actual pitcher stats by game.parquet"

    @property
    def draftkings_mlb_lines(self) -> Path:
        return self.out_dir / "draftkings mlb lines.parquet"

    @property
    def draftkings_mlb_game_lines(self) -> Path:
        return self.out_dir / "draftkings mlb game lines.parquet"

    @property
    def draftkings_mlb_player_props(self) -> Path:
        return self.out_dir / "draftkings mlb player props.parquet"

    @property
    def draftkings_mlb_priced_board(self) -> Path:
        return self.out_dir / "draftkings mlb priced board.parquet"

    @property
    def draftkings_mlb_betting_report(self) -> Path:
        return self.out_dir / "draftkings mlb betting report.csv"

    @property
    def fanduel_mlb_lines(self) -> Path:
        return self.out_dir / "fanduel mlb lines.parquet"

    @property
    def fanduel_mlb_game_lines(self) -> Path:
        return self.out_dir / "fanduel mlb game lines.parquet"

    @property
    def fanduel_mlb_player_props(self) -> Path:
        return self.out_dir / "fanduel mlb player props.parquet"

    @property
    def fanduel_mlb_priced_board(self) -> Path:
        return self.out_dir / "fanduel mlb priced board.parquet"

    @property
    def fanduel_mlb_betting_report(self) -> Path:
        return self.out_dir / "fanduel mlb betting report.csv"

    @property
    def mlb_two_book_comparison_parquet(self) -> Path:
        return self.out_dir / "mlb two book comparison.parquet"

    @property
    def mlb_two_book_comparison_csv(self) -> Path:
        return self.out_dir / "mlb two book comparison.csv"

    @property
    def mlb_best_price_by_market_parquet(self) -> Path:
        return self.out_dir / "mlb best price by market.parquet"

    @property
    def mlb_best_price_by_market_csv(self) -> Path:
        return self.out_dir / "mlb best price by market.csv"

    @property
    def history_trained_model_summary(self) -> Path:
        return self.out_dir / "history trained model summary.csv"

    @property
    def latest_player_form(self) -> Path:
        return self.out_dir / "latest player form.parquet"

    @property
    def latest_team_form(self) -> Path:
        return self.out_dir / "latest team form.parquet"

    @property
    def player_prop_fair_prices(self) -> Path:
        return self.out_dir / "player prop fair prices.parquet"

    @property
    def game_fair_prices(self) -> Path:
        return self.out_dir / "game fair prices.parquet"

    @property
    def best_bets(self) -> Path:
        return self.out_dir / "best bets.csv"

    @property
    def game_best_bets(self) -> Path:
        return self.out_dir / "game best bets.csv"

    @property
    def player_and_team_volatility(self) -> Path:
        return self.out_dir / "player and team volatility.parquet"

    @property
    def same_game_relationship_guide(self) -> Path:
        return self.out_dir / "same game relationship guide.parquet"
