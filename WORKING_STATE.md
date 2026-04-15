# MLB Working State

This repo contains the current working MLB live-market workflow.

## Main flow

1. Export neutral model projections
2. Build `mlb_live_projections.csv`
3. Score the live Unabated board
4. Save latest outputs and dated copies

## Daily commands

### 1) Export neutral projections
```powershell
py ".\\export_mlb_model_projections.py" --player-stack-input "artifacts\\draftkings full stack projected player props.csv" --game-stack-input "artifacts\\draftkings full stack projected game lines.csv" --player-out "artifacts\\mlb_model_player_projections.csv" --game-out "artifacts\\mlb_model_game_projections.csv"
```

### 2) Build live projections
```powershell
py ".\\build_mlb_live_projections.py" --player-input "artifacts\\mlb_model_player_projections.csv" --game-input "artifacts\\mlb_model_game_projections.csv" --out "artifacts\\mlb_live_projections.csv"
```

### 3) Run nightly live board
```powershell
py ".\\run_mlb_unabated_nightly.py" --unabated-props-input "artifacts\\unabated_propodds_raw.json" --unabated-game-input "artifacts\\unabated_gameodds_raw.json" --out-dir "artifacts" --iterations 50000
```

## Current known-good outputs

- `artifacts\\mlb_live_projections.csv`
- `artifacts\\mlb_live_market_hub.csv`
- `artifacts\\mlb_live_scored_board.csv`
- `artifacts\\mlb_live_player_board.csv`
- `artifacts\\mlb_live_team_board.csv`
- `artifacts\\mlb_live_game_board.csv`
- `artifacts\\mlb_live_score_status_summary.csv`
- `artifacts\\mlb_live_market_family_summary.csv`
- `artifacts\\mlb_tonight_bet_card.csv`
- `artifacts\\mlb_tonight_bet_card_best_book.csv`

## Notes

- Unabated is the live odds source.
- Neutral model projections are built separately, then attached to the live board.
- The nightly runner preserves the normal latest filenames and also creates dated copies for the current day.
