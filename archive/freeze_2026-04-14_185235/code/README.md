
MLB daily fix package

What this fixes
• Reads the real column names from your full-stack DraftKings CSVs
• Writes a clean mlb_live_projections.csv file
• Matches side markets by selection so moneyline/spread can attach the right probabilities
• Uses easy names where possible:
  • game_moneyline → moneyline
  • game_spread → spread
  • tb_ou → total_bases

Daily use

1) Put both files in:
   C:\Users\User\Desktop\mlb

2) Build projections:
   py ".\build_mlb_live_projections.py" --player-props "artifacts\draftkings full stack projected player props.csv" --game-lines "artifacts\draftkings full stack projected game lines.csv" --out "artifacts\mlb_live_projections.csv"

3) Rerun the Unabated board:
   py ".\run_mlb_live_board.py" --unabated-props-input "artifacts\unabated_propodds_raw.json" --unabated-game-input "artifacts\unabated_gameodds_raw.json" --projection-file "artifacts\mlb_live_projections.csv" --out-dir "artifacts" --iterations 50000

4) Check:
   Import-Csv ".\artifacts\mlb_live_score_status_summary.csv"
   Import-Csv ".\artifacts\mlb_live_market_family_summary.csv" | Select-Object -First 30
