# MLB Full-Stack Market Engine

This engine is built to project and price **every confirmed DraftKings MLB market family** you mapped, without double-counting the same information from multiple angles.

## What it does
- projects every market on `draftkings mlb api all markets.csv`
- uses your existing history-trained `.joblib` models where a direct family model exists
- adds **Bayesian shrinkage** on top of base projections
- supports **XGBoost** and **LightGBM** residual calibrators when you train them on labeled history
- uses **Monte Carlo simulation** for pricing O/U, milestones, game lines, team totals, and first-inning markets
- computes **Pythagorean win expectancy** from projected home/away runs for game-context sanity

## No-double-counting rule
The engine keeps these in separate lanes:
- structural model signal
- market anchor signal
- boosted residual signal
- Bayesian shrinkage

It does **not** add raw line + raw odds + fair prob + derived market mean as separate pieces of evidence. Those collapse into a single `market anchor mean`.

## Confirmed market families covered
### Game and team
- Moneyline / Run Line / Game Total bundle
- Team Total Runs
- Alternate Team Total Runs
- Runs - 1st Inning

### Batter O/U
- Hits O/U
- Total Bases O/U
- RBIs O/U
- Runs O/U
- Hits + Runs + RBIs O/U
- Singles O/U
- Doubles O/U

### Pitcher O/U
- Strikeouts Thrown O/U
- Outs O/U
- Hits Allowed O/U
- Earned Runs Allowed O/U

### Milestones
- Home Runs
- Hits
- Strikeouts Thrown
- Batter Strikeouts
- Batter Walks
- Triples

## Run it
```powershell
cd "C:\Users\User\Desktop\mlb_model_clean"

py ".\mlb_market_full_stack_engine.py" ^
  --board-file "artifacts\draftkings mlb api all markets.csv" ^
  --out-dir "artifacts" ^
  --player-form-file "artifacts\latest player form.parquet" ^
  --team-form-file "artifacts\latest team form.parquet" ^
  --model-dir "artifacts\history trained models" ^
  --iterations 50000
```

## Outputs
- `draftkings full stack projected all markets.csv`
- `draftkings full stack projected player props.csv`
- `draftkings full stack projected game lines.csv`
- `draftkings full stack best edges.csv`
- `draftkings full stack summary.csv`

## Train XGBoost / LightGBM / Bayesian residual calibrators
You need a labeled history file with at least:
- `market family`
- `actual value`
- feature columns

Then run:
```powershell
py ".\mlb_market_full_stack_engine.py" ^
  --fit-calibrators-from-history "artifacts\market_history_training.csv" ^
  --calibrator-dir "artifacts\market_calibrators"
```

That writes per-family calibrators like:
- `hits_ou_xgb_reg.joblib`
- `hits_ou_lgbm_reg.joblib`
- `hits_ou_bayes_reg.joblib`

## Important note
This engine is built so the live run works **today** with:
- joblib base models
- Bayesian shrinkage
- Monte Carlo
- Pythagorean game context

And the XGBoost / LightGBM layer plugs in **cleanly** once you give it labeled history, instead of fake-training on today’s odds.
