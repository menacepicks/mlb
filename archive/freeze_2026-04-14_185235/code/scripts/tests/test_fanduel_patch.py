
from pathlib import Path
import shutil
import pandas as pd
import numpy as np

from mlb_betting_data.sportsbooks.fanduel import normalize_fanduel_payloads
from mlb_betting_data.config import AppConfig
from mlb_betting_data.io import save_table
from mlb_betting_data.pipelines.compare_books import compare_mlb_books

payload = {
    "__fd_captured_at": "2026-04-14T00:00:00Z",
    "attachments": {
        "events": {
            "soc1": {"eventTypeId": "999999", "name": "Wolves @ Leeds", "openDate": "2026-04-15T00:00:00Z"},
            "mlb1": {"eventTypeId": "7511", "name": "Boston Red Sox @ New York Yankees", "openDate": "2026-04-15T00:00:00Z"},
        },
        "markets": {
            "m1": {"eventId":"soc1","eventTypeId":"999999","marketName":"Moneyline (3 Way)","marketType":"MONEY_LINE","marketStatus":"OPEN","runners":[
                {"runnerName":"Wolves","winRunnerOdds":{"americanDisplayOdds":{"americanOddsInt":-110}},"result":{"type":"HOME"},"handicap":0},
                {"runnerName":"Draw","winRunnerOdds":{"americanDisplayOdds":{"americanOddsInt":250}},"result":{"type":"DRAW"},"handicap":0},
                {"runnerName":"Leeds","winRunnerOdds":{"americanDisplayOdds":{"americanOddsInt":180}},"result":{"type":"AWAY"},"handicap":0},
            ]},
            "m2": {"eventId":"mlb1","eventTypeId":"7511","marketName":"Money Line","marketType":"MONEY_LINE","marketStatus":"OPEN","runners":[
                {"runnerName":"Boston Red Sox","winRunnerOdds":{"americanDisplayOdds":{"americanOddsInt":130}},"result":{"type":"AWAY"}},
                {"runnerName":"New York Yankees","winRunnerOdds":{"americanDisplayOdds":{"americanOddsInt":-150}},"result":{"type":"HOME"}},
            ]},
            "m3": {"eventId":"mlb1","eventTypeId":"7511","marketName":"Total Runs","marketType":"TOTAL_RUNS","marketStatus":"OPEN","runners":[
                {"runnerName":"Over","winRunnerOdds":{"americanDisplayOdds":{"americanOddsInt":-105}},"result":{"type":"OVER"},"handicap":8.5},
                {"runnerName":"Under","winRunnerOdds":{"americanDisplayOdds":{"americanOddsInt":-115}},"result":{"type":"UNDER"},"handicap":8.5},
            ]}
        }
    },
    "layout": {}
}
df = normalize_fanduel_payloads([payload], sport_hint="mlb")
assert len(df) == 4, df
assert set(df["sport"]) == {"mlb"}
assert not df["home_team"].astype(str).str.contains("Wolves|Leeds", case=False, regex=True).any()

tmp = Path("test_artifacts_compare")
if tmp.exists():
    shutil.rmtree(tmp)
tmp.mkdir()
cfg = AppConfig(out_dir=tmp)
dk = pd.DataFrame([
    {"sport":"MLB","event id":"dk1","market":"moneyline","selection":"home","line":np.nan,"player":"","team":"NY Yankees","home team":"NY Yankees","away team":"BOS Red Sox","book":"DraftKings","american odds":-145,"fetched at":"2026-04-14T00:00:00Z"},
    {"sport":"MLB","event id":"dk1","market":"moneyline","selection":"away","line":np.nan,"player":"","team":"BOS Red Sox","home team":"NY Yankees","away team":"BOS Red Sox","book":"DraftKings","american odds":125,"fetched at":"2026-04-14T00:00:00Z"},
])
fd = pd.DataFrame([
    {"sport":"MLB","event id":"fd9","market":"moneyline","selection":"home","line":np.nan,"player":"","team":"New York Yankees","home team":"New York Yankees","away team":"Boston Red Sox","book":"FanDuel","american odds":-140,"fetched at":"2026-04-14T00:00:00Z"},
    {"sport":"MLB","event id":"fd9","market":"moneyline","selection":"away","line":np.nan,"player":"","team":"Boston Red Sox","home team":"New York Yankees","away team":"Boston Red Sox","book":"FanDuel","american odds":130,"fetched at":"2026-04-14T00:00:00Z"},
])
save_table(dk, cfg.draftkings_mlb_lines)
save_table(fd, cfg.fanduel_mlb_lines)
compare_mlb_books(cfg)
best = pd.read_csv(cfg.mlb_best_price_by_market_csv)
assert len(best) == 2, best
assert set(best["books found"]) == {2}, best
assert set(best["best book"]) == {"FanDuel"}, best
print("ok")
