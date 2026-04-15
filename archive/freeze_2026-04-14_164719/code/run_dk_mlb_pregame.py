import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(r"C:\Users\User\Desktop\mlb_model_clean")
DK_PATH = ROOT / "mlb_betting_data" / "sportsbooks" / "draftkings.py"
OUT_PATH = ROOT / "artifacts" / "draftkings_mlb_pregame_live_odds.csv"

spec = importlib.util.spec_from_file_location("dk", DK_PATH)
dk = importlib.util.module_from_spec(spec)
sys.modules["dk"] = dk
spec.loader.exec_module(dk)

payloads, meta = dk.capture_draftkings_payloads(sport="mlb", use_browser=True)
df = dk.normalize_draftkings_payloads(payloads, sport_hint="mlb")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)

print(df.shape)
print(meta)
print(OUT_PATH)
