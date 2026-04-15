import subprocess
import sys
import pathlib
from datetime import datetime

ROOT = pathlib.Path(r"C:\Users\User\Desktop\mlb_model_clean")

# Replace these filenames with your real scripts.
STEPS = [
    ROOT / "run_mlb_lineups_refresh.py",
    ROOT / "run_mlb_pitcher_refresh.py",
    ROOT / "run_mlb_weather_refresh.py",
    ROOT / "run_mlb_model_refresh.py",
    ROOT / "run_mlb_edges_refresh.py",
    ROOT / "run_mlb_health_check.py",
]

print(f"[{datetime.now().isoformat(timespec='seconds')}] Starting MLB pregame pipeline")

for step in STEPS:
    if not step.exists():
        raise FileNotFoundError(f"Missing step script: {step}")

    print(f"Running: {step.name}")
    subprocess.run([sys.executable, str(step)], check=True, cwd=ROOT)

print(f"[{datetime.now().isoformat(timespec='seconds')}] MLB pregame pipeline finished")