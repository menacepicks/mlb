from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd

BET_CARD_COLUMNS = [
    'rank', 'market_family', 'scope', 'segment', 'event_id', 'book', 'market', 'player', 'team',
    'line', 'side', 'market_odds', 'fair_prob', 'fair_american', 'model_prob', 'edge_pct_points', 'confidence', 'note',
]

FAMILY_MIN_EDGE = {
    'moneyline': 2.0,
    'spread': 2.0,
    'game_total_runs': 2.0,
    'team_total_runs': 2.0,
    'pitcher_strikeouts': 3.0,
    'pitching_outs': 3.0,
    'earned_runs': 3.0,
    'hits_allowed': 3.0,
    'walks_allowed': 3.0,
    'hits': 3.0,
    'total_bases': 3.0,
    'runs': 3.0,
    'rbis': 3.0,
    'hits_runs_rbis': 3.0,
    'doubles': 3.5,
    'triples': 5.0,
    'stolen_bases': 4.0,
    'home_runs': 5.0,
    'inning_1_runs': 2.5,
}

VOLATILE_FAMILIES = {'home_runs', 'triples', 'stolen_bases'}


def _load_table(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    if p.suffix.lower() == '.csv':
        return pd.read_csv(p)
    if p.suffix.lower() in {'.parquet', '.pq'}:
        return pd.read_parquet(p)
    raise ValueError(f'Unsupported file: {p}')


def _safe_float(v: Any) -> float | None:
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return float(v)
    except Exception:
        return None


def _fair_american(prob: float | None) -> int | None:
    if prob is None:
        return None
    p = min(max(float(prob), 1e-9), 1 - 1e-9)
    if p >= 0.5:
        return int(round(-(100.0 * p) / (1.0 - p)))
    return int(round((100.0 * (1.0 - p)) / p))


def _canonical_team(value: Any) -> str:
    text = str(value or '').strip().lower().replace('.', '').replace('_', ' ')
    return ''.join(text.split())


def _event_key_from_cols(df: pd.DataFrame) -> pd.Series:
    home = df.get('home_team', pd.Series([''] * len(df), index=df.index)).map(_canonical_team)
    away = df.get('away_team', pd.Series([''] * len(df), index=df.index)).map(_canonical_team)
    return away + '|' + home


def _merge_weather(card: pd.DataFrame, weather: pd.DataFrame | None) -> pd.DataFrame:
    if weather is None or weather.empty:
        return card
    wx = weather.copy()
    wx.columns = [str(c).strip().lower() for c in wx.columns]
    if 'event_id' not in wx.columns:
        wx['event_id'] = ''
    if 'event_key' not in wx.columns:
        wx['event_key'] = ''
    if ('home_team' in wx.columns or 'away_team' in wx.columns) and wx['event_key'].astype(str).str.len().eq(0).any():
        wx['event_key'] = _event_key_from_cols(wx)
    wind_col = next((c for c in wx.columns if 'wind' in c), None)
    temp_col = next((c for c in wx.columns if 'temp' in c), None)
    keep = ['event_id', 'event_key'] + [c for c in [wind_col, temp_col] if c]
    keep = [c for c in keep if c in wx.columns]
    wx = wx[keep].drop_duplicates()
    out = card.merge(wx, on=['event_id'], how='left', suffixes=('', '_wx_id'))
    if 'event_key' in wx.columns:
        miss = out[wind_col].isna() if wind_col else out[temp_col].isna()
        fill = card.merge(wx, on=['event_key'], how='left', suffixes=('', '_wx_key'))
        for c in [wind_col, temp_col]:
            if c and c in out.columns and c in fill.columns:
                out.loc[miss, c] = fill.loc[miss, c]
    return out


def _merge_volatility(card: pd.DataFrame, volatility: pd.DataFrame | None) -> pd.DataFrame:
    if volatility is None or volatility.empty:
        return card
    vol = volatility.copy()
    vol.columns = [str(c).strip().lower() for c in vol.columns]
    score_col = next((c for c in vol.columns if 'volatility' in c or 'std' in c), None)
    if not score_col:
        return card
    player_col = next((c for c in vol.columns if c in {'player','participant','player_name','name'}), None)
    team_col = next((c for c in vol.columns if c in {'team','team_name'}), None)
    out = card.copy()
    if player_col:
        tmp = vol[[player_col, score_col]].dropna().drop_duplicates()
        tmp[player_col] = tmp[player_col].map(_canonical_team)
        out['_player_key'] = out['player'].map(_canonical_team)
        out = out.merge(tmp.rename(columns={player_col:'_player_key', score_col:'player_volatility'}), on='_player_key', how='left')
        out = out.drop(columns=['_player_key'])
    if team_col:
        tmp = vol[[team_col, score_col]].dropna().drop_duplicates()
        tmp[team_col] = tmp[team_col].map(_canonical_team)
        out['_team_key'] = out['team'].map(_canonical_team)
        out = out.merge(tmp.rename(columns={team_col:'_team_key', score_col:'team_volatility'}), on='_team_key', how='left')
        out = out.drop(columns=['_team_key'])
    return out


def _confidence(row: pd.Series) -> float:
    edge = max(_safe_float(row.get('edge_pct_points')) or 0.0, 0.0)
    fair = _safe_float(row.get('fair_prob')) or 0.0
    model = _safe_float(row.get('model_prob')) or 0.0
    status = str(row.get('score_status') or '')
    base = 50.0
    if status == 'scored_probability':
        base += 18.0
    elif status == 'scored_monte_carlo':
        base += 10.0
    base += min(edge, 15.0) * 1.6
    base += min(abs(model - fair) * 100.0, 12.0)
    if str(row.get('market_family') or '') in VOLATILE_FAMILIES:
        base -= 8.0
    line = _safe_float(row.get('line'))
    if str(row.get('market_family') or '') == 'game_total_runs' and str(row.get('segment') or '') == 'full_game' and line is not None and line < 5:
        base -= 40.0
    pvol = _safe_float(row.get('player_volatility')) or 0.0
    tvol = _safe_float(row.get('team_volatility')) or 0.0
    base -= min(max(pvol, tvol), 10.0) * 1.2
    return max(1.0, min(base, 99.0))


def _note(row: pd.Series) -> str:
    notes: list[str] = []
    if str(row.get('score_status') or '') == 'scored_probability':
        notes.append('Model probability')
    elif str(row.get('score_status') or '') == 'scored_monte_carlo':
        notes.append('Monte Carlo')
    else:
        notes.append('Market only')
    if str(row.get('market_family') or '') in VOLATILE_FAMILIES:
        notes.append('High variance')
    if str(row.get('scope') or '') == 'team':
        notes.append('Team market')
    if str(row.get('scope') or '') == 'game':
        notes.append('Game market')
    wind = _safe_float(row.get('wind_mph')) or _safe_float(row.get('wind'))
    temp = _safe_float(row.get('temp_f')) or _safe_float(row.get('temperature'))
    if wind is not None and wind >= 12 and str(row.get('market_family') or '') in {'game_total_runs','team_total_runs','home_runs'}:
        notes.append(f'Wind {wind:.0f} mph')
    if temp is not None and temp <= 45:
        notes.append(f'Temp {temp:.0f}F')
    return ' | '.join(notes)


def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # kill obviously bad/trivial markets
    line = pd.to_numeric(out.get('line'), errors='coerce')
    fam = out.get('market_family', pd.Series('', index=out.index)).astype(str)
    seg = out.get('segment', pd.Series('', index=out.index)).astype(str)
    out = out.loc[~((fam == 'game_total_runs') & (seg == 'full_game') & line.notna() & (line < 3.0))].copy()
    out = out.loc[~((fam == 'team_total_runs') & line.notna() & (line < 1.0))].copy()
    out = out.loc[~((fam == 'moneyline') & out.get('selection').astype(str).str.lower().isin(['over', 'under']))].copy()
    # keep only book offers that actually have prices
    out = out.loc[pd.to_numeric(out.get('price_american'), errors='coerce').notna()].copy()
    # de-dupe exact market rows
    dedupe_cols = ['event_id', 'market_family', 'scope', 'segment', 'participant', 'team_name', 'selection', 'line', 'book', 'price_american']
    dedupe_cols = [c for c in dedupe_cols if c in out.columns]
    out = out.drop_duplicates(subset=dedupe_cols, keep='first').copy()
    return out


def _apply_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['edge_pct_points'] = pd.to_numeric(out['edge_pct_points'], errors='coerce')
    out['fair_prob'] = pd.to_numeric(out['fair_prob'], errors='coerce')
    out['model_prob'] = pd.to_numeric(out['model_prob'], errors='coerce')
    out = out.loc[out['score_status'].isin(['scored_probability', 'scored_monte_carlo'])].copy()
    out = out.loc[out['edge_pct_points'].between(1.5, 25.0, inclusive='both')].copy()
    mins = out['market_family'].map(lambda f: FAMILY_MIN_EDGE.get(str(f), 3.5))
    out = out.loc[out['edge_pct_points'] >= mins].copy()
    return out


def _best_per_market(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    sort_cols = ['event_id', 'market_family', 'scope', 'segment', 'participant', 'team_name', 'selection', 'line']
    sort_cols = [c for c in sort_cols if c in out.columns]
    out = out.sort_values(sort_cols + ['edge_pct_points'], ascending=[True] * len(sort_cols) + [False])
    out = out.drop_duplicates(subset=sort_cols, keep='first').copy()
    return out


def build_bet_card(scored: pd.DataFrame, volatility: pd.DataFrame | None = None, weather: pd.DataFrame | None = None) -> pd.DataFrame:
    out = _sanitize(scored)
    out = _apply_thresholds(out)
    out = _best_per_market(out)
    out = _merge_weather(out, weather)
    out = _merge_volatility(out, volatility)
    if out.empty:
        return pd.DataFrame(columns=BET_CARD_COLUMNS)

    out['confidence'] = out.apply(_confidence, axis=1).round(1)
    out['note'] = out.apply(_note, axis=1)
    out['fair_american'] = out['model_prob'].map(_fair_american)
    out['market'] = out['market_family'].astype(str)
    out['player'] = out['participant'].fillna('')
    out['team'] = out['team_name'].fillna('')
    out['side'] = out['selection'].astype(str).str.lower()
    out['market_odds'] = pd.to_numeric(out['price_american'], errors='coerce').round(0).astype('Int64')
    out['fair_prob'] = pd.to_numeric(out['fair_prob'], errors='coerce').round(4)
    out['model_prob'] = pd.to_numeric(out['model_prob'], errors='coerce').round(4)
    out['edge_pct_points'] = pd.to_numeric(out['edge_pct_points'], errors='coerce').round(2)
    out['line'] = pd.to_numeric(out['line'], errors='coerce').round(2)

    out = out.sort_values(['confidence', 'edge_pct_points'], ascending=[False, False]).reset_index(drop=True)
    out['rank'] = range(1, len(out) + 1)
    return out[BET_CARD_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a clean daily MLB bet card from the scored live board')
    parser.add_argument('--scored-board', required=True)
    parser.add_argument('--out-csv', default='artifacts/mlb_daily_bet_card.csv')
    parser.add_argument('--out-xlsx', default='artifacts/mlb_daily_bet_card.xlsx')
    parser.add_argument('--volatility-input', default=None)
    parser.add_argument('--weather-input', default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scored = _load_table(args.scored_board)
    vol = _load_table(args.volatility_input)
    weather = _load_table(args.weather_input)
    card = build_bet_card(scored, vol, weather)
    csv_path = Path(args.out_csv)
    xlsx_path = Path(args.out_xlsx)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    card.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        card.to_excel(writer, index=False, sheet_name='Bet Card')
    print(f'bet_card_rows={len(card)}')
    print(csv_path)
    print(xlsx_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
