from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

import pandas as pd

UNIFIED_COLUMNS = [
    "book",
    "sport",
    "event_id",
    "market_id",
    "selection_id",
    "fetched_at",
    "market_name",
    "market_family",
    "scope",
    "segment",
    "is_milestone",
    "milestone_value",
    "line",
    "selection",
    "participant",
    "team_name",
    "home_team",
    "away_team",
    "price_american",
    "implied_prob",
    "fair_prob",
    "market_group_key",
    "outcome_key",
    "projection_mean",
    "projection_prob",
    "model_prob",
    "edge_pct_points",
    "score_status",
]

SEGMENT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b1st\s*inning\b|\bfirst\s*inning\b|\bnrfi\b|\byrfi\b", re.I), "inning_1"),
    (re.compile(r"\b2nd\s*inning\b|\bsecond\s*inning\b", re.I), "inning_2"),
    (re.compile(r"\b3rd\s*inning\b|\bthird\s*inning\b", re.I), "inning_3"),
    (re.compile(r"\b4th\s*inning\b|\bfourth\s*inning\b", re.I), "inning_4"),
    (re.compile(r"\b5th\s*inning\b|\bfifth\s*inning\b", re.I), "inning_5"),
    (re.compile(r"\b6th\s*inning\b|\bsixth\s*inning\b", re.I), "inning_6"),
    (re.compile(r"\b7th\s*inning\b|\bseventh\s*inning\b", re.I), "inning_7"),
    (re.compile(r"\b8th\s*inning\b|\beighth\s*inning\b", re.I), "inning_8"),
    (re.compile(r"\b9th\s*inning\b|\bninth\s*inning\b", re.I), "inning_9"),
    (re.compile(r"\bfirst\s*5\s*innings\b|\b1st\s*5\s*innings\b|\bf5\b", re.I), "innings_1_5"),
    (re.compile(r"\binnings\b", re.I), "innings"),
]

FAMILY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bmoneyline\b|\bto win\b", re.I), "moneyline"),
    (re.compile(r"\brun line\b|\bspread\b|\balt run line\b", re.I), "spread"),
    (re.compile(r"\btotal runs\b|\bgame total\b|\bover/under total runs\b", re.I), "game_total_runs"),
    (re.compile(r"\bteam total runs\b|\bteam runs\b", re.I), "team_total_runs"),
    (re.compile(r"\bteam total hits\b|\bteam hits\b", re.I), "team_hits"),
    (re.compile(r"\bhits\s*o/?u\b|\bhits over under\b|\bhits\b", re.I), "hits"),
    (re.compile(r"\btotal bases\b", re.I), "total_bases"),
    (re.compile(r"\bruns scored\b|\bruns\s*o/?u\b|\bruns\b", re.I), "runs"),
    (re.compile(r"\brbi\b|\brbis\b|\bruns batted in\b", re.I), "rbis"),
    (re.compile(r"\bhits\s*[+ ]\s*runs\s*[+ ]\s*rbis\b|\bhits runs rbis\b", re.I), "hits_runs_rbis"),
    (re.compile(r"\bhome runs\b|\bhome run\b", re.I), "home_runs"),
    (re.compile(r"\bstrikeouts thrown\b|\bpitcher strikeouts\b|\bstrikeouts\b", re.I), "pitcher_strikeouts"),
    (re.compile(r"\bouts recorded\b|\bpitching outs\b", re.I), "pitching_outs"),
    (re.compile(r"\bearned runs allowed\b|\bearned runs\b", re.I), "earned_runs"),
    (re.compile(r"\bhits allowed\b", re.I), "hits_allowed"),
    (re.compile(r"\bwalks allowed\b", re.I), "walks_allowed"),
    (re.compile(r"\bstolen bases\b", re.I), "stolen_bases"),
    (re.compile(r"\bsingles\b", re.I), "singles"),
    (re.compile(r"\bdoubles\b", re.I), "doubles"),
    (re.compile(r"\btriples\b", re.I), "triples"),
    (re.compile(r"\bto score in the first inning\b|\bnrfi\b|\byrfi\b", re.I), "inning_1_runs"),
]

TEAM_HINT_PATTERNS = re.compile(
    r"\b(team total|team runs|team hits|moneyline|run line|total runs|nrfi|yrfi|first inning|game total)\b",
    re.I,
)

MILESTONE_IN_NAME = re.compile(r"(?:\b(\d+)\+\b)|(?:at least\s+(\d+))", re.I)


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    for token in ["_", "/", "(", ")", ","]:
        text = text.replace(token, " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_american_to_prob(odds: Any) -> float | None:
    if odds in (None, ""):
        return None
    text = str(odds).strip().replace("−", "-")
    try:
        value = float(text)
    except ValueError:
        return None
    if value > 0:
        return 100.0 / (value + 100.0)
    if value < 0:
        return (-value) / ((-value) + 100.0)
    return None


def parse_line(value: Any) -> float | None:
    if value in (None, ""):
        return None
    text = str(value).strip().replace("−", "-")
    m = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def detect_segment(market_name: str) -> str:
    for pattern, segment in SEGMENT_PATTERNS:
        if pattern.search(market_name or ""):
            return segment
    return "full_game"


def detect_market_family(market_name: str) -> str:
    for pattern, family in FAMILY_PATTERNS:
        if pattern.search(market_name or ""):
            return family
    return norm_text(market_name).replace(" ", "_") or "unknown"


def detect_milestone(market_name: str, selection: str, line: float | None) -> tuple[bool, float | None]:
    selection_norm = norm_text(selection)
    if selection_norm in {"over", "under", "home", "away", "yes", "no"}:
        if line is not None and abs(line - round(line)) < 1e-9 and line >= 2 and selection_norm in {"yes", "no"}:
            return True, float(line)
        return False, None
    match = MILESTONE_IN_NAME.search(market_name or "")
    if match:
        raw = match.group(1) or match.group(2)
        if raw:
            return True, float(raw)
    if line is not None and selection_norm not in {"over", "under"} and line >= 2:
        return True, float(line)
    return False, None


def detect_scope(market_name: str, participant: str, team_name: str, selection: str) -> str:
    market_text = norm_text(market_name)
    if participant:
        return "player"
    if team_name:
        return "team"
    if TEAM_HINT_PATTERNS.search(market_text):
        if "team" in market_text:
            return "team"
        return "game"
    selection_text = norm_text(selection)
    if selection_text in {"home", "away"}:
        return "game"
    return "game"



def infer_entity_from_market_name(market_name: str, market_family: str) -> tuple[str, str]:
    raw = str(market_name or '').strip()
    if not raw:
        return '', ''
    player_keywords = [
        ' Hits O/U', ' Hits', ' Total Bases', ' Runs O/U', ' Runs', ' RBI O/U', ' RBIs',
        ' Home Runs', ' Strikeouts', ' Outs Recorded', ' Earned Runs', ' Hits Allowed',
        ' Walks Allowed', ' Stolen Bases', ' Singles', ' Doubles', ' Triples'
    ]
    team_keywords = [' Team Total Runs', ' Team Runs', ' Team Total Hits', ' Team Hits']
    for token in team_keywords:
        if token.lower() in raw.lower():
            idx = raw.lower().find(token.lower())
            return '', raw[:idx].strip()
    if market_family in {'moneyline', 'spread', 'game_total_runs', 'inning_1_runs'}:
        return '', ''
    for token in player_keywords:
        if token.lower() in raw.lower():
            idx = raw.lower().find(token.lower())
            return raw[:idx].strip(), ''
    return '', ''

def canonical_team(team_name: str) -> str:
    return norm_text(team_name).replace(" ", "_")


def safe_str(value: Any) -> str:
    return "" if value is None else str(value).strip()


def build_market_group_key(
    *,
    event_id: str,
    scope: str,
    segment: str,
    market_family: str,
    participant: str,
    team_name: str,
    line: float | None,
    is_milestone: bool,
    milestone_value: float | None,
) -> str:
    threshold = milestone_value if is_milestone else line
    threshold_text = "" if threshold is None or (isinstance(threshold, float) and math.isnan(threshold)) else f"{threshold:g}"
    entity = participant or team_name or "game"
    return "|".join(
        [
            safe_str(event_id),
            safe_str(scope),
            safe_str(segment),
            safe_str(market_family),
            canonical_team(entity),
            "milestone" if is_milestone else "ou_or_side",
            threshold_text,
        ]
    )


def build_outcome_key(market_group_key: str, selection: str) -> str:
    return f"{market_group_key}|{norm_text(selection)}"


def ensure_unified_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in UNIFIED_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[UNIFIED_COLUMNS].copy()
