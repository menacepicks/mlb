from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import glob
import re
import zipfile

import pandas as pd


EVENT_FILE_RE = re.compile(r"\.(EV[AN]|EB[AN])$", re.IGNORECASE)
ROSTER_FILE_RE = re.compile(r"\.ROS$", re.IGNORECASE)
ADV_RE = re.compile(r"([B123])([\-X])([123H])")
SB_RE = re.compile(r"SB([23H])")
CS_RE = re.compile(r"(?:PO)?CS([23H])")


@dataclass(slots=True)
class GameState:
    game_id: str
    date: int
    doubleheader_number: int
    away_team_id: str
    home_team_id: str
    day_night: str | None
    ballpark_id: str | None
    current_pitcher: dict[int, str | None]
    starting_pitcher: dict[int, str | None]
    bases: dict[int, str | None]
    last_half: tuple[int, int] | None
    outs_in_half: int
    team_runs: dict[int, int]


def list_zip_paths(zip_dir: Path | None, zip_glob: str | None) -> list[Path]:
    paths: list[Path] = []
    if zip_dir is not None:
        paths.extend(sorted(zip_dir.glob("*.zip")))
    if zip_glob:
        paths.extend(sorted(Path(p) for p in glob.glob(zip_glob)))
    unique: dict[str, Path] = {}
    for path in paths:
        unique[str(path.resolve())] = path
    return list(unique.values())


def _parse_date(raw: str) -> int:
    return int(raw.replace("/", ""))


def _iter_games_from_event_lines(lines: list[str]) -> list[list[str]]:
    games: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.startswith("id,"):
            if current:
                games.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        games.append(current)
    return games


def _parse_roster_file(name: str, raw_bytes: bytes) -> list[dict[str, object]]:
    season_match = re.search(r"(\d{4})", name)
    season = int(season_match.group(1)) if season_match else None
    rows: list[dict[str, object]] = []
    for raw_line in raw_bytes.decode("latin1").splitlines():
        parts = raw_line.split(",")
        if len(parts) < 7:
            continue
        rows.append(
            {
                "player id": parts[0],
                "last name": parts[1],
                "first name": parts[2],
                "full name": f"{parts[2]} {parts[1]}".strip(),
                "bats": parts[3],
                "throws": parts[4],
                "team id": parts[5],
                "primary position": parts[6],
                "season": season,
            }
        )
    return rows


def _empty_hitter_row(player_id: str) -> dict[str, object]:
    return {
        "player id": player_id,
        "plate appearances": 0,
        "at bats": 0,
        "hits": 0,
        "singles": 0,
        "doubles": 0,
        "triples": 0,
        "home runs": 0,
        "total bases": 0,
        "walks": 0,
        "hit by pitch": 0,
        "strikeouts": 0,
        "runs scored": 0,
        "runs batted in": 0,
        "stolen bases": 0,
        "caught stealing": 0,
    }


def _empty_pitcher_row(player_id: str) -> dict[str, object]:
    return {
        "player id": player_id,
        "batters faced": 0,
        "outs recorded": 0,
        "innings pitched": 0.0,
        "hits allowed": 0,
        "walks allowed": 0,
        "hit by pitch allowed": 0,
        "strikeouts": 0,
        "home runs allowed": 0,
        "runs allowed": 0,
        "earned runs": None,
        "was starting pitcher": False,
        "got quality start": None,
        "got win": None,
    }


def _parse_play_event(event: str) -> dict[str, object]:
    main, *rest = event.split(".", 1)
    adv = rest[0] if rest else ""
    stats = {
        "plate appearance": 1,
        "at bat": 1,
        "hit": 0,
        "single": 0,
        "double": 0,
        "triple": 0,
        "home run": 0,
        "total bases": 0,
        "walk": 0,
        "hit by pitch": 0,
        "strikeout": 0,
        "outs on main play": 0,
        "default batter destination": None,
        "event main": main,
        "adv": adv,
    }

    if main == "NP":
        stats["plate appearance"] = 0
        stats["at bat"] = 0
        return stats

    if main in {"BK", "DI", "OA", "PB", "WP"} or main.startswith("PO") or main.startswith("CS") or main.startswith("SB"):
        stats["plate appearance"] = 0
        stats["at bat"] = 0
        return stats

    if main.startswith("IW") or main == "I" or main.startswith("W"):
        stats["at bat"] = 0
        stats["walk"] = 1
        stats["default batter destination"] = 1
    elif main.startswith("HP"):
        stats["at bat"] = 0
        stats["hit by pitch"] = 1
        stats["default batter destination"] = 1
    elif main.startswith("HR"):
        stats["hit"] = 1
        stats["home run"] = 1
        stats["total bases"] = 4
        stats["default batter destination"] = "H"
    elif main.startswith("T"):
        stats["hit"] = 1
        stats["triple"] = 1
        stats["total bases"] = 3
        stats["default batter destination"] = 3
    elif main.startswith("D") and not main.startswith("DI"):
        stats["hit"] = 1
        stats["double"] = 1
        stats["total bases"] = 2
        stats["default batter destination"] = 2
    elif main.startswith("S") and not main.startswith("SB"):
        stats["hit"] = 1
        stats["single"] = 1
        stats["total bases"] = 1
        stats["default batter destination"] = 1
    elif main.startswith("K"):
        stats["strikeout"] = 1
        stats["outs on main play"] = 0 if "B-" in adv else 1
    elif main.startswith("C"):
        stats["at bat"] = 0
        stats["outs on main play"] = 1
    elif main.startswith("E") or "/E" in main:
        stats["default batter destination"] = 1
    elif main.startswith("FC"):
        stats["default batter destination"] = 1
    elif "SH" in main or "SF" in main:
        stats["at bat"] = 0
        stats["outs on main play"] = 0 if "B-" in adv else 1
    else:
        stats["outs on main play"] = 0 if "B-" in adv else 1

    return stats


def _parse_adv_tokens(adv: str) -> list[tuple[str, str, str, str]]:
    if not adv:
        return []
    tokens = [token for token in adv.split(";") if token]
    parsed: list[tuple[str, str, str, str]] = []
    for token in tokens:
        match = ADV_RE.match(token)
        if match:
            parsed.append((match.group(1), match.group(2), match.group(3), token))
    return parsed


def _explicit_batter_destination(tokens: list[tuple[str, str, str, str]]) -> str | None:
    for src, sep, dest, _ in tokens:
        if src == "B" and sep == "-":
            return dest
    return None


def _process_game_lines(game_lines: list[str]) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
    info: dict[str, str] = {}
    current_pitcher: dict[int, str | None] = {0: None, 1: None}
    starting_pitcher: dict[int, str | None] = {0: None, 1: None}
    player_team_side: dict[str, int] = {}
    hitter_rows: dict[str, dict[str, object]] = {}
    pitcher_rows: dict[str, dict[str, object]] = {}
    earned_runs_by_pitcher: dict[str, int] = {}
    starting_lineup_count: dict[int, int] = defaultdict(int)
    game_id_raw = None

    for line in game_lines:
        parts = line.split(",")
        if parts[0] == "id":
            game_id_raw = parts[1]
        elif parts[0] == "info":
            info[parts[1]] = parts[2]
        elif parts[0] in {"start", "sub"}:
            player_id = parts[1]
            side = int(parts[3])
            batting_order = int(parts[4])
            field_pos = int(parts[5])
            player_team_side[player_id] = side
            if parts[0] == "start" and 1 <= batting_order <= 9:
                starting_lineup_count[side] += 1
            if field_pos == 1:
                current_pitcher[side] = player_id
                if parts[0] == "start" and starting_pitcher[side] is None:
                    starting_pitcher[side] = player_id
                    pitcher_rows.setdefault(player_id, _empty_pitcher_row(player_id))["was starting pitcher"] = True
        elif parts[0] == "data" and parts[1] == "er":
            earned_runs_by_pitcher[parts[2]] = int(parts[3])

    game_state = GameState(
        game_id=game_id_raw or "",
        date=_parse_date(info.get("date", "1900/01/01")),
        doubleheader_number=int(info.get("number", "0")),
        away_team_id=info.get("visteam", ""),
        home_team_id=info.get("hometeam", ""),
        day_night=info.get("daynight"),
        ballpark_id=info.get("site"),
        current_pitcher=current_pitcher,
        starting_pitcher=starting_pitcher,
        bases={1: None, 2: None, 3: None},
        last_half=None,
        outs_in_half=0,
        team_runs={0: 0, 1: 0},
    )

    def team_id_for_side(side: int) -> str:
        return game_state.away_team_id if side == 0 else game_state.home_team_id

    for line in game_lines:
        parts = line.split(",")
        if parts[0] == "sub":
            player_id = parts[1]
            side = int(parts[3])
            field_pos = int(parts[5])
            player_team_side[player_id] = side
            if field_pos == 1:
                game_state.current_pitcher[side] = player_id
        elif parts[0] == "play":
            inning = int(parts[1])
            batting_side = int(parts[2])
            batter_id = parts[3]
            event = parts[6]

            if event == "NP":
                continue

            if game_state.last_half != (inning, batting_side):
                game_state.bases = {1: None, 2: None, 3: None}
                game_state.outs_in_half = 0
                game_state.last_half = (inning, batting_side)

            player_team_side[batter_id] = batting_side
            pitching_side = 1 - batting_side
            pitcher_id = game_state.current_pitcher[pitching_side]
            parsed = _parse_play_event(event)
            adv_tokens = _parse_adv_tokens(str(parsed["adv"]))

            hitter_row = hitter_rows.setdefault(batter_id, _empty_hitter_row(batter_id))
            hitter_row["plate appearances"] += int(parsed["plate appearance"])
            hitter_row["at bats"] += int(parsed["at bat"])
            hitter_row["hits"] += int(parsed["hit"])
            hitter_row["singles"] += int(parsed["single"])
            hitter_row["doubles"] += int(parsed["double"])
            hitter_row["triples"] += int(parsed["triple"])
            hitter_row["home runs"] += int(parsed["home run"])
            hitter_row["total bases"] += int(parsed["total bases"])
            hitter_row["walks"] += int(parsed["walk"])
            hitter_row["hit by pitch"] += int(parsed["hit by pitch"])
            hitter_row["strikeouts"] += int(parsed["strikeout"])

            if pitcher_id is not None and int(parsed["plate appearance"]) > 0:
                pitcher_row = pitcher_rows.setdefault(pitcher_id, _empty_pitcher_row(pitcher_id))
                pitcher_row["batters faced"] += 1
                pitcher_row["hits allowed"] += int(parsed["hit"])
                pitcher_row["walks allowed"] += int(parsed["walk"])
                pitcher_row["hit by pitch allowed"] += int(parsed["hit by pitch"])
                pitcher_row["strikeouts"] += int(parsed["strikeout"])
                pitcher_row["home runs allowed"] += int(parsed["home run"])

            full_event = event

            for match in SB_RE.finditer(full_event):
                dest = match.group(1)
                src = {"2": 1, "3": 2, "H": 3}[dest]
                runner = game_state.bases.get(src)
                if runner:
                    hitter_rows.setdefault(runner, _empty_hitter_row(runner))["stolen bases"] += 1
                    if dest == "H":
                        hitter_rows.setdefault(runner, _empty_hitter_row(runner))["runs scored"] += 1
                        game_state.team_runs[batting_side] += 1
                        if pitcher_id is not None:
                            pitcher_rows.setdefault(pitcher_id, _empty_pitcher_row(pitcher_id))["runs allowed"] += 1
                        game_state.bases[src] = None
                    else:
                        game_state.bases[src] = None
                        game_state.bases[int(dest)] = runner

            for match in CS_RE.finditer(full_event):
                dest = match.group(1)
                src = {"2": 1, "3": 2, "H": 3}[dest]
                runner = game_state.bases.get(src)
                if runner:
                    hitter_rows.setdefault(runner, _empty_hitter_row(runner))["caught stealing"] += 1
                    game_state.bases[src] = None
                    if pitcher_id is not None:
                        pitcher_rows.setdefault(pitcher_id, _empty_pitcher_row(pitcher_id))["outs recorded"] += 1
                    game_state.outs_in_half += 1

            outs = int(parsed["outs on main play"])
            new_bases = game_state.bases.copy()
            moved_sources: list[int] = []
            batter_rbi = 0

            for src, sep, dest, raw_token in adv_tokens:
                runner_id = batter_id if src == "B" else game_state.bases.get(int(src))
                if src != "B":
                    moved_sources.append(int(src))

                if sep == "X":
                    outs += 1
                else:
                    if dest == "H":
                        if runner_id:
                            hitter_rows.setdefault(runner_id, _empty_hitter_row(runner_id))["runs scored"] += 1
                        game_state.team_runs[batting_side] += 1
                        if pitcher_id is not None:
                            pitcher_rows.setdefault(pitcher_id, _empty_pitcher_row(pitcher_id))["runs allowed"] += 1
                        if runner_id and runner_id != batter_id and "NR" not in raw_token and not (str(parsed["event main"]).startswith("E") or "/E" in str(parsed["event main"])):
                            batter_rbi += 1
                    else:
                        if runner_id:
                            new_bases[int(dest)] = runner_id

            for source in moved_sources:
                new_bases[source] = None

            if int(parsed["plate appearance"]) > 0 and _explicit_batter_destination(adv_tokens) is None:
                default_dest = parsed["default batter destination"]
                if default_dest in {1, 2, 3}:
                    new_bases[int(default_dest)] = batter_id
                elif default_dest == "H":
                    hitter_rows.setdefault(batter_id, _empty_hitter_row(batter_id))["runs scored"] += 1
                    game_state.team_runs[batting_side] += 1
                    if pitcher_id is not None:
                        pitcher_rows.setdefault(pitcher_id, _empty_pitcher_row(pitcher_id))["runs allowed"] += 1

            if int(parsed["home run"]) == 1:
                additional = sum(1 for src, sep, dest, raw in adv_tokens if src != "B" and sep == "-" and dest == "H")
                batter_rbi += 1 + additional

            hitter_rows.setdefault(batter_id, _empty_hitter_row(batter_id))["runs batted in"] += batter_rbi

            if pitcher_id is not None:
                pitcher_rows.setdefault(pitcher_id, _empty_pitcher_row(pitcher_id))["outs recorded"] += outs

            game_state.outs_in_half += outs
            game_state.bases = new_bases

            if game_state.outs_in_half >= 3:
                game_state.bases = {1: None, 2: None, 3: None}

    hitter_out: list[dict[str, object]] = []
    for player_id, row in hitter_rows.items():
        side = player_team_side.get(player_id)
        if side is None:
            continue
        hitter_out.append(
            {
                "game id": game_state.game_id,
                "date": game_state.date,
                "season": int(str(game_state.date)[:4]),
                "doubleheader number": game_state.doubleheader_number,
                "player id": player_id,
                "team id": team_id_for_side(side),
                "opponent team id": team_id_for_side(1 - side),
                "is home": side == 1,
                "ballpark id": game_state.ballpark_id,
                "day night": game_state.day_night,
                **row,
                "got a hit": int(row["hits"] > 0),
                "got 2plus total bases": int(row["total bases"] >= 2),
                "scored a run": int(row["runs scored"] > 0),
                "got an rbi": int(row["runs batted in"] > 0),
            }
        )

    pitcher_out: list[dict[str, object]] = []
    for player_id, row in pitcher_rows.items():
        side = player_team_side.get(player_id)
        if side is None:
            continue
        earned_runs = earned_runs_by_pitcher.get(player_id)
        innings_pitched = row["outs recorded"] / 3.0
        was_starting_pitcher = bool(row["was starting pitcher"])
        got_quality_start = None
        if was_starting_pitcher and earned_runs is not None:
            got_quality_start = int(innings_pitched >= 6.0 and earned_runs <= 3)

        pitcher_out.append(
            {
                "game id": game_state.game_id,
                "date": game_state.date,
                "season": int(str(game_state.date)[:4]),
                "doubleheader number": game_state.doubleheader_number,
                "player id": player_id,
                "team id": team_id_for_side(side),
                "opponent team id": team_id_for_side(1 - side),
                "is home": side == 1,
                "ballpark id": game_state.ballpark_id,
                "day night": game_state.day_night,
                **row,
                "earned runs": earned_runs,
                "innings pitched": innings_pitched,
                "got quality start": got_quality_start,
                "got win": None,
            }
        )

    away_runs = game_state.team_runs[0]
    home_runs = game_state.team_runs[1]
    game_row = {
        "game id": game_state.game_id,
        "date": game_state.date,
        "season": int(str(game_state.date)[:4]),
        "doubleheader number": game_state.doubleheader_number,
        "away team id": game_state.away_team_id,
        "home team id": game_state.home_team_id,
        "away runs": away_runs,
        "home runs": home_runs,
        "total runs": away_runs + home_runs,
        "home team won": int(home_runs > away_runs),
        "winning team id": game_state.home_team_id if home_runs > away_runs else game_state.away_team_id,
        "day night": game_state.day_night,
        "ballpark id": game_state.ballpark_id,
        "away starting pitcher id": game_state.starting_pitcher[0],
        "home starting pitcher id": game_state.starting_pitcher[1],
        "away starting lineup size": starting_lineup_count.get(0, 0),
        "home starting lineup size": starting_lineup_count.get(1, 0),
    }

    team_game_out = [
        {
            "game id": game_state.game_id,
            "date": game_state.date,
            "season": int(str(game_state.date)[:4]),
            "doubleheader number": game_state.doubleheader_number,
            "team id": game_state.away_team_id,
            "opponent team id": game_state.home_team_id,
            "is home": False,
            "runs scored": away_runs,
            "runs allowed": home_runs,
            "won game": int(away_runs > home_runs),
            "starting pitcher id": game_state.starting_pitcher[0],
            "starting lineup size": starting_lineup_count.get(0, 0),
            "day night": game_state.day_night,
            "ballpark id": game_state.ballpark_id,
        },
        {
            "game id": game_state.game_id,
            "date": game_state.date,
            "season": int(str(game_state.date)[:4]),
            "doubleheader number": game_state.doubleheader_number,
            "team id": game_state.home_team_id,
            "opponent team id": game_state.away_team_id,
            "is home": True,
            "runs scored": home_runs,
            "runs allowed": away_runs,
            "won game": int(home_runs > away_runs),
            "starting pitcher id": game_state.starting_pitcher[1],
            "starting lineup size": starting_lineup_count.get(1, 0),
            "day night": game_state.day_night,
            "ballpark id": game_state.ballpark_id,
        },
    ]

    return hitter_out, pitcher_out, game_row, team_game_out


def build_historical_betting_data(zip_paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    player_directory_rows: list[dict[str, object]] = []
    game_rows: list[dict[str, object]] = []
    team_game_rows: list[dict[str, object]] = []
    hitter_rows: list[dict[str, object]] = []
    pitcher_rows: list[dict[str, object]] = []

    seen_event_members: set[tuple[str, str]] = set()
    seen_roster_members: set[tuple[str, str]] = set()

    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as archive:
            for member_name in archive.namelist():
                key = (member_name, str(zip_path.name))
                if ROSTER_FILE_RE.search(member_name):
                    if key in seen_roster_members:
                        continue
                    seen_roster_members.add(key)
                    player_directory_rows.extend(_parse_roster_file(member_name, archive.read(member_name)))
                elif EVENT_FILE_RE.search(member_name):
                    if key in seen_event_members:
                        continue
                    seen_event_members.add(key)
                    lines = archive.read(member_name).decode("latin1").splitlines()
                    for game_lines in _iter_games_from_event_lines(lines):
                        game_hitter_rows, game_pitcher_rows, game_row, game_team_rows = _process_game_lines(game_lines)
                        hitter_rows.extend(game_hitter_rows)
                        pitcher_rows.extend(game_pitcher_rows)
                        game_rows.append(game_row)
                        team_game_rows.extend(game_team_rows)

    player_directory = pd.DataFrame(player_directory_rows).drop_duplicates()
    actual_game_results = pd.DataFrame(game_rows).drop_duplicates(subset=["game id"])
    actual_team_game_results = pd.DataFrame(team_game_rows).drop_duplicates(subset=["game id", "team id"])
    actual_hitter_stats_by_game = pd.DataFrame(hitter_rows)
    actual_pitcher_stats_by_game = pd.DataFrame(pitcher_rows)

    return (
        player_directory,
        actual_game_results,
        actual_team_game_results,
        actual_hitter_stats_by_game,
        actual_pitcher_stats_by_game,
    )
