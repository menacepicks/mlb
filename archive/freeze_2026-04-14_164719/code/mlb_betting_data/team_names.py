from __future__ import annotations

import re
import unicodedata

TEAM_NAME_TO_ID = {
    "arizona diamondbacks": "ARI",
    "diamondbacks": "ARI",
    "dbacks": "ARI",
    "ari": "ARI",
    "atlanta braves": "ATL",
    "braves": "ATL",
    "atl": "ATL",
    "baltimore orioles": "BAL",
    "orioles": "BAL",
    "bal": "BAL",
    "boston red sox": "BOS",
    "red sox": "BOS",
    "bos": "BOS",
    "chicago cubs": "CHN",
    "cubs": "CHN",
    "chn": "CHN",
    "chicago white sox": "CHA",
    "white sox": "CHA",
    "cha": "CHA",
    "cincinnati reds": "CIN",
    "reds": "CIN",
    "cin": "CIN",
    "cleveland guardians": "CLE",
    "cleveland indians": "CLE",
    "guardians": "CLE",
    "indians": "CLE",
    "cle": "CLE",
    "colorado rockies": "COL",
    "rockies": "COL",
    "col": "COL",
    "detroit tigers": "DET",
    "tigers": "DET",
    "det": "DET",
    "houston astros": "HOU",
    "astros": "HOU",
    "hou": "HOU",
    "kansas city royals": "KCA",
    "royals": "KCA",
    "kc": "KCA",
    "kca": "KCA",
    "los angeles angels": "ANA",
    "anaheim angels": "ANA",
    "angels": "ANA",
    "la angels": "ANA",
    "ana": "ANA",
    "los angeles dodgers": "LAN",
    "dodgers": "LAN",
    "la dodgers": "LAN",
    "lan": "LAN",
    "miami marlins": "MIA",
    "florida marlins": "FLO",
    "marlins": "MIA",
    "mia": "MIA",
    "flo": "FLO",
    "milwaukee brewers": "MIL",
    "brewers": "MIL",
    "mil": "MIL",
    "minnesota twins": "MIN",
    "twins": "MIN",
    "min": "MIN",
    "new york mets": "NYN",
    "mets": "NYN",
    "nyn": "NYN",
    "new york yankees": "NYA",
    "yankees": "NYA",
    "nya": "NYA",
    "oakland athletics": "OAK",
    "athletics": "OAK",
    "a's": "OAK",
    "as": "OAK",
    "oak": "OAK",
    "philadelphia phillies": "PHI",
    "phillies": "PHI",
    "phi": "PHI",
    "pittsburgh pirates": "PIT",
    "pirates": "PIT",
    "pit": "PIT",
    "san diego padres": "SDN",
    "padres": "SDN",
    "sdn": "SDN",
    "san francisco giants": "SFN",
    "giants": "SFN",
    "sfn": "SFN",
    "seattle mariners": "SEA",
    "mariners": "SEA",
    "sea": "SEA",
    "st. louis cardinals": "SLN",
    "st louis cardinals": "SLN",
    "cardinals": "SLN",
    "sln": "SLN",
    "tampa bay rays": "TBA",
    "rays": "TBA",
    "tba": "TBA",
    "texas rangers": "TEX",
    "rangers": "TEX",
    "tex": "TEX",
    "toronto blue jays": "TOR",
    "blue jays": "TOR",
    "jays": "TOR",
    "tor": "TOR",
    "washington nationals": "WAS",
    "nationals": "WAS",
    "nats": "WAS",
    "was": "WAS",
    "montreal expos": "MON",
    "expos": "MON",
    "mon": "MON",
}

NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]+")


def _norm(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = text.replace("@", " ").replace(" at ", " ").replace(" versus ", " ").replace(" vs ", " ")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = NON_ALNUM_RE.sub(" ", text)
    text = " ".join(text.split())
    return text


def team_name_to_id(name: str) -> str:
    text = _norm(name)
    if not text:
        return ""
    if text in TEAM_NAME_TO_ID:
        return TEAM_NAME_TO_ID[text]
    words = text.split()
    if words:
        last_two = " ".join(words[-2:])
        if last_two in TEAM_NAME_TO_ID:
            return TEAM_NAME_TO_ID[last_two]
        last_one = words[-1]
        if last_one in TEAM_NAME_TO_ID:
            return TEAM_NAME_TO_ID[last_one]
    return ""
