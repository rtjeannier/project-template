"""Fetch raw projection data from the FantasyPros API.

UNVERIFIED: written without access to the API (the dev environment's network blocks
api.fantasypros.com). It saves raw JSON responses only. The converter into the
league-state format will be written once real responses are in hand, so it doesn't
guess at field names.

Key: set FANTASYPROS_API_KEY in the environment or in a git-ignored .env file.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://api.fantasypros.com/public/v2/json/nfl/{season}/{endpoint}"
POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")


def _api_key() -> str:
    key = os.environ.get("FANTASYPROS_API_KEY")
    if not key and Path(".env").exists():
        for line in Path(".env").read_text().splitlines():
            if line.strip().startswith("FANTASYPROS_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"')
    if not key:
        raise SystemExit("FANTASYPROS_API_KEY is not set (environment or .env)")
    return key


def get(season: int, endpoint: str, **params) -> dict:
    url = BASE.format(season=season, endpoint=endpoint)
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"x-api-key": _api_key()})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def fetch_raw(season: int, week: int, out_dir: str | Path, scoring: str = "PPR") -> list[Path]:
    """Save this week's projections and ROS rankings for each position as raw JSON."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for pos in POSITIONS:
        for name, endpoint, params in [
            (f"proj_wk{week}_{pos}", "projections", {"position": pos, "week": week, "scoring": scoring}),
            (f"ros_rank_{pos}", "consensus-rankings", {"position": pos, "type": "ros", "scoring": scoring}),
        ]:
            try:
                data = get(season, endpoint, **params)
            except Exception as e:  # keep going so one bad endpoint doesn't lose the rest
                data = {"error": repr(e), "endpoint": endpoint, "params": params}
            path = out / f"{name}.json"
            path.write_text(json.dumps(data, indent=1))
            written.append(path)
    return written
