"""Core data structures and the league-state file format.

A league state is a single JSON file (see ``examples/`` and ``guillotine sample``):

{
  "settings": {
    "season": 2026, "current_week": 10, "last_week": 17,
    "roster_slots": {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1, "K": 1, "DEF": 1},
    "roster_size": 16, "eliminations_per_week": 1, "min_bid": 0
  },
  "my_team_id": "t1",
  "teams": [{"id": "t1", "name": "...", "faab": 412, "players": ["p1", ...],
             "aggression": 1.0}],
  "waiver_pool": ["p99", ...],      # players released by the team eliminated last week
  "free_agents": ["p120", ...],     # optional: notable unrostered players
  "players": [{"id": "p1", "name": "...", "pos": "RB", "nfl_team": "DET", "bye": 8,
               "ros_points": 112.5, "weekly": {"10": 14.1}, "trade_value": 31}]
}

``current_week`` is the next week to be played. The waiver round on ``waiver_pool``
happens before that week's games, matching how guillotine leagues process the
eliminated roster.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

FLEX_ELIGIBILITY = {
    "FLEX": ("RB", "WR", "TE"),
    "W/R/T": ("RB", "WR", "TE"),
    "W/R": ("RB", "WR"),
    "W/T": ("WR", "TE"),
    "R/W": ("RB", "WR"),
    "SUPERFLEX": ("QB", "RB", "WR", "TE"),
    "Q/W/R/T": ("QB", "RB", "WR", "TE"),
}
NON_STARTING_SLOTS = {"BN", "IR", "BENCH"}


@dataclass
class Player:
    id: str
    name: str
    pos: str
    nfl_team: str = ""
    bye: int | None = None
    ros_points: float = 0.0
    weekly: dict[int, float] = field(default_factory=dict)
    trade_value: float | None = None


@dataclass
class Team:
    id: str
    name: str
    faab: float
    players: list[str]
    aggression: float = 1.0  # >1 bids more aggressively than the league norm


@dataclass
class Settings:
    season: int
    current_week: int
    last_week: int
    roster_slots: dict[str, int]
    roster_size: int = 16
    eliminations_per_week: int = 1
    min_bid: int = 0

    @property
    def weeks(self) -> list[int]:
        return list(range(self.current_week, self.last_week + 1))


@dataclass
class LeagueState:
    settings: Settings
    teams: list[Team]
    players: dict[str, Player]
    my_team_id: str
    waiver_pool: list[str] = field(default_factory=list)
    free_agents: list[str] = field(default_factory=list)

    def team(self, team_id: str) -> Team:
        return next(t for t in self.teams if t.id == team_id)

    @classmethod
    def from_dict(cls, d: dict) -> "LeagueState":
        s = d["settings"]
        settings = Settings(
            season=s["season"],
            current_week=s["current_week"],
            last_week=s["last_week"],
            roster_slots=s["roster_slots"],
            roster_size=s.get("roster_size", 16),
            eliminations_per_week=s.get("eliminations_per_week", 1),
            min_bid=s.get("min_bid", 0),
        )
        players = {}
        for p in d["players"]:
            players[p["id"]] = Player(
                id=p["id"],
                name=p["name"],
                pos=p["pos"].upper().replace("DST", "DEF"),
                nfl_team=p.get("nfl_team", ""),
                bye=p.get("bye"),
                ros_points=float(p.get("ros_points", 0.0)),
                weekly={int(k): float(v) for k, v in p.get("weekly", {}).items()},
                trade_value=p.get("trade_value"),
            )
        teams = [
            Team(
                id=t["id"],
                name=t["name"],
                faab=float(t["faab"]),
                players=list(t["players"]),
                aggression=float(t.get("aggression", 1.0)),
            )
            for t in d["teams"]
        ]
        state = cls(
            settings=settings,
            teams=teams,
            players=players,
            my_team_id=d["my_team_id"],
            waiver_pool=list(d.get("waiver_pool", [])),
            free_agents=list(d.get("free_agents", [])),
        )
        state.validate()
        return state

    def to_dict(self) -> dict:
        """The single-file state format (what from_dict reads)."""
        st = self.settings
        return {
            "settings": {"season": st.season, "current_week": st.current_week, "last_week": st.last_week,
                         "roster_slots": st.roster_slots, "roster_size": st.roster_size,
                         "eliminations_per_week": st.eliminations_per_week, "min_bid": st.min_bid},
            "my_team_id": self.my_team_id,
            "teams": [{"id": t.id, "name": t.name, "faab": t.faab, "players": t.players, "aggression": t.aggression}
                      for t in self.teams],
            "waiver_pool": self.waiver_pool, "free_agents": self.free_agents,
            "players": [{"id": p.id, "name": p.name, "pos": p.pos, "nfl_team": p.nfl_team, "bye": p.bye,
                         "ros_points": p.ros_points, "weekly": {str(w): v for w, v in p.weekly.items()},
                         **({"trade_value": p.trade_value} if p.trade_value is not None else {})}
                        for p in self.players.values()],
        }

    @classmethod
    def load(cls, path: str | Path) -> "LeagueState":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def validate(self) -> None:
        known = set(self.players)
        ids = [pid for t in self.teams for pid in t.players] + self.waiver_pool + self.free_agents
        missing = sorted(set(ids) - known)
        if missing:
            raise ValueError(f"player ids referenced but not defined: {missing[:10]}")
        if self.my_team_id not in {t.id for t in self.teams}:
            raise ValueError(f"my_team_id {self.my_team_id!r} is not one of the teams")
        for slot in self.settings.roster_slots:
            if slot not in FLEX_ELIGIBILITY and slot not in NON_STARTING_SLOTS:
                if slot not in {"QB", "RB", "WR", "TE", "K", "DEF"}:
                    raise ValueError(f"unknown roster slot {slot!r}")
