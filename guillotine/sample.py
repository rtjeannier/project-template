"""Synthetic league generator, used for demos and tests.

All players and teams are made up. The point is to have a realistic-shaped league
(9 teams left at week 10, with uneven budgets) to exercise the simulator before
real data is wired in.
"""

from __future__ import annotations

import numpy as np

# (position, players in the league, ROS points per game for the best player, decay per rank)
POSITION_TIERS = [
    ("QB", 40, 24.0, 0.035),
    ("RB", 90, 20.0, 0.022),
    ("WR", 120, 19.0, 0.016),
    ("TE", 45, 14.0, 0.035),
    ("K", 32, 9.5, 0.010),
    ("DEF", 32, 10.0, 0.014),
]
NFL_TEAMS = [f"T{i:02d}" for i in range(32)]


def make_sample_league(
    seed: int = 7,
    n_teams_alive: int = 9,
    current_week: int = 10,
    last_week: int = 17,
    roster_size: int = 16,
) -> dict:
    rng = np.random.default_rng(seed)
    weeks = list(range(current_week, last_week + 1))
    players = []
    for pos, n, top, decay in POSITION_TIERS:
        for r in range(n):
            ppg = max(1.0, top * np.exp(-decay * r) * rng.uniform(0.9, 1.1))
            bye = int(rng.integers(5, 15))
            games = sum(1 for w in weeks if w != bye)
            trade_value = round(max(0.0, (ppg - 0.45 * top) * 4.0) * games / len(weeks), 1)
            players.append({
                "id": f"{pos.lower()}{r + 1}",
                "name": f"Sample {pos} {r + 1}",
                "pos": pos,
                "nfl_team": NFL_TEAMS[int(rng.integers(0, 32))],
                "bye": bye,
                "ros_points": round(ppg * games, 1),
                "trade_value": trade_value,
            })

    # Snake-ish allocation so rosters are uneven but plausible: n_teams_alive teams
    # plus one team that was just eliminated (its roster is this week's waiver pool).
    by_pos = {pos: [p for p in players if p["pos"] == pos] for pos, *_ in POSITION_TIERS}
    need = {"QB": 2, "RB": 5, "WR": 5, "TE": 2, "K": 1, "DEF": 1}
    n_rosters = n_teams_alive + 1
    rosters = [[] for _ in range(n_rosters)]
    for pos, count in need.items():
        pool = by_pos[pos]
        # Shuffle among nearby ranks so talent isn't perfectly balanced.
        noisy = sorted(range(len(pool)), key=lambda i: i + rng.normal(0, 4))
        picks = iter(noisy)
        for _ in range(count):
            order = rng.permutation(n_rosters)
            for t in order:
                rosters[t].append(pool[next(picks)]["id"])

    rostered = {pid for r in rosters for pid in r}
    free_agents = [p["id"] for p in players if p["id"] not in rostered]

    names = ["You", "Gridiron Ghosts", "Faab Hoarders", "Waiver Wolves", "Late Round QB",
             "Bench Mob", "Sunday Scaries", "Bye Week Blues", "Kicker Truthers", "Chopping Block"]
    teams = []
    for i in range(n_teams_alive):
        teams.append({
            "id": f"team{i + 1}",
            "name": names[i],
            "faab": int(rng.integers(150, 900)),
            "players": rosters[i],
        })

    return {
        "settings": {
            "season": 2026,
            "current_week": current_week,
            "last_week": last_week,
            "roster_slots": {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1, "K": 1, "DEF": 1, "BN": 6},
            "roster_size": roster_size,
            "eliminations_per_week": 1,
            "min_bid": 0,
        },
        "my_team_id": "team1",
        "teams": teams,
        "waiver_pool": rosters[n_teams_alive],
        "free_agents": free_agents,
        "players": players,
    }
