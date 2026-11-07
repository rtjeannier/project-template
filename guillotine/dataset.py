"""Generate the full simulated dataset: every bid, award, drop, score and cut.

``python -m guillotine simulate-data LEAGUE.json --seasons 5000 --out sim_data``

Every season is played straight through from the league file. Every team bids its fair
bid (the rational bid, no noise); a share of bids (noise_share) are multiplied by a random
draw from Lognormal(0, bid_sd) (median 1x, always positive), capped so an overpay eats at
most noise_cap of what a fair bid would leave. Lock bids have no multiplier (empty),
independently for each bid, so there is no fixed strategy and the same cut sequence
produces different waivers each time. Nothing is analyzed here: the files are the raw
record, to crawl afterwards.

Files (gzip CSV):
- ``bids.csv.gz``: season, week, team, player, the player this claim drops (blank = an
  open roster spot), what the bid was for (``lineup`` = starts this week or in
  week 17, ``bench`` = a backup), fair bid,
  multiplier, bid, FAAB at the time, the team's elimination odds that week, whether it
  won the player, and the price.
- ``weeks.csv.gz``: each alive team's score each week, whether it was cut, and its FAAB
  after waivers.
- ``drops.csv.gz``: players released to make room after a pickup.
- ``seasons.csv.gz``: the champion and the order teams were cut.
- ``injuries.csv.gz``: every injury to a rostered player: the week he got hurt (he plays
  that game), his team, how many remaining weeks he misses, and whether it ends his
  season.
- ``players.csv``, ``teams.csv``, ``league.json``: lookup tables and the starting state,
  so any roster at any point can be rebuilt from the starting rosters plus awards and drops.
"""

from __future__ import annotations

import csv
import gzip
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from .models import LeagueState
from .simulate import Simulator, SimResults, UserPolicy

_SIM: Simulator | None = None


def _scratch(sim: Simulator) -> SimResults:
    n_w = len(sim.proj.weeks)
    return SimResults(n_sims=0, team_ids=[t.id for t in sim.state.teams], weeks=sim.proj.weeks,
                      elim_counts={t.id: np.zeros(n_w) for t in sim.state.teams},
                      champion_counts=defaultdict(int), opp_max_bid=defaultdict(list), user_gain_now={})


def _play(season_range: tuple[int, int]) -> dict[str, list[tuple]]:
    """Play seasons [a, b) and turn their traces into rows (runs in a worker)."""
    sim = _SIM
    weeks = sim.proj.weeks
    out = {"bids": [], "weeks": [], "drops": [], "seasons": [], "injuries": []}
    for s in range(*season_range):
        sim.trace = []
        sim._run_one(s, UserPolicy("auto"), _scratch(sim))
        odds, won, cuts, champ, drop_of = {}, {}, [], None, {}
        for e in sim.trace:
            if e[0] == "claim":
                drop_of[(e[2], e[3], e[4])] = e[5]
            elif e[0] == "odds":
                odds[e[2]] = e[3]
            elif e[0] == "award":
                won[(e[2], e[3], e[4])] = e[5]
            elif e[0] == "week":
                _, _, j, scores, losers, faab = e
                for t, sc in scores.items():
                    out["weeks"].append((s, weeks[j], t, round(sc, 2), t in losers, faab.get(t)))
                cuts += [f"{weeks[j]}:{t}" for t in losers]
            elif e[0] == "drop":
                out["drops"].append((s, weeks[e[2]], e[3], e[4]))
            elif e[0] == "injury":
                _, _, j, pid, tid, weeks_out, ending = e
                out["injuries"].append((s, weeks[j], tid, pid, weeks_out, ending))
            elif e[0] == "champ":
                champ = e[2]
        for e in sim.trace:
            if e[0] == "bid":
                _, _, j, pid, tid, amt, fair, m, faab, kind = e
                price = won.get((j, pid, tid))
                out["bids"].append((s, weeks[j], tid, pid, kind, drop_of.get((j, pid, tid)), fair,
                                    None if m is None else round(m, 4), amt, faab,
                                    round(odds.get(j, {}).get(tid, 0.0), 4), price is not None, price))
        out["seasons"].append((s, champ, " ".join(cuts)))
    sim.trace = None
    return out


HEADERS = {
    "bids": ("season", "week", "team_id", "player_id", "kind", "drop_id", "fair_bid", "multiplier", "bid", "faab",
             "cut_odds", "won", "price"),
    "weeks": ("season", "week", "team_id", "score", "cut", "faab_after_waivers"),
    "drops": ("season", "week", "team_id", "player_id"),
    "seasons": ("season", "champion_id", "cuts"),
    "injuries": ("season", "week", "team_id", "player_id", "weeks_out", "season_ending"),
}


def generate(state: LeagueState, league_path: str | Path, n_seasons: int, out_dir: str | Path,
             seed: int = 0, workers: int = 1, chunk: int = 50) -> list[Path]:
    global _SIM
    from .analysis import final_edge

    sim = Simulator(state, seed=seed)
    _SIM = sim
    ranges = [(a, min(a + chunk, n_seasons)) for a in range(0, n_seasons, chunk)]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {k: gzip.open(out / f"{k}.csv.gz", "wt", newline="") for k in HEADERS}
    writers = {k: csv.writer(f) for k, f in files.items()}
    for k, w in writers.items():
        w.writerow(HEADERS[k])

    def sink(part):
        for k, rows in part.items():
            writers[k].writerows(rows)

    if workers > 1:
        import multiprocessing as mp

        with mp.get_context("fork").Pool(workers) as pool:
            for part in pool.imap(_play, ranges):
                sink(part)
    else:
        for r in ranges:
            sink(_play(r))
    for f in files.values():
        f.close()

    edge = final_edge(sim)
    with (out / "players.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("player_id", "name", "pos", "nfl_team", "bye", "ros_points", "final_edge"))
        for p in state.players.values():
            w.writerow((p.id, p.name, p.pos, p.nfl_team, p.bye, p.ros_points, round(edge.get(p.id, 0.0), 2)))
    with (out / "teams.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("team_id", "name", "start_faab", "is_you"))
        for t in state.teams:
            w.writerow((t.id, t.name, t.faab, t.id == state.my_team_id))
    shutil.copy(league_path, out / "league.json")
    (out / "meta.json").write_text(json.dumps({
        "seasons": n_seasons, "seed": seed, "bid_sd": sim.model.bid_sd, "noise_share": sim.model.noise_share,
        "noise_cap": sim.model.noise_cap, "weeks": sim.proj.weeks,
        "you": state.my_team_id}))
    return sorted(out.iterdir())
