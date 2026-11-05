"""Build the static results explorer (GitHub Pages) from a simulated dataset.

``python -m guillotine build-site sim_data --out docs/explorer``

The raw bid log is millions of rows, too big for a browser, so it is pre-aggregated:

- ``cube.json``: bids grouped by how far over/under fair they were and by the situation
  at the time (player's final-week edge, position, week, bidder's FAAB vs the richest
  rival's, bidder's cut odds, fair-bid size, you vs everyone), with how many of those
  bidders went on to win the league. The page sums any combination of filters.
- ``players.json``: the same over/under split for each player with enough bids.
- ``seasons.json``: a sample of complete seasons (every bid with fair value and
  multiplier, every award, weekly scores, cuts and FAAB) to read one by one.

Only bids where the bidder's fair bid was above $0 count as over/under bids: a $0 fair
bid means the player was no use to that team.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

MULT_EDGES = [-np.inf, 0.0, 0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2.0, np.inf]
MULT_LABELS = ["$0 bid (draw ≤ 0)", "0–0.5×", "0.5–0.75×", "0.75–0.9×", "0.9–1.1× (fair)", "1.1–1.25×",
               "1.25–1.5×", "1.5–2×", "2×+"]
DIMS = {
    "kind": ("What the bid was for", ["lineup upgrade", "keep from rivals", "better backup ($1)"]),
    "edge": ("Player in the final", ["replaceable (edge ≤ 0)", "edge 0–2.5", "irreplaceable (edge > 2.5)"]),
    "pos": ("Position", ["QB", "RB", "WR", "TE"]),
    "week": ("Week", None),
    "ratio": ("Bidder's FAAB vs richest rival", ["under half", "half to equal", "1–2×", "over 2×"]),
    "cut": ("Bidder's cut risk that week", ["low (<8%)", "medium (8–15%)", "high (>15%)"]),
    "fair": ("Fair bid", ["$1–10", "$10–50", "$50–150", "$150+"]),
    "team": ("Bidder", None),
}


def _faab_before(weeks: pd.DataFrame, teams: pd.DataFrame, week_list: list[int]) -> pd.DataFrame:
    """FAAB each alive team had going into each week's waivers."""
    first = week_list[0]
    alive = weeks[["season", "week", "team_id"]]
    prev = weeks[["season", "week", "team_id", "faab_after_waivers"]].copy()
    prev["week"] = prev["week"].map(lambda w: week_list[week_list.index(w) + 1] if w != week_list[-1] else -1)
    out = alive.merge(prev.rename(columns={"faab_after_waivers": "faab_before"}), on=["season", "week", "team_id"], how="left")
    start = dict(zip(teams.team_id, teams.start_faab))
    m = out["week"] == first
    out.loc[m, "faab_before"] = out.loc[m, "team_id"].map(start)
    return out


def build(data_dir: str | Path, out_dir: str | Path, sample_seasons: int = 60, anonymize: bool = False) -> list[Path]:
    d, out = Path(data_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((d / "meta.json").read_text())
    bids = pd.read_csv(d / "bids.csv.gz", dtype={"player_id": str, "team_id": str})
    weeks = pd.read_csv(d / "weeks.csv.gz", dtype={"team_id": str})
    seasons = pd.read_csv(d / "seasons.csv.gz", dtype={"champion_id": str})
    players = pd.read_csv(d / "players.csv", dtype={"player_id": str})
    teams = pd.read_csv(d / "teams.csv", dtype={"team_id": str})
    you = str(meta["you"])
    names = dict(zip(teams.team_id, teams.name))
    if anonymize:
        names = {t: ("You" if t == you else f"Team {i + 1}") for i, t in enumerate(teams.team_id)}

    # --- situation at each bid
    fb = _faab_before(weeks, teams, meta["weeks"])
    top = fb.sort_values("faab_before", ascending=False).groupby(["season", "week"])
    first = top.nth(0)[["season", "week", "team_id", "faab_before"]].rename(columns={"team_id": "t1", "faab_before": "f1"})
    second = top.nth(1)[["season", "week", "faab_before"]].rename(columns={"faab_before": "f2"})
    rich = first.merge(second, on=["season", "week"], how="left")
    n_alive = weeks.groupby(["season", "week"]).size().rename("alive").reset_index()
    b = bids[bids.fair_bid > 0].merge(rich, on=["season", "week"], how="left").merge(n_alive, on=["season", "week"], how="left")
    b["richest_rival"] = np.where(b.team_id == b.t1, b.f2, b.f1)
    b = b.merge(players[["player_id", "pos", "final_edge"]], on="player_id", how="left")
    b = b.merge(seasons[["season", "champion_id"]], on="season", how="left")
    b["won_league"] = b.team_id == b.champion_id
    b["mult"] = pd.cut(b.multiplier, MULT_EDGES, labels=False, right=True)
    b["edge"] = pd.cut(b.final_edge, [-np.inf, 0, 2.5, np.inf], labels=False)
    b["pos"] = b.pos.map({p: i for i, p in enumerate(DIMS["pos"][1])}).fillna(-1).astype(int)
    ratio = b.faab / b.richest_rival.clip(lower=1)
    b["ratio"] = pd.cut(ratio, [-np.inf, 0.5, 1.0, 2.0, np.inf], labels=False)
    b["cut"] = pd.cut(b.cut_odds, [-np.inf, 0.08, 0.15, np.inf], labels=False)
    b["fair"] = pd.cut(b.fair_bid, [0, 10, 50, 150, np.inf], labels=False)
    b["you"] = (b.team_id == you).astype(int)
    team_order = list(teams.team_id)
    b["team"] = b.team_id.map({t: i for i, t in enumerate(team_order)})
    b["kind"] = b.kind.map({"lineup": 0, "deny": 1, "bench": 2})
    b = b[b.pos >= 0]

    keys = ["mult", "kind", "edge", "pos", "week", "ratio", "cut", "fair", "team"]
    cube = b.groupby(keys).agg(n=("won_league", "size"), w=("won_league", "sum")).reset_index()
    week_labels = [str(w) for w in meta["weeks"]]
    cube["week"] = cube["week"].map({w: i for i, w in enumerate(meta["weeks"])})
    team_labels = [names[t] for t in team_order]
    dims = {k: {"title": t, "labels": (labs if labs else (team_labels if k == "team" else week_labels))}
            for k, (t, labs) in DIMS.items()}
    champs = seasons.champion_id.value_counts(normalize=True)
    cube_out = {"keys": keys, "rows": cube[keys + ["n", "w"]].astype(int).values.tolist(),
                "mult_labels": MULT_LABELS, "dims": dims, "you_team": team_order.index(you),
                "team_p_champion": [float(champs.get(t, 0.0)) for t in team_order]}
    (out / "cube.json").write_text(json.dumps(cube_out, separators=(",", ":")))

    # --- per player
    pn = dict(zip(players.player_id, players.name))
    pp = b.groupby(["player_id", "mult", "team"]).agg(n=("won_league", "size"), w=("won_league", "sum")).reset_index()
    counts = b.groupby("player_id").size()
    keep = counts[counts >= 300].index
    plist = []
    for pid in keep:
        rows = pp[pp.player_id == pid]
        sub = players[players.player_id == pid].iloc[0]
        plist.append({"id": str(pid), "name": pn[pid], "pos": sub.pos, "edge": round(float(sub.final_edge), 2),
                      "n": int(counts[pid]), "median_fair": float(b[b.player_id == pid].fair_bid.median()),
                      "cells": rows[["mult", "team", "n", "w"]].astype(int).values.tolist()})
    plist.sort(key=lambda p: -p["n"])
    (out / "players.json").write_text(json.dumps(plist, separators=(",", ":")))

    # --- sample seasons
    samp = []
    allb = bids[bids.season < sample_seasons]
    for s in range(min(sample_seasons, len(seasons))):
        sb = allb[allb.season == s]
        sw = weeks[weeks.season == s]
        champ = str(seasons.loc[seasons.season == s, "champion_id"].iloc[0])
        wk = []
        for w in meta["weeks"]:
            ww = sw[sw.week == w]
            if ww.empty:
                break
            wb = sb[(sb.week == w) & ((sb.bid > 0) | sb.won)]
            awards = []
            for pid, g in wb.groupby("player_id"):
                g = g.sort_values("bid", ascending=False)
                winner = g[g.won]
                awards.append({"player": pn.get(pid, str(pid)), "pos": players.set_index("player_id").pos.get(pid, ""),
                               "winner": names[str(winner.team_id.iloc[0])] if len(winner) else None,
                               "price": int(winner.price.iloc[0]) if len(winner) else None,
                               "bids": [[names[str(r.team_id)], int(r.bid), int(r.fair_bid), round(float(r.multiplier), 2)]
                                        for r in g.itertuples()]})
            awards.sort(key=lambda a: -(a["price"] or 0))
            wk.append({"week": int(w), "awards": awards,
                       "scores": [[names[str(r.team_id)], round(float(r.score), 1), bool(r.cut), int(r.faab_after_waivers)]
                                  for r in ww.sort_values("score", ascending=False).itertuples()]})
        samp.append({"season": s, "champion": names[champ], "weeks": wk})
    (out / "seasons.json").write_text(json.dumps(samp, separators=(",", ":")))

    summary = {"seasons": int(meta["seasons"]), "bids": int(len(bids)), "analyzed_bids": int(len(b)),
               "bid_sd": meta["bid_sd"], "you": names[you],
               "teams": [{"name": names[t], "faab": float(f), "p_champion": float(champs.get(t, 0.0)),
                          "is_you": bool(t == you)} for t, f in zip(teams.team_id, teams.start_faab)]}
    (out / "summary.json").write_text(json.dumps(summary))
    shutil.copy(Path(__file__).with_name("site_template.html"), out / "index.html")
    return sorted(out.iterdir())
