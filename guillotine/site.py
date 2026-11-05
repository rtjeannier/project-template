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


FAAB_BINS = [-1, 0.10, 0.25, 0.50, 0.75, 10]
FAAB_LABELS = ["under 10%", "10–25%", "25–50%", "50–75%", "75%+"]
WHEN_BINS = [(9, 11, "weeks 9–11"), (12, 14, "weeks 12–14"), (15, 17, "weeks 15–17")]


def _paths(bids, weeks, seasons, teams, names, pname, ppos, week_list, min_avail: int = 40) -> dict:
    """Paths to victory, per team and over time.

    - by week: seasons the team is still alive going into that week, how many it went on
      to win, split by FAAB left after that week's waivers (share of its FAAB today), and
      FAAB-left quartiles in seasons it won vs lost.
    - players: for each player who hit waivers while the team was alive (first time he
      hit waivers that season), the team's win rate when it got him vs when it didn't,
      overall and by when he became available.
    """
    champ = dict(zip(seasons.season, seasons.champion_id))
    start = dict(zip(teams.team_id, teams.start_faab))
    w = weeks.copy()
    w["won"] = w.season.map(champ) == w.team_id
    w["frac"] = w.faab_after_waivers / w.team_id.map(start).clip(lower=1)
    w["bin"] = pd.cut(w.frac, FAAB_BINS, labels=False)
    out = {"weeks": week_list, "faab_labels": FAAB_LABELS, "when_labels": [x[2] for x in WHEN_BINS], "teams": []}

    # first week each player hit waivers (any bid on him) in each season
    first = bids[["season", "week", "player_id"]].drop_duplicates().groupby(["season", "player_id"]).week.min().reset_index()
    got = bids[bids.won][["season", "player_id", "team_id"]].drop_duplicates()
    for t in teams.team_id:
        tw = w[w.team_id == t]
        by_week = []
        for wk in week_list:
            x = tw[tw.week == wk]
            cells = x.groupby("bin").won.agg(["size", "sum"]).reindex(range(len(FAAB_LABELS)), fill_value=0)
            q = lambda df: [float(v) for v in df.faab_after_waivers.quantile([0.25, 0.5, 0.75])] if len(df) else None
            by_week.append({"week": int(wk), "alive": int(len(x)), "wins": int(x.won.sum()),
                            "faab_bins": cells.astype(int).values.tolist(),
                            "faab_won": q(x[x.won]), "faab_lost": q(x[~x.won])})
        # players available while this team was alive
        alive = tw[["season", "week"]]
        av = first.merge(alive, on=["season", "week"])
        g = got[got.team_id == t][["season", "player_id"]].assign(got=True)
        av = av.merge(g, on=["season", "player_id"], how="left")
        av["got"] = av.got.fillna(False).astype(bool)
        av["won"] = av.season.map(champ) == t
        av["when"] = 0
        for i, (a, b2, _) in enumerate(WHEN_BINS):
            av.loc[(av.week >= a) & (av.week <= b2), "when"] = i
        stats = av.groupby(["player_id", "when", "got"]).won.agg(["size", "sum"]).reset_index()
        counts = av.groupby("player_id").size()
        plist = []
        for pid in counts[counts >= min_avail].index:
            sp = stats[stats.player_id == pid]
            plist.append({"id": str(pid), "name": pname.get(pid, str(pid)), "pos": ppos.get(pid, ""),
                          "cells": sp[["when", "got", "size", "sum"]].astype(int).values.tolist()})
        out["teams"].append({"name": names[t], "seasons": int(len(seasons)), "wins": int((seasons.champion_id == t).sum()),
                             "start_faab": float(start[t]), "by_week": by_week, "players": plist})
    return out


def build(data_dir: str | Path, out_dir: str | Path, sample_seasons: int = 60, anonymize: bool = False) -> list[Path]:
    d, out = Path(data_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((d / "meta.json").read_text())
    bids = pd.read_csv(d / "bids.csv.gz", dtype={"player_id": str, "team_id": str, "drop_id": str})
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
    paths = _paths(bids, weeks, seasons, teams, names, pn_all := dict(zip(players.player_id, players.name)),
                   dict(zip(players.player_id, players.pos)), meta["weeks"])
    (out / "paths.json").write_text(json.dumps(paths, separators=(",", ":")))

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
                               "drop": (pn.get(winner.drop_id.iloc[0], "open spot") if len(winner) and isinstance(winner.drop_id.iloc[0], str)
                                        else ("open spot" if len(winner) else None)),
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
