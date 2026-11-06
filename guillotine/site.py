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

N_CONDITIONS = 14  # conditions shown per team (at most 31: the page uses 32-bit masks)
N_TIER = 6  # of which reserved for "lands an elite RB by week k" style conditions
TIER_WEEKS = (11, 14)  # the "by week k" windows for tier conditions in "What has to go right"
MIN_SEASONS = 200  # a condition needs this many seasons on each side to be shown
SHARE_EDGES = [-np.inf, 0.05, 0.10, 0.25, 0.50, 0.75, np.inf]
SHARE_LABELS = ["under 5%", "5–10%", "10–25%", "25–50%", "50–75%", "75%+"]
MULT_EDGES = [-np.inf, 0.0, 0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2.0, np.inf]
MULT_LABELS = ["$0 bid (draw ≤ 0)", "0–0.5×", "0.5–0.75×", "0.75–0.9×", "0.9–1.1× (fair)", "1.1–1.25×",
               "1.25–1.5×", "1.5–2×", "2×+"]
DIMS = {
    "share": ("Price paid (% of FAAB)", SHARE_LABELS),
    "kind": ("What the bid was for", ["starter (this week or week 17)", "backup"]),
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
    out = {"weeks": week_list, "faab_labels": FAAB_LABELS, "teams": []}

    # first week each player hit waivers (any bid on him) in each season
    first = bids[["season", "week", "player_id"]].drop_duplicates().groupby(["season", "player_id"]).week.min().reset_index()
    won_rows = bids[bids.won].drop_duplicates(["season", "player_id"])
    won_rows = won_rows.assign(pct=won_rows.price / won_rows.faab.clip(lower=1))
    got = won_rows[["season", "player_id", "team_id"]]
    for t in teams.team_id:
        tw = w[w.team_id == t]
        by_week = []
        for wk in week_list:
            x = tw[tw.week == wk]
            cells = x.groupby("bin").won.agg(["size", "sum"]).reindex(range(len(FAAB_LABELS)), fill_value=0)
            q = lambda df: [float(v) for v in df.frac.quantile([0.25, 0.5, 0.75])] if len(df) else None
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
        stats = av.groupby(["player_id", "got"]).won.agg(["size", "sum"]).reset_index()
        counts = av.groupby("player_id").size()
        # Who won him, and for how much, in those same seasons (first time he hit waivers).
        sales = av[["season", "player_id"]].merge(won_rows, on=["season", "player_id"])
        plist = []
        for pid in counts[counts >= min_avail].index:
            sp = stats[stats.player_id == pid]
            sold = sales[sales.player_id == pid]
            mine = sold[sold.team_id == t]
            top = sold.team_id.value_counts()
            plist.append({"id": str(pid), "name": pname.get(pid, str(pid)), "pos": ppos.get(pid, ""),
                          "cells": sp[["got", "size", "sum"]].astype(int).values.tolist(),
                          "paid": float(mine.price.median()) if len(mine) else None,
                          "paid_pct": float(mine.pct.median()) if len(mine) else None,
                          "price": float(sold.price.median()) if len(sold) else None,
                          "winners": [[names[u], round(float(c) / int(counts[pid]), 3)] for u, c in top.head(3).items()]})
        out["teams"].append({"name": names[t], "seasons": int(len(seasons)), "wins": int((seasons.champion_id == t).sum()),
                             "start_faab": float(start[t]), "by_week": by_week, "players": plist})
    return out



SURVIVAL_DIMS = {
    "aggr": ("Bidding vs fair that week", ["no bids", "underbid (avg < 0.9×)", "about fair (0.9–1.1×)", "overbid (avg > 1.1×)"]),
    "spend": ("FAAB spent that week", ["$0", "up to 10%", "10–30%", "over 30%"]),
    "pickup": ("Pickups that week", ["no lineup upgrade", "landed a lineup upgrade"]),
}


def _scenarios(weeks, seasons, bids, injuries, players, teams, names, week_list, start_rosters) -> dict:
    """What has to go right for each team: the conditions that move its title odds most.

    Each condition is judged only on the seasons where it could matter to the team, so
    the plain benefit of surviving long doesn't get credited to it:
    - "Rival U is cut by week k": seasons where the team is still alive after week k.
    - "Team lands player X": seasons where X hit waivers while the team was alive.
    - "Team lands an elite RB by week k" (any player of that tier and position): seasons
      where the team is alive for week k's waivers. Tiers come from the week-17 edge:
      elite = edge > 2.5, starter-grade or better = edge > 0.
    - "Team loses a starter-grade player for the season by week k": same seasons.
    - Injuries (rivals' players get hurt, the team's own stay healthy): all seasons.
    Each condition also says when it happens: the middle half of the weeks it happened in.
    For each: how often it happens within those seasons, and the team's title odds when it
    does vs when it doesn't. The top N_CONDITIONS by swing x uncertainty are kept, with
    per-season bitmasks (where it counts, and whether it happened) so the page can check
    any combination.
    """
    season_ids = seasons.season.values
    n = len(season_ids)
    pos_of = {sid: i for i, sid in enumerate(season_ids)}
    champ = seasons.set_index("season").champion_id.reindex(season_ids).values
    team_ids = list(teams.team_id)
    big = max(week_list) + 1
    cut = weeks[weeks.cut].groupby(["season", "team_id"]).week.min().unstack()
    cut = cut.reindex(index=season_ids, columns=team_ids).fillna(big)  # never cut = alive to the end
    edge = dict(zip(players.player_id, players.final_edge))
    pname = dict(zip(players.player_id, players.name))
    notable = [pid for pid, e in edge.items() if e > -3]  # elite and starter tiers
    owner = {pid: t for t, pids in start_rosters.items() for pid in pids}

    def mask_of(sids):
        m = np.zeros(n, dtype=bool)
        m[[pos_of[sid] for sid in sids]] = True
        return m

    everywhere = np.ones(n, dtype=bool)
    nowhere = np.zeros(n, dtype=bool)
    hurt = {pid: mask_of(g.season.unique()) for pid, g in injuries.groupby("player_id")} if len(injuries) else {}
    # first week each notable player hit waivers in each season (np.inf = never)
    nb = bids[bids.player_id.isin(notable)]
    first_wk = nb.groupby(["player_id", "season"]).week.min()
    avail_week = {pid: g.droplevel(0).reindex(season_ids).fillna(np.inf).values for pid, g in first_wk.groupby(level=0)}
    landed = {(t, pid): mask_of(g.season.unique()) for (t, pid), g in bids[bids.won].groupby(["team_id", "player_id"])}

    def week_of(df, cols):
        """First week per key and season, as {key: array over seasons (nan = never)}."""
        g = df.groupby(cols + ["season"]).week.min()
        out = {}
        for key, x in g.groupby(level=list(range(len(cols)))):
            out[key] = x.droplevel(list(range(len(cols)))).reindex(season_ids).values.astype(float)
        return out

    won_b = bids[bids.won].merge(players[["player_id", "pos", "final_edge"]], on="player_id", how="left")
    landed_wk = week_of(won_b, ["team_id", "player_id"])
    hurt_wk = {pid: v for pid, v in week_of(injuries, ["player_id"]).items()} if len(injuries) else {}
    hurt_wk = {(k[0] if isinstance(k, tuple) else k): v for k, v in hurt_wk.items()}
    tiers = [("an elite", 2.5), ("a starter-grade or better", 0.0)]
    tier_wk = {}  # (team, tier label, position) -> first week it landed one, per season
    for lab, lo in tiers:
        x = won_b[won_b.final_edge > lo]
        for (t, pos), v in week_of(x, ["team_id", "pos"]).items():
            tier_wk[(t, lab, pos)] = v
        for t, v in week_of(x[x.pos.isin(["RB", "WR", "TE"])].assign(any="ANY"), ["team_id"]).items():
            tier_wk[((t[0] if isinstance(t, tuple) else t), lab, "ANY")] = v
    lost = injuries[injuries.season_ending].merge(players[["player_id", "final_edge"]], on="player_id", how="left") if len(injuries) else injuries
    lost_wk = {}
    for lab, lo in tiers:
        if len(lost):
            for t, v in week_of(lost[lost.final_edge > lo], ["team_id"]).items():
                lost_wk[((t[0] if isinstance(t, tuple) else t), lab)] = v

    out = {"seasons": n, "teams": []}
    for t in team_ids:
        win = champ == t
        my_cut = cut[t].values
        cands = []

        def score(pop, ev):
            k, m = int(pop.sum()), int((pop & ev).sum())
            if m < MIN_SEASONS or k - m < MIN_SEASONS:
                return None
            f = m / k
            return abs(win[pop & ev].mean() - win[pop & ~ev].mean()) * (f * (1 - f)) ** 0.5 * (k / n) ** 0.5

        def add(label, where, kind, pop, ev, when=None):
            sc = score(pop, ev)
            if sc is not None:
                cands.append((sc, label, where, kind, pop, ev, when))

        def by_week(label_of, where_of, kind, wk_arr):
            """'By week k' versions of an event that first happens in week wk_arr, for an early
            and a middle window, judged on seasons the team is alive for week k's waivers."""
            if wk_arr is None:
                return
            for k in TIER_WEEKS:
                add(label_of(k), where_of(k), kind, my_cut >= k, wk_arr <= k, wk_arr)

        for u in team_ids:
            if u == t:
                continue
            opts = []
            for k in week_list[:-1]:
                pop = my_cut > k  # you're still alive after week k
                sc = score(pop, cut[u].values <= k)
                if sc is not None:
                    opts.append((sc, k))
            if opts:
                _, k = max(opts)
                add(f"{names[u]} is cut by week {k}", f"seasons you're alive after week {k}", "results",
                    my_cut > k, cut[u].values <= k, cut[u].values)
        for pid in notable:
            h = hurt.get(pid, nowhere)
            if owner.get(pid) == t:
                add(f"{pname[pid]} (yours) stays healthy", "all seasons", "injury", everywhere, ~h)
            else:
                who = f" ({names[owner[pid]]})" if pid in owner else ""
                add(f"{pname[pid]}{who} gets hurt", "all seasons", "injury", everywhere, h, hurt_wk.get(pid))
            aw = avail_week.get(pid)
            if aw is not None:
                pop = my_cut >= aw  # he hit waivers while you were alive
                add(f"{names[t]} lands {pname[pid]}", "seasons he hit waivers while you were alive", "pickup",
                    pop, landed.get((t, pid), nowhere), landed_wk.get((t, pid)))
        alive_k = lambda k: f"seasons you're alive for week {k} waivers"
        for lab, _ in tiers:
            for pos in ("RB", "WR", "TE", "QB", "ANY"):
                what = "RB, WR or TE" if pos == "ANY" else pos
                by_week(lambda k, lab=lab, what=what: f"{names[t]} lands {lab} {what} by week {k}", alive_k,
                        "tier", tier_wk.get((t, lab, pos)))
            by_week(lambda k, lab=lab: f"{names[t]} loses {lab} player for the season by week {k}", alive_k,
                    "injury", lost_wk.get((t, lab)))
        cands.sort(key=lambda c: -c[0])
        # Keep room for the tier-level conditions, so named players don't crowd them out.
        tier_c = [c for c in cands if c[3] == "tier"][:N_TIER]
        other = [c for c in cands if c[3] != "tier"][:N_CONDITIONS - len(tier_c)]
        top = sorted(tier_c + other, key=lambda c: -c[0])
        pop_bits = np.zeros(n, dtype=np.int64)
        ev_bits = np.zeros(n, dtype=np.int64)
        conds = []
        for i, (_, label, where, kind, pop, ev, when) in enumerate(top):
            pop_bits |= pop.astype(np.int64) << i
            ev_bits |= (pop & ev).astype(np.int64) << i
            wk = None
            if when is not None:
                w = np.asarray(when, dtype=float)[pop & ev]
                w = w[np.isfinite(w) & (w < big)]
                if len(w):
                    wk = [int(np.percentile(w, 25)), int(np.median(w)), int(np.percentile(w, 75))]
            conds.append({"label": label, "where": where, "kind": kind, "when": wk, "share": round(float(pop.mean()), 4),
                          "freq": round(float(ev[pop].mean()), 4),
                          "p_yes": round(float(win[pop & ev].mean()), 4), "p_no": round(float(win[pop & ~ev].mean()), 4)})
        out["teams"].append({"name": names[t], "p_win": round(float(win.mean()), 4), "conditions": conds,
                             "pop": pop_bits.tolist(), "ev": ev_bits.tolist(), "won": np.flatnonzero(win).tolist()})
    return out


def build(data_dir: str | Path, out_dir: str | Path, sample_seasons: int = 200, anonymize: bool = False) -> list[Path]:
    d, out = Path(data_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((d / "meta.json").read_text())
    bids = pd.read_csv(d / "bids.csv.gz", dtype={"player_id": str, "team_id": str, "drop_id": str})
    weeks = pd.read_csv(d / "weeks.csv.gz", dtype={"team_id": str})
    seasons = pd.read_csv(d / "seasons.csv.gz", dtype={"champion_id": str})
    inj_path = d / "injuries.csv.gz"
    injuries = (pd.read_csv(inj_path, dtype={"team_id": str, "player_id": str}) if inj_path.exists()
                else pd.DataFrame(columns=["season", "week", "team_id", "player_id", "weeks_out", "season_ending"]))
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
    b["kind"] = b.kind.map({"lineup": 0, "bench": 1})
    b = b[b.pos >= 0]
    # Price paid as a share of the FAAB the team had when it bid.
    b["share"] = pd.cut(b.bid / b.faab.clip(lower=1), SHARE_EDGES, labels=False)
    # Only bids that went through: did paying more or less, when you got him, change
    # your chance of winning the league? Losing bids are left out.
    wb = b[b.won]

    keys = ["mult", "share", "kind", "edge", "pos", "week", "ratio", "cut", "fair", "team"]
    cube = wb.groupby(keys).agg(n=("won_league", "size"), w=("won_league", "sum")).reset_index()
    week_labels = [str(w) for w in meta["weeks"]]
    cube["week"] = cube["week"].map({w: i for i, w in enumerate(meta["weeks"])})
    team_labels = [names[t] for t in team_order]
    dims = {k: {"title": t, "labels": (labs if labs else (team_labels if k == "team" else week_labels))}
            for k, (t, labs) in DIMS.items()}
    paths = _paths(bids, weeks, seasons, teams, names, pn_all := dict(zip(players.player_id, players.name)),
                   dict(zip(players.player_id, players.pos)), meta["weeks"])
    (out / "paths.json").write_text(json.dumps(paths, separators=(",", ":")))
    league = json.loads((d / "league.json").read_text())
    start_rosters = {str(t["id"]): [str(p) for p in t["players"]] for t in league["teams"]}
    scen = _scenarios(weeks, seasons, bids, injuries, players, teams, names, meta["weeks"], start_rosters)
    (out / "scenarios.json").write_text(json.dumps(scen, separators=(",", ":")))

    champs = seasons.champion_id.value_counts(normalize=True)
    cube_out = {"keys": keys, "rows": cube[keys + ["n", "w"]].astype(int).values.tolist(),
                "mult_labels": MULT_LABELS, "share_labels": SHARE_LABELS, "dims": dims, "you_team": team_order.index(you),
                "team_p_champion": [float(champs.get(t, 0.0)) for t in team_order]}
    (out / "cube.json").write_text(json.dumps(cube_out, separators=(",", ":")))

    # --- per player
    pn = dict(zip(players.player_id, players.name))
    pp = wb.groupby(["player_id", "mult", "share", "team"]).agg(n=("won_league", "size"), w=("won_league", "sum")).reset_index()
    counts = wb.groupby("player_id").size()
    keep = counts[counts >= 100].index
    fair_pct = (b.fair_bid / b.faab.clip(lower=1)).groupby(b.player_id).median()
    paid_pct = (wb.bid / wb.faab.clip(lower=1)).groupby(wb.player_id).median()
    med_week = b.groupby("player_id").week.median()
    plist = []
    for pid in keep:
        rows = pp[pp.player_id == pid]
        sub = players[players.player_id == pid].iloc[0]
        plist.append({"id": str(pid), "name": pn[pid], "pos": sub.pos, "edge": round(float(sub.final_edge), 2),
                      "n": int(counts[pid]), "median_week": float(med_week[pid]),
                      "median_fair_pct": float(fair_pct[pid]), "median_paid_pct": float(paid_pct[pid]),
                      "median_paid": float(wb[wb.player_id == pid].bid.median()),
                      "cells": rows[["mult", "share", "team", "n", "w"]].astype(int).values.tolist()})
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
            # every claim on players someone bid on above $0 or won, including $0 claims
            wk_b = sb[sb.week == w]
            contested = set(wk_b[(wk_b.bid > 0) | wk_b.won].player_id)
            wb = wk_b[wk_b.player_id.isin(contested)]
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
            hurt = injuries[(injuries.season == s) & (injuries.week == w)]
            inj = [[pn.get(r.player_id, r.player_id), names[str(r.team_id)], int(r.weeks_out), bool(r.season_ending)]
                   for r in hurt.itertuples()]
            wk.append({"week": int(w), "awards": awards, "injuries": inj,
                       "scores": [[names[str(r.team_id)], round(float(r.score), 1), bool(r.cut), int(r.faab_after_waivers)]
                                  for r in ww.sort_values("score", ascending=False).itertuples()]})
        samp.append({"season": s, "champion": names[champ], "weeks": wk})
    (out / "seasons.json").write_text(json.dumps(samp, separators=(",", ":")))

    overbid = _overbid(bids, seasons, teams, names, you, meta["weeks"], meta.get("bid_sd"))

    import datetime as _dt
    summary = {"built_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "seasons": int(meta["seasons"]), "bids": int(len(bids)), "analyzed_bids": int(len(b)),
               "bid_sd": meta["bid_sd"], "you": names[you], "overbid": overbid,
               "teams": [{"name": names[t], "faab": float(f), "p_champion": float(champs.get(t, 0.0)),
                          "is_you": bool(t == you)} for t, f in zip(teams.team_id, teams.start_faab)]}
    (out / "summary.json").write_text(json.dumps(summary))
    page = Path(__file__).with_name("site_template.html").read_text()
    (out / "index.html").write_text(page.replace("__BUILD__", summary["built_at"].replace(" ", "_")))
    return sorted(out.iterdir())


# Distance from the fair price, the same in both directions: 1.6x over is as far as 1/1.6x
# under (and, with lognormal noise, as likely).
OVERBID_STEPS = [1.1, 1.25, 1.6, 2.0]
OVERBID_LABELS = ["within 10%", "10–25%", "25–60% (up to 1.6×)", "1.6–2×", "2× or more"]
STUD_FAIR = 150
OVERBID_CAP = 1.6  # the "without crazy overbids" ceiling


def _overbid(bids, seasons, teams, names, you, weeks, bid_sd) -> dict:
    """What over- and underbidding did to title chances, from actual simulated outcomes.

    Bid multipliers are random, so grouping bids by multiplier is a fair experiment. Each
    bid is grouped by direction (over/under) and distance from fair, and compared with bids
    within 10% of fair. Lock bids (exactly 1x: topping every rival's whole FAAB) aren't
    random and are left out, as are small bids (fair < $10).

    "Without crazy overbids": each team's title odds if its stud bids (fair $150+) had never
    been drawn above OVERBID_CAP x fair, rivals unchanged. Seasons are reweighted (importance
    weighting): a season where one of its stud draws was above the cap gets weight 0, and
    otherwise 1 / P(draw <= cap) per stud draw, which is exactly the odds under noise that
    never goes above the cap."""
    from math import erf, log, sqrt

    b = bids[(bids.multiplier != 1.0) & (bids.fair_bid >= 10)].merge(
        seasons[["season", "champion_id"]], on="season")
    b["title"] = b.team_id.astype(str) == b.champion_id.astype(str)
    lm = np.log(b.multiplier.clip(lower=1e-6))
    steps = np.log(OVERBID_STEPS)
    b["dist"] = np.searchsorted(steps, np.abs(lm), side="right")  # 0 = within 10%, ... 4 = 2x or more
    b["over"] = lm > 0

    def side(x, base, n0):
        g = x.groupby("dist").agg(n=("title", "size"), got=("won", "mean"), title=("title", "mean"), paid=("price", "mean"))
        out = []
        for i in range(len(OVERBID_LABELS)):
            if i in g.index and g.loc[i, "n"] > 0:
                r = g.loc[i]
                se = float(np.sqrt(base * (1 - base) * (1 / r.n + 1 / max(n0, 1))))
                out.append([int(r.n), float(r.got), None if pd.isna(r.paid) else float(r.paid), float(r.title - base), se])
            else:
                out.append([0, None, None, None, None])
        return out

    def table(x):
        near = x[x.dist == 0]
        base = float(near.title.mean()) if len(near) else 0.0
        return {"fair": base, "n_fair": int(len(near)),
                "under": side(x[~x.over], base, len(near)), "over": side(x[x.over], base, len(near))}

    ys = b[b.team_id.astype(str) == str(you)]
    tables = {"all": table(b), "all_studs": table(b[b.fair_bid >= STUD_FAIR]),
              "you": table(ys), "you_studs": table(ys[ys.fair_bid >= STUD_FAIR])}

    # Without crazy overbids, team by team.
    sd = bid_sd or 0.42
    p_ok = 0.5 * (1 + erf(log(OVERBID_CAP) / sd / sqrt(2)))
    st = b[b.fair_bid >= STUD_FAIR]
    sids = seasons.season.values
    champ = seasons.set_index("season").champion_id.astype(str).reindex(sids)
    capped = []
    for t in teams.team_id:
        x = st[st.team_id.astype(str) == str(t)]
        g = x.groupby("season").multiplier.agg(n="size", bad=lambda m: bool((m > OVERBID_CAP).any()))
        w = pd.Series(1.0, index=sids)
        w.loc[g.index] = np.where(g.bad, 0.0, p_ok ** -g.n.astype(float))
        y = (champ == str(t)).astype(float).values
        ess = float(w.sum() ** 2 / (w ** 2).sum())
        pw = float((w.values * y).sum() / w.sum())
        capped.append({"team": names[t], "is_you": str(t) == str(you), "actual": float(y.mean()), "capped": pw,
                       "se": float(np.sqrt(pw * (1 - pw) / max(ess, 1))), "ess": ess,
                       "crazy_share": float(g.bad.sum()) / len(sids)})

    # Each team's first week (every team alive and bidding): how much it overbid overall,
    # weighted by fair price, in fifths of its own seasons.
    w0 = b[b.week == weeks[0]].copy()
    w0["lm"] = np.log(w0.multiplier)
    ts = w0.groupby(["season", "team_id"]).apply(lambda x: np.average(x.lm, weights=x.fair_bid), include_groups=False).rename("a").reset_index()
    ts = ts.merge(seasons[["season", "champion_id"]], on="season")
    ts["title"] = ts.team_id.astype(str) == ts.champion_id.astype(str)
    ts["q"] = ts.groupby("team_id")["a"].transform(lambda v: pd.qcut(v, 5, labels=False, duplicates="drop"))
    style = []
    for t in teams.team_id:
        x = ts[ts.team_id.astype(str) == str(t)]
        if len(x) < 500:
            continue
        g = x.groupby("q").title.agg(["mean", "size"])
        style.append({"team": names[t], "is_you": str(t) == str(you),
                      "rates": [float(g.loc[q, "mean"]) if q in g.index else None for q in range(5)],
                      "n": [int(g.loc[q, "size"]) if q in g.index else 0 for q in range(5)]})
    return {"labels": OVERBID_LABELS, "steps": OVERBID_STEPS, "stud_fair": STUD_FAIR, "cap": OVERBID_CAP,
            "tables": tables, "capped": capped, "style": style, "week": int(weeks[0])}
