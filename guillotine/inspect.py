"""Export every simulated scenario for one target so it can be inspected.

``python -m guillotine inspect LEAGUE.json --player "Jahmyr Gibbs"`` runs the player's
bid levels (multiples of your fair bid, plus none / all-in / lock) on the same random
draws and writes, to ``--out``:

- ``seasons.csv``: one row per (bid level, simulated season): whether he hit waivers
  while you were alive, your fair bid and actual bid, the top rival bid, who won him and
  at what price, your FAAB after vs the richest rival's, the week you were cut, the
  champion, and whether you won the league.
- ``bids.csv``: every team's bid on him each time he was awarded, with their FAAB.
- ``timelines.json``: week-by-week detail for the first ``--timelines`` seasons of each
  bid level: every award (with all bids), scores, cuts and FAAB balances.
- ``inspect.json``: all of the above in one file (used by the HTML viewer).
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from .analysis import TARGET_RULES
from .models import LeagueState
from .simulate import Simulator, UserPolicy


def _label(rule) -> str:
    if isinstance(rule, str) and rule.startswith("x"):
        return f"{rule[1:]}x fair"
    return str(rule)


def inspect_player(state: LeagueState, player: str, n_sims: int = 300, n_timelines: int = 40,
                   seed: int = 0, rules=TARGET_RULES) -> dict:
    by_name = {p.name.lower(): p.id for p in state.players.values()}
    pid = by_name.get(player.lower(), player)
    if pid not in state.players:
        raise SystemExit(f"unknown player {player!r}")
    sim = Simulator(state, seed=seed)
    me = sim.user
    names = {t.id: t.name for t in state.teams}
    pname = {p.id: p.name for p in state.players.values()}
    seasons, bids_rows, timelines, summary = [], [], {}, []

    for rule in rules:
        sim.trace = []
        res = sim.run(n_sims, UserPolicy("auto", rules={pid: rule}))
        ev = defaultdict(list)
        for e in sim.trace:
            ev[e[1]].append(e)
        label = _label(rule)
        for s in range(n_sims):
            rows = ev[s]
            rule_ev = [e for e in rows if e[0] == "rule" and e[3] == pid]
            award = next((e for e in rows if e[0] == "award" and e[3] == pid), None)
            weeks = [e for e in rows if e[0] == "week"]
            cut = next((sim.proj.weeks[e[2]] for e in weeks if me in e[4]), None)
            champ = next((e[2] for e in rows if e[0] == "champ"), None)
            row = {
                "bid_level": label, "season": s,
                "could_bid": bool(rule_ev),
                "week_available": sim.proj.weeks[rule_ev[0][2]] if rule_ev else None,
                "fair_bid": rule_ev[0][4] if rule_ev else None,
                "your_bid": rule_ev[0][5] if rule_ev else None,
                "won_by": names.get(award[4]) if award else None,
                "week_won": sim.proj.weeks[award[2]] if award else None,
                "price": award[5] if award else None,
                "top_rival_bid": None, "top_rival": None, "your_faab_before": None,
                "your_faab_after": None, "richest_rival_faab_after": None,
                "you_cut_week": cut, "champion": names.get(champ), "you_won_league": champ == me,
            }
            if award:
                all_bids, faab_before = award[6], award[7]
                rivals = {t: b for t, b in all_bids.items() if t != me}
                if rivals:
                    top = max(rivals, key=rivals.get)
                    row["top_rival_bid"], row["top_rival"] = rivals[top], names[top]
                if me in faab_before:
                    row["your_faab_before"] = faab_before[me]
                    after = dict(faab_before)
                    after[award[4]] -= award[5]
                    row["your_faab_after"] = after[me]
                    row["richest_rival_faab_after"] = max((v for t, v in after.items() if t != me), default=0)
                for t, b in sorted(all_bids.items(), key=lambda kv: -kv[1]):
                    bids_rows.append({"bid_level": label, "season": s, "week": sim.proj.weeks[award[2]],
                                      "team": names[t], "bid": b, "faab": faab_before.get(t),
                                      "won": t == award[4], "is_you": t == me})
            seasons.append(row)
            if s < n_timelines:
                tl = []
                for e in rows:
                    if e[0] == "award":
                        tl.append({"type": "award", "week": sim.proj.weeks[e[2]], "player": pname[e[3]],
                                   "team": names[e[4]], "price": e[5],
                                   "bids": {names[t]: b for t, b in sorted(e[6].items(), key=lambda kv: -kv[1])},
                                   "target": e[3] == pid})
                    elif e[0] == "rule":
                        tl.append({"type": "your_bid", "week": sim.proj.weeks[e[2]], "player": pname[e[3]],
                                   "fair": e[4], "bid": e[5]})
                    elif e[0] == "week":
                        tl.append({"type": "week", "week": sim.proj.weeks[e[2]],
                                   "scores": {names[t]: round(v, 1) for t, v in sorted(e[3].items(), key=lambda kv: -kv[1])},
                                   "cut": [names[t] for t in e[4]],
                                   "faab": {names[t]: v for t, v in e[5].items()}})
                    elif e[0] == "champ":
                        tl.append({"type": "champion", "team": names[e[2]]})
                timelines[f"{label}|{s}"] = tl
        mine = [r for r in seasons if r["bid_level"] == label]
        could = [r for r in mine if r["could_bid"]]
        summary.append({
            "bid_level": label, "p_win_league": res.p_champion(me),
            "p_could_bid": len(could) / n_sims,
            "p_get_when_could": (sum(r["won_by"] == names[me] for r in could) / len(could)) if could else 0.0,
        })
        sim.trace = None

    return {"player": pname[pid], "you": names[me], "n_sims": n_sims, "summary": summary,
            "seasons": seasons, "bids": bids_rows, "timelines": timelines}


def write(data: dict, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, rows in (("seasons.csv", data["seasons"]), ("bids.csv", data["bids"])):
        p = out / name
        with p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["empty"])
            w.writeheader()
            w.writerows(rows)
        paths.append(p)
    (out / "timelines.json").write_text(json.dumps(data["timelines"]))
    (out / "inspect.json").write_text(json.dumps(data))
    return paths + [out / "timelines.json", out / "inspect.json"]
