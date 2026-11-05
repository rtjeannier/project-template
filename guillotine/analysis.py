"""Turn simulation runs into decisions.

Compares scenarios on the same random draws:
  * baseline — everyone, including you, bids using the model
  * pass     — you sit out this week's waivers
  * bid $Y on X — your sealed bid enters the auction; you win X only if it beats the field
  * bid sheet — all recommended bids at once (you never skip a player: a $1 bid can steal one)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .models import LeagueState
from .simulate import SimResults, Simulator, UserPolicy

WIN_TARGETS = (0.5, 0.75, 0.9)  # bid levels reported: P(beat every opponent)
GRID_TARGETS = (0.25, 0.5, 0.75, 0.9)  # bid levels tried, besides $1


@dataclass
class Candidate:
    player_id: str
    gain: float  # weekly starting-lineup points you'd gain
    bid_for: dict[float, int]  # target win probability -> bid needed
    suggested_bid: int  # the bid (of those tried) with the best P(win league)
    p_win_at_suggested: float
    curve: list[tuple[int, float, float]] = field(default_factory=list)  # (bid, P(win bid), P(win league))
    d_champion: float = 0.0  # change in P(win league) vs passing
    d_survive_3: float = 0.0  # change in P(alive after the next 3 weeks) vs passing
    champion: float = 0.0
    survive_3: float = 0.0


@dataclass
class Analysis:
    state: LeagueState
    sim: Simulator
    baseline: SimResults
    passed: SimResults
    candidates: list[Candidate] = field(default_factory=list)
    blocking: list[dict] = field(default_factory=list)
    sheet: dict[str, int] = field(default_factory=dict)
    sheet_result: SimResults | None = None
    holes: list[dict] = field(default_factory=list)
    future_targets: list[dict] = field(default_factory=list)
    opponents: list[dict] = field(default_factory=list)


def bid_needed(opp_bids: np.ndarray, p_win: float) -> int:
    """Smallest whole-dollar bid that beats the top opposing bid with prob ``p_win``.

    Ties are treated as losses (tiebreak usually goes against you or is a coin flip).
    """
    return int(np.quantile(opp_bids, p_win)) + 1


def p_win_at(opp_bids: np.ndarray, bid: int) -> float:
    return float(np.mean(opp_bids < bid))


def survive_through(res: SimResults, team_id: str, n_weeks: int) -> float:
    alive = res.p_alive_after(team_id)
    return float(alive[min(n_weeks, len(alive)) - 1])


FUTURE_SHARES = (0.25, 0.5, 0.75, 1.0)  # share of your FAAB bid when a future target hits waivers


def blocking_analysis(sim: Simulator, baseline: SimResults, targets: list[dict], n_sims: int) -> list[dict]:
    """For players likely to hit waivers later: is it worth overbidding to keep them from rivals?

    Compares, on the same random draws:
      pass    - you never bid on him; rivals bid as usual
      embargo - he's removed when he hits waivers (nobody gets him)
      bid X%  - whenever he hits waivers you bid X% of your FAAB at that time
    blocking value = P(win | nobody gets him) - P(win | you pass)
    own-use value  = P(win | best bid) - P(win | nobody gets him)
    """
    me = sim.user
    pid_of = {p.name: p.id for p in sim.state.players.values()}
    rows = []
    for t in targets:
        pid = pid_of[t["player"]]
        passed = sim.run(n_sims, UserPolicy("auto", future_bids={pid: 0.0}))
        embargo = sim.run(n_sims, UserPolicy("auto", embargo=frozenset({pid})))
        curve = []
        for share in FUTURE_SHARES:
            r = sim.run(n_sims, UserPolicy("auto", future_bids={pid: share}))
            got = sum(1 for s, j, p, tid, a in r.awards if p == pid and tid == me and j > 0)
            paid = [a for s, j, p, tid, a in r.awards if p == pid and tid == me and j > 0]
            could = len(r.user_could_bid.get(pid, ()))  # you can't miss him if you're already out
            curve.append({"share": share, "p_champ": r.p_champion(me), "p_get": got / could if could else 0.0,
                          "p_chance": could / n_sims,
                          "avg_paid": float(np.mean(paid)) if paid else 0.0})
        # Who ends up with him when you pass, and how strong they are
        owners = [tid for s, j, p, tid, a in passed.awards if p == pid and j > 0]
        top_owner = max(set(owners), key=owners.count) if owners else None
        best = max(curve, key=lambda c: c["p_champ"])
        rows.append({
            "player": t["player"], "pos": t["pos"], "typical_week": t["typical_week"],
            "p_pass": passed.p_champion(me), "p_embargo": embargo.p_champion(me),
            "block_value": embargo.p_champion(me) - passed.p_champion(me),
            "own_value": best["p_champ"] - embargo.p_champion(me),
            "curve": curve, "best_share": best["share"],
            "usual_owner": next((x.name for x in sim.state.teams if x.id == top_owner), "-"),
        })
    return rows


def analyze(state: LeagueState, n_sims: int = 400, top_k: int = 6, seed: int = 0,
            sim: Simulator | None = None, n_future: int = 3) -> Analysis:
    sim = sim or Simulator(state, seed=seed)
    me = state.my_team_id
    my_faab = int(state.team(me).faab)

    baseline = sim.run(n_sims, UserPolicy("auto"))
    passed = sim.run(n_sims, UserPolicy("pass"))
    out = Analysis(state=state, sim=sim, baseline=baseline, passed=passed)

    # ---- this week's waiver targets
    gains = sorted(baseline.user_gain_now.items(), key=lambda kv: -kv[1])
    for pid, gain in gains:
        if gain < sim.model.min_gain or len(out.candidates) >= top_k:
            continue
        opp = np.array(baseline.opp_max_bid[pid])
        bid_for = {t: bid_needed(opp, t) for t in WIN_TARGETS}
        levels = sorted({max(1, state.settings.min_bid)} | {min(bid_needed(opp, t), my_faab) for t in GRID_TARGETS})
        curve = []
        for amt in levels:
            r = sim.run(n_sims, UserPolicy("bid", bids={pid: amt}))
            curve.append((amt, p_win_at(opp, amt), r.p_champion(me), survive_through(r, me, 3)))
        best = max(curve, key=lambda c: (round(c[2], 4), -c[0]))
        c = Candidate(
            player_id=pid,
            gain=gain,
            bid_for=bid_for,
            suggested_bid=best[0],
            p_win_at_suggested=best[1],
            curve=[(a_, pw, ch) for a_, pw, ch, _ in curve],
            champion=best[2],
            survive_3=best[3],
        )
        c.d_champion = c.champion - passed.p_champion(me)
        c.d_survive_3 = c.survive_3 - survive_through(passed, me, 3)
        out.candidates.append(c)

    # Never skip: bid on every candidate at its best level, then run the whole sheet.
    out.sheet = {c.player_id: c.suggested_bid for c in out.candidates}
    if out.sheet:
        out.sheet_result = sim.run(n_sims, UserPolicy("bid", bids=out.sheet))

    out.holes = roster_holes(sim)
    out.future_targets = future_targets(sim, baseline)
    if n_future > 0:
        out.blocking = blocking_analysis(sim, baseline, out.future_targets[:n_future], n_sims)
    out.opponents = opponent_table(sim, baseline)
    return out


def roster_holes(sim: Simulator) -> list[dict]:
    """Your starter at each slot vs the median starter at that slot across the league."""
    proj, spec = sim.proj, sim.spec
    vb = proj.value_basis[0]
    lineups = {tid: spec.assign(r, vb, proj.pos) for tid, r in sim.init_rosters.items()}
    mine = lineups[sim.user]
    rows = []
    for k, (label, p) in enumerate(mine):
        league = [vb[lu[k][1]] if lu[k][1] is not None else 0.0 for lu in lineups.values()]
        you = vb[p] if p is not None else 0.0
        rows.append({
            "slot": label,
            "player": sim.state.players[proj.ids[p]].name if p is not None else "(empty)",
            "you": you,
            "league_median": float(np.median(league)),
            "gap": you - float(np.median(league)),
            "rank": 1 + sum(v > you for v in league),
            "of": len(league),
        })
    return rows


def future_targets(sim: Simulator, res: SimResults, top: int = 15) -> list[dict]:
    """Players on other rosters who are likely to hit waivers later, and what they'd cost."""
    me = sim.user
    mine = set(sim.state.team(me).players)
    prices: dict[tuple[int, int, str], tuple[str, int]] = {
        (s, j, pid): (tid, amt) for s, j, pid, tid, amt in res.awards
    }
    rows = []
    for pid, entries in res.pool_entries.items():
        if pid in mine or pid in sim.state.waiver_pool:
            continue
        sims_seen = {s for s, _, _ in entries}
        first = {}
        for s, j, g in entries:
            first.setdefault(s, (j, g))
        gains = [g for j, g in first.values() if not np.isnan(g)]
        if not gains:
            continue
        med_gain = float(np.median(gains))
        if med_gain < sim.model.min_gain:
            continue
        paid = [prices[(s, j, pid)][1] for s, (j, _) in first.items() if (s, j, pid) in prices]
        won = sum(1 for s, (j, _) in first.items() if prices.get((s, j, pid), ("",))[0] == me)
        weeks = [res.weeks[min(j, len(res.weeks) - 1)] for j, _ in first.values()]
        owner = next(t.name for t in sim.state.teams if pid in t.players)
        rows.append({
            "player": sim.state.players[pid].name,
            "pos": sim.state.players[pid].pos,
            "owner": owner,
            "p_available": len(sims_seen) / res.n_sims,
            "typical_week": int(np.median(weeks)),
            "your_gain": med_gain,
            "median_price": float(np.median(paid)) if paid else 0.0,
            "p_you_win": won / len(res.user_could_bid[pid]) if res.user_could_bid.get(pid) else 0.0,
        })
    rows.sort(key=lambda r: -(r["p_available"] * r["your_gain"]))
    return rows[:top]


def opponent_table(sim: Simulator, res: SimResults) -> list[dict]:
    proj, spec = sim.proj, sim.spec
    vb = proj.value_basis[0]
    lineups = {tid: spec.assign(r, vb, proj.pos) for tid, r in sim.init_rosters.items()}
    n_slots = len(next(iter(lineups.values())))
    medians = [np.median([vb[lu[k][1]] if lu[k][1] is not None else 0 for lu in lineups.values()])
               for k in range(n_slots)]
    rows = []
    for t in sim.state.teams:
        lu = lineups[t.id]
        gaps = [((vb[p] if p is not None else 0) - medians[k], label) for k, (label, p) in enumerate(lu)]
        weakest = sorted(gaps)[:2]
        rows.append({
            "team": t.name,
            "is_you": t.id == sim.user,
            "faab": t.faab,
            "proj_weekly": float(sum(vb[p] for _, p in lu if p is not None)),
            "p_elim_next": float(res.elim_counts[t.id][0] / res.n_sims),
            "p_champion": res.p_champion(t.id),
            "needs": ", ".join(label for g, label in weakest if g < -0.5) or "-",
        })
    rows.sort(key=lambda r: -r["proj_weekly"])
    return rows
