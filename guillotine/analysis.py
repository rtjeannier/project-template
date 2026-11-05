"""Turn simulation runs into decisions.

Compares scenarios on the same random draws:
  * baseline — everyone, including you, bids using the model
  * pass     — you sit out this week's waivers
  * bid $Y on X — your sealed bid enters the auction; you win X only if it beats the field
  * targets — for this week's best pool players and the irreplaceable players likely to
    hit waivers later: "whenever he's available and I'm alive, bid by this rule", for
    several X (X = 0 is never bidding; the smallest is a $1 lottery ticket). The best X
    for each target forms the plan, which is then run as a whole.
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
    targets: list[dict] = field(default_factory=list)
    plan: dict[str, str | float] = field(default_factory=dict)
    plan_result: SimResults | None = None
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


# Bids tried for each target, whenever he's on waivers and you're alive: multiples of your
# fair bid (the rational bid for you at that moment, no noise), plus none, all-in, and lock.
# Where P(win league) peaks shows whether over- or under-bidding pays for that player.
TARGET_RULES = ("none", "x0.5", "x0.75", "x1", "x1.25", "x1.5", "x2", "x3", "all-in", "lock")


def final_edge(sim: Simulator) -> dict[str, float]:
    """Each player's projected points per week above week-17 replacement (two teams left).
    Positive = irreplaceable in the final: he counts for you and against your opponent."""
    proj = sim.proj
    vb = proj.value_basis[-1]
    repl = sim._replacement(vb, set(range(len(proj.ids))), 2)
    return {pid: float(vb[i] - repl.get(proj.pos[i], 0.0)) for i, pid in enumerate(proj.ids)}


def pick_targets(sim: Simulator, baseline: SimResults, n_now: int, n_future: int) -> list[str]:
    """This week's best pool players for you, plus the irreplaceable players on other
    rosters most likely to hit waivers while you can still bid."""
    me = sim.user
    edge = final_edge(sim)
    now = [pid for pid, g in sorted(baseline.user_gain_now.items(), key=lambda kv: -kv[1])
           if g >= sim.model.min_gain or edge[pid] > 0][:n_now]
    mine = set(sim.state.team(me).players) | set(sim.state.waiver_pool) | set(sim.state.free_agents)
    future = []
    for pid, sims_seen in baseline.user_could_bid.items():
        if pid in mine or edge.get(pid, 0) <= 0:
            continue
        future.append((len(sims_seen) / baseline.n_sims * edge[pid], pid))
    future = [pid for _, pid in sorted(future, reverse=True)[:n_future]]
    return now + [p for p in future if p not in now]


_SIM: Simulator | None = None  # shared with forked workers


def _rule_point(args) -> dict:
    """One (target, bid) scenario; runs in a worker process."""
    pid, rule, n_sims = args
    sim = _SIM
    me = sim.user
    r = sim.run(n_sims, UserPolicy("auto", rules={pid: rule}))
    mine = [(j, a) for s, j, p, tid, a in r.awards if p == pid and tid == me]
    could = len(r.user_could_bid.get(pid, ()))
    after = r.user_rule_after.get(pid, [])
    return {
        "player_id": pid, "rule": rule, "p_champ": r.p_champion(me), "p_alive3": survive_through(r, me, 3),
        "p_get": len(mine) / could if could else 0.0, "p_chance": could / n_sims,
        "avg_paid": float(np.mean([a for _, a in mine])) if mine else 0.0,
        "weeks": [r.weeks[j] for j, _ in mine],
        "bid_median": float(np.median(r.user_rule_bids[pid])) if r.user_rule_bids.get(pid) else 0.0,
        "fair_median": float(np.median(r.user_fair_bids[pid])) if r.user_fair_bids.get(pid) else None,
        "left": float(np.median([x for x, _ in after])) if after else None,
        "richest": float(np.median([y for _, y in after])) if after else None,
    }


def target_analysis(sim: Simulator, targets: list[str], n_sims: int, workers: int = 1) -> list[dict]:
    """For each target, simulate each bid in TARGET_RULES ("whenever he's on waivers and
    I'm alive, bid this") on the same random draws, and keep the one with the best
    P(win league). Scenarios are independent, so they run in parallel."""
    global _SIM
    _SIM = sim
    tasks = [(pid, rule, n_sims) for pid in targets for rule in TARGET_RULES]
    if workers > 1:
        import multiprocessing as mp

        with mp.get_context("fork").Pool(workers) as pool:
            points = pool.map(_rule_point, tasks, chunksize=1)
    else:
        points = [_rule_point(t) for t in tasks]
    edge = final_edge(sim)
    rows = []
    for pid in targets:
        curve = [c for c in points if c["player_id"] == pid]
        best = max(curve, key=lambda c: (round(c["p_champ"], 4), -TARGET_RULES.index(c["rule"])))
        owner = next((t.name for t in sim.state.teams if pid in t.players), "waivers now")
        fairs = [c["fair_median"] for c in curve if c.get("fair_median") is not None]
        rows.append({"player_id": pid, "owner": owner, "edge": edge.get(pid, 0.0), "curve": curve,
                     "fair": float(np.median(fairs)) if fairs else None,
                     "best": best, "gain_vs_pass": best["p_champ"] - curve[0]["p_champ"]})
    rows.sort(key=lambda r: -r["gain_vs_pass"])
    return rows


def analyze(state: LeagueState, n_sims: int = 400, top_k: int = 6, seed: int = 0,
            sim: Simulator | None = None, n_future: int = 5, workers: int = 1) -> Analysis:
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
        out.candidates.append(Candidate(player_id=pid, gain=gain, bid_for=bid_for,
                                        suggested_bid=0, p_win_at_suggested=0.0))

    # Who to target, and how much of your FAAB to bid whenever each one is available
    targets = pick_targets(sim, baseline, n_now=2, n_future=n_future)
    out.targets = target_analysis(sim, targets, n_sims, workers)
    out.plan = {t["player_id"]: t["best"]["rule"] for t in out.targets}
    out.plan_result = sim.run(n_sims, UserPolicy("auto", rules=dict(out.plan)))

    out.holes = roster_holes(sim)
    out.future_targets = future_targets(sim, baseline)
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
        owner = next((t.name for t in sim.state.teams if pid in t.players), "free agent now")
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
