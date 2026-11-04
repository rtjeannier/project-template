"""Monte Carlo simulation of the rest of a guillotine season.

Each simulated week:
  1. Waivers: every active team bids on the pool (last week's eliminated roster plus
     leftover free agents). The highest bids are processed first, budgets are
     enforced, and a team that wins re-values the remaining players before its
     next claim is considered.
  2. Games: each team starts its best projected lineup; scores are sampled.
  3. Guillotine: the lowest scorer(s) are eliminated and their roster becomes the
     next pool. Their unspent FAAB disappears.

The simulation can force a decision for your team in the first waiver round
("pass", or "win player X for $Y") so scenarios can be compared on identical
random draws.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .bidding import BidModel
from .lineup import LineupSpec
from .models import LeagueState
from .projections import Projections

FA_POOL_KEEP = 30  # leftover free agents carried between weeks (by projection)
ROS_ERROR_SD = 0.20  # lognormal sd of a player's season-long deviation from his projection
FLEX_SHARE = {"RB": 0.4, "WR": 0.5, "TE": 0.1}  # how flex starts split by position


@dataclass
class UserPolicy:
    """What your team does in the first waiver round. Later rounds always use the model."""

    mode: str = "auto"  # "auto" | "pass" | "force"
    player: str | None = None
    price: int = 0


@dataclass
class SimResults:
    n_sims: int
    team_ids: list[str]
    weeks: list[int]
    elim_counts: dict[str, np.ndarray]  # team -> count eliminated in week index j
    champion_counts: dict[str, int]
    # First waiver round
    opp_max_bid: dict[str, list[int]]  # player -> top opposing bid per sim
    user_gain_now: dict[str, float]  # player -> weekly lineup points gain for you
    # All waiver awards: (sim, week_idx, player, team, price)
    awards: list[tuple[int, int, str, str, int]] = field(default_factory=list)
    # Players entering a waiver pool after week 0: player -> list of (sim, week_idx, your gain)
    pool_entries: dict[str, list[tuple[int, int, float]]] = field(default_factory=dict)
    user_final_faab: list[float] = field(default_factory=list)

    def p_alive_after(self, team_id: str) -> np.ndarray:
        """P(team still alive after each week)."""
        return 1.0 - np.cumsum(self.elim_counts[team_id]) / self.n_sims

    def p_champion(self, team_id: str) -> float:
        return self.champion_counts.get(team_id, 0) / self.n_sims


class Simulator:
    def __init__(self, state: LeagueState, bid_model: BidModel | None = None, seed: int = 0):
        self.state = state
        self.model = bid_model or BidModel()
        self.seed = seed
        self.proj = Projections(state)
        self.spec = LineupSpec(state.settings.roster_slots)
        self.user = state.my_team_id
        idx = self.proj.index
        self.init_rosters = {t.id: [idx[p] for p in t.players] for t in state.teams}
        self.init_pool = [idx[p] for p in state.waiver_pool]
        self.init_fa = [idx[p] for p in state.free_agents if p not in set(state.waiver_pool)]
        # Keep the initial free-agent list small: only the best by value matter.
        self.init_fa.sort(key=lambda i: -self.proj.value_basis[0][i])
        self.init_fa = self.init_fa[:FA_POOL_KEEP]
        self.bid_log: list | None = None  # set to [] to record (sim, week_idx, player, team, bid, faab)

    # ------------------------------------------------------------------ public

    def run(self, n_sims: int = 500, policy: UserPolicy | None = None) -> SimResults:
        policy = policy or UserPolicy()
        n_weeks = len(self.proj.weeks)
        res = SimResults(
            n_sims=n_sims,
            team_ids=[t.id for t in self.state.teams],
            weeks=self.proj.weeks,
            elim_counts={t.id: np.zeros(n_weeks) for t in self.state.teams},
            champion_counts=defaultdict(int),
            opp_max_bid=defaultdict(list),
            user_gain_now={},
        )
        for s in range(n_sims):
            self._run_one(s, policy, res)
        res.champion_counts = dict(res.champion_counts)
        res.opp_max_bid = dict(res.opp_max_bid)
        return res

    # --------------------------------------------------------------- one season

    def _run_one(self, sim: int, policy: UserPolicy, res: SimResults):
        rng = np.random.default_rng((self.seed, sim, 0))  # bidding / manager noise
        # Player game outcomes come from a separate stream so every scenario sees the
        # same "luck" for each player-week; scenarios then differ only by decisions.
        luck = np.random.default_rng((self.seed, sim, 1))
        z = luck.standard_normal(self.proj.means.shape)
        # Season-long projection error: a player's true level differs from his projection
        # for the whole season (breakouts, busts, role changes), on top of weekly noise.
        true_level = luck.lognormal(-ROS_ERROR_SD**2 / 2, ROS_ERROR_SD, size=(self.proj.means.shape[0], 1))
        points = np.clip(self.proj.means * true_level + self.proj.sd * z, 0, None)
        proj, spec, model = self.proj, self.spec, self.model
        rosters = {tid: list(r) for tid, r in self.init_rosters.items()}
        faab = {t.id: t.faab for t in self.state.teams}
        aggression = {
            t.id: t.aggression * (1.0 if t.id == self.user else rng.lognormal(0, model.manager_sd))
            for t in self.state.teams
        }
        active = [t.id for t in self.state.teams]
        fa_pool = list(self.init_fa)
        pool = list(self.init_pool)
        last_scores: dict[str, float] = {}

        for j in range(len(proj.weeks)):
            if len(active) <= 1:
                break
            pool, fa_pool = self._waiver_round(
                sim, j, rng, rosters, faab, aggression, active, pool, fa_pool, policy, res
            )

            # Games
            scores = {}
            means = proj.means[:, j]
            for tid in active:
                _, starters = spec.best(rosters[tid], means, proj.pos)
                scores[tid] = float(points[starters, j].sum())
            last_scores = scores

            n_elim = min(self.state.settings.eliminations_per_week, len(active) - 1)
            losers = sorted(active, key=lambda t: scores[t])[:n_elim]
            dumped = []
            for tid in losers:
                res.elim_counts[tid][j] += 1
                active.remove(tid)
                dumped.extend(rosters[tid])
                rosters[tid] = []
                faab[tid] = 0
            pool = dumped
            for p in dumped:
                res.pool_entries.setdefault(proj.ids[p], []).append((sim, j + 1, np.nan))

        if len(active) == 1:
            champ = active[0]
        else:
            champ = max(active, key=lambda t: last_scores.get(t, 0.0))
        res.champion_counts[champ] += 1
        res.user_final_faab.append(faab[self.user])

    # ------------------------------------------------------------ waiver round

    def _waiver_round(self, sim, j, rng, rosters, faab, aggression, active, pool, fa_pool, policy, res):
        proj, spec, model = self.proj, self.spec, self.model
        settings = self.state.settings
        vb = proj.value_basis[j]
        n_weeks = len(proj.weeks)
        weeks_left = n_weeks - j

        candidates = sorted(set(pool) | set(fa_pool), key=lambda p: -vb[p])[: model.pool_consider]
        remaining = set(candidates)

        universe = set(pool) | set(fa_pool) | {p for t in active for p in rosters[t]}
        e = settings.eliminations_per_week
        # Rounds still to play (this week = r 0), with the number of teams alive in each.
        n_alive = [len(active) - r * e for r in range(weeks_left) if len(active) - r * e >= 2]
        repl = {n: self._replacement(vb, universe, n) for n in set(n_alive)}
        totals_now = {tid: spec.total(rosters[tid], proj.means[:, j], proj.pos) for tid in active}
        weight = self._elimination_weights(model.p_cut(totals_now), n_alive)
        future = self._future_pool(vb, rosters, active, totals_now)
        priority = {tid: r for r, tid in enumerate(rng.permutation(active))}
        claimed: set[int] = set()

        def free_gains(tid: str, base: float) -> list[dict[str, float]]:
            """Per round: lineup gain from the best player at each position nobody will
            pay for (the replacement level for that many teams alive)."""
            vals = {p: vb[p] for p in rosters[tid]}
            pos = {p: proj.pos[p] for p in rosters[tid]}
            cache = {}
            for n in set(n_alive):
                out = {}
                for ps, v in repl[n].items():
                    vals[-1], pos[-1] = v, ps
                    out[ps] = spec.total(list(rosters[tid]) + [-1], vals, pos) - base
                cache[n] = out
            return [cache[n] for n in n_alive]

        def weighted_par(tid: str, gain: float, ps: str, free: list[dict[str, float]], start: int = 0) -> float:
            """Points above replacement in each round, weighted by P(this team is the one
            cut that round). Points only matter when they decide who is eliminated."""
            w = weight[tid]
            return float(sum(w[r] * max(0.0, gain - free[r].get(ps, 0.0)) for r in range(start, len(n_alive))))

        def opportunity(tid: str, base: float, free: list[dict[str, float]]) -> float:
            """Weighted upgrades this team expects to buy in later rounds instead."""
            best, best_pos = 0.0, None
            for p in future:
                g = spec.total(rosters[tid] + [p], vb, proj.pos) - base
                if g > best:
                    best, best_pos = g, proj.pos[p]
            if best_pos is None:
                return 0.0
            return sum(min(1.0, model.future_share / n_alive[r]) * weighted_par(tid, best, best_pos, free, r)
                       for r in range(1, len(n_alive)))

        def team_bids(tid: str) -> dict[int, int]:
            out = {}
            base = spec.total(rosters[tid], vb, proj.pos)
            free = free_gains(tid, base)
            opp = opportunity(tid, base, free)
            for p in remaining:
                gain = spec.total(rosters[tid] + [p], vb, proj.pos) - base
                if tid == self.user and j > 0:
                    self._note_user_gain(res, sim, j, proj.ids[p], gain)
                if gain - free[0].get(proj.pos[p], 0.0) < model.min_gain and gain - free[-1].get(proj.pos[p], 0.0) < model.min_gain:
                    continue
                value = weighted_par(tid, gain, proj.pos[p], free)
                if value <= 0:
                    continue
                amt = model.bid(
                    value=value,
                    opportunity=opp,
                    faab=faab[tid],
                    aggression=aggression[tid],
                    noise=rng.lognormal(0, model.bid_noise_sd),
                    min_bid=settings.min_bid,
                )
                if amt >= 0:
                    out[p] = amt
                    if self.bid_log is not None:
                        self.bid_log.append((sim, j, proj.ids[p], tid, amt, faab[tid]))
            return out

        bids = {tid: team_bids(tid) for tid in active}

        if j == 0:
            if sim == 0:
                base = spec.total(rosters[self.user], vb, proj.pos)
                for p in candidates:
                    g = spec.total(rosters[self.user] + [p], vb, proj.pos) - base
                    res.user_gain_now[proj.ids[p]] = g
            for p in candidates:
                opp = [bids[t][p] for t in active if t != self.user and p in bids[t]]
                res.opp_max_bid[proj.ids[p]].append(max(opp) if opp else -1)

            if policy.mode in ("pass", "force") and self.user in bids:
                bids[self.user] = {}
            if policy.mode == "force" and self.user in active:
                p = proj.index[policy.player]
                if p in remaining and faab[self.user] >= policy.price:
                    self._award(sim, j, p, self.user, policy.price, rosters, faab, fa_pool, remaining, claimed, res, vb)

        while True:
            best = None
            for tid in active:
                for p, amt in bids[tid].items():
                    if p not in remaining or amt > faab[tid]:
                        continue
                    key = (amt, -priority[tid])
                    if best is None or key > best[0]:
                        best = (key, tid, p)
            if best is None:
                break
            (amt, _), tid, p = best
            self._award(sim, j, p, tid, amt, rosters, faab, fa_pool, remaining, claimed, res, vb)
            # Priority rolls to the back after a successful claim.
            priority[tid] = max(priority.values()) + 1
            if not (j == 0 and tid == self.user and policy.mode != "auto"):
                bids[tid] = team_bids(tid)

        leftovers = [p for p in set(pool) | set(fa_pool) if p not in claimed]
        leftovers.sort(key=lambda p: -vb[p])
        return [], leftovers[:FA_POOL_KEEP]

    def _award(self, sim, j, p, tid, amt, rosters, faab, fa_pool, remaining, claimed, res, vb):
        proj = self.proj
        faab[tid] -= amt
        rosters[tid].append(p)
        remaining.discard(p)
        claimed.add(p)
        if p in fa_pool:
            fa_pool.remove(p)
        res.awards.append((sim, j, proj.ids[p], tid, int(amt)))
        if len(rosters[tid]) > self.state.settings.roster_size:
            _, starters = self.spec.best(rosters[tid], vb, proj.pos)
            bench = [q for q in rosters[tid] if q not in starters and q != p]
            if bench:
                drop = min(bench, key=lambda q: vb[q])
                rosters[tid].remove(drop)
                fa_pool.append(drop)

    def _note_user_gain(self, res, sim, j, pid, gain):
        entries = res.pool_entries.get(pid)
        if entries and entries[-1][0] == sim and entries[-1][1] == j and np.isnan(entries[-1][2]):
            entries[-1] = (sim, j, gain)

    def _replacement(self, vb, universe, n_active: int) -> dict[str, float]:
        """Dynamic replacement level: the best player at each position the remaining
        teams won't start. It rises as teams are cut and their rosters hit waivers."""
        slots = self.state.settings.roster_slots
        flex = sum(c for s, c in slots.items() if s in ("FLEX", "W/R/T", "W/R", "W/T", "R/W"))
        superflex = sum(c for s, c in slots.items() if s in ("SUPERFLEX", "Q/W/R/T"))
        by_pos: dict[str, list[float]] = {}
        for p in universe:
            by_pos.setdefault(self.proj.pos[p], []).append(vb[p])
        out = {}
        for pos, vals in by_pos.items():
            starters = slots.get(pos, 0) + flex * FLEX_SHARE.get(pos, 0)
            starters += superflex * (0.8 if pos == "QB" else 0.07)
            k = int(round(n_active * starters))
            vals.sort(reverse=True)
            out[pos] = vals[k] if k < len(vals) else 0.0
        return out

    @staticmethod
    def _elimination_weights(p_cut: dict[str, float], n_alive: list[int]) -> dict[str, list[float]]:
        """P(team t is the one eliminated in round r), for each remaining round.

        This week's elimination chances carry forward: each later round removes the
        teams most likely to have been cut and renormalizes the rest. The last round
        (two teams) is the final, where "eliminated" means losing it.
        """
        order = sorted(p_cut, key=lambda t: -p_cut[t])
        out = {}
        for t in p_cut:
            others = [u for u in order if u != t]
            alive, w = 1.0, []
            for n in n_alive:
                rivals = others[len(others) - (n - 1):] if n - 1 > 0 else []
                denom = p_cut[t] + sum(p_cut[u] for u in rivals)
                q = p_cut[t] / denom if denom > 0 else 0.0
                w.append(alive * q)
                alive *= 1 - q
            out[t] = w
        return out

    def _future_pool(self, vb, rosters, active, totals) -> list[int]:
        """Players likely to hit waivers later: the weakest teams' best players."""
        if len(active) <= 2:
            return []
        weak = sorted(active, key=lambda t: totals[t])[: max(1, len(active) // 3)]
        n_keep = self.spec.n_starters + 2
        out = []
        for t in weak:
            out += sorted(rosters[t], key=lambda p: -vb[p])[:n_keep]
        return out
