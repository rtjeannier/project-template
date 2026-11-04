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
        self.kappa = self.model.pace_scale if self.model.pace_scale is not None else 1.0
        if self.model.pace_scale is None:
            self.calibrate()

    def calibrate(self, n_probe: int = 60, iters: int = 4) -> float:
        """Pick the bid scale so the first round's actual spend matches the pacing plan."""
        for _ in range(iters):
            actual = planned = 0.0
            for s in range(n_probe):
                a, p = self._run_one(10_000 + s, UserPolicy("auto"), self._scratch_results(), probe=True)
                actual += a
                planned += p
            if actual <= 0 or planned <= 0:
                break
            self.kappa *= float(np.clip(planned / actual, 0.25, 4.0))
        return self.kappa

    def _scratch_results(self) -> SimResults:
        n_weeks = len(self.proj.weeks)
        return SimResults(
            n_sims=0, team_ids=[], weeks=self.proj.weeks,
            elim_counts={t.id: np.zeros(n_weeks) for t in self.state.teams},
            champion_counts=defaultdict(int), opp_max_bid=defaultdict(list), user_gain_now={},
        )

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

    def _run_one(self, sim: int, policy: UserPolicy, res: SimResults, probe: bool = False):
        rng = np.random.default_rng((self.seed, sim, 0))  # bidding / manager noise
        # Player game outcomes come from a separate stream so every scenario sees the
        # same "luck" for each player-week; scenarios then differ only by decisions.
        z = np.random.default_rng((self.seed, sim, 1)).standard_normal(self.proj.means.shape)
        points = np.clip(self.proj.means + self.proj.sd * z, 0, None)
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
            start_money = sum(faab[t] for t in active)
            pool, fa_pool = self._waiver_round(
                sim, j, rng, rosters, faab, aggression, active, pool, fa_pool, policy, res
            )
            if probe:
                return start_money - sum(faab[t] for t in active), self._planned_spend

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
        gv = proj.generic_value[:, j]
        weeks_left = len(proj.weeks) - j

        candidates = sorted(set(pool) | set(fa_pool), key=lambda p: -vb[p])[: model.pool_consider]
        remaining = set(candidates)

        baselines = {tid: spec.total(rosters[tid], vb, proj.pos) for tid in active}
        order = sorted(active, key=lambda t: baselines[t])
        danger = {tid: model.danger(order.index(tid), len(active)) for tid in active}
        avg_faab = float(np.mean([faab[t] for t in active])) if active else 0.0
        price = self._price_per_unit(j, candidates, rosters, active, faab, baselines)
        priority = {tid: r for r, tid in enumerate(rng.permutation(active))}
        claimed: set[int] = set()

        def team_bids(tid: str) -> dict[int, int]:
            out = {}
            base = spec.total(rosters[tid], vb, proj.pos)
            for p in remaining:
                gain = spec.total(rosters[tid] + [p], vb, proj.pos) - base
                if tid == self.user and j > 0:
                    self._note_user_gain(res, sim, j, proj.ids[p], gain)
                if gain < model.min_gain:
                    continue
                amt = model.bid(
                    price_per_unit=price,
                    generic_value=gv[p],
                    need=model.need(gain, proj.weekly_vor[j, p]),
                    danger=danger[tid],
                    faab=faab[tid],
                    avg_faab=avg_faab,
                    aggression=aggression[tid],
                    noise=rng.lognormal(0, model.bid_noise_sd),
                    weeks_left=weeks_left,
                    min_bid=settings.min_bid,
                )
                if amt >= 0:
                    out[p] = amt
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

    def _price_per_unit(self, j, candidates, rosters, active, faab, baselines) -> float:
        """Dollars per unit of generic value, given money left vs value still to come."""
        proj = self.proj
        money = sum(faab[t] for t in active)
        supply = float(sum(proj.generic_value[p, j] for p in candidates))
        n_weeks = len(proj.weeks)
        if j + 1 < n_weeks and len(active) > 2:
            # Future pools look like the rosters of the currently weakest teams.
            weak = sorted(active, key=lambda t: baselines[t])[: max(1, len(active) // 3)]
            n_keep = self.spec.n_starters + 2
            for k in range(j + 1, n_weeks):
                if len(active) - (k - j) * self.state.settings.eliminations_per_week < 2:
                    break
                vals = []
                for t in weak:
                    top = sorted((proj.generic_value[p, k] for p in rosters[t]), reverse=True)
                    vals.append(sum(top[:n_keep]))
                supply += float(np.mean(vals)) * self.state.settings.eliminations_per_week
        if supply <= 0:
            self._planned_spend = 0.0
            return 0.0
        now = float(sum(proj.generic_value[p, j] for p in candidates))
        self._planned_spend = self.model.spend_ratio * money * now / supply
        return self.kappa * self.model.spend_ratio * money / supply
