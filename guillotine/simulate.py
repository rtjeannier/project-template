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
from statistics import NormalDist

import numpy as np

from .bidding import BidModel
from .lineup import LineupSpec
from .models import LeagueState
from .projections import Projections

FA_POOL_KEEP = 30  # leftover free agents carried between weeks (by projection)
# Projection uncertainty grows with the horizon: a player's true level is a random walk
# around his projection. About ±10% for this week, growing to about ±25% by week 17.
PROJ_ERROR_NOW = 0.10  # lognormal sd of this week's true level vs projection
PROJ_ERROR_STEP = 0.08  # extra lognormal sd added per week further out


def horizon_sd(weeks_ahead: int) -> float:
    """Projection uncertainty (lognormal sd) for a week this many weeks ahead."""
    return float(np.sqrt(PROJ_ERROR_NOW**2 + max(weeks_ahead, 0) * PROJ_ERROR_STEP**2))


_NORMAL = NormalDist()
REPL_DRAWS = np.random.default_rng(7).standard_normal((40, 400))  # fixed draws for uncertain replacement levels
# Injuries: weekly chance a player gets hurt in a game. A share of injuries end his
# season; the rest keep him out 1-4 weeks. Kept modest because ROS projections already
# price in some missed time.
INJURY_RATE = {"QB": 0.015, "RB": 0.025, "WR": 0.02, "TE": 0.02, "K": 0.005, "DEF": 0.0}
SEASON_ENDING_SHARE = 0.35
SHORT_INJURY_WEEKS = (1, 4)
ODDS_DRAWS = np.random.default_rng(20261104).standard_normal((2000, 32))  # fixed draws for P(lowest score)
FLEX_SHARE = {"RB": 0.4, "WR": 0.5, "TE": 0.1}  # how flex starts split by position


@dataclass
class UserPolicy:
    """What your team does in the first waiver round. Later rounds always use the model."""

    mode: str = "auto"  # "auto" | "pass" | "force" | "bid"
    player: str | None = None
    price: int = 0
    bids: dict[str, int] | None = None  # mode "bid": your sealed bids {player_id: amount}; others bid as modeled
    # Standing rules: whenever one of these players is on waivers (and you're alive), bid by
    # the rule: "none" (never bid), "$1", a number q in (0, 1) = enough to beat every
    # interested rival with probability q, estimated from what you can see (their FAAB,
    # their need, typical bidding; not their actual bids), or "lock" = top the richest
    # interested rival's whole budget. Applies on top of any mode.
    rules: dict[str, str | float] | None = None
    embargo: frozenset[str] = frozenset()  # players removed from the game when they hit waivers


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
    # player -> sims where he was on waivers in a later round while you were still alive to bid
    user_could_bid: dict[str, set[int]] = field(default_factory=dict)
    user_rule_bids: dict[str, list[int]] = field(default_factory=dict)  # what your standing rules bid

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
        self.inj_rate = np.array([INJURY_RATE.get(p, 0.02) for p in self.proj.pos])
        n_w = len(self.proj.weeks)
        self._avail = np.ones((len(self.proj.ids), n_w))  # per-sim: 1 if the player can play week k
        self._back = np.zeros(len(self.proj.ids), dtype=int)  # per-sim: first week index he's healthy again
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
        n_p, n_w = self.proj.means.shape
        steps = np.concatenate([luck.normal(-PROJ_ERROR_NOW**2 / 2, PROJ_ERROR_NOW, (n_p, 1)),
                                luck.normal(-PROJ_ERROR_STEP**2 / 2, PROJ_ERROR_STEP, (n_p, n_w - 1))], axis=1)
        true_level = np.exp(np.cumsum(steps, axis=1))  # random walk: later weeks are less certain
        points = np.clip(self.proj.means * true_level + self.proj.sd * z, 0, None)
        self._draw_injuries(luck)
        points *= self._avail
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
            means = proj.means[:, j] * self._avail[:, j]  # injured players are known before kickoff
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

    def _draw_injuries(self, luck) -> None:
        """At most one injury per player per season. He gets hurt in a game in week s and
        misses s+1 .. s+d (to the end of the season if season-ending). Managers know about
        an injury once it has happened, never in advance."""
        n_p, n_w = self.proj.means.shape
        hurt = luck.random((n_p, n_w)) < self.inj_rate[:, None]
        s = np.where(hurt.any(axis=1), hurt.argmax(axis=1), n_w)
        ending = luck.random(n_p) < SEASON_ENDING_SHARE
        d = np.where(ending, n_w, luck.integers(SHORT_INJURY_WEEKS[0], SHORT_INJURY_WEEKS[1] + 1, n_p))
        k = np.arange(n_w)[None, :]
        out = (k > s[:, None]) & (k <= (s + d)[:, None])
        self._avail = (~out).astype(float)
        self._back = np.where(s < n_w, np.minimum(s + d + 1, n_w), 0)

    def _known_value(self, j: int) -> np.ndarray:
        """Waiver value basis at week j, given injuries that have already happened: a hurt
        player is worth only the share of the remaining weeks he'll be back for."""
        vb = self.proj.value_basis[j].copy()
        hurt_now = self._avail[:, j] == 0
        if hurt_now.any():
            left = len(self.proj.weeks) - j
            vb[hurt_now] *= np.clip(len(self.proj.weeks) - self._back[hurt_now], 0, left) / left
        return vb

    @staticmethod
    def _beat_field(q: float, rivals: list[tuple[float, float]], sd: float, cap: int) -> int:
        """Smallest bid that tops every rival with probability q. Each rival's bid is
        lognormal around your estimate of it (sd = manager + bid noise) and can't exceed
        that rival's FAAB."""
        def p_top_all(b: int) -> float:
            p = 1.0
            for est, budget in rivals:
                if b > budget or est <= 0:
                    continue  # can't outbid you, or isn't really bidding
                p *= _NORMAL.cdf(np.log(b / est) / sd)
            return p
        lo, hi = 1, max(cap, 1)
        if p_top_all(hi) < q:
            return hi
        while lo < hi:
            mid = (lo + hi) // 2
            if p_top_all(mid) >= q:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def _cut_odds(self, rosters, active, j) -> dict[str, float]:
        """P(each team posts the lowest score in week j), from score distributions.

        Each team's score is approximately normal: the mean is its best lineup's
        projection, and the variance comes from its starters' weekly spread plus
        season-long projection error. A boom-or-bust lineup carries more cut risk
        than a steady one with the same projection.
        """
        proj = self.proj
        means = proj.means[:, j] * self._avail[:, j]
        mu, sd = [], []
        for t in active:
            _, starters = self.spec.best(rosters[t], means, proj.pos)
            m = means[starters]
            mu.append(m.sum())
            sd.append(float(np.sqrt((proj.sd[starters, j] ** 2).sum() + ((m * PROJ_ERROR_NOW) ** 2).sum())))
        scores = np.array(mu) + np.array(sd) * ODDS_DRAWS[:, : len(active)]
        counts = np.bincount(scores.argmin(axis=1), minlength=len(active)) + 0.5  # floor: never exactly 0
        return {t: float(c) / counts.sum() for t, c in zip(active, counts)}

    def _waiver_round(self, sim, j, rng, rosters, faab, aggression, active, pool, fa_pool, policy, res):
        proj, spec, model = self.proj, self.spec, self.model
        settings = self.state.settings
        vb = self._known_value(j)
        n_weeks = len(proj.weeks)
        weeks_left = n_weeks - j

        if policy.embargo:
            gone = {proj.index[pid] for pid in policy.embargo}
            pool = [p for p in pool if p not in gone]
            fa_pool = [p for p in fa_pool if p not in gone]
        candidates = sorted(set(pool) | set(fa_pool), key=lambda p: -vb[p])[: model.pool_consider]
        remaining = set(candidates)
        if self.user in active:
            for p in candidates:
                res.user_could_bid.setdefault(proj.ids[p], set()).add(sim)

        universe = set(pool) | set(fa_pool) | {p for t in active for p in rosters[t]}
        e = settings.eliminations_per_week
        # Rounds still to play (this week = r 0), with the number of teams alive in each.
        n_alive = [len(active) - r * e for r in range(weeks_left) if len(active) - r * e >= 2]
        # Replacement level for each round, as seen from now: further-out rounds are less
        # certain, so the best free player is likelier to be someone who broke out.
        repl = {n: self._replacement(vb, universe, n, horizon_sd(r)) for r, n in enumerate(n_alive)}
        # Odds as they stand before waivers (this week's projections, current rosters).
        # They're recalculated after every pickup below.
        odds = self._cut_odds(rosters, active, j)
        weight = self._elimination_weights(odds, n_alive)
        future = self._future_pool(vb, rosters, active, {t: -odds[t] for t in active})
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

        def weighted_par(tid: str, gain: float, ps: str, free: list[dict[str, float]], v: float,
                         start: int = 0) -> float:
            """Value of holding a player in each round, weighted by P(this team is the one
            cut that round). Points only matter when they decide who is eliminated.

            Own use: lineup gain over the free replacement.
            Denial: every player you hold is one a rival doesn't. His points above that
            round's replacement level would otherwise go to one of the other n-1 teams, so
            denial counts 1/(n-1) of it. In the final (n = 2) it counts in full: an
            irreplaceable player is worth his edge twice, once for you and once against
            your opponent. A replaceable player is worth nothing to deny.
            """
            # A player bought now has to stay healthy until round r to help then.
            w, healthy = weight[tid], 1.0 - INJURY_RATE.get(ps, 0.02)
            total = 0.0
            for r in range(start, len(n_alive)):
                own = max(0.0, gain - free[r].get(ps, 0.0))
                deny = max(0.0, v - repl[n_alive[r]].get(ps, 0.0)) / (n_alive[r] - 1)
                total += w[r] * healthy ** (r - start) * (own + model.denial_weight * deny)
            return float(total)

        def opportunity(tid: str, base: float, free: list[dict[str, float]]) -> float:
            """Weighted upgrades this team expects to buy in later rounds instead."""
            best, best_p = 0.0, None
            for p in future:
                g = spec.total(rosters[tid] + [p], vb, proj.pos) - base
                if g > best:
                    best, best_p = g, p
            if best_p is None:
                return 0.0
            return sum(min(1.0, model.future_share / n_alive[r])
                       * weighted_par(tid, best, proj.pos[best_p], free, vb[best_p], r)
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
                ps = proj.pos[p]
                own = max(gain - free[0].get(ps, 0.0), gain - free[-1].get(ps, 0.0))
                deny = vb[p] - repl[n_alive[-1]].get(ps, 0.0)
                if own < model.min_gain and deny <= 0:
                    continue  # no use to us and replaceable: nothing to deny
                value = weighted_par(tid, gain, ps, free, vb[p])
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
                    if tid != self.user:
                        # What you could estimate about this rival's bid: no noise, typical aggression.
                        expected[(tid, p)] = model.bid(value=value, opportunity=opp, faab=faab[tid],
                                                       aggression=1.0, noise=1.0, min_bid=settings.min_bid)
            return out

        expected: dict[tuple[str, int], int] = {}
        rival_sd = float(np.hypot(model.manager_sd, model.bid_noise_sd))

        def apply_rules(out: dict[int, int]) -> dict[int, int]:
            if not policy.rules or (j == 0 and policy.mode != "auto"):
                return out
            for pid, rule in policy.rules.items():
                p = proj.index[pid]
                if p not in remaining:
                    continue
                res.user_rule_bids.setdefault(pid, [])
                if rule == "none":
                    out.pop(p, None)
                    continue
                rivals = [(t, expected[(t, p)]) for t in active if t != self.user and (t, p) in expected]
                if rule == "$1" or not rivals:
                    amt = 1
                elif rule == "lock":
                    amt = int(max(faab[t] for t, _ in rivals)) + 1
                else:
                    amt = self._beat_field(float(rule), [(e, faab[t]) for t, e in rivals], rival_sd,
                                           int(faab[self.user]))
                amt = int(min(max(amt, 1), faab[self.user]))
                out[p] = amt
                res.user_rule_bids[pid].append(amt)
            return out

        bids = {tid: team_bids(tid) for tid in active}
        if self.user in bids:
            bids[self.user] = apply_rules(bids[self.user])

        if j == 0:
            if sim == 0:
                base = spec.total(rosters[self.user], vb, proj.pos)
                for p in candidates:
                    g = spec.total(rosters[self.user] + [p], vb, proj.pos) - base
                    res.user_gain_now[proj.ids[p]] = g
            for p in candidates:
                opp = [bids[t][p] for t in active if t != self.user and p in bids[t]]
                res.opp_max_bid[proj.ids[p]].append(max(opp) if opp else -1)

            if policy.mode in ("pass", "force", "bid") and self.user in bids:
                bids[self.user] = {}
            if policy.mode == "bid" and self.user in active:
                bids[self.user] = {proj.index[pid]: amt for pid, amt in (policy.bids or {}).items()
                                   if proj.index[pid] in remaining}
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
            # The pickup changes everyone's elimination odds: recalculate before more claims.
            odds = self._cut_odds(rosters, active, j)
            weight = self._elimination_weights(odds, n_alive)
            # Priority rolls to the back after a successful claim.
            priority[tid] = max(priority.values()) + 1
            if not (j == 0 and tid == self.user and policy.mode != "auto"):
                bids[tid] = team_bids(tid)
                if tid == self.user:
                    bids[tid] = apply_rules(bids[tid])

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

    def _replacement(self, vb, universe, n_active: int, sigma: float = 0.0) -> dict[str, float]:
        """Dynamic replacement level: the best player at each position the remaining
        teams won't start. It rises as teams are cut and their rosters hit waivers.

        With projection uncertainty ``sigma`` (for a round that far ahead), it is the
        expected value of that order statistic: among many uncertain players, the k-th
        best is likely to be someone outperforming his projection. So a stud's edge over
        replacement is smaller far out, and firms up as the final approaches."""
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
            if k >= len(vals):
                out[pos] = 0.0
            elif sigma <= 0:
                out[pos] = sorted(vals, reverse=True)[k]
            else:
                v = np.array(vals)
                drawn = v * np.exp(sigma * REPL_DRAWS[:, : len(v)] - sigma**2 / 2)
                out[pos] = float(np.mean(-np.partition(-drawn, k, axis=1)[:, k]))
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
