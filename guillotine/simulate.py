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

FUTURE_TOP_PLAYERS = 4  # per rival roster and position, players considered as possible future pickups
MONEY_GRID = 41  # money levels at which a team values what its FAAB can still buy
LOSE_GRID = 6  # lost-bid sizes checked for the "lose" side (interpolated between)
COVER_OPTIONS = 8  # replacements considered for each starter who might get hurt
FA_KEEP_PER_POS = 15  # free agents carried between weeks: the best this many at each position
ALT_PER_POS = 3  # free players per position checked as the alternative to a claim
# Projection uncertainty grows with the horizon: a player's true level is a random walk
# around his projection. About ±10% for this week, growing to about ±25% by week 17.
PROJ_ERROR_NOW = 0.10  # lognormal sd of this week's true level vs projection
PROJ_ERROR_STEP = 0.08  # extra lognormal sd added per week further out


def horizon_sd(weeks_ahead: int) -> float:
    """Projection uncertainty (lognormal sd) for a week this many weeks ahead."""
    return float(np.sqrt(PROJ_ERROR_NOW**2 + max(weeks_ahead, 0) * PROJ_ERROR_STEP**2))


_NORMAL = NormalDist()


def _phi(x):
    """Standard normal CDF, vectorized (tanh approximation, error < 0.0002)."""
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))

# Injuries: weekly chance a player gets hurt in a game. A share of injuries end his
# season; the rest keep him out 1-4 weeks. Kept modest because ROS projections already
# price in some missed time.
INJURY_RATE = {"QB": 0.015, "RB": 0.025, "WR": 0.02, "TE": 0.02, "K": 0.005, "DEF": 0.0}
SEASON_ENDING_SHARE = 0.35
SHORT_INJURY_WEEKS = (1, 4)
# Bench rules by slot: at most 1 backup QB and 2 backup TEs (TEs can still start at
# flex); always keep at least 1 backup RB and 1 backup WR.
BENCH_MAX = {"QB": 1, "TE": 2}
BENCH_MIN = {"RB": 1, "WR": 1}
FLEX_POS = ("RB", "WR", "TE")
# Backups a team typically holds at each position (7 bench spots: 1 QB, 1 TE, the rest
# RB/WR). Used to scale how many players below the starters are blocked.
BACKUPS_PER_TEAM = {"QB": 1.0, "TE": 1.0, "RB": 2.5, "WR": 2.5}
# A backup is only worth a claim if he's clearly better than who he replaces: at least
# 10% and half a point a week. Stops swapping near-identical backups week after week.
BENCH_MARGIN_PCT, BENCH_MARGIN_PTS = 0.10, 0.5


def _clearly_better(new: float, old: float) -> bool:
    return new >= old * (1 + BENCH_MARGIN_PCT) and new - old >= BENCH_MARGIN_PTS
ODDS_DRAWS = np.random.default_rng(20261104).standard_normal((2000, 32))  # fixed draws for P(lowest score)
SEASON_DRAWS = np.random.default_rng(20261105).standard_normal((1000, 10, 32))  # fixed draws for the rest-of-season odds
FLEX_SHARE = {"RB": 0.4, "WR": 0.5, "TE": 0.1}  # how flex starts split by position


@dataclass
class UserPolicy:
    """What your team does in the first waiver round. Later rounds always use the model."""

    mode: str = "auto"  # "auto" | "pass" | "force" | "bid"
    player: str | None = None
    price: int = 0
    bids: dict[str, int] | None = None  # mode "bid": your sealed bids {player_id: amount}; others bid as modeled
    # Standing rules: whenever one of these players is on waivers (and you're alive), bid by
    # the rule: "none" (never bid), "$1", "x1.5" = 1.5 times your fair (rational) bid, an int
    # = that many dollars, "all-in", a number q in (0, 1) = enough to beat every
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
    user_fair_bids: dict[str, list[int]] = field(default_factory=dict)  # your rational bid each time he was available
    # after you win a target: (your FAAB left, richest rival's FAAB)
    user_rule_after: dict[str, list[tuple[float, float]]] = field(default_factory=dict)

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
        # Keep the best free agents at each position (this week + week 17).
        self.init_fa = self._best_per_pos(self.init_fa, self.proj.means[:, 0] + self.proj.means[:, -1], FA_KEEP_PER_POS)
        self.inj_rate = np.array([INJURY_RATE.get(p, 0.02) for p in self.proj.pos])
        n_w = len(self.proj.weeks)
        self._avail = np.ones((len(self.proj.ids), n_w))  # per-sim: 1 if the player can play week k
        self._back = np.zeros(len(self.proj.ids), dtype=int)  # per-sim: first week index he's healthy again
        self._hurt_week = np.full(len(self.proj.ids), n_w)  # per-sim: week index he got hurt (n_w = never)
        self.bid_log: list | None = None  # set to [] to record (sim, week_idx, player, team, bid, faab)
        self.trace: list | None = None  # set to [] to record every award, rule bid, score and cut
        self.why_no_bid: list | None = None  # set to [] to record why teams skipped a player (diagnostics)
        # set to [] to record every valuation, in title chances: (sim, week_idx, player, team,
        # points gained now, points gained wk17, value from this week, from weeks between, from
        # the final, total V, value of waiting for this slot, value of all the money kept,
        # slot, FAAB, fair bid)
        self.value_log: list | None = None
        self.money_log: list | None = None  # set to [] to record what each team's money is worth (diagnostics)
        self._added_this_round: dict = {}  # players each team added in the current waiver round
        # Week-17 value of each player as known in the current round (0 if he's out for the
        # rest of the season). Bench players are judged on this: a backup's job is the final.
        self._final = self.proj.means[:, -1].copy()
        self._now = self.proj.means[:, 0].copy()  # this week's projections (per waiver round)

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
            if self.trace is not None:
                n_w = len(proj.weeks)
                for tid in active:
                    for p in rosters[tid]:
                        if self._hurt_week[p] == j:
                            out = int(min(self._hurt_len[p], n_w - 1 - j))
                            ending = bool(j + self._hurt_len[p] >= n_w - 1)
                            self.trace.append(("injury", sim, j, proj.ids[p], tid, out, ending))

            n_elim = min(self.state.settings.eliminations_per_week, len(active) - 1)
            losers = sorted(active, key=lambda t: scores[t])[:n_elim]
            if self.trace is not None:
                self.trace.append(("week", sim, j, dict(scores), list(losers), {t: faab[t] for t in active}))
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
        if self.trace is not None:
            self.trace.append(("champ", sim, champ))

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
        self._hurt_week, self._hurt_len = s, d  # week index hurt (n_w = never) and weeks out

    def _final_value(self, j: int) -> np.ndarray:
        """Each player's projected week-17 score, given injuries that have already happened."""
        fv = self.proj.means[:, -1].copy()
        hurt_now = self._avail[:, j] == 0
        fv[hurt_now & (self._back >= len(self.proj.weeks))] = 0.0
        return fv

    def _out_for_final(self, j: int) -> np.ndarray:
        """P(each player misses week 17 through an injury that hasn't happened yet), seen
        from waiver round j. He can get hurt in any game from week j to week 16; a
        season-ending injury, or a short one still running in week 17, keeps him out.
        Injuries that already happened are known (one per player per season)."""
        n_w = len(self.proj.weeks)
        last = n_w - 1
        lo, hi = SHORT_INJURY_WEEKS
        rate = self.inj_rate
        out = np.zeros(len(rate))
        for g in range(j, last):  # hurt in the game of week index g
            k = last - g  # he misses g+1 .. g+d, so week 17 if d >= k
            p_long = SEASON_ENDING_SHARE + (1 - SEASON_ENDING_SHARE) * max(0, hi - max(k, lo) + 1) / (hi - lo + 1)
            out += rate * (1 - rate) ** (g - j) * p_long
        out[self._hurt_week < j] = 0.0
        return out

    def _final_gain(self, roster: list[int], x: int, fin, q, base: float, without: dict) -> float:
        """Expected week-17 lineup gain from adding player x to this roster.

        If he'd start in week 17: his gain over the starter he pushes out, times P(he's
        healthy then). If he wouldn't start, he's a backup: for each starter s, P(s is out)
        x how much better the lineup without s is with him as the next man up.
        ``base`` = week-17 lineup total now; ``without[s]`` = total without starter s."""
        spec, pos = self.spec, self.proj.pos
        with_x = spec.total(roster + [x], fin, pos)
        if with_x > base + 1e-9:
            return (1.0 - q[x]) * (with_x - base)
        gain = 0.0
        for st, wo in without.items():
            if q[st] > 0 and pos[st] in (FLEX_POS if pos[x] in FLEX_POS else (pos[x],)):
                gain += q[st] * max(0.0, spec.total([y for y in roster if y != st] + [x], fin, pos) - wo)
        return gain

    def _slot_group(self, roster: list[int], p: int, fin) -> tuple[str, ...]:
        """The slot p upgrades for this team in week 17, as the positions that could fill it.

        The lineup fills the harder slots first (QB, RB, WR, TE), then flex takes the best
        of the rest. So the slot p really upgrades is the one held by the starter who drops
        out of the week-17 lineup when he's added: with two great RBs and two great WRs, a
        new RB or WR pushes out the worst flex starter, and the slot is flex (any RB/WR/TE
        can fill it). If nobody drops out, it's the empty slot he fills. A player who
        wouldn't start is a backup, compared with backups like him (QB with QBs, RB/WR/TE
        with flex players)."""
        ps, pos = self.proj.pos[p], self.proj.pos
        before = {x: lab for lab, x in self.spec.assign(roster, fin, pos) if x is not None}
        after = {x: lab for lab, x in self.spec.assign(roster + [p], fin, pos) if x is not None}
        if p not in after:
            return (ps,) if ps not in FLEX_POS else FLEX_POS
        out = [x for x in before if x not in after]
        label = before[out[0]] if out else after[p]
        if label in ("FLEX", "SUPERFLEX"):
            return FLEX_POS
        return (label,)

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
        self._final = self._final_value(j)
        n_weeks = len(proj.weeks)
        weeks_left = n_weeks - j

        if policy.embargo:
            gone = {proj.index[pid] for pid in policy.embargo}
            pool = [p for p in pool if p not in gone]
            fa_pool = [p for p in fa_pool if p not in gone]
        self._now = proj.means[:, j] * self._avail[:, j]
        # The pool players worth a look: best by this week + week 17.
        worth = self._now + self._final
        candidates = sorted(set(pool) | set(fa_pool), key=lambda p: -worth[p])[: model.pool_consider]
        remaining = set(candidates)
        if self.user in active:
            for p in candidates:
                res.user_could_bid.setdefault(proj.ids[p], set()).add(sim)

        e = settings.eliminations_per_week
        # Rounds still to play (this week = r 0), with the number of teams alive in each.
        n_alive = [len(active) - r * e for r in range(weeks_left) if len(active) - r * e >= 2]
        # Odds as they stand before waivers (this week's projections, current rosters).
        # They're recalculated after every pickup below.
        odds = self._cut_odds(rosters, active, j)
        priority = {tid: r for r, tid in enumerate(rng.permutation(active))}
        claimed: set[int] = set()

        # Valuation inputs for this round. Everything is valued in title chances: a player
        # or a dollar is worth what it adds to P(this team wins the league).
        now = self._now
        fin = self._final
        q_out = self._out_for_final(j)
        one_left = len(n_alive) == 1  # the final: this week is week 17 and nothing comes later
        rounds = len(n_alive)
        n_mid = max(rounds - 2, 0)  # weeks between this one and the final
        # This week's score distribution of every team (as in _cut_odds), and its week-17
        # lineup: mean and spread.
        col = {t: i for i, t in enumerate(active)}
        mu_now, sd_now, mu17, sd17 = {}, {}, {}, {}
        err17 = horizon_sd(n_weeks - 1 - j)
        for t in active:
            _, st_now = spec.best(rosters[t], now, proj.pos)
            m = now[st_now]
            mu_now[t] = float(m.sum())
            sd_now[t] = float(np.sqrt((proj.sd[st_now, j] ** 2).sum() + ((m * PROJ_ERROR_NOW) ** 2).sum()))
            tot, st17 = spec.best(rosters[t], fin, proj.pos)
            mu17[t] = float(tot)
            sd17[t] = float(np.sqrt((proj.sd[st17, -1] ** 2).sum() + ((fin[st17] * err17) ** 2).sum()))
        draws = {t: mu_now[t] + sd_now[t] * ODDS_DRAWS[:, col[t]] for t in active}
        # Rest-of-season odds: play the remaining weeks 1,000 times with these score
        # distributions (this week's lineups now, week-17 lineups after). weight[t][r] =
        # P(t is cut in round r); in the last round, P(t loses the final).
        weight = self._season_odds(active, n_alive, e, mu_now, sd_now, mu17, sd17)
        # P(a rival's player reaches waivers later): his team is cut in round r (not the
        # final, after which there are no waivers). Whether this team then gets him depends
        # on its money: see share() in team_bids.
        reach = {u: sum(weight[u][r] for r in range(len(n_alive) - 1)) for u in active}
        SLOTS = {"QB": ("QB",), "RB": ("RB",), "WR": ("WR",), "TE": ("TE",), "FLEX": FLEX_POS}
        # P(team reaches the final) and P(team wins the league)
        p_final = {t: 1.0 if rounds == 1 else max(0.0, 1.0 - sum(weight[t][: rounds - 1])) for t in active}
        p_title = {t: max(0.0, 1.0 - sum(weight[t])) for t in active}
        alive_after = {v: 1.0 - np.cumsum(weight[v][:rounds]) for v in active}  # P(still alive after round r)

        available = [x for x in set(pool) | set(fa_pool)]
        alt_fin = self._best_per_pos(available, fin, ALT_PER_POS)  # free week-17 alternatives

        # Title chance per extra point, for every team.
        #   This week: P(cut) as a function of extra points, against every other team's score
        #   distribution. A point matters only if it can change who's lowest.
        #   Weeks before the final: a point is worth about what it's worth this week x P(title).
        #   The final: P(reach it) x density of the final margin against the likely opponent
        #   (rivals' average, weighted by their chance of reaching the final).
        cut_fn, pp_mid, pp_fin = {}, {}, {}
        for t in active:
            if not one_left:
                omin = np.min(np.vstack([draws[u] for u in active if u != t]), axis=0)
                cut_fn[t] = (lambda om, mu, sd: (lambda extra: float(_phi((om - mu - extra) / max(sd, 1e-6)).mean())))(
                    omin, mu_now[t], sd_now[t])
                pp_mid[t] = (cut_fn[t](-0.5) - cut_fn[t](0.5)) * p_title[t]
            else:
                pp_mid[t] = 0.0
            rivals_t = [u for u in active if u != t]
            wsum = sum(p_final[u] for u in rivals_t)
            if wsum > 0:
                opp_mu = sum(p_final[u] * mu17[u] for u in rivals_t) / wsum
                opp_sd = sum(p_final[u] * sd17[u] for u in rivals_t) / wsum
            else:
                opp_mu, opp_sd = mu17[t], sd17[t]
            sig = float(np.hypot(sd17[t], opp_sd)) or 1.0
            z = (mu17[t] - opp_mu) / sig
            pp_fin[t] = p_final[t] * float(np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)) / sig

        # Players who could come back later: the best few at each position on every roster.
        # One reaches waivers only if his team is cut before the final (P = reach).
        opts = []  # (player, holder, position)
        if not one_left:
            for u in active:
                if reach[u] <= 0:
                    continue
                by_pos: dict[str, list[int]] = {}
                for x in rosters[u]:
                    if fin[x] > 0:
                        by_pos.setdefault(proj.pos[x], []).append(x)
                for ps, xs in by_pos.items():
                    for x in sorted(xs, key=lambda y: -fin[y])[:FUTURE_TOP_PLAYERS]:
                        opts.append((x, u, ps))
        n_opt = len(opts)
        opt_pos = np.array([o[2] for o in opts], dtype=object)
        opt_reach = np.array([reach[o[1]] for o in opts], dtype=float)
        # Weeks before the final he'd play for his new team (freed after round r, plays r+1 on).
        opt_mid = np.array([sum(weight[o[1]][r] * max(0, n_mid - r) for r in range(rounds - 1)) / reach[o[1]]
                            if reach[o[1]] > 0 else 0.0 for o in opts], dtype=float)
        # What each one is worth to each team, in title chances if he lands there: his week-17
        # gain over the better of its starter and the best free player, for the final and the
        # weeks he'd play before it.
        gain_opt, val_opt = {}, {}
        for t in active:
            with_free = rosters[t] + alt_fin
            base_free = spec.total(with_free, fin, proj.pos)
            g = np.array([0.0 if o[1] == t else (1.0 - q_out[o[0]]) * max(0.0, spec.total(with_free + [o[0]], fin, proj.pos) - base_free)
                          for o in opts], dtype=float)
            gain_opt[t] = g
            val_opt[t] = g * (pp_fin[t] + pp_mid[t] * opt_mid)
        # Each team's willingness to pay for each of them, in dollars: his value x what a
        # dollar is worth to that team (its FAAB / the value of the best player it could hope
        # to add in each slot), capped by its FAAB.
        POS_SLOTS = ("QB", "RB", "WR", "TE")
        will = {}
        for t in active:
            hope = sum(float((opt_reach * val_opt[t])[opt_pos == ps].max(initial=0.0)) for ps in POS_SLOTS)
            rate = float(faab[t]) / hope if hope > 0 else 0.0
            will[t] = np.minimum(float(faab[t]), val_opt[t] * rate)
        # For each option and each team that might compete for him: P(that team is still
        # alive when he comes back), averaged over when his team would be cut.
        alive_at = {}
        for v in active:
            alive_at[v] = np.array([sum(weight[o[1]][r] * max(float(alive_after[v][r]), 0.0) for r in range(rounds - 1)) / reach[o[1]]
                                    if reach[o[1]] > 0 and o[1] != v else 0.0 for o in opts], dtype=float)

        bid_sd = model.bid_sd if model.bid_sd else 0.61

        def win_chance(tid: str, budgets: np.ndarray, cut_rival: str | None = None, cut_by: float = 0.0) -> np.ndarray:
            """P(tid wins each option with each budget), options x budgets. Every rival still
            alive when he comes back bids its willingness x Normal(1, bid_sd) (as rivals bid in
            this simulation); tid wins if its budget beats all of them. cut_rival has cut_by
            less money (it won a bid tid lost)."""
            if n_opt == 0:
                return np.zeros((0, len(budgets)))
            out = np.ones((n_opt, len(budgets)))
            for v in active:
                if v == tid:
                    continue
                w = will[v] if v != cut_rival else np.minimum(will[v], max(float(faab[v]) - cut_by, 0.0))
                cap = float(faab[v]) - (cut_by if v == cut_rival else 0.0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    beat = _phi((budgets[None, :] / np.maximum(w[:, None], 1e-9) - 1.0) / bid_sd)
                beat = np.where(w[:, None] <= 0, 1.0, beat)
                beat = np.where(budgets[None, :] > cap, 1.0, beat)  # can't bid more than it has
                a_v = alive_at[v][:, None]
                out *= (1.0 - a_v) + a_v * beat
            return out

        def team_bids(tid: str) -> dict[int, int]:
            out = {}
            roster = rosters[tid]
            if not one_left:
                p_cut = cut_fn[tid]
                cut0 = p_cut(0.0)
                title_if_survive = p_title[tid] / max(1.0 - weight[tid][0], 1e-9)
            per_point_mid, per_point_final = pp_mid[tid], pp_fin[tid]
            base_now = spec.total(roster, now, proj.pos)
            base_fin, starters = spec.best(roster, fin, proj.pos)
            without = {st: spec.total([y for y in roster if y != st], fin, proj.pos) for st in starters}

            # What money buys later: the best set of later pickups this team can afford, at
            # most one per position, plus a replacement for each starter who might get hurt
            # before week 17. Each pickup costs what it takes to beat the most willing rival
            # still alive then. Money spent now shrinks the set it can afford.
            money = float(max(faab[tid], 0))
            m_grid = np.linspace(0.0, money, MONEY_GRID) if money > 0 else np.zeros(1)
            step = money / (len(m_grid) - 1) if len(m_grid) > 1 else 1.0
            chance = win_chance(tid, m_grid)  # options x budget set aside for him
            my_val = opt_reach * val_opt[tid]  # expected title chance from each, if bought
            slot_mask = {ps: (opt_pos == ps) & (my_val > 0) for ps in POS_SLOTS}
            # Injury cover: the best few replacements for each starter, valued as P(he's out)
            # x the replacement's gain over the next man up (or the best free player).
            cover_items = []  # (starter, positions, option indices, gains)
            if n_opt:
                labels = {x: lab for lab, x in spec.assign(roster, fin, proj.pos) if x is not None}
                for st, lab in labels.items():
                    if q_out[st] <= 0:
                        continue
                    grp = FLEX_POS if lab in ("FLEX", "SUPERFLEX") else (lab,)
                    idx = [i for i in np.argsort(-fin[[o[0] for o in opts]]) if opts[i][2] in grp and opts[i][1] != tid][:COVER_OPTIONS]
                    if not idx:
                        continue
                    rest_free = [y for y in roster if y != st] + alt_fin
                    base_rf = spec.total(rest_free, fin, proj.pos)
                    g_rep = np.array([(1.0 - q_out[opts[i][0]]) * max(0.0, spec.total(rest_free + [opts[i][0]], fin, proj.pos) - base_rf)
                                      for i in idx])
                    cover_items.append((st, grp, np.array(idx, dtype=int), g_rep, base_rf))

            def budget_value(chance, slot_vals=None, cover_vals=None) -> np.ndarray:
                """Best total title chance from later pickups, for every budget on m_grid. The
                budget is split across positions (and injury cover); within each, the money set
                aside goes to whichever player gives the most value x P(winning him with it)."""
                f = np.zeros(len(m_grid))
                if n_opt == 0 or money <= 0:
                    return f
                curves = []
                for ps in POS_SLOTS:
                    vals = my_val if slot_vals is None or ps not in slot_vals else slot_vals[ps]
                    m = slot_mask[ps] & (vals > 0)
                    if m.any():
                        curves.append((vals[m][:, None] * chance[m]).max(axis=0))
                for c, (st, grp, idx, g_rep, _) in enumerate(cover_items):
                    gr = g_rep if cover_vals is None or c not in cover_vals else cover_vals[c]
                    vals = opt_reach[idx] * q_out[st] * per_point_final * gr
                    if (vals > 0).any():
                        curves.append((vals[:, None] * chance[idx]).max(axis=0))
                n = len(m_grid)
                ii = np.arange(n)
                src = ii[:, None] - ii[None, :]  # budget left for the other positions
                for g in curves:
                    cand = np.where(src >= 0, f[np.clip(src, 0, None)] + g[None, :], -np.inf)
                    f = np.maximum(f, cand.max(axis=1))
                return f

            future_all = budget_value(chance)  # what this team's money is worth, by amount kept
            # If it loses a bid of b, it keeps all its money and the winner (likely its richest
            # rival) has b less, so later prices may drop.
            richest = max((u for u in active if u != tid), key=lambda u: faab[u], default=None)
            b_pts = np.linspace(0.0, money, LOSE_GRID) if money > 0 else np.zeros(1)
            lost_vals = np.array([budget_value(win_chance(tid, m_grid, richest, bb))[-1] for bb in b_pts])
            future_lost = np.interp(money - m_grid, b_pts, lost_vals)  # by the bid lost: b = money - m_grid
            if self.money_log is not None and submitting:
                self.money_log.append((sim, j, tid, money, m_grid.copy(), future_all.copy(), per_point_final, per_point_mid,
                                       [(proj.ids[o[0]], o[2], float(gain_opt[tid][i]), float(opt_reach[i]), float(val_opt[tid][i]), float(chance[i, -1]))
                                        for i, o in enumerate(opts)]))

            # This week's alternative: every other player on waivers or free agency now.
            # A slot that only needs filling this week (a bye, an injury) can be filled by
            # the next-best available player, so a claim only gets credit for beating him.
            gain_now = {}
            for x in available:
                if now[x] > 0:
                    g = spec.total(roster + [x], now, proj.pos) - base_now
                    if g > 1e-9:
                        gain_now[x] = g
            # Week 17 too: a claim only counts above the best other player available now at
            # that slot. With many free QBs on waivers, none of them is worth much.
            gain_fin = {x: self._final_gain(roster, x, fin, q_out, base_fin, without)
                        for x in set(remaining) | set(alt_fin)}
            why = self.why_no_bid
            for p in remaining:
                g_now = gain_now.get(p, 0.0)
                if g_now > 0:
                    group_now = self._slot_group(roster, p, now)
                    alt = max((g for x, g in gain_now.items() if x != p and proj.pos[x] in group_now), default=0.0)
                    g_now = max(0.0, g_now - alt)
                g_fin = gain_fin[p]
                if g_fin > 0:
                    group_fin = self._slot_group(roster, p, fin)
                    alt = max((g for x, g in gain_fin.items() if x != p and proj.pos[x] in group_fin), default=0.0)
                    g_fin = max(0.0, g_fin - alt)
                if tid == self.user and j > 0:
                    self._note_user_gain(res, sim, j, proj.ids[p], g_fin)
                slot = self._slot_check(roster, p)
                if slot == "blocked":
                    if why is not None:
                        why.append((sim, j, proj.ids[p], tid, "blocked by roster limits", g_now, g_fin, 0.0, 0.0))
                    continue  # he'd be the backup dropped to stay within the QB/TE limits
                v_now = (cut0 - p_cut(g_now)) * title_if_survive if (g_now > 0 and not one_left) else 0.0
                # Weeks before the final count only if he'd start: a backup's value is injury
                # cover for the final, not points every week.
                starts17 = spec.total(roster + [p], fin, proj.pos) > base_fin + 1e-9
                v_mid = per_point_mid * n_mid * g_fin if starts17 else 0.0
                v_fin = per_point_final * g_fin
                value = v_now + v_mid + v_fin
                group = self._slot_group(roster, p, fin)
                key = "FLEX" if group == FLEX_POS else group[0]
                # Win vs lose. Lose: keep all the money (later upgrades in every slot, including
                # this one, plus injury cover). Win at price b: his value now, and only FAAB - b
                # for later, where this slot's later option only counts if he's better than p.
                # Winning him: later pickups at his position(s) only count above him, and if a
                # starter he could replace gets hurt, he plays, so a bought replacement only
                # counts above him too.
                # His week-17 gain on the same footing as the later options: over the better of
                # the current starter and the best free player (other than him).
                free_wo = roster + [x for x in alt_fin if x != p]
                g_p = (1.0 - q_out[p]) * max(0.0, spec.total(free_wo + [p], fin, proj.pos) - spec.total(free_wo, fin, proj.pos))
                slot_vals = {}
                for ps in POS_SLOTS:
                    if ps in group:
                        g_after = np.maximum(gain_opt[tid] - g_p, 0.0)
                        slot_vals[ps] = opt_reach * g_after * (per_point_final + per_point_mid * opt_mid)
                cover_vals = {}
                for c, (st, grp, idx, g_rep, base_rf) in enumerate(cover_items):
                    if proj.pos[p] in grp:
                        rest_free = [y for y in roster if y != st] + alt_fin
                        b_s = max(0.0, spec.total(rest_free + [p], fin, proj.pos) - base_rf)
                        if b_s > 0:
                            cover_vals[c] = np.maximum(g_rep - b_s, 0.0)
                future_win = budget_value(chance, slot_vals, cover_vals)
                lose = future_lost  # by the bid lost: b = money - m_grid
                win = value + future_win  # by money kept: m_grid
                if win[-1] <= lose[-1]:
                    if why is not None:
                        why.append((sim, j, proj.ids[p], tid, "can do as well later", g_now, g_fin, value, float(lose[-1] - future_win[-1])))
                    continue
                # Fair bid: the most he can pay and still be better off winning than losing.
                # Paying b means winning with money - b, versus losing a bid of b (a rival paid
                # at least that). The diff rises as b falls; find where it crosses 0.
                diff = win - lose
                ok = np.nonzero(diff >= 0)[0]
                k = int(ok[0])
                if k == 0:
                    keep_min = 0.0
                else:  # interpolate between grid points k-1 (not enough) and k (enough)
                    d0, d1 = diff[k - 1], diff[k]
                    keep_min = m_grid[k - 1] + (m_grid[k] - m_grid[k - 1]) * (-d0) / max(d1 - d0, 1e-12)
                det = int(max(0.0, money - keep_min))
                lock = int(max((faab[u] for u in active if u != tid), default=0)) + 1
                if lock <= det:
                    det = lock  # topping every rival's whole FAAB is enough to guarantee the win
                det = max(det, settings.min_bid)
                if det < 0:
                    continue
                # A lineup claim starts this week or in week 17; otherwise he's a backup.
                kind = "lineup" if (gain_now.get(p, 0.0) > 0 or spec.total(roster + [p], fin, proj.pos) > base_fin + 1e-9) else "bench"
                locked = det == lock and lock <= faab[tid]
                if locked:
                    # Guaranteeing the win is worth it: bid exactly the lock price (top every
                    # other team's whole FAAB). No noise: a lock that misses isn't a lock.
                    m, amt = 1.0, int(det)
                elif model.bid_sd is not None:
                    # Every bid is fair x a random multiplier, normal around 1. Bidding is free
                    # (you only pay if you win), so a draw at or below 0 is a $0 claim.
                    m = float(rng.normal(1.0, model.bid_sd))
                    amt = int(min(max(m, 0.0) * det * aggression[tid], faab[tid]))
                else:
                    m = None
                    amt = int(min(det * aggression[tid] * rng.lognormal(0, model.bid_noise_sd), faab[tid]))
                if submitting and self.trace is not None:
                    self.trace.append(("bid", sim, j, proj.ids[p], tid, max(amt, 0), det, m, faab[tid], kind))
                if self.value_log is not None and submitting:
                    self.value_log.append((sim, j, proj.ids[p], tid, g_now, g_fin, v_now, v_mid, v_fin, value,
                                           float(lose[-1] - future_win[-1]), float(lose[-1]), key, faab[tid], det))
                fair_all[(tid, p)] = det
                kinds[(tid, p)] = kind
                if tid != self.user:
                    expected[(tid, p)] = det
                else:
                    fair[p] = det
                if amt >= 0:
                    out[p] = amt
                    if self.bid_log is not None:
                        self.bid_log.append((sim, j, proj.ids[p], tid, amt, faab[tid]))
            return out

        expected: dict[tuple[str, int], int] = {}
        fair: dict[int, int] = {}
        fair_all: dict[tuple[str, int], int] = {}
        kinds: dict[tuple[str, int], str] = {}
        rival_sd = float(np.hypot(model.manager_sd, model.bid_noise_sd))
        submitting = True  # bids are logged once, when submitted; later re-checks only cancel claims

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
                    if self.trace is not None:
                        self.trace.append(("rule", sim, j, pid, fair.get(p, 0), 0))
                    continue
                rivals = [(t, expected[(t, p)]) for t in active if t != self.user and (t, p) in expected]
                if isinstance(rule, str) and rule.startswith("x"):
                    # A multiple of your fair bid: x1 = bid exactly what the player is worth to you.
                    f = fair.get(p, 0)
                    res.user_fair_bids.setdefault(pid, []).append(f)
                    amt = max(1, int(round(float(rule[1:]) * f)))
                elif rule == "all-in":
                    amt = int(faab[self.user])
                elif isinstance(rule, int):
                    amt = rule  # a fixed dollar amount (capped at what you have)
                elif rule == "$1" or not rivals:
                    amt = 1
                elif rule == "lock":
                    amt = int(max(faab[t] for t, _ in rivals)) + 1
                else:
                    amt = self._beat_field(float(rule), [(e, faab[t]) for t, e in rivals], rival_sd,
                                           int(faab[self.user]))
                amt = int(min(max(amt, 1), faab[self.user]))
                out[p] = amt
                res.user_rule_bids[pid].append(amt)
                if self.trace is not None:
                    self.trace.append(("rule", sim, j, pid, fair.get(p, 0), amt))
            return out

        if self.trace is not None:
            self.trace.append(("odds", sim, j, dict(odds)))
        bids = {tid: team_bids(tid) for tid in active}
        if self.user in bids:
            bids[self.user] = apply_rules(bids[self.user])
        submitting = False

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
            self._added_this_round = {}
            if policy.mode == "force" and self.user in active:
                p = proj.index[policy.player]
                if p in remaining and faab[self.user] >= policy.price:
                    self._award(sim, j, p, self.user, policy.price, rosters, faab, fa_pool, remaining, claimed, res)

        # Every claim is "add X / drop Y", planned once, after all bids are in.
        drops = {tid: self._plan_drops(rosters[tid], bids[tid], fair_all, tid)
                 for tid in active}
        # A backup-only claim must be clearly better for week 17 than the player it drops.
        # Planning several claims together can assign a different drop than the one the
        # claim was judged against, so check again and withdraw the claim if not.
        for tid in active:
            for p, d in list(drops[tid].items()):
                if (kinds.get((tid, p)) == "bench" and isinstance(d, (int, np.integer))
                        and not _clearly_better(self._final[p], self._final[d])):
                    bids[tid].pop(p, None)
                    del drops[tid][p]
        if self.trace is not None:
            for tid in active:
                for p, d in drops[tid].items():
                    self.trace.append(("claim", sim, j, proj.ids[p], tid,
                                       proj.ids[d] if isinstance(d, (int, np.integer)) else None))
        gone: dict[str, set] = {tid: set() for tid in active}  # drops already used by a processed claim
        self._added_this_round = {}

        while True:
            best = None
            for tid in active:
                for p, amt in bids[tid].items():
                    if p not in remaining or amt > faab[tid]:
                        continue  # taken already, or the team can no longer afford it
                    d = drops[tid].get(p)
                    if d in gone[tid] or (isinstance(d, (int, np.integer)) and d not in rosters[tid]):
                        continue  # invalid claim due to previous transaction: its drop is gone
                    key = (amt, -priority[tid])
                    if best is None or key > best[0]:
                        best = (key, tid, p)
            if best is None:
                break
            (amt, _), tid, p = best
            d = drops[tid].get(p)
            if self._would_cut_new(rosters[tid], p, d, self._added_this_round.get(tid, set())):
                bids[tid].pop(p, None)  # invalid: it would push out a player won this same round
                continue
            if self.trace is not None:
                all_bids = {t: bids[t][p] for t in active if p in bids[t]}
                self.trace.append(("award", sim, j, proj.ids[p], tid, int(amt), all_bids,
                                   {t: faab[t] for t in active}))
            gone[tid].add(d)
            self._award(sim, j, p, tid, amt, rosters, faab, fa_pool, remaining, claimed, res, drop=d)
            if tid == self.user and policy.rules and proj.ids[p] in policy.rules:
                richest = max((faab[u] for u in active if u != tid), default=0)
                res.user_rule_after.setdefault(proj.ids[p], []).append((faab[tid], richest))
            # Priority rolls to the back after a successful claim. Bids and drops were fixed
            # when submitted, so nothing is re-priced or re-planned.
            priority[tid] = max(priority.values()) + 1

        leftovers = [p for p in set(pool) | set(fa_pool) if p not in claimed]
        return [], self._best_per_pos(leftovers, worth, FA_KEEP_PER_POS)

    def _award(self, sim, j, p, tid, amt, rosters, faab, fa_pool, remaining, claimed, res, drop=None):
        proj = self.proj
        faab[tid] -= amt
        rosters[tid].append(p)
        if isinstance(drop, (int, np.integer)) and drop in rosters[tid]:
            rosters[tid].remove(drop)
            fa_pool.append(drop)
            if self.trace is not None:
                self.trace.append(("drop", sim, j, tid, proj.ids[drop]))
        remaining.discard(p)
        claimed.add(p)
        if p in fa_pool:
            fa_pool.remove(p)
        res.awards.append((sim, j, proj.ids[p], tid, int(amt)))
        self._added_this_round.setdefault(tid, set()).add(p)
        self._enforce_roster(sim, j, tid, p, rosters, fa_pool, protected=self._added_this_round[tid])

    def _best_per_pos(self, players, worth, k: int) -> list[int]:
        """The best k players at each position, by ``worth``."""
        by_pos: dict[str, list[int]] = {}
        for x in players:
            by_pos.setdefault(self.proj.pos[x], []).append(x)
        out = []
        for xs in by_pos.values():
            out += sorted(xs, key=lambda x: -worth[x])[:k]
        return sorted(out, key=lambda x: -worth[x])

    def _would_cut_new(self, roster: list[int], p: int, drop, added: set) -> bool:
        """Would adding p (dropping ``drop``) force the roster rules to cut p or a player
        this team added earlier in the same round?"""
        trial = [q for q in roster if q != drop] + [p]
        fa: list[int] = []
        trace, self.trace = self.trace, None
        try:
            self._enforce_roster(0, 0, "_", p, {"_": trial}, fa, protected=frozenset(added | {p}))
        finally:
            self.trace = trace
        return any(q in added or q == p for q in fa)

    def _drop_for(self, roster: list[int], p: int):
        """Who this team would cut to fit p: the first player _enforce_roster removes, or
        ("open", roster size) if there's room. Open spots are tokens, so two claims for the
        same open spot are linked like two claims dropping the same player."""
        trial, fa = {"_": list(roster) + [p]}, []
        trace, self.trace = self.trace, None
        try:
            self._enforce_roster(0, 0, "_", p, trial, fa)
        finally:
            self.trace = trace
        return fa[0] if fa else ("open", len(roster))

    def _plan_drops(self, roster: list[int], claims: dict[int, int], fair_all, tid) -> dict:
        """Give every claim a drop, as in Yahoo: a player on the roster right now (or an
        open roster spot). Never a player the team might add in this same round.

        Going from the highest fair bid down and assuming the claims above it win: if the
        player would still help (lineup upgrade, better backup, or worth keeping from
        rivals), he gets a different drop from the claims above, so both can go through.
        If not, he's a backup claim that shares the drop of the claim he'd replace, so only
        one of them processes."""
        proj, spec, model = self.proj, self.spec, self.model
        plan: dict[int, object] = {}
        n_open = max(0, self.state.settings.roster_size - len(roster))
        open_used = 0
        hypothetical = list(roster)  # roster if the claims above win (only to judge usefulness)
        remaining = list(roster)  # players still free to name as a drop
        added: set[int] = set()  # players that higher claims would add
        for p in sorted(claims, key=lambda q: -fair_all.get((tid, q), 0)):
            # Still useful if the claims above win: he'd start this week or in week 17, or
            # he's a clearly better backup than who he'd replace.
            useful = (self._starts(hypothetical, p) or self._slot_check(hypothetical, p) == "upgrade")
            if not useful:
                plan[p] = self._drop_for(roster, p)  # backup claim: same drop as the claim it backs up
                continue
            d = self._drop_for(hypothetical, p)
            if isinstance(d, tuple):  # fits in an open roster spot
                if open_used >= n_open:
                    plan[p] = self._drop_for(roster, p)
                    continue
                d = ("open", open_used)
                open_used += 1
            elif d in added:
                # He'd only fit by cutting a player another claim adds: he's an alternative
                # to that claim, so he shares its drop and only one of them can process.
                plan[p] = plan[d]
                continue
            elif d in remaining:
                remaining.remove(d)
            else:
                plan[p] = self._drop_for(roster, p)
                continue
            plan[p] = d
            added.add(p)
            hypothetical = [q for q in hypothetical if q != d] + [p]
        return plan

    def _slot_check(self, roster: list[int], p: int) -> str:
        """Can player p join this roster, and would he be a better backup?

        He's compared with the player the team would actually have to drop to fit him
        (the roster rules decide who: lowest week-17 value, keeping 1 backup RB and 1
        backup WR and at most 1 backup QB / 2 backup TEs), on week-17 projection, because
        a bench spot is for the final. A bye or a short slump now doesn't make a backup worse.

        "upgrade": clearly better for week 17 than that player, or he fits in an open spot.
        "blocked": the rules would drop him again (he'd be the worst backup QB/TE).
        "no": he fits, but isn't clearly better for week 17.
        """
        d = self._drop_for(roster, p)
        if isinstance(d, tuple):  # open roster spot
            return "upgrade"
        if d == p:
            return "blocked"
        return "upgrade" if _clearly_better(self._final[p], self._final[d]) else "no"

    def _starters(self, roster: list[int]) -> set[int]:
        """Players who start this week or in week 17: they're never cut to make room."""
        pos = self.proj.pos
        return set(self.spec.best(roster, self._now, pos)[1]) | set(self.spec.best(roster, self._final, pos)[1])

    def _starts(self, roster: list[int], p: int) -> bool:
        return p in self._starters(roster + [p])

    def _enforce_roster(self, sim, j, tid, newest, rosters, fa_pool, protected=frozenset()) -> None:
        """Drop bench players until the roster fits: at most 1 backup QB and 2 backup TEs,
        at most roster_size players, keeping at least 1 backup RB and 1 backup WR. The bench
        player cut is the one with the lowest week-17 value. Starters (this week or week 17)
        are never cut."""
        proj = self.proj
        roster = rosters[tid]
        while True:
            starters = self._starters(roster)
            bench = [q for q in roster if q not in starters]
            drop = None
            for pos, cap in BENCH_MAX.items():
                same = [q for q in bench if proj.pos[q] == pos]
                if len(same) > cap:
                    pick = [q for q in same if q not in protected] or same
                    drop = min(pick, key=lambda q: self._final[q])
                    break
            if drop is None and len(roster) > self.state.settings.roster_size:
                count = {pos: sum(proj.pos[q] == pos for q in bench) for pos in BENCH_MIN}
                ok = [q for q in bench if q != newest and q not in protected
                      and not (proj.pos[q] in BENCH_MIN and count[proj.pos[q]] <= BENCH_MIN[proj.pos[q]])]
                pool = ok or [q for q in bench if q != newest and q not in protected] or bench
                drop = min(pool, key=lambda q: self._final[q]) if pool else None
            if drop is None:
                return
            roster.remove(drop)
            fa_pool.append(drop)
            if self.trace is not None:
                self.trace.append(("drop", sim, j, tid, proj.ids[drop]))

    def _note_user_gain(self, res, sim, j, pid, gain):
        entries = res.pool_entries.get(pid)
        if entries and entries[-1][0] == sim and entries[-1][1] == j and np.isnan(entries[-1][2]):
            entries[-1] = (sim, j, gain)

    def _replacement(self, vb, universe, n_active: int) -> dict[str, float]:
        """Dynamic replacement level: what a team can expect to get for free at each
        position. Every team alive starts some players and holds backups, so the free
        player is somewhere below all of them: this averages from the first non-starter
        down through (teams alive x backups per team) more (with 2 teams left, QBs 3-5).
        It rises as teams are cut and their rosters hit waivers.

        No "breakout" adjustment: an earlier version took the expected k-th best of
        randomly perturbed projections, which at deep positions put replacement above
        every real player (Josh Allen looked replaceable). Uncertainty is handled by the
        injury discount and by players' true levels drifting during the season."""
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
            ranked = sorted(vals, reverse=True)
            # Every team alive also holds backups, which blocks some of the players below
            # the starters. Average from the first non-starter down through everyone's
            # backups: with 2 teams left, QBs 3-5; with 10, QBs 11-21.
            depth = int(round(n_active * BACKUPS_PER_TEAM.get(pos, 1.0)))
            below = ranked[k:k + depth + 1]
            out[pos] = float(np.mean(below)) if below else 0.0
        return out

    @staticmethod
    def _season_odds(active, n_alive, per_week, mu_now, sd_now, mu17, sd17) -> dict[str, list[float]]:
        """P(each team is cut in each remaining round), from simulating the rest of the
        season with fixed draws. Round 0 uses this week's score distributions, later rounds
        the week-17 lineups. The last round is the final: "cut" = losing it."""
        k, n = SEASON_DRAWS.shape[0], len(active)
        mu_a = np.array([[mu_now[t] for t in active], [mu17[t] for t in active]])
        sd_a = np.array([[sd_now[t] for t in active], [sd17[t] for t in active]])
        dead = np.zeros((k, n), dtype=bool)
        counts = np.zeros((len(n_alive), n))
        rows = np.arange(k)
        for r in range(len(n_alive)):
            i = 0 if r == 0 else 1
            sc = mu_a[i] + sd_a[i] * SEASON_DRAWS[:, r % SEASON_DRAWS.shape[1], :n]
            sc = np.where(dead, np.inf, sc)
            for _ in range(per_week if r < len(n_alive) - 1 else 1):
                lo = sc.argmin(axis=1)
                counts[r] += np.bincount(lo, minlength=n)
                dead[rows, lo] = True
                sc[rows, lo] = np.inf
        p = counts / k
        return {t: [float(x) for x in p[:, c]] for c, t in enumerate(active)}

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

