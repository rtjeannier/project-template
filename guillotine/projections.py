"""Turn season-level projections into week-by-week score distributions.

Projection sources usually give rest-of-season (ROS) totals plus a projection for
the current week only. For later weeks we spread the ROS total evenly over the
player's remaining games, scoring zero on the bye.

Elimination depends on bad weeks, not averages, so each player-week gets a spread
as well as a mean. The spread is a position-level coefficient of variation:
receivers and tight ends are boom/bust, quarterbacks are steadier.
"""

from __future__ import annotations

import numpy as np

from .models import LeagueState

# Week-to-week standard deviation as a fraction of the projected mean.
POSITION_CV = {"QB": 0.40, "RB": 0.55, "WR": 0.60, "TE": 0.65, "K": 0.45, "DEF": 0.65}

# Projection blend used when valuing players for waivers: managers care about the
# next few weeks (survival) more than the distant ones.
NEAR_TERM_WEEKS = 3
NEAR_TERM_WEIGHT = 0.6


class Projections:
    def __init__(self, state: LeagueState):
        self.weeks = state.settings.weeks
        self.ids = list(state.players)
        self.index = {pid: i for i, pid in enumerate(self.ids)}
        self.pos = [state.players[pid].pos for pid in self.ids]
        n_weeks = len(self.weeks)

        self.means = np.zeros((len(self.ids), n_weeks))
        for i, pid in enumerate(self.ids):
            p = state.players[pid]
            games = [w for w in self.weeks if w != p.bye]
            per_game = p.ros_points / len(games) if games else 0.0
            for j, w in enumerate(self.weeks):
                if w == p.bye:
                    continue
                self.means[i, j] = p.weekly.get(w, per_game)

        cv = np.array([POSITION_CV.get(pos, 0.6) for pos in self.pos])
        self.sd = self.means * cv[:, None]

        # value_basis[j] = per-player weekly value as seen from sim week j.
        self.value_basis = np.zeros((n_weeks, len(self.ids)))
        for j in range(n_weeks):
            near = self.means[:, j : j + NEAR_TERM_WEEKS].mean(axis=1)
            ros = self.means[:, j:].mean(axis=1)
            self.value_basis[j] = NEAR_TERM_WEIGHT * near + (1 - NEAR_TERM_WEIGHT) * ros

        self._init_generic_value(state)

    def _init_generic_value(self, state: LeagueState) -> None:
        """Generic (team-independent) market value of each player at each sim week.

        Uses supplied trade values when most players have them (they decay linearly as
        the season runs out). Otherwise falls back to value over replacement.
        """
        slots = state.settings.roster_slots
        n_teams = len(state.teams)
        flex = sum(c for s, c in slots.items() if s in ("FLEX", "W/R/T", "W/R", "W/T", "R/W"))
        superflex = sum(c for s, c in slots.items() if s in ("SUPERFLEX", "Q/W/R/T"))
        flex_share = {"RB": 0.4, "WR": 0.5, "TE": 0.1}
        ros_avg = self.means.mean(axis=1)

        self.replacement = {}
        for pos in set(self.pos):
            starters = slots.get(pos, 0) + flex * flex_share.get(pos, 0)
            starters += superflex * (0.8 if pos == "QB" else 0.07)
            k = max(1, int(round(n_teams * starters)))
            vals = sorted((ros_avg[i] for i, p in enumerate(self.pos) if p == pos), reverse=True)
            self.replacement[pos] = vals[k] if k < len(vals) else 0.0

        repl = np.array([self.replacement[p] for p in self.pos])
        weekly_vor = np.clip(self.means - repl[:, None], 0, None)
        vor_cum = np.cumsum(weekly_vor[:, ::-1], axis=1)[:, ::-1]  # VOR from week j onward
        self.weekly_vor = np.clip(self.value_basis - repl[None, :], 0, None)

        tv = np.array(
            [state.players[pid].trade_value if state.players[pid].trade_value is not None else np.nan
             for pid in self.ids],
            dtype=float,
        )
        have_tv = ~np.isnan(tv)
        if have_tv.mean() >= 0.5:
            # Fill missing trade values from VOR, scaled into trade-value units.
            scale_mask = have_tv & (vor_cum[:, 0] > 0)
            ratio = np.median(tv[scale_mask] / vor_cum[scale_mask, 0]) if scale_mask.any() else 1.0
            tv = np.where(have_tv, tv, vor_cum[:, 0] * ratio)
            n_weeks = len(self.weeks)
            decay = np.array([(n_weeks - j) / n_weeks for j in range(n_weeks)])
            self.generic_value = np.clip(tv, 0, None)[:, None] * decay[None, :]
            self.value_source = "trade_value"
        else:
            self.generic_value = vor_cum
            self.value_source = "vor"
