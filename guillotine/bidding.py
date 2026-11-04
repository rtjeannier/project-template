"""How managers bid FAAB on players released by the eliminated team.

The model is built from four ideas:

1. **Market price per unit of value.** Each player has a generic value (trade value
   or value over replacement). The league's going rate converts value to dollars:

       price_per_unit = spend_ratio * FAAB held by active teams
                        / (value in this week's pool + value expected in future pools)

   This is what makes managers save money. Early on, many rosters are still to be
   released, so each dollar must stretch further and bids are a fraction of a
   player's "worth". Near the end, few pools remain and the same money chases less
   value, so prices rise. Money left when a team is eliminated is wasted, and the
   model reflects that.

2. **Team-specific need.** A team bids on how much a player improves *its*
   starting lineup (best-in-slot), compared with how much he'd help a generic team.
   A stud RB is worth more to a team starting a replacement-level RB2, and close to
   nothing to a team whose RBs he wouldn't start over.

3. **Danger.** Teams projected near the bottom value immediate points more.

4. **Wealth and personality.** Richer teams bid more. Each manager has a per-sim
   aggressiveness draw, and each bid gets noise. No team commits more than a capped
   share of its budget to one player until the last weeks.

The winning bid is the *maximum* of several noisy bids, so it sits well above the
average bid (the winner's curse). Without a correction the simulated league would
blow its budget in the first couple of weeks. ``pace_scale`` scales every bid so
that what the league actually pays in a round matches the pacing plan above. It is
calibrated automatically (see ``Simulator.calibrate``) unless you set it, e.g.
from your league's real spending history.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class BidModel:
    spend_ratio: float = 0.85  # share of active FAAB the league actually spends on what remains
    manager_sd: float = 0.25  # lognormal sd of a manager's aggressiveness (drawn once per sim)
    bid_noise_sd: float = 0.30  # lognormal sd of noise on each individual bid
    max_need: float = 1.6  # cap on need multiplier
    danger_boost: float = 0.5  # extra multiplier for the team projected lowest
    wealth_exponent: float = 0.5  # how strongly a big budget inflates bids
    pool_consider: int = 15  # only the top-N pool players by projection get bids
    min_gain: float = 0.1  # weekly points gain below which a team doesn't bother
    pace_scale: float | None = None  # None = auto-calibrate to the pacing plan

    def cap_fraction(self, weeks_left: int) -> float:
        """Most of its remaining budget a team will put on a single player."""
        if weeks_left <= 2:
            return 1.0
        return min(1.0, 1.5 / math.sqrt(weeks_left))

    def danger(self, rank_from_bottom: int, n_active: int) -> float:
        if n_active <= 1:
            return 1.0
        pct = rank_from_bottom / (n_active - 1)  # 0 = projected lowest
        return 1.0 + self.danger_boost * (1.0 - pct) ** 2

    def need(self, team_gain: float, generic_gain: float) -> float:
        if generic_gain <= 0.01:
            return 1.0
        return min(self.max_need, team_gain / generic_gain)

    def bid(
        self,
        *,
        price_per_unit: float,
        generic_value: float,
        need: float,
        danger: float,
        faab: float,
        avg_faab: float,
        aggression: float,
        noise: float,
        weeks_left: int,
        min_bid: int,
    ) -> int:
        wealth = (faab / avg_faab) ** self.wealth_exponent if avg_faab > 0 else 1.0
        amount = price_per_unit * generic_value * need * danger * wealth * aggression * noise
        amount = min(amount, faab * self.cap_fraction(weeks_left), faab)
        return max(int(amount), min_bid) if faab >= min_bid else -1
