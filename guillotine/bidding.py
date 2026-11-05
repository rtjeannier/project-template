"""Bid model settings. The valuation is in ``Simulator._waiver_round``:

    V        = P(cut this week) x this week's gain over the best player available now
             + P(reach the final and lose it) x expected week-17 lineup gain
    money(m) = what m dollars still buy later (each slot's best returning player x the
               team's share of claims, plus injury cover), weighted by P(lose the final)
    fair bid = the most b with  V + money(FAAB - b | won)  >=  money(FAAB | lost)
    bid      = fair bid x multiplier ~ Normal(1, bid_sd) x aggression   (capped at FAAB)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BidModel:
    manager_sd: float = 0.0  # lognormal sd of per-manager aggressiveness; 0 = every manager is rational
    bid_noise_sd: float = 0.3  # lognormal noise on each bid, used only when bid_sd is None
    # Every bid = fair bid x a multiplier drawn from Normal(1, bid_sd). 0.61 puts the 95th
    # percentile at 2x fair; a draw at or below 0 is a $0 bid. None = old lognormal noise.
    bid_sd: float | None = 0.61
    pool_consider: int = 15  # only the top-N pool players (this week + week 17) get bids
    min_gain: float = 0.1  # weekly points gain over the free alternative below which a team doesn't bother

    def bid(self, *, value: float, opportunity: float, faab: float, aggression: float,
            noise: float, min_bid: int, lock_price: int | None = None) -> int:
        if faab < min_bid:
            return -1
        if value <= 0:
            return min_bid
        willingness = faab * value / (value + max(opportunity, 0.0))
        # Guarantee the win when it's worth it: topping every other team's whole budget
        # can't lose, so a rational manager pays exactly that, with no noise.
        if lock_price is not None and lock_price <= min(willingness, faab):
            return max(int(lock_price), min_bid)
        amount = min(willingness * aggression * noise, faab)
        return max(int(amount), min_bid)
