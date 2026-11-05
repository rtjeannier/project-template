"""How managers bid FAAB on players released by the eliminated team.

Each manager prices a player in *their own budget*, not in a league-wide currency:

    willingness = FAAB × V / (V + future opportunity)
    bid         = willingness × shade × aggressiveness × noise      (capped at FAAB)

- **V (value now)** = how much the player raises *this team's* starting lineup per
  week, minus what the team could add for free at that position, times the number
  of weeks the team expects to stay alive. So Walker replacing Henry counts even
  though both are good RBs. The free alternative is the **dynamic replacement
  level**: the best player at the position that the remaining teams won't roster.
  It rises every week as eliminated rosters flood waivers and fewer teams compete.
  By the last weeks a solid starter can be at or below replacement.
- **Future opportunity** = upgrades the team expects to buy from later pools
  (estimated from the weakest teams' rosters), discounted by the chance the team
  survives to use them, and by competition for them. FAAB is use-it-or-lose-it: it
  disappears when the team is cut and is worthless after the final round. So
  opportunity shrinks as the season runs out, and in the last round a team will
  spend everything on an upgrade.
- **Survival**: P(cut) comes from each team's score distribution (see
  ``Simulator._cut_odds``). Teams in danger discount the future more and spend now.
- **shade**: in a sealed first-price auction people bid below what the player is
  worth to them. **aggressiveness** is drawn once per manager per season, **noise**
  once per bid.

Knobs live on ``BidModel``. If your league has a history (e.g. "studs go for 30–40%
of budget in week 2"), these are what to tune, or set ``aggression`` per team in the
league file for known big spenders or hoarders.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BidModel:
    # shade / manager_sd / bid_noise_sd are fit to a real guillotine league's FAB offers: every
    # bid on each round's top player, as a share of the bidder's FAAB, was p25 0.10, median 0.23,
    # p75 0.41, p90 0.56 (Sep-Oct 2026, 11-17 teams alive).
    shade: float = 0.6  # bid as a share of willingness to pay
    manager_sd: float = 0.75  # lognormal sd of a manager's aggressiveness (drawn once per sim)
    bid_noise_sd: float = 0.4  # lognormal sd of noise on each individual bid
    future_share: float = 3.0  # a team expects to land about future_share / n_alive of each future pool's best upgrade
    pool_consider: int = 15  # only the top-N pool players by projection get bids
    min_gain: float = 0.1  # weekly points gain over the free alternative below which a team doesn't bother

    def bid(self, *, value: float, opportunity: float, faab: float, aggression: float,
            noise: float, min_bid: int) -> int:
        if faab < min_bid:
            return -1
        if value <= 0:
            return min_bid
        willingness = faab * value / (value + max(opportunity, 0.0))
        amount = min(willingness * self.shade * aggression * noise, faab)
        return max(int(amount), min_bid)
