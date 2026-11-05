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
- **Denial**: every player a team holds is one its rivals don't. Points above the
  round's replacement level count 1/(teams alive − 1) extra, reaching the full amount
  in the final, where an irreplaceable player counts twice (for you and against your
  opponent) and a replaceable one is worth nothing to deny.
- **Future opportunity** = upgrades the team expects to buy from later pools
  (estimated from the weakest teams' rosters), discounted by the chance the team
  survives to use them, and by competition for them. FAAB is use-it-or-lose-it: it
  disappears when the team is cut and is worthless after the final round. So
  opportunity shrinks as the season runs out, and in the last round a team will
  spend everything on an upgrade.
- **Survival**: P(cut) comes from each team's score distribution (see
  ``Simulator._cut_odds``). Teams in danger discount the future more and spend now.
- **shade**: in a sealed first-price auction a rational bidder bids below what the
  player is worth to them. **noise** is drawn once per bid so bids aren't
  predictable. Managers are otherwise identical and rational (``manager_sd`` = 0).
  Set ``aggression`` per team in the league file to model a known hoarder or spender.
- **lock**: if topping every other team's whole FAAB costs less than the player is
  worth to a manager, they pay exactly that and guarantee the win.

Knobs live on ``BidModel``. If your league has a history (e.g. "studs go for 30–40%
of budget in week 2"), these are what to tune, or set ``aggression`` per team in the
league file for known big spenders or hoarders.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BidModel:
    shade: float = 0.6  # bid as a share of willingness to pay (rational first-price shading)
    manager_sd: float = 0.0  # lognormal sd of per-manager aggressiveness; 0 = every manager is rational
    bid_noise_sd: float = 0.3  # lognormal noise on each bid, used only when bid_sd is None
    # Every bid = fair bid x a multiplier drawn from Normal(1, bid_sd). 0.61 puts the 95th
    # percentile at 2x fair; a draw at or below 0 is a $0 bid. None = old lognormal noise.
    bid_sd: float | None = 0.61
    future_share: float = 3.0  # a team expects to land about future_share / n_alive of each future pool's best upgrade
    denial_weight: float = 1.0  # how much keeping a player from rivals counts (0 = ignore denial)
    pool_consider: int = 15  # only the top-N pool players by projection get bids
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
        amount = min(willingness * self.shade * aggression * noise, faab)
        return max(int(amount), min_bid)
