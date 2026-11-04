# Guillotine league simulator

Simulates the rest of a guillotine fantasy football season thousands of times, with
waiver bidding included, to answer:

- What are my chances of surviving each week, and of winning the league?
- Where are my roster holes compared with the teams still alive?
- Who in this week's waiver pool should I bid on, how much do I need to bid to beat
  the other managers, and is the player worth the FAAB?
- Which players on other rosters are likely to hit waivers later, and what will they cost?

## Quick start

```bash
pip install numpy pytest
python -m guillotine sample --out examples/sample_league.json   # synthetic league
python -m guillotine analyze examples/sample_league.json --sims 400 --out report.md
python -m pytest
```

See `examples/sample_report.md` for output from the synthetic league.

## How a simulated week works

1. **Waivers.** The pool is last week's eliminated roster plus leftover free agents.
   Every active team bids (model below). The highest bid anywhere is processed
   first; ties go to rolling waiver priority. Bids above a team's remaining budget
   are skipped. A team that wins someone re-values the remaining players before its
   next claim, so it won't buy three RBs it can't start. Over-full rosters drop their
   least valuable bench player.
2. **Games.** Each team starts its best projected lineup. Each player-week score is
   drawn from a normal distribution (floored at 0) with the projected mean and a
   position-based spread (QB 40% of the mean ... TE/DEF 65%). Byes score 0.
3. **Guillotine.** The lowest score is cut. That roster becomes next week's pool and
   the team's unspent FAAB is lost.

## How managers bid (`guillotine/bidding.py`)

Before the final, points only matter if they decide who gets cut. So each manager
values a player by **elimination-weighted points above replacement**:

```
value(player, team) = Σ over remaining rounds r of
      P(team is the one cut in round r) × max(0, lineup gain − free gain at that point)
```

- **Lineup gain**: how much the player raises *that team's* starting lineup. Walker
  replacing Henry counts even though both are good RBs.
- **Free gain / dynamic replacement**: what the team could add for $0. The
  replacement level at a position is the best player the remaining teams won't
  start. It rises every round as eliminated rosters flood waivers. With few teams
  left, a solid starter can be at or below replacement.
- **P(cut in round r)**: nobody knows next week's odds before waivers run, so bids
  use last week's post-waiver elimination odds, with the eliminated team removed and
  the rest renormalized. After the round's pickups, the odds are recalculated from
  the new lineups, and those carry into next week's bidding. Week 1 of the
  simulation has no previous week, so it uses the current rosters (`cut_sd` = 10
  matches FantasyPros' week 9 safe %). Later rounds within one valuation repeat the
  same rule: remove the team most likely to be cut, then renormalize. The last round
  (two teams) is the final, and "cut" means losing it. A safe team's weight sits in
  the final, where replacement is highest, so it waits for studs. A team on the
  bubble values points now.

A manager then spends a share of **their own budget**:

```
willingness = FAAB × value / (value + future opportunity)
bid         = willingness × shade × aggressiveness × noise        (capped at FAAB)
```

**Future opportunity** is the same weighted value for the best upgrade expected in
later pools (from the weakest teams' rosters), scaled by competition
(`future_share / teams alive`). FAAB is use-it-or-lose-it, so it disappears when a
team is cut. In the last round there's no future, so a team will spend its whole
budget on an upgrade.

`shade`, `manager_sd` and `bid_noise_sd` are fit to a real league's FAB offers:
every bid on each round's top player, as a share of the bidder's remaining FAAB,
was p25 0.10, median 0.23, p75 0.41, p90 0.56. The simulator matches that within
about 0.02 for weeks 9–12. Its 90th percentile is a bit high (0.67), because a few
managers go all-in. For a known big spender or hoarder, set `aggression` per team in
the league file. Set `Simulator.bid_log = []` to record every simulated bid for
re-calibration.

Scores include a season-long projection error per player (`ROS_ERROR_SD` = 20%) on
top of weekly noise, so breakouts and busts happen and long shots keep a real chance.

## Decisions (`guillotine/analysis.py`)

Scenarios run on **common random numbers**: each simulated season gets the same
player outcomes in every scenario, so differences come from decisions, not luck.

- **pass**: you skip this week's waivers (the baseline for comparisons).
- **auto**: you bid like the model.
- **bid $Y on X**: your sealed bid enters the auction like everyone else's. You win X
  only if it beats the field (ties go to waiver priority).

You never skip a player: for each candidate the analysis tries $1 and the bids that
beat the field 25/50/75/90% of the time. It recommends the bid with the highest
P(win league), even if that's a $1 lottery ticket. All recommended bids are then
submitted together as a **bid sheet**, with its own P(win league). Budgets are
enforced the way Yahoo does it: a claim you can no longer afford is skipped.

## League state file

See the docstring in `guillotine/models.py`. One JSON file holds settings, teams
(FAAB, roster), this week's waiver pool, notable free agents, and player projections
(`ros_points`, optional per-week `weekly`, optional `trade_value`).

## Data sources

- **FantasyPros**: `python -m guillotine fetch-fantasypros --season 2026 --week 10`
  saves raw responses to `data/raw/fantasypros/` (git-ignored). Needs
  `FANTASYPROS_API_KEY` in the environment or `.env`. Unverified against the live
  API; the converter into the league file is the next step.
- **Yahoo (rosters, FAAB, eliminations)**: not wired up yet. Until then, fill in
  `teams`/`waiver_pool` by hand.

## Known limitations

- One elimination round per week. Multi-elimination weeks are supported by the
  setting, but not tested against real Yahoo rules.
- Free agents can only be claimed through the weekly waiver round. There is no
  free-agent pickup after waivers clear.
- You can be forced to win only one player per scenario. Combinations aren't
  evaluated yet.
- Player scores are independent: no QB/WR stacking correlation, no injuries beyond
  the projection spread.
