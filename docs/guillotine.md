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
      P(team is the one cut in round r) × P(player still healthy)
      × [ max(0, lineup gain − free gain)  +  max(0, points − replacement) / (teams alive − 1) ]
```

- **Denial**: every player you hold is one a rival doesn't. Points above the round's
  replacement level count an extra 1/(teams alive − 1), reaching the full amount in
  the final. Picture your final opponent as the best team that can be built from
  everyone else: an irreplaceable player counts twice (for you and against them),
  and a replaceable one is worth nothing to deny (`denial_weight`).
- **Lineup gain**: how much the player raises *that team's* starting lineup. Walker
  replacing Henry counts even though both are good RBs.
- **Free gain / dynamic replacement**: what the team could add for $0. The
  replacement level at a position is the best player the remaining teams won't
  start. It rises every round as eliminated rosters flood waivers. With few teams
  left, a solid starter can be at or below replacement.
- **P(cut in round r)**: each team's weekly score is treated as roughly normal. The
  mean is its best lineup's projection. The spread comes from its starters' weekly
  variance plus season-long projection error, so a boom-or-bust lineup carries more
  cut risk than a steady one with the same projection. P(cut) is the chance of
  posting the lowest score, from 2,000 fixed draws. Given FantasyPros' own team
  projections, this reproduces their week 9 safe % within 0.9 points on average,
  with no tuning. Odds are computed before waivers from current rosters and this
  week's projections, so byes and injuries known ahead of time count. They're
  recalculated after every pickup, so later claims in the same round see the new
  odds. For later rounds, the team most likely to be cut is removed and the rest
  renormalized. The last round (two teams) is the final, and "cut" means losing it.
  A safe team's weight sits in the final, where replacement is highest, so it waits
  for studs. A team on the bubble values points now.

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

**Injuries.** Each player has a weekly chance of getting hurt in a game
(`INJURY_RATE`: RB 2.5%, WR/TE 2%, QB 1.5%). 35% of injuries end the season; the rest
cost 1–4 weeks. Over weeks 9–17 that's about an 18% chance an RB misses time, and 8%
of players are out for the final. Managers only learn about an injury after it
happens: hurt players are benched, and their waiver value shrinks to the weeks
they'll be back for. When valuing a player, a manager discounts each later round by
the chance he's still healthy then, (1 − rate)^weeks. A stud bought for the final
has more weeks to get hurt in than one bought for this week. Rates are kept modest
because ROS projections already price in some missed time.

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

**Who to target.** The report's main table is a plan. Candidates are this week's best
pool players plus the players on other rosters who are **irreplaceable in the final**:
projected above week-17 replacement (two teams left) and likely to hit waivers while
you can still bid. For each, the analysis tries the standing rule "whenever he's on
waivers and I'm alive, bid X% of my FAAB" for X = never, $1, 15, 30, 50, 75%, on the
same random draws. It keeps the X with the best P(win league), then runs all the
rules together, since they compete for the same budget. "P(you get him)" only counts
seasons where he hit waivers while you were still alive. Rivals' budgets and bids come
from the same simulation. `--future N` sets how many players from other rosters are
evaluated (default 6).

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
