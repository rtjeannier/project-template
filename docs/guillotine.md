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

```
bid = price_per_unit × value(player) × need(team) × danger(team) × wealth(team)
      × aggressiveness(manager) × noise × pace_scale
```

- **value**: trade value (decays as weeks run out) or value over replacement.
- **price_per_unit** = 85% of active teams' FAAB ÷ (value in this pool + expected
  value of future pools). Future pools are estimated from the weakest teams' rosters.
  This is the "save money for later" mechanism: with many pools still to come, each
  dollar has to stretch further.
- **need**: how much the player improves *that team's* starting lineup, relative to
  a generic team. Best-in-slot: zero if he wouldn't start.
- **danger**: teams projected near the bottom pay up to 50% more for points now.
- **wealth**: √(team FAAB ÷ league average FAAB).
- **aggressiveness/noise**: per-manager and per-bid lognormal randomness.
- **cap**: at most 1.5/√(weeks left) of the remaining budget on one player, and no
  cap in the last two weeks.
- **pace_scale**: the winning bid is the *maximum* of several noisy bids, so it runs
  hot. `pace_scale` is auto-calibrated so actual first-round spending matches the
  pacing plan.

Knobs live on `BidModel`. If your league has a history (e.g. "studs go for 30–40%
of budget in week 2"), these are what to tune, or set `aggression` per team in the
league file for known big spenders or hoarders.

## Decisions (`guillotine/analysis.py`)

Scenarios run on **common random numbers**: each simulated season gets the same
player outcomes in every scenario, so differences come from decisions, not luck.

- **pass**: you skip this week's waivers.
- **auto**: you bid like the model.
- **force X at $Y**: you win X at the bid that beats the field 75% of the time.

The waiver table reports the bid needed to beat every opponent 50/75/90% of the
time, and the change in P(win league) and P(alive in 3 weeks) vs passing.

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
