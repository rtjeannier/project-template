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

Managers are **rational with noise**: everyone uses this valuation, and each bid gets
random noise (`bid_noise_sd`) so it isn't predictable. There are no per-manager
personalities (`manager_sd` = 0), but you can set `aggression` per team in the league
file for a known hoarder or spender. **Lock:** if topping every other team's whole
FAAB costs less than the player is worth to a manager, they pay exactly that and
guarantee the win. Set `Simulator.bid_log = []` to record every simulated bid.

**Projection uncertainty grows with the horizon.** A player's true level is a random
walk around his projection: about ±10% for this week (`PROJ_ERROR_NOW`), with ±8% added
per week (`PROJ_ERROR_STEP`), so about ±25% by week 17. Managers account for it too:
the replacement level for a round far ahead is the expected k-th best of many uncertain
players, so some breakout is likely to be free. A stud's edge over replacement is
smaller and less certain far out, and firms up toward the final. For example, Walker's
week-17 edge is +1.3 pts seen from 8 weeks out and +2.3 in the final week.

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

**Who to target, and what to bid.** The report's main table is a plan. Candidates
are this week's best pool players plus the players on other rosters who are
**irreplaceable in the final**: projected above week-17 replacement (two teams left)
and likely to hit waivers while you can still bid. Whenever one is on waivers and you're alive, the
analysis first computes your **fair bid**: the rational bid for you at that moment,
using the same valuation rivals use, with no noise. It then simulates bidding **0.5×,
0.75×, 1×, 1.25×, 1.5×, 2× and 3× fair**, plus none, all-in and lock (top the richest
rival's whole budget), on the same random draws. If P(win league) peaks above 1×,
overbidding pays for that player (to block a rival, or because he's worth more than
later options); below 1×, underbidding does (let rivals overpay and keep the money). For every
bid it also shows how often you get him, and your FAAB left against the richest
rival's right after you win him: with twice their money, you can take later players
they can't afford.

The trade-offs are counted directly. Paying up now leaves less for later studs.
Letting a rival overpay makes later studs cheaper for you. Waiting only pays if you
survive to collect. The best bid per target forms the plan, which then runs as a
whole because the targets compete for the same budget. "Get him" only counts seasons
where he hit waivers while you were alive. `--future N` sets how many players from
other rosters are evaluated (default 5).

## Roster rules

Starting lineup: QB, 2 RB, 2 WR, TE, 2 W/R/T flex (a TE can start at flex). Bench rules,
by slot:

- at most 1 backup QB, and at most 2 backup TEs
- always keep at least 1 backup RB and 1 backup WR
- 15 players max. When over, a team drops its lowest-value bench player that keeps
  these rules, never a starter and never the player it just won.

Teams bid on a player if he improves their starting lineup, if keeping him from rivals
is worth something, or if he's a better backup than their worst bench player in his
slot (QB vs bench QBs; RB/WR/TE vs flex-eligible bench players). A team never bids on a
QB or TE who would just be dropped again to stay under the limits. A backup-only
upgrade has a fair bid of $1 and is bid at random as $0, $1 or $2, unless denial is
worth more. Each bid in the dataset is tagged `lineup`, `deny` or `bench`.

**Claims are add/drop pairs, as in Yahoo.** Once all bids are in, each team's claims
are given drops, going from the highest fair bid down and assuming the claims above
win. If the player would still help (lineup upgrade, better backup, or worth keeping
from rivals), his claim drops the next player (or uses the next open spot), so both
can go through. If not, it is a backup claim that shares the drop of the claim it
would replace. Claims are then processed league-wide from the highest bid down. A
claim is skipped if its drop is already gone ("invalid claim due to previous
transaction") or the team can no longer afford it. Each claim's fair bid uses the
team's whole FAAB, so total claims can exceed the budget and second choices aren't
bid artificially low. Bids and drops never change after submission.

## Simulated dataset

```bash
python -m guillotine simulate-data league.json --seasons 20000 --out sim_data
```

Plays every season straight through. Every team bids its fair bid times a random
multiplier drawn from Normal(1, 0.61), so the 95th percentile is 2× fair. Bidding is
free, so a draw at or below 0 (about 5%) is a $0 bid, not a skipped one. Each bid is drawn independently: there is no fixed strategy, and
the same cut sequence produces different waivers in each season. Since the multipliers
are random, comparing outcomes across them later is a fair experiment.

It writes the raw record and analyzes nothing: `bids.csv.gz` (every bid, with fair
bid, multiplier, FAAB, cut odds, won/price), `weeks.csv.gz` (scores, cuts, FAAB),
`drops.csv.gz`, `seasons.csv.gz` (champion, cut order), plus `players.csv`,
`teams.csv` and the starting `league.json`. Rosters at any point can be rebuilt from
the starting rosters plus awards and drops.

## Results explorer (GitHub Pages)

```bash
python -m guillotine build-site sim_data --out docs/explorer
```

This builds a static site from the simulated dataset into `docs/explorer/`. It has
`index.html` plus small pre-aggregated JSON files, because the raw bid log is too big
for a browser. To host it, go to the repo's Settings → Pages → Deploy from a branch,
pick this branch, and set the folder to `/docs`. The page is then at
`https://<user>.github.io/<repo>/explorer/`. Locally, run `python -m http.server`
in that folder.

The page has:

- **What a good bid looks like:** every real claim (not $0–2 backup pickups), win or
  lose, described only by observable things: bid size as a % of the bidder's FAAB,
  player tier (elite / starter / depth by final-week edge), time window, and FAAB vs
  the richest rival. For a chosen situation, it shows the change in the bidder's
  chance of winning the league at each bid size (vs the same team's average in that
  situation) and how often that bid landed the player, plus a "good bid by
  situation" table. No model "fair bid" is used. With data where bids were fair ×
  noise, bid % isn't randomized, so treat it as association until a run with bids
  drawn uniformly from 0–100% of FAAB.
- **Week by week, survival by choices:** pick a team and a week. Every season the team
  is alive going into that week is grouped by what it did that week (average bid vs
  fair, FAAB spent, whether it landed a lineup upgrade), and each group gets a survival
  curve (still alive after each later cut) plus its chance of winning the league.
  Bidding vs fair is randomized, so those differences are caused by bidding.
- **Paths to victory:** pick a team and see, week by week, how its winning seasons
  differ from its losing ones: win rate by FAAB left after each week's waivers (only
  seasons it's still alive), median FAAB left in won vs lost seasons, and the cost of
  missing each player (its win rate when it got him minus when it didn't, counting
  only seasons he hit waivers while it was alive), overall and by when he became
  available. These are associations: unlike bid multipliers, getting a player isn't
  randomized.
- **Players:** most irreplaceable first: median week bid on, times won, and median
  winning price as a % of the winner's FAAB.
- **Seasons:** sample seasons week by week, with every bid's fair value and multiplier,
  every award, scores and cuts.

Team names are shown as they are in the league (`--anonymize` replaces them with Team
1..N if ever needed).

## Inspecting scenarios

To check the algorithm, or to see when over- or under-bidding pays, export every
simulated season for one target:

```bash
python -m guillotine inspect league.json --player "Jahmyr Gibbs" --sims 300 --out inspect_gibbs
python -m guillotine viewer inspect_gibbs/inspect.json --out inspector.html
```

`inspect` runs every bid level (none, 0.5×–3× fair, all-in, lock) on the same random
draws and writes:

- `seasons.csv`: one row per (bid level, season). It has when he hit waivers, your fair
  bid and actual bid, the top rival bid, who won him and at what price, your FAAB after
  vs the richest rival's, when you were cut, and the champion.
- `bids.csv`: every team's bid on him, with their FAAB at the time.
- `timelines.json`: week-by-week detail (every award with all bids, scores, cuts,
  FAAB) for the first `--timelines` seasons of each level.

`viewer` builds one HTML page from any number of exports. It shows the bid curve with
a noise band, a filterable season table with sanity flags (an affordable higher bid
losing, a price that differs from your bid, a lock losing when you could afford to top
it, negative FAAB), the same season at every bid level, and the full timeline.

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
