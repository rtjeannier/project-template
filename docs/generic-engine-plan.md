# Plan: a generic guillotine simulator fed by a weekly data process

Goal: one engine that simulates any guillotine league from three inputs, a separate data
layer that produces those inputs (Yahoo first), and a runner that does fetch → simulate →
report every week without hand edits.

```
league config ──┐
                ├─► data adapters (Yahoo, FantasyPros) ─► snapshot + projections (saved per week)
credentials ────┘                                              │
                                                               ▼
                                     engine (no network) ─► sim data ─► report / explorer
```

## 1. The inputs

The engine reads three files and nothing else. Each has a version number and a validator
that rejects anything it can't simulate correctly, with a message naming the field.

### 1a. League rules (`rules`): fixed for a season

| Group | Field | Examples |
|---|---|---|
| Season | `first_week`, `last_week` | 1, 17 |
| Roster | `slots` | `{"QB":1,"RB":2,"WR":2,"TE":1,"W/R/T":2}`; K, DEF, SUPERFLEX allowed |
| | `flex` | eligibility per flex slot: `{"W/R/T":["RB","WR","TE"]}` |
| | `bench`, `ir` | bench size; IR slots and which injury statuses may use them |
| Scoring | `scoring` | stat → points (`pass_yd: 0.04`, `rec: 0.5`, `rec_te: 1.0` for TE premium...) |
| Eliminations | `cuts` | per-week count, e.g. 1 a week, or `{"default":1,"10":2}` |
| | `cut_tiebreak` | what decides a tied lowest score (bench points, season points, ...) |
| | `final` | `head_to_head` (last 2 play the last week) or `last_standing` (cuts run to 1) or `top_score` (N teams in the last week, best score wins) |
| Waivers | `type` | `faab` or `priority` |
| | `budget`, `min_bid`, `zero_bids` | 1000, 0, true |
| | `priority` | `rolling` (winner goes last), `fixed`, `reverse_standings`; and the tiebreak between equal bids |
| | `cut_roster_clears` | when a cut team's players can be claimed (next waiver run) |
| | `free_agency` | whether unclaimed players can be added free between runs |

### 1b. League snapshot (`snapshot`): one per week, taken right before a waiver run

- week, teams (id, name, FAAB, waiver priority, roster with slot incl. IR)
- teams already cut (and the week)
- waiver pool (players from the latest cut roster) and free agents
- injuries: status and expected return, per player
- optional: transactions so far (claims, prices, drops) for the season replay

### 1c. Projections (`projections`)

Per player per remaining week, either stat lines (preferred: the engine applies the league's
scoring) or points already scored. Source and as-of date recorded.

### 1d. Model assumptions (not league rules)

Injury rates, projection error, bid noise, bench keep counts. Defaults ship with the engine;
a league can override them. Kept separate so "the league's rules" and "our modeling
choices" never mix.

## 2. Engine changes (what is hard-coded today)

| Today | Becomes |
|---|---|
| Final = 2 teams, head to head, in week 17 | `final` rule; "week 17" becomes "the final week" everywhere, including the report text |
| One cut a week assumed in odds and round count | `cuts` schedule drives the rounds and survival odds |
| Money pots fixed at QB, TE, RB1, RB2, WR1, WR2, FLEX1, FLEX2 | One pot per starting slot from `slots`, flex slots last; K/DEF included when the league starts them |
| Position lists in injury rates, bench limits, backups, flex constants | Read from the roster rules; defaults per position in the model assumptions |
| Rolling priority starting from a random order | `priority` rule, starting from the snapshot's real priority |
| FAAB only | `priority` waivers too (a claim's cost is the priority, not money) |
| Points per week given | Points computed from stat projections with `scoring` (points still accepted) |
| Your team = `my_team_id` in the state file | Optional "focus team" for the report; the report works with none |

Every change is checked two ways: synthetic leagues that exercise the rule (superflex,
K/DEF, 2 cuts a week, last-standing final, priority waivers, TE premium), and a regression
run: the week-9 Guillotine 2025 inputs converted to the new format give the same results as
today with the same seed (or the differences are explained, one by one).

## 3. Data layer (Yahoo first)

- **Yahoo Fantasy API**: league settings → `rules`; rosters, FAAB, priority, cut teams →
  `snapshot`; transactions → history. Needs a Yahoo developer app (you register it) and an
  OAuth refresh token, stored as a credential on the cloud environment, never in the repo.
- **Projections**: FantasyPros API (needs `FANTASYPROS_API_KEY`; the claude.ai connector
  can't be called from a script). Week-by-week stat projections for every remaining week.
- **Player IDs**: Yahoo and FantasyPros IDs differ. A mapping table built by name + team +
  position, with unmatched players listed in the run log instead of silently zeroed.
- Every fetch is saved raw, then converted, so a bad week can be re-converted without
  refetching.

## 4. Weekly runner

```
guillotine run leagues/guillotine-2025.json            # fetch, save, simulate, build, publish
guillotine run leagues/guillotine-2025.json --no-fetch # rerun the saved snapshot
```

- Saves `data/<league>/<season>/week-NN/{rules,snapshot,projections}.json`, so the season's
  history builds itself.
- Number of seasons is a setting; it gets faster as teams are cut (~2.5 h for 20k seasons
  on 4 cores with 10 teams left).
- Scheduling: a scheduled Claude task (no time limit) or GitHub Actions (6 h job limit,
  needs the credentials as repo secrets). Decided when the data layer works.

## 5. Season replay (after the weekly runner)

From the saved weekly snapshots (or Yahoo's transaction log for past weeks): each week's
title odds per team, the model's fair price for every claim made, and each manager's
moves graded by what was known at the time, with luck (actual vs projected scores) shown
separately.

## Order of work

1. Input formats + validator + converter for the current week-9 file; regression baseline.
2. Engine generalization, one rule group at a time, each with its synthetic-league tests.
3. Yahoo + FantasyPros adapters.
4. Weekly runner and scheduling.
5. Season replay.

## Questions to settle along the way

- Your league's tiebreak for the lowest score, and whether Yahoo applies a minimum bid.
- Does the data branch hold league data (snapshots, reports) or only the engine? Proposal:
  engine repo holds code + a tiny example league; each real league gets its own data
  folder or repo that the runner writes to.
