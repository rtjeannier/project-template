# League inputs: the format the simulator reads

The simulator never fetches data. Each run reads one **league folder** with three JSON
files, written by whoever gathers the data (a person, or an agent using Yahoo,
FantasyPros, etc.):

```
leagues/guillotine-2025/week-10/
  rules.json         the league's rules (same every week of a season)
  snapshot.json      rosters, FAAB and the waiver pool, right before this week's waiver run
  projections.json   every relevant player's projection for every remaining week
```

Then:

```bash
python -m guillotine validate leagues/guillotine-2025/week-10
```

```bash
python -m guillotine run leagues/guillotine-2025/week-10 --seasons 20000
```

`validate` lists **every** problem at once (file, field and what is wrong) and exits with
code 2 if there are any; warnings don't stop a run. `run` validates, simulates, and writes
`sim_data/` and the report (`report/index.html` + data) inside the folder (`--data` and
`--site` change where).

A complete example: [`examples/league/`](../examples/league).

## Checklist for whoever writes the folder

1. `rules.json`: copy last week's; it only changes if the league changes its settings.
2. `snapshot.json`, taken **after** the latest cut and **before** the waiver run that
   follows it:
   - only teams still alive, each with its current FAAB and full roster (bench included);
   - `waiver_pool` = the players the just-cut team released (they're claimable this run);
   - `free_agents` = the best unrostered players (about 15 per position is plenty);
   - every player listed in a roster, the pool or `free_agents` defined once in `players`;
   - injuries: status and expected return week.
3. `projections.json`: for every player on a roster or in the pool, a projection for each
   remaining week from `snapshot.week` through `rules.last_week` (skip his bye). Use the
   league's scoring, or give stat lines and let `rules.scoring` turn them into points.
4. Run `validate` and fix everything it reports.

Ids are any strings, but must be the same across the three files. Using one source's ids
(e.g. Yahoo player keys) for everything avoids mismatches.

## rules.json

```json
{
  "format": "guillotine-rules/1",
  "name": "Guillotine 2025",
  "season": 2025,
  "last_week": 17,
  "roster": {
    "starters": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "W/R/T": 2},
    "bench": 7
  },
  "scoring": {
    "description": "half PPR",
    "stats": {"pass_yds": 0.04, "pass_td": 4, "pass_int": -1, "rush_yds": 0.1, "rush_td": 6,
              "rec": 0.5, "rec_yds": 0.1, "rec_td": 6, "fum_lost": -2, "two_pt": 2}
  },
  "eliminations": {"per_week": 1},
  "waivers": {"type": "faab", "budget": 1000, "min_bid": 0, "priority": "rolling"}
}
```

| Field | Meaning |
|---|---|
| `last_week` | The final week. The champion is the top scorer that week among the teams left. |
| `roster.starters` | Starting slots and counts. Positions: `QB RB WR TE K DEF`. Flex slots: `W/R/T`, `FLEX` (RB/WR/TE), `W/R`, `R/W`, `W/T`, `SUPERFLEX` and `Q/W/R/T` (QB/RB/WR/TE). |
| `roster.flex` | Optional: define your own flex slot, e.g. `{"RB/TE": ["RB", "TE"]}`. |
| `roster.bench` | Bench spots. Roster size = starters + bench. |
| `roster.ir` | IR spots. **Only 0 is supported for now.** |
| `scoring.stats` | Points per unit of each stat, used when projections are stat lines. Stat names are free-form but must match the projections. Optional if projections are points. |
| `scoring.by_position` | Extra points per stat for one position, e.g. TE premium `{"TE": {"rec": 0.5}}`. |
| `eliminations.per_week` | Teams cut each week. |
| `eliminations.by_week` | Different counts for some weeks, e.g. `{"10": 2}`. **Not supported yet.** |
| `waivers.type` | `faab`. (`priority`, waivers without money, is planned.) |
| `waivers.budget` | Season FAAB budget per team. |
| `waivers.min_bid` | Smallest bid allowed (0 if $0 bids are allowed). |
| `waivers.priority` | How ties between equal bids are broken: `rolling` (a winner moves to the back). |

For now the cuts must leave exactly 2 teams for the last week (a head-to-head final):
teams alive in `snapshot.week` minus the cuts in every week before `last_week` must be 2.
`validate` says so if they don't.

## snapshot.json

```json
{
  "format": "guillotine-snapshot/1",
  "week": 10,
  "focus_team": "9",
  "teams": [
    {"id": "9", "name": "Roland the headless", "faab": 412, "players": ["31883", "33393", "..."]}
  ],
  "eliminated": [{"id": "4", "name": "Beefcakes", "week": 9}],
  "waiver_pool": ["30123", "..."],
  "free_agents": ["28754", "..."],
  "players": [
    {"id": "31883", "name": "Jahmyr Gibbs", "pos": "RB", "nfl_team": "DET", "bye": 8},
    {"id": "30123", "name": "Some Receiver", "pos": "WR", "nfl_team": "NYJ", "bye": 9,
     "injury": {"status": "OUT", "return_week": 12, "note": "hamstring"}}
  ]
}
```

| Field | Meaning |
|---|---|
| `week` | The next week to be played. The waiver run being simulated happens before its games. |
| `focus_team` | The team the report calls "You". |
| `teams` | Teams still alive only: id, name, FAAB left, every rostered player id. |
| `eliminated` | Optional record of cut teams; not used by the simulation. |
| `waiver_pool` | Players released by the team cut most recently: claimable this run. |
| `free_agents` | Other unrostered players worth tracking. |
| `players[].bye` | Bye week, or `null`. |
| `players[].injury` | Optional. `status` `OUT`, `IR`, `PUP`, `SUSP` or `NFI` means he scores 0 until `return_week` (`null` = out for the season). Other statuses (`Q`, `D`) change nothing: build them into the projections instead. |

## projections.json

```json
{
  "format": "guillotine-projections/1",
  "source": "FantasyPros weekly projections, half PPR",
  "as_of": "2025-11-04",
  "players": {
    "31883": {"10": 19.4, "11": 18.9, "12": 18.7},
    "30123": {"12": {"rec": 5.1, "rec_yds": 61.0, "rec_td": 0.4}}
  }
}
```

Each player maps week → either **points** (a number, already in the league's scoring) or
a **stat line** (an object, scored with `rules.scoring`). Mixing both is fine. Weeks before
`snapshot.week` are ignored, so the file can carry the whole season.

Missing weeks are an error for rostered and waiver players (the bye, and weeks he's
ruled out by `injury`, don't need one) and a warning for free agents, who then count as 0.
Players with projections but not in `snapshot.players` are an error.

## What the simulator adds on its own

Week-to-week spread around each projection, projections drifting over the season,
random injuries, and how teams bid. These are modeling choices, not league facts, and
live in the code (`guillotine/projections.py`, `guillotine/simulate.py`,
`guillotine/bidding.py`).
