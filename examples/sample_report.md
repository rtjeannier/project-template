# Guillotine analysis: week 10, 2026

9 teams alive, 8 weeks left, 400 simulated seasons. Player values from **trade_value**.

## You: $814 FAAB

- Chance to win the league: **22%** if you follow the model's bids, 22% if you pass this week
- Chance you're still alive after each week: wk10 91%, wk11 83%, wk12 77%, wk13 72%, wk14 64%, wk15 53%, wk16 40%, wk17 22%

### Roster holes (your starter vs the league median at that slot, avg pts/week)

| Slot | Starter | You | League median | Gap | Rank |
|---|---|---:|---:|---:|---:|
| QB | Sample QB 4 | 19.5 | 18.9 | +0.5 | 4/9 |
| RB | Sample RB 2 | 19.8 | 16.5 | +3.2 | 1/9 |
| RB | Sample RB 21 | 11.0 | 12.9 | -1.8 | 9/9 |
| WR | Sample WR 17 | 15.6 | 16.6 | -1.0 | 7/9 |
| WR | Sample WR 2 | 13.0 | 13.1 | -0.2 | 6/9 |
| WR | Sample WR 24 | 12.8 | 12.8 | +0.0 | 5/9 |
| TE | Sample TE 2 | 14.4 | 10.3 | +4.1 | 1/9 |
| K | Sample K 1 | 9.8 | 8.3 | +1.5 | 1/9 |
| DEF | Sample DEF 9 | 6.7 | 7.4 | -0.7 | 8/9 |
| FLEX | Sample WR 35 | 10.2 | 11.4 | -1.2 | 9/9 |

## This week's waiver targets

**Bid for X%** = the bid that beats every other manager in X% of simulations. Effects compare winning the player at the suggested bid against passing on waivers this week.

| Player | Pos | +pts/wk for you | Bid for 50% | Bid for 75% | Bid for 90% | Suggested | Δ P(win league) | Δ P(alive 3 wks) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Sample RB 4 | RB | 8.8 | $251 | $275 | $322 | $275 (80%) | +0.0 pts | +5.8 pts |
| Sample DEF 1 | DEF | 3.2 | $176 | $212 | $271 | $212 (75%) | +1.5 pts | +5.2 pts |
| Sample WR 10 | WR | 2.8 | $147 | $178 | $214 | $178 (75%) | +0.3 pts | +0.0 pts |
| Sample WR 15 | WR | 1.8 | $113 | $136 | $173 | $136 (75%) | -2.0 pts | +0.7 pts |
| Sample TE 8 | TE | 1.5 | $134 | $164 | $210 | $164 (75%) | -1.5 pts | -2.2 pts |
| Sample RB 27 | RB | 0.7 | $0 | $0 | $0 | $0 (100%) | -1.0 pts | +1.5 pts |

Baseline if you pass this week: P(win league) 22%, P(alive after 3 wks) 74%.
Differences smaller than about ±2.9 pts are within simulation noise at 400 sims; rerun with more `--sims` before close calls.

## Future targets: players likely to hit waivers later

| Player | Pos | Current team | P(hits waivers) | Typical week | +pts/wk for you | Median price | P(you get him) |
|---|---|---|---:|---:|---:|---:|---:|
| Sample RB 5 | RB | Kicker Truthers | 92% | 14 | 5.2 | $55 | 16% |
| Sample WR 11 | WR | Kicker Truthers | 92% | 14 | 3.9 | $49 | 7% |
| Sample RB 11 | RB | Waiver Wolves | 90% | 15 | 3.2 | $19 | 11% |
| Sample RB 1 | RB | Bench Mob | 94% | 13 | 3.1 | $45 | 14% |
| Sample WR 4 | WR | Waiver Wolves | 90% | 15 | 3.1 | $28 | 16% |
| Sample RB 14 | RB | Late Round QB | 87% | 14 | 2.5 | $42 | 8% |
| Sample WR 7 | WR | Bye Week Blues | 93% | 15 | 2.3 | $26 | 12% |
| Sample RB 8 | RB | Faab Hoarders | 88% | 14 | 2.3 | $35 | 10% |
| Sample RB 3 | RB | Sunday Scaries | 88% | 15 | 2.3 | $18 | 10% |
| Sample WR 8 | WR | Faab Hoarders | 88% | 14 | 2.3 | $29 | 7% |
| Sample WR 1 | WR | Sunday Scaries | 88% | 15 | 2.3 | $19 | 12% |
| Sample WR 18 | WR | Late Round QB | 78% | 13 | 2.3 | $34 | 4% |
| Sample RB 7 | RB | Late Round QB | 92% | 14 | 1.9 | $32 | 16% |
| Sample RB 15 | RB | Kicker Truthers | 58% | 12 | 2.6 | $68 | 7% |
| Sample WR 3 | WR | Gridiron Ghosts | 86% | 15 | 1.5 | $14 | 10% |

## League board

| Team | FAAB | Proj pts/wk | P(cut this week) | P(win league) | Weakest slots |
|---|---:|---:|---:|---:|---|
| Sunday Scaries | $523 | 138.4 | 3% | 12% | RB |
| **You** | $814 | 132.8 | 9% | 22% | RB, FLEX |
| Gridiron Ghosts | $656 | 130.8 | 6% | 14% | WR, WR |
| Late Round QB | $243 | 128.1 | 8% | 8% | TE, WR |
| Faab Hoarders | $509 | 127.8 | 10% | 12% | QB, WR |
| Waiver Wolves | $503 | 126.9 | 12% | 10% | WR, QB |
| Kicker Truthers | $335 | 125.7 | 22% | 8% | QB, WR |
| Bench Mob | $517 | 122.6 | 18% | 6% | RB, K |
| Bye Week Blues | $724 | 122.0 | 12% | 7% | QB, RB |
