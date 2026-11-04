import numpy as np
import pytest

from guillotine.analysis import analyze, bid_needed, p_win_at
from guillotine.lineup import LineupSpec
from guillotine.models import LeagueState
from guillotine.sample import make_sample_league
from guillotine.simulate import Simulator, UserPolicy


@pytest.fixture(scope="module")
def state():
    return LeagueState.from_dict(make_sample_league(seed=3, n_teams_alive=5, current_week=13))


@pytest.fixture(scope="module")
def sim(state):
    return Simulator(state, seed=1)


def test_lineup_fills_dedicated_then_flex():
    spec = LineupSpec({"QB": 1, "RB": 2, "WR": 2, "FLEX": 1, "BN": 5})
    pos = ["QB", "RB", "RB", "RB", "WR", "WR", "WR", "TE"]
    vals = [20, 15, 12, 11, 14, 9, 8, 10]
    total, starters = spec.best(list(range(8)), vals, pos)
    # QB 20, RB 15+12, WR 14+9, FLEX = best of RB 11 / WR 8 / TE 10 -> 11
    assert total == 20 + 15 + 12 + 14 + 9 + 11
    assert set(starters) == {0, 1, 2, 3, 4, 5}


def test_lineup_reports_empty_slots():
    spec = LineupSpec({"QB": 1, "TE": 1})
    slots = spec.assign([0], [10.0], ["QB"])
    assert slots == [("QB", 0), ("TE", None)]


def test_bye_week_scores_zero(state, sim):
    p = next(p for p in state.players.values() if p.bye in state.settings.weeks)
    i, j = sim.proj.index[p.id], state.settings.weeks.index(p.bye)
    assert sim.proj.means[i, j] == 0


def test_season_accounting(state, sim):
    res = sim.run(40)
    n_weeks = len(res.weeks)
    # every sim crowns exactly one champion
    assert sum(res.champion_counts.values()) == 40
    # at most (teams - 1) eliminations per sim, one per week here
    total_elims = sum(c.sum() for c in res.elim_counts.values())
    assert total_elims == 40 * min(n_weeks, len(state.teams) - 1)


def test_budgets_never_go_negative(state, sim):
    res = sim.run(30)
    start = {t.id: t.faab for t in state.teams}
    spent = {}
    for s, _, _, tid, amt in res.awards:
        spent[(s, tid)] = spent.get((s, tid), 0) + amt
        assert amt >= state.settings.min_bid
    for (s, tid), total in spent.items():
        assert total <= start[tid]


def test_forced_award_is_applied(state, sim):
    pid = max(sim.run(5).user_gain_now.items(), key=lambda kv: kv[1])[0]
    res = sim.run(10, UserPolicy("force", pid, 1))
    first_round = [(s, p, t, a) for s, j, p, t, a in res.awards if j == 0 and p == pid]
    assert len(first_round) == 10
    assert all(t == state.my_team_id and a == 1 for _, _, t, a in first_round)


def test_calibration_matches_pacing_plan(sim):
    actual = planned = 0.0
    for s in range(40):
        a, p = sim._run_one(50_000 + s, UserPolicy("auto"), sim._scratch_results(), probe=True)
        actual += a
        planned += p
    assert actual == pytest.approx(planned, rel=0.35)


def test_bid_curve_helpers():
    opp = np.array([0, 10, 20, 30, 40])
    assert p_win_at(opp, 25) == 0.6
    assert p_win_at(opp, bid_needed(opp, 0.5)) >= 0.5


def test_analyze_end_to_end(state):
    a = analyze(state, n_sims=30, top_k=2, seed=0)
    assert a.holes and a.opponents
    assert len(a.candidates) <= 2
