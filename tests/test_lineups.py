"""Leagues with different lineups: the money plan and slot logic follow the league's slots."""
import numpy as np
import pytest

from guillotine.lineup import LineupSpec
from guillotine.models import LeagueState
from guillotine.sample import make_sample_league
from guillotine.simulate import Simulator


def test_money_plan_has_one_pot_per_starting_spot():
    names = lambda slots: [n for n, _ in LineupSpec(slots).upgrade_slots()]
    # QB (no flex can take it) first, then the scarcest flex-eligible spots, flex last.
    assert names({"QB": 1, "RB": 2, "WR": 2, "TE": 1, "W/R/T": 2}) == \
        ["QB", "TE", "RB1", "RB2", "WR1", "WR2", "W/R/T#1", "W/R/T#2"]
    assert names({"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1, "K": 1, "DEF": 1, "BN": 6}) == \
        ["QB", "K", "DEF", "TE", "RB1", "RB2", "WR1", "WR2", "WR3", "FLEX#1"]
    sf = LineupSpec({"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "SUPERFLEX": 1}).upgrade_slots()
    assert sf[-1] == ("SUPERFLEX#1", ("QB", "RB", "WR", "TE"))
    assert len(sf) == 8


def test_positions_compete_only_through_the_leagues_flex_slots():
    wr = LineupSpec({"QB": 1, "RB": 2, "WR": 2, "TE": 1, "W/R": 1})
    assert wr.flex_group("RB") == ("RB", "WR") and wr.flex_group("TE") == ("TE",)
    assert wr.group("W/R") == ("RB", "WR") and wr.group("TE") == ("TE",)
    sf = LineupSpec({"QB": 1, "RB": 2, "WR": 2, "TE": 1, "SUPERFLEX": 1})
    assert sf.flex_group("QB") == ("QB", "RB", "WR", "TE")
    assert sf.flex_counts() == (0, 1)


def _league(slots: dict, seed: int = 3) -> LeagueState:
    d = make_sample_league(seed=seed, n_teams_alive=5, current_week=13)
    d["settings"]["roster_slots"] = slots
    d["settings"]["roster_size"] = sum(v for k, v in slots.items() if k != "BN") + 6
    return LeagueState.from_dict(d)


@pytest.mark.parametrize("slots", [
    {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "SUPERFLEX": 1},
    {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "W/R": 1, "K": 1, "DEF": 1},
])
def test_other_lineups_simulate_with_sane_bids(slots):
    sim = Simulator(_league(slots), seed=2)
    sim.value_log = []
    res = sim.run(n_sims=3)
    assert sum(res.champion_counts.values()) == 3
    assert sim.value_log
    for row in sim.value_log:
        slot, faab, fair = row[12], row[13], row[14]
        assert 0 <= fair <= max(faab, 0) + 1
        if "SUPERFLEX" not in slots:
            assert slot in {"QB", "RB", "WR", "TE", "K", "DEF", "RB/WR"}


def test_superflex_makes_a_second_qb_worth_money():
    # The same league with and without a superflex spot: a QB who'd only be a backup in a
    # 1-QB lineup can start in the superflex, so teams value QBs more.
    base = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}
    qb_value = {}
    for name, slots in (("1qb", base), ("sf", {**base, "SUPERFLEX": 1})):
        sim = Simulator(_league(slots), seed=4)
        sim.value_log = []
        sim.run(n_sims=3)
        qbs = [r[9] for r in sim.value_log if sim.proj.pos[sim.proj.index[r[2]]] == "QB"]
        qb_value[name] = float(np.sum(qbs))
    assert qb_value["sf"] > qb_value["1qb"]
