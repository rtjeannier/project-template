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


def test_last_round_bid_uses_most_of_budget():
    from guillotine.bidding import BidModel
    m = BidModel(bid_noise_sd=0.0)
    # No future rounds left: an upgrade is worth (almost) the whole budget.
    assert m.bid(value=10, opportunity=0, faab=500, aggression=1, noise=1, min_bid=0) == int(500 * m.shade)
    # Plenty of future opportunity: same player gets a smaller share.
    assert m.bid(value=10, opportunity=90, faab=500, aggression=1, noise=1, min_bid=0) == int(50 * m.shade)


def test_replacement_level_rises_as_teams_are_cut(state, sim):
    vb = sim.proj.value_basis[0]
    universe = set(range(len(sim.proj.ids)))
    many, few = sim._replacement(vb, universe, 10), sim._replacement(vb, universe, 3)
    assert all(few[p] >= many[p] for p in many)
    assert few["RB"] > many["RB"]


def test_bid_curve_helpers():
    opp = np.array([0, 10, 20, 30, 40])
    assert p_win_at(opp, 25) == 0.6
    assert p_win_at(opp, bid_needed(opp, 0.5)) >= 0.5


def test_analyze_end_to_end(state):
    a = analyze(state, n_sims=30, top_k=2, seed=0)
    assert a.holes and a.opponents
    assert len(a.candidates) <= 2


def test_sealed_bid_only_wins_when_highest(state, sim):
    base = sim.run(20)
    pid = max(base.user_gain_now.items(), key=lambda kv: kv[1])[0]
    res = sim.run(20, UserPolicy("bid", bids={pid: 1}))
    opp = base.opp_max_bid[pid]
    won = {s for s, j, p, t, a in res.awards if j == 0 and p == pid and t == state.my_team_id}
    # A $1 bid wins exactly in the sims where nobody else bid more than $1 (ties may go either way).
    assert all(opp[s] <= 1 for s in won)
    assert all(s in won for s in range(20) if opp[s] < 1)


def test_elimination_weights_sum_to_at_most_one():
    w = Simulator._elimination_weights({"a": 0.5, "b": 0.3, "c": 0.2}, [3, 2])
    for t, ws in w.items():
        assert 0 <= sum(ws) <= 1 + 1e-9
    # Someone is cut every round: the weights for each round sum to 1 across teams.
    assert abs(sum(w[t][0] for t in w) - 1) < 1e-9


def test_injured_players_score_zero_and_lose_waiver_value(state, sim):
    sim._draw_injuries(np.random.default_rng(5))
    out = np.argwhere(sim._avail == 0)
    assert len(out) > 0
    p, k = out[0]
    assert sim._known_value(k)[p] <= sim.proj.value_basis[k][p]
    # Nobody is out in the first week: an injury happens in a game and costs later weeks.
    assert sim._avail[:, 0].all()


def test_lock_rule_tops_every_interested_rival(state, sim):
    base = sim.run(15)
    pid = max(base.user_gain_now.items(), key=lambda kv: kv[1])[0]
    res = sim.run(15, UserPolicy("auto", rules={pid: "lock"}))
    me = state.my_team_id
    my_faab = state.team(me).faab
    rich = max(t.faab for t in state.teams if t.id != me)
    if my_faab > rich:
        # Topping the richest rival's whole budget can't lose this week.
        assert sum(1 for s, j, p, t, a in res.awards if j == 0 and p == pid and t == me) == 15
    assert all(b <= my_faab for b in res.user_rule_bids.get(pid, []))


def test_uncertain_replacement_is_higher_far_out(state, sim):
    vb = sim.proj.value_basis[-1]
    universe = set(range(len(sim.proj.ids)))
    near, far = sim._replacement(vb, universe, 2, 0.1), sim._replacement(vb, universe, 2, 0.3)
    assert far["RB"] > near["RB"]


def test_inspect_export_and_viewer(state, tmp_path):
    from guillotine.inspect import inspect_player, write
    from guillotine.viewer import build

    sim = Simulator(state, seed=1)
    pid = max(sim.run(5).user_gain_now.items(), key=lambda kv: kv[1])[0]
    data = inspect_player(state, pid, n_sims=6, n_timelines=2, rules=("none", "x1", "x2"))
    assert len(data["seasons"]) == 18 and len(data["summary"]) == 3
    for row in data["seasons"]:
        if row["price"] is not None and row["won_by"] == state.team(state.my_team_id).name:
            assert row["price"] == row["your_bid"]
    paths = write(data, tmp_path)
    html = build([paths[-1]], tmp_path / "inspector.html").read_text()
    assert "/*DATA*/" not in html and data["player"] in html


def test_winning_price_equals_submitted_bid(state):
    # Sealed bids: whoever wins pays exactly what they bid, never a re-priced amount.
    sim = Simulator(state, seed=2)
    sim.trace = []
    sim.run(15)
    for e in sim.trace:
        if e[0] == "award":
            _, s, j, pid, tid, amt, all_bids, faab = e
            assert all_bids[tid] == amt


def test_simulated_dataset_is_consistent(state, tmp_path):
    import csv
    import gzip
    import json

    from guillotine.dataset import generate

    league = tmp_path / "league.json"
    league.write_text(json.dumps(make_sample_league(seed=3, n_teams_alive=5, current_week=13)))
    generate(state, league, n_seasons=8, out_dir=tmp_path / "data", chunk=3)
    read = lambda n: list(csv.DictReader(gzip.open(tmp_path / "data" / n, "rt")))
    bids, seasons = read("bids.csv.gz"), read("seasons.csv.gz")
    assert len(seasons) == 8 and bids
    for b in bids:
        assert int(b["bid"]) <= float(b["faab"])
        if b["won"] == "True":
            assert int(b["price"]) == int(b["bid"])
    # exactly one winner per (season, week, player) awarded
    winners = {}
    for b in bids:
        if b["won"] == "True":
            key = (b["season"], b["week"], b["player_id"])
            assert key not in winners
            winners[key] = b["team_id"]


def test_site_build_from_simulated_data(state, tmp_path):
    import json

    from guillotine.dataset import generate
    from guillotine.site import build

    league = tmp_path / "league.json"
    league.write_text(json.dumps(make_sample_league(seed=3, n_teams_alive=5, current_week=13)))
    generate(state, league, n_seasons=10, out_dir=tmp_path / "data", chunk=5)
    files = {p.name for p in build(tmp_path / "data", tmp_path / "site", sample_seasons=3, anonymize=True)}
    assert {"index.html", "cube.json", "players.json", "seasons.json", "summary.json", "paths.json"} <= files
    summary = json.loads((tmp_path / "site" / "summary.json").read_text())
    assert summary["you"] == "You" and all(t["name"].startswith(("Team ", "You")) for t in summary["teams"])


def test_free_replacement_never_beats_real_players_this_week(state, sim):
    # This week's free replacement is chosen by projection, so it can't exceed the best
    # player actually available.
    from guillotine.simulate import revealed_sd

    vb = sim.proj.value_basis[0]
    universe = set(range(len(sim.proj.ids)))
    repl = sim._replacement(vb, universe, 2, revealed_sd(0))
    for pos, v in repl.items():
        assert v <= max(vb[p] for p in universe if sim.proj.pos[p] == pos)


def test_roster_rules_by_slot(state, sim):
    from guillotine.simulate import BENCH_MAX, BENCH_MIN

    proj, vb = sim.proj, sim.proj.value_basis[0]
    tid = state.my_team_id
    roster = list(sim.init_rosters[tid])
    others = [i for i in range(len(proj.ids)) if i not in roster]
    # Pile on extra QBs and TEs, then enforce the rules.
    extra = [i for i in others if proj.pos[i] == "QB"][:3] + [i for i in others if proj.pos[i] == "TE"][:4]
    roster += extra
    rosters, fa = {tid: roster}, []
    sim._enforce_roster(0, 0, tid, extra[-1], rosters, fa, vb)
    _, starters = sim.spec.best(rosters[tid], vb, proj.pos)
    bench = [q for q in rosters[tid] if q not in starters]
    for pos, cap in BENCH_MAX.items():
        assert sum(proj.pos[q] == pos for q in bench) <= cap
    assert len(rosters[tid]) <= state.settings.roster_size
    for pos, need in BENCH_MIN.items():
        if any(proj.pos[q] == pos for q in sim.init_rosters[tid] if q not in starters):
            assert sum(proj.pos[q] == pos for q in bench) >= need
    assert set(fa) == set(roster) - set(rosters[tid]) or fa


def test_bench_upgrades_bid_zero_to_two(state, tmp_path):
    import csv
    import gzip
    import json

    from guillotine.dataset import generate

    league = tmp_path / "league.json"
    league.write_text(json.dumps(make_sample_league(seed=3, n_teams_alive=5, current_week=13)))
    generate(state, league, n_seasons=6, out_dir=tmp_path / "data", chunk=3)
    rows = list(csv.DictReader(gzip.open(tmp_path / "data" / "bids.csv.gz", "rt")))
    bench = [r for r in rows if r["kind"] == "bench"]
    assert all(int(r["fair_bid"]) == 1 and 0 <= int(r["bid"]) <= 2 for r in bench)
    assert {r["kind"] for r in rows} <= {"lineup", "deny", "bench"}


def test_claims_sharing_a_drop_process_at_most_once(state):
    sim = Simulator(state, seed=4)
    sim.trace = []
    sim.run(12)
    claims = {(e[1], e[2], e[3], e[4]): e[5] for e in sim.trace if e[0] == "claim"}
    used = {}
    for e in sim.trace:
        if e[0] != "award":
            continue
        _, s, j, pid, tid, amt, all_bids, faab = e
        assert (s, j, pid, tid) in claims  # every award came from a planned add/drop claim
        d = claims[(s, j, pid, tid)]
        if d is not None:  # a real player dropped (not an open spot)
            key = (s, j, tid, d)
            assert key not in used, "two claims dropping the same player both processed"
            used[key] = pid
        assert amt <= faab[tid]  # the winning claim was affordable when it processed
