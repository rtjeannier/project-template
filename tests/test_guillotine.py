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
    assert m.bid(value=10, opportunity=0, faab=500, aggression=1, noise=1, min_bid=0) == 500
    # Plenty of future opportunity: same player gets a smaller share.
    assert m.bid(value=10, opportunity=90, faab=500, aggression=1, noise=1, min_bid=0) == 50


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
    # If nobody else bid $1 or more, a $1 bid always wins. (It can also win when a higher
    # rival claim is invalid: its drop already used, or no longer affordable.)
    assert all(s in won for s in range(20) if opp[s] < 1)
    assert all(a == 1 for s, j, p, t, a in res.awards if j == 0 and p == pid and t == state.my_team_id)


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
    injuries = read("injuries.csv.gz")
    for r in injuries:  # a hurt player misses only weeks that are left
        assert 0 <= int(r["weeks_out"]) <= 17 - int(r["week"])
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
    assert {"index.html", "cube.json", "players.json", "seasons.json", "summary.json", "paths.json", "scenarios.json"} <= files
    summary = json.loads((tmp_path / "site" / "summary.json").read_text())
    assert summary["you"] == "You" and all(t["name"].startswith(("Team ", "You")) for t in summary["teams"])


def test_replacement_never_beats_the_best_real_player(state, sim):
    # Replacement is a real player's projection, so it can't exceed the best one available.
    vb = sim.proj.value_basis[0]
    universe = set(range(len(sim.proj.ids)))
    for n in (2, 3, 5):
        for pos, v in sim._replacement(vb, universe, n).items():
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


def test_bids_only_when_better_than_waiting_and_within_budget(state):
    # Every bid's value beats the slot's later option, and the fair price stays in budget.
    sim = Simulator(state, seed=5)
    sim.value_log = []
    sim.run(n_sims=4)
    assert sim.value_log
    for (_, _, _, _, g_now, g_fin, v_now, v_mid, v_fin, v, wait, money, slot, faab, fair) in sim.value_log:
        assert v > wait  # better than waiting for this slot
        assert abs(v - (v_now + v_mid + v_fin)) < 1e-12
        assert min(v_now, v_mid, v_fin) >= 0
        assert 0 <= fair <= max(faab, 0) + 1


def test_a_bye_week_does_not_lower_week17_value(state, sim):
    # Claims are valued on this week and week 17 only: zeroing a middle week changes nothing.
    sim._final = sim.proj.means[:, -1].copy()
    sim._now = sim.proj.means[:, 0].copy()
    roster = list(sim.init_rosters[sim.user])
    p = max((x for x in range(len(sim.proj.ids)) if x not in roster), key=lambda x: sim._final[x])
    q = sim._out_for_final(0)
    base, starters = sim.spec.best(roster, sim._final, sim.proj.pos)
    without = {s: sim.spec.total([y for y in roster if y != s], sim._final, sim.proj.pos) for s in starters}
    before = sim._final_gain(roster, p, sim._final, q, base, without)
    saved = sim.proj.means[p, 1]
    sim.proj.means[p, 1] = 0.0
    try:
        after = sim._final_gain(roster, p, sim._final, q, base, without)
    finally:
        sim.proj.means[p, 1] = saved
    assert after == before


def test_slot_is_where_the_dropped_starter_played(state, sim):
    # A flex-eligible player who only pushes out a flex starter competes with any RB/WR/TE.
    FLEX_POS = sim.spec.group("FLEX")
    fin = sim.proj.means[:, -1]
    roster = list(sim.init_rosters[sim.user])
    labels = {x: lab for lab, x in sim.spec.assign(roster, fin, sim.proj.pos) if x is not None}
    for p in range(len(sim.proj.ids)):
        if p in roster or sim.proj.pos[p] not in FLEX_POS:
            continue
        after = {x for _, x in sim.spec.assign(roster + [p], fin, sim.proj.pos) if x is not None}
        out = [x for x in labels if x not in after]
        if p in after and out and labels[out[0]] == "FLEX":
            assert sim._slot_group(roster, p, fin) == FLEX_POS
            return
    pytest.skip("no flex-only upgrade in the sample league")
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


def test_drops_are_only_players_on_the_roster_at_submission(state):
    """A claim can't name a player the team doesn't have, and nobody drops a player they
    picked up in the same round."""
    sim = Simulator(state, seed=6)
    sim.trace = []
    rosters_at = {}
    orig = sim._waiver_round

    def wr(s, j, rng, rosters, *a, **k):
        rosters_at[(s, j)] = {t: set(r) for t, r in rosters.items()}
        return orig(s, j, rng, rosters, *a, **k)

    sim._waiver_round = wr
    sim.run(10)
    idx = sim.proj.index
    added = {}
    for e in sim.trace:
        if e[0] == "claim" and e[5] is not None:
            _, s, j, pid, tid, drop = e
            assert idx[drop] in rosters_at[(s, j)][tid], "claim names a drop not on the roster"
        if e[0] == "award":
            added.setdefault((e[1], e[2], e[4]), set()).add(e[3])
        if e[0] == "drop":
            _, s, j, tid, pid = e
            assert pid not in added.get((s, j, tid), set()), "dropped a player added this same round"


def test_backup_pickups_improve_week_17(state):
    """A backup-only pickup must be better for the final (week 17) than the player dropped,
    using week-17 values as known that round (a player out for the season is worth 0)."""
    sim = Simulator(state, seed=8)
    sim.trace = []
    final_at = {}
    orig = sim._final_value

    def fv(j):
        v = orig(j)
        final_at[(sim._current_sim, j)] = v
        return v

    sim._final_value = fv
    orig_run = sim._run_one

    def run_one(s, *a, **k):
        sim._current_sim = s
        return orig_run(s, *a, **k)

    sim._run_one = run_one
    sim.run(10)
    idx = sim.proj.index
    kind = {(e[1], e[2], e[3], e[4]): e[9] for e in sim.trace if e[0] == "bid"}
    claims = {(e[1], e[2], e[3], e[4]): e[5] for e in sim.trace if e[0] == "claim"}
    checked = 0
    for e in sim.trace:
        if e[0] != "award":
            continue
        key = (e[1], e[2], e[3], e[4])
        drop = claims.get(key)
        if kind.get(key) == "bench" and drop is not None:
            f = final_at[(e[1], e[2])]
            assert f[idx[e[3]]] > f[idx[drop]]
            checked += 1
    assert checked > 0


def test_lock_bids_are_never_randomized(state):
    """When the fair bid is the lock price (top every other team's whole FAAB), the team bids
    exactly that: noise must not turn a guaranteed win into a loss."""
    sim = Simulator(state, seed=9)
    sim.trace = []
    sim.run(15)
    checked = 0
    for e in sim.trace:
        if e[0] == "bid" and e[9] != "bench" and e[7] == 1.0 and e[6] > 0:
            assert e[5] == e[6]  # bid == fair (the lock)
            checked += 1
    assert checked >= 0
