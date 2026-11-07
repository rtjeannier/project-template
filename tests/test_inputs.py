import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from guillotine.inputs import InputError, convert_legacy, load_league
from guillotine.models import LeagueState
from guillotine.projections import Projections

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "league"


def _copy(tmp_path) -> Path:
    d = tmp_path / "lg"
    shutil.copytree(EXAMPLE, d)
    return d


def _edit(d: Path, name: str, fn) -> None:
    f = d / name
    data = json.loads(f.read_text())
    fn(data)
    f.write_text(json.dumps(data))


def test_example_league_loads():
    lg = load_league(EXAMPLE)
    st = lg.state
    assert st.settings.current_week == 10 and st.settings.last_week == 17
    assert len(st.teams) == 9 and st.my_team_id in {t.id for t in st.teams}
    assert lg.rules.roster_size == st.settings.roster_size


def test_stat_lines_use_league_scoring_and_position_bonus(tmp_path):
    d = _copy(tmp_path)
    _edit(d, "rules.json", lambda r: r.update(scoring={
        "stats": {"rec": 0.5, "rec_yds": 0.1, "rec_td": 6}, "by_position": {"TE": {"rec": 0.5}}}))
    snap = json.loads((d / "snapshot.json").read_text())
    te = next(p for p in snap["players"] if p["pos"] == "TE" and p["id"] in snap["teams"][0]["players"])
    wk = next(str(w) for w in range(10, 18) if w != te.get("bye"))
    line = {"rec": 6, "rec_yds": 70, "rec_td": 0.5, "targets": 8}
    _edit(d, "projections.json", lambda p: p["players"][te["id"]].update({wk: line}))
    lg = load_league(d)
    assert lg.state.players[te["id"]].weekly[int(wk)] == pytest.approx(6 * 1.0 + 7.0 + 3.0)
    assert any("targets" in w for w in lg.warnings)  # unscored stat reported, not fatal


def test_injury_and_bye_weeks_score_zero(tmp_path):
    d = _copy(tmp_path)
    snap = json.loads((d / "snapshot.json").read_text())
    pid = snap["teams"][0]["players"][0]
    for p in snap["players"]:
        if p["id"] == pid:
            p["injury"] = {"status": "OUT", "return_week": 12}
            bye = p.get("bye")
    (d / "snapshot.json").write_text(json.dumps(snap))
    w = load_league(d).state.players[pid].weekly
    assert w[10] == 0 and w[11] == 0 and (bye is None or bye < 10 or w[bye] == 0)
    assert any(w[k] > 0 for k in range(12, 18) if k != bye)


def test_every_problem_is_reported_at_once(tmp_path):
    d = _copy(tmp_path)
    snap = json.loads((d / "snapshot.json").read_text())
    rostered = snap["teams"][1]["players"][0]
    snap["teams"][0]["players"].append("nobody")          # undefined player
    snap["teams"][2]["players"].append(rostered)           # on two rosters
    snap["teams"][3]["colour"] = "red"                     # unknown field
    (d / "snapshot.json").write_text(json.dumps(snap))
    _edit(d, "projections.json", lambda p: p["players"][rostered].pop("15", None) or p["players"][rostered].pop("16"))
    _edit(d, "rules.json", lambda r: r["eliminations"].update(per_week=2))
    with pytest.raises(InputError) as e:
        load_league(d)
    text = "\n".join(e.value.problems)
    for needle in ("'nobody'", "is also on", "'colour'", "no projection for week", "2-team final"):
        assert needle in text, needle


def test_legacy_conversion_matches_legacy_projections(tmp_path):
    legacy = EXAMPLE.parent / "sample_league.json"
    out = convert_legacy(legacy, tmp_path / "conv")
    a = Projections(LeagueState.load(legacy))
    b = Projections(load_league(out).state)
    assert a.ids == b.ids
    np.testing.assert_allclose(a.means, b.means, atol=1e-3)
