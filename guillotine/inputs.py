"""The simulator's inputs: a league folder with three JSON files.

    rules.json        the league's rules, fixed for the season
    snapshot.json     rosters, FAAB and the waiver pool right before a waiver run
    projections.json  each player's projection for every remaining week

The full format is in docs/inputs.md. ``load_league`` checks all three files, reports
every problem at once (``InputError``), and turns them into the ``LeagueState`` the
simulator runs on. Nothing here fetches data: whoever writes the folder (a person or an
agent) is the data source.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .models import FLEX_ELIGIBILITY, LeagueState

RULES_FORMAT = "guillotine-rules/1"
SNAPSHOT_FORMAT = "guillotine-snapshot/1"
PROJECTIONS_FORMAT = "guillotine-projections/1"

POSITIONS = ("QB", "RB", "WR", "TE", "K", "DEF")
WAIVER_TYPES = ("faab",)  # "priority" (no money) is planned
PRIORITY_RULES = ("rolling",)  # "fixed", "reverse_score" are planned
OUT_STATUSES = ("OUT", "IR", "PUP", "SUSP", "NFI")  # statuses that mean "won't play until return_week"


class InputError(ValueError):
    """Every problem found in a league folder, one per line."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__(f"{len(problems)} problem(s) in the league inputs:\n" + "\n".join(f"  - {p}" for p in problems))


@dataclass
class Rules:
    name: str
    season: int
    last_week: int
    starters: dict[str, int]
    flex: dict[str, tuple[str, ...]]
    bench: int
    ir: int
    scoring: dict[str, float]
    scoring_by_position: dict[str, dict[str, float]]
    cuts_per_week: int
    cuts_by_week: dict[int, int]
    waiver_type: str
    budget: float
    min_bid: int
    priority: str
    notes: str = ""

    def cuts(self, week: int) -> int:
        return self.cuts_by_week.get(week, self.cuts_per_week)

    @property
    def roster_size(self) -> int:
        return sum(self.starters.values()) + self.bench


@dataclass
class League:
    rules: Rules
    state: LeagueState
    warnings: list[str] = field(default_factory=list)


# ------------------------------------------------------------------------ helpers

class _Check:
    """Collects problems instead of stopping at the first one."""

    def __init__(self):
        self.problems: list[str] = []
        self.warnings: list[str] = []

    def err(self, where: str, msg: str) -> None:
        self.problems.append(f"{where}: {msg}")

    def warn(self, where: str, msg: str) -> None:
        self.warnings.append(f"{where}: {msg}")

    def need(self, d: dict, key: str, kind, where: str, default=...):
        if not isinstance(d, dict) or key not in d:
            if default is ...:
                self.err(where, f"missing required field {key!r}")
            return None if default is ... else default
        v = d[key]
        kinds = kind if isinstance(kind, tuple) else (kind,)
        # JSON has no int/float split for whole numbers; bools are not numbers here.
        if float in kinds and isinstance(v, int) and not isinstance(v, bool):
            return float(v)
        if not isinstance(v, kinds) or (isinstance(v, bool) and bool not in kinds):
            self.err(where, f"{key!r} should be {' or '.join(k.__name__ for k in kinds)}, got {type(v).__name__} {v!r:.60}")
            return None if default is ... else default
        return v

    def unknown(self, d: dict, allowed: set[str], where: str) -> None:
        if isinstance(d, dict):
            for k in d:
                if k not in allowed:
                    self.err(where, f"unknown field {k!r} (allowed: {', '.join(sorted(allowed))})")


def _read(path: Path, check: _Check):
    if not path.exists():
        check.err(path.name, "file not found")
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        check.err(path.name, f"not valid JSON ({e})")
        return None


# ------------------------------------------------------------------------- rules

def _parse_rules(d: dict, c: _Check) -> Rules | None:
    W = "rules.json"
    if not isinstance(d, dict):
        c.err(W, "should be a JSON object")
        return None
    c.unknown(d, {"format", "name", "season", "last_week", "roster", "scoring", "eliminations", "waivers", "notes"}, W)
    if d.get("format") != RULES_FORMAT:
        c.err(W, f"'format' should be {RULES_FORMAT!r}, got {d.get('format')!r}")
    name = c.need(d, "name", str, W, "")
    season = c.need(d, "season", int, W)
    last_week = c.need(d, "last_week", int, W)

    r = c.need(d, "roster", dict, W) or {}
    c.unknown(r, {"starters", "flex", "bench", "ir"}, f"{W} roster")
    starters = c.need(r, "starters", dict, f"{W} roster") or {}
    flex_custom = c.need(r, "flex", dict, f"{W} roster", {})
    flex: dict[str, tuple[str, ...]] = {}
    for slot, n in starters.items():
        where = f"{W} roster.starters.{slot}"
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            c.err(where, f"count should be a whole number >= 0, got {n!r}")
        if slot in POSITIONS:
            continue
        elig = flex_custom.get(slot, FLEX_ELIGIBILITY.get(slot))
        if elig is None:
            c.err(where, f"{slot!r} is neither a position ({', '.join(POSITIONS)}) nor a known flex slot "
                         f"({', '.join(FLEX_ELIGIBILITY)}); define it in roster.flex, e.g. \"{slot}\": [\"RB\", \"WR\"]")
            continue
        bad = [p for p in elig if p not in POSITIONS]
        if bad or not elig:
            c.err(f"{W} roster.flex.{slot}", f"eligible positions should come from {', '.join(POSITIONS)}, got {elig!r}")
            continue
        flex[slot] = tuple(elig)
    for slot in flex_custom:
        if slot not in starters:
            c.err(f"{W} roster.flex.{slot}", "defined in roster.flex but not used in roster.starters")
    bench = c.need(r, "bench", int, f"{W} roster")
    ir = c.need(r, "ir", int, f"{W} roster", 0)
    if ir:
        c.err(f"{W} roster.ir", "IR slots are not supported yet (planned); use 0 and leave IR players off the roster")

    scoring, by_pos = {}, {}
    s = c.need(d, "scoring", dict, W, {})
    c.unknown(s, {"stats", "by_position", "description"}, f"{W} scoring")
    for k, v in (s.get("stats") or {}).items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            c.err(f"{W} scoring.stats.{k}", f"points per unit should be a number, got {v!r}")
        else:
            scoring[k] = float(v)
    for pos, m in (s.get("by_position") or {}).items():
        if pos not in POSITIONS or not isinstance(m, dict):
            c.err(f"{W} scoring.by_position.{pos}", f"should be a position ({', '.join(POSITIONS)}) mapping stat -> extra points")
            continue
        by_pos[pos] = {k: float(v) for k, v in m.items() if isinstance(v, (int, float))}

    e = c.need(d, "eliminations", dict, W) or {}
    c.unknown(e, {"per_week", "by_week"}, f"{W} eliminations")
    per_week = c.need(e, "per_week", int, f"{W} eliminations")
    by_week = {}
    for k, v in (e.get("by_week") or {}).items():
        if not str(k).isdigit() or not isinstance(v, int) or v < 0:
            c.err(f"{W} eliminations.by_week", f"should map week number -> teams cut, got {k!r}: {v!r}")
        else:
            by_week[int(k)] = v

    wv = c.need(d, "waivers", dict, W) or {}
    c.unknown(wv, {"type", "budget", "min_bid", "priority"}, f"{W} waivers")
    wtype = c.need(wv, "type", str, f"{W} waivers")
    if wtype is not None and wtype not in WAIVER_TYPES:
        c.err(f"{W} waivers.type", f"{wtype!r} is not supported yet; supported: {', '.join(WAIVER_TYPES)}")
    budget = c.need(wv, "budget", float, f"{W} waivers")
    min_bid = c.need(wv, "min_bid", int, f"{W} waivers", 0)
    priority = c.need(wv, "priority", str, f"{W} waivers", "rolling")
    if priority not in PRIORITY_RULES:
        c.err(f"{W} waivers.priority", f"{priority!r} is not supported yet; supported: {', '.join(PRIORITY_RULES)}")

    if None in (season, last_week, bench, per_week, budget):
        return None
    return Rules(name=name, season=season, last_week=last_week, starters=dict(starters), flex=flex,
                 bench=bench, ir=ir or 0, scoring=scoring, scoring_by_position=by_pos,
                 cuts_per_week=per_week, cuts_by_week=by_week, waiver_type=wtype or "faab",
                 budget=budget, min_bid=min_bid or 0, priority=priority, notes=d.get("notes", ""))


# --------------------------------------------------------------------- snapshot

def _points(stats: dict, pos: str, rules: Rules, where: str, c: _Check, unscored: set) -> float:
    total = 0.0
    extra = rules.scoring_by_position.get(pos, {})
    for k, v in stats.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            c.err(where, f"stat {k!r} should be a number, got {v!r}")
            continue
        if k not in rules.scoring and k not in extra:
            unscored.add(k)
            continue
        total += v * (rules.scoring.get(k, 0.0) + extra.get(k, 0.0))
    return round(total, 3)


def _build(rules: Rules, snap: dict, proj: dict, c: _Check) -> LeagueState | None:
    S, P = "snapshot.json", "projections.json"
    if not isinstance(snap, dict) or not isinstance(proj, dict):
        c.err(S if not isinstance(snap, dict) else P, "should be a JSON object")
        return None
    c.unknown(snap, {"format", "week", "focus_team", "teams", "eliminated", "waiver_pool", "free_agents", "players", "notes"}, S)
    c.unknown(proj, {"format", "source", "as_of", "players", "notes"}, P)
    if snap.get("format") != SNAPSHOT_FORMAT:
        c.err(S, f"'format' should be {SNAPSHOT_FORMAT!r}, got {snap.get('format')!r}")
    if proj.get("format") != PROJECTIONS_FORMAT:
        c.err(P, f"'format' should be {PROJECTIONS_FORMAT!r}, got {proj.get('format')!r}")
    week = c.need(snap, "week", int, S)
    if week is None:
        return None
    if week > rules.last_week:
        c.err(f"{S} week", f"week {week} is after the last week ({rules.last_week})")
        return None
    weeks = list(range(week, rules.last_week + 1))

    # players
    players, pos_of = {}, {}
    for i, p in enumerate(c.need(snap, "players", list, S) or []):
        where = f"{S} players[{i}]"
        c.unknown(p, {"id", "name", "pos", "nfl_team", "bye", "injury"}, where)
        pid = c.need(p, "id", str, where)
        if pid is None:
            continue
        where = f"{S} player {pid}"
        if pid in players:
            c.err(where, "defined twice")
        pos = c.need(p, "pos", str, where, "")
        if pos not in POSITIONS:
            c.err(where, f"pos {pos!r} should be one of {', '.join(POSITIONS)}")
        bye = c.need(p, "bye", (int, type(None)), where, None)
        inj = c.need(p, "injury", (dict, type(None)), where, None)
        out_until = None
        if inj:
            c.unknown(inj, {"status", "return_week", "note"}, f"{where} injury")
            status = str(inj.get("status", "")).upper()
            ret = inj.get("return_week")
            if ret is not None and (not isinstance(ret, int) or isinstance(ret, bool)):
                c.err(f"{where} injury.return_week", f"should be a week number or null, got {ret!r}")
                ret = None
            if status in OUT_STATUSES:
                out_until = ret if ret is not None else rules.last_week + 1  # null = out for the season
        players[pid] = {"id": pid, "name": c.need(p, "name", str, where, pid), "pos": pos,
                        "nfl_team": p.get("nfl_team", ""), "bye": bye, "_out_until": out_until}
        pos_of[pid] = pos

    # projections
    unscored: set[str] = set()
    pp = c.need(proj, "players", dict, P) or {}
    for pid, by_week in pp.items():
        where = f"{P} player {pid}"
        if pid not in players:
            c.err(where, "has projections but is not in snapshot.players")
            continue
        if not isinstance(by_week, dict):
            c.err(where, "should map week -> points or stat line")
            continue
        weekly = {}
        for wk, v in by_week.items():
            if not str(wk).isdigit():
                c.err(where, f"week key {wk!r} should be a week number")
                continue
            w = int(wk)
            if w not in weeks:
                continue  # past weeks or beyond the season: ignored
            if isinstance(v, dict):
                if not rules.scoring:
                    c.err(f"{where} week {w}", "is a stat line but rules.scoring.stats is empty; give points or add scoring")
                    continue
                weekly[w] = _points(v, pos_of[pid], rules, f"{where} week {w}", c, unscored)
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                weekly[w] = float(v)
            else:
                c.err(f"{where} week {w}", f"should be points (number) or a stat line (object), got {v!r}")
        players[pid]["weekly"] = weekly
    if unscored:
        c.warn(P, f"stats with no points in rules.scoring were ignored: {', '.join(sorted(unscored))}")

    # teams
    teams, seen = [], {}
    for i, t in enumerate(c.need(snap, "teams", list, S) or []):
        where = f"{S} teams[{i}]"
        c.unknown(t, {"id", "name", "faab", "priority", "players"}, where)
        tid = c.need(t, "id", str, where)
        if tid is None:
            continue
        where = f"{S} team {tid}"
        if any(x["id"] == tid for x in teams):
            c.err(where, "defined twice")
        faab = c.need(t, "faab", float, where, 0.0)
        if faab < 0 or faab > rules.budget:
            c.err(where, f"faab {faab} should be between 0 and the budget ({rules.budget})")
        roster = c.need(t, "players", list, where, [])
        for pid in roster:
            if pid not in players:
                c.err(where, f"player {pid!r} is not in snapshot.players")
            elif pid in seen:
                c.err(where, f"player {pid!r} is also on {seen[pid]}")
            seen[pid] = f"team {tid}"
        if len(roster) > rules.roster_size:
            c.err(where, f"{len(roster)} players, more than the roster size ({rules.roster_size} = starters + bench)")
        if "priority" in t:
            c.warn(where, "waiver priority is not used yet (each round's order is drawn at random)")
        teams.append({"id": tid, "name": c.need(t, "name", str, where, tid), "faab": faab, "players": list(roster)})
    if len(teams) < 2:
        c.err(S, "needs at least 2 teams still alive")

    pool = c.need(snap, "waiver_pool", list, S, [])
    fas = c.need(snap, "free_agents", list, S, [])
    for name, lst in (("waiver_pool", pool), ("free_agents", fas)):
        for pid in lst:
            if pid not in players:
                c.err(f"{S} {name}", f"player {pid!r} is not in snapshot.players")
            elif pid in seen and seen[pid] != "waiver_pool":
                c.err(f"{S} {name}", f"player {pid!r} is also on {seen[pid]}")
            seen.setdefault(pid, name)

    focus = c.need(snap, "focus_team", str, S)
    if focus is not None and focus not in {t["id"] for t in teams}:
        c.err(f"{S} focus_team", f"{focus!r} is not one of the teams still alive")

    # Every player who can matter needs a projection for every remaining week he plays.
    for pid, p in players.items():
        if pid not in seen:
            continue
        missing = [w for w in weeks if w != p["bye"] and w not in p.get("weekly", {})
                   and not (p["_out_until"] and w < p["_out_until"])]
        if missing:
            msg = f"no projection for week(s) {', '.join(map(str, missing))} ({p['name']})"
            if seen[pid] == "free_agents":
                c.warn(f"{P} player {pid}", msg + "; counted as 0")
            else:
                c.err(f"{P} player {pid}", msg + " (rostered and waiver players need every week they play)")

    # Cuts: the engine plays a two-team final in the last week, one cut a week before it.
    alive = len(teams)
    for w in weeks[:-1]:
        alive -= rules.cuts(w)
    if rules.cuts_by_week:
        c.err("rules.json eliminations.by_week", "week-by-week cut counts are not supported yet (planned)")
    if alive != 2:
        c.err("rules.json eliminations", f"{len(teams)} teams alive in week {week} and {rules.cuts_per_week} cut a week "
                                         f"leaves {alive} for the last week; only a 2-team final is supported yet")

    if c.problems:
        return None

    out_players = []
    for pid, p in players.items():
        weekly = dict(p.get("weekly", {}))
        if p["_out_until"]:
            for w in weeks:
                if w < p["_out_until"]:
                    weekly[w] = 0.0
        if p["bye"] in weekly:
            weekly[p["bye"]] = 0.0
        for w in weeks:  # free agents with gaps: 0, already warned
            if w != p["bye"]:
                weekly.setdefault(w, 0.0)
        out_players.append({"id": pid, "name": p["name"], "pos": p["pos"], "nfl_team": p["nfl_team"],
                            "bye": p["bye"], "ros_points": round(sum(weekly.values()), 1),
                            "weekly": {str(w): v for w, v in sorted(weekly.items())}})
    slots = dict(rules.starters)
    legacy = {
        "settings": {"season": rules.season, "current_week": week, "last_week": rules.last_week,
                     "roster_slots": slots, "roster_size": rules.roster_size,
                     "eliminations_per_week": rules.cuts_per_week, "min_bid": rules.min_bid},
        "my_team_id": focus, "teams": teams, "waiver_pool": pool, "free_agents": fas, "players": out_players,
    }
    for slot, elig in rules.flex.items():
        FLEX_ELIGIBILITY.setdefault(slot, elig)  # custom flex names (same names keep built-in meaning)
        if tuple(FLEX_ELIGIBILITY[slot]) != tuple(elig):
            c.err(f"rules.json roster.flex.{slot}", f"{slot!r} is a built-in flex name meaning {FLEX_ELIGIBILITY[slot]}; "
                                                    "use another name for a different set")
            return None
    try:
        return LeagueState.from_dict(legacy)
    except ValueError as e:  # a gap in the checks above
        c.err("league", str(e))
        return None


def load_league(folder: str | Path) -> League:
    """Read, check and convert a league folder. Raises InputError listing every problem."""
    folder = Path(folder)
    c = _Check()
    rules_d = _read(folder / "rules.json", c)
    snap = _read(folder / "snapshot.json", c)
    proj = _read(folder / "projections.json", c)
    rules = _parse_rules(rules_d, c) if rules_d is not None else None
    state = _build(rules, snap, proj, c) if rules and snap is not None and proj is not None and not c.problems else None
    if c.problems or state is None:
        raise InputError(c.problems or ["could not build the league"])
    return League(rules=rules, state=state, warnings=c.warnings)


def load_state(path: str | Path) -> LeagueState:
    """A league folder (the input format) or a single legacy state file."""
    path = Path(path)
    if path.is_dir():
        return load_league(path).state
    return LeagueState.load(path)


# ---------------------------------------------------------------- legacy → folder

def convert_legacy(state_path: str | Path, out: str | Path, name: str = "", bench: int | None = None) -> Path:
    """Write a league folder from a single legacy state file (points-based projections)."""
    d = json.loads(Path(state_path).read_text())
    s = d["settings"]
    starters = {k: v for k, v in s["roster_slots"].items() if k not in ("BN", "IR", "BENCH")}
    n_start = sum(starters.values())
    rules = {
        "format": RULES_FORMAT, "name": name or f"League {s['season']}", "season": s["season"],
        "last_week": s["last_week"],
        "roster": {"starters": starters, "bench": bench if bench is not None else s.get("roster_size", 16) - n_start},
        "eliminations": {"per_week": s.get("eliminations_per_week", 1)},
        "waivers": {"type": "faab", "budget": max([t["faab"] for t in d["teams"]] + [100]),
                    "min_bid": s.get("min_bid", 0), "priority": "rolling"},
    }
    weeks = range(s["current_week"], s["last_week"] + 1)
    players, proj = [], {}
    for p in d["players"]:
        players.append({k: p[k] for k in ("id", "name", "pos", "nfl_team", "bye") if k in p})
        games = [w for w in weeks if w != p.get("bye")]
        per_game = p.get("ros_points", 0.0) / len(games) if games else 0.0
        wk = {int(k): v for k, v in p.get("weekly", {}).items()}
        proj[p["id"]] = {str(w): wk.get(w, round(per_game, 3)) for w in games}
    snap = {"format": SNAPSHOT_FORMAT, "week": s["current_week"], "focus_team": d["my_team_id"],
            "teams": [{"id": t["id"], "name": t["name"], "faab": t["faab"], "players": t["players"]} for t in d["teams"]],
            "waiver_pool": d.get("waiver_pool", []), "free_agents": d.get("free_agents", []), "players": players}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rules.json").write_text(json.dumps(rules, indent=1))
    (out / "snapshot.json").write_text(json.dumps(snap, indent=1))
    (out / "projections.json").write_text(json.dumps({"format": PROJECTIONS_FORMAT, "source": f"converted from {Path(state_path).name}",
                                                      "players": proj}, indent=1))
    return out
