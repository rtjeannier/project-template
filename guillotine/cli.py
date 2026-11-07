"""Command line: ``python -m guillotine <command>``."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .analysis import analyze
from .inputs import InputError, load_league, load_state, convert_legacy
from .models import LeagueState
from .report import render
from .sample import make_sample_league


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="guillotine", description="Guillotine league FAAB simulator")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="simulate the rest of the season and print a report")
    a.add_argument("league", help="league folder (rules.json, snapshot.json, projections.json) or a legacy state file")
    a.add_argument("--sims", type=int, default=400, help="simulated seasons per scenario")
    a.add_argument("--top", type=int, default=6, help="waiver candidates to evaluate")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="parallel processes for scenarios")
    a.add_argument("--future", type=int, default=5, help="irreplaceable players on other rosters to evaluate as targets")
    a.add_argument("--out", help="also write the report to this file")

    s = sub.add_parser("sample", help="write a synthetic league state file")
    s.add_argument("--out", default="-", help="output path (default stdout)")
    s.add_argument("--seed", type=int, default=7)

    f = sub.add_parser("fetch-fantasypros", help="save raw FantasyPros projections (needs API key)")
    f.add_argument("--season", type=int, required=True)
    f.add_argument("--week", type=int, required=True)
    f.add_argument("--scoring", default="PPR")
    f.add_argument("--out", default="data/raw/fantasypros")

    i = sub.add_parser("inspect", help="export every simulated scenario for one target player")
    i.add_argument("league", help="league folder (rules.json, snapshot.json, projections.json) or a legacy state file")
    i.add_argument("--player", required=True, help="player name, e.g. 'Jahmyr Gibbs'")
    i.add_argument("--sims", type=int, default=300)
    i.add_argument("--timelines", type=int, default=40, help="seasons per bid level with full week-by-week detail")
    i.add_argument("--seed", type=int, default=0)
    i.add_argument("--out", default="inspect_out")

    v = sub.add_parser("viewer", help="build the Scenario Inspector page from inspect.json exports")
    v.add_argument("files", nargs="+", help="inspect.json files from `inspect`")
    v.add_argument("--out", default="inspector.html")

    d = sub.add_parser("simulate-data", help="play many seasons and write every bid, award, drop, score and cut")
    d.add_argument("league", help="league folder (rules.json, snapshot.json, projections.json) or a legacy state file")
    d.add_argument("--seasons", type=int, default=5000)
    d.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--out", default="sim_data")

    w = sub.add_parser("build-site", help="build the static results explorer (GitHub Pages) from simulate-data output")
    w.add_argument("data", help="directory written by simulate-data")
    w.add_argument("--out", default="docs/explorer")
    w.add_argument("--sample-seasons", type=int, default=200, help="full seasons to include for reading one by one")
    w.add_argument("--anonymize", action="store_true", help="replace team names with Team 1..N (you stay 'You')")

    c = sub.add_parser("validate", help="check a league folder and list every problem")
    c.add_argument("league", help="league folder")

    r = sub.add_parser("run", help="validate a league folder, simulate it and build the report")
    r.add_argument("league", help="league folder")
    r.add_argument("--seasons", type=int, default=5000)
    r.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--data", help="where to write the simulation data (default <league>/sim_data)")
    r.add_argument("--site", help="where to write the report (default <league>/report)")

    k = sub.add_parser("convert-legacy", help="turn a single-file league state into a league folder")
    k.add_argument("state", help="legacy state JSON file")
    k.add_argument("--out", required=True, help="league folder to write")
    k.add_argument("--name", default="")

    args = ap.parse_args(argv)

    if args.cmd in ("validate", "run"):
        try:
            league = load_league(args.league)
        except InputError as e:
            print(e, file=sys.stderr)
            return 2
        for w in league.warnings:
            print(f"warning: {w}", file=sys.stderr)
        st = league.state
        print(f"ok: {league.rules.name or 'league'}, week {st.settings.current_week}-{st.settings.last_week}, "
              f"{len(st.teams)} teams alive, {len(st.waiver_pool)} on waivers, {len(st.players)} players", file=sys.stderr)
        if args.cmd == "validate":
            return 0
        from .dataset import generate
        from .site import build as build_site

        data = Path(args.data or Path(args.league) / "sim_data")
        site = Path(args.site or Path(args.league) / "report")
        t0 = time.time()
        generate(st, args.league, args.seasons, data, args.seed, args.workers)
        print(f"{args.seasons} seasons in {time.time() - t0:.0f}s", file=sys.stderr)
        for path in build_site(data, site):
            print(path)
        return 0

    if args.cmd == "convert-legacy":
        print(convert_legacy(args.state, args.out, args.name))
        return 0

    if args.cmd == "build-site":
        from .site import build as build_site

        for path in build_site(args.data, args.out, args.sample_seasons, args.anonymize):
            print(path)
        return 0

    if args.cmd == "simulate-data":
        from .dataset import generate

        t0 = time.time()
        for path in generate(load_state(args.league), args.league, args.seasons, args.out,
                             args.seed, args.workers):
            print(path)
        print(f"{args.seasons} seasons in {time.time() - t0:.0f}s", file=sys.stderr)
        return 0

    if args.cmd == "viewer":
        from .viewer import build

        print(build(args.files, args.out))
        return 0

    if args.cmd == "inspect":
        from .inspect import inspect_player, write

        data = inspect_player(load_state(args.league), args.player, args.sims, args.timelines, args.seed)
        for path in write(data, args.out):
            print(path)
        for row in data["summary"]:
            print(f"{row['bid_level']:>12}  P(win league) {100 * row['p_win_league']:5.1f}%  "
                  f"could bid {100 * row['p_could_bid']:4.0f}%  got him {100 * row['p_get_when_could']:4.0f}%")
        return 0

    if args.cmd == "fetch-fantasypros":
        from .providers.fantasypros import fetch_raw

        for path in fetch_raw(args.season, args.week, args.out, args.scoring):
            print(path)
        return 0

    if args.cmd == "sample":
        text = json.dumps(make_sample_league(seed=args.seed), indent=1)
        if args.out == "-":
            print(text)
        else:
            Path(args.out).write_text(text)
        return 0

    state = load_state(args.league)
    t0 = time.time()
    result = analyze(state, n_sims=args.sims, top_k=args.top, seed=args.seed, n_future=args.future, workers=args.workers)
    report = render(result)
    print(report)
    print(f"\n_{time.time() - t0:.1f}s_", file=sys.stderr)
    if args.out:
        Path(args.out).write_text(report)
    return 0
