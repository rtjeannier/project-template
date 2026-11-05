"""Command line: ``python -m guillotine <command>``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .analysis import analyze
from .models import LeagueState
from .report import render
from .sample import make_sample_league


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="guillotine", description="Guillotine league FAAB simulator")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="simulate the rest of the season and print a report")
    a.add_argument("league", help="league state JSON file")
    a.add_argument("--sims", type=int, default=400, help="simulated seasons per scenario")
    a.add_argument("--top", type=int, default=6, help="waiver candidates to evaluate")
    a.add_argument("--seed", type=int, default=0)
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

    args = ap.parse_args(argv)

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

    state = LeagueState.load(args.league)
    t0 = time.time()
    result = analyze(state, n_sims=args.sims, top_k=args.top, seed=args.seed, n_future=args.future)
    report = render(result)
    print(report)
    print(f"\n_{time.time() - t0:.1f}s_", file=sys.stderr)
    if args.out:
        Path(args.out).write_text(report)
    return 0
