"""Markdown report for an Analysis."""

from __future__ import annotations

from .analysis import WIN_TARGETS, Analysis, survive_through


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _signed_pct(x: float) -> str:
    return f"{100 * x:+.1f} pts"


def _rule(r) -> str:
    if r == "none":
        return "don't bid"
    if r == "$1":
        return "$1 lottery ticket"
    if r == "lock":
        return "lock: top richest rival's budget"
    if r == "all-in":
        return "all-in"
    if isinstance(r, str) and r.startswith("x"):
        m = float(r[1:])
        return "fair bid" if m == 1 else f"{r[1:]}× fair ({'over' if m > 1 else 'under'}bid)"
    if isinstance(r, int):
        return f"${r}"
    return f"beat the field {100 * float(r):.0f}%"


def _level(r) -> str:
    if isinstance(r, str) and r.startswith("x"):
        return f"{r[1:]}× fair"
    if isinstance(r, int):
        return f"${r}"
    return r if isinstance(r, str) else f"beat {100 * float(r):.0f}%"


def render(a: Analysis) -> str:
    st = a.state
    me = st.my_team_id
    my = st.team(me)
    players = st.players
    res = a.baseline
    lines = []
    add = lines.append

    add(f"# Guillotine plan: week {st.settings.current_week}, {st.settings.season}")
    add("")
    add(f"{len(st.teams)} teams alive, {len(res.weeks)} weeks left, {res.n_sims} simulated seasons per scenario.")
    add("")
    add(f"## {my.name}: ${my.faab:.0f} FAAB")
    add("")
    if a.plan_result is not None:
        add(f"- Chance to win the league: **{_pct(a.plan_result.p_champion(me))}** following the plan below, "
            f"{_pct(res.p_champion(me))} bidding like a typical manager")
    alive = (a.plan_result or res).p_alive_after(me)
    add("- Chance you're still alive after each week: " + ", ".join(
        f"wk{w} {_pct(p)}" for w, p in zip(res.weeks, alive)))
    p = res.p_champion(me)
    noise = 100 * (2 * p * (1 - p) / res.n_sims) ** 0.5
    add(f"- Differences under about ±{noise:.1f} pts are simulation noise at {res.n_sims} sims.")
    add("")

    add("## Who to target, and how much to bid")
    add("")
    add("**Fair bid** = the rational bid for you whenever he's on waivers: his elimination-weighted points above "
        "replacement plus denial, against the upgrades you could buy later instead (same valuation rivals use, no "
        "noise). The simulation then bids 0.5×–3× that, none, all-in, and *lock* (top the richest rival's whole "
        "budget), and keeps the bid with the best P(win league): if it's above 1× fair, overbidding pays for him. "
        "Rivals bid rationally with random noise, and lock when guaranteeing the win is worth it to them. "
        "Candidates are this week's best pool players plus the players on other rosters who are irreplaceable in "
        "the final (projected above week-17 replacement) and likely to hit waivers while you can bid. "
        "**Final edge** = his projected points/week above week-17 replacement (allowing for how uncertain week 17 "
        "is from here): in the final he counts for you *and* against your opponent. Effects are vs never bidding on him.")
    add("")
    add("| Player | Pos | Now on | Final edge | Fair bid | Best bid | Median bid placed | Δ P(win league) | P(you get him, when you can bid) | Typical price paid | Usually available |")
    add("|---|---|---|---:|---:|---|---:|---:|---:|---:|---|")
    for t in a.targets:
        pl = players[t["player_id"]]
        b = t["best"]
        wk = f"wk {sorted(b['weeks'])[len(b['weeks']) // 2]}" if b["weeks"] else "-"
        bid = f"${b['bid_median']:.0f}" if b["rule"] != "none" else "-"
        fair = f"${t['fair']:.0f}" if t.get("fair") is not None else "-"
        add(f"| {pl.name} | {pl.pos} | {t['owner']} | {t['edge']:+.1f} | {fair} | {_rule(b['rule'])} | {bid} | "
            f"{_signed_pct(t['gain_vs_pass'])} | {_pct(b['p_get'])} | "
            f"{'$' + format(b['avg_paid'], '.0f') if b['avg_paid'] else '-'} | {wk} "
            f"({_pct(b['p_chance'])} of seasons) |")
    add("")
    add("### Every bid tried")
    add("")
    add("P(win league) for each bid; in brackets, how often you get him when you can bid, then your FAAB left vs "
        "the richest rival's right after you win him (medians).")
    add("")
    for t in a.targets:
        add(f"**{players[t['player_id']].name}**")
        add("")
        add("| Bid | Median $ | P(win league) | Get him | Your FAAB left | Richest rival |")
        add("|---|---:|---:|---:|---:|---:|")
        for c in t["curve"]:
            left = f"${c['left']:.0f}" if c.get("left") is not None else "-"
            rich = f"${c['richest']:.0f}" if c.get("richest") is not None else "-"
            dollars = f"${c['bid_median']:.0f}" if c["rule"] != "none" else "-"
            mark = " ◀ best" if c["rule"] == t["best"]["rule"] else ""
            add(f"| {_level(c['rule'])}{mark} | {dollars} | {_pct(c['p_champ'])} | {_pct(c['p_get'])} | {left} | {rich} |")
        add("")

    add("## This week's pool: what it takes to win each player")
    add("")
    add("**Bid for X%** = the bid that beats every other manager in X% of simulations.")
    add("")
    hdr = " | ".join(f"Bid for {int(t * 100)}%" for t in WIN_TARGETS)
    add(f"| Player | Pos | +pts/wk for you | {hdr} |")
    add("|---|---|---:|" + "---:|" * len(WIN_TARGETS))
    for c in a.candidates:
        pl = players[c.player_id]
        add(f"| {pl.name} | {pl.pos} | {c.gain:.1f} | " + " | ".join(f"${c.bid_for[t]}" for t in WIN_TARGETS) + " |")
    add("")

    add("### Roster holes (your starter vs the league median at that slot, avg pts/week)")
    add("")
    add("| Slot | Starter | You | League median | Gap | Rank |")
    add("|---|---|---:|---:|---:|---:|")
    for h in a.holes:
        flag = " ⚠️" if h["gap"] < -2 else ""
        add(f"| {h['slot']} | {h['player']} | {h['you']:.1f} | {h['league_median']:.1f} | "
            f"{h['gap']:+.1f}{flag} | {h['rank']}/{h['of']} |")
    add("")

    add("## League board")
    add("")
    add("| Team | FAAB | Proj pts/wk | P(cut this week) | P(win league) | Weakest slots |")
    add("|---|---:|---:|---:|---:|---|")
    for o in a.opponents:
        name = f"**{o['team']}**" if o["is_you"] else o["team"]
        add(f"| {name} | ${o['faab']:.0f} | {o['proj_weekly']:.1f} | {_pct(o['p_elim_next'])} | "
            f"{_pct(o['p_champion'])} | {o['needs']} |")
    add("")
    return "\n".join(lines)
