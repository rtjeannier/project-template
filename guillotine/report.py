"""Markdown report for an Analysis."""

from __future__ import annotations

from .analysis import WIN_TARGETS, Analysis, survive_through


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _signed_pct(x: float) -> str:
    return f"{100 * x:+.1f} pts"


def _share(s: float, faab: float) -> str:
    if s <= 0:
        return "don't bid"
    if s < 0.01:
        return "$1"
    return f"{100 * s:.0f}% of FAAB (${s * faab:.0f} today)"


def _level(s: float) -> str:
    return "none" if s <= 0 else "$1" if s < 0.01 else f"{100 * s:.0f}%"


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
    add("Each row is a standing rule: **whenever he's on waivers and you're still alive, bid this much**. "
        "Candidates are this week's best pool players plus the players on other rosters who are irreplaceable in "
        "the final (projected above week-17 replacement) and likely to hit waivers while you can bid. "
        "**Final edge** = his projected points/week above week-17 replacement: in the final he counts for you "
        "*and* against your opponent. Effects are vs never bidding on him.")
    add("")
    add("| Player | Pos | Now on | Final edge | Bid | Δ P(win league) | P(you get him, when you can bid) | Typical price | Usually available |")
    add("|---|---|---|---:|---|---:|---:|---:|---|")
    for t in a.targets:
        pl = players[t["player_id"]]
        b = t["best"]
        wk = f"wk {sorted(b['weeks'])[len(b['weeks']) // 2]}" if b["weeks"] else "-"
        add(f"| {pl.name} | {pl.pos} | {t['owner']} | {t['edge']:+.1f} | {_share(b['share'], my.faab)} | "
            f"{_signed_pct(t['gain_vs_pass'])} | {_pct(b['p_get'])} | "
            f"{'$' + format(b['avg_paid'], '.0f') if b['avg_paid'] else '-'} | {wk} "
            f"({_pct(b['p_chance'])} of seasons) |")
    add("")
    add("P(win league) at each bid level tried (share of your FAAB at the time):")
    add("")
    for t in a.targets:
        add(f"- {players[t['player_id']].name}: " + ", ".join(
            f"{_level(c['share'])} → {_pct(c['p_champ'])}" for c in t["curve"]))
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
