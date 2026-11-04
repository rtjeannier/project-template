"""Markdown report for an Analysis."""

from __future__ import annotations

from .analysis import WIN_TARGETS, Analysis, survive_through


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _signed_pct(x: float) -> str:
    return f"{100 * x:+.1f} pts"


def render(a: Analysis) -> str:
    st = a.state
    me = st.my_team_id
    my = st.team(me)
    players = st.players
    res = a.baseline
    lines = []
    add = lines.append

    add(f"# Guillotine analysis: week {st.settings.current_week}, {st.settings.season}")
    add("")
    add(f"{len(st.teams)} teams alive, {len(res.weeks)} weeks left, {res.n_sims} simulated seasons. "
        f"Player values from **{a.sim.proj.value_source}**.")
    add("")

    add(f"## {my.name}: ${my.faab:.0f} FAAB")
    add("")
    add(f"- Chance to win the league: **{_pct(res.p_champion(me))}** if you follow the model's bids, "
        f"{_pct(a.passed.p_champion(me))} if you pass this week")
    alive = res.p_alive_after(me)
    add("- Chance you're still alive after each week: " + ", ".join(
        f"wk{w} {_pct(p)}" for w, p in zip(res.weeks, alive)))
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

    add("## This week's waiver targets")
    add("")
    add("**Bid for X%** = the bid that beats every other manager in X% of simulations. Each tried bid "
        "($1 and the 25/50/75/90% levels) is entered into the auction for real: you only win the player if it "
        "beats the field. **Best bid** = the one with the highest P(win league). Effects are vs passing.")
    add("")
    hdr = " | ".join(f"Bid for {int(t * 100)}%" for t in WIN_TARGETS)
    add(f"| Player | Pos | +pts/wk for you | {hdr} | Best bid (P win bid) | Δ P(win league) | Δ P(alive 3 wks) |")
    add("|---|---|---:|" + "---:|" * len(WIN_TARGETS) + "---:|---:|---:|")
    for c in a.candidates:
        p = players[c.player_id]
        bids = " | ".join(f"${c.bid_for[t]}" for t in WIN_TARGETS)
        add(f"| {p.name} | {p.pos} | {c.gain:.1f} | {bids} | ${c.suggested_bid} "
            f"({_pct(c.p_win_at_suggested)}) | {_signed_pct(c.d_champion)} | {_signed_pct(c.d_survive_3)} |")
    if not a.candidates:
        add("| (nobody in the pool would start for you) |")
    add("")
    if a.candidates:
        add("P(win league) at each bid tried:")
        add("")
        for c in a.candidates:
            add(f"- {players[c.player_id].name}: " + ", ".join(
                f"${amt} → {_pct(ch)} (win bid {_pct(pw)})" for amt, pw, ch in c.curve))
        add("")
    if a.sheet_result is not None:
        sheet = ", ".join(f"{players[pid].name} ${amt}" for pid, amt in
                          sorted(a.sheet.items(), key=lambda kv: -kv[1]))
        add(f"**Bid sheet:** {sheet}. Submitted together: P(win league) "
            f"**{_pct(a.sheet_result.p_champion(me))}**, P(alive after 3 wks) "
            f"{_pct(survive_through(a.sheet_result, me, 3))}.")
        add("")
    add(f"Baseline if you pass this week: P(win league) {_pct(a.passed.p_champion(me))}, "
        f"P(alive after 3 wks) {_pct(survive_through(a.passed, me, 3))}.")
    p = a.passed.p_champion(me)
    noise = 100 * (2 * p * (1 - p) / res.n_sims) ** 0.5
    add(f"Differences smaller than about ±{noise:.1f} pts are within simulation noise at "
        f"{res.n_sims} sims; rerun with more `--sims` before close calls.")
    add("")

    add("## Future targets: players likely to hit waivers later")
    add("")
    add("| Player | Pos | Current team | P(hits waivers) | Typical week | +pts/wk for you | Median price | P(you get him) |")
    add("|---|---|---|---:|---:|---:|---:|---:|")
    for r in a.future_targets:
        add(f"| {r['player']} | {r['pos']} | {r['owner']} | {_pct(r['p_available'])} | {r['typical_week']} | "
            f"{r['your_gain']:.1f} | ${r['median_price']:.0f} | {_pct(r['p_you_win'])} |")
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
