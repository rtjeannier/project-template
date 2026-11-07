"""Optimal starting lineup for a roster.

Dedicated slots are filled first with the best player at each position, then flex
slots (narrowest eligibility first) take the best remaining eligible player. For
standard nested flex structures this greedy fill is optimal.
"""

from __future__ import annotations

from .models import FLEX_ELIGIBILITY, NON_STARTING_SLOTS


class LineupSpec:
    def __init__(self, roster_slots: dict[str, int]):
        self.dedicated = []
        self.flex = []
        for slot, count in roster_slots.items():
            if slot in NON_STARTING_SLOTS or count <= 0:
                continue
            if slot in FLEX_ELIGIBILITY:
                # A tuple, not a set: on a tie the first position listed wins, so lineups don't
                # depend on Python's per-process ordering of a set of strings.
                self.flex.append((tuple(FLEX_ELIGIBILITY[slot]), count))
            else:
                self.dedicated.append((slot, count))
        self.flex.sort(key=lambda fc: len(fc[0]))
        self.n_starters = sum(c for _, c in self.dedicated) + sum(c for _, c in self.flex)

    def assign(self, players: list[int], values, positions: list[str]) -> list[tuple[str, int | None]]:
        """Fill every starting slot: list of (slot label, player index or None if empty).

        ``values`` is indexable by player index, ``positions`` maps index -> position.
        """
        by_pos: dict[str, list[tuple[float, int]]] = {}
        for p in players:
            by_pos.setdefault(positions[p], []).append((values[p], p))
        for lst in by_pos.values():
            lst.sort(reverse=True)

        out: list[tuple[str, int | None]] = []
        leftovers: dict[str, list[tuple[float, int]]] = {}
        for pos, count in self.dedicated:
            lst = by_pos.get(pos, [])
            for k in range(count):
                out.append((pos, lst[k][1] if k < len(lst) else None))
            leftovers[pos] = lst[count:]
        for pos, lst in by_pos.items():
            leftovers.setdefault(pos, lst)

        for eligible, count in self.flex:
            label = "FLEX" if set(eligible) != {"QB", "RB", "WR", "TE"} else "SUPERFLEX"
            for _ in range(count):
                best_pos = None
                best_v = None
                for pos in eligible:
                    lst = leftovers.get(pos)
                    if lst and (best_v is None or lst[0][0] > best_v):
                        best_v, best_pos = lst[0][0], pos
                if best_pos is None:
                    out.append((label, None))
                    continue
                out.append((label, leftovers[best_pos].pop(0)[1]))
        return out

    def best(self, players: list[int], values, positions: list[str]) -> tuple[float, list[int]]:
        """Return (total projected points, starter indices)."""
        starters = [p for _, p in self.assign(players, values, positions) if p is not None]
        return float(sum(values[p] for p in starters)), starters

    def total(self, players: list[int], values, positions: list[str]) -> float:
        return self.best(players, values, positions)[0]

    def adder(self, players: list[int], values, positions: list[str]):
        """Fast "lineup total if we add player x" for one roster: the roster's values are
        sorted once, and each call only inserts x's value. Same result as
        total(players + [x], ...). Call with x=None for the roster's own total."""
        by_pos: dict[str, list[float]] = {}
        for p in players:
            by_pos.setdefault(positions[p], []).append(float(values[p]))
        for lst in by_pos.values():
            lst.sort(reverse=True)
        dedicated, flex = self.dedicated, self.flex

        def fill(lists: dict[str, list[float]]) -> float:
            total = 0.0
            left: dict[str, list[float]] = {}
            for pos, count in dedicated:
                lst = lists.get(pos, [])
                total += sum(lst[:count])
                left[pos] = lst[count:]
            for pos, lst in lists.items():
                left.setdefault(pos, lst)
            heads = {pos: 0 for pos in left}
            for eligible, count in flex:
                for _ in range(count):
                    best_pos, best_v = None, None
                    for pos in eligible:
                        lst = left.get(pos)
                        if lst is not None and heads[pos] < len(lst) and (best_v is None or lst[heads[pos]] > best_v):
                            best_v, best_pos = lst[heads[pos]], pos
                    if best_pos is None:
                        continue
                    total += best_v
                    heads[best_pos] += 1
            return total

        base = fill(by_pos)

        def add(x=None) -> float:
            if x is None:
                return base
            pos = positions[x]
            lst = list(by_pos.get(pos, []))
            v = float(values[x])
            i = 0
            while i < len(lst) and lst[i] >= v:
                i += 1
            lst.insert(i, v)
            lists = dict(by_pos)
            lists[pos] = lst
            return fill(lists)

        return add
