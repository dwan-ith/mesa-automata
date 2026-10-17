"""A small, generic engine for synchronous 5-neighbour (von Neumann) CA.

The engine is deliberately rule-agnostic. A :class:`RuleSet` collects
transition rules of the form "if the centre cell is in set ``C`` and the
north/east/south/west neighbours are in sets ``N``/``E``/``S``/``W`` then the
cell takes state ``new``". That is exactly the table format used by Golly's
``.rule`` files and by the von Neumann model literature.

The rule set is **compiled** into a flat lookup table of ``n_states ** 5``
bytes, indexed by ``(((c*n + north)*n + east)*n + south)*n + west``. Stepping a
whole grid then reduces to a single vectorised gather, which is both fast and
trivially correct: there is exactly one place where the dynamics are defined.

Rule semantics (matching Golly):

* rules are evaluated in the order they were added — **first match wins**;
* a neighbourhood matching no rule leaves the cell **unchanged**.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

Domain = Tuple[int, ...]
CompiledRule = Tuple[Tuple[Domain, Domain, Domain, Domain, Domain], int]


class RuleSet:
    """An ordered collection of 5-neighbour transition rules.

    Rules are stored in priority order. Each position accepts any iterable of
    state codes; iterables are de-duplicated and sorted on entry.
    """

    def __init__(self, n_states: int) -> None:
        if n_states < 1 or n_states > 256:
            raise ValueError("n_states must be in 1..256")
        self.n_states = int(n_states)
        self.rules: List[CompiledRule] = []

    def add(
        self,
        center: Iterable[int],
        north: Iterable[int],
        east: Iterable[int],
        south: Iterable[int],
        west: Iterable[int],
        new: int,
    ) -> None:
        """Append a rule. Neighbour order is (north, east, south, west)."""
        if not 0 <= int(new) < self.n_states:
            raise ValueError(f"new state {new} out of range")
        domains = tuple(
            tuple(sorted({int(x) for x in group}))
            for group in (center, north, east, south, west)
        )
        if any(len(d) == 0 for d in domains):
            raise ValueError("rule domains must be non-empty")
        if any(not 0 <= v < self.n_states for d in domains for v in d):
            raise ValueError("rule domain contains out-of-range state")
        self.rules.append((domains, int(new)))

    def __len__(self) -> int:
        return len(self.rules)

    def compile(self) -> np.ndarray:
        """Return the flat ``uint8`` transition table for this rule set."""
        n = self.n_states
        if n > 255:
            raise ValueError("compiled table requires n_states <= 255")
        strides = (n ** 4, n ** 3, n ** 2, n, 1)

        table = np.empty((n,) * 5, dtype=np.uint8)
        # Default behaviour: every cell keeps its current state.
        table[:] = np.arange(n, dtype=np.uint8)[:, None, None, None, None]

        flat = table.reshape(-1)
        # Apply in reverse so the earliest, highest-priority rule wins.
        for domains, new in reversed(self.rules):
            idx = 0
            for axis, (domain, stride) in enumerate(zip(domains, strides)):
                coords = np.fromiter(domain, dtype=np.int64, count=len(domain))
                shape = [1] * 5
                shape[axis] = len(domain)
                idx = idx + coords.reshape(shape) * stride
            flat[idx.ravel()] = new
        return flat


class TableAutomaton:
    """A concrete CA defined by a compiled transition table."""

    def __init__(self, n_states: int, table: np.ndarray) -> None:
        self.n_states = int(n_states)
        table = np.ascontiguousarray(table, dtype=np.uint8)
        if table.size != self.n_states ** 5:
            raise ValueError(
                f"table size {table.size} != n_states**5 ({self.n_states ** 5})"
            )
        self.table = table

    @classmethod
    def from_ruleset(cls, ruleset: RuleSet) -> "TableAutomaton":
        return cls(ruleset.n_states, ruleset.compile())

    # -- single cell --------------------------------------------------------
    def next_state(self, center: int, north: int, east: int, south: int, west: int) -> int:
        n = self.n_states
        index = (((int(center) * n + int(north)) * n + int(east)) * n + int(south)) * n + int(west)
        return int(self.table[index])

    # -- whole grid ---------------------------------------------------------
    def neighbour_arrays(self, grid: np.ndarray, torus: bool = False):
        """Return ``(north, east, south, west)`` neighbour arrays for ``grid``.

        ``grid`` is indexed ``grid[y, x]`` (Mesa Cartesian convention, y up), so
        north is ``(x, y + 1)``.
        """
        grid = np.asarray(grid)
        if torus:
            north = np.roll(grid, -1, axis=0)
            south = np.roll(grid, 1, axis=0)
            east = np.roll(grid, -1, axis=1)
            west = np.roll(grid, 1, axis=1)
        else:
            padded = np.pad(grid, 1, mode="constant", constant_values=0)
            north = padded[2:, 1:-1]
            south = padded[:-2, 1:-1]
            west = padded[1:-1, :-2]
            east = padded[1:-1, 2:]
        return north, east, south, west

    def step(self, grid: np.ndarray, torus: bool = False) -> np.ndarray:
        """Advance a whole grid one synchronous tick."""
        grid = np.asarray(grid)
        if grid.ndim != 2:
            raise ValueError("grid must be 2-D")
        if not np.issubdtype(grid.dtype, np.integer):
            grid = grid.astype(np.int64)  # e.g. float input is truncated to states
        out_dtype = grid.dtype
        north, east, south, west = self.neighbour_arrays(grid, torus=torus)
        n = self.n_states
        c = grid.astype(np.int64)
        index = ((((c * n + north) * n + east) * n + south) * n + west)
        return self.table[index].astype(out_dtype)


def compile_rules(n_states: int, rules: Sequence[CompiledRule]) -> np.ndarray:
    """Convenience: compile an explicit list of rules into a flat table."""
    ruleset = RuleSet(n_states)
    for domains, new in rules:
        ruleset.add(*domains, new)
    return ruleset.compile()
