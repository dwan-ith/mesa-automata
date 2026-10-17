"""Von Neumann's 29-state transition function, *generated* from the model.

Rather than hand-writing one branch per state (which is how the automaton is
usually transcribed, and how it was previously implemented here), this module
*describes* the automaton in terms of von Neumann's own concepts and lets
:mod:`vn_automata.engine` compile that description into a lookup table.

The vocabulary is directional. An ordinary/special transmission cell "points"
at one of its four neighbours; an excited transmission cell injects its pulse
into the neighbour it points at. So, for a cell, the only neighbours that can
feed it are those whose direction points *at* it, i.e. a neighbour on side
``s`` must have direction ``opposite(s)``.

From that single idea the whole rule set follows:

* **Ground** becomes sensitized when any neighbour injects an excited pulse.
* **Sensitized** cells walk a binary decision tree (``S -> S0/S1 -> ...``),
  branching on whether an excited pulse arrives this tick, until they settle
  into one of the nine functional states.
* **Transmission** cells (ordinary and special) are mutually antagonistic:
  an excited pulse of the *other* family erases the cell to ground. Otherwise
  they relay an incoming pulse one cell onward (they are fed by any neighbour
  except the one they point at) and relax to quiescent when no pulse arrives.
* **Confluent** cells are 2-bit delay buffers / AND gates. They sample the set
  of ordinary-transmission lines pointing at them and emit to all four
  neighbours only when *every* such line is excited and at least one exists.

The generated rule set is checked, over the entire neighbourhood space, against
the canonical Golly ``JvN29.rule`` table — see ``tests/test_golden_golly.py``.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Mapping, Optional, Tuple

import numpy as np

from .engine import RuleSet, TableAutomaton
from .states import (
    CONFLUENT,
    CONFLUENT_EMITTING,
    Direction,
    OPPOSITE_DIRECTION,
    OTS_EXCITED_BY_DIR,
    OTS_QUIESCENT_BY_DIR,
    STATE_COUNT,
    STS_EXCITED_BY_DIR,
    STS_QUIESCENT_BY_DIR,
    VNState,
)

ALL_STATES: FrozenSet[int] = frozenset(range(STATE_COUNT))
DIRECTIONS: Tuple[Direction, ...] = (
    Direction.EAST,
    Direction.NORTH,
    Direction.WEST,
    Direction.SOUTH,
)

#: Neighbour position -> engine parameter name.
_SIDE_NAME = {
    Direction.NORTH: "north",
    Direction.EAST: "east",
    Direction.SOUTH: "south",
    Direction.WEST: "west",
}

#: The sensitized bifurcation tree: state -> (branch on bit 0, branch on bit 1).
_TREE = {
    VNState.S: (VNState.S0, VNState.S1),
    VNState.S0: (VNState.S00, VNState.S01),
    VNState.S1: (VNState.S10, VNState.S11),
    VNState.S00: (VNState.S000, VNState.OT_W),
    VNState.S01: (VNState.OT_S, VNState.ST_E),
    VNState.S10: (VNState.ST_N, VNState.ST_W),
    VNState.S11: (VNState.ST_S, VNState.C00),
    VNState.S000: (VNState.OT_E, VNState.OT_N),
}

#: Confluent 2-bit delay buffer: state -> (bit emitted now, bit buffered next).
_CONFLUENT_BITS = {
    VNState.C00: (0, 0),
    VNState.C01: (0, 1),
    VNState.C10: (1, 0),
    VNState.C11: (1, 1),
}
_BITS_TO_CONFLUENT = {bits: state for state, bits in _CONFLUENT_BITS.items()}


def _pointing_into(by_direction: Mapping[Direction, int], side: Direction) -> FrozenSet[int]:
    """States of one directional family, on neighbour ``side``, that point at us."""
    return frozenset({by_direction[OPPOSITE_DIRECTION[side]]})


def _pulse_in(side: Direction) -> FrozenSet[int]:
    """Any excited ordinary or special transmission cell pointing into ``side``."""
    return _pointing_into(OTS_EXCITED_BY_DIR, side) | _pointing_into(STS_EXCITED_BY_DIR, side)


def build_rule_set() -> RuleSet:
    """Construct the complete von Neumann 29-state rule set."""
    rs = RuleSet(STATE_COUNT)

    def rule(center, new, domains: Optional[Dict[Direction, FrozenSet[int]]] = None) -> None:
        domains = domains or {}
        rs.add(
            center=center,
            north=domains.get(Direction.NORTH, ALL_STATES),
            east=domains.get(Direction.EAST, ALL_STATES),
            south=domains.get(Direction.SOUTH, ALL_STATES),
            west=domains.get(Direction.WEST, ALL_STATES),
            new=int(new),
        )

    # -- 1. Ground ---------------------------------------------------------
    # Unexcitable background becomes sensitized when fed an excited pulse.
    for s in DIRECTIONS:
        rule({VNState.U}, VNState.S, {s: _pulse_in(s)})

    # -- 2. Sensitized bifurcation tree ------------------------------------
    # A sensitized cell reads one bit per tick and walks towards the addressed
    # functional state; if no pulse arrives it takes the "0" branch.
    for state, (bit0, bit1) in _TREE.items():
        rule({state}, bit0, {s: ALL_STATES - _pulse_in(s) for s in DIRECTIONS})
        for s in DIRECTIONS:
            rule({state}, bit1, {s: _pulse_in(s)})

    # -- 3. Transmission families (ordinary and special) -------------------
    _add_transmission_rules(
        rule,
        quiescent_by_dir=OTS_QUIESCENT_BY_DIR,
        excited_by_dir=OTS_EXCITED_BY_DIR,
        killer_by_dir=STS_EXCITED_BY_DIR,  # special pulses erase ordinary cells
    )
    _add_transmission_rules(
        rule,
        quiescent_by_dir=STS_QUIESCENT_BY_DIR,
        excited_by_dir=STS_EXCITED_BY_DIR,
        killer_by_dir=OTS_EXCITED_BY_DIR,  # ordinary pulses erase special cells
    )

    # -- 4. Confluent (delay buffer + AND gate) ----------------------------
    _add_confluent_rules(rule)
    return rs


def _add_transmission_rules(rule, *, quiescent_by_dir, excited_by_dir, killer_by_dir) -> None:
    """Rules shared by the ordinary and special transmission families."""

    def feed_in(s: Direction) -> FrozenSet[int]:
        # Fed by an excited cell of our own family, or by an emitting confluent.
        return _pointing_into(excited_by_dir, s) | CONFLUENT_EMITTING

    def killer_in(s: Direction) -> FrozenSet[int]:
        return _pointing_into(killer_by_dir, s)

    for d in DIRECTIONS:
        qui = quiescent_by_dir[d]
        exc = excited_by_dir[d]

        # (a) Antagonistic erasure by the other family takes priority.
        for s in DIRECTIONS:
            rule({qui, exc}, VNState.U, {s: killer_in(s)})

        # (b) Relaying: a pulse from any side *other than* the one we point at
        #     (our downstream neighbour) excites us.
        for s in DIRECTIONS:
            if s != d:
                rule({qui, exc}, exc, {s: feed_in(s)})

        # (c) An already-excited cell relaxes to quiescent unless still fed.
        domains: Dict[Direction, FrozenSet[int]] = {}
        for s in DIRECTIONS:
            if s == d:
                # Downstream neighbour imposes no feed condition.
                domains[s] = ALL_STATES - killer_in(s)
            else:
                domains[s] = ALL_STATES - (feed_in(s) | killer_in(s))
        rule({exc}, qui, domains)


def _add_confluent_rules(rule) -> None:
    def excited_line_in(s: Direction) -> FrozenSet[int]:
        return _pointing_into(OTS_EXCITED_BY_DIR, s)

    def quiescent_line_in(s: Direction) -> FrozenSet[int]:
        return _pointing_into(OTS_QUIESCENT_BY_DIR, s)

    # Antagonistic erasure by an excited special pulse.
    for s in DIRECTIONS:
        rule(CONFLUENT, VNState.U, {s: _pointing_into(STS_EXCITED_BY_DIR, s)})

    def advance(center: VNState, input_bit: int) -> int:
        _, buffered = _CONFLUENT_BITS[center]
        return int(_BITS_TO_CONFLUENT[(buffered, input_bit)])

    # Input = 1: at least one excited line, and *no* quiescent line anywhere
    # (i.e. every input line is excited — von Neumann's AND).
    for feed in DIRECTIONS:
        domains = {
            s: (excited_line_in(s) if s == feed else ALL_STATES - quiescent_line_in(s))
            for s in DIRECTIONS
        }
        for center in CONFLUENT:
            rule({center}, advance(center, 1), domains)

    # Input = 0: a quiescent line is present ...
    for quiet in DIRECTIONS:
        domains = {quiet: quiescent_line_in(quiet)}
        for center in CONFLUENT:
            rule({center}, advance(center, 0), domains)
    # ... or there is no excited line at all.
    domains = {s: ALL_STATES - excited_line_in(s) for s in DIRECTIONS}
    for center in CONFLUENT:
        rule({center}, advance(center, 0), domains)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_AUTOMATON: Optional[TableAutomaton] = None


def automaton() -> TableAutomaton:
    """Return the compiled von Neumann automaton (built once, lazily)."""
    global _AUTOMATON
    if _AUTOMATON is None:
        _AUTOMATON = TableAutomaton.from_ruleset(build_rule_set())
    return _AUTOMATON


def cell_next_state(center: int, north: int, east: int, south: int, west: int) -> int:
    """Synchronous next state of one cell from its 5-cell von Neumann neighbourhood."""
    return automaton().next_state(center, north, east, south, west)


def step_grid(grid: np.ndarray, torus: bool = False) -> np.ndarray:
    """Advance a 2-D ``grid[y, x]`` of states by one synchronous tick."""
    return automaton().step(grid, torus=torus)


def rule_count() -> int:
    """Number of generated transition rules (diagnostics/tests)."""
    return len(build_rule_set())
