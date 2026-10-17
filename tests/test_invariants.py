"""Structural invariants of the 29-state vocabulary and the generated rule set.

These tests protect the *data model* the whole package is built on: the state
groups must partition the state space exactly, the directional lookup tables
must agree with the metadata, and the compiled automaton must have sane shape.
"""

import numpy as np
import pytest

from vn_automata import states
from vn_automata.golly import parse_colors
from vn_automata.rules import automaton, build_rule_set, rule_count
from vn_automata.states import (
    CONFLUENT,
    CONFLUENT_EMITTING,
    Direction,
    GROUND,
    OPPOSITE_DIRECTION,
    OTS,
    OTS_EXCITED,
    OTS_EXCITED_BY_DIR,
    OTS_QUIESCENT,
    OTS_QUIESCENT_BY_DIR,
    SENSITIZED,
    STS,
    STS_EXCITED,
    STS_EXCITED_BY_DIR,
    STS_QUIESCENT,
    STS_QUIESCENT_BY_DIR,
    STATE_COLORS,
    STATE_COUNT,
    STATE_INFO,
    VNState,
    is_excited,
)

GROUPS = {
    "ground": GROUND,
    "sensitized": SENSITIZED,
    "ots": OTS,
    "sts": STS,
    "confluent": CONFLUENT,
}


def test_state_groups_partition_the_state_space():
    every = set()
    for name, group in GROUPS.items():
        assert not (every & group), f"{name} overlaps a previous group"
        every |= group
    assert every == set(range(STATE_COUNT)) == set(VNState)


def test_excited_states_are_exactly_the_emitting_ones():
    expected = OTS_EXCITED | STS_EXCITED | CONFLUENT_EMITTING
    actual = {VNState(s) for s in VNState if is_excited(s)}
    assert actual == expected
    # Ground, sensitized and quiescent cells never count as pulses.
    assert not any(is_excited(s) for s in GROUND | SENSITIZED)


def test_directional_lookup_tables_match_metadata():
    for table, category in (
        (OTS_QUIESCENT_BY_DIR, states.StateCategory.ORDINARY_TRANSMISSION),
        (OTS_EXCITED_BY_DIR, states.StateCategory.ORDINARY_TRANSMISSION),
        (STS_QUIESCENT_BY_DIR, states.StateCategory.SPECIAL_TRANSMISSION),
        (STS_EXCITED_BY_DIR, states.StateCategory.SPECIAL_TRANSMISSION),
    ):
        assert set(table) == {Direction.EAST, Direction.NORTH, Direction.WEST, Direction.SOUTH}
        for direction, state in table.items():
            meta = STATE_INFO[VNState(state)]
            assert meta.direction == direction
            assert meta.category == category


def test_quiescent_and_excited_families_are_parallel_numbering():
    """Canonical numbering: excited = quiescent + 4 within each family."""
    for direction in Direction:
        if direction == Direction.NONE:
            continue
        assert OTS_EXCITED_BY_DIR[direction] - OTS_QUIESCENT_BY_DIR[direction] == 4
        assert STS_EXCITED_BY_DIR[direction] - STS_QUIESCENT_BY_DIR[direction] == 4


def test_opposite_direction_is_an_involution():
    assert set(OPPOSITE_DIRECTION) == {Direction.EAST, Direction.NORTH, Direction.WEST, Direction.SOUTH}
    for d, opp in OPPOSITE_DIRECTION.items():
        assert OPPOSITE_DIRECTION[opp] == d
        assert d != opp


def test_metadata_is_complete_and_consistent():
    colors = set()
    symbols = set()
    for state in VNState:
        meta = STATE_INFO[state]
        assert meta.code == int(state)
        assert meta.rgb == tuple(int(meta.hex_color[i:i + 2], 16) for i in (1, 3, 5))
        colors.add(meta.hex_color)
        symbols.add(meta.symbol)
    assert len(colors) == 29, "hex colours must be unique so every state is distinguishable"
    assert len(symbols) == 29, "symbols must be unique so every state is distinguishable"


def test_palette_matches_golly_canonical_colors():
    golly = parse_colors()
    for state in VNState:
        assert STATE_INFO[state].rgb == golly[int(state)]


def test_non_directional_states_have_no_direction():
    for state in GROUND | SENSITIZED | CONFLUENT:
        assert STATE_INFO[VNState(state)].direction == Direction.NONE


def test_automaton_is_lazy_singleton():
    assert automaton() is automaton()
    table = automaton().table
    assert table.dtype == np.uint8
    assert table.size == STATE_COUNT ** 5
    assert automaton().n_states == STATE_COUNT


def test_rule_set_shape_and_stability():
    rs = build_rule_set()
    assert rs.n_states == STATE_COUNT
    assert len(rs) == rule_count()
    assert 100 <= len(rs) < 250, "the generated rule set should stay human-scale"
    # Determinism: repeated generation must produce the identical rule list.
    assert build_rule_set().rules == rs.rules


def test_quiescent_configurations_are_fixed_points():
    """Orderly tissue is stable: ground, quiescent wires and C00 never move."""
    from vn_automata.rules import step_grid

    rng = np.random.default_rng(1)
    quiescent = np.array(sorted(GROUND | OTS_QUIESCENT | STS_QUIESCENT | {min(CONFLUENT)}))
    for _ in range(6):
        grid = rng.choice(quiescent, size=(12, 12))
        assert np.array_equal(step_grid(grid), grid)
