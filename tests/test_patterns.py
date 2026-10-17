"""Behavioural tests: every preset pattern must exhibit the phenomenon it claims.

These run the *real* automaton (``step_grid``) over the pattern and assert on
the emergent behaviour, rather than trusting the pattern layout.
"""

import numpy as np

from vn_automata.patterns import (
    make_confluent_and_gate,
    make_confluent_splitter,
    make_construction_arm_1011,
    make_destruction_demo,
    make_programmable_builder,
    make_pulse_loop,
)
from vn_automata.rules import step_grid
from vn_automata.states import VNState, is_excited

TARGET_STATE = {
    "10000": VNState.OT_E,
    "10001": VNState.OT_N,
    "1001": VNState.OT_W,
    "1010": VNState.OT_S,
    "1011": VNState.ST_E,
    "1100": VNState.ST_N,
    "1101": VNState.ST_W,
    "1110": VNState.ST_S,
    "1111": VNState.C00,
}


def _to_array(pattern, width, height):
    array = np.zeros((height, width), dtype=np.int64)
    for (x, y), state in pattern.items():
        if 0 <= x < width and 0 <= y < height:
            array[y, x] = int(state)
    return array


def _excited(array):
    return {
        (x, y)
        for y in range(array.shape[0])
        for x in range(array.shape[1])
        if is_excited(int(array[y, x]))
    }


def test_pulse_loop_circulates_without_decay():
    """A single pulse must circulate forever, returning to its start."""
    start = (13, 12)
    array = _to_array(make_pulse_loop(12, 12, 8, 8), 40, 40)
    assert int(array[start[1], start[0]]) == VNState.OT_E_EX
    ring_cells = int((array != VNState.U).sum())

    for _ in range(40):
        array = step_grid(array)
        assert len(_excited(array)) == 1, "loop pulse must not decay or duplicate"
        assert int((array != VNState.U).sum()) == ring_cells

    # Perimeter of this ring is 28 cells, so the pulse returns after 28 ticks.
    array = _to_array(make_pulse_loop(12, 12, 8, 8), 40, 40)
    for _ in range(28):
        array = step_grid(array)
    assert _excited(array) == {start}


def test_confluent_splitter_fans_one_pulse_into_two():
    """One input pulse must become simultaneous pulses on both branches."""
    array = _to_array(make_confluent_splitter(10, 15), 35, 35)
    for _ in range(5):
        array = step_grid(array)
    excited = _excited(array)
    # Ordinary branch heads north from the node, special branch heads east.
    assert any(x == 14 and y >= 16 for x, y in excited), excited
    assert any(y == 15 and x >= 15 for x, y in excited), excited
    assert len(excited) == 2


def test_destruction_demo_erases_ordinary_cell():
    """An excited special pulse must erase the target ordinary cell to ground."""
    array = _to_array(make_destruction_demo(10, 15), 35, 35)
    assert int(array[15, 13]) == VNState.OT_N
    for _ in range(3):
        array = step_grid(array)
    assert int(array[15, 13]) == VNState.U


def test_and_gate_requires_both_inputs():
    """Output fires only when both input pulses arrive together."""
    both = _to_array(make_confluent_and_gate(10, 15), 35, 35)
    for _ in range(8):
        both = step_grid(both)
    # A signal propagated past the gate and sensitized the ground beyond it.
    assert int(both[15, 16]) != VNState.U

    single = _to_array(make_confluent_and_gate(10, 15), 35, 35)
    single[13, 13] = VNState.OT_N  # remove one incoming pulse, keep the wire
    for _ in range(8):
        single = step_grid(single)
    assert int(single[15, 16]) == VNState.U
    assert not any(y == 15 and x >= 14 for x, y in _excited(single))


def test_construction_arm_extends_by_one_cell():
    """The 1011 sequence must grow the arm by turning ground into ST_E."""
    array = _to_array(make_construction_arm_1011(10, 15), 35, 35)
    assert int(array[15, 15]) == VNState.U
    for _ in range(12):
        array = step_grid(array)
    assert int(array[15, 15]) == VNState.ST_E


def test_programmable_builder_synthesises_every_target():
    """Each canonical binary sequence must build its exact target state."""
    x, y = 5, 15
    for sequence, expected in TARGET_STATE.items():
        n = len(sequence)
        target = (x + n, y)
        array = _to_array(make_programmable_builder(sequence, x, y), 40, 40)
        for _ in range(n + 6):
            array = step_grid(array)
        assert int(array[target[1], target[0]]) == expected, (
            sequence, VNState(int(array[target[1], target[0]])).name
        )
