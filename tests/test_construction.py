"""Tests for construction bifurcation tree and pulse-encoded synthesis.

Verifies that all 9 valid quiescent target states are synthesized by their
canonical binary pulse sequence:
- 10000 -> OT_E
- 10001 -> OT_N
- 1001  -> OT_W
- 1010  -> OT_S
- 1011  -> ST_E
- 1100  -> ST_N
- 1101  -> ST_W
- 1110  -> ST_S
- 1111  -> C00
"""

import pytest
from vn_automata.states import VNState
from vn_automata.rules import cell_next_state


def simulate_pulse_train_on_cell(bits: str) -> int:
    """Feed a binary sequence of pulses into a ground cell from an adjacent wire."""
    state = VNState.U

    for bit in bits:
        # A bit 1 represents an incoming excited transmission pulse from West
        # A bit 0 represents a quiescent transmission cell from West
        west = VNState.ST_E_EX if (bit == "1") else VNState.ST_E
        state = cell_next_state(
            center=state,
            north=VNState.U,
            east=VNState.U,
            south=VNState.U,
            west=west,
        )

    return state


@pytest.mark.parametrize(
    "sequence, expected_state",
    [
        ("10000", VNState.OT_E),   # East ordinary transmission quiescent
        ("10001", VNState.OT_N),   # North ordinary transmission quiescent
        ("1001",  VNState.OT_W),   # West ordinary transmission quiescent
        ("1010",  VNState.OT_S),   # South ordinary transmission quiescent
        ("1011",  VNState.ST_E),   # East special transmission quiescent (arm extension)
        ("1100",  VNState.ST_N),   # North special transmission quiescent
        ("1101",  VNState.ST_W),   # West special transmission quiescent
        ("1110",  VNState.ST_S),   # South special transmission quiescent
        ("1111",  VNState.C00),    # Confluent state quiescent
    ]
)
def test_all_nine_construction_targets(sequence: str, expected_state: VNState):
    """Verify that every sequence builds its exact corresponding quiescent state."""
    final_state = simulate_pulse_train_on_cell(sequence)
    assert final_state == expected_state, (
        f"Sequence {sequence} resulted in {VNState(final_state).name}, expected {expected_state.name}"
    )


def test_construction_arm_step_by_step_1011():
    """Verify the exact intermediate sensitized states for sequence 1011."""    # Step 0: Target is Ground U
    state = VNState.U

    # Pulse 1: U -> S
    state = cell_next_state(state, VNState.U, VNState.U, VNState.U, VNState.ST_E_EX)
    assert state == VNState.S

    # Pulse 0: S -> S0
    state = cell_next_state(state, VNState.U, VNState.U, VNState.U, VNState.ST_E)
    assert state == VNState.S0

    # Pulse 1: S0 -> S01
    state = cell_next_state(state, VNState.U, VNState.U, VNState.U, VNState.ST_E_EX)
    assert state == VNState.S01

    # Pulse 1: S01 -> ST_E
    state = cell_next_state(state, VNState.U, VNState.U, VNState.U, VNState.ST_E_EX)
    assert state == VNState.ST_E
