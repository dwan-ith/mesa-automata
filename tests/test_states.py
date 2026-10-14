"""Tests for state definitions, metadata, and color mappings."""

import pytest
from vn_automata.states import VNState, StateCategory, STATE_INFO, STATE_COLORS, Direction


def test_twenty_nine_states_exist():
    """Verify that exactly 29 states are defined with values 0 through 28."""
    assert len(VNState) == 29
    assert set(s.value for s in VNState) == set(range(29))


def test_state_metadata_completeness():
    """Verify that all 29 states have corresponding metadata."""
    for state in VNState:
        assert state in STATE_INFO
        meta = STATE_INFO[state]
        assert meta.code == state.value
        assert len(meta.rgb) == 3
        assert all(0 <= c <= 255 for c in meta.rgb)
        assert meta.hex_color.startswith("#")
        assert len(meta.hex_color) == 7


def test_categories():
    """Verify grouping of states into functional categories."""
    assert STATE_INFO[VNState.U].category == StateCategory.GROUND
    
    sensitized_states = [VNState(i) for i in range(1, 9)]
    for s in sensitized_states:
        assert STATE_INFO[s].category == StateCategory.SENSITIZED

    ots_states = [VNState(i) for i in range(9, 17)]
    for s in ots_states:
        assert STATE_INFO[s].category == StateCategory.ORDINARY_TRANSMISSION

    sts_states = [VNState(i) for i in range(17, 25)]
    for s in sts_states:
        assert STATE_INFO[s].category == StateCategory.SPECIAL_TRANSMISSION

    confluent_states = [VNState(i) for i in range(25, 29)]
    for s in confluent_states:
        assert STATE_INFO[s].category == StateCategory.CONFLUENT


def test_directional_properties():
    """Verify directional transmission states have proper direction properties."""
    assert STATE_INFO[VNState.OT_E].direction == Direction.EAST
    assert STATE_INFO[VNState.OT_N].direction == Direction.NORTH
    assert STATE_INFO[VNState.OT_W].direction == Direction.WEST
    assert STATE_INFO[VNState.OT_S].direction == Direction.SOUTH

    assert STATE_INFO[VNState.ST_E].direction == Direction.EAST
    assert STATE_INFO[VNState.ST_N].direction == Direction.NORTH
    assert STATE_INFO[VNState.ST_W].direction == Direction.WEST
    assert STATE_INFO[VNState.ST_S].direction == Direction.SOUTH
