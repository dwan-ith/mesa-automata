"""Tests for Golly Extended RLE interoperability."""

import pytest

from vn_automata.rle import (
    RLEError,
    export_rle,
    parse_rle,
    state_to_token,
    token_to_state,
)
from vn_automata.states import VNState


def test_token_encoding_matches_golly_spec():
    """State tokens must follow the documented Golly encoding exactly."""
    assert state_to_token(0) == "."
    assert state_to_token(1) == "A"
    assert state_to_token(24) == "X"
    # 29-state rule: 25..28 spill into the "p" block.
    assert state_to_token(25) == "pA"
    assert state_to_token(26) == "pB"
    assert state_to_token(27) == "pC"
    assert state_to_token(28) == "pD"


def test_token_roundtrip_all_states():
    for state in range(29):
        assert token_to_state(state_to_token(state)) == state


def test_token_accepts_two_state_aliases():
    assert token_to_state("b") == 0
    assert token_to_state("o") == 1


def test_parse_simple_rle_body():
    # Two rows: row0 "A.A" (state1, dead, state1), row1 single "B".
    rle = "x = 3, y = 2, rule = JvN29\nA.A$B!\n"
    parsed = parse_rle(rle)
    assert parsed.width == 3 and parsed.height == 2
    # Top row is y=1 in Cartesian coordinates.
    assert parsed.pattern == {
        (0, 1): VNState.S,
        (2, 1): VNState.S,
        (0, 0): VNState.S0,
    }


def test_parse_multistate_block_tokens():
    rle = "x = 2, y = 1, rule = JvN29\npApD!\n"
    parsed = parse_rle(rle)
    assert parsed.pattern[(0, 0)] == VNState.C00
    assert parsed.pattern[(1, 0)] == VNState.C11


def test_roundtrip_arbitrary_pattern():
    pattern = {
        (0, 0): VNState.U,
        (1, 0): VNState.OT_E_EX,
        (2, 0): VNState.C10,
        (0, 1): VNState.ST_S_EX,
        (3, 2): VNState.S000,
    }
    text = export_rle(pattern, rule="JvN29", comments=["von Neumann demo"])
    parsed = parse_rle(text)
    assert parsed.rule == "JvN29"
    assert parsed.pattern == {k: v for k, v in pattern.items() if v != VNState.U}
    assert parsed.comments == ["von Neumann demo"]


def test_roundtrip_every_state():
    pattern = {(x, 0): VNState(x) for x in range(1, 29)}
    parsed = parse_rle(export_rle(pattern))
    assert parsed.pattern == pattern


def test_out_of_range_state_rejected():
    with pytest.raises(RLEError):
        token_to_state("yP")  # beyond state 255
