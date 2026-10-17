"""Fuzz round-trips and charset invariants for both serialisation formats."""

import random

import numpy as np
import pytest

from vn_automata.loader import (
    CHAR_TO_STATE,
    STATE_TO_CHAR,
    export_to_ascii,
    parse_ascii_pattern,
)
from vn_automata.rle import N_STATES, export_rle, parse_rle, state_to_token, token_to_state
from vn_automata.states import VNState


def _random_pattern(rng, max_side=12):
    width = rng.randint(1, max_side)
    height = rng.randint(1, max_side)
    pattern = {}
    for x in range(width):
        for y in range(height):
            if rng.random() < 0.35:
                pattern[(x, y)] = rng.choice([int(s) for s in VNState])
    return pattern


def test_rle_tokens_are_unique_and_injective():
    tokens = [state_to_token(s) for s in range(N_STATES)]
    assert len(set(tokens)) == N_STATES
    for state in range(N_STATES):
        assert token_to_token_roundtrip(state) == state


def token_to_token_roundtrip(state):
    return token_to_state(state_to_token(state))


def test_rle_roundtrip_fuzz():
    rng = random.Random(1234)
    for _ in range(150):
        pattern = _random_pattern(rng)
        parsed = parse_rle(export_rle(pattern, rule="JvN29"))
        assert parsed.pattern == {k: v for k, v in pattern.items() if v != 0}


def test_rle_roundtrip_fuzz_with_padding():
    rng = random.Random(5678)
    for _ in range(60):
        pattern = {k: v for k, v in _random_pattern(rng, max_side=8).items() if v != 0}
        width = rng.randint(1, 14)
        height = rng.randint(1, 14)
        # export grows the declared box if the pattern does not fit.
        bw = max((x for x, _ in pattern), default=-1) + 1
        bh = max((y for _, y in pattern), default=-1) + 1
        parsed = parse_rle(export_rle(pattern, width=width, height=height, rule="JvN29"))
        assert parsed.width == max(width, bw)
        assert parsed.height == max(height, bh)
        assert parsed.pattern == pattern


def test_rle_multidigit_run_counts():
    parsed = parse_rle("x = 205, y = 1, rule = JvN29\n205A!")
    assert len(parsed.pattern) == 205
    assert set(parsed.pattern.values()) == {int(VNState.S)}
    assert parsed.pattern[(204, 0)] == int(VNState.S)


def test_rle_row_skip_counts():
    parsed = parse_rle("x = 1, y = 4, rule = JvN29\n3$A!")
    # Row 3 of 4 is the bottom row in top-first RLE -> Cartesian y = 0.
    assert parsed.pattern == {(0, 0): int(VNState.S)}


def test_rle_exclamation_ends_parsing():
    parsed = parse_rle("x = 3, y = 1, rule = JvN29\nA.B!")
    # '.' writes a dead cell which still advances the cursor.
    assert parsed.pattern == {(0, 0): int(VNState.S), (2, 0): int(VNState.S0)}


def test_rle_header_without_rule_and_comments():
    text = "#C a comment\n#N nice name\nx = 2, y = 1\nBA!\n"
    parsed = parse_rle(text)
    assert parsed.rule is None
    assert parsed.comments == ["C a comment", "N nice name"]
    assert parsed.pattern == {(0, 0): int(VNState.S0), (1, 0): int(VNState.S)}


def test_rle_rejects_invalid_tokens():
    for body in ("Z!", "a!", "p!", "qA!", "yA!"):  # unknown, lowercase, dangling block, out of 29-state range
        with pytest.raises(ValueError):
            parse_rle(f"x = 1, y = 1, rule = JvN29\n{body}")


def test_rle_empty_pattern_roundtrip():
    parsed = parse_rle(export_rle({}, rule="JvN29"))
    assert parsed.pattern == {}


def test_ascii_roundtrip_fuzz():
    rng = random.Random(99)
    for _ in range(80):
        pattern = _random_pattern(rng)
        width = max(x for x, _ in pattern) + 1 if pattern else 1
        height = max(y for _, y in pattern) + 1 if pattern else 1
        text = export_to_ascii(pattern, width, height)
        back = parse_ascii_pattern(text)
        assert back == {k: v for k, v in pattern.items() if v != 0}


def test_ascii_origin_offset():
    text = "Cr"
    parsed = parse_ascii_pattern(text, origin=(3, 2))
    assert parsed == {(3, 2): int(VNState.C00), (4, 2): int(VNState.OT_E)}


def test_charsets_cover_every_state_exactly_once():
    # Canonical export: exactly one char per state, all distinct.
    assert len(STATE_TO_CHAR) == 29
    assert set(STATE_TO_CHAR) == set(VNState)
    assert len(set(STATE_TO_CHAR.values())) == 29
    # Parse map must decode every canonical char back to the same state.
    for state, char in STATE_TO_CHAR.items():
        assert CHAR_TO_STATE[char] == state
    # Every state is reachable through the parse map.
    assert {v for v in CHAR_TO_STATE.values()} == set(VNState)
