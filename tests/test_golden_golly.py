"""Golden tests validating the engine against the canonical Golly JvN29 rule.

Unlike the other test modules (which check self-consistency), these tests pin
the implementation to an *external* trusted artefact: the ``JvN29.rule`` table
distributed with Golly. Two things are checked independently:

1. the *generated* rule set equals the *parsed* reference rule set, exactly,
   over the entire 29^5 neighbourhood space; and
2. the rule compiler itself agrees with an obvious, slow, first-match
   interpreter (so the fast path cannot silently diverge).
"""

import numpy as np
import pytest

from vn_automata.golly import DEFAULT, parse_colors, parse_rule_file
from vn_automata.patterns import make_construction_arm_1011
from vn_automata.rules import automaton, build_rule_set, cell_next_state, step_grid
from vn_automata.states import STATE_INFO, VNState

N = 29


def _numba_available():
    try:
        import numba  # noqa: F401

        return True
    except ImportError:
        return False


def test_golly_reference_loads():
    """The bundled rule file must parse into a complete 29-state rule set."""
    variables, rules = parse_rule_file()
    colors = parse_colors()
    assert len(rules) > 90
    assert len(colors) == 29
    assert set(colors) == set(range(29))


def test_our_palette_matches_golly_rule():
    """states.py colours must match the canonical Golly palette exactly."""
    colors = parse_colors()
    for state in VNState:
        assert STATE_INFO[state].rgb == colors[int(state)], (
            f"{state.name}: states.py {STATE_INFO[state].rgb} != Golly {colors[int(state)]}"
        )


def test_generated_rule_set_is_compact():
    """The generated rule set should stay small and comprehensible."""
    assert build_rule_set().n_states == N
    # Sanity: clearly far smaller than the 29^5 = 20M possible neighbourhoods.
    assert len(build_rule_set()) < 250


def test_generated_table_equals_reference_table():
    """The generated table must equal the Golly rule file compiled identically."""
    generated = automaton().table
    reference = DEFAULT.compiled_table()
    mismatches = np.nonzero(generated != reference)[0]
    assert mismatches.size == 0, (
        f"{mismatches.size} / {N**5} neighbourhoods differ from the Golly reference; "
        f"first: {int(mismatches[0])}"
    )


def test_cell_next_state_matches_reference_interpreter():
    """Random sample: table lookup must match the independent interpreter."""
    rng = np.random.default_rng(1234)
    sample = rng.integers(0, N, size=(200_000, 5))
    for c, n, e, s, w in sample:
        assert cell_next_state(int(c), int(n), int(e), int(s), int(w)) == \
            DEFAULT.next_state(int(c), int(n), int(e), int(s), int(w))


@pytest.mark.skipif(not _numba_available(), reason="numba not installed")
def test_full_state_space_matches_reference_interpreter():
    """Exhaustive: all 29^5 neighbourhoods vs the independent interpreter."""
    from numba import njit

    masks, news = DEFAULT.masks()
    mc = np.ascontiguousarray(masks[:, 0]); mn = np.ascontiguousarray(masks[:, 1])
    me = np.ascontiguousarray(masks[:, 2]); ms = np.ascontiguousarray(masks[:, 3])
    mw = np.ascontiguousarray(masks[:, 4]); new_states = np.ascontiguousarray(news)
    table = np.ascontiguousarray(automaton().table)

    @njit
    def kernel(c, n, e, s, w, mc, mn, me, ms, mw, ns, table):
        out = np.empty_like(c)
        nr = ns.shape[0]
        for i in range(c.shape[0]):
            cc, nn, ee, ss, ww = c[i], n[i], e[i], s[i], w[i]
            res = cc
            for r in range(nr):
                if ((mc[r] >> cc) & 1) and ((mn[r] >> nn) & 1) and ((me[r] >> ee) & 1) \
                        and ((ms[r] >> ss) & 1) and ((mw[r] >> ww) & 1):
                    res = ns[r]
                    break
            idx = (((cc * 29 + nn) * 29 + ee) * 29 + ss) * 29 + ww
            out[i] = table[idx] - res  # 0 iff identical
        return out

    mismatch = 0
    for center in range(N):
        idx = np.arange(N ** 4, dtype=np.int64)
        w = idx % N; s = (idx // N) % N; e = (idx // N ** 2) % N; n = (idx // N ** 3) % N
        c = np.full(idx.shape[0], center, dtype=np.int64)
        mismatch += int(np.count_nonzero(
            kernel(c, n, e, s, w, mc, mn, me, ms, mw, new_states, table)))
    assert mismatch == 0, f"{mismatch} neighbourhoods differ from the reference"


@pytest.mark.parametrize("torus", [False, True])
def test_grid_evolution_matches_golly(torus):
    """Multi-step evolution on random grids must match the reference exactly."""
    rng = np.random.default_rng(7)
    for _ in range(6):
        grid = rng.integers(0, N, size=(9, 11))
        ours = grid.copy()
        theirs = grid.copy()
        for _ in range(8):
            ours = step_grid(ours, torus=torus)
            theirs = DEFAULT.step_grid(theirs, torus=torus)
            assert np.array_equal(ours, theirs)


def test_construction_arm_1011_matches_golly():
    """The 1011 arm extension must evolve identically in both engines."""
    width = height = 35
    grid = np.zeros((height, width), dtype=np.int64)
    for (x, y), state in make_construction_arm_1011(10, 15).items():
        grid[y, x] = int(state)
    ours = grid.copy()
    theirs = grid.copy()
    for _ in range(12):
        ours = step_grid(ours, torus=False)
        theirs = DEFAULT.step_grid(theirs, torus=False)
        assert np.array_equal(ours, theirs)
    # And it must actually construct an east special-transmission cell.
    assert ours[15, 15] == VNState.ST_E
