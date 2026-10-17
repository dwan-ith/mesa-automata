"""Differential testing: our engine vs the Golly JvN29 rule file.

The reference evaluation here uses a *third* independent strategy — for every
rule from the parsed Golly file, a vectorised boolean match over the whole grid
applied in reverse file order (so the first rule in file order wins last).
This is neither the compiled lookup table nor the per-cell interpreter, so a
shared bug cannot hide. Neighbour extraction is also re-implemented inline.
"""

import numpy as np
import pytest

from vn_automata.golly import DEFAULT
from vn_automata.patterns import (
    make_construction_arm_1011,
    make_pulse_loop,
    make_programmable_builder,
)
from vn_automata.rules import step_grid
from vn_automata.states import VNState

N = 29


def _neighbours(grid, torus):
    g = np.asarray(grid, dtype=np.int64)
    if torus:
        return (np.roll(g, -1, 0), np.roll(g, -1, 1), np.roll(g, 1, 0), np.roll(g, 1, 1))
    p = np.pad(g, 1, mode="constant", constant_values=0)
    return (p[2:, 1:-1], p[1:-1, 2:], p[:-2, 1:-1], p[1:-1, :-2])


def ref_step(grid, torus=False):
    """Independent vectorised first-match evaluator for the Golly rule file."""
    g = np.asarray(grid, dtype=np.int64)
    north, east, south, west = _neighbours(g, torus)
    out = g.copy()
    for (mc, mn, me, ms, mw), new in reversed(list(zip(DEFAULT.masks()[0], DEFAULT.masks()[1]))):
        match = (
            (((mc >> g) & 1) & ((mn >> north) & 1) & ((me >> east) & 1)
             & ((ms >> south) & 1) & ((mw >> west) & 1))
        ).astype(bool)
        out = np.where(match, new, out)
    return out


def _dense_noise(rng, h, w):
    return rng.integers(0, N, size=(h, w))


def _sparse_circuit(rng, h, w, density=0.15):
    palette = np.array(
        [1, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
         21, 22, 23, 24, 25, 26, 27, 28], dtype=np.int64,
    )
    g = np.zeros((h, w), dtype=np.int64)
    mask = rng.random((h, w)) < density
    g[mask] = rng.choice(palette, size=int(mask.sum()))
    return g


@pytest.mark.parametrize("torus", [False, True])
@pytest.mark.parametrize("kind", ["dense", "sparse"])
def test_random_grid_evolution_matches_reference(torus, kind):
    rng = np.random.default_rng({"dense": 11, "sparse": 22}[kind] + int(torus))
    maker = _dense_noise if kind == "dense" else _sparse_circuit
    for _ in range(3):
        grid = maker(rng, 13, 17)
        ours = grid.copy()
        theirs = grid.copy()
        for _ in range(25):
            ours = step_grid(ours, torus=torus)
            theirs = ref_step(theirs, torus=torus)
            assert np.array_equal(ours, theirs)


def test_pulse_loop_long_run_matches_reference():
    grid = np.zeros((26, 26), dtype=np.int64)
    for (x, y), state in make_pulse_loop(4, 4, 12, 10).items():
        grid[y, x] = int(state)
    ours, theirs = grid.copy(), grid.copy()
    for _ in range(120):
        ours = step_grid(ours)
        theirs = ref_step(theirs)
        assert np.array_equal(ours, theirs)


def test_all_nine_construction_sequences_match_reference():
    for sequence in ("10000", "10001", "1001", "1010", "1011", "1100", "1101", "1110", "1111"):
        grid = np.zeros((10 + 10, 45), dtype=np.int64)
        for (x, y), state in make_programmable_builder(sequence, 5, 15).items():
            grid[y, x] = int(state)
        ours, theirs = grid.copy(), grid.copy()
        for _ in range(len(sequence) + 6):
            ours = step_grid(ours)
            theirs = ref_step(theirs)
        assert np.array_equal(ours, theirs), sequence
        target = (5 + len(sequence), 15)
        assert int(ours[target[1], target[0]]) != VNState.U


def test_non_torus_boundary_matches_reference_with_ground_padding():
    """Excited pulses driven into the wall must behave identically in both engines."""
    rng = np.random.default_rng(77)
    grid = _sparse_circuit(rng, 8, 8, density=0.5)
    grid[0, :] = VNState.OT_W_EX  # firing straight at the west... ensure boundary activity
    ours, theirs = grid.copy(), grid.copy()
    for _ in range(20):
        ours = step_grid(ours, torus=False)
        theirs = ref_step(theirs, torus=False)
        assert np.array_equal(ours, theirs)


def test_arm_1011_matches_reference_at_boundary():
    """The construction arm running right up to the grid edge vs the reference."""
    grid = np.zeros((20, 16), dtype=np.int64)
    for (x, y), state in make_construction_arm_1011(10, 15).items():
        grid[y, x] = int(state)
    ours, theirs = grid.copy(), grid.copy()
    for _ in range(30):
        ours = step_grid(ours, torus=False)
        theirs = ref_step(theirs, torus=False)
        assert np.array_equal(ours, theirs)
