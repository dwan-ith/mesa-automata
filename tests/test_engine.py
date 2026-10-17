"""Rigorous tests for the generic rule engine itself.

``vn_automata.engine`` must be correct for *any* rule set, not just von
Neumann's: the JvN golden tests would be meaningless if the compiler had a bug
that happened to cancel out. These tests pin the engine's contract:

* rules are evaluated in insertion order (first match wins);
* unmatched neighbourhoods leave the cell unchanged;
* compiled-table stepping agrees with per-cell evaluation;
* neighbour extraction matches the documented coordinate convention.
"""

import numpy as np
import pytest

from vn_automata.engine import RuleSet, TableAutomaton, compile_rules

ALL = [0, 1]


def test_first_match_wins():
    """The earliest inserted rule must take priority over later overlaps."""
    rs = RuleSet(3)
    rs.add(center=[0], north=[0, 1, 2], east=[0, 1, 2], south=[0, 1, 2], west=[0, 1, 2], new=1)
    rs.add(center=[0, 1], north=[0, 1, 2], east=[0, 1, 2], south=[0, 1, 2], west=[0, 1, 2], new=2)
    table = rs.compile().reshape(3, 3, 3, 3, 3)
    assert int(table[0, 0, 0, 0, 0]) == 1  # first rule wins
    assert int(table[1, 0, 0, 0, 0]) == 2  # only second rule matches
    assert int(table[2, 0, 0, 0, 0]) == 2  # nothing matches -> unchanged


def test_unmatched_neighbourhood_is_unchanged():
    rs = RuleSet(4)
    rs.add(center=[1], north=[2], east=[3], south=[0], west=[1], new=0)
    ta = TableAutomaton.from_ruleset(rs)
    for c in range(4):
        for n in range(4):
            for e in range(4):
                for s in range(4):
                    for w in range(4):
                        matched = (c, n, e, s, w) == (1, 2, 3, 0, 1)
                        expected = 0 if matched else c
                        assert ta.next_state(c, n, e, s, w) == expected


def test_compiler_defaults_to_identity_when_no_rules():
    """A rule set with zero rules needs no entry to change — table is identity."""
    table = RuleSet(29).compile()
    assert np.array_equal(table, np.repeat(np.arange(29, dtype=np.uint8), 29 ** 4))


def test_domains_are_deduplicated_and_order_insensitive():
    a = RuleSet(2)
    a.add(center=[1, 1, 0, 0], north=[0], east=[0], south=[0], west=[0], new=1)
    b = RuleSet(2)
    b.add(center=[0, 1], north=[0], east=[0], south=[0], west=[0], new=1)
    assert np.array_equal(a.compile(), b.compile())


def test_rule_validation_errors():
    with pytest.raises(ValueError):
        RuleSet(0)
    with pytest.raises(ValueError):
        RuleSet(257)
    rs = RuleSet(2)
    with pytest.raises(ValueError):
        rs.add([0], [0], [0], [0], [0], new=2)  # new out of range
    with pytest.raises(ValueError):
        rs.add([2], [0], [0], [0], [0], new=1)  # domain value out of range
    with pytest.raises(ValueError):
        rs.add([], [0], [0], [0], [0], new=1)  # empty domain


def test_compile_rules_convenience():
    ruleset = RuleSet(2)
    ruleset.add([0], [0], [0], [0], [0], new=1)
    table = compile_rules(2, [(([0], [0], [0], [0], [0]), 1)])
    assert np.array_equal(ruleset.compile(), table)
    ta = TableAutomaton(2, table)
    # Cell in state 0 with all-ground neighbours takes 1...
    assert ta.next_state(0, 0, 0, 0, 0) == 1
    # ...and a cell in state 1 is untouched (unmatched -> unchanged).
    assert ta.next_state(1, 0, 0, 0, 0) == 1


def test_table_automaton_validation():
    with pytest.raises(ValueError):
        TableAutomaton(3, np.zeros(4, dtype=np.uint8))  # wrong size
    ta = TableAutomaton(2, np.arange(32, dtype=np.uint8))
    with pytest.raises(ValueError):
        ta.step(np.zeros(4, dtype=np.int64))  # not 2-D


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int16, np.int32, np.int64])
def test_step_preserves_integer_dtypes(dtype):
    ta = TableAutomaton(2, np.arange(32, dtype=np.uint8))
    out = ta.step(np.zeros((3, 3), dtype=dtype))
    assert out.dtype == dtype


def test_step_float_input_returns_int64():
    ta = TableAutomaton(2, np.arange(32, dtype=np.uint8))
    out = ta.step(np.zeros((3, 3), dtype=np.float64))
    assert out.dtype == np.int64


def _random_ruleset(rng, n, n_rules):
    rs = RuleSet(n)
    for _ in range(n_rules):
        domains = []
        for _ in range(5):
            size = int(rng.integers(1, n + 1))
            states = rng.choice(n, size=size, replace=False)
            domains.append([int(s) for s in states])
        rs.add(*domains, new=int(rng.integers(0, n)))
    return rs


def test_compiler_matches_naive_first_match_interpreter():
    """Fuzz: compiled tables must equal an obvious slow matcher, exactly."""
    rng = np.random.default_rng(2024)
    for trial in range(25):
        n = int(rng.integers(2, 6))
        rs = _random_ruleset(rng, n, n_rules=int(rng.integers(1, 12)))
        table = rs.compile().reshape((n,) * 5)
        for _ in range(40):
            c, n_, e, s, w = (int(x) for x in rng.integers(0, n, size=5))
            expected = c
            for domains, new in rs.rules:
                if (c in domains[0] and n_ in domains[1] and e in domains[2]
                        and s in domains[3] and w in domains[4]):
                    expected = new
                    break
            assert int(table[c, n_, e, s, w]) == expected, (trial, c, n_, e, s, w)


def test_signal_shift_rule_genericity():
    """A non-JvN rule ('cell becomes its west neighbour') must shift signals."""
    rs = RuleSet(2)
    for west in (0, 1):
        rs.add(center=ALL, north=ALL, east=ALL, south=ALL, west=[west], new=west)
    ta = TableAutomaton.from_ruleset(rs)

    grid = np.array([[0, 1, 0, 0, 0, 0]], dtype=np.int64)
    for k in range(1, 5):
        grid = ta.step(grid)
        assert int(grid[0, 1 + k]) == 1 and grid.sum() == 1
    # k=5: the signal walks off the east edge of a non-torus grid and dies.
    grid = ta.step(grid)
    assert grid.sum() == 0

    # With a torus the signal wraps around and returns after `width` steps.
    grid = np.array([[0, 1, 0, 0, 0, 0]], dtype=np.int64)
    for _ in range(6):
        grid = ta.step(grid, torus=True)
    assert np.array_equal(grid, np.array([[0, 1, 0, 0, 0, 0]]))


def _manual_neighbours(grid, torus):
    """Independent re-implementation of the documented neighbour convention."""
    h, w = grid.shape
    north = np.empty_like(grid)
    east = np.empty_like(grid)
    south = np.empty_like(grid)
    west = np.empty_like(grid)
    for y in range(h):
        for x in range(w):
            ny = (y + 1) % h if torus else (y + 1 if y + 1 < h else None)
            sx = (x + 1) % w if torus else (x + 1 if x + 1 < w else None)
            sy = (y - 1) % h if torus else (y - 1 if y - 1 >= 0 else None)
            wx = (x - 1) % w if torus else (x - 1 if x - 1 >= 0 else None)
            north[y, x] = grid[ny, x] if ny is not None else 0
            east[y, x] = grid[y, sx] if sx is not None else 0
            south[y, x] = grid[sy, x] if sy is not None else 0
            west[y, x] = grid[y, wx] if wx is not None else 0
    return north, east, south, west


@pytest.mark.parametrize("torus", [False, True])
def test_neighbour_arrays_match_manual_convention(torus):
    rng = np.random.default_rng(99)
    for _ in range(5):
        grid = rng.integers(0, 29, size=(6, 7))
        got = TableAutomaton(29, np.zeros(29 ** 5, dtype=np.uint8)).neighbour_arrays(grid, torus)
        expected = _manual_neighbours(grid, torus)
        for g, e in zip(got, expected):
            assert np.array_equal(g, e)


@pytest.mark.parametrize("torus", [False, True])
def test_grid_step_equals_per_cell_evaluation(torus):
    """Table stepping must equal per-cell lookups on the same neighbourhoods."""
    rng = np.random.default_rng(5)
    ta = TableAutomaton(29, (np.arange(29 ** 5, dtype=np.uint8) % 29))  # arbitrary table
    for _ in range(4):
        grid = rng.integers(0, 29, size=(5, 8))
        out = ta.step(grid, torus=torus)
        north, east, south, west = _manual_neighbours(grid, torus)
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                assert out[y, x] == ta.next_state(
                    int(grid[y, x]), int(north[y, x]), int(east[y, x]),
                    int(south[y, x]), int(west[y, x]),
                )


def test_boundary_cells_see_ground_off_grid():
    """Off-grid neighbours of a non-torus grid must read as ground (state 0)."""
    rs = RuleSet(2)
    rs.add(center=[1], north=[0], east=[0], south=[0], west=[0], new=0)
    ta = TableAutomaton.from_ruleset(rs)
    grid = np.array([[0, 0], [0, 1]], dtype=np.int64)  # only cell (1,1) alive
    out = ta.step(grid, torus=False)
    assert int(out[1, 1]) == 0  # east neighbour was off-grid ground -> rule fired
