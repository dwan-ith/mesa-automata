"""Parity tests: the Mesa agent-based model and the array engine must agree.

``VonNeumannCA`` steps cell-by-cell through Mesa agents;
``step_grid`` does a single vectorised gather. Two implementations of the same
dynamics in the same package must produce byte-identical trajectories.
"""

import numpy as np
import pytest

from vn_automata.model import VonNeumannCA
from vn_automata.patterns import make_pulse_loop, make_programmable_builder
from vn_automata.rules import step_grid
from vn_automata.states import is_excited

N = 29


def _pattern_from_grid(grid):
    return {
        (x, y): int(state)
        for y, x in np.ndindex(grid.shape[0], grid.shape[1])
        if (state := int(grid[y, x])) != 0
    }


CONFIGS = [
    (14, 9, False, "dense"),
    (16, 16, True, "dense"),
    (20, 12, False, "sparse"),
    (12, 18, True, "sparse"),
]


@pytest.mark.parametrize("width,height,torus,kind", CONFIGS)
def test_model_and_array_engine_stay_in_lockstep(width, height, torus, kind):
    rng = np.random.default_rng(hash((width, height, kind)) % 2 ** 32)
    if kind == "dense":
        grid = rng.integers(0, N, size=(height, width))
    else:
        grid = np.zeros((height, width), dtype=np.int64)
        palette = np.array([1, 9, 10, 13, 14, 17, 21, 25, 27, 28])
        mask = rng.random((height, width)) < 0.2
        grid[mask] = rng.choice(palette, size=int(mask.sum()))

    model = VonNeumannCA(width=width, height=height, torus=torus,
                         initial_pattern=_pattern_from_grid(grid), seed=3)
    array = grid.copy()
    for _ in range(25):
        model.step()
        array = step_grid(array, torus=torus)
        assert np.array_equal(model.to_numpy(), array)


def test_model_matches_array_engine_on_pulse_loop():
    pattern = make_pulse_loop(2, 2, 8, 6)
    model = VonNeumannCA(width=14, height=12, initial_pattern=pattern, seed=1)
    array = np.zeros((12, 14), dtype=np.int64)
    for (x, y), state in pattern.items():
        array[y, x] = int(state)

    for _ in range(60):  # two full laps of the ring
        model.step()
        array = step_grid(array)
        assert np.array_equal(model.to_numpy(), array)
        assert model.count_active_pulses() == 1


def test_model_matches_array_engine_on_construction():
    sequence = "1101"
    pattern = make_programmable_builder(sequence, 3, 5)
    model = VonNeumannCA(width=20, height=12, initial_pattern=pattern, seed=1)
    array = np.zeros((12, 20), dtype=np.int64)
    for (x, y), state in pattern.items():
        array[y, x] = int(state)

    for _ in range(len(sequence) + 6):
        model.step()
        array = step_grid(array)
        assert np.array_equal(model.to_numpy(), array)

    target = (3 + len(sequence), 5)
    assert model.get_cell(*target) == array[target[1], target[0]] != 0


def test_model_is_deterministic_for_a_fixed_seed():
    pattern = make_pulse_loop(2, 2, 8, 6)
    trajectories = []
    for _ in range(2):
        model = VonNeumannCA(width=14, height=12, initial_pattern=dict(pattern), seed=42)
        trace = []
        for _ in range(15):
            model.step()
            trace.append(model.to_numpy().copy())
        trajectories.append(trace)
    for a, b in zip(*trajectories):
        assert np.array_equal(a, b)


def test_metrics_reflect_grid_state():
    rng = np.random.default_rng(4)
    grid = rng.integers(0, N, size=(10, 10))
    model = VonNeumannCA(width=10, height=10, initial_pattern=_pattern_from_grid(grid), seed=5)
    excited = np.array([s for s in range(N) if is_excited(s)])
    assert model.count_active_pulses() == int(np.isin(grid, excited).sum())
    assert model.count_non_ground_cells() == int((grid != 0).sum())
    assert model.count_sensitized_cells() == int(((grid >= 1) & (grid <= 8)).sum())
    assert model.count_confluent_cells() == int(((grid >= 25) & (grid <= 28)).sum())


def test_numpy_roundtrip_and_pattern_loading():
    model = VonNeumannCA(width=6, height=5, seed=0)
    arr = np.arange(30, dtype=np.int32).reshape(5, 6) % 29
    model.from_numpy(arr)
    assert np.array_equal(model.to_numpy(), arr)

    pattern = {(3, 3): 13, (4, 4): 28}
    model.load_pattern(pattern, offset=(1, 0))
    assert model.get_cell(4, 3) == 13
    assert model.get_cell(5, 4) == 28


def test_set_cell_rejects_out_of_bounds():
    model = VonNeumannCA(width=5, height=5, seed=0)
    model.set_cell(5, 0, 7)   # ignored silently (documented behaviour)
    model.set_cell(0, -1, 7)
    assert model.get_cell(0, 0) == 0
    assert model.get_cell(2, 2) == 0
