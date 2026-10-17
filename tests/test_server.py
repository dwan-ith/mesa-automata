"""Tests for the visualizer's simulation backend (no HTTP involved).

``SimulationState`` owns all the mutable state the REST API exposes, so its
validation and bookkeeping are tested directly.
"""

import pytest

from vn_automata.states import VNState
from vn_automata.visualizer.server import SimulationState


def test_snapshot_shape_and_metrics():
    sim = SimulationState("pulse_loop", 40, 30)
    snap = sim.get_snapshot()
    assert set(snap) == {"step", "width", "height", "pattern", "cells", "metrics"}
    assert set(snap["metrics"]) == {
        "active_pulses", "non_ground", "sensitized", "confluent",
    }
    assert snap["step"] == 0
    assert snap["cells"], "a pulse loop must contain non-ground cells"


def test_reset_expands_grid_to_fit_pattern():
    sim = SimulationState("construction_arm_1011", 10, 10)
    assert sim.width >= 16 and sim.height >= 17  # arm spans x..x+4 at y=15


def test_step_clamps_to_at_least_one():
    sim = SimulationState("pulse_loop", 30, 30)
    before = sim.get_snapshot()["step"]
    sim.step(0)
    assert sim.get_snapshot()["step"] == before + 1
    sim.step(3)
    assert sim.get_snapshot()["step"] == before + 4


def test_set_cell_validates_state_range():
    sim = SimulationState("pulse_loop", 30, 30)
    with pytest.raises(ValueError):
        sim.set_cell(0, 0, 29)
    with pytest.raises(ValueError):
        sim.set_cell(0, 0, -1)
    sim.set_cell(3, 3, int(VNState.ST_E))
    assert sim.model.get_cell(3, 3) == int(VNState.ST_E)


@pytest.mark.parametrize("bad", ["", "abc", "10a1", "1 0", "1" * 33])
def test_sequence_rejects_malformed_input(bad):
    sim = SimulationState("pulse_loop", 40, 30)
    with pytest.raises(ValueError):
        sim.load_sequence_arm(bad)


def test_sequence_arm_actually_constructs():
    sim = SimulationState("pulse_loop", 40, 30)
    sim.load_sequence_arm("1011")
    assert sim.pattern_key == "Custom Sequence: 1011"
    model = sim.model
    n = 4
    for _ in range(n + 6):
        model.step()
    y = 30 // 2
    assert model.get_cell(8 + n, y) == int(VNState.ST_E)


def test_unknown_pattern_reset_leaves_clean_ground():
    sim = SimulationState("no_such_pattern", 20, 15)
    snap = sim.get_snapshot()
    assert snap["cells"] == {}
    assert snap["metrics"]["non_ground"] == 0
