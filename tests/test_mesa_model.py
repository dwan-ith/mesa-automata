"""Tests for Mesa 3.x integration, Agent lifecycle, and DataCollector."""

import pytest
from vn_automata.states import VNState
from vn_automata.model import VonNeumannCA
from vn_automata.patterns import make_pulse_loop, make_construction_arm_1011


def test_model_initialization():
    """Verify that Mesa grid and CellAgents are correctly initialized."""
    model = VonNeumannCA(width=20, height=20)
    assert model.width == 20
    assert model.height == 20
    assert len(model.agents) == 400
    assert model.step_count == 0
    assert model.count_non_ground_cells() == 0


def test_model_stepping_with_pattern():
    """Verify that Mesa model steps synchronously and advances pulses."""
    pattern = make_pulse_loop(5, 5, 4, 4)
    model = VonNeumannCA(width=20, height=20, initial_pattern=pattern)

    # Initial state should have exactly 1 active pulse
    assert model.count_active_pulses() == 1
    initial_pulse_pos = [a.pos for a in model.agents if a.is_excited][0]

    # Step the model
    model.step()
    assert model.step_count == 1
    assert model.count_active_pulses() == 1
    next_pulse_pos = [a.pos for a in model.agents if a.is_excited][0]

    # The pulse moved along the wire
    assert next_pulse_pos != initial_pulse_pos


def test_model_datacollector():
    """Verify Mesa DataCollector records CA metrics."""
    pattern = make_construction_arm_1011(5, 10)
    model = VonNeumannCA(width=25, height=25, initial_pattern=pattern)

    for _ in range(5):
        model.step()

    df = model.datacollector.get_model_vars_dataframe()
    assert len(df) == 6  # Step 0 + 5 steps
    assert "ActivePulses" in df.columns
    assert "NonGroundCells" in df.columns
    assert "SensitizedCells" in df.columns
