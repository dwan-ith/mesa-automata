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


def test_model_torus_steps_without_crash():
    """Regression: torus models must wrap instead of raising IndexError."""
    model = VonNeumannCA(width=10, height=10, torus=True)
    model.step()
    assert model.step_count == 1
    # Pulse placed at edge must wrap sensitization to opposite edge
    model.set_cell(9, 5, VNState.OT_E_EX)
    model.step()
    assert model.get_cell(0, 5) == VNState.S


def test_model_grid_parity_with_step_grid():
    """Mesa model stepping must match step_grid kernel on the same pattern."""
    import numpy as np
    from vn_automata.rules import step_grid

    pattern = make_pulse_loop(5, 5, 4, 4)
    model = VonNeumannCA(width=20, height=20, initial_pattern=pattern)
    before = model.to_numpy()
    model.step()
    expected = step_grid(before, torus=False)
    assert (model.to_numpy() == expected).all()


def test_active_pulses_excludes_sensitized():
    """Sensitized intermediates must not count as active pulses."""
    model = VonNeumannCA(width=10, height=10)
    model.set_cell(5, 5, VNState.S)
    model.set_cell(4, 5, VNState.S01)
    model.set_cell(3, 5, VNState.OT_E_EX)
    assert model.count_active_pulses() == 1
