"""Tests for synchronous transition rules across neighborhoods."""

import numpy as np
import pytest
from vn_automata.states import VNState
from vn_automata.rules import cell_next_state, step_grid


def test_ground_state_stays_quiescent():
    """Ground state with no incoming excitation remains ground."""
    next_st = cell_next_state(VNState.U, VNState.U, VNState.U, VNState.U, VNState.U)
    assert next_st == VNState.U


def test_ground_state_sensitizes():
    """Ground state with incoming excited OTS from West becomes Sensitized (S)."""
    # West neighbor is OT_E_EX pointing East towards center
    next_st = cell_next_state(VNState.U, VNState.U, VNState.U, VNState.U, VNState.OT_E_EX)
    assert next_st == VNState.S

    # South neighbor is OT_N_EX pointing North towards center
    next_st = cell_next_state(VNState.U, VNState.U, VNState.U, VNState.OT_N_EX, VNState.U)
    assert next_st == VNState.S


def test_ordinary_wire_pulse_propagation():
    """Verify that an excited pulse moves one step East along an OTS wire."""
    # Wire layout along row: (x-1)=OT_E_EX, x=OT_E, (x+1)=OT_E
    # Cell at x receives input from west (OT_E_EX)
    next_st = cell_next_state(
        center=VNState.OT_E,
        north=VNState.U,
        east=VNState.OT_E,
        south=VNState.U,
        west=VNState.OT_E_EX,
    )
    assert next_st == VNState.OT_E_EX

    # Next cycle for the sending cell at x-1 (west neighbor is now quiet OTS):
    send_next = cell_next_state(
        center=VNState.OT_E_EX,
        north=VNState.U,
        east=VNState.OT_E,
        south=VNState.U,
        west=VNState.OT_E,
    )
    assert send_next == VNState.OT_E


def test_confluent_node_delay_and_emission():
    """Verify 2-cycle confluent delay and emission to transmission line."""
    # Step 1: Input line (OT_E_EX) feeds into C00 from West
    c_next = cell_next_state(
        center=VNState.C00,
        north=VNState.U,
        east=VNState.OT_E,
        south=VNState.U,
        west=VNState.OT_E_EX,
    )
    # Becomes C01 (quiescent now, excited next tick)
    assert c_next == VNState.C01

    # Step 2: Next tick with no further input
    c_next2 = cell_next_state(
        center=VNState.C01,
        north=VNState.U,
        east=VNState.OT_E,
        south=VNState.U,
        west=VNState.OT_E,
    )
    # Becomes C10 (excited now, emitting pulse)
    assert c_next2 == VNState.C10

    # Step 3: Adjacent transmission wire to the East receives emission from C10
    wire_next = cell_next_state(
        center=VNState.OT_E,
        north=VNState.U,
        east=VNState.OT_E,
        south=VNState.U,
        west=VNState.C10,  # Emitting confluent node to the West
    )
    assert wire_next == VNState.OT_E_EX


def test_antagonistic_destruction_sts_erases_ots():
    """Special transmission excited pulse destroys an adjacent ordinary transmission cell."""
    # Center is OTS pointing North. West neighbor is STS pointing East with excited pulse.
    next_st = cell_next_state(
        center=VNState.OT_N,
        north=VNState.OT_N,
        east=VNState.U,
        south=VNState.U,
        west=VNState.ST_E_EX,
    )
    assert next_st == VNState.U


def test_antagonistic_destruction_ots_erases_sts():
    """Ordinary transmission excited pulse destroys an adjacent special transmission cell."""
    # Center is STS pointing North. West neighbor is OTS pointing East with excited pulse.
    next_st = cell_next_state(
        center=VNState.ST_N,
        north=VNState.ST_N,
        east=VNState.U,
        south=VNState.U,
        west=VNState.OT_E_EX,
    )
    assert next_st == VNState.U


def test_antagonistic_destruction_sts_erases_confluent():
    """Special transmission excited pulse destroys a confluent cell."""
    next_st = cell_next_state(
        center=VNState.C00,
        north=VNState.U,
        east=VNState.U,
        south=VNState.U,
        west=VNState.ST_E_EX,
    )
    assert next_st == VNState.U


def test_north_south_wire_propagation():
    """Regression: N/S wires must propagate (catches step_grid N/S swap)."""
    # North-pointing wire fed from the south
    assert (
        cell_next_state(VNState.OT_N, VNState.U, VNState.U, VNState.OT_N_EX, VNState.U)
        == VNState.OT_N_EX
    )
    # South-pointing wire fed from the north
    assert (
        cell_next_state(VNState.OT_S, VNState.OT_S_EX, VNState.U, VNState.U, VNState.U)
        == VNState.OT_S_EX
    )


def test_step_grid_matches_single_cell_rule_all_directions():
    """step_grid must agree with cell_next_state for E/W/N/S pulses."""
    # Vertical wire: pulse at south (row 1) feeding center (row 2)
    g = np.full((5, 5), VNState.U, dtype=np.int32)
    g[1, 2] = VNState.OT_N_EX
    g[2, 2] = VNState.OT_N
    g2 = step_grid(g, torus=False)
    assert g2[2, 2] == VNState.OT_N_EX

    # Horizontal wire: pulse at west feeding center
    g = np.full((5, 5), VNState.U, dtype=np.int32)
    g[2, 1] = VNState.OT_E_EX
    g[2, 2] = VNState.OT_E
    g2 = step_grid(g, torus=False)
    assert g2[2, 2] == VNState.OT_E_EX


def test_step_grid_torus_wraps():
    """Torus mode must wrap pulses across edges."""
    g = np.full((4, 4), VNState.U, dtype=np.int32)
    # Pulse at x=3 (east edge) pointing east wraps to x=0 ground
    g[2, 3] = VNState.OT_E_EX
    g2 = step_grid(g, torus=True)
    # Ground at (2, 0) has a wrapped west neighbor pointing east -> sensitized
    assert g2[2, 0] == VNState.S


def test_step_kernel_matches_reference_on_random_grids():
    """Accelerated kernel must match cell_next_state on randomized states."""
    rng = np.random.default_rng(42)
    for torus in (False, True):
        for _ in range(3):
            g = rng.integers(0, 29, size=(10, 12)).astype(np.int32)
            got = step_grid(g, torus=torus)
            # Reference: per-cell Python evaluation with matching neighborhoods
            if torus:
                n = np.roll(g, -1, axis=0)
                s = np.roll(g, 1, axis=0)
                e = np.roll(g, -1, axis=1)
                w = np.roll(g, 1, axis=1)
            else:
                p = np.pad(g, 1, constant_values=VNState.U)
                n, s, w, e = p[2:, 1:-1], p[:-2, 1:-1], p[1:-1, :-2], p[1:-1, 2:]
            for y in range(g.shape[0]):
                for x in range(g.shape[1]):
                    assert got[y, x] == cell_next_state(
                        int(g[y, x]), int(n[y, x]), int(e[y, x]),
                        int(s[y, x]), int(w[y, x]),
                    )


def test_pattern_validation():
    """Invalid builder inputs must raise, not silently produce garbage."""
    import pytest
    from vn_automata.patterns import make_programmable_builder, make_pulse_loop

    with pytest.raises(ValueError):
        make_programmable_builder("")
    with pytest.raises(ValueError):
        make_programmable_builder("10a01")
    with pytest.raises(ValueError):
        make_programmable_builder("1" * 33)
    with pytest.raises(ValueError):
        make_pulse_loop(0, 0, 2, 2)
