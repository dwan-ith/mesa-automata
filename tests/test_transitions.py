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
