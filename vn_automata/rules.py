"""Deterministic, synchronous transition rules for von Neumann's 29-state CA.

Implements the exact 5-cell neighborhood rules:
- Ground excitation into Sensitized tree (S)
- Sensitized bifurcation tree (S -> S0/S1 -> S00/S01/S10/S11 -> target quiescent states)
- Ordinary Transmission States (OTS): directional pulse transfer, confluent input, STS destruction
- Special Transmission States (STS): directional pulse transfer, confluent input, OTS destruction
- Confluent States (C): 2-bit delay buffer, OTS AND-gate input, multi-line fan-out, STS destruction
"""

from typing import Tuple
import numpy as np
from .states import VNState

try:
    from numba import njit as _njit
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover - optional acceleration
    _HAS_NUMBA = False

    def _njit(*a, **k):
        def wrap(f):
            return f

        return wrap if a and callable(a[0]) else wrap


# Directions relative to center (x, y):
# north: (x, y+1)
# east:  (x+1, y)
# south: (x, y-1)
# west:  (x-1, y)

# Excited transmission states pointing towards center:
# A west neighbor (x-1, y) points towards center if its direction is EAST.
# A south neighbor (x, y-1) points towards center if its direction is NORTH.
# A east neighbor (x+1, y) points towards center if its direction is WEST.
# A north neighbor (x, y+1) points towards center if its direction is SOUTH.

OTS_EAST_EXC = VNState.OT_E_EX  # 13
OTS_NORTH_EXC = VNState.OT_N_EX  # 14
OTS_WEST_EXC = VNState.OT_W_EX  # 15
OTS_SOUTH_EXC = VNState.OT_S_EX  # 16

STS_EAST_EXC = VNState.ST_E_EX  # 21
STS_NORTH_EXC = VNState.ST_N_EX  # 22
STS_WEST_EXC = VNState.ST_W_EX  # 23
STS_SOUTH_EXC = VNState.ST_S_EX  # 24

OTS_EAST_ALL = (VNState.OT_E, VNState.OT_E_EX)  # 9, 13
OTS_NORTH_ALL = (VNState.OT_N, VNState.OT_N_EX)  # 10, 14
OTS_WEST_ALL = (VNState.OT_W, VNState.OT_W_EX)  # 11, 15
OTS_SOUTH_ALL = (VNState.OT_S, VNState.OT_S_EX)  # 12, 16

CONFLUENT_EMITTING = (VNState.C10, VNState.C11)  # 26, 28


def has_sts_pointing_excited(north: int, east: int, south: int, west: int) -> bool:
    """Check if any adjacent STS cell points to center and is excited (destruction pulse)."""
    return (
        west == STS_EAST_EXC
        or south == STS_NORTH_EXC
        or east == STS_WEST_EXC
        or north == STS_SOUTH_EXC
    )


def has_ots_pointing_excited(north: int, east: int, south: int, west: int) -> bool:
    """Check if any adjacent OTS cell points to center and is excited (antagonistic destruction pulse)."""
    return (
        west == OTS_EAST_EXC
        or south == OTS_NORTH_EXC
        or east == OTS_WEST_EXC
        or north == OTS_SOUTH_EXC
    )


def has_any_pointing_excited(north: int, east: int, south: int, west: int) -> bool:
    """Check if any adjacent OTS or STS cell points to center and is excited (sensitization input)."""
    return (
        west in (OTS_EAST_EXC, STS_EAST_EXC)
        or south in (OTS_NORTH_EXC, STS_NORTH_EXC)
        or east in (OTS_WEST_EXC, STS_WEST_EXC)
        or north in (OTS_SOUTH_EXC, STS_SOUTH_EXC)
    )


def cell_next_state(center: int, north: int, east: int, south: int, west: int) -> int:
    """Compute the synchronous next state for a single cell given its 4 orthogonal neighbors."""
    c = center

    # 1. GROUND STATE (U = 0)
    if c == VNState.U:
        if has_any_pointing_excited(north, east, south, west):
            return VNState.S
        return VNState.U

    # 2. SENSITIZED STATES (1..8)
    if 1 <= c <= 8:
        # Input bit is 1 if any adjacent pointing transmission cell is excited
        inp = 1 if has_any_pointing_excited(north, east, south, west) else 0

        if c == VNState.S:
            return VNState.S1 if inp else VNState.S0
        elif c == VNState.S0:
            return VNState.S01 if inp else VNState.S00
        elif c == VNState.S1:
            return VNState.S11 if inp else VNState.S10
        elif c == VNState.S00:
            return VNState.OT_W if inp else VNState.S000
        elif c == VNState.S01:
            return VNState.ST_E if inp else VNState.OT_S
        elif c == VNState.S10:
            return VNState.ST_W if inp else VNState.ST_N
        elif c == VNState.S11:
            return VNState.C00 if inp else VNState.ST_S
        elif c == VNState.S000:
            return VNState.OT_N if inp else VNState.OT_E

    # 3. ORDINARY TRANSMISSION STATES (9..16)
    if 9 <= c <= 16:
        # Destruction: STS excited pulse destroys OTS
        if has_sts_pointing_excited(north, east, south, west):
            return VNState.U

        # Determine center direction & base quiescent state
        if c in (VNState.OT_E, VNState.OT_E_EX):
            base_qui = VNState.OT_E
            direction = "EAST"
        elif c in (VNState.OT_N, VNState.OT_N_EX):
            base_qui = VNState.OT_N
            direction = "NORTH"
        elif c in (VNState.OT_W, VNState.OT_W_EX):
            base_qui = VNState.OT_W
            direction = "WEST"
        else:
            base_qui = VNState.OT_S
            direction = "SOUTH"

        # Check for incoming excited ordinary transmission or confluent emission
        # (excluding the neighbor center is pointing towards)
        excited = False

        # From West (neighbor pointing East)
        if direction != "WEST":
            if west == OTS_EAST_EXC or (west in CONFLUENT_EMITTING):
                excited = True
        # From South (neighbor pointing North)
        if not excited and direction != "SOUTH":
            if south == OTS_NORTH_EXC or (south in CONFLUENT_EMITTING):
                excited = True
        # From East (neighbor pointing West)
        if not excited and direction != "EAST":
            if east == OTS_WEST_EXC or (east in CONFLUENT_EMITTING):
                excited = True
        # From North (neighbor pointing South)
        if not excited and direction != "NORTH":
            if north == OTS_SOUTH_EXC or (north in CONFLUENT_EMITTING):
                excited = True

        return (base_qui + 4) if excited else base_qui

    # 4. SPECIAL TRANSMISSION STATES (17..24)
    if 17 <= c <= 24:
        # Destruction: OTS excited pulse destroys STS
        if has_ots_pointing_excited(north, east, south, west):
            return VNState.U

        # Determine center direction & base quiescent state
        if c in (VNState.ST_E, VNState.ST_E_EX):
            base_qui = VNState.ST_E
            direction = "EAST"
        elif c in (VNState.ST_N, VNState.ST_N_EX):
            base_qui = VNState.ST_N
            direction = "NORTH"
        elif c in (VNState.ST_W, VNState.ST_W_EX):
            base_qui = VNState.ST_W
            direction = "WEST"
        else:
            base_qui = VNState.ST_S
            direction = "SOUTH"

        # Check for incoming excited special transmission or confluent emission
        # (excluding the neighbor center is pointing towards)
        excited = False

        if direction != "WEST":
            if west == STS_EAST_EXC or (west in CONFLUENT_EMITTING):
                excited = True
        if not excited and direction != "SOUTH":
            if south == STS_NORTH_EXC or (south in CONFLUENT_EMITTING):
                excited = True
        if not excited and direction != "EAST":
            if east == STS_WEST_EXC or (east in CONFLUENT_EMITTING):
                excited = True
        if not excited and direction != "NORTH":
            if north == STS_SOUTH_EXC or (north in CONFLUENT_EMITTING):
                excited = True

        return (base_qui + 4) if excited else base_qui

    # 5. CONFLUENT STATES (25..28)
    if 25 <= c <= 28:
        # Destruction: STS excited pulse destroys Confluent
        if has_sts_pointing_excited(north, east, south, west):
            return VNState.U

        # Confluent takes inputs from OTS cells pointing towards it
        # and applies AND logic across all feeding lines.
        input_lines = 0
        excited_lines = 0

        # West neighbor pointing East into C
        if west in OTS_EAST_ALL:
            input_lines += 1
            if west == OTS_EAST_EXC:
                excited_lines += 1

        # South neighbor pointing North into C
        if south in OTS_NORTH_ALL:
            input_lines += 1
            if south == OTS_NORTH_EXC:
                excited_lines += 1

        # East neighbor pointing West into C
        if east in OTS_WEST_ALL:
            input_lines += 1
            if east == OTS_WEST_EXC:
                excited_lines += 1

        # North neighbor pointing South into C
        if north in OTS_SOUTH_ALL:
            input_lines += 1
            if north == OTS_SOUTH_EXC:
                excited_lines += 1

        # Input bit: 1 if at least one input line exists and ALL feeding lines are excited
        inp = 1 if (input_lines > 0 and excited_lines == input_lines) else 0

        # 2-bit delay buffer update:
        # State:
        # C00 (25): now=0, next=0
        # C10 (26): now=1, next=0
        # C01 (27): now=0, next=1
        # C11 (28): now=1, next=1
        if c in (VNState.C00, VNState.C10):
            # next bit from t becomes now bit (0)
            return VNState.C01 if inp else VNState.C00
        else:
            # next bit from t was 1, becomes now bit (1)
            return VNState.C11 if inp else VNState.C10

    return VNState.U


def step_grid(grid: np.ndarray, torus: bool = False) -> np.ndarray:
    """Synchronously advance a 2D NumPy array of cell states by one clock tick.
    
    Parameters:
        grid: 2D integer array with shape (height, width)
        torus: whether to wrap boundaries (default False, infinite/zero-padded universe)
    
    Returns:
        New 2D integer array of states at time t+1.
    """
    height, width = grid.shape
    new_grid = np.zeros_like(grid)

    if torus:
        # np.roll(shift=-1) moves row y+1 into row y, i.e. the north
        # neighbor (x, y+1) in Mesa Cartesian coordinates where arr[y, x].
        north = np.roll(grid, shift=-1, axis=0)
        south = np.roll(grid, shift=1, axis=0)
        east = np.roll(grid, shift=-1, axis=1)
        west = np.roll(grid, shift=1, axis=1)
    else:
        # Zero-pad boundaries (ground state outside).
        # padded[y + 1, x + 1] == grid[y, x], so:
        #   north (x, y+1) -> padded row y+2 -> padded[2:, :]
        #   south (x, y-1) -> padded row y   -> padded[:-2, :]
        padded = np.pad(grid, 1, mode="constant", constant_values=VNState.U)
        north = padded[2:, 1:-1]
        south = padded[:-2, 1:-1]
        west = padded[1:-1, :-2]
        east = padded[1:-1, 2:]

    # Apply transitions: Numba-compiled kernel when available (same rules as
    # cell_next_state, plain ints so njit can compile it), else Python loops.
    if _HAS_NUMBA:
        _step_kernel_nb(grid, north, east, south, west, new_grid)
    else:
        for y in range(height):
            for x in range(width):
                new_grid[y, x] = cell_next_state(
                    grid[y, x],
                    north[y, x],
                    east[y, x],
                    south[y, x],
                    west[y, x],
                )

    return new_grid


@_njit(cache=True)
def _step_kernel_nb(grid, north, east, south, west, out):
    """Numba fast path mirroring cell_next_state with int state codes."""
    h, w = grid.shape
    for y in range(h):
        for x in range(w):
            c = grid[y, x]
            n = north[y, x]
            e = east[y, x]
            s = south[y, x]
            ww = west[y, x]

            # 1. Ground
            if c == 0:
                out[y, x] = 1 if (
                    ww == 13 or ww == 21
                    or s == 14 or s == 22
                    or e == 15 or e == 23
                    or n == 16 or n == 24
                ) else 0
                continue

            # 2. Sensitized 1..8
            if 1 <= c <= 8:
                inp = 1 if (
                    ww == 13 or ww == 21
                    or s == 14 or s == 22
                    or e == 15 or e == 23
                    or n == 16 or n == 24
                ) else 0
                if c == 1:
                    out[y, x] = 3 if inp else 2
                elif c == 2:
                    out[y, x] = 5 if inp else 4
                elif c == 3:
                    out[y, x] = 7 if inp else 6
                elif c == 4:
                    out[y, x] = 11 if inp else 8
                elif c == 5:
                    out[y, x] = 17 if inp else 12
                elif c == 6:
                    out[y, x] = 19 if inp else 18
                elif c == 7:
                    out[y, x] = 25 if inp else 20
                else:
                    out[y, x] = 10 if inp else 9
                continue

            # 3. Ordinary transmission 9..16
            if 9 <= c <= 16:
                # STS destruction
                if ww == 21 or s == 22 or e == 23 or n == 24:
                    out[y, x] = 0
                    continue
                if c == 9 or c == 13:
                    base = 9
                    direction = 1
                elif c == 10 or c == 14:
                    base = 10
                    direction = 2
                elif c == 11 or c == 15:
                    base = 11
                    direction = 3
                else:
                    base = 12
                    direction = 4
                excited = False
                if direction != 3 and (ww == 13 or ww == 26 or ww == 28):
                    excited = True
                if not excited and direction != 4 and (s == 14 or s == 26 or s == 28):
                    excited = True
                if not excited and direction != 1 and (e == 15 or e == 26 or e == 28):
                    excited = True
                if not excited and direction != 2 and (n == 16 or n == 26 or n == 28):
                    excited = True
                out[y, x] = base + 4 if excited else base
                continue

            # 4. Special transmission 17..24
            if 17 <= c <= 24:
                if ww == 13 or s == 14 or e == 15 or n == 16:
                    out[y, x] = 0
                    continue
                if c == 17 or c == 21:
                    base = 17
                    direction = 1
                elif c == 18 or c == 22:
                    base = 18
                    direction = 2
                elif c == 19 or c == 23:
                    base = 19
                    direction = 3
                else:
                    base = 20
                    direction = 4
                excited = False
                if direction != 3 and (ww == 21 or ww == 26 or ww == 28):
                    excited = True
                if not excited and direction != 4 and (s == 22 or s == 26 or s == 28):
                    excited = True
                if not excited and direction != 1 and (e == 23 or e == 26 or e == 28):
                    excited = True
                if not excited and direction != 2 and (n == 24 or n == 26 or n == 28):
                    excited = True
                out[y, x] = base + 4 if excited else base
                continue

            # 5. Confluent 25..28
            if 25 <= c <= 28:
                if ww == 21 or s == 22 or e == 23 or n == 24:
                    out[y, x] = 0
                    continue
                nlines = 0
                elines = 0
                if ww == 9 or ww == 13:
                    nlines += 1
                    if ww == 13:
                        elines += 1
                if s == 10 or s == 14:
                    nlines += 1
                    if s == 14:
                        elines += 1
                if e == 11 or e == 15:
                    nlines += 1
                    if e == 15:
                        elines += 1
                if n == 12 or n == 16:
                    nlines += 1
                    if n == 16:
                        elines += 1
                inp = 1 if (nlines > 0 and elines == nlines) else 0
                if c == 25 or c == 26:
                    out[y, x] = 27 if inp else 25
                else:
                    out[y, x] = 28 if inp else 26
                continue

            out[y, x] = 0
