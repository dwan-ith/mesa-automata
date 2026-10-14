"""Pattern loaders and serialization for Von Neumann 29-State CA."""

from typing import Dict, List, Tuple
from .states import VNState, STATE_INFO


CHAR_TO_STATE: Dict[str, int] = {
    ".": VNState.U,
    " ": VNState.U,
    "0": VNState.U,
    "S": VNState.S,
    "r": VNState.OT_E,
    "u": VNState.OT_N,
    "l": VNState.OT_W,
    "d": VNState.OT_S,
    "R": VNState.OT_E_EX,
    "U": VNState.OT_N_EX,
    "L": VNState.OT_W_EX,
    "D": VNState.OT_S_EX,
    "e": VNState.ST_E,
    "n": VNState.ST_N,
    "w": VNState.ST_W,
    "s": VNState.ST_S,
    "E": VNState.ST_E_EX,
    "N": VNState.ST_N_EX,
    "W": VNState.ST_W_EX,
    "V": VNState.ST_S_EX,
    "C": VNState.C00,
    "*": VNState.C10,
    "+": VNState.C01,
    "#": VNState.C11,
}


def parse_ascii_pattern(text: str, origin: Tuple[int, int] = (0, 0)) -> Dict[Tuple[int, int], int]:
    """Parse a multiline string using compact ASCII representation into cell dictionary.
    
    Coordinate convention: lines top-to-bottom map to decreasing y (Cartesian) or row index.
    Here row 0 is top (y = lines - 1), bottom is y = 0.
    """
    pattern: Dict[Tuple[int, int], int] = {}
    lines = [line.rstrip() for line in text.strip().splitlines() if line and not line.startswith("//")]
    num_rows = len(lines)
    ox, oy = origin

    for row_idx, line in enumerate(lines):
        y = oy + (num_rows - 1 - row_idx)
        for col_idx, char in enumerate(line):
            x = ox + col_idx
            if char in CHAR_TO_STATE:
                state = CHAR_TO_STATE[char]
                if state != VNState.U:
                    pattern[(x, y)] = state

    return pattern


def export_to_ascii(pattern: Dict[Tuple[int, int], int], width: int, height: int) -> str:
    """Render a pattern dictionary as an ASCII diagram."""
    state_to_char = {v: k for k, v in CHAR_TO_STATE.items()}
    grid = [["." for _ in range(width)] for _ in range(height)]

    for (x, y), st in pattern.items():
        if 0 <= x < width and 0 <= y < height:
            row = height - 1 - y
            grid[row][x] = state_to_char.get(st, "?")

    return "\n".join("".join(row) for row in grid)
