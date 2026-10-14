"""Pattern loaders and serialization for Von Neumann 29-State CA."""

from typing import Dict, Tuple
from .states import VNState


CHAR_TO_STATE: Dict[str, int] = {
    ".": VNState.U,
    " ": VNState.U,
    "0": VNState.U,
    # Sensitized intermediates (full tree so tapes/construction can be saved)
    "S": VNState.S,
    "a": VNState.S0,
    "b": VNState.S1,
    "c": VNState.S00,
    "o": VNState.S01,
    "f": VNState.S10,
    "g": VNState.S11,
    "h": VNState.S000,
    # Ordinary transmission (quiescent lower / excited upper)
    "r": VNState.OT_E,
    "u": VNState.OT_N,
    "l": VNState.OT_W,
    "d": VNState.OT_S,
    "R": VNState.OT_E_EX,
    "U": VNState.OT_N_EX,
    "L": VNState.OT_W_EX,
    "D": VNState.OT_S_EX,
    # Special transmission
    "e": VNState.ST_E,
    "n": VNState.ST_N,
    "w": VNState.ST_W,
    "s": VNState.ST_S,
    "E": VNState.ST_E_EX,
    "N": VNState.ST_N_EX,
    "W": VNState.ST_W_EX,
    "V": VNState.ST_S_EX,
    # Confluent
    "C": VNState.C00,
    "*": VNState.C10,
    "+": VNState.C01,
    "#": VNState.C11,
}

# Canonical export: exactly one char per state (aliases map to ground ".").
STATE_TO_CHAR: Dict[int, str] = {
    VNState.U: ".",
    VNState.S: "S",
    VNState.S0: "a",
    VNState.S1: "b",
    VNState.S00: "c",
    VNState.S01: "o",
    VNState.S10: "f",
    VNState.S11: "g",
    VNState.S000: "h",
    VNState.OT_E: "r",
    VNState.OT_N: "u",
    VNState.OT_W: "l",
    VNState.OT_S: "d",
    VNState.OT_E_EX: "R",
    VNState.OT_N_EX: "U",
    VNState.OT_W_EX: "L",
    VNState.OT_S_EX: "D",
    VNState.ST_E: "e",
    VNState.ST_N: "n",
    VNState.ST_W: "w",
    VNState.ST_S: "s",
    VNState.ST_E_EX: "E",
    VNState.ST_N_EX: "N",
    VNState.ST_W_EX: "W",
    VNState.ST_S_EX: "V",
    VNState.C00: "C",
    VNState.C10: "*",
    VNState.C01: "+",
    VNState.C11: "#",
}


def parse_ascii_pattern(
    text: str, origin: Tuple[int, int] = (0, 0), strict: bool = False
) -> Dict[Tuple[int, int], int]:
    """Parse a multiline string using compact ASCII representation into cell dictionary.

    Coordinate convention: lines top-to-bottom map to decreasing y (Cartesian).
    Row 0 is top (y = num_rows - 1 + oy), bottom is y = oy.
    Lines starting with "//" are comments and skipped.

    Args:
        text: multiline pattern.
        origin: (ox, oy) offset added to parsed coordinates.
        strict: if True, raise ValueError on unknown non-whitespace chars;
            if False (default), unknown chars are ignored for back-compat.
    """
    pattern: Dict[Tuple[int, int], int] = {}
    lines = [line.rstrip("\n") for line in text.strip().splitlines()]
    lines = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("//")]
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
            elif strict and char not in (" ", "\t"):
                raise ValueError(f"Unknown pattern char {char!r} at row {row_idx}, col {col_idx}")

    return pattern


def export_to_ascii(pattern: Dict[Tuple[int, int], int], width: int, height: int) -> str:
    """Render a pattern dictionary as an ASCII diagram (round-trips via parse)."""
    grid = [["." for _ in range(width)] for _ in range(height)]

    for (x, y), st in pattern.items():
        if 0 <= x < width and 0 <= y < height:
            row = height - 1 - y
            grid[row][x] = STATE_TO_CHAR.get(int(st), "?")

    return "\n".join("".join(row) for row in grid)
