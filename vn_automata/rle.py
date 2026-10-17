"""Golly Extended RLE reader/writer for von Neumann patterns.

Golly — the reference simulator for John von Neumann's 29-state automaton —
stores patterns as *Extended RLE*. Supporting it means patterns made for (or
exported from) Golly load directly here and vice versa.

State encoding (from the Golly manual):

* ``.`` is state 0 (and ``b`` for two-state rules);
* states 1..24 are the single characters ``A``..``X``;
* state ``25 + 24*k`` .. ``48 + 24*k`` use a two-character token: a block letter
  (``p`` for 25..48, ``q`` for 49..72, ...) followed by ``A``..``X``.

So for the 29-state rule, states 25..28 appear as ``pA``..``pD``.

Coordinates are Cartesian (``y`` increases upwards), matching the rest of the
package; RLE itself is written top row first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

Coordinate = Tuple[int, int]
Pattern = Dict[Coordinate, int]

N_STATES = 29  # this package implements the 29-state rule

_BLOCK_BASE = {"p": 25, "q": 49, "r": 73, "s": 97, "t": 121,
               "u": 145, "v": 169, "w": 193, "x": 217, "y": 241}

_HEADER_RE = re.compile(r"x\s*=\s*(\d+)\s*,\s*y\s*=\s*(\d+)(?:\s*,\s*rule\s*=\s*(\S+))?")


class RLEError(ValueError):
    """Raised for malformed RLE input."""


@dataclass
class RLEPattern:
    pattern: Pattern
    width: int
    height: int
    rule: Optional[str] = None
    comments: List[str] = field(default_factory=list)


def state_to_token(state: int) -> str:
    """Render a single state as its Golly Extended RLE token."""
    state = int(state)
    if state == 0:
        return "."
    if not 1 <= state <= 255:
        raise RLEError(f"state {state} not representable in RLE")
    block, letter = divmod(state - 1, 24)
    if block == 0:
        return chr(ord("A") + letter)
    prefix = chr(ord("p") + block - 1)
    if prefix not in _BLOCK_BASE:
        raise RLEError(f"state {state} outside the encodable range")
    return prefix + chr(ord("A") + letter)


def token_to_state(token: str) -> int:
    """Inverse of :func:`state_to_token` (also accepts ``b``/``o``)."""
    if token in (".", "b"):
        return 0
    if token == "o":
        return 1
    if len(token) == 1 and "A" <= token <= "X":
        return ord(token) - ord("A") + 1
    if len(token) == 2 and token[0] in _BLOCK_BASE and "A" <= token[1] <= "X":
        state = _BLOCK_BASE[token[0]] + (ord(token[1]) - ord("A"))
        if state > 255:
            raise RLEError(f"state {state} out of the 0..255 range")
        return state
    raise RLEError(f"invalid RLE token {token!r}")


def parse_rle(text: str) -> RLEPattern:
    """Parse Golly Extended RLE text into a :class:`RLEPattern`."""
    comments: List[str] = []
    header_rule: Optional[str] = None
    header_w = header_h = 0
    body_parts: List[str] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            comments.append(line[1:].strip())
            continue
        if "=" in line and header_w == 0:
            match = _HEADER_RE.search(line)
            if match:
                header_w = int(match.group(1))
                header_h = int(match.group(2))
                header_rule = match.group(3)
                continue
        body_parts.append(line)

    body = "".join(body_parts)
    if not body.endswith("!"):
        body += "!"

    cells: Dict[Coordinate, int] = {}
    x = row = 0
    max_row = 0
    i = 0
    length = len(body)
    while i < length:
        char = body[i]
        if char == "!":
            break
        if char.isspace():
            i += 1
            continue
        # optional run count
        count = 0
        while i < length and body[i].isdigit():
            count = count * 10 + int(body[i])
            i += 1
        count = count or 1
        if i >= length:
            break
        char = body[i]

        if char == "$":
            row += count
            x = 0
            i += 1
            continue
        if char in ".bo":
            token = char
            i += 1
        elif char in _BLOCK_BASE:
            token = body[i:i + 2]
            i += 2
        else:
            token = char
            i += 1
        state = token_to_state(token)
        if state >= N_STATES:
            raise RLEError(f"state {state} out of range for the 29-state rule")
        for _ in range(count):
            cells[(x, row)] = state
            x += 1
        max_row = max(max_row, row)

    # Convert top-first rows to Cartesian y (y increases upward).
    height = header_h or (max_row + 1)
    pattern = {
        (px, (height - 1) - py): state
        for (px, py), state in cells.items()
        if state != 0
    }
    width = header_w or (max((px for px, _ in pattern), default=-1) + 1)
    return RLEPattern(pattern=pattern, width=width, height=height,
                      rule=header_rule, comments=comments)


def _bounding_box(pattern: Pattern) -> Tuple[int, int]:
    if not pattern:
        return 0, 0
    return max(x for x, _ in pattern) + 1, max(y for _, y in pattern) + 1


def export_rle(
    pattern: Pattern,
    width: Optional[int] = None,
    height: Optional[int] = None,
    rule: str = "JvN29",
    comments: Sequence[str] = (),
) -> str:
    """Serialise a pattern dictionary to Golly Extended RLE text."""
    bw, bh = _bounding_box(pattern)
    width = bw if width is None else max(int(width), bw)
    height = bh if height is None else max(int(height), bh)

    grid = [[0] * width for _ in range(height)]
    for (x, y), state in pattern.items():
        if 0 <= x < width and 0 <= y < height:
            grid[height - 1 - y][x] = int(state)  # top row first

    rows: List[str] = []
    for row in grid:
        last = width - 1
        while last >= 0 and row[last] == 0:
            last -= 1
        if last < 0:
            rows.append("")
            continue
        pieces: List[str] = []
        i = 0
        while i <= last:
            state = row[i]
            j = i + 1
            while j <= last and row[j] == state:
                j += 1
            run = j - i
            pieces.append((str(run) if run > 1 else "") + state_to_token(state))
            i = j
        rows.append("".join(pieces))

    while rows and rows[-1] == "":
        rows.pop()

    out: List[str] = [f"# {c}" for c in comments]
    out.append(f"x = {width}, y = {height}, rule = {rule}")
    out.append("$".join(rows) + "!")
    return "\n".join(out) + "\n"
