"""Independent reference implementation of von Neumann's 29-state CA.

This module parses the canonical Golly ``JvN29.rule`` table (the de-facto
reference used by researchers and by Golly itself) and re-implements the
transition function *independently* of :mod:`vn_automata.rules`. It exists so
that the fast production engine can be validated against a trusted external
artefact rather than only against itself.

Reference artefact:
    ``vn_automata/data/JvN29.rule`` -- the JvN29 rule distributed with Golly,
    mirrored from https://github.com/EZLiang/vonNeumannTuringMachine

Neighborhood order in Golly tables for ``neighborhood:vonNeumann`` is::

    center, north, east, south, west

which matches the ``(center, north, east, south, west)`` signature of
:func:`vn_automata.rules.cell_next_state`.
"""

from __future__ import annotations

import os
import re
from typing import Dict, FrozenSet, List, Tuple

import numpy as np

from .engine import compile_rules

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "JvN29.rule")

_VAR_RE = re.compile(r"^var\s+([A-Za-z])\s*=\s*\{([^}]*)\}")
_RULE_RE = re.compile(r"^[-0-9A-Za-z,\s]+$")


def parse_rule_text(text: str) -> Tuple[Dict[str, Tuple[int, ...]], List[Tuple[Tuple[FrozenSet[int], ...], int]]]:
    """Parse the ``@TABLE`` section of a Golly rule file.

    Returns ``(variables, rules)`` where each rule is
    ``((center, north, east, south, west), new_state)`` and every component is a
    frozenset of admissible states.
    """
    variables: Dict[str, Tuple[int, ...]] = {}
    rules: List[Tuple[Tuple[FrozenSet[int], ...], int]] = []

    in_table = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("@"):
            in_table = line.startswith("@TABLE")
            continue
        if not in_table:
            continue

        match = _VAR_RE.match(line)
        if match:
            values = tuple(
                int(x) for x in match.group(2).split(",") if x.strip() != ""
            )
            variables[match.group(1)] = values
            continue

        if ":" in line:  # n_states:29, neighborhood:..., symmetries:none
            continue
        if not _RULE_RE.match(line):
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:
            continue

        domains: List[FrozenSet[int]] = []
        for token in parts[:5]:
            if token in variables:
                domains.append(frozenset(variables[token]))
            else:
                domains.append(frozenset((int(token),)))
        rules.append((tuple(domains), int(parts[5])))

    return variables, rules


def parse_rule_file(path: str = DATA_PATH):
    with open(path, "r", encoding="utf-8") as handle:
        return parse_rule_text(handle.read())


def parse_colors(path: str = DATA_PATH) -> Dict[int, Tuple[int, int, int]]:
    """Parse the ``@COLORS`` section into ``{state: (r, g, b)}``."""
    colors: Dict[int, Tuple[int, int, int]] = {}
    in_colors = False
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("@"):
                in_colors = line.startswith("@COLORS")
                continue
            if in_colors and line:
                parts = line.split()
                if len(parts) >= 4:
                    colors[int(parts[0])] = (int(parts[1]), int(parts[2]), int(parts[3]))
    return colors


class GollyJVN29:
    """Independent transition engine built from the Golly JvN29 rule table.

    Unmatched neighborhoods leave the cell unchanged, mirroring Golly's
    behaviour. Rules are evaluated in file order (first match wins).
    """

    def __init__(self, path: str = DATA_PATH) -> None:
        self.variables, self.rules = parse_rule_file(path)
        self.colors = parse_colors(path)
        self._masks = [
            (tuple(sum(1 << v for v in domain) for domain in domains), new)
            for domains, new in self.rules
        ]

    @property
    def n_states(self) -> int:
        return 29

    def compiled_table(self) -> np.ndarray:
        """The rule file compiled into the same flat table format as the engine."""
        return compile_rules(self.n_states, self.rules)

    def next_state(self, center: int, north: int, east: int, south: int, west: int) -> int:
        for domains, new in self.rules:
            if (
                center in domains[0]
                and north in domains[1]
                and east in domains[2]
                and south in domains[3]
                and west in domains[4]
            ):
                return new
        return center

    def masks(self):
        """Return ``(mask_matrix, new_states)`` for vectorized comparison.

        ``mask_matrix`` has shape ``(n_rules, 5)`` of 29-bit admissible-state
        masks in ``(center, north, east, south, west)`` order.
        """
        masks = np.array([[m for m in rule_masks] for rule_masks, _ in self._masks], dtype=np.int64)
        news = np.array([new for _, new in self._masks], dtype=np.int64)
        return masks, news

    def step_grid(self, grid: np.ndarray, torus: bool = False) -> np.ndarray:
        """Step a ``(height, width)`` array with the reference rules.

        Coordinates follow the Mesa Cartesian convention used everywhere in
        this project: ``grid[y, x]``, north is ``(x, y + 1)``.
        """
        grid = np.asarray(grid, dtype=np.int64)
        height, width = grid.shape
        out = np.empty_like(grid)

        if torus:
            north = np.roll(grid, -1, axis=0)
            south = np.roll(grid, 1, axis=0)
            east = np.roll(grid, -1, axis=1)
            west = np.roll(grid, 1, axis=1)
        else:
            padded = np.pad(grid, 1, mode="constant", constant_values=0)
            north = padded[2:, 1:-1]
            south = padded[:-2, 1:-1]
            west = padded[1:-1, :-2]
            east = padded[1:-1, 2:]

        for y in range(height):
            for x in range(width):
                out[y, x] = self.next_state(
                    int(grid[y, x]),
                    int(north[y, x]),
                    int(east[y, x]),
                    int(south[y, x]),
                    int(west[y, x]),
                )
        return out


DEFAULT = GollyJVN29()
