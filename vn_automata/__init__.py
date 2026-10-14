"""Von Neumann 29-State Cellular Automaton in Mesa."""

from .states import VNState, StateCategory, STATE_INFO, STATE_COLORS
from .rules import cell_next_state, step_grid
from .model import VonNeumannCA, CellAgent
from .patterns import PATTERNS, get_pattern, make_programmable_builder
from .loader import parse_ascii_pattern, export_to_ascii, CHAR_TO_STATE, STATE_TO_CHAR

__version__ = "0.1.0"
__all__ = [
    "VNState",
    "StateCategory",
    "STATE_INFO",
    "STATE_COLORS",
    "cell_next_state",
    "step_grid",
    "VonNeumannCA",
    "CellAgent",
    "PATTERNS",
    "get_pattern",
    "make_programmable_builder",
    "parse_ascii_pattern",
    "export_to_ascii",
    "CHAR_TO_STATE",
    "STATE_TO_CHAR",
]
