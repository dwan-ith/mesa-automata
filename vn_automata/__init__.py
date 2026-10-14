"""Von Neumann 29-State Cellular Automaton in Mesa."""

from .states import VNState, StateCategory, STATE_INFO, STATE_COLORS
from .rules import cell_next_state, step_grid
from .model import VonNeumannCA, CellAgent
from .patterns import PATTERNS, get_pattern

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
]
