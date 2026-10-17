"""Von Neumann's 29-state self-reproducing cellular automaton, in Mesa.

The package is layered:

* :mod:`vn_automata.states` — the 29 states and their functional vocabulary.
* :mod:`vn_automata.engine` — a generic, rule-agnostic 5-neighbour CA engine
  that compiles rule sets into lookup tables.
* :mod:`vn_automata.rules` — von Neumann's rule set, *generated* from the model
  and exposed as ``cell_next_state`` / ``step_grid``.
* :mod:`vn_automata.model` — the Mesa 3.x model and cell agents.
* :mod:`vn_automata.patterns`, :mod:`vn_automata.loader` — demos and I/O.
* :mod:`vn_automata.golly` — an independent reference used for validation.
"""

from .states import (
    CONFLUENT,
    CONFLUENT_EMITTING,
    Direction,
    GROUND,
    OTS,
    OTS_EXCITED,
    OTS_QUIESCENT,
    SENSITIZED,
    STATE_COLORS,
    STATE_COUNT,
    STATE_INFO,
    STS,
    STS_EXCITED,
    STS_QUIESCENT,
    StateCategory,
    VNState,
    is_excited,
)
from .engine import RuleSet, TableAutomaton, compile_rules
from .rules import (
    automaton,
    build_rule_set,
    cell_next_state,
    rule_count,
    step_grid,
)
from .model import CellAgent, VonNeumannCA
from .patterns import PATTERNS, get_pattern, make_programmable_builder
from .loader import CHAR_TO_STATE, STATE_TO_CHAR, export_to_ascii, parse_ascii_pattern
from .rle import RLEPattern, export_rle, parse_rle
from .golly import GollyJVN29, parse_colors, parse_rule_file

__version__ = "0.2.0"
__all__ = [
    # states
    "VNState",
    "StateCategory",
    "Direction",
    "STATE_INFO",
    "STATE_COLORS",
    "STATE_COUNT",
    "GROUND",
    "SENSITIZED",
    "OTS",
    "OTS_QUIESCENT",
    "OTS_EXCITED",
    "STS",
    "STS_QUIESCENT",
    "STS_EXCITED",
    "CONFLUENT",
    "CONFLUENT_EMITTING",
    "is_excited",
    # engine
    "RuleSet",
    "TableAutomaton",
    "compile_rules",
    # rules
    "automaton",
    "build_rule_set",
    "cell_next_state",
    "step_grid",
    "rule_count",
    # model
    "VonNeumannCA",
    "CellAgent",
    # patterns / io
    "PATTERNS",
    "get_pattern",
    "make_programmable_builder",
    "parse_ascii_pattern",
    "export_to_ascii",
    "CHAR_TO_STATE",
    "STATE_TO_CHAR",
    "parse_rle",
    "export_rle",
    "RLEPattern",
    # reference
    "GollyJVN29",
    "parse_rule_file",
    "parse_colors",
]
