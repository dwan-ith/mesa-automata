"""Von Neumann 29-State Cellular Automaton state definitions, categories, and colors.

Canonical encoding matches the standard JvN29 rule specification:
- 0: Ground / Unexcitable (U)
- 1-8: Sensitized / Transition states (S, S0, S1, S00, S01, S10, S11, S000)
- 9-12: Ordinary Transmission States (Quiescent: East, North, West, South)
- 13-16: Ordinary Transmission States (Excited: East, North, West, South)
- 17-20: Special Transmission States (Quiescent: East, North, West, South)
- 21-24: Special Transmission States (Excited: East, North, West, South)
- 25-28: Confluent states (C00, C10, C01, C11)
"""

from enum import IntEnum, unique
from typing import NamedTuple, Optional, Tuple


class StateCategory(IntEnum):
    GROUND = 0
    SENSITIZED = 1
    ORDINARY_TRANSMISSION = 2
    SPECIAL_TRANSMISSION = 3
    CONFLUENT = 4


class Direction(IntEnum):
    NONE = 0
    EAST = 1
    NORTH = 2
    WEST = 3
    SOUTH = 4


@unique
class VNState(IntEnum):
    # Ground state
    U = 0

    # Sensitized / Transition states
    S = 1
    S0 = 2
    S1 = 3
    S00 = 4
    S01 = 5
    S10 = 6
    S11 = 7
    S000 = 8

    # Ordinary transmission quiescent (OTS_QUI)
    OT_E = 9
    OT_N = 10
    OT_W = 11
    OT_S = 12

    # Ordinary transmission excited (OTS_EXC)
    OT_E_EX = 13
    OT_N_EX = 14
    OT_W_EX = 15
    OT_S_EX = 16

    # Special transmission quiescent (STS_QUI)
    ST_E = 17
    ST_N = 18
    ST_W = 19
    ST_S = 20

    # Special transmission excited (STS_EXC)
    ST_E_EX = 21
    ST_N_EX = 22
    ST_W_EX = 23
    ST_S_EX = 24

    # Confluent states
    C00 = 25  # Quiescent now, quiescent next
    C10 = 26  # Excited now, quiescent next
    C01 = 27  # Quiescent now, excited next
    C11 = 28  # Excited now, excited next


class StateMetadata(NamedTuple):
    code: int
    name: str
    symbol: str
    category: StateCategory
    direction: Direction
    is_excited: bool
    rgb: Tuple[int, int, int]
    hex_color: str
    description: str


STATE_INFO = {
    VNState.U: StateMetadata(
        0, "U", "·", StateCategory.GROUND, Direction.NONE, False,
        (48, 48, 48), "#303030", "Ground (unexcitable background space)"
    ),
    VNState.S: StateMetadata(
        1, "S", "S", StateCategory.SENSITIZED, Direction.NONE, True,
        (255, 0, 0), "#ff0000", "Sensitized (newly sensitized from U)"
    ),
    VNState.S0: StateMetadata(
        2, "S0", "S₀", StateCategory.SENSITIZED, Direction.NONE, False,
        (255, 125, 0), "#ff7d00", "Sensitized (after pulse 0)"
    ),
    VNState.S1: StateMetadata(
        3, "S1", "S₁", StateCategory.SENSITIZED, Direction.NONE, True,
        (255, 150, 25), "#ff9619", "Sensitized (after pulse 1)"
    ),
    VNState.S00: StateMetadata(
        4, "S00", "S₀₀", StateCategory.SENSITIZED, Direction.NONE, False,
        (255, 175, 50), "#ffaf32", "Sensitized (after pulse 00)"
    ),
    VNState.S01: StateMetadata(
        5, "S01", "S₀₁", StateCategory.SENSITIZED, Direction.NONE, True,
        (255, 200, 75), "#ffc84b", "Sensitized (after pulse 01)"
    ),
    VNState.S10: StateMetadata(
        6, "S10", "S₁₀", StateCategory.SENSITIZED, Direction.NONE, False,
        (255, 255, 100), "#ffff64", "Sensitized (after pulse 10)"
    ),
    VNState.S11: StateMetadata(
        7, "S11", "S₁₁", StateCategory.SENSITIZED, Direction.NONE, True,
        (255, 250, 125), "#fffa7d", "Sensitized (after pulse 11)"
    ),
    VNState.S000: StateMetadata(
        8, "S000", "S₀₀₀", StateCategory.SENSITIZED, Direction.NONE, False,
        (251, 255, 0), "#fbff00", "Sensitized (after pulse 000)"
    ),

    # Ordinary transmission quiescent
    VNState.OT_E: StateMetadata(
        9, "OT_E", "→", StateCategory.ORDINARY_TRANSMISSION, Direction.EAST, False,
        (89, 89, 255), "#5959ff", "Ordinary transmission East (quiescent wire)"
    ),
    VNState.OT_N: StateMetadata(
        10, "OT_N", "↑", StateCategory.ORDINARY_TRANSMISSION, Direction.NORTH, False,
        (106, 106, 255), "#6a6aff", "Ordinary transmission North (quiescent wire)"
    ),
    VNState.OT_W: StateMetadata(
        11, "OT_W", "←", StateCategory.ORDINARY_TRANSMISSION, Direction.WEST, False,
        (122, 122, 255), "#7a7aff", "Ordinary transmission West (quiescent wire)"
    ),
    VNState.OT_S: StateMetadata(
        12, "OT_S", "↓", StateCategory.ORDINARY_TRANSMISSION, Direction.SOUTH, False,
        (139, 139, 255), "#8b8bff", "Ordinary transmission South (quiescent wire)"
    ),

    # Ordinary transmission excited
    VNState.OT_E_EX: StateMetadata(
        13, "OT_E_EX", "▸", StateCategory.ORDINARY_TRANSMISSION, Direction.EAST, True,
        (27, 176, 27), "#1bb01b", "Ordinary transmission East (excited pulse)"
    ),
    VNState.OT_N_EX: StateMetadata(
        14, "OT_N_EX", "▴", StateCategory.ORDINARY_TRANSMISSION, Direction.NORTH, True,
        (36, 200, 36), "#24c824", "Ordinary transmission North (excited pulse)"
    ),
    VNState.OT_W_EX: StateMetadata(
        15, "OT_W_EX", "◂", StateCategory.ORDINARY_TRANSMISSION, Direction.WEST, True,
        (73, 255, 73), "#49ff49", "Ordinary transmission West (excited pulse)"
    ),
    VNState.OT_S_EX: StateMetadata(
        16, "OT_S_EX", "▾", StateCategory.ORDINARY_TRANSMISSION, Direction.SOUTH, True,
        (106, 255, 106), "#6aff6a", "Ordinary transmission South (excited pulse)"
    ),

    # Special transmission quiescent
    VNState.ST_E: StateMetadata(
        17, "ST_E", "⇒", StateCategory.SPECIAL_TRANSMISSION, Direction.EAST, False,
        (235, 36, 36), "#eb2424", "Special transmission East (construction arm wire)"
    ),
    VNState.ST_N: StateMetadata(
        18, "ST_N", "⇑", StateCategory.SPECIAL_TRANSMISSION, Direction.NORTH, False,
        (255, 56, 56), "#ff3838", "Special transmission North (construction arm wire)"
    ),
    VNState.ST_W: StateMetadata(
        19, "ST_W", "⇐", StateCategory.SPECIAL_TRANSMISSION, Direction.WEST, False,
        (255, 73, 73), "#ff4949", "Special transmission West (construction arm wire)"
    ),
    VNState.ST_S: StateMetadata(
        20, "ST_S", "⇓", StateCategory.SPECIAL_TRANSMISSION, Direction.SOUTH, False,
        (255, 89, 89), "#ff5959", "Special transmission South (construction arm wire)"
    ),

    # Special transmission excited
    VNState.ST_E_EX: StateMetadata(
        21, "ST_E_EX", "▶", StateCategory.SPECIAL_TRANSMISSION, Direction.EAST, True,
        (185, 56, 255), "#b938ff", "Special transmission East (excited construction pulse)"
    ),
    VNState.ST_N_EX: StateMetadata(
        22, "ST_N_EX", "▲", StateCategory.SPECIAL_TRANSMISSION, Direction.NORTH, True,
        (191, 73, 255), "#bf49ff", "Special transmission North (excited construction pulse)"
    ),
    VNState.ST_W_EX: StateMetadata(
        23, "ST_W_EX", "◀", StateCategory.SPECIAL_TRANSMISSION, Direction.WEST, True,
        (197, 89, 255), "#c559ff", "Special transmission West (excited construction pulse)"
    ),
    VNState.ST_S_EX: StateMetadata(
        24, "ST_S_EX", "▼", StateCategory.SPECIAL_TRANSMISSION, Direction.SOUTH, True,
        (203, 106, 255), "#cb6aff", "Special transmission South (excited construction pulse)"
    ),

    # Confluent states
    VNState.C00: StateMetadata(
        25, "C00", "©", StateCategory.CONFLUENT, Direction.NONE, False,
        (0, 255, 128), "#00ff80", "Confluent node (quiescent now, quiescent next)"
    ),
    VNState.C10: StateMetadata(
        26, "C10", "✱", StateCategory.CONFLUENT, Direction.NONE, True,
        (255, 255, 128), "#ffff80", "Confluent node (emitting pulse now, quiescent next)"
    ),
    VNState.C01: StateMetadata(
        27, "C01", "⊙", StateCategory.CONFLUENT, Direction.NONE, False,
        (255, 128, 64), "#ff8040", "Confluent node (quiescent now, emitting pulse next)"
    ),
    VNState.C11: StateMetadata(
        28, "C11", "✪", StateCategory.CONFLUENT, Direction.NONE, True,
        (33, 215, 215), "#21d7d7", "Confluent node (emitting pulse now, emitting pulse next)"
    ),
}

STATE_COLORS = {s: info.hex_color for s, info in STATE_INFO.items()}
STATE_NAMES = {s: info.name for s, info in STATE_INFO.items()}
