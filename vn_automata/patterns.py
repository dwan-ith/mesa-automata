"""Pre-configured canonical patterns demonstrating Von Neumann 29-state CA phenomena.

Included patterns:
1. pulse_loop: Continuous signal circulation in an ordinary transmission loop.
2. confluent_splitter: Signal duplication and ordinary-to-special conversion through a confluent node.
3. construction_arm_1011: Special transmission wire extending itself by 1 cell using pulse 1011.
4. destruction_demo: Antagonistic erasure of ordinary wire by special transmission pulse.
5. confluent_and_gate: 2-input confluent AND gate requiring synchronous simultaneous pulses.
6. programmable_builder: Parameterized constructor line emitting any binary construction sequence.
"""

from typing import Dict, List, Tuple
from .states import VNState


def make_pulse_loop(x: int = 10, y: int = 10, width: int = 6, height: int = 6) -> Dict[Tuple[int, int], int]:
    """Create a rectangular ordinary transmission loop with a single circulating pulse."""
    if width < 3 or height < 3:
        raise ValueError(f"Loop must be at least 3x3, got {width}x{height}")
    pattern: Dict[Tuple[int, int], int] = {}

    # Bottom edge (East)
    for i in range(x, x + width - 1):
        pattern[(i, y)] = VNState.OT_E
    # Right edge (North)
    for j in range(y, y + height - 1):
        pattern[(x + width - 1, j)] = VNState.OT_N
    # Top edge (West)
    for i in range(x + width - 1, x, -1):
        pattern[(i, y + height - 1)] = VNState.OT_W
    # Left edge (South)
    for j in range(y + height - 1, y, -1):
        pattern[(x, j)] = VNState.OT_S

    # Inject one excited pulse on the bottom wire
    pattern[(x + 1, y)] = VNState.OT_E_EX
    return pattern


def make_confluent_splitter(x: int = 10, y: int = 15) -> Dict[Tuple[int, int], int]:
    """Create a confluent node taking input from an OTS wire and splitting to OTS & STS branches.
    
    Wire layout:
    - Input wire (OTS East) approaching from west: (x, y) -> (x+3, y)
    - Confluent cell at (x+4, y)
    - Output branch 1 (OTS North): (x+4, y+1) -> (x+4, y+4)
    - Output branch 2 (STS East): (x+5, y) -> (x+8, y)
    """
    pattern: Dict[Tuple[int, int], int] = {}

    # Input line (OTS East)
    pattern[(x, y)] = VNState.OT_E
    pattern[(x + 1, y)] = VNState.OT_E_EX  # Leading pulse
    pattern[(x + 2, y)] = VNState.OT_E
    pattern[(x + 3, y)] = VNState.OT_E

    # Confluent node
    pattern[(x + 4, y)] = VNState.C00

    # Output branch 1 (Ordinary Transmission pointing North)
    pattern[(x + 4, y + 1)] = VNState.OT_N
    pattern[(x + 4, y + 2)] = VNState.OT_N
    pattern[(x + 4, y + 3)] = VNState.OT_N

    # Output branch 2 (Special Transmission pointing East)
    pattern[(x + 5, y)] = VNState.ST_E
    pattern[(x + 6, y)] = VNState.ST_E
    pattern[(x + 7, y)] = VNState.ST_E

    return pattern


def make_construction_arm_1011(x: int = 10, y: int = 15) -> Dict[Tuple[int, int], int]:
    """Demonstrate the canonical von Neumann construction process.
    
    A special transmission arm pointing East sends the sequence 1011 (1, 0, 1, 1)
    into the quiescent ground state cell at the tip.
    Over 4 cycles of excitation, the ground cell transitions through:
    U -> S -> S0 -> S01 -> ST_E
    extending the arm by one cell!
    
    Spacing: To deliver 1, 0, 1, 1 consecutively at the tip, pulses travel at 1 cell/tick.
    At t=0:
    - Tip of wire at (x+4, y)
    - Target ground cell at (x+5, y)
    - Pulse 1 at (x+4, y) [arrives at t=1]
    - Space 0 at (x+3, y) [arrives at t=2]
    - Pulse 1 at (x+2, y) [arrives at t=3]
    - Pulse 1 at (x+1, y) [arrives at t=4]
    """
    pattern: Dict[Tuple[int, int], int] = {}

    # Special transmission line pointing East with pulse train 1011
    pattern[(x, y)] = VNState.ST_E
    pattern[(x + 1, y)] = VNState.ST_E_EX  # bit 1 (arrives t=4)
    pattern[(x + 2, y)] = VNState.ST_E_EX  # bit 1 (arrives t=3)
    pattern[(x + 3, y)] = VNState.ST_E     # bit 0 (arrives t=2)
    pattern[(x + 4, y)] = VNState.ST_E_EX  # bit 1 (arrives t=1)

    # (x+5, y) is left as Ground state U (target to be constructed)
    return pattern


def make_destruction_demo(x: int = 10, y: int = 15) -> Dict[Tuple[int, int], int]:
    """Demonstrate antagonistic destruction / cell clearing.
    
    A Special Transmission line points towards an Ordinary Transmission wire.
    When the STS excited pulse collides with the OTS wire, the OTS cell is erased back to U (ground).
    """
    pattern: Dict[Tuple[int, int], int] = {}

    # Special transmission line pointing East
    pattern[(x, y)] = VNState.ST_E
    pattern[(x + 1, y)] = VNState.ST_E_EX  # Destruction pulse
    pattern[(x + 2, y)] = VNState.ST_E

    # Target Ordinary Transmission wire perpendicular (pointing North)
    pattern[(x + 3, y - 1)] = VNState.OT_N
    pattern[(x + 3, y)] = VNState.OT_N     # Cell to be erased
    pattern[(x + 3, y + 1)] = VNState.OT_N

    return pattern


def make_confluent_and_gate(x: int = 10, y: int = 15) -> Dict[Tuple[int, int], int]:
    """Demonstrate a 2-input Confluent AND gate.
    
    Two OTS wires feed into a single Confluent cell.
    The Confluent cell only becomes excited if BOTH incoming pulses arrive simultaneously.
    """
    pattern: Dict[Tuple[int, int], int] = {}

    # Input 1: from West pointing East
    pattern[(x, y)] = VNState.OT_E
    pattern[(x + 1, y)] = VNState.OT_E_EX  # Pulse 1
    pattern[(x + 2, y)] = VNState.OT_E

    # Input 2: from South pointing North
    pattern[(x + 3, y - 3)] = VNState.OT_N
    pattern[(x + 3, y - 2)] = VNState.OT_N_EX  # Pulse 2 (same arrival time)
    pattern[(x + 3, y - 1)] = VNState.OT_N

    # Confluent gate at (x+3, y)
    pattern[(x + 3, y)] = VNState.C00

    # Output line pointing East
    pattern[(x + 4, y)] = VNState.OT_E
    pattern[(x + 5, y)] = VNState.OT_E

    return pattern


def make_programmable_builder(
    sequence: str,
    x: int = 5,
    y: int = 15,
    max_len: int = 32,
) -> Dict[Tuple[int, int], int]:
    """Construct an arm with any custom binary sequence (e.g., '1011', '10000', '1111').

    The sequence is ordered chronologically (first bit arrives first).

    Raises:
        ValueError: if sequence is empty, exceeds max_len, or contains chars
            other than '0'/'1'.
    """
    seq = sequence.strip()
    if not seq:
        raise ValueError("Sequence must not be empty")
    if len(seq) > max_len:
        raise ValueError(f"Sequence too long ({len(seq)} > {max_len})")
    if any(bit not in ("0", "1") for bit in seq):
        raise ValueError(f"Sequence must be binary ('0'/'1' only), got {sequence!r}")
    pattern: Dict[Tuple[int, int], int] = {}
    n = len(seq)

    # Place wire leading up to tip
    # sequence[0] should be at the tip (x + n, y)
    # sequence[1] at (x + n - 1, y), etc.
    for i, bit in enumerate(seq):
        pos_x = x + (n - 1 - i)
        is_1 = (bit == "1")
        pattern[(pos_x, y)] = VNState.ST_E_EX if is_1 else VNState.ST_E

    return pattern


PATTERNS = {
    "pulse_loop": {
        "name": "Continuous Pulse Loop",
        "description": "Signal circulating infinitely in an ordinary transmission wire loop.",
        "factory": lambda: make_pulse_loop(12, 12, 8, 8),
        "width": 35,
        "height": 35,
    },
    "confluent_splitter": {
        "name": "Confluent Signal Splitter",
        "description": "Confluent node duplicating pulse into ordinary and special transmission branches.",
        "factory": lambda: make_confluent_splitter(10, 15),
        "width": 35,
        "height": 35,
    },
    "construction_arm_1011": {
        "name": "Construction Arm Elongation (1011)",
        "description": "Special transmission arm extending itself by 1 cell using sequence 1011.",
        "factory": lambda: make_construction_arm_1011(10, 15),
        "width": 35,
        "height": 35,
    },
    "destruction_demo": {
        "name": "Antagonistic Wire Erasure",
        "description": "Special transmission pulse erasing an ordinary transmission cell to ground U.",
        "factory": lambda: make_destruction_demo(10, 15),
        "width": 35,
        "height": 35,
    },
    "confluent_and_gate": {
        "name": "Confluent AND Gate",
        "description": "Two inputs converging into a confluent cell requiring simultaneous pulses.",
        "factory": lambda: make_confluent_and_gate(10, 15),
        "width": 35,
        "height": 35,
    },
}


def get_pattern(key: str) -> Dict[Tuple[int, int], int]:
    """Retrieve pattern dictionary by key."""
    if key in PATTERNS:
        return PATTERNS[key]["factory"]()
    raise KeyError(f"Unknown pattern key '{key}'. Available: {list(PATTERNS.keys())}")
