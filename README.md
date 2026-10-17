# Von Neumann 29-State Self-Reproducing Automaton in Mesa 3.x

A faithful, executable, and interactive implementation of **John von Neumann's 29-state Cellular Automaton** built on the [Mesa](https://github.com/projectmesa/mesa) Agent-Based Modeling framework.

---

## 1. Background & Motivation

In the late 1940s, mathematician **John von Neumann** (with suggestions from Stanislaw Ulam) posed a profound question:
> *Can a physical or formal system composed of simple, locally interacting identical components build an exact duplicate of itself by executing a description of its own structure?*

While Conway's *Game of Life* (1970) later demonstrated that simple 2-state local rules can yield rich complexity, von Neumann's 1966 model tackled **universal construction and machine self-replication**:
- **Stored Description (DNA/Tape):** A passive linear encoding of the system's own architecture.
- **Universal Constructor (Ribosome/Arm):** A machine capable of reading any blueprint tape and mechanically constructing the specified structure in empty cell space.
- **Universal Copier:** A mechanism that duplicates the description tape and attaches it to the offspring, avoiding the infinite regress paradox.

This repository provides a Python/Mesa implementation of the 29-state rule-set, transmission/construction demonstrator patterns, test validation suite, and an interactive real-time visualizer.

The transition function is **generated from a description of the model**, not transcribed as a list of cases — see [§2.1](#21-how-the-rules-are-implemented). It is then validated, exhaustively, against the canonical Golly `JvN29.rule` table.

> Scope note: this covers the verified transmission rules, confluent logic, antagonistic erasure, and single-cell synthesis via the sensitized tree (the 9 canonical pulse sequences). A full universal constructor + tape + self-replicating configuration (100k+ cells, never completely specified in the literature) is out of scope.

---

## 2. The 29 States & Transition Logic

Von Neumann's cellular automaton operates on a 2D Cartesian grid with a **5-cell orthogonal neighborhood** (the cell itself + North, East, South, West). Every cell synchronously transitions between 29 states:

| Category | States | Count | Functional Role |
| :--- | :--- | :--- | :--- |
| **Ground State** | `U` (0) | 1 | Unexcitable background space. Blank slate for construction. |
| **Sensitized / Transition** | `S, S0, S1, S00, S01, S10, S11, S000` (1–8) | 8 | Transient differentiation tree directing cell synthesis from incoming pulse trains. |
| **Ordinary Transmission (OTS)** | `OT_E, OT_N, OT_W, OT_S` (9–12: Quiescent)<br>`OT_E_EX, OT_N_EX, OT_W_EX, OT_S_EX` (13–16: Excited) | 8 | Directional wires that route binary control and computational pulses. |
| **Special Transmission (STS)** | `ST_E, ST_N, ST_W, ST_S` (17–20: Quiescent)<br>`ST_E_EX, ST_N_EX, ST_W_EX, ST_S_EX` (21–24: Excited) | 8 | Construction arm wires. Pulses emerging at the tip synthesize new cells. |
| **Confluent Nodes** | `C00, C01, C10, C11` (25–28) | 4 | Signal splitters (fan-out), ordinary-to-special conversion bridges, and 2-input AND gates with 1-cycle delay. Numbering and colours follow the canonical Golly `JvN29.rule` table. |

### The Construction Bifurcation Tree

When an excited pulse reaches a ground cell `U`, it awakens into state `S`. A binary pulse train subsequently guides the cell along a decision tree into any of the 9 quiescent functional states:

```
                  U (Ground)
                      | [1]
                      S
                     / \
              [0]  /     \  [1]
                 S0       S1
                /  \     /  \
         [0]  /   [1]\ [0]\  \[1]
            S00   S01 S10   S11
            / \   / \ / \   / \
           /   \ /   \   \ /   \
       S000  OT_W OT_S ST_E ST_N ST_W ST_S C00
       /  \  [1001][1010][1011][1100][1101][1110][1111]
     OT_E OT_N
    [10000][10001]
```

### Destruction / Antagonistic Erasure
To retract the construction arm or clear obstacles, the ordinary and special transmission subsets are **mutually antagonistic**:
- An excited **Special Transmission** pulse entering an **Ordinary Transmission** or **Confluent** cell erases it back to ground `U`.
- An excited **Ordinary Transmission** pulse entering a **Special Transmission** cell erases it back to ground `U`.

### 2.1 How the rules are implemented

There is no giant `if state == 13: ... elif state == 14: ...` cascade. Instead the
rule set is *described* in the language of von Neumann's model and **compiled**
into a lookup table:

1. **Vocabulary** (`states.py`) — the 29 states are grouped by role:
   `GROUND`, `SENSITIZED`, `OTS_QUIESCENT/OTS_EXCITED`, `STS_QUIESCENT/STS_EXCITED`,
   `CONFLUENT`, and `CONFLUENT_EMITTING`, plus directional lookup tables.
2. **Semantics** (`rules.py`) — the transition function is written once, in terms
   of *directions*: a transmission cell is fed by any neighbour whose direction
   points at it, is erased by an excited pulse of the antagonistic family, and a
   confluent cell ANDs the ordinary lines pointing at it. `build_rule_set()`
   emits ~150 rules from this description.
3. **Engine** (`engine.py`) — a generic, rule-agnostic 5-neighbour CA compiler.
   `RuleSet.compile()` turns the rules into a flat `uint8[29^5]` table; stepping
   a grid is then a single vectorised gather (≈57M cells/s, pure NumPy, no
   native extensions).

This is both faster and more trustworthy than a hand-written cascade: the
dynamics exist in exactly one place, and that place is checked against an
external reference. Golly **Extended RLE** import/export (`rle.py`) lets real
Golly patterns load here and vice versa.

---

## 3. Project Structure

```
Mesa-von-Neumann/
├── pyproject.toml              # Build & dependency metadata
├── README.md                   # Complete architectural guide
├── PROPOSAL.md                 # Upstream proposal for projectmesa/mesa-examples
├── CONTRIBUTING.md             # Contribution guidelines
├── run_visualization.py        # CLI entry point to launch interactive visualizer
├── vn_automata/
│   ├── states.py               # The 29 states, metadata, colours, functional groups
│   ├── engine.py               # Generic 5-neighbour CA rule compiler + table engine
│   ├── rules.py                # Von Neumann's rule set, generated from the model
│   ├── model.py                # Mesa 3.x Model (VonNeumannCA) and CellAgent
│   ├── patterns.py             # Presets: Pulse loop, Confluent splitter, 1011 arm, Erasure
│   ├── loader.py               # Compact ASCII pattern parser/serializer
│   ├── rle.py                  # Golly Extended RLE import/export
│   ├── golly.py                # Independent JvN29 reference engine (validates the rules)
│   ├── data/JvN29.rule         # Bundled canonical Golly rule table (see data/NOTICE.md)
│   └── visualizer/
│       ├── server.py           # Simulation backend & REST API
│       └── static/index.html   # Canvas UI with pan/zoom, inspector, and pulse builder
└── tests/
    ├── test_states.py          # State space integrity tests
    ├── test_transitions.py     # Propagation, delay, and erasure rule tests
    ├── test_construction.py    # Complete 9-branch bifurcation tree tests
    ├── test_patterns.py        # Behavioural tests of every preset pattern
    ├── test_mesa_model.py      # Mesa Model, Agent lifecycle, and DataCollector tests
    ├── test_rle.py             # Golly Extended RLE round-trips
    └── test_golden_golly.py    # Golden tests against the canonical Golly JvN29 table
```

---

## 4. Quickstart

### Prerequisites
- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) or `pip`

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/dwan-ith/mesa-automata.git
cd mesa-automata

# Create virtual environment and install in editable mode
uv venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
uv pip install -e ".[dev]"
```

### 2. Run the Test Suite
```bash
pytest -v
```
The test suite covers the 29-state space, signal transmission (E/W/N/S),
confluent delay/splitting, antagonistic erasure, pulse-train synthesis across
all 9 target states, emergent behaviour of every preset pattern,
`step_grid`/model parity, torus wrapping, Golly RLE round-trips, and input
validation.

#### Correctness validation against Golly
Rather than trusting the generated rules, the engine is checked against the
canonical `JvN29.rule` table shipped with [Golly](http://golly.sourceforge.net/)
(bundled at `vn_automata/data/JvN29.rule`, parsed independently by
`vn_automata/golly.py`). `tests/test_golden_golly.py` asserts that the
generated rule set compiles to a table **identical to the reference over all
29⁵ = 20,511,149 neighbourhoods**, that the compiled table agrees with an
independent first-match interpreter, and that multi-step evolution of random
grids and a known construction configuration matches. This is what pins the
"faithful to von Neumann's 29-state rule" claim to an external artefact.

### 3. Launch the Interactive Visualizer
```bash
python run_visualization.py --port 8521
```
Open your browser to [http://localhost:8521](http://localhost:8521).

#### Visualizer Features:
- **Interactive Canvas:** High-DPI pan and zoom across the cell grid.
- **Playback Controls:** Play, pause, single-step, and adjust simulation speed (1–60 fps).
- **Pulse Sequence Builder:** Type any binary sequence (e.g. `1011`, `10000`, `1111`) to immediately watch the arm construct the corresponding cell.
- **Cell Inspector:** Hover over any coordinate to view its exact state, category, direction, and excitation status.
- **Interactive Brush:** Click or drag to draw or erase any of the 29 states directly on the grid.

---

## 5. Mesa 3.x Architectural Implementation

In accordance with Mesa 3.x best practices:
1. **Synchronous 2-Phase Updates:** 
   In `model.step()`, all cell agents evaluate their neighborhood in Phase 1 (`determine_next_state()`) and commit their state simultaneously in Phase 2 (`advance()`), preventing race conditions.
2. **Grid Space:** Utilizes `mesa.space.SingleGrid` with boundary options (infinite ground or torus).
3. **Data Collection:** Integrated `mesa.DataCollector` tracking `ActivePulses`, `NonGroundCells`, `SensitizedCells`, and `ConfluentCells` over time.

```python
from vn_automata import VonNeumannCA
from vn_automata.patterns import make_construction_arm_1011

# Initialize a 40x40 Mesa model with the 1011 construction arm
pattern = make_construction_arm_1011(10, 15)
model = VonNeumannCA(width=40, height=40, initial_pattern=pattern)

# Step synchronously
for step in range(5):
    model.step()
    print(f"Tick {model.step_count}: {model.count_active_pulses()} active pulses, {model.count_sensitized_cells()} sensitized cells")
```

For array-level work, or to exchange patterns with Golly:

```python
from vn_automata import cell_next_state, step_grid, parse_rle, export_rle

# One synchronous step of a whole grid (y, x), torus optional
grid2 = step_grid(grid, torus=False)

# Golly Extended RLE interoperability (understandable by Golly's JvN29)
pattern = parse_rle("x = 3, y = 1, rule = JvN29\npApD!").pattern
text = export_rle(pattern, rule="JvN29")
```

---

## 6. License

MIT License. Contributions and feedback are welcome!
