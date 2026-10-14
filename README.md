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
| **Confluent Nodes** | `C00, C10, C01, C11` (25–28) | 4 | Signal splitters (fan-out), ordinary-to-special conversion bridges, and 2-input AND gates with 1-cycle delay. |

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
│   ├── states.py               # Enum of 29 states, metadata, symbols, and colors
│   ├── rules.py                # Pure, deterministic transition functions & Numba-accelerated kernel
│   ├── model.py                # Mesa 3.x Model (VonNeumannCA) and CellAgent
│   ├── patterns.py             # Presets: Pulse loop, Confluent splitter, 1011 arm, Erasure
│   ├── loader.py               # ASCII / RLE format parser and serializer
│   └── visualizer/
│       ├── server.py           # Simulation backend & REST API
│       └── static/index.html   # Canvas UI with pan/zoom, inspector, and pulse builder
└── tests/
    ├── test_states.py          # State space integrity tests
    ├── test_transitions.py     # Propagation, delay, and erasure rule tests
    ├── test_construction.py    # Complete 9-branch bifurcation tree tests
    └── test_mesa_model.py      # Mesa Model, Agent lifecycle, and DataCollector tests
```

---

## 4. Quickstart

### Prerequisites
- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) or `pip`

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/your-username/mesa-von-neumann.git
cd mesa-von-neumann

# Create virtual environment and install in editable mode
uv venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
uv pip install -e ".[dev]"
```

### 2. Run the Test Suite
```bash
pytest -v
```
All 30+ unit and integration tests validate the 29-state space, signal transmission (E/W/N/S), confluent delay/splitting, antagonistic erasure, pulse-train synthesis across all 9 target states, `step_grid`/model parity, torus wrapping, loader round-trips, and input validation.

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

---

## 6. License

MIT License. Contributions and feedback are welcome!
