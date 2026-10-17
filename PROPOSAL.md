# Proposal: Von Neumann 29-State Self-Reproducing Cellular Automaton Example for Mesa

**Proposed Destination:** `mesa-examples` (Showcase / Advanced Cellular Automata)  
**Authors:** dwan-ith  
**Target Mesa Version:** Mesa 3.0+  
**Contact:** `maintainers@projectmesa.dev` / [GitHub Discussions](https://github.com/projectmesa/mesa/discussions)

---

## 1. Summary

We propose contributing a complete, verified, and interactive implementation of **John von Neumann's 29-state self-reproducing cellular automaton** to the Mesa ecosystem (specifically `mesa-examples`).

While Mesa already features Conway's Game of Life as an introductory cellular automaton, von Neumann's automaton addresses a fundamentally different and profound milestone in computational science: **universal construction, stored descriptions, and machine self-replication**.

This implementation provides:
1. A **faithful, deterministic transition engine** for all 29 states in a 5-cell von Neumann neighborhood.
2. A canonical **Mesa 3.x Agent-Based Model** with synchronous two-phase stepping (`determine_next_state` -> `advance`) and `DataCollector` metrics.
3. An **interactive canvas visualizer** enabling users to inspect states, follow the construction arm, build pulse sequences, and witness cell synthesis.
4. A **comprehensive test suite** validating the 9-branch construction tree, signal propagation, confluent delay/fan-out, and antagonistic erasure.

---

## 2. Why Von Neumann CA Belongs in Mesa

### A. Beyond Conway's Game of Life
In typical 2-state cellular automata like Conway's Life, complex behavior emerges passively from simple random seeds or handcrafted oscillators. In contrast, von Neumann's automaton is an engineered formal system demonstrating:
- **Universal Computation:** Logic gates (AND, OR, NOT, delays) constructed from directional transmission wires and confluent nodes.
- **Universal Construction:** An active arm directed by binary pulse trains that dynamically synthesizes new cellular structures out of quiescent ground cells (`U`).
- **Self-Replication:** Replicating systems that separate machine function from a stored tape description, laying the mathematical groundwork for both molecular biology (DNA/ribosomes) and computer architecture.

### B. Demonstrating Mesa 3.x Capabilities
This example showcases how Mesa 3.x gracefully handles:
- **Multi-state discrete spaces:** 29 distinct states with directional and excitation attributes.
- **Strict synchronous updates:** Zero race conditions across large grid spaces.
- **Custom domain metrics:** Tracking active pulses, sensitized cells undergoing construction, and non-ground structures via `mesa.DataCollector`.

---

## 3. Scope & Included Deliverables

| Component | Description |
| :--- | :--- |
| **`states.py`** | Type-safe `IntEnum` of all 29 states, directional metadata, functional groups, and canonical color palette matching Golly/literature standards. |
| **`engine.py`** | A generic, rule-agnostic 5-neighbour CA engine that compiles a rule set into a flat `uint8[29^5]` lookup table and steps grids with vectorised NumPy (no native extensions). |
| **`rules.py`** | Von Neumann's rule set *generated from the model* (directions, families, confluent logic) rather than transcribed, exposed as `cell_next_state(...)` / `step_grid(...)`. |
| **`model.py`** | `VonNeumannCA(mesa.Model)` and `CellAgent(mesa.Agent)` using `SingleGrid(torus=False)`. |
| **`patterns.py`** | Curated pedagogical presets: Continuous Pulse Loop, Confluent Splitter/Bridge, Construction Arm (1011), Antagonistic Wire Erasure, and Programmable Sequence Builder. |
| **`loader.py` / `rle.py`** | Compact ASCII patterns plus **Golly Extended RLE** import/export for real interoperation with the reference simulator. |
| **`golly.py` + `data/JvN29.rule`** | An independent parser/re-implementation of the canonical Golly rule table, used as an external correctness oracle. |
| **`visualizer/`** | Interactive HTML5 Canvas dashboard with pan/zoom, hover inspector, real-time metrics, and custom pulse sequence synthesizer. |
| **`tests/`** | Automated unit/integration tests verifying the 29-state definitions, wire transmission, confluent delay, the full 9-branch construction tree, emergent behaviour of every preset, RLE round-trips, and exact agreement with Golly over the complete state-transition space. |

---

## 4. Verification & Correctness

Correctness is verified against the canonical specification (Burks 1966, Buckley 2008, Golly JvN29):

0. **Exhaustive Golly Agreement (Oracle):**  
   `vn_automata/golly.py` parses the canonical `JvN29.rule` from Golly and re-implements the transition function independently. `tests/test_golden_golly.py` then asserts that the generated rule set (a) compiles to a lookup table identical to the reference over **all 29⁵ = 20,511,149 neighborhoods**, and (b) agrees with an independent first-match interpreter on random samples, and additionally matches the reference on multi-step evolution of randomized grids and a known construction configuration. The state numbering (25–28) and colour palette are taken from this same table.
1. **Sensitization Tree Verification:**  
   Every target quiescent state is verified by automated pytest parameterized tests:
   - `10000` $\to$ East Ordinary Transmission
   - `10001` $\to$ North Ordinary Transmission
   - `1001` $\to$ West Ordinary Transmission
   - `1010` $\to$ South Ordinary Transmission
   - `1011` $\to$ East Special Transmission (Arm Elongation)
   - `1100` $\to$ North Special Transmission
   - `1101` $\to$ West Special Transmission
   - `1110` $\to$ South Special Transmission
   - `1111` $\to$ Confluent State ($C_{00}$)
2. **Antagonistic Erasure:** Special and Ordinary transmission pulses mutually destroy each other back to ground state `U`.
3. **Synchronous Integrity:** Two-phase execution prevents evaluation order artifacts.

---

## 5. Proposed Integration Plan

We recommend introducing this into `mesa-examples` under `examples/von_neumann_automata/` or `examples/cellular_automata/von_neumann/`:

- **Step 1:** Community discussion on [GitHub Discussions](https://github.com/projectmesa/mesa/discussions).
- **Step 2:** Open a Pull Request on `projectmesa/mesa-examples` containing the model, test suite, and visualizer.
- **Step 3:** Provide Solara / JupyterViz bindings if requested by maintainers for alignment with Mesa's standard notebook examples.

We invite the Mesa maintainers and community to review this proposal and provide feedback on directory structure and visualization preferences.
