"""Mesa 3.x model and cell agents for Von Neumann 29-State CA."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import mesa

from .states import VNState, StateCategory, STATE_INFO
from .rules import cell_next_state


class CellAgent(mesa.Agent):
    """An individual finite-state cellular agent in von Neumann's 29-state space."""

    def __init__(
        self,
        model: mesa.Model,
        state: int = VNState.U,
    ) -> None:
        super().__init__(model)
        self.state: int = state
        self.next_state: int = state

    @property
    def category(self) -> StateCategory:
        return STATE_INFO[VNState(self.state)].category

    @property
    def is_excited(self) -> bool:
        return STATE_INFO[VNState(self.state)].is_excited

    def determine_next_state(self) -> None:
        """Phase 1: synchronous evaluation based on 5-cell von Neumann neighborhood."""
        x, y = self.pos
        w, h = self.model.width, self.model.height
        torus = self.model.grid.torus
        cell_map = self.model.cell_map

        # Von Neumann neighborhood coordinates (Mesa Cartesian, y up):
        # north: (x, y+1), east: (x+1, y), south: (x, y-1), west: (x-1, y)
        if torus:
            n_val = cell_map[(x, (y + 1) % h)].state
            e_val = cell_map[((x + 1) % w, y)].state
            s_val = cell_map[(x, (y - 1) % h)].state
            w_val = cell_map[((x - 1) % w, y)].state
        else:
            n_val = cell_map[(x, y + 1)].state if y + 1 < h else VNState.U
            e_val = cell_map[(x + 1, y)].state if x + 1 < w else VNState.U
            s_val = cell_map[(x, y - 1)].state if y - 1 >= 0 else VNState.U
            w_val = cell_map[(x - 1, y)].state if x - 1 >= 0 else VNState.U

        self.next_state = cell_next_state(self.state, n_val, e_val, s_val, w_val)

    def advance(self) -> None:
        """Phase 2: commit state update synchronously."""
        self.state = self.next_state


class VonNeumannCA(mesa.Model):
    """Mesa Model implementing John von Neumann's 29-State Self-Reproducing Automaton.
    
    Provides synchronous 2-phase updates, grid management, pattern loading,
    and metrics collection adhering to Mesa 3.x standards.
    """

    def __init__(
        self,
        width: int = 50,
        height: int = 50,
        torus: bool = False,
        initial_pattern: Optional[Dict[Tuple[int, int], int]] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(seed=seed)
        self.width = width
        self.height = height
        self.torus = torus
        self.step_count = 0

        # Initialize single-occupancy grid
        self.grid = mesa.space.SingleGrid(width, height, torus=torus)

        # Place cells at all grid coordinates
        self.cell_map: Dict[Tuple[int, int], CellAgent] = {}
        for x in range(width):
            for y in range(height):
                state = VNState.U
                if initial_pattern and (x, y) in initial_pattern:
                    state = initial_pattern[(x, y)]
                agent = CellAgent(self, state=state)
                self.grid.place_agent(agent, (x, y))
                self.cell_map[(x, y)] = agent

        # Setup Mesa DataCollector for quantitative CA metrics
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "ActivePulses": lambda m: m.count_active_pulses(),
                "NonGroundCells": lambda m: m.count_non_ground_cells(),
                "SensitizedCells": lambda m: m.count_sensitized_cells(),
                "ConfluentCells": lambda m: m.count_confluent_cells(),
            }
        )
        self.datacollector.collect(self)

    def count_active_pulses(self) -> int:
        """Count cells currently carrying an excited pulse.

        Only ordinary/special transmission excited states and emitting
        confluent states (C10, C11) count. Sensitized intermediates (S..S000)
        are construction transients, not pulses.
        """
        return sum(
            1
            for a in self.agents
            if a.state in (13, 14, 15, 16, 21, 22, 23, 24, 26, 28)
        )

    def count_non_ground_cells(self) -> int:
        """Count cells that are not in the quiescent ground state U."""
        return sum(1 for a in self.agents if a.state != VNState.U)

    def count_sensitized_cells(self) -> int:
        """Count cells currently undergoing construction differentiation (states S..S000)."""
        return sum(1 for a in self.agents if 1 <= a.state <= 8)

    def count_confluent_cells(self) -> int:
        """Count cells in any confluent state (C00, C01, C10, C11)."""
        return sum(1 for a in self.agents if 25 <= a.state <= 28)

    def step(self) -> None:
        """Execute one complete synchronous clock tick across the entire cell space."""
        # Phase 1: all cells determine their next state based on current generation
        for agent in self.agents:
            agent.determine_next_state()

        # Phase 2: all cells synchronously commit their state
        for agent in self.agents:
            agent.advance()

        self.step_count += 1
        self.datacollector.collect(self)

    def set_cell(self, x: int, y: int, state: int) -> None:
        """Directly set the state of a cell at (x, y)."""
        if 0 <= x < self.width and 0 <= y < self.height:
            agent = self.cell_map[(x, y)]
            agent.state = state
            agent.next_state = state

    def get_cell(self, x: int, y: int) -> int:
        """Get the state of the cell at (x, y)."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.cell_map[(x, y)].state
        return VNState.U

    def load_pattern(self, pattern: Dict[Tuple[int, int], int], offset: Tuple[int, int] = (0, 0)) -> None:
        """Load a pattern dictionary of (x, y) -> state into the grid at given offset."""
        ox, oy = offset
        for (x, y), st in pattern.items():
            tx, ty = x + ox, y + oy
            if 0 <= tx < self.width and 0 <= ty < self.height:
                self.set_cell(tx, ty, st)

    def to_numpy(self) -> np.ndarray:
        """Export grid states as a 2D numpy array with shape (height, width)."""
        arr = np.zeros((self.height, self.width), dtype=np.int32)
        for (x, y), agent in self.cell_map.items():
            arr[y, x] = agent.state
        return arr

    def from_numpy(self, arr: np.ndarray) -> None:
        """Load grid states from a 2D numpy array."""
        h, w = arr.shape
        for y in range(min(h, self.height)):
            for x in range(min(w, self.width)):
                self.set_cell(x, y, int(arr[y, x]))
