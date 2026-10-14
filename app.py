"""Mesa Solara visualization application (optional).

Requires the `viz` extra:

    pip install -e ".[viz]"

Run with:
    solara run app.py
"""

from typing import Dict, Any

try:
    from mesa.visualization import (
        SolaraViz,
        make_space_component,
        make_plot_component,
        Slider,
    )
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "Solara visualization requires the 'viz' extra: pip install -e \".[viz]\""
    ) from exc

from vn_automata.model import VonNeumannCA
from vn_automata.states import VNState, STATE_INFO
from vn_automata.patterns import get_pattern


def cell_portrayal(agent) -> Dict[str, Any]:
    """Portrayal function mapping CellAgent state to graphical properties."""
    meta = STATE_INFO[VNState(agent.state)]
    
    # Render non-ground cells with distinct solid squares; ground cells as subtle dots
    is_ground = (agent.state == VNState.U)
    return {
        "color": meta.hex_color,
        "size": 15 if is_ground else 55,
        "marker": "s" if not is_ground else "o",
        "alpha": 0.25 if is_ground else 1.0,
    }


def model_factory(width: int = 30, height: int = 30, pattern_name: str = "construction_arm_1011") -> VonNeumannCA:
    """Factory to initialize model with selected preset pattern."""
    pat = get_pattern(pattern_name)
    return VonNeumannCA(width=width, height=height, initial_pattern=pat)


# Model parameters configurable via Solara UI
model_params = {
    "width": Slider("Grid Width", 35, 15, 60, 5),
    "height": Slider("Grid Height", 30, 15, 60, 5),
}

# Visualization components
space_component = make_space_component(cell_portrayal)
plot_component = make_plot_component(
    {
        "ActivePulses": "#49ff49",
        "NonGroundCells": "#58a6ff",
        "SensitizedCells": "#ff9619",
        "ConfluentCells": "#00ff80",
    }
)

# Initial model instance
initial_model = model_factory(35, 30, "construction_arm_1011")

# Solara visualization page
page = SolaraViz(
    initial_model,
    components=[space_component, plot_component],
    model_params=model_params,
    name="Von Neumann 29-State Self-Reproducing Automaton",
)
