"""JIT-compiled helpers for the Streamlit app runtime.

Keeping these helpers in an importable module rather than defining them inside
``app_streamlit.py`` lets JAX keep the compiled executables across Streamlit
reruns.  The app supplies a SimulationConfig with ``n_steps`` set to the desired
chunk length; the helpers advance the current state by that chunk in one XLA
program.
"""
from __future__ import annotations

from .config import MachineConfig, ActuatorConfig, SimulationConfig
from .profiles import PlasmaState
from .solver import simulate_final_jit
from .zero_d import simulate_0d_final_with_elapsed_jit


def advance_radial_chunk_jit(
    state: PlasmaState,
    machine: MachineConfig,
    actuator: ActuatorConfig,
    sim: SimulationConfig,
) -> PlasmaState:
    """Advance a 1.5D state by ``sim.n_steps`` with one JIT-compiled call."""
    return simulate_final_jit(machine, actuator, sim, state0=state)


def advance_zero_d_chunk_with_time_jit(
    state: PlasmaState,
    machine: MachineConfig,
    actuator: ActuatorConfig,
    sim: SimulationConfig,
):
    """Advance a 0.5D state and return ``(new_state, elapsed_physical_time_s)``."""
    return simulate_0d_final_with_elapsed_jit(machine, actuator, sim, state0=state)
