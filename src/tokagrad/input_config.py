"""JSON input-file helpers for TokaGrad demo scripts.

The input format is intentionally small and dependency-free.  A file may contain
``machine``, ``actuator``, ``simulation`` and, for waveform runs, ``waveform``
sections.  Unknown keys raise a ValueError so typos do not silently change a run.

Example::

    {
      "machine": {"R0": 6.2, "a": 2.0, "Ip_MA": 15.0, "Bt": 5.3},
      "actuator": {"P_aux_MW": 50.0},
      "simulation": {"nr": 48, "n_steps": 500},
      "waveform": {
        "name": "iter_like",
        "times": [0, 2, 6, 10],
        "interpolation": {"default": "linear", "P_aux_MW": "step"},
        "controls": {"Ip_MA": [2, 5, 15, 15], "P_aux_MW": [0, 5, 20, 50]}
      }
    }
"""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
import json
import math
from typing import Any, Mapping

import jax.numpy as jnp

from .config import MachineConfig, ActuatorConfig, SimulationConfig
from .waveforms import (
    ScenarioWaveform,
    make_waveform,
    iter_baseline_like_waveform,
    default_interpolation_mode_for_control,
    normalize_interpolation_mode,
)

from .solver import greenwald_target_density_1e20

from .zero_d import zero_d_enabled, ipb98y2_tau_E

def _dataclass_field_names(obj_or_cls) -> set[str]:
    return {f.name for f in fields(obj_or_cls)}


def _resolve_path(value: str, base_dir: Path) -> str:
    if not value:
        return value
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return str(p)


def _min_physical_cell_width(sim: SimulationConfig, machine: MachineConfig) -> float:
    """Return min radial cell width [m] using the same face grid as grid.py."""
    nr = max(int(sim.nr), 1)
    if getattr(sim, "radial_grid", "uniform") in ("uniform", "linear"):
        return float(machine.a) / float(nr)
    if getattr(sim, "radial_grid", "uniform") in ("edge_cluster_sqrt", "edge_clustered", "sqrt_edge"):
        p = float(getattr(sim, "edge_cluster_power", 2.0))
        xs = [i / nr for i in range(nr + 1)]
        faces = [math.sqrt(max(1.0 - (1.0 - x) ** p, 0.0)) for x in xs]
        return float(machine.a) * min(max(faces[i + 1] - faces[i], 1.0e-12) for i in range(nr))
    raise ValueError(f"Unknown radial_grid={getattr(sim, 'radial_grid', None)!r}")


def resolve_time_discretization(sim: SimulationConfig, machine: MachineConfig, actuator: ActuatorConfig) -> SimulationConfig:
    """Resolve dt/n_steps from dt_mode and end_time_s.

    This keeps old input files working while allowing two new workflows:
      * dt_mode="cfl": dt = CFL * min(dx)^2 / D_ref, with D_ref known only as
        a pre-run order-of-magnitude estimate.
      * end_time_s>0: derive n_steps from the resolved dt.  By default, dt is
        reduced slightly so n_steps*dt equals end_time_s exactly.
    """
    dt = float(sim.dt)
    if zero_d_enabled(sim):
        nbar20_proxy = greenwald_target_density_1e20(machine, actuator, sim)
        tau_E_proxy = ipb98y2_tau_E(machine, nbar20_proxy, actuator.P_aux_MW, sim)
        dt = min(sim.zero_d_dt_fraction_tauE * tau_E_proxy, dt)
    else:
        mode = str(getattr(sim, "dt_mode", "fixed")).lower()
        if mode in ("cfl", "diffusion_cfl", "auto_cfl"):
            dx = _min_physical_cell_width(sim, machine)
            Dref = max(float(getattr(sim, "cfl_diffusivity_ref_m2_s", 1.0)), 1.0e-12)
            cfl = max(float(getattr(sim, "cfl_number", 0.25)), 1.0e-12)
            dt = cfl * dx * dx / Dref
        elif mode in ("fixed", "manual", "constant"):
            dt = max(dt, 1.0e-12)
        else:
            raise ValueError(f"Unknown SimulationConfig.dt_mode={mode!r}; use 'fixed' or 'cfl'.")

    end_time = float(getattr(sim, "end_time_s", 0.0))
    n_steps = int(sim.n_steps)
    if end_time > 0.0:
        n_steps = max(1, int(math.ceil(end_time / max(dt, 1.0e-12))))
        if bool(getattr(sim, "adjust_dt_to_end_time", True)):
            dt = end_time / n_steps
    return replace(sim, dt=float(dt), n_steps=int(n_steps))


def _update_machine(machine: MachineConfig, updates: Mapping[str, Any]) -> MachineConfig:
    allowed = _dataclass_field_names(MachineConfig)
    out: dict[str, Any] = {}
    for key, val in updates.items():
        if key == "Ip_MA":
            out["Ip"] = float(val) * 1.0e6
        elif key == "impurity_Z_eff_for_line_rad" and "impurity_Z" in allowed:
            out["impurity_Z"] = val
        elif key in allowed:
            out[key] = val
        else:
            raise ValueError(f"Unknown MachineConfig key in input file: {key!r}")
    return replace(machine, **out) if out else machine


def _update_actuator(actuator: ActuatorConfig, updates: Mapping[str, Any]) -> ActuatorConfig:
    allowed = _dataclass_field_names(ActuatorConfig)
    out: dict[str, Any] = {}
    for key, val in updates.items():
        if key in allowed:
            out[key] = val
        elif key == "nbar20":
            # Deprecated absolute density input; prefer greenwald_fraction_target.
            continue
        elif key == "edge_ne20":
            # Deprecated edge density input; now derived from Greenwald fraction.
            continue
        else:
            raise ValueError(f"Unknown ActuatorConfig key in input file: {key!r}")
    return replace(actuator, **out) if out else actuator


def _update_simulation(sim: SimulationConfig, updates: Mapping[str, Any], base_dir: Path) -> SimulationConfig:
    allowed = _dataclass_field_names(SimulationConfig)
    out: dict[str, Any] = {}
    deprecated_ignored = {"zero_d_fast_ion_tau_s", "derive_density_from_greenwald"}
    for key, val in updates.items():
        if key in deprecated_ignored:
            # Slowing-down time is now computed from the reconstructed profiles.
            continue
        if key == "end_time":
            key = "end_time_s"
        # Short aliases for expensive-submodel update cadences and common option names.
        skip_aliases = {
            "geqdsk_q_source": "geqdsk_q_profile_source",
            "q_profile_source": "geqdsk_q_profile_source",
            "equilibrium_skip": "equilibrium_skip_steps",
            "source_skip": "source_skip_steps",
            "transport_skip": "transport_skip_steps",
            "pedestal_skip": "pedestal_skip_steps",
            "lh_gate_skip": "pedestal_skip_steps",
            "pedestal_lh_gate_skip": "pedestal_skip_steps",
            "equilibrium_update_interval": "equilibrium_skip_steps",
            "source_update_interval": "source_skip_steps",
            "transport_update_interval": "transport_skip_steps",
            "pedestal_update_interval": "pedestal_skip_steps",
        }
        key = skip_aliases.get(key, key)
        if key not in allowed:
            raise ValueError(f"Unknown SimulationConfig key in input file: {key!r}")
        if key == "save_result_times_s":
            if val is None or val == "":
                out[key] = ()
            elif isinstance(val, (list, tuple)):
                out[key] = tuple(float(x) for x in val)
            else:
                out[key] = (float(val),)
        elif key.endswith("_path") or key in {"geqdsk_path"}:
            out[key] = _resolve_path(str(val), base_dir)
        else:
            out[key] = val
    return replace(sim, **out) if out else sim




def _move_density_controls_from_sim_to_actuator(data: dict[str, Any]) -> None:
    """Backward-compatible migration for old input files.

    Older JSON files stored density targets under ``simulation``.  Density
    targets now live under ``actuator``.  Likewise, the 0.5D NBI birth energy
    has been unified with the actuator-level NBI/fast-ion birth energy used by
    auxiliary heating partitioning.
    """
    sim_sec = data.get("simulation") or {}
    if not isinstance(sim_sec, Mapping):
        return
    act_sec = data.setdefault("actuator", {})
    if not isinstance(act_sec, dict):
        return
    for key in ("greenwald_fraction_target", "greenwald_edge_density_fraction"):
        if key in sim_sec and key not in act_sec:
            act_sec[key] = sim_sec[key]
        if key in sim_sec:
            sim_sec.pop(key, None)

def _load_json(path: str | Path) -> tuple[dict[str, Any], Path]:
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Input file must contain a JSON object: {p}")
    return data, p.parent


def load_static_input(path: str | Path,
                      machine: MachineConfig | None = None,
                      actuator: ActuatorConfig | None = None,
                      sim: SimulationConfig | None = None):
    """Load machine/actuator/simulation sections from a JSON input file."""
    data, base_dir = _load_json(path)
    _move_density_controls_from_sim_to_actuator(data)
    machine = MachineConfig() if machine is None else machine
    actuator = ActuatorConfig() if actuator is None else actuator
    sim = SimulationConfig() if sim is None else sim

    if "machine" in data:
        machine = _update_machine(machine, data["machine"] or {})
    if "actuator" in data:
        actuator = _update_actuator(actuator, data["actuator"] or {})
    if "simulation" in data:
        sim = _update_simulation(sim, data["simulation"] or {}, base_dir)
    sim = resolve_time_discretization(sim, machine, actuator)
    return machine, actuator, sim


def waveform_from_input_section(section: Mapping[str, Any] | None, t_end: float | None = None) -> ScenarioWaveform:
    """Build a ScenarioWaveform from a JSON ``waveform`` section.

    Supported compact format::

        {"times": [0, 1], "controls": {"Ip_MA": [2, 15]}}

    Per-control times and interpolation modes are also accepted::

        {"controls": {"Ip_MA": {"times": [0, 1], "values": [2, 15], "interpolation": "linear"}}}

    A section-wide interpolation map can be used too::

        {"interpolation": {"default": "linear", "P_aux_MW": "step"}, ...}

    If no explicit mode is provided, most controls default to ``linear`` while
    ``P_aux_MW`` defaults to zero-order-hold ``step``.
    """
    if not section:
        return iter_baseline_like_waveform(t_end=20.0 if t_end is None else float(t_end))
    name = str(section.get("name", "input_waveform"))
    global_times = section.get("times", None)
    interp_section = section.get("interpolation", section.get("interpolation_modes", None))
    if interp_section is None:
        interp_default = "linear"
        interp_by_key = {}
    elif isinstance(interp_section, str):
        interp_default = normalize_interpolation_mode(interp_section)
        interp_by_key = {}
    elif isinstance(interp_section, Mapping):
        interp_default = normalize_interpolation_mode(interp_section.get("default", "linear"))
        interp_by_key = {str(k): normalize_interpolation_mode(v) for k, v in interp_section.items() if str(k) != "default"}
    else:
        raise ValueError("waveform.interpolation must be a string or JSON object")
    controls_in = section.get("controls", {})
    if not isinstance(controls_in, Mapping):
        raise ValueError("waveform.controls must be a JSON object")
    controls = {}
    for key, spec in controls_in.items():
        key_str = str(key)
        explicit_mode = None
        hold_left = True
        hold_right = True
        if isinstance(spec, Mapping):
            times = spec.get("times", global_times)
            values = spec.get("values", None)
            explicit_mode = spec.get("interpolation", spec.get("mode", None))
            hold_left = bool(spec.get("hold_left", True))
            hold_right = bool(spec.get("hold_right", True))
            if values is None:
                raise ValueError(f"waveform control {key!r} needs a values array")
        else:
            times = global_times
            values = spec
        if times is None:
            raise ValueError(f"waveform control {key!r} needs times, either global or per-control")
        if explicit_mode is not None:
            mode = normalize_interpolation_mode(explicit_mode)
        elif key_str in interp_by_key:
            mode = interp_by_key[key_str]
        else:
            mode = default_interpolation_mode_for_control(key_str, interp_default)
        controls[key_str] = make_waveform(
            jnp.asarray(times, dtype=jnp.float32),
            jnp.asarray(values, dtype=jnp.float32),
            interpolation=mode,
            hold_left=hold_left,
            hold_right=hold_right,
        )
    return ScenarioWaveform(controls=controls, name=name)


def load_waveform_input(path: str | Path,
                        machine: MachineConfig | None = None,
                        actuator: ActuatorConfig | None = None,
                        sim: SimulationConfig | None = None):
    """Load machine/actuator/simulation/waveform sections from a JSON input file."""
    data, base_dir = _load_json(path)
    _move_density_controls_from_sim_to_actuator(data)
    machine = MachineConfig() if machine is None else machine
    actuator = ActuatorConfig() if actuator is None else actuator
    sim = SimulationConfig() if sim is None else sim

    if "machine" in data:
        machine = _update_machine(machine, data["machine"] or {})
    if "actuator" in data:
        actuator = _update_actuator(actuator, data["actuator"] or {})
    if "simulation" in data:
        sim = _update_simulation(sim, data["simulation"] or {}, base_dir)
    sim = resolve_time_discretization(sim, machine, actuator)
    waveform = waveform_from_input_section(data.get("waveform"), t_end=float(sim.dt) * int(sim.n_steps))
    return machine, actuator, sim, waveform


__all__ = ["load_static_input", "load_waveform_input", "waveform_from_input_section", "resolve_time_discretization"]
