"""Time-dependent scenario waveforms for TokaGrad.

This module provides a lightweight waveform layer above the legacy static
MachineConfig/ActuatorConfig/SimulationConfig API.

A ScenarioWaveform stores control trajectories.  Each control can use either
piecewise-linear interpolation or zero-order-hold/step interpolation.  At each
time step, the solver samples the waveform and replaces selected
machine/actuator fields.  This is intentionally simple and JAX-friendly enough
for future AD control-point optimization.

Supported controls:
    machine:  Ip_MA, Bt, R0, a, kappa, delta
    actuator: P_aux_MW, greenwald_fraction_target, heat_center, heat_width, f_e_heat, nbi_birth_energy_MeV

Additional controls can be added by extending apply_waveform_controls.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import jax.numpy as jnp

from .config import MachineConfig, ActuatorConfig


@dataclass(frozen=True)
class Waveform:
    times: jnp.ndarray
    values: jnp.ndarray
    hold_left: bool = True
    hold_right: bool = True
    interpolation: str = "linear"

    def __call__(self, t):
        mode = normalize_interpolation_mode(self.interpolation)
        if mode == "step":
            return interp1d_step_hold(t, self.times, self.values, self.hold_left, self.hold_right)
        return interp1d_linear_hold(t, self.times, self.values, self.hold_left, self.hold_right)


@dataclass(frozen=True)
class ScenarioWaveform:
    controls: Mapping[str, Waveform]
    name: str = "custom"

    def value(self, key: str, t, default=None):
        wf = self.controls.get(key)
        if wf is None:
            return default
        return wf(t)


def normalize_interpolation_mode(mode: str | None) -> str:
    """Return canonical waveform interpolation mode.

    Supported user-facing aliases:
      - ``linear`` / ``interp`` / ``piecewise_linear``
      - ``step`` / ``hold`` / ``zoh`` / ``previous`` / ``piecewise_constant``

    The step mode is a left-hold / previous-value waveform: the value at
    ``times[i] <= t < times[i+1]`` is ``values[i]``.  At a knot, the new value
    is applied immediately, e.g. ``t == times[i]`` returns ``values[i]``.
    """
    m = "linear" if mode is None else str(mode).strip().lower().replace("-", "_")
    if m in {"linear", "interp", "interpolate", "piecewise_linear", "piecewise_linear_interp"}:
        return "linear"
    if m in {"step", "hold", "zero_order_hold", "zoh", "previous", "previous_hold", "piecewise_constant"}:
        return "step"
    raise ValueError(f"Unknown waveform interpolation mode: {mode!r}. Use 'linear' or 'step'.")


def interp1d_linear_hold(t, times, values, hold_left=True, hold_right=True):
    """Piecewise-linear 1D interpolation with optional endpoint hold.

    JAX's jnp.interp is used for simplicity.  It is piecewise differentiable
    with respect to values, and adequate for waveform-control optimization.
    """
    times = jnp.asarray(times)
    values = jnp.asarray(values)
    t = jnp.asarray(t)
    left = values[0] if hold_left else jnp.nan
    right = values[-1] if hold_right else jnp.nan
    return jnp.interp(t, times, values, left=left, right=right)


def interp1d_step_hold(t, times, values, hold_left=True, hold_right=True):
    """Zero-order-hold interpolation with optional endpoint hold.

    This uses the previous control value over each interval.  In other words,
    for ``times[i] <= t < times[i+1]`` it returns ``values[i]``.
    """
    times = jnp.asarray(times)
    values = jnp.asarray(values)
    t = jnp.asarray(t)
    idx = jnp.searchsorted(times, t, side="right") - 1
    idx = jnp.clip(idx, 0, values.shape[0] - 1)
    out = values[idx]
    if not hold_left:
        out = jnp.where(t < times[0], jnp.nan, out)
    if not hold_right:
        out = jnp.where(t > times[-1], jnp.nan, out)
    return out


def default_interpolation_mode_for_control(key: str, global_default: str = "linear") -> str:
    """Default interpolation mode for a waveform control.

    Most actuator/machine controls are smoothly ramped, so ``linear`` is the
    global default.  Auxiliary heating power is often commanded as on/off or
    piecewise-constant power steps, so ``P_aux_MW`` defaults to ``step`` unless
    a JSON input explicitly overrides it.
    """
    if str(key) == "P_aux_MW":
        return "step"
    return normalize_interpolation_mode(global_default)


def make_waveform(times, values, interpolation: str = "linear", hold_left: bool = True, hold_right: bool = True):
    return Waveform(
        jnp.asarray(times, dtype=jnp.float32),
        jnp.asarray(values, dtype=jnp.float32),
        hold_left=bool(hold_left),
        hold_right=bool(hold_right),
        interpolation=normalize_interpolation_mode(interpolation),
    )


def apply_waveform_controls(
    machine: MachineConfig,
    actuator: ActuatorConfig,
    waveform: ScenarioWaveform | None,
    t,
):
    """Return machine/actuator configs sampled at time t."""
    if waveform is None:
        return machine, actuator

    m_updates = {}
    a_updates = {}

    val = waveform.value("Ip_MA", t, None)
    if val is not None:
        m_updates["Ip"] = val * 1.0e6

    val = waveform.value("Bt", t, None)
    if val is not None:
        m_updates["Bt"] = val

    for key in ["R0", "a", "kappa", "delta"]:
        val = waveform.value(key, t, None)
        if val is not None:
            m_updates[key] = val

    for key in ["P_aux_MW", "greenwald_fraction_target", "heat_center", "heat_width", "f_e_heat", "nbi_birth_energy_MeV", "nbi_fast_ion_A", "nbi_fast_ion_Z"]:
        val = waveform.value(key, t, None)
        if val is not None:
            a_updates[key] = val

    m = replace(machine, **m_updates) if m_updates else machine
    a = replace(actuator, **a_updates) if a_updates else actuator
    return m, a


def iter_baseline_like_waveform(t_end: float = 20.0) -> ScenarioWaveform:
    """Example ITER-like scenario waveform.

    This is not an official ITER scenario.  It is a pedagogical waveform:
      - ramp-up Ip from 2 MA to 15 MA
      - sequentially increase auxiliary heating
      - hold flat-top
      - ramp-down Ip and heating
    """
    # Common time anchors [s]
    times = jnp.asarray([0.0, 2.0, 6.0, 10.0, 15.0, 18.0, t_end], dtype=jnp.float32)

    controls = {
        "Ip_MA": make_waveform(times, [2.0, 5.0, 15.0, 15.0, 15.0, 8.0, 2.0], interpolation="linear"),
        "P_aux_MW": make_waveform(times, [0.0, 5.0, 20.0, 50.0, 50.0, 10.0, 0.0], interpolation="step"),
        # Shape controls. R0 and a are nearly fixed by default; kappa/delta
        # ramp up during current ramp-up and relax during ramp-down.
        "R0": make_waveform(times, [6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2]),
        "a": make_waveform(times, [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
        "kappa": make_waveform(times, [1.25, 1.45, 1.70, 1.75, 1.75, 1.50, 1.25]),
        "delta": make_waveform(times, [0.10, 0.20, 0.32, 0.35, 0.35, 0.22, 0.10]),
        "greenwald_fraction_target": make_waveform(times, [0.65, 0.75, 0.85, 0.90, 0.90, 0.75, 0.65]),
        "heat_center": make_waveform(times, [0.35, 0.35, 0.30, 0.25, 0.25, 0.32, 0.35]),
        "heat_width": make_waveform(times, [0.35, 0.34, 0.30, 0.28, 0.28, 0.32, 0.35]),
        "f_e_heat": make_waveform(times, [0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60]),
        "nbi_birth_energy_MeV": make_waveform(times, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    }
    return ScenarioWaveform(controls=controls, name="iter_baseline_like")
