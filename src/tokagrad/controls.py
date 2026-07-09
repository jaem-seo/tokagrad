"""JAX pytree control/config helpers for TokaGrad.

The original MachineConfig/ActuatorConfig dataclasses are convenient Python
configuration objects.  This module adds pytree-compatible dynamic counterparts
whose numerical leaves can participate in JAX transforms.  The objects retain
the same attribute names, so they can be passed to most existing TokaGrad
functions.

Design goal:
    - static/non-differentiable fields such as species strings are auxiliary
      metadata;
    - numerical fields are pytree leaves and may be JAX tracers;
    - conversion from the legacy configs is explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import jax
import jax.numpy as jnp

from .config import MachineConfig, ActuatorConfig


_MACHINE_NUMERIC_FIELDS = (
    "R0",
    "a",
    "kappa",
    "delta",
    "Bt",
    "Ip",
    "Zeff",
    "lnLambda",
    "dt_fraction_D",
    "dt_fraction_T",
    "impurity_Z",
)

_MACHINE_STATIC_FIELDS = (
    "plasma_species",
)

_ACTUATOR_NUMERIC_FIELDS = (
    "P_aux_MW",
    "f_e_heat",
    "nbi_birth_energy_MeV",
    "nbi_fast_ion_A",
    "nbi_fast_ion_Z",
    "heat_center",
    "heat_width",
    "greenwald_fraction_target",
    "greenwald_edge_density_fraction",
    "edge_Te_keV",
    "edge_Ti_keV",
    "cd_fraction",
    "cd_efficiency_20",
    "cd_fraction_max",
    "cd_center",
    "cd_width",
)

_ACTUATOR_STATIC_FIELDS = ("aux_partition_model",)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PytreeMachineConfig:
    R0: Any = 6.2
    a: Any = 2.0
    kappa: Any = 1.7
    delta: Any = 0.33
    Bt: Any = 5.3
    Ip: Any = 15.0e6
    Zeff: Any = 1.7
    lnLambda: Any = 17.0
    plasma_species: str = "DT"
    dt_fraction_D: Any = 0.5
    dt_fraction_T: Any = 0.5
    impurity_Z: Any = 74.0

    @classmethod
    def from_config(cls, cfg: MachineConfig):
        return cls(**{f.name: getattr(cfg, f.name) for f in fields(cls)})

    def tree_flatten(self):
        children = tuple(jnp.asarray(getattr(self, name)) for name in _MACHINE_NUMERIC_FIELDS)
        aux = {name: getattr(self, name) for name in _MACHINE_STATIC_FIELDS}
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        data = dict(zip(_MACHINE_NUMERIC_FIELDS, children))
        data.update(aux_data)
        return cls(**data)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PytreeActuatorConfig:
    P_aux_MW: Any = 50.0
    f_e_heat: Any = 0.6
    aux_partition_model: str = "fixed"
    nbi_birth_energy_MeV: Any = 1.0
    nbi_fast_ion_A: Any = 2.0
    nbi_fast_ion_Z: Any = 1.0
    heat_center: Any = 0.25
    heat_width: Any = 0.28
    greenwald_fraction_target: Any = 0.9
    greenwald_edge_density_fraction: Any = 0.25
    edge_Te_keV: Any = 0.25
    edge_Ti_keV: Any = 0.25
    cd_fraction: Any = 0.0
    cd_efficiency_20: Any = 0.30
    cd_fraction_max: Any = 0.5
    cd_center: Any = 0.35
    cd_width: Any = 0.25

    @classmethod
    def from_config(cls, cfg: ActuatorConfig):
        return cls(**{f.name: getattr(cfg, f.name) for f in fields(cls)})

    def tree_flatten(self):
        children = tuple(jnp.asarray(getattr(self, name)) for name in _ACTUATOR_NUMERIC_FIELDS)
        aux = {name: getattr(self, name) for name in _ACTUATOR_STATIC_FIELDS}
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        data = dict(zip(_ACTUATOR_NUMERIC_FIELDS, children))
        data.update(aux_data)
        return cls(**data)


def pytree_configs_from_legacy(machine: MachineConfig, actuator: ActuatorConfig):
    return PytreeMachineConfig.from_config(machine), PytreeActuatorConfig.from_config(actuator)


def replace_pytree_machine(machine: PytreeMachineConfig, **kwargs):
    data = {f.name: getattr(machine, f.name) for f in fields(PytreeMachineConfig)}
    data.update(kwargs)
    return PytreeMachineConfig(**data)


def replace_pytree_actuator(actuator: PytreeActuatorConfig, **kwargs):
    data = {f.name: getattr(actuator, f.name) for f in fields(PytreeActuatorConfig)}
    data.update(kwargs)
    return PytreeActuatorConfig(**data)


def sigmoid_bounded(z, lo, hi):
    """Smooth map from unconstrained z to [lo, hi]."""
    return lo + (hi - lo) * jax.nn.sigmoid(z)


def logit_from_bounded(x, lo, hi, eps=1.0e-6):
    y = (jnp.asarray(x) - lo) / (hi - lo + 1e-12)
    y = jnp.clip(y, eps, 1.0 - eps)
    return jnp.log(y / (1.0 - y))


CONTROL_NAMES = (
    "Ip_MA",
    "Bt",
    "greenwald_fraction_target",
    "P_aux_MW",
    "R0",
    "a",
    "kappa",
    "delta",
)

DEFAULT_CONTROL_BOUNDS = {
    "Ip_MA": (8.0, 17.0),
    "Bt": (3.0, 7.0),
    "greenwald_fraction_target": (0.5, 1.05),
    "P_aux_MW": (5.0, 120.0),
    "R0": (4.5, 7.5),
    "a": (1.0, 2.5),
    "kappa": (1.2, 2.1),
    "delta": (0.0, 0.55),
}


def control_bounds_arrays(bounds=DEFAULT_CONTROL_BOUNDS, names=CONTROL_NAMES):
    lo = jnp.asarray([bounds[n][0] for n in names], dtype=jnp.float32)
    hi = jnp.asarray([bounds[n][1] for n in names], dtype=jnp.float32)
    return lo, hi


def controls_from_unconstrained(z, bounds=DEFAULT_CONTROL_BOUNDS, names=CONTROL_NAMES):
    lo, hi = control_bounds_arrays(bounds, names)
    return sigmoid_bounded(jnp.asarray(z, dtype=jnp.float32), lo, hi)


def unconstrained_from_controls(x_phys, bounds=DEFAULT_CONTROL_BOUNDS, names=CONTROL_NAMES):
    lo, hi = control_bounds_arrays(bounds, names)
    return logit_from_bounded(jnp.asarray(x_phys, dtype=jnp.float32), lo, hi)


def apply_control_vector(
    x_phys,
    machine: PytreeMachineConfig,
    actuator: PytreeActuatorConfig,
    sim,
    names=CONTROL_NAMES,
):
    """Apply physical controls to pytree configs.

    `sim` remains the legacy frozen SimulationConfig because model choices and
    grid sizes are intentionally static.  If a control modifies a SimulationConfig
    scalar, a dataclass replace is used; the scalar may still be a JAX tracer.
    """
    c = {name: x_phys[i] for i, name in enumerate(names)}
    m = machine
    a = actuator
    s = sim

    if "Ip_MA" in c:
        m = replace_pytree_machine(m, Ip=c["Ip_MA"] * 1.0e6)
    if "Bt" in c:
        m = replace_pytree_machine(m, Bt=c["Bt"])
    if "R0" in c:
        m = replace_pytree_machine(m, R0=c["R0"])
    if "a" in c:
        m = replace_pytree_machine(m, a=c["a"])
    if "kappa" in c:
        m = replace_pytree_machine(m, kappa=c["kappa"])
    if "delta" in c:
        m = replace_pytree_machine(m, delta=c["delta"])
    if "P_aux_MW" in c:
        a = replace_pytree_actuator(a, P_aux_MW=c["P_aux_MW"])
    if "heat_center" in c:
        a = replace_pytree_actuator(a, heat_center=c["heat_center"])
    if "greenwald_fraction_target" in c:
        a = replace_pytree_actuator(a, greenwald_fraction_target=c["greenwald_fraction_target"])
    return m, a, s
