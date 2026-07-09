"""Density target helpers for Greenwald-based controls.

ActuatorConfig is the owner of density targets.  This module keeps backward
compatibility with older input files that may still contain nbar20/edge_ne20
or SimulationConfig.greenwald_fraction_target.

Reference: [M. Greenwald et al., Nucl. Fusion 28, 2199 (1988)].
Target fractions, edge fractions, and feedback/rescaling laws are TokaGrad
control closures rather than a predictive density-limit model.
"""
from __future__ import annotations

import jax.numpy as jnp


def greenwald_density_1e20(machine):
    """Greenwald density in units of 1e20 m^-3: Ip[MA] / (pi a^2)."""
    return (machine.Ip / 1.0e6) / (jnp.pi * machine.a**2 + 1.0e-12)


def greenwald_fraction_target(actuator, sim=None):
    if hasattr(actuator, "greenwald_fraction_target"):
        return jnp.asarray(actuator.greenwald_fraction_target)
    if sim is not None and hasattr(sim, "greenwald_fraction_target"):
        return jnp.asarray(sim.greenwald_fraction_target)
    return jnp.asarray(0.9)


def greenwald_edge_density_fraction(actuator, sim=None):
    if hasattr(actuator, "greenwald_edge_density_fraction"):
        return jnp.asarray(actuator.greenwald_edge_density_fraction)
    if sim is not None and hasattr(sim, "greenwald_edge_density_fraction"):
        return jnp.asarray(sim.greenwald_edge_density_fraction)
    return jnp.asarray(0.25)


def target_nbar20(machine, actuator, sim=None):
    if hasattr(actuator, "nbar20") and not hasattr(actuator, "greenwald_fraction_target"):
        return jnp.asarray(actuator.nbar20)
    return greenwald_fraction_target(actuator, sim) * greenwald_density_1e20(machine)


def target_edge_ne20(machine, actuator, sim=None):
    if hasattr(actuator, "edge_ne20") and not hasattr(actuator, "greenwald_edge_density_fraction"):
        return jnp.asarray(actuator.edge_ne20)
    return greenwald_edge_density_fraction(actuator, sim) * target_nbar20(machine, actuator, sim)
