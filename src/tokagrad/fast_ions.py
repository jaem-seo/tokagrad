"""Reduced fast-ion stored-energy and beta proxies.

These helpers are intentionally lightweight and JAX-compatible.  They are used
by both the 0.5D fast model and the ordinary 1.5D diagnostics/pedestal inputs so
that beta_N can consistently mean thermal + fast beta when requested.

Reference: [J. D. Gaffey, J. Plasma Phys. 16, 149 (1976)].
The stored-energy estimate W_fast=P_fast*tau_s and the clipping bounds are
reduced TokaGrad proxies rather than a Fokker--Planck fast-ion calculation.
"""

import jax.numpy as jnp

from .grid import make_grid_from_config, axis_augmented_profile, axis_augmented_volume_element, axis_augmented_volume_element_from_dV_drho
from .heating import total_heating_sources, species_fractions, impurity_electron_fraction


def _fast_ion_critical_energy_keV(Te_keV, machine, A_fast):
    """Reduced critical energy for fast-ion slowing down [keV].

    Reference: [J. D. Gaffey, J. Plasma Phys. 16, 149 (1976)].
    """
    fH, fD, fT, fHe = species_fractions(machine)
    x_imp = impurity_electron_fraction(machine)
    Zimp = jnp.maximum(jnp.asarray(machine.impurity_Z), 1.0001)
    # Approximate sum_j n_j Z_j^2/A_j normalized to ne, with singly charged fuel.
    main_sum = fH / 1.0 + fD / 2.0 + fT / 3.0 + 4.0 * fHe / 4.0
    imp_sum = 0.5 * Zimp * x_imp
    sum_Z2_over_A = (1.0 - x_imp) * main_sum + imp_sum
    ratio = sum_Z2_over_A
    Te = jnp.maximum(Te_keV, 0.03)
    return 14.8 * A_fast * Te * ratio ** (2.0 / 3.0)


def fast_ion_slowing_down_time_profile(state, machine, sim, *, birth_energy_MeV, A_fast, Z_fast):
    """Profile-dependent reduced slowing-down time [s].

    Normalized so alpha particles in ITER-like core conditions have tau_s~0.2 s.
    The birth-energy dependence is included through a logarithmic slowdown factor.
    """
    Te = jnp.maximum(0.5 * (state.Te + state.Ti), 0.03)
    ne20 = jnp.maximum(state.ne20, 0.02)
    lnL = jnp.maximum(machine.lnLambda, 5.0)
    A = jnp.asarray(A_fast, dtype=Te.dtype)
    Z = jnp.maximum(jnp.asarray(Z_fast, dtype=Te.dtype), 0.5)
    tau_char = (
        0.20
        * (A / 4.0)
        * (2.0 / Z) ** 2
        * (Te / 10.0) ** 1.5
        / (ne20 * (lnL / 17.0) + 1.0e-12)
    )
    Ecrit = jnp.maximum(_fast_ion_critical_energy_keV(Te, machine, A), 1.0)
    Eb_keV = jnp.maximum(jnp.asarray(birth_energy_MeV, dtype=Te.dtype) * 1000.0, 1.0)
    slowdown_factor = jnp.log1p((Eb_keV / Ecrit) ** 1.5) / 3.0
    return jnp.clip(tau_char * slowdown_factor, 1.0e-5, 10.0)


def _weighted_power_tau_MJ(rho, power_density_MW_m3, tau_s, machine, dV_drho=None):
    _, w_aug = axis_augmented_profile(rho, power_density_MW_m3 * tau_s)
    if dV_drho is None:
        dV = axis_augmented_volume_element(rho, machine)
    else:
        dV = axis_augmented_volume_element_from_dV_drho(rho, dV_drho)
    return jnp.sum(w_aug * dV)


def fast_ion_energy_components_MJ(state, machine, actuator, sim, eq=None, dV_drho=None):
    """Return NBI and alpha fast-ion stored-energy proxies [MJ]."""
    rho, _, _ = make_grid_from_config(state.Te.size, machine.a, sim)
    if dV_drho is None and eq is not None:
        dV_drho = eq.dV_drho
    _Se, _Si, h = total_heating_sources(rho, state, machine, actuator, sim, eq=eq, dV_drho=dV_drho)
    Paux = h["Paux_e"] + h["Paux_i"]
    Palpha = h["Palpha_e"] + h["Palpha_i"]
    tau_nbi = fast_ion_slowing_down_time_profile(
        state, machine, sim,
        birth_energy_MeV=getattr(actuator, "nbi_birth_energy_MeV", 1.0),
        A_fast=getattr(actuator, "nbi_fast_ion_A", 2.0),
        Z_fast=getattr(actuator, "nbi_fast_ion_Z", 1.0),
    )
    tau_alpha = fast_ion_slowing_down_time_profile(
        state, machine, sim,
        birth_energy_MeV=getattr(sim, "zero_d_alpha_birth_energy_MeV", 3.52),
        A_fast=4.0,
        Z_fast=2.0,
    )
    W_nbi = getattr(sim, "zero_d_fast_nbi_fraction", 1.0) * _weighted_power_tau_MJ(rho, Paux, tau_nbi, machine, dV_drho=dV_drho)
    W_alpha = _weighted_power_tau_MJ(rho, Palpha, tau_alpha, machine, dV_drho=dV_drho)
    return jnp.maximum(W_nbi, 0.0), jnp.maximum(W_alpha, 0.0)


def fast_ion_energy_MJ(state, machine, actuator, sim, eq=None, dV_drho=None):
    W_nbi, W_alpha = fast_ion_energy_components_MJ(state, machine, actuator, sim, eq=eq, dV_drho=dV_drho)
    return W_nbi + W_alpha


def beta_normalized_total_from_thermal(beta_N_thermal, Wth_MJ, Wfast_MJ):
    return beta_N_thermal * (1.0 + Wfast_MJ / (Wth_MJ + 1.0e-12))
