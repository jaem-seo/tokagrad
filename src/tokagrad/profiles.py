"""Plasma state and reduced initial-profile construction.

References:
  [M. Greenwald et al., Nucl. Fusion 28, 2199 (1988)] -- density normalization.
  [ITER Physics Basis Editors, Nucl. Fusion 39, 2175 (1999)] -- ITER profiles.

The H-mode/parabolic shapes and beta_N rescaling are initialization heuristics;
they are not solutions of transport or equilibrium equations.
"""

import jax.numpy as jnp
from typing import NamedTuple
from .current import initial_psi_from_total_current, initial_psi_iter_hmode
from .density import target_nbar20 as _target_nbar20_helper, target_edge_ne20 as _target_edge_ne20_helper

MU0 = 4.0e-7 * jnp.pi
KEV_TO_J = 1.602176634e-16

class PlasmaState(NamedTuple):
    Te: jnp.ndarray        # keV
    Ti: jnp.ndarray        # keV
    ne20: jnp.ndarray      # 1e20 m^-3
    psi_ind: jnp.ndarray   # poloidal-flux proxy; face grid length nr+1 by default
    psi_edge: jnp.ndarray  # scalar edge/transformer-flux boundary state
    Phi_b_prev: jnp.ndarray = jnp.asarray(0.0)
    # Reduced toroidal flux at the previous step, used to estimate Phi_dot/Phi.
    dV_drho_prev: jnp.ndarray = jnp.asarray(0.0)
    # Previous-step V'(rho) [m^3], required by conservative n and n*T transients.

def parabolic_profile(rho, core, edge, alpha=1.6):
    return edge + (core - edge) * (1.0 - rho**2) ** alpha

def pedestal_profile(rho, core, pedestal_top, edge, rho_top=0.95, width=0.045, core_alpha=1.4):
    """Smooth ITER-H-mode-like core plus pedestal profile."""
    core_shape = pedestal_top + (core - pedestal_top) * jnp.maximum(1.0 - (rho / rho_top) ** 2, 0.0) ** core_alpha
    #ped_s = 0.5 * (1.0 - jnp.tanh((rho - rho_top) / (0.5 * width + 1e-8)))
    rho_ped = 1.0 - 0.5 * (1.0 - rho_top) # center of the pedestal
    ped_s = 0.5 * (1.0 - jnp.tanh((rho - rho_ped) / (0.5 * width + 1e-8)))
    return edge + (core_shape - edge) * ped_s


def _infer_rho_faces(rho):
    """Infer normalized cell faces from cell centers.

    This mirrors tokagrad.grid.infer_rho_faces but is kept local to avoid making
    profile initialization depend on solver utilities.  It works for both uniform
    and edge-clustered cell-centered grids.
    """
    rho = jnp.asarray(rho)
    mids = 0.5 * (rho[:-1] + rho[1:])
    return jnp.concatenate([
        jnp.asarray([0.0], dtype=rho.dtype),
        mids,
        jnp.asarray([1.0], dtype=rho.dtype),
    ])


def _volume_weights(rho):
    """Reduced shell-volume weights proportional to d(rho^2)."""
    faces = _infer_rho_faces(rho)
    return jnp.maximum(faces[1:] ** 2 - faces[:-1] ** 2, 1.0e-12)


def _weighted_avg(y, rho):
    w = _volume_weights(rho)
    return jnp.sum(y * w) / (jnp.sum(w) + 1.0e-12)


def _cell_average_profile(rho, profile_fn, sim=None):
    """Return finite-volume cell averages of a smooth radial profile.

    Very coarse grids sample the pedestal poorly if a profile is evaluated only
    at cell centers.  For example, with nr=4 the outer cell center is rho=0.875,
    so a steep edge pedestal is missed almost completely.  Initial conditions are
    therefore built from sub-cell quadrature averages over each cell.  This keeps
    the coarse-grid density/temperature amplitudes much closer to high-nr runs
    while still returning one cell-centered state value per radial cell.
    """
    faces = _infer_rho_faces(rho)
    n_q = int(getattr(sim, "initial_profile_quadrature_points", 16)) if sim is not None else 16
    n_q = max(n_q, 1)
    # Midpoint quadrature inside each cell.  Weight by rho to approximate the
    # toroidal shell-volume average; the common geometric prefactor cancels.
    xi = (jnp.arange(n_q, dtype=rho.dtype) + 0.5) / float(n_q)
    rr = faces[:-1, None] + (faces[1:] - faces[:-1])[:, None] * xi[None, :]
    vals = profile_fn(rr)
    weights = jnp.maximum(rr, 1.0e-8)
    return jnp.sum(vals * weights, axis=1) / (jnp.sum(weights, axis=1) + 1.0e-12)


def _target_nbar20(machine, actuator, sim):
    return _target_nbar20_helper(machine, actuator, sim)


def _target_edge_ne20(machine, actuator, sim):
    return _target_edge_ne20_helper(machine, actuator, sim)


def _thermal_energy_MJ_from_profiles(rho, Te, Ti, ne20, machine):
    w = _volume_weights(rho)
    # Convert reduced rho^2 weights into approximate volume using total torus volume.
    volume = 2.0 * jnp.pi**2 * machine.R0 * machine.kappa * machine.a**2
    dV = volume * w / (jnp.sum(w) + 1.0e-12)
    W_J = 1.5 * jnp.sum(ne20 * 1.0e20 * (Te + Ti) * KEV_TO_J * dV)
    return W_J / 1.0e6


def _beta_N_thermal_from_profiles(rho, Te, Ti, ne20, machine):
    w = _volume_weights(rho)
    p_avg_keV20 = jnp.sum(ne20 * (Te + Ti) * w) / (jnp.sum(w) + 1.0e-12)
    p_avg = p_avg_keV20 * 1.0e20 * KEV_TO_J
    beta = 2.0 * MU0 * p_avg / (machine.Bt**2 + 1.0e-12)
    beta_percent = 100.0 * beta
    return beta_percent * machine.a * machine.Bt / (machine.Ip / 1.0e6 + 1.0e-12)


def _beta_N_total_from_profiles(rho, Te, Ti, ne20, machine, actuator, sim):
    beta_th = _beta_N_thermal_from_profiles(rho, Te, Ti, ne20, machine)
    try:
        from .fast_ions import fast_ion_energy_MJ, beta_normalized_total_from_thermal
        psi_edge = jnp.asarray(0.0)
        psi_ind = initial_psi_iter_hmode(rho, machine, psi_edge=psi_edge, sim=sim)
        tmp = PlasmaState(Te, Ti, ne20, psi_ind, psi_edge)
        Wth = _thermal_energy_MJ_from_profiles(rho, Te, Ti, ne20, machine)
        Wfast = fast_ion_energy_MJ(tmp, machine, actuator, sim)
        return beta_normalized_total_from_thermal(beta_th, Wth, Wfast)
    except Exception:
        return beta_th


def _target_pressure_avg_from_beta_N(machine, sim):
    betaN = getattr(sim, "initial_beta_N_target", 1.82) if sim is not None else 1.82
    Ip_MA = machine.Ip / 1.0e6
    beta_frac = betaN * Ip_MA / (machine.a * machine.Bt + 1.0e-12) / 100.0
    return beta_frac * machine.Bt**2 / (2.0 * MU0 + 1.0e-30)


def _rescale_density_shape_to_greenwald(rho, base_ne20, machine, actuator, sim):
    """Keep ITER-like ne shape but enforce target f_G and edge ne."""
    if sim is None or not getattr(sim, "derive_density_from_greenwald", True):
        return base_ne20
    nbar_tgt = _target_nbar20(machine, actuator, sim)
    edge_tgt = _target_edge_ne20(machine, actuator, sim)
    # Use the original shape only through its excess above the *physical* edge
    # target, not base_ne20[-1].  On coarse grids the last cell center can sit
    # well inside the pedestal top (e.g. rho=0.875 for nr=4), so treating the
    # last cell as the edge makes the excess almost vanish and causes a large
    # artificial density amplification.
    excess = jnp.maximum(base_ne20 - edge_tgt, 0.0)
    avg_excess = _weighted_avg(excess, rho)
    amp = jnp.maximum((nbar_tgt - edge_tgt) / (avg_excess + 1.0e-12), 0.0)
    return jnp.maximum(edge_tgt + amp * excess, 1.0e-5)


def _rescale_temperature_to_beta_N(rho, Te, Ti, ne20, machine, actuator, sim):
    """Keep Te/Ti shapes but rescale excess to approximately preserve total beta_N.

    The first estimate matches the requested beta_N thermally.  A few cheap
    correction iterations then account for the reduced fast-ion beta proxy from
    NBI and alpha power, so ``initial_beta_N_target`` is interpreted as
    thermal+fast beta_N.
    """
    if sim is None or not getattr(sim, "initial_temperature_rescale_to_beta_N", True):
        return Te, Ti
    p_tgt = _target_pressure_avg_from_beta_N(machine, sim)
    beta_tgt = jnp.asarray(getattr(sim, "initial_beta_N_target", 1.82), dtype=Te.dtype)
    Te_edge = jnp.asarray(actuator.edge_Te_keV)
    Ti_edge = jnp.asarray(actuator.edge_Ti_keV)
    Te_excess = jnp.maximum(Te - Te_edge, 0.0)
    Ti_excess = jnp.maximum(Ti - Ti_edge, 0.0)
    edge_term = _weighted_avg(ne20 * (Te_edge + Ti_edge), rho)
    excess_term = _weighted_avg(ne20 * (Te_excess + Ti_excess), rho)
    target_keV_ne20 = p_tgt / (1.0e20 * KEV_TO_J + 1.0e-30)
    amp = (target_keV_ne20 - edge_term) / (excess_term + 1.0e-12)
    amp = jnp.clip(amp, 0.02, 20.0)

    # Correct the amplitude so beta_N_total, not just beta_N_thermal, matches.
    # Two iterations are enough for this initial-condition rescale and keep the
    # function JAX-friendly.
    for _ in range(2):
        Te_try = Te_edge + amp * Te_excess
        Ti_try = Ti_edge + amp * Ti_excess
        beta_tot = _beta_N_total_from_profiles(rho, Te_try, Ti_try, ne20, machine, actuator, sim)
        corr = jnp.clip(beta_tgt / (beta_tot + 1.0e-12), 0.2, 5.0)
        amp = jnp.clip(amp * corr, 0.02, 20.0)
    return Te_edge + amp * Te_excess, Ti_edge + amp * Ti_excess


def make_iter_hmode_initial_profiles(rho, actuator, machine, sim=None):
    """ITER-H-mode-like initial profiles, rescaled to the supplied machine.

    The base shape is ITER-like, but if sim.derive_density_from_greenwald=True
    the density amplitude is rescaled so the volume-averaged density satisfies
    the requested Greenwald fraction for the current machine Ip and minor radius.
    If sim.initial_temperature_rescale_to_beta_N=True, Te/Ti excess above edge is
    also rescaled to keep the requested beta_N approximately fixed.  This makes
    ramp-up waveforms start from low-density/low-pressure profiles when Ip is low.
    """
    edge_ne20 = _target_edge_ne20(machine, actuator, sim)
    Te = _cell_average_profile(
        rho,
        lambda rr: pedestal_profile(
            rr, core=15.0, pedestal_top=5.0, edge=actuator.edge_Te_keV,
            rho_top=0.95, width=0.035, core_alpha=1.2
        ),
        sim=sim,
    )
    Ti = _cell_average_profile(
        rho,
        lambda rr: pedestal_profile(
            rr, core=14.0, pedestal_top=5.0, edge=actuator.edge_Ti_keV,
            rho_top=0.95, width=0.035, core_alpha=1.1
        ),
        sim=sim,
    )
    if sim == None:
        ne_base = _cell_average_profile(
            rho,
            lambda rr: pedestal_profile(
                rr, core=1.10, pedestal_top=1.00, edge=edge_ne20,
                rho_top=0.95, width=0.04, core_alpha=0.45
            ),
            sim=sim,
        )
    else:
        ne_base = _cell_average_profile(
            rho,
            lambda rr: pedestal_profile(
                rr, core=getattr(sim, "greenwald_tanh_core_factor", 1.1), pedestal_top=getattr(sim, "greenwald_tanh_pedestal_factor", 1.0), edge=edge_ne20,
                rho_top=0.95, width=0.04, core_alpha=0.45
            ),
            sim=sim,
        )

    ne20 = _rescale_density_shape_to_greenwald(rho, ne_base, machine, actuator, sim)
    Te, Ti = _rescale_temperature_to_beta_N(rho, Te, Ti, ne20, machine, actuator, sim)
    psi_edge = jnp.asarray(0.0)
    psi_ind = initial_psi_iter_hmode(rho, machine, psi_edge=psi_edge, sim=sim)
    return Te, Ti, ne20, psi_ind, psi_edge

def make_initial_profiles(rho, actuator, machine, sim=None):
    if sim is not None and getattr(sim, "initial_profile_model", "h_mode") == "h_mode":
        return make_iter_hmode_initial_profiles(rho, actuator, machine, sim=sim)

    edge_ne20 = _target_edge_ne20(machine, actuator, sim)
    Te = _cell_average_profile(
        rho, lambda rr: parabolic_profile(rr, core=8.0, edge=actuator.edge_Te_keV, alpha=1.4), sim=sim
    )
    Ti = _cell_average_profile(
        rho, lambda rr: parabolic_profile(rr, core=8.0, edge=actuator.edge_Ti_keV, alpha=1.4), sim=sim
    )
    ne_base = _cell_average_profile(
        rho,
        lambda rr: parabolic_profile(rr, core=jnp.maximum(_target_nbar20(machine, actuator, sim) * 1.25, edge_ne20), edge=edge_ne20, alpha=0.8),
        sim=sim,
    )
    ne20 = _rescale_density_shape_to_greenwald(rho, ne_base, machine, actuator, sim)
    Te, Ti = _rescale_temperature_to_beta_N(rho, Te, Ti, ne20, machine, actuator, sim)
    psi_edge = jnp.asarray(0.0)
    psi_ind = initial_psi_from_total_current(rho, machine, psi_edge=psi_edge, sim=sim)
    return Te, Ti, ne20, psi_ind, psi_edge
