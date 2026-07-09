"""0.5D fast tokamak model: 0D energy balance + reconstructed profiles.

This module is deliberately separate from the 1.5D diffusion solver.  It evolves
only the scalar thermal stored energy W_th, but after every step reconstructs
cell-centered Te=Ti and ne profiles so that the existing TokaGrad diagnostics
(fusion power, radiation, bootstrap/current, q, equilibrium surfaces, etc.) can
be reused.

References:
  [ITER Physics Basis Editors, Nucl. Fusion 39, 2175 (1999)] -- IPB98(y,2).
  [M. Greenwald et al., Nucl. Fusion 28, 2199 (1988)] -- density scaling.
The reconstructed radial shapes and adaptive effective timestep are reduced
TokaGrad closures rather than additional empirical scaling laws.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import MachineConfig, ActuatorConfig, SimulationConfig
from .profiles import PlasmaState, make_initial_profiles
from .grid import make_grid_from_config, axis_augmented_profile, axis_augmented_volume_element, axis_augmented_volume_element_from_dV_drho
from .heating import (
    KEV_TO_J,
    effective_ion_mass_amu,
    total_heating_sources,
)
from .diagnostics import compute_diagnostics, power_integral_axis_augmented, beta_normalized_total
from .equilibrium import solve_fixed_boundary_equilibrium
from .pedestal import pedestal_target_profiles, martin_lh_threshold_power_MW, delabie_lh_threshold_power_MW, lh_transition_gate
from .solver import (
    rescale_tanh_density_to_greenwald,
    rescale_initial_density_to_greenwald,
    density_model_uses_tanh_rescale,
    greenwald_target_density_1e20,
)
from .current import psi_inductive_update
from .waveforms import apply_waveform_controls
from .density import target_nbar20, target_edge_ne20


def zero_d_enabled(sim: SimulationConfig) -> bool:
    mode = str(getattr(sim, "simulation_model", "1.5d")).lower().replace("_", "")
    return mode in ("0d", "05d", "0.5d", "0dfast", "05dfast", "0.5dfast", "zerod", "zerodfast")


def ipb98y2_tau_E(machine: MachineConfig, nbar20, P_loss_MW, sim: SimulationConfig | None = None):
    """IPB98(y,2) thermal energy confinement time [s].

    Reference: [ITER Physics Basis Editors, Nucl. Fusion 39, 2175 (1999)].

    tau_E = 0.0562 Ip^0.93 Bt^0.15 n19^0.41 P^-0.69 R^1.97 eps^0.58 kappa^0.78 M^0.19 H
    with Ip in MA, Bt in T, n19 in 1e19 m^-3, P in MW.
    """
    H = 1.0 if sim is None else getattr(sim, "zero_d_H_factor", 1.0)
    Ip_MA = jnp.maximum(machine.Ip / 1.0e6, 0.05)
    Bt = jnp.maximum(jnp.abs(machine.Bt), 0.05)
    n19 = jnp.maximum(10.0 * nbar20, 0.2)
    P = jnp.maximum(P_loss_MW, 0.05)
    R = jnp.maximum(machine.R0, 0.2)
    eps = jnp.maximum(machine.a / (machine.R0 + 1e-12), 0.03)
    kappa = jnp.maximum(machine.kappa, 0.5)
    M = jnp.maximum(effective_ion_mass_amu(machine), 1.0)
    tau = 0.0562 * (Ip_MA ** 0.93) * (Bt ** 0.15) * (n19 ** 0.41) * (P ** -0.69) * (R ** 1.97) * (eps ** 0.58) * (kappa ** 0.78) * (M ** 0.19) * H
    if sim is not None:
        tau = jnp.clip(tau, getattr(sim, "zero_d_tauE_min", 1.0e-3), getattr(sim, "zero_d_tauE_max", 20.0))
    return tau


def iter89p_tau_E(machine: MachineConfig, nbar20, P_loss_MW, sim: SimulationConfig | None = None):
    """ITER89-P L-mode thermal energy confinement time [s].

    Reference: [P. N. Yushmanov et al., Nucl. Fusion 30, 1999 (1990)].

    A compact ITER89-P form is used as the 0.5D L-mode branch:

        tau_E = 0.038 Ip^0.85 R^1.5 eps^0.3 kappa^0.5 n19^0.1 Bt^0.2 M^0.5 P^-0.5 L

    with Ip in MA, R/a in m, n19 in 1e19 m^-3, Bt in T, P in MW.
    The optional ``zero_d_L_factor`` is kept separate from the H-mode H98 factor.
    """
    L = 1.0 if sim is None else getattr(sim, "zero_d_L_factor", 1.0)
    Ip_MA = jnp.maximum(machine.Ip / 1.0e6, 0.05)
    Bt = jnp.maximum(jnp.abs(machine.Bt), 0.05)
    #n20 = jnp.maximum(nbar20, 0.02)
    n19 = jnp.maximum(10.0 * nbar20, 0.2)
    P = jnp.maximum(P_loss_MW, 0.05)
    R = jnp.maximum(machine.R0, 0.2)
    #a = jnp.maximum(machine.a, 0.05)
    eps = jnp.maximum(machine.a / (machine.R0 + 1e-12), 0.03)
    kappa = jnp.maximum(machine.kappa, 0.5)
    M = jnp.maximum(effective_ion_mass_amu(machine), 1.0)
    tau = 0.038 * (Ip_MA ** 0.85) * (R ** 1.5) * (eps ** 0.30) * (kappa ** 0.50) * (n19 ** 0.10) * (Bt ** 0.20) * (M ** 0.50) * (P ** -0.50) * L
    if sim is not None:
        tau = jnp.clip(tau, getattr(sim, "zero_d_tauE_min", 1.0e-3), getattr(sim, "zero_d_tauE_max", 20.0))
    return tau


def zero_d_lh_gate_from_heat_diag(rho, state: PlasmaState, machine: MachineConfig, actuator: ActuatorConfig, sim: SimulationConfig, heat_diag, dV_drho=None):
    """AD-friendly L-H gate and associated powers for 0.5D closures."""
    pint = lambda key: power_integral_axis_augmented(heat_diag[key], rho, machine, dV_drho=dV_drho)
    P_aux = pint("Paux_e") + pint("Paux_i")
    P_ohm = pint("Pohm_e")
    P_alpha = pint("Palpha_e") + pint("Palpha_i")
    P_rad = pint("Prad_e")
    P_abs = P_aux + P_ohm + P_alpha
    if sim.pedestal_lh_power_basis == "absorbed_heating":
        P_eff = P_abs
    elif sim.pedestal_lh_power_basis == "net_separatrix":
        P_eff = jnp.maximum(P_abs - P_rad, 0.0)
    else:
        raise ValueError(
            f"Unknown pedestal_lh_power_basis={sim.pedestal_lh_power_basis!r}. "
            'Use "net_separatrix" or "absorbed_heating".'
        )
    nbar20 = _volume_average(rho, state.ne20, machine, dV_drho=dV_drho)
    if sim.pedestal_lh_threshold_model == "delabie":
        P_LH = delabie_lh_threshold_power_MW(nbar20, machine, sim)
    else:
        P_LH = martin_lh_threshold_power_MW(nbar20, machine, sim)
    gate = lh_transition_gate(P_eff, P_LH, sim)
    return gate, P_eff, P_LH, P_abs, P_rad


def _lmode_density_profile(rho, machine: MachineConfig, actuator: ActuatorConfig, sim: SimulationConfig, dV_drho=None):
    """Smooth no-pedestal density shape at the requested Greenwald average and edge density."""
    target = target_nbar20(machine, actuator, sim)
    edge = target_edge_ne20(machine, actuator, sim)
    excess_shape = jnp.maximum(1.0 - rho**2, 0.0) ** 0.45
    avg_excess = _volume_average(rho, excess_shape, machine, dV_drho=dV_drho)
    amp = jnp.maximum((target - edge) / (avg_excess + 1.0e-12), 0.0)
    return jnp.maximum(edge + amp * excess_shape, 1.0e-5)


def _blend_lh_density(rho, ne_hmode, machine: MachineConfig, actuator: ActuatorConfig, sim: SimulationConfig, lh_gate, dV_drho=None):
    #if not getattr(sim, "pedestal_lh_transition_control", False):
    if getattr(sim, "pedestal_lh_threshold_model", "none") == "none":
        return ne_hmode
    ne_lmode = _lmode_density_profile(rho, machine, actuator, sim, dV_drho=dV_drho)
    return (1.0 - lh_gate) * ne_lmode + lh_gate * ne_hmode


def _axis_dV(rho, machine, dV_drho=None):
    if dV_drho is None:
        return axis_augmented_volume_element(rho, machine)
    return axis_augmented_volume_element_from_dV_drho(rho, dV_drho)


def _volume_average(rho, y, machine, dV_drho=None):
    _, y_aug = axis_augmented_profile(rho, y)
    dV = _axis_dV(rho, machine, dV_drho=dV_drho)
    return jnp.sum(y_aug * dV) / (jnp.sum(dV) + 1e-12)


def _thermal_energy_from_T(rho, ne20, T_keV, machine, dV_drho=None):
    _, ne_aug = axis_augmented_profile(rho, ne20)
    _, T_aug = axis_augmented_profile(rho, T_keV)
    dV = _axis_dV(rho, machine, dV_drho=dV_drho)
    # Te=Ti=T -> W = 1.5 ne (Te+Ti) = 3 ne T
    W_J = 3.0 * jnp.sum(ne_aug * 1.0e20 * T_aug * KEV_TO_J * dV)
    return W_J / 1.0e6


def _thermal_energy_state(state: PlasmaState, machine, sim, dV_drho=None):
    rho, _, _ = make_grid_from_config(state.Te.size, machine.a, sim)
    _, Te_aug = axis_augmented_profile(rho, state.Te)
    _, Ti_aug = axis_augmented_profile(rho, state.Ti)
    _, ne_aug = axis_augmented_profile(rho, state.ne20)
    dV = _axis_dV(rho, machine, dV_drho=dV_drho)
    W_J = 1.5 * jnp.sum(ne_aug * 1.0e20 * (Te_aug + Ti_aug) * KEV_TO_J * dV)
    return W_J / 1.0e6


def _pedestal_floor_temperature(rho, dummy_state, machine, actuator, sim, q, beta_N_proxy=None, lh_gate=1.0):
    """Return a Te=Ti pedestal underlay and pedestal info."""
    edge_T = 0.5 * (actuator.edge_Te_keV + actuator.edge_Ti_keV)
    if getattr(sim, "pedestal_model", "none") == "none":
        width = jnp.asarray(getattr(sim, "greenwald_tanh_default_width", 0.04), dtype=rho.dtype)
        T_floor = jnp.zeros_like(rho) + edge_T
        return T_floor, {"width": width, "rho_top": 1.0 - width, "lh_gate": lh_gate}
    try:
        Te_tgt, Ti_tgt, _ne_tgt, info = pedestal_target_profiles(rho, dummy_state, machine, actuator, sim, q=q, beta_N_proxy=beta_N_proxy if beta_N_proxy is not None else 1.5)
        T_h = 0.5 * (Te_tgt + Ti_tgt)
        # Smoothly switch the tanh pedestal underlay on only in H-mode.
        # Below the Martin threshold, the 0D reconstructed profile has no
        # pedestal floor; stored energy is carried by the core-shape amplitude.
        T_floor = edge_T + lh_gate * jnp.maximum(T_h - edge_T, 0.0)
        if isinstance(info, dict):
            info = {**info, "lh_gate": lh_gate}
        return T_floor, info
    except Exception:
        if getattr(sim, "eped1nn_fail_mode", "fallback") == "raise":
            raise
        width = jnp.asarray(getattr(sim, "greenwald_tanh_default_width", 0.04), dtype=rho.dtype)
        T_floor = jnp.zeros_like(rho) + edge_T
        return T_floor, {"width": width, "rho_top": 1.0 - width, "lh_gate": lh_gate}


def reconstruct_profile_from_Wth(Wth_MJ, machine: MachineConfig, actuator: ActuatorConfig, sim: SimulationConfig, prev_state: PlasmaState | None = None):
    """Reconstruct Te=Ti, ne, and saturated current state from scalar W_th.

    Density uses the Greenwald-rescaled tanh+core shape by default.  Temperature
    is a pedestal underlay plus a core excess shape whose amplitude is chosen so
    the axis-augmented thermal stored energy equals Wth_MJ.
    """
    rho, _, _ = make_grid_from_config(sim.nr, machine.a, sim)
    # Fast path: do not rebuild full ITER-like initial profiles during every
    # scalar 0.5D reconstruction.  make_initial_profiles() performs beta_N
    # rescaling and initial-current construction and is comparatively expensive;
    # after the first state is available, the previous reconstructed state is a
    # better dummy profile for pedestal-width and density-shape estimates.
    if prev_state is not None and getattr(sim, "zero_d_profile_reconstruction_fast_path", True):
        Te0, Ti0, ne0 = prev_state.Te, prev_state.Ti, prev_state.ne20
        psi0, psi_edge0 = prev_state.psi_ind, prev_state.psi_edge
    else:
        Te0, Ti0, ne0, psi0, psi_edge0 = make_initial_profiles(rho, actuator, machine, sim)
        if prev_state is None:
            prev_state = PlasmaState(Te0, Ti0, ne0, psi0, psi_edge0)

    # Use a cheap dummy equilibrium/q to get pedestal width and target.  The
    # final current/q are recomputed after T and n are reconstructed unless
    # zero_d_reconstruct_current=False.
    dummy = PlasmaState(
        Te0, Ti0, ne0, prev_state.psi_ind, prev_state.psi_edge,
        prev_state.Phi_b_prev, prev_state.dV_drho_prev,
    )
    eq_dummy = solve_fixed_boundary_equilibrium(dummy, machine, actuator, sim)

    if density_model_uses_tanh_rescale(sim) or getattr(sim, "density_evolution_model", "") in ("greenwald_rescale_initial_shape", "greenwald_initial_shape_rescale", "initial_shape_rescale"):
        if density_model_uses_tanh_rescale(sim):
            ne20 = rescale_tanh_density_to_greenwald(rho, dummy, machine, actuator, sim, eq_dummy.dV_drho, q=eq_dummy.q, max_ne=5.0)
        else:
            ne20 = rescale_initial_density_to_greenwald(rho, machine, actuator, sim, eq_dummy.dV_drho, max_ne=5.0)
    else:
        # Fast mode defaults to a profile closure rather than density diffusion.
        ne20 = rescale_tanh_density_to_greenwald(rho, dummy, machine, actuator, sim, eq_dummy.dV_drho, q=eq_dummy.q, max_ne=5.0)

    dummy_n = PlasmaState(
        Te0, Ti0, ne20, prev_state.psi_ind, prev_state.psi_edge,
        prev_state.Phi_b_prev, prev_state.dV_drho_prev,
    )
    eq_n = solve_fixed_boundary_equilibrium(dummy_n, machine, actuator, sim)
    _Se_gate, _Si_gate, heat_gate = total_heating_sources(rho, dummy_n, machine, actuator, sim, eq=eq_n)
    lh_gate, _P_eff_lh, _P_LH_lh, _P_abs_lh, _P_rad_lh = zero_d_lh_gate_from_heat_diag(
        rho, dummy_n, machine, actuator, sim, heat_gate, dV_drho=eq_n.dV_drho
    )
    ne20 = _blend_lh_density(rho, ne20, machine, actuator, sim, lh_gate, dV_drho=eq_n.dV_drho)
    dummy_n = PlasmaState(
        Te0, Ti0, ne20, prev_state.psi_ind, prev_state.psi_edge,
        prev_state.Phi_b_prev, prev_state.dV_drho_prev,
    )
    eq_n = solve_fixed_boundary_equilibrium(dummy_n, machine, actuator, sim)
    beta_N = beta_normalized_total(dummy_n, machine, actuator, sim, eq=eq_n)
    T_floor, info = _pedestal_floor_temperature(rho, dummy_n, machine, actuator, sim, eq_n.q, beta_N_proxy=beta_N, lh_gate=lh_gate)
    T_floor = jnp.maximum(T_floor, actuator.edge_Te_keV)

    core_shape = jnp.maximum(1.0 - rho**1.5, 0.0) ** getattr(sim, "zero_d_core_shape_power", 1.5)
    # Keep the reconstructed temperature equal to the boundary at rho=1; the
    # actual plot helper connects to actuator.edge_Te/Ti.
    W_floor = _thermal_energy_from_T(rho, ne20, T_floor, machine, dV_drho=eq_n.dV_drho)
    W_target = jnp.maximum(jnp.asarray(Wth_MJ, dtype=rho.dtype), 1.0e-8)
    W_core_unit = _thermal_energy_from_T(rho, ne20, core_shape, machine, dV_drho=eq_n.dV_drho)
    amp = jnp.maximum((W_target - W_floor) / (W_core_unit + 1e-12), 0.0)
    # If requested energy is below the pedestal floor, scale the pedestal excess
    # above the edge down smoothly rather than creating negative core excess.
    edge_T = 0.5 * (actuator.edge_Te_keV + actuator.edge_Ti_keV)
    T_excess_floor = jnp.maximum(T_floor - edge_T, 0.0)
    W_edge = _thermal_energy_from_T(rho, ne20, jnp.zeros_like(rho) + edge_T, machine, dV_drho=eq_n.dV_drho)
    W_excess = jnp.maximum(W_floor - W_edge, 1e-12)
    floor_scale = jnp.minimum(1.0, jnp.maximum((W_target - W_edge) / W_excess, 0.0))
    T_base = edge_T + floor_scale * T_excess_floor
    T = T_base + amp * core_shape
    T = jnp.clip(T, edge_T, getattr(sim, "zero_d_temperature_max_keV", 80.0))

    tmp = PlasmaState(
        T, T, ne20, prev_state.psi_ind, prev_state.psi_edge,
        prev_state.Phi_b_prev, prev_state.dV_drho_prev,
    )
    if not getattr(sim, "zero_d_reconstruct_current", True):
        # Fastest 0.5D mode: update only W/Te/Ti/ne and retain the previous
        # current state. Diagnostics that depend on q/current will then be
        # approximate, but energy and fusion/radiation scans become very cheap.
        return PlasmaState(
            T, T, ne20, prev_state.psi_ind, prev_state.psi_edge,
            eq_n.Phi_b, eq_n.dV_drho,
        )
    eq = solve_fixed_boundary_equilibrium(tmp, machine, actuator, sim)
    psi_ind, psi_edge = psi_inductive_update(rho, tmp, T, T, ne20, machine, actuator, sim, eq=eq)
    tmp2 = PlasmaState(T, T, ne20, psi_ind, psi_edge, eq.Phi_b, eq.dV_drho)
    eq2 = solve_fixed_boundary_equilibrium(tmp2, machine, actuator, sim)
    return PlasmaState(T, T, ne20, psi_ind, psi_edge, eq2.Phi_b, eq2.dV_drho)


def zero_d_powers(state: PlasmaState, machine: MachineConfig, actuator: ActuatorConfig, sim: SimulationConfig, eq=None):
    rho, _, _ = make_grid_from_config(state.Te.size, machine.a, sim)
    if eq is None:
        eq = solve_fixed_boundary_equilibrium(state, machine, actuator, sim)
    dV_drho = eq.dV_drho
    _Se, _Si, h = total_heating_sources(rho, state, machine, actuator, sim, eq=eq)
    pint = lambda key: power_integral_axis_augmented(h[key], rho, machine, dV_drho=dV_drho)
    P_aux = pint("Paux_e") + pint("Paux_i")
    P_ohm = pint("Pohm_e")
    P_alpha = pint("Palpha_e") + pint("Palpha_i")
    P_rad = pint("Prad_e")
    P_heat = P_aux + P_ohm + P_alpha
    nbar20 = _volume_average(rho, state.ne20, machine, dV_drho=dV_drho)
    P_loss = jnp.maximum(P_heat - P_rad, 0.05)
    tau_H = ipb98y2_tau_E(machine, nbar20, P_loss, sim)
    tau_L = iter89p_tau_E(machine, nbar20, P_loss, sim)
    lh_gate, P_eff, P_LH, P_abs, P_rad_lh = zero_d_lh_gate_from_heat_diag(
        rho, state, machine, actuator, sim, h, dV_drho=dV_drho
    )
    tau_E = (1.0 - lh_gate) * tau_L + lh_gate * tau_H
    nbar20_proxy = greenwald_target_density_1e20(machine, actuator, sim)
    tau_E_proxy = ipb98y2_tau_E(machine, nbar20_proxy, actuator.P_aux_MW, sim)
    return {
        "P_aux_MW": P_aux, "P_ohm_MW": P_ohm, "P_alpha_MW": P_alpha, "P_rad_MW": P_rad,
        "P_heat_MW": P_heat, "tau_E_s": tau_E, "tau_E_H98_s": tau_H, "tau_E_ITER89_s": tau_L,
        "lh_gate": lh_gate, "P_LH_eff_MW": P_eff, "P_LH_Martin_MW": P_LH, "nbar20": nbar20,
        "tau_E_proxy_s": tau_E_proxy,
    }


def zero_d_effective_timestep_from_powers(powers, sim: SimulationConfig):
    """Return the adaptive 0.5D physical timestep used by the energy update [s].

    The fast 0.5D model uses ``sim.dt`` as an upper bound, but the actual
    physical advance is limited to ``zero_d_dt_fraction_tauE * tau_E``.  Keep
    this helper separate so scripts and UI code can report the same time axis
    that the solver actually used.
    """
    #tau = powers["tau_E_s"]
    tau = powers["tau_E_proxy_s"]
    return jnp.minimum(
        jnp.asarray(sim.dt, dtype=jnp.asarray(tau).dtype),
        jnp.asarray(getattr(sim, "zero_d_dt_fraction_tauE", 0.1), dtype=jnp.asarray(tau).dtype) * tau,
    )


def zero_d_step_with_dt(state: PlasmaState, machine: MachineConfig, actuator: ActuatorConfig, sim: SimulationConfig):
    """Advance one 0.5D step and return ``(new_state, dt_eff_s)``.

    ``dt_eff_s`` is the actual physical timestep used in the semi-analytic
    stored-energy update.  It can be smaller than ``sim.dt`` when tau_E is
    short, so downstream history/plot/optimization code should accumulate this
    value rather than assuming a uniform ``i * sim.dt`` time grid.
    """
    eq_state = solve_fixed_boundary_equilibrium(state, machine, actuator, sim)
    W = _thermal_energy_state(state, machine, sim, dV_drho=eq_state.dV_drho)
    p = zero_d_powers(state, machine, actuator, sim, eq=eq_state)
    tau = p["tau_E_s"]
    dt_eff = zero_d_effective_timestep_from_powers(p, sim)
    P_net = p["P_heat_MW"] - p["P_rad_MW"]
    # Semi-analytic update for dW/dt = P_net - W/tau with frozen powers.
    # This is much more robust than forward Euler when an automatically chosen
    # 0.5D timestep is a sizeable fraction of tau_E during a ramp-up.
    a = jnp.exp(-dt_eff / (tau + 1e-12))
    W_ss = tau * P_net
    W_new = a * W + (1.0 - a) * W_ss
    W_new = jnp.maximum(W_new, 1.0e-6)
    return reconstruct_profile_from_Wth(W_new, machine, actuator, sim, prev_state=state), dt_eff


def initial_state_0d(machine: MachineConfig, actuator: ActuatorConfig, sim: SimulationConfig):
    from .solver import initial_state
    s0 = initial_state(machine, actuator, sim)
    eq0 = solve_fixed_boundary_equilibrium(s0, machine, actuator, sim)
    return reconstruct_profile_from_Wth(_thermal_energy_state(s0, machine, sim, dV_drho=eq0.dV_drho), machine, actuator, sim, prev_state=s0)


def _history_dict_from_states_and_dts(states: PlasmaState, dt_eff_s):
    dt_eff_s = jnp.asarray(dt_eff_s)
    return {
        "states": states,
        # Post-step physical times.  Frame 0 is added by plotting helpers.
        "time": jnp.cumsum(dt_eff_s),
        "dt_eff_s": dt_eff_s,
        "time_is_post_step": True,
    }


def simulate_0d(machine: MachineConfig, actuator: ActuatorConfig, sim: SimulationConfig, state0: PlasmaState | None = None):
    if state0 is None:
        state0 = initial_state_0d(machine, actuator, sim)
    if getattr(sim, "zero_d_use_lax_scan", True):
        def body(carry, _i):
            new_state, dt_eff = zero_d_step_with_dt(carry, machine, actuator, sim)
            return new_state, {"states": new_state, "dt_eff_s": dt_eff}
        final_state, sample = jax.lax.scan(body, state0, jnp.arange(sim.n_steps))
        return final_state, _history_dict_from_states_and_dts(sample["states"], sample["dt_eff_s"])
    state = state0
    states = []
    dts = []
    for _ in range(int(sim.n_steps)):
        state, dt_eff = zero_d_step_with_dt(state, machine, actuator, sim)
        states.append(state)
        dts.append(dt_eff)
    if states:
        states_hist = PlasmaState(
            jnp.stack([s.Te for s in states]),
            jnp.stack([s.Ti for s in states]),
            jnp.stack([s.ne20 for s in states]),
            jnp.stack([s.psi_ind for s in states]),
            jnp.stack([s.psi_edge for s in states]),
            jnp.stack([s.Phi_b_prev for s in states]),
            jnp.stack([s.dV_drho_prev for s in states]),
        )
        dt_hist = jnp.stack(dts)
    else:
        states_hist = PlasmaState(*(jnp.expand_dims(x, 0) for x in state0))
        dt_hist = jnp.zeros((0,), dtype=jnp.asarray(sim.dt).dtype)
    return state, _history_dict_from_states_and_dts(states_hist, dt_hist)


def simulate_0d_final_with_elapsed(machine: MachineConfig, actuator: ActuatorConfig, sim: SimulationConfig, state0: PlasmaState | None = None):
    """Return ``(final_state, elapsed_physical_time_s)`` for a 0.5D rollout."""
    if state0 is None:
        state0 = initial_state_0d(machine, actuator, sim)
    if getattr(sim, "zero_d_use_lax_scan", True):
        def body(carry, _i):
            state, t_elapsed = carry
            new_state, dt_eff = zero_d_step_with_dt(state, machine, actuator, sim)
            return (new_state, t_elapsed + dt_eff), None
        (final_state, elapsed), _ = jax.lax.scan(
            body,
            (state0, jnp.asarray(0.0, dtype=state0.Te.dtype)),
            jnp.arange(sim.n_steps),
        )
        return final_state, elapsed
    state = state0
    elapsed = jnp.asarray(0.0, dtype=state0.Te.dtype)
    for _ in range(int(sim.n_steps)):
        state, dt_eff = zero_d_step_with_dt(state, machine, actuator, sim)
        elapsed = elapsed + dt_eff
    return state, elapsed


def simulate_0d_final(machine: MachineConfig, actuator: ActuatorConfig, sim: SimulationConfig, state0: PlasmaState | None = None):
    final_state, _elapsed = simulate_0d_final_with_elapsed(machine, actuator, sim, state0=state0)
    return final_state


simulate_0d_jit = jax.jit(simulate_0d, static_argnames=("machine", "actuator", "sim"))
simulate_0d_final_jit = jax.jit(simulate_0d_final, static_argnames=("machine", "actuator", "sim"))
simulate_0d_final_with_elapsed_jit = jax.jit(simulate_0d_final_with_elapsed, static_argnames=("machine", "actuator", "sim"))

# AD/optimization variants.  Pytree machine/actuator values are dynamic so JAX
# can differentiate controls; only the model/numerics configuration is static.
simulate_0d_final_dynamic_jit = jax.jit(
    simulate_0d_final, static_argnames=("sim",)
)
simulate_0d_final_with_elapsed_dynamic_jit = jax.jit(
    simulate_0d_final_with_elapsed, static_argnames=("sim",)
)


def simulate_waveform_0d(machine: MachineConfig, actuator: ActuatorConfig, sim: SimulationConfig, waveform, state0: PlasmaState | None = None):
    if state0 is None:
        m0, a0 = apply_waveform_controls(machine, actuator, waveform, 0.0)
        state0 = initial_state_0d(m0, a0, sim)
    if getattr(sim, "zero_d_use_lax_scan", True):
        def body(carry, _i):
            state, t = carry
            mt, at = apply_waveform_controls(machine, actuator, waveform, t)
            new_state, dt_eff = zero_d_step_with_dt(state, mt, at, sim)
            t_new = t + dt_eff
            sample = {
                "states": new_state,
                "time": t_new,
                "time_sample": t,
                "dt_eff_s": dt_eff,
                "Ip_MA": mt.Ip / 1.0e6,
                "Bt": mt.Bt,
                "P_aux_MW": at.P_aux_MW,
                "greenwald_fraction_target": at.greenwald_fraction_target,
                "heat_center": at.heat_center,
                "heat_width": at.heat_width,
            }
            return (new_state, t_new), sample
        (final_state, _t_final), hist = jax.lax.scan(
            body,
            (state0, jnp.asarray(0.0, dtype=state0.Te.dtype)),
            jnp.arange(sim.n_steps),
        )
        hist = dict(hist)
        hist["time_is_post_step"] = True
        return final_state, hist
    state = state0
    t = jnp.asarray(0.0, dtype=state0.Te.dtype)
    samples = []
    for _ in range(int(sim.n_steps)):
        mt, at = apply_waveform_controls(machine, actuator, waveform, t)
        state, dt_eff = zero_d_step_with_dt(state, mt, at, sim)
        t_new = t + dt_eff
        samples.append((state, t_new, t, dt_eff, mt, at))
        t = t_new
    if samples:
        states = [x[0] for x in samples]
        hist = {
            "states": PlasmaState(
                jnp.stack([s.Te for s in states]),
                jnp.stack([s.Ti for s in states]),
                jnp.stack([s.ne20 for s in states]),
                jnp.stack([s.psi_ind for s in states]),
                jnp.stack([s.psi_edge for s in states]),
                jnp.stack([s.Phi_b_prev for s in states]),
                jnp.stack([s.dV_drho_prev for s in states]),
            ),
            "time": jnp.stack([x[1] for x in samples]),
            "time_sample": jnp.stack([x[2] for x in samples]),
            "dt_eff_s": jnp.stack([x[3] for x in samples]),
            "Ip_MA": jnp.stack([x[4].Ip / 1.0e6 for x in samples]),
            "Bt": jnp.stack([x[4].Bt for x in samples]),
            "P_aux_MW": jnp.stack([x[5].P_aux_MW for x in samples]),
            "greenwald_fraction_target": jnp.stack([x[5].greenwald_fraction_target for x in samples]),
            "heat_center": jnp.stack([x[5].heat_center for x in samples]),
            "heat_width": jnp.stack([x[5].heat_width for x in samples]),
            "time_is_post_step": True,
        }
    else:
        hist = {"states": state0, "time": jnp.zeros((0,)), "dt_eff_s": jnp.zeros((0,)), "time_is_post_step": True}
    return state, hist


from .fast_ions import (
    fast_ion_energy_components_MJ,
)


def zero_d_diagnostics(state: PlasmaState, machine: MachineConfig, actuator: ActuatorConfig, sim: SimulationConfig):
    eq = solve_fixed_boundary_equilibrium(state, machine, actuator, sim)
    out = dict(compute_diagnostics(state, machine, actuator, sim))
    p = zero_d_powers(state, machine, actuator, sim, eq=eq)
    W_nbi, W_alpha = fast_ion_energy_components_MJ(state, machine, actuator, sim, eq=eq)
    out.update({
        "tau_E_s": p["tau_E_s"],
        "tau_E_H98_s": p.get("tau_E_H98_s", p["tau_E_s"]),
        "tau_E_ITER89_s": p.get("tau_E_ITER89_s", p["tau_E_s"]),
        "ped_lh_gate": p.get("lh_gate", 1.0),
        "P_LH_eff_MW": p.get("P_LH_eff_MW", 0.0),
        "P_LH_Martin_MW": p.get("P_LH_Martin_MW", 0.0),
        "W_fast_MJ": W_nbi + W_alpha,
        "W_fast_NBI_MJ": W_nbi,
        "W_fast_alpha_MJ": W_alpha,
        "beta_N_total": beta_normalized_total(state, machine, actuator, sim, eq=eq),
    })
    return out
