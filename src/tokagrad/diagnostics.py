"""Integrated scalar diagnostics derived from radial plasma profiles.

References:
  [F. Troyon et al., Plasma Phys. Control. Fusion 26, 209 (1984)] -- beta_N.
  [M. Greenwald et al., Nucl. Fusion 28, 2199 (1988)] -- density fraction.
  [H.-S. Bosch and G. M. Hale, Nucl. Fusion 32, 611 (1992)] -- fusion power.

Axis extrapolation and fast-ion additions to beta_N are numerical/reduced
TokaGrad conventions and are not part of the original empirical limits.
"""

import jax.numpy as jnp
from .grid import make_grid_from_config, infer_rho_faces, axis_extrapolated_value, axis_augmented_profile, axis_augmented_volume_element, axis_augmented_volume_element_from_dV_drho
from .equilibrium import solve_fixed_boundary_equilibrium
from .heating import total_heating_sources, fusion_power_density_DT_MW_m3
from .current import effective_current_drive_fraction
from .pedestal import pedestal_diagnostics, martin_lh_threshold_power_MW, delabie_lh_threshold_power_MW, lh_transition_gate
from .current import (
    q_profile,
    neoclassical_resistivity,
    shape_factor,
    current_components_from_state,
    total_current,
    edge_q_from_boundary_geometry,
)

MU0 = 4.0e-7 * jnp.pi

def _axis_volume_weights(rho, machine, dV_drho=None):
    if dV_drho is None:
        return axis_augmented_volume_element(rho, machine)
    return axis_augmented_volume_element_from_dV_drho(rho, dV_drho)


def volume_average_axis_augmented(y, rho, machine, dV_drho=None):
    rho_aug, y_aug = axis_augmented_profile(rho, y)
    dV = _axis_volume_weights(rho, machine, dV_drho=dV_drho)
    return jnp.sum(y_aug * dV) / (jnp.sum(dV) + 1e-12)


def line_average_profile(y, rho):
    """Center-chord radial line average, matching solver feedback convention."""
    rho_f = infer_rho_faces(rho)
    drho = rho_f[1:] - rho_f[:-1]
    return jnp.sum(y * drho) / (jnp.sum(drho) + 1.0e-12)


def density_average_for_greenwald_metric(y, rho, machine, sim, dV_drho=None):
    """Density average used for the public ``greenwald_fraction`` diagnostic."""
    basis = str(getattr(sim, "greenwald_feedback_average_basis", "volume")).lower()
    if basis in ("line", "line_average", "line-averaged", "chord", "chord_average"):
        return line_average_profile(y, rho)
    return volume_average_axis_augmented(y, rho, machine, dV_drho=dV_drho)


def power_integral_axis_augmented(y, rho, machine, dV_drho=None):
    rho_aug, y_aug = axis_augmented_profile(rho, y)
    dV = _axis_volume_weights(rho, machine, dV_drho=dV_drho)
    return jnp.sum(y_aug * dV)


def thermal_energy_MJ(state, machine, sim=None, eq=None, dV_drho=None):
    rho, _, _ = make_grid_from_config(state.Te.size, machine.a, sim)
    rho_aug, Te_aug = axis_augmented_profile(rho, state.Te)
    _, Ti_aug = axis_augmented_profile(rho, state.Ti)
    _, ne_aug20 = axis_augmented_profile(rho, state.ne20)
    if dV_drho is None and eq is not None:
        dV_drho = eq.dV_drho
    dV = _axis_volume_weights(rho, machine, dV_drho=dV_drho)
    keV_to_J = 1.602176634e-16
    ne = ne_aug20 * 1.0e20
    W = 1.5 * jnp.sum(ne * (Te_aug + Ti_aug) * keV_to_J * dV)
    return W / 1.0e6

def beta_normalized(state, machine, sim=None, eq=None, dV_drho=None):
    rho, _, _ = make_grid_from_config(state.Te.size, machine.a, sim)
    rho_aug, Te_aug = axis_augmented_profile(rho, state.Te)
    _, Ti_aug = axis_augmented_profile(rho, state.Ti)
    _, ne_aug20 = axis_augmented_profile(rho, state.ne20)
    if dV_drho is None and eq is not None:
        dV_drho = eq.dV_drho
    dV = _axis_volume_weights(rho, machine, dV_drho=dV_drho)
    volume = jnp.sum(dV)
    keV_to_J = 1.602176634e-16
    pressure = ne_aug20 * 1e20 * (Te_aug + Ti_aug) * keV_to_J
    p_avg = jnp.sum(pressure * dV) / (volume + 1e-12)
    beta = 2.0 * MU0 * p_avg / (machine.Bt**2 + 1e-12)
    beta_percent = 100.0 * beta
    Ip_MA = machine.Ip / 1e6
    beta_N = beta_percent * machine.a * machine.Bt / (Ip_MA + 1e-12)
    return beta_N


def beta_normalized_total(state, machine, actuator=None, sim=None, eq=None, dV_drho=None):
    """Thermal + reduced fast-ion normalized beta.

    The thermal part is the ordinary profile pressure beta_N.  When an
    actuator is supplied, add a fast-ion stored-energy proxy from NBI and
    alpha power using the same slowing-down-time model as the 0.5D fast mode.
    """
    beta_th = beta_normalized(state, machine, sim, eq=eq, dV_drho=dV_drho)
    if actuator is None or sim is None:
        return beta_th
    try:
        from .fast_ions import fast_ion_energy_MJ, beta_normalized_total_from_thermal
        Wth = thermal_energy_MJ(state, machine, sim, eq=eq, dV_drho=dV_drho)
        Wfast = fast_ion_energy_MJ(state, machine, actuator, sim, eq=eq, dV_drho=dV_drho)
        return beta_normalized_total_from_thermal(beta_th, Wth, Wfast)
    except Exception:
        return beta_th


def fast_ion_energy_MJ_diagnostic(state, machine, actuator, sim, eq=None, dV_drho=None):
    try:
        from .fast_ions import fast_ion_energy_MJ
        return fast_ion_energy_MJ(state, machine, actuator, sim, eq=eq, dV_drho=dV_drho)
    except Exception:
        return jnp.asarray(0.0)

def fusion_power_MW(state, machine, sim=None, eq=None, dV_drho=None):
    """Reduced D-T fusion power proxy [MW], including copied rho=0."""
    rho, _, _ = make_grid_from_config(state.Te.size, machine.a, sim)
    if dV_drho is None and eq is not None:
        dV_drho = eq.dV_drho
    pfus = fusion_power_density_DT_MW_m3(state, machine)
    return power_integral_axis_augmented(pfus, rho, machine, dV_drho=dV_drho)

def actual_pedestal_alpha_from_profile(state, machine, actuator, sim, eq):
    """Compute actual ballooning alpha in the resolved pedestal window."""
    rho, _, _ = make_grid_from_config(state.Te.size, machine.a, sim)
    from .pedestal import actual_alpha_profile, pedestal_alpha_window_max, pedestal_diagnostics

    alpha_profile = actual_alpha_profile(rho, state, machine, eq.q)
    try:
        ped = pedestal_diagnostics(rho, state, machine, actuator, sim, q=eq.q)
        width = ped.get("ped_width", 0.05)
        rho_top = ped.get("ped_rho_top", 0.95)
    except Exception:
        width = 0.05
        rho_top = 0.95

    alpha_ped_max = pedestal_alpha_window_max(rho, alpha_profile, width)
    idx = jnp.argmin(jnp.abs(rho - rho_top))
    alpha_at_top = alpha_profile[idx]
    return alpha_ped_max, alpha_at_top


def compute_diagnostics(
    state,
    machine,
    actuator=None,
    sim=None,
    include_equilibrium=True,
    *,
    rho=None,
    eq=None,
    current_components=None,
    heat_diag=None,
):
    from .config import ActuatorConfig, SimulationConfig
    if actuator is None:
        actuator = ActuatorConfig()
    if sim is None:
        sim = SimulationConfig(nr=state.Te.size)

    if rho is None:
        rho, _, _ = make_grid_from_config(state.Te.size, machine.a, sim)
    # Build the requested equilibrium before volume-integrated diagnostics so
    # geqdsk_prescribed and reduced_fixed_boundary use their own V'(rho).
    # In app/live loops, pass a precomputed eq to avoid recomputing geometry.
    eq_for_alpha = eq if eq is not None else solve_fixed_boundary_equilibrium(state, machine, actuator, sim)
    dV_diag = eq_for_alpha.dV_drho

    if current_components is None:
        j_ind, j_bs, j_cd, j_total = current_components_from_state(rho, state, machine, actuator, sim, eq=eq_for_alpha)
    else:
        j_ind, j_bs, j_cd, j_total = current_components
    # Use the equilibrium q if prescribed (GEQDSK) and geometry-aware q/current
    # integrals otherwise.
    q = eq_for_alpha.q if hasattr(eq_for_alpha, "q") else q_profile(rho, j_total, machine, sim, eq=eq_for_alpha)
    eta_neo = neoclassical_resistivity(state.Te, rho, machine, eq=eq_for_alpha)
    q95_idx = min(int(0.95 * state.Te.size), state.Te.size - 1)

    I_ind = total_current(rho, j_ind, machine, eq=eq_for_alpha)
    I_bs = total_current(rho, j_bs, machine, eq=eq_for_alpha)
    I_cd = total_current(rho, j_cd, machine, eq=eq_for_alpha)
    I_total = total_current(rho, j_total, machine, eq=eq_for_alpha)

    q_edge_lcfs = jnp.maximum(edge_q_from_boundary_geometry(machine, I_total, eq=eq_for_alpha, sim=sim), q[-1])
    rho_q = jnp.concatenate([rho, jnp.ones(1, dtype=rho.dtype)])
    q_q = jnp.concatenate([q, jnp.asarray([q_edge_lcfs], dtype=q.dtype)])
    q95_value = jnp.interp(jnp.asarray(0.95, dtype=rho.dtype), rho_q, q_q)
    try:
        from .current import loop_voltage_from_ip_error
        V_loop, _ = loop_voltage_from_ip_error(rho, state, machine, actuator, sim, eq=eq_for_alpha)
    except Exception:
        V_loop = jnp.asarray(0.0)

    # Integrated power diagnostics.  In app/live loops, pass precomputed
    # heating to avoid evaluating auxiliary/alpha/radiation sources twice.
    if heat_diag is None:
        _, _, heat_diag = total_heating_sources(rho, state, machine, actuator, sim, eq=eq_for_alpha)
    def pint(name):
        return power_integral_axis_augmented(heat_diag[name], rho, machine, dV_drho=dV_diag)

    P_aux_abs = pint("Paux_e") + pint("Paux_i")
    P_ohm = pint("Pohm_e")
    P_alpha = pint("Palpha_e") + pint("Palpha_i")
    P_rad = pint("Prad_e")
    # External fusion gain.  Use absorbed auxiliary heating in the denominator;
    # do not include Ohmic or alpha self-heating, otherwise ignition-like states
    # would artificially suppress the reported Q.
    P_fus = fusion_power_MW(state, machine, sim, eq=eq_for_alpha)
    #Q_ext = P_fus / (P_aux_abs + 1.0e-12)
    Q_ext = P_fus / (actuator.P_aux_MW + 1.0e-12)
    P_abs = P_aux_abs + P_ohm + P_alpha
    if sim.pedestal_lh_power_basis == "absorbed_heating":
        P_lh_eff = P_abs
    else:
        P_lh_eff = jnp.maximum(P_abs - P_rad, 0.0)
    if sim.pedestal_lh_threshold_model == "delabie":
        P_LH = delabie_lh_threshold_power_MW(volume_average_axis_augmented(state.ne20, rho, machine, dV_drho=dV_diag), machine, sim)
    else:
        P_LH = martin_lh_threshold_power_MW(volume_average_axis_augmented(state.ne20, rho, machine, dV_drho=dV_diag), machine, sim)
    ped_lh_gate = lh_transition_gate(P_lh_eff, P_LH, sim)

    # Equilibrium is needed both for optional equilibrium diagnostics and
    # for the actual profile-based pedestal alpha.
    ped_alpha_actual, ped_alpha_actual_at_top = actual_pedestal_alpha_from_profile(
        state, machine, actuator, sim, eq_for_alpha
    )

    nG_1e20 = (machine.Ip / 1.0e6) / (jnp.pi * machine.a**2 + 1e-12)
    nbar_vol_1e20 = volume_average_axis_augmented(state.ne20, rho, machine, dV_drho=dV_diag)
    nbar_line_1e20 = line_average_profile(state.ne20, rho)
    nbar_feedback_1e20 = density_average_for_greenwald_metric(
        state.ne20, rho, machine, sim, dV_drho=dV_diag
    )

    Te_axis = axis_extrapolated_value(state.Te, rho)
    Ti_axis = axis_extrapolated_value(state.Ti, rho)
    ne_axis = axis_extrapolated_value(state.ne20, rho)

    out = {
        "Te0_keV": Te_axis,
        "Ti0_keV": Ti_axis,
        "ne0_1e20": ne_axis,
        "Te_first_cell_keV": state.Te[0],
        "Ti_first_cell_keV": state.Ti[0],
        "ne_first_cell_1e20": state.ne20[0],
        "Te_avg_keV": volume_average_axis_augmented(state.Te, rho, machine, dV_drho=dV_diag),
        "Ti_avg_keV": volume_average_axis_augmented(state.Ti, rho, machine, dV_drho=dV_diag),
        "ne_avg_1e20": nbar_vol_1e20,
        "ne_line_avg_1e20": nbar_line_1e20,
        "ne_greenwald_avg_1e20": nbar_feedback_1e20,
        "n_greenwald_1e20": nG_1e20,
        "greenwald_fraction": nbar_feedback_1e20 / (nG_1e20 + 1e-12),
        "greenwald_fraction_volume": nbar_vol_1e20 / (nG_1e20 + 1e-12),
        "greenwald_fraction_line": nbar_line_1e20 / (nG_1e20 + 1e-12),
        "q0": q[0],
        "q95": q95_value,
        "q_edge": q_edge_lcfs,
        "j_total_min_MA_m2": jnp.min(j_total) / 1e6,
        "j_edge_MA_m2": j_total[-1] / 1e6,
        "I_ind_MA": I_ind / 1e6,
        "I_bs_MA": I_bs / 1e6,
        "I_cd_MA": I_cd / 1e6,
        "I_total_MA": I_total / 1e6,
        "V_loop_V": V_loop,
        "shape_factor": shape_factor(machine, rho),
        "eta_neo_axis": eta_neo[0],
        "eta_neo_edge": eta_neo[-1],
        "Wth_MJ": thermal_energy_MJ(state, machine, sim, eq=eq_for_alpha),
        "W_fast_MJ": fast_ion_energy_MJ_diagnostic(state, machine, actuator, sim, eq=eq_for_alpha),
        "beta_N_thermal": beta_normalized(state, machine, sim, eq=eq_for_alpha),
        "beta_N": beta_normalized_total(state, machine, actuator, sim, eq=eq_for_alpha),
        "beta_N_total": beta_normalized_total(state, machine, actuator, sim, eq=eq_for_alpha),
        "P_fus_MW": P_fus,
        "Q": Q_ext,
        "Q_fusion_gain": Q_ext,
        "P_aux_abs_MW": P_aux_abs,
        "P_ohmic_MW": P_ohm,
        "P_alpha_MW": P_alpha,
        "P_abs_for_lh_MW": P_abs,
        "P_LH_eff_MW": P_lh_eff,
        "P_LH_Martin_MW": P_LH,
        "ped_lh_gate": ped_lh_gate,
        "P_rad_MW": P_rad,
        "P_brem_MW": pint("Pbrem"),
        "P_line_MW": pint("Pline"),
        "P_sync_MW": pint("Psync"),
        "I_cd_auto_fraction": effective_current_drive_fraction(actuator, machine),
        "ped_alpha_actual": ped_alpha_actual,
        "ped_alpha_actual_at_top": ped_alpha_actual_at_top,
    }

    if include_equilibrium:
        eq = eq_for_alpha
        out.update({
            "eq_beta_p": eq.beta_p,
            "eq_li_proxy": eq.li_proxy,
            "eq_volume_m3": eq.V[-1],
            "eq_R_axis_m": eq.R[0].mean(),
            "eq_R_out_m": jnp.max(eq.R[-1]),
            "eq_R_in_m": jnp.min(eq.R[-1]),
            "eq_Z_top_m": jnp.max(eq.Z[-1]),
            "eq_Z_bot_m": jnp.min(eq.Z[-1]),
        })
        out.update(pedestal_diagnostics(rho, state, machine, actuator, sim, q=eq.q, beta_N_proxy=out["beta_N"]))
    return out
