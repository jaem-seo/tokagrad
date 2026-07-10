"""Analytic/NN pedestal closures and empirical L--H transition gates.

References:
  [P. B. Snyder et al., Nucl. Fusion 51, 103016 (2011)] -- EPED model.
  [Y. R. Martin et al., J. Phys.: Conf. Ser. 123, 012033 (2008)] -- ITPA08.
  [E. Delabie et al., Nucl. Fusion (2026), doi:10.1088/1741-4326/ae39f2]
    -- ITPA TC-26 metal-wall threshold scaling.
  [T. Onjun et al., Phys. Plasmas 9, 5018 (2002)] -- alpha-critical model

"""

import jax.numpy as jnp
from .grid import radial_gradient
from .density import target_edge_ne20
from .heating import effective_ion_mass_amu

MU0 = 4.0e-7 * jnp.pi
KEV_TO_J = 1.602176634e-16

def _smooth_lower(x, lo, width=1.0e-3):
    w = jnp.maximum(width, 1.0e-8)
    return lo + w * jnp.logaddexp(0.0, (x - lo) / w)

def _smooth_upper(x, hi, width=1.0e-3):
    w = jnp.maximum(width, 1.0e-8)
    return hi - w * jnp.logaddexp(0.0, (hi - x) / w)

def _smooth_bounded(x, lo, hi, width=1.0e-3):
    return _smooth_upper(_smooth_lower(x, lo, width), hi, width)

def _smooth_abs(x, width=1.0e-8):
    return jnp.sqrt(x*x + width*width)

def _smooth_max(x, sharpness=30.0):
    w = 1.0 / sharpness
    z = x / w
    zmax = jnp.max(z)
    return w * (zmax + jnp.log(jnp.mean(jnp.exp(z - zmax))))

def _maybe_lower(x, lo, sim, width=1.0e-3):
    if getattr(sim, 'differentiable_smooth_mode', False):
        return _smooth_lower(x, lo, width)
    return jnp.maximum(x, lo)


def _require_equilibrium_q(q, rho, context):
    """Return a q profile supplied by the caller, or fail loudly.

    Pedestal closures should use the q profile from the active Equilibrium
    (``eq.q``).  The old fallback ``1 + 2*rho**2`` hid missing-q call paths
    and could make EPED/alpha-critical targets inconsistent with the selected
    equilibrium.
    """
    if q is None:
        raise ValueError(f"{context} requires q=eq.q from the active Equilibrium.")
    return jnp.clip(jnp.abs(jnp.asarray(q, dtype=rho.dtype) + jnp.zeros_like(rho)), 0.05, 20.0)

def _maybe_bound(x, lo, hi, sim, width=1.0e-3):
    if getattr(sim, 'differentiable_smooth_mode', False):
        return _smooth_bounded(x, lo, hi, width)
    return jnp.clip(x, lo, hi)

def _maybe_abs(x, sim, width=1.0e-8):
    if getattr(sim, 'differentiable_smooth_mode', False):
        return _smooth_abs(x, width)
    return jnp.abs(x)

def _maybe_rate_limit(x, limit, sim, width=1.0e-2):
    if getattr(sim, 'differentiable_smooth_mode', False):
        return limit * jnp.tanh(x / (limit + width))
    return jnp.clip(x, -limit, limit)


def _resolve_beta_N_proxy(state, machine, actuator, sim, beta_N_proxy=None, eq=None, dV_drho=None):
    """Resolve the EPED beta_N input using the diagnostics convention.

    EPED1-NN uses beta_N as an input feature.  When the caller does not supply
    one explicitly, use diagnostics.beta_normalized_total(), including the same
    optional equilibrium V'(rho) weighting and fast-ion proxy.  This helper is
    local-imported to avoid a module-level circular import with diagnostics.py.
    """
    if beta_N_proxy is not None:
        return jnp.asarray(beta_N_proxy)
    try:
        from .diagnostics import beta_normalized_total
        return beta_normalized_total(state, machine, actuator, sim, eq=eq, dV_drho=dV_drho)
    except Exception:
        return jnp.asarray(1.5)


def _grad(y, rho, a):
    return radial_gradient(y, rho, a)

def alpha_critical_shape_proxy(machine, q_top, s_top, k_top, d_top, sim):
    """Reduced shape-based empirical alpha_critical proxy.

    Reference: [T. Onjun et al., Phys. Plasmas 9, 5018 (2002)].
    """
    k = k_top
    d = d_top
    s = _maybe_lower(s_top, 0.0, sim, 1e-3)
    q = _maybe_lower(q_top, 0.5, sim, 1e-3)
    alpha = sim.pedestal_alpha_scale * (
        0.4 * s * (1.0 + k**2 * (1.0 + 5.0 * d**2))
    )
    return _maybe_bound(alpha, sim.pedestal_alpha_min, sim.pedestal_alpha_max, sim, 1e-2)

def kbm_width(beta_p_ped, sim):
    """KBM-like width scaling: Delta ~ G sqrt(beta_p,ped).

    Reference: [P. B. Snyder et al., Nucl. Fusion 51, 103016 (2011)].
    """
    beta_pos = _maybe_lower(beta_p_ped, 1e-8, sim, 1e-5)
    width = sim.pedestal_kbm_width_coeff * jnp.sqrt(beta_pos)
    return _maybe_bound(width, sim.pedestal_width_min, sim.pedestal_width_max, sim, 1e-3)

def _magnetic_shear(q, rho):
    nr = rho.size
    drho = 1.0 / nr
    qp = jnp.concatenate([q[0:1], q, q[-1:]])
    dq = (qp[2:] - qp[:-2]) / (2.0 * drho)
    return rho * dq / _smooth_lower(q, 1e-6, 1e-6)


def _interp_at_rho(rho, y, rho_query):
    """Smooth linear interpolation of y(rho) at scalar rho_query."""
    # Gaussian kernel interpolation avoids non-differentiable argmin.
    width = 1.0 / rho.size
    w = jnp.exp(-0.5 * ((rho - rho_query) / (width + 1e-8)) ** 2)
    return jnp.sum(w * y) / (jnp.sum(w) + 1e-12)


def pedestal_pressure_height_from_alpha(rho, machine, q, sim, beta_p_guess=0.05):
    """Compute pedestal pressure height from alpha_crit times KBM width.

    In smooth mode, quantities at the pedestal top are obtained by smooth
    interpolation rather than argmin indexing.
    """
    kappa_r = 1.0 + (machine.kappa - 1.0) * rho ** sim.elongation_profile_power
    delta_r = machine.delta * rho ** sim.triangularity_profile_power
    k95 = _interp_at_rho(rho, kappa_r, 0.95)
    d95 = _interp_at_rho(rho, delta_r, 0.95)

    width = kbm_width(beta_p_guess, sim)
    rho_top = 1.0 - width

    s = _magnetic_shear(q, rho)
    q_top = _interp_at_rho(rho, q, rho_top)
    s_top = _interp_at_rho(rho, s, rho_top)
    alpha_c = alpha_critical_shape_proxy(machine, q_top, s_top, k95, d95, sim)

    B = machine.Bt
    R = machine.R0
    q_top_safe = _smooth_lower(q_top, 0.3, 1e-3)
    dpdr_crit = alpha_c * B**2 / (2.0 * MU0 * R * q_top_safe**2 + 1e-12)
    dp_ped = dpdr_crit * machine.a * width

    beta_p_ped = 2.0 * MU0 * dp_ped / (
        (MU0 * machine.Ip / (2.0 * jnp.pi * machine.a + 1e-12)) ** 2 + 1e-12
    )

    width = kbm_width(beta_p_ped, sim)
    rho_top = 1.0 - width
    q_top = _interp_at_rho(rho, q, rho_top)
    s_top = _interp_at_rho(rho, s, rho_top)
    q_top_safe = _smooth_lower(q_top, 0.3, 1e-3)
    alpha_c = alpha_critical_shape_proxy(machine, q_top, s_top, k95, d95, sim)
    dpdr_crit = alpha_c * B**2 / (2.0 * MU0 * R * q_top_safe**2 + 1e-12)
    dp_ped = dpdr_crit * machine.a * width

    return {
        "dp_ped": dp_ped,
        "width": width,
        "rho_top": rho_top,
        "alpha_crit": alpha_c,
        "q_top": q_top,
        "s_top": s_top,
        "beta_p_ped": beta_p_ped,
    }


def _tanh_pedestal_profile(rho, edge_value, height, width, sharpness):
    #rho_top = 1.0 - width
    #s = 0.5 * (1.0 - jnp.tanh((rho - rho_top) / (0.5 * width + 1e-8)))
    rho_ped = 1.0 - 0.5 * width # center of the pedestal
    s = 0.5 * (1.0 - jnp.tanh((rho - rho_ped) / (0.5 * width + 1e-8)))
    return edge_value + height * s

def _targets_from_pressure_height(rho, machine, actuator, sim, dp_ped, width):
    ne_height = sim.pedestal_density_height20
    ne_target = _tanh_pedestal_profile(
        rho, target_edge_ne20(machine, actuator, sim), ne_height, width, sim.pedestal_transition_sharpness
    )
    ne_top20 = target_edge_ne20(machine, actuator, sim) + ne_height
    p_e = sim.pedestal_te_fraction * dp_ped
    p_i = (1.0 - sim.pedestal_te_fraction) * dp_ped
    Te_height = p_e / (_smooth_lower(ne_top20, 0.02, 1e-4) * 1.0e20 * KEV_TO_J)
    Ti_height = p_i / (_smooth_lower(ne_top20, 0.02, 1e-4) * 1.0e20 * KEV_TO_J)

    Te_target = _tanh_pedestal_profile(
        rho, actuator.edge_Te_keV, Te_height, width, sim.pedestal_transition_sharpness
    )
    Ti_target = _tanh_pedestal_profile(
        rho, actuator.edge_Ti_keV, Ti_height, width, sim.pedestal_transition_sharpness
    )
    return Te_target, Ti_target, ne_target

def alpha_critical_pedestal_targets(rho, state, machine, actuator, sim, q):
    info = pedestal_pressure_height_from_alpha(rho, machine, q, sim)

    # First build the discrete target from the analytic pressure-height estimate.
    Te0, Ti0, ne0 = _targets_from_pressure_height(
        rho, machine, actuator, sim, info["dp_ped"], info["width"]
    )

    # Calibrate the discrete target so its finite-difference alpha actually
    # matches alpha_crit. This removes mismatch from tanh width, grid spacing,
    # and rho-vs-r conversion.
    class _S:
        pass
    st = _S()
    st.Te, st.Ti, st.ne20 = Te0, Ti0, ne0
    alpha_tgt = pedestal_alpha_window_max(
        rho, actual_alpha_profile(rho, st, machine, q), info["width"]
    )
    raw_discrete_scale = info["alpha_crit"] / _smooth_lower(alpha_tgt, 1e-6, 1e-6)

    if getattr(sim, "pedestal_alpha_resolution_guard", False):
        raw_discrete_scale = _maybe_bound(
            raw_discrete_scale,
            0.05,
            getattr(sim, "pedestal_alpha_discrete_scale_max", 5.0),
            sim,
            1e-2,
        )

        # Optional coarse-grid guard.  If the predicted pedestal width is
        # substantially smaller than one radial cell, the finite-difference
        # alpha of the discrete tanh target is not meaningful.  In that case,
        # smoothly blend the alpha calibration and alpha-tracking feedback back
        # toward unity instead of inflating the pedestal pressure to compensate
        # for the unresolved gradient.  Kept optional because very coarse grids
        # may otherwise under-drive the edge pedestal in long transients.
        rho_size = jnp.asarray(rho.size, dtype=rho.dtype)
        if rho.size > 1:
            drho_eff = (jnp.max(rho) - jnp.min(rho)) / jnp.maximum(rho_size - 1.0, 1.0)
        else:
            drho_eff = jnp.asarray(1.0, dtype=rho.dtype)
        # For approximately uniform grids this is the local cell width; for
        # edge-clustered grids it is a conservative global proxy.
        drho_eff = jnp.maximum(drho_eff, 1.0 / jnp.maximum(rho_size, 1.0))
        width_cells = info["width"] / (drho_eff + 1e-12)
        min_cells = getattr(sim, "pedestal_alpha_calibration_min_width_cells", 0.75)
        trans_cells = getattr(sim, "pedestal_alpha_calibration_width_cells", 0.25)
        alpha_resolution_factor = 0.5 * (
            1.0 + jnp.tanh((width_cells - min_cells) / (trans_cells + 1e-8))
        )
        discrete_scale = 1.0 + alpha_resolution_factor * (raw_discrete_scale - 1.0)
    else:
        raw_discrete_scale = _maybe_bound(raw_discrete_scale, 0.05, 20.0, sim, 1e-2)
        discrete_scale = raw_discrete_scale
        width_cells = jnp.asarray(jnp.nan, dtype=rho.dtype)
        alpha_resolution_factor = jnp.asarray(1.0, dtype=rho.dtype)

    feedback_scale, alpha_actual = alpha_tracking_pressure_rescale(
        rho, state, machine, sim, q, info
    )
    if getattr(sim, "pedestal_alpha_resolution_guard", False):
        feedback_scale = 1.0 + alpha_resolution_factor * (feedback_scale - 1.0)

    dp_eff = info["dp_ped"] * discrete_scale * feedback_scale
    Te_tgt, Ti_tgt, ne_tgt = _targets_from_pressure_height(
        rho, machine, actuator, sim, dp_eff, info["width"]
    )

    info = dict(info)
    info["dp_ped_eff"] = dp_eff
    info["alpha_target_discrete"] = alpha_tgt
    info["alpha_discrete_scale"] = discrete_scale
    info["alpha_discrete_scale_raw"] = raw_discrete_scale
    info["alpha_resolution_factor"] = alpha_resolution_factor
    info["alpha_width_cells"] = width_cells
    info["alpha_tracking_scale"] = feedback_scale
    info["alpha_actual"] = alpha_actual
    return Te_tgt, Ti_tgt, ne_tgt, info

def eped1_nn_jax_pedestal_targets(rho, state, machine, actuator, sim, q, beta_N_proxy=None, eq=None, dV_drho=None):
    """JAX-native EPED1-NN pedestal target with alpha-critical fallback.

    This path parses GA BrainFUSE/FANN ``brainfuse_*.net`` files and evaluates
    the ensemble as a JAX MLP.  Therefore gradients can pass from the pedestal
    target back to the EPED1-NN input features and, through them, to tokagrad
    controls such as Ip, Bt, shape, and density.
    """
    beta_N_input = _resolve_beta_N_proxy(state, machine, actuator, sim, beta_N_proxy, eq=eq, dV_drho=dV_drho)
    try:
        from .eped1nn_adapter import build_eped1nn_input
        from .eped1nn_jax import predict_eped1nn_jax
        max_nets = int(getattr(sim, "eped1nn_jax_max_nets", 0))
        x = build_eped1nn_input(machine, actuator, sim, beta_N_proxy=beta_N_input)
        y = predict_eped1nn_jax(
            x,
            getattr(sim, "eped1nn_model_dir", "external_models/neural"),
            getattr(sim, "eped1nn_model_name", "EPED1_H_superH"),
            max_nets=max_nets,
        )
    except Exception:
        if getattr(sim, "eped1nn_fail_mode", "fallback") == "raise":
            raise
        Te_tgt, Ti_tgt, ne_tgt, info = alpha_critical_pedestal_targets(
            rho, state, machine, actuator, sim, q
        )
        info = dict(info)
        info["model_used"] = jnp.asarray(0.0)  # 0 = alpha fallback
        return Te_tgt, Ti_tgt, ne_tgt, info

    # Low-level BrainFUSE EPED1_H_superH output names are
    #   OUT_p_E1_0, OUT_p_E1_2, OUT_wid_E1_0, OUT_wid_E1_2.
    # The pressure outputs are MPa-like (0.008 -> 8 kPa in the GA samples),
    # while the width outputs are normalized psi widths.  Average the two EPED
    # branches and infer Te/Ti from total pedestal pressure and ne_ped.
    p_ped_MPa = 0.5 * (y[0] + y[1])
    width_raw = 0.5 * (y[2] + y[3])

    width = _maybe_bound(
        width_raw, sim.pedestal_width_min, sim.pedestal_width_max, sim, 1e-3
    )
    ptotped_Pa = _maybe_lower(p_ped_MPa, 0.0, sim, 1e-3) * 1.0e6 * sim.eped1nn_output_scale

    ne_height = sim.pedestal_density_height20
    neped20 = _maybe_lower(target_edge_ne20(machine, actuator, sim) + ne_height, 0.05, sim, 1e-3)
    Tsum_keV = ptotped_Pa / _smooth_lower(neped20 * 1.0e20 * KEV_TO_J, 1.0e-30, 1.0e-30)
    f_e = _maybe_bound(sim.pedestal_te_fraction, 1.0e-6, 1.0 - 1.0e-6, sim, 1.0e-6)
    Te_ped_keV = f_e * Tsum_keV
    Ti_ped_keV = (1.0 - f_e) * Tsum_keV
    Te_height = _maybe_lower(Te_ped_keV - actuator.edge_Te_keV, 0.0, sim, 1e-3)
    Ti_height = _maybe_lower(Ti_ped_keV - actuator.edge_Ti_keV, 0.0, sim, 1e-3)

    Te_tgt = _tanh_pedestal_profile(
        rho, actuator.edge_Te_keV, Te_height, width, sim.pedestal_transition_sharpness
    )
    Ti_tgt = _tanh_pedestal_profile(
        rho, actuator.edge_Ti_keV, Ti_height, width, sim.pedestal_transition_sharpness
    )
    ne_tgt = _tanh_pedestal_profile(
        rho, target_edge_ne20(machine, actuator, sim), ne_height, width, sim.pedestal_transition_sharpness
    )

    info = {
        "dp_ped": jnp.asarray(ptotped_Pa),
        "dp_ped_eff": jnp.asarray(ptotped_Pa),
        "width": jnp.asarray(width),
        "rho_top": jnp.asarray(1.0 - width),
        "alpha_crit": jnp.asarray(-2.0),
        "q_top": _interp_at_rho(rho, q, 1.0 - width),
        "s_top": _interp_at_rho(rho, _magnetic_shear(q, rho), 1.0 - width),
        "beta_p_ped": jnp.asarray(0.0),
        "model_used": jnp.asarray(2.0),  # 2 = JAX EPED1-NN
        "eped1nn_beta_N": jnp.asarray(beta_N_input),
        "eped1nn_p_ped_MPa": jnp.asarray(p_ped_MPa),
        "eped1nn_ptotped_kPa": jnp.asarray(ptotped_Pa / 1.0e3),
        "eped1nn_width_raw": jnp.asarray(width_raw),
        "eped1nn_Te_ped_keV": jnp.asarray(Te_ped_keV),
        "eped1nn_Ti_ped_keV": jnp.asarray(Ti_ped_keV),
    }
    return Te_tgt, Ti_tgt, ne_tgt, info


def pedestal_target_profiles(rho, state, machine, actuator, sim, q=None, beta_N_proxy=None, eq=None, dV_drho=None):
    """Return target pedestal profiles and metadata for the selected closure.

    This centralizes the choice of alpha-critical, external EPED1-NN, or
    JAX-native EPED1-NN.  Enforcement modes (soft source, smooth blend, etc.)
    should call this helper rather than duplicating target construction.
    """
    if sim.pedestal_model == "none":
        z = jnp.zeros_like(rho)
        info = {
            "dp_ped": jnp.asarray(0.0),
            "dp_ped_eff": jnp.asarray(0.0),
            "width": jnp.asarray(0.0),
            "rho_top": jnp.asarray(1.0),
            "alpha_crit": jnp.asarray(0.0),
            "q_top": jnp.asarray(0.0),
            "s_top": jnp.asarray(0.0),
            "beta_p_ped": jnp.asarray(0.0),
            "model_used": jnp.asarray(-1.0),
        }
        return z, z, z, info

    q = _require_equilibrium_q(q, rho, "pedestal_target_profiles")

    if sim.pedestal_model == "alpha_critical":
        return alpha_critical_pedestal_targets(rho, state, machine, actuator, sim, q)
    if sim.pedestal_model == "eped1_nn_jax":
        return eped1_nn_jax_pedestal_targets(
            rho, state, machine, actuator, sim, q, beta_N_proxy=beta_N_proxy, eq=eq, dV_drho=dV_drho
        )
    raise ValueError(
        f"Unknown pedestal_model={sim.pedestal_model!r}. "
        'Use "alpha_critical", "eped1_nn_jax", or "none".'
    )


def _pedestal_enforcement_mode(sim):
    return getattr(sim, "pedestal_enforcement", "tanh_blend")


def _pedestal_source_enabled(sim):
    mode = _pedestal_enforcement_mode(sim)
    return mode == "soft_source"


def _pedestal_blend_enabled(sim):
    mode = _pedestal_enforcement_mode(sim)
    return mode == "tanh_blend"


def _pedestal_underlay_enabled(sim):
    mode = _pedestal_enforcement_mode(sim)
    return mode == "tanh_underlay"


def _pedestal_enforcement_start(width, sim):
    """Inner edge of the imposed pedestal region.

    Earlier versions turned the pedestal mask on at ``rho_top - 2*width``.  For
    wide early-ramp pedestals this reached far into the core and caused a
    transient temperature bump around rho~0.7.  The direct enforcement should
    instead be localized near the pedestal top, while the target profile itself
    still supplies the edge-to-top tanh shape.
    """
    width = _maybe_bound(width, sim.pedestal_width_min, sim.pedestal_width_max, sim, 1e-3)
    rho_top = 1.0 - width
    sharp = getattr(sim, "pedestal_blend_mask_sharpness", 0.02)
    sharp = jnp.where(sharp > 0.0, sharp, getattr(sim, "pedestal_transition_sharpness", 0.012))
    factor = getattr(sim, "pedestal_blend_core_width_factor", 2.0)
    return jnp.clip(rho_top - factor * sharp, 0.0, 1.0), sharp


def _smooth_pedestal_blend_mask(rho, width, sim):
    """Smooth mask that is ~0 in the core and ~1 in the pedestal/edge region."""
    start, sharp = _pedestal_enforcement_start(width, sim)
    return 0.5 * (1.0 + jnp.tanh((rho - start) / (sharp + 1e-8)))




def _limit_profile_correction(delta, limit, sim, width=1.0e-2):
    """Limit one direct pedestal-profile correction in AD-friendly form."""
    limit = getattr(sim, limit, 0.0) if isinstance(limit, str) else limit
    if limit is None or float(limit) <= 0.0:
        return delta
    if getattr(sim, "differentiable_smooth_mode", False):
        return limit * jnp.tanh(delta / (limit + width))
    return jnp.clip(delta, -limit, limit)



def _smooth_max_pair(a, b, width):
    """AD-friendly smooth approximation to max(a, b)."""
    w = jnp.maximum(jnp.asarray(width), 1.0e-8)
    x = (a - b) / w
    return b + w * jnp.logaddexp(0.0, x)


def apply_tanh_pedestal_underlay(
    rho, Te, Ti, ne20, psi_ind, machine, actuator, sim, q=None, lh_gate=1.0, beta_N_proxy=None
):
    """Impose a whole-profile tanh pedestal underlay.

    Unlike tanh_blend, this mode does not multiply the target by a pedestal-region
    mask.  It treats the target tanh pedestal as a smooth radial floor:

        y_floor = smooth_max(y_old, y_tanh_target)
        y_new   = y_old + f * (y_floor - y_old)

    Thus the pedestal component is present from core to edge, while the pre-existing
    core excess sits on top of it.  This avoids a non-monotonic dip just inside the
    pedestal top when the old profile lies below the imposed pedestal plateau.
    """
    if sim.pedestal_model == "none":
        return Te, Ti, ne20
    q = _require_equilibrium_q(q, rho, "apply_tanh_pedestal_underlay")

    class _S:
        pass
    state = _S()
    state.Te = Te
    state.Ti = Ti
    state.ne20 = ne20
    state.psi_ind = psi_ind

    Te_tgt, Ti_tgt, ne_tgt, _info = pedestal_target_profiles(
        rho, state, machine, actuator, sim, q=q, beta_N_proxy=beta_N_proxy
    )
    frac = _maybe_bound(getattr(sim, "pedestal_blend_fraction", 1.0), 0.0, 1.0, sim, 1e-4)
    strength = jnp.clip(jnp.asarray(lh_gate), 0.0, 1.0)
    f = strength * frac

    wT = getattr(sim, "pedestal_underlay_smooth_keV", 0.05)
    wn = getattr(sim, "pedestal_underlay_smooth_ne20", 0.005)
    Te_floor = _smooth_max_pair(Te, f * Te_tgt, wT)
    Ti_floor = _smooth_max_pair(Ti, f * Ti_tgt, wT)
    dTe = _limit_profile_correction(Te_floor - Te, "pedestal_blend_max_delta_keV", sim)
    dTi = _limit_profile_correction(Ti_floor - Ti, "pedestal_blend_max_delta_keV", sim)
    Te_new = Te + dTe
    Ti_new = Ti + dTi

    if getattr(sim, "pedestal_blend_include_density", True):
        #ne_floor = _smooth_max_pair(ne20, ne_tgt, wn)
        #dne = _limit_profile_correction(
        #    f * (ne_floor - ne20), "pedestal_blend_max_delta_ne20", sim, width=1.0e-3
        #)
        ne_floor = _smooth_max_pair(ne20, f * ne_tgt, wn)
        dne = _limit_profile_correction(
            ne_floor - ne20, "pedestal_blend_max_delta_ne20", sim, width=1.0e-3
        )
        ne_new = ne20 + dne
    else:
        ne_new = ne20

    return (
        _maybe_bound(Te_new, actuator.edge_Te_keV, 80.0, sim, 1e-3),
        _maybe_bound(Ti_new, actuator.edge_Ti_keV, 80.0, sim, 1e-3),
        _maybe_bound(ne_new, 1.0e-4, 5.0, sim, 1e-4),
    )

def apply_tanh_pedestal_blend(
    rho, Te, Ti, ne20, psi_ind, machine, actuator, sim, q=None, lh_gate=1.0, beta_N_proxy=None
):
    """Smoothly impose a tanh pedestal target by additive/blending correction.

    The update is AD-friendly:
        y_new = y + f * mask(rho) * (y_target - y)
    where mask is a smooth tanh function.  This is equivalent to blending toward
    the target profile, but can also be interpreted as adding a smooth pedestal
    correction.  No hard index or moving-boundary branch is used.
    """
    if sim.pedestal_model == "none":
        return Te, Ti, ne20
    q = _require_equilibrium_q(q, rho, "apply_tanh_pedestal_blend")

    class _S:
        pass
    state = _S()
    state.Te = Te
    state.Ti = Ti
    state.ne20 = ne20
    state.psi_ind = psi_ind

    Te_tgt, Ti_tgt, ne_tgt, info = pedestal_target_profiles(
        rho, state, machine, actuator, sim, q=q, beta_N_proxy=beta_N_proxy
    )
    mask = _smooth_pedestal_blend_mask(rho, info["width"], sim)
    frac = _maybe_bound(getattr(sim, "pedestal_blend_fraction", 1.0), 0.0, 1.0, sim, 1e-4)
    strength = jnp.clip(jnp.asarray(lh_gate), 0.0, 1.0)
    f = strength * frac
    dTe = _limit_profile_correction(mask * (f * Te_tgt - Te), "pedestal_blend_max_delta_keV", sim)
    dTi = _limit_profile_correction(mask * (f * Ti_tgt - Ti), "pedestal_blend_max_delta_keV", sim)
    Te_new = Te + dTe
    Ti_new = Ti + dTi
    if getattr(sim, "pedestal_blend_include_density", True):
        dne = _limit_profile_correction(mask * (f * ne_tgt - ne20), "pedestal_blend_max_delta_ne20", sim, width=1.0e-3)
        ne_new = ne20 + dne
    else:
        ne_new = ne20

    return (
        _maybe_bound(Te_new, actuator.edge_Te_keV, 80.0, sim, 1e-3),
        _maybe_bound(Ti_new, actuator.edge_Ti_keV, 80.0, sim, 1e-3),
        _maybe_bound(ne_new, 1.0e-4, 5.0, sim, 1e-4),
    )


def actual_alpha_profile(rho, state, machine, q):
    """Ballooning alpha profile from actual pressure gradient."""
    ne = _smooth_lower(state.ne20, 0.0, 1.0e-4) * 1.0e20
    p = ne * (state.Te + state.Ti) * KEV_TO_J
    dpdr = _grad(p, rho, machine.a)
    qpos = _smooth_lower(q, 0.1, 1e-3)
    alpha = (
        2.0 * MU0 * machine.R0 * qpos ** 2
        / (machine.Bt ** 2 + 1e-12)
        * _smooth_abs(dpdr, 1e-8)
    )
    return alpha

def pedestal_alpha_window_max(rho, alpha_profile, width):
    """Smooth max alpha in a resolved pedestal window."""
    rho_top = 1.0 - width
    rho_min = _smooth_lower(rho_top - 2.0 * width, 0.0, 1e-3)
    # Smooth window and log-sum-exp max for AD mode.
    window = 0.5 * (1.0 + jnp.tanh((rho - rho_min) / (0.5 * width + 1e-6)))
    weighted = window * alpha_profile
    return _smooth_max(weighted, sharpness=40.0)

def alpha_tracking_pressure_rescale(rho, state, machine, sim, q, info):
    """Mild feedback factor from actual alpha toward alpha_crit.

    The main alpha calibration is done on the discrete target profile in
    alpha_critical_pedestal_targets. This feedback is deliberately mild to
    avoid oscillatory overshoot.
    """
    if not sim.pedestal_alpha_tracking:
        return 1.0, 0.0

    alpha_prof = actual_alpha_profile(rho, state, machine, q)
    alpha_actual = pedestal_alpha_window_max(rho, alpha_prof, info["width"])
    alpha_crit = _smooth_lower(info["alpha_crit"], 1e-6, 1e-6)
    ratio = alpha_crit / _smooth_lower(alpha_actual, 0.1 * alpha_crit, 1e-4)
    scale = ratio ** (0.25 * sim.pedestal_alpha_tracking_gain)
    scale = _maybe_bound(scale, 0.6, 2.0, sim, 1e-2)
    return scale, alpha_actual


def martin_lh_surface_area(machine):
    """Approximate plasma surface area S [m^2] for Martin/ITPA08 scaling."""
    return (2.0 * jnp.pi) ** 2 * machine.a * machine.R0 * jnp.sqrt(_smooth_lower(machine.kappa, 0.2, 1e-3))

def martin_lh_threshold_power_MW(nbar20, machine, sim):
    """Martin/ITPA08 L-H threshold power scaling [MW].

    Reference: [Y. R. Martin et al., J. Phys.: Conf. Ser. 123, 012033 (2008)].

    P_LH = 0.0488 n20^0.717 B_T^0.803 S^0.941
    where n20 is line-/volume-averaged density in 1e20 m^-3, B_T in T,
    and S is plasma surface area in m^2.
    """
    S = martin_lh_surface_area(machine)
    return sim.martin_lh_coeff * (
        _smooth_lower(nbar20, 1e-4, 1e-4) ** sim.martin_lh_n_exp
    ) * (
        _smooth_lower(machine.Bt, 0.1, 1e-3) ** sim.martin_lh_bt_exp
    ) * (
        _smooth_lower(S, 1e-4, 1e-4) ** sim.martin_lh_s_exp
    )

def delabie_lh_threshold_power_MW(nbar20, machine, sim):
    """Delabie/ITPA26 L-H threshold power scaling [MW].

    Reference: [E. Delabie et al., Nucl. Fusion (2026),
    doi:10.1088/1741-4326/ae39f2].
    
    P_LH = 0.0441 n20^1.08 B_T^0.580 (2/M_eff^0.975) D S
    where n20 is line-/volume-averaged density in 1e20 m^-3, B_T in T,
    S is plasma surface area in m^2, 
    and D is divertor configuration (1 for HT-like, 1.93 for VT-like).
    """
    S = martin_lh_surface_area(machine)
    return sim.delabie_lh_coeff * (
         _smooth_lower(nbar20, 1e-4, 1e-4) ** sim.delabie_lh_n_exp
     ) * (
         _smooth_lower(machine.Bt, 0.1, 1e-3) ** sim.delabie_lh_bt_exp
     ) * (
         2.0 / effective_ion_mass_amu(machine) ** sim.delabie_lh_meff_exp
     ) * (
         sim.delabie_lh_d_exp
     ) * (
         _smooth_lower(S, 1e-4, 1e-4) ** sim.delabie_lh_s_exp
    )

def lh_transition_gate(P_eff_MW, P_LH_MW, sim):
    """Continuous L-H gate in [pedestal_lh_min_gate, 1]."""
    #if not sim.pedestal_lh_transition_control:
    if sim.pedestal_lh_threshold_model == "none":
        return jnp.asarray(1.0)
    x = (P_eff_MW - sim.pedestal_lh_threshold_scale * P_LH_MW) / (
        sim.pedestal_lh_margin * _smooth_lower(P_LH_MW, 1e-6, 1e-6) + 1e-12
    )
    gate = 0.5 * (1.0 + jnp.tanh(x))
    return sim.pedestal_lh_min_gate + (1.0 - sim.pedestal_lh_min_gate) * gate


def pedestal_cached_targets(
    rho, state, machine, actuator, sim, q=None, beta_N_proxy=None, lh_gate=1.0
):
    """Return LH-gated pedestal targets for cached stepping.

    The expensive part of a pedestal update is usually ``pedestal_target_profiles``
    (EPED1-NN or alpha-critical target construction).  A cached solver should
    recompute those targets only every ``pedestal_skip_steps`` steps, but the
    cheap relaxation source and the cheap tanh projection must still be applied
    against the *current* profiles every transport time step.  Otherwise a
    stale source/projection can overshoot or disappear between refreshes.

    Returns
    -------
    Te_tgt, Ti_tgt, ne_tgt : arrays
        LH-gated target profiles used by the soft pedestal source.
    active : scalar
        0/1-like multiplier for the source when the L-H gate is below cutoff.
    Te_goal, Ti_goal, ne_goal : arrays
        Final profile goals used by tanh_blend/tanh_underlay direct projection.
        These include the projection fraction and reproduce the existing
        project_pedestal_alpha logic without recomputing the target model.
    width : scalar
        Pedestal width used by the blend mask.
    """
    z = jnp.zeros_like(rho)
    if sim.pedestal_model == "none":
        return z, z, z, jnp.asarray(0.0), z, z, z, jnp.asarray(0.05)

    q = _require_equilibrium_q(q, rho, "pedestal_cached_targets")
    Te_tgt, Ti_tgt, ne_tgt, info = pedestal_target_profiles(
        rho, state, machine, actuator, sim, q=q, beta_N_proxy=beta_N_proxy
    )
    frac = _maybe_bound(getattr(sim, "pedestal_blend_fraction", 1.0), 0.0, 1.0, sim, 1e-4)
    strength = jnp.clip(jnp.asarray(lh_gate), 0.0, 1.0)
    active = jnp.asarray(1.0, dtype=strength.dtype)
    projection_factor = strength * frac

    return (
        Te_tgt,
        Ti_tgt,
        ne_tgt,
        active,
        projection_factor * Te_tgt,
        projection_factor * Ti_tgt,
        projection_factor * ne_tgt,
        info["width"],
    )


def pedestal_sources_from_cached_targets(
    rho, state, sim, Te_tgt, Ti_tgt, ne_tgt, active, width
):
    """Cheap pedestal relaxation source using cached target profiles."""
    z = jnp.zeros_like(rho)
    if sim.pedestal_model == "none" or (not _pedestal_source_enabled(sim)):
        return z, z, z
    mask = _smooth_pedestal_blend_mask(rho, width, sim)
    tau = sim.pedestal_pressure_relax_tau
    Te_rate = mask * (Te_tgt - state.Te) / tau
    Ti_rate = mask * (Ti_tgt - state.Ti) / tau
    ne_rate = mask * (ne_tgt - state.ne20) / tau
    max_T_rate = sim.pedestal_source_max_delta_keV / (sim.dt + 1e-12)
    Te_rate = active * _maybe_rate_limit(Te_rate, max_T_rate, sim, 1e-2)
    Ti_rate = active * _maybe_rate_limit(Ti_rate, max_T_rate, sim, 1e-2)
    ne_rate = active * _maybe_rate_limit(ne_rate, 0.05 / (sim.dt + 1e-12), sim, 1e-3)
    return Te_rate, Ti_rate, ne_rate


def project_pedestal_alpha_from_cached_targets(
    rho, Te, Ti, ne20, machine, actuator, sim, Te_goal, Ti_goal, ne_goal, width
):
    """Cheap tanh pedestal projection using cached target profiles.

    This function intentionally supports the smooth tanh modes, where the target
    profile is the expensive part and the actual projection is cheap.  Legacy
    hard alpha projection still falls back to the non-cached path by simply not
    doing anything here.
    """
    if sim.pedestal_model == "none":
        return Te, Ti, ne20
    mode = _pedestal_enforcement_mode(sim)
    if mode in ("none", "off"):
        return Te, Ti, ne20

    if mode == "tanh_underlay":
        wT = getattr(sim, "pedestal_underlay_smooth_keV", 0.05)
        wn = getattr(sim, "pedestal_underlay_smooth_ne20", 0.005)
        Te_floor = _smooth_max_pair(Te, Te_goal, wT)
        Ti_floor = _smooth_max_pair(Ti, Ti_goal, wT)
        dTe = _limit_profile_correction(Te_floor - Te, "pedestal_blend_max_delta_keV", sim)
        dTi = _limit_profile_correction(Ti_floor - Ti, "pedestal_blend_max_delta_keV", sim)
        Te_new = Te + dTe
        Ti_new = Ti + dTi
        if getattr(sim, "pedestal_blend_include_density", True):
            ne_floor = _smooth_max_pair(ne20, ne_goal, wn)
            dne = _limit_profile_correction(
                ne_floor - ne20, "pedestal_blend_max_delta_ne20", sim, width=1.0e-3
            )
            ne_new = ne20 + dne
        else:
            ne_new = ne20
        return (
            _maybe_bound(Te_new, actuator.edge_Te_keV, 80.0, sim, 1e-3),
            _maybe_bound(Ti_new, actuator.edge_Ti_keV, 80.0, sim, 1e-3),
            _maybe_bound(ne_new, 1.0e-4, 5.0, sim, 1e-4),
        )

    if mode == "tanh_blend":
        mask = _smooth_pedestal_blend_mask(rho, width, sim)
        dTe = _limit_profile_correction(mask * (Te_goal - Te), "pedestal_blend_max_delta_keV", sim)
        dTi = _limit_profile_correction(mask * (Ti_goal - Ti), "pedestal_blend_max_delta_keV", sim)
        Te_new = Te + dTe
        Ti_new = Ti + dTi
        if getattr(sim, "pedestal_blend_include_density", True):
            dne = _limit_profile_correction(mask * (ne_goal - ne20), "pedestal_blend_max_delta_ne20", sim, width=1.0e-3)
            ne_new = ne20 + dne
        else:
            ne_new = ne20
        return (
            _maybe_bound(Te_new, actuator.edge_Te_keV, 80.0, sim, 1e-3),
            _maybe_bound(Ti_new, actuator.edge_Ti_keV, 80.0, sim, 1e-3),
            _maybe_bound(ne_new, 1.0e-4, 5.0, sim, 1e-4),
        )

    return Te, Ti, ne20


def pedestal_sources(rho, state, machine, actuator, sim, q=None, beta_N_proxy=None, lh_gate=1.0):
    """Return explicit pedestal relaxation sources.

    For tanh_blend enforcement the pedestal is imposed by a smooth post-transport
    profile correction, so explicit pedestal source terms are disabled.
    """
    z = jnp.zeros_like(rho)
    if sim.pedestal_model == "none" or (not _pedestal_source_enabled(sim)):
        return z, z, z

    q = _require_equilibrium_q(q, rho, "pedestal_sources")

    Te_tgt, Ti_tgt, ne_tgt, info = pedestal_target_profiles(
        rho, state, machine, actuator, sim, q=q, beta_N_proxy=beta_N_proxy
    )
    strength = jnp.clip(jnp.asarray(lh_gate), 0.0, 1.0)
    active = jnp.asarray(1.0, dtype=strength.dtype)

    width = info["width"]
    mask = _smooth_pedestal_blend_mask(rho, width, sim)
    tau = sim.pedestal_pressure_relax_tau
    Te_rate = mask * (Te_tgt - state.Te) / tau
    Ti_rate = mask * (Ti_tgt - state.Ti) / tau
    ne_rate = mask * (ne_tgt - state.ne20) / tau

    # Keep pedestal source strong enough to realize alpha_crit, but bounded.
    max_T_rate = sim.pedestal_source_max_delta_keV / (sim.dt + 1e-12)
    Te_rate = active * _maybe_rate_limit(Te_rate, max_T_rate, sim, 1e-2)
    Ti_rate = active * _maybe_rate_limit(Ti_rate, max_T_rate, sim, 1e-2)
    ne_rate = active * _maybe_rate_limit(ne_rate, 0.05 / (sim.dt + 1e-12), sim, 1e-3)
    return Te_rate, Ti_rate, ne_rate

def pedestal_diagnostics(rho, state, machine, actuator, sim, q=None, beta_N_proxy=None):
    if sim.pedestal_model == "none":
        return {}
    q = _require_equilibrium_q(q, rho, "pedestal_diagnostics")
    _, _, _, info = pedestal_target_profiles(rho, state, machine, actuator, sim, q=q, beta_N_proxy=beta_N_proxy)
    alpha_prof = actual_alpha_profile(rho, state, machine, q)
    alpha_actual = pedestal_alpha_window_max(rho, alpha_prof, info["width"])
    return {
        "ped_alpha_crit": info["alpha_crit"],
        "ped_alpha_actual": alpha_actual,
        "ped_alpha_tracking_scale": info.get("alpha_tracking_scale", jnp.asarray(1.0)),
        "ped_alpha_discrete_scale": info.get("alpha_discrete_scale", jnp.asarray(1.0)),
        "ped_alpha_target_discrete": info.get("alpha_target_discrete", jnp.asarray(0.0)),
        "ped_width": info["width"],
        "ped_rho_top": info["rho_top"],
        "ped_beta_p": info["beta_p_ped"],
        "ped_dp_Pa": info["dp_ped"],
        "ped_dp_eff_Pa": info.get("dp_ped_eff", info["dp_ped"]),
        "ped_q_top": info["q_top"],
        "ped_s_top": info["s_top"],
        "ped_model_used": info.get("model_used", jnp.asarray(0.0)),
        "eped1nn_teped_eV": info.get("eped1nn_teped_eV", jnp.asarray(0.0)),
        "eped1nn_p_ped_MPa": info.get("eped1nn_p_ped_MPa", jnp.asarray(0.0)),
        "eped1nn_ptotped_kPa": info.get("eped1nn_ptotped_kPa", jnp.asarray(0.0)),
        "eped1nn_width_raw": info.get("eped1nn_width_raw", jnp.asarray(0.0)),
        "eped1nn_Te_ped_keV": info.get("eped1nn_Te_ped_keV", jnp.asarray(0.0)),
        "eped1nn_Ti_ped_keV": info.get("eped1nn_Ti_ped_keV", jnp.asarray(0.0)),
        "eped1nn_beta_N": info.get("eped1nn_beta_N", jnp.asarray(0.0)),
    }


def project_pedestal_alpha(rho, Te, Ti, ne20, psi_ind, machine, actuator, sim, q=None, lh_gate=1.0, beta_N_proxy=None):
    """Apply the selected direct pedestal profile enforcement.

    Historically this function implemented a fast direct projection that
    accompanied the soft pedestal source.  It now also handles the AD-friendly
    tanh_blend mode, where the pedestal target is imposed through a smooth
    additive/blending correction instead of through stiff source terms.
    """
    if sim.pedestal_model == "none":
        return Te, Ti, ne20

    mode = _pedestal_enforcement_mode(sim)
    if mode in ("none", "off"):
        return Te, Ti, ne20

    if _pedestal_underlay_enabled(sim):
        return apply_tanh_pedestal_underlay(
            rho, Te, Ti, ne20, psi_ind, machine, actuator, sim, q=q, lh_gate=lh_gate, beta_N_proxy=beta_N_proxy
        )

    if _pedestal_blend_enabled(sim):
        return apply_tanh_pedestal_blend(
            rho, Te, Ti, ne20, psi_ind, machine, actuator, sim, q=q, lh_gate=lh_gate, beta_N_proxy=beta_N_proxy
        )

    # Legacy soft-source + optional projection behavior.  The direct projection
    # remains controlled by pedestal_alpha_tracking and pedestal_projection_fraction.
    if (not sim.pedestal_alpha_tracking) or getattr(sim, "pedestal_projection_fraction", 0.0) <= 0.0:
        return Te, Ti, ne20

    q = _require_equilibrium_q(q, rho, "project_pedestal_alpha")

    class _S:
        pass
    state = _S()
    state.Te = Te
    state.Ti = Ti
    state.ne20 = ne20
    state.psi_ind = psi_ind

    Te_tgt, Ti_tgt, ne_tgt, info = pedestal_target_profiles(rho, state, machine, actuator, sim, q=q, beta_N_proxy=beta_N_proxy)

    width = info["width"]
    # Blend only near/inside the pedestal region.  This avoids forcing the
    # high pedestal-top target deep into the core during early ramp-up.
    mask = _smooth_pedestal_blend_mask(rho, width, sim)
    strength = jnp.clip(jnp.asarray(lh_gate), 0.0, 1.0)
    f = strength * jnp.clip(sim.pedestal_projection_fraction, 0.0, 1.0) * mask

    dTe = _limit_profile_correction(f * (Te_tgt - Te), "pedestal_blend_max_delta_keV", sim)
    dTi = _limit_profile_correction(f * (Ti_tgt - Ti), "pedestal_blend_max_delta_keV", sim)
    dne = _limit_profile_correction(f * (ne_tgt - ne20), "pedestal_blend_max_delta_ne20", sim, width=1.0e-3)
    Te_new = Te + dTe
    Ti_new = Ti + dTi
    ne_new = ne20 + dne
    return (
        jnp.clip(Te_new, actuator.edge_Te_keV, 80.0),
        jnp.clip(Ti_new, actuator.edge_Ti_keV, 80.0),
        jnp.clip(ne_new, target_edge_ne20(machine, actuator, sim), 5.0),
    )
