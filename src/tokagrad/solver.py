"""Coupled radial transport stepping and nonlinear time-integration schemes.

References:
  [J. Citrin et al., arXiv:2406.06718 (2024)] -- differentiable 1-D tokamak
    transport equations and JAX implementation context.
  [S. V. Patankar, Numerical Heat Transfer and Fluid Flow (1980)] -- finite
    volume diffusion and conservative discretization.

Rate limiters, source clipping, cadence caches, pedestal enforcement, and
Greenwald profile rescaling are TokaGrad robustness/control closures.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .grid import make_grid_from_config, infer_rho_faces, axis_augmented_profile, axis_augmented_volume_element, axis_augmented_volume_element_from_dV_drho, volume_element_from_dV_drho
from .profiles import PlasmaState, make_initial_profiles, pedestal_profile
from .heating import total_heating_sources, volume_element_1d
from .transport import compute_diffusivity, transport_fluxes
from .current import psi_inductive_update, current_components_from_state, total_current, psi_from_current_density, ensure_psi_state_grid, saturated_conductivity_current_components, is_psi_diffusion_model, initial_current_shape, normalize_current_profile, poloidal_area_weights
from .equilibrium import solve_fixed_boundary_equilibrium
from .pedestal import pedestal_sources, project_pedestal_alpha, martin_lh_threshold_power_MW, delabie_lh_threshold_power_MW, lh_transition_gate, pedestal_target_profiles, pedestal_cached_targets, pedestal_sources_from_cached_targets, project_pedestal_alpha_from_cached_targets
from .waveforms import apply_waveform_controls
from .density import greenwald_density_1e20, target_nbar20, target_edge_ne20



class StepCache(NamedTuple):
    """Cached expensive submodel outputs for update-cadence acceleration."""
    eq: object
    chi_e: jnp.ndarray
    chi_i: jnp.ndarray
    Dn: jnp.ndarray
    Te_source_base: jnp.ndarray
    Ti_source_base: jnp.ndarray
    ne_source_base: jnp.ndarray
    heat_diag: dict
    # Cached pedestal target data.  The target model and L-H gate are refreshed
    # only every pedestal_skip_steps, but the cheap source/projection is applied
    # every transport step against the current profiles.
    ped_Te_tgt: jnp.ndarray
    ped_Ti_tgt: jnp.ndarray
    ped_ne_tgt: jnp.ndarray
    ped_active: jnp.ndarray
    ped_Te_goal: jnp.ndarray
    ped_Ti_goal: jnp.ndarray
    ped_ne_goal: jnp.ndarray
    ped_width: jnp.ndarray
    lh_gate: jnp.ndarray


def _skip_period(sim, name: str) -> int:
    """Return integer update cadence for a cached submodel."""
    val = getattr(sim, name, 1)
    try:
        return max(1, int(val))
    except Exception:
        return 1


def _skip_enabled(sim) -> bool:
    return any(
        _skip_period(sim, name) > 1
        for name in (
            "equilibrium_skip_steps",
            "source_skip_steps",
            "transport_skip_steps",
            "pedestal_skip_steps",
        )
    )


def _refresh_mask(i, period: int):
    return jnp.equal(jnp.mod(i, period), 0)


def _safe_tridiagonal_denominator(x):
    """Avoid exact-zero pivots without changing ordinary pivots."""
    tiny = jnp.asarray(1.0e-30, dtype=x.dtype)
    return jnp.where(jnp.abs(x) > tiny, x, jnp.where(x >= 0.0, tiny, -tiny))


def solve_tridiagonal(lower, diag, upper, rhs):
    """Differentiable Thomas solve for a tridiagonal linear system.

    Parameters
    ----------
    lower, upper : arrays with shape ``(n-1,)``
        Sub- and super-diagonal entries.
    diag, rhs : arrays with shape ``(n,)``
        Main diagonal and right-hand side.

    This is O(n) and avoids building/factorizing a dense ``n x n`` matrix for
    the 1D radial diffusion operators used by the semi-implicit solver.
    """
    n = diag.shape[0]
    if n == 1:
        return rhs / _safe_tridiagonal_denominator(diag)

    upper_full = jnp.concatenate([upper, jnp.zeros((1,), dtype=diag.dtype)])
    d0 = _safe_tridiagonal_denominator(diag[0])
    c0 = upper_full[0] / d0
    r0 = rhs[0] / d0

    def forward(carry, i):
        c_prev, r_prev = carry
        denom = _safe_tridiagonal_denominator(diag[i] - lower[i - 1] * c_prev)
        c_i = upper_full[i] / denom
        r_i = (rhs[i] - lower[i - 1] * r_prev) / denom
        return (c_i, r_i), (c_i, r_i)

    (_, _), (c_tail, r_tail) = jax.lax.scan(forward, (c0, r0), jnp.arange(1, n))
    cprime = jnp.concatenate([jnp.asarray([c0], dtype=diag.dtype), c_tail])
    rprime = jnp.concatenate([jnp.asarray([r0], dtype=diag.dtype), r_tail])

    def backward(x_next, i):
        x_i = rprime[i] - cprime[i] * x_next
        return x_i, x_i

    x_last = rprime[-1]
    _, x_rev = jax.lax.scan(backward, x_last, jnp.arange(n - 2, -1, -1))
    return jnp.concatenate([x_rev[::-1], jnp.asarray([x_last], dtype=diag.dtype)])


def _tridiagonal_to_dense(lower, diag, upper):
    """Dense matrix view of tridiagonal coefficients, for diagnostics/legacy uses."""
    n = diag.shape[0]
    L = jnp.diag(diag)
    L = L + jnp.diag(upper, k=1)
    L = L + jnp.diag(lower, k=-1)
    return L

def diffusion_tridiagonal_and_edge_source(diffusivity, rho, a, edge_value, dV_drho=None):
    """Tridiagonal radial diffusion operator and edge source.

    Returns ``lower, main, upper, b`` for the operator ``L`` such that
    ``d y / dt = L y + b``.  ``lower`` and ``upper`` have length ``nr-1``.
    """
    nr = rho.size
    rho_f = infer_rho_faces(rho)
    drho_cell = rho_f[1:] - rho_f[:-1]
    r_c = a * rho
    r_f = a * rho_f

    d_ext = jnp.concatenate([diffusivity[0:1], diffusivity, diffusivity[-1:]])
    d_face = 0.5 * (d_ext[:-1] + d_ext[1:])

    # Distance from cell center to neighboring center/edge in normalized rho.
    dist_lo = jnp.concatenate([(rho[0] - rho_f[0])[None], rho[1:] - rho[:-1]])
    dist_hi = jnp.concatenate([rho[1:] - rho[:-1], (rho_f[-1] - rho[-1])[None]])

    if dV_drho is None:
        lo = r_f[:-1] * d_face[:-1] / ((r_c + 1e-12) * a**2 * drho_cell * (dist_lo + 1e-12))
        hi = r_f[1:] * d_face[1:] / ((r_c + 1e-12) * a**2 * drho_cell * (dist_hi + 1e-12))
    else:
        Vp = jnp.maximum(dV_drho, 1e-12)
        Vp_ext = jnp.concatenate([Vp[0:1], Vp, Vp[-1:]])
        Vp_face = 0.5 * (Vp_ext[:-1] + Vp_ext[1:])
        G = Vp_face * d_face / (a**2 + 1e-12)
        lo = G[:-1] / (Vp * drho_cell * (dist_lo + 1e-12))
        hi = G[1:] / (Vp * drho_cell * (dist_hi + 1e-12))

    lo = lo.at[0].set(0.0)  # axis no-flux

    main = -(lo + hi)
    upper = hi[:-1]
    lower = lo[1:]

    b = jnp.zeros(nr, dtype=rho.dtype)
    b = b.at[-1].set(hi[-1] * edge_value)
    return lower, main, upper, b


def diffusion_matrix_and_edge_source(diffusivity, rho, a, edge_value, dV_drho=None):
    """Build radial diffusion operator on uniform or non-uniform rho grids.

    This legacy helper returns a dense matrix.  Time stepping uses the
    tridiagonal coefficients from ``diffusion_tridiagonal_and_edge_source`` to
    avoid dense linear solves.
    """
    lower, main, upper, b = diffusion_tridiagonal_and_edge_source(
        diffusivity, rho, a, edge_value, dV_drho=dV_drho
    )
    return _tridiagonal_to_dense(lower, main, upper), b

def finite_volume_diffusion(y, diffusivity, rho, a, edge_value, dV_drho=None):
    L, b = diffusion_matrix_and_edge_source(diffusivity, rho, a, edge_value, dV_drho=dV_drho)
    return L @ y + b

def finite_volume_flux_divergence(face_flux, rho, a, dV_drho=None):
    """Return -div(F) for face fluxes with length nr+1."""
    rho_f = infer_rho_faces(rho)
    drho_cell = rho_f[1:] - rho_f[:-1]
    if dV_drho is None:
        r_c = a * rho
        r_f = a * rho_f
        return - (r_f[1:] * face_flux[1:] - r_f[:-1] * face_flux[:-1]) / (
            (r_c + 1e-12) * a * drho_cell
        )
    Vp = jnp.maximum(dV_drho, 1e-12)
    Vp_ext = jnp.concatenate([Vp[0:1], Vp, Vp[-1:]])
    Vp_face = 0.5 * (Vp_ext[:-1] + Vp_ext[1:])
    return - (Vp_face[1:] * face_flux[1:] - Vp_face[:-1] * face_flux[:-1]) / (Vp * drho_cell)

def face_average(y):
    """Face values with endpoint hold, length nr+1."""
    return jnp.concatenate([y[0:1], 0.5 * (y[:-1] + y[1:]), y[-1:]])


def reduced_g1(dV_drho, machine):
    """Fallback reduced g1 metric when an Equilibrium object is absent."""
    Vp = jnp.maximum(dV_drho, 1e-12)
    return (Vp * Vp) / (machine.a**2 + 1e-12)



def _centered_transport_tridiagonal_for_conservative_variable(rho, A_cell, B_cell, scale_cell, C_cell, edge_value):
    """Tridiagonal FV operator for conservative-variable transport.

    This is the structured version of
    ``_centered_transport_matrix_for_conservative_variable``.  It returns
    ``lower, main, upper, b`` for ``dU/dt = L U + b``.
    """
    nr = rho.size
    rho_f = infer_rho_faces(rho)
    drho_cell = rho_f[1:] - rho_f[:-1]
    A_face = face_average(A_cell)
    B_face = face_average(B_cell)
    C = jnp.maximum(C_cell, 1.0e-30)
    scale = scale_cell

    lower = jnp.zeros((max(nr - 1, 0),), dtype=rho.dtype)
    upper = jnp.zeros((max(nr - 1, 0),), dtype=rho.dtype)
    main = jnp.zeros((nr,), dtype=rho.dtype)
    b = jnp.zeros((nr,), dtype=rho.dtype)

    if nr > 1:
        # Interior faces: face k separates cell k-1 and k.
        d = rho[1:] - rho[:-1] + 1.0e-12
        A = A_face[1:-1]
        B = B_face[1:-1]
        cL = -A / (d * C[:-1]) - 0.5 * B / C[:-1]
        cR =  A / (d * C[1:])  - 0.5 * B / C[1:]

        upper = scale[:-1] * cR / (drho_cell[:-1] + 1.0e-12)
        lower = -scale[1:] * cL / (drho_cell[1:] + 1.0e-12)
        main = main.at[:-1].add(scale[:-1] * cL / (drho_cell[:-1] + 1.0e-12))
        main = main.at[1:].add(-scale[1:] * cR / (drho_cell[1:] + 1.0e-12))

    # Edge face: y_edge is prescribed.  Use the same centered convective face
    # value convention as interior faces.
    d_edge = (rho_f[-1] - rho[-1]) + 1.0e-12
    A_edge = A_face[-1]
    B_edge = B_face[-1]
    cN = -A_edge / (d_edge * C[-1]) - 0.5 * B_edge / C[-1]
    cE =  A_edge / d_edge - 0.5 * B_edge
    main = main.at[-1].add(scale[-1] * cN / (drho_cell[-1] + 1.0e-12))
    b = b.at[-1].add(scale[-1] * cE * edge_value / (drho_cell[-1] + 1.0e-12))
    return lower, main, upper, b


def _conservative_advection_rate(U, rho, phi_rate):
    """Explicit A2 Phi_b-dot advection term: +(phi_dot/2phi) d(rho U)/drho."""
    phi_rate = jnp.asarray(phi_rate, dtype=U.dtype)
    face_val = face_average(rho * U)
    return 0.5 * phi_rate * _div_center_flux(face_val, rho)


def heat_convection_profile(rho, sim, species):
    """Return heat-convection coefficient q_conv for one species.

    The default is zero, matching the previous solver.  Nonzero constant
    coefficients can be enabled later through config without changing the
    PDE discretization.
    """
    model = getattr(sim, "heat_convection_model", "none")
    if model in ("none", None, ""):
        return jnp.zeros_like(rho)
    if model == "constant":
        val = getattr(sim, "heat_convection_e_base", 0.0) if species == "e" else getattr(sim, "heat_convection_i_base", 0.0)
        return jnp.ones_like(rho) * val
    raise ValueError(f"Unknown heat_convection_model={model!r}.")


def particle_convection_profile(rho, sim):
    """Return particle pinch/convection coefficient V_e."""
    model = getattr(sim, "particle_convection_model", "none")
    if model in ("none", None, ""):
        return jnp.zeros_like(rho)
    if model == "constant":
        return jnp.ones_like(rho) * getattr(sim, "particle_convection_base", 0.0)
    raise ValueError(f"Unknown particle_convection_model={model!r}.")


def _div_center_flux(face_flux, rho):
    rho_f = infer_rho_faces(rho)
    drho_cell = rho_f[1:] - rho_f[:-1]
    return (face_flux[1:] - face_flux[:-1]) / (drho_cell + 1e-12)



def effective_phi_dot_over_phi(state, eq, sim):
    """Return Phi_b_dot/Phi_b used by transport equations.

    Automatic part uses the previous state's reduced Phi_b and the current
    equilibrium Phi_b.  Manual sim.phi_dot_over_phi is added on top.
    """
    manual = jnp.asarray(getattr(sim, "phi_dot_over_phi", 0.0), dtype=eq.Phi_b.dtype)
    if not getattr(sim, "auto_phi_dot_over_phi", True):
        return manual
    prev = jnp.asarray(getattr(state, "Phi_b_prev", 0.0), dtype=eq.Phi_b.dtype)
    dt = jnp.asarray(getattr(sim, "dt", 1.0), dtype=eq.Phi_b.dtype)
    auto = jnp.where(
        prev > 1.0e-12,
        (eq.Phi_b - prev) / (dt * prev + 1.0e-12),
        0.0,
    )
    lim = jnp.asarray(getattr(sim, "phi_dot_over_phi_clip", 5.0), dtype=eq.Phi_b.dtype)
    auto = jnp.clip(auto, -lim, lim)
    return manual + auto


def previous_dV_drho(state, eq):
    """Return the stored old-time V'(rho), falling back for legacy states."""
    prev = jnp.asarray(
        getattr(state, "dV_drho_prev", jnp.asarray(0.0)),
        dtype=eq.dV_drho.dtype,
    )
    # Array shapes are static under JIT, so this compatibility branch is
    # resolved while tracing. Newly initialized states always take the first
    # branch; scalar legacy/external states use the current geometry once.
    if prev.ndim == 1 and prev.shape == eq.dV_drho.shape:
        return jnp.maximum(prev, 1.0e-12)
    return jnp.maximum(eq.dV_drho, 1.0e-12)


def semi_implicit_heat_update(
    T,
    chi,
    n20,
    source,
    rho,
    machine,
    sim,
    edge_value,
    dt,
    eq,
    phi_rate,
    q_conv=None,
    *,
    n20_old=None,
    n20_new=None,
    dV_drho_old=None,
):
    """Semi-implicit heat equation in conservative form.

    References: [J. Citrin et al., arXiv:2406.06718 (2024)];
    [S. V. Patankar, Numerical Heat Transfer and Fluid Flow (1980)].

    The solved variable is
        U_s = V'^(5/3) n_s T_s.

    The implemented equation is the heat PDE without expanding the
    Phi_b-dot operator into the effective source/convection terms:

        (3/2) V'^(-5/3) (d/dt - phidot/2 d/drho rho)
        [V'^(5/3) n_s T_s]
          = (1/V') d/drho[ chi_s n_s g1/V' dT_s/drho
                            - g0 q_conv,s T_s ] + Q_s.

    ``n20`` and ``eq`` are the coefficient-evaluation state. ``n20_old`` and
    ``dV_drho_old`` define the conservative variable at the start of the step;
    ``n20_new`` and ``eq`` define it at the end.  Keeping these separate is
    essential: with zero heat flux/source, an increase in density or V' must
    reduce T so that V'^(5/3)*n*T remains constant.

    ``source`` is the existing temperature-rate source Q_s/(1.5 n_s) [keV/s]
    evaluated with ``n20``. It is converted back to the conservative-variable
    RHS with the same coefficient-evaluation density and geometry.
    """
    Vp = jnp.maximum(eq.dV_drho, 1e-12)
    Vp_old = jnp.maximum(
        eq.dV_drho if dV_drho_old is None else dV_drho_old,
        1e-12,
    )
    g1 = jnp.maximum(getattr(eq, "g1", reduced_g1(Vp, machine)), 1e-12)
    g0 = jnp.maximum(getattr(eq, "g0", Vp), 1e-12)
    n = jnp.maximum(n20, 1e-6)
    n_old = jnp.maximum(n20 if n20_old is None else n20_old, 1e-6)
    n_new = jnp.maximum(n20 if n20_new is None else n20_new, 1e-6)
    q_conv = jnp.zeros_like(T) if q_conv is None else q_conv

    A = chi * n * g1 / Vp
    B = g0 * q_conv

    C_old = (Vp_old ** (5.0 / 3.0)) * n_old
    C_new = (Vp ** (5.0 / 3.0)) * n_new
    C_source = (Vp ** (5.0 / 3.0)) * n
    U_old = C_old * T
    scale = (2.0 / 3.0) * (Vp ** (2.0 / 3.0))
    lower, main, upper, b = _centered_transport_tridiagonal_for_conservative_variable(
        rho, A, B, scale, C_new, edge_value
    )

    source_U = C_source * source
    phidot_U = _conservative_advection_rate(U_old, rho, phi_rate)
    rhs = U_old + dt * (source_U + phidot_U + b)
    U_new = solve_tridiagonal(-dt * lower, 1.0 - dt * main, -dt * upper, rhs)
    return U_new / (C_new + 1.0e-30)


def semi_implicit_density_update(
    n20,
    Dn,
    source,
    rho,
    machine,
    sim,
    edge_value,
    dt,
    eq,
    phi_rate,
    V_e=None,
    *,
    dV_drho_old=None,
):
    """Semi-implicit particle equation in conservative A2 form.

    References: [J. Citrin et al., arXiv:2406.06718 (2024)];
    [S. V. Patankar, Numerical Heat Transfer and Fluid Flow (1980)].

    The solved variable is N = V' n_e:

        (d/dt - phidot/2 d/drho rho)[n_e V']
          = d/drho[ D_e g1/V' dn_e/drho - g0 V_e n_e ] + V' S_n.

    This fixes the previous reduced form where the diffusive coefficient was
    effectively D*n*g1/V'.  ``Dn`` is now an ordinary particle diffusivity.
    When ``dV_drho_old`` is supplied, the transient is discretized as
    ``(V'_new*n_new - V'_old*n_old)/dt``.
    """
    Vp = jnp.maximum(eq.dV_drho, 1e-12)
    Vp_old = jnp.maximum(
        eq.dV_drho if dV_drho_old is None else dV_drho_old,
        1e-12,
    )
    g1 = jnp.maximum(getattr(eq, "g1", reduced_g1(Vp, machine)), 1e-12)
    g0 = jnp.maximum(getattr(eq, "g0", Vp), 1e-12)
    V_e = jnp.zeros_like(n20) if V_e is None else V_e

    A = Dn * g1 / Vp
    B = g0 * V_e
    C_new = Vp
    N_old = Vp_old * n20
    scale = jnp.ones_like(n20)
    lower, main, upper, b = _centered_transport_tridiagonal_for_conservative_variable(
        rho, A, B, scale, C_new, edge_value
    )

    source_N = Vp * source
    phidot_N = _conservative_advection_rate(N_old, rho, phi_rate)
    rhs = N_old + dt * (source_N + phidot_N + b)
    N_new = solve_tridiagonal(-dt * lower, 1.0 - dt * main, -dt * upper, rhs)
    return N_new / (C_new + 1.0e-30)


def particle_source_profile(rho, machine, actuator, sim):
    edge = target_edge_ne20(machine, actuator, sim)
    nbar = target_nbar20(machine, actuator, sim)
    return edge + (nbar * 1.25 - edge) * (1.0 - rho**2) ** 0.8





def _cell_volume_weights_for_integral(rho, machine, dV_drho=None):
    """Return per-cell volume weights [m^3] for cell-centered profiles.

    ``eq.dV_drho`` is a derivative with respect to normalized rho, not a shell
    volume.  Multiply it by the cell width before integrating power-density
    profiles.  This helper intentionally mirrors diagnostics.power_integral_axis_augmented
    without adding an explicit zero-volume axis sample.
    """
    if dV_drho is None:
        return volume_element_1d(rho, machine)
    return volume_element_from_dV_drho(rho, dV_drho)


def _integrate_power_profile(rho, profile, machine, dV_drho=None):
    return jnp.sum(profile * _cell_volume_weights_for_integral(rho, machine, dV_drho=dV_drho))


def pedestal_lh_gate_from_heating(rho, state, machine, actuator, sim, heat_diag, dV_drho=None):
    """Compute L-H pedestal gate from heating diagnostics.

    If equilibrium geometry is available, pass ``dV_drho=eq.dV_drho`` so the
    threshold power basis and line-averaged density use the same cell-volume
    weights as diagnostics.  Note that ``dV_drho`` is V'(rho), so it must be
    converted to finite-volume shell volumes before integrating MW/m^3 profiles.
    """
    P_aux = _integrate_power_profile(rho, heat_diag["Paux_e"] + heat_diag["Paux_i"], machine, dV_drho=dV_drho)
    P_ohm = _integrate_power_profile(rho, heat_diag["Pohm_e"], machine, dV_drho=dV_drho)
    P_alpha = _integrate_power_profile(rho, heat_diag["Palpha_e"] + heat_diag["Palpha_i"], machine, dV_drho=dV_drho)
    P_rad = _integrate_power_profile(rho, heat_diag["Prad_e"], machine, dV_drho=dV_drho)
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
    nbar20 = volume_average_profile(state.ne20, _cell_volume_weights_for_integral(rho, machine, dV_drho=dV_drho))
    if sim.pedestal_lh_threshold_model == "delabie":
        P_LH = delabie_lh_threshold_power_MW(nbar20, machine, sim)
    else:
        P_LH = martin_lh_threshold_power_MW(nbar20, machine, sim)

    gate = lh_transition_gate(P_eff, P_LH, sim)
    #jax.debug.print("gate={x} (Pohm={Pohm}, Paux={Paux}, Pa={Pa}, Prad={Prad}, Pabs={Pabs}, Psep={Psep}, PLH={PLH})", x=gate, Pohm=P_ohm, Pabs=P_abs, Pa=P_alpha, Prad=P_rad, Psep=P_eff, PLH=P_LH, Paux=P_aux)
    return gate, P_eff, P_LH, P_abs, P_rad

def volume_average_profile(y, dV_drho):
    return jnp.sum(y * dV_drho) / (jnp.sum(dV_drho) + 1e-12)

def greenwald_target_density_1e20(machine, actuator_or_sim, sim=None):
    """Target volume-averaged density from the requested Greenwald fraction."""
    if sim is None:
        # Backward-compatible call: greenwald_target_density_1e20(machine, actuator, sim)
        sim = actuator_or_sim
        actuator = None
    else:
        actuator = actuator_or_sim
    if actuator is None:
        return getattr(sim, "greenwald_fraction_target", 0.9) * greenwald_density_1e20(machine)
    return target_nbar20(machine, actuator, sim)


def density_model_uses_initial_shape_rescale(sim):
    """True if density should bypass particle diffusion and keep the initial shape."""
    return getattr(sim, "density_evolution_model", "diffusive") in (
        "greenwald_rescale_initial_shape",
        "greenwald_initial_shape_rescale",
        "initial_shape_rescale",
    )


def density_model_uses_tanh_rescale(sim):
    """True if density should use a Greenwald-rescaled tanh+core shape."""
    return getattr(sim, "density_evolution_model", "diffusive") in (
        "greenwald_rescale_tanh",
        "greenwald_tanh_rescale",
        "tanh_rescale",
    )


def density_model_uses_direct_rescale(sim):
    """True if density is prescribed by an algebraic Greenwald rescale closure."""
    return density_model_uses_initial_shape_rescale(sim) or density_model_uses_tanh_rescale(sim)


def rescale_initial_density_to_greenwald(rho, machine, actuator, sim, dV_drho, max_ne=5.0):
    """Preserve the initial density shape and rescale it to target f_G.

    This closure is useful for fast final-state scans where particle transport is
    not the quantity of interest.  It bypasses the density diffusion equation and
    sets

        n_e(rho) = C n_e,initial(rho),

    with C chosen so the volume-average density equals
    actuator.greenwald_fraction_target * n_G.  The profile shape is therefore exactly
    inherited from the selected initial_profile_model up to the optional safety
    ceiling/floor.
    """
    _Te0, _Ti0, ne0, _psi0, _psi_edge0 = make_initial_profiles(rho, actuator, machine, sim)
    shape = jnp.maximum(ne0, 1.0e-8)
    # Match the app/diagnostic Greenwald fraction definition, which uses an
    # axis-augmented volume integral.  The dV_drho argument is kept for API
    # compatibility with the diffusion closures, but the direct shape-rescale
    # mode intentionally targets the user-facing volume average.
    _, shape_aug = axis_augmented_profile(rho, shape)
    if dV_drho is None:
        dV_aug = axis_augmented_volume_element(rho, machine)
    else:
        dV_aug = axis_augmented_volume_element_from_dV_drho(rho, dV_drho)
    nbar_shape = jnp.sum(shape_aug * dV_aug) / (jnp.sum(dV_aug) + 1.0e-12)
    target = greenwald_target_density_1e20(machine, actuator, sim)
    scaled = shape * (target / (nbar_shape + 1.0e-12))
    return maybe_clip(scaled, 1.0e-4, max_ne, sim)


def _volume_average_axis_augmented_shape(rho, shape, machine, dV_drho=None):
    _, shape_aug = axis_augmented_profile(rho, shape)
    if dV_drho is None:
        dV_aug = axis_augmented_volume_element(rho, machine)
    else:
        dV_aug = axis_augmented_volume_element_from_dV_drho(rho, dV_drho)
    return jnp.sum(shape_aug * dV_aug) / (jnp.sum(dV_aug) + 1.0e-12)


def _rescale_shape_with_fixed_edge_to_greenwald(rho, base_shape, edge_tgt, target, machine, sim, max_ne=5.0, dV_drho=None):
    """Scale a dimensionless/physical density shape to match edge and f_G."""
    # Only the excess above the chosen edge value is scaled.  This keeps the
    # separatrix density tied to greenwald_edge_density_fraction while the volume
    # average is exactly matched in the user-facing axis-augmented convention.
    excess = jnp.maximum(base_shape - edge_tgt, 0.0)
    avg_excess = _volume_average_axis_augmented_shape(rho, excess, machine, dV_drho=dV_drho)
    amp = jnp.maximum((target - edge_tgt) / (avg_excess + 1.0e-12), 0.0)
    out = edge_tgt + amp * excess
    return maybe_clip(out, 1.0e-4, max_ne, sim)


def _pedestal_width_for_tanh_density(rho, state, machine, actuator, sim, q=None, dV_drho=None):
    """Return a pedestal width for greenwald_rescale_tanh.

    If a pedestal model is active, use the same pedestal width that is used for
    temperature pedestal targets.  Otherwise use a small ITER-H-mode-like
    fallback width.  The density mode should not fail just because the optional
    EPED backend is unavailable; in that case the configured fail_mode still
    controls whether the pedestal target helper raises or falls back.
    """
    width = jnp.asarray(getattr(sim, "greenwald_tanh_default_width", 0.04), dtype=rho.dtype)
    if getattr(sim, "pedestal_model", "none") == "none":
        return width
    if q is None:
        raise ValueError("_pedestal_width_for_tanh_density requires q=eq.q from the active Equilibrium.")
    try:
        try:
            from .diagnostics import beta_normalized_total
            beta_proxy = beta_normalized_total(state, machine, actuator, sim, dV_drho=dV_drho)
        except Exception:
            beta_proxy = 1.5
        _Te_tgt, _Ti_tgt, _ne_tgt, info = pedestal_target_profiles(
            rho, state, machine, actuator, sim, q=q, beta_N_proxy=beta_proxy
        )
        width = info.get("width", width) if isinstance(info, dict) else width
    except Exception:
        if getattr(sim, "eped1nn_fail_mode", "fallback") == "raise":
            raise
    return jnp.clip(jnp.asarray(width, dtype=rho.dtype), 1.0e-4, 0.5)


def rescale_tanh_density_to_greenwald(rho, state, machine, actuator, sim, dV_drho=None, q=None, max_ne=5.0):
    """Use a pedestal-width-aware tanh+core density shape at target f_G.

    This is similar in spirit to greenwald_rescale_initial_shape, but the shape is
    rebuilt analytically each call:

      edge density = greenwald_edge_density_fraction * target nbar
      pedestal top = greenwald_tanh_pedestal_factor (dimensionless)
      core = greenwald_tanh_core_factor (dimensionless)
      pedestal-top location = 1 - pedestal_width

    If alpha-critical or EPED1-NN provides a pedestal width, that width moves the
    tanh transition.  The profile is then rescaled so the axis-augmented volume
    average equals greenwald_fraction_target * n_G.
    """
    target = greenwald_target_density_1e20(machine, actuator, sim)
    edge_frac = getattr(actuator, "greenwald_edge_density_fraction", 0.25)
    edge_tgt = edge_frac * target
    width = _pedestal_width_for_tanh_density(rho, state, machine, actuator, sim, q=q, dV_drho=dV_drho)
    rho_top = 1.0 - width
    transition = jnp.maximum(
        getattr(sim, "pedestal_transition_sharpness", 0.012),
        getattr(sim, "greenwald_tanh_transition_fraction", 0.25) * width,
    )
    # Build a dimensionless shape with edge value equal to edge_tgt so that the
    # fixed-edge rescale can preserve the requested boundary density exactly.
    edge_shape = edge_tgt
    ped_shape = jnp.maximum(getattr(sim, "greenwald_tanh_pedestal_factor", 1.0) * target, edge_shape)
    core_shape = jnp.maximum(getattr(sim, "greenwald_tanh_core_factor", 1.10) * target, ped_shape)
    base_shape = pedestal_profile(
        rho,
        core=core_shape,
        pedestal_top=ped_shape,
        edge=edge_shape,
        rho_top=rho_top,
        width=transition,
        core_alpha=0.45,
    )
    return _rescale_shape_with_fixed_edge_to_greenwald(
        rho, base_shape, edge_tgt, target, machine, sim, max_ne=max_ne, dV_drho=dV_drho
    )


def rescale_density_model_to_greenwald(rho, state, machine, actuator, sim, eq, max_ne=5.0):
    """Apply the active algebraic Greenwald-rescale density closure."""
    if density_model_uses_tanh_rescale(sim):
        return rescale_tanh_density_to_greenwald(
            rho, state, machine, actuator, sim,
            getattr(eq, "dV_drho", None),
            q=getattr(eq, "q", None),
            max_ne=max_ne,
        )
    return rescale_initial_density_to_greenwald(
        rho, machine, actuator, sim, getattr(eq, "dV_drho", None), max_ne=max_ne
    )

def apply_reflective_particle_conservation(rho, ne20, ne_old, dV_drho, sim):
    """Apply edge density source so total particle content is conserved."""
    nbar_old = volume_average_profile(ne_old, dV_drho)
    nbar_new = volume_average_profile(ne20, dV_drho)
    dr = rho[-1] - rho[-2]
    shape = jnp.exp(-0.5 * ((rho - (1.0 - dr)) / (sim.density_boundary_source_width + 1e-8)) ** 2)
    shape = shape / (volume_average_profile(shape, dV_drho) + 1e-12)
    return ne20 + (nbar_old - nbar_new) * shape

def apply_greenwald_matching_feedback(rho, ne20, dV_drho, machine, actuator, sim):
    """Legacy post-step edge correction to match target Greenwald fraction."""
    nbar = volume_average_profile(ne20, dV_drho)
    n_target = target_nbar20(machine, actuator, sim)
    delta = n_target - nbar
    shape = jnp.exp(-0.5 * ((rho - 1.0) / (sim.density_boundary_source_width + 1e-8)) ** 2)
    shape = shape / (volume_average_profile(shape, dV_drho) + 1e-12)
    return ne20 + shape * delta


def apply_greenwald_matching_feedback_implicit_response(
    rho,
    ne_free,
    ne_old,
    Dn,
    eq,
    machine,
    actuator,
    sim,
):
    """Apply Greenwald feedback as a density-solver source response.

    The legacy ``greenwald_feedback`` implementation added a normalized edge
    Gaussian directly after the diffusion solve.  For large dt that produces a
    split-operator pulse: diffusion removes density over the whole step, then a
    large localized correction is deposited at the very end of the step and is
    not diffused until the next step.  This helper computes the response of the
    same implicit density operator to a unit edge source and adds the correction
    in that diffused response shape instead.

    The volume-average error is relaxed by ``1-exp(-dt/tau)`` in physical time,
    so changing dt changes the time discretization error but not the intended
    feedback law.
    """
    dV = eq.dV_drho
    nbar = volume_average_profile(ne_free, dV)
    n_target = target_nbar20(machine, actuator, sim)
    tau = jnp.maximum(jnp.asarray(getattr(sim, "density_feedback_tau", 1.0e-3), dtype=ne_free.dtype), 1.0e-12)
    dt = jnp.asarray(sim.dt, dtype=ne_free.dtype)
    relax = 1.0 - jnp.exp(-dt / tau)
    avg_delta = relax * (n_target - nbar)
    avg_delta = maybe_limit_delta(avg_delta, getattr(sim, "density_source_max_delta", 0.25), sim)

    shape = jnp.exp(-0.5 * ((rho - 1.0) / (sim.density_boundary_source_width + 1.0e-8)) ** 2)
    shape = shape / (volume_average_profile(shape, dV) + 1.0e-12)

    zero = jnp.zeros_like(ne_old)
    response = semi_implicit_density_update(
        zero,
        Dn,
        shape,
        rho,
        machine,
        sim,
        jnp.asarray(0.0, dtype=ne_free.dtype),
        sim.dt,
        eq,
        jnp.asarray(0.0, dtype=ne_free.dtype),
        V_e=jnp.zeros_like(ne_old),
        dV_drho_old=dV,
    )
    response_avg = volume_average_profile(response, dV)
    correction = response * (avg_delta / (response_avg + 1.0e-30))
    return ne_free + correction


def greenwald_feedback_source_rate(rho, ne_ref, dV_drho, machine, actuator, sim):
    """Continuous-time Greenwald feedback source rate for implicit density solve."""
    nbar = volume_average_profile(ne_ref, dV_drho)
    n_target = target_nbar20(machine, actuator, sim)
    tau = jnp.maximum(jnp.asarray(getattr(sim, "density_feedback_tau", 1.0e-3), dtype=ne_ref.dtype), 1.0e-12)
    dt = jnp.asarray(sim.dt, dtype=ne_ref.dtype)
    relax = 1.0 - jnp.exp(-dt / tau)
    avg_delta = relax * (n_target - nbar)
    avg_delta = maybe_limit_delta(avg_delta, getattr(sim, "density_source_max_delta", 0.25), sim)

    shape = jnp.exp(-0.5 * ((rho - 1.0) / (sim.density_boundary_source_width + 1.0e-8)) ** 2)
    shape = shape / (volume_average_profile(shape, dV_drho) + 1.0e-12)
    return shape * avg_delta / (dt + 1.0e-12)


def density_model_uses_implicit_source_feedback(sim):
    model = getattr(sim, "density_evolution_model", "diffusive")
    method = str(getattr(sim, "density_feedback_method", "implicit_source")).lower()
    return model == "greenwald_feedback" and method in ("implicit_source", "source", "in_solve")

def greenwald_boundary_particle_source(rho, ne20, dV_drho, machine, actuator, sim):
    """Edge-localized source rate to maintain target Greenwald fraction."""
    nbar = volume_average_profile(ne20, dV_drho)
    n_target = target_nbar20(machine, actuator, sim)
    delta = n_target - nbar
    
    shape = jnp.exp(-0.5 * ((rho - 1.0) / (sim.density_boundary_source_width + 1e-8)) ** 2)
    shape = shape / (volume_average_profile(shape, dV_drho) + 1e-12)

    # Source is in 1e20 m^-3 / s and changes volume average by delta/tau.
    source = shape * delta / (sim.density_feedback_tau + 1e-12)
    #source = shape * delta / (sim.dt + 1e-12)
    max_rate = sim.density_source_max_delta / (sim.dt + 1e-12)
    if getattr(sim, "differentiable_smooth_mode", False):
        return smooth_symmetric_limit(source, max_rate, getattr(sim, "smooth_rate_width", 1.0e-2) / (sim.dt + 1e-12))
    return jnp.clip(source, 0.0, max_rate)

def apply_density_model_after_update(ne_candidate, state, rho, Dn, ne_source, eq, machine, actuator, sim, semi_implicit=True):
    """Apply selected density closure."""
    model = getattr(sim, "density_evolution_model", "diffusive")

    if model == "diffusive":
        return maybe_clip(ne_candidate, target_edge_ne20(machine, actuator, sim), 5.0, sim)

    if model == "fixed_initial":
        init = initial_state(machine, actuator, sim)
        return init.ne20

    if density_model_uses_initial_shape_rescale(sim):
        return rescale_initial_density_to_greenwald(
            rho, machine, actuator, sim, eq.dV_drho, max_ne=5.0
        )

    if density_model_uses_tanh_rescale(sim):
        return rescale_tanh_density_to_greenwald(
            rho, state, machine, actuator, sim, eq.dV_drho, q=getattr(eq, "q", None), max_ne=5.0
        )

    if model == "reflective":
        return apply_reflective_particle_conservation(
            rho, maybe_clip(ne_candidate, 1e-4, 5.0, sim), state.ne20, eq.dV_drho, sim
        )

    if model == "greenwald_feedback_source":
        # Keep ordinary diffusion, but add an edge fueling/recycling feedback source.
        src = greenwald_boundary_particle_source(rho, state.ne20, eq.dV_drho, machine, actuator, sim)
        src = source_limited_rate(state.ne20, src, 1e-4, sim, ymax=5.0, max_delta=sim.density_source_max_delta)
        if semi_implicit:
            # Apply the feedback as a local source after the transport solve.
            out = ne_candidate + sim.dt * src
        else:
            out = ne_candidate + sim.dt * src
        return maybe_clip(out, 1e-4, 5.0, sim)

    if model == "greenwald_feedback":
        method = str(getattr(sim, "density_feedback_method", "implicit_source")).lower()
        if method in ("implicit_source", "source", "in_solve"):
            return maybe_clip(ne_candidate, 1.0e-4, 5.0, sim)
        if method in ("post_correction", "legacy", "direct"):
            return apply_greenwald_matching_feedback(
                rho, ne_candidate, eq.dV_drho, machine, actuator, sim
            )
        if method in ("implicit_response", "implicit", "response"):
            out = apply_greenwald_matching_feedback_implicit_response(
                rho, ne_candidate, state.ne20, Dn, eq, machine, actuator, sim
            )
            return maybe_clip(out, 1.0e-4, 5.0, sim)
        raise ValueError(
            f"Unknown density_feedback_method={method!r}. "
            'Use "implicit_response" or "post_correction".'
        )

    raise ValueError(
        f"Unknown density_evolution_model={model!r}. "
        'Use "diffusive", "fixed_initial", "reflective", "greenwald_feedback", '
        '"greenwald_rescale_initial_shape", or "greenwald_rescale_tanh".'
    )

def apply_temperature_freeze_if_requested(Te, Ti, machine, actuator, sim):
    if not getattr(sim, "freeze_temperature_profiles", False):
        return Te, Ti
    init = initial_state(machine, actuator, sim)
    return init.Te, init.Ti



def smooth_clip(x, lo, hi, width=1.0e-2):
    """Differentiable approximation to jnp.clip.

    Outside [lo, hi] it approaches the corresponding bound smoothly; inside it
    behaves nearly like the identity.  This is not a hard safety projection, so
    use ordinary mode for production-like robust simulations.
    """
    w = jnp.maximum(width, 1.0e-8)
    y = lo + w * jax.nn.softplus((x - lo) / w)
    y = hi - w * jax.nn.softplus((hi - y) / w)
    return y


def maybe_clip(x, lo, hi, sim, width=None):
    if getattr(sim, "differentiable_smooth_mode", False):
        if width is None:
            width = getattr(sim, "smooth_clip_width", 1.0e-2)
        return smooth_clip(x, lo, hi, width)
    return jnp.clip(x, lo, hi)


def smooth_symmetric_limit(x, max_abs, width=1.0e-2):
    """Smooth approximation to clip(x, -max_abs, max_abs)."""
    m = jnp.maximum(max_abs, 1.0e-12)
    w = jnp.maximum(width, 1.0e-8)
    return m * jnp.tanh(x / (m + w))


def maybe_limit_delta(delta, max_delta, sim):
    if getattr(sim, "differentiable_smooth_mode", False):
        return smooth_symmetric_limit(delta, max_delta, getattr(sim, "smooth_rate_width", 1.0e-2))
    return jnp.clip(delta, -max_delta, max_delta)

def compute_transport_and_sources(state: PlasmaState, machine, actuator, sim):
    rho, _, _ = make_grid_from_config(sim.nr, machine.a, sim)

    # q is required by QLKNN-style transport and pedestal features.
    eq_for_transport = solve_fixed_boundary_equilibrium(state, machine, actuator, sim)
    chi_e, chi_i, Dn = compute_diffusivity(
        rho, state.Te, state.Ti, state.ne20, machine, sim, q=eq_for_transport.q, eq=eq_for_transport
    )

    Te_source, Ti_source, _heat_diag = total_heating_sources(
        rho, state, machine, actuator, sim, eq=eq_for_transport
    )
    lh_gate, _, _, _, _ = pedestal_lh_gate_from_heating(
        rho, state, machine, actuator, sim, _heat_diag, dV_drho=eq_for_transport.dV_drho
    )

    # Base particle source. The default "diffusive" mode has no volumetric
    # density feedback; particles can diffuse out through the edge boundary.
    target_ne = particle_source_profile(rho, machine, actuator, sim)
    if getattr(sim, "density_evolution_model", "diffusive") == "legacy_target_source":
        ne_source = (target_ne - state.ne20) / 0.08
    else:
        ne_source = jnp.zeros_like(state.ne20)

    # Pedestal source, replaceable by EPED1-NN / alpha-critical closures.
    # Use the same total-beta convention as diagnostics and 0.5D mode.
    try:
        from .diagnostics import beta_normalized_total
        beta_proxy = beta_normalized_total(state, machine, actuator, sim, eq=eq_for_transport)
    except Exception:
        beta_proxy = jnp.maximum(jnp.mean(state.Te + state.Ti) / 10.0, 0.0)
    ped_Te, ped_Ti, ped_ne = pedestal_sources(
        rho, state, machine, actuator, sim, q=eq_for_transport.q, beta_N_proxy=beta_proxy, lh_gate=lh_gate
    )
    Te_source = Te_source + ped_Te
    Ti_source = Ti_source + ped_Ti
    ne_source = ne_source + ped_ne

    return rho, chi_e, chi_i, Dn, Te_source, Ti_source, ne_source



def _base_source_terms(rho, state: PlasmaState, eq, machine, actuator, sim):
    """Compute non-pedestal source terms and heating diagnostics."""
    Te_source, Ti_source, heat_diag = total_heating_sources(
        rho, state, machine, actuator, sim, eq=eq
    )
    target_ne = particle_source_profile(rho, machine, actuator, sim)
    if getattr(sim, "density_evolution_model", "diffusive") == "legacy_target_source":
        ne_source = (target_ne - state.ne20) / 0.08
    else:
        ne_source = jnp.zeros_like(state.ne20)
    return Te_source, Ti_source, ne_source, heat_diag


def _transport_coefficients_for_state(rho, state: PlasmaState, eq, machine, sim):
    return compute_diffusivity(
        rho, state.Te, state.Ti, state.ne20, machine, sim, q=eq.q, eq=eq
    )


def _pedestal_cached_terms(rho, state: PlasmaState, eq, heat_diag, machine, actuator, sim):
    """Compute cached pedestal target data and instantaneous source terms.

    ``pedestal_skip_steps`` should skip the expensive target/L-H-gate model, not
    the cheap enforcement itself.  Therefore we cache gated target profiles and
    recompute the relaxation source from those targets and the current profiles
    every transport step.
    """
    lh_gate, _, _, _, _ = pedestal_lh_gate_from_heating(
        rho, state, machine, actuator, sim, heat_diag, dV_drho=eq.dV_drho
    )
    from .diagnostics import beta_normalized_total
    beta_proxy = beta_normalized_total(state, machine, actuator, sim, eq=eq)
    (
        ped_Te_tgt,
        ped_Ti_tgt,
        ped_ne_tgt,
        ped_active,
        ped_Te_goal,
        ped_Ti_goal,
        ped_ne_goal,
        ped_width,
    ) = pedestal_cached_targets(
        rho, state, machine, actuator, sim, q=eq.q, beta_N_proxy=beta_proxy, lh_gate=lh_gate
    )
    return (
        ped_Te_tgt,
        ped_Ti_tgt,
        ped_ne_tgt,
        ped_active,
        ped_Te_goal,
        ped_Ti_goal,
        ped_ne_goal,
        ped_width,
        lh_gate,
    )


def initialize_step_cache(state: PlasmaState, machine, actuator, sim) -> StepCache:
    """Build a cache for expensive submodels from the current state."""
    rho, _, _ = make_grid_from_config(sim.nr, machine.a, sim)
    eq = solve_fixed_boundary_equilibrium(state, machine, actuator, sim)
    chi_e, chi_i, Dn = _transport_coefficients_for_state(rho, state, eq, machine, sim)
    Te_src, Ti_src, ne_src, heat_diag = _base_source_terms(rho, state, eq, machine, actuator, sim)
    (
        ped_Te_tgt,
        ped_Ti_tgt,
        ped_ne_tgt,
        ped_active,
        ped_Te_goal,
        ped_Ti_goal,
        ped_ne_goal,
        ped_width,
        lh_gate,
    ) = _pedestal_cached_terms(rho, state, eq, heat_diag, machine, actuator, sim)
    return StepCache(
        eq,
        chi_e,
        chi_i,
        Dn,
        Te_src,
        Ti_src,
        ne_src,
        heat_diag,
        ped_Te_tgt,
        ped_Ti_tgt,
        ped_ne_tgt,
        ped_active,
        ped_Te_goal,
        ped_Ti_goal,
        ped_ne_goal,
        ped_width,
        lh_gate,
    )


def update_step_cache(state: PlasmaState, cache: StepCache, i, machine, actuator, sim) -> StepCache:
    """Refresh cached submodels according to the configured update cadences."""
    rho, _, _ = make_grid_from_config(sim.nr, machine.a, sim)

    eq_period = _skip_period(sim, "equilibrium_skip_steps")
    src_period = _skip_period(sim, "source_skip_steps")
    tr_period = _skip_period(sim, "transport_skip_steps")
    ped_period = _skip_period(sim, "pedestal_skip_steps")

    eq = jax.lax.cond(
        _refresh_mask(i, eq_period),
        lambda _: solve_fixed_boundary_equilibrium(state, machine, actuator, sim),
        lambda _: cache.eq,
        operand=None,
    )

    chi_e, chi_i, Dn = jax.lax.cond(
        _refresh_mask(i, tr_period),
        lambda _: _transport_coefficients_for_state(rho, state, eq, machine, sim),
        lambda _: (cache.chi_e, cache.chi_i, cache.Dn),
        operand=None,
    )

    Te_src, Ti_src, ne_src, heat_diag = jax.lax.cond(
        _refresh_mask(i, src_period),
        lambda _: _base_source_terms(rho, state, eq, machine, actuator, sim),
        lambda _: (cache.Te_source_base, cache.Ti_source_base, cache.ne_source_base, cache.heat_diag),
        operand=None,
    )

    (
        ped_Te_tgt,
        ped_Ti_tgt,
        ped_ne_tgt,
        ped_active,
        ped_Te_goal,
        ped_Ti_goal,
        ped_ne_goal,
        ped_width,
        lh_gate,
    ) = jax.lax.cond(
        _refresh_mask(i, ped_period),
        lambda _: _pedestal_cached_terms(rho, state, eq, heat_diag, machine, actuator, sim),
        lambda _: (
            cache.ped_Te_tgt,
            cache.ped_Ti_tgt,
            cache.ped_ne_tgt,
            cache.ped_active,
            cache.ped_Te_goal,
            cache.ped_Ti_goal,
            cache.ped_ne_goal,
            cache.ped_width,
            cache.lh_gate,
        ),
        operand=None,
    )

    return StepCache(
        eq,
        chi_e,
        chi_i,
        Dn,
        Te_src,
        Ti_src,
        ne_src,
        heat_diag,
        ped_Te_tgt,
        ped_Ti_tgt,
        ped_ne_tgt,
        ped_active,
        ped_Te_goal,
        ped_Ti_goal,
        ped_ne_goal,
        ped_width,
        lh_gate,
    )

def limit_transport_rate(rate, max_delta, sim):
    return jnp.clip(rate, -max_delta / (sim.dt + 1e-12), max_delta / (sim.dt + 1e-12))

def source_limited_rate(y, source, edge_value, sim, ymax=80.0, max_delta=None):
    """Return source rate after semi-implicit local sink treatment and delta limiting.

    In differentiable_smooth_mode, the hard per-step delta limiter and final
    floor/ceiling are replaced by smooth approximations.  This improves AD/JVP
    behavior but should not be interpreted as a strict safety limiter.
    """
    if max_delta is None:
        max_delta = sim.source_max_delta_keV
    if sim.source_implicitness <= 0.0:
        dy = maybe_limit_delta(sim.dt * source, max_delta, sim)
        y_new = maybe_clip(y + dy, edge_value, ymax, sim)
        return (y_new - y) / (sim.dt + 1e-12)
    pos = jnp.maximum(source, 0.0)
    neg = jnp.maximum(-source, 0.0)
    excess = jnp.maximum(y - edge_value, 1e-4)
    k = neg / excess
    y_new = (y + sim.dt * pos + sim.dt * sim.source_implicitness * k * edge_value) / (1.0 + sim.dt * sim.source_implicitness * k)
    dy = maybe_limit_delta(y_new - y, max_delta, sim)
    y_new = maybe_clip(y + dy, edge_value, ymax, sim)
    return (y_new - y) / (sim.dt + 1e-12)


def explicit_step(state: PlasmaState, machine, actuator, sim):
    rho, chi_e, chi_i, Dn, Te_source, Ti_source, ne_source = compute_transport_and_sources(
        state, machine, actuator, sim
    )
    Te_source = source_limited_rate(state.Te, Te_source, actuator.edge_Te_keV, sim, ymax=80.0)
    Ti_source = source_limited_rate(state.Ti, Ti_source, actuator.edge_Ti_keV, sim, ymax=80.0)
    ne_source = source_limited_rate(state.ne20, ne_source, target_edge_ne20(machine, actuator, sim), sim, ymax=5.0, max_delta=0.02)

    eq = solve_fixed_boundary_equilibrium(state, machine, actuator, sim)
    _, _, heat_diag_for_lh = total_heating_sources(rho, state, machine, actuator, sim, eq=eq)
    lh_gate_step, _, _, _, _ = pedestal_lh_gate_from_heating(
        rho, state, machine, actuator, sim, heat_diag_for_lh, dV_drho=eq.dV_drho
    )
    dV = eq.dV_drho
    direct_density_rescale = density_model_uses_direct_rescale(sim)
    if getattr(sim, "transport_mode", "diffusive") == "flux":
        Qe_f, Qi_f, Gam_f = transport_fluxes(rho, state.Te, state.Ti, state.ne20, eq.q, machine, sim, eq=eq)
        dTe_tr = limit_transport_rate(
            finite_volume_flux_divergence(Qe_f, rho, machine.a, dV_drho=dV),
            sim.transport_flux_max_delta_keV,
            sim,
        )
        dTi_tr = limit_transport_rate(
            finite_volume_flux_divergence(Qi_f, rho, machine.a, dV_drho=dV),
            sim.transport_flux_max_delta_keV,
            sim,
        )
        dTe = dTe_tr + Te_source
        dTi = dTi_tr + Ti_source
        if direct_density_rescale:
            dne = jnp.zeros_like(state.ne20)
        else:
            dne_tr = limit_transport_rate(
                finite_volume_flux_divergence(Gam_f, rho, machine.a, dV_drho=dV),
                sim.transport_flux_max_delta_ne20,
                sim,
            )
            dne = dne_tr + ne_source
    else:
        dTe = finite_volume_diffusion(state.Te, chi_e, rho, machine.a, actuator.edge_Te_keV, dV_drho=dV) + Te_source
        dTi = finite_volume_diffusion(state.Ti, chi_i, rho, machine.a, actuator.edge_Ti_keV, dV_drho=dV) + Ti_source
        if direct_density_rescale:
            dne = jnp.zeros_like(state.ne20)
        else:
            dne = finite_volume_diffusion(state.ne20, Dn, rho, machine.a, target_edge_ne20(machine, actuator, sim), dV_drho=dV) + ne_source

    Te = jnp.clip(state.Te + sim.dt * dTe, actuator.edge_Te_keV, 80.0)
    Ti = jnp.clip(state.Ti + sim.dt * dTi, actuator.edge_Ti_keV, 80.0)
    if direct_density_rescale:
        ne_candidate = rescale_density_model_to_greenwald(rho, state, machine, actuator, sim, eq, max_ne=5.0)
    else:
        ne_candidate = jnp.clip(state.ne20 + sim.dt * dne, target_edge_ne20(machine, actuator, sim), 5.0)
    ne20 = apply_density_model_after_update(
        ne_candidate, state, rho, Dn, ne_source, eq, machine, actuator, sim, semi_implicit=False
    )

    Te, Ti = apply_temperature_freeze_if_requested(Te, Ti, machine, actuator, sim)

    from .diagnostics import beta_normalized_total
    _ped_tmp = PlasmaState(
        Te, Ti, ne20, state.psi_ind, state.psi_edge,
        state.Phi_b_prev, state.dV_drho_prev,
    )
    beta_proxy_step = beta_normalized_total(_ped_tmp, machine, actuator, sim)
    Te, Ti, ne20 = project_pedestal_alpha(
        rho, Te, Ti, ne20, state.psi_ind, machine, actuator, sim, q=eq.q, lh_gate=lh_gate_step, beta_N_proxy=beta_proxy_step
    )

    Te, Ti = apply_temperature_freeze_if_requested(Te, Ti, machine, actuator, sim)
    eq_for_current = eq
    if getattr(sim, "density_evolution_model", "diffusive") == "fixed_initial":
        ne20 = initial_state(machine, actuator, sim).ne20
    elif density_model_uses_direct_rescale(sim):
        tmp_density = PlasmaState(
            Te, Ti, ne20, state.psi_ind, state.psi_edge,
            state.Phi_b_prev, state.dV_drho_prev,
        )
        eq_for_current = solve_fixed_boundary_equilibrium(tmp_density, machine, actuator, sim)
        ne20 = rescale_density_model_to_greenwald(rho, tmp_density, machine, actuator, sim, eq_for_current, max_ne=5.0)
    psi_ind, psi_edge = psi_inductive_update(rho, state, Te, Ti, ne20, machine, actuator, sim, eq=eq_for_current)
    return PlasmaState(
        Te, Ti, ne20, psi_ind, psi_edge,
        eq_for_current.Phi_b, eq_for_current.dV_drho,
    )


def _pack_profiles(state: PlasmaState):
    return jnp.concatenate([state.Te, state.Ti, state.ne20])


def _state_from_profile_vector(x, template: PlasmaState):
    nr = template.Te.shape[0]
    return PlasmaState(
        x[:nr],
        x[nr:2 * nr],
        x[2 * nr:3 * nr],
        template.psi_ind,
        template.psi_edge,
        template.Phi_b_prev,
        template.dV_drho_prev,
    )


def _clip_profile_state(state: PlasmaState, machine, actuator, sim):
    Te = maybe_clip(state.Te, actuator.edge_Te_keV, 80.0, sim)
    Ti = maybe_clip(state.Ti, actuator.edge_Ti_keV, 80.0, sim)
    ne20 = maybe_clip(state.ne20, 1.0e-4, 5.0, sim)
    return PlasmaState(
        Te, Ti, ne20, state.psi_ind, state.psi_edge,
        state.Phi_b_prev, state.dV_drho_prev,
    )


def _blend_profile_states(a: PlasmaState, b: PlasmaState, alpha):
    """Return a + alpha*(b-a) for profiles; preserve/blend psi only as a carry."""
    alpha = jnp.asarray(alpha, dtype=a.Te.dtype)
    return PlasmaState(
        a.Te + alpha * (b.Te - a.Te),
        a.Ti + alpha * (b.Ti - a.Ti),
        a.ne20 + alpha * (b.ne20 - a.ne20),
        a.psi_ind + alpha * (b.psi_ind - a.psi_ind),
        a.psi_edge + alpha * (b.psi_edge - a.psi_edge),
        a.Phi_b_prev + alpha * (b.Phi_b_prev - a.Phi_b_prev),
        a.dV_drho_prev + alpha * (b.dV_drho_prev - a.dV_drho_prev),
    )


def _refresh_current_for_profiles(old_state: PlasmaState, profile_state: PlasmaState, machine, actuator, sim):
    """Recompute the current/psi update after a nonlinear profile solve."""
    rho, _, _ = make_grid_from_config(sim.nr, machine.a, sim)
    eq = solve_fixed_boundary_equilibrium(profile_state, machine, actuator, sim)
    psi_ind, psi_edge = psi_inductive_update(
        rho,
        old_state,
        profile_state.Te,
        profile_state.Ti,
        profile_state.ne20,
        machine,
        actuator,
        sim,
        eq=eq,
    )
    return PlasmaState(
        profile_state.Te, profile_state.Ti, profile_state.ne20,
        psi_ind, psi_edge, eq.Phi_b, eq.dV_drho,
    )


def _implicit_profile_step_with_reference(
    state: PlasmaState,
    reference_state: PlasmaState,
    machine,
    actuator,
    sim,
    update_current: bool = True,
):
    """One backward-Euler transport update with nonlinear coefficients from reference_state.

    ``state`` supplies the old-time profiles appearing in the time derivative.
    ``reference_state`` supplies the nonlinear coefficients, geometry, sources,
    L-H gate, and pedestal targets.  If ``reference_state is state`` this reduces
    to the ordinary semi-implicit step.  Repeating this map as a fixed-point
    iteration gives a practical full-implicit/Picard solve.
    """
    rho, _, _ = make_grid_from_config(sim.nr, machine.a, sim)

    eq = solve_fixed_boundary_equilibrium(reference_state, machine, actuator, sim)
    chi_e, chi_i, Dn = _transport_coefficients_for_state(rho, reference_state, eq, machine, sim)
    Te_source, Ti_source, ne_source, heat_diag = _base_source_terms(
        rho, reference_state, eq, machine, actuator, sim
    )
    (
        ped_Te_tgt,
        ped_Ti_tgt,
        ped_ne_tgt,
        ped_active,
        ped_Te_goal,
        ped_Ti_goal,
        ped_ne_goal,
        ped_width,
        lh_gate_step,
    ) = _pedestal_cached_terms(rho, reference_state, eq, heat_diag, machine, actuator, sim)
    ped_Te, ped_Ti, ped_ne = pedestal_sources_from_cached_targets(
        rho,
        reference_state,
        sim,
        ped_Te_tgt,
        ped_Ti_tgt,
        ped_ne_tgt,
        ped_active,
        ped_width,
    )
    Te_source = Te_source + ped_Te
    Ti_source = Ti_source + ped_Ti
    ne_source = ne_source + ped_ne

    # Keep the per-step limiter anchored to the old state.  The rates themselves
    # are evaluated from the nonlinear reference state.
    Te_source = source_limited_rate(state.Te, Te_source, actuator.edge_Te_keV, sim, ymax=80.0)
    Ti_source = source_limited_rate(state.Ti, Ti_source, actuator.edge_Ti_keV, sim, ymax=80.0)
    ne_source = source_limited_rate(
        state.ne20, ne_source, target_edge_ne20(machine, actuator, sim), sim, ymax=5.0, max_delta=0.02
    )

    dV = eq.dV_drho
    phi_rate = effective_phi_dot_over_phi(reference_state, eq, sim)
    dV_old = previous_dV_drho(state, eq)
    if density_model_uses_implicit_source_feedback(sim):
        ne_source = ne_source + greenwald_feedback_source_rate(
            rho, reference_state.ne20, dV, machine, actuator, sim
        )

    # Advance density first because the heat equation evolves
    # V'^(5/3)*n*T, not T alone.  The updated density therefore belongs in the
    # end-of-step transient coefficient of both heat equations.
    if density_model_uses_direct_rescale(sim):
        ne_candidate = rescale_density_model_to_greenwald(rho, reference_state, machine, actuator, sim, eq, max_ne=5.0)
    else:
        V_e = particle_convection_profile(rho, sim)
        ne_candidate = semi_implicit_density_update(
            state.ne20, Dn, ne_source, rho, machine, sim,
            target_edge_ne20(machine, actuator, sim), sim.dt, eq, phi_rate,
            V_e=V_e, dV_drho_old=dV_old,
        )
    ne_candidate = maybe_clip(ne_candidate, 1.0e-4, 5.0, sim)
    ne20 = apply_density_model_after_update(
        ne_candidate, reference_state, rho, Dn, ne_source, eq, machine, actuator, sim, semi_implicit=True
    )

    qconv_e = heat_convection_profile(rho, sim, "e")
    qconv_i = heat_convection_profile(rho, sim, "i")
    n_for_heat = jnp.maximum(reference_state.ne20, 1.0e-6)
    Te = semi_implicit_heat_update(
        state.Te, chi_e, n_for_heat, Te_source, rho, machine, sim,
        actuator.edge_Te_keV, sim.dt, eq, phi_rate, q_conv=qconv_e,
        n20_old=state.ne20, n20_new=ne20, dV_drho_old=dV_old,
    )
    Ti = semi_implicit_heat_update(
        state.Ti, chi_i, n_for_heat, Ti_source, rho, machine, sim,
        actuator.edge_Ti_keV, sim.dt, eq, phi_rate, q_conv=qconv_i,
        n20_old=state.ne20, n20_new=ne20, dV_drho_old=dV_old,
    )

    Te = maybe_clip(Te, actuator.edge_Te_keV, 80.0, sim)
    Ti = maybe_clip(Ti, actuator.edge_Ti_keV, 80.0, sim)

    Te, Ti = apply_temperature_freeze_if_requested(Te, Ti, machine, actuator, sim)

    _ped_mode = getattr(sim, "pedestal_enforcement", "tanh_blend")
    _smooth_ped = _ped_mode in ("tanh_blend", "tanh_underlay")
    if (not getattr(sim, "differentiable_smooth_mode", False)) or _smooth_ped:
        Te, Ti, ne20 = project_pedestal_alpha_from_cached_targets(
            rho,
            Te,
            Ti,
            ne20,
            machine,
            actuator,
            sim,
            ped_Te_goal,
            ped_Ti_goal,
            ped_ne_goal,
            ped_width,
        )

    Te, Ti = apply_temperature_freeze_if_requested(Te, Ti, machine, actuator, sim)

    profile_state = PlasmaState(
        Te, Ti, ne20, state.psi_ind, state.psi_edge,
        state.Phi_b_prev, state.dV_drho_prev,
    )
    eq_for_current = eq
    if getattr(sim, "density_evolution_model", "diffusive") == "fixed_initial":
        ne20 = initial_state(machine, actuator, sim).ne20
        profile_state = PlasmaState(
            Te, Ti, ne20, state.psi_ind, state.psi_edge,
            state.Phi_b_prev, state.dV_drho_prev,
        )
    elif density_model_uses_direct_rescale(sim):
        # Preserve the old behavior where the density rescale can use geometry
        # consistent with the post-temperature candidate.
        eq_for_current = solve_fixed_boundary_equilibrium(profile_state, machine, actuator, sim)
        ne20 = rescale_density_model_to_greenwald(rho, profile_state, machine, actuator, sim, eq_for_current, max_ne=5.0)
        profile_state = PlasmaState(
            Te, Ti, ne20, state.psi_ind, state.psi_edge,
            state.Phi_b_prev, state.dV_drho_prev,
        )

    if not update_current:
        return profile_state

    psi_ind, psi_edge = psi_inductive_update(
        rho, state, Te, Ti, ne20, machine, actuator, sim, eq=eq_for_current
    )
    return PlasmaState(
        Te, Ti, ne20, psi_ind, psi_edge,
        eq_for_current.Phi_b, eq_for_current.dV_drho,
    )


def semi_implicit_step(state: PlasmaState, machine, actuator, sim):
    """Mainline time step: semi-implicit diffusive transport plus explicit nonlinear coefficients."""
    return _implicit_profile_step_with_reference(state, state, machine, actuator, sim, update_current=True)


def full_implicit_picard_step(state: PlasmaState, machine, actuator, sim):
    """Backward-Euler nonlinear Picard iteration for the profile update.

    This is a practical full-implicit option: geometry, transport coefficients,
    sources, and pedestal targets are re-evaluated from the latest iterate while
    each inner solve keeps the tridiagonal diffusion structure.
    """
    n_iter = max(1, int(getattr(sim, "implicit_nonlinear_iters", 3)))
    alpha = jnp.asarray(getattr(sim, "implicit_relaxation", 1.0), dtype=state.Te.dtype)
    guess = _implicit_profile_step_with_reference(state, state, machine, actuator, sim, update_current=False)
    for _ in range(n_iter):
        cand = _implicit_profile_step_with_reference(state, guess, machine, actuator, sim, update_current=False)
        cand = _clip_profile_state(cand, machine, actuator, sim)
        guess = _blend_profile_states(guess, cand, alpha)
        guess = _clip_profile_state(guess, machine, actuator, sim)
    return _refresh_current_for_profiles(state, guess, machine, actuator, sim)


def predictor_corrector_step(state: PlasmaState, machine, actuator, sim):
    """Predictor-corrector option.

    Predictor: ordinary semi-implicit step using old-state coefficients.
    Corrector: recompute nonlinear coefficients/sources from the predictor and
    solve the backward-Euler linearized system once more from the old state.
    """
    pred = _implicit_profile_step_with_reference(state, state, machine, actuator, sim, update_current=False)
    corr = _implicit_profile_step_with_reference(state, pred, machine, actuator, sim, update_current=False)
    alpha = jnp.asarray(getattr(sim, "predictor_corrector_relaxation", 1.0), dtype=state.Te.dtype)
    prof = _blend_profile_states(pred, corr, alpha)
    prof = _clip_profile_state(prof, machine, actuator, sim)
    return _refresh_current_for_profiles(state, prof, machine, actuator, sim)


def newton_raphson_step(state: PlasmaState, machine, actuator, sim):
    """Newton solve for the nonlinear one-step profile fixed point x = G(x).

    The unknown vector is [Te, Ti, ne].  Current/psi is updated once after the
    nonlinear profile solve.  This option is intended for small-nr experiments
    and method comparisons; it uses a dense 3*nr Jacobian solve.
    """
    pred = _implicit_profile_step_with_reference(state, state, machine, actuator, sim, update_current=False)
    x = _pack_profiles(pred)
    damping = jnp.asarray(getattr(sim, "newton_damping", 0.7), dtype=x.dtype)
    reg = jnp.asarray(getattr(sim, "newton_jacobian_regularization", 1.0e-6), dtype=x.dtype)
    n_iter = max(1, int(getattr(sim, "newton_iters", 4)))

    def fixed_point_residual(x_vec):
        ref = _state_from_profile_vector(x_vec, state)
        mapped = _implicit_profile_step_with_reference(state, ref, machine, actuator, sim, update_current=False)
        return x_vec - _pack_profiles(mapped)

    for _ in range(n_iter):
        r = fixed_point_residual(x)
        J = jax.jacfwd(fixed_point_residual)(x)
        J = J + reg * jnp.eye(J.shape[0], dtype=J.dtype)
        dx = jnp.linalg.solve(J, -r)
        x = x + damping * dx
        x = _pack_profiles(_clip_profile_state(_state_from_profile_vector(x, state), machine, actuator, sim))

    prof = _state_from_profile_vector(x, state)
    prof = _clip_profile_state(prof, machine, actuator, sim)
    return _refresh_current_for_profiles(state, prof, machine, actuator, sim)


def semi_implicit_step_cached(state: PlasmaState, cache: StepCache, i, machine, actuator, sim):
    """Semi-implicit step using cached expensive submodel outputs.

    This is used only when at least one *_skip_steps option is >1.  With all
    skip cadences equal to 1, the legacy semi_implicit_step path is used so the
    default behavior remains unchanged.
    """
    rho, _, _ = make_grid_from_config(sim.nr, machine.a, sim)
    cache = update_step_cache(state, cache, i, machine, actuator, sim)

    chi_e, chi_i, Dn = cache.chi_e, cache.chi_i, cache.Dn
    ped_Te, ped_Ti, ped_ne = pedestal_sources_from_cached_targets(
        rho,
        state,
        sim,
        cache.ped_Te_tgt,
        cache.ped_Ti_tgt,
        cache.ped_ne_tgt,
        cache.ped_active,
        cache.ped_width,
    )
    Te_source = cache.Te_source_base + ped_Te
    Ti_source = cache.Ti_source_base + ped_Ti
    ne_source = cache.ne_source_base + ped_ne

    Te_source = source_limited_rate(state.Te, Te_source, actuator.edge_Te_keV, sim, ymax=80.0)
    Ti_source = source_limited_rate(state.Ti, Ti_source, actuator.edge_Ti_keV, sim, ymax=80.0)
    ne_source = source_limited_rate(state.ne20, ne_source, target_edge_ne20(machine, actuator, sim), sim, ymax=5.0, max_delta=0.02)

    eq = cache.eq
    dV = eq.dV_drho
    phi_rate = effective_phi_dot_over_phi(state, eq, sim)
    dV_old = previous_dV_drho(state, eq)
    if density_model_uses_implicit_source_feedback(sim):
        ne_source = ne_source + greenwald_feedback_source_rate(
            rho, state.ne20, dV, machine, actuator, sim
        )

    # Density supplies the new-time transient coefficient C=V'^(5/3)*n in
    # the heat solve, so update it before Te and Ti.
    if density_model_uses_direct_rescale(sim):
        ne_candidate = rescale_density_model_to_greenwald(rho, state, machine, actuator, sim, eq, max_ne=5.0)
    else:
        V_e = particle_convection_profile(rho, sim)
        ne_candidate = semi_implicit_density_update(
            state.ne20, Dn, ne_source, rho, machine, sim,
            target_edge_ne20(machine, actuator, sim), sim.dt, eq, phi_rate,
            V_e=V_e, dV_drho_old=dV_old,
        )
    ne_candidate = maybe_clip(ne_candidate, 1.0e-4, 5.0, sim)
    ne20 = apply_density_model_after_update(
        ne_candidate, state, rho, Dn, ne_source, eq, machine, actuator, sim, semi_implicit=True
    )

    qconv_e = heat_convection_profile(rho, sim, "e")
    qconv_i = heat_convection_profile(rho, sim, "i")
    Te = semi_implicit_heat_update(
        state.Te, chi_e, state.ne20, Te_source, rho, machine, sim,
        actuator.edge_Te_keV, sim.dt, eq, phi_rate, q_conv=qconv_e,
        n20_old=state.ne20, n20_new=ne20, dV_drho_old=dV_old,
    )
    Ti = semi_implicit_heat_update(
        state.Ti, chi_i, state.ne20, Ti_source, rho, machine, sim,
        actuator.edge_Ti_keV, sim.dt, eq, phi_rate, q_conv=qconv_i,
        n20_old=state.ne20, n20_new=ne20, dV_drho_old=dV_old,
    )

    Te = maybe_clip(Te, actuator.edge_Te_keV, 80.0, sim)
    Ti = maybe_clip(Ti, actuator.edge_Ti_keV, 80.0, sim)

    Te, Ti = apply_temperature_freeze_if_requested(Te, Ti, machine, actuator, sim)

    _ped_mode = getattr(sim, "pedestal_enforcement", "tanh_blend")
    _smooth_ped = _ped_mode in ("tanh_blend", "tanh_underlay")
    allow_projection = ((not getattr(sim, "differentiable_smooth_mode", False)) or _smooth_ped)

    def _project(vals):
        Te0, Ti0, ne0 = vals
        return project_pedestal_alpha_from_cached_targets(
            rho,
            Te0,
            Ti0,
            ne0,
            machine,
            actuator,
            sim,
            cache.ped_Te_goal,
            cache.ped_Ti_goal,
            cache.ped_ne_goal,
            cache.ped_width,
        )

    def _no_project(vals):
        return vals

    Te, Ti, ne20 = jax.lax.cond(
        jnp.asarray(allow_projection),
        _project,
        _no_project,
        operand=(Te, Ti, ne20),
    )

    Te, Ti = apply_temperature_freeze_if_requested(Te, Ti, machine, actuator, sim)
    eq_for_current = eq
    if getattr(sim, "density_evolution_model", "diffusive") == "fixed_initial":
        ne20 = initial_state(machine, actuator, sim).ne20
    elif density_model_uses_direct_rescale(sim):
        # Respect the equilibrium cadence: use the cached equilibrium for the
        # density rescale unless the equilibrium cache is refreshed this step.
        tmp_density = PlasmaState(
            Te, Ti, ne20, state.psi_ind, state.psi_edge,
            state.Phi_b_prev, state.dV_drho_prev,
        )
        eq_for_current = jax.lax.cond(
            _refresh_mask(i, _skip_period(sim, "equilibrium_skip_steps")),
            lambda _: solve_fixed_boundary_equilibrium(tmp_density, machine, actuator, sim),
            lambda _: eq,
            operand=None,
        )
        ne20 = rescale_density_model_to_greenwald(rho, tmp_density, machine, actuator, sim, eq_for_current, max_ne=5.0)

    psi_ind, psi_edge = psi_inductive_update(rho, state, Te, Ti, ne20, machine, actuator, sim, eq=eq_for_current)
    new_state = PlasmaState(
        Te, Ti, ne20, psi_ind, psi_edge,
        eq_for_current.Phi_b, eq_for_current.dV_drho,
    )
    return new_state, cache


def step_cached(state: PlasmaState, cache: StepCache, i, machine, actuator, sim):
    scheme = str(sim.diffusion_scheme).lower()
    if scheme == "semi_implicit":
        return semi_implicit_step_cached(state, cache, i, machine, actuator, sim)
    # Nonlinear schemes re-evaluate submodels inside their inner iterations.
    # The semi-implicit cache is therefore not reused here; keep the cache carry
    # shape stable and run the requested uncached step.
    new_state = step(state, machine, actuator, sim)
    return new_state, cache

def step(state: PlasmaState, machine, actuator, sim):
    scheme = str(sim.diffusion_scheme).lower()
    if scheme == "semi_implicit":
        return semi_implicit_step(state, machine, actuator, sim)
    if scheme in ("full_implicit", "implicit", "picard", "full_implicit_picard"):
        return full_implicit_picard_step(state, machine, actuator, sim)
    if scheme in ("predictor_corrector", "pc"):
        return predictor_corrector_step(state, machine, actuator, sim)
    if scheme in ("newton", "newton_raphson", "newton-raphson"):
        return newton_raphson_step(state, machine, actuator, sim)
    if scheme == "explicit":
        # Legacy path retained for tests only.
        return explicit_step(state, machine, actuator, sim)
    raise ValueError(
        f"Unknown diffusion_scheme={sim.diffusion_scheme!r}. "
        'Use "semi_implicit", "full_implicit", "predictor_corrector", "newton", or "explicit".'
    )

def _ensure_state_psi_grid(state: PlasmaState, machine, actuator, sim):
    """Normalize ``state.psi_ind`` to the configured cell/face grid length.

    This is mainly needed when ``psi_state_grid="face"`` and an initial or
    externally supplied state still carries a legacy cell-grid psi.  Without this
    normalization, current-update branches such as ``saturated_conductivity``
    return an ``nr+1`` face-grid psi after the first step while the scan carry
    entered with length ``nr``.
    """
    rho, _, _ = make_grid_from_config(sim.nr, machine.a, sim)
    psi = ensure_psi_state_grid(
        rho,
        state.psi_ind,
        machine,
        psi_edge=getattr(state, "psi_edge", 0.0),
        sim=sim,
        eq=None,
    )
    dV_prev = jnp.asarray(
        getattr(state, "dV_drho_prev", jnp.asarray(0.0)),
        dtype=state.Te.dtype,
    )
    normalized = PlasmaState(
        state.Te, state.Ti, state.ne20, psi, state.psi_edge,
        state.Phi_b_prev, dV_prev,
    )
    if dV_prev.ndim != 1 or dV_prev.shape != rho.shape:
        eq = solve_fixed_boundary_equilibrium(normalized, machine, actuator, sim)
        normalized = PlasmaState(
            state.Te, state.Ti, state.ne20, psi, state.psi_edge,
            eq.Phi_b, eq.dV_drho,
        )
    return normalized

def initial_state(machine, actuator, sim):
    rho, _, _ = make_grid_from_config(sim.nr, machine.a, sim)
    Te, Ti, ne20, psi_ind, psi_edge = make_initial_profiles(rho, actuator, machine, sim)
    psi_ind = ensure_psi_state_grid(rho, psi_ind, machine, psi_edge=psi_edge, sim=sim)

    # ITER H-mode initial state: make total current close to Ip from the start.
    # This is not q regularization during evolution; it only prevents an initial
    # over-current transient from forcing a large negative loop voltage and
    # artificial edge current reversal.
    if getattr(sim, "initial_profile_model", "h_mode") == "h_mode":
        tmp = PlasmaState(Te, Ti, ne20, psi_ind, psi_edge)
        j_ind, j_bs, j_cd, _ = current_components_from_state(rho, tmp, machine, actuator, sim)
        I_nonind = total_current(rho, j_bs + j_cd, machine)
        I_ind_target = jnp.maximum(machine.Ip - I_nonind, 0.0)
        I_ind = total_current(rho, j_ind, machine)
        j_ind = j_ind * I_ind_target / (I_ind + 1e-12)
        psi_ind = psi_from_current_density(rho, j_ind, machine, psi_edge=psi_edge, sim=sim)

    tmp = PlasmaState(Te, Ti, ne20, psi_ind, psi_edge)
    eq0 = solve_fixed_boundary_equilibrium(tmp, machine, actuator, sim)

    initial_current_model = str(
        getattr(sim, "initial_current_profile_model", "saturated_components")
    ).lower()
    shape_models = ("total_current_shape", "initial_current_shape", "shape")
    saturated_models = ("saturated_components", "saturated", "components")
    if initial_current_model not in shape_models + saturated_models:
        raise ValueError(
            f"Unknown initial_current_profile_model={initial_current_model!r}. "
            "Use 'saturated_components' or 'total_current_shape'."
        )

    # For the psi diffusion branch, psi represents the total-current
    # poloidal flux.  Do not initialize it from an arbitrary polynomial total
    # current and then subtract bootstrap in diagnostics; that creates an
    # artificial negative Ohmic pedestal layer.  Instead build a saturated
    # Ohmic+bootstrap+CD split and store the corresponding total-current psi.
    if initial_current_model in shape_models:
        psi_is_total_current = is_psi_diffusion_model(
            getattr(sim, "current_evolution_model", "")
        )

        def state_current_for_target(j_target, eq_target):
            """Convert a requested total j to the psi convention in use."""
            if psi_is_total_current:
                return j_target
            # Legacy saturated-current mode stores Ohmic/inductive psi.  Remove
            # the non-inductive pieces so its reconstructed *total* still equals
            # initial_current_shape(), as requested by this option.
            psi_probe = psi_from_current_density(
                rho, j_target, machine, psi_edge=psi_edge, sim=sim, eq=eq_target
            )
            probe = PlasmaState(
                Te, Ti, ne20, psi_probe, psi_edge,
                eq_target.Phi_b, eq_target.dV_drho,
            )
            _, j_bs_probe, j_cd_probe, _ = current_components_from_state(
                rho, probe, machine, actuator, sim,
                eq=eq_target, q=eq_target.q,
            )
            return j_target - j_bs_probe - j_cd_probe

        # A few cheap Picard passes keep the target normalization, bootstrap
        # split, q, and current-dependent reduced geometry mutually consistent.
        for _ in range(5):
            area0 = poloidal_area_weights(rho, machine, eq=eq0)
            j0_total = normalize_current_profile(
                rho, initial_current_shape(rho), machine.Ip,
                machine.a, machine.kappa, area=area0,
            )
            j0_state = state_current_for_target(j0_total, eq0)
            psi_ind = psi_from_current_density(
                rho, j0_state, machine, psi_edge=psi_edge, sim=sim, eq=eq0
            )
            tmp = PlasmaState(
                Te, Ti, ne20, psi_ind, psi_edge, eq0.Phi_b, eq0.dV_drho,
            )
            eq0 = solve_fixed_boundary_equilibrium(tmp, machine, actuator, sim)
    elif is_psi_diffusion_model(getattr(sim, "current_evolution_model", "")):
        j0_ohm, j0_bs, j0_cd, j0_total = saturated_conductivity_current_components(
            rho, Te, Ti, ne20, machine, actuator, sim, eq=eq0, q=eq0.q
        )
        psi_ind = psi_from_current_density(
            rho, j0_total, machine, psi_edge=psi_edge, sim=sim, eq=eq0
        )
        tmp = PlasmaState(
            Te, Ti, ne20, psi_ind, psi_edge, eq0.Phi_b, eq0.dV_drho,
        )
        eq0 = solve_fixed_boundary_equilibrium(tmp, machine, actuator, sim)
        j0_ohm, j0_bs, j0_cd, j0_total = saturated_conductivity_current_components(
            rho, Te, Ti, ne20, machine, actuator, sim, eq=eq0, q=eq0.q
        )
        psi_ind = psi_from_current_density(
            rho, j0_total, machine, psi_edge=psi_edge, sim=sim, eq=eq0
        )

    psi_ind = ensure_psi_state_grid(rho, psi_ind, machine, psi_edge=psi_edge, sim=sim, eq=eq0)
    return PlasmaState(
        Te, Ti, ne20, psi_ind, psi_edge, eq0.Phi_b, eq0.dV_drho,
    )


def simulate(machine, actuator, sim, state0=None):
    if state0 is None:
        state0 = initial_state(machine, actuator, sim)
    else:
        state0 = _ensure_state_psi_grid(state0, machine, actuator, sim)

    if _skip_enabled(sim):
        cache0 = initialize_step_cache(state0, machine, actuator, sim)

        def body(carry, i):
            state, cache = carry
            new_state, new_cache = step_cached(state, cache, i, machine, actuator, sim)
            return (new_state, new_cache), new_state

        (final_state, _cache_final), hist = jax.lax.scan(body, (state0, cache0), jnp.arange(sim.n_steps))
        return final_state, hist

    def body(carry, i):
        new_state = step(carry, machine, actuator, sim)
        return new_state, new_state

    final_state, hist = jax.lax.scan(body, state0, jnp.arange(sim.n_steps))
    return final_state, hist

simulate_jit = jax.jit(simulate, static_argnames=("machine", "actuator", "sim"))


def simulate_final(machine, actuator, sim, state0=None):
    """Return only final state using lax.fori_loop, avoiding full history storage."""
    if state0 is None:
        state0 = initial_state(machine, actuator, sim)
    else:
        state0 = _ensure_state_psi_grid(state0, machine, actuator, sim)

    if _skip_enabled(sim):
        cache0 = initialize_step_cache(state0, machine, actuator, sim)

        def body(i, carry):
            state, cache = carry
            return step_cached(state, cache, i, machine, actuator, sim)

        final_state, _cache_final = jax.lax.fori_loop(0, sim.n_steps, body, (state0, cache0))
        return final_state

    def body(i, carry):
        return step(carry, machine, actuator, sim)

    return jax.lax.fori_loop(0, sim.n_steps, body, state0)


simulate_final_jit = jax.jit(
    simulate_final, static_argnames=("machine", "actuator", "sim")
)

# Optimization/AD variant: machine and actuator must remain dynamic pytrees so
# gradients can flow through their numerical leaves.  SimulationConfig contains
# model switches and loop/grid sizes, and therefore remains static.
simulate_final_dynamic_jit = jax.jit(
    simulate_final, static_argnames=("sim",)
)



def simulate_waveform(machine, actuator, sim, waveform, state0=None):
    """Run a time-dependent scenario waveform.

    At step i, controls are sampled at t=i*dt and applied to machine/actuator
    before calling the ordinary one-step solver.  The returned history is a
    lightweight tuple of state history plus sampled controls.

    Returns:
        final_state, hist
    where hist is a dict-like tuple:
        {
          "states": PlasmaState time history,
          "time": [n_steps],
          "Ip_MA": [n_steps],
          "P_aux_MW": [n_steps],
          "greenwald_fraction_target": [n_steps],
          "heat_center": [n_steps],
        }
    """
    if state0 is None:
        # Initial state should be consistent with t=0 controls.
        m0, a0 = apply_waveform_controls(machine, actuator, waveform, 0.0)
        state0 = initial_state(m0, a0, sim)
    else:
        m0, a0 = apply_waveform_controls(machine, actuator, waveform, 0.0)
        state0 = _ensure_state_psi_grid(state0, m0, a0, sim)

    if _skip_enabled(sim):
        m0, a0 = apply_waveform_controls(machine, actuator, waveform, 0.0)
        cache0 = initialize_step_cache(state0, m0, a0, sim)

        def body(carry, i):
            state, cache = carry
            t = i.astype(jnp.float32) * sim.dt
            mt, at = apply_waveform_controls(machine, actuator, waveform, t)
            new_state, new_cache = step_cached(state, cache, i, mt, at, sim)
            sample = {
                "states": new_state,
                "time": t,
                "Ip_MA": mt.Ip / 1.0e6,
                "Bt": mt.Bt,
                "P_aux_MW": at.P_aux_MW,
                "greenwald_fraction_target": at.greenwald_fraction_target,
                "heat_center": at.heat_center,
                "heat_width": at.heat_width,
            }
            return (new_state, new_cache), sample

        (final_state, _cache_final), hist = jax.lax.scan(body, (state0, cache0), jnp.arange(sim.n_steps))
        return final_state, hist

    def body(carry, i):
        t = i.astype(jnp.float32) * sim.dt
        mt, at = apply_waveform_controls(machine, actuator, waveform, t)
        new_state = step(carry, mt, at, sim)
        sample = {
            "states": new_state,
            "time": t,
            "Ip_MA": mt.Ip / 1.0e6,
            "Bt": mt.Bt,
            "P_aux_MW": at.P_aux_MW,
            "greenwald_fraction_target": at.greenwald_fraction_target,
            "heat_center": at.heat_center,
            "heat_width": at.heat_width,
        }
        return new_state, sample

    final_state, hist = jax.lax.scan(body, state0, jnp.arange(sim.n_steps))
    return final_state, hist
