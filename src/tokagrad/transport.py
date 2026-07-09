"""Turbulent/surrogate transport dispatch and radial flux construction.

References:
  [M. Erba et al., Nucl. Fusion 38, 1013 (1998)] -- mixed Bohm/gyro-Bohm.
  [K. L. van de Plassche et al., Phys. Plasmas 27, 022310 (2020)] -- QLKNN.
  [G. M. Staebler et al., Phys. Plasmas 12, 102508 (2005)] -- TGLF basis.

Flux-to-effective-diffusivity conversion, clipping, and fallback behavior are
TokaGrad integrated-model approximations and are documented at their call sites.
"""

import jax.numpy as jnp

from .qlknn_adapter import (
    fusion_surrogates_effective_diffusivities,
    fusion_surrogates_normalized_fluxes,
)
from .tglfnn_jax import tglfnn_jax_effective_diffusivities
from .neoclassical import angioni_neoclassical_diffusivities
from .grid import radial_gradient as grid_radial_gradient, infer_rho_faces
from .heating import effective_ion_mass_amu

E_CHARGE = 1.602176634e-19
KEV_TO_J = 1.602176634e-16
M_DEUTERON = 3.3435837724e-27

def _smooth_lower(x, lo, width=1.0e-3):
    w = jnp.maximum(width, 1.0e-8)
    return lo + w * jnp.logaddexp(0.0, (x - lo) / w)

def _smooth_upper(x, hi, width=1.0e-3):
    w = jnp.maximum(width, 1.0e-8)
    return hi - w * jnp.logaddexp(0.0, (hi - x) / w)

def _smooth_bounded(x, lo, hi, width=1.0e-3):
    return _smooth_upper(_smooth_lower(x, lo, width), hi, width)

def _maybe_lower(x, lo, sim, width=1.0e-3):
    if getattr(sim, 'differentiable_smooth_mode', False):
        return _smooth_lower(x, lo, width)
    return jnp.maximum(x, lo)

def _maybe_bound(x, lo, hi, sim, width=1.0e-3):
    if getattr(sim, 'differentiable_smooth_mode', False):
        return _smooth_bounded(x, lo, hi, width)
    return jnp.clip(x, lo, hi)

def _smooth_abs(x, width=1.0e-8):
    return jnp.sqrt(x*x + width*width)

def _bohm_gyrobohm_geometry_scales(rho, machine, eq=None):
    """Return local minor radius and unnormalized rho_tor [m]."""
    if eq is not None and hasattr(eq, "R"):
        rmid = 0.5 * (jnp.max(eq.R, axis=1) - jnp.min(eq.R, axis=1))
    else:
        rmid = machine.a * rho

    if eq is not None and hasattr(eq, "Phi_b"):
        # rho_tor=sqrt(Phi/(pi*B0)); rho_b is its LCFS value.
        rho_b = jnp.sqrt(
            jnp.abs(jnp.asarray(eq.Phi_b, dtype=rho.dtype))
            / (jnp.pi * jnp.maximum(jnp.abs(machine.Bt), 1.0e-6))
        )
    else:
        rho_b = jnp.asarray(machine.a, dtype=rho.dtype)
    return jnp.maximum(rmid, 0.0), jnp.maximum(rho_b, 1.0e-6)


def bohm_gyrobohm_chi(rho, Te, Ti, ne20, q, machine, sim, eq=None):
    """Bohm--gyroBohm turbulent transport closure [m^2/s].

    Reference: [M. Erba et al., Nucl. Fusion 38, 1013 (1998)].

      chi_e,B  = r_mid*q^2/(e*B0*n_e) *
                 (|dn_e/drho_tor|*T_e + n_e*|dT_e/drho_tor|),
      chi_i,B  = 2*chi_e,B,
      chi_e,gB = sqrt(A_i/2)*sqrt(T_e[eV])/B0^2 *
                 |dT_e[eV]/drho_tor|,
      chi_i,gB = 0.5*chi_e,gB.

    Species-specific calibration coefficients combine the two contributions.
    The electron particle diffusivity is the TORAX/Garzotti harmonic form
    ``eta*chi_e*chi_i/(chi_e+chi_i)``.  ``Ti`` is intentionally absent from
    this model's dependencies and remains in the signature for transport-API
    compatibility.
    """
    del Ti
    q = _maybe_bound(jnp.abs(q), 0.05, 20.0, sim, 1e-2)
    B0 = _maybe_lower(jnp.abs(machine.Bt), 1.0e-3, sim, 1e-3)
    rmid, rho_b = _bohm_gyrobohm_geometry_scales(rho, machine, eq=eq)

    Te_pos = _maybe_lower(Te, 1.0e-6, sim, 1e-6)
    ne_m3 = _maybe_lower(ne20, 1.0e-8, sim, 1e-8) * 1.0e20
    dTe_drho_tor = grid_radial_gradient(Te, rho, rho_b)
    dne_drho_tor = grid_radial_gradient(ne20 * 1.0e20, rho, rho_b)
    if getattr(sim, "differentiable_smooth_mode", False):
        abs_dTe = _smooth_abs(dTe_drho_tor, 1.0e-8)
        abs_dne = _smooth_abs(dne_drho_tor, 1.0e8)
    else:
        abs_dTe = jnp.abs(dTe_drho_tor)
        abs_dne = jnp.abs(dne_drho_tor)

    # Product-rule form for |dp_e/drho_tor|. Temperatures are converted
    # from keV to joules before division by the elementary charge.
    pressure_gradient_magnitude = (
        abs_dne * Te_pos + abs_dTe * ne_m3
    ) * KEV_TO_J
    chi_e_bohm_scale = (
        rmid * q**2 * pressure_gradient_magnitude
        / (E_CHARGE * B0 * ne_m3 + 1.0e-30)
    )
    chi_i_bohm_scale = 2.0 * chi_e_bohm_scale

    Ai = jnp.maximum(effective_ion_mass_amu(machine), 1.0e-6)
    Te_eV = Te_pos * 1.0e3
    abs_dTe_eV = abs_dTe * 1.0e3
    chi_e_gyrobohm_scale = (
        jnp.sqrt(Ai / 2.0) * jnp.sqrt(Te_eV) * abs_dTe_eV / (B0**2)
    )
    chi_i_gyrobohm_scale = 0.5 * chi_e_gyrobohm_scale

    chi_e = (
        sim.bohm_chi_e_bohm_coeff * chi_e_bohm_scale
        + sim.bohm_chi_e_gyrobohm_coeff * chi_e_gyrobohm_scale
    )
    chi_i = (
        sim.bohm_chi_i_bohm_coeff * chi_i_bohm_scale
        + sim.bohm_chi_i_gyrobohm_coeff * chi_i_gyrobohm_scale
    )

    weighting = sim.bohm_particle_c1 + (
        sim.bohm_particle_c2 - sim.bohm_particle_c1
    ) * rho
    Dn = weighting * chi_e * chi_i / (chi_e + chi_i + 1.0e-30)

    chi_e = _maybe_bound(chi_e, sim.bohm_chi_min, sim.bohm_chi_max, sim, 1e-2)
    chi_i = _maybe_bound(chi_i, sim.bohm_chi_min, sim.bohm_chi_max, sim, 1e-2)
    Dn = _maybe_bound(Dn, sim.bohm_particle_min, sim.bohm_particle_max, sim, 1e-2)
    return chi_e, chi_i, Dn

def compute_diffusivity(rho, Te, Ti, ne20, machine, sim, q=None, eq=None):
    """Dispatch turbulent/surrogate transport and add neoclassical transport."""
    if q is None:
        raise ValueError("Diffusivity requires q=eq.q from the active Equilibrium.")
    q = jnp.clip(jnp.abs(jnp.asarray(q, dtype=rho.dtype) + jnp.zeros_like(rho)), 0.05, 20.0)

    if sim.transport_model == "bohm_gyrobohm":
        chi_e, chi_i, Dn = bohm_gyrobohm_chi(rho, Te, Ti, ne20, q, machine, sim, eq=eq)
    elif sim.transport_model in ("fusion_surrogates", "qlknn"):
        chi_e, chi_i, Dn = fusion_surrogates_effective_diffusivities(rho, Te, Ti, ne20, q, machine, sim, eq=eq)
    elif sim.transport_model == "tglfnn_jax":
        chi_e, chi_i, Dn = tglfnn_jax_effective_diffusivities(rho, Te, Ti, ne20, q, machine, sim, eq=eq)
    else:
        raise ValueError(
            f"Unknown transport_model={sim.transport_model!r}. "
            'Use "bohm_gyrobohm", "fusion_surrogates", or "tglfnn_jax".'
        )

    chi_e_nc, chi_i_nc, Dn_nc = angioni_neoclassical_diffusivities(
        rho, Te, Ti, ne20, q, machine, sim, eq=eq
    )
    return chi_e + chi_e_nc, chi_i + chi_i_nc, Dn + Dn_nc


def cell_to_face_average(y):
    """Map cell-centered profile to faces with zero edge flux convention handled outside."""
    return jnp.concatenate([
        y[0:1],
        0.5 * (y[:-1] + y[1:]),
        y[-1:],
    ])

def gradient_face(y, rho, machine):
    rho_f = infer_rho_faces(rho)
    r = machine.a * rho
    r_f = machine.a * rho_f
    yf = jnp.concatenate([y[0:1], y, y[-1:]])
    # face gradients: axis no-gradient, edge one-sided to ghost=edge cell
    dist = jnp.concatenate([(r[0]-r_f[0])[None], r[1:] - r[:-1], (r_f[-1]-r[-1])[None]])
    return (yf[1:] - yf[:-1]) / (dist + 1e-12)

def diffusivity_to_face_fluxes(rho, Te, Ti, ne20, chi_e, chi_i, Dn, machine):
    """Return outward face fluxes generated from diffusivities."""
    chi_e_f = cell_to_face_average(chi_e)
    chi_i_f = cell_to_face_average(chi_i)
    Dn_f = cell_to_face_average(Dn)
    Qe = -chi_e_f * gradient_face(Te, rho, machine)
    Qi = -chi_i_f * gradient_face(Ti, rho, machine)
    Gamma = -Dn_f * gradient_face(ne20, rho, machine)
    # axis no flux
    Qe = Qe.at[0].set(0.0)
    Qi = Qi.at[0].set(0.0)
    Gamma = Gamma.at[0].set(0.0)
    return Qe, Qi, Gamma

def normalized_fluxes_to_face_fluxes(rho, Te, Ti, ne20, qe_norm, qi_norm, gamma_norm, machine, sim):
    """Convert cell-centered normalized QLKNN fluxes to face fluxes.

    This is still a stage-1 scaling: normalized fluxes are mapped to
    m^2/s-gradient-equivalent units using a gyroBohm-inspired local scale.
    The important structural change is that the solver consumes flux divergence
    directly instead of flux/gradient diffusivities.
    """
    # Reuse a local gyroBohm-ish m^2/s scale, similar magnitude to diffusivity closures.
    Ti_J = _maybe_lower(Ti, 0.03, sim, 1e-3) * 1.0e3 * E_CHARGE
    vthi = jnp.sqrt(2.0 * Ti_J / M_DEUTERON)
    # Use rho_i = v_thi / Omega_ci to avoid float32 underflow in sqrt(M*T).
    omega_ci = E_CHARGE * _maybe_lower(machine.Bt, 0.2, sim, 1e-3) / M_DEUTERON
    rhoi = vthi / (omega_ci + 1.0e-30)
    chi_gb = rhoi**2 * vthi / jnp.maximum(machine.a, 0.1)
    chi_gb = jnp.clip(chi_gb, sim.chi_clip_min, sim.chi_clip_max)

    qe_f = cell_to_face_average(qe_norm * chi_gb)
    qi_f = cell_to_face_average(qi_norm * chi_gb)
    gamma_f = cell_to_face_average(gamma_norm * chi_gb)

    # Direction: QLKNN positive flux is outward down-gradient. Here flux variable
    # is the transported-variable flux entering dY/dt = -div(F) + S.
    grad_Te = gradient_face(Te, rho, machine)
    grad_Ti = gradient_face(Ti, rho, machine)
    grad_n = gradient_face(ne20, rho, machine)

    Qe = qe_f * jnp.maximum(jnp.abs(grad_Te), 0.1)
    Qi = qi_f * jnp.maximum(jnp.abs(grad_Ti), 0.1)
    Gamma = gamma_f * jnp.maximum(jnp.abs(grad_n), 0.01)

    # Preserve outward direction; no flux at magnetic axis.
    Qe = Qe.at[0].set(0.0)
    Qi = Qi.at[0].set(0.0)
    Gamma = Gamma.at[0].set(0.0)
    return sim.transport_flux_scale * Qe, sim.transport_flux_scale * Qi, sim.transport_flux_scale * Gamma

def transport_fluxes(rho, Te, Ti, ne20, q, machine, sim, eq=None):
    """Return outward face fluxes for Te, Ti, and ne.

    If QLKNN-like or fusion_surrogates is selected, use model fluxes directly
    in transport_mode="flux". Otherwise construct equivalent diffusive fluxes.
    """
    if q is None:
        raise ValueError("transport_fluxes requires q=eq.q from the active Equilibrium.")
    q = jnp.clip(jnp.abs(jnp.asarray(q, dtype=rho.dtype) + jnp.zeros_like(rho)), 0.05, 20.0)

    if sim.transport_model in ("fusion_surrogates", "qlknn"):
        qe, qi, gam = fusion_surrogates_normalized_fluxes(rho, Te, Ti, ne20, q, machine, sim, eq=eq)
        return normalized_fluxes_to_face_fluxes(rho, Te, Ti, ne20, qe, qi, gam, machine, sim)

    if sim.transport_model == "tglfnn_jax":
        # The public TGLFNN-JAX adapter currently returns effective
        # diffusivities rather than raw normalized fluxes.  In flux mode, use
        # those diffusivities to construct face fluxes instead of accidentally
        # falling through a second transport dispatch or another fallback path.
        chi_e, chi_i, Dn = tglfnn_jax_effective_diffusivities(rho, Te, Ti, ne20, q, machine, sim, eq=eq)
        return diffusivity_to_face_fluxes(rho, Te, Ti, ne20, chi_e, chi_i, Dn, machine)

    chi_e, chi_i, Dn = compute_diffusivity(rho, Te, Ti, ne20, machine, sim, q=q, eq=eq)
    return diffusivity_to_face_fluxes(rho, Te, Ti, ne20, chi_e, chi_i, Dn, machine)
