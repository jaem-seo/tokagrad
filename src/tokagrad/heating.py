"""Heating, collisional exchange, fusion, and radiation source models.

References:
  [S. I. Braginskii, Rev. Plasma Phys. 1, 205 (1965)] -- e-i exchange.
  [H.-S. Bosch and G. M. Hale, Nucl. Fusion 32, 611 (1992)] -- D-T reactivity.
  [D. R. Mikkelsen, Nucl. Technol./Fusion 4, 237 (1983)] -- alpha partition.
  [F. Albajar et al., Nucl. Fusion 41, 665 (2001)] -- synchrotron loss.

Gaussian actuator deposition and the analytic impurity line-cooling curve are
reduced TokaGrad closures; the latter is not an ADAS charge-state calculation.
"""

import jax.numpy as jnp

from .current import current_components_from_state, q_profile, sauter_neoclassical_resistivity_1999
from .grid import cell_widths, volume_element_from_dV_drho

MU0 = 4.0e-7 * jnp.pi
KEV_TO_J = 1.602176634e-16
ELEMENTARY_CHARGE = 1.602176634e-19
EPS0 = 8.8541878128e-12
ELECTRON_MASS = 9.1093837015e-31
ATOMIC_MASS_UNIT = 1.66053906660e-27

def _smooth_lower(x, lo, width=1.0e-3):
    w = jnp.maximum(width, 1.0e-8)
    return lo + w * jnp.logaddexp(0.0, (x - lo) / w)

def _smooth_upper(x, hi, width=1.0e-3):
    w = jnp.maximum(width, 1.0e-8)
    return hi - w * jnp.logaddexp(0.0, (hi - x) / w)

def _smooth_bounded(x, lo, hi, width=1.0e-3):
    return _smooth_upper(_smooth_lower(x, lo, width), hi, width)

def _smooth_symmetric_limit(x, max_abs, width=1.0e-2):
    m = jnp.maximum(max_abs, 1.0e-12)
    return m * jnp.tanh(x / (m + width))


def _maybe_bound(x, lo, hi, sim, width=1.0e-2):
    if getattr(sim, "differentiable_smooth_mode", False):
        return _smooth_bounded(x, lo, hi, width)
    return jnp.clip(x, lo, hi)


def _maybe_lower(x, lo, sim, width=1.0e-3):
    if getattr(sim, "differentiable_smooth_mode", False):
        return _smooth_lower(x, lo, width)
    return jnp.maximum(x, lo)


def _maybe_limit(x, max_abs, sim, width=1.0e-2):
    if getattr(sim, "differentiable_smooth_mode", False):
        return _smooth_symmetric_limit(x, max_abs, width)
    return jnp.clip(x, -max_abs, max_abs)


def volume_element_1d(rho, machine):
    """Approximate toroidal shell volume per radial cell [m^3]."""
    dr = cell_widths(rho, machine.a)
    r = machine.a * rho
    return 4.0 * jnp.pi**2 * machine.R0 * machine.kappa * r * dr


def _volume_weights(rho, machine, dV_drho=None):
    if dV_drho is None:
        return volume_element_1d(rho, machine)
    return volume_element_from_dV_drho(rho, dV_drho)


def normalize_profile_to_power_MW(rho, profile, P_MW, machine, dV_drho=None):
    """Return volumetric power density [MW/m^3] normalized to total P_MW."""
    dV = _volume_weights(rho, machine, dV_drho=dV_drho)
    norm = jnp.sum(profile * dV) + 1e-20
    return profile * P_MW / norm

def gaussian_deposition(rho, center, width):
    return jnp.exp(-0.5 * ((rho - center) / (width + 1e-8)) ** 2)

def auxiliary_heating(rho, actuator, machine, state=None, sim=None, dV_drho=None):
    """Auxiliary electron/ion heating in MW/m^3.

    ``aux_partition_model="fixed"`` preserves the legacy behavior and splits
    the deposited auxiliary power using ``actuator.f_e_heat``.

    ``aux_partition_model="slowing_down"`` treats the auxiliary power as a
    fast-ion/NBI source with birth energy ``actuator.nbi_birth_energy_MeV`` and
    splits the energy deposition by the same classical slowing-down drag
    partition used for alpha particles.  The default fast ion is a deuteron
    (A=2, Z=1).  A ``state`` is required for this mode because the split depends
    on local Te and composition; direct legacy calls without a state fall back
    to the fixed split for backward compatibility.
    """
    prof = gaussian_deposition(rho, actuator.heat_center, actuator.heat_width)
    P = normalize_profile_to_power_MW(rho, prof, actuator.P_aux_MW, machine, dV_drho=dV_drho)
    mode = str(getattr(actuator, "aux_partition_model", "fixed")).lower()
    if mode in ("fixed", "fixed_fraction", "manual", "f_e_heat") or state is None:
        fe = jnp.clip(jnp.asarray(getattr(actuator, "f_e_heat", 0.6), dtype=P.dtype), 0.0, 1.0)
        fi = 1.0 - fe
    elif mode in ("slowing_down", "fast_ion", "nbi"):
        fe, fi = fast_ion_slowing_down_partition(
            state.Te,
            state.ne20,
            machine,
            birth_energy_MeV=getattr(actuator, "nbi_birth_energy_MeV", 1.0),
            A_fast=getattr(actuator, "nbi_fast_ion_A", 2.0),
            Z_fast=getattr(actuator, "nbi_fast_ion_Z", 1.0),
        )
    else:
        raise ValueError(f"Unknown aux_partition_model={mode!r}; use 'fixed' or 'slowing_down'.")
    return fe * P, fi * P

def species_fractions(machine):
    """Return H, D, T, He main-ion fractions."""
    s = machine.plasma_species.upper().replace("-", "").replace("_", "")
    if s == "H":
        return 1.0, 0.0, 0.0, 0.0
    if s == "D":
        return 0.0, 1.0, 0.0, 0.0
    if s == "T":
        return 0.0, 0.0, 1.0, 0.0
    if s in ("HE", "HE4", "ALPHA"):
        return 0.0, 0.0, 0.0, 1.0
    if s in ("DT", "D50T50"):
        fd = machine.dt_fraction_D
        ft = machine.dt_fraction_T
        total = fd + ft + 1e-12
        return 0.0, fd / total, ft / total, 0.0
    # Conservative fallback to D-T.
    fd = machine.dt_fraction_D
    ft = machine.dt_fraction_T
    total = fd + ft + 1e-12
    return 0.0, fd / total, ft / total, 0.0

def effective_ion_mass_amu(machine):
    fH, fD, fT, fHe = species_fractions(machine)
    return fH * 1.0 + fD * 2.0 + fT * 3.0 + fHe * 4.0


def impurity_electron_fraction(machine):
    """Return impurity electron fraction x = Z_imp n_imp / n_e from Zeff.

    Assumes singly charged main fuel ions plus one representative impurity with
    charge ``machine.impurity_Z``.  Then

        Zeff = 1 + (Z_imp - 1) x,  x = Z_imp n_imp / n_e.

    The returned value is clipped to [0, 0.95] for robustness.
    """
    Zimp = jnp.maximum(jnp.asarray(machine.impurity_Z), 1.0001)
    Zeff = jnp.maximum(jnp.asarray(machine.Zeff), 1.0)
    x = (Zeff - 1.0) / (Zimp - 1.0 + 1e-12)
    return jnp.clip(x, 0.0, 0.95)


def fuel_ion_densities_20(ne20, machine):
    """Return H, D, T, He, impurity densities in 1e20 m^-3.

    ``ne20`` is the electron density.  Main fuel ion densities are reduced when
    Zeff>1 by allocating a fraction of electron density to a representative
    impurity species.  For D-T plasmas this gives nD+nT < ne, as required by
    quasineutrality with impurity dilution.
    """
    ne20 = jnp.asarray(ne20)
    fH, fD, fT, fHe = species_fractions(machine)
    x_imp = impurity_electron_fraction(machine)
    # For H/D/T main plasmas, main ions are singly charged. For He main plasmas,
    # use Z=2 in the quasineutral allocation. Mixed DT remains Z=1.
    main_Z = jnp.where(fHe > 0.5, 2.0, 1.0)
    n_main_total = ne20 * (1.0 - x_imp) / main_Z
    nH = fH * n_main_total
    nD = fD * n_main_total
    nT = fT * n_main_total
    nHe = fHe * n_main_total
    Zimp = jnp.maximum(jnp.asarray(machine.impurity_Z), 1.0001)
    nimp = ne20 * x_imp / Zimp
    return nH, nD, nT, nHe, nimp


def ohmic_heating(rho, state, machine, actuator, sim, eq=None):
    """Ohmic electron heating eta_neo * j_ind^2 [MW/m^3].

    Smooth mode uses softened Btheta/q bounds and the simpler resistivity proxy
    to avoid the non-smooth Sauter branch during AD optimization.
    """
    if not sim.include_ohmic_heating:
        return jnp.zeros_like(rho)
    j_ind, _, _, j_tot = current_components_from_state(rho, state, machine, actuator, sim, eq=eq)
    if eq is not None and hasattr(eq, "q"):
        q = jnp.clip(eq.q, 0.5, 10.0)
    else:
        q = q_profile(rho, j_ind, machine, sim, eq=eq)
        q = _maybe_bound(q, 0.5, 10.0, sim, 1e-2)

    if getattr(sim, "differentiable_smooth_mode", False):
        from .current import neoclassical_resistivity
        eta = neoclassical_resistivity(state.Te, rho, machine, eq=eq, q=q)
    else:
        eta = sauter_neoclassical_resistivity_1999(state.Te, state.Ti, state.ne20, q, rho, machine)
    return _maybe_bound(getattr(sim, "resistivity_multiplier", 1.0) * eta * j_ind * j_tot / 1.0e6, 0.0, 50.0, sim, 1e-2)


def electron_ion_exchange_rate_s(Te_keV, Ti_keV, ne20, machine, sim=None):
    """Electron-ion temperature equilibration rate [1/s].

    Uses the Spitzer/Braginskii energy-exchange coefficient for a Maxwellian
    electron population exchanging energy with all thermal ion species present
    in ``fuel_ion_densities_20``:

        dT_e/dt = sum_i nu_E,ei (T_i - T_e),

        nu_E,ei = [8 sqrt(2 pi) / 3] n_i Z_i^2 e^4 lnLambda
                  / [(4 pi eps0)^2 m_e m_i
                     (kT_e/m_e + kT_i/m_i)^(3/2)].

    Temperatures are supplied in keV, densities in 1e20 m^-3.  Unlike the old
    reduced closure, this does not use ``Zeff**2`` or a fitted 1.5-second
    prefactor; impurity dilution and ion masses enter through the explicit
    ``sum_i n_i Z_i^2/m_i`` energy-exchange rate.
    """
    if sim is not None:
        ne20 = _maybe_lower(ne20, 1.0e-8, sim, 1.0e-8)
        Te = _maybe_lower(Te_keV, 0.03, sim, 1e-3)
        Ti = _maybe_lower(Ti_keV, 0.03, sim, 1e-3)
    else:
        ne20 = jnp.maximum(ne20, 1.0e-8)
        Te = jnp.maximum(Te_keV, 0.03)
        Ti = jnp.maximum(Ti_keV, 0.03)

    nH20, nD20, nT20, nHe20, nimp20 = fuel_ion_densities_20(ne20, machine)
    Zimp = jnp.maximum(jnp.asarray(machine.impurity_Z), 1.0001)
    Aimp = 2.0 * Zimp

    # Stable n20/keV form of the SI expression above.  The coefficient is
    #
    #   [8 sqrt(2 pi) / 3] * 1e20 * e^4
    #   / [(4 pi eps0)^2 m_e m_u (1 keV / m_e)^(3/2)]
    #
    # = 10.0835362363 s^-1 for lnLambda=1, n20=1, T_keV=1, A=1.
    # Writing it this way avoids float32 underflow in e^4 and overflow in the
    # SI thermal-speed term.
    coeff_n20_kev = jnp.asarray(10.083536236342505, dtype=Te.dtype) * jnp.asarray(machine.lnLambda, dtype=Te.dtype)
    Te32 = Te ** 1.5

    def one_species_rate(n_i20, Z_i, A_i):
        A_i = jnp.asarray(A_i, dtype=Te.dtype)
        # Full Spitzer thermal-speed denominator gives a small correction
        # [1 + (m_e/m_i) Ti/Te]^(-3/2).  It is usually very close to one but
        # is kept for coefficient fidelity.
        finite_Ti_correction = (
            1.0
            + (ELECTRON_MASS / (A_i * ATOMIC_MASS_UNIT)) * (Ti / (Te + 1.0e-30))
        ) ** (-1.5)
        return (
            coeff_n20_kev
            * n_i20
            * Z_i**2
            / (A_i * Te32 + 1.0e-30)
            * finite_Ti_correction
        )

    rate = (
        one_species_rate(nH20, 1.0, 1.0)
        + one_species_rate(nD20, 1.0, 2.0)
        + one_species_rate(nT20, 1.0, 3.0)
        + one_species_rate(nHe20, 2.0, 4.0)
        + one_species_rate(nimp20, Zimp, Aimp)
    )
    return jnp.maximum(rate, 0.0)


def electron_ion_exchange_time_s(Te_keV, Ti_keV, ne20, machine, sim=None):
    """Electron-ion temperature equilibration time [s]."""
    rate = electron_ion_exchange_rate_s(Te_keV, Ti_keV, ne20, machine, sim=sim)
    return 1.0 / (rate + 1.0e-30)


def electron_ion_exchange_sources(rho, state, machine, sim):
    """Electron-ion collisional heat exchange in source units [keV/s].

    Positive ``Se`` heats electrons.  The ion source is set to ``-Se`` because
    the present reduced heat equations use the same density factor for electron
    and ion thermal energies.  This preserves total thermal energy within the
    code's single-ion-temperature approximation.
    """
    if not sim.include_ei_exchange:
        z = jnp.zeros_like(rho)
        return z, z

    ne20 = _maybe_lower(state.ne20, 1.0e-8, sim, 1e-8)
    Te = _maybe_lower(state.Te, 0.03, sim, 1e-3)
    Ti = _maybe_lower(state.Ti, 0.03, sim, 1e-3)
    rate = electron_ion_exchange_rate_s(Te, Ti, ne20, machine, sim=sim)

    Se = sim.ei_exchange_multiplier * rate * (Ti - Te)
    Se = _maybe_limit(Se, 300.0, sim, 1e-2)
    Si = -Se
    return Se, Si


def bosch_hale_dt_reactivity(T_keV):
    """Bosch-Hale D-T Maxwellian reactivity <sigma v> [m^3/s].

    Reference: [H.-S. Bosch and G. M. Hale, Nucl. Fusion 32, 611 (1992)].
    """
    T = _smooth_bounded(T_keV, 0.2, 200.0, 1.0e-2)
    C0 = 6.4341e-20
    C1 = 6.6610
    C2 = 1.5136e-2
    C3 = 7.5189e-2
    C4 = 4.6064e-3
    C5 = 1.3500e-2
    C6 = -1.0675e-4
    C7 = 1.3660e-5
    zeta = 1.0 - (C2 * T + C4 * T**2 + C6 * T**3) / (1.0 + C3 * T + C5 * T**2 + C7 * T**3)
    xi = C1 / (T ** (1.0 / 3.0))
    return C0 * zeta ** (-5.0 / 6.0) * xi**2 * jnp.exp(-3.0 * zeta ** (1.0 / 3.0) * xi)

def fusion_power_density_DT_MW_m3(state, machine):
    """Bosch-Hale D-T fusion power density [MW/m^3]."""
    species = machine.plasma_species.upper().replace("-", "").replace("_", "")
    if species not in ("DT", "D50T50"):
        return jnp.zeros_like(state.Ti)
    ne20 = _smooth_lower(state.ne20, 0.0, 1.0e-4)
    _nH20, nD20, nT20, _nHe20, _nimp20 = fuel_ion_densities_20(ne20, machine)
    sigmav = bosch_hale_dt_reactivity(state.Ti)
    Efus_J = 17.6e6 * 1.602176634e-19
    # Compute using n20 units to avoid float32 overflow: (1e20)^2 factor.
    coeff = 1.0e40 * Efus_J / 1.0e6
    return nD20 * nT20 * sigmav * coeff

def _fast_ion_critical_energy_keV(Te_keV, ne20, machine, A_fast=4.0):
    """Critical fast-ion energy [keV] separating electron and ion drag.

    Uses the standard fast-ion estimate

        E_c = 14.8 A_f T_e [sum_i n_i Z_i^2 / (A_i n_e)]^(2/3),

    with T_e and E_c in keV.  The representative impurity mass is approximated
    as A_imp ≈ 2 Z_imp.
    """
    ne20 = _smooth_lower(ne20, 1.0e-8, 1.0e-8)
    nH20, nD20, nT20, nHe20, nimp20 = fuel_ion_densities_20(ne20, machine)
    Zimp = jnp.maximum(jnp.asarray(machine.impurity_Z), 1.0001)
    Aimp = 2.0 * Zimp
    sum_Z2_over_A = (
        nH20 * 1.0**2 / 1.0
        + nD20 * 1.0**2 / 2.0
        + nT20 * 1.0**2 / 3.0
        + nHe20 * 2.0**2 / 4.0
        + nimp20 * Zimp**2 / Aimp
    ) / (ne20 + 1.0e-20)
    sum_Z2_over_A = _smooth_lower(sum_Z2_over_A, 1.0e-6, 1.0e-8)
    Te_pos = _smooth_lower(Te_keV, 0.03, 1e-3)
    A = jnp.asarray(A_fast, dtype=Te_pos.dtype)
    return 14.8 * A * Te_pos * sum_Z2_over_A ** (2.0 / 3.0)


def _alpha_ion_energy_fraction_from_Ecrit(Ecrit_keV, Ealpha_keV=3500.0):
    """Ion energy fraction from classical slowing-down drag partition.

    At alpha energy E, the reduced drag split is

        P_i/P_total = 1 / [1 + (E/E_c)^(3/2)].

    The deposited ion fraction is the energy integral of this ratio from 0 to
    E_alpha.  A fixed midpoint quadrature is used so the function remains JAX
    compatible and differentiable.
    """
    Ecrit = _smooth_lower(Ecrit_keV, 1.0e-6, 1.0e-8)
    n_quad = 64
    x = (jnp.arange(n_quad, dtype=Ecrit.dtype) + 0.5) / float(n_quad)
    shape = (n_quad,) + (1,) * Ecrit.ndim
    E = Ealpha_keV * jnp.reshape(x, shape)
    frac_E = 1.0 / (1.0 + (E / Ecrit) ** 1.5)
    return jnp.mean(frac_E, axis=0)


def fast_ion_slowing_down_partition(Te_keV, ne20, machine, *, birth_energy_MeV=1.0, A_fast=2.0, Z_fast=1.0):
    """Return electron/ion deposition fractions for a born fast-ion source.

    The split is computed from the same critical-energy drag partition used for
    alpha heating, but with a configurable birth energy and fast-ion mass.
    ``Z_fast`` is accepted for API symmetry with fast-ion stored-energy helpers;
    in this reduced critical-energy expression the split depends on ``A_fast``
    and background composition.
    """
    Ecrit_keV = _fast_ion_critical_energy_keV(Te_keV, ne20, machine, A_fast=A_fast)
    Eb_keV = jnp.maximum(jnp.asarray(birth_energy_MeV, dtype=Ecrit_keV.dtype) * 1000.0, 1.0e-6)
    f_i = _alpha_ion_energy_fraction_from_Ecrit(Ecrit_keV, Eb_keV)
    f_i = jnp.clip(f_i, 0.0, 1.0)
    f_e = 1.0 - f_i
    return f_e, f_i


def alpha_slowing_down_partition(Te_keV, ne20, machine):
    """Return electron/ion alpha heating fractions from classical slowing down.

    This replaces the older empirical ``0.35*sqrt(E_c/E_alpha)`` proxy with
    the critical-energy formula and the actual reduced energy integral of the
    electron/ion drag split.  For a 50-50 D-T plasma at T_e≈10 keV, it gives an
    ion fraction of order 0.15--0.2, increasing with T_e.
    """
    return fast_ion_slowing_down_partition(
        Te_keV, ne20, machine, birth_energy_MeV=3.52, A_fast=4.0, Z_fast=2.0
    )


def mikkelsen_alpha_partition(Te_keV):
    """Mikkelsen alpha-heating partition (for controlled benchmark with TORAX).

    Reference: [D. R. Mikkelsen, Nucl. Technol./Fusion 4, 237 (1983)].

    This follows equations 5 and 26 of Mikkelsen (1983), including the missing
    ``2*x`` correction in the published logarithm noted by TORAX.
    """
    Te = _smooth_lower(Te_keV, 0.03, 1.0e-3)
    birth_energy_keV = jnp.asarray(3520.0, dtype=Te.dtype)
    alpha_mass_amu = jnp.asarray(4.002602, dtype=Te.dtype)
    energy_ratio = birth_energy_keV / (10.0 * alpha_mass_amu * Te)
    x = jnp.sqrt(energy_ratio)
    frac_i = (
        2.0
        * (
            jnp.log(
                (1.0 - x + energy_ratio)
                / (1.0 + 2.0 * x + energy_ratio)
            )
            / 6.0
            + (
                jnp.arctan((2.0 * x - 1.0) / jnp.sqrt(3.0))
                + jnp.pi / 6.0
            )
            / jnp.sqrt(3.0)
        )
        / energy_ratio
    )
    frac_i = jnp.clip(frac_i, 0.0, 1.0)
    return 1.0 - frac_i, frac_i

def alpha_heating(rho, state, machine, sim):
    """D-T alpha heating split into electron/ion channels [MW/m^3].

    The total alpha power uses the 3.5/17.6 fusion-energy fraction. 
    Electron/ion partition is selected independently.
    """
    if not sim.include_alpha_heating:
        z = jnp.zeros_like(rho)
        return z, z

    P_fus = fusion_power_density_DT_MW_m3(state, machine)
    P_alpha = sim.alpha_heating_scale * (3.5 / 17.6) * P_fus

    if sim.alpha_partition_model == "fixed":
        fe = sim.alpha_electron_fraction_fixed
        fi = sim.alpha_ion_fraction_fixed
        norm = fe + fi + 1e-12
        fe, fi = fe / norm, fi / norm
    elif sim.alpha_partition_model == "slowing_down":
        fe, fi = alpha_slowing_down_partition(state.Te, state.ne20, machine)
    elif sim.alpha_partition_model in ("mikkelsen", "torax", "stix_mikkelsen"):
        # for controlled benchmark with TORAX
        fe, fi = mikkelsen_alpha_partition(state.Te)
    else:
        raise ValueError(
            f"Unknown alpha_partition_model={sim.alpha_partition_model!r}; "
            "use 'mikkelsen', 'slowing_down', or 'fixed'."
        )

    return fe * P_alpha, fi * P_alpha

def bremsstrahlung_loss(rho, state, machine, sim):
    """Non-relativistic electron-ion bremsstrahlung loss [MW/m^3].

    Reference: [J. D. Huba, NRL Plasma Formulary (2013)].

    Uses the standard hydrogenic/NRL-style scaling

        P_br = 5.35e3 Z_eff n20^2 sqrt(T_keV)  [W/m^3],

    with a Gaunt-factor-level coefficient suitable for simple reactor-scale
    modelling.  Electron-electron and relativistic corrections are intentionally
    omitted for this reduced DT-oriented model.
    """
    if not sim.include_radiation_losses:
        return jnp.zeros_like(rho)
    ne20 = _smooth_lower(state.ne20, 0.0, 1.0e-4)
    Te = _maybe_lower(state.Te, 0.03, sim, 1e-3)
    P_W_m3 = 5.35e3 * machine.Zeff * ne20**2 * jnp.sqrt(Te)
    return _maybe_bound(sim.bremsstrahlung_scale * P_W_m3 / 1.0e6, 0.0, 50.0, sim, 1e-2)


def line_radiation_loss(rho, state, machine, sim):
    """Reduced impurity line-radiation loss [MW/m^3].

    The model is deliberately simple but unit-consistent:

        P_line = n_e n_imp L_z(T_e),

    where L_z is a reduced coronal cooling coefficient [W m^3].  The default
    L_z is C-like and low-keV calibrated; high-Z/non-coronal line radiation
    should be handled by changing ``line_radiation_scale`` or replacing this
    with ADAS/radiative-cooling tables.
    """
    if not sim.include_radiation_losses:
        return jnp.zeros_like(rho)
    ne20 = _smooth_lower(state.ne20, 0.0, 1.0e-4)
    Te = _maybe_lower(state.Te, 0.01, sim, 1e-3)
    Zimp = jnp.maximum(jnp.asarray(machine.impurity_Z), 1.0001)
    x_imp = impurity_electron_fraction(machine)
    # Keep the arithmetic in n20 units.  Directly forming n_e*n_imp can
    # overflow float32 for ITER-like densities before the small cooling
    # coefficient Lz is applied (1e20*1e18--1e20 ~ 1e38--1e40).  The
    # equivalent scaled form is
    #   n_e n_imp Lz / 1e6 = ne20^2 * (x_imp/Zimp) * Lz * 1e34  [MW/m^3].
    # This avoids inf*0 -> NaN in the hot core where the temperature window
    # underflows.
    imp_number_fraction = x_imp / Zimp

    Lz_ref = jnp.asarray(getattr(sim, "line_radiation_Lz_ref_W_m3", 3.0e-35))
    Tcut = jnp.maximum(jnp.asarray(getattr(sim, "line_radiation_Tcut_keV", 1.5)), 1.0e-3)
    zpow = jnp.asarray(getattr(sim, "line_radiation_Z_scaling_power", 0.5))
    edge_power = jnp.asarray(getattr(sim, "line_radiation_edge_power", 3.0))
    temp_window = jnp.exp(-Te / Tcut) / jnp.sqrt(Te + 0.02)
    z_scale = (Zimp / 6.0) ** zpow
    edge = rho ** edge_power
    Lz = Lz_ref * z_scale * temp_window
    P_MW_m3 = ne20**2 * imp_number_fraction * Lz * 1.0e34 * edge
    return _maybe_bound(sim.line_radiation_scale * P_MW_m3, 0.0, 50.0, sim, 1e-2)


def synchrotron_loss(rho, state, machine, sim):
    """Thermal synchrotron/cyclotron electron loss [MW/m^3].

    Reference context: [F. Albajar et al., Nucl. Fusion 41, 665 (2001)].
    Unlike the Albajar--Artaud fit, this implementation is an optically thin
    Larmor estimate multiplied by configurable escape and reflection factors.

    Starts from the optically thin non-relativistic thermal average

        P_sync = 6.211e-3 n20 B_T^2 T_keV  [MW/m^3],

    derived from the Thomson/Larmor synchrotron power.  Tokamak plasmas are
    optically thick at low harmonics, so the net loss is multiplied by an
    explicit escape/self-absorption factor and by (1 - wall_reflectivity).
    """
    if not sim.include_radiation_losses:
        return jnp.zeros_like(rho)
    ne20 = _smooth_lower(state.ne20, 0.0, 1.0e-4)
    Te = _maybe_lower(state.Te, 0.03, sim, 1e-3)
    Rwall = _maybe_bound(sim.synchrotron_wall_reflectivity, 0.0, 0.999, sim, 1e-3)
    escape = _maybe_bound(getattr(sim, "synchrotron_escape_fraction", 0.05), 0.0, 1.0, sim, 1e-3)
    P_thin_MW_m3 = 6.211e-3 * ne20 * machine.Bt**2 * Te
    return _maybe_bound(
        sim.synchrotron_scale * escape * (1.0 - Rwall) * P_thin_MW_m3,
        0.0,
        50.0,
        sim,
        1e-2,
    )

def radiation_losses(rho, state, machine, sim):
    """Total electron radiation losses [MW/m^3]."""
    brem = bremsstrahlung_loss(rho, state, machine, sim)
    line = line_radiation_loss(rho, state, machine, sim)
    sync = synchrotron_loss(rho, state, machine, sim)
    return brem + line + sync, {"brem": brem, "line": line, "sync": sync}

def MWm3_to_keV_per_s(P_MW_m3, ne20, sim=None):
    """Convert volumetric heating [MW/m^3] to temperature source [keV/s]."""
    if sim is not None and getattr(sim, "differentiable_smooth_mode", False):
        ne20_safe = _smooth_lower(ne20, 0.02, 1e-4)
    else:
        ne20_safe = jnp.maximum(ne20, 0.02)
    ne = ne20_safe * 1.0e20
    return P_MW_m3 * 1.0e6 / (1.5 * ne * KEV_TO_J)

def total_heating_sources(rho, state, machine, actuator, sim, eq=None, dV_drho=None):
    """Return electron/ion temperature sources [keV/s] and diagnostics.

    Includes by default:
      - auxiliary heating,
      - ohmic heating from inductive current,
      - electron-ion exchange,
      - alpha heating for D-T plasma,
      - bremsstrahlung, line, and synchrotron radiation losses.
    """
    if dV_drho is None and eq is not None:
        dV_drho = eq.dV_drho
    Paux_e, Paux_i = auxiliary_heating(rho, actuator, machine, state=state, sim=sim, dV_drho=dV_drho)
    Pohm_e = ohmic_heating(rho, state, machine, actuator, sim, eq=eq)
    Palpha_e, Palpha_i = alpha_heating(rho, state, machine, sim)
    Prad_e, rad_parts = radiation_losses(rho, state, machine, sim)
    Sei, Sii = electron_ion_exchange_sources(rho, state, machine, sim)

    Pe_net = _maybe_bound(Paux_e + Pohm_e + Palpha_e - Prad_e, -50.0, 100.0, sim, 1e-2)
    Pi_net = _maybe_bound(Paux_i + Palpha_i, -50.0, 100.0, sim, 1e-2)

    Se = MWm3_to_keV_per_s(Pe_net, state.ne20, sim) + Sei
    Si = MWm3_to_keV_per_s(Pi_net, state.ne20, sim) + Sii
    Se = _maybe_bound(Se, -500.0, 500.0, sim, 1e-2)
    Si = _maybe_bound(Si, -500.0, 500.0, sim, 1e-2)

    diag = {
        "Paux_e": Paux_e,
        "Paux_i": Paux_i,
        "Pohm_e": Pohm_e,
        "Palpha_e": Palpha_e,
        "Palpha_i": Palpha_i,
        "Prad_e": Prad_e,
        "Pbrem": rad_parts["brem"],
        "Pline": rad_parts["line"],
        "Psync": rad_parts["sync"],
        "Pe_net": Pe_net,
        "Pi_net": Pi_net,
        "Sei_keV_s": Sei,
        "Sii_keV_s": Sii,
        "tau_ei_s": electron_ion_exchange_time_s(state.Te, state.Ti, state.ne20, machine, sim=sim),
    }
    return Se, Si, diag
