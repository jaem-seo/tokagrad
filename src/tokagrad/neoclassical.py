"""Reduced neoclassical heat and particle transport closures.

References:
  [C. Angioni and O. Sauter, Phys. Plasmas 7, 1224 (2000)] -- K/L matrices.
  [C. S. Chang and F. L. Hinton, Phys. Fluids 25, 1493 (1982)] -- reduced
    analytic ion thermal diffusivity fit.
  [O. Sauter et al., Phys. Plasmas 6, 2834 (1999)] -- bootstrap/conductivity.
  [W. A. Houlberg et al., Phys. Plasmas 4, 3230 (1997)] -- arbitrary regime.

The optional Shaing near-axis term and its radial blend are reduced extensions;
the blending location and multiplier are calibration parameters.
"""

import jax.numpy as jnp
from .current import (
    sauter_ln_lambda_e,
    sauter_ln_lambda_ii,
)
from .heating import effective_ion_mass_amu
from .grid import radial_gradient as grid_radial_gradient

E_CHARGE = 1.602176634e-19
KEV_TO_J = 1.602176634e-16
M_E = 9.1093837015e-31
M_AMU = 1.66053906660e-27


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


def _maybe_abs(x, sim, width=1.0e-8):
    if getattr(sim, 'differentiable_smooth_mode', False):
        return jnp.sqrt(x * x + width * width)
    return jnp.abs(x)


def _main_ion_density_fraction(machine):
    """Return n_main / n_e for one main singly charged ion plus one impurity."""
    Zimp = jnp.maximum(jnp.asarray(getattr(machine, "impurity_Z", 6.0)), 1.01)
    Zeff = jnp.maximum(jnp.asarray(getattr(machine, "Zeff", 1.0)), 1.0)
    # Quasi-neutrality and Zeff = sum n_s Z_s^2 / n_e with Zi=1:
    # n_imp/n_e = (Zeff-1)/(Zimp*(Zimp-1)).
    f_imp = jnp.clip((Zeff - 1.0) / (Zimp * (Zimp - 1.0) + 1.0e-12), 0.0, 0.95 / Zimp)
    return jnp.clip(1.0 - Zimp * f_imp, 1.0e-3, 1.0)


def _impurity_strength_alpha_I(machine):
    """Angioni-Sauter single-impurity convention: alpha_I = Zeff - 1."""
    return jnp.maximum(jnp.asarray(getattr(machine, "Zeff", 1.0)) - 1.0, 0.0)


def _flux_surface_average(theta, R, jac, value):
    weight = R * jac
    num = jnp.trapezoid(value * weight, theta, axis=1)
    den = jnp.trapezoid(weight, theta, axis=1)
    return num / (den + 1.0e-30)


def _take_surface_value(A, idx):
    return jnp.take_along_axis(A, idx[:, None], axis=1)[:, 0]


def _geometry_profiles(rho, machine, sim, eq=None):
    """Return geometry profiles needed by Angioni-Sauter.

    The quantities gm4=<B^-2> and gm5=<B^2> are dimensional here, so
    gm4*B0^2 and B0^2/gm5 reproduce the dimensionless factors.
    """
    rho = jnp.asarray(rho)
    Bt_abs = _maybe_lower(jnp.abs(jnp.asarray(machine.Bt)), 0.2, sim, 1.0e-3)
    if eq is not None and all(hasattr(eq, name) for name in ("R", "Z", "jac", "theta")):
        R2d = jnp.asarray(eq.R)
        Z2d = jnp.asarray(eq.Z)
        Rmax = jnp.max(R2d, axis=1)
        Rmin = jnp.min(R2d, axis=1)
        Zmax = jnp.max(Z2d, axis=1)
        Zmin = jnp.min(Z2d, axis=1)
        Rmaj = _maybe_lower(0.5 * (Rmax + Rmin), 0.1, sim, 1.0e-3)
        rminor = _maybe_lower(0.5 * (Rmax - Rmin), 1.0e-4 * machine.a, sim, 1.0e-5)
        epsilon = _maybe_bound(rminor / Rmaj, 1.0e-4, 0.95, sim, 1.0e-4)
        idx_top = jnp.argmax(Z2d, axis=1)
        R_top = _take_surface_value(R2d, idx_top)
        delta = _maybe_bound((Rmaj - R_top) / rminor, -0.95, 0.95, sim, 1.0e-3)
        if hasattr(eq, "Bmag"):
            Bmag = jnp.asarray(eq.Bmag)
        elif hasattr(eq, "Btor") and hasattr(eq, "Bpol"):
            Bmag = jnp.sqrt(jnp.asarray(eq.Btor) ** 2 + jnp.asarray(eq.Bpol) ** 2)
        else:
            Bmag = Bt_abs + jnp.zeros_like(R2d)
        B2_avg = _flux_surface_average(eq.theta, R2d, eq.jac, Bmag**2)
        Bm2_avg = _flux_surface_average(eq.theta, R2d, eq.jac, 1.0 / (Bmag**2 + 1.0e-30))
        F = jnp.asarray(eq.F) if hasattr(eq, "F") else machine.R0 * machine.Bt + jnp.zeros_like(rho)
        if hasattr(eq, "psi_pol"):
            psi_pol = jnp.asarray(eq.psi_pol)
            dpsi_dr = grid_radial_gradient(psi_pol, rho, machine.a)
        else:
            dpsi_dr = 2.0 * jnp.pi * machine.Bt * machine.a * rho / jnp.maximum(jnp.abs(rho * 0.0 + 1.0), 1.0)
        return Rmaj, rminor, epsilon, delta, B2_avg, Bm2_avg, F, dpsi_dr, Bt_abs

    epsilon = _maybe_bound(machine.a * rho / machine.R0, 1.0e-4, 0.95, sim, 1.0e-4)
    tri_pow = float(getattr(sim, "triangularity_profile_power", 1.0)) if sim is not None else 1.0
    delta = machine.delta * jnp.maximum(rho, 1.0e-4) ** tri_pow
    Rmaj = jnp.asarray(machine.R0) + jnp.zeros_like(rho)
    rminor = machine.a * rho
    B2_avg = Bt_abs**2 + jnp.zeros_like(rho)
    Bm2_avg = 1.0 / (Bt_abs**2 + 1.0e-30) + jnp.zeros_like(rho)
    F = machine.R0 * machine.Bt + jnp.zeros_like(rho)
    dpsi_dr = 2.0 * jnp.pi * machine.Bt * machine.a * rho / jnp.maximum(jnp.abs(rho * 0.0 + 1.0), 1.0)
    return Rmaj, rminor, epsilon, delta, B2_avg, Bm2_avg, F, dpsi_dr, Bt_abs


def _signed_floor(x, floor):
    sign = jnp.where(x < 0.0, -1.0, 1.0)
    return jnp.where(jnp.abs(x) < floor, sign * floor, x)


def _axis_copy_first(y):
    if y.size <= 1:
        return y
    return y.at[0].set(y[1])


def _trapped_fractions(epsilon, delta, B2_avg, Bm2_avg):
    """Effective trapped fractions f_t and f_t^d."""
    eps = jnp.clip(epsilon, 1.0e-5, 0.95)
    aa = (1.0 - eps) / (1.0 + eps)
    epseff = 0.67 * (1.0 - 1.4 * jnp.abs(delta) * delta) * eps
    epseff = jnp.clip(epseff, 1.0e-8, 0.99)
    ftrap = 1.0 - jnp.sqrt(jnp.maximum(aa, 1.0e-12)) * (1.0 - epseff) / (1.0 + 2.0 * jnp.sqrt(epseff))
    B2Bm2 = jnp.maximum(B2_avg * Bm2_avg, 1.0 + 1.0e-8)
    ftrap_d = 1.0 - (1.0 - ftrap) / B2Bm2
    return jnp.clip(ftrap, 0.0, 0.995), jnp.clip(ftrap_d, 0.0, 0.995), B2Bm2


def _Fmn_X(X, Z_eff):
    """Angioni-Sauter Eq. (24) F_mn matrix elements.

    Reference: [C. Angioni and O. Sauter, Phys. Plasmas 7, 1224 (2000)].
    """
    X = jnp.clip(X, 0.0, 0.995)
    Z = jnp.maximum(Z_eff, 1.0)
    F11 = X + X * (0.9 + X * (-1.9 + X * (1.6 - 0.6 * X))) / (Z + 0.5)
    F12 = X + X * (0.6 + X * (-0.95 + X * (0.3 + 0.05 * X))) / (Z + 0.5)
    F22 = X + X * (-0.11 + X * (0.08 + 0.03 * X)) / (Z + 0.5)
    return F11, F12, F22


def _Kmn_coeffs(Z_eff):
    """Correction coefficients for all-collisionality extension."""
    Z = jnp.maximum(Z_eff, 1.0)
    a11 = (1.0 + 3.0 * Z) / (0.77 + 1.22 * Z)
    a12 = (0.72 + 0.42 * Z) / (1.0 + 0.5 * Z)
    a22 = 0.46 * jnp.ones_like(Z)
    b11 = (1.0 + 1.1 * Z) / (1.37 * Z)
    b12 = (1.0 + Z) / (2.99 * Z)
    b22 = Z / (-3.0 + 5.32 * Z)
    c11 = (0.1 + 0.34 * Z) / (1.65 * Z)
    c12 = (0.27 + 0.4 * Z) / (1.0 + 3.0 * Z)
    c22 = (0.22 + 0.55 * Z) / (-1.0 + 7.0 * Z)
    d11 = 0.23 * Z / (-1.0 + 3.85 * Z)
    d12 = (0.22 + 0.38 * Z) / (1.0 + 6.1 * Z)
    d22 = (0.25 + 0.05 * Z) / (1.0 + 0.82 * Z)
    return (a11, a12, a22), (b11, b12, b22), (c11, c12, c22), (d11, d12, d22)


def _sauter_L31(ft, nue, Z):
    X = ft / (1.0 + (1.0 - 0.1 * ft) * jnp.sqrt(nue) + 0.5 * (1.0 - ft) * nue / Z)
    return (1.0 + 1.4 / (Z + 1.0)) * X - 1.9 / (Z + 1.0) * X**2 + 0.3 / (Z + 1.0) * X**3 + 0.2 / (Z + 1.0) * X**4


def _sauter_L32(ft, nue, Z):
    X = ft / (1.0 + 0.26 * (1.0 - ft) * jnp.sqrt(nue) + 0.18 * (1.0 - 0.37 * ft) * nue / jnp.sqrt(Z))
    Y = ft / (1.0 + (1.0 + 0.6 * ft) * jnp.sqrt(nue) + 0.85 * (1.0 - 0.37 * ft) * nue * (1.0 + Z))
    F32ee = (
        (0.05 + 0.62 * Z) / (Z * (1.0 + 0.44 * Z)) * (X - X**4)
        + 1.0 / (1.0 + 0.22 * Z) * (X**2 - X**4 - 1.2 * (X**3 - X**4))
        + 1.2 / (1.0 + 0.5 * Z) * X**4
    )
    F32ei = (
        -(0.56 + 1.93 * Z) / (Z * (1.0 + 0.44 * Z)) * (Y - Y**4)
        + 4.95 / (1.0 + 2.48 * Z) * (Y**2 - Y**4 - 0.55 * (Y**3 - Y**4))
        - 1.2 / (1.0 + 0.5 * Z) * Y**4
    )
    return F32ee + F32ei


def _calculate_Kmn(ftrap, ftrap_d, Z_eff, B2Bm2, nue_star, nui_star, alpha_I):
    F11_ft, F12_ft, F22_ft = _Fmn_X(ftrap, Z_eff)
    F11_ftd, F12_ftd, F22_ftd = _Fmn_X(ftrap_d, Z_eff)
    (a11, a12, a22), (b11, b12, b22), (c11, c12, c22), (d11, d12, d22) = _Kmn_coeffs(Z_eff)

    nue_eff = nue_star / (1.0 + 7.0 * ftrap**2)
    sqrt_nue_eff = jnp.sqrt(jnp.maximum(nue_eff, 0.0))
    nui_eff = nui_star / (1.0 + 7.0 * ftrap**2)
    FPS = 1.0 - 1.0 / jnp.maximum(B2Bm2, 1.0 + 1.0e-12)
    FPS4 = B2Bm2 - 1.0

    # Banana-regime electron coefficients, Eq. (23), and H transforms Eq. (30c,f).
    K11e0 = -0.5 * F11_ftd
    K12e0 = 0.75 * F12_ftd
    K22e0 = -(13.0 / 8.0 + 1.0 / (jnp.sqrt(2.0) * Z_eff)) * F22_ftd
    K14e0 = -0.5 * F11_ft
    K24e0 = 0.75 * F12_ft

    H11_0 = K11e0
    H12_0 = K12e0 + 2.5 * K11e0
    H22_0 = K22e0 + 5.0 * K12e0 - 6.25 * K11e0
    H41_0 = K14e0
    H42_0 = K24e0 + 2.5 * K14e0

    temp1 = nue_eff * ftrap_d**3 * (1.0 + ftrap_d**6)
    H11 = H11_0 / (1.0 + a11 * sqrt_nue_eff + b11 * nue_eff) - d11 * temp1 / (1.0 + c11 * temp1 + 1.0e-30) * FPS
    H12 = H12_0 / (1.0 + a12 * sqrt_nue_eff + b12 * nue_eff) - d12 * temp1 / (1.0 + c12 * temp1 + 1.0e-30) * FPS
    H22 = H22_0 / (1.0 + a22 * sqrt_nue_eff + b22 * nue_eff) - d22 * temp1 / (1.0 + c22 * temp1 + 1.0e-30) * FPS

    temp2 = 1.0 / (1.0 + nue_eff**2 * ftrap**12)
    temp4 = nue_eff * ftrap**3 * (1.0 + 0.8 * ftrap**3)
    H41 = (H41_0 / (1.0 + a11 * sqrt_nue_eff + b11 * nue_eff) - d11 * temp4 / (1.0 + c11 * temp4 + 1.0e-30) * FPS4) * temp2
    H42 = (H42_0 / (1.0 + a12 * sqrt_nue_eff + b12 * nue_eff) - d12 * temp4 / (1.0 + c12 * temp4 + 1.0e-30) * FPS4) * temp2

    n = ftrap.shape[0]
    Kmn_e = jnp.zeros((n, 4, 4), dtype=ftrap.dtype)
    Kmn_e = Kmn_e.at[:, 0, 0].set(H11)
    Kmn_e = Kmn_e.at[:, 0, 1].set(H12 - 2.5 * H11)
    Kmn_e = Kmn_e.at[:, 1, 0].set(Kmn_e[:, 0, 1])
    Kmn_e = Kmn_e.at[:, 1, 1].set(H22 - 5.0 * H12 + 6.25 * H11)
    Kmn_e = Kmn_e.at[:, 0, 3].set(H41)
    Kmn_e = Kmn_e.at[:, 3, 0].set(Kmn_e[:, 0, 3])
    Kmn_e = Kmn_e.at[:, 1, 3].set(H42 - 2.5 * H41)
    Kmn_e = Kmn_e.at[:, 3, 1].set(Kmn_e[:, 1, 3])
    # Ware-pinch/current-drive coupling uses Sauter 1999 bootstrap coefficients.
    Kmn_e = Kmn_e.at[:, 0, 2].set(-_sauter_L31(ftrap, nue_star, Z_eff))
    Kmn_e = Kmn_e.at[:, 2, 0].set(Kmn_e[:, 0, 2])
    Kmn_e = Kmn_e.at[:, 1, 2].set(-_sauter_L32(ftrap, nue_star, Z_eff))
    Kmn_e = Kmn_e.at[:, 2, 1].set(Kmn_e[:, 1, 2])

    alpha = -(0.62 + 1.5 * alpha_I) / (0.53 + alpha_I + 1.0e-30) * ((1.0 - ftrap) / (1.0 - 0.22 * ftrap - 0.19 * ftrap**2 + 1.0e-30))
    F22_i = (1.0 - 0.55) * (1.0 + 1.54 * alpha_I) * ftrap_d + ftrap_d**2 * (0.75 + ftrap_d * (-0.7 + 0.5 * ftrap_d)) * (1.0 + 2.92 * alpha_I)
    K11_i = 1.0 / (0.11 + 1.7 * ftrap - 1.25 * ftrap**2 + 0.44 * ftrap**3 + 1.0e-30) - 1.0
    mu_i_eff = nui_eff * (1.0 + 1.54 * alpha_I)
    Hp = 1.0 + 1.33 * alpha_I * (1.0 + 0.60 * alpha_I) / (1.0 + 1.79 * alpha_I + 1.0e-30)
    fgeom = ftrap_d**3 * (1.0 + ftrap_d**6)
    a2, b2, c2, d2 = 1.03, 0.31, 0.22, 0.175
    K22_i = -F22_i / (1.0 + a2 * jnp.sqrt(jnp.maximum(mu_i_eff, 0.0)) + b2 * mu_i_eff) - d2 * mu_i_eff * fgeom / (1.0 + c2 * mu_i_eff * fgeom + 1.0e-30) * Hp * FPS

    Kmn_i = jnp.zeros((n, 2, 2), dtype=ftrap.dtype)
    Kmn_i = Kmn_i.at[:, 0, 0].set(K11_i)
    Kmn_i = Kmn_i.at[:, 1, 1].set(K22_i)
    Kmn_i = Kmn_i.at[:, 0, 1].set(-alpha)
    Kmn_i = Kmn_i.at[:, 1, 0].set(alpha)
    return Kmn_e, Kmn_i


def _calculate_Lmn(Kmn_e, Kmn_i, Te, Ti, ne_m3, ni_m3, q, Rmaj, epsilon, F, dpsi_dr, B0, B2_avg, Bm2_avg, Ai, Zi, nue_star, nui_star):
    Te_J = _maybe_lower(Te, 0.03, None) * KEV_TO_J
    Ti_J = _maybe_lower(Ti, 0.03, None) * KEV_TO_J
    vte = jnp.sqrt(2.0 * Te_J / M_E)
    vti = jnp.sqrt(2.0 * Ti_J / (Ai * M_AMU))
    tau_e = q * Rmaj / (nue_star * epsilon**1.5 * vte + 1.0e-30)
    tau_i = q * Rmaj / (nui_star * epsilon**1.5 * vti + 1.0e-30)
    B0_safe = _maybe_lower(jnp.abs(B0), 1.0e-4, None)
    r_larmor_e = M_E * vte / (E_CHARGE * B0_safe)
    r_larmor_i = M_AMU * Ai * vti / (E_CHARGE * Zi * B0_safe)
    dpsi_dr = _signed_floor(dpsi_dr, 1.0e-10)
    Ld = ne_m3 * r_larmor_e**2 / (tau_e + 1.0e-30) * dpsi_dr**2
    Ldi = ni_m3 * r_larmor_i**2 / (tau_i + 1.0e-30) * dpsi_dr**2
    Lb = F * ne_m3
    Lbi = F * ni_m3
    Lsi = ni_m3 * (E_CHARGE * Zi) ** 2 * tau_i * B0**2 / (Ai * M_AMU * Ti_J + 1.0e-30)
    gm4 = Bm2_avg
    gm5 = B2_avg

    n = Te.shape[0]
    Lmn_e = jnp.zeros((n, 4, 4), dtype=Te.dtype)
    Lmn_e = Lmn_e.at[:, 0, 0].set(Kmn_e[:, 0, 0] * Ld * gm4 * B0**2)
    Lmn_e = Lmn_e.at[:, 0, 1].set(Kmn_e[:, 0, 1] * Ld * gm4 * B0**2)
    Lmn_e = Lmn_e.at[:, 0, 2].set(Kmn_e[:, 0, 2] * Lb)
    Lmn_e = Lmn_e.at[:, 0, 3].set(Kmn_e[:, 0, 3] * Ld / (gm5 + 1.0e-30) * B0**2)
    Lmn_e = Lmn_e.at[:, 1, 0].set(Lmn_e[:, 0, 1])
    Lmn_e = Lmn_e.at[:, 1, 1].set(Kmn_e[:, 1, 1] * Ld * gm4 * B0**2)
    Lmn_e = Lmn_e.at[:, 1, 2].set(Kmn_e[:, 1, 2] * Lb)
    Lmn_e = Lmn_e.at[:, 1, 3].set(Kmn_e[:, 1, 3] * Ld / (gm5 + 1.0e-30) * B0**2)
    Lmn_e = Lmn_e.at[:, 2, 0].set(Lmn_e[:, 0, 2])
    Lmn_e = Lmn_e.at[:, 2, 1].set(Lmn_e[:, 1, 2])
    Lmn_e = Lmn_e.at[:, 2, 3].set(Kmn_e[:, 2, 3] * Lb)
    Lmn_e = Lmn_e.at[:, 3, 0].set(Lmn_e[:, 0, 3])
    Lmn_e = Lmn_e.at[:, 3, 1].set(Lmn_e[:, 1, 3])
    Lmn_e = Lmn_e.at[:, 3, 2].set(Lmn_e[:, 2, 3])
    Lmn_e = Lmn_e.at[:, 3, 3].set(Kmn_e[:, 3, 3] * Ld / (gm5 + 1.0e-30) * B0**2)

    Lmn_i = jnp.zeros((n, 2, 2), dtype=Te.dtype)
    Lmn_i = Lmn_i.at[:, 0, 0].set(Kmn_i[:, 0, 0] * Lsi * gm5 / (B0**2 + 1.0e-30))
    Lmn_i = Lmn_i.at[:, 0, 1].set(Kmn_i[:, 0, 1] * Lbi)
    Lmn_i = Lmn_i.at[:, 1, 0].set(Kmn_i[:, 1, 0] * Lbi)
    Lmn_i = Lmn_i.at[:, 1, 1].set(Kmn_i[:, 1, 1] * Ldi * gm4 * B0**2)
    return Lmn_e, Lmn_i



def _elongation_profile_from_geometry(rho, machine, sim, eq=None):
    """Return a positive elongation profile for the Shaing near-axis model."""
    rho = jnp.asarray(rho)
    if eq is not None and all(hasattr(eq, name) for name in ("R", "Z")):
        R2d = jnp.asarray(eq.R)
        Z2d = jnp.asarray(eq.Z)
        rminor = _maybe_lower(0.5 * (jnp.max(R2d, axis=1) - jnp.min(R2d, axis=1)), 1.0e-4 * machine.a, sim, 1.0e-5)
        zhalf = _maybe_lower(0.5 * (jnp.max(Z2d, axis=1) - jnp.min(Z2d, axis=1)), 1.0e-4 * machine.a, sim, 1.0e-5)
        return _maybe_bound(zhalf / rminor, 0.3, 5.0, sim, 1.0e-3)
    return _maybe_bound(jnp.asarray(machine.kappa) + jnp.zeros_like(rho), 0.3, 5.0, sim, 1.0e-3)


def _shaing_near_axis_chi_i(rho, Ti, ne20, q, machine, sim, Rmaj, epsilon, F, dpsi_dr, B0, Ai, Zi, nui_star, eq=None):
    """Shaing-Hazeltine-Zarnstorff near-axis ion thermal diffusivity.

    This mirrors the optional ion correction in Angioni-Sauter module.
    The model is intended only for the magnetic-axis/very-low-epsilon region;
    callers should blend or localize it before combining with Angioni-Sauter.
    """
    rho = jnp.asarray(rho)
    Ti_J = _maybe_lower(Ti, 0.03, sim, 1.0e-3) * KEV_TO_J
    m_ion = _maybe_lower(Ai, 1.0, sim, 1.0e-3) * M_AMU
    q_abs = _maybe_bound(jnp.abs(q), 0.05, 50.0, sim, 1.0e-2)
    R_safe = _maybe_lower(Rmaj, 0.1, sim, 1.0e-3)
    kappa = _elongation_profile_from_geometry(rho, machine, sim, eq=eq)
    F_abs = _maybe_lower(jnp.abs(F), 1.0e-6, sim, 1.0e-8)
    B0_abs = _maybe_lower(jnp.abs(B0), 0.05, sim, 1.0e-3)
    v_ti = jnp.sqrt(2.0 * Ti_J / (m_ion + 1.0e-30))
    # nu_i* = q R nu_ii / (epsilon^(3/2) v_ti).
    nu_ii = jnp.maximum(nui_star, 0.0) * _maybe_lower(epsilon, 1.0e-5, sim, 1.0e-5) ** 1.5 * v_ti / (q_abs * R_safe + 1.0e-30)
    omega0 = E_CHARGE * _maybe_lower(Zi, 0.1, sim, 1.0e-3) * B0_abs / (m_ion + 1.0e-30)

    # Shaing et al. large-aspect-ratio near-axis forms.
    C1 = jnp.sqrt(2.0 * q_abs / (kappa * F_abs * R_safe + 1.0e-30))
    f_t_ion = (F_abs * v_ti * C1**2 / (omega0 + 1.0e-30)) ** (1.0 / 3.0)
    delta_psi_ion = (F_abs**2 * v_ti**2 * C1 / (omega0**2 + 1.0e-30)) ** (2.0 / 3.0)

    dpsi = _signed_floor(dpsi_dr, 1.0e-12)
    if dpsi.size > 1:
        dpsi = dpsi.at[0].set(dpsi[1])
    conversion = 1.0 / ((dpsi / (2.0 * jnp.pi)) ** 2 + 1.0e-30)
    chi_shaing = nu_ii * delta_psi_ion**2 / (_maybe_lower(f_t_ion, 1.0e-12, sim, 1.0e-12)) * conversion
    mult = float(getattr(sim, "neoclassical_shaing_ion_multiplier", 1.8)) if sim is not None else 1.8
    return mult * chi_shaing


def _shaing_blend_alpha(rho, sim):
    """Weight alpha for Angioni in (1-alpha)*Shaing + alpha*Angioni."""
    start = float(getattr(sim, "neoclassical_shaing_blend_start", 0.2)) if sim is not None else 0.2
    rate = float(getattr(sim, "neoclassical_shaing_blend_rate", 5.0)) if sim is not None else 5.0
    return 1.0 / (1.0 + jnp.exp(-2.0 * rate * (jnp.asarray(rho) - start)))


def chang_hinton_neoclassical_diffusivities(rho, Te, Ti, ne20, q, machine, sim, eq=None):
    """Reduced positive scalar neoclassical diffusivities [m^2/s].

    The ion heat channel uses the common Chang-Hinton all-collisionality fit,
    written as a positive scalar diffusivity proportional to
    ``q^2 rho_i^2 nu_ii / epsilon^(3/2)`` with banana/plateau/Pfirsch-Schlueter
    interpolation terms [C. S. Chang and F. L. Hinton, Phys. Fluids 25, 1493
    (1982)].  This avoids the sign-indefinite scalar projection of the full
    Angioni-Sauter flux-force matrix and is intended as a robust analytic
    comparison model.

    Electron heat and particle channels are reduced random-walk estimates using
    the same collisionality interpolation scale.  They are positive by
    construction and deliberately simple; tune ``neoclassical_chi_scale`` and
    ``neoclassical_D_scale`` when matching a higher-fidelity neoclassical code.
    """
    rho = jnp.asarray(rho)
    Te = _maybe_lower(jnp.asarray(Te), 0.05, sim, 1.0e-3)
    Ti = _maybe_lower(jnp.asarray(Ti), 0.05, sim, 1.0e-3)
    ne20 = _maybe_lower(jnp.asarray(ne20), 0.03, sim, 1.0e-4)
    q_abs = _maybe_bound(jnp.abs(jnp.asarray(q)), 0.2, 20.0, sim, 1.0e-2)
    Zeff = _maybe_lower(jnp.asarray(getattr(machine, "Zeff", 1.0)), 1.0, sim, 1.0e-3) + jnp.zeros_like(rho)
    Zi = jnp.ones_like(rho)
    Ai = effective_ion_mass_amu(machine) + jnp.zeros_like(rho)

    Rmaj, _rminor, epsilon, _delta, _B2_avg, _Bm2_avg, _F, _dpsi_dr, B0 = _geometry_profiles(
        rho, machine, sim, eq=eq
    )
    eps_min = float(getattr(sim, "chang_hinton_epsilon_min", 0.03)) if sim is not None else 0.03
    eps = _maybe_bound(epsilon, eps_min, 0.95, sim, 1.0e-4)
    R_safe = _maybe_lower(Rmaj, 0.1, sim, 1.0e-3)
    B_abs = _maybe_lower(jnp.abs(B0), 0.05, sim, 1.0e-3)

    f_main = _main_ion_density_fraction(machine)
    ne_m3 = ne20 * 1.0e20
    ni_m3 = ne_m3 * f_main
    Te_eV = Te * 1.0e3
    Ti_eV = Ti * 1.0e3
    Te_J = Te * KEV_TO_J
    Ti_J = Ti * KEV_TO_J

    lnLe = sauter_ln_lambda_e(ne_m3, Te_eV)
    lnLii = sauter_ln_lambda_ii(ni_m3, Ti_eV, Zi)
    nue_star = 6.921e-18 * q_abs * R_safe * ne_m3 * Zeff * lnLe / (Te_eV**2 * eps**1.5 + 1.0e-30)
    nui_star = 4.90e-18 * q_abs * R_safe * ni_m3 * Zi**4 * lnLii / (Ti_eV**2 * eps**1.5 + 1.0e-30)
    nue_star = jnp.maximum(nue_star, 0.0)
    nui_star = jnp.maximum(nui_star, 0.0)

    meff_i = Ai * M_AMU
    vte = jnp.sqrt(2.0 * Te_J / (M_E + 1.0e-30))
    vti = jnp.sqrt(2.0 * Ti_J / (meff_i + 1.0e-30))
    nue = nue_star * eps**1.5 * vte / (q_abs * R_safe + 1.0e-30)
    nui = nui_star * eps**1.5 * vti / (q_abs * R_safe + 1.0e-30)

    rho_e = M_E * vte / (E_CHARGE * B_abs + 1.0e-30)
    rho_i = meff_i * vti / (E_CHARGE * Zi * B_abs + 1.0e-30)
    base_e = q_abs**2 * rho_e**2 * nue / (eps**1.5 + 1.0e-30)
    base_i = q_abs**2 * rho_i**2 * nui / (eps**1.5 + 1.0e-30)

    sqrt_eps = jnp.sqrt(eps)
    sqrt_nui = jnp.sqrt(jnp.maximum(nui_star, 0.0))
    alpha_I = _impurity_strength_alpha_I(machine) + jnp.zeros_like(rho)

    banana_plateau = (
        0.66 * (1.0 + 1.54 * alpha_I)
        + (1.88 * sqrt_eps - 1.54 * eps) * (1.0 + 3.75 * alpha_I)
    ) / (1.0 + 1.03 * sqrt_nui + 0.31 * nui_star + 1.0e-30)
    ps_factor = 1.0 + 1.33 * alpha_I * (1.0 + 0.60 * alpha_I) / (1.0 + 1.79 * alpha_I + 1.0e-30)
    pfirsch_schlueter = 0.59 * eps * nui_star / (1.0 + 0.74 * eps**1.5 * nui_star + 1.0e-30) * ps_factor
    chi_i = base_i * jnp.maximum(banana_plateau + pfirsch_schlueter, 0.0)

    # Reduced positive electron/particle channels.  The same interpolation shape
    # is used for electrons, but with a conservative coefficient because the
    # Chang-Hinton fit itself is an ion heat conductivity model.
    sqrt_nue = jnp.sqrt(jnp.maximum(nue_star, 0.0))
    e_interp = (
        0.66 + 1.88 * sqrt_eps - 1.54 * eps
    ) / (1.0 + 1.03 * sqrt_nue + 0.31 * nue_star + 1.0e-30)
    e_interp = e_interp + 0.59 * eps * nue_star / (1.0 + 0.74 * eps**1.5 * nue_star + 1.0e-30)
    chi_e = 0.5 * base_e * jnp.maximum(e_interp, 0.0)

    particle_fraction = float(getattr(sim, "chang_hinton_particle_fraction", 0.2)) if sim is not None else 0.2
    Dn = particle_fraction * jnp.sqrt(jnp.maximum(chi_e * chi_i, 0.0))

    chi_e = _axis_copy_first(chi_e)
    chi_i = _axis_copy_first(chi_i)
    Dn = _axis_copy_first(Dn)

    chi_e = getattr(sim, "neoclassical_chi_scale", 1.0) * chi_e
    chi_i = getattr(sim, "neoclassical_chi_scale", 1.0) * chi_i
    Dn = getattr(sim, "neoclassical_D_scale", 1.0) * Dn
    hi = getattr(sim, "neoclassical_chi_max", 5.0)
    return (
        _maybe_bound(chi_e, 0.0, hi, sim, 1.0e-2),
        _maybe_bound(chi_i, 0.0, hi, sim, 1.0e-2),
        _maybe_bound(Dn, 0.0, hi, sim, 1.0e-2),
    )

def angioni_neoclassical_diffusivities(rho, Te, Ti, ne20, q, machine, sim, eq=None, _allow_neonn_fallback: bool = True):
    """Angioni-Sauter neoclassical chi_e, chi_i, and D_e:
    construct the Angioni-Sauter K_mn matrices, dimensional L_mn matrices,
    form the heat and particle flux responses, then extract effective
    ``chi_neo_e``, ``chi_neo_i``, and ``D_neo_e``.  The Ware pinch and
    convective density pinch are not returned because tokagrad's present
    transport interface only carries a scalar density diffusivity ``Dn``.
    """
    model = getattr(sim, "neoclassical_transport_model", "angioni")
    if model == "none":
        z = jnp.zeros_like(rho)
        return z, z, z
    if model == "neonn_jax" and _allow_neonn_fallback:
        from .neonn_jax import neonn_jax_diffusivities
        return neonn_jax_diffusivities(rho, Te, Ti, ne20, q, machine, sim)
    if model in ("chang_hinton", "chang-hinton", "hinton_chang"):
        return chang_hinton_neoclassical_diffusivities(rho, Te, Ti, ne20, q, machine, sim, eq=eq)

    rho = jnp.asarray(rho)
    Te = _maybe_lower(jnp.asarray(Te), 0.05, sim, 1.0e-3)
    Ti = _maybe_lower(jnp.asarray(Ti), 0.05, sim, 1.0e-3)
    ne20 = _maybe_lower(jnp.asarray(ne20), 0.03, sim, 1.0e-4)
    q = _maybe_bound(jnp.abs(jnp.asarray(q)), 0.2, 20.0, sim, 1.0e-2)
    Zeff = jnp.maximum(jnp.asarray(getattr(machine, "Zeff", 1.0)), 1.0) + jnp.zeros_like(rho)
    Zi = jnp.ones_like(rho)
    Ai = effective_ion_mass_amu(machine) + jnp.zeros_like(rho)

    Rmaj, _rminor, epsilon, delta, B2_avg, Bm2_avg, F, dpsi_dr, B0 = _geometry_profiles(rho, machine, sim, eq=eq)
    ftrap, ftrap_d, B2Bm2 = _trapped_fractions(epsilon, delta, B2_avg, Bm2_avg)

    f_main = _main_ion_density_fraction(machine)
    ne_m3 = ne20 * 1.0e20
    ni_m3 = ne_m3 * f_main
    Te_eV = Te * 1.0e3
    Ti_eV = Ti * 1.0e3
    lnLe = sauter_ln_lambda_e(ne_m3, Te_eV)
    lnLii = sauter_ln_lambda_ii(ni_m3, Ti_eV, Zi)
    nue_star = 6.921e-18 * q * Rmaj * ne_m3 * Zeff * lnLe / (Te_eV**2 * epsilon**1.5 + 1.0e-30)
    nui_star = 4.90e-18 * q * Rmaj * ni_m3 * Zi**4 * lnLii / (Ti_eV**2 * epsilon**1.5 + 1.0e-30)
    nue_star = jnp.maximum(nue_star, 0.0)
    nui_star = jnp.maximum(nui_star, 0.0)

    alpha_I = _impurity_strength_alpha_I(machine) + jnp.zeros_like(rho)
    Kmn_e, Kmn_i = _calculate_Kmn(ftrap, ftrap_d, Zeff, B2Bm2, nue_star, nui_star, alpha_I)
    Lmn_e, Lmn_i = _calculate_Lmn(Kmn_e, Kmn_i, Te, Ti, ne_m3, ni_m3, q, Rmaj, epsilon, F, dpsi_dr, B0, B2_avg, Bm2_avg, Ai, Zi, nue_star, nui_star)

    dne_dr = grid_radial_gradient(ne_m3, rho, machine.a)
    dni_dr = grid_radial_gradient(ni_m3, rho, machine.a)
    dte_dr = grid_radial_gradient(Te, rho, machine.a)
    dti_dr = grid_radial_gradient(Ti, rho, machine.a)
    dpsi_dr_safe = _signed_floor(dpsi_dr, 1.0e-10)
    dlnne_dpsi = (dne_dr / (ne_m3 + 1.0e-30)) / dpsi_dr_safe
    dlnni_dpsi = (dni_dr / (ni_m3 + 1.0e-30)) / dpsi_dr_safe
    dlnte_dpsi = (dte_dr / (Te + 1.0e-30)) / dpsi_dr_safe
    dlnti_dpsi = (dti_dr / (Ti + 1.0e-30)) / dpsi_dr_safe

    pe = ne_m3 * Te * KEV_TO_J
    pi = ni_m3 * Ti * KEV_TO_J
    Rpe = pe / (pe + pi + 1.0e-30)
    alpha = -Kmn_i[:, 0, 1]
    E_parallel = jnp.zeros_like(rho)

    Be2 = (
        Lmn_e[:, 1, 0] * dlnne_dpsi
        + (Lmn_e[:, 1, 0] + Lmn_e[:, 1, 1]) * dlnte_dpsi
        + (1.0 - Rpe) / (Rpe + 1.0e-30) * Lmn_e[:, 1, 0] * dlnni_dpsi
        + (1.0 - Rpe) / (Rpe + 1.0e-30) * (Lmn_e[:, 1, 0] + alpha * Lmn_e[:, 1, 3]) * dlnti_dpsi
        + Lmn_e[:, 1, 2] * E_parallel / (B0 + 1.0e-30)
    )
    Bi2 = (
        alpha * Lmn_e[:, 3, 0] * dlnne_dpsi
        + alpha * (Lmn_e[:, 3, 0] + Lmn_e[:, 3, 1]) * dlnte_dpsi
        + alpha * (1.0 - Rpe) / (Rpe + 1.0e-30) * Lmn_e[:, 3, 0] * dlnni_dpsi
        + alpha * Lmn_e[:, 3, 2] * E_parallel / (B0 + 1.0e-30)
        + (Lmn_i[:, 1, 1] + (1.0 - Rpe) / (Rpe + 1.0e-30) * alpha**2 / Zi * Lmn_e[:, 3, 3]) * dlnti_dpsi
    )

    dpsi2 = dpsi_dr_safe**2
    chi_e = -Be2 / (ne_m3 * dlnte_dpsi * dpsi2 + 1.0e-30)
    chi_i = -Bi2 / (ni_m3 * dlnti_dpsi * dpsi2 + 1.0e-30)
    Dn = -Lmn_e[:, 0, 0] / (ne_m3 * dpsi2 + 1.0e-30)

    # Follow near-axis constant extrapolation for quantities obtained
    # by division through gradients/psi'.
    chi_e = _axis_copy_first(chi_e)
    chi_i = _axis_copy_first(chi_i)
    Dn = _axis_copy_first(Dn)

    if bool(getattr(sim, "neoclassical_abs_effective_diffusivity", True)):
        chi_e = _maybe_abs(chi_e, sim, 1.0e-8)
        chi_i = _maybe_abs(chi_i, sim, 1.0e-8)
        Dn = _maybe_abs(Dn, sim, 1.0e-10)

    # Optional Shaing-Hazeltine-Zarnstorff near-axis correction for ion heat
    # transport.  Shaing is a near-axis ion model,
    # while electron heat and density channels remain Angioni-Sauter.
    shaing_mode = str(getattr(sim, "neoclassical_shaing_ion_mode", "off")).lower()
    if shaing_mode not in ("off", "none", "false", "0"):
        chi_i_shaing = _shaing_near_axis_chi_i(
            rho=rho, Ti=Ti, ne20=ne20, q=q, machine=machine, sim=sim,
            Rmaj=Rmaj, epsilon=epsilon, F=F, dpsi_dr=dpsi_dr_safe, B0=B0,
            Ai=Ai, Zi=Zi, nui_star=nui_star, eq=eq,
        )
        alpha_blend = _shaing_blend_alpha(rho, sim)
        if shaing_mode in ("blend", "correction"):
            chi_i = (1.0 - alpha_blend) * chi_i_shaing + alpha_blend * chi_i
        elif shaing_mode in ("add", "add_blended", "additive"):
            # Localized additive correction: strongest near axis, negligible outside.
            chi_i = chi_i + (1.0 - alpha_blend) * chi_i_shaing
        elif shaing_mode in ("add_full", "full_add"):
            chi_i = chi_i + chi_i_shaing
        elif shaing_mode in ("replace", "shaing"):
            chi_i = chi_i_shaing
        else:
            raise ValueError(
                f"Unknown neoclassical_shaing_ion_mode={shaing_mode!r}. "
                'Use "off", "blend", "add", "add_full", or "replace".'
            )

    chi_e = sim.neoclassical_chi_scale * chi_e
    chi_i = sim.neoclassical_chi_scale * chi_i
    Dn = sim.neoclassical_D_scale * Dn
    return (
        _maybe_bound(chi_e, 0.0, sim.neoclassical_chi_max, sim, 1.0e-2),
        _maybe_bound(chi_i, 0.0, sim.neoclassical_chi_max, sim, 1.0e-2),
        _maybe_bound(Dn, 0.0, sim.neoclassical_chi_max, sim, 1.0e-2),
    )
