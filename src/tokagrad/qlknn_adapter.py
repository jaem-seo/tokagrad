"""Adapter for Google DeepMind fusion_surrogates QLKNN_7_11.

From the inspected source code, the concrete API is:

    from fusion_surrogates.qlknn import qlknn_model
    model = qlknn_model.QLKNNModel.load_default_model()
    fluxes = model.predict(inputs)

The default model name is qlknn_7_11_v1. The model input order is:

    ['Ati', 'Ate', 'Ane', 'Ani', 'q', 'smag', 'x',
     'Ti_Te', 'LogNuStar', 'normni']

where Ati=A_i=R/L_Ti, Ate=R/L_Te, Ane=R/L_ne, Ani=R/L_ni,
smag is magnetic shear, x is local inverse aspect ratio r/R0, and
normni=n_i/n_e.

The model flux dictionary contains:
    efiITG, efeITG, pfeITG,
    efeTEM, efiTEM, pfeTEM,
    efeETG, gamma_max

The QLKNN outputs are gyroBohm-normalized flux-like quantities, not directly
m^2/s diffusivities. For the current stage-1 solver we convert them into
effective diffusivities by dividing heat fluxes by the corresponding normalized
temperature gradients. This is a pragmatic integrated-model closure, not a
replacement for a full flux-gradient inversion.

References:
  [J. Citrin et al., Nucl. Fusion 55, 092001 (2015)] -- early QuaLiKiz NN.
  [K. L. van de Plassche et al., Phys. Plasmas 27, 022310 (2020)] -- QLKNN.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional, Tuple

import jax.numpy as jnp
from .grid import radial_gradient as grid_radial_gradient


QLKNN_INPUT_NAMES = [
    "Ati",
    "Ate",
    "Ane",
    "Ani",
    "q",
    "smag",
    "x",
    "Ti_Te",
    "LogNuStar",
    "normni",
]

QLKNN_FLUX_NAMES = [
    "efiITG",
    "efeITG",
    "pfeITG",
    "efeTEM",
    "efiTEM",
    "pfeTEM",
    "efeETG",
    "gamma_max",
]



def _smooth_lower(x, lo, width=1.0e-3):
    w = jnp.maximum(width, 1.0e-8)
    return lo + w * jnp.logaddexp(0.0, (x - lo) / w)


def _smooth_upper(x, hi, width=1.0e-3):
    w = jnp.maximum(width, 1.0e-8)
    return hi - w * jnp.logaddexp(0.0, (hi - x) / w)


def _smooth_bounded(x, lo, hi, width=1.0e-3):
    return _smooth_upper(_smooth_lower(x, lo, width), hi, width)


def _smooth_abs(x, width=1.0e-8):
    return jnp.sqrt(x * x + width * width)


def _maybe_lower(x, lo, sim=None, width=1.0e-3):
    if sim is not None and getattr(sim, "differentiable_smooth_mode", False):
        return _smooth_lower(x, lo, width)
    return jnp.maximum(x, lo)


def _maybe_bound(x, lo, hi, sim=None, width=1.0e-3):
    if sim is not None and getattr(sim, "differentiable_smooth_mode", False):
        return _smooth_bounded(x, lo, hi, width)
    return jnp.clip(x, lo, hi)


def _maybe_abs(x, sim=None, width=1.0e-8):
    if sim is not None and getattr(sim, "differentiable_smooth_mode", False):
        return _smooth_abs(x, width)
    return jnp.abs(x)


def main_ion_density_fraction(machine):
    """Return n_i/n_e from Zeff and a single impurity-Z approximation.

    The old QLKNN adapter used normni=1.0, which implicitly assumed no
    impurity dilution.  With the same quasineutral single-impurity model used
    by the TGLF adapter,

        n_Z/n_e = (Zeff-1)/(Z*(Z-1)),
        n_i/n_e = 1 - Z*n_Z/n_e.

    This is still a reduced composition model, but it uses the configured Zeff
    and impurity_Z instead of a hard-coded pure-main-ion proxy.
    """
    Z = jnp.maximum(jnp.asarray(getattr(machine, "impurity_Z", 6.0)), 2.0)
    Zeff = jnp.clip(jnp.asarray(machine.Zeff), 1.0, Z - 1.0e-6)
    nimp = (Zeff - 1.0) / (Z * (Z - 1.0) + 1.0e-12)
    return jnp.clip(1.0 - Z * nimp, 0.0, 1.0)


def _local_inverse_aspect_ratio(rho, machine, eq=None, sim=None):
    """Return local r/R from equilibrium surfaces when available."""
    rho = jnp.asarray(rho)
    if eq is None or not hasattr(eq, "R"):
        return machine.a * rho / machine.R0
    R = jnp.asarray(eq.R)
    Rmax = jnp.max(R, axis=1)
    Rmin = jnp.min(R, axis=1)
    r_minor = 0.5 * (Rmax - Rmin)
    Rmaj = 0.5 * (Rmax + Rmin)
    return r_minor / _maybe_lower(Rmaj, 0.1, sim, 1.0e-3)


@dataclass(frozen=True)
class FusionSurrogatesStatus:
    available: bool
    message: str
    module_repr: str = ""


def normalized_log_gradient(y, rho, R0, a, floor=1e-8, sim=None):
    """Return R/L_y = -R0 d ln y / dr."""
    grad = grid_radial_gradient(y, rho, a)
    return -R0 * grad / _maybe_lower(y, floor, sim, 1e-4)


def magnetic_shear_from_q(q, rho, sim=None):
    """Return s_hat = rho/q dq/drho."""
    # gradient wrt rho, so set a=1.0
    dq = grid_radial_gradient(q, rho, 1.0)
    return rho * dq / _maybe_lower(q, 1e-6, sim, 1e-6)


def collisionality_proxy(Te_keV, ne20, q, rho, machine, sim=None, eq=None):
    """Crude logarithmic ion-electron normalized collisionality proxy.

    QLKNN expects LogNuStar. This is not a full QuaLiKiz preprocessing
    implementation, but it gives the correct qualitative scaling:
      nu* increases with density, q, Zeff, R0 and decreases with Te^2 eps^(3/2).
    """
    eps = _maybe_lower(_local_inverse_aspect_ratio(rho, machine, eq=eq, sim=sim), 1e-3, sim, 1e-4)
    Te_keV = _maybe_lower(Te_keV, 0.03, sim, 1e-4)
    nu_star = 1e-3 * machine.Zeff * ne20 * _maybe_lower(q, 0.2, sim, 1e-3) * machine.R0 / (
        Te_keV**2 * eps**1.5 + 1e-12
    )
    return jnp.log(_maybe_lower(nu_star, 1e-8, sim, 1e-8))


def build_qlknn_features(rho, Te, Ti, ne20, q, machine, sim=None, eq=None):
    """Construct QLKNN_7_11 inputs in the exact model order.

    Returns:
      features: dict of named 1D arrays
      x: [nr, 10] array ordered as QLKNN_INPUT_NAMES
    """
    Ate = normalized_log_gradient(Te, rho, machine.R0, machine.a, floor=0.03, sim=sim)
    Ati = normalized_log_gradient(Ti, rho, machine.R0, machine.a, floor=0.03, sim=sim)
    Ane = normalized_log_gradient(ne20, rho, machine.R0, machine.a, floor=0.02, sim=sim)
    Ani = Ane
    smag = magnetic_shear_from_q(q, rho, sim=sim)
    x_minor = _local_inverse_aspect_ratio(rho, machine, eq=eq, sim=sim)
    Ti_Te = _maybe_lower(Ti, 0.03, sim, 1e-4) / _maybe_lower(Te, 0.03, sim, 1e-4)
    LogNuStar = collisionality_proxy(Te, ne20, q, rho, machine, sim=sim, eq=eq)
    normni = main_ion_density_fraction(machine) + 0.0 * rho

    features = {
        "Ati": Ati,
        "Ate": Ate,
        "Ane": Ane,
        "Ani": Ani,
        "q": q,
        "smag": smag,
        "x": x_minor,
        "Ti_Te": Ti_Te,
        "LogNuStar": LogNuStar,
        "normni": normni,
    }
    x = jnp.stack([features[name] for name in QLKNN_INPUT_NAMES], axis=-1)
    return features, x


def clip_inputs_to_qlknn_ranges(x, sim=None):
    """Clip inputs to broad QLKNN_7_11 training ranges from model README.

    This avoids pathological extrapolation in the interactive GUI. The ranges
    combine the 11D core and 7D edge datasets.
    """
    mins = jnp.array([
        1.0e-8,   # Ati
        1.0e-8,   # Ate
        -5.0,     # Ane
        -15.0,    # Ani
        0.66,     # q
        -1.0,     # smag
        0.10,     # x
        0.25,     # Ti_Te
        -5.1,     # LogNuStar
        0.5,      # normni
    ], dtype=x.dtype)
    maxs = jnp.array([
        150.0,    # Ati
        150.0,    # Ate
        110.0,    # Ane
        110.0,    # Ani
        30.0,     # q
        40.0,     # smag
        0.95,     # x
        2.5,      # Ti_Te
        0.50,     # LogNuStar
        1.0,      # normni
    ], dtype=x.dtype)
    return _maybe_bound(x, mins, maxs, sim, 1e-3)


def fusion_surrogates_status() -> FusionSurrogatesStatus:
    try:
        import fusion_surrogates  # type: ignore
        from fusion_surrogates.qlknn import qlknn_model  # type: ignore
        return FusionSurrogatesStatus(
            available=True,
            message=(
                "fusion_surrogates import succeeded; "
                "QLKNNModel API is available."
            ),
            module_repr=repr(fusion_surrogates),
        )
    except Exception as exc:
        return FusionSurrogatesStatus(
            available=False,
            message=f"fusion_surrogates QLKNN import failed: {exc}",
        )


@lru_cache(maxsize=1)
def load_fusion_surrogates_qlknn_model():
    """Load DeepMind QLKNN_7_11 default model."""
    from fusion_surrogates.qlknn import qlknn_model  # type: ignore
    return qlknn_model.QLKNNModel.load_default_model()


def try_load_fusion_surrogates_model() -> Tuple[Optional[Any], FusionSurrogatesStatus]:
    """Compatibility wrapper retained for older scripts."""
    status = fusion_surrogates_status()
    if not status.available:
        return None, status
    try:
        model = load_fusion_surrogates_qlknn_model()
        msg = (
            f"Loaded {model.name} via "
            "fusion_surrogates.qlknn.qlknn_model.QLKNNModel.load_default_model(). "
            f"Inputs={model.config.input_names}; targets={model.config.target_names}."
        )
        return model, FusionSurrogatesStatus(True, msg, repr(model))
    except Exception as exc:
        return None, FusionSurrogatesStatus(False, f"QLKNN load failed: {exc}")


def _squeeze_model_profile(value):
    """Return a 1D radial profile from fusion_surrogates output arrays.

    Different fusion_surrogates releases have used slightly different output
    shapes, e.g. [1, nr, 1], [1, nr], or [nr, 1].  The previous adapter assumed
    exactly [1, nr, 1], so a harmless shape change could throw inside the
    transport call and silently fall back to qlknn_like.  This helper keeps the
    real backend active when the model loaded correctly.
    """
    arr = jnp.asarray(value)
    if arr.ndim >= 1 and arr.shape[0] == 1:
        arr = arr[0]
    return jnp.squeeze(arr)


def predict_fusion_surrogates_fluxes(x, sim=None):
    """Predict QLKNN fluxes using the real fusion_surrogates model.

    Args:
      x: array with shape [nr, 10] in QLKNN_INPUT_NAMES order.

    Returns:
      dict mapping flux names to 1D arrays of length nr.
    """
    model = load_fusion_surrogates_qlknn_model()
    x = clip_inputs_to_qlknn_ranges(x, sim)
    # The model supports arbitrary leading batch dimensions. Use [1, nr, 10]
    # to match the package tests and then remove singleton batch/channel axes.
    fluxes = model.predict(x[None, ...])
    out = {}
    for name, value in fluxes.items():
        out[name] = _squeeze_model_profile(value)
    return out



def predict_fusion_surrogates_fluxes_smooth(x, sim=None):
    """Predict QLKNN fluxes with a smooth flux-map nonlinearity.

    DeepMind's QLKNNModel.predict is JAX-native, but its flux-map applies
    hard jnp.clip(target, min=0) to leading flux targets. For smooth AD mode we
    call predict_targets and reconstruct the flux dictionary with softplus-like
    positive parts instead.
    """
    model = load_fusion_surrogates_qlknn_model()
    x = clip_inputs_to_qlknn_ranges(x, sim)
    targets = model.predict_targets(x[None, ...])

    def positive_part(y):
        if sim is not None and getattr(sim, "differentiable_smooth_mode", False):
            return _smooth_lower(y, 0.0, 1.0e-3)
        return jnp.clip(y, min=0.0)

    out = {}
    for flux_name, fmap in model.config.flux_map.items():
        target_name = fmap["target"]
        denominator_name = fmap["denominator"]
        target_idx = model.config.target_names.index(target_name)
        if denominator_name is None:
            flux = positive_part(targets[..., target_idx])
        else:
            denominator_idx = model.config.target_names.index(denominator_name)
            flux = targets[..., target_idx] * positive_part(targets[..., denominator_idx])
        out[flux_name] = _squeeze_model_profile(flux)
    return out


def qlknn_fluxes_to_effective_diffusivities(fluxes, features, sim):
    """Convert QLKNN gyroBohm fluxes to effective diffusivities.

    QLKNN outputs normalized fluxes. For this stage-1 solver we use:

      chi_i ~ (efiITG + efiTEM) / max(Ati, floor)
      chi_e ~ (efeITG + efeTEM + efeETG) / max(Ate, floor)
      Dn    ~ |pfeITG + pfeTEM| / max(|Ane|, floor)

    multiplied by qlknn_gb_to_chi and transport_surrogate_scale.

    This is a pragmatic closure for interactive integrated modeling. A more
    physical version should convert gyroBohm heat/particle fluxes using local
    gyroBohm units and solve a flux-gradient relation.
    """
    floor = sim.gradient_floor
    qi = _maybe_lower(fluxes.get("efiITG", 0.0), 0.0, sim, 1e-3) + _maybe_lower(fluxes.get("efiTEM", 0.0), 0.0, sim, 1e-3)
    qe = (
        _maybe_lower(fluxes.get("efeITG", 0.0), 0.0, sim, 1e-3)
        + _maybe_lower(fluxes.get("efeTEM", 0.0), 0.0, sim, 1e-3)
        + _maybe_lower(fluxes.get("efeETG", 0.0), 0.0, sim, 1e-3)
    )
    pf = fluxes.get("pfeITG", 0.0) + fluxes.get("pfeTEM", 0.0)

    chi_i = qi / _maybe_lower(features["Ati"], floor, sim, 1e-3)
    chi_e = qe / _maybe_lower(features["Ate"], floor, sim, 1e-3)
    Dn = _maybe_abs(pf, sim, 1e-8) / _maybe_lower(_maybe_abs(features["Ane"], sim, 1e-8), floor, sim, 1e-3)

    scale = sim.transport_surrogate_scale * sim.qlknn_gb_to_chi
    chi_i = scale * _maybe_bound(chi_i, sim.chi_clip_min, sim.chi_clip_max, sim, 1e-2)
    chi_e = scale * _maybe_bound(chi_e, sim.chi_clip_min, sim.chi_clip_max, sim, 1e-2)
    Dn = scale * _maybe_bound(Dn, sim.chi_clip_min, sim.chi_clip_max, sim, 1e-2)
    return chi_e, chi_i, Dn


def fusion_surrogates_effective_diffusivities(rho, Te, Ti, ne20, q, machine, sim, eq=None):
    """Real QLKNN_7_11-backed effective diffusivity closure.

    This function uses the actual DeepMind fusion_surrogates JAX model if the
    package is installed. It can be used inside the solver, but the first call
    will load model weights and may be slow. If loading fails and
    sim.fusion_surrogates_fail_mode == "fallback", use bohm_gyrobohm instead.
    """
    features, x = build_qlknn_features(rho, Te, Ti, ne20, q, machine, sim=sim, eq=eq)
    try:
        if getattr(sim, "differentiable_smooth_mode", False):
            fluxes = predict_fusion_surrogates_fluxes_smooth(x, sim=sim)
        else:
            fluxes = predict_fusion_surrogates_fluxes(x, sim=sim)
        return qlknn_fluxes_to_effective_diffusivities(fluxes, features, sim)
    except Exception:
        if sim.fusion_surrogates_fail_mode == "raise":
            raise
        from .transport import bohm_gyrobohm_chi
        return bohm_gyrobohm_chi(rho, Te, Ti, ne20, q, machine, sim, eq=eq)


def qlknn_like_normalized_fluxes(rho, Te, Ti, ne20, q, machine, sim, eq=None):
    """Pure-JAX QLKNN-like normalized fluxes.

    Returns cell-centered outward normalized fluxes:
      qe_norm, qi_norm, gamma_norm

    These are dimensionless toy/QLKNN-like fluxes. The transport solver converts
    them into physical-ish face fluxes with a simple gyroBohm-inspired scale.
    """
    f, _ = build_qlknn_features(rho, Te, Ti, ne20, q, machine, sim=sim, eq=eq)

    drive_i = _maybe_lower(f["Ati"] - (3.0 + 0.35 * f["smag"]), 0.0, sim, 1e-3)
    drive_e = _maybe_lower(f["Ate"] - (3.5 + 0.25 * f["smag"]), 0.0, sim, 1e-3)
    drive_n = _maybe_lower(f["Ane"] - 1.5, 0.0, sim, 1e-3)

    qfac = jnp.clip(f["q"] / 2.5, 0.3, 3.0)
    epsfac = jnp.sqrt(jnp.maximum(f["x"], 1e-3) / 0.3)

    qi = qfac**1.3 * epsfac * drive_i**2 / (1.0 + drive_i**2)
    qe = qfac**1.2 * epsfac * (
        0.55 * drive_e**2 / (1.0 + drive_e**2)
        + 0.45 * drive_n**2 / (1.0 + drive_n**2)
    )
    gamma = 0.35 * qfac * epsfac * drive_n / (1.0 + jnp.abs(drive_n))

    scale = sim.transport_surrogate_scale
    qi = scale * jnp.clip(qi, 0.0, sim.transport_flux_clip)
    qe = scale * jnp.clip(qe, 0.0, sim.transport_flux_clip)
    gamma = scale * jnp.clip(gamma, -sim.transport_flux_clip, sim.transport_flux_clip)
    return qe, qi, gamma


def fusion_surrogates_normalized_fluxes(rho, Te, Ti, ne20, q, machine, sim, eq=None):
    """Real QLKNN normalized flux closure when fusion_surrogates is available.

    Returns cell-centered outward normalized fluxes:
      qe_norm, qi_norm, gamma_norm.

    If the package is unavailable and fail_mode is "fallback", returns the
    pure-JAX QLKNN-like normalized fluxes.
    """
    features, x = build_qlknn_features(rho, Te, Ti, ne20, q, machine, sim=sim, eq=eq)
    try:
        if getattr(sim, "differentiable_smooth_mode", False):
            fluxes = predict_fusion_surrogates_fluxes_smooth(x, sim=sim)
        else:
            fluxes = predict_fusion_surrogates_fluxes(x, sim=sim)
        qi = _maybe_lower(fluxes.get("efiITG", 0.0), 0.0, sim, 1e-3) + _maybe_lower(fluxes.get("efiTEM", 0.0), 0.0, sim, 1e-3)
        qe = (
            jnp.maximum(fluxes.get("efeITG", 0.0), 0.0)
            + jnp.maximum(fluxes.get("efeTEM", 0.0), 0.0)
            + jnp.maximum(fluxes.get("efeETG", 0.0), 0.0)
        )
        gamma = fluxes.get("pfeITG", 0.0) + fluxes.get("pfeTEM", 0.0)
        scale = sim.transport_surrogate_scale
        qi = scale * jnp.clip(qi, 0.0, sim.transport_flux_clip)
        qe = scale * jnp.clip(qe, 0.0, sim.transport_flux_clip)
        gamma = scale * jnp.clip(gamma, -sim.transport_flux_clip, sim.transport_flux_clip)
        return qe, qi, gamma
    except Exception:
        if sim.fusion_surrogates_fail_mode == "raise":
            raise
        return qlknn_like_normalized_fluxes(rho, Te, Ti, ne20, q, machine, sim, eq=eq)
