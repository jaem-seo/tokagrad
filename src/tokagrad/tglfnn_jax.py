"""JAX-native BrainFUSE TGLF-NN adapter.

This module reuses the BrainFUSE/FANN parser used by EPED1-NN and provides a
lightweight tokagrad local-feature adapter for the public
``DIIID_ion_stiffness_60_rotation`` TGLFNN ensemble.  The feature construction is
an integrated-model proxy rather than the full TGLF preprocessing pipeline; the
important property for tokagrad is that the whole path is JAX differentiable.

Reference: [G. M. Staebler et al., Phys. Plasmas 12, 102508 (2005)].
The public NN weights define the surrogate itself; TokaGrad's local-feature
construction and flux-to-diffusivity mapping are reduced adapters, not TGLF.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

import jax
import jax.numpy as jnp

from .eped1nn_jax import (
    BrainfuseJaxEnsemble,
    parse_brainfuse_net,
    brainfuse_ensemble_forward,
)
from .qlknn_adapter import (
    _maybe_abs,
    _maybe_bound,
    _maybe_lower,
)
from .grid import radial_gradient as grid_radial_gradient


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _candidate_roots(model_dir: str | Path) -> list[Path]:
    raw = Path(model_dir).expanduser()
    roots = [raw]
    if not raw.is_absolute():
        project = _project_root()
        roots.extend([Path.cwd() / raw, project / raw, project / "scripts" / raw, project / "src" / raw])
    out: list[Path] = []
    seen: set[str] = set()
    for r in roots:
        key = str(r.resolve(strict=False))
        if key not in seen:
            seen.add(key); out.append(r)
    return out


def _candidate_model_dirs(root: Path, model_name: str) -> list[Path]:
    return [
        root / "tglfnn" / "models" / model_name,
        root / "neural" / "tglfnn" / "models" / model_name,
        root / "external_models" / "neural" / "tglfnn" / "models" / model_name,
        root / "models" / model_name,
        root / model_name,
        root,
    ]


def find_tglfnn_model_dir(model_dir: str | Path, model_name: str = "DIIID_ion_stiffness_60_rotation") -> Path | None:
    for root in _candidate_roots(model_dir):
        for cand in _candidate_model_dirs(root, model_name):
            if cand.exists() and any(cand.glob("brainfuse_*.net")):
                return cand.resolve()
    return None


@lru_cache(maxsize=8)
def load_tglfnn_ensemble(model_dir: str, model_name: str = "DIIID_ion_stiffness_60_rotation", max_nets: int = 0) -> BrainfuseJaxEnsemble:
    model_path = find_tglfnn_model_dir(model_dir, model_name)
    if model_path is None:
        raise FileNotFoundError(f"No TGLFNN brainfuse_*.net files found for {model_name!r} under {model_dir!r}.")
    files = sorted(model_path.glob("brainfuse_*.net"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
    if max_nets and max_nets > 0:
        files = files[:max_nets]
    nets = tuple(parse_brainfuse_net(p) for p in files)
    first = nets[0]
    return BrainfuseJaxEnsemble(str(model_path), nets, first.input_names, first.output_names)


def tglfnn_jax_status(model_dir: str, model_name: str = "DIIID_ion_stiffness_60_rotation", max_nets: int = 0):
    try:
        ens = load_tglfnn_ensemble(str(model_dir), str(model_name), int(max_nets))
        return True, f"Loaded {len(ens.nets)} TGLFNN JAX BrainFUSE nets from {ens.model_path}."
    except Exception as exc:
        return False, str(exc)


def predict_tglfnn_jax(x_phys, model_dir: str, model_name: str = "DIIID_ion_stiffness_60_rotation", max_nets: int = 0):
    ens = load_tglfnn_ensemble(str(model_dir), str(model_name), int(max_nets))
    x_phys = jnp.asarray(x_phys)
    if x_phys.ndim == 1:
        return brainfuse_ensemble_forward(ens, x_phys)
    return jax.vmap(lambda row: brainfuse_ensemble_forward(ens, row))(x_phys)



KEV_TO_J = 1.602176634e-16
MU0 = 4.0e-7 * jnp.pi
M_DEUTERON = 3.3435837724e-27
E_CHARGE = 1.602176634e-19


def _a_over_L(y, rho, sim=None, floor=1e-8):
    """TGLF normalized inverse scale length: -a/y dy/dr = -1/y dy/drho."""
    dy_drho = grid_radial_gradient(y, rho, 1.0)
    return -dy_drho / _maybe_lower(y, floor, sim, 1.0e-4)


def _main_ion_and_impurity_fractions(machine):
    """Return approximate AS_2=n_i/ne and AS_3=n_Z/ne from Zeff.

    TGLF AS_s variables are species density ratios n_s/n_e.  With one main
    singly charged ion and a single impurity charge proxy Z, quasineutrality and
    Zeff give
        n_Z/ne = (Zeff-1)/(Z*(Z-1)),
        n_i/ne = 1 - Z*n_Z/ne = (Z-Zeff)/(Z-1).
    This reproduces the familiar DIII-D-like values AS_2~0.8, AS_3~0.033 for
    Zeff=2 and carbon-like Z=6.
    """
    Z = jnp.maximum(jnp.asarray(getattr(machine, "impurity_Z", 6.0)), 2.0)
    Zeff = jnp.clip(jnp.asarray(machine.Zeff), 1.0, Z - 1.0e-6)
    nimp = (Zeff - 1.0) / (Z * (Z - 1.0) + 1.0e-12)
    nmain = 1.0 - Z * nimp
    return nmain, nimp


def _take_surface_value(A, idx):
    return jnp.take_along_axis(A, idx[:, None], axis=1)[:, 0]


def _tglf_shape_proxies(rho, machine, sim=None, eq=None):
    """Reduced Miller-like local geometry in TGLF normalization.

    When an active equilibrium object is supplied, use the actual flux-surface
    geometry reconstructed by ``equilibrium.py`` instead of machine-only shape
    proxies.  This gives TGLFNN the same local elongation, triangularity,
    Shafranov-shift derivative, and local minor/major radius seen by the rest of
    the solver.  If no equilibrium is available, fall back to the smooth reduced
    Miller-like proxies used previously.
    """
    rho = jnp.asarray(rho)
    rho_pos = _maybe_lower(rho, 1.0e-4, sim, 1.0e-5)
    if eq is not None and hasattr(eq, "R") and hasattr(eq, "Z"):
        R = jnp.asarray(eq.R)
        Z = jnp.asarray(eq.Z)
        Rmax = jnp.max(R, axis=1)
        Rmin = jnp.min(R, axis=1)
        Zmax = jnp.max(Z, axis=1)
        Zmin = jnp.min(Z, axis=1)
        r_minor = _maybe_lower(0.5 * (Rmax - Rmin), 1.0e-4 * machine.a, sim, 1.0e-5)
        Rmaj = 0.5 * (Rmax + Rmin)
        kappa_loc = _maybe_bound(0.5 * (Zmax - Zmin) / r_minor, 0.2, 5.0, sim, 1.0e-3)
        idx_top = jnp.argmax(Z, axis=1)
        R_top = _take_surface_value(R, idx_top)
        delta_loc = _maybe_bound((Rmaj - R_top) / r_minor, -0.9, 0.9, sim, 1.0e-3)
        RMAJ_LOC = Rmaj / _maybe_lower(machine.a, 0.1, sim, 1.0e-3)
        RMIN_LOC = r_minor / _maybe_lower(machine.a, 0.1, sim, 1.0e-3)
        dkappa = grid_radial_gradient(kappa_loc, rho, 1.0)
        s_kappa = rho_pos * dkappa / _maybe_lower(kappa_loc, 0.2, sim, 1.0e-4)
        DRMAJDX_LOC = grid_radial_gradient(RMAJ_LOC, rho, 1.0)
        return RMAJ_LOC, RMIN_LOC, delta_loc, kappa_loc, s_kappa, DRMAJDX_LOC

    tri_pow = float(getattr(sim, "triangularity_profile_power", 1.0)) if sim is not None else 1.0
    kap_pow = float(getattr(sim, "elongation_profile_power", 0.5)) if sim is not None else 0.5
    delta_loc = machine.delta * rho_pos ** tri_pow
    kappa_loc = 1.0 + (machine.kappa - 1.0) * rho_pos ** kap_pow
    dkap_drho = (machine.kappa - 1.0) * kap_pow * rho_pos ** (kap_pow - 1.0)
    s_kappa = rho_pos * dkap_drho / _maybe_lower(kappa_loc, 0.2, sim, 1.0e-4)
    Rmaj_over_a = machine.R0 / _maybe_lower(machine.a, 0.1, sim, 1.0e-3)
    dRmajdx = 0.0 * rho_pos
    return Rmaj_over_a + 0.0 * rho, rho_pos, delta_loc, kappa_loc, s_kappa, dRmajdx


def _tglf_Bunit_profile(rho, machine, sim=None, eq=None):
    """Use flux-surface RMS |B| from the active equilibrium when available."""
    if eq is not None and hasattr(eq, "Bmag") and hasattr(eq, "jac") and hasattr(eq, "R"):
        B2 = jnp.asarray(eq.Bmag) ** 2
        w = jnp.maximum(jnp.asarray(eq.R) * jnp.asarray(eq.jac), 0.0)
        return jnp.sqrt(jnp.sum(B2 * w, axis=1) / (jnp.sum(w, axis=1) + 1.0e-30))
    return jnp.abs(machine.Bt) + 0.0 * jnp.asarray(rho)


def _tglf_collision_frequency_xnue(Te_keV, ne20, machine, sim=None):
    """TGLF XNUE proxy: nu_ei / (c_s/a).

    Uses the standard Spitzer electron-ion collision frequency with ne in cm^-3
    and Te in eV, normalized by c_s/a.  This is closer to TGLF XNUE than the
    QLKNN log-nustar proxy used in the first implementation.
    """
    Te_eV = _maybe_lower(Te_keV, 0.03, sim, 1.0e-4) * 1.0e3
    ne_cm3 = _maybe_lower(ne20, 1.0e-4, sim, 1.0e-6) * 1.0e14
    nu_ei = 2.91e-6 * ne_cm3 * machine.Zeff * machine.lnLambda / (Te_eV ** 1.5 + 1.0e-30)
    Te_J = Te_keV * 1.0e3 * E_CHARGE
    cs = jnp.sqrt(_maybe_lower(Te_J, 1.0e-21, sim, 1.0e-22) / M_DEUTERON)
    return nu_ei * machine.a / (cs + 1.0e-30)


def _gyrobohm_diffusivity_scale(Te_keV, machine, sim=None, Bunit=None):
    """rho_s^2 c_s / a [m^2/s] using deuterium sound gyroradius.

    Avoid the algebraically equivalent ``sqrt(M*T) / eB`` form.  With JAX's
    default float32 dtype, ``M*T`` is O(1e-42) for keV plasmas and can
    underflow to zero.  Computing rho_s as ``c_s / Omega_ci`` keeps all
    intermediate values in a normal range and prevents TGLFNN turbulent
    diffusivities from collapsing to the lower clip/floor.
    """
    Te_J = _maybe_lower(Te_keV, 0.03, sim, 1.0e-4) * 1.0e3 * E_CHARGE
    Braw = jnp.abs(machine.Bt) if Bunit is None else jnp.abs(Bunit)
    B = _maybe_lower(Braw, 0.2, sim, 1.0e-3)
    cs = jnp.sqrt(Te_J / M_DEUTERON)
    omega_ci = E_CHARGE * B / M_DEUTERON
    rho_s = cs / (omega_ci + 1.0e-30)
    return rho_s * rho_s * cs / _maybe_lower(machine.a, 0.1, sim, 1.0e-3)


def _clip_tglfnn_inputs_to_training(features, ens, sim=None):
    if not bool(getattr(sim, "tglfnn_clip_inputs_to_training", True)):
        return features
    nsig = float(getattr(sim, "tglfnn_training_clip_sigma", 5.0))
    ref = ens.nets[0]
    mu = jnp.asarray(ref.scale_mean_in)
    sig = _maybe_lower(jnp.asarray(ref.scale_deviation_in), 1.0e-12, sim, 1.0e-12)
    lo = mu - nsig * sig
    hi = mu + nsig * sig
    return _maybe_bound(features, lo, hi, sim, 1.0e-3)


def _profile_like(value, rho):
    """Broadcast scalar TGLF inputs to the radial profile shape.

    Some TGLF local inputs, such as RMAJ_LOC or rotation defaults, are scalars
    in this reduced model.  JAX ``stack`` requires exactly matching shapes, so
    convert all scalar leaves to ``zeros_like(rho)+value`` before assembling the
    feature matrix.
    """
    arr = jnp.asarray(value)
    if arr.shape == ():
        return jnp.zeros_like(rho) + arr
    return arr + jnp.zeros_like(rho)


def build_tglfnn_features(rho, Te, Ti, ne20, q, machine, sim=None, eq=None):
    """Build TGLFNN inputs closer to the public TGLF convention.

    The public DIII-D rotation TGLFNN model takes the TGLF local variables named
    in each ``brainfuse_*.net`` file.  This adapter now uses TGLF's normalized
    geometry/gradient convention rather than the earlier QLKNN-style proxy:

      * RLTS/RLNS = -a/y dy/dr = -d ln(y)/d rho,
      * RMAJ_LOC = R_maj/a, RMIN_LOC = r/a,
      * Q_PRIME_LOC = q/rho * dq/d rho,
      * P_PRIME_LOC is formed from the normalized pressure gradient,
      * BETAE = 2 mu0 p_e / B_unit^2,
      * AS_2/AS_3 are quasineutrality/Zeff-based species density ratios,
      * XNUE = nu_ei / (c_s/a).

    Quantities that are not present in this reduced 1.5D model, especially
    rotation, are kept as smooth configurable proxies/defaults.
    """
    rho = jnp.asarray(rho); Te = jnp.asarray(Te); Ti = jnp.asarray(Ti); ne20 = jnp.asarray(ne20); q = jnp.asarray(q)
    rho_pos = _maybe_lower(rho, 1.0e-4, sim, 1.0e-5)
    q_abs = _maybe_lower(jnp.abs(q), 0.2, sim, 1.0e-3)

    RMAJ_LOC, RMIN_LOC, DELTA_LOC, KAPPA_LOC, S_KAPPA_LOC, DRMAJDX_LOC = _tglf_shape_proxies(rho, machine, sim, eq=eq)

    RLTS_1 = _a_over_L(Te, rho, sim, floor=0.03)
    RLTS_2 = _a_over_L(Ti, rho, sim, floor=0.03)
    RLNS_1 = _a_over_L(ne20, rho, sim, floor=0.02)
    # Main ion and impurity gradients follow the electron-density shape in this
    # reduced model unless future impurity-density evolution is added.
    RLNS_2 = RLNS_1
    RLNS_3 = RLNS_1

    dq_drho = grid_radial_gradient(q_abs, rho, 1.0)
    Q_PRIME_LOC = q_abs * dq_drho / rho_pos

    ne_m3 = ne20 * 1.0e20
    p_e = ne_m3 * _maybe_lower(Te, 0.03, sim, 1.0e-4) * KEV_TO_J
    p_tot = ne_m3 * (_maybe_lower(Te, 0.03, sim, 1.0e-4) + _maybe_lower(Ti, 0.03, sim, 1.0e-4)) * KEV_TO_J
    Bunit = _maybe_lower(_tglf_Bunit_profile(rho, machine, sim=sim, eq=eq), 0.2, sim, 1.0e-3)
    BETAE = 2.0 * MU0 * p_e / (Bunit * Bunit + 1.0e-30)
    beta_tot = 2.0 * MU0 * p_tot / (Bunit * Bunit + 1.0e-30)
    dbeta_drho = grid_radial_gradient(beta_tot, rho, 1.0)
    # TGLF P_PRIME_LOC = q a^2/(r B_unit^2) dp/dr.  Using normalized beta pressure
    # gives the same local sign/drive in a dimensionless form appropriate for
    # this reduced model.
    P_PRIME_LOC = q_abs * dbeta_drho / rho_pos

    AS_2, AS_3 = _main_ion_and_impurity_fractions(machine)
    AS_2 = AS_2 + 0.0 * rho
    AS_3 = AS_3 + 0.0 * rho
    TAUS_2 = _maybe_lower(Ti, 0.03, sim, 1.0e-4) / _maybe_lower(Te, 0.03, sim, 1.0e-4)
    XNUE = _tglf_collision_frequency_xnue(Te, ne20, machine, sim)

    # Rotation is absent from the current state vector.  Keep configurable smooth
    # defaults so future Vtor evolution can be inserted without changing the API.
    VPAR_1 = getattr(sim, "tglfnn_vpar_1", 0.0) + 0.0 * rho
    VPAR_SHEAR_1 = getattr(sim, "tglfnn_vpar_shear_1", 0.0) + 0.0 * rho
    VEXB_SHEAR = getattr(sim, "tglfnn_vexb_shear", 0.0) + 0.0 * rho

    vals = {
        "AS_2": AS_2,
        "AS_3": AS_3,
        "BETAE": BETAE,
        "DELTA_LOC": DELTA_LOC,
        "DRMAJDX_LOC": DRMAJDX_LOC,
        "KAPPA_LOC": KAPPA_LOC,
        "P_PRIME_LOC": P_PRIME_LOC,
        "Q_LOC": q_abs,
        "Q_PRIME_LOC": Q_PRIME_LOC,
        "RLNS_1": RLNS_1,
        "RLNS_2": RLNS_2,
        "RLNS_3": RLNS_3,
        "RLTS_1": RLTS_1,
        "RLTS_2": RLTS_2,
        "RMAJ_LOC": RMAJ_LOC,
        "RMIN_LOC": RMIN_LOC,
        "S_KAPPA_LOC": S_KAPPA_LOC,
        "TAUS_2": TAUS_2,
        "VEXB_SHEAR": VEXB_SHEAR,
        "VPAR_1": VPAR_1,
        "VPAR_SHEAR_1": VPAR_SHEAR_1,
        "XNUE": XNUE,
        "ZEFF": machine.Zeff + 0.0 * rho,
        "_BUNIT": Bunit,
    }
    if sim is not None:
        ens = load_tglfnn_ensemble(str(getattr(sim, "tglfnn_model_dir", "external_models/neural")), str(getattr(sim, "tglfnn_model_name", "DIIID_ion_stiffness_60_rotation")), int(getattr(sim, "tglfnn_jax_max_nets", 0)))
        names = ens.input_names
    else:
        ens = None
        names = tuple(vals.keys())
    features = jnp.stack([_profile_like(vals.get(name, 0.0), rho) for name in names], axis=-1)
    if ens is not None:
        features = _clip_tglfnn_inputs_to_training(features, ens, sim)
    return features, vals

def tglfnn_jax_effective_diffusivities(rho, Te, Ti, ne20, q, machine, sim, eq=None):
    features, proxy = build_tglfnn_features(rho, Te, Ti, ne20, q, machine, sim=sim, eq=eq)
    try:
        out = predict_tglfnn_jax(
            features,
            getattr(sim, "tglfnn_model_dir", "external_models/neural"),
            getattr(sim, "tglfnn_model_name", "DIIID_ion_stiffness_60_rotation"),
            int(getattr(sim, "tglfnn_jax_max_nets", 0)),
        )
        ens = load_tglfnn_ensemble(
            str(getattr(sim, "tglfnn_model_dir", "external_models/neural")),
            str(getattr(sim, "tglfnn_model_name", "DIIID_ion_stiffness_60_rotation")),
            int(getattr(sim, "tglfnn_jax_max_nets", 0)),
        )
        od = {name: out[:, i] for i, name in enumerate(ens.output_names)}
        qe = _maybe_lower(od.get("OUT_ENERGY_FLUX_1", 0.0), 0.0, sim, 1e-3)
        qi = _maybe_lower(od.get("OUT_ENERGY_FLUX_i", 0.0), 0.0, sim, 1e-3)
        gam = od.get("OUT_PARTICLE_FLUX_1", 0.0)
        floor = getattr(sim, "gradient_floor", 0.5)

        # TGLFNN fluxes are dimensionless gyroBohm-normalized flux-like outputs.
        # Convert them to an effective diffusivity using chi_gB=rho_s^2 c_s/a
        # and the TGLF a/L gradients.  A legacy dimensionless mode can be
        # recovered by setting tglfnn_use_gyrobohm_scale=False.
        use_gb = bool(getattr(sim, "tglfnn_use_gyrobohm_scale", True))
        gb = _gyrobohm_diffusivity_scale(Te, machine, sim, Bunit=proxy.get("_BUNIT", None)) if use_gb else (1.0 + 0.0 * rho)
        chi_e = gb * qe / _maybe_lower(proxy["RLTS_1"], floor, sim, 1e-3)
        chi_i = gb * qi / _maybe_lower(proxy["RLTS_2"], floor, sim, 1e-3)
        Dn = gb * _maybe_abs(gam, sim, 1e-8) / _maybe_lower(_maybe_abs(proxy["RLNS_1"], sim, 1e-8), floor, sim, 1e-3)

        scale = getattr(sim, "transport_surrogate_scale", 1.0) * getattr(sim, "tglfnn_gb_to_chi", 1.0)
        chi_e = scale * _maybe_bound(chi_e, getattr(sim, "chi_clip_min", 0.03), getattr(sim, "chi_clip_max", 30.0), sim, 1e-2)
        chi_i = scale * _maybe_bound(chi_i, getattr(sim, "chi_clip_min", 0.03), getattr(sim, "chi_clip_max", 30.0), sim, 1e-2)
        Dn = scale * _maybe_bound(Dn, getattr(sim, "chi_clip_min", 0.03), getattr(sim, "chi_clip_max", 30.0), sim, 1e-2)
        return chi_e, chi_i, Dn
    except Exception:
        if getattr(sim, "tglfnn_fail_mode", "fallback") == "raise":
            raise
        from .transport import bohm_gyrobohm_chi
        return bohm_gyrobohm_chi(rho, Te, Ti, ne20, q, machine, sim, eq=eq)
