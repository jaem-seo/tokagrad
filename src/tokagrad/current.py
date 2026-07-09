"""Toroidal-current, bootstrap, resistivity, q, and poloidal-flux evolution.

Principal references for the implemented physics:
  [O. Sauter et al., Phys. Plasmas 6, 2834 (1999)] -- conductivity/bootstrap.
  [O. Sauter et al., Phys. Plasmas 9, 5140 (2002)] -- Sauter corrections.
  [O. Sauter, Fusion Eng. Des. 112, 633 (2016)] -- global shape factors.
  [F. Felici, EPFL Thesis 5203 (2011)] -- 1-D poloidal-flux diffusion form.

Circular q estimates, saturated-conductivity profiles, source smoothing, and
axis regularization are explicitly reduced TokaGrad closures, not full
Grad--Shafranov or kinetic solutions.
"""

import jax
import jax.numpy as jnp
from .density import target_nbar20
from .grid import infer_rho_faces, radial_gradient as grid_radial_gradient, cell_widths

MU0 = 4.0e-7 * jnp.pi
KEV_TO_J = 1.602176634e-16

def _safe_tridiagonal_denominator_current(x):
    """Avoid exact-zero pivots in the Thomas algorithm."""
    tiny = jnp.asarray(1.0e-30, dtype=x.dtype)
    return jnp.where(jnp.abs(x) > tiny, x, jnp.where(x >= 0.0, tiny, -tiny))


def solve_tridiagonal_current(lower, diag, upper, rhs):
    """Differentiable O(n) Thomas solve for tridiagonal current operators.

    ``lower`` and ``upper`` have shape ``(n-1,)``; ``diag`` and ``rhs`` have
    shape ``(n,)``.  This local copy avoids importing ``solver.py`` from this
    module, which would create a circular dependency.
    """
    n = diag.shape[0]
    if n == 1:
        return rhs / _safe_tridiagonal_denominator_current(diag)

    upper_full = jnp.concatenate([upper, jnp.zeros((1,), dtype=diag.dtype)])
    d0 = _safe_tridiagonal_denominator_current(diag[0])
    c0 = upper_full[0] / d0
    r0 = rhs[0] / d0

    def forward(carry, i):
        c_prev, r_prev = carry
        denom = _safe_tridiagonal_denominator_current(diag[i] - lower[i - 1] * c_prev)
        c_i = upper_full[i] / denom
        r_i = (rhs[i] - lower[i - 1] * r_prev) / denom
        return (c_i, r_i), (c_i, r_i)

    (_, _), (c_tail, r_tail) = jax.lax.scan(forward, (c0, r0), jnp.arange(1, n))
    cprime = jnp.concatenate([jnp.asarray([c0], dtype=diag.dtype), c_tail])
    rprime = jnp.concatenate([jnp.asarray([r0], dtype=diag.dtype), r_tail])

    def backward(x_next, i_rev):
        i = n - 2 - i_rev
        x_i = rprime[i] - cprime[i] * x_next
        return x_i, x_i

    x_last = rprime[-1]
    _, x_rev_tail = jax.lax.scan(backward, x_last, jnp.arange(n - 1))
    return jnp.concatenate([x_rev_tail[::-1], jnp.asarray([x_last], dtype=diag.dtype)])


def solve_implicit_tridiagonal_current(lower_L, main_L, upper_L, rhs, dt):
    """Solve ``(I - dt*L) x = rhs`` for tridiagonal ``L``."""
    return solve_tridiagonal_current(-dt * lower_L, 1.0 - dt * main_L, -dt * upper_L, rhs)


def is_psi_diffusion_model(model):
    """Return True for the current total-current psi diffusion branch.

    ``new_psi_diffusion`` was the historical name for this branch before the
    legacy cylindrical psi solver was removed.  Keep it as an alias so old
    input files do not silently fall through to no current update.
    """
    name = str(model or "").lower()
    return name in ("psi_diffusion", "new_psi_diffusion")


def _shape_factor(machine, rho):
    """Legacy ITER q-scaling correction for shaped flux surfaces.
    Reference: [ITER Physics Group, Nucl. Fusion 39, 2137 (1999)].
    This is the standard empirical product

      [1 + kappa^2 (1 + 2 delta^2 - 1.2 delta^3)] / 2
      * (1.17 - 0.65 epsilon) / (1 - epsilon^2)^2.

    It is useful as an LCFS/q95 scaling and as a geometry-free fallback.  The
    radial kappa/delta interpolation below is only a reduced local-profile
    heuristic; geometry-aware paths use the explicit flux-surface metrics.
    """
    #eps = machine.a / machine.R0
    #k = machine.kappa
    #d = machine.delta
    eps = rho * machine.a / machine.R0
    k = 1.0 + (machine.kappa - 1.0) * rho ** 0.5
    d = machine.delta * rho ** 1.0
    elong_tri = 0.5 * (1.0 + k**2 * (1.0 + 2.0 * d**2 - 1.2 * d**3))
    aspect = (1.17 - 0.65 * eps) / ((1.0 - eps**2) ** 2 + 1e-12)
    return elong_tri * aspect

def shape_factor(machine, rho):
    """Reduced q-shaping correction.

    Geometry reference: [O. Sauter, Fusion Eng. Des. 112, 633 (2016)].
    The radial interpolation of edge shape is a TokaGrad approximation.
    """
    eps = rho * machine.a / machine.R0
    k = 1.0 + (machine.kappa - 1.0) * rho ** 0.5 
    d = machine.delta * rho ** 1.0

    elong = 1.0 + 1.2 * (k - 1.0) + 0.56 * (k - 1.0)**2
    tri = 1.0 + 0.09 * d + 0.16 * d**2
    aspect = (1.0 + 0.45 * d * eps) / (1.0 - 0.74 * eps)
    square = 1.0
    return elong * tri * aspect * square

def area_weights(rho, a, kappa):
    """Analytic poloidal cross-section cell areas [m^2].

    This is the legacy large-aspect-ratio/Miller approximation.  When an
    Equilibrium object is available, prefer ``poloidal_area_weights(..., eq)``
    so current integrals use the same prescribed or reduced flux-surface
    geometry as volume integrals.
    """
    r_faces = a * infer_rho_faces(rho)
    return jnp.pi * kappa * (r_faces[1:]**2 - r_faces[:-1]**2)


def poloidal_area_weights_from_jac(rho, theta, jac):
    """Poloidal cross-section cell-area weights from flux-surface geometry.

    ``jac = |d(R,Z)/d(rho,theta)|`` gives ``dA/drho = ∫ J dtheta``.  The
    returned weights are finite-volume cell areas, i.e. ``dA/drho * Δrho``.
    These are the correct weights for toroidal-current integrals
    ``I = ∫ j_phi dA``.
    """
    dtheta = theta[1] - theta[0] if theta.size > 1 else 2.0 * jnp.pi
    dA_drho = jnp.sum(jnp.maximum(jac, 0.0), axis=1) * dtheta
    drho = cell_widths(rho, 1.0)
    return jnp.maximum(dA_drho * drho, 1.0e-18)


def poloidal_area_weights(rho, machine, eq=None):
    """Current-integration cell areas, using equilibrium geometry if present."""
    if eq is not None and hasattr(eq, "jac") and hasattr(eq, "theta"):
        try:
            return poloidal_area_weights_from_jac(rho, eq.theta, eq.jac)
        except Exception:
            pass
    return area_weights(rho, machine.a, machine.kappa)


def normalize_current_profile(rho, j_shape, Ip, a, kappa, area=None):
    w = area_weights(rho, a, kappa) if area is None else area
    total = jnp.sum(j_shape * w) + 1e-12
    return j_shape * Ip / total

def enclosed_current(rho, j_A_m2, a, kappa):
    """Current enclosed at cell centers with axis-consistent integration."""
    r_c = a * rho
    r_faces = a * infer_rho_faces(rho)
    full_cell_area = jnp.pi * kappa * (r_faces[1:]**2 - r_faces[:-1]**2)
    prev_full = jnp.concatenate([
        jnp.zeros(1, dtype=rho.dtype),
        jnp.cumsum(j_A_m2[:-1] * full_cell_area[:-1]),
    ])
    partial_area = jnp.pi * kappa * jnp.maximum(r_c**2 - r_faces[:-1]**2, 0.0)
    return prev_full + j_A_m2 * partial_area

def total_current(rho, j_A_m2, machine, eq=None):
    return jnp.sum(j_A_m2 * poloidal_area_weights(rho, machine, eq=eq))


def enclosed_current_faces(rho, j_A_m2, machine, eq=None):
    """Finite-volume enclosed current on radial faces from cell j.

    Length is nr+1 with I(0)=0 and I(1)=total current.  If ``eq`` is supplied,
    the face current is based on the equilibrium poloidal cross-section cell
    areas rather than the analytic elliptical-area approximation.
    """
    area = poloidal_area_weights(rho, machine, eq=eq)
    return jnp.concatenate([
        jnp.zeros(1, dtype=j_A_m2.dtype),
        jnp.cumsum(j_A_m2 * area),
    ])


def _center_partial_area_fraction(rho):
    """Area-coordinate fraction from each left face to the cell center.

    For a cell-centered rho grid the first cell center is at rho=Δrho/2.
    The enclosed poloidal area near the magnetic axis scales as rho**2, not
    linearly in rho.  Using a fixed 0.5 fraction therefore overestimates the
    magnetic-axis enclosed current by about a factor of two and creates an
    artificial low-q notch in the first one or two cells.  The rho**2 fraction
    is exact for circular/Miller-like nested surfaces near the axis and is a
    robust finite-volume proxy for shaped surfaces.
    """
    rho_f = infer_rho_faces(rho)
    num = rho**2 - rho_f[:-1]**2
    den = rho_f[1:]**2 - rho_f[:-1]**2
    return jnp.clip(num / (den + 1.0e-30), 0.0, 1.0)


def enclosed_current_centers(rho, j_A_m2, machine, eq=None):
    """Approximate enclosed current at cell centers using geometry weights.

    The partial-cell contribution is evaluated in area coordinate rather than
    by blindly taking half of the cell current.  This is important on the first
    cell because A(<rho) ~ rho^2 near the magnetic axis; the old 0.5 factor made
    I_enc(rho_0) too large and pushed q(rho_0) down artificially.
    """
    area = poloidal_area_weights(rho, machine, eq=eq)
    I_faces = enclosed_current_faces(rho, j_A_m2, machine, eq=eq)
    frac = _center_partial_area_fraction(rho)
    return I_faces[:-1] + frac * j_A_m2 * area


def current_density_from_enclosed_faces(rho, I_faces, machine, eq=None):
    """Cell-centered current density from face enclosed-current profile."""
    area = poloidal_area_weights(rho, machine, eq=eq)
    return (I_faces[1:] - I_faces[:-1]) / (area + 1e-12)


def _miller_lcfs_boundary(theta, machine):
    """Reduced fixed-boundary LCFS curve at rho=1.

    This is used only for boundary scalar estimates such as q_edge when the
    transport grid is cell-centered and rho[-1] is inside the LCFS.  The
    Shafranov shift used by the reduced equilibrium vanishes at the fixed
    boundary, so the LCFS is just the prescribed Miller boundary.
    """
    th = theta
    angle = th + machine.delta * jnp.sin(th)
    Rb = machine.R0 + machine.a * jnp.cos(angle)
    Zb = machine.kappa * machine.a * jnp.sin(th)
    return Rb, Zb


def _periodic_boundary_arc_integrals(Rb, Zb):
    """Return geometric arc integrals on a closed R-Z boundary.

    The uniform-Bp edge-q estimate uses
        Lp = ∮ dl,   G = ∮ dl/R^2.
    For a circular large-aspect-ratio plasma this reduces to the familiar
    q_a = 2π a^2 B_t / (μ0 R0 I_p).
    """
    Rn = jnp.roll(Rb, -1)
    Zn = jnp.roll(Zb, -1)
    dR = Rn - Rb
    dZ = Zn - Zb
    dl = jnp.sqrt(dR**2 + dZ**2 + 1.0e-30)
    Rmid = jnp.maximum(0.5 * (Rb + Rn), 1.0e-6)
    Lp = jnp.sum(dl)
    int_dl_over_R2 = jnp.sum(dl / (Rmid**2 + 1.0e-12))
    return Lp, int_dl_over_R2


def edge_q_from_boundary_geometry(machine, I_total_A, eq=None, sim=None):
    """Estimate q at the LCFS from boundary geometry and total current.

    This is intended for diagnostics/constraints when the transport grid is
    cell-centered.  In that case q[-1] is the value at the last cell center,
    not at rho=1, which can be noticeably inside the LCFS for small nr.

    The estimate assumes an approximately uniform poloidal field on the LCFS,
    Bp_edge ≈ μ0 I_p / Lp, and computes

        q_edge ≈ F_edge Lp/(2π μ0 I_p) ∮ dl/R²,

    where F=R B_phi.  The circular limit recovers the standard large-aspect
    ratio expression.  For geqdsk_prescribed, use the outer prescribed surface;
    for reduced_fixed_boundary, use the fixed Miller boundary at rho=1 rather
    than the last cell-center surface.
    """
    theta = None
    if eq is not None and hasattr(eq, "theta"):
        theta = eq.theta
    if theta is None:
        theta = jnp.linspace(0.0, 2.0 * jnp.pi, 64, endpoint=False)

    model = getattr(sim, "equilibrium_model", "") if sim is not None else ""
    if eq is not None and model == "geqdsk_prescribed":
        Rb, Zb = eq.rbbbs, eq.zbbbs
    else:
        Rb, Zb = _miller_lcfs_boundary(theta, machine)

    Lp, int_dl_over_R2 = _periodic_boundary_arc_integrals(Rb, Zb)
    if eq is not None and hasattr(eq, "F"):
        F_edge = jnp.ravel(eq.F)[-1]
    else:
        F_edge = machine.R0 * machine.Bt
    I = jnp.maximum(jnp.abs(I_total_A), 1.0)
    q_edge = jnp.abs(F_edge) * Lp * int_dl_over_R2 / (2.0 * jnp.pi * MU0 * I + 1.0e-12)
    return jnp.clip(q_edge, 0.05, 20.0)


def _as_rho_profile(value, rho):
    """Broadcast a scalar/profile-like value to the rho grid."""
    arr = jnp.asarray(value, dtype=rho.dtype)
    return arr + jnp.zeros_like(rho)


def q_from_equilibrium_or_profile(rho, eq=None, q=None, fallback=None, lo=0.05, hi=20.0):
    """Return the active q profile, preferring an Equilibrium object's q.

    The main simulation path constructs an Equilibrium every step.  Any physics
    closure that needs q should use ``eq.q`` rather than reconstructing a
    circular/cylindrical estimate from current.  The fallback is kept only for
    initialization or legacy helper calls where no Equilibrium exists yet.
    """
    if q is not None:
        q_use = _as_rho_profile(q, rho)
    elif eq is not None and hasattr(eq, "q"):
        q_use = _as_rho_profile(eq.q, rho)
    elif fallback is not None:
        q_use = _as_rho_profile(fallback, rho)
    else:
        # Last-resort legacy fallback for bootstrap/current helpers called before
        # an Equilibrium is available.  Ordinary simulation/diagnostics paths
        # should pass eq so this branch is not used.
        q_use = jnp.ones_like(rho) * 3.0
    return jnp.clip(jnp.abs(q_use), lo, hi)


def _q_profile_from_current(rho, j_A_m2, machine, sim, eq=None):
    """Reconstruct q with the calibrated circular-plus-shaping approximation.

    The reduced metrics are transport/current-diffusion proxies rather than a
    self-consistent Grad--Shafranov equilibrium.  Keep q on the established
    ITER shape-factor convention; prescribed GEQDSK q is still preferred by
    ``q_profile`` when available.
    """
    Ienc = enclosed_current_centers(rho, j_A_m2, machine, eq=eq)
    has_flux_metrics = eq is not None and all(
        hasattr(eq, name) for name in ("F", "Phi_b", "g2", "g3")
    )
    if has_flux_metrics and not getattr(sim, "reduced_geometry_metrics", True):
        S = _ampere_cell_factor(rho, eq)
        Phi_b = jnp.abs(jnp.asarray(eq.Phi_b, dtype=rho.dtype))
        q_metric = 2.0 * Phi_b * rho * S / _signed_floor(Ienc, 1.0)
        return jnp.maximum(jnp.abs(q_metric), 0.05)
    
    r = machine.a * rho
    denom = MU0 * machine.R0 * jnp.maximum(jnp.abs(Ienc), 1.0)
    q_circ = 2.0 * jnp.pi * r**2 * jnp.abs(machine.Bt) / (denom + 1.0e-30)
    return jnp.maximum(shape_factor(machine, rho) * q_circ, 0.05)


def q_profile(rho, j_A_m2, machine, sim, eq=None):
    """Return q on the transport grid, preferring ``eq.q`` when available.

    If ``eq.q`` exists it remains the source of truth.  Otherwise use the
    calibrated circular-plus-shape-factor reconstruction.
    """
    if eq is not None and hasattr(eq, "q"):
        return q_from_equilibrium_or_profile(rho, eq=eq)
    return _q_profile_from_current(rho, j_A_m2, machine, sim, eq=eq)


def is_face_psi_state(rho, psi):
    return psi.size == rho.size + 1

def face_radii(rho, machine):
    nr = rho.size
    return machine.a * jnp.arange(nr + 1, dtype=rho.dtype) / nr


def _signed_floor(x, floor):
    """Sign-preserving floor for quantities that may carry convention sign."""
    x = jnp.asarray(x)
    sign = jnp.where(x < 0.0, -1.0, 1.0)
    return jnp.where(jnp.abs(x) < floor, sign * floor, x)


def _binomial_smooth_cell_profile(y, passes=1):
    """Light [1,2,1]/4 smoothing on a cell-centered profile.

    This is used only as a numerical roughness damper for explicit source
    profiles.  Boundary values are reflected, so no artificial edge sink/source
    is introduced by a zero ghost value.
    """
    out = y
    for _ in range(max(0, int(passes))):
        yp = jnp.concatenate([out[:1], out, out[-1:]])
        out = 0.25 * yp[:-2] + 0.50 * yp[1:-1] + 0.25 * yp[2:]
    return out


def _binomial_smooth_enclosed_current_faces(I_faces, passes=1):
    """Smooth enclosed-current faces while preserving I(0) and I(LCFS)."""
    out = I_faces
    for _ in range(max(0, int(passes))):
        if out.size <= 2:
            return out
        interior = 0.25 * out[:-2] + 0.50 * out[1:-1] + 0.25 * out[2:]
        out = jnp.concatenate([out[:1], interior, out[-1:]])
    return out



def _axis_regularize_enclosed_current_faces(I_faces, rho, n_faces=1):
    """Enforce smooth magnetic-axis behavior I(rho)=a*rho**2+b*rho**4+... .

    The psi solve can leave a small persistent first-cell notch when
    psi is stored on faces including the magnetic axis.  The physical toroidal
    current density is an even function of rho near the axis, hence the enclosed
    current must be smooth in x=rho**2.  This conservative local repair replaces
    only the first few inner enclosed-current faces by a quadratic-in-x
    extrapolation from the neighboring interior faces.  I(0) and I(LCFS) are
    preserved exactly.
    """
    n = int(max(0, n_faces))
    if n <= 0 or I_faces.size < 5:
        return I_faces
    rho_f = infer_rho_faces(rho)
    x = rho_f * rho_f
    # Fit I/x = a + b*x using robust interior faces.  Face 0 is exactly zero;
    # face 1 is the potentially contaminated first annulus and is not used.
    i0 = 2
    i1 = min(I_faces.size - 1, 5)
    idx = jnp.arange(i0, i1)
    xi = x[idx]
    yi = I_faces[idx] / jnp.maximum(xi, 1.0e-30)
    # Closed-form least-squares fit to y = a + b*x.
    m = jnp.asarray(idx.size, dtype=I_faces.dtype)
    sx = jnp.sum(xi)
    sy = jnp.sum(yi)
    sxx = jnp.sum(xi * xi)
    sxy = jnp.sum(xi * yi)
    det = jnp.maximum(m * sxx - sx * sx, 1.0e-30)
    a0 = (sxx * sy - sx * sxy) / det
    b0 = (m * sxy - sx * sy) / det
    out = I_faces.at[0].set(0.0)
    max_face = min(n + 1, I_faces.size - 1)
    upper = 0.999 * jnp.maximum(I_faces[max_face], 0.0)
    for k in range(1, max_face):
        pred = x[k] * (a0 + b0 * x[k])
        # Do not allow the local extrapolation to violate monotonic enclosed
        # current between the axis and the first untouched fitting face.
        pred = jnp.clip(pred, 0.0, upper)
        out = out.at[k].set(pred)
    return out

def _maybe_smooth_new_psi_current(rho, psi, psi_edge, machine, sim, eq):
    """Optional conservative post-step roughness filter for psi_diffusion.

    The filter acts on the enclosed-current face profile, not directly on j.
    Therefore the magnetic-axis and LCFS enclosed currents are preserved exactly.
    It is disabled by default and intended only to damp grid-scale noise in long
    physical-resistivity runs.
    """
    passes = int(getattr(sim, "current_enclosed_smoothing_passes", 0))
    axis_faces = int(getattr(sim, "current_axis_regularization_faces", 2))
    if eq is None or (passes <= 0 and axis_faces <= 0):
        return psi, psi_edge
    I_faces = enclosed_current_faces_from_psi(
        rho, psi, machine, psi_edge=psi_edge, eq=eq
    )
    I_smooth = _binomial_smooth_enclosed_current_faces(I_faces, passes=passes)
    I_smooth = _axis_regularize_enclosed_current_faces(
        I_smooth, rho, n_faces=axis_faces
    )
    j_smooth = current_density_from_enclosed_faces(rho, I_smooth, machine, eq=eq)
    psi_smooth = psi_from_current_density(
        rho, j_smooth, machine, psi_edge=psi_edge, sim=sim, eq=eq
    )
    if is_face_psi_state(rho, psi_smooth):
        psi_smooth = psi_smooth.at[-1].set(psi_edge)
    return psi_smooth, psi_edge



def _regularized_current_from_psi_for_components(rho, psi, psi_edge, machine, sim, eq):
    """Current reconstruction with optional conservative diagnostics filter.

    ``psi`` is smoother than its reconstructed current density: the
    current is obtained by first forming an enclosed-current face profile
    from psi gradients and then differencing that profile over annular areas.
    This is effectively a second radial derivative of psi, so harmless
    grid-scale components in psi can appear as visible cell-to-cell jitter in j.

    For diagnostics and slowly coupled source terms, smooth the cumulative
    enclosed-current profile I(rho) rather than j itself.  This preserves the
    exact axis and LCFS currents and avoids changing the net plasma current.
    """
    if eq is None or sim is None:
        return current_from_psi(rho, psi, machine, psi_edge=psi_edge, sim=sim, eq=eq)
    passes = int(getattr(sim, "current_diagnostic_smoothing_passes", 1))
    axis_faces = int(getattr(sim, "current_axis_regularization_faces", 2))
    if passes <= 0 and axis_faces <= 0:
        return current_from_psi(rho, psi, machine, psi_edge=psi_edge, sim=sim, eq=eq)
    I_faces = enclosed_current_faces_from_psi(
        rho, psi, machine, psi_edge=psi_edge, eq=eq
    )
    I_faces = _binomial_smooth_enclosed_current_faces(I_faces, passes=passes)
    I_faces = _axis_regularize_enclosed_current_faces(I_faces, rho, n_faces=axis_faces)
    return current_density_from_enclosed_faces(rho, I_faces, machine, eq=eq)

def _ampere_cell_factor(rho, eq):
    """Return the factor mapping psi gradient to enclosed current.

    The face plasma-current profile can be reconstructed as

        I_face = dpsi/drho_norm * (g2*g3/rho_norm)_face * F_face
                 / (16*pi^3*mu0*Phi_b).

    This helper returns the factor at cell centers. Face-grid states interpolate
    it to the outer current faces with ``_ampere_outer_face_factor``. The sign convention:
    for positive ``F`` and positive ``Ip``, psi increases outward.
    """
    rho_safe = jnp.maximum(rho, 1.0e-4)
    F = _signed_floor(_as_rho_profile(eq.F, rho), 1.0e-8)
    Phi_b = jnp.maximum(jnp.asarray(eq.Phi_b, dtype=rho.dtype), 1.0e-12)
    if hasattr(eq, "g2g3_over_rho"):
        g2g3_over_rho = jnp.maximum(
            _as_rho_profile(eq.g2g3_over_rho, rho), 1.0e-30
        )
    else:
        g2g3_over_rho = jnp.maximum(eq.g2 * eq.g3 / rho_safe, 1.0e-30)
    return F * g2g3_over_rho / (16.0 * jnp.pi**3 * MU0 * Phi_b + 1.0e-30)


def _ampere_face_factor(rho, eq):
    """Face-centered enclosed-current factor for cell-centered psi."""
    S_cell = _ampere_cell_factor(rho, eq)
    return _face_average_current(S_cell)


def _ampere_outer_face_factor(rho, eq):
    """Ampere factor on faces 1..nr for a face-grid psi state.

    A difference of two adjacent face values of psi is a derivative over the
    intervening cell.  In the face-grid current representation, however, that
    backward difference is used to impose the cumulative current at the outer
    face of the cell.  The metric factor must therefore be evaluated at that
    same outer face.  The old implementation multiplied by a cell-centred
    factor.  Since the factor is proportional to rho near the magnetic axis,
    this made the first value too small by nearly a factor of two and turned
    the radial psi integral into a biased right-endpoint sum.

    Linear interpolation/extrapolation is exact for the regular near-axis
    behaviour S~rho and keeps this map an exact inverse of
    ``face_psi_from_current_density`` on the discrete grid.
    """
    S_cell = _ampere_cell_factor(rho, eq)
    if S_cell.size == 1:
        return S_cell
    interior = 0.5 * (S_cell[:-1] + S_cell[1:])
    edge = S_cell[-1] + 0.5 * (S_cell[-1] - S_cell[-2])
    return jnp.concatenate([interior, edge[None]])


def enclosed_current_faces_from_psi(rho, psi, machine, psi_edge=0.0, eq=None):
    """Finite-volume enclosed toroidal current from poloidal flux.

    Returns ``I_faces`` with length ``nr+1`` and ``I_faces[0]=0``.  The formula
    is the current-integrated counterpart of the operator

        d/drho[(g2*g3/rho) dpsi/drho].
    """
    if eq is None:
        # Legacy cylindrical fallback; kept so old callers without an
        # Equilibrium still work.  Handle both face-grid and cell-grid psi
        # directly here.  Do not call current_from_psi(), because that calls
        # this function and would recurse for cell-grid psi states.
        rho_f = infer_rho_faces(rho)
        rf = machine.a * rho_f
        if is_face_psi_state(rho, psi):
            dr = machine.a * jnp.maximum(rho_f[1:] - rho_f[:-1], 1.0e-12)
            dpsi_dr_cell = (psi[1:] - psi[:-1]) / dr
            I_outer = -2.0 * jnp.pi * rf[1:] * dpsi_dr_cell / (MU0 * machine.R0)
            return jnp.concatenate([jnp.zeros(1, dtype=rho.dtype), I_outer])

        psi_edge_arr = jnp.asarray(psi_edge, dtype=psi.dtype)
        grad = jnp.zeros(rho.size + 1, dtype=psi.dtype)
        if rho.size > 1:
            dr_mid = machine.a * jnp.maximum(rho[1:] - rho[:-1], 1.0e-12)
            grad = grad.at[1:-1].set((psi[1:] - psi[:-1]) / dr_mid)
        grad = grad.at[0].set(0.0)
        dr_edge = machine.a * jnp.maximum(rho_f[-1] - rho[-1], 1.0e-12)
        grad = grad.at[-1].set((psi_edge_arr - psi[-1]) / dr_edge)
        return -2.0 * jnp.pi * rf * grad / (MU0 * machine.R0)

    if is_face_psi_state(rho, psi):
        rho_f = infer_rho_faces(rho)
        drho = jnp.maximum(rho_f[1:] - rho_f[:-1], 1.0e-12)
        grad_cell = (psi[1:] - psi[:-1]) / drho
        S_outer = _ampere_outer_face_factor(rho, eq)
        I_outer = S_outer * grad_cell
        return jnp.concatenate([jnp.zeros(1, dtype=rho.dtype), I_outer])

    rho_f = infer_rho_faces(rho)
    S_face = _ampere_face_factor(rho, eq)
    psi_edge_arr = jnp.asarray(psi_edge, dtype=psi.dtype)
    grad = jnp.zeros(rho.size + 1, dtype=psi.dtype)
    if rho.size > 1:
        grad = grad.at[1:-1].set((psi[1:] - psi[:-1]) / jnp.maximum(rho[1:] - rho[:-1], 1.0e-12))
    grad = grad.at[0].set(0.0)
    grad = grad.at[-1].set((psi_edge_arr - psi[-1]) / jnp.maximum(rho_f[-1] - rho[-1], 1.0e-12))
    return S_face * grad


def current_from_psi(rho, psi, machine, psi_edge=0.0, sim=None, eq=None):
    """Equivalent toroidal current density from psi.

    The poloidal flux equation naturally reconstructs the flux-surface projected
    current ``<B.j>``.  For the rest of TokaGrad we convert it to a cell-averaged
    toroidal-current-density proxy by differencing the corresponding enclosed
    current and dividing by the actual poloidal cross-section cell area from the
    equilibrium Jacobian.  Consequently ``total_current(..., eq)`` of the
    returned profile equals the LCFS enclosed current from the Neumann relation.
    """
    I_faces = enclosed_current_faces_from_psi(
        rho, psi, machine, psi_edge=psi_edge, eq=eq
    )
    return current_density_from_enclosed_faces(rho, I_faces, machine, eq=eq)

def _legacy_face_psi_from_current_density(rho, j_A_m2, machine, psi_edge=0.0, eq=None):
    """Large-aspect-ratio fallback psi inversion for calls without geometry."""
    nr = rho.size
    dr = machine.a / nr
    rf = face_radii(rho, machine)
    I_faces = enclosed_current_faces(rho, j_A_m2, machine, eq=eq)
    # Legacy cylindrical Ampere relation, used only when no Equilibrium object
    # is available.  The active psi-diffusion branch should normally pass eq.
    dpsi_dr_cell = -MU0 * machine.R0 * I_faces[1:] / (
        2.0 * jnp.pi * jnp.maximum(rf[1:], 1.0e-6)
    )
    rev_increments = dpsi_dr_cell[::-1] * dr
    psi_rev = jnp.asarray(psi_edge, dtype=rho.dtype) - jnp.cumsum(rev_increments)
    psi_inner_to_outer = psi_rev[::-1]
    return jnp.concatenate([psi_inner_to_outer, jnp.asarray(psi_edge, dtype=rho.dtype)[None]])


def face_psi_from_current_density(rho, j_A_m2, machine, psi_edge=0.0, eq=None):
    """Construct face-grid psi from a cell toroidal-current profile.

    When an Equilibrium object is supplied this is the exact inverse of
    ``enclosed_current_faces_from_psi`` using the flux-coordinate Ampere factor
    ``(g2*g3/rho) F /(16*pi^3*mu0*Phi_b)``.  If no equilibrium is available,
    fall back to the legacy cylindrical relation.
    """
    if eq is None:
        return _legacy_face_psi_from_current_density(
            rho, j_A_m2, machine, psi_edge=psi_edge, eq=eq
        )
    rho_f = infer_rho_faces(rho)
    drho = jnp.maximum(rho_f[1:] - rho_f[:-1], 1.0e-12)
    I_faces = enclosed_current_faces(rho, j_A_m2, machine, eq=eq)
    # Preserve the sign convention carried by F/Phi_b.  For ITER GEQDSK inputs
    # F can be negative; using jnp.maximum(S, eps) would flip a negative factor
    # to +eps and create |psi|~1e37, followed by inf currents/NaNs.
    S_outer = _signed_floor(_ampere_outer_face_factor(rho, eq), 1.0e-30)
    grad_cell = I_faces[1:] / S_outer
    rev_increments = (grad_cell * drho)[::-1]
    psi_rev = jnp.asarray(psi_edge, dtype=rho.dtype) - jnp.cumsum(rev_increments)
    psi_inner_to_outer = psi_rev[::-1]
    return jnp.concatenate([psi_inner_to_outer, jnp.asarray(psi_edge, dtype=rho.dtype)[None]])


def psi_from_current_density(rho, j_A_m2, machine, psi_edge=0.0, sim=None, eq=None):
    """Invert the psi->current relation in the active psi grid."""
    psi_face = face_psi_from_current_density(
        rho, j_A_m2, machine, psi_edge=psi_edge, eq=eq
    )
    if sim is not None and getattr(sim, "psi_state_grid", "face") == "face":
        return psi_face
    # Cell-centered compatibility: use face averages while keeping psi_edge as
    # the separate LCFS gauge variable.  The face grid is preferred for the
    # current density model because the Neumann Ip condition is exact there.
    return 0.5 * (psi_face[:-1] + psi_face[1:])


def ensure_psi_state_grid(rho, psi_ind, machine, psi_edge=0.0, sim=None, eq=None):
    """Return ``psi_ind`` with the length required by ``sim.psi_state_grid``.

    The plasma state stores temperature/density on cell centers, but psi may be
    stored either on cell centers (length ``nr``) or faces (length ``nr+1``).
    JAX loop carries require a fixed shape, so initial states, cached states,
    and externally supplied states must be normalized before entering a scan.

    Conversion is conservative in the sense that it reconstructs a current
    density from the supplied psi on its native grid and then inverts that
    current profile back to the requested grid.
    """
    psi_ind = jnp.asarray(psi_ind)
    target = getattr(sim, "psi_state_grid", "face") if sim is not None else "cell"
    nr = rho.size
    if target == "face":
        if psi_ind.size == nr + 1:
            return psi_ind
        if psi_ind.size == nr:
            j = current_from_psi(rho, psi_ind, machine, psi_edge=psi_edge, sim=sim, eq=eq)
            return psi_from_current_density(rho, j, machine, psi_edge=psi_edge, sim=sim, eq=eq)
    else:
        if psi_ind.size == nr:
            return psi_ind
        if psi_ind.size == nr + 1:
            j = current_from_psi(rho, psi_ind, machine, psi_edge=psi_edge, sim=sim, eq=eq)
            return psi_from_current_density(rho, j, machine, psi_edge=psi_edge, sim=sim, eq=eq)
    raise ValueError(
        f"psi_ind has incompatible length {psi_ind.size} for nr={nr} and "
        f"psi_state_grid={target!r}; expected {nr} or {nr + 1}."
    )

def initial_current_shape(rho, peaking=2.0, floor=0.0):
    return (1.0 - rho**2) ** peaking + floor

def _smooth_iter_q_profile(rho_like, q0=1.5, q95=3.3, qedge=3.5):
    """Smooth ITER-like q target evaluated at arbitrary normalized radius."""
    x = jnp.clip(rho_like / 0.95, 0.0, 1.0)
    q_mid = q0 + (q95 - q0) * x**2.2
    edge_ramp = jnp.clip((rho_like - 0.95) / 0.05, 0.0, 1.0)
    return q_mid + (qedge - q95) * edge_ramp**2 * (3.0 - 2.0 * edge_ramp)


def iter_hmode_initial_current_density(rho, machine, q0=1.5, q95=3.3, qedge=3.5):
    """ITER-H-mode-like initial inductive current profile from a smooth q guess.

    The old implementation differentiated I_enc(r) at cell centers and then
    divided by r.  Near the magnetic axis this is numerically singular: with
    cell centers r0, r1, a forward derivative gives dI/dr ~ C(r0+r1), which
    after division by r0 overestimates the central cell by roughly
    (1+r1/r0)/2.  On a uniform grid this is about a factor of two.

    Here we construct I_enc on cell faces and compute the cell-average current
    from finite-volume annular differences.  This enforces I_enc(0)=0 exactly
    and removes the artificial one-cell spike at rho=0 while preserving the
    intended q-like peaking shape.
    """
    rho_f = infer_rho_faces(rho)
    r_f = machine.a * rho_f
    q_f = jnp.maximum(_smooth_iter_q_profile(rho_f, q0=q0, q95=q95, qedge=qedge), 0.2)
    F = shape_factor(machine, rho_f)
    I_f = F * 2.0 * jnp.pi * r_f**2 * machine.Bt / (MU0 * machine.R0 * q_f)
    I_f = I_f.at[0].set(0.0)
    area = area_weights(rho, machine.a, machine.kappa)
    j = (I_f[1:] - I_f[:-1]) / (area + 1e-12)
    j = jnp.maximum(j, 1e3)
    return normalize_current_profile(rho, j, machine.Ip, machine.a, machine.kappa)

def initial_psi_from_current_density(rho, machine, j_A_m2, psi_edge=0.0, sim=None, eq=None):
    return psi_from_current_density(rho, j_A_m2, machine, psi_edge=psi_edge, sim=sim, eq=eq)

def initial_psi_iter_hmode(rho, machine, psi_edge=0.0, sim=None):
    return initial_psi_from_current_density(
        rho, machine, iter_hmode_initial_current_density(rho, machine), psi_edge=psi_edge, sim=sim
    )

def initial_psi_from_total_current(rho, machine, psi_edge=0.0, sim=None, eq=None):
    """Initial psi on the configured state grid.

    Historically this helper returned a cell-centred psi because ``sim`` was not
    passed through this path.  With ``psi_state_grid="face"`` the current
    update branches return a face-grid state of length ``nr+1``.  Returning a
    cell-grid initial state of length ``nr`` then makes ``jax.lax.scan`` fail
    because the carry shape changes after the first step.
    """
    j0 = normalize_current_profile(
        rho,
        initial_current_shape(rho),
        machine.Ip,
        machine.a,
        machine.kappa,
    )
    return psi_from_current_density(rho, j0, machine, psi_edge=psi_edge, sim=sim, eq=eq)

# ---------------------------------------------------------------------------
# Sauter-Angioni-Lin-Liu neoclassical coefficients, Eqs. (13)-(18).
# Reference: [O. Sauter et al., Phys. Plasmas 6, 2834 (1999)].
# ---------------------------------------------------------------------------

def sauter_ln_lambda_e(ne_m3, Te_eV):
    ne = jnp.maximum(ne_m3, 1e14)
    Te = jnp.maximum(Te_eV, 10.0)
    return 31.3 - jnp.log(jnp.sqrt(ne) / Te)

def sauter_ln_lambda_ii(ni_m3, Ti_eV, Z):
    ni = jnp.maximum(ni_m3, 1e14)
    Ti = jnp.maximum(Ti_eV, 10.0)
    Z = jnp.maximum(Z, 1.0)
    return 30.0 - jnp.log((Z**3 * jnp.sqrt(ni)) / (Ti**1.5))

def sauter_N_Z(Z):
    Z = jnp.maximum(Z, 1.0)
    return 0.58 + 0.74 / (0.76 + Z)

def sauter_trapped_fraction(rho, machine):
    """Approximate effective trapped-particle fraction f_t.

    Sauter 1999 defines f_t by a flux-surface integral. In this reduced
    fixed-boundary code we use a bounded Miller/circular proxy based on
    eps = a rho / R0.
    """
    eps = jnp.clip(machine.a * rho / machine.R0, 1e-5, 0.95)
    fc = (1.0 - eps)**2 / (jnp.sqrt(1.0 - eps**2 + 1e-12) * (1.0 + 1.46 * jnp.sqrt(eps)))
    return jnp.clip(1.0 - fc, 0.0, 0.98)


def sauter_collisionalities(Te_keV, Ti_keV, ne20, ni20, q, rho, machine):
    """nu_e*, nu_i* from Sauter 1999 Eq. (18b,c)."""
    eps = jnp.clip(machine.a * rho / machine.R0, 1e-4, 0.95)
    Z = jnp.maximum(machine.Zeff, 1.0)
    Te_eV = jnp.maximum(Te_keV * 1.0e3, 10.0)
    Ti_eV = jnp.maximum(Ti_keV * 1.0e3, 10.0)
    ne_m3 = jnp.maximum(ne20, 1e-6) * 1.0e20
    ni_m3 = jnp.maximum(ni20, 1e-6) * 1.0e20
    lnLe = sauter_ln_lambda_e(ne_m3, Te_eV)
    lnLii = sauter_ln_lambda_ii(ni_m3, Ti_eV, Z)
    q = jnp.maximum(q, 0.2)
    nue = 6.921e-18 * q * machine.R0 * ne_m3 * Z * lnLe / (Te_eV**2 * eps**1.5)
    nui = 4.90e-18 * q * machine.R0 * ni_m3 * Z**4 * lnLii / (Ti_eV**2 * eps**1.5)
    return jnp.maximum(nue, 0.0), jnp.maximum(nui, 0.0), lnLe, lnLii

def sauter_F33(X, Z):
    Z = jnp.maximum(Z, 1.0)
    return 1.0 - (1.0 + 0.36 / Z) * X + 0.59 / Z * X**2 - 0.23 / Z * X**3

def sauter_F31(X, Z):
    Z = jnp.maximum(Z, 1.0)
    return (
        (1.0 + 1.4 / (Z + 1.0)) * X
        - 1.9 / (Z + 1.0) * X**2
        + 0.3 / (Z + 1.0) * X**3
        + 0.2 / (Z + 1.0) * X**4
    )

def sauter_L32_terms(X, Y, Z):
    Z = jnp.maximum(Z, 1.0)
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
    return F32ee, F32ei

def sauter_alpha_coeff(ft, nui_star):
    """Ion-temperature-gradient bootstrap coefficient alpha.

    Sauter--Angioni--Lin-Liu 1999 Eq. (17b) is

        alpha = [a0 + 0.25(1-ft^2)sqrt(nui*)/(1+0.5sqrt(nui*))
                 - 0.315 nui*^2 ft^6]
                / [1 + 0.15 nui*^2 ft^6].

    A previous implementation accidentally divided ``a0`` by
    ``1+0.5*sqrt(nui*)`` as well.  That changes the plateau/high-
    collisionality behavior and is not the published Sauter fit.
    """
    ft = jnp.clip(ft, 0.0, 0.98)
    nui = jnp.maximum(nui_star, 0.0)
    sqrt_nui = jnp.sqrt(nui)
    a0 = -1.17 * (1.0 - ft) / (1.0 - 0.22 * ft - 0.19 * ft**2 + 1e-12)
    alpha = (
        a0
        + 0.25 * (1.0 - ft**2) * sqrt_nui / (1.0 + 0.5 * sqrt_nui)
        - 0.315 * nui**2 * ft**6
    ) / (1.0 + 0.15 * nui**2 * ft**6)
    return jnp.clip(alpha, -5.0, 5.0)

def sauter_coefficients_1999(rho, Te_keV, Ti_keV, ne20, q, machine):
    """Return Sauter 1999 F33,L31,L32,L34,alpha and related quantities."""
    Z = jnp.maximum(machine.Zeff, 1.0)
    ft = sauter_trapped_fraction(rho, machine)
    ni20 = ne20 / Z
    nue, nui, lnLe, lnLii = sauter_collisionalities(Te_keV, Ti_keV, ne20, ni20, q, rho, machine)

    ft33 = ft / (1.0 + (0.55 - 0.1 * ft) * jnp.sqrt(nue) + 0.45 * (1.0 - ft) * nue / (Z**1.5))
    ft31 = ft / (1.0 + (1.0 - 0.1 * ft) * jnp.sqrt(nue) + 0.5 * (1.0 - ft) * nue / Z)
    ft32ee = ft / (1.0 + 0.26 * (1.0 - ft) * jnp.sqrt(nue) + 0.18 * (1.0 - 0.37 * ft) * nue / jnp.sqrt(Z))
    ft32ei = ft / (1.0 + (1.0 + 0.6 * ft) * jnp.sqrt(nue) + 0.85 * (1.0 - 0.37 * ft) * nue * (1.0 + Z))
    ft34 = ft / (1.0 + (1.0 - 0.1 * ft) * jnp.sqrt(nue) + 0.5 * (1.0 - 0.5 * ft) * nue / Z)

    F33 = jnp.clip(sauter_F33(ft33, Z), 0.02, 1.0)
    L31 = sauter_F31(ft31, Z)
    F32ee, F32ei = sauter_L32_terms(ft32ee, ft32ei, Z)
    L32 = F32ee + F32ei
    L34 = sauter_F31(ft34, Z)
    alpha = sauter_alpha_coeff(ft, nui)

    Te_eV = jnp.maximum(Te_keV * 1.0e3, 10.0)
    sigma_sp = 1.9012e4 * Te_eV**1.5 / (Z * sauter_N_Z(Z) * lnLe + 1e-12)
    sigma_neo = sigma_sp * F33
    return {
        "F33": F33, "L31": L31, "L32": L32, "L34": L34, "alpha": alpha,
        "ft": ft, "nue_star": nue, "nui_star": nui, "lnLe": lnLe, "lnLii": lnLii,
        "sigma_sp": sigma_sp, "sigma_neo": sigma_neo,
    }

def sauter_neoclassical_resistivity_1999(Te_keV, Ti_keV, ne20, q, rho, machine):
    coeff = sauter_coefficients_1999(rho, Te_keV, Ti_keV, ne20, q, machine)
    return jnp.clip(1.0 / (coeff["sigma_neo"] + 1e-30), 1e-10, 2e-5)


def neoclassical_resistivity(Te_keV, rho, machine, eq=None, q=None):
    q_use = q_from_equilibrium_or_profile(rho, eq=eq, q=q)
    ne20 = jnp.ones_like(rho)
    return sauter_neoclassical_resistivity_1999(Te_keV, Te_keV, ne20, q_use, rho, machine)

def effective_current_drive_fraction(actuator, machine, nbar20=None):
    """Driven-current fraction of Ip.

    This uses the conventional normalized current-drive figure of merit; the
    Gaussian deposition shape and clipping are actuator-level approximations.

    If actuator.cd_fraction > 0, use it as a manual fraction of Ip.
    Otherwise use the standard normalized current-drive efficiency

        eta20 = n20 * I_CD[A] * R0[m] / P_aux[W],

    hence

        I_CD = eta20 * P_aux[W] / (n20 * R0[m]).

    Here n20 defaults to the actuator Greenwald density target.
    """
    n20 = target_nbar20(machine, actuator, None) if nbar20 is None else nbar20
    Icd = actuator.cd_efficiency_20 * actuator.P_aux_MW * 1.0e6 / (
        jnp.maximum(n20, 0.05) * machine.R0 + 1e-12
    )
    automatic_frac = Icd / (machine.Ip + 1e-12)
    # JAX-compatible selection: actuator fields are abstract tracers in the
    # sim-only-static optimization JIT, even when cd_fraction is not optimized.
    frac = jnp.where(actuator.cd_fraction > 0.0, actuator.cd_fraction, automatic_frac)
    return jnp.clip(frac, 0.0, actuator.cd_fraction_max)

def current_drive_density(rho, actuator, machine):
    """Externally driven non-inductive current density [A/m^2].

    If actuator.cd_fraction > 0, use it as a manual fraction of Ip.
    Otherwise compute I_CD = P_aux * cd_efficiency_20 and clip by
    cd_fraction_max.
    """
    frac = effective_current_drive_fraction(actuator, machine)
    shape = jnp.exp(-0.5 * ((rho - actuator.cd_center) / (actuator.cd_width + 1e-8)) ** 2)
    return normalize_current_profile(
        rho, shape + 1e-8, frac * machine.Ip, machine.a, machine.kappa
    )



def _flux_surface_average(theta, R, jac, y):
    """Volume-weighted flux-surface average of a 2D scalar y(rho,theta)."""
    dtheta = theta[1] - theta[0] if theta.size > 1 else 2.0 * jnp.pi
    w = jnp.maximum(R, 1.0e-6) * jnp.maximum(jac, 0.0)
    return jnp.sum(y * w, axis=1) * dtheta / (jnp.sum(w, axis=1) * dtheta + 1.0e-30)


def noninductive_B_projection(rho, j_ni_A_m2, machine, eq=None):
    """Approximate ``<B.j_ni>`` from a toroidal-current proxy.

    The bootstrap/CD closures return a cell-averaged toroidal-current-density
    proxy ``j_phi`` because the rest of the framework integrates current as
    ``I = ∫ j_phi dA``.  The psi equation instead needs the flux-surface
    projection ``<B.j_ni>``.

    Assuming the non-inductive current is predominantly field-aligned,

        j_phi = j_parallel * B_phi / B,
        B.j   = j_parallel * B = j_phi * B**2 / B_phi.

    Therefore the local conversion factor is ``B**2/B_phi`` rather than simply
    ``B_phi``.  This keeps the correct sign from the toroidal-field function F
    and includes the finite-Bp correction when Bpol is available.
    """
    if eq is not None and hasattr(eq, "R") and hasattr(eq, "F"):
        R = jnp.maximum(eq.R, 1.0e-6)
        F = _as_rho_profile(eq.F, rho)
        Bphi = F[:, None] / R
        if hasattr(eq, "Btor"):
            Bphi = eq.Btor
        if hasattr(eq, "Bpol"):
            Bp = eq.Bpol
        else:
            # Large-aspect-ratio fallback from q if a 2D Bpol proxy is absent.
            q = q_from_equilibrium_or_profile(rho, eq=eq, lo=0.5, hi=20.0)
            Bp_1d = machine.a * rho * machine.Bt / (machine.R0 * (q + 1.0e-8))
            Bp = Bp_1d[:, None] * jnp.ones_like(Bphi)
        B2 = Bphi * Bphi + Bp * Bp
        coeff_2d = B2 / _signed_floor(Bphi, 1.0e-8)
        if hasattr(eq, "theta") and hasattr(eq, "jac"):
            Bproj = _flux_surface_average(eq.theta, R, eq.jac, coeff_2d)
        else:
            Bproj = jnp.mean(coeff_2d, axis=1)
    elif eq is not None and hasattr(eq, "F"):
        F = _as_rho_profile(eq.F, rho)
        # Without 2D geometry, retain the correct field sign and magnitude.
        Bproj = jnp.sign(F) * jnp.abs(machine.Bt)
    else:
        Bproj = jnp.ones_like(rho) * machine.Bt
    return Bproj * j_ni_A_m2

def radial_gradient(y, machine, rho=None):
    if rho is None:
        rho = (jnp.arange(y.size, dtype=y.dtype) + 0.5) / y.size
    return grid_radial_gradient(y, rho, machine.a)

def sauter_bootstrap_current_density(rho, Te_keV, Ti_keV, ne20, j_ref, machine, sim, eq=None, q=None):
    """Sauter-1999 analytic bootstrap current density [A/m^2].

    The Sauter paper gives flux-surface-averaged <j_parallel B> coefficients.
    This reduced 1D closure uses the fitted coefficients and a local Btheta/q
    proxy to produce a toroidal-current-density source.
    """
    ne = jnp.maximum(ne20, 1e-6) * 1.0e20
    pressure = ne * (Te_keV + Ti_keV) * KEV_TO_J
    dpdr = radial_gradient(pressure, machine, rho)
    dlnp = dpdr / jnp.maximum(pressure, 1.0)
    dlnTe = radial_gradient(jnp.log(jnp.maximum(Te_keV, 0.03)), machine, rho)
    dlnTi = radial_gradient(jnp.log(jnp.maximum(Ti_keV, 0.03)), machine, rho)

    Ienc = enclosed_current(rho, j_ref, machine.a, machine.kappa)
    r = machine.a * rho
    Btheta_current = MU0 * Ienc / (2.0 * jnp.pi * jnp.maximum(r, 1e-3))
    q_current = jnp.clip(
        machine.a * machine.Bt * rho / (
            machine.R0 * jnp.maximum(jnp.abs(Btheta_current), sim.bootstrap_btheta_floor)
        ),
        0.5, 10.0
    )
    q_use = q_from_equilibrium_or_profile(rho, eq=eq, q=q, fallback=q_current, lo=0.5, hi=10.0)
    Btheta = machine.a * machine.Bt * rho / (machine.R0 * jnp.maximum(q_use, 1.0e-6))

    c = sauter_coefficients_1999(rho, Te_keV, Ti_keV, ne20, q_use, machine)
    drive = c["L31"] * dlnp + c["L32"] * dlnTe + c["L34"] * c["alpha"] * dlnTi

    jbs = -sim.bootstrap_multiplier * pressure * drive / jnp.maximum(jnp.abs(Btheta), sim.bootstrap_btheta_floor)
    axis_taper = 1.0 - jnp.exp(-(rho / 0.08)**2)
    jbs = axis_taper * jbs
    return jnp.clip(jbs, -8.0e6, 8.0e6)


def bootstrap_current_density(rho, Te_keV, Ti_keV, ne20, j_ref, machine, sim, eq=None, q=None):
    """Bootstrap-current closure selected by ``sim.bootstrap_model``.

    Default is the Sauter analytic fit.  If ``bootstrap_model="neonn_jax"`` is
    selected, this attempts to use a NEO/BrainFUSE model output.  The bundled
    public model in this repository currently exposes transport outputs only,
    so the default fail mode falls back to Sauter.
    """
    model = str(getattr(sim, "bootstrap_model", "sauter") or "sauter").lower()
    if model in ("sauter", "sauter1999", "analytic", "reduced"):
        return sauter_bootstrap_current_density(rho, Te_keV, Ti_keV, ne20, j_ref, machine, sim, eq=eq, q=q)
    if model in ("neonn", "neonn_jax", "neo_nn", "neo"):
        try:
            q_use = q_from_equilibrium_or_profile(rho, eq=eq, q=q, fallback=None, lo=0.5, hi=20.0)
            from .neonn_jax import neonn_jax_bootstrap_current_density
            return neonn_jax_bootstrap_current_density(rho, Te_keV, Ti_keV, ne20, q_use, machine, sim, eq=eq)
        except Exception:
            if str(getattr(sim, "neonn_bootstrap_fail_mode", "fallback")).lower() == "raise":
                raise
            return sauter_bootstrap_current_density(rho, Te_keV, Ti_keV, ne20, j_ref, machine, sim, eq=eq, q=q)
    raise ValueError(f"Unknown bootstrap_model={model!r}. Use 'sauter' or 'neonn_jax'.")


def current_components_from_state(rho, state, machine, actuator, sim, eq=None, q=None):
    """Return j_ohm, j_bs, j_cd, j_total."""
    psi_edge_for_j = getattr(state, "psi_edge", 0.0)
    if is_psi_diffusion_model(getattr(sim, "current_evolution_model", "")):
        initial_current_model = str(
            getattr(sim, "initial_current_profile_model", "saturated_components")
        ).lower()
        preserve_shape = initial_current_model in (
            "total_current_shape", "initial_current_shape", "shape"
        )
        if preserve_shape:
            # This option promises that initial_current_shape() is the actual
            # total-current profile.  The optional enclosed-current diagnostic
            # smoother otherwise adds a visible central bump to an already
            # smooth polynomial.  Reconstruct the stored total-current psi
            # exactly for this explicit profile mode.
            j_from_psi = current_from_psi(
                rho, state.psi_ind, machine,
                psi_edge=psi_edge_for_j, sim=sim, eq=eq,
            )
        else:
            j_from_psi = _regularized_current_from_psi_for_components(
                rho, state.psi_ind, psi_edge_for_j, machine, sim, eq
            )
    else:
        j_from_psi = current_from_psi(
            rho, state.psi_ind, machine, psi_edge=psi_edge_for_j, sim=sim, eq=eq
        )
    j_cd = current_drive_density(rho, actuator, machine)
        
    if is_psi_diffusion_model(getattr(sim, "current_evolution_model", "")):
        # Convention: psi -> total current.  j_ref is only used as a
        # fallback q/Btheta proxy inside the reduced bootstrap closure; when an
        # equilibrium object is supplied, eq.q remains the source of truth.
        j_bs = bootstrap_current_density(
            rho, state.Te, state.Ti, state.ne20, j_from_psi,
            machine, sim, eq=eq, q=q
        )
        j_total = j_from_psi
        j_ohm = j_total - j_bs - j_cd
        return j_ohm, j_bs, j_cd, j_total

    # Legacy convention: psi -> inductive/Ohmic current.
    j_ind = j_from_psi
    j_ref = j_ind + j_cd
    j_bs = bootstrap_current_density(rho, state.Te, state.Ti, state.ne20, j_ref, machine, sim, eq=eq, q=q)
    j_total = j_ind + j_bs + j_cd
    return j_ind, j_bs, j_cd, j_total


def loop_voltage_from_ip_error(rho, state, machine, actuator, sim, eq=None, q=None):
    """Scalar loop voltage controller from total Ip error."""
    _, _, _, j_total = current_components_from_state(rho, state, machine, actuator, sim, eq=eq, q=q)
    I_total = total_current(rho, j_total, machine, eq=eq)
    err_MA = (machine.Ip - I_total) / 1.0e6
    V_loop = sim.loop_voltage_gain * err_MA
    return jnp.clip(V_loop, -sim.loop_voltage_max, sim.loop_voltage_max), I_total



def smooth_profile_3pt(y, strength):
    """Small conservative-ish local smoother for profile noise."""
    if y.size < 3 or strength <= 0.0:
        return y
    ys = y
    mid = 0.25 * y[:-2] + 0.5 * y[1:-1] + 0.25 * y[2:]
    ys = ys.at[1:-1].set((1.0 - strength) * y[1:-1] + strength * mid)
    return ys



def saturated_conductivity_current_components(
    rho, Te, Ti, ne20, machine, actuator, sim, eq=None, q=None, n_iter=3
):
    """Return a self-consistent saturated Ohmic+non-inductive current split.

    This is intended for initializing the ``psi_diffusion``
    branch.  In that branch the stored ``psi`` represents the total plasma
    current, not the Ohmic current alone.  A polynomial total-current profile
    therefore gives an artificial negative Ohmic layer wherever bootstrap is
    large.  Instead initialize from the saturated Ohmic relation

        j_ohm ∝ sigma_parallel,
        ∫(j_ohm + j_bs + j_cd) dA = Ip.

    The bootstrap closure mostly uses ``eq.q`` when an equilibrium is supplied,
    so the fixed-point loop is mainly a harmless robustness measure for legacy
    fallback calls.
    """
    q_use = q_from_equilibrium_or_profile(rho, eq=eq, q=q, lo=0.5, hi=10.0)
    eta = sauter_neoclassical_resistivity_1999(Te, Ti, ne20, q_use, rho, machine)
    sigma = 1.0 / jnp.maximum(eta, 1.0e-12)
    sigma_mean = jnp.mean(sigma)
    sigma_shape = jnp.maximum(sigma, sim.saturated_current_sigma_floor * sigma_mean)
    sigma_shape = sigma_shape ** sim.saturated_current_conductivity_power
    sigma_shape = smooth_profile_3pt(sigma_shape, sim.saturated_current_smooth)
    area = poloidal_area_weights(rho, machine, eq=eq)

    j_cd = current_drive_density(rho, actuator, machine)
    I_cd = total_current(rho, j_cd, machine, eq=eq)
    j_ohm = normalize_current_profile(
        rho, sigma_shape + 1.0e-12,
        jnp.maximum(machine.Ip - I_cd, 0.0),
        machine.a, machine.kappa, area=area,
    )
    j_bs = jnp.zeros_like(rho)
    for _ in range(int(n_iter)):
        j_ref = j_ohm + j_bs + j_cd
        j_bs = bootstrap_current_density(
            rho, Te, Ti, ne20, j_ref, machine, sim, eq=eq, q=q_use
        )
        I_nonind = total_current(rho, j_bs + j_cd, machine, eq=eq)
        I_ohm_target = jnp.maximum(machine.Ip - I_nonind, 0.0)
        j_ohm = normalize_current_profile(
            rho, sigma_shape + 1.0e-12, I_ohm_target,
            machine.a, machine.kappa, area=area,
        )
    j_total = j_ohm + j_bs + j_cd
    # Remove tiny integral drift from the fixed-point/bootstrap approximation by
    # scaling only the Ohmic part; keep bootstrap and CD as computed sources.
    I_total = total_current(rho, j_total, machine, eq=eq)
    I_ohm = total_current(rho, j_ohm, machine, eq=eq)
    j_ohm = j_ohm * (machine.Ip - total_current(rho, j_bs + j_cd, machine, eq=eq)) / (I_ohm + 1.0e-12)
    j_total = j_ohm + j_bs + j_cd
    return j_ohm, j_bs, j_cd, j_total

def saturated_conductivity_current_density(rho, state, Te, Ti, ne20, machine, actuator, sim, eq=None, q=None):
    """Fully saturated inductive-current closure.

    Assumption: resistive current has saturated to a common loop electric field,
    so

        j_ind(r) = sigma_neo(r) * E_parallel,

    and the scalar E_parallel is chosen so that

        integral(j_ind) + I_BS + I_CD = Ip.

    No q-profile target is imposed. The profile follows conductivity, which is
    mostly Te^(3/2) with neoclassical corrections.
    """
    # Previous current gives only a q estimate for neoclassical coefficients.
    try:
        j_prev = current_from_psi(
            rho, state.psi_ind, machine, psi_edge=getattr(state, "psi_edge", 0.0), sim=sim, eq=eq
        )
    except Exception:
        j_prev = normalize_current_profile(
            rho, initial_current_shape(rho, peaking=1.5, floor=0.0), machine.Ip, machine.a, machine.kappa
        )

    q_fallback = _q_profile_from_current(rho, j_prev, machine, sim)
    q_use = q_from_equilibrium_or_profile(rho, eq=eq, q=q, fallback=q_fallback, lo=0.5, hi=10.0)

    eta = sauter_neoclassical_resistivity_1999(Te, Ti, ne20, q_use, rho, machine)
    sigma = 1.0 / jnp.maximum(eta, 1e-12)

    # Conductivity floor and optional power allow a controlled saturated shape.
    sigma_mean = jnp.mean(sigma)
    sigma_shape = jnp.maximum(sigma, sim.saturated_current_sigma_floor * sigma_mean)
    sigma_shape = sigma_shape ** sim.saturated_current_conductivity_power
    sigma_shape = smooth_profile_3pt(sigma_shape, sim.saturated_current_smooth)

    j_cd = current_drive_density(rho, actuator, machine)

    # Bootstrap with provisional conductivity-shaped inductive current.
    j_ind_prov = normalize_current_profile(
        rho, sigma_shape + 1e-12,
        jnp.maximum(machine.Ip - total_current(rho, j_cd, machine, eq=eq), 0.0),
        machine.a, machine.kappa,
        area=poloidal_area_weights(rho, machine, eq=eq),
    )
    j_ref = j_ind_prov + j_cd
    j_bs = bootstrap_current_density(rho, Te, Ti, ne20, j_ref, machine, sim, eq=eq, q=q)

    I_nonind = total_current(rho, j_bs + j_cd, machine, eq=eq)
    I_ind_target = jnp.maximum(machine.Ip - I_nonind, 0.0)
    j_ind = normalize_current_profile(
        rho, sigma_shape + 1e-12, I_ind_target, machine.a, machine.kappa,
        area=poloidal_area_weights(rho, machine, eq=eq),
    )
    return j_ind

def saturated_conductivity_psi_update(rho, state, Te, Ti, ne20, machine, actuator, sim, eq=None, q=None):
    """Return psi corresponding to saturated-conductivity current; no diffusion solve."""
    j_ind = saturated_conductivity_current_density(
        rho, state, Te, Ti, ne20, machine, actuator, sim, eq=eq, q=q
    )
    psi_edge = getattr(state, "psi_edge", 0.0)
    psi_ind = psi_from_current_density(rho, j_ind, machine, psi_edge=psi_edge, sim=sim, eq=eq)
    return psi_ind, psi_edge


def _face_average_current(y):
    return jnp.concatenate([y[0:1], 0.5 * (y[:-1] + y[1:]), y[-1:]])


def _cell_to_inner_face_current(y):
    """Interpolate a cell profile to face unknowns 0..nr-1.

    The LCFS face is excluded because the fixed-Ip Neumann condition supplies
    its flux-gradient.  Face 0 uses the first cell value; interior faces use the
    adjacent-cell average.
    """
    if y.size <= 1:
        return y
    return jnp.concatenate([y[0:1], 0.5 * (y[:-1] + y[1:])])


def _edge_dpsi_drho_from_current(I_edge_A, machine, dtype=None, rho=None, eq=None):
    """LCFS Neumann gradient corresponding to an enclosed edge current.

    With an Equilibrium object this is the exact inverse of the relation

        I_p = [dpsi/drho_norm * (g2*g3/rho_norm) * F
               / (16*pi^3*mu0*Phi_b)]_LCFS.

    Without geometry, fall back to the old circular large-aspect-ratio formula
    and its legacy sign convention.
    """
    I = jnp.asarray(I_edge_A, dtype=dtype)
    if rho is not None and eq is not None:
        S_edge = _signed_floor(_ampere_outer_face_factor(rho, eq)[-1], 1.0e-30)
        val = I / S_edge
    else:
        val = -MU0 * machine.R0 * I / (2.0 * jnp.pi + 1.0e-30)
    return jnp.asarray(val, dtype=dtype) if dtype is not None else val


def _face_operator_fixed_gradient_tridiag(rho, C_cell, A_cell, grad_edge):
    """Tridiagonal coefficients for ``_face_operator_fixed_gradient``."""
    nr = rho.size
    d = 1.0 / nr
    C_face = jnp.maximum(_cell_to_inner_face_current(C_cell), 1.0e-30)
    A_cell = jnp.maximum(A_cell, 0.0)
    main = jnp.zeros(nr, dtype=rho.dtype)
    lower = jnp.zeros(max(nr - 1, 0), dtype=rho.dtype)
    upper = jnp.zeros(max(nr - 1, 0), dtype=rho.dtype)
    b = jnp.zeros(nr, dtype=rho.dtype)
    if nr == 1:
        b = b.at[0].set(A_cell[-1] * grad_edge / (C_face[0] * d + 1.0e-30))
        return lower, main, upper, b
    hi0 = 2.0 * A_cell[0] / (C_face[0] * d**2 + 1.0e-30)
    main = main.at[0].set(-hi0)
    upper = upper.at[0].set(hi0)
    for k in range(1, nr):
        lo = A_cell[k - 1] / (C_face[k] * d**2 + 1.0e-30)
        lower = lower.at[k - 1].set(lo)
        if k == nr - 1:
            bnd = A_cell[k] * grad_edge / (C_face[k] * d + 1.0e-30)
            main = main.at[k].set(-lo)
            b = b.at[k].set(bnd)
        else:
            hi = A_cell[k] / (C_face[k] * d**2 + 1.0e-30)
            main = main.at[k].set(-(lo + hi))
            upper = upper.at[k].set(hi)
    return lower, main, upper, b


def _cell_operator_fixed_gradient_tridiag(rho, C_cell, A_cell, grad_edge):
    """Tridiagonal coefficients for ``_cell_operator_fixed_gradient``."""
    rho_f = infer_rho_faces(rho)
    drho_cell = rho_f[1:] - rho_f[:-1]
    A_face = _face_average_current(A_cell)
    dist_lo = jnp.concatenate([(rho[0] - rho_f[0])[None], rho[1:] - rho[:-1]])
    dist_hi = jnp.concatenate([rho[1:] - rho[:-1], (rho_f[-1] - rho[-1])[None]])
    C = jnp.maximum(C_cell, 1.0e-30)
    lo = A_face[:-1] / (C * drho_cell * (dist_lo + 1.0e-12))
    hi = A_face[1:] / (C * drho_cell * (dist_hi + 1.0e-12))
    lo = lo.at[0].set(0.0)
    hi_L = hi.at[-1].set(0.0)
    main = -(lo + hi_L)
    upper = hi_L[:-1]
    lower = lo[1:]
    b = jnp.zeros_like(rho).at[-1].set(
        A_face[-1] * grad_edge / (C[-1] * drho_cell[-1] + 1.0e-30)
    )
    return lower, main, upper, b


def _dirichlet_operator_cell_tridiag(rho, C_cell, A_cell, edge_value):
    """Tridiagonal coefficients for ``_dirichlet_operator_cell``."""
    rho_f = infer_rho_faces(rho)
    drho_cell = rho_f[1:] - rho_f[:-1]
    A_face = _face_average_current(A_cell)
    dist_lo = jnp.concatenate([(rho[0] - rho_f[0])[None], rho[1:] - rho[:-1]])
    dist_hi = jnp.concatenate([rho[1:] - rho[:-1], (rho_f[-1] - rho[-1])[None]])
    C = jnp.maximum(C_cell, 1.0e-30)
    lo = A_face[:-1] / (C * drho_cell * (dist_lo + 1.0e-12))
    hi = A_face[1:] / (C * drho_cell * (dist_hi + 1.0e-12))
    lo = lo.at[0].set(0.0)
    main = -(lo + hi)
    upper = hi[:-1]
    lower = lo[1:]
    b = jnp.zeros_like(rho).at[-1].set(hi[-1] * edge_value)
    return lower, main, upper, b


def _dirichlet_operator_face_tridiag(rho, C_cell, A_cell, edge_value):
    """Tridiagonal coefficients for ``_dirichlet_operator_face``."""
    nr = rho.size
    d = 1.0 / nr
    C_face = jnp.maximum(_cell_to_inner_face_current(C_cell), 1.0e-30)
    A_cell = jnp.maximum(A_cell, 0.0)
    main = jnp.zeros(nr, dtype=rho.dtype)
    lower = jnp.zeros(max(nr - 1, 0), dtype=rho.dtype)
    upper = jnp.zeros(max(nr - 1, 0), dtype=rho.dtype)
    b = jnp.zeros(nr, dtype=rho.dtype)
    if nr == 1:
        hi = A_cell[-1] / (C_face[0] * d**2 + 1.0e-30)
        main = main.at[0].set(-hi)
        b = b.at[0].set(hi * edge_value)
        return lower, main, upper, b
    hi0 = 2.0 * A_cell[0] / (C_face[0] * d**2 + 1.0e-30)
    main = main.at[0].set(-hi0)
    upper = upper.at[0].set(hi0)
    for k in range(1, nr):
        lo = A_cell[k - 1] / (C_face[k] * d**2 + 1.0e-30)
        lower = lower.at[k - 1].set(lo)
        if k == nr - 1:
            hi = A_cell[k] / (C_face[k] * d**2 + 1.0e-30)
            main = main.at[k].set(-(lo + hi))
            b = b.at[k].set(hi * edge_value)
        else:
            hi = A_cell[k] / (C_face[k] * d**2 + 1.0e-30)
            main = main.at[k].set(-(lo + hi))
            upper = upper.at[k].set(hi)
    return lower, main, upper, b


def _current_phi_dot_over_phi(state, eq, sim, dtype):
    """Manual + automatic Phi_b_dot/Phi_b used by the current equation."""
    manual = jnp.asarray(getattr(sim, "phi_dot_over_phi", 0.0), dtype=dtype)
    if not getattr(sim, "auto_phi_dot_over_phi", True):
        return manual
    prev_phi = jnp.asarray(getattr(state, "Phi_b_prev", 0.0), dtype=dtype)
    phi_b = jnp.asarray(eq.Phi_b, dtype=dtype)
    dt = jnp.asarray(getattr(sim, "dt", 1.0), dtype=dtype)
    auto_phi = jnp.where(
        prev_phi > 1.0e-12,
        (phi_b - prev_phi) / (dt * prev_phi + 1.0e-12),
        0.0,
    )
    lim = jnp.asarray(getattr(sim, "phi_dot_over_phi_clip", 5.0), dtype=dtype)
    return manual + jnp.clip(auto_phi, -lim, lim)


def _gradient_face_unknowns_with_edge_gradient(rho, psi_u, grad_edge):
    """dpsi/drho on face unknowns 0..nr-1 with axis and LCFS Neumann data."""
    nr = rho.size
    d = 1.0 / nr
    if nr == 1:
        return jnp.zeros_like(psi_u).at[-1].set(grad_edge)
    g = jnp.zeros_like(psi_u)
    # Magnetic-axis symmetry.
    g = g.at[0].set(0.0)
    if nr > 2:
        g = g.at[1:-1].set((psi_u[2:] - psi_u[:-2]) / (2.0 * d))
    # The eliminated LCFS face is reconstructed with the prescribed gradient.
    g = g.at[-1].set(grad_edge)
    return g


def _gradient_cell_unknowns_with_edge_gradient(rho, psi_u, grad_edge):
    """dpsi/drho on cell centers with zero-axis and fixed-LCFS gradients."""
    nr = rho.size
    if nr == 1:
        return jnp.zeros_like(psi_u).at[-1].set(grad_edge)
    g = grid_radial_gradient(psi_u, rho, 1.0)
    g = g.at[0].set(0.0)
    g = g.at[-1].set(grad_edge)
    return g


def psi_diffusion_step(rho, state, Te, Ti, ne20, machine, actuator, sim, eq, q=None):
    """Full poloidal-flux diffusion step.

    Reference: [F. Felici, EPFL Thesis 5203 (2011), Ch. 6].
    The finite-volume staggering, fixed-Ip boundary, and explicit source
    smoothing are TokaGrad numerical choices.

    This branch implements the psi-diffusion equation in normalized toroidal-flux
    coordinate rho,

        C_psi (dpsi/dt - 0.5*rho*(Phi_b_dot/Phi_b)*dpsi/drho)
          = d/drho[ A*dpsi/drho ] - Q_ni,

    with

        C_psi = 16*pi^2*mu0*sigma_parallel*rho*Phi_b^2/F^2,
        A     = g2*g3/rho,
        Q_ni = 8*pi^2*V'*mu0*Phi_b/F^2 * <B.j_ni>.

    """
    if eq is None:
        raise ValueError("psi_diffusion_step requires an Equilibrium object.")

    psi_state = state.psi_ind
    psi_is_face = psi_state.size == rho.size + 1

    q_eq = q_from_equilibrium_or_profile(rho, eq=eq, q=q, lo=0.5, hi=10.0)
    eta = sauter_neoclassical_resistivity_1999(Te, Ti, ne20, q_eq, rho, machine)
    sigma = 1.0 / jnp.maximum(eta, 1.0e-12)

    # A scalar multiplier is useful for accelerated demonstrations.  It changes
    # the penetration time but not the saturated conductivity-shaped solution.
    diff_mult = jnp.maximum(
        jnp.asarray(getattr(sim, "resistivity_multiplier", 1.0), dtype=rho.dtype),
        1.0e-12,
    )
    sigma_eff = sigma / diff_mult

    F = jnp.maximum(jnp.abs(_as_rho_profile(eq.F, rho)), 1.0e-6)
    Phi_b = jnp.maximum(jnp.asarray(eq.Phi_b, dtype=rho.dtype), 1.0e-12)
    Vp = jnp.maximum(eq.dV_drho, 1.0e-12)
    rho_safe = jnp.maximum(rho, 1.0e-4)

    Cpsi = 16.0 * jnp.pi**2 * MU0 * sigma_eff * rho_safe * Phi_b**2 / (F**2 + 1.0e-30)
    if hasattr(eq, "g2g3_over_rho"):
        A = jnp.maximum(_as_rho_profile(eq.g2g3_over_rho, rho), 1.0e-30)
    else:
        A = jnp.maximum(eq.g2 * eq.g3 / rho_safe, 1.0e-30)

    # Non-inductive source.  The reduced bootstrap/CD closures return toroidal
    # current-density proxies; Bavg projects them approximately to <B.j_ni>.
    j_total_for_q = current_from_psi(
        rho, psi_state, machine,
        psi_edge=getattr(state, "psi_edge", 0.0), sim=sim, eq=eq,
    )
    j_cd = current_drive_density(rho, actuator, machine)
    j_bs = bootstrap_current_density(rho, Te, Ti, ne20, j_total_for_q, machine, sim, eq=eq, q=q_eq)
    Bdotjni = noninductive_B_projection(rho, j_bs + j_cd, machine, eq=eq)
    Qni = 8.0 * jnp.pi**2 * Vp * MU0 * Phi_b * Bdotjni / (F**2 + 1.0e-30)
    source_cell = -Qni / (Cpsi + 1.0e-30)
    source_cell = _binomial_smooth_cell_profile(
        source_cell, passes=getattr(sim, "current_source_smoothing_passes", 1)
    )

    phi_rate = _current_phi_dot_over_phi(state, eq, sim, rho.dtype)
    boundary_model = getattr(sim, "new_psi_boundary_model", "fixed_ip_neumann")

    if boundary_model == "fixed_ip_neumann":
        # Neumann condition fixes the total plasma current represented
        # by psi.  A legacy compatibility option is provided for old workflows
        # that stored only inductive current in psi.
        I_edge = machine.Ip
        grad_edge = _edge_dpsi_drho_from_current(I_edge, machine, dtype=rho.dtype, rho=rho, eq=eq)

        if psi_is_face:
            psi_u = psi_state[:-1]
            lower, main, upper, b_neu = _face_operator_fixed_gradient_tridiag(rho, Cpsi, A, grad_edge)
            source = _cell_to_inner_face_current(source_cell)
            rho_face_u = infer_rho_faces(rho)[:-1]
            psi_grad = _gradient_face_unknowns_with_edge_gradient(rho, psi_u, grad_edge)
            phidot_rate = 0.5 * phi_rate * rho_face_u * psi_grad
            rhs = psi_u + sim.dt * (source + phidot_rate + b_neu)
            psi_u_new = solve_implicit_tridiagonal_current(lower, main, upper, rhs, sim.dt)
            d_edge = infer_rho_faces(rho)[-1] - infer_rho_faces(rho)[-2]
            psi_edge_new = psi_u_new[-1] + grad_edge * d_edge
            psi_new = jnp.concatenate([psi_u_new, psi_edge_new[None]])
            psi_new, psi_edge_new = _maybe_smooth_new_psi_current(
                rho, psi_new, psi_edge_new, machine, sim, eq
            )
            return psi_new, psi_edge_new

        psi_u = psi_state
        lower, main, upper, b_neu = _cell_operator_fixed_gradient_tridiag(rho, Cpsi, A, grad_edge)
        psi_grad = _gradient_cell_unknowns_with_edge_gradient(rho, psi_u, grad_edge)
        phidot_rate = 0.5 * phi_rate * rho * psi_grad
        rhs = psi_u + sim.dt * (source_cell + phidot_rate + b_neu)
        psi_new = solve_implicit_tridiagonal_current(lower, main, upper, rhs, sim.dt)
        d_edge = infer_rho_faces(rho)[-1] - rho[-1]
        psi_edge_new = psi_new[-1] + grad_edge * d_edge
        psi_new, psi_edge_new = _maybe_smooth_new_psi_current(
            rho, psi_new, psi_edge_new, machine, sim, eq
        )
        return psi_new, psi_edge_new

    if boundary_model != "edge_psi_dirichlet":
        raise ValueError(
            f"Unknown new_psi_boundary_model={boundary_model!r}. "
            'Use "fixed_ip_neumann" or "edge_psi_dirichlet".'
        )

    # Alternative boundary: prescribe LCFS psi at t+dt.  When the standard
    # loop-voltage feedback model is active, advance that boundary by V_loop.
    edge_psi_old = getattr(state, "psi_edge", psi_state[-1] if psi_is_face else 0.0)
    if getattr(sim, "current_feedback_model", "") == "psi_boundary_loop_voltage":
        V_loop, _ = loop_voltage_from_ip_error(rho, state, machine, actuator, sim, eq=eq, q=q_eq)
        edge_psi_value = edge_psi_old + V_loop * sim.dt / (2.0 * jnp.pi)
    else:
        edge_psi_value = edge_psi_old

    if psi_is_face:
        psi_u = psi_state[:-1]
        lower, main, upper, b_dir = _dirichlet_operator_face_tridiag(rho, Cpsi, A, edge_psi_value)
        source = _cell_to_inner_face_current(source_cell)
        rho_face_u = infer_rho_faces(rho)[:-1]
        # Use the current edge value for the explicit Phi-dot gradient.
        psi_edge_full = jnp.concatenate([psi_u, jnp.asarray(edge_psi_value, dtype=psi_u.dtype)[None]])
        psi_grad_full = grid_radial_gradient(psi_edge_full, infer_rho_faces(rho), 1.0)
        phidot_rate = 0.5 * phi_rate * rho_face_u * psi_grad_full[:-1]
        psi_u_new = solve_implicit_tridiagonal_current(lower, main, upper, psi_u + sim.dt * (source + phidot_rate + b_dir), sim.dt)
        psi_new = jnp.concatenate([psi_u_new, jnp.asarray(edge_psi_value, dtype=psi_u.dtype)[None]])
        psi_new, edge_psi_value = _maybe_smooth_new_psi_current(
            rho, psi_new, edge_psi_value, machine, sim, eq
        )
        return psi_new, edge_psi_value

    lower, main, upper, b_dir = _dirichlet_operator_cell_tridiag(rho, Cpsi, A, edge_psi_value)
    psi_grad = grid_radial_gradient(psi_state, rho, 1.0)
    phidot_rate = 0.5 * phi_rate * rho * psi_grad
    psi_new = solve_implicit_tridiagonal_current(
        lower, main, upper,
        psi_state + sim.dt * (source_cell + phidot_rate + b_dir),
        sim.dt,
    )
    psi_new, edge_psi_value = _maybe_smooth_new_psi_current(
        rho, psi_new, edge_psi_value, machine, sim, eq
    )
    return psi_new, edge_psi_value



def psi_inductive_update(rho, state, Te_new, Ti_new, ne_new, machine, actuator, sim, eq=None):
    """Update psi_ind and psi_edge through resistive diffusion plus loop voltage."""
    current_model = getattr(sim, "current_evolution_model", "saturated_conductivity")
    if current_model == "saturated_conductivity":
        return saturated_conductivity_psi_update(
            rho, state, Te_new, Ti_new, ne_new, machine, actuator, sim, eq=eq
        )
    if is_psi_diffusion_model(current_model):
        if eq is None:
            raise ValueError("psi_diffusion requires an Equilibrium object.")
        return psi_diffusion_step(
            rho, state, Te_new, Ti_new, ne_new, machine, actuator, sim, eq
        )
    raise ValueError(
        f"Unknown current_evolution_model={current_model!r}. "
        "Use 'saturated_conductivity' or 'psi_diffusion'."
    )
