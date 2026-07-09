"""Fast fixed-boundary geometry and prescribed G-EQDSK reconstruction.

References:
  [R. L. Miller et al., Phys. Plasmas 5, 973 (1998)] -- Miller surfaces.
  [L. L. Lao et al., Nucl. Fusion 25, 1611 (1985)] -- EFIT/G-EQDSK context.
  [O. Sauter, Fusion Eng. Des. 112, 633 (2016)] -- shaped-plasma geometry.

The internal ``reduced_fixed_boundary`` branch is a moment/geometry closure;
it is not a Grad--Shafranov equilibrium solve.  Its Shafranov shift and reduced
metric options should therefore be read as calibrated approximations.
"""

import jax.numpy as jnp
from typing import NamedTuple

from .grid import make_grid_from_config, cell_widths, infer_rho_faces
from .current import (
    MU0,
    q_profile,
    enclosed_current,
    enclosed_current_centers,
    current_components_from_state,
    total_current,
    shape_factor,
)

KEV_TO_J = 1.602176634e-16

class Equilibrium(NamedTuple):
    """Fast fixed-boundary equilibrium / geometry object.

    All arrays are JAX arrays. The 2D arrays have shape [nr, ntheta].
    """
    rho: jnp.ndarray
    theta: jnp.ndarray
    R: jnp.ndarray
    Z: jnp.ndarray
    jac: jnp.ndarray             # |d(R,Z)/d(rho,theta)| [m^2]
    dV_drho: jnp.ndarray         # dV/drho [m^3]
    V: jnp.ndarray               # enclosed volume [m^3]
    g0: jnp.ndarray              # reduced <|grad V|> metric [m^2]
    g1: jnp.ndarray              # reduced <|grad V|^2> metric [m^4]
    g2: jnp.ndarray              # reduced <|grad V|^2/R^2> metric [m^2]
    g3: jnp.ndarray              # reduced <1/R^2> metric [m^-2]
    g2g3_over_rho: jnp.ndarray  # composite psi/current metric [m^-2]
    Phi_b: jnp.ndarray           # reduced toroidal flux at LCFS [Wb]
    psi_norm: jnp.ndarray        # normalized poloidal flux proxy
    psi_pol: jnp.ndarray         # Wb/rad-like proxy, zero on axis
    q: jnp.ndarray
    p: jnp.ndarray               # pressure [Pa]
    F: jnp.ndarray               # R B_phi ~ R0 Bt
    Btor: jnp.ndarray            # 2D toroidal field
    Bpol: jnp.ndarray            # flux-surface averaged Bp proxy on 2D grid
    Bmag: jnp.ndarray            # total |B|
    j_total: jnp.ndarray
    j_ind: jnp.ndarray
    j_bs: jnp.ndarray
    j_cd: jnp.ndarray
    I_total: jnp.ndarray
    I_enclosed: jnp.ndarray
    beta_p: jnp.ndarray
    li_proxy: jnp.ndarray
    shape_factor: jnp.ndarray
    rbbbs: jnp.ndarray
    zbbbs: jnp.ndarray

def pressure_profile(state):
    """Thermal pressure p = ne (Te+Ti) in Pa."""
    ne = state.ne20 * 1.0e20
    return ne * (state.Te + state.Ti) * KEV_TO_J

def estimate_beta_p_and_li(rho, p, j_total, machine):
    """Fast scalar equilibrium proxies for shaping/shift.

    These are not full equilibrium quantities; they are useful geometry
    modifiers for a fast fixed-boundary solver.
    """
    r = machine.a * rho
    dr = cell_widths(rho, machine.a)

    # Simple volume-like weighting.
    w = rho + 1e-12
    p_avg = jnp.sum(p * w) / jnp.sum(w)

    # Edge poloidal field estimate.
    Bpa = MU0 * machine.Ip / (2.0 * jnp.pi * machine.a + 1e-12)
    beta_p = 2.0 * MU0 * p_avg / (Bpa**2 + 1e-12)

    # Internal inductance proxy: <Bp^2> / Bpa^2.
    Ienc = enclosed_current(rho, j_total, machine.a, machine.kappa)
    Bp = MU0 * Ienc / (2.0 * jnp.pi * r + 1e-12)
    li = 2.0 * jnp.sum((Bp / (Bpa + 1e-12)) ** 2 * rho) / (jnp.sum(rho) + 1e-12)
    return beta_p, li


def analytic_shafranov_shift_axis(machine, sim, beta_p, li):
    """Large-aspect-ratio analytic Shafranov-shift closure.

    Background: [J. Wesson, Tokamaks, 4th ed. (2011)].  The clipped scalar
    expression used here is a TokaGrad reduced approximation.

    Uses the reduced analytic estimate

        Delta0/a = 0.5 * epsilon * (beta_p + li/2 - 1/2),

    clipped to a non-negative, numerically bounded displacement.  The manual
    Shafranov-shift mode has been removed so reduced fixed-boundary geometry
    is controlled by this single analytic closure.
    """
    eps = machine.a / machine.R0
    delta_a = 0.5 * eps * (beta_p + 0.5 * li - 0.5)
    return jnp.clip(delta_a, 0.0, sim.shafranov_shift_max)

def miller_moment_geometry(rho, theta, machine, sim, beta_p=0.0, li=1.0):
    """Generate simple fixed-boundary Miller/moment flux surfaces.

    Boundary is fixed by (a, kappa, delta). Interior surfaces are generated
    with smooth rho-dependent elongation, triangularity, and Shafranov shift.

    This is intentionally fast and robust rather than a full GS solve.
    """
    rr = machine.a * rho[:, None]
    th = theta[None, :]

    # Interior shape profiles: kappa and delta vanish smoothly near axis.
    kappa_r = 1.0 + (machine.kappa - 1.0) * rho[:, None] ** sim.elongation_profile_power
    delta_r = machine.delta * rho[:, None] ** sim.triangularity_profile_power

    # Axis shift is internal only and goes to zero at boundary.
    shift_axis = analytic_shafranov_shift_axis(machine, sim, beta_p, li)
    shift0 = machine.a * shift_axis
    shift = shift0 * (1.0 - rho[:, None] ** 2)

    angle = th + delta_r * jnp.sin(th)
    R = machine.R0 + shift + rr * jnp.cos(angle)
    Z = kappa_r * rr * jnp.sin(th)
    return R, Z

def numerical_jacobian(R, Z, rho, theta):
    """Compute |d(R,Z)/d(rho,theta)| on uniform or non-uniform rho grids."""
    dtheta = theta[1] - theta[0]

    def grad_rho(A):
        if rho.size < 3:
            Ap = jnp.concatenate([A[0:1, :], A, A[-1:, :]], axis=0)
            dr = jnp.maximum(rho[1] - rho[0], 1e-12)
            return (Ap[2:, :] - Ap[:-2, :]) / (2.0 * dr)
        g0 = (A[1:2, :] - A[0:1, :]) / (rho[1] - rho[0] + 1e-12)
        gi = (A[2:, :] - A[:-2, :]) / ((rho[2:] - rho[:-2])[:, None] + 1e-12)
        gN = (A[-1:, :] - A[-2:-1, :]) / (rho[-1] - rho[-2] + 1e-12)
        return jnp.concatenate([g0, gi, gN], axis=0)

    def grad_theta(A):
        return (jnp.roll(A, -1, axis=1) - jnp.roll(A, 1, axis=1)) / (2.0 * dtheta)

    R_r = grad_rho(R)
    R_t = grad_theta(R)
    Z_r = grad_rho(Z)
    Z_t = grad_theta(Z)

    jac = jnp.abs(R_r * Z_t - R_t * Z_r)
    return jnp.maximum(jac, 1e-12)


def _cumtrapz_on_rho(y, rho):
    """Cumulative trapezoid integral with a first-axis extrapolation."""
    if rho.size < 2:
        return y * rho
    dr = rho[1:] - rho[:-1]
    seg = 0.5 * (y[1:] + y[:-1]) * dr
    y_axis = y[0] - (y[1] - y[0]) * rho[0] / (rho[1] - rho[0] + 1e-12)
    first = 0.5 * (y_axis + y[0]) * rho[0]
    return jnp.concatenate([jnp.asarray([first], dtype=y.dtype), first + jnp.cumsum(seg)])


def volume_elements(R, jac, theta, rho=None):
    """Compute dV/drho = ∫ 2π R J dtheta and enclosed volume."""
    dtheta = theta[1] - theta[0]
    dV_drho = jnp.sum(2.0 * jnp.pi * R * jac, axis=1) * dtheta
    if rho is None:
        V = jnp.cumsum(dV_drho) / dV_drho.size
    else:
        V = _cumtrapz_on_rho(dV_drho, rho)
    return dV_drho, V


def _surface_gradient_metrics(R, Z, jac, rho, theta, dV_drho, machine):
    """Return flux-surface metric moments from nested surfaces.

    For coordinates ``(rho, theta)``, ``|grad rho| = dl_theta / jac``.  This
    distinction matters for shaped plasmas: replacing ``|grad rho|`` by the
    circular proxy ``1/a`` makes the q correction scale approximately as
    kappa**2 instead of the familiar elliptical ``(1+kappa**2)/2``.
    """
    dtheta = theta[1] - theta[0]
    R_theta = (jnp.roll(R, -1, axis=1) - jnp.roll(R, 1, axis=1)) / (2.0 * dtheta)
    Z_theta = (jnp.roll(Z, -1, axis=1) - jnp.roll(Z, 1, axis=1)) / (2.0 * dtheta)
    dl_dtheta = jnp.sqrt(R_theta**2 + Z_theta**2 + 1.0e-30)
    grad_rho = dl_dtheta / jnp.maximum(jac, 1.0e-12)

    # A flux-surface average is weighted by dV = 2*pi*R*J dtheta drho.
    weight = jnp.maximum(2.0 * jnp.pi * R * jac, 0.0)
    weight_sum = jnp.maximum(jnp.sum(weight, axis=1), 1.0e-30)

    def fs_average(value):
        return jnp.sum(weight * value, axis=1) / weight_sum

    grad_V = jnp.maximum(dV_drho, 1.0e-12)[:, None] * grad_rho
    g0 = fs_average(grad_V)
    g1 = fs_average(grad_V**2)
    g2 = fs_average(grad_V**2 / jnp.maximum(R**2, 1.0e-12))
    g3 = fs_average(1.0 / jnp.maximum(R**2, 1.0e-12))
    return g0, g1, g2, g3


def reduced_flux_metrics(R, Z, jac, theta, rho, machine, sim):
    """Reduced geometric metrics from fixed-boundary surfaces.

    V', g0, g1, g2, g3 need to be obtained from a Grad-Shafranov equilibrium.
    Retain the original calibrated Miller/moment closure.  These quantities are
    transport/current-diffusion proxies rather than exact GS metrics:
      g0 ~ V'/a, g1 ~ (V'/a)^2,
      g3 ~ <1/R^2>, g2 ~ g1*g3.
    """
    dV_drho, V = volume_elements(R, jac, theta, rho=rho)
    if getattr(sim, "reduced_geometry_metrics", True):
        Vp = jnp.maximum(dV_drho, 1.0e-12)
        a = jnp.maximum(machine.a, 1.0e-6)
        g0 = Vp / a
        g1 = (Vp / a) ** 2
        g3 = jnp.mean(1.0 / (R**2 + 1.0e-12), axis=1)
        g2 = g1 * g3
    else:
        g0, g1, g2, g3 = _surface_gradient_metrics(
            R, Z, jac, rho, theta, dV_drho, machine
        )

    # Reduced toroidal flux through the LCFS poloidal cross-section.
    # For Miller-like surfaces this is approximately B0*pi*a^2*kappa.
    area_b = jnp.maximum(V[-1] / (2.0 * jnp.pi * jnp.maximum(machine.R0, 1e-6)), 1e-12)
    Phi_b = jnp.maximum(machine.Bt * area_b, 1e-12)
    return dV_drho, V, g0, g1, g2, g3, Phi_b


def psi_from_q_profile(rho, q, machine):
    """Construct a monotonic poloidal-flux proxy from q.

    For fast geometry purposes, use toroidal flux Phi_tor ≈ pi Bt a^2 rho^2
    and q = dPhi_tor/dpsi_pol, hence dpsi/drho = dPhi/drho / q.
    """
    nr = rho.size
    drho = 1.0 / nr
    dPhi_drho = 2.0 * jnp.pi * machine.Bt * machine.a**2 * rho
    dpsi_drho = dPhi_drho / (q + 1e-8)
    psi = jnp.cumsum(dpsi_drho) * drho
    psi = psi - psi[0]
    psi_norm = psi / (psi[-1] + 1e-12)
    return psi_norm, psi

def approximate_B_fields(R, rho, q, psi_pol, machine):
    """Approximate B_phi, B_p and |B| on Miller surfaces.

    Btor = F/R with F=R0*Bt.
    Bpol is estimated from q relation Bp ~ r Bt/(R0 q), then mapped to 2D.
    """
    F = machine.R0 * machine.Bt
    Btor = F / (R + 1e-12)

    r_minor = machine.a * rho
    Bp_1d = r_minor * machine.Bt / (machine.R0 * (q + 1e-8))
    Bp_2d = Bp_1d[:, None] * jnp.ones_like(R)
    Bmag = jnp.sqrt(Btor**2 + Bp_2d**2)
    return F * jnp.ones_like(rho), Btor, Bp_2d, Bmag


def _read_geqdsk_numbers(path):
    """Read a G-EQDSK file into header and numeric token list.

    This parser handles standard EFIT/G-EQDSK fixed-width scientific notation,
    including Fortran D exponents. It is intentionally dependency-light.
    """
    import re
    with open(path, "r") as f:
        lines = f.readlines()
    header = lines[0].rstrip("\n") if lines else ""
    # Most g-files end the first line with three integers: idum, nw, nh.
    ints = re.findall(r"[-+]?\d+", header)
    if len(ints) < 2:
        raise ValueError(f"Could not infer nw, nh from first G-EQDSK line: {header!r}")
    nw = int(ints[-2])
    nh = int(ints[-1])

    text = "".join(lines[1:]).replace("D", "E").replace("d", "E")
    nums = [float(x) for x in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?", text)]
    return header, nw, nh, nums


def _polygon_area_np(r, z):
    """Signed polygon area in the R-Z plane."""
    import numpy as np
    if len(r) < 3:
        return 0.0
    return 0.5 * float(np.sum(r * np.roll(z, -1) - np.roll(r, -1) * z))


def _point_in_polygon_np(x, y, r, z):
    """Ray-casting point-in-polygon test without optional dependencies."""
    import numpy as np
    r = np.asarray(r)
    z = np.asarray(z)
    inside = False
    j = len(r) - 1
    for i in range(len(r)):
        zi, zj = z[i], z[j]
        ri, rj = r[i], r[j]
        crosses = ((zi > y) != (zj > y)) and (x < (rj - ri) * (y - zi) / (zj - zi + 1.0e-300) + ri)
        if crosses:
            inside = not inside
        j = i
    return inside


def _detect_geqdsk_xpoint_flux(g):
    """Estimate diverted-boundary flux from a saddle point of psi(R,Z).

    The standard EFIT ``sibry`` value is usually already the X-point/separatrix
    flux.  Some processed files, however, omit the boundary point list while
    still carrying a full ``psirz`` grid.  This routine scans the 2D flux grid
    for a saddle-like point near normalized flux one and returns its local flux
    value.  If a reliable saddle is not found, the caller should fall back to
    ``sibry``.
    """
    import numpy as np

    R = np.asarray(g["R_grid"])
    Z = np.asarray(g["Z_grid"])
    psi = np.asarray(g["psirz"])
    simag = float(g["simag"])
    sibry = float(g["sibry"])
    span = sibry - simag
    if psi.size == 0 or abs(span) < 1.0e-14 or R.size < 5 or Z.size < 5:
        return None, None

    dpsi_dz, dpsi_dr = np.gradient(psi, Z, R, edge_order=2)
    d2psi_dz2, d2psi_drdz = np.gradient(dpsi_dz, Z, R, edge_order=2)
    d2psi_dzdr, d2psi_dr2 = np.gradient(dpsi_dr, Z, R, edge_order=2)
    det_h = d2psi_dr2 * d2psi_dz2 - 0.25 * (d2psi_drdz + d2psi_dzdr) ** 2

    psin = (psi - simag) / (span + 1.0e-300)
    RR, ZZ = np.meshgrid(R, Z)
    radius = np.sqrt((RR - float(g["rmaxis"])) ** 2 + (ZZ - float(g["zmaxis"])) ** 2)
    dr_grid = abs(R[1] - R[0]) if R.size > 1 else 0.0
    dz_grid = abs(Z[1] - Z[0]) if Z.size > 1 else 0.0
    min_radius = max(3.0 * max(dr_grid, dz_grid), 0.02)

    grad = np.sqrt(dpsi_dr**2 + dpsi_dz**2)
    finite = np.isfinite(grad) & np.isfinite(psin) & np.isfinite(det_h)
    near_sep = np.abs(psin - 1.0) < 0.35
    saddle = det_h < 0.0
    away_axis = radius > min_radius
    interior = np.zeros_like(finite, dtype=bool)
    interior[1:-1, 1:-1] = True
    mask = finite & near_sep & saddle & away_axis & interior
    if not np.any(mask):
        return None, None

    gscale = np.nanmedian(grad[mask]) + 1.0e-300
    score = grad / gscale + 8.0 * np.abs(psin - 1.0) + 0.05 / (radius + 1.0e-6)
    score = np.where(mask, score, np.inf)
    iz, ir = np.unravel_index(int(np.argmin(score)), score.shape)
    psi_x = float(psi[iz, ir])
    psin_x = float(psin[iz, ir])
    if not np.isfinite(psi_x) or abs(psin_x - 1.0) > 0.45:
        return None, None
    return psi_x, (float(R[ir]), float(Z[iz]), psin_x)


def _extract_geqdsk_contour_boundary(g, level=None):
    """Extract an LCFS-like boundary from the 2D psi contour.

    The chosen contour is the closed contour at the requested flux level that
    encloses the magnetic axis.  This is used when EQDSK omits the explicit
    BBBS/LCFS point list.
    """
    import numpy as np

    if level is None:
        psi_x, xpt = _detect_geqdsk_xpoint_flux(g)
        level = psi_x if psi_x is not None else float(g["sibry"])
    R = np.asarray(g["R_grid"])
    Z = np.asarray(g["Z_grid"])
    psi = np.asarray(g["psirz"])
    if R.size < 2 or Z.size < 2 or psi.shape != (Z.size, R.size):
        return np.asarray([]), np.asarray([]), False

    pmin, pmax = float(np.nanmin(psi)), float(np.nanmax(psi))
    if not (min(pmin, pmax) <= float(level) <= max(pmin, pmax)):
        return np.asarray([]), np.asarray([]), False

    try:
        # Use Matplotlib's object-oriented Agg canvas only for contour generation.
        # Do not call matplotlib.use("Agg", force=True): that changes the global
        # backend and breaks later interactive plt.show() calls in demo scripts.
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        fig = Figure()
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        cs = ax.contour(R, Z, psi, levels=[float(level)])
        segs = cs.allsegs[0] if cs.allsegs else []
        fig.clear()
    except Exception:
        return np.asarray([]), np.asarray([]), False

    candidates = []
    dr_grid = abs(R[1] - R[0]) if R.size > 1 else 0.0
    dz_grid = abs(Z[1] - Z[0]) if Z.size > 1 else 0.0
    close_tol = 3.0 * max(dr_grid, dz_grid, 1.0e-12)
    axis = (float(g["rmaxis"]), float(g["zmaxis"]))
    for seg in segs:
        seg = np.asarray(seg)
        if seg.ndim != 2 or seg.shape[0] < 8 or seg.shape[1] != 2:
            continue
        r = seg[:, 0]
        z = seg[:, 1]
        if np.hypot(r[0] - r[-1], z[0] - z[-1]) > close_tol:
            r = np.concatenate([r, r[:1]])
            z = np.concatenate([z, z[:1]])
        if not _point_in_polygon_np(axis[0], axis[1], r, z):
            continue
        area = abs(_polygon_area_np(r, z))
        length = float(np.sum(np.hypot(np.diff(r), np.diff(z))))
        if area <= 0.0 or length <= 0.0:
            continue
        candidates.append((area, length, r, z))

    if not candidates:
        # Last-resort fallback: pick the largest closed contour.  This keeps the
        # parser useful for unusual files but marks the extraction as weak.
        for seg in segs:
            seg = np.asarray(seg)
            if seg.ndim != 2 or seg.shape[0] < 8 or seg.shape[1] != 2:
                continue
            r = seg[:, 0]
            z = seg[:, 1]
            if np.hypot(r[0] - r[-1], z[0] - z[-1]) > close_tol:
                continue
            area = abs(_polygon_area_np(r, z))
            length = float(np.sum(np.hypot(np.diff(r), np.diff(z))))
            if area > 0.0 and length > 0.0:
                candidates.append((area, length, r, z))
    if not candidates:
        return np.asarray([]), np.asarray([]), False

    # For a separatrix-level contour, the axis-enclosing LCFS should normally be
    # the largest valid closed contour around the magnetic axis.
    area, length, r, z = max(candidates, key=lambda t: t[0])
    return np.asarray(r), np.asarray(z), True


def read_geqdsk(path):
    """Parse standard EFIT G-EQDSK/g-file.

    Returns a dict with core geometry, 1D profiles and 2D poloidal flux grid.
    The parser covers the conventional EFIT order:
      rdim,zdim,rcentr,rleft,zmid; rmaxis,zmaxis,simag,sibry,bcentr;
      current, simag, xdum, rmaxis, xdum; zmaxis, xdum, sibry, xdum, xdum;
      fpol, pres, ffprim, pprime, psirz, qpsi, boundary, limiter.
    """
    header, nw, nh, nums = _read_geqdsk_numbers(path)
    if len(nums) < 20 + 5 * nw + nw * nh:
        raise ValueError(
            f"G-EQDSK numeric data too short: got {len(nums)}, expected at least {20 + 5*nw + nw*nh}"
        )

    k = 0
    rdim, zdim, rcentr, rleft, zmid = nums[k:k+5]; k += 5
    rmaxis, zmaxis, simag, sibry, bcentr = nums[k:k+5]; k += 5
    # Standard G-EQDSK has exactly 20 scalar values before the 1-D profiles:
    #   line 1: rdim,zdim,rcentr,rleft,zmid
    #   line 2: rmaxis,zmaxis,simag,sibry,bcentr
    #   line 3: current,simag,xdum,rmaxis,xdum
    #   line 4: zmaxis,xdum,sibry,xdum,xdum
    # The previous parser skipped 25 scalars here, which shifted fpol/pres/... by
    # five entries.  At high nr this contaminated the edge F=RBphi interpolation
    # with pressure-like values (O(1e5--1e6)), causing pathological current
    # diffusion and edge Ohmic heating.
    current = nums[k]
    k += 5  # consume the rest of scalar line 3
    k += 5  # consume scalar line 4

    import numpy as np
    fpol = np.asarray(nums[k:k+nw]); k += nw
    pres = np.asarray(nums[k:k+nw]); k += nw
    ffprim = np.asarray(nums[k:k+nw]); k += nw
    pprime = np.asarray(nums[k:k+nw]); k += nw
    psirz = np.asarray(nums[k:k+nw*nh]).reshape((nh, nw)); k += nw * nh
    qpsi = np.asarray(nums[k:k+nw]); k += nw

    nbbbs = int(round(nums[k])) if k < len(nums) else 0
    limitr = int(round(nums[k+1])) if k + 1 < len(nums) else 0
    k += 2
    rbbbs = zbbbs = rlim = zlim = np.asarray([])
    if nbbbs > 0 and k + 2 * nbbbs <= len(nums):
        bd = np.asarray(nums[k:k+2*nbbbs]).reshape((nbbbs, 2)); k += 2 * nbbbs
        rbbbs, zbbbs = bd[:, 0], bd[:, 1]
    if limitr > 0 and k + 2 * limitr <= len(nums):
        lm = np.asarray(nums[k:k+2*limitr]).reshape((limitr, 2)); k += 2 * limitr
        rlim, zlim = lm[:, 0], lm[:, 1]

    R_grid = rleft + np.arange(nw) * rdim / max(nw - 1, 1)
    Z_grid = zmid - 0.5 * zdim + np.arange(nh) * zdim / max(nh - 1, 1)
    psin_1d = np.linspace(0.0, 1.0, nw)

    boundary_source = "bbbs" if len(rbbbs) >= 4 else "missing"
    psi_boundary_level = sibry
    xpoint = None
    if len(rbbbs) < 4:
        # Build a temporary dictionary so the contour/X-point routines can use
        # the parsed psi grid before the final return object is assembled.
        gtmp = {
            "R_grid": R_grid,
            "Z_grid": Z_grid,
            "psirz": psirz,
            "simag": simag,
            "sibry": sibry,
            "rmaxis": rmaxis,
            "zmaxis": zmaxis,
        }
        psi_x, xpoint = _detect_geqdsk_xpoint_flux(gtmp)
        level = psi_x if psi_x is not None else sibry
        psi_boundary_level = level
        rct, zct, ok = _extract_geqdsk_contour_boundary(gtmp, level=level)
        if ok and len(rct) >= 4:
            rbbbs, zbbbs = rct, zct
            boundary_source = "psirz_xpoint_contour" if psi_x is not None else "psirz_sibry_contour"

    return {
        "header": header,
        "nw": nw,
        "nh": nh,
        "rdim": rdim,
        "zdim": zdim,
        "rcentr": rcentr,
        "rleft": rleft,
        "zmid": zmid,
        "rmaxis": rmaxis,
        "zmaxis": zmaxis,
        "simag": simag,
        "sibry": sibry,
        "bcentr": bcentr,
        "current": current,
        "R_grid": R_grid,
        "Z_grid": Z_grid,
        "psirz": psirz,
        "psin_1d": psin_1d,
        "psi_boundary_level": psi_boundary_level,
        "fpol": fpol,
        "pres": pres,
        "ffprim": ffprim,
        "pprime": pprime,
        "qpsi": qpsi,
        "rbbbs": rbbbs,
        "zbbbs": zbbbs,
        "boundary_source": boundary_source,
        "xpoint": xpoint,
        "rlim": rlim,
        "zlim": zlim,
    }


_GEQDSK_CACHE = {}


def get_cached_geqdsk(path):
    path = str(path)
    if path not in _GEQDSK_CACHE:
        _GEQDSK_CACHE[path] = read_geqdsk(path)
    return _GEQDSK_CACHE[path]


def _periodic_interp_boundary(rbd, zbd, theta, raxis, zaxis, a_fallback, kappa_fallback):
    """Map boundary points to theta and interpolate R,Z boundary at target theta."""
    import numpy as np
    if len(rbd) < 4:
        rb = raxis + a_fallback * np.cos(theta)
        zb = zaxis + kappa_fallback * a_fallback * np.sin(theta)
        return rb, zb
    ang = np.arctan2(zbd - zaxis, rbd - raxis)
    ang = np.mod(ang, 2.0 * np.pi)
    order = np.argsort(ang)
    ang = ang[order]
    r = rbd[order]
    z = zbd[order]
    # remove duplicate angles
    keep = np.concatenate([[True], np.diff(ang) > 1e-6])
    ang, r, z = ang[keep], r[keep], z[keep]
    ang_ext = np.concatenate([ang - 2.0 * np.pi, ang, ang + 2.0 * np.pi])
    r_ext = np.concatenate([r, r, r])
    z_ext = np.concatenate([z, z, z])
    return np.interp(theta, ang_ext, r_ext), np.interp(theta, ang_ext, z_ext)


def _jnp_interp1(x, xp, fp):
    """JAX 1D interpolation wrapper."""
    return jnp.interp(x, jnp.asarray(xp), jnp.asarray(fp), left=jnp.asarray(fp)[0], right=jnp.asarray(fp)[-1])


def _normalize_theta_alignment(mode):
    """Normalize user-facing theta-grid alignment strings."""
    mode = str(mode or "auto").lower()
    aliases = {
        "default": "auto",
        "uniform": "cardinal",
        "cardinal_uniform": "cardinal",
        "miller": "cardinal",
        "xpoint": "lower_xpoint",
        "lower_x": "lower_xpoint",
        "lower-x": "lower_xpoint",
        "lower_x_point": "lower_xpoint",
        "bottom": "lower_xpoint",
        "min_z": "lower_xpoint",
        "min-z": "lower_xpoint",
    }
    return aliases.get(mode, mode)


def _uniform_theta_grid_np(ntheta, anchor=None):
    """Uniform periodic theta grid, optionally shifted to include ``anchor``.

    The returned array is sorted in [0, 2π) and has exactly uniform spacing in
    the periodic sense.  Keeping theta uniform is important because the reduced
    metric and Jacobian routines use periodic finite differences and rectangle
    sums in theta.
    """
    import numpy as np
    n = max(int(ntheta), 4)
    if anchor is None:
        return np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=float)
    dtheta = 2.0 * np.pi / float(n)
    theta = np.mod(float(anchor) + dtheta * np.arange(n, dtype=float), 2.0 * np.pi)
    theta.sort()
    return theta


def _uniform_theta_grid_jax(ntheta, anchor=None):
    """JAX counterpart of _uniform_theta_grid_np for fixed-boundary geometry."""
    n = max(int(ntheta), 4)
    if anchor is None:
        return jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    dtheta = 2.0 * jnp.pi / float(n)
    theta = jnp.mod(jnp.asarray(anchor) + dtheta * jnp.arange(n), 2.0 * jnp.pi)
    return jnp.sort(theta)


def fixed_boundary_theta_grid(sim):
    """Theta grid for reduced fixed-boundary Miller geometry.

    The default cardinal grid starts at theta=0.  For ntheta divisible by four
    (including the default ntheta=16), it contains the outboard, top, inboard,
    and bottom Miller points exactly.
    """
    mode = _normalize_theta_alignment(getattr(sim, "theta_grid_alignment", "auto"))
    # A fixed-boundary Miller shape has its cardinal extrema at 0, π/2, π, 3π/2.
    # Use the cardinal grid in auto mode.  A lower-X alignment request is ignored
    # here because the reduced Miller shape does not carry an X-point.
    return _uniform_theta_grid_jax(getattr(sim, "ntheta", 16), anchor=None)


def _geqdsk_lower_anchor_theta_np(g):
    """Return polar angle of the lower X-point, falling back to min-Z LCFS point."""
    import numpy as np
    raxis = float(g.get("rmaxis", 0.0))
    zaxis = float(g.get("zmaxis", 0.0))

    xpt = g.get("xpoint", None)
    if xpt is not None and len(xpt) >= 2:
        rx, zx = float(xpt[0]), float(xpt[1])
        if np.isfinite(rx) and np.isfinite(zx) and zx <= zaxis:
            return float(np.mod(np.arctan2(zx - zaxis, rx - raxis), 2.0 * np.pi))

    rbd = np.asarray(g.get("rbbbs", []), dtype=float)
    zbd = np.asarray(g.get("zbbbs", []), dtype=float)
    if rbd.size >= 4 and zbd.size == rbd.size:
        finite = np.isfinite(rbd) & np.isfinite(zbd)
        if np.any(finite):
            idxs = np.nonzero(finite)[0]
            i = idxs[int(np.argmin(zbd[finite]))]
            return float(np.mod(np.arctan2(zbd[i] - zaxis, rbd[i] - raxis), 2.0 * np.pi))

    # Analytic Miller fallback bottom point.
    return 1.5 * np.pi


def _geqdsk_theta_grid_np(g, machine, sim):
    """Theta grid for prescribed GEQDSK surfaces.

    Exact inclusion of top/bottom/inboard/outboard extrema for an arbitrary
    extracted LCFS would require a non-uniform theta grid.  The geometry code
    currently assumes uniform theta spacing for periodic finite differences and
    flux-surface sums, so in auto mode we use the user's requested fallback:
    include the lower X-point/min-Z boundary point exactly and distribute all
    other points uniformly around the poloidal angle.
    """
    mode = _normalize_theta_alignment(getattr(sim, "theta_grid_alignment", "auto"))
    if mode in ("auto", "lower_xpoint"):
        anchor = _geqdsk_lower_anchor_theta_np(g)
        return _uniform_theta_grid_np(getattr(sim, "ntheta", 16), anchor=anchor)
    if mode == "cardinal":
        return _uniform_theta_grid_np(getattr(sim, "ntheta", 16), anchor=None)
    raise ValueError(
        f"Unknown theta_grid_alignment={getattr(sim, 'theta_grid_alignment', None)!r}. "
        "Use 'auto', 'cardinal', or 'lower_xpoint'."
    )


def _rho_centers_np(nr, radial_grid="uniform", edge_cluster_power=2.0):
    """NumPy copy of the static rho-grid construction used for cached GEQDSK surfaces."""
    import numpy as np
    x = np.linspace(0.0, 1.0, int(nr) + 1)
    if radial_grid in ("uniform", "linear"):
        faces = x
    elif radial_grid in ("edge_cluster_sqrt", "edge_clustered", "sqrt_edge"):
        p = float(edge_cluster_power)
        faces = np.sqrt(np.maximum(1.0 - (1.0 - x) ** p, 0.0))
    else:
        raise ValueError(f"Unknown radial_grid={radial_grid!r}")
    return 0.5 * (faces[:-1] + faces[1:])


def _resample_curve_by_theta_np(r, z, theta, raxis, zaxis, a_fallback=1.0, kappa_fallback=1.0):
    """Resample a closed R-Z curve at requested polar angles around the magnetic axis."""
    import numpy as np
    rb, zb = _periodic_interp_boundary(
        np.asarray(r), np.asarray(z), np.asarray(theta),
        float(raxis), float(zaxis), float(a_fallback), float(kappa_fallback),
    )
    return np.asarray(rb), np.asarray(zb)


def _axis_hessian_flux_surface_np(g, psin, theta, machine):
    """Local near-axis flux surface from the Hessian of psirz at the magnetic axis.

    This is used only when a requested very-small flux contour is not resolved by
    the GEQDSK mesh.  It avoids imposing the edge elongation/triangularity all the
    way to rho~0 and gives a smooth regular magnetic-axis limit.
    """
    import numpy as np

    Rg = np.asarray(g["R_grid"])
    Zg = np.asarray(g["Z_grid"])
    psi = np.asarray(g["psirz"])
    raxis = float(g["rmaxis"])
    zaxis = float(g["zmaxis"])
    simag = float(g["simag"])
    psib = float(g.get("psi_boundary_level", g["sibry"]))
    dpsi = abs(float(psin) * (psib - simag))
    if Rg.size < 3 or Zg.size < 3 or psi.shape != (Zg.size, Rg.size) or dpsi <= 0.0:
        rr = float(machine.a) * np.sqrt(max(float(psin), 0.0))
        return raxis + rr * np.cos(theta), zaxis + rr * np.sin(theta)

    ir = int(np.argmin(np.abs(Rg - raxis)))
    iz = int(np.argmin(np.abs(Zg - zaxis)))
    ir = min(max(ir, 1), Rg.size - 2)
    iz = min(max(iz, 1), Zg.size - 2)
    dr = float(Rg[ir + 1] - Rg[ir - 1]) * 0.5
    dz = float(Zg[iz + 1] - Zg[iz - 1]) * 0.5
    if abs(dr) <= 0.0 or abs(dz) <= 0.0:
        rr = float(machine.a) * np.sqrt(max(float(psin), 0.0))
        return raxis + rr * np.cos(theta), zaxis + rr * np.sin(theta)

    d2rr = (psi[iz, ir + 1] - 2.0 * psi[iz, ir] + psi[iz, ir - 1]) / (dr * dr)
    d2zz = (psi[iz + 1, ir] - 2.0 * psi[iz, ir] + psi[iz - 1, ir]) / (dz * dz)
    d2rz = (
        psi[iz + 1, ir + 1] - psi[iz + 1, ir - 1]
        - psi[iz - 1, ir + 1] + psi[iz - 1, ir - 1]
    ) / (4.0 * dr * dz)
    H = np.asarray([[d2rr, d2rz], [d2rz, d2zz]], dtype=float)
    # Make the axis extremum positive definite in the direction from axis to LCFS.
    H = H * (1.0 if (psib - simag) >= 0.0 else -1.0)
    try:
        evals, evecs = np.linalg.eigh(H)
    except Exception:
        evals = np.asarray([np.nan, np.nan])
        evecs = np.eye(2)
    if (not np.all(np.isfinite(evals))) or np.min(evals) <= 1.0e-30:
        rr = float(machine.a) * np.sqrt(max(float(psin), 0.0))
        return raxis + rr * np.cos(theta), zaxis + rr * np.sin(theta)

    rad = np.sqrt(2.0 * dpsi / np.maximum(evals, 1.0e-30))
    xy = evecs @ np.vstack([rad[0] * np.cos(theta), rad[1] * np.sin(theta)])
    return raxis + xy[0], zaxis + xy[1]


def _axis_to_boundary_surfaces_np(g, rho, theta, machine):
    """Old GEQDSK geometry: straight interpolation from magnetic axis to LCFS."""
    import numpy as np
    rbd = np.asarray(g.get("rbbbs", []))
    zbd = np.asarray(g.get("zbbbs", []))
    if len(rbd) < 4:
        thb = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
        rbd = float(g["rmaxis"]) + machine.a * np.cos(thb + machine.delta * np.sin(thb))
        zbd = float(g["zmaxis"]) + machine.kappa * machine.a * np.sin(thb)
    rb, zb = _resample_curve_by_theta_np(
        rbd, zbd, theta, g["rmaxis"], g["zmaxis"], machine.a, machine.kappa
    )
    R = float(g["rmaxis"]) + rho[:, None] * (rb[None, :] - float(g["rmaxis"]))
    Z = float(g["zmaxis"]) + rho[:, None] * (zb[None, :] - float(g["zmaxis"]))
    return R, Z


def _geqdsk_psirz_contour_surfaces_np(
    g, rho, theta, machine, inner_fallback="axis_hessian", lcfs_margin=0.0
):
    """Build cell-centered nested flux surfaces from GEQDSK psirz contours.

    ``rho`` is the transport-grid coordinate and is normally cell-centered.
    The returned R/Z surfaces must therefore correspond to the *same* rho
    locations.  Earlier versions rescaled the outermost cell-centered surface
    to the LCFS so coarse plots looked nicer, but then ``dV/drho`` was evaluated
    at a boundary-like surface and later integrated again from the last cell
    center to rho=1.  For small ``nr`` this double-counted the edge shell and
    made volume-based quantities, especially 0D stored-energy reconstruction
    and P_fus, resolution dependent.

    The physical LCFS is still available from ``rbbbs/zbbbs`` and is overlaid
    separately in plotting.  Geometry metrics and volume/surface integrals use
    the true cell-centered contours here.  ``lcfs_margin`` can still move the
    prescribed boundary slightly inward for numerical robustness, but it does
    not remap the grid so the outer cell center remains an outer cell center.
    """
    import numpy as np

    rho = np.asarray(rho, dtype=float)
    theta = np.asarray(theta, dtype=float)
    simag = float(g["simag"])
    psib = float(g.get("psi_boundary_level", g["sibry"]))
    margin = float(max(lcfs_margin, 0.0))
    rho_contour = np.clip(rho * max(1.0 - margin, 1.0e-6), 0.0, 1.0)
    psin_targets = np.clip(rho_contour**2, 0.0, 1.0)

    # Always have an LCFS representation for the outer fallback and for filling
    # the last level if the separatrix contour is numerically awkward.
    R_old, Z_old = _axis_to_boundary_surfaces_np(g, rho, theta, machine)
    rb_old = R_old[-1]
    zb_old = Z_old[-1]

    R_list = []
    Z_list = []
    ok = []
    for psin in psin_targets:
        level = simag + float(psin) * (psib - simag)
        # Use the actual psirz contour up to the separatrix whenever possible.
        # The tabulated rbbbs boundary can be slightly inside/outside the
        # extracted psirz separatrix.  For edge-clustered rho grids, a
        # near-LCFS cell/face can otherwise have a larger enclosed volume than
        # rbbbs, and the finite-volume volume correction collapses the last
        # cell volume to zero.
        rct, zct, good = _extract_geqdsk_contour_boundary(g, level=level)
        if (not good) and psin >= 1.0 - 1.0e-10:
            rct, zct = np.asarray(g.get("rbbbs", [])), np.asarray(g.get("zbbbs", []))
            good = len(rct) >= 4
        if good and len(rct) >= 4:
            rb, zb = _resample_curve_by_theta_np(
                rct, zct, theta, g["rmaxis"], g["zmaxis"], machine.a, machine.kappa
            )
            R_list.append(rb)
            Z_list.append(zb)
            ok.append(True)
        else:
            R_list.append(np.full_like(theta, np.nan, dtype=float))
            Z_list.append(np.full_like(theta, np.nan, dtype=float))
            ok.append(False)

    R = np.vstack(R_list)
    Z = np.vstack(Z_list)
    ok = np.asarray(ok, dtype=bool)

    # Fill unresolved inner surfaces without imposing the LCFS triangularity.
    for i, psin in enumerate(psin_targets):
        if ok[i]:
            continue
        if inner_fallback == "axis_hessian":
            R[i], Z[i] = _axis_hessian_flux_surface_np(g, psin, theta, machine)
            ok[i] = True
        elif inner_fallback == "nearest_contour" and np.any(ok):
            outer = np.where(ok & (psin_targets > max(psin, 0.0)))[0]
            ref = int(outer[0]) if outer.size else int(np.where(ok)[0][0])
            scale = np.sqrt((psin + 1.0e-12) / (psin_targets[ref] + 1.0e-12))
            R[i] = float(g["rmaxis"]) + scale * (R[ref] - float(g["rmaxis"]))
            Z[i] = float(g["zmaxis"]) + scale * (Z[ref] - float(g["zmaxis"]))
            ok[i] = True
        else:
            # Last-resort old behavior, but only for this failed contour.
            R[i], Z[i] = R_old[i], Z_old[i]
            ok[i] = True

    # Guard against rare contour-pathology NaNs.
    bad = ~(np.all(np.isfinite(R), axis=1) & np.all(np.isfinite(Z), axis=1))
    if np.any(bad):
        R[bad] = R_old[bad]
        Z[bad] = Z_old[bad]
    return R, Z


def _closed_curve_toroidal_volume_np(Rb, Zb):
    """Axisymmetric volume enclosed by a closed poloidal R-Z contour.

    Uses Pappus' centroid theorem: V = 2*pi*R_centroid*A_poloidal.
    The curve is assumed to enclose the magnetic axis and to be ordered
    periodically.  Degenerate inner surfaces return zero volume.
    """
    import numpy as np
    Rb = np.asarray(Rb, dtype=float)
    Zb = np.asarray(Zb, dtype=float)
    if Rb.size < 3 or Zb.size < 3:
        return 0.0
    Rn = np.roll(Rb, -1)
    Zn = np.roll(Zb, -1)
    cross = Rb * Zn - Rn * Zb
    A_signed = 0.5 * np.sum(cross)
    if not np.isfinite(A_signed) or abs(A_signed) < 1.0e-14:
        return 0.0
    R_cent = np.sum((Rb + Rn) * cross) / (6.0 * A_signed)
    A = abs(A_signed)
    return float(max(2.0 * np.pi * max(R_cent, 0.0) * A, 0.0))


_GEQDSK_SURFACE_CACHE = {}


def get_cached_geqdsk_surfaces(path, machine, sim):
    """Return cached R(rho,theta), Z(rho,theta) surfaces for prescribed GEQDSK mode."""
    import numpy as np
    path = str(path)
    mode = getattr(sim, "geqdsk_surface_geometry", "psirz_contours")
    radial_grid = getattr(sim, "radial_grid", "uniform")
    edge_cluster_power = float(getattr(sim, "edge_cluster_power", 2.0))
    # Do not include machine numeric values in the cache key.  In prescribed
    # GEQDSK mode the surfaces come from the file, and optimizer controls may be
    # JAX tracers.  Calling float(machine.a) here would break reverse-mode AD.
    key = (
        path, int(sim.nr), int(sim.ntheta), radial_grid, edge_cluster_power,
        _normalize_theta_alignment(getattr(sim, "theta_grid_alignment", "auto")),
        mode, getattr(sim, "geqdsk_inner_contour_fallback", "axis_hessian"),
        float(getattr(sim, "geqdsk_lcfs_margin", 0.0)),
    )
    if key in _GEQDSK_SURFACE_CACHE:
        return _GEQDSK_SURFACE_CACHE[key]

    g = get_cached_geqdsk(path)
    rho = _rho_centers_np(int(sim.nr), radial_grid, edge_cluster_power)
    theta = _geqdsk_theta_grid_np(g, machine, sim)
    cell_volumes = None
    if mode in ("axis_to_boundary", "straight_line", "boundary_scaled"):
        R, Z = _axis_to_boundary_surfaces_np(g, rho, theta, machine)
        source = "axis_to_boundary"
        # For straight axis-to-boundary surfaces, analytic/midpoint V' is already
        # well behaved enough; no separate face-volume correction is needed.
    elif mode in ("psirz_contours", "contours", "flux_contours"):
        inner_fallback = getattr(sim, "geqdsk_inner_contour_fallback", "axis_hessian")
        margin = float(getattr(sim, "geqdsk_lcfs_margin", 0.0))
        R, Z = _geqdsk_psirz_contour_surfaces_np(
            g, rho, theta, machine,
            inner_fallback=inner_fallback,
            lcfs_margin=margin,
        )
        # Build finite-volume shell volumes from the actual cell faces, including
        # rho=1/LCFS.  This is crucial at small nr: using only center V'(rho) can
        # miss a large part of the edge shell, while earlier remapping of the
        # outer center to the LCFS double-counted that shell.
        faces = np.asarray([0.0, *list(0.5 * (rho[:-1] + rho[1:])), 1.0], dtype=float)
        Rf, Zf = _geqdsk_psirz_contour_surfaces_np(
            g, faces, theta, machine,
            inner_fallback=inner_fallback,
            lcfs_margin=margin,
        )
        V_faces = np.asarray([_closed_curve_toroidal_volume_np(Rf[i], Zf[i]) for i in range(faces.size)], dtype=float)
        V_faces[0] = 0.0
        # Numerical contour/pathology guard: enforce monotonic enclosed volume.
        V_faces = np.maximum.accumulate(np.maximum(V_faces, 0.0))
        cell_volumes = np.maximum(V_faces[1:] - V_faces[:-1], 0.0)
        source = "psirz_contours"
    else:
        raise ValueError(
            f"Unknown geqdsk_surface_geometry={mode!r}. Use 'psirz_contours' or 'axis_to_boundary'."
        )
    out = {"rho": rho, "theta": theta, "R": R, "Z": Z, "surface_source": source}
    if cell_volumes is not None:
        out["cell_volumes"] = cell_volumes
    _GEQDSK_SURFACE_CACHE[key] = out
    return out


def geqdsk_prescribed_equilibrium(state, machine, actuator, sim):
    """Build a TokaGrad Equilibrium from a prescribed G-EQDSK file.

    This version is JAX-compatible inside `lax.scan`: all interpolation and
    geometry construction after loading the cached EQDSK is done with jnp
    arrays, not NumPy conversions of traced arrays.
    """
    if not getattr(sim, "geqdsk_path", ""):
        raise ValueError('sim.geqdsk_path must be set when equilibrium_model="geqdsk_prescribed".')

    g = get_cached_geqdsk(sim.geqdsk_path)

    # In GEQDSK mode, build flux surfaces from the 2D psirz contours by default.
    # The previous behavior, straight interpolation from magnetic axis to the LCFS,
    # is still available with geqdsk_surface_geometry="axis_to_boundary".
    surfaces = get_cached_geqdsk_surfaces(sim.geqdsk_path, machine, sim)
    rho = jnp.asarray(surfaces["rho"])
    theta = jnp.asarray(surfaces["theta"])
    R = jnp.asarray(surfaces["R"])
    Z = jnp.asarray(surfaces["Z"])
    rbbbs, zbbbs = jnp.asarray(g["rbbbs"]), jnp.asarray(g["zbbbs"])

    psin = jnp.clip(rho**2, 0.0, 1.0)

    jac = numerical_jacobian(R, Z, rho, theta)
    dV_drho, V, g0, g1, g2, g3, Phi_b = reduced_flux_metrics(R, Z, jac, theta, rho, machine, sim)

    # If available, use finite-volume shell volumes computed from actual GEQDSK
    # face contours.  This makes ``sum(dV_drho * Δrho)`` include the LCFS volume
    # even when ``nr`` is small, without pretending that the last cell-center
    # surface is itself the LCFS.  Downstream code already converts V'(rho) to
    # cell volumes through ``volume_element_from_dV_drho``.
    if "cell_volumes" in surfaces:
        dV_cell = jnp.asarray(surfaces["cell_volumes"], dtype=rho.dtype)
        rho_f = infer_rho_faces(rho)
        drho_cell = jnp.maximum(rho_f[1:] - rho_f[:-1], 1.0e-12)
        dV_drho = dV_cell / drho_cell
        V_faces = jnp.concatenate([jnp.zeros(1, dtype=rho.dtype), jnp.cumsum(dV_cell)])
        V = V_faces[:-1] + 0.5 * dV_cell
        a_eff = jnp.maximum(machine.a, 1.0e-6)
        g0 = jnp.maximum(dV_drho, 1.0e-12) / a_eff
        g1 = g0 * g0
        g3 = jnp.mean(1.0 / (R * R + 1.0e-12), axis=1)
        g2 = g1 * g3
        area_b = jnp.maximum(V_faces[-1] / (2.0 * jnp.pi * jnp.maximum(machine.R0, 1.0e-6)), 1.0e-12)
        Phi_b = jnp.maximum(machine.Bt * area_b, 1.0e-12)

    q_raw = _jnp_interp1(psin, g["psin_1d"], g["qpsi"])
    # Some processed EQDSK/JINTRAC files contain pathological q values outside
    # the LCFS or sign-convention artifacts.  Clip for reduced-model use.
    q_eqdsk = jnp.clip(jnp.abs(q_raw), 0.1, 20.0)

    F = _jnp_interp1(psin, g["psin_1d"], g["fpol"])
    if getattr(sim, "geqdsk_use_pressure", False):
        p = _jnp_interp1(psin, g["psin_1d"], g["pres"])
    else:
        p = pressure_profile(state)

    psi_norm = psin
    psi_pol = g["simag"] + psin * (g["sibry"] - g["simag"])
    g2g3_over_rho = g2 * g3 / jnp.maximum(rho, 1.0e-4)
    
    # Current integrals use the actual shell areas and current diffusion uses
    # the reduced flux metrics.  Deliberately omit q so q_profile() reconstructs
    # it from evolving current with the circular-plus-shape-factor convention
    # rather than reusing qpsi.
    eq_area = type("EqGeom", (), {
        "theta": theta,
        "jac": jac,
        "R": R,
        "F": F,
        "Phi_b": Phi_b,
        "g2": g2,
        "g3": g3,
        "g2g3_over_rho": g2g3_over_rho,
        "dV_drho": dV_drho,
    })()

    q_source = str(getattr(sim, "geqdsk_q_profile_source", "eqdsk")).lower()
    use_current_q = q_source in ("current", "current_density", "psi", "from_current", "evolving")

    def _metric_with_q(q_in):
        return type("EqMetric", (), {
            "theta": theta,
            "jac": jac,
            "R": R,
            "F": F,
            "Phi_b": Phi_b,
            "g2": g2,
            "g3": g3,
            "g2g3_over_rho": g2g3_over_rho,
            "dV_drho": dV_drho,
            "q": q_in,
        })()

    if use_current_q:
        # Start from the EFIT qpsi interpolation for Sauter/bootstrap closures,
        # reconstruct q from the resulting total current, and repeat a small
        # Picard pass.  This mirrors the fixed-boundary equilibrium path while
        # retaining the prescribed GEQDSK flux-surface geometry and F profile.
        q = q_eqdsk
        eq_metric = _metric_with_q(q)
        j_ind, j_bs, j_cd, j_total = current_components_from_state(
            rho, state, machine, actuator, sim, eq=eq_metric, q=q
        )
        q = jnp.clip(q_profile(rho, j_total, machine, sim, eq=eq_area), 0.1, 20.0)
        eq_metric = _metric_with_q(q)
        j_ind, j_bs, j_cd, j_total = current_components_from_state(
            rho, state, machine, actuator, sim, eq=eq_metric, q=q
        )
        q = jnp.clip(q_profile(rho, j_total, machine, sim, eq=eq_area), 0.1, 20.0)
    else:
        q = q_eqdsk
        # Preserve the legacy prescribed-EQDSK behavior: q is the g-file qpsi
        # interpolation, while current components use the same q in closures.
        j_ind, j_bs, j_cd, j_total = current_components_from_state(
            rho, state, machine, actuator, sim, q=q
        )

    Btor = F[:, None] / (R + 1e-12)
    r_minor = jnp.maximum(machine.a * rho, 1e-4)
    Bp_1d = r_minor * jnp.maximum(machine.Bt, 0.1) / (
        jnp.maximum(machine.R0, 0.1) * (q + 1e-8)
    )
    Bpol = Bp_1d[:, None] * jnp.ones_like(R)
    Bmag = jnp.sqrt(Btor**2 + Bpol**2)

    I_total = total_current(rho, j_total, machine, eq=eq_area)
    Ienc = enclosed_current_centers(rho, j_total, machine, eq=eq_area)
    beta_p, li = estimate_beta_p_and_li(rho, pressure_profile(state), j_total, machine)

    return Equilibrium(
        rho=rho,
        theta=theta,
        R=R,
        Z=Z,
        jac=jac,
        dV_drho=dV_drho,
        V=V,
        g0=g0,
        g1=g1,
        g2=g2,
        g3=g3,
        g2g3_over_rho=g2g3_over_rho,
        Phi_b=Phi_b,
        psi_norm=psi_norm,
        psi_pol=psi_pol,
        q=q,
        p=p,
        F=F,
        Btor=Btor,
        Bpol=Bpol,
        Bmag=Bmag,
        j_total=j_total,
        j_ind=j_ind,
        j_bs=j_bs,
        j_cd=j_cd,
        I_total=I_total,
        I_enclosed=Ienc,
        beta_p=beta_p,
        li_proxy=jnp.clip(li, 0.2, 3.0),
        shape_factor=shape_factor(machine, rho),
        rbbbs=rbbbs,
        zbbbs=zbbbs,
    )

def reduced_fixed_boundary_equilibrium(state, machine, actuator, sim):
    """Fast fixed-boundary equilibrium reconstruction.

    Inputs:
      state: Te, Ti, ne, psi_ind
      machine: R0, a, kappa, delta, Bt, Ip
      actuator/sim: current components and geometry settings

    Output:
      Equilibrium object with flux-surface geometry, metric, q, p, B, j.

    This is a fast moment/Miller fixed-boundary equilibrium closure. It is
    suitable for grid generation and qualitative integrated simulations, not
    for replacing EFIT/CHEASE/HELENA/free-boundary GS solvers.
    """
    rho, _, _ = make_grid_from_config(sim.nr, machine.a, sim)
    theta = fixed_boundary_theta_grid(sim)

    j_ind, j_bs, j_cd, j_total = current_components_from_state(
        rho, state, machine, actuator, sim
    )
    p = pressure_profile(state)

    beta_p, li = estimate_beta_p_and_li(rho, p, j_total, machine)
    R, Z = miller_moment_geometry(rho, theta, machine, sim, beta_p=beta_p, li=li)
    jac = numerical_jacobian(R, Z, rho, theta)
    dV_drho, V, g0, g1, g2, g3, Phi_b = reduced_flux_metrics(R, Z, jac, theta, rho, machine, sim)

    if getattr(sim, "torax_circular_psi_geometry", False):
        # TORAX's testing-only circular geometry deliberately uses Phi=pi*B*r^2
        # even for its ad-hoc radially varying elongation.  It also defines the
        # composite current metric directly from V' instead of g2*g3/rho.
        # Keep this opt-in because it is a benchmark convention, not a general
        # Miller/Grad-Shafranov identity.
        Phi_b = jnp.pi * jnp.abs(machine.Bt) * machine.a**2
        g2g3_over_rho = (
            4.0 * jnp.pi**2 * dV_drho * g3
            / jnp.maximum(machine.R0, 1.0e-6)
        )
    else:
        g2g3_over_rho = g2 * g3 / jnp.maximum(rho, 1.0e-4)

    # Use the just-built reduced geometry for current integrals and the
    # calibrated circular-plus-shape-factor q estimate.
    # For psi_diffusion the stored psi is a *total-current* flux,
    # so current reconstruction needs the reduced metric factors even in
    # the fixed-boundary model.  A geometry-only proxy is enough for this Picard
    # pass; q is then recomputed from the resulting total current magnitude.
    F_reduced = machine.R0 * machine.Bt * jnp.ones_like(rho)
    eq_geom = type("EqGeom", (), {
        "theta": theta,
        "jac": jac,
        "R": R,
        "F": F_reduced,
        "Phi_b": Phi_b,
        "g2": g2,
        "g3": g3,
        "g2g3_over_rho": g2g3_over_rho,
        "dV_drho": dV_drho,
    })()
    q = q_profile(rho, j_total, machine, sim, eq=eq_geom)
    eq_metric = type("EqMetric", (), {
        "theta": theta,
        "jac": jac,
        "R": R,
        "F": F_reduced,
        "Phi_b": Phi_b,
        "g2": g2,
        "g3": g3,
        "g2g3_over_rho": g2g3_over_rho,
        "dV_drho": dV_drho,
        "q": q,
    })()
    # One Picard-like consistency pass: once q and the reduced metric are
    # available, use the same geometry in current_from_psi and in Sauter/bootstrap
    # closures rather than falling back to a cylindrical current convention.
    j_ind, j_bs, j_cd, j_total = current_components_from_state(
        rho, state, machine, actuator, sim, eq=eq_metric, q=q
    )
    q = q_profile(rho, j_total, machine, sim, eq=eq_geom)
    eq_metric = type("EqMetric", (), {
        "theta": theta,
        "jac": jac,
        "R": R,
        "F": F_reduced,
        "Phi_b": Phi_b,
        "g2": g2,
        "g3": g3,
        "g2g3_over_rho": g2g3_over_rho,
        "dV_drho": dV_drho,
        "q": q,
    })()
    j_ind, j_bs, j_cd, j_total = current_components_from_state(
        rho, state, machine, actuator, sim, eq=eq_metric, q=q
    )
    q = q_profile(rho, j_total, machine, sim, eq=eq_geom)
    psi_norm, psi_pol = psi_from_q_profile(rho, q, machine)
    F, Btor, Bpol, Bmag = approximate_B_fields(R, rho, q, psi_pol, machine)

    I_total = total_current(rho, j_total, machine, eq=eq_geom)
    Ienc = enclosed_current_centers(rho, j_total, machine, eq=eq_geom)

    return Equilibrium(
        rho=rho,
        theta=theta,
        R=R,
        Z=Z,
        jac=jac,
        dV_drho=dV_drho,
        V=V,
        g0=g0,
        g1=g1,
        g2=g2,
        g3=g3,
        g2g3_over_rho=g2g3_over_rho,
        Phi_b=Phi_b,
        psi_norm=psi_norm,
        psi_pol=psi_pol,
        q=q,
        p=p,
        F=F,
        Btor=Btor,
        Bpol=Bpol,
        Bmag=Bmag,
        j_total=j_total,
        j_ind=j_ind,
        j_bs=j_bs,
        j_cd=j_cd,
        I_total=I_total,
        I_enclosed=Ienc,
        beta_p=beta_p,
        li_proxy=jnp.clip(li, 0.2, 3.0),
        shape_factor=shape_factor(machine, rho),
        rbbbs=R[-1],
        zbbbs=Z[-1],
    )


def solve_fixed_boundary_equilibrium(state, machine, actuator, sim):
    """Dispatch equilibrium model.

    - reduced_fixed_boundary: fast internal Miller/moment closure.
    - geqdsk_prescribed: read prescribed G-EQDSK geometry/q/F profiles.
    """
    model = getattr(sim, "equilibrium_model", "reduced_fixed_boundary")
    if model == "reduced_fixed_boundary":
        return reduced_fixed_boundary_equilibrium(state, machine, actuator, sim)
    if model == "geqdsk_prescribed":
        return geqdsk_prescribed_equilibrium(state, machine, actuator, sim)
    raise ValueError(
        f"Unknown equilibrium_model={model!r}. "
        'Use "reduced_fixed_boundary" or "geqdsk_prescribed".'
    )
