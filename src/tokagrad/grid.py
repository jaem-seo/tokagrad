"""Radial grids, derivatives, and finite-volume integration helpers.

Reference: [S. V. Patankar, Numerical Heat Transfer and Fluid Flow (1980)].
The square-root edge clustering and axis polynomial extrapolation are TokaGrad
grid choices introduced to resolve pedestals on small radial meshes.
"""

import jax.numpy as jnp


def make_rho_faces(nr: int, radial_grid: str = "uniform", edge_cluster_power: float = 2.0):
    """Return radial cell faces in normalized rho.

    radial_grid="uniform": rho_f = x.
    radial_grid="edge_cluster_sqrt": rho_f = sqrt(1 - (1-x)^p).

    The latter clusters cell centers near rho=1.  For p=2, half of the
    uniformly-spaced x interval maps into the outer ~13% of rho, giving much
    better pedestal resolution.
    """
    x = jnp.linspace(0.0, 1.0, nr + 1)
    if radial_grid in ("uniform", "linear"):
        return x
    if radial_grid in ("edge_cluster_sqrt", "edge_clustered", "sqrt_edge"):
        p = edge_cluster_power
        return jnp.sqrt(jnp.maximum(1.0 - (1.0 - x) ** p, 0.0))
    raise ValueError(f"Unknown radial_grid={radial_grid!r}")


def make_grid(nr: int, a: float, radial_grid: str = "uniform", edge_cluster_power: float = 2.0):
    """Cell-centered radial grid.

    Returns
    -------
    rho : [nr]
        Cell-center normalized radius.
    r : [nr]
        Minor radius a*rho.
    dr : scalar or [nr]
        Physical cell width. For uniform grid this remains a scalar for
        backward compatibility; for non-uniform grid it is an array.

    Note
    ----
    The solver state remains cell-centered.  For non-uniform edge-clustered
    grids the first cell center can be far from rho=0; axis values for
    diagnostics/plots and selected derivative stencils use a copied axis value
    y(0)=y(rho[0]) rather than a polynomial extrapolation.
    """
    faces = make_rho_faces(nr, radial_grid, edge_cluster_power)
    rho = 0.5 * (faces[:-1] + faces[1:])
    r = a * rho
    drho = faces[1:] - faces[:-1]
    dr = a * drho
    if radial_grid in ("uniform", "linear"):
        dr = a / nr
    return rho, r, dr


def make_grid_from_config(nr: int, a: float, sim=None):
    if sim is None:
        return make_grid(nr, a)
    return make_grid(
        nr,
        a,
        getattr(sim, "radial_grid", "uniform"),
        getattr(sim, "edge_cluster_power", 2.0),
    )




def _polyfit_axis_value(rho, y, degree=2, n_points=4):
    """Legacy polynomial least-squares extrapolation to rho=0.

    This is retained for debugging/backward comparison, but the default
    tokagrad convention is now axis-copy: y(rho=0) = y[0].  This avoids
    coarse-grid overshoots in plots and scalar diagnostics.
    """
    y = jnp.asarray(y)
    rho = jnp.asarray(rho)
    m = min(int(n_points), int(y.size))
    deg = min(int(degree), max(m - 1, 0))
    rr = rho[:m]
    yy = y[:m]
    if m <= 1 or deg == 0:
        return y[0]
    A = jnp.stack([rr**k for k in range(deg + 1)], axis=1)
    coeff, *_ = jnp.linalg.lstsq(A, yy, rcond=None)
    return coeff[0]


def axis_extrapolated_value(y, rho, method="copy", n_points=4):
    """Return the diagnostic/plot value at rho=0.

    The default is no extrapolation: copy the first cell-centered value.
    Legacy linear/quadratic/cubic extrapolations are retained as explicit
    options for comparison only.
    """
    y = jnp.asarray(y)
    rho = jnp.asarray(rho)
    if y.size < 2:
        return y[0]
    if method in (None, "copy", "constant", "nearest", "first"):
        return y[0]
    if method == "linear":
        slope = (y[1] - y[0]) / (rho[1] - rho[0] + 1e-12)
        return y[0] - slope * rho[0]
    if method in ("quadratic", "poly2"):
        return _polyfit_axis_value(rho, y, degree=2, n_points=n_points)
    if method in ("cubic", "poly3"):
        return _polyfit_axis_value(rho, y, degree=3, n_points=max(n_points, 4))
    raise ValueError(f"Unknown axis extrapolation method={method!r}")


def axis_augmented_profile(rho, y, method="copy", n_points=4):
    """Return (rho_aug, y_aug) with rho=0 prepended as y[0]."""
    y0 = axis_extrapolated_value(y, rho, method=method, n_points=n_points)
    rho0 = jnp.asarray([0.0], dtype=rho.dtype)
    return jnp.concatenate([rho0, rho]), jnp.concatenate([y0[None], jnp.asarray(y)])


def boundary_augmented_profile(rho, y, edge_value=None, method="copy", n_points=4):
    """Return profile points including rho=0 and rho=1 for plotting.

    The rho=0 value is copied from the first cell by default.  If edge_value
    is None, the rho=1 value is copied from the last cell; otherwise the
    supplied physical boundary condition is appended.
    """
    rho_aug, y_aug = axis_augmented_profile(rho, y, method=method, n_points=n_points)
    edge = y_aug[-1] if edge_value is None else jnp.asarray(edge_value, dtype=y_aug.dtype)
    rho1 = jnp.asarray([1.0], dtype=rho_aug.dtype)
    return jnp.concatenate([rho_aug, rho1]), jnp.concatenate([y_aug, edge[None]])


def left_axis_ghost_value(y, rho):
    """Ghost value at rho=0: zero-gradient copy of the first cell."""
    return jnp.asarray(y)[0]


def axis_augmented_cell_widths(rho, machine):
    """Volume weights for an axis-augmented diagnostic/profile integral.

    The first sample is at rho=0 with a copied axis value.  This keeps scalar
    diagnostics consistent with the plotting convention while preserving the
    cell-centered state.
    """
    rho = jnp.asarray(rho)
    rho_aug = jnp.concatenate([jnp.asarray([0.0], dtype=rho.dtype), rho])
    faces = jnp.concatenate([
        jnp.asarray([0.0], dtype=rho.dtype),
        0.5 * (rho_aug[:-1] + rho_aug[1:]),
        jnp.asarray([1.0], dtype=rho.dtype),
    ])
    drho = faces[1:] - faces[:-1]
    return machine.a * drho


def axis_augmented_volume_element(rho, machine):
    """Approximate toroidal shell volumes for axis-augmented profiles."""
    rho_aug = jnp.concatenate([jnp.asarray([0.0], dtype=rho.dtype), rho])
    dr = axis_augmented_cell_widths(rho, machine)
    r = machine.a * rho_aug
    return 4.0 * jnp.pi**2 * machine.R0 * machine.kappa * r * dr


def volume_element_from_dV_drho(rho, dV_drho):
    """Cell volumes from an equilibrium V'(rho)=dV/drho profile.

    ``dV_drho`` is assumed to be evaluated at the same cell centers as ``rho``
    and to be a derivative with respect to normalized rho.  This helper converts
    it to per-cell shell volumes by multiplying by the inferred cell widths in
    normalized rho.
    """
    rho = jnp.asarray(rho)
    dV_drho = jnp.asarray(dV_drho)
    faces = infer_rho_faces(rho)
    drho = faces[1:] - faces[:-1]
    return jnp.maximum(dV_drho, 0.0) * drho


def axis_augmented_volume_element_from_dV_drho(rho, dV_drho):
    """Axis-augmented volume weights from an equilibrium V'(rho).

    The prepended rho=0 sample has zero volume weight; the remaining entries are
    the finite-volume shell volumes corresponding to the cell-centered state.
    This keeps the plotting/diagnostic convention y(0)=y[0] while preserving the
    full equilibrium volume integral.
    """
    dV = volume_element_from_dV_drho(rho, dV_drho)
    return jnp.concatenate([jnp.asarray([0.0], dtype=dV.dtype), dV])


def infer_rho_faces(rho):
    """Infer monotonic cell faces from cell centers, with exact 0 and 1 edges."""
    mids = 0.5 * (rho[:-1] + rho[1:])
    return jnp.concatenate([jnp.asarray([0.0], dtype=rho.dtype), mids, jnp.asarray([1.0], dtype=rho.dtype)])


def cell_widths(rho, a=1.0):
    faces = infer_rho_faces(rho)
    return a * (faces[1:] - faces[:-1])


def radial_gradient(y, rho, a):
    """Finite-difference gradient dy/dr on a non-uniform radial grid.

    The first cell uses a copied rho=0 axis point, imposing a zero-gradient
    center ghost for robust coarse-grid behavior.
    """
    r = a * rho
    y_axis = left_axis_ghost_value(y, rho)
    g0 = (y[0] - y_axis) / (r[0] + 1e-12)
    gN = (y[-1] - y[-2]) / (r[-1] - r[-2] + 1e-12)
    gi = (y[2:] - y[:-2]) / (r[2:] - r[:-2] + 1e-12)
    return jnp.concatenate([g0[None], gi, gN[None]])
