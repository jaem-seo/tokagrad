"""Differentiable-control objectives for TokaGrad.

This module provides a deliberately small differentiable optimization path for
control studies.  It treats selected scalar controls as JAX variables while
keeping the model choices and grid size static:

    x = [Ip_MA, Bt, R0, a, kappa, delta, P_aux_MW, greenwald_fraction_target, ...]

The default helper configuration avoids optional/non-JAX external dependencies
and uses the reduced fixed-boundary equilibrium.  It is intended for gradient
checks, algorithm prototyping, and sensitivity studies rather than validated
ITER optimization.

Optimization reference: [D. P. Kingma and J. Ba, ICLR (2015)] for Adam.
Constraint penalties, smooth clips, regularization, and control bounds are
TokaGrad optimization choices rather than plasma-physics scalings.
"""

from dataclasses import replace
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp

from .config import MachineConfig, ActuatorConfig, SimulationConfig
from .solver import initial_state, simulate_final_dynamic_jit
from .zero_d import zero_d_enabled, initial_state_0d, simulate_0d_final_dynamic_jit
from .diagnostics import fusion_power_MW, beta_normalized_total, volume_average_axis_augmented
from .grid import make_grid_from_config
from .current import current_components_from_state, q_profile, total_current, edge_q_from_boundary_geometry
from .equilibrium import solve_fixed_boundary_equilibrium
from .controls import (
    PytreeMachineConfig,
    PytreeActuatorConfig,
    pytree_configs_from_legacy,
    controls_from_unconstrained,
    unconstrained_from_controls,
    apply_control_vector,
    DEFAULT_CONTROL_BOUNDS,
    CONTROL_NAMES as PYTREE_CONTROL_NAMES,
)


def make_gradient_friendly_sim(
    base_sim: SimulationConfig | None = None,
    *,
    nr: int = 8,
    ntheta: int = 8,
    dt: float = 2.0e-3,
    n_steps: int = 4,
):
    """Return a differentiability-friendly reduced SimulationConfig.

    The defaults avoid optional external surrogates and hard file-based
    prescribed equilibria.  Piecewise smooth limiters/clips remain, so gradients
    should be interpreted as local sensitivities of the reduced model.
    """
    if base_sim is None:
        base_sim = SimulationConfig()

    return replace(
        base_sim,
        nr=nr,
        ntheta=ntheta,
        dt=dt,
        n_steps=n_steps,
        equilibrium_model="reduced_fixed_boundary",
        transport_model=base_sim.transport_model,
        fusion_surrogates_fail_mode=base_sim.fusion_surrogates_fail_mode,
        transport_mode="diffusive",
        diffusion_scheme="semi_implicit",
        current_evolution_model=base_sim.current_evolution_model,
        density_evolution_model=base_sim.density_evolution_model,
        density_feedback_tau=base_sim.density_feedback_tau,
        density_source_max_delta=base_sim.density_source_max_delta,
        differentiable_smooth_mode=True,
        #pedestal_lh_transition_control=True,
        pedestal_lh_margin=0.20,
        pedestal_lh_min_gate=0.05,
        pedestal_model=base_sim.pedestal_model,
        pedestal_alpha_tracking=True,
        pedestal_projection_fraction=0.0,
        include_ohmic_heating=True,
        include_ei_exchange=True,
        include_radiation_losses=True,
        include_alpha_heating=True,
        alpha_partition_model="slowing_down",
        neoclassical_transport_model="angioni",
        freeze_temperature_profiles=False,
    )


def differentiable_metrics(
    final_state,
    machine: MachineConfig,
    actuator: ActuatorConfig,
    sim: SimulationConfig,
):
    """Small metrics dict that avoids expensive/nonessential diagnostics."""
    rho, _, _ = make_grid_from_config(sim.nr, machine.a, sim)
    eq = solve_fixed_boundary_equilibrium(final_state, machine, actuator, sim)
    j_ind, j_bs, j_cd, j_total = current_components_from_state(
        rho, final_state, machine, actuator, sim, eq=eq
    )
    q = eq.q if hasattr(eq, "q") else q_profile(rho, j_total, machine, sim, eq=eq)
    #q95_idx = min(int(0.95 * sim.nr), sim.nr - 1)

    I_total = total_current(rho, j_total, machine, eq=eq)
    q_edge_lcfs = edge_q_from_boundary_geometry(machine, I_total, eq=eq, sim=sim)
    rho_q = jnp.concatenate([rho, jnp.ones(1, dtype=rho.dtype)])
    q_q = jnp.concatenate([q, jnp.asarray([q_edge_lcfs], dtype=q.dtype)])
    q95_value = jnp.interp(jnp.asarray(0.95, dtype=rho.dtype), rho_q, q_q)

    P_fus = fusion_power_MW(final_state, machine, sim=sim, eq=eq)
    P_aux = jnp.maximum(actuator.P_aux_MW, 1e-6)
    Q = P_fus / P_aux
    beta_N = beta_normalized_total(final_state, machine, actuator, sim=sim, eq=eq)
    ne_avg = volume_average_axis_augmented(final_state.ne20, rho, machine, dV_drho=eq.dV_drho)
    nG_1e20 = (machine.Ip / 1.0e6) / (jnp.pi * machine.a**2 + 1e-12)
    fG = ne_avg / (nG_1e20 + 1e-12)

    return {
        "P_fus_MW": P_fus,
        "P_aux_MW": actuator.P_aux_MW,
        "Q": Q,
        "beta_N": beta_N,
        "q0": q[0],
        "q95": q95_value,
        "q_edge": q_edge_lcfs,
        "greenwald_fraction": fG,
        "Te0_keV": final_state.Te[0],
        "Ti0_keV": final_state.Ti[0],
        "ne_avg_1e20": ne_avg,
        "I_total_MA": total_current(rho, j_total, machine, eq=eq) / 1.0e6,
        "Ip_MA": machine.Ip / 1.0e6,
    }


# ---------------------------------------------------------------------------
# PyTree + optax preparation path
# ---------------------------------------------------------------------------

def pytree_base_configs(
    base_machine: MachineConfig | None = None,
    base_actuator: ActuatorConfig | None = None,
):
    """Return pytree-compatible machine/actuator configs."""
    if base_machine is None:
        base_machine = MachineConfig()
    if base_actuator is None:
        base_actuator = ActuatorConfig()
    return pytree_configs_from_legacy(base_machine, base_actuator)



def simulate_final_unrolled_ad(state0, machine, actuator, sim):
    """AD-friendly final-state rollout using a Python-unrolled static loop.

    `jax.lax.fori_loop` lowers to scan-like primitives whose JVP/VJP can be
    fragile for the current NamedTuple state and piecewise closures.  For the
    small optimization/demo cases, a static Python loop is more transparent to
    JAX AD and avoids the scan-level tangent NaNs observed in early tests.
    """
    from .solver import step
    state = state0
    for _ in range(int(sim.n_steps)):
        state = step(state, machine, actuator, sim)
    return state


# Preserve the unrolled smooth-AD path (it avoids scan/fori-loop tangent NaNs)
# while compiling the complete rollout.  Machine and actuator remain dynamic.
simulate_final_unrolled_ad_jit = jax.jit(
    simulate_final_unrolled_ad, static_argnames=("sim",)
)


def rollout_from_unconstrained_controls_pytree(
    z,
    base_machine: PytreeMachineConfig | None = None,
    base_actuator: PytreeActuatorConfig | None = None,
    base_sim: SimulationConfig | None = None,
    bounds=DEFAULT_CONTROL_BOUNDS,
    names=PYTREE_CONTROL_NAMES,
):
    """Rollout using a smooth unconstrained control parameterization.

    z is mapped through a sigmoid to physical bounded controls.  Machine and
    actuator configs are pytrees, so their numerical leaves can be JAX dynamic
    values.  The SimulationConfig remains mostly static, except scalar controls
    such as greenwald_fraction_target may be replaced by tracer values.
    """
    if base_machine is None or base_actuator is None:
        base_machine, base_actuator = pytree_base_configs()
    if base_sim is None:
        base_sim = make_gradient_friendly_sim()

    x_phys = controls_from_unconstrained(z, bounds, names)
    machine, actuator, sim = apply_control_vector(
        x_phys, base_machine, base_actuator, base_sim, names
    )
    if zero_d_enabled(sim):
        state0 = initial_state_0d(machine, actuator, sim)
        final_state = simulate_0d_final_dynamic_jit(machine, actuator, sim, state0=state0)
    else:
        state0 = initial_state(machine, actuator, sim)
        if getattr(sim, "differentiable_smooth_mode", False):
            final_state = simulate_final_unrolled_ad_jit(state0, machine, actuator, sim)
        else:
            final_state = simulate_final_dynamic_jit(machine, actuator, sim, state0=state0)
    if getattr(sim, "differentiable_smooth_mode", False):
        metrics = differentiable_metrics_ad_clean(final_state, machine, actuator, sim)
    else:
        metrics = differentiable_metrics(final_state, machine, actuator, sim)
    metrics = dict(metrics)
    metrics["controls_phys"] = x_phys
    return final_state, machine, actuator, sim, metrics



def smooth_lower(x, lo, width=1.0e-3):
    """Smooth approximation to max(x, lo)."""
    w = jnp.maximum(width, 1.0e-8)
    return lo + w * jax.nn.softplus((x - lo) / w)


def smooth_upper(x, hi, width=1.0e-3):
    """Smooth approximation to min(x, hi)."""
    w = jnp.maximum(width, 1.0e-8)
    return hi - w * jax.nn.softplus((hi - x) / w)


def smooth_bounded(x, lo, hi, width=1.0e-3):
    return smooth_upper(smooth_lower(x, lo, width), hi, width)


def q_profile_ad_clean(rho, j_A_m2, machine, eq=None):
    """AD-clean q helper, preferring the active Equilibrium q profile.

    The circular current-derived proxy is retained only as a legacy fallback for
    callers that have no Equilibrium object.  Optimization metrics now pass/use
    ``eq.q`` directly.
    """
    if eq is not None and hasattr(eq, "q"):
        return smooth_bounded(jnp.abs(jnp.asarray(eq.q, dtype=rho.dtype) + jnp.zeros_like(rho)), 0.05, 30.0, 1.0e-2)
    # Local import avoids changing the production current module.
    from .current import enclosed_current, MU0, shape_factor

    r = machine.a * rho
    Ienc = enclosed_current(rho, j_A_m2, machine.a, machine.kappa)
    Ienc_safe = smooth_lower(Ienc, 1.0e3, 1.0e2)
    Bp = MU0 * Ienc_safe / (2.0 * jnp.pi * smooth_lower(r, 1.0e-3, 1.0e-4))
    q = r * machine.Bt * shape_factor(machine, rho) / (machine.R0 * smooth_lower(Bp, 1.0e-5, 1.0e-6))
    return smooth_bounded(q, 0.05, 30.0, 1.0e-2)



DEFAULT_CONSTRAINTS = (
    # Constraint convention:
    #   lower: metric >= value
    #   upper: metric <= value
    # Penalty is weight * smooth_relu(violation)^2.
    {"metric": "q_edge", "kind": "lower", "value": 3.0, "weight": 10.0},
    {"metric": "q95", "kind": "lower", "value": 2.7, "weight": 10.0},
    {"metric": "q0", "kind": "lower", "value": 0.9, "weight": 10.0},
    {"metric": "greenwald_fraction", "kind": "upper", "value": 1.0, "weight": 10.0},
    {"metric": "beta_N", "kind": "upper", "value": 4.0, "weight": 10.0},
)


def smooth_relu(x, width=1.0e-2):
    """Smooth approximation to max(x, 0), AD-friendly."""
    w = jnp.maximum(width, 1.0e-8)
    return w * jax.nn.softplus(x / w)


def constraint_violation(metric_value, kind, threshold):
    """Positive when a constraint is violated."""
    if kind == "lower":
        return threshold - metric_value
    if kind == "upper":
        return metric_value - threshold
    raise ValueError(f"Unknown constraint kind={kind!r}")


def evaluate_constraints(metrics, constraints=DEFAULT_CONSTRAINTS, smooth_width=1.0e-2):
    """Evaluate differentiable soft-constraint penalty.

    Returns
    -------
    total_penalty, details
        `total_penalty` is sum_i weight_i * smooth_relu(violation_i)^2.
        `details` contains raw and smoothed violations for diagnostics.
    """
    total = 0.0
    details = {}
    if constraints is None:
        return jnp.asarray(0.0), details

    for c in constraints:
        name = c["metric"]
        if name not in metrics:
            # Missing constraints are ignored but recorded as NaN.
            details[f"constraint_{name}_missing"] = jnp.asarray(1.0)
            continue
        kind = c.get("kind", "lower")
        threshold = jnp.asarray(c["value"], dtype=jnp.float32)
        weight = jnp.asarray(c.get("weight", 1.0), dtype=jnp.float32)
        raw = constraint_violation(metrics[name], kind, threshold)
        viol = smooth_relu(raw, smooth_width)
        pen = weight * viol**2
        total = total + pen
        details[f"constraint_{name}_raw_violation"] = raw
        details[f"constraint_{name}_violation"] = viol
        details[f"constraint_{name}_penalty"] = pen
    details["constraint_penalty"] = total
    return total, details




def _parse_objective_terms(objective, objective_weights=None):
    """Return [(name, weight), ...] and reject deprecated constrained aliases.

    Supported inputs:
      objective="Q"
      objective=("Q", "Pfus"), objective_weights=(1.0, 0.01)
      objective="Q+0.01*Pfus"
    Constraints are deliberately not encoded in the objective name anymore;
    they are always applied through the separate `constraints` argument.
    """
    def _reject(name):
        if str(name).strip().lower() in ("constrained_q", "constrained_q_integral"):
            raise ValueError(
                "constrained_q has been removed. Use objective='Q' and pass "
                "constraints=... separately."
            )

    if isinstance(objective, str):
        expr = objective.strip()
        _reject(expr)
        parts = [p.strip() for p in expr.split("+") if p.strip()]
        out = []
        for part in parts:
            if "*" in part:
                a, b = part.split("*", 1)
                try:
                    weight = float(a.strip())
                    name = b.strip()
                except ValueError:
                    name = a.strip()
                    weight = float(b.strip())
            else:
                name = part
                weight = 1.0
            _reject(name)
            out.append((name, weight))
        if not out:
            raise ValueError("Empty objective specification.")
        return out

    names = tuple(objective)
    for name in names:
        _reject(name)
    if objective_weights is None:
        weights = (1.0,) * len(names)
    else:
        weights = tuple(objective_weights)
        if len(weights) != len(names):
            raise ValueError("objective_weights must have the same length as objective.")
    return [(name, float(w)) for name, w in zip(names, weights)]


def _objective_metric_name(name, *, waveform=False):
    key = str(name).strip().lower().replace("-", "_")
    key_compact = key.replace("_", "")
    if key_compact in ("q", "qgain"):
        return "Q_integral" if waveform else "Q"
    if key_compact in ("qintegral", "qwaveform"):
        return "Q_integral"
    if key_compact in ("qfinal",):
        return "Q"
    if key_compact in ("pfus", "pfusion", "pfusmw", "pfusavg"):
        return "P_fus_avg_MW" if waveform else "P_fus_MW"
    if key_compact in ("pfusintegral", "pfusaverage"):
        return "P_fus_avg_MW"
    if key_compact in ("pfusfinal",):
        return "P_fus_MW"
    if key_compact in ("ti0", "ti0kev", "centralti", "centraltikev"):
        return "Ti0_keV"
    if key_compact in ("te0", "te0kev", "centralte", "centraltekev"):
        return "Te0_keV"
    if key_compact in ("betan", "betanormalized"):
        return "beta_N"
    raise ValueError(f"Unknown objective term {name!r}.")


def _weighted_objective_from_metrics(metrics, objective, objective_weights=None, *, waveform=False):
    value = jnp.asarray(0.0, dtype=jnp.float32)
    details = {}
    for name, weight in _parse_objective_terms(objective, objective_weights):
        metric_name = _objective_metric_name(name, waveform=waveform)
        term = jnp.asarray(weight, dtype=jnp.float32) * metrics[metric_name]
        value = value + term
        details[f"objective_term_{metric_name}"] = term
    details["objective_performance"] = value
    return value, details

def differentiable_metrics_ad_clean(
    final_state,
    machine: MachineConfig,
    actuator: ActuatorConfig,
    sim: SimulationConfig,
):
    """AD-clean metrics for the pytree optimization path.

    The production diagnostics are intentionally rich but contain several hard
    safety clips/projections.  This compact metric path avoids non-smooth q
    constraints and uses smooth positive/bound transforms where possible.
    """
    rho, _, _ = make_grid_from_config(sim.nr, machine.a, sim)
    eq = solve_fixed_boundary_equilibrium(final_state, machine, actuator, sim)
    j_ind, j_bs, j_cd, j_total = current_components_from_state(
        rho, final_state, machine, actuator, sim, eq=eq
    )
    q = eq.q if hasattr(eq, "q") else q_profile(rho, j_total, machine, sim, eq=eq)
    #q95_idx = min(int(0.95 * sim.nr), sim.nr - 1)

    I_total = total_current(rho, j_total, machine, eq=eq)
    q_edge_lcfs = edge_q_from_boundary_geometry(machine, I_total, eq=eq, sim=sim)
    rho_q = jnp.concatenate([rho, jnp.ones(1, dtype=rho.dtype)])
    q_q = jnp.concatenate([q, jnp.asarray([q_edge_lcfs], dtype=q.dtype)])
    q95_value = jnp.interp(jnp.asarray(0.95, dtype=rho.dtype), rho_q, q_q)

    P_fus = fusion_power_MW(final_state, machine, sim=sim, eq=eq)
    P_aux = smooth_lower(actuator.P_aux_MW, 1e-6, 1e-6)
    Q = P_fus / P_aux

    # Smooth beta_N proxy, same expression as beta_normalized but without hard operations.
    beta_N = beta_normalized_total(final_state, machine, actuator, sim=sim, eq=eq)

    ne_avg = volume_average_axis_augmented(final_state.ne20, rho, machine, dV_drho=eq.dV_drho)
    nG_1e20 = (machine.Ip / 1.0e6) / (jnp.pi * machine.a**2 + 1e-12)
    fG = ne_avg / smooth_lower(nG_1e20, 1e-6, 1e-6)

    return {
        "P_fus_MW": P_fus,
        "P_aux_MW": actuator.P_aux_MW,
        "Q": Q,
        "beta_N": beta_N,
        "q0": q[0],
        "q95": q95_value,
        "q_edge": q_edge_lcfs,
        "greenwald_fraction": fG,
        "Te0_keV": final_state.Te[0],
        "Ti0_keV": final_state.Ti[0],
        "ne_avg_1e20": ne_avg,
        "I_total_MA": total_current(rho, j_total, machine, eq=eq) / 1.0e6,
        "Ip_MA": machine.Ip / 1.0e6,
    }


def pytree_objective_from_z(
    z,
    objective="Q",
    base_machine: PytreeMachineConfig | None = None,
    base_actuator: PytreeActuatorConfig | None = None,
    base_sim: SimulationConfig | None = None,
    bounds=DEFAULT_CONTROL_BOUNDS,
    names=PYTREE_CONTROL_NAMES,
    constraints=DEFAULT_CONSTRAINTS,
    constraint_smooth_width: float = 1.0e-2,
    objective_weights=None,
    fixed_control_values=None,
    fixed_start: bool = False,
    fixed_final: bool = False,
    fixed_endpoint_keys=None,
):
    """AD-compatible scalar-control objective.

    Performance objective and constraints are deliberately separated:

        value = weighted_objective(metrics) - constraint_penalty

    Thus the old `constrained_q` objective is no longer needed.
    """
    _, _, _, _, metrics = rollout_from_unconstrained_controls_pytree(
        z, base_machine, base_actuator, base_sim, bounds, names
    )
    perf_value, objective_details = _weighted_objective_from_metrics(
        metrics, objective, objective_weights, waveform=False
    )
    penalty_val, constraint_details = evaluate_constraints(
        metrics,
        constraints=constraints,
        smooth_width=constraint_smooth_width,
    )
    value = perf_value - penalty_val
    metrics = dict(metrics)
    metrics.update(objective_details)
    metrics.update(constraint_details)
    metrics["objective_value"] = value
    return value, metrics

def autodiff_value_and_grad_pytree(
    z,
    objective="Q",
    base_machine: PytreeMachineConfig | None = None,
    base_actuator: PytreeActuatorConfig | None = None,
    base_sim: SimulationConfig | None = None,
    bounds=DEFAULT_CONTROL_BOUNDS,
    names=PYTREE_CONTROL_NAMES,
    constraints=DEFAULT_CONSTRAINTS,
    constraint_smooth_width: float = 1.0e-2,
    objective_weights=None,
):
    """AD value/grad using JAX over pytree configs."""
    return jax.value_and_grad(
        lambda zz: pytree_objective_from_z(
            zz, objective, base_machine, base_actuator, base_sim, bounds, names,
            constraints, constraint_smooth_width, objective_weights
        ),
        has_aux=True,
    )(z)


def finite_difference_gradient_z(
    z,
    objective="Q",
    base_machine: PytreeMachineConfig | None = None,
    base_actuator: PytreeActuatorConfig | None = None,
    base_sim: SimulationConfig | None = None,
    bounds=DEFAULT_CONTROL_BOUNDS,
    names=PYTREE_CONTROL_NAMES,
    eps: float = 1.0e-3,
    constraints=DEFAULT_CONSTRAINTS,
    constraint_smooth_width: float = 1.0e-2,
    objective_weights=None,
):
    """Central finite-difference gradient in unconstrained z-space."""
    z = jnp.asarray(z, dtype=jnp.float32)
    y0, metrics0 = pytree_objective_from_z(
        z, objective, base_machine, base_actuator, base_sim, bounds, names,
        constraints, constraint_smooth_width, objective_weights
    )
    grads = []
    eye = jnp.eye(z.size, dtype=jnp.float32)
    for i in range(z.size):
        yp, _ = pytree_objective_from_z(
            z + eps * eye[i], objective, base_machine, base_actuator, base_sim, bounds, names,
            constraints, constraint_smooth_width, objective_weights
        )
        ym, _ = pytree_objective_from_z(
            z - eps * eye[i], objective, base_machine, base_actuator, base_sim, bounds, names,
            constraints, constraint_smooth_width, objective_weights
        )
        grads.append((yp - ym) / (2.0 * eps))
    return y0, metrics0, jnp.asarray(grads)


def _adam_fallback_update(grad, m, v, t, learning_rate=1.0e-2, b1=0.9, b2=0.999, eps=1.0e-8):
    """Small Adam update used when optax is not installed."""
    m = b1 * m + (1.0 - b1) * grad
    v = b2 * v + (1.0 - b2) * (grad * grad)
    mh = m / (1.0 - b1 ** t)
    vh = v / (1.0 - b2 ** t)
    step = learning_rate * mh / (jnp.sqrt(vh) + eps)
    return step, m, v


def _array_global_norm(x):
    """Finite global norm for optimizer diagnostics/clipping."""
    x = jnp.asarray(x, dtype=jnp.float32)
    xf = jnp.where(jnp.isfinite(x), x, 0.0)
    return jnp.sqrt(jnp.sum(xf * xf) + 1.0e-30)


def _sanitize_and_clip_gradient(
    grad,
    *,
    max_grad_norm: float | None = None,
):
    """Replace non-finite gradient entries by zero and optionally clip norm.

    The 0.5D waveform objective contains many smooth-but-stiff reduced physics
    pieces: EPED pedestal floors, L-H gates, current/q reconstruction and
    radiation/fusion powers.  At higher radial resolution JAX can occasionally
    return a non-finite local derivative even when the forward rollout and
    metrics are finite.  For an interactive optimizer it is better to report the
    issue, zero only those entries, and continue with the finite gradient
    components than to abort the whole run.
    """
    finite = jnp.isfinite(grad)
    nonfinite_count = int(grad.size - jnp.sum(finite))
    grad = jnp.where(finite, grad, 0.0)
    norm_before = _array_global_norm(grad)
    applied_clip = False
    if max_grad_norm is not None and max_grad_norm > 0.0:
        scale = jnp.minimum(1.0, jnp.asarray(max_grad_norm, dtype=grad.dtype) / (norm_before + 1.0e-12))
        applied_clip = bool(float(scale) < 0.999999)
        grad = grad * scale
    norm_after = _array_global_norm(grad)
    return grad, {
        "grad_finite": finite,
        "nonfinite_grad_count": nonfinite_count,
        "grad_norm_before_clip": norm_before,
        "grad_norm": norm_after,
        "grad_clip_applied": applied_clip,
    }


def _clip_update_delta(z_old, z_new, max_update_norm: float | None = None):
    """Clip one optimizer step in unconstrained z-space."""
    if max_update_norm is None or max_update_norm <= 0.0:
        return z_new, {"update_norm_before_clip": _array_global_norm(z_new - z_old), "update_norm": _array_global_norm(z_new - z_old), "update_clip_applied": False}
    delta = z_new - z_old
    norm_before = _array_global_norm(delta)
    scale = jnp.minimum(1.0, jnp.asarray(max_update_norm, dtype=delta.dtype) / (norm_before + 1.0e-12))
    clipped = z_old + delta * scale
    return clipped, {
        "update_norm_before_clip": norm_before,
        "update_norm": _array_global_norm(clipped - z_old),
        "update_clip_applied": bool(float(scale) < 0.999999),
    }


def optimize_controls_optax(
    x0_phys,
    objective="Q",
    base_machine: MachineConfig | PytreeMachineConfig | None = None,
    base_actuator: ActuatorConfig | PytreeActuatorConfig | None = None,
    base_sim: SimulationConfig | None = None,
    bounds=DEFAULT_CONTROL_BOUNDS,
    names=PYTREE_CONTROL_NAMES,
    n_iter: int = 10,
    learning_rate: float = 1.0e-2,
    gradient_mode: str = "autodiff",
    constraints=DEFAULT_CONSTRAINTS,
    constraint_smooth_width: float = 1.0e-2,
    objective_weights=None,
    profile_timing: bool = False,
    max_grad_norm: float | None = None,
    max_update_norm: float | None = None,
    iteration_callback: Callable[[dict[str, Any]], None] | None = None,
):
    """Optimize controls using optax when available.

    Parameters
    ----------
    gradient_mode:
        "autodiff" uses JAX value_and_grad and is the default path.
        "finite_difference" is retained only as a debugging cross-check.
    """
    if base_sim is None:
        base_sim = make_gradient_friendly_sim()
    if isinstance(base_machine, PytreeMachineConfig):
        pm = base_machine
    else:
        pm = PytreeMachineConfig.from_config(base_machine or MachineConfig())
    if isinstance(base_actuator, PytreeActuatorConfig):
        pa = base_actuator
    else:
        pa = PytreeActuatorConfig.from_config(base_actuator or ActuatorConfig())

    z = unconstrained_from_controls(x0_phys, bounds, names)

    try:
        import optax  # type: ignore
        tx = optax.adam(learning_rate)
        opt_state = tx.init(z)
        use_optax = True
    except Exception:
        optax = None
        tx = None
        opt_state = None
        use_optax = False
        m = jnp.zeros_like(z)
        v = jnp.zeros_like(z)

    history = []
    timing_rows = []
    objective_eval = lambda zz: pytree_objective_from_z(
        zz, objective, pm, pa, base_sim, bounds, names,
        constraints, constraint_smooth_width, objective_weights,
    )
    # Construct these transformed callables once so their compiled executables
    # are reused across optimizer iterations.  Recreating value_and_grad in the
    # loop retraces the objective and leaves substantial Python dispatch around
    # the already-jitted rollout.
    if gradient_mode == "autodiff":
        objective_eval_jit = jax.jit(objective_eval)
        value_and_grad_eval_jit = jax.jit(jax.value_and_grad(objective_eval, has_aux=True))
    else:
        objective_eval_jit = None
        value_and_grad_eval_jit = None
    for it in range(n_iter):
        forward_time = None
        grad_total_time = None
        if gradient_mode == "autodiff":
            if profile_timing:
                t_f = time.perf_counter()
                _fv, _fm = objective_eval_jit(z)
                try:
                    jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, (_fv, _fm))
                except Exception:
                    pass
                forward_time = time.perf_counter() - t_f
            t_g = time.perf_counter()
            (value, metrics), grad = value_and_grad_eval_jit(z)
            try:
                jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, (value, metrics, grad))
            except Exception:
                pass
            grad_total_time = time.perf_counter() - t_g
        elif gradient_mode == "finite_difference":
            t_g = time.perf_counter()
            value, metrics, grad = finite_difference_gradient_z(
                z, objective, pm, pa, base_sim, bounds, names,
                constraints=constraints, constraint_smooth_width=constraint_smooth_width,
                objective_weights=objective_weights
            )
            try:
                jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, (value, metrics, grad))
            except Exception:
                pass
            grad_total_time = time.perf_counter() - t_g
        else:
            raise ValueError('gradient_mode must be "finite_difference" or "autodiff".')

        grad, grad_info = _sanitize_and_clip_gradient(grad, max_grad_norm=max_grad_norm)

        z_before = z
        x_phys_before = controls_from_unconstrained(z_before, bounds, names)
        # We maximize objective, while optax optimizers are descent-style.
        if use_optax:
            updates, opt_state = tx.update(-grad, opt_state, z)
            z_candidate = optax.apply_updates(z, updates)
        else:
            step, m, v = _adam_fallback_update(-grad, m, v, it + 1, learning_rate)
            z_candidate = z - step
        z, update_info = _clip_update_delta(z_before, z_candidate, max_update_norm=max_update_norm)

        x_phys = controls_from_unconstrained(z, bounds, names)
        control_delta_phys = x_phys - x_phys_before
        row = {
            "iter": it,
            "objective": value,
            "control_names": tuple(names),
            "controls_phys": x_phys,
            "controls_phys_before_update": x_phys_before,
            "controls_delta_phys": control_delta_phys,
            "grad_z": grad,
            "metrics": metrics,
            "used_optax": use_optax,
            **grad_info,
            **update_info,
        }
        history.append(row)
        if iteration_callback is not None:
            iteration_callback(row)
        if profile_timing:
            timing_rows.append({
                "iter": it,
                "forward_time_s": forward_time,
                "value_and_grad_time_s": grad_total_time,
                "backprop_estimate_s": None if forward_time is None or grad_total_time is None else max(grad_total_time - forward_time, 0.0),
            })

    if objective_eval_jit is not None:
        final_value, final_metrics = objective_eval_jit(z)
    else:
        final_value, final_metrics = objective_eval(z)
    return {
        "z": z,
        "x_phys": controls_from_unconstrained(z, bounds, names),
        "objective": final_value,
        "metrics": final_metrics,
        "history": history,
        "used_optax": use_optax,
        "gradient_mode": gradient_mode,
        "timing": timing_rows,
    }



# ---------------------------------------------------------------------------
# Waveform-control optimization scaffold
# ---------------------------------------------------------------------------

def controls_from_waveform_z(z, lo, hi):
    """Map unconstrained waveform parameters to bounded control points."""
    z = jnp.asarray(z, dtype=jnp.float32)
    lo = jnp.asarray(lo, dtype=jnp.float32)
    hi = jnp.asarray(hi, dtype=jnp.float32)
    return lo + (hi - lo) * jax.nn.sigmoid(z)


def z_from_waveform_controls(x, lo, hi, eps=1e-6):
    x = jnp.asarray(x, dtype=jnp.float32)
    lo = jnp.asarray(lo, dtype=jnp.float32)
    hi = jnp.asarray(hi, dtype=jnp.float32)
    y = jnp.clip((x - lo) / (hi - lo + 1e-12), eps, 1.0 - eps)
    return jnp.log(y / (1.0 - y))


def make_piecewise_waveform_from_controls(
    times,
    control_values,
    name="optimized_waveform",
):
    """Create ScenarioWaveform from dense control point arrays.

    control_values is a dict with optional keys supported by
    waveforms.apply_waveform_controls, including machine controls
    Ip_MA, Bt, R0, a, kappa, delta and actuator controls
    P_aux_MW, greenwald_fraction_target, heat_center, heat_width, f_e_heat.
    """
    from .waveforms import ScenarioWaveform, make_waveform, default_interpolation_mode_for_control
    controls = {
        key: make_waveform(times, val, interpolation=default_interpolation_mode_for_control(key))
        for key, val in control_values.items()
    }
    return ScenarioWaveform(controls=controls, name=name)


def waveform_regularization(control_values, bounds, keys, smoothness_weight=0.0, slew_weight=0.0):
    """Dimensionless waveform regularization.

    - slew penalty: first differences of normalized control points.
    - smoothness penalty: second differences of normalized control points.
    """
    penalty = 0.0
    details = {}
    for key in keys:
        x = jnp.asarray(control_values[key], dtype=jnp.float32)
        lo, hi = bounds[key]
        xn = (x - lo) / (hi - lo + 1e-12)
        slew = jnp.diff(xn)
        smooth = jnp.diff(xn, n=2) if x.size >= 3 else jnp.zeros((0,), dtype=x.dtype)
        slew_pen = jnp.mean(slew**2) if slew.size > 0 else 0.0
        smooth_pen = jnp.mean(smooth**2) if smooth.size > 0 else 0.0
        penalty = penalty + slew_weight * slew_pen + smoothness_weight * smooth_pen
        details[f"{key}_slew_penalty"] = slew_pen
        details[f"{key}_smoothness_penalty"] = smooth_pen
    return penalty, details


def _waveform_objective_needs_time_integrals(objective, objective_weights=None) -> bool:
    """Return True when waveform objective terms require stepwise accumulation."""
    return any(
        _objective_metric_name(name, waveform=True) in ("Q_integral", "P_fus_avg_MW")
        for name, _weight in _parse_objective_terms(objective, objective_weights)
    )


def _normalize_waveform_constraint_mode(mode, *, needs_time_integrals: bool) -> str:
    """Return ``"time_average"`` or ``"final"`` for waveform constraint penalties.

    ``auto`` preserves the original time-averaged semantics for time-integral
    waveform objectives, but uses final constraints for explicitly final
    objectives such as ``Qfinal``/``Pfusfinal``.  This avoids a very expensive
    per-step diagnostics pass when the requested objective is final-only.
    """
    key = str(mode or "auto").strip().lower().replace("-", "_")
    if key in ("auto", ""):
        return "time_average" if needs_time_integrals else "final"
    if key in ("time_average", "time_avg", "average", "integral", "trajectory"):
        return "time_average"
    if key in ("final", "terminal", "endpoint"):
        return "final"
    raise ValueError(
        f"Unknown waveform_constraint_mode={mode!r}; use 'auto', 'final', or 'time_average'."
    )


def waveform_objective_from_z(
    z,
    times,
    bounds,
    base_machine=None,
    base_actuator=None,
    base_sim=None,
    objective="Q",
    keys=("Ip_MA", "P_aux_MW"),
    smoothness_weight: float = 0.0,
    slew_weight: float = 0.0,
    constraints=DEFAULT_CONSTRAINTS,
    constraint_smooth_width: float = 1.0e-2,
    objective_weights=None,
    fixed_control_values=None,
    fixed_start: bool = False,
    fixed_final: bool = False,
    fixed_endpoint_keys=None,
    waveform_constraint_mode: str = "auto",
):
    """AD-ready waveform objective.

    z has shape [num_keys, num_control_points].  This objective integrates
    P_fus and P_aux over the rollout and supports regularization of waveform
    smoothness/slew.
    """
    from .waveforms import apply_waveform_controls
    from .solver import initial_state, step
    from .zero_d import zero_d_enabled, initial_state_0d, zero_d_step_with_dt

    if base_machine is None:
        base_machine = MachineConfig()
    if base_actuator is None:
        base_actuator = ActuatorConfig()
    if base_sim is None:
        base_sim = make_gradient_friendly_sim(
            n_steps=int(len(times)),
            dt=float(times[1] - times[0]) if len(times) > 1 else 1e-3,
        )

    z = jnp.asarray(z, dtype=jnp.float32)
    control_values = {}
    for i, key in enumerate(keys):
        lo, hi = bounds[key]
        control_values[key] = controls_from_waveform_z(z[i], lo, hi)
    control_values = apply_fixed_endpoint_values(
        control_values, fixed_control_values, keys,
        fixed_start=fixed_start, fixed_final=fixed_final,
        fixed_endpoint_keys=fixed_endpoint_keys,
    )

    waveform = make_piecewise_waveform_from_controls(times, control_values)
    m0, a0 = apply_waveform_controls(base_machine, base_actuator, waveform, times[0])
    state = initial_state_0d(m0, a0, base_sim) if zero_d_enabled(base_sim) else initial_state(m0, a0, base_sim)

    is_zero_d = zero_d_enabled(base_sim)
    needs_time_integrals = _waveform_objective_needs_time_integrals(objective, objective_weights)
    constraint_mode = _normalize_waveform_constraint_mode(
        waveform_constraint_mode, needs_time_integrals=needs_time_integrals
    )
    use_time_constraints = constraint_mode == "time_average"
    needs_step_metrics = needs_time_integrals or use_time_constraints
    t0 = jnp.asarray(0.0, dtype=jnp.float32)
    zero = jnp.asarray(0.0, dtype=jnp.float32)

    def body(carry, i):
        state_i, t_i, P_fus_int_i, P_aux_int_i, constraint_penalty_int_i, last_Q_i, last_sample_t_i = carry
        # Sample the optimized control waveform on the actual simulation grid.
        # For 0.5D, the physical grid is the accumulated adaptive dt_eff, not
        # simply i*sim.dt.  For 1.5D, dt_eff == sim.dt.
        mt, at = apply_waveform_controls(base_machine, base_actuator, waveform, t_i)
        if is_zero_d:
            state_next, dt_step = zero_d_step_with_dt(state_i, mt, at, base_sim)
        else:
            state_next = step(state_i, mt, at, base_sim)
            dt_step = jnp.asarray(base_sim.dt, dtype=jnp.float32)

        if needs_step_metrics:
            metrics_i = differentiable_metrics_ad_clean(state_next, mt, at, base_sim)
            P_fus_int_next = P_fus_int_i + metrics_i["P_fus_MW"] * dt_step
            P_aux_int_next = P_aux_int_i + jnp.maximum(at.P_aux_MW, 1.0) * dt_step
            if use_time_constraints:
                penalty_i, _ = evaluate_constraints(
                    metrics_i,
                    constraints=constraints,
                    smooth_width=constraint_smooth_width,
                )
                constraint_penalty_int_next = constraint_penalty_int_i + penalty_i * dt_step
            else:
                constraint_penalty_int_next = constraint_penalty_int_i
            last_Q_next = metrics_i["Q"]
        else:
            P_fus_int_next = P_fus_int_i
            P_aux_int_next = P_aux_int_i
            constraint_penalty_int_next = constraint_penalty_int_i
            last_Q_next = last_Q_i
        return (
            state_next,
            t_i + dt_step,
            P_fus_int_next,
            P_aux_int_next,
            constraint_penalty_int_next,
            last_Q_next,
            t_i,
        ), None

    init_carry = (state, t0, zero, zero, zero, zero, t0)
    final_carry, _ = jax.lax.scan(body, init_carry, jnp.arange(int(base_sim.n_steps)))
    state, t, P_fus_int, P_aux_int, constraint_penalty_int, last_Q, last_sample_t = final_carry

    mf, af = apply_waveform_controls(base_machine, base_actuator, waveform, last_sample_t)
    metrics = dict(differentiable_metrics_ad_clean(state, mf, af, base_sim))
    metrics["last_step_Q"] = jnp.where(needs_step_metrics, last_Q, metrics["Q"])

    duration = jnp.maximum(t, 1e-12)
    if needs_time_integrals:
        P_fus_avg = P_fus_int / duration
        P_aux_avg = P_aux_int / duration
        Q_integral = P_fus_int / (P_aux_int + 1e-6)
    else:
        P_fus_avg = metrics["P_fus_MW"]
        P_aux_avg = jnp.maximum(jnp.asarray(af.P_aux_MW, dtype=metrics["P_fus_MW"].dtype), 1.0)
        Q_integral = metrics["Q"]
    constraint_penalty_avg = constraint_penalty_int / duration

    reg_penalty, reg_details = waveform_regularization(
        control_values, bounds, keys,
        smoothness_weight=smoothness_weight,
        slew_weight=slew_weight,
    )

    metrics["P_fus_integral_MJ_equiv"] = P_fus_int
    metrics["P_aux_integral_MJ_equiv"] = P_aux_int
    metrics["P_fus_avg_MW"] = P_fus_avg
    metrics["P_aux_avg_MW"] = P_aux_avg
    metrics["Q_integral"] = Q_integral
    metrics["constraint_penalty_integral_avg"] = constraint_penalty_avg
    metrics["waveform_regularization"] = reg_penalty
    metrics["rollout_duration_s"] = duration
    metrics["waveform_constraint_mode_final"] = jnp.asarray(1.0 if constraint_mode == "final" else 0.0)
    final_constraint_penalty, final_constraint_details = evaluate_constraints(
        metrics,
        constraints=constraints,
        smooth_width=constraint_smooth_width,
    )
    metrics["constraint_penalty_final"] = final_constraint_penalty
    metrics.update(final_constraint_details)
    metrics.update(reg_details)

    perf_value, objective_details = _weighted_objective_from_metrics(
        metrics, objective, objective_weights, waveform=True
    )
    # Constraints are always separate from the performance objective.  For
    # waveform optimization use the time-averaged penalty accumulated during
    # the rollout, plus waveform regularization.
    penalty_used = final_constraint_penalty if constraint_mode == "final" else constraint_penalty_avg
    value = perf_value - penalty_used - reg_penalty
    metrics.update(objective_details)
    metrics["constraint_penalty_used"] = penalty_used
    metrics["objective_value"] = value
    return value, metrics


def value_and_grad_waveform_objective(
    z,
    times,
    bounds,
    base_machine=None,
    base_actuator=None,
    base_sim=None,
    objective="Q",
    keys=("Ip_MA", "P_aux_MW"),
    gradient_mode="autodiff",
    eps=1.0e-3,
    smoothness_weight: float = 0.0,
    slew_weight: float = 0.0,
    constraints=DEFAULT_CONSTRAINTS,
    constraint_smooth_width: float = 1.0e-2,
    objective_weights=None,
    fixed_control_values=None,
    fixed_start: bool = False,
    fixed_final: bool = False,
    fixed_endpoint_keys=None,
):
    """Return (value, metrics, grad) for waveform objective."""
    if gradient_mode == "autodiff":
        (value, metrics), grad = jax.value_and_grad(
            lambda zz: waveform_objective_from_z(
                zz, times, bounds, base_machine, base_actuator, base_sim,
                objective, keys, smoothness_weight, slew_weight,
                constraints, constraint_smooth_width, objective_weights,
                fixed_control_values, fixed_start, fixed_final, fixed_endpoint_keys,
                waveform_constraint_mode,
            ),
            has_aux=True,
        )(z)
        return value, metrics, grad

    if gradient_mode != "finite_difference":
        raise ValueError('gradient_mode must be "autodiff" or "finite_difference".')

    z = jnp.asarray(z, dtype=jnp.float32)
    value, metrics = waveform_objective_from_z(
        z, times, bounds, base_machine, base_actuator, base_sim,
        objective, keys, smoothness_weight, slew_weight,
        constraints, constraint_smooth_width, objective_weights,
        fixed_control_values, fixed_start, fixed_final, fixed_endpoint_keys,
        waveform_constraint_mode,
    )
    flat = z.reshape(-1)
    grads = []
    eye = jnp.eye(flat.size, dtype=jnp.float32)
    for i in range(flat.size):
        zp = (flat + eps * eye[i]).reshape(z.shape)
        zm = (flat - eps * eye[i]).reshape(z.shape)
        yp, _ = waveform_objective_from_z(
            zp, times, bounds, base_machine, base_actuator, base_sim,
            objective, keys, smoothness_weight, slew_weight,
            constraints, constraint_smooth_width, objective_weights,
            fixed_control_values, fixed_start, fixed_final, fixed_endpoint_keys,
            waveform_constraint_mode,
        )
        ym, _ = waveform_objective_from_z(
            zm, times, bounds, base_machine, base_actuator, base_sim,
            objective, keys, smoothness_weight, slew_weight,
            constraints, constraint_smooth_width, objective_weights,
            fixed_control_values, fixed_start, fixed_final, fixed_endpoint_keys,
            waveform_constraint_mode,
        )
        grads.append((yp - ym) / (2.0 * eps))
    grad = jnp.asarray(grads).reshape(z.shape)
    return value, metrics, grad


def waveform_controls_from_z(z, bounds, keys=("Ip_MA", "P_aux_MW")):
    """Return dict of physical waveform control-point arrays."""
    z = jnp.asarray(z, dtype=jnp.float32)
    out = {}
    for i, key in enumerate(keys):
        lo, hi = bounds[key]
        out[key] = controls_from_waveform_z(z[i], lo, hi)
    return out


def initial_waveform_z_from_values(control_values, bounds, keys=("Ip_MA", "P_aux_MW")):
    """Build unconstrained waveform z from physical control-point arrays."""
    z_list = []
    for key in keys:
        lo, hi = bounds[key]
        z_list.append(z_from_waveform_controls(control_values[key], lo, hi))
    return jnp.stack(z_list, axis=0)


def _normalize_fixed_endpoint_keys(fixed_endpoint_keys, keys=("Ip_MA", "P_aux_MW")):
    """Return tuple of waveform keys whose endpoints should be fixed."""
    if fixed_endpoint_keys is None:
        return tuple(keys)
    if isinstance(fixed_endpoint_keys, str):
        txt = fixed_endpoint_keys.strip()
        if txt.lower() in {"", "all", "*"}:
            return tuple(keys)
        return tuple(k.strip() for k in txt.split(",") if k.strip())
    return tuple(fixed_endpoint_keys)


def apply_fixed_endpoint_mask(
    z,
    z_initial,
    fixed_endpoints=True,
    fixed_endpoint_keys=None,
    keys=("Ip_MA", "P_aux_MW"),
    *,
    fixed_start: bool | None = None,
    fixed_final: bool | None = None,
):
    """Keep selected waveform endpoint control points fixed in z-space."""
    if fixed_start is None and fixed_final is None:
        fixed_start = bool(fixed_endpoints)
        fixed_final = bool(fixed_endpoints)
    else:
        fixed_start = bool(fixed_start)
        fixed_final = bool(fixed_final)
    if not fixed_start and not fixed_final:
        return z
    z_new = z
    fixed_keys = _normalize_fixed_endpoint_keys(fixed_endpoint_keys, keys)
    for i, key in enumerate(keys):
        if key in fixed_keys:
            if fixed_start:
                z_new = z_new.at[i, 0].set(z_initial[i, 0])
            if fixed_final:
                z_new = z_new.at[i, -1].set(z_initial[i, -1])
    return z_new


def apply_fixed_endpoint_gradient_mask(
    grad,
    fixed_endpoint_keys=None,
    keys=("Ip_MA", "P_aux_MW"),
    *,
    fixed_start: bool = False,
    fixed_final: bool = False,
):
    """Zero gradients for endpoints that are fixed."""
    if not fixed_start and not fixed_final:
        return grad
    out = grad
    fixed_keys = _normalize_fixed_endpoint_keys(fixed_endpoint_keys, keys)
    for i, key in enumerate(keys):
        if key in fixed_keys:
            if fixed_start:
                out = out.at[i, 0].set(0.0)
            if fixed_final:
                out = out.at[i, -1].set(0.0)
    return out


def count_free_waveform_variables(
    control_values,
    keys=("Ip_MA", "P_aux_MW"),
    *,
    fixed_start: bool = False,
    fixed_final: bool = False,
    fixed_endpoint_keys=None,
):
    """Count waveform scalar variables that are actually free to update."""
    fixed_keys = set(_normalize_fixed_endpoint_keys(fixed_endpoint_keys, keys))
    n = 0
    for key in keys:
        size = int(jnp.asarray(control_values[key]).size)
        fixed = 0
        if key in fixed_keys and size > 0:
            fixed += int(bool(fixed_start))
            if size > 1:
                fixed += int(bool(fixed_final))
            elif fixed_final and not fixed_start:
                fixed += 1
            fixed = min(fixed, size)
        n += max(size - fixed, 0)
    return n


def apply_fixed_endpoint_values(
    control_values,
    fixed_control_values=None,
    keys=("Ip_MA", "P_aux_MW"),
    *,
    fixed_start: bool = False,
    fixed_final: bool = False,
    fixed_endpoint_keys=None,
):
    """Set selected physical endpoint values exactly.

    This is needed because fixed ramp-up endpoints such as P_aux_MW=0 can lie
    outside the interior optimization bounds.
    """
    if fixed_control_values is None or (not fixed_start and not fixed_final):
        return control_values
    fixed_keys = set(_normalize_fixed_endpoint_keys(fixed_endpoint_keys, keys))
    out = dict(control_values)
    for key in keys:
        if key not in fixed_keys or key not in out or key not in fixed_control_values:
            continue
        arr = jnp.asarray(out[key], dtype=jnp.float32)
        fixed_arr = jnp.asarray(fixed_control_values[key], dtype=jnp.float32)
        if fixed_start and arr.size > 0 and fixed_arr.size > 0:
            arr = arr.at[0].set(fixed_arr[0])
        if fixed_final and arr.size > 0 and fixed_arr.size > 0:
            arr = arr.at[-1].set(fixed_arr[-1])
        out[key] = arr
    return out


def optimize_waveform_optax(
    z0=None,
    times=None,
    initial_control_values=None,
    bounds=None,
    objective="Q",
    keys=("Ip_MA", "P_aux_MW"),
    base_machine=None,
    base_actuator=None,
    base_sim=None,
    n_iter: int = 10,
    learning_rate: float = 1.0e-2,
    gradient_mode: str = "autodiff",
    smoothness_weight: float = 0.0,
    slew_weight: float = 0.0,
    constraints=DEFAULT_CONSTRAINTS,
    constraint_smooth_width: float = 1.0e-2,
    objective_weights=None,
    profile_timing: bool = False,
    fixed_endpoints: bool = True,
    fixed_endpoint_keys=None,
    fixed_start: bool | None = None,
    fixed_final: bool | None = None,
    waveform_constraint_mode: str = "auto",
    max_grad_norm: float | None = None,
    max_update_norm: float | None = None,
    iteration_callback: Callable[[dict[str, Any]], None] | None = None,
):
    """Optimize bounded piecewise-linear waveform control points.

    Supports AD gradients, endpoint fixing, and smoothness/slew penalties.
    """
    if times is None:
        times = jnp.linspace(0.0, 1.0, 8)
    else:
        times = jnp.asarray(times, dtype=jnp.float32)

    if bounds is None:
        bounds = {
            **DEFAULT_CONTROL_BOUNDS,
            "Ip_MA": (2.0, 17.0),
            "P_aux_MW": (0.0, 120.0),
            "greenwald_fraction_target": (0.5, 1.05),
            "heat_center": (0.05, 0.75),
            "heat_width": (0.10, 0.60),
            "f_e_heat": (0.2, 0.9),
        }

    if base_machine is None:
        base_machine = MachineConfig()
    if base_actuator is None:
        base_actuator = ActuatorConfig()
    if base_sim is None:
        dt = float(times[1] - times[0]) if len(times) > 1 else 1.0e-3
        base_sim = make_gradient_friendly_sim(n_steps=int(len(times)), dt=dt)

    if z0 is None:
        if initial_control_values is None:
            initial_control_values = {}
            for key in keys:
                lo, hi = bounds[key]
                initial_control_values[key] = 0.5 * (lo + hi) * jnp.ones_like(times)
        z = initial_waveform_z_from_values(initial_control_values, bounds, keys)
    else:
        z = jnp.asarray(z0, dtype=jnp.float32)

    fixed_control_values = {
        key: jnp.asarray(initial_control_values[key], dtype=jnp.float32)
        for key in keys
        if initial_control_values is not None and key in initial_control_values
    }

    if fixed_start is None and fixed_final is None:
        effective_fixed_start = bool(fixed_endpoints)
        effective_fixed_final = bool(fixed_endpoints)
    else:
        effective_fixed_start = bool(fixed_start)
        effective_fixed_final = bool(fixed_final)

    z_initial = z
    z = apply_fixed_endpoint_mask(
        z, z_initial, fixed_endpoints, fixed_endpoint_keys, keys,
        fixed_start=effective_fixed_start, fixed_final=effective_fixed_final,
    )

    try:
        import optax  # type: ignore
        tx = optax.adam(learning_rate)
        opt_state = tx.init(z)
        use_optax = True
    except Exception:
        optax = None
        tx = None
        opt_state = None
        use_optax = False
        m = jnp.zeros_like(z)
        v = jnp.zeros_like(z)

    history = []
    timing_rows = []
    objective_eval = lambda zz: waveform_objective_from_z(
        zz, times, bounds, base_machine, base_actuator, base_sim,
        objective, keys, smoothness_weight, slew_weight, constraints,
        constraint_smooth_width, objective_weights,
        fixed_control_values, effective_fixed_start, effective_fixed_final, fixed_endpoint_keys,
        waveform_constraint_mode,
    )
    if gradient_mode == "autodiff":
        objective_eval_jit = jax.jit(objective_eval)
        value_and_grad_eval_jit = jax.jit(jax.value_and_grad(objective_eval, has_aux=True))
    else:
        objective_eval_jit = None
        value_and_grad_eval_jit = None

    for it in range(n_iter):
        forward_time = None
        grad_total_time = None
        if profile_timing and gradient_mode == "autodiff":
            t_f = time.perf_counter()
            _fv, _fm = objective_eval_jit(z)
            try:
                jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, (_fv, _fm))
            except Exception:
                pass
            forward_time = time.perf_counter() - t_f
        t_g = time.perf_counter()
        if gradient_mode == "autodiff":
            (value, metrics), grad = value_and_grad_eval_jit(z)
        else:
            value, metrics, grad = value_and_grad_waveform_objective(
                z, times, bounds, base_machine, base_actuator, base_sim,
                objective, keys, gradient_mode=gradient_mode,
                smoothness_weight=smoothness_weight,
                slew_weight=slew_weight,
                constraints=constraints,
                constraint_smooth_width=constraint_smooth_width,
                objective_weights=objective_weights,
                fixed_control_values=fixed_control_values,
                fixed_start=effective_fixed_start,
                fixed_final=effective_fixed_final,
                fixed_endpoint_keys=fixed_endpoint_keys,
                waveform_constraint_mode=waveform_constraint_mode,
            )
        try:
            jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, (value, metrics, grad))
        except Exception:
            pass
        grad_total_time = time.perf_counter() - t_g
        grad, grad_info = _sanitize_and_clip_gradient(grad, max_grad_norm=max_grad_norm)

        grad = apply_fixed_endpoint_gradient_mask(
            grad, fixed_endpoint_keys, keys,
            fixed_start=effective_fixed_start, fixed_final=effective_fixed_final,
        )

        z_before = z
        controls_before = apply_fixed_endpoint_values(
            waveform_controls_from_z(z_before, bounds, keys),
            fixed_control_values, keys,
            fixed_start=effective_fixed_start, fixed_final=effective_fixed_final,
            fixed_endpoint_keys=fixed_endpoint_keys,
        )

        # Maximize objective; optax is descent-style.
        if use_optax:
            updates, opt_state = tx.update(-grad, opt_state, z)
            z_candidate = optax.apply_updates(z, updates)
        else:
            step, m, v = _adam_fallback_update(-grad, m, v, it + 1, learning_rate)
            z_candidate = z - step

        z, update_info = _clip_update_delta(z_before, z_candidate, max_update_norm=max_update_norm)
        z = apply_fixed_endpoint_mask(
            z, z_initial, fixed_endpoints, fixed_endpoint_keys, keys,
            fixed_start=effective_fixed_start, fixed_final=effective_fixed_final,
        )

        controls_phys = apply_fixed_endpoint_values(
            waveform_controls_from_z(z, bounds, keys),
            fixed_control_values, keys,
            fixed_start=effective_fixed_start, fixed_final=effective_fixed_final,
            fixed_endpoint_keys=fixed_endpoint_keys,
        )
        row = {
            "iter": it,
            "objective": value,
            "metrics": metrics,
            "grad_z": grad,
            "controls_phys": controls_phys,
            "controls_phys_before_update": controls_before,
            "used_optax": use_optax,
            **grad_info,
            **update_info,
        }
        history.append(row)
        if iteration_callback is not None:
            iteration_callback(row)
        if profile_timing:
            timing_rows.append({
                "iter": it,
                "forward_time_s": forward_time,
                "value_and_grad_time_s": grad_total_time,
                "backprop_estimate_s": None if forward_time is None or grad_total_time is None else max(grad_total_time - forward_time, 0.0),
            })

    if objective_eval_jit is not None:
        final_value, final_metrics = objective_eval_jit(z)
    else:
        final_value, final_metrics = objective_eval(z)
    final_controls = apply_fixed_endpoint_values(
        waveform_controls_from_z(z, bounds, keys),
        fixed_control_values, keys,
        fixed_start=effective_fixed_start, fixed_final=effective_fixed_final,
        fixed_endpoint_keys=fixed_endpoint_keys,
    )
    final_waveform = make_piecewise_waveform_from_controls(times, final_controls, name="optimized_waveform")

    return {
        "z": z,
        "times": times,
        "controls_phys": final_controls,
        "waveform": final_waveform,
        "objective": final_value,
        "metrics": final_metrics,
        "history": history,
        "used_optax": use_optax,
        "gradient_mode": gradient_mode,
        "keys": keys,
        "bounds": bounds,
        "fixed_endpoints": fixed_endpoints,
        "fixed_start": effective_fixed_start,
        "fixed_final": effective_fixed_final,
        "fixed_endpoint_keys": _normalize_fixed_endpoint_keys(fixed_endpoint_keys, keys),
        "waveform_constraint_mode": waveform_constraint_mode,
        "smoothness_weight": smoothness_weight,
        "slew_weight": slew_weight,
        "timing": timing_rows,
    }
