import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from tokagrad import MachineConfig, ActuatorConfig, SimulationConfig
from tokagrad.density import target_edge_ne20, target_nbar20, greenwald_fraction_target
from tokagrad.controls import (
    PytreeMachineConfig,
    PytreeActuatorConfig,
    apply_control_vector,
    controls_from_unconstrained,
    unconstrained_from_controls,
)
from tokagrad.current import current_components_from_state
from tokagrad.diagnostics import compute_diagnostics, fusion_power_MW, beta_normalized, volume_average_axis_augmented
from tokagrad.equilibrium import solve_fixed_boundary_equilibrium, read_geqdsk
from tokagrad.grid import make_grid_from_config, axis_augmented_profile, boundary_augmented_profile
from tokagrad.heating import total_heating_sources
from tokagrad.solver import initial_state, step, simulate_final, effective_phi_dot_over_phi
from tokagrad.zero_d import zero_d_step_with_dt, initial_state_0d, simulate_0d_final, ipb98y2_tau_E
from tokagrad.app_runtime import advance_radial_chunk_jit, advance_zero_d_chunk_with_time_jit
from tokagrad.transport import compute_diffusivity
from tokagrad.waveforms import iter_baseline_like_waveform, apply_waveform_controls
from tokagrad.optim_differentiable import q_profile_ad_clean, simulate_final_unrolled_ad


st.set_page_config(page_title="TokaGrad Digital Twin", layout="wide")
st.title("TokaGrad interactive differentiable digital twin")


CONTROL_OPTIONS = ("Ip_MA", "Bt", "greenwald_fraction_target", "P_aux_MW", "R0", "a", "kappa", "delta")
DEFAULT_BOUNDS = {
    "Ip_MA": (8.0, 17.0),
    "Bt": (3.0, 7.0),
    "greenwald_fraction_target": (0.5, 1.05),
    "P_aux_MW": (5.0, 120.0),
    "R0": (4.5, 7.5),
    "a": (1.0, 2.5),
    "kappa": (1.2, 2.1),
    "delta": (0.0, 0.55),
}


def _np(x):
    return np.asarray(x)


def _axis_aug_np(rho, y, edge_value=None):
    rr, yy = boundary_augmented_profile(rho, y, edge_value=edge_value)
    return _np(rr), _np(yy)


def _close_curve(x):
    x = _np(x)
    return np.r_[x, x[0]] if x.size else x


def _plot_surfaces_app(ax, eq, sim):
    R = _np(eq.R)
    Z = _np(eq.Z)
    stride = max(1, int(sim.nr) // 10)
    for i in range(0, int(sim.nr), stride):
        ax.plot(_close_curve(R[i]), _close_curve(Z[i]), lw=0.75, alpha=0.55)
    ax.plot(_close_curve(R[-1]), _close_curve(Z[-1]), "k-", lw=1.8, label="computed LCFS")
    if getattr(sim, "equilibrium_model", "") == "geqdsk_prescribed" and getattr(sim, "geqdsk_path", ""):
        try:
            g = read_geqdsk(getattr(sim, "geqdsk_path"))
            rbd = np.asarray(g.get("rbbbs", []), dtype=float)
            zbd = np.asarray(g.get("zbbbs", []), dtype=float)
            if rbd.size >= 4 and zbd.size == rbd.size:
                src = str(g.get("boundary_source", "G-EQDSK"))
                ax.plot(_close_curve(rbd), _close_curve(zbd), "r--", lw=1.2, label=f"G-EQDSK ({src})")
            if "rmaxis" in g and "zmaxis" in g:
                ax.plot([float(g["rmaxis"])], [float(g["zmaxis"])], "kx", ms=5, label="axis")
        except Exception as exc:
            ax.text(0.02, 0.98, f"G-EQDSK boundary unavailable: {exc}", transform=ax.transAxes,
                    va="top", ha="left", fontsize=7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=7, loc="best")


def ipb98y2_tau(machine, actuator, sim):
    """Approximate IPB98(y,2) confinement time [s].

    tau = 0.0562 Ip^0.93 Bt^0.15 P^-0.69 n^0.41 M^0.19 R^1.97 eps^0.58 kappa^0.78
    Units: MA, T, MW, 1e19 m^-3, amu, m.
    """
    Ip_MA = max(float(machine.Ip) / 1e6, 0.1)
    Bt = max(float(machine.Bt), 0.1)
    P = max(float(actuator.P_aux_MW), 1.0)
    n19 = max(10.0 * float(target_nbar20(machine, actuator, sim)), 0.5)
    # D-T effective ion mass.
    M = 2.5 if str(machine.plasma_species).upper().replace("-", "") == "DT" else 2.0
    R = max(float(machine.R0), 0.5)
    eps = max(float(machine.a) / R, 0.05)
    kappa = max(float(machine.kappa), 0.5)
    tau = 0.0562 * Ip_MA**0.93 * Bt**0.15 * P**(-0.69) * n19**0.41 * M**0.19 * R**1.97 * eps**0.58 * kappa**0.78
    return float(np.clip(tau, 0.01, 10.0))


def make_demo_sim(machine, actuator, nr=None, dt=None, relax_multiplier=1.0):
    """Live-simulation config based on config.py defaults.

    The live app advances one explicit user-controlled step at a time from the
    fixed actuator setting.  Therefore this function no longer constructs an
    ITER ramp waveform or a 5-tau relaxation scenario; it simply takes the
    default SimulationConfig and applies UI overrides for grid and dt.
    """
    base = SimulationConfig()
    return replace(
        base,
        nr=int(nr if nr is not None else base.nr),
        ntheta=int(base.ntheta),
        dt=float(dt if dt is not None else base.dt),
        n_steps=1,
    )


def make_ad_relax_sim(machine, actuator, base_sim=None, nr=4, max_steps=16):
    """Small AD-safe simulation for one confinement time.

    Sensitivity/optimization tabs use the final state after 1*tau_E of
    time evolution.  The number of unrolled steps is capped for UI
    responsiveness, and dt is adjusted so the total simulated time is tau_E.
    """
    if base_sim is None:
        base_sim = SimulationConfig()
    tau = ipb98y2_tau(machine, actuator, base_sim)
    total_time = 1.0 * tau
    n_steps = int(max(2, max_steps))
    dt_eff = float(total_time / n_steps)
    return replace(
        base_sim,
        nr=int(nr),
        ntheta=min(int(getattr(base_sim, "ntheta", 4)), 8),
        dt=dt_eff,
        n_steps=n_steps,
        differentiable_smooth_mode=True,
        # Avoid direct non-differentiable pedestal projection in app AD demos.
        pedestal_projection_fraction=0.0,
    )


def compact_sim(machine, actuator, dt=2e-3, nr=8, relax_multiplier=5.0):
    tau = ipb98y2_tau(machine, actuator, None)
    n_steps = int(np.clip(np.ceil(relax_multiplier * tau / dt), 2, 80))
    base = make_demo_sim(machine, actuator, nr=nr, dt=dt, relax_multiplier=relax_multiplier)
    return replace(base, n_steps=n_steps)


def build_sidebar():
    st.sidebar.header("Live controls")

    st.sidebar.subheader("Run")
    model_choice = st.sidebar.selectbox(
        "Simulation model",
        ["1.5d", "0d_fast"],
        index=0,
        format_func=lambda x: "1.5D radial diffusion" if x == "1.5d" else "0.5D fast energy balance",
    )
    running = st.sidebar.toggle("Run continuously", value=st.session_state.get("running", False))
    st.session_state.running = running
    steps_per_refresh = st.sidebar.slider("Steps per refresh", 1, 500, 100, 1)
    #refresh_delay = st.sidebar.slider("Refresh delay [s]", 0.0, 0.5, 0.0, 0.01)
    use_jit_chunk = st.sidebar.checkbox(
        "Use JIT chunk advance",
        value=bool(st.session_state.get("use_jit_chunk", True)),
        help=(
            "Advance steps_per_refresh steps in one JAX/XLA chunk. "
            "The first call after changing numerics may compile for a few seconds, "
            "then subsequent refreshes are much faster."
        ),
    )
    st.session_state.use_jit_chunk = bool(use_jit_chunk)
    reset = st.sidebar.button("Reset simulation", width='stretch')
    step_once = st.sidebar.button("Step once", width='stretch')

    # The app now starts from fixed actuators defined by config.py defaults.
    # No baseline waveform is applied unless a script explicitly asks for it.
    follow_waveform = False
    waveform_horizon = 20.0
    show_waveform = False

    default_machine = MachineConfig()
    default_actuator = ActuatorConfig()
    default_sim = SimulationConfig()

    st.sidebar.subheader("Machine")
    R0 = st.sidebar.slider("R0 [m]", 4.0, 8.0, float(default_machine.R0), 0.1)
    a = st.sidebar.slider("a [m]", 0.8, 3.0, float(default_machine.a), 0.05)
    kappa = st.sidebar.slider("kappa", 1.0, 2.4, float(default_machine.kappa), 0.05)
    delta = st.sidebar.slider("delta", 0.0, 0.6, float(default_machine.delta), 0.01)
    Bt = st.sidebar.slider("Bt [T]", 2.0, 8.0, float(default_machine.Bt), 0.1)
    Ip_MA = st.sidebar.slider("Ip [MA]", 2.0, 18.0, float(default_machine.Ip) / 1e6, 0.1)

    st.sidebar.subheader("Actuators")
    P_aux = st.sidebar.slider("P_aux [MW]", 0.0, 150.0, float(default_actuator.P_aux_MW), 1.0)
    greenwald_target = st.sidebar.slider("Greenwald target", 0.4, 1.1, float(default_actuator.greenwald_fraction_target), 0.02)
    edge_frac = st.sidebar.slider("Edge density / nbar", 0.05, 0.6, float(default_actuator.greenwald_edge_density_fraction), 0.01)

    st.sidebar.subheader("Numerics")
    nr = st.sidebar.select_slider("nr", options=[8, 16, 32, 64], value=int(default_sim.nr) if int(default_sim.nr) in [16,24,32,48] else 48)
    dt_options = [1e-3, 2e-3, 5e-3, 1e-2]
    dt = st.sidebar.select_slider("dt [s]", options=dt_options, value=float(default_sim.dt) if float(default_sim.dt) in dt_options else 2e-3)
    st.sidebar.subheader("Pedestal")

    density_evolution_model = st.sidebar.selectbox(
        "Density evolution model",
        [
            "greenwald_feedback",
            "greenwald_rescale_initial_shape",
            "greenwald_rescale_tanh",
            "diffusive",
            "fixed_initial",
            "reflective",
        ],
        index=[
            "greenwald_feedback",
            "greenwald_rescale_initial_shape",
            "greenwald_rescale_tanh",
            "diffusive",
            "fixed_initial",
            "reflective",
        ].index(default_sim.density_evolution_model) if default_sim.density_evolution_model in [
            "greenwald_feedback",
            "greenwald_rescale_initial_shape",
            "greenwald_rescale_tanh",
            "diffusive",
            "fixed_initial",
            "reflective",
        ] else 0,
        format_func=lambda x: {
            "greenwald_feedback": "Diffusion + Greenwald feedback source",
            "greenwald_rescale_initial_shape": "Initial shape rescaled to target f_G",
            "greenwald_rescale_tanh": "Pedestal-width tanh+core shape rescaled to target f_G",
            "diffusive": "Pure density diffusion",
            "fixed_initial": "Fixed initial density",
            "reflective": "Reflective / particle conserving",
        }.get(x, x),
    )

    st.sidebar.divider()
    st.sidebar.header("Optimization setup")
    obj_pfus = st.sidebar.checkbox("Objective: P_fus", value=True)
    obj_q = st.sidebar.checkbox("Objective: Q", value=False)
    opt_controls = st.sidebar.multiselect(
        "Controls to optimize",
        CONTROL_OPTIONS,
        default=["P_aux_MW", "greenwald_fraction_target"],
    )

    bounds = {}
    with st.sidebar.expander("Control bounds", expanded=False):
        for name in CONTROL_OPTIONS:
            lo0, hi0 = DEFAULT_BOUNDS[name]
            c1, c2 = st.columns(2)
            lo = c1.number_input(f"{name} min", value=float(lo0), key=f"bound_{name}_lo")
            hi = c2.number_input(f"{name} max", value=float(hi0), key=f"bound_{name}_hi")
            bounds[name] = (float(lo), float(hi))

    constraint_qedge = st.sidebar.checkbox("Constraint: q_edge range", value=True)
    qedge_min = st.sidebar.number_input("q_edge min", value=3.0, step=0.1)
    qedge_max = st.sidebar.number_input("q_edge max", value=20.0, step=0.5)
    constraint_beta = st.sidebar.checkbox("Constraint: beta_N range", value=True)
    beta_min = st.sidebar.number_input("beta_N min", value=0.0, step=0.1)
    beta_max = st.sidebar.number_input("beta_N max", value=4.0, step=0.1)
    opt_iter = st.sidebar.slider("Optimization iterations", 1, 50, 8, 1)
    lr = st.sidebar.number_input("Learning rate", value=1e-1, format="%.4g")

    machine = MachineConfig(R0=R0, a=a, kappa=kappa, delta=delta, Bt=Bt, Ip=Ip_MA * 1e6)
    actuator = ActuatorConfig(P_aux_MW=P_aux, greenwald_fraction_target=greenwald_target, greenwald_edge_density_fraction=edge_frac)
    if model_choice == "0d_fast":
        tau_guess = ipb98y2_tau_E(machine, target_nbar20(machine, actuator, default_sim), max(P_aux, 0.05), default_sim)
        dt = float(np.clip(float(default_sim.zero_d_dt_fraction_tauE) * float(tau_guess), 1.0e-3, 1.0))
        st.sidebar.caption(f"0.5D dt auto = 0.1 τE ≈ {dt:.3g} s")
    sim = make_demo_sim(machine, actuator, nr=nr, dt=dt)
    sim = replace(
        sim,
        simulation_model=model_choice,
        density_evolution_model=("greenwald_rescale_tanh" if model_choice == "0d_fast" else density_evolution_model),
        n_steps=1,
    )

    controls = {
        "simulation_model": model_choice,
        "running": running,
        "steps_per_refresh": steps_per_refresh,
        #"refresh_delay": refresh_delay,
        "use_jit_chunk": bool(use_jit_chunk),
        "reset": reset,
        "step_once": step_once,
        "follow_waveform": follow_waveform,
        "waveform_horizon": waveform_horizon,
        "show_waveform": show_waveform,
        "opt_objectives": {"P_fus": obj_pfus, "Q": obj_q},
        "opt_controls": tuple(opt_controls),
        "bounds": bounds,
        "constraint_qedge": constraint_qedge,
        "qedge_min": qedge_min,
        "qedge_max": qedge_max,
        "constraint_beta": constraint_beta,
        "beta_min": beta_min,
        "beta_max": beta_max,
        "opt_iter": opt_iter,
        "learning_rate": float(lr),
        "density_evolution_model": density_evolution_model,
    }
    return machine, actuator, sim, controls


def control_signature(machine, actuator, sim):
    return (
        round(float(machine.R0), 5),
        round(float(machine.a), 5),
        round(float(machine.kappa), 5),
        round(float(machine.delta), 5),
        round(float(machine.Bt), 5),
        round(float(machine.Ip) / 1e6, 5),
        round(float(actuator.P_aux_MW), 5),
        round(float(target_nbar20(machine, actuator, sim)), 5),
        round(float(actuator.greenwald_fraction_target), 5),
    )


def state_layout_signature(sim):
    """Settings that determine the shape/meaning of the persistent live state."""
    return (
        str(getattr(sim, "simulation_model", "1.5d")).lower(),
        int(sim.nr),
        str(getattr(sim, "radial_grid", "uniform")),
        float(getattr(sim, "edge_cluster_power", 2.0)),
        str(getattr(sim, "psi_state_grid", "face")),
    )


def state_matches_layout(state, sim):
    """Return whether a saved state can be used with the current radial layout."""
    if state is None:
        return False
    nr = int(sim.nr)
    psi_size = nr + 1 if str(getattr(sim, "psi_state_grid", "face")).lower() == "face" else nr
    try:
        return (
            int(state.Te.size) == nr
            and int(state.Ti.size) == nr
            and int(state.ne20.size) == nr
            and int(state.psi_ind.size) == psi_size
        )
    except Exception:
        return False


def waveform_effective_configs(manual_machine, manual_actuator, manual_sim, t, horizon):
    wf = iter_baseline_like_waveform(t_end=horizon)
    m, a = apply_waveform_controls(manual_machine, manual_actuator, wf, t)
    nG = (m.Ip / 1.0e6) / (np.pi * m.a**2 + 1e-12)
    fG = float(np.clip(float(getattr(a, "greenwald_fraction_target", 0.9)), 0.4, 1.1))
    return m, a, manual_sim, wf


def init_history():
    return {k: [] for k in [
        "t", "Ip_MA", "Bt", "P_aux_MW", "greenwald_fraction_target", "R0", "a", "kappa", "delta",
        "Te0_keV", "Ti0_keV", "ne_avg_1e20", "P_fus_MW", "Q", "q95", "q_edge", "beta_N", "greenwald_fraction",
        "P_fus_MW/100"
    ]}


def reset_state(machine, actuator, sim):
    if str(getattr(sim, "simulation_model", "1.5d")).lower() in ("0d_fast", "0.5d", "0d"):
        st.session_state.state = initial_state_0d(machine, actuator, sim)
    else:
        st.session_state.state = initial_state(machine, actuator, sim)
    st.session_state.t = 0.0
    st.session_state.step_count = 0
    st.session_state.history = init_history()
    st.session_state.user_override = False
    st.session_state.last_control_signature = None
    st.session_state.surface_svg = None
    st.session_state.surface_svg_sig = None
    st.session_state.state_layout_signature = state_layout_signature(sim)



def _block_until_ready_tree(x):
    """Synchronize a JAX PyTree and return it."""
    return jax.block_until_ready(x)


def advance_live_state_chunk(state, machine, actuator, sim, n_steps, *, use_jit=True):
    """Advance the app state by ``n_steps`` and report timing metadata.

    The JIT path builds a temporary config with ``n_steps`` equal to the live
    chunk length and calls an import-module-level jitted helper.  Keeping the
    helper outside this Streamlit script avoids recompiling only because the app
    reran.  If a particular configuration is not JIT-compatible, the function
    falls back to the old eager Python loop and returns the exception message in
    the metadata instead of killing the app.
    """
    n_steps = int(max(1, n_steps))
    chunk_sim = replace(sim, n_steps=n_steps)
    mode = str(getattr(chunk_sim, "simulation_model", "1.5d")).lower()
    is_zero_d = mode in ("0d_fast", "0.5d", "0d")
    t0 = time.perf_counter()
    used_jit = False
    fallback_reason = ""
    if use_jit:
        try:
            if is_zero_d:
                new_state, simulated_time_s = advance_zero_d_chunk_with_time_jit(state, machine, actuator, chunk_sim)
            else:
                new_state = advance_radial_chunk_jit(state, machine, actuator, chunk_sim)
                simulated_time_s = float(sim.dt) * n_steps
            _block_until_ready_tree((new_state, simulated_time_s))
            used_jit = True
            elapsed = time.perf_counter() - t0
            return new_state, {
                "used_jit": used_jit,
                "elapsed_s": elapsed,
                "steps": n_steps,
                "per_step_s": elapsed / max(n_steps, 1),
                "simulated_time_s": float(simulated_time_s),
                "per_step_physical_s": float(simulated_time_s) / max(n_steps, 1),
                "fallback_reason": fallback_reason,
            }
        except Exception as exc:
            # Some experimental branches may contain Python-side logic that is
            # not JIT-compatible.  Keep the app usable and show the reason.
            fallback_reason = f"JIT chunk fallback: {type(exc).__name__}: {exc}"

    new_state = state
    simulated_time_s = 0.0
    for _ in range(n_steps):
        if is_zero_d:
            new_state, dt_eff = zero_d_step_with_dt(new_state, machine, actuator, sim)
            simulated_time_s += float(dt_eff)
        else:
            new_state = step(new_state, machine, actuator, sim)
            simulated_time_s += float(sim.dt)
    _block_until_ready_tree(new_state)
    elapsed = time.perf_counter() - t0
    return new_state, {
        "used_jit": used_jit,
        "elapsed_s": elapsed,
        "steps": n_steps,
        "per_step_s": elapsed / max(n_steps, 1),
        "simulated_time_s": float(simulated_time_s),
        "per_step_physical_s": float(simulated_time_s) / max(n_steps, 1),
        "fallback_reason": fallback_reason,
    }



def compute_live_precomputed(state, machine, actuator, sim):
    """Compute live diagnostics inputs once and share them across UI panels."""
    rho, _, _ = make_grid_from_config(sim.nr, machine.a, sim)
    eq = solve_fixed_boundary_equilibrium(state, machine, actuator, sim)
    currents = current_components_from_state(rho, state, machine, actuator, sim, eq=eq)
    _, _, heat = total_heating_sources(rho, state, machine, actuator, sim, eq=eq)
    diag = compute_diagnostics(
        state,
        machine,
        actuator,
        sim,
        rho=rho,
        eq=eq,
        current_components=currents,
        heat_diag=heat,
    )
    phi_rate = effective_phi_dot_over_phi(state, eq, sim)
    return {
        "rho": rho,
        "eq": eq,
        "currents": currents,
        "heat": heat,
        "diag": diag,
        "phi_rate": phi_rate,
    }


def surface_cache_signature(machine, actuator, sim):
    return (
        round(float(machine.R0), 5),
        round(float(machine.a), 5),
        round(float(machine.kappa), 5),
        round(float(machine.delta), 5),
        round(float(machine.Bt), 5),
        round(float(machine.Ip) / 1e6, 5),
        int(getattr(sim, "nr", 0)),
        int(getattr(sim, "ntheta", 0)),
        str(getattr(sim, "equilibrium_model", "")),
        str(getattr(sim, "geqdsk_path", "")),
    )


def _svg_points(xs, ys, xmin, xmax, ymin, ymax, width=360, height=310, pad=18):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    denom_x = max(float(xmax - xmin), 1e-12)
    denom_y = max(float(ymax - ymin), 1e-12)
    xp = pad + (xs - xmin) / denom_x * (width - 2 * pad)
    yp = height - (pad + (ys - ymin) / denom_y * (height - 2 * pad))
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xp, yp))


def make_surface_svg(eq, sim, width=310, height=310):
    """Lightweight static SVG for the boundary/surface panel."""
    R = _np(eq.R)
    Z = _np(eq.Z)
    curves = []
    stride = max(1, int(getattr(sim, "nr", R.shape[0])) // 7)
    for i in range(0, R.shape[0], stride):
        curves.append((R[i], Z[i], "surface"))
    curves.append((R[-1], Z[-1], "lcfs"))

    if getattr(sim, "equilibrium_model", "") == "geqdsk_prescribed" and getattr(sim, "geqdsk_path", ""):
        try:
            g = read_geqdsk(getattr(sim, "geqdsk_path"))
            rbd = np.asarray(g.get("rbbbs", []), dtype=float)
            zbd = np.asarray(g.get("zbbbs", []), dtype=float)
            if rbd.size >= 4 and zbd.size == rbd.size:
                curves.append((rbd, zbd, "geqdsk"))
        except Exception:
            pass

    all_R = np.concatenate([np.asarray(c[0], dtype=float).ravel() for c in curves if len(c[0])])
    all_Z = np.concatenate([np.asarray(c[1], dtype=float).ravel() for c in curves if len(c[1])])
    if all_R.size == 0 or all_Z.size == 0:
        return "<div style='font-size:0.9rem;color:#666'>No surface data.</div>"
    xmin, xmax = float(np.nanmin(all_R)), float(np.nanmax(all_R))
    ymin, ymax = float(np.nanmin(all_Z)), float(np.nanmax(all_Z))
    dx, dy = xmax - xmin, ymax - ymin
    if dx > dy:
        mid = 0.5 * (ymin + ymax)
        ymin, ymax = mid - 0.5 * dx, mid + 0.5 * dx
    else:
        mid = 0.5 * (xmin + xmax)
        xmin, xmax = mid - 0.5 * dy, mid + 0.5 * dy

    polylines = []
    for x, y, kind in curves:
        xx = _close_curve(x)
        yy = _close_curve(y)
        pts = _svg_points(xx, yy, xmin, xmax, ymin, ymax, width=width, height=height)
        if kind == "lcfs":
            polylines.append(f'<polyline points="{pts}" fill="none" stroke="#111827" stroke-width="2.2"/>')
        elif kind == "geqdsk":
            polylines.append(f'<polyline points="{pts}" fill="none" stroke="#b91c1c" stroke-width="1.7" stroke-dasharray="5 3"/>')
        else:
            polylines.append(f'<polyline points="{pts}" fill="none" stroke="#64748b" stroke-opacity="0.52" stroke-width="0.9"/>')
    polylines_str = "".join(polylines)
    return (
        f'<div style="width:100%;">'
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Plasma boundary and nested flux surfaces">'
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="10" fill="white" stroke="#e5e7eb"/>'
        f'{polylines_str}'
        f'<text x="14" y="24" font-size="12" fill="#334155">Boundary / surfaces</text>'
        f'<text x="{width-55}" y="{height-10}" font-size="10" fill="#64748b">R [m]</text>'
        f'<text x="8" y="{height//2}" font-size="10" fill="#64748b" transform="rotate(-90, 8, {height//2})">Z [m]</text>'
        f'</svg></div>'
    )


def get_surface_svg(eq, machine, actuator, sim):
    sig = surface_cache_signature(machine, actuator, sim)
    if st.session_state.get("surface_svg") is None or st.session_state.get("surface_svg_sig") != sig:
        st.session_state.surface_svg = make_surface_svg(eq, sim)
        st.session_state.surface_svg_sig = sig
    return st.session_state.surface_svg

def append_history(machine, actuator, sim, diag):
    h = st.session_state.history
    h["t"].append(float(st.session_state.t))
    h["Ip_MA"].append(float(machine.Ip) / 1e6)
    h["Bt"].append(float(machine.Bt))
    h["P_aux_MW"].append(float(actuator.P_aux_MW))
    h["greenwald_fraction_target"].append(float(actuator.greenwald_fraction_target))
    h["R0"].append(float(machine.R0))
    h["a"].append(float(machine.a))
    h["kappa"].append(float(machine.kappa))
    h["delta"].append(float(machine.delta))
    for k in ["Te0_keV", "Ti0_keV", "ne_avg_1e20", "P_fus_MW", "q95", "q_edge", "beta_N", "greenwald_fraction"]:
        h[k].append(float(diag.get(k, np.nan)))
    h["P_fus_MW/100"].append(float(diag.get("P_fus_MW")) / 100)
    q = float(diag.get("P_fus_MW", 0.0)) / max(float(actuator.P_aux_MW), 1e-6)
    h["Q"].append(q)
    for k in h:
        if len(h[k]) > 800:
            h[k] = h[k][-800:]


def render_waveform_reference(wf, horizon, tnow):
    t = np.linspace(0.0, horizon, 200)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
    for key in ["Ip_MA", "P_aux_MW", "greenwald_fraction_target"]:
        if key in wf.controls:
            ax[0].plot(t, [float(wf.controls[key](tt)) for tt in t], "--", label=key)
    for key in ["kappa", "delta"]:
        if key in wf.controls:
            ax[1].plot(t, [float(wf.controls[key](tt)) for tt in t], "--", label=key)
    for a in ax:
        a.axvline(tnow, color="k", lw=0.8, alpha=0.5)
        a.grid(alpha=0.25)
        a.legend(fontsize=8)
        a.set_xlabel("time [s]")
    ax[0].set_title("Actuator waveform")
    ax[1].set_title("Shape waveform")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def render_history():
    h = st.session_state.history
    if len(h["t"]) == 0:
        st.info("No history yet.")
        return
    df = pd.DataFrame(h)
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Controls")
        _stable_line_chart(df, x="t", y=["Ip_MA", "Bt", "P_aux_MW", "greenwald_fraction_target"], height=230, key="live_history_controls")
    with c2:
        st.caption("Plasma response")
        _stable_line_chart(df, x="t", y=["Te0_keV", "Ti0_keV", "P_fus_MW/100", "Q", "q95", "beta_N"], height=230, key="live_history_response")



def _line_chart_df(rho, columns):
    data = {"rho": np.asarray(rho, dtype=float)}
    for name, values in columns.items():
        data[name] = np.asarray(values, dtype=float)
    return pd.DataFrame(data)


def _stable_line_chart(df, *, x, y, height, key):
    """Render a line chart with a stable element key when supported.

    Streamlit's frontend can leave stale chart elements during rapid reruns if
    several auto-keyed charts are created inside columns.  Providing stable keys
    gives the browser a fixed identity for each live chart.  Older Streamlit
    versions may not support the ``key`` keyword for st.line_chart, so fall back
    gracefully.
    """
    try:
        st.line_chart(df, x=x, y=y, height=height, key=key)
    except TypeError:
        st.line_chart(df, x=x, y=y, height=height)


def render_profiles(state, machine, actuator, sim, *, live=None, surface_svg=None):
    """Render live profiles using Streamlit's lightweight charts.

    The function consumes precomputed equilibrium/current/heating data from
    ``compute_live_precomputed`` so it does not recompute expensive physics.
    The boundary/surface panel is a cached SVG and is not regenerated during
    normal running.
    """
    t0 = time.perf_counter()
    if live is None:
        live = compute_live_precomputed(state, machine, actuator, sim)
    rho = live["rho"]
    eq = live["eq"]
    j_ind, j_bs, j_cd, j_total = live["currents"]
    heat = live["heat"]
    if surface_svg is None:
        surface_svg = get_surface_svg(eq, machine, actuator, sim)

    rr, Te = _axis_aug_np(rho, state.Te, actuator.edge_Te_keV)
    _, Ti = _axis_aug_np(rho, state.Ti, actuator.edge_Ti_keV)
    _, ne19 = _axis_aug_np(rho, state.ne20 * 10.0, target_edge_ne20(machine, actuator, sim) * 10.0)
    df_prof = _line_chart_df(rr, {
        "Te [keV]": Te,
        "Ti [keV]": Ti,
        "ne [1e19/m3]": ne19,
    })

    rrh, Paux = _axis_aug_np(rho, heat["Paux_e"] + heat["Paux_i"])
    _, Palpha = _axis_aug_np(rho, heat["Palpha_e"] + heat["Palpha_i"])
    _, Pohm = _axis_aug_np(rho, heat["Pohm_e"])
    _, Prad = _axis_aug_np(rho, -heat["Prad_e"])
    _, Pnet_e = _axis_aug_np(rho, heat["Pe_net"])
    _, Pnet_i = _axis_aug_np(rho, heat["Pi_net"])
    df_heat = _line_chart_df(rrh, {
        "Aux": Paux,
        "Alpha": Palpha,
        "Ohmic": Pohm,
        "Radiation": Prad,
        #"Net e": Pnet_e,
        #"Net i": Pnet_i,
    })

    rrj, jtot = _axis_aug_np(rho, j_total / 1.0e6)
    rqq, qv = _axis_aug_np(rho, eq.q)
    _, jind = _axis_aug_np(rho, j_ind / 1.0e6)
    _, jbs = _axis_aug_np(rho, j_bs / 1.0e6)
    _, jcd = _axis_aug_np(rho, j_cd / 1.0e6)
    df_cur = _line_chart_df(rrj, {
        "total": jtot,
        "ohmic": jind,
        "BS": jbs,
        "CD": jcd,
        "q": qv,
    })

    c1, c2, c3, c4 = st.columns([1.0, 1.0, 1.0, 0.5])
    with c1:
        st.caption("Profiles")
        _stable_line_chart(df_prof, x="rho", y=["Te [keV]", "Ti [keV]", "ne [1e19/m3]"], height=260, key="live_profiles_temperature_density")
    with c2:
        st.caption("Heating/loss [MW m⁻³]")
        _stable_line_chart(df_heat, x="rho", y=["Aux", "Alpha", "Ohmic", "Radiation"], height=260, key="live_profiles_heating")
    with c3:
        st.caption("Current [MA m⁻²] & q")
        _stable_line_chart(df_cur, x="rho", y=["total", "ohmic", "BS", "CD", "q"], height=260, key="live_profiles_current_q")
    with c4:
        st.caption("Static boundary/surfaces")
        st.markdown(surface_svg, unsafe_allow_html=True)

    return time.perf_counter() - t0


def get_physical_control_values(machine, actuator, sim, names):
    values = []
    for name in names:
        if name == "Ip_MA":
            values.append(float(machine.Ip) / 1e6)
        elif name == "Bt":
            values.append(float(machine.Bt))
        elif name == "greenwald_fraction_target":
            values.append(float(actuator.greenwald_fraction_target))
        elif name == "P_aux_MW":
            values.append(float(actuator.P_aux_MW))
        elif name == "R0":
            values.append(float(machine.R0))
        elif name == "a":
            values.append(float(machine.a))
        elif name == "kappa":
            values.append(float(machine.kappa))
        elif name == "delta":
            values.append(float(machine.delta))
        else:
            raise ValueError(name)
    return jnp.asarray(values, dtype=jnp.float32)


def metrics_from_controls_z(z, base_machine, base_actuator, base_sim, bounds, names, evaluator="pseudo_time"):
    x = controls_from_unconstrained(z, bounds=bounds, names=names)
    pm = PytreeMachineConfig.from_config(base_machine)
    pa = PytreeActuatorConfig.from_config(base_actuator)
    m, a, s = apply_control_vector(x, pm, pa, base_sim, names=names)
    if str(getattr(s, "simulation_model", "1.5d")).lower() in ("0d_fast", "0.5d", "0d"):
        state0 = initial_state_0d(m, a, s)
        final_state = simulate_0d_final(m, a, s, state0=state0)
    else:
        state0 = initial_state(m, a, s)
        # Use a Python-unrolled loop to keep the VJP/JVP path transparent.
        final_state = simulate_final_unrolled_ad(state0, m, a, s)
    rho, _, _ = make_grid_from_config(s.nr, m.a, s)
    eq = solve_fixed_boundary_equilibrium(final_state, m, a, s)
    j_ind, j_bs, j_cd, j_total = current_components_from_state(rho, final_state, m, a, s, eq=eq)
    q = eq.q
    q95_idx = min(int(0.95 * s.nr), s.nr - 1)
    P_fus = fusion_power_MW(final_state, m, s)
    P_aux = jnp.maximum(a.P_aux_MW, 1e-6)
    Q = P_fus / P_aux
    beta_N = beta_normalized(final_state, m, s)
    return {
        "P_fus_MW": P_fus,
        "Q": Q,
        "q_edge": q[-1],
        "q95": q[q95_idx],
        "beta_N": beta_N,
        "x_phys": x,
    }


def q_profile_app(rho, j_total, machine):
    return q_profile_ad_clean(rho, j_total, machine)


def normalized_jacobian(machine, actuator, sim, names, bounds, evaluator="pseudo_time"):
    x0 = get_physical_control_values(machine, actuator, sim, names)
    z0 = unconstrained_from_controls(x0, bounds=bounds, names=names)

    def yvec(zz):
        met = metrics_from_controls_z(zz, machine, actuator, sim, bounds, names, evaluator=evaluator)
        return jnp.asarray([met["P_fus_MW"], met["Q"]])

    y0 = yvec(z0)
    dy_dz = jax.jacobian(yvec)(z0)

    # Convert dz derivative to dx derivative using finite local slope of x(z).
    eps = 1.0e-3
    x_plus = controls_from_unconstrained(z0 + eps, bounds=bounds, names=names)
    x_minus = controls_from_unconstrained(z0 - eps, bounds=bounds, names=names)
    dx_dz = (x_plus - x_minus) / (2 * eps)
    dy_dx = dy_dz / (dx_dz[None, :] + 1e-12)

    norm = dy_dx * (x0[None, :] / (y0[:, None] + 1e-12))
    return _np(y0), _np(norm), _np(x0)


def constraints_from_sidebar(controls):
    cons = []
    if controls["constraint_qedge"]:
        cons.append({"metric": "q_edge", "kind": "lower", "value": float(controls["qedge_min"]), "weight": 20.0})
        cons.append({"metric": "q_edge", "kind": "upper", "value": float(controls["qedge_max"]), "weight": 20.0})
    if controls["constraint_beta"]:
        cons.append({"metric": "beta_N", "kind": "lower", "value": float(controls["beta_min"]), "weight": 10.0})
        cons.append({"metric": "beta_N", "kind": "upper", "value": float(controls["beta_max"]), "weight": 10.0})
    return tuple(cons)


def optimize_scalar_app(machine, actuator, sim, names, bounds, objectives, constraints, n_iter=8, lr=1e-2, evaluator="pseudo_time"):
    x0 = get_physical_control_values(machine, actuator, sim, names)
    z = unconstrained_from_controls(x0, bounds=bounds, names=names)
    baseline = metrics_from_controls_z(z, machine, actuator, sim, bounds, names, evaluator=evaluator)
    P0 = jnp.maximum(baseline["P_fus_MW"], 1e-6)
    Q0 = jnp.maximum(baseline["Q"], 1e-6)

    def smooth_relu(x, width=1e-2):
        return width * jax.nn.softplus(x / width)

    def obj(zz):
        met = metrics_from_controls_z(zz, machine, actuator, sim, bounds, names, evaluator=evaluator)
        value = 0.0
        if objectives.get("P_fus", False):
            value = value + met["P_fus_MW"] / P0
        if objectives.get("Q", False):
            value = value + met["Q"] / Q0
        penalty = 0.0
        for c in constraints:
            v = c["value"] - met[c["metric"]] if c["kind"] == "lower" else met[c["metric"]] - c["value"]
            penalty = penalty + c.get("weight", 1.0) * smooth_relu(v) ** 2
        return value - penalty, met

    grad_fn = jax.value_and_grad(lambda zz: obj(zz)[0])
    history = []
    m = jnp.zeros_like(z)
    v = jnp.zeros_like(z)
    for it in range(int(n_iter)):
        val, grad = grad_fn(z)
        grad_finite = jnp.isfinite(grad)
        if not bool(jnp.all(grad_finite)):
            raise FloatingPointError(f"Non-finite AD gradient in app scalar optimizer: {grad}")
        # Adam ascent
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad * grad
        mh = m / (1 - 0.9 ** (it + 1))
        vh = v / (1 - 0.999 ** (it + 1))
        z = z + lr * mh / (jnp.sqrt(vh) + 1e-8)
        val2, met2 = obj(z)
        history.append({
            "iter": it,
            "objective": float(val2),
            "P_fus_MW": float(met2["P_fus_MW"]),
            "Q": float(met2["Q"]),
            "q_edge": float(met2["q_edge"]),
            "beta_N": float(met2["beta_N"]),
            "grad_norm": float(jnp.linalg.norm(grad)),
        })
    final_val, final_met = obj(z)
    return {
        "z": z,
        "x_phys": controls_from_unconstrained(z, bounds=bounds, names=names),
        "metrics": final_met,
        "objective": final_val,
        "history": history,
        "baseline": baseline,
    }


manual_machine, manual_actuator, manual_sim, ui = build_sidebar()

sig = control_signature(manual_machine, manual_actuator, manual_sim)
if st.session_state.get("last_control_signature") is None:
    st.session_state.last_control_signature = sig
elif sig != st.session_state.last_control_signature:
    if ui["follow_waveform"]:
        st.session_state.user_override = True
    st.session_state.last_control_signature = sig

# Fixed-actuator interactive mode.  The live simulation starts from the
# config.py default state and responds directly to sidebar values; no baseline
# ramp waveform is applied in the app.
use_waveform = False
machine, actuator, sim = manual_machine, manual_actuator, manual_sim
waveform = None

layout_sig = state_layout_signature(sim)
saved_layout_sig = st.session_state.get("state_layout_signature")
layout_changed = saved_layout_sig is not None and saved_layout_sig != layout_sig
state_invalid = not state_matches_layout(st.session_state.get("state"), sim)
if "state" not in st.session_state or ui["reset"] or layout_changed or state_invalid:
    reset_state(machine, actuator, sim)
else:
    st.session_state.state_layout_signature = layout_sig

st.session_state.machine = machine
st.session_state.actuator = actuator
st.session_state.sim = sim

tab1, tab2, tab3 = st.tabs(["1. Live simulation", "2. AD sensitivity", "3. Scalar AD optimization"])

with tab1:
    st.subheader("Live fixed-actuator simulation")

    should_step = ui["running"] or ui["step_once"]
    advance_status_slot = st.empty()
    if should_step:
        n = int(ui["steps_per_refresh"] if ui["running"] else 1)
        mt, at, stmp = manual_machine, manual_actuator, manual_sim
        with advance_status_slot.container():
            progress = st.progress(0.0)
            progress.progress(0.02)
            with st.spinner("Advancing simulation chunk..." if ui.get("use_jit_chunk", True) else "Advancing simulation..."):
                new_state, timing = advance_live_state_chunk(
                    st.session_state.state,
                    mt,
                    at,
                    stmp,
                    n,
                    use_jit=bool(ui.get("use_jit_chunk", True)),
                )
            progress.progress(1.0)
        advance_status_slot.empty()
        st.session_state.state = new_state
        st.session_state.t += float(timing.get("simulated_time_s", float(stmp.dt) * n))
        st.session_state.step_count += n
        st.session_state.last_advance_timing = timing
        machine, actuator, sim = mt, at, stmp
        if timing.get("fallback_reason"):
            st.warning(timing["fallback_reason"])

    state = st.session_state.state
    live_t0 = time.perf_counter()
    live = compute_live_precomputed(state, machine, actuator, sim)
    diag = live["diag"]
    postprocess_s = time.perf_counter() - live_t0
    append_history(machine, actuator, sim, diag)

    mcols = st.columns(8)
    eq_live = live["eq"]
    phi_rate_live = live["phi_rate"]
    vals = [
        ("t [s]", st.session_state.t),
        ("step", st.session_state.step_count),
        ("P_fus", diag.get("P_fus_MW", np.nan)),
        ("Q", float(diag.get("P_fus_MW", 0.0)) / max(float(actuator.P_aux_MW), 1e-6)),
        ("q_edge", diag.get("q_edge", np.nan)),
        ("beta_N", diag.get("beta_N", np.nan)),
        ("f_G", diag.get("greenwald_fraction", np.nan)),
        ("Phi_dot/Phi", phi_rate_live),
    ]
    for c, (lab, val) in zip(mcols, vals):
        c.metric(lab, f"{float(val):.3g}")

    mode_label = "0.5D fast energy-balance" if str(getattr(sim, "simulation_model", "1.5d")).lower() in ("0d_fast", "0.5d", "0d") else "1.5D radial diffusion"
    st.info(f"Mode: {mode_label}. Fixed actuator from config.py defaults + sidebar overrides.  IPB98(y,2) tau_E ≈ {ipb98y2_tau(machine, actuator, sim):.3f} s")
    if "last_advance_timing" in st.session_state:
        lt = st.session_state.last_advance_timing
        mode_txt = "JIT chunk" if lt.get("used_jit", False) else "eager fallback"
        st.caption(
            f"Last advance: {mode_txt}, {int(lt.get('steps', 0))} steps in "
            f"{float(lt.get('elapsed_s', 0.0)):.3g} s "
            f"({1.0e3 * float(lt.get('per_step_s', 0.0)):.3g} ms/step, including first-call compile if applicable). "
            f"Advanced physical time: {float(lt.get('simulated_time_s', 0.0)):.3g} s "
            f"({float(lt.get('per_step_physical_s', 0.0)):.3g} s/solver step)."
        )

    history_slot = st.empty()
    with history_slot.container():
        render_history()

    st.subheader("Radial profiles")
    surface_svg = get_surface_svg(eq_live, machine, actuator, sim)
    profile_slot = st.empty()
    with profile_slot.container():
        render_s = render_profiles(state, machine, actuator, sim, live=live, surface_svg=surface_svg)
    st.caption(
        f"Live postprocess: shared diagnostics/equilibrium/current/heating {postprocess_s:.3g} s; "
        f"lightweight chart rendering call {render_s:.3g} s. Boundary/surface SVG is cached."
    )

with tab2:
    st.subheader("Local AD sensitivity / normalized Jacobian")
    names = tuple(ui["opt_controls"]) or ("P_aux_MW",)
    bounds = {k: ui["bounds"][k] for k in names}
    ad_sim = make_ad_relax_sim(machine, actuator, base_sim=sim, nr=8, max_steps=12)
    evaluator_label = "1 tau_E time evolution"
    st.caption(f"Sensitivity uses {evaluator_label}; AD demo grid nr={ad_sim.nr}, n_steps={ad_sim.n_steps}, dt={ad_sim.dt:.3g} s.")

    if st.button("Compute AD Jacobian", width='stretch'):
        with st.spinner("Computing AD Jacobian..."):
            y0, norm_jac, x0 = normalized_jacobian(machine, actuator, ad_sim, names, bounds, evaluator="pseudo_time")
        st.session_state.last_jacobian = {"names": names, "y0": y0, "norm_jac": norm_jac, "x0": x0}

    if "last_jacobian" in st.session_state:
        r = st.session_state.last_jacobian
        st.write(f"Baseline: P_fus = {r['y0'][0]:.4g} MW, Q = {r['y0'][1]:.4g}")
        dfp = pd.DataFrame({"control": r["names"], "normalized sensitivity": r["norm_jac"][0]})
        dfq = pd.DataFrame({"control": r["names"], "normalized sensitivity": r["norm_jac"][1]})
        c1, c2 = st.columns(2)
        with c1:
            st.caption(r"Normalized ∂P_fus/∂x × x/P_fus")
            st.bar_chart(dfp, x="control", y="normalized sensitivity", height=320)
        with c2:
            st.caption(r"Normalized ∂Q/∂x × x/Q")
            st.bar_chart(dfq, x="control", y="normalized sensitivity", height=320)
        st.dataframe(pd.DataFrame({"control": r["names"], "x": r["x0"], "dlogPfus_dlogx": r["norm_jac"][0], "dlogQ_dlogx": r["norm_jac"][1]}), width='stretch')
    else:
        st.info("Click Compute AD Jacobian.")

with tab3:
    st.subheader("Scalar AD optimization")
    names = tuple(ui["opt_controls"]) or ("P_aux_MW",)
    bounds = {k: ui["bounds"][k] for k in names}
    constraints = constraints_from_sidebar(ui)
    opt_sim = make_ad_relax_sim(machine, actuator, base_sim=sim, nr=8, max_steps=12)
    evaluator_label = "1 tau_E time evolution"
    st.caption(f"Each objective evaluation uses {evaluator_label}. AD demo grid nr={opt_sim.nr}, n_steps={opt_sim.n_steps}, dt={opt_sim.dt:.3g} s.")

    st.write("Selected objectives:", [k for k, v in ui["opt_objectives"].items() if v])
    st.write("Selected controls:", list(names))
    st.write("Constraints:", constraints if constraints else "None")

    if not any(ui["opt_objectives"].values()):
        st.warning("Select at least one objective in the sidebar.")
    elif st.button("Run scalar AD optimization", width='stretch'):
        with st.spinner("Running scalar AD optimization..."):
            result = optimize_scalar_app(
                machine, actuator, opt_sim, names, bounds,
                ui["opt_objectives"], constraints,
                n_iter=ui["opt_iter"], lr=ui["learning_rate"], evaluator="pseudo_time"
            )
        st.session_state.last_opt_result = {"result": result, "names": names}

    if "last_opt_result" in st.session_state:
        obj = st.session_state.last_opt_result
        result = obj["result"]
        names = obj["names"]
        base = result["baseline"]
        met = result["metrics"]

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Baseline P_fus [MW]", f"{float(base['P_fus_MW']):.4g}")
            st.metric("Baseline Q", f"{float(base['Q']):.4g}")
            st.metric("Baseline q_edge", f"{float(base['q_edge']):.4g}")
            st.metric("Baseline beta_N", f"{float(base['beta_N']):.4g}")
        with c2:
            st.metric("Optimized P_fus [MW]", f"{float(met['P_fus_MW']):.4g}")
            st.metric("Optimized Q", f"{float(met['Q']):.4g}")
            st.metric("Optimized q_edge", f"{float(met['q_edge']):.4g}")
            st.metric("Optimized beta_N", f"{float(met['beta_N']):.4g}")

        st.subheader("Optimized controls")
        st.dataframe(pd.DataFrame({"control": names, "optimized value": _np(result["x_phys"])}), width='stretch')

        hist = pd.DataFrame(result["history"])
        st.subheader("Optimization history")
        st.line_chart(hist, x="iter", y=["objective", "P_fus_MW", "Q", "q_edge", "beta_N"], height=320)
        st.dataframe(hist, width='stretch')

        if st.button("Apply optimized controls to live simulation", width='stretch'):
            x = {name: float(val) for name, val in zip(names, _np(result["x_phys"]))}
            # Store as suggestions; Streamlit widgets cannot be programmatically reassigned after creation,
            # but users can copy these values to sidebar sliders.
            st.session_state.optimized_control_suggestion = x
            st.success("Optimized controls stored below. Copy them to the sidebar sliders.")

    if "optimized_control_suggestion" in st.session_state:
        st.json(st.session_state.optimized_control_suggestion)

if ui["running"]:
    #time.sleep(ui["refresh_delay"])
    st.rerun()
