"""Unified TokaGrad simulation entry point.

The input JSON decides what is run:
  - simulation.simulation_model = "1.5d" or "0d_fast"
  - presence of a top-level "waveform" section selects waveform simulation;
    otherwise a fixed-actuator steady/pseudo-time simulation is run.

Examples:
  PYTHONPATH=src python scripts/run_simulation.py --input-file inputs/iter_flattop_tglfnn.json
  PYTHONPATH=src python scripts/run_simulation.py --input-file inputs/iter_waveform_transition.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import cProfile
import pstats
import io
from dataclasses import replace, asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from tokagrad import MachineConfig, ActuatorConfig, SimulationConfig
from tokagrad.input_config import load_static_input, load_waveform_input, resolve_time_discretization
from tokagrad.grid import make_grid_from_config, boundary_augmented_profile, infer_rho_faces
from tokagrad.solver import initial_state, simulate, simulate_jit, simulate_waveform
from tokagrad.zero_d import (
    zero_d_enabled,
    initial_state_0d,
    simulate_0d,
    simulate_0d_jit,
    simulate_waveform_0d,
    zero_d_diagnostics,
)
from tokagrad.diagnostics import compute_diagnostics
from tokagrad.waveforms import iter_baseline_like_waveform, apply_waveform_controls
from tokagrad.profiles import PlasmaState
from tokagrad.current import current_components_from_state
from tokagrad.heating import total_heating_sources, fuel_ion_densities_20
from tokagrad.equilibrium import solve_fixed_boundary_equilibrium, read_geqdsk
from tokagrad.density import target_edge_ne20
from tokagrad.transport import compute_diffusivity


class StageTimer:
    def __init__(self):
        self.rows: list[tuple[str, float]] = []
        self._last = time.perf_counter()
        self._t0 = self._last

    def mark(self, name: str):
        now = time.perf_counter()
        self.rows.append((name, now - self._last))
        self._last = now

    @property
    def total(self):
        return time.perf_counter() - self._t0

    def print(self):
        print("\nWall-time breakdown:")
        for name, dt in self.rows:
            print(f"  {name:<34s} {dt:9.4f} s")
        print(f"  {'total':<34s} {self.total:9.4f} s")


def _block_until_ready(obj: Any):
    try:
        leaves = jax.tree_util.tree_leaves(obj)
    except Exception:
        leaves = [obj]
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return obj


def _backend_can_show() -> bool:
    try:
        import matplotlib
        backend = matplotlib.get_backend().lower()
        return not any(x in backend for x in ("agg", "pdf", "svg", "ps"))
    except Exception:
        return True


def _np(x):
    return np.asarray(x)


def _close_curve(y):
    a = np.asarray(y)
    if a.size == 0:
        return a
    return np.r_[a, a[0]]


def _state_at_history(state0, final_state, hist, idx: int):
    """Return a PlasmaState for slider frame idx, with idx=0 as the initial state."""
    if idx <= 0 or hist is None:
        return state0
    states = hist["states"] if isinstance(hist, dict) and "states" in hist else hist
    try:
        n_hist = int(np.asarray(states.Te).shape[0])
    except Exception:
        return final_state
    if n_hist <= 0:
        return final_state
    if idx - 1 >= n_hist:
        return final_state
    return PlasmaState(
        states.Te[idx - 1],
        states.Ti[idx - 1],
        states.ne20[idx - 1],
        states.psi_ind[idx - 1],
        states.psi_edge[idx - 1],
        states.Phi_b_prev[idx - 1],
        states.dV_drho_prev[idx - 1],
    )


def _simulation_times(sim, hist):
    if isinstance(hist, dict) and "time" in hist:
        t_hist = np.asarray(hist["time"], dtype=float)
        if t_hist.size:
            # 0.5D histories store post-step physical times accumulated from
            # dt_eff.  1.5D waveform histories store the pre-step control sample
            # time, so their post-step frame is still t_hist + sim.dt.
            if bool(hist.get("time_is_post_step", False)):
                return np.r_[0.0, t_hist]
            return np.r_[0.0, t_hist + float(sim.dt)]
    n_hist = 0
    if hist is not None:
        states = hist["states"] if isinstance(hist, dict) and "states" in hist else hist
        try:
            n_hist = int(np.asarray(states.Te).shape[0])
        except Exception:
            n_hist = 0
    n = max(n_hist + 1, 2)
    return np.arange(n, dtype=float) * float(sim.dt)

def _active_configs(machine, actuator, sim, waveform, t):
    if waveform is None:
        return machine, actuator
    return apply_waveform_controls(machine, actuator, waveform, float(t))


def _aug_profile(rho, y, edge_value=None):
    rr, yy = boundary_augmented_profile(rho, y, edge_value=edge_value)
    return np.asarray(rr), np.asarray(yy)


def _mark_pedestal_top(ax, diag, curves):
    """Mark Te, Ti, and ne at the already-diagnosed pedestal-top radius."""
    if float(diag["ped_lh_gate"]) < 0.5:
        return
    if "ped_rho_top" not in diag:
        return
    try:
        rho_top = float(diag["ped_rho_top"])
    except (TypeError, ValueError):
        return
    if not np.isfinite(rho_top) or not (0.0 < rho_top < 1.0):
        return
    for i, (line, rr, yy) in enumerate(curves):
        y_top = float(np.interp(rho_top, np.asarray(rr), np.asarray(yy)))
        ax.plot(
            [rho_top], [y_top], marker="x", linestyle="none",
            markersize=7, markeredgewidth=1.6, color=line.get_color(),
            label="pedestal top" if i == 0 else None,
        )


def _plot_psi_profile(ax, rho, state):
    """Plot poloidal flux on its native face or cell radial grid."""
    rho_np = np.asarray(rho, dtype=float)
    psi = np.asarray(_np(state.psi_ind), dtype=float)
    psi_edge = float(np.asarray(_np(state.psi_edge)))

    if psi.size == rho_np.size + 1:
        rho_psi = np.asarray(_np(infer_rho_faces(jnp.asarray(rho))), dtype=float)
        psi_plot = psi
    elif psi.size == rho_np.size:
        rho_psi = np.r_[rho_np, 1.0]
        psi_plot = np.r_[psi, psi_edge]
    else:
        raise ValueError(
            f"psi_ind has length {psi.size}; expected nr={rho_np.size} "
            f"or nr+1={rho_np.size + 1}."
        )

    ax.plot(rho_psi, psi_plot, color="tab:purple", label=r"$\psi$")
    ax.axhline(0.0, color="0.5", lw=0.8, alpha=0.7)
    ax.set_title("Poloidal flux")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$\psi$ [Wb]")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)


def _plot_state_summary(axs, rho, diag, state, machine, actuator, sim, t, waveform=None, hist=None):
    for ax in axs.flat:
        ax.cla()
    eq = solve_fixed_boundary_equilibrium(state, machine, actuator, sim)
    j_ind, j_bs, j_cd, j_tot = current_components_from_state(rho, state, machine, actuator, sim, eq=eq)
    _Se, _Si, h = total_heating_sources(rho, state, machine, actuator, sim, eq=eq)
    nH, nD, nT, nHe, nimp = fuel_ion_densities_20(state.ne20, machine)
    _eH20, eD20, eT20, _eHe20, _eimp20 = fuel_ion_densities_20(target_edge_ne20(machine, actuator, sim), machine)

    edge_ne = target_edge_ne20(machine, actuator, sim)
    ax = axs.flat[0]
    rr_te, yy_te = _aug_profile(rho, state.Te, actuator.edge_Te_keV); line_te, = ax.plot(rr_te, yy_te, label=r"$T_e$ [keV]")
    rr_ti, yy_ti = _aug_profile(rho, state.Ti, actuator.edge_Ti_keV); line_ti, = ax.plot(rr_ti, yy_ti, label=r"$T_i$ [keV]")
    rr_ne, yy_ne = _aug_profile(rho, state.ne20 * 10.0, float(edge_ne) * 10.0); line_ne, = ax.plot(rr_ne, yy_ne, label=r"$n_e$ [$10^{19}$ m$^{-3}$]")
    #_mark_pedestal_top(
    #    ax, diag,
    #    [(line_te, rr_te, yy_te), (line_ti, rr_ti, yy_ti), (line_ne, rr_ne, yy_ne)],
    #)
    rr, yy = _aug_profile(rho, nD * 10.0, eD20 * 10.0); ax.plot(rr, yy, "--", lw=1.0, label=r"$n_D$ [$10^{19}$ m$^{-3}$]")
    rr, yy = _aug_profile(rho, nT * 10.0, eT20 * 10.0); ax.plot(rr, yy, ":", lw=1.2, label=r"$n_T$ [$10^{19}$ m$^{-3}$]")
    ax.set_title("Profiles")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel("T [keV] or n [$10^{19}$ m$^{-3}$]")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)

    ax = axs.flat[1]
    for label, y, ls in [
        ("Auxiliary", h["Paux_e"] + h["Paux_i"], "-"),
        ("Alpha", h["Palpha_e"] + h["Palpha_i"], "--"),
        ("Ohmic", h["Pohm_e"], ":"),
        ("Radiation loss", h["Prad_e"], "-."),
        #("Brem", -h.get("Pbrem", 0.0 * h["Prad_e"]), (0, (2, 1))),
        #("Line", -h.get("Pline", 0.0 * h["Prad_e"]), (0, (4, 1))),
        #("Sync", -h.get("Psync", 0.0 * h["Prad_e"]), (0, (1, 1))),
        ("Net e", h["Pe_net"], (0, (3, 1, 1, 1))),
        ("Net i", h["Pi_net"], (0, (1, 1))),
    ]:
        rr, yy = _aug_profile(rho, np.asarray(y), 0.0)
        ax.plot(rr, yy, ls=ls, label=label)
    ax.set_title("Heating/loss")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"P [MW m$^{-3}$]")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)

    ax = axs.flat[2]
    for label, y, ls in [
        ("total j", j_tot, "-"),
        ("inductive", j_ind, "--"),
        ("bootstrap", j_bs, ":"),
        ("CD", j_cd, "-."),
    ]:
        rr, yy = _aug_profile(rho, np.asarray(y), 0.0)
        ax.plot(rr, yy / 1e6, ls=ls, label=label)
    ax.set_title("Current density and q")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"j [MA m$^{-2}$] or q")
    ax.grid(alpha=0.25, which="both")

    rr, yy = _aug_profile(rho, np.asarray(eq.q), float(diag.get('q_edge', 0.0)))
    ax.plot(rr, yy, color='k', label="q")
    ax.set_ylim([None, max(np.max(j_tot) / 1e6, float(diag.get('q95', 0.0))) + 0.5])
    ax.legend(fontsize=8)

    show_diff = bool(getattr(sim, "plot_diffusivity", False))
    show_psi = bool(getattr(sim, "plot_psi", False))
    next_idx = 3
    if show_diff:
        ax = axs.flat[next_idx]
        next_idx += 1
        prof = _final_profiles_at_state(rho, state, machine, actuator, sim, include_diffusivity=True)
        has_negative = False
        for name, y, ls in [(r"$\chi_e$", prof["chi_e"], "-"), (r"$\chi_i$", prof["chi_i"], "--"), (r"$D_n$", prof["Dn"], ":")]:
            rrd, yy = _axis_aug_np(rho, y)
            yy = _np(yy)
            has_negative = has_negative or bool(np.any(yy < 0.0))
            ax.plot(rrd, yy, linestyle=ls, label=name)
        '''
        if has_negative:
            ax.set_yscale("symlog", linthresh=1.0e-3)
            ax.axhline(0.0, color="0.5", lw=0.8)
        else:
            ax.set_yscale("log")
        '''
        ax.set_title("Diffusivity")
        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(r"D [m$^2$ s$^{-1}$]")
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8)

    if show_psi:
        _plot_psi_profile(axs.flat[next_idx], rho, state)
        next_idx += 1

    ax = axs.flat[next_idx]
    next_idx += 1
    '''stride = max(1, len(rho) // 10)
    for i in range(0, len(rho), stride):
        ax.plot(_close_curve(eq.R[i]), _close_curve(eq.Z[i]), lw=0.8)
    ax.set_title("Magnetic surfaces")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_aspect("equal", adjustable="box")'''
    _plot_surfaces_with_geqdsk(ax, eq, sim)
    ax.set_title("Boundary/surfaces")
    ax.grid(alpha=0.25, which="both")

    ax = axs.flat[next_idx]
    ax.axis("off")
    lines = [
        f"t = {float(t):.4g} s",
        f"model = {getattr(sim, 'simulation_model', '1.5d')}",
        f"Ip = {float(machine.Ip/1e6):.3g} MA",
        f"Bt = {float(machine.Bt):.3g} T",
        f"Paux = {float(actuator.P_aux_MW):.3g} MW",
        f"f_G,target = {float(actuator.greenwald_fraction_target):.3g}",
        "",
        f"Q = {float(diag.get('Q', 0.0)):.4g}",
        f"P_fus = {float(diag.get('P_fus_MW', 0.0)):.4g} MW",
        f"beta_N = {float(diag.get('beta_N', 0.0)):.4g}",
        f"q95 = {float(diag.get('q95', 0.0)):.4g}",
        f"q_edge = {float(diag.get('q_edge', 0.0)):.4g}",
        f"Pohm = {float(diag.get('P_ohmic_MW')):.4g} MW",
        f"Palp = {float(diag.get('P_alpha_MW')):.4g} MW",
        f"Pabs = {float(diag.get('P_abs_for_lh_MW')):.4g} MW",
        f"Prad = {float(diag.get('P_rad_MW')):.4g} MW",
        f"Psep = {(float(diag.get('P_abs_for_lh_MW')) - float(diag.get('P_rad_MW'))):.4g} MW",
        f"P_LH = {float(diag.get('P_LH_Martin_MW')):.4g} MW",
        f"LH gate = {float(diag.get('ped_lh_gate', 1.0)):.3g}",
        f"I_total = {float(diag.get('I_total_MA')):.4g} MA",
        f"I_ind = {float(diag.get('I_ind_MA')):.4g} MA",
        f"I_bs = {float(diag.get('I_bs_MA')):.4g} MA",
        f"I_cd = {float(diag.get('I_cd_MA')):.4g} MA",
    ]
    ax.text(0.02, 0.98, "\n".join([x for x in lines if x != ""]), va="top", ha="left", family="monospace", fontsize=9)


def plot_simulation_time_slider(machine, actuator, sim, waveform, rho, state0, final_state, hist, *, show=True, save_path=""):
    """Unified plotting helper with the time slider initially at the final time."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button

    times = _simulation_times(sim, hist)
    nframes = len(times)
    show_diff = bool(getattr(sim, "plot_diffusivity", False))
    show_psi = bool(getattr(sim, "plot_psi", False))
    ncols = 5 + int(show_diff) + int(show_psi)
    fig, axs = plt.subplots(1, ncols, figsize=(21 + 4 * (ncols - 5), 5))
    fig.subplots_adjust(bottom=0.2, wspace=0.32, hspace=0.36, left=0.035, right=0.985)
    # Leave room for one-step navigation buttons next to the time slider.
    slider_ax = fig.add_axes([0.12, 0.05, 0.66, 0.035])
    back_ax = fig.add_axes([0.855, 0.05, 0.045, 0.035])
    fwd_ax = fig.add_axes([0.908, 0.05, 0.045, 0.035])
    slider = Slider(
        slider_ax,
        "time [s]",
        float(times[0]),
        float(times[-1]) if nframes > 1 else max(float(times[0]), 1.0e-12),
        valinit=float(times[-1]),
        valstep=times if nframes > 1 else None,
        valfmt="%.4g s",
    )

    current_index = {"idx": nframes - 1 if nframes else 0}

    def _time_to_index(t_value):
        if nframes <= 1:
            return 0
        return int(np.argmin(np.abs(times - float(t_value))))

    def _set_frame_index(idx):
        if nframes <= 0:
            return
        idx = int(np.clip(idx, 0, nframes - 1))
        slider.set_val(float(times[idx]))

    def update(val):
        t = float(slider.val)
        idx = _time_to_index(t)
        current_index["idx"] = idx
        st = _state_at_history(state0, final_state, hist, idx)
        mt, at = _active_configs(machine, actuator, sim, waveform, times[idx] if nframes else t)
        diag = zero_d_diagnostics(st, mt, at, sim) if zero_d_enabled(sim) else compute_diagnostics(st, mt, at, sim)
        _plot_state_summary(axs, rho, diag, st, mt, at, sim, times[idx] if nframes else t, waveform=waveform, hist=hist)
        fig.suptitle(
            "TokaGrad simulation: fixed actuator  "
            f"t={t:.4g} s  ({idx}/{nframes-1})  |  "
            f"Ip={float(mt.Ip) / 1e6:.3g} MA, Bt={float(mt.Bt):.3g} T, P_aux={float(at.P_aux_MW):.3g} MW  |  "
            f"Q={float(diag.get('Q', 0.0)):.3g}, "
            f"P_fus={float(diag.get('P_fus_MW', 0.0)):.3g} MW, "
            f"beta_N={float(diag.get('beta_N', 0.0)):.3g}, "
            f"f_G={float(diag.get('greenwald_fraction', 0.0)):.3g}",
            fontsize=12,
        )

        fig.canvas.draw_idle()

    back_button = Button(back_ax, "-1")
    fwd_button = Button(fwd_ax, "+1")
    back_button.on_clicked(lambda _event: _set_frame_index(current_index["idx"] - 1))
    fwd_button.on_clicked(lambda _event: _set_frame_index(current_index["idx"] + 1))

    slider.on_changed(update)
    update(float(times[-1]) if nframes else 0.0)
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_final_state_static(machine, actuator, sim, waveform, rho, state0, final_state, hist, *, show=True, save_path=""):
    """Plot only the final state without Matplotlib widgets.

    This intentionally avoids importing ``matplotlib.widgets``.  It is useful
    on remote/X11 servers where the simulation runs correctly but GUI input
    extensions such as XInput 2 are unavailable and interactive sliders fail.
    """
    import matplotlib.pyplot as plt

    times = _simulation_times(sim, hist)
    t_final = float(times[-1]) if len(times) else float(sim.dt) * int(sim.n_steps)
    mt, at = _active_configs(machine, actuator, sim, waveform, t_final)
    diag = zero_d_diagnostics(final_state, mt, at, sim) if zero_d_enabled(sim) else compute_diagnostics(final_state, mt, at, sim)

    show_diff = bool(getattr(sim, "plot_diffusivity", False))
    show_psi = bool(getattr(sim, "plot_psi", False))
    ncols = 5 + int(show_diff) + int(show_psi)
    fig, axs = plt.subplots(1, ncols, figsize=(21 + 4 * (ncols - 5), 5))
    fig.subplots_adjust(bottom=0.12, wspace=0.32, hspace=0.36, left=0.035, right=0.985)

    _plot_state_summary(axs, rho, diag, final_state, mt, at, sim, t_final, waveform=waveform, hist=hist)
    fig.suptitle(
        "TokaGrad final-state summary  "
        f"t={t_final:.4g} s  |  "
        f"Ip={float(mt.Ip) / 1e6:.3g} MA, Bt={float(mt.Bt):.3g} T, P_aux={float(at.P_aux_MW):.3g} MW  |  "
        f"Q={float(diag.get('Q', 0.0)):.3g}, "
        f"P_fus={float(diag.get('P_fus_MW', 0.0)):.3g} MW, "
        f"beta_N={float(diag.get('beta_N', 0.0)):.3g}, "
        f"f_G={float(diag.get('greenwald_fraction', 0.0)):.3g}",
        fontsize=12,
    )

    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def _axis_aug_np(rho, y, edge_value=None):
    rr, yy = boundary_augmented_profile(rho, y, edge_value=edge_value)
    return _np(rr), _np(yy)


def _stack_waveform_state_history(state0, wf_hist, sim):
    """Return state-time arrays for interactive waveform replay."""
    states = wf_hist["states"]
    n_steps = int(_np(states.Te).shape[0])
    if isinstance(wf_hist, dict) and "time" in wf_hist:
        t_hist = np.asarray(wf_hist["time"], dtype=float)
        if t_hist.size == n_steps:
            if bool(wf_hist.get("time_is_post_step", False)):
                times = np.r_[0.0, t_hist]
            else:
                times = np.r_[0.0, t_hist + float(sim.dt)]
        else:
            times = np.arange(n_steps + 1, dtype=float) * float(sim.dt)
    else:
        # State0 is the t=0 condition.  The state saved at scan step i is after
        # one dt update using controls sampled at t=i*dt, so place it at
        # (i+1)*dt for ordinary 1.5D waveform histories.
        times = np.arange(n_steps + 1, dtype=float) * float(sim.dt)
    arr = {
        "Te": np.concatenate([_np(state0.Te)[None, :], _np(states.Te)], axis=0),
        "Ti": np.concatenate([_np(state0.Ti)[None, :], _np(states.Ti)], axis=0),
        "ne20": np.concatenate([_np(state0.ne20)[None, :], _np(states.ne20)], axis=0),
        "psi_ind": np.concatenate([_np(state0.psi_ind)[None, :], _np(states.psi_ind)], axis=0),
        "psi_edge": np.concatenate([np.asarray(_np(state0.psi_edge)).reshape(1), np.asarray(_np(states.psi_edge)).reshape(-1)]),
        "Phi_b_prev": np.concatenate([np.asarray(_np(state0.Phi_b_prev)).reshape(1), np.asarray(_np(states.Phi_b_prev)).reshape(-1)]),
        "dV_drho_prev": np.concatenate(
            [_np(state0.dV_drho_prev)[None, :], _np(states.dV_drho_prev)],
            axis=0,
        ),
    }
    return times, arr
    

def _state_from_arrays(arr, idx):
    return PlasmaState(
        arr["Te"][idx],
        arr["Ti"][idx],
        arr["ne20"][idx],
        arr["psi_ind"][idx],
        np.asarray(arr["psi_edge"])[idx],
        np.asarray(arr["Phi_b_prev"])[idx],
        np.asarray(arr["dV_drho_prev"])[idx]
        if "dV_drho_prev" in arr else jnp.asarray(0.0),
    )


def _final_profiles_at_state(rho, state, machine, actuator, sim, *, include_diffusivity=None):
    if include_diffusivity is None:
        include_diffusivity = bool(getattr(sim, "plot_diffusivity", False))
    eq = solve_fixed_boundary_equilibrium(state, machine, actuator, sim)
    j_ind, j_bs, j_cd, j_total = current_components_from_state(rho, state, machine, actuator, sim, eq=eq)
    _, _, heat = total_heating_sources(rho, state, machine, actuator, sim, eq=eq)
    out = {
        "eq": eq,
        "j_ind": j_ind,
        "j_bs": j_bs,
        "j_cd": j_cd,
        "j_total": j_total,
        "heat": heat,
    }
    if include_diffusivity:
        chi_e, chi_i, Dn = compute_diffusivity(rho, state.Te, state.Ti, state.ne20, machine, sim, q=eq.q, eq=eq)
        out.update({"chi_e": chi_e, "chi_i": chi_i, "Dn": Dn})
    return out



def _save_time_slice_results(rho, state0, final_state, hist, machine, actuator, sim, *, waveform=None, input_file=None):
    """Save requested time slices to a compressed NumPy archive.

    The archive stores one row per requested saved time.  Requested times are
    mapped to the nearest available history frame.  Profiles are stored as
    arrays of shape (n_saved, nr); diagnostics are stored as scalar time series.
    A small JSON sidecar is written next to the NPZ for discoverability.
    """
    if not bool(getattr(sim, "save_results_enabled", False)):
        return None
    fmt = str(getattr(sim, "save_result_format", "npz")).lower()
    if fmt not in ("npz", "numpy", "numpy_npz"):
        raise ValueError(f"Unsupported save_result_format={fmt!r}; currently only 'npz' is supported.")

    times = np.asarray(_simulation_times(sim, hist), dtype=float)
    if times.size == 0:
        times = np.asarray([0.0, float(sim.dt) * int(sim.n_steps)], dtype=float)

    requested = np.asarray(tuple(getattr(sim, "save_result_times_s", ()) or ()), dtype=float)
    if requested.size == 0:
        requested = np.asarray([float(times[-1])], dtype=float)
    requested = np.clip(requested, float(times[0]), float(times[-1]))
    indices = np.asarray([int(np.argmin(np.abs(times - t))) for t in requested], dtype=int)
    # Keep user order, but if two requested times map to the same frame, save it once.
    unique_indices = []
    unique_requested = []
    seen = set()
    for req, idx in zip(requested, indices):
        key = int(idx)
        if key not in seen:
            unique_indices.append(key)
            unique_requested.append(float(req))
            seen.add(key)
    indices = np.asarray(unique_indices, dtype=int)
    requested = np.asarray(unique_requested, dtype=float)
    saved_times = times[indices]

    data: dict[str, np.ndarray] = {
        "rho": np.asarray(rho, dtype=float),
        "time_requested_s": requested,
        "time_saved_s": saved_times,
        "frame_index": indices,
    }
    diag_series: dict[str, list[np.ndarray]] = {}
    heat_series: dict[str, list[np.ndarray]] = {}
    profile_series: dict[str, list[np.ndarray]] = {
        "Te_keV": [],
        "Ti_keV": [],
        "ne20": [],
        "q": [],
        "j_ind_A_m2": [],
        "j_bs_A_m2": [],
        "j_cd_A_m2": [],
        "j_total_A_m2": [],
        "chi_e_m2_s": [],
        "chi_i_m2_s": [],
        "Dn_m2_s": [],
    }
    control_series = {"Ip_MA": [], "Bt_T": [], "P_aux_MW": [], "greenwald_fraction_target": []}

    for idx in indices:
        t = float(times[idx])
        st = _state_at_history(state0, final_state, hist, int(idx))
        mt, at = _active_configs(machine, actuator, sim, waveform, t) if waveform is not None else (machine, actuator)
        # Saving explicitly requests diffusivities even though the default plot
        # path skips this expensive debug-only computation.
        prof = _final_profiles_at_state(rho, st, mt, at, sim, include_diffusivity=True)
        diag = zero_d_diagnostics(st, mt, at, sim) if zero_d_enabled(sim) else compute_diagnostics(st, mt, at, sim)

        profile_series["Te_keV"].append(np.asarray(_np(st.Te), dtype=float))
        profile_series["Ti_keV"].append(np.asarray(_np(st.Ti), dtype=float))
        profile_series["ne20"].append(np.asarray(_np(st.ne20), dtype=float))
        profile_series["q"].append(np.asarray(_np(prof["eq"].q), dtype=float))
        profile_series["j_ind_A_m2"].append(np.asarray(_np(prof["j_ind"]), dtype=float))
        profile_series["j_bs_A_m2"].append(np.asarray(_np(prof["j_bs"]), dtype=float))
        profile_series["j_cd_A_m2"].append(np.asarray(_np(prof["j_cd"]), dtype=float))
        profile_series["j_total_A_m2"].append(np.asarray(_np(prof["j_total"]), dtype=float))
        profile_series["chi_e_m2_s"].append(np.asarray(_np(prof["chi_e"]), dtype=float))
        profile_series["chi_i_m2_s"].append(np.asarray(_np(prof["chi_i"]), dtype=float))
        profile_series["Dn_m2_s"].append(np.asarray(_np(prof["Dn"]), dtype=float))

        for hk, hv in prof["heat"].items():
            heat_series.setdefault(str(hk), []).append(np.asarray(_np(hv), dtype=float))
        for dk, dv in diag.items():
            try:
                arr = np.asarray(_np(dv), dtype=float)
            except Exception:
                continue
            if arr.ndim == 0 or arr.size == 1:
                diag_series.setdefault(str(dk), []).append(arr.reshape(()))
            elif arr.shape == np.asarray(rho).shape:
                # Keep any radial diagnostic profile if one is added later.
                diag_series.setdefault(str(dk), []).append(arr)
        control_series["Ip_MA"].append(float(mt.Ip) / 1.0e6)
        control_series["Bt_T"].append(float(mt.Bt))
        control_series["P_aux_MW"].append(float(at.P_aux_MW))
        control_series["greenwald_fraction_target"].append(float(at.greenwald_fraction_target))

    for key, rows in profile_series.items():
        data[key] = np.stack(rows, axis=0)
    for key, rows in heat_series.items():
        data[f"heat_{key}_MW_m3"] = np.stack(rows, axis=0)
    for key, rows in diag_series.items():
        try:
            data[f"diag_{key}"] = np.stack(rows, axis=0)
        except Exception:
            pass
    for key, rows in control_series.items():
        data[f"control_{key}"] = np.asarray(rows, dtype=float)

    out = Path(str(getattr(sim, "save_result_file", "outputs/tokagrad_results.npz"))).expanduser()
    if not out.is_absolute():
        out = (Path.cwd() / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)

    meta = {
        "format": "tokagrad_time_slices_npz_v1",
        "file": str(out),
        "input_file": str(input_file) if input_file is not None else "",
        "requested_times_s": requested.tolist(),
        "saved_times_s": saved_times.tolist(),
        "frame_indices": indices.tolist(),
        "n_saved": int(len(indices)),
        "dt_s": float(sim.dt),
        "n_steps": int(sim.n_steps),
        "end_time_s": float(times[-1]),
        "profile_keys": sorted([k for k in data.keys() if k not in {"rho", "time_requested_s", "time_saved_s", "frame_index"} and not k.startswith("diag_")]),
        "diagnostic_keys": sorted([k.removeprefix("diag_") for k in data.keys() if k.startswith("diag_")]),
        "units": {
            "rho": "normalized toroidal-flux radius",
            "time_saved_s": "s",
            "Te_keV": "keV",
            "Ti_keV": "keV",
            "ne20": "1e20 m^-3",
            "q": "dimensionless",
            "j_*_A_m2": "A m^-2",
            "heat_*_MW_m3": "MW m^-3",
            "chi_*_m2_s": "m^2 s^-1",
            "Dn_m2_s": "m^2 s^-1",
        },
        "machine": asdict(machine),
        "actuator_initial": asdict(actuator),
        "simulation": asdict(sim),
    }
    if bool(getattr(sim, "save_result_include_metadata_json", True)):
        meta_path = out.with_suffix(out.suffix + ".json")
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Saved time-slice results to {out}")
        print(f"Saved result metadata to {meta_path}")
    else:
        print(f"Saved time-slice results to {out}")
    return out


def _plot_surfaces_with_geqdsk(ax, eq, sim):
    R = _np(eq.R)
    Z = _np(eq.Z)
    stride = max(1, int(sim.nr) // 10)
    for i in range(0, int(sim.nr), stride):
        ax.plot(_close_curve(R[i]), _close_curve(Z[i]), lw=0.75, alpha=0.55)
    ax.plot(_close_curve(R[-1]), _close_curve(Z[-1]), lw=0.75, alpha=0.55)
    #ax.plot(_close_curve(R[-1]), _close_curve(Z[-1]), "k-", lw=1.8, label="computed LCFS")
    ax.plot(_close_curve(_np(eq.rbbbs)), _close_curve(_np(eq.zbbbs)), "k-", lw=1.8, label=r"LCFS ($\rho=1$)")
    ax.plot([R[0].mean()], [Z[0].mean()], "kx", ms=5, label=r"axis ($\rho=0$)")
    '''if getattr(sim, "equilibrium_model", "") == "geqdsk_prescribed" and getattr(sim, "geqdsk_path", ""):
        try:
            g = read_geqdsk(getattr(sim, "geqdsk_path"))
            rbd = np.asarray(g.get("rbbbs", []), dtype=float)
            zbd = np.asarray(g.get("zbbbs", []), dtype=float)
            if rbd.size >= 4 and rbd.size == zbd.size:
                src = str(g.get("boundary_source", "G-EQDSK"))
                ax.plot(_close_curve(rbd), _close_curve(zbd), "r--", lw=1.3, label=f"G-EQDSK ({src})")
            if "rmaxis" in g and "zmaxis" in g:
                ax.plot([float(g["rmaxis"])], [float(g["zmaxis"])], "kx", ms=5, label="axis")
        except Exception as exc:
            ax.text(0.02, 0.98, f"G-EQDSK boundary unavailable: {exc}", transform=ax.transAxes,
                    va="top", ha="left", fontsize=7)'''
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=7, loc="best")


def _waveform_series(machine, actuator, sim, waveform, t_grid):
    Ip, Paux, Bt, fG = [], [], [], []
    for tt in t_grid:
        mt, at = apply_waveform_controls(machine, actuator, waveform, float(tt))
        Ip.append(float(mt.Ip) / 1e6)
        Paux.append(float(at.P_aux_MW))
        Bt.append(float(mt.Bt))
        fG.append(float(at.greenwald_fraction_target))
    return np.asarray(Ip), np.asarray(Paux), np.asarray(Bt), np.asarray(fG)


def plot_waveform_single_figure_time_slider(rho, state0, final_state, machine, actuator, sim, waveform, wf_hist,
                                            *, show=True, save_path=""):
    """One interactive figure: waveform row plus scalar-demo-style profile panels."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button
    rho = _np(rho)
    times, arr = _stack_waveform_state_history(state0, wf_hist, sim)
    nframes = len(times)

    m0, at0 = apply_waveform_controls(machine, actuator, waveform, 0.0)
    rho_Te0, Te0 = _axis_aug_np(rho, state0.Te, at0.edge_Te_keV)
    _, Ti0 = _axis_aug_np(rho, state0.Ti, at0.edge_Ti_keV)
    _, ne0 = _axis_aug_np(rho, state0.ne20 * 10.0, target_edge_ne20(m0, at0, sim) * 10.0)

    show_diff = bool(getattr(sim, "plot_diffusivity", False))
    show_psi = bool(getattr(sim, "plot_psi", False))
    ncols = 5 + int(show_diff) + int(show_psi)
    fig = plt.figure(figsize=(21.0 + 4.0 * (ncols - 5), 8.0))
    gs = fig.add_gridspec(
        2, ncols,
        height_ratios=[1.0, 3.0],
        left=0.035,
        right=0.965,
        top=0.86,
        bottom=0.16,
        hspace=0.45,
        wspace=0.35,
    )
    ax_wf = fig.add_subplot(gs[0, :])
    ax_wf2 = ax_wf.twinx()
    ax_prof = fig.add_subplot(gs[1, 0])
    ax_heat = fig.add_subplot(gs[1, 1])
    ax_cur = fig.add_subplot(gs[1, 2])
    next_col = 3
    if show_diff:
        ax_diff = fig.add_subplot(gs[1, next_col])
        next_col += 1
    else:
        ax_diff = None
    if show_psi:
        ax_psi = fig.add_subplot(gs[1, next_col])
        next_col += 1
    else:
        ax_psi = None
    ax_surf = fig.add_subplot(gs[1, next_col])
    ax_text = fig.add_subplot(gs[1, next_col + 1])

    tmax = float(times[-1]) if nframes > 1 else 0.0
    slider_ax = fig.add_axes([0.12, 0.055, 0.66, 0.035])
    back_ax = fig.add_axes([0.855, 0.055, 0.045, 0.035])
    fwd_ax = fig.add_axes([0.908, 0.055, 0.045, 0.035])
    slider = Slider(
        slider_ax,
        "time [s]",
        0.0,
        max(tmax, 1.0e-12),
        valinit=max(tmax, 1.0e-12),
        valstep=times if nframes > 1 else None,
        valfmt="%.4g s",
    )

    dense_t = np.linspace(0.0, max(tmax, 1.0e-12), 400)
    Ip_dense, Paux_dense, Bt_dense, fG_dense = _waveform_series(machine, actuator, sim, waveform, dense_t)

    current_index = {"idx": nframes - 1 if nframes else 0}

    def _time_to_index(t_value):
        if nframes <= 1:
            return 0
        return int(np.argmin(np.abs(times - float(t_value))))

    def _set_frame_index(idx):
        if nframes <= 0:
            return
        idx = int(np.clip(idx, 0, nframes - 1))
        slider.set_val(float(times[idx]))

    def draw(time_value):
        idx = _time_to_index(time_value)
        current_index["idx"] = idx
        tnow = float(times[idx])
        mt, at = apply_waveform_controls(machine, actuator, waveform, tnow)
        st = _state_from_arrays(arr, idx)
        prof = _final_profiles_at_state(rho, st, mt, at, sim, include_diffusivity=show_diff)
        diag = compute_diagnostics(st, mt, at, sim)

        axes_to_clear = [ax_wf, ax_wf2, ax_prof, ax_heat, ax_cur, ax_surf, ax_text]
        if ax_diff is not None:
            axes_to_clear.insert(5, ax_diff)
        if ax_psi is not None:
            axes_to_clear.insert(-2, ax_psi)
        for ax in axes_to_clear:
            ax.cla()

        # Top waveform panel.
        ax_wf.plot(dense_t, Ip_dense, 'k', label="Ip [MA]")
        ax_wf.set_ylabel("Ip [MA]")
        ax_wf2.plot(dense_t, Paux_dense, "--", color='grey', label="P_aux [MW]")
        ax_wf2.set_ylabel("P_aux [MW]")
        ax_wf2.yaxis.set_label_position("right")
        ax_wf.axvline(tnow, color="k", ls="--", lw=1.0, alpha=0.75)
        ax_wf.set_xlim(0.0, max(tmax, 1.0e-12))
        ax_wf.set_ylim(0.0, None)
        ax_wf2.set_ylim(0.0, None)
        ax_wf.set_title("Actuator waveform")
        ax_wf.set_xlabel("time [s]")
        ax_wf.grid(alpha=0.25)
        lines, labels = ax_wf.get_legend_handles_labels()
        lines2, labels2 = ax_wf2.get_legend_handles_labels()
        ax_wf.legend(lines + lines2, labels + labels2, fontsize=8, ncol=4, loc="upper left")

        # Panel 1: temperature and density.
        rr, Te = _axis_aug_np(rho, st.Te, at.edge_Te_keV)
        _, Ti = _axis_aug_np(rho, st.Ti, at.edge_Ti_keV)
        _, ne19 = _axis_aug_np(rho, st.ne20 * 10.0, target_edge_ne20(mt, at, sim) * 10.0)
        line_te, = ax_prof.plot(rr, Te, label="Te [keV]")
        line_ti, = ax_prof.plot(rr, Ti, label="Ti [keV]")
        line_ne, = ax_prof.plot(rr, ne19, label=r"ne [$10^{19}$ m$^{-3}$]")
        #_mark_pedestal_top(
        #    ax_prof, diag,
        #    [(line_te, rr, Te), (line_ti, rr, Ti), (line_ne, rr, ne19)],
        #)
        _nH20, nD20, nT20, _nHe20, _nimp20 = fuel_ion_densities_20(st.ne20, mt)
        _eH20, eD20, eT20, _eHe20, _eimp20 = fuel_ion_densities_20(target_edge_ne20(mt, at, sim), mt)
        if float(np.max(_np(nD20))) > 0.0:
            _, nD19 = _axis_aug_np(rho, nD20 * 10.0, eD20 * 10.0)
            ax_prof.plot(rr, nD19, "--", lw=1.1, label=r"nD [$10^{19}$ m$^{-3}$]")
        if float(np.max(_np(nT20))) > 0.0:
            _, nT19 = _axis_aug_np(rho, nT20 * 10.0, eT20 * 10.0)
            ax_prof.plot(rr, nT19, ":", lw=1.3, label=r"nT [$10^{19}$ m$^{-3}$]")
        ax_prof.plot(rho_Te0, Te0, "k--", lw=1.0, label="initial Te")
        ax_prof.plot(rho_Te0, Ti0, "k-.", lw=1.0, label="initial Ti")
        ax_prof.plot(rho_Te0, ne0, "k:", lw=1.2, label="initial ne")
        ax_prof.set_title("Profiles")
        ax_prof.set_xlabel(r"$\rho$")
        ax_prof.set_ylabel("T [keV] or n [$10^{19}$ m$^{-3}$]")
        ax_prof.grid(alpha=0.25)
        ax_prof.legend(fontsize=7)

        h = prof["heat"]
        for name, y, ls in [
            ("Auxiliary", h["Paux_e"] + h["Paux_i"], "-"),
            ("Alpha", h["Palpha_e"] + h["Palpha_i"], "--"),
            ("Ohmic", h["Pohm_e"], ":"),
            ("Radiation loss", h["Prad_e"], "-."),
            #("Brem", -h.get("Pbrem", 0.0 * h["Prad_e"]), (0, (2, 1))),
            #("Line", -h.get("Pline", 0.0 * h["Prad_e"]), (0, (4, 1))),
            #("Sync", -h.get("Psync", 0.0 * h["Prad_e"]), (0, (1, 1))),
            ("Net e", h["Pe_net"], (0, (3, 1, 1, 1))),
            ("Net i", h["Pi_net"], (0, (1, 1))),
        ]:
            rrh, yy = _axis_aug_np(rho, y)
            ax_heat.plot(rrh, yy, linestyle=ls, label=name)
        ax_heat.axhline(0.0, color="0.5", lw=0.8)
        ax_heat.set_title("Heating/loss")
        ax_heat.set_xlabel(r"$\rho$")
        ax_heat.set_ylabel(r"P [MW m$^{-3}$]")
        ax_heat.grid(alpha=0.25)
        ax_heat.legend(fontsize=7)

        for name, y, ls in [
            ("total j", prof["j_total"] / 1.0e6, "-"),
            ("inductive", prof["j_ind"] / 1.0e6, "--"),
            ("bootstrap", prof["j_bs"] / 1.0e6, ":"),
            ("CD", prof["j_cd"] / 1.0e6, "-."),
        ]:
            rrj, yy = _axis_aug_np(rho, y, 0.0)
            ax_cur.plot(rrj, yy, linestyle=ls, label=name)
        rqq, qv = _axis_aug_np(rho, prof["eq"].q, float(diag.get('q_edge', 0.0)))
        ax_cur.plot(rqq, qv, color="0.25", lw=1.6, label="q")
        ax_cur.set_title("Current density and q")
        ax_cur.set_xlabel(r"$\rho$")
        ax_cur.set_ylabel(r"j [MA m$^{-2}$] or q")
        ax_cur.set_ylim([None, max(np.max(prof["j_total"]) / 1.0e6, float(diag.get('q95', 0.0))) + 0.5])
        ax_cur.grid(alpha=0.25)
        lines, labels = ax_cur.get_legend_handles_labels()
        #lines2, labels2 = ax_q.get_legend_handles_labels()
        #ax_cur.legend(lines + lines2, labels + labels2, fontsize=7)
        ax_cur.legend(lines, labels, fontsize=7)

        if ax_diff is not None:
            has_negative = False
            for name, y, ls in [(r"$\chi_e$", prof["chi_e"], "-"), (r"$\chi_i$", prof["chi_i"], "--"), (r"$D_n$", prof["Dn"], ":")]:
                rrd, yy = _axis_aug_np(rho, y)
                yy = _np(yy)
                has_negative = has_negative or bool(np.any(yy < 0.0))
                ax_diff.plot(rrd, yy, linestyle=ls, label=name)
            '''
            if has_negative:
                ax_diff.set_yscale("symlog", linthresh=1.0e-3)
                ax_diff.axhline(0.0, color="0.5", lw=0.8)
            else:
                ax_diff.set_yscale("log")
            '''
            ax_diff.set_title("Diffusivity")
            ax_diff.set_xlabel(r"$\rho$")
            ax_diff.set_ylabel(r"D [m$^2$ s$^{-1}$]")
            ax_diff.grid(alpha=0.25, which="both")
            ax_diff.legend(fontsize=8)
        if ax_psi is not None:
            _plot_psi_profile(ax_psi, rho, st)
        _plot_surfaces_with_geqdsk(ax_surf, prof["eq"], sim)
        ax_surf.set_title("Boundary/surfaces")

        ax_text.axis("off")
        lines = [
            f"t = {float(tnow):.4g} s",
            f"model = {getattr(sim, 'simulation_model', '1.5d')}",
            f"Ip = {float(mt.Ip/1e6):.3g} MA",
            f"Bt = {float(mt.Bt):.3g} T",
            f"Paux = {float(at.P_aux_MW):.3g} MW",
            f"f_G,target = {float(at.greenwald_fraction_target):.3g}",
            "",
            f"Q = {float(diag.get('Q', 0.0)):.4g}",
            f"P_fus = {float(diag.get('P_fus_MW', 0.0)):.4g} MW",
            f"beta_N = {float(diag.get('beta_N', 0.0)):.4g}",
            f"q95 = {float(diag.get('q95', 0.0)):.4g}",
            f"q_edge = {float(diag.get('q_edge', 0.0)):.4g}",
            f"Pohm = {float(diag.get('P_ohmic_MW')):.4g} MW",
            f"Palp = {float(diag.get('P_alpha_MW')):.4g} MW",
            f"Pabs = {float(diag.get('P_abs_for_lh_MW')):.4g} MW",
            f"Prad = {float(diag.get('P_rad_MW')):.4g} MW",
            f"Psep = {(float(diag.get('P_abs_for_lh_MW')) - float(diag.get('P_rad_MW'))):.4g} MW",
            f"P_LH = {float(diag.get('P_LH_Martin_MW')):.4g} MW",
            f"LH gate = {float(diag.get('ped_lh_gate', 1.0)):.3g}",
            f"I_total = {float(diag.get('I_total_MA')):.4g} MA",
            f"I_ind = {float(diag.get('I_ind_MA')):.4g} MA",
            f"I_bs = {float(diag.get('I_bs_MA')):.4g} MA",
            f"I_cd = {float(diag.get('I_cd_MA')):.4g} MA",
        ]
        ax_text.text(0.02, 0.98, "\n".join([x for x in lines if x != ""]), va="top", ha="left", family="monospace", fontsize=9)

        ip_ma = float(mt.Ip) / 1e6
        fig.suptitle(
            "TokaGrad simulation: waveform actuator  "
            f"t={tnow:.4g} s  ({idx}/{nframes-1})  |  "
            f"Ip={ip_ma:.3g} MA, Bt={float(mt.Bt):.3g} T, P_aux={float(at.P_aux_MW):.3g} MW  |  "
            f"Q={float(diag.get('Q', 0.0)):.3g}, "
            f"P_fus={float(diag.get('P_fus_MW', 0.0)):.3g} MW, "
            f"beta_N={float(diag.get('beta_N', 0.0)):.3g}, "
            f"f_G={float(diag.get('greenwald_fraction', 0.0)):.3g}",
            fontsize=12,
        )
        fig.canvas.draw_idle()

    back_button = Button(back_ax, "-1")
    fwd_button = Button(fwd_ax, "+1")
    back_button.on_clicked(lambda _event: _set_frame_index(current_index["idx"] - 1))
    fwd_button.on_clicked(lambda _event: _set_frame_index(current_index["idx"] + 1))

    slider.on_changed(draw)
    draw(tmax)
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig



def _default_input_file() -> Path | None:
    root = Path(__file__).resolve().parents[1]
    for name in ("iter_flattop_tglfnn.json", "iter_waveform_transition.json"):
        p = root / "inputs" / name
        if p.exists():
            return p
    return None


def _input_has_waveform(path: Path | None) -> bool:
    if path is None or not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return isinstance(data, dict) and "waveform" in data
    except Exception:
        return False


def _print_case_summary(kind: str, machine, actuator, sim, input_file: Path | None):
    print("TokaGrad unified simulation")
    print(f"  input_file        : {input_file if input_file else '(built-in fallback)'}")
    print(f"  case              : {kind}")
    print(f"  simulation_model  : {getattr(sim, 'simulation_model', '1.5d')}")
    print(f"  nr, ntheta        : {sim.nr}, {sim.ntheta}")
    print(f"  dt, n_steps       : {float(sim.dt):.6g} s, {int(sim.n_steps)}")
    print(f"  end_time          : {float(sim.dt) * int(sim.n_steps):.6g} s  (dt_mode={getattr(sim, 'dt_mode', 'fixed')})")
    print(f"  equilibrium_model : {sim.equilibrium_model}")
    if not zero_d_enabled(sim):
        print(f"  transport_model   : {getattr(sim, 'transport_model', '(0D none)')}")
    print(f"  pedestal_model    : {getattr(sim, 'pedestal_model', 'none')}")
    print(f"  Ip, Bt, P_aux     : {float(machine.Ip)/1e6:.4g} MA, {float(machine.Bt):.4g} T, {float(actuator.P_aux_MW):.4g} MW")


def _run_static(machine, actuator, sim):
    if zero_d_enabled(sim):
        state0 = initial_state_0d(machine, actuator, sim)
        final_state, hist = simulate_0d_jit(machine, actuator, sim, state0=state0)
    else:
        state0 = initial_state(machine, actuator, sim)
        final_state, hist = simulate_jit(machine, actuator, sim, state0=state0)
    return state0, final_state, hist


def _run_waveform(machine, actuator, sim, waveform):
    m0, a0 = apply_waveform_controls(machine, actuator, waveform, 0.0)
    if zero_d_enabled(sim):
        state0 = initial_state_0d(m0, a0, sim)
        final_state, wf_hist = simulate_waveform_0d(machine, actuator, sim, waveform, state0=state0)
    else:
        state0 = initial_state(m0, a0, sim)
        final_state, wf_hist = simulate_waveform(machine, actuator, sim, waveform, state0=state0)
    return state0, final_state, wf_hist


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run a TokaGrad simulation selected by an input JSON file.")
    p.add_argument("--input-file", default="", help="Input JSON. If omitted, inputs/waveform_iter_like.json is used when available.")
    p.add_argument("--save-figure", default="", help="Optional output PNG/PDF path.")
    p.add_argument("--no-show", action="store_true", help="Do not call plt.show().")
    p.add_argument("--no-plot", action="store_true", help="Run simulation and diagnostics only; skip plotting.")
    p.add_argument("--simple-final-plot", action="store_true", help="Plot only the final state without Matplotlib sliders/buttons. Useful on X11 servers without XInput 2.")
    p.add_argument("--plot-diffusivity", action="store_true", help="Include the optional chi_e/chi_i/Dn diffusivity debug panel in plots.")
    p.add_argument("--plot-psi", action="store_true", help="Include the poloidal-flux psi profile panel in plots.")
    p.add_argument("--quiet-timing", action="store_true", help="Suppress wall-time breakdown.")
    p.add_argument("--warmup", action="store_true", help="Run a one-step warm-up before timed solve to populate JAX/NN/GEQDSK caches.")
    p.add_argument("--profile-internals", action="store_true", help="Print cProfile top cumulative-time functions for the transport/evolution solve.")
    p.add_argument("--n-steps", type=int, default=None, help="Override simulation.n_steps after loading input.")
    p.add_argument("--dt", type=float, default=None, help="Override simulation.dt after loading input.")
    p.add_argument("--end-time", type=float, default=None, help="Override simulation.end_time_s and derive n_steps from dt/CFL.")
    p.add_argument("--cfl-dt", action="store_true", help="Set simulation.dt_mode='cfl' and derive dt from CFL*dx^2/D_ref.")
    p.add_argument("--cfl-number", type=float, default=None, help="Override simulation.cfl_number when using CFL dt mode.")
    p.add_argument("--cfl-diffusivity", type=float, default=None, help="Override reference diffusivity D_ref [m^2/s] for CFL dt mode.")
    p.add_argument("--nr", type=int, default=None, help="Override simulation.nr after loading input.")
    p.add_argument("--ntheta", type=int, default=None, help="Override simulation.ntheta after loading input.")
    p.add_argument("--zero-d-skip-current", action="store_true", help="0.5D profiling mode: keep previous psi/current instead of reconstructing current/q every step.")
    p.add_argument("--equilibrium-skip", type=int, default=None, help="Recompute equilibrium every N transport steps; default 1.")
    p.add_argument("--source-skip", type=int, default=None, help="Recompute source/heating terms every N transport steps; default 1.")
    p.add_argument("--transport-skip", type=int, default=None, help="Recompute transport coefficients every N transport steps; default 1.")
    p.add_argument("--pedestal-skip", type=int, default=None, help="Recompute L-H gate/pedestal terms every N transport steps; default 1.")
    p.add_argument("--zero-d-python-loop", action="store_true", help="0.5D debugging mode: use Python loop instead of lax.scan.")
    p.add_argument("--save-results", action="store_true", help="Save selected time slices to an NPZ file after the run.")
    p.add_argument("--save-result-times", default="", help="Comma-separated physical times [s] to save, e.g. '0,1,2.5'.")
    p.add_argument("--save-result-file", default="", help="Output .npz file for saved time slices.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    timer = StageTimer()

    input_file = Path(args.input_file).expanduser().resolve() if args.input_file else _default_input_file()
    has_waveform = _input_has_waveform(input_file)
    if input_file is not None and input_file.exists():
        if has_waveform:
            machine, actuator, sim, waveform = load_waveform_input(input_file)
        else:
            machine, actuator, sim = load_static_input(input_file)
            waveform = None
    else:
        machine, actuator, sim = MachineConfig(), ActuatorConfig(), SimulationConfig()
        waveform = iter_baseline_like_waveform(float(sim.dt) * int(sim.n_steps))
        has_waveform = True
        input_file = None

    # Lightweight command-line overrides for fast profiling.
    updates = {}
    if args.n_steps is not None:
        updates["n_steps"] = int(args.n_steps)
        updates["end_time_s"] = 0.0
    if args.dt is not None:
        updates["dt"] = float(args.dt)
        if not args.cfl_dt:
            updates["dt_mode"] = "fixed"
    if args.end_time is not None: updates["end_time_s"] = float(args.end_time)
    if args.cfl_dt: updates["dt_mode"] = "cfl"
    if args.cfl_number is not None: updates["cfl_number"] = float(args.cfl_number)
    if args.cfl_diffusivity is not None: updates["cfl_diffusivity_ref_m2_s"] = float(args.cfl_diffusivity)
    if args.plot_diffusivity: updates["plot_diffusivity"] = True
    if args.plot_psi: updates["plot_psi"] = True
    if args.nr is not None: updates["nr"] = int(args.nr)
    if args.ntheta is not None: updates["ntheta"] = int(args.ntheta)
    if args.zero_d_skip_current:
        updates["zero_d_reconstruct_current"] = False
    if args.equilibrium_skip is not None:
        updates["equilibrium_skip_steps"] = max(1, int(args.equilibrium_skip))
    if args.source_skip is not None:
        updates["source_skip_steps"] = max(1, int(args.source_skip))
    if args.transport_skip is not None:
        updates["transport_skip_steps"] = max(1, int(args.transport_skip))
    if args.pedestal_skip is not None:
        updates["pedestal_skip_steps"] = max(1, int(args.pedestal_skip))
    if args.zero_d_python_loop:
        updates["zero_d_use_lax_scan"] = False
    if args.save_results:
        updates["save_results_enabled"] = True
    if args.save_result_times:
        updates["save_result_times_s"] = tuple(float(x.strip()) for x in args.save_result_times.split(",") if x.strip())
    if args.save_result_file:
        updates["save_result_file"] = str(args.save_result_file)
        updates["save_results_enabled"] = True
    if updates:
        sim = replace(sim, **updates)
        sim = resolve_time_discretization(sim, machine, actuator)
    timer.mark("preprocessing / input loading")

    kind = ("0.5D" if zero_d_enabled(sim) else "1.5D") + (" waveform" if has_waveform else " fixed-actuator")
    _print_case_summary(kind, machine, actuator, sim, input_file)

    if args.warmup:
        # Populate Python caches, GEQDSK surfaces, neural surrogate parameters, and
        # JAX/XLA lowering caches before timing the requested discharge.  For 0.5D
        # lax.scan the scan length is part of the compiled program, so a one-step
        # warm-up does not remove the actual compile cost.  Use the same n_steps
        # for 0.5D scan warm-up; use one step elsewhere to avoid doing a full
        # expensive 1.5D discharge twice.
        warm_steps = int(sim.n_steps) if (zero_d_enabled(sim) and getattr(sim, "zero_d_use_lax_scan", True)) else 1
        warm_sim = replace(sim, n_steps=warm_steps)
        if has_waveform:
            _block_until_ready(_run_waveform(machine, actuator, warm_sim, waveform))
        else:
            _block_until_ready(_run_static(machine, actuator, warm_sim))
        timer.mark("warm-up / cache fill")

    prof = cProfile.Profile() if args.profile_internals else None
    if prof is not None:
        prof.enable()
    if has_waveform:
        # Initial state is built inside the waveform runner at t=0.
        state0, final_state, hist = _run_waveform(machine, actuator, sim, waveform)
    else:
        state0, final_state, hist = _run_static(machine, actuator, sim)
    _block_until_ready((state0, final_state, hist))
    if prof is not None:
        prof.disable()
    timer.mark("transport/evolution solve")
    if prof is not None:
        sio = io.StringIO()
        pstats.Stats(prof, stream=sio).strip_dirs().sort_stats("cumtime").print_stats(35)
        print("\nTransport/evolution cProfile top cumulative-time calls:")
        print(sio.getvalue())

    if zero_d_enabled(sim):
        diag0 = zero_d_diagnostics(state0, machine, actuator, sim)
        diagf = zero_d_diagnostics(final_state, machine, actuator, sim)
    else:
        diag0 = compute_diagnostics(state0, machine, actuator, sim)
        diagf = compute_diagnostics(final_state, machine, actuator, sim)
    _block_until_ready((diag0, diagf))
    timer.mark("diagnostics")

    if bool(getattr(sim, "save_results_enabled", False)):
        rho_save, _, _ = make_grid_from_config(sim.nr, machine.a, sim)
        _save_time_slice_results(
            np.asarray(rho_save), state0, final_state, hist, machine, actuator, sim,
            waveform=waveform if has_waveform else None,
            input_file=input_file,
        )
        timer.mark("result saving")

    print("\nInitial diagnostics:")
    for k in ("Te0_keV", "Ti0_keV", "ne_avg_1e20", "greenwald_fraction", "Q", "P_fus_MW", "beta_N"):
        if k in diag0:
            print(f"  {k:<22s} {float(diag0[k]):.6g}")
    print("Final diagnostics:")
    for k in ("Te0_keV", "Ti0_keV", "ne_avg_1e20", "greenwald_fraction", "Q", "P_fus_MW", "P_aux_abs_MW", "tau_E_s", "beta_N"):
        if k in diagf:
            print(f"  {k:<22s} {float(diagf[k]):.6g}")
    if zero_d_enabled(sim):
        try:
            t_axis = _simulation_times(sim, hist)
            print(f"  actual_time_end_s    {float(t_axis[-1]):.6g}  (0.5D accumulated dt_eff)")
        except Exception:
            pass

    if not args.quiet_timing:
        timer.print()

    if not args.no_plot:
        rho, _, dr = make_grid_from_config(sim.nr, machine.a, sim)
        show = (not args.no_show) and _backend_can_show()
        if args.simple_final_plot:
            plot_final_state_static(
                machine, actuator, sim, waveform if has_waveform else None, rho, state0, final_state, hist,
                show=show, save_path=args.save_figure
            )
        elif has_waveform:
            plot_waveform_single_figure_time_slider(
                rho, state0, final_state, machine, actuator, sim, waveform, hist,
                show=show,
                save_path=args.save_figure,
            )
        else:
            plot_simulation_time_slider(
                machine, actuator, sim, waveform if has_waveform else None, rho, state0, final_state, hist,
                show=show, save_path=args.save_figure
            )


if __name__ == "__main__":
    main()
