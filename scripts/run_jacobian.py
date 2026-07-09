"""Plot final-objective sensitivities to scalar optimization controls.

The input format and control/objective names are the same as those used by
``scripts/run_optimization.py``.  AD first computes derivatives with respect
to physical controls, then the plotted entries are normalized as
``abs(x / y) * dy/dx`` so sensitivities of differently dimensioned controls
and objectives can be compared directly.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp
import numpy as np

from tokagrad.controls import apply_control_vector, pytree_configs_from_legacy
from tokagrad.input_config import load_static_input
from tokagrad.optim_differentiable import (
    _objective_metric_name,
    _parse_objective_terms,
    differentiable_metrics,
    differentiable_metrics_ad_clean,
    simulate_final_unrolled_ad_jit,
)
from tokagrad.solver import initial_state, simulate_final_dynamic_jit
from tokagrad.zero_d import (
    initial_state_0d,
    simulate_0d_final_dynamic_jit,
    zero_d_enabled,
)


SUPPORTED_CONTROLS = frozenset(
    {
        "Ip_MA",
        "Bt",
        "R0",
        "a",
        "kappa",
        "delta",
        "P_aux_MW",
        "heat_center",
        "greenwald_fraction_target",
    }
)
MACHINE_CONTROLS = frozenset({"Bt", "R0", "a", "kappa", "delta"})
CONTROL_UNITS = {
    "Ip_MA": "MA",
    "Bt": "T",
    "R0": "m",
    "a": "m",
    "P_aux_MW": "MW",
}


def _default_input_file() -> Path:
    return ROOT / "inputs" / "iter_flattop_1d.json"


def _load_raw(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("The input JSON root must be an object.")
    return data


def _as_tuple(value, default) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        return tuple(x.strip() for x in value.split(",") if x.strip())
    return tuple(str(x) for x in value)


def _initial_control_value(name, machine, actuator) -> float:
    if name == "Ip_MA":
        return float(machine.Ip) / 1.0e6
    if name in MACHINE_CONTROLS:
        return float(getattr(machine, name))
    return float(getattr(actuator, name))


def _objective_names(opt_section: Mapping[str, Any]) -> tuple[str, ...]:
    objective = opt_section.get("objective", ("Q", "Pfus"))
    weights = opt_section.get("objective_weights")
    # Parse with the optimization implementation so aliases and expressions
    # have exactly the same meaning.  We intentionally discard the weights:
    # this script differentiates every requested final metric independently.
    return tuple(name for name, _weight in _parse_objective_terms(objective, weights))


def _make_value_and_jacobian(
    machine,
    actuator,
    sim,
    controls: tuple[str, ...],
    metric_names: tuple[str, ...],
):
    base_machine, base_actuator = pytree_configs_from_legacy(machine, actuator)

    def final_objectives(x_phys):
        run_machine, run_actuator, run_sim = apply_control_vector(
            x_phys, base_machine, base_actuator, sim, names=controls
        )
        if zero_d_enabled(run_sim):
            state0 = initial_state_0d(run_machine, run_actuator, run_sim)
            final_state = simulate_0d_final_dynamic_jit(
                run_machine, run_actuator, run_sim, state0=state0
            )
        else:
            state0 = initial_state(run_machine, run_actuator, run_sim)
            if getattr(run_sim, "differentiable_smooth_mode", False):
                final_state = simulate_final_unrolled_ad_jit(
                    state0, run_machine, run_actuator, run_sim
                )
            else:
                final_state = simulate_final_dynamic_jit(
                    run_machine, run_actuator, run_sim, state0=state0
                )

        metric_fn = (
            differentiable_metrics_ad_clean
            if getattr(run_sim, "differentiable_smooth_mode", False)
            else differentiable_metrics
        )
        metrics = metric_fn(final_state, run_machine, run_actuator, run_sim)
        return jnp.stack([jnp.asarray(metrics[name]) for name in metric_names])

    def value_and_jacobian(x_phys):
        # One primal rollout, followed by one reverse pullback per objective.
        # vmap batches those pullbacks into an [n_objective, n_control] matrix.
        values, pullback = jax.vjp(final_objectives, x_phys)
        basis = jnp.eye(values.shape[0], dtype=values.dtype)
        jacobian = jax.vmap(lambda cotangent: pullback(cotangent)[0])(basis)
        return values, jacobian

    return jax.jit(value_and_jacobian)


def _control_labels(controls: tuple[str, ...]) -> list[str]:
    return [
        f"{name}\n[{CONTROL_UNITS[name]}]" if name in CONTROL_UNITS else name
        for name in controls
    ]


def _plot_jacobian(
    objective_names: tuple[str, ...],
    metric_names: tuple[str, ...],
    controls: tuple[str, ...],
    values: np.ndarray,
    normalized_jacobian: np.ndarray,
):
    import matplotlib.pyplot as plt

    n_objectives = len(objective_names)
    width = max(6.5, 1.35 * len(controls) + 2.5)
    fig, axes = plt.subplots(
        n_objectives,
        1,
        figsize=(width, max(3.5, 3.4 * n_objectives)),
        squeeze=False,
        constrained_layout=True,
    )
    x = np.arange(len(controls))
    labels = _control_labels(controls)

    for i, (objective, metric) in enumerate(zip(objective_names, metric_names)):
        ax = axes[i, 0]
        row = normalized_jacobian[i]
        colors = ["tab:blue" if value >= 0.0 else "tab:red" for value in row]
        bars = ax.bar(x, row, color=colors, alpha=0.85)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(x, labels)
        ax.set_ylabel(r"$|x/y|\,\partial y/\partial x$")
        metric_note = "" if objective == metric else f" ({metric})"
        ax.set_title(f"Final {objective}{metric_note} = {values[i]:.6g}")
        ax.grid(axis="y", alpha=0.25)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

        for bar, derivative in zip(bars, row):
            if not np.isfinite(derivative):
                label = str(derivative)
            else:
                label = f"{derivative:.3g}"
            ax.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2.0, derivative),
                xytext=(0, 4 if derivative >= 0.0 else -4),
                textcoords="offset points",
                ha="center",
                va="bottom" if derivative >= 0.0 else "top",
                fontsize=9,
            )

    fig.suptitle("Dimensionless final-objective Jacobian")
    return fig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot AD Jacobians of final optimization objectives with respect to controls."
    )
    parser.add_argument(
        "--input-file",
        default=str(_default_input_file()),
        help="Static simulation JSON containing an optimization section.",
    )
    parser.add_argument("--simulation-model", choices=["1.5d", "0d_fast"], default="")
    parser.add_argument("--nr", type=int, default=None)
    parser.add_argument("--ntheta", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--save-figure", default="", help="Optional PNG/PDF output path.")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    input_file = Path(args.input_file).expanduser().resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    raw = _load_raw(input_file)
    if "waveform" in raw:
        raise NotImplementedError(
            "run_jacobian.py currently accepts scalar fixed-actuator controls only; "
            "a waveform control has multiple independent control-point values."
        )
    opt_section = raw.get("optimization")
    if not isinstance(opt_section, Mapping):
        raise ValueError("The input JSON must contain an 'optimization' object.")

    controls = _as_tuple(
        opt_section.get("controls"),
        ("Ip_MA", "P_aux_MW", "greenwald_fraction_target"),
    )
    if not controls:
        raise ValueError("optimization.controls must not be empty.")
    unsupported = sorted(set(controls) - SUPPORTED_CONTROLS)
    if unsupported:
        raise ValueError(
            f"Unsupported differentiable control(s): {unsupported}. "
            f"Supported controls are {sorted(SUPPORTED_CONTROLS)}."
        )

    objective_names = _objective_names(opt_section)
    metric_names = tuple(
        _objective_metric_name(name, waveform=False) for name in objective_names
    )
    supported_metrics = {"Q", "P_fus_MW", "Ti0_keV", "Te0_keV", "beta_N"}
    unsupported_metrics = sorted(set(metric_names) - supported_metrics)
    if unsupported_metrics:
        raise ValueError(
            "Unsupported final objective metric(s) "
            f"{unsupported_metrics}; supported metrics are {sorted(supported_metrics)}."
        )

    machine, actuator, sim = load_static_input(input_file)
    updates = {}
    if args.simulation_model:
        updates["simulation_model"] = args.simulation_model
    if args.nr is not None:
        updates["nr"] = args.nr
    if args.ntheta is not None:
        updates["ntheta"] = args.ntheta
    if args.n_steps is not None:
        updates["n_steps"] = args.n_steps
    if args.dt is not None:
        updates["dt"] = args.dt
    if updates:
        sim = replace(sim, **updates)

    x0 = jnp.asarray(
        [_initial_control_value(name, machine, actuator) for name in controls]
    )
    evaluate = _make_value_and_jacobian(
        machine, actuator, sim, controls, metric_names
    )

    print("TokaGrad final-objective Jacobian")
    print(f"  input_file       : {input_file}")
    print(f"  simulation_model : {sim.simulation_model}")
    print(f"  nr, ntheta       : {sim.nr}, {sim.ntheta}")
    print(f"  n_steps, dt      : {sim.n_steps}, {sim.dt}")
    print(f"  objectives       : {list(objective_names)}")
    print(f"  controls         : {list(controls)}")
    print("  derivatives      : physical control coordinates (not bounded optimizer coordinates)")

    start = time.perf_counter()
    values_jax, jacobian_jax = evaluate(x0)
    jax.block_until_ready(jacobian_jax)
    elapsed = time.perf_counter() - start
    values = np.asarray(jax.device_get(values_jax), dtype=float)
    jacobian = np.asarray(jax.device_get(jacobian_jax), dtype=float)
    controls_phys = np.asarray(jax.device_get(x0), dtype=float)
    # Dimensionless sensitivity requested for plotting:
    #     |x / y| * dy/dx
    # The absolute value applies only to the scale factor, so the derivative's
    # direction (sign) is retained.  It is undefined for a zero objective.
    objective_scale = np.full_like(values, np.nan, dtype=float)
    nonzero_objective = np.abs(values) > np.finfo(float).tiny
    objective_scale[nonzero_objective] = 1.0 / np.abs(values[nonzero_objective])
    normalized_jacobian = (
        jacobian * np.abs(controls_phys)[None, :] * objective_scale[:, None]
    )

    print("\nFinal objective values:")
    for objective, metric, value in zip(objective_names, metric_names, values):
        print(f"  {objective:<12s} ({metric:<10s}) {value:.8g}")
    print("\nJacobian rows=objectives, columns=controls:")
    print("  " + " ".join(f"{name:>20s}" for name in controls))
    for objective, row in zip(objective_names, jacobian):
        print(f"  {objective:<12s}" + " ".join(f"{value:20.8e}" for value in row))
    print("\nDimensionless Jacobian |x/y| * dy/dx (used in plot):")
    print("  " + " ".join(f"{name:>20s}" for name in controls))
    for objective, row in zip(objective_names, normalized_jacobian):
        print(f"  {objective:<12s}" + " ".join(f"{value:20.8e}" for value in row))
    print(f"\ncompile_and_AD_time = {elapsed:.4f} s")

    fig = _plot_jacobian(
        objective_names, metric_names, controls, values, normalized_jacobian
    )
    if args.save_figure:
        output = Path(args.save_figure).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=args.dpi)
        print(f"Saved figure: {output}")
    if not args.no_show:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
