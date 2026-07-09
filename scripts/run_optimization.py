"""Unified TokaGrad optimization entry point.

The input JSON decides whether the optimization is fixed-actuator or waveform:
  - no top-level "waveform" section: scalar fixed-actuator optimization
  - with "waveform" section: waveform-control optimization
  - simulation.simulation_model selects 1.5D or 0.5D fast rollout

Optional top-level "optimization" section can override objective, controls,
bounds, iterations, learning rate, and constraints.  Without it, conservative
small defaults are used.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import math
import os
import copy
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax.numpy as jnp

from tokagrad import MachineConfig, ActuatorConfig, SimulationConfig
from tokagrad.input_config import load_static_input, load_waveform_input, waveform_from_input_section
from tokagrad.optim_differentiable import (
    make_gradient_friendly_sim,
    optimize_controls_optax,
    optimize_waveform_optax,
    count_free_waveform_variables,
)
from tokagrad.controls import DEFAULT_CONTROL_BOUNDS

MACHINE_SCALAR_CONTROLS = frozenset({"Bt", "R0", "a", "kappa", "delta", "Zeff", "impurity_Z"})


def _load_raw(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _default_input_file() -> Path | None:
    root = Path(__file__).resolve().parents[1]
    for name in ("iter_flattop_tglfnn.json", "iter_waveform_transition.json"):
        p = root / "inputs" / name
        if p.exists():
            return p
    return None


def _as_tuple(x, default):
    if x is None:
        return tuple(default)
    if isinstance(x, str):
        return tuple(s.strip() for s in x.split(",") if s.strip())
    return tuple(x)


def _bounds_from_opt(keys, opt_sec):
    b = {k: DEFAULT_CONTROL_BOUNDS[k] for k in keys if k in DEFAULT_CONTROL_BOUNDS}
    user = opt_sec.get("bounds", {}) if isinstance(opt_sec, Mapping) else {}
    for k, v in user.items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            b[k] = (float(v[0]), float(v[1]))
    missing = [k for k in keys if k not in b]
    if missing:
        raise ValueError(
            f"Missing optimization bounds for control(s) {missing}. "
            "Add them under optimization.bounds as [min, max]."
        )
    return b


def _objective_from_opt(opt_sec):
    obj = opt_sec.get("objective", ["Q", "Pfus"])
    weights = opt_sec.get("objective_weights", [1.0, 0.01] if not isinstance(obj, str) else None)
    return obj, weights


def _constraints_from_opt(opt_sec):
    if "constraints" in opt_sec:
        return tuple(opt_sec["constraints"] or [])
    return (
        {"metric": "q_edge", "kind": "lower", "value": 3.0, "weight": 10.0},
        {"metric": "q95", "kind": "lower", "value": 2.7, "weight": 10.0},
        {"metric": "greenwald_fraction", "kind": "upper", "value": 1.0, "weight": 10.0},
        {"metric": "beta_N", "kind": "upper", "value": 4.0, "weight": 10.0},
    )


def _fixed_endpoint_options(opt_sec):
    """Parse waveform endpoint-fixing options from optimization section."""
    if not isinstance(opt_sec, Mapping):
        return False, False, None
    if "fixed_endpoints" in opt_sec:
        fixed_start = bool(opt_sec.get("fixed_endpoints", False))
        fixed_final = bool(opt_sec.get("fixed_endpoints", False))
    else:
        fixed_start = bool(opt_sec.get("fixed_start", opt_sec.get("fix_start", opt_sec.get("fixed_initial", False))))
        fixed_final = bool(opt_sec.get("fixed_final", opt_sec.get("fix_final", opt_sec.get("fixed_end", False))))
    fixed_keys = opt_sec.get("fixed_endpoint_keys", opt_sec.get("fixed_control_keys", None))
    if isinstance(fixed_keys, str):
        txt = fixed_keys.strip()
        if txt.lower() in {"", "all", "*"}:
            fixed_keys = None
        else:
            fixed_keys = tuple(k.strip() for k in txt.split(",") if k.strip())
    elif fixed_keys is not None:
        fixed_keys = tuple(fixed_keys)
    return fixed_start, fixed_final, fixed_keys


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        try:
            return float(jnp.asarray(x))
        except Exception:
            return math.nan


def _fmt_num(x, precision=6):
    v = _safe_float(x)
    if math.isnan(v) or math.isinf(v):
        return str(v)
    return f"{v:.{precision}g}"


def _print_metrics(metrics, keys, prefix="  "):
    for k in keys:
        if k in metrics:
            print(f"{prefix}{k:<32s} {_fmt_num(metrics[k])}", flush=True)


def _print_control_dict(control_dict, prefix="  "):
    for k, v in control_dict.items():
        vals = [_safe_float(x) for x in list(jnp.asarray(v))]
        print(f"{prefix}{k}: {[float(f'{x:.6g}') if math.isfinite(x) else x for x in vals]}", flush=True)


def _print_scalar_control_update(row, prefix="    "):
    names = tuple(row.get("control_names", ()))
    before = row.get("controls_phys_before_update")
    after = row.get("controls_phys")
    delta = row.get("controls_delta_phys")
    if not names or before is None or after is None:
        return False
    before_arr = list(jnp.asarray(before))
    after_arr = list(jnp.asarray(after))
    delta_arr = list(jnp.asarray(delta)) if delta is not None else [
        jnp.asarray(a) - jnp.asarray(b) for b, a in zip(before_arr, after_arr)
    ]
    for name, b, a, d in zip(names, before_arr, after_arr, delta_arr):
        print(
            f"{prefix}{name:<28s} {_fmt_num(b)} -> {_fmt_num(a)} "
            f"(Δ={_fmt_num(d)})",
            flush=True,
        )
    return True


def _print_scalar_gradient(row, prefix="    "):
    names = tuple(row.get("control_names", ()))
    grad = row.get("grad_z")
    finite = row.get("grad_finite")
    if not names or grad is None:
        return False
    grad_arr = list(jnp.asarray(grad))
    finite_arr = list(jnp.asarray(finite)) if finite is not None else [jnp.isfinite(g) for g in grad_arr]
    for name, g, ok in zip(names, grad_arr, finite_arr):
        status = "finite" if bool(ok) else "non-finite→0"
        print(f"{prefix}{name:<28s} grad_z={_fmt_num(g)} ({status})", flush=True)
    return True


def _print_waveform_control_update(row, prefix="    "):
    before = row.get("controls_phys_before_update")
    after = row.get("controls_phys")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return False
    for key in after:
        b = jnp.asarray(before.get(key, after[key]))
        a = jnp.asarray(after[key])
        d = a - b
        b_vals = [_safe_float(x) for x in list(b)]
        a_vals = [_safe_float(x) for x in list(a)]
        d_vals = [_safe_float(x) for x in list(d)]
        b_fmt = [float(f"{x:.6g}") if math.isfinite(x) else x for x in b_vals]
        a_fmt = [float(f"{x:.6g}") if math.isfinite(x) else x for x in a_vals]
        d_fmt = [float(f"{x:.6g}") if math.isfinite(x) else x for x in d_vals]
        print(f"{prefix}{key}: {b_fmt} -> {a_fmt} (Δ={d_fmt})", flush=True)
    return True


def _static_control_value(machine, actuator, key: str) -> float:
    """Return the physical scalar value used by optimization controls."""
    if key == "Ip_MA":
        return float(machine.Ip) / 1.0e6
    if key in MACHINE_SCALAR_CONTROLS:
        return float(getattr(machine, key))
    return float(getattr(actuator, key))


def _iteration_logger(kind, metric_keys):
    def cb(row):
        it = int(row.get("iter", -1))
        obj = _fmt_num(row.get("objective", math.nan))
        gnorm = _fmt_num(row.get("grad_norm", math.nan))
        unclip = _fmt_num(row.get("grad_norm_before_clip", math.nan))
        upnorm = _fmt_num(row.get("update_norm", math.nan))
        nbad = int(row.get("nonfinite_grad_count", 0) or 0)
        print(f"\n[{kind} iter {it:03d}] objective={obj} grad_norm={gnorm} raw_grad_norm={unclip} update_norm={upnorm} nonfinite_grad={nbad}", flush=True)
        if bool(row.get("grad_clip_applied", False)):
            print("  note: gradient norm clipped", flush=True)
        if bool(row.get("update_clip_applied", False)):
            print("  note: update step clipped", flush=True)
        if nbad > 0 and _safe_float(row.get("grad_norm", 0.0)) <= 0.0:
            print(
                "  warning: all usable AD gradients are zero after non-finite-gradient sanitization; "
                "try smoother models/settings or --gradient-mode finite_difference.",
                flush=True,
            )
        metrics = row.get("metrics", {}) or {}
        _print_metrics(metrics, metric_keys, prefix="  metric ")
        print("  controls update this iteration:", flush=True)
        printed = _print_waveform_control_update(row, prefix="    ")
        if not printed:
            printed = _print_scalar_control_update(row, prefix="    ")
        if not printed:
            controls = row.get("controls_phys_before_update", row.get("controls_phys", {}))
            if isinstance(controls, Mapping):
                _print_control_dict(controls, prefix="    ")
        if kind == "scalar":
            print("  scalar z-gradients:", flush=True)
            _print_scalar_gradient(row, prefix="    ")
    return cb




def _to_jsonable(x):
    """Convert JAX/NumPy/scalar containers to plain JSON values."""
    try:
        arr = jnp.asarray(x)
        if arr.ndim == 0:
            return float(arr)
        return [float(v) for v in list(arr)]
    except Exception:
        if isinstance(x, Mapping):
            return {str(k): _to_jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_to_jsonable(v) for v in x]
        try:
            return float(x)
        except Exception:
            return x


def _default_optimized_input_path(input_file: Path | None) -> Path:
    root = Path(__file__).resolve().parents[1]
    if input_file is not None:
        return input_file.with_name(f"optimized_{input_file.stem}.json")
    out_dir = root / "outputs"
    out_dir.mkdir(exist_ok=True)
    return out_dir / "optimized_simulation_input.json"


def _rebase_relative_simulation_paths(data: dict[str, Any], *, old_base: Path | None, new_base: Path) -> None:
    """Keep relative model/GEQDSK paths valid if optimized JSON is written elsewhere."""
    if old_base is None:
        return
    sim_sec = data.get("simulation", {})
    if not isinstance(sim_sec, dict):
        return
    path_keys = (
        "geqdsk_path",
        "eped1nn_model_dir",
        "tglfnn_model_dir",
        "neonn_model_dir",
        "fusion_surrogates_model_dir",
    )
    for key in path_keys:
        val = sim_sec.get(key)
        if not isinstance(val, str) or not val.strip():
            continue
        p = Path(val)
        if p.is_absolute():
            continue
        abs_path = (old_base / p).resolve()
        try:
            sim_sec[key] = os.path.relpath(abs_path, new_base)
        except Exception:
            sim_sec[key] = str(abs_path)


def _set_static_control_value(data: dict[str, Any], key: str, value: float) -> None:
    machine_sec = data.setdefault("machine", {})
    actuator_sec = data.setdefault("actuator", {})
    if key == "Ip_MA":
        machine_sec["Ip_MA"] = float(value)
    elif key == "Bt":
        machine_sec["Bt"] = float(value)
    elif key in {"R0", "a", "kappa", "delta", "Zeff", "impurity_Z"}:
        machine_sec[key] = float(value)
    else:
        actuator_sec[key] = float(value)


def _set_waveform_control_values(data: dict[str, Any], key: str, values) -> None:
    wf_sec = data.setdefault("waveform", {})
    controls = wf_sec.setdefault("controls", {})
    vals = [float(v) for v in list(jnp.asarray(values))]
    old = controls.get(key)
    if isinstance(old, Mapping):
        new_spec = dict(old)
        new_spec["values"] = vals
        controls[key] = new_spec
    else:
        controls[key] = vals


def _build_optimized_simulation_input(raw: Mapping[str, Any], *, has_waveform: bool, result: Mapping[str, Any], keys, sim, input_file: Path | None, output_file: Path) -> dict[str, Any]:
    """Return an input JSON dictionary that run_simulation.py can execute with optimized controls."""
    data = copy.deepcopy(dict(raw)) if isinstance(raw, Mapping) else {}
    data.setdefault("machine", {})
    data.setdefault("actuator", {})
    sim_sec = data.setdefault("simulation", {})
    if isinstance(sim_sec, dict):
        # Persist common CLI overrides so the post-simulation exactly mirrors the optimized rollout.
        sim_sec["simulation_model"] = getattr(sim, "simulation_model", sim_sec.get("simulation_model", "1.5d"))
        sim_sec["nr"] = int(getattr(sim, "nr", sim_sec.get("nr", 48)))
        sim_sec["ntheta"] = int(getattr(sim, "ntheta", sim_sec.get("ntheta", 16)))
        sim_sec["n_steps"] = int(getattr(sim, "n_steps", sim_sec.get("n_steps", 1)))
        sim_sec["dt"] = float(getattr(sim, "dt", sim_sec.get("dt", 1.0e-3)))
    if has_waveform:
        wf_sec = data.setdefault("waveform", {})
        name = str(wf_sec.get("name", "waveform"))
        if not name.endswith("_optimized"):
            wf_sec["name"] = f"{name}_optimized"
        for k, v in (result.get("controls_phys", {}) or {}).items():
            _set_waveform_control_values(data, str(k), v)
    else:
        x_phys = list(jnp.asarray(result.get("x_phys", [])))
        for k, v in zip(keys, x_phys):
            _set_static_control_value(data, str(k), float(v))
    # Keep optimization info for provenance, but mark the file as simulation-ready.
    data["post_optimization"] = {
        "source_input_file": str(input_file) if input_file is not None else "",
        "objective": _to_jsonable(result.get("objective")),
        "metrics": _to_jsonable(result.get("metrics", {})),
        "note": "This file was generated by run_optimization.py and can be passed directly to scripts/run_simulation.py.",
    }
    _rebase_relative_simulation_paths(data, old_base=input_file.parent if input_file is not None else None, new_base=output_file.parent)
    return data


def _write_optimized_simulation_input(data: Mapping[str, Any], output_file: Path) -> Path:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    return output_file


def _run_post_optimization_simulation(input_json: Path, *, save_figure: str = "", no_show: bool = False) -> int:
    run_sim = Path(__file__).resolve().with_name("run_simulation.py")
    cmd = [sys.executable, str(run_sim), "--input-file", str(input_json)]
    if save_figure:
        cmd.extend(["--save-figure", str(Path(save_figure).expanduser())])
    if no_show:
        cmd.append("--no-show")
    print("\nRunning post-optimization simulation:")
    print("  " + " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        print(f"Post-optimization simulation exited with code {completed.returncode}", flush=True)
    return int(completed.returncode)

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run TokaGrad optimization selected by input JSON.")
    p.add_argument("--input-file", default="", help="Input JSON. Presence of waveform section selects waveform optimization.")
    p.add_argument("--simulation-model", choices=["1.5d", "0d_fast"], default="", help="Optional override for simulation.simulation_model.")
    p.add_argument("--n-iter", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--gradient-mode", choices=["autodiff", "finite_difference"], default="")
    p.add_argument("--nr", type=int, default=None)
    p.add_argument("--ntheta", type=int, default=None)
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--no-profile-timing", action="store_true", help="Do not run extra forward timing probes inside optimization iterations.")
    p.add_argument("--no-iteration-log", action="store_true", help="Suppress per-iteration objective/metric/control logging.")
    p.add_argument("--no-post-simulation", action="store_true", help="Do not automatically run run_simulation.py after optimization.")
    p.add_argument("--post-sim-no-show", action="store_true", help="Pass --no-show to the automatic post-optimization simulation.")
    p.add_argument("--post-sim-save-figure", default="", help="Optional figure path for the automatic post-optimization simulation.")
    p.add_argument("--optimized-input-file", default="", help="Where to write the optimized simulation input JSON. Default: next to the input file as optimized_<name>.json.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    t0 = time.perf_counter()
    input_file = Path(args.input_file).expanduser().resolve() if args.input_file else _default_input_file()
    raw = _load_raw(input_file) if input_file else {}
    opt_sec = raw.get("optimization", {}) if isinstance(raw.get("optimization", {}), Mapping) else {}
    has_waveform = "waveform" in raw

    if input_file and input_file.exists():
        if has_waveform:
            machine, actuator, sim, waveform = load_waveform_input(input_file)
        else:
            machine, actuator, sim = load_static_input(input_file)
            waveform = None
    else:
        machine, actuator, sim, waveform = MachineConfig(), ActuatorConfig(), SimulationConfig(), None
        has_waveform = False

    if args.simulation_model:
        sim = replace(sim, simulation_model=args.simulation_model)
    sim_updates = {}
    if args.nr is not None: sim_updates["nr"] = args.nr
    if args.ntheta is not None: sim_updates["ntheta"] = args.ntheta
    if args.n_steps is not None: sim_updates["n_steps"] = args.n_steps
    if args.dt is not None: sim_updates["dt"] = args.dt
    if sim_updates:
        sim = replace(sim, **sim_updates)

    gradient_mode = args.gradient_mode or opt_sec.get("gradient_mode", "autodiff")
    n_iter = int(args.n_iter if args.n_iter is not None else opt_sec.get("n_iter", 3 if not has_waveform else 1))
    lr = float(args.learning_rate if args.learning_rate is not None else opt_sec.get("learning_rate", 1.0e-2 if not has_waveform else 5.0e-3))
    objective, objective_weights = _objective_from_opt(opt_sec)
    constraints = _constraints_from_opt(opt_sec)
    max_grad_norm = opt_sec.get("max_grad_norm", 10.0)
    max_update_norm = opt_sec.get("max_update_norm", 0.75)
    max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
    max_update_norm = None if max_update_norm is None else float(max_update_norm)
    ad_smooth_mode = bool(opt_sec.get("ad_smooth_mode", False))
    if gradient_mode == "autodiff" and ad_smooth_mode:
        smooth_updates = {"differentiable_smooth_mode": True}
        if hasattr(sim, "pedestal_projection_fraction"):
            smooth_updates["pedestal_projection_fraction"] = 0.0
        sim = replace(sim, **smooth_updates)

    # Keep optimization runs lighter than full demos unless explicitly overridden.
    #sim = make_gradient_friendly_sim(base_sim=sim, nr=sim.nr, ntheta=sim.ntheta, dt=sim.dt, n_steps=sim.n_steps)

    print("TokaGrad unified optimization")
    print(f"  input_file       : {input_file if input_file else '(built-in defaults)'}")
    print(f"  case             : {'waveform' if has_waveform else 'fixed-actuator'}")
    print(f"  simulation_model : {sim.simulation_model}")
    print(f"  gradient_mode    : {gradient_mode}")
    print(f"  n_iter, lr        : {n_iter}, {lr}")
    print(f"  max_grad_norm    : {max_grad_norm}")
    print(f"  max_update_norm  : {max_update_norm}")
    if gradient_mode == "autodiff":
        print(f"  ad_smooth_mode   : {ad_smooth_mode} (differentiable_smooth_mode={getattr(sim, 'differentiable_smooth_mode', False)})")
    profile_timing = not args.no_profile_timing

    if has_waveform:
        keys = _as_tuple(opt_sec.get("controls"), ("Ip_MA", "P_aux_MW", "greenwald_fraction_target"))
        bounds = _bounds_from_opt(keys, opt_sec)
        # Use waveform control points from the input file as optimization variables.
        times = None
        initial = {}
        wf_sec = raw.get("waveform", {})
        global_times = wf_sec.get("times", None)
        controls = wf_sec.get("controls", {})
        for k in keys:
            spec = controls.get(k)
            if spec is None:
                # Fallback two-point waveform around the base config.
                if times is None:
                    times = jnp.asarray([0.0, float(sim.dt) * max(int(sim.n_steps) - 1, 1)], dtype=jnp.float32)
                base = _static_control_value(machine, actuator, k)
                initial[k] = jnp.asarray([base, base], dtype=jnp.float32)
                continue
            if isinstance(spec, Mapping):
                vals = spec.get("values")
                t = spec.get("times", global_times)
            else:
                vals = spec
                t = global_times
            if times is None:
                times = jnp.asarray(t, dtype=jnp.float32)
            initial[k] = jnp.asarray(vals, dtype=jnp.float32)
        if times is None:
            times = jnp.asarray([0.0, float(sim.dt) * max(int(sim.n_steps) - 1, 1)], dtype=jnp.float32)
        # Important: waveform control-point times define the actuator knots only.
        # They must not overwrite the simulation time grid.  The rollout uses
        # sim.n_steps and sim.dt from the input JSON or explicit CLI overrides.
        fixed_start, fixed_final, fixed_endpoint_keys = _fixed_endpoint_options(opt_sec)
        n_ctrl_values_total = int(sum(initial[k].size for k in keys if k in initial))
        n_ctrl_values_free = count_free_waveform_variables(
            initial, keys,
            fixed_start=fixed_start,
            fixed_final=fixed_final,
            fixed_endpoint_keys=fixed_endpoint_keys,
        )
        n_objective_terms = len(objective) if isinstance(objective, (list, tuple)) else len(str(objective).split("+"))
        if gradient_mode == "autodiff":
            print(f"  rollout call estimate: {n_iter} value_and_grad calls + 1 final forward; weighted objectives share one backprop per iteration")
        else:
            print(f"  rollout call estimate: finite-difference about {n_iter * (1 + 2*n_ctrl_values_free) + 1} forward objective calls for {n_ctrl_values_free} free waveform variables")
        print(f"  objective terms   : {n_objective_terms}; waveform scalar variables: {n_ctrl_values_total} total, {n_ctrl_values_free} free")
        print(f"  fixed endpoints   : start={fixed_start}, final={fixed_final}, keys={fixed_endpoint_keys if fixed_endpoint_keys is not None else 'all optimized controls'}")
        iter_cb = None if args.no_iteration_log else _iteration_logger(
            "waveform",
            ["objective_performance", "Q_integral", "P_fus_avg_MW", "P_aux_avg_MW", "Q", "P_fus_MW", "Te0_keV", "Ti0_keV", "q95", "q_edge", "greenwald_fraction", "beta_N", "constraint_penalty_used", "constraint_penalty_final", "constraint_penalty_integral_avg", "waveform_constraint_mode_final", "waveform_regularization"],
        )
        result = optimize_waveform_optax(
            times=times,
            initial_control_values=initial,
            bounds=bounds,
            objective=objective,
            objective_weights=objective_weights,
            keys=keys,
            base_machine=machine,
            base_actuator=actuator,
            base_sim=sim,
            n_iter=n_iter,
            learning_rate=lr,
            gradient_mode=gradient_mode,
            constraints=constraints,
            smoothness_weight=float(opt_sec.get("smoothness_weight", 0.05)),
            slew_weight=float(opt_sec.get("slew_weight", 0.02)),
            fixed_endpoints=bool(opt_sec.get("fixed_endpoints", False)),
            fixed_start=fixed_start,
            fixed_final=fixed_final,
            fixed_endpoint_keys=fixed_endpoint_keys,
            waveform_constraint_mode=str(opt_sec.get("waveform_constraint_mode", "auto")),
            profile_timing=profile_timing,
            max_grad_norm=max_grad_norm,
            max_update_norm=max_update_norm,
            iteration_callback=iter_cb,
        )
        print("objective =", float(result["objective"]))
        print("metrics:")
        _print_metrics(result["metrics"], ["Q_integral", "P_fus_avg_MW", "P_aux_avg_MW", "rollout_duration_s", "Te0_keV", "Ti0_keV", "q95", "greenwald_fraction", "beta_N", "constraint_penalty_used", "constraint_penalty_final", "constraint_penalty_integral_avg", "waveform_constraint_mode_final", "waveform_regularization", "objective_performance"])
        print("optimized controls:")
        for k, v in result["controls_phys"].items():
            print(f"  {k}: {[float(x) for x in v]}")
    else:
        keys = _as_tuple(opt_sec.get("controls"), ("Ip_MA", "P_aux_MW", "greenwald_fraction_target"))
        bounds = _bounds_from_opt(keys, opt_sec)
        x0 = [_static_control_value(machine, actuator, k) for k in keys]
        n_ctrl_values = len(keys)
        n_objective_terms = len(objective) if isinstance(objective, (list, tuple)) else len(str(objective).split("+"))
        if gradient_mode == "autodiff":
            print(f"  rollout call estimate: {n_iter} value_and_grad calls + 1 final forward; weighted objectives share one backprop per iteration")
        else:
            print(f"  rollout call estimate: finite-difference about {n_iter * (1 + 2*n_ctrl_values) + 1} forward objective calls for {n_ctrl_values} controls")
        print(f"  objective terms   : {n_objective_terms}; optimized scalar variables: {n_ctrl_values}")
        iter_cb = None if args.no_iteration_log else _iteration_logger(
            "scalar",
            ["objective_performance", "Q", "P_fus_MW", "P_aux_MW", "Te0_keV", "Ti0_keV", "q_edge", "q95", "q0", "greenwald_fraction", "beta_N", "constraint_penalty"],
        )
        result = optimize_controls_optax(
            jnp.asarray(x0, dtype=jnp.float32),
            objective=objective,
            objective_weights=objective_weights,
            base_machine=machine,
            base_actuator=actuator,
            base_sim=sim,
            names=keys,
            bounds=bounds,
            n_iter=n_iter,
            learning_rate=lr,
            gradient_mode=gradient_mode,
            constraints=constraints,
            profile_timing=profile_timing,
            max_grad_norm=max_grad_norm,
            max_update_norm=max_update_norm,
            iteration_callback=iter_cb,
        )
        print("objective =", float(result["objective"]))
        print("x_phys =", [float(x) for x in result["x_phys"]])
        print("metrics:")
        _print_metrics(result["metrics"], ["Q", "P_fus_MW", "P_aux_MW", "Te0_keV", "Ti0_keV", "q_edge", "q95", "q0", "greenwald_fraction", "beta_N", "constraint_penalty", "objective_performance"])

    timing_rows = result.get("timing", []) or []
    if timing_rows:
        print("timing by optimization iteration:")
        for tr in timing_rows:
            ft = tr.get("forward_time_s")
            gt = tr.get("value_and_grad_time_s")
            bt = tr.get("backprop_estimate_s")
            if ft is None:
                print(f"  iter {tr['iter']:02d}: gradient/eval={gt:.4f}s")
            else:
                print(f"  iter {tr['iter']:02d}: forward={ft:.4f}s, value+grad={gt:.4f}s, backprop_est≈{bt:.4f}s")
    print("history:")
    for row in result.get("history", []):
        print(f"  iter {row['iter']:02d}: obj={float(row['objective']):.6g}")

    opt_wall_time = time.perf_counter() - t0
    print(f"optimization_wall_time = {opt_wall_time:.4f} s")

    if not args.no_post_simulation:
        optimized_input = Path(args.optimized_input_file).expanduser().resolve() if args.optimized_input_file else _default_optimized_input_path(input_file)
        sim_input_data = _build_optimized_simulation_input(
            raw,
            has_waveform=has_waveform,
            result=result,
            keys=keys,
            sim=sim,
            input_file=input_file,
            output_file=optimized_input,
        )
        _write_optimized_simulation_input(sim_input_data, optimized_input)
        print(f"Wrote optimized simulation input: {optimized_input}")
        _run_post_optimization_simulation(
            optimized_input,
            save_figure=args.post_sim_save_figure,
            no_show=args.post_sim_no_show,
        )
    else:
        print("post-optimization simulation skipped (--no-post-simulation)")

    print(f"wall_time_total = {time.perf_counter() - t0:.4f} s")


if __name__ == "__main__":
    main()
