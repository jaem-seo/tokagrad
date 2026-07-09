#!/usr/bin/env python3
"""Inspect and smoke-test the JAX-native EPED1-NN BrainFUSE backend.

Example:
  PYTHONPATH=src python scripts/check_eped1nn_jax.py \
      --model-dir external_models/neural --model-name EPED1_H_superH

If the optional external BrainFUSE executable is also available, this script can
be extended to compare JAX outputs with the legacy black-box adapter.  The core
purpose here is to verify that the brainfuse_*.net files can be parsed and that
JAX can differentiate the MLP output with respect to the EPED1-NN input vector.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax
import jax.numpy as jnp

from tokagrad.config import MachineConfig, ActuatorConfig, SimulationConfig
from tokagrad.eped1nn_adapter import build_eped1nn_input
from tokagrad.eped1nn_jax import (
    eped1nn_jax_status,
    load_brainfuse_ensemble,
    brainfuse_ensemble_forward,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="../external_models/neural/eped1nn/models/EPED1_H_superH/")
    ap.add_argument("--model-name", default="EPED1_H_superH")
    ap.add_argument("--max-nets", type=int, default=1, help="Use 1 for fast smoke test; 0 for full ensemble.")
    args = ap.parse_args()

    ok, msg = eped1nn_jax_status(args.model_dir, args.model_name, args.max_nets)
    print(msg)
    if not ok:
        return 2

    ens = load_brainfuse_ensemble(args.model_dir, args.model_name, args.max_nets)
    print(f"input_names  = {ens.input_names}")
    print(f"output_names = {ens.output_names}")
    print(f"n_nets       = {len(ens.nets)}")

    machine = MachineConfig()
    actuator = ActuatorConfig()
    sim = SimulationConfig(
        pedestal_model="eped1_nn_jax",
        eped1nn_model_dir=args.model_dir,
        eped1nn_model_name=args.model_name,
        eped1nn_jax_max_nets=args.max_nets,
    )
    x = build_eped1nn_input(machine, actuator, sim, beta_N_proxy=1.8)
    y = brainfuse_ensemble_forward(ens, x)
    jac = jax.jacobian(lambda z: brainfuse_ensemble_forward(ens, z))(x)
    print("sample x =", jnp.asarray(x))
    print("sample y =", y)
    print("dy/dx shape =", jac.shape)
    print("finite output:", bool(jnp.all(jnp.isfinite(y))))
    print("finite jacobian:", bool(jnp.all(jnp.isfinite(jac))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
