"""AD gradient smoke test for the smooth TokaGrad optimization path.

Run:
    PYTHONPATH=src python scripts/check_autodiff_gradients.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax.numpy as jnp
from dataclasses import replace
from tokagrad import SimulationConfig

from tokagrad.controls import unconstrained_from_controls
from tokagrad.optim_differentiable import (
    make_gradient_friendly_sim,
    autodiff_value_and_grad_pytree,
)


def main():
    names = ("Ip_MA", "P_aux_MW")
    sim = make_gradient_friendly_sim(base_sim=SimulationConfig(), nr=4, ntheta=4, n_steps=1, dt=2e-3)
    z = unconstrained_from_controls(jnp.asarray([15.0, 50.0], dtype=jnp.float32), names=names)

    for objective in ["Pfus", "Q", ("Q", "Pfus")]:
        (value, metrics), grad = autodiff_value_and_grad_pytree(
            z, objective=objective, objective_weights=(1.0, 0.01) if isinstance(objective, tuple) else None, base_sim=sim, names=names
        )
        print(
            objective,
            "value=", float(value),
            "grad=", [float(g) for g in grad],
            "finite=", bool(jnp.all(jnp.isfinite(grad))),
        )


if __name__ == "__main__":
    main()
    import sys as _sys, os as _os
    _sys.stdout.flush(); _sys.stderr.flush()
    _os._exit(0)
