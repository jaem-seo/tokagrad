"""Input-vector adapter shared by EPED1-NN validation scripts.

Physics reference: [P. B. Snyder et al., Nucl. Fusion 51, 103016 (2011)].
The exact input ordering follows the distributed BrainFUSE model files.
"""

from __future__ import annotations

import jax.numpy as jnp

from .density import target_edge_ne20
from .heating import effective_ion_mass_amu


EPED1NN_INPUT_NAMES = [
    "a", "betaN", "bt", "delta", "ip_MA", "kappa", "mi",
    "neped_1e19", "R", "zeffped",
]


def build_eped1nn_input(machine, actuator, sim, beta_N_proxy=1.5, neped20=None):
    """Build the GA EPED1-NN 10-input vector.

    Runtime pedestal calls should pass ``neped20`` from the current density
    profile at the pedestal top.  The fallback here is only for standalone
    validation scripts that do not have a profile state available.
    """
    if neped20 is None:
        neped20 = jnp.maximum(target_edge_ne20(machine, actuator, sim), 0.05)
    return jnp.array([
        machine.a,
        beta_N_proxy,
        machine.Bt,
        machine.delta,
        machine.Ip / 1.0e6,
        machine.kappa,
        effective_ion_mass_amu(machine),
        neped20 * 10.0,
        machine.R0,
        machine.Zeff,
    ])
