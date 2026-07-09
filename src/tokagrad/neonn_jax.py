"""JAX-native BrainFUSE NEOjbs-NN adapter for reduced neoclassical transport.

Physics references for fallback/output interpretation:
  [O. Sauter et al., Phys. Plasmas 6, 2834 (1999)].
  [C. Angioni and O. Sauter, Phys. Plasmas 7, 1224 (2000)].
The network files, rather than an analytic formula here, define the surrogate.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import re
import jax
import jax.numpy as jnp

from .eped1nn_jax import BrainfuseJaxEnsemble, parse_brainfuse_net, brainfuse_ensemble_forward
from .qlknn_adapter import collisionality_proxy, _maybe_bound, _maybe_lower, _maybe_abs


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _candidate_roots(model_dir: str | Path) -> list[Path]:
    raw = Path(model_dir).expanduser(); roots=[raw]
    if not raw.is_absolute():
        project=_project_root(); roots += [Path.cwd()/raw, project/raw, project/'scripts'/raw, project/'src'/raw]
    out=[]; seen=set()
    for r in roots:
        key=str(r.resolve(strict=False))
        if key not in seen:
            seen.add(key); out.append(r)
    return out


def _candidate_model_dirs(root: Path, model_name: str) -> list[Path]:
    return [
        root / 'neonn' / model_name,
        root / 'neural' / 'neonn' / model_name,
        root / 'external_models' / 'neural' / 'neonn' / model_name,
        root / model_name,
        root,
    ]


def find_neonn_model_dir(model_dir: str | Path, model_name: str = 'jbsnn') -> Path | None:
    for root in _candidate_roots(model_dir):
        for cand in _candidate_model_dirs(root, model_name):
            if cand.exists() and any(cand.glob('brainfuse_*.net')):
                return cand.resolve()
    return None


@lru_cache(maxsize=8)
def load_neonn_ensemble(model_dir: str, model_name: str = 'jbsnn', max_nets: int = 0) -> BrainfuseJaxEnsemble:
    model_path = find_neonn_model_dir(model_dir, model_name)
    if model_path is None:
        raise FileNotFoundError(f"No NEOjbs-NN brainfuse_*.net files found for {model_name!r} under {model_dir!r}.")
    files = sorted(model_path.glob('brainfuse_*.net'), key=lambda p: int(re.findall(r'\d+', p.stem)[-1]))
    if max_nets and max_nets > 0:
        files = files[:max_nets]
    nets = tuple(parse_brainfuse_net(p) for p in files)
    first = nets[0]
    return BrainfuseJaxEnsemble(str(model_path), nets, first.input_names, first.output_names)


def neonn_jax_status(model_dir: str, model_name: str = 'jbsnn', max_nets: int = 0):
    try:
        ens=load_neonn_ensemble(str(model_dir), str(model_name), int(max_nets))
        return True, f"Loaded {len(ens.nets)} NEOjbs-NN JAX BrainFUSE nets from {ens.model_path}."
    except Exception as exc:
        return False, str(exc)


def predict_neonn_jax(x_phys, model_dir: str, model_name: str = 'jbsnn', max_nets: int = 0):
    ens=load_neonn_ensemble(str(model_dir), str(model_name), int(max_nets))
    x_phys=jnp.asarray(x_phys)
    if x_phys.ndim == 1:
        return brainfuse_ensemble_forward(ens, x_phys)
    return jax.vmap(lambda row: brainfuse_ensemble_forward(ens, row))(x_phys)


def build_neonn_features(rho, Te, Ti, ne20, q, machine, sim=None):
    rho=jnp.asarray(rho); Te=jnp.asarray(Te); Ti=jnp.asarray(Ti); ne20=jnp.asarray(ne20)
    eps = _maybe_lower(machine.a * rho / (machine.R0 + 1e-12), 1.0e-4, sim, 1e-5)
    lognu = collisionality_proxy(Te, ne20, q, rho, machine, sim=sim) / jnp.log(10.0)
    t2 = _maybe_lower(Ti / _maybe_lower(Te, 0.03, sim, 1e-4), 0.05, sim, 1e-3)
    # Reduced proxies for second-ion density fraction and geometry factor.
    n2 = jnp.ones_like(rho)
    geo = jnp.sqrt(_maybe_lower(eps, 1e-4, sim, 1e-5)) * (1.0 + 0.25 * (machine.kappa - 1.0))
    vals = {
        'in1_eps': eps,
        'in2_q': q,
        'in3_nu': lognu,
        'in4_n2': n2,
        'in5_t2': t2,
        'in6_geo': geo,
    }
    ens=load_neonn_ensemble(str(getattr(sim, 'neonn_model_dir', 'external_models/neural')), str(getattr(sim, 'neonn_model_name', 'jbsnn')), int(getattr(sim, 'neonn_jax_max_nets', 0))) if sim is not None else None
    names = ens.input_names if ens is not None else tuple(vals.keys())
    return jnp.stack([vals.get(name, jnp.zeros_like(rho)) for name in names], axis=-1), vals



def _auto_bootstrap_output_name(output_names, requested: str = '') -> str | None:
    """Find a likely bootstrap-current output name, if one exists.

    The public BrainFUSE model bundled with this repository currently exposes
    neoclassical transport coefficient-like outputs (OUT_cne, OUT_cte, ...).
    Some NEO/BrainFUSE exports may include a bootstrap-current output; this
    helper keeps support explicit and conservative.
    """
    if requested:
        return requested if requested in output_names else None
    candidates = []
    for name in output_names:
        low = str(name).lower()
        if any(tok in low for tok in ('jbs', 'j_boot', 'jboot', 'bootstrap')):
            candidates.append(name)
    return candidates[0] if candidates else None


def neonn_jax_bootstrap_current_density(rho, Te, Ti, ne20, q, machine, sim, eq=None):
    """Optional NEO/BrainFUSE bootstrap-current predictor [A/m^2].

    This function is intentionally conservative.  It only returns a value when
    the loaded model exposes a recognized bootstrap-current output and the output
    units are known or inferable from the output name.  The bundled ``jbsnn``
    BrainFUSE files in this repository do not expose such an output; they expose
    transport coefficient-like outputs and should continue to use the Sauter
    bootstrap closure.
    """
    model_dir = getattr(sim, 'neonn_model_dir', 'external_models/neural')
    model_name = getattr(sim, 'neonn_model_name', 'jbsnn')
    max_nets = int(getattr(sim, 'neonn_jax_max_nets', 0))
    ens = load_neonn_ensemble(str(model_dir), str(model_name), max_nets)
    requested = str(getattr(sim, 'neonn_bootstrap_output_name', '') or '')
    out_name = _auto_bootstrap_output_name(ens.output_names, requested=requested)
    if out_name is None:
        raise KeyError(
            'Loaded NEO/BrainFUSE model does not expose a bootstrap-current output. '
            f'Available outputs: {ens.output_names}'
        )
    features, _vals = build_neonn_features(rho, Te, Ti, ne20, q, machine, sim=sim)
    out = predict_neonn_jax(features, model_dir, model_name, max_nets)
    od = {name: out[:, i] for i, name in enumerate(ens.output_names)}
    raw = od[out_name]
    units = str(getattr(sim, 'neonn_bootstrap_output_units', 'auto') or 'auto').lower()
    name_low = out_name.lower()
    if units in ('a_m2', 'a/m2', 'a_per_m2') or 'a_m2' in name_low or 'a/m2' in name_low:
        jbs = raw
    elif units in ('ma_m2', 'ma/m2', 'ma_per_m2') or 'ma_m2' in name_low or 'ma/m2' in name_low:
        jbs = raw * 1.0e6
    else:
        raise ValueError(
            f'Bootstrap output {out_name!r} was found, but its units are not known. '
            'Set sim.neonn_bootstrap_output_units to "A_m2" or "MA_m2".'
        )
    scale = getattr(sim, 'neonn_bootstrap_scale', 1.0)
    return scale * jbs

def neonn_jax_diffusivities(rho, Te, Ti, ne20, q, machine, sim):
    features, vals = build_neonn_features(rho, Te, Ti, ne20, q, machine, sim=sim)
    try:
        out = predict_neonn_jax(features, getattr(sim, 'neonn_model_dir', 'external_models/neural'), getattr(sim, 'neonn_model_name', 'jbsnn'), int(getattr(sim, 'neonn_jax_max_nets', 0)))
        ens = load_neonn_ensemble(str(getattr(sim, 'neonn_model_dir', 'external_models/neural')), str(getattr(sim, 'neonn_model_name', 'jbsnn')), int(getattr(sim, 'neonn_jax_max_nets', 0)))
        od = {name: out[:, i] for i, name in enumerate(ens.output_names)}
        cne = _maybe_abs(od.get('OUT_cne', 0.0), sim, 1e-8)
        cte = _maybe_abs(od.get('OUT_cte', 0.0), sim, 1e-8)
        cni = 0.5 * (_maybe_abs(od.get('OUT_cni1', 0.0), sim, 1e-8) + _maybe_abs(od.get('OUT_cni2', 0.0), sim, 1e-8))
        cti = 0.5 * (_maybe_abs(od.get('OUT_cti1', 0.0), sim, 1e-8) + _maybe_abs(od.get('OUT_cti2', 0.0), sim, 1e-8))
        # Map dimensionless coefficient-like outputs to the same order of m^2/s
        # as the previous Angioni closure.  This is a differentiable reduced
        # closure; calibrate neonn_transport_scale against a trusted NEO run.
        eps = vals['in1_eps']
        banana = jnp.sqrt(_maybe_lower(eps, 1e-4, sim, 1e-5))
        temp_fac = jnp.sqrt(_maybe_lower(Ti, 0.05, sim, 1e-4) / 5.0)
        base = getattr(sim, 'neonn_transport_scale', 1.0) * (0.6 + 2.5 * banana) * temp_fac
        chi_e = getattr(sim, 'neoclassical_chi_scale', 1.0) * base * cte
        chi_i = getattr(sim, 'neoclassical_chi_scale', 1.0) * base * 0.5 * (cni + cti)
        Dn = getattr(sim, 'neoclassical_D_scale', 1.0) * base * cne
        hi = getattr(sim, 'neoclassical_chi_max', 5.0)
        return (_maybe_bound(chi_e, 0.0, hi, sim, 1e-2), _maybe_bound(chi_i, 0.0, hi, sim, 1e-2), _maybe_bound(Dn, 0.0, hi, sim, 1e-2))
    except Exception:
        if getattr(sim, 'neonn_fail_mode', 'fallback') == 'raise':
            raise
        from .neoclassical import angioni_neoclassical_diffusivities
        return angioni_neoclassical_diffusivities(rho, Te, Ti, ne20, q, machine, sim, _allow_neonn_fallback=False)
