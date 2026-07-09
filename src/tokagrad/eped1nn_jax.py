"""JAX-native reader/evaluator for GA BrainFUSE/FANN EPED1-NN models.

This module parses the text ``brainfuse_*.net`` files used by the public
``gafusion/neural`` EPED1-NN models and reconstructs the FANN forward pass with
JAX operations.  It is intentionally conservative: when parsing fails the caller
can fall back to the analytic alpha-critical pedestal model.

The implementation follows the BrainFUSE wrapper convention:

    x_scaled = (x - scale_mean_in) / scale_deviation_in
    y_scaled = fann_run(x_scaled)
    y = y_scaled * scale_deviation_out + scale_mean_out
    y *= prod_i x_i ** norm_output[o, i]

Only feed-forward FANN networks whose saved connection graph points to earlier
neurons are supported, which is the case for standard and shortcut MLPs saved by
FANN/BrainFUSE.

Physics reference: [P. B. Snyder et al., Nucl. Fusion 51, 103016 (2011)].
The saved BrainFUSE/FANN files define the NN surrogate; parsing, scaling, and
ensemble averaging here do not constitute an independent pedestal model.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import jax
import jax.numpy as jnp


_FLOAT = r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eEdD][-+]?\d+)?"


@dataclass(frozen=True)
class BrainfuseJaxNet:
    path: str
    num_input: int
    num_output: int
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    scale_mean_in: jnp.ndarray
    scale_deviation_in: jnp.ndarray
    scale_mean_out: jnp.ndarray
    scale_deviation_out: jnp.ndarray
    norm_output: jnp.ndarray
    num_inputs_per_neuron: tuple[int, ...]
    layer_sizes: tuple[int, ...]
    output_indices: tuple[int, ...]
    dense_weights: tuple[jnp.ndarray, ...]
    dense_biases: tuple[jnp.ndarray, ...]
    dense_activation_function: tuple[int, ...]
    dense_activation_steepness: tuple[float, ...]
    activation_function: tuple[int, ...]
    activation_steepness: tuple[float, ...]
    conn_ptr: tuple[int, ...]
    conn_to: jnp.ndarray
    conn_weight: jnp.ndarray


@dataclass(frozen=True)
class BrainfuseJaxEnsemble:
    model_path: str
    nets: tuple[BrainfuseJaxNet, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]


def _numbers_from_line(text: str) -> list[float]:
    return [float(x.replace("D", "E").replace("d", "e")) for x in re.findall(_FLOAT, text)]


def _ints_from_line(text: str) -> list[int]:
    return [int(round(v)) for v in _numbers_from_line(text)]


def _find_value_line(text: str, key: str) -> str | None:
    m = re.search(rf"(?:^|\n){re.escape(key)}\s*=\s*([^\n\r]*)", text)
    return None if m is None else m.group(1).strip()


def _parse_numeric_vector(text: str, key: str, n: int | None = None, default: float = 0.0) -> np.ndarray:
    raw = _find_value_line(text, key)
    if raw is None:
        if n is None:
            return np.asarray([], dtype=float)
        return np.full(n, default, dtype=float)
    vals = np.asarray(_numbers_from_line(raw), dtype=float)
    if n is not None and vals.size != n:
        if vals.size < n:
            vals = np.pad(vals, (0, n - vals.size), mode="constant", constant_values=default)
        else:
            vals = vals[:n]
    return vals


def _parse_quoted_names(text: str, key: str, n: int | None = None, prefix: str = "x") -> tuple[str, ...]:
    raw = _find_value_line(text, key)
    if raw is None:
        if n is None:
            return tuple()
        return tuple(f"{prefix}{i}" for i in range(n))
    names = re.findall(r"'([^']*)'|\"([^\"]*)\"", raw)
    flat = [a or b for a, b in names]
    if not flat:
        # BrainFUSE save() uses Python repr strings separated by spaces.  This
        # fallback accepts simple unquoted tokens as well.
        flat = [tok for tok in raw.split() if tok]
    if n is not None and len(flat) != n:
        if len(flat) < n:
            flat = flat + [f"{prefix}{i}" for i in range(len(flat), n)]
        else:
            flat = flat[:n]
    return tuple(flat)


def _parse_int_scalar(text: str, key: str, default: int) -> int:
    raw = _find_value_line(text, key)
    if raw is None:
        return default
    nums = _ints_from_line(raw)
    return nums[0] if nums else default


def _section_between(text: str, start_key: str, end_key: str | None = None) -> str:
    start = text.find(start_key)
    if start < 0:
        return ""
    if end_key is None:
        return text[start:]
    end = text.find(end_key, start + len(start_key))
    return text[start:] if end < 0 else text[start:end]


def _parse_neuron_specs(text: str) -> list[tuple[int, int, float]]:
    sec = _section_between(text, "neurons", "connections")
    triples = re.findall(rf"\(\s*(\d+)\s*,\s*(\d+)\s*,\s*({_FLOAT})\s*\)", sec)
    return [(int(a), int(b), float(c.replace("D", "E").replace("d", "e"))) for a, b, c in triples]


def _parse_connections(text: str) -> list[tuple[int, float]]:
    sec = _section_between(text, "connections", None)
    # Drop possible section headers following connections in nonstandard files.
    # FANN places all connection tuples after the connections header.
    pairs = re.findall(rf"\(\s*(\d+)\s*,\s*({_FLOAT})\s*\)", sec)
    return [(int(a), float(b.replace("D", "E").replace("d", "e"))) for a, b in pairs]


def _activation(code: int, steepness: float, z):
    """Approximate FANN activation functions using JAX operations.

    FANN's most common EPED1-NN activations are sigmoid/symmetric sigmoid and
    linear.  Less common functions are included to avoid hard failure if a model
    uses them, but exact stepwise integer approximations are intentionally not
    reproduced because they are not useful for AD.
    """
    s = jnp.asarray(steepness, dtype=z.dtype)
    u = s * z
    # FANN sigmoid uses a 2*s factor internally; symmetric sigmoid maps to [-1,1].
    sigmoid = jax.nn.sigmoid(2.0 * u)
    sigmoid_sym = 2.0 * sigmoid - 1.0
    gaussian = jnp.exp(-(u * u))
    elliot = 0.5 * (u / (1.0 + jnp.abs(u)) + 1.0)
    elliot_sym = u / (1.0 + jnp.abs(u))
    linear = u
    return jnp.select(
        [
            code == 0,  # FANN_LINEAR
            code == 3,  # FANN_SIGMOID
            code == 4,  # FANN_SIGMOID_STEPWISE, smoothed here
            code == 5,  # FANN_SIGMOID_SYMMETRIC
            code == 6,  # FANN_SIGMOID_SYMMETRIC_STEPWISE, smoothed here
            code == 7,  # FANN_GAUSSIAN
            code == 8,  # FANN_GAUSSIAN_SYMMETRIC
            code == 9,  # FANN_ELLIOT
            code == 10, # FANN_ELLIOT_SYMMETRIC
            code == 11, # FANN_LINEAR_PIECE
            code == 12, # FANN_LINEAR_PIECE_SYMMETRIC
            code == 13, # FANN_SIN_SYMMETRIC
            code == 14, # FANN_COS_SYMMETRIC
            code == 15, # FANN_SIN
            code == 16, # FANN_COS
        ],
        [
            linear,
            sigmoid,
            sigmoid,
            sigmoid_sym,
            sigmoid_sym,
            gaussian,
            2.0 * gaussian - 1.0,
            elliot,
            elliot_sym,
            jnp.clip(linear, 0.0, 1.0),
            jnp.clip(linear, -1.0, 1.0),
            jnp.sin(u),
            jnp.cos(u),
            0.5 * (jnp.sin(u) + 1.0),
            0.5 * (jnp.cos(u) + 1.0),
        ],
        default=linear,
    )



def _try_build_dense_layers(
    layer_sizes: tuple[int, ...],
    n_in_per: Sequence[int],
    act: Sequence[int],
    steep: Sequence[float],
    ptr: Sequence[int],
    connections: Sequence[tuple[int, float]],
    num_input: int,
    num_output: int,
):
    """Convert a standard fully connected FANN graph into dense JAX arrays.

    FANN stores explicit bias neurons inside layer_sizes.  BrainFUSE EPED1-NN
    models use dense feed-forward layers with one bias/sentinel neuron per
    non-output layer and a final output-layer sentinel.  Dense reconstruction is
    much faster to trace/differentiate than evaluating every neuron as a Python
    graph, while the original graph metadata is still kept as a fallback.
    """
    if not layer_sizes or sum(layer_sizes) != len(n_in_per):
        return tuple(), tuple(), tuple(), tuple()

    starts = [0]
    for s in layer_sizes[:-1]:
        starts.append(starts[-1] + int(s))

    weights = []
    biases = []
    layer_acts = []
    layer_steeps = []

    prev_real = list(range(starts[0], starts[0] + int(num_input)))
    for layer_idx in range(1, len(layer_sizes)):
        cur_start = starts[layer_idx]
        cur_end = cur_start + int(layer_sizes[layer_idx])
        cur_layer = list(range(cur_start, min(cur_end, len(n_in_per))))
        cur_real = [i for i in cur_layer if int(n_in_per[i]) > 0]
        if layer_idx == len(layer_sizes) - 1:
            cur_real = cur_real[:num_output]
        if not cur_real:
            return tuple(), tuple(), tuple(), tuple()

        prev_start = starts[layer_idx - 1]
        prev_end = prev_start + int(layer_sizes[layer_idx - 1])
        prev_layer = list(range(prev_start, min(prev_end, len(n_in_per))))
        prev_bias = [i for i in prev_layer if int(n_in_per[i]) == 0 and i not in prev_real]
        prev_bias_idx = prev_bias[-1] if prev_bias else None
        prev_map = {idx: k for k, idx in enumerate(prev_real)}

        W = np.zeros((len(prev_real), len(cur_real)), dtype=float)
        b = np.zeros((len(cur_real),), dtype=float)
        for j, neuron_idx in enumerate(cur_real):
            s, e = int(ptr[neuron_idx]), int(ptr[neuron_idx + 1])
            for src, weight in connections[s:e]:
                src = int(src)
                if src in prev_map:
                    W[prev_map[src], j] += float(weight)
                elif prev_bias_idx is not None and src == prev_bias_idx:
                    b[j] += float(weight)
                else:
                    # Non-layer-local or shortcut connection: keep the graph
                    # fallback to avoid silently changing the model.
                    return tuple(), tuple(), tuple(), tuple()

        # EPED1-NN models use uniform activation per dense layer.  If not, the
        # graph fallback remains correct and differentiable, just slower.
        acts = {int(act[i]) for i in cur_real}
        steeps = {float(steep[i]) for i in cur_real}
        if len(acts) != 1 or len(steeps) != 1:
            return tuple(), tuple(), tuple(), tuple()
        weights.append(np.asarray(W, dtype=float))
        biases.append(np.asarray(b, dtype=float))
        layer_acts.append(next(iter(acts)))
        layer_steeps.append(next(iter(steeps)))
        prev_real = cur_real

    return tuple(weights), tuple(biases), tuple(layer_acts), tuple(layer_steeps)

def parse_brainfuse_net(path: str | Path) -> BrainfuseJaxNet:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="ignore")

    # GA/BrainFUSE EPED1-NN FANN files do not always include explicit
    # ``num_input=``/``num_output=`` header fields.  The public
    # EPED1_H_superH files, for example, provide ``layer_sizes=11 31 31 5``
    # where the input and hidden sizes include FANN bias neurons, while the
    # physical dimensions are given by scale vectors and input/output names:
    #   scale_mean_in  -> 10 physical inputs
    #   scale_mean_out -> 4 physical outputs
    # Treat scale/name metadata as authoritative and use layer_sizes only for
    # locating the real output neurons inside the saved FANN graph.
    layer_sizes = tuple(_ints_from_line(_find_value_line(text, "layer_sizes") or ""))
    scale_mean_in_raw = _parse_numeric_vector(text, "scale_mean_in", None, default=0.0)
    scale_dev_in_raw = _parse_numeric_vector(text, "scale_deviation_in", None, default=1.0)
    scale_mean_out_raw = _parse_numeric_vector(text, "scale_mean_out", None, default=0.0)
    scale_dev_out_raw = _parse_numeric_vector(text, "scale_deviation_out", None, default=1.0)
    input_names_raw = _parse_quoted_names(text, "input_names", None, prefix="x")
    output_names_raw = _parse_quoted_names(text, "output_names", None, prefix="y")

    num_input = _parse_int_scalar(text, "num_input", 0)
    if num_input <= 0:
        if scale_mean_in_raw.size:
            num_input = int(scale_mean_in_raw.size)
        elif input_names_raw:
            num_input = len(input_names_raw)
        elif layer_sizes:
            # FANN layer_sizes normally include one bias neuron in the input
            # layer.  This fallback is used only for non-BrainFUSE files.
            num_input = max(int(layer_sizes[0]) - 1, 1)

    num_output = _parse_int_scalar(text, "num_output", 0)
    if num_output <= 0:
        if scale_mean_out_raw.size:
            num_output = int(scale_mean_out_raw.size)
        elif output_names_raw:
            num_output = len(output_names_raw)
        elif layer_sizes:
            num_output = int(layer_sizes[-1])

    neurons = _parse_neuron_specs(text)
    connections = _parse_connections(text)
    if not neurons or not connections:
        raise ValueError(f"Could not parse FANN neurons/connections from {path}")
    if num_input <= 0 or num_output <= 0:
        raise ValueError(
            f"Could not determine BrainFUSE input/output sizes in {path}. "
            "Expected num_input/num_output, scale_mean_in/scale_mean_out, "
            "input_names/output_names, or layer_sizes metadata."
        )

    n_in_per = [n[0] for n in neurons]
    act = [n[1] for n in neurons]
    steep = [n[2] for n in neurons]

    if layer_sizes and sum(layer_sizes) != len(neurons):
        # Do not fail hard: some FANN variants omit sentinel neurons.  We can
        # still evaluate the graph; output_indices will fall back below.
        layer_sizes = tuple()

    if layer_sizes:
        out_start = int(sum(layer_sizes[:-1]))
        out_end = out_start + int(layer_sizes[-1])
        out_layer = list(range(out_start, min(out_end, len(neurons))))
        # Public EPED1-NN FANN files often store a zero-input sentinel/bias as
        # the last output-layer neuron.  The real outputs are the connected
        # neurons in the output layer, matching scale_mean_out/output_names.
        real_outputs = [i for i in out_layer if n_in_per[i] > 0]
        if len(real_outputs) >= num_output:
            output_indices = tuple(real_outputs[:num_output])
        else:
            output_indices = tuple(out_layer[:num_output])
    else:
        real_outputs = [i for i, n in enumerate(n_in_per) if n > 0]
        output_indices = tuple(real_outputs[-num_output:])

    if len(output_indices) != num_output:
        raise ValueError(
            f"Could not locate {num_output} output neurons in {path}; "
            f"found indices {output_indices}."
        )

    ptr = [0]
    for n in n_in_per:
        ptr.append(ptr[-1] + int(n))
    n_conn_needed = ptr[-1]
    if n_conn_needed > len(connections):
        raise ValueError(
            f"FANN file {path} declares {n_conn_needed} incoming connections but "
            f"only {len(connections)} tuples were parsed."
        )
    connections = connections[:n_conn_needed]

    dense_weights, dense_biases, dense_act, dense_steep = _try_build_dense_layers(
        layer_sizes, n_in_per, act, steep, ptr, connections, int(num_input), int(num_output)
    )

    scale_mean_in = _parse_numeric_vector(text, "scale_mean_in", num_input, default=0.0)
    scale_deviation_in = _parse_numeric_vector(text, "scale_deviation_in", num_input, default=1.0)
    scale_mean_out = _parse_numeric_vector(text, "scale_mean_out", num_output, default=0.0)
    scale_deviation_out = _parse_numeric_vector(text, "scale_deviation_out", num_output, default=1.0)
    input_names = _parse_quoted_names(text, "input_names", num_input, prefix="x")
    output_names = _parse_quoted_names(text, "output_names", num_output, prefix="y")

    norm_raw = _parse_numeric_vector(text, "norm_output", None, default=0.0)
    if norm_raw.size == num_output * num_input:
        norm_output = norm_raw.reshape((num_output, num_input))
    else:
        norm_output = np.zeros((num_output, num_input), dtype=float)

    conn_to = np.asarray([c[0] for c in connections], dtype=np.int32)
    conn_weight = np.asarray([c[1] for c in connections], dtype=float)

    return BrainfuseJaxNet(
        path=str(path),
        num_input=int(num_input),
        num_output=int(num_output),
        input_names=tuple(input_names),
        output_names=tuple(output_names),
        scale_mean_in=np.asarray(scale_mean_in, dtype=float),
        scale_deviation_in=np.asarray(scale_deviation_in, dtype=float),
        scale_mean_out=np.asarray(scale_mean_out, dtype=float),
        scale_deviation_out=np.asarray(scale_deviation_out, dtype=float),
        norm_output=np.asarray(norm_output, dtype=float),
        num_inputs_per_neuron=tuple(int(v) for v in n_in_per),
        layer_sizes=tuple(int(v) for v in layer_sizes),
        output_indices=tuple(int(v) for v in output_indices),
        dense_weights=dense_weights,
        dense_biases=dense_biases,
        dense_activation_function=tuple(int(v) for v in dense_act),
        dense_activation_steepness=tuple(float(v) for v in dense_steep),
        activation_function=tuple(int(v) for v in act),
        activation_steepness=tuple(float(v) for v in steep),
        conn_ptr=tuple(int(v) for v in ptr),
        conn_to=np.asarray(conn_to, dtype=np.int32),
        conn_weight=np.asarray(conn_weight, dtype=float),
    )


def brainfuse_net_forward(net: BrainfuseJaxNet, x_phys):
    x_phys = jnp.asarray(x_phys)
    if x_phys.ndim > 1:
        return jax.vmap(lambda row: brainfuse_net_forward(net, row))(x_phys)
    x = (x_phys - jnp.asarray(net.scale_mean_in)) / (jnp.asarray(net.scale_deviation_in) + 1e-30)

    if net.dense_weights:
        h = x
        for W, b, act, steep in zip(
            net.dense_weights,
            net.dense_biases,
            net.dense_activation_function,
            net.dense_activation_steepness,
        ):
            h = _activation(int(act), float(steep), h @ jnp.asarray(W) + jnp.asarray(b))
        y_scaled = h[: net.num_output]
    else:
        # Sequential graph fallback.  Zero-input neurons beyond the physical
        # input vector are FANN bias/sentinel neurons and emit 1.
        values = []
        for i, n_in in enumerate(net.num_inputs_per_neuron):
            if i < net.num_input:
                values.append(x[i])
            elif n_in == 0:
                values.append(jnp.asarray(1.0, dtype=x.dtype))
            else:
                start, end = net.conn_ptr[i], net.conn_ptr[i + 1]
                idx = jnp.asarray(net.conn_to[start:end])
                w = jnp.asarray(net.conn_weight[start:end])
                prev = jnp.asarray(values)
                z = jnp.sum(prev[idx] * w)
                values.append(_activation(net.activation_function[i], net.activation_steepness[i], z))
        y_scaled = jnp.stack([values[i] for i in net.output_indices])

    y = y_scaled * jnp.asarray(net.scale_deviation_out) + jnp.asarray(net.scale_mean_out)

    # BrainFUSE norm_output denormalization; guard x<=0 because some powers may
    # be fractional.  EPED inputs are physically positive except delta, whose
    # norm_output can be nonzero.  A smooth positive floor avoids NaNs in AD.
    x_pos = jnp.maximum(x_phys, 1.0e-30)
    norm = jnp.prod(x_pos[None, :] ** jnp.asarray(net.norm_output), axis=1)
    return y * norm


def _candidate_model_dirs(root: Path, model_name: str) -> list[Path]:
    return [
        root / "eped1nn" / "models" / model_name,
        root / "neural" / "eped1nn" / "models" / model_name,
        root / model_name,
        root,
    ]


def _project_root() -> Path:
    # .../src/tokagrad/eped1nn_jax.py -> project root
    return Path(__file__).resolve().parents[2]


def _candidate_roots(model_dir: str | Path) -> list[Path]:
    """Return robust interpretations of a user-supplied EPED1-NN path.

    Users often pass either the cloned neural repo root, the eped1nn model
    directory itself, or a path written relative to scripts/run_simulation.py
    such as "../external_models/...".  Runtime code should not depend on the
    current working directory, so we try all common anchors.
    """
    raw = Path(model_dir).expanduser()
    roots = [raw]
    if not raw.is_absolute():
        project = _project_root()
        roots.extend([
            Path.cwd() / raw,
            project / raw,
            project / "scripts" / raw,
            project / "src" / raw,
        ])
    out: list[Path] = []
    seen: set[str] = set()
    for r in roots:
        try:
            key = str(r.resolve(strict=False))
        except Exception:
            key = str(r)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def find_eped1nn_model_dir(model_dir: str | Path, model_name: str = "EPED1_H_superH") -> Path | None:
    for root in _candidate_roots(model_dir):
        for cand in _candidate_model_dirs(root, model_name):
            if cand.exists() and any(cand.glob("brainfuse_*.net")):
                return cand.resolve()
    return None


@lru_cache(maxsize=8)
def load_brainfuse_ensemble(model_dir: str, model_name: str = "EPED1_H_superH", max_nets: int = 0) -> BrainfuseJaxEnsemble:
    model_path = find_eped1nn_model_dir(model_dir, model_name)
    if model_path is None:
        raise FileNotFoundError(
            f"No brainfuse_*.net files found for model {model_name!r} under {model_dir!r}."
        )
    files = sorted(model_path.glob("brainfuse_*.net"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
    if max_nets and max_nets > 0:
        files = files[:max_nets]
    nets = tuple(parse_brainfuse_net(p) for p in files)
    if not nets:
        raise FileNotFoundError(f"No brainfuse_*.net files found in {model_path}")
    first = nets[0]
    return BrainfuseJaxEnsemble(
        model_path=str(model_path),
        nets=nets,
        input_names=first.input_names,
        output_names=first.output_names,
    )


def brainfuse_ensemble_forward(ensemble: BrainfuseJaxEnsemble, x_phys, return_std: bool = False):
    outs = jnp.stack([brainfuse_net_forward(net, x_phys) for net in ensemble.nets], axis=0)
    mean = jnp.mean(outs, axis=0)
    if return_std:
        return mean, jnp.std(outs, axis=0)
    return mean


def eped1nn_jax_status(model_dir: str, model_name: str = "EPED1_H_superH", max_nets: int = 0):
    try:
        ens = load_brainfuse_ensemble(str(model_dir), model_name, int(max_nets))
        return True, f"Loaded {len(ens.nets)} EPED1-NN JAX BrainFUSE nets from {ens.model_path}."
    except Exception as exc:
        return False, str(exc)


def predict_eped1nn_jax(x_phys, model_dir: str, model_name: str = "EPED1_H_superH", max_nets: int = 0):
    ens = load_brainfuse_ensemble(str(model_dir), str(model_name), int(max_nets))
    return brainfuse_ensemble_forward(ens, x_phys)
