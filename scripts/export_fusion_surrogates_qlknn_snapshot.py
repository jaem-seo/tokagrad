"""Export DeepMind fusion_surrogates QLKNN model metadata/params snapshot.

This is optional. The current TokaGrad AD path can call the JAX-native
fusion_surrogates QLKNNModel directly. This exporter is provided only if you
later want to build a dependency-free pure-JAX loader or archive the exact model
snapshot used in a study.

Run on a machine where fusion_surrogates is installed:

    PYTHONPATH=src python scripts/export_fusion_surrogates_qlknn_snapshot.py \
        --out qlknn_snapshot.npz

The params are serialized with Flax msgpack and stored as a byte array inside
the npz, together with input/target names and stats arrays.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from flax import serialization

from fusion_surrogates.qlknn import qlknn_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="qlknn_snapshot.npz")
    args = parser.parse_args()

    model = qlknn_model.QLKNNModel.load_default_model()
    cfg = model.config
    stats = cfg.stats_data

    payload = {
        "model_name": np.asarray(model.name),
        "version": np.asarray(model.version),
        "input_names_json": np.asarray(json.dumps(cfg.input_names)),
        "target_names_json": np.asarray(json.dumps(cfg.target_names)),
        "flux_map_json": np.asarray(json.dumps(cfg.flux_map)),
        "network_type": np.asarray(str(cfg.network_type.value)),
        "network_config_json": np.asarray(json.dumps(cfg.network_config.__dict__)),
        "config_msgpack": np.frombuffer(cfg.serialize(), dtype=np.uint8),
        "params_msgpack": np.frombuffer(serialization.msgpack_serialize(model._params), dtype=np.uint8),
    }

    if stats is not None:
        payload.update({
            "input_mean": np.asarray(stats.input_mean),
            "input_stddev": np.asarray(stats.input_stddev),
            "target_mean": np.asarray(stats.target_mean),
            "target_stddev": np.asarray(stats.target_stddev),
            "input_min": np.asarray(stats.input_min),
            "input_max": np.asarray(stats.input_max),
        })

    np.savez(args.out, **payload)
    print(f"wrote {Path(args.out).resolve()}")
    print("inputs:", cfg.input_names)
    print("targets:", cfg.target_names)
    print("fluxes:", list(cfg.flux_map.keys()))


if __name__ == "__main__":
    main()
