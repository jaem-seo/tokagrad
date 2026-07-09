"""Check fusion_surrogates QLKNN_7_11 installation and API."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax.numpy as jnp

from tokagrad.qlknn_adapter import (
    QLKNN_INPUT_NAMES,
    fusion_surrogates_status,
    try_load_fusion_surrogates_model,
)

status = fusion_surrogates_status()
print("Import available:", status.available)
print("Message:", status.message)

model, model_status = try_load_fusion_surrogates_model()
print("Model status:", model_status.message)

if model is not None:
    print("Model name:", model.name)
    print("Model version:", model.version)
    print("Input names:", model.config.input_names)
    print("Expected input names:", QLKNN_INPUT_NAMES)
    print("Target names:", model.config.target_names)
    print("Flux names:", list(model.config.flux_map.keys()))
    x = jnp.ones((1, 4, model.num_inputs))
    fluxes = model.predict(x)
    print("Example flux output shapes:")
    for k, v in fluxes.items():
        print(f"  {k}: {v.shape}")
