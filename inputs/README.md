# TokaGrad input files

`run_simulation.py` and `run_optimization.py` read JSON files from this directory.

A file without a top-level `waveform` section is treated as a fixed-actuator / steady-style case.
A file with a top-level `waveform` section is treated as a waveform case.

The simulation dimensionality is selected by:

```json
"simulation": {"simulation_model": "1.5d"}
```

or

```json
"simulation": {"simulation_model": "0d_fast"}
```

Examples:

```bash
PYTHONPATH=src python scripts/run_simulation.py --input-file inputs/iter_flattop_1d.json
PYTHONPATH=src python scripts/run_simulation.py --input-file inputs/iter_waveform_1d.json
```

Optimization uses the same input files.  A top-level optional `optimization` section may contain fields like:

```json
"optimization": {
  "controls": ["Ip_MA", "P_aux_MW", "greenwald_fraction_target"],
  "objective": ["Q", "Pfus"],
  "objective_weights": [1.0, 0.01],
  "n_iter": 3,
  "learning_rate": 0.01,
  "gradient_mode": "autodiff",
  "bounds": {
    "Ip_MA": [8.0, 17.0],
    "P_aux_MW": [5.0, 120.0],
    "greenwald_fraction_target": [0.5, 1.05]
  },
  "constraints": [
    {"metric": "q95", "kind": "lower", "value": 2.7, "weight": 10.0},
    {"metric": "beta_N", "kind": "upper", "value": 4.0, "weight": 10.0}
  ]
}
```

## Waveform interpolation modes

Each waveform control can use either linear interpolation or zero-order-hold
step interpolation.  The default is linear, except `P_aux_MW`, which defaults to
step/hold because auxiliary heating is often switched in discrete power levels.

Section-wide syntax:

```json
"waveform": {
  "times": [0, 2, 6, 10],
  "interpolation": {"default": "linear", "P_aux_MW": "step"},
  "controls": {
    "Ip_MA": [2, 5, 15, 15],
    "P_aux_MW": [0, 5, 20, 50]
  }
}
```

Per-control syntax:

```json
"P_aux_MW": {
  "times": [0, 2, 6, 10],
  "values": [0, 5, 20, 50],
  "interpolation": "step"
}
```

`step` means previous-value hold: the value at `times[i] <= t < times[i+1]`
is `values[i]`, and at a knot the new value is applied immediately.

### Post-optimization simulation

`run_optimization.py` now writes an optimized simulation input JSON after the
optimizer finishes and automatically launches `scripts/run_simulation.py` with
that file.  By default the optimized input is written next to the original input
as `optimized_<input-name>.json`.

Useful options:

```bash
python scripts/run_optimization.py --input-file inputs/iter_waveform_1d.json \
  --post-sim-save-figure outputs/optimized_waveform.png --post-sim-no-show
```

To only optimize and skip the follow-up simulation:

```bash
python scripts/run_optimization.py --input-file inputs/iter_waveform_1d.json \
  --no-post-simulation
```

To choose the optimized simulation input path explicitly:

```bash
python scripts/run_optimization.py --input-file inputs/iter_waveform_1d.json \
  --optimized-input-file inputs/my_optimized_case.json
```
