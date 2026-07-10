# TokaGrad

TokaGrad is a fast, differentiable integrated tokamak simulator built with
JAX. It couples reduced models for radial transport, heating and radiation,
fixed-boundary equilibrium geometry, pedestal evolution, density control, and
current diffusion. The same model is used for command-line simulation,
automatic-differentiation-based actuator optimization, and an interactive
Streamlit digital-twin interface.

TokaGrad is intended for rapid scenario exploration, control studies, and
reactor design optimization. It is not a replacement for validated integrated
modelling systems such as ASTRA, JINTRAC, RAPTOR, TORAX, or TRANSP, nor for a
full Grad-Shafranov or gyrokinetic solver.

## Main capabilities

- 1.5D radial evolution of electron temperature, ion temperature, electron
  density, and a poloidal-flux/current state.
- Fast 0.5D energy-balance mode with reconstructed radial profiles.
- Static-actuator and time-dependent waveform simulations.
- Semi-implicit finite-volume transport on cylindrical or flux geometry.
- Reduced fixed-boundary Miller geometry or prescribed G-EQDSK surfaces.
- JAX-native TGLF-NN, NEO-NN, and EPED1-NN inference from bundled BrainFUSE
  model files. QLKNN from fusion_surrogates also supported.
- Auxiliary, Ohmic, and alpha heating; electron-ion exchange; bremsstrahlung,
  line-radiation, and synchrotron losses.
- Sauter bootstrap current, current drive, neoclassical resistivity, and
  poloidal-flux diffusion.
- L-H threshold gating and tanh pedestal underlay/blending.
- Reverse-mode AD optimization of scalar controls and waveform control points.
- Interactive live simulation, sensitivity analysis, and scalar optimization.

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install -e .
```

This installs the core simulator, Streamlit UI, Optax optimization backend, and
the `fusion_surrogates` QLKNN adapter dependencies.

JAX-native TGLF-NN, NEO-NN, and EPED1-NN modes expect the GA `neural`
repository to be available under `external_models/neural`. If it is not already
present, clone it into the project-local `external_models/` directory:

```bash
mkdir -p external_models
git clone https://github.com/gafusion/neural.git external_models/neural
```

The default input files use this path through settings such as
`tglfnn_model_dir`, `neonn_model_dir`, and `eped1nn_model_dir`.
Standard `pyproject.toml` dependencies cannot clone a non-package model-asset
repository into a project-local path during `pip install`; keep this clone step
separate, or vendor the repository as a Git submodule if you want it tracked
with the source tree.

## Running simulations

The unified entry point is `scripts/run_simulation.py`. The JSON input selects
the dimensionality and whether the run uses fixed actuators or waveforms.

```bash
# Fixed-actuator 1.5D simulation
python scripts/run_simulation.py \
  --input-file inputs/iter_flattop_tglfnn.json

# Time-dependent 1.5D simulation
python scripts/run_simulation.py \
  --input-file inputs/iter_waveform_transition.json

# Fast 0.5D simulation
python scripts/run_simulation.py \
  --input-file inputs/iter_flattop_0d.json
```
<p align="center">
  <img src="https://github.com/jaem-seo/tokagrad/blob/main/images/_run_simulation.png">
</p>

Useful command-line overrides include:

```bash
python scripts/run_simulation.py \
  --input-file inputs/iter_flattop_tglfnn.json \
  --nr 32 --ntheta 32 --dt 1e-3 --end-time 10 \
  --save-results \
  --save-result-times "0.0, 5.0, 10.0" \
  --no-show
```

Use `--save-results`, `--save-result-times`, and `--save-result-file` to save
selected physical-time slices to a compressed NumPy file.
Add `--plot-psi` or `--plot-diffusivity` to include 
the poloidal-flux or diffusivity profile in the interactive plot.
On remote X11 servers where Matplotlib sliders fail because XInput 2 is not
available, use `--simple-final-plot` to draw only a static final-state summary
without slider/button widgets:

```bash
python scripts/run_simulation.py \
  --input-file inputs/iter_flattop_tglfnn.json \
  --simple-final-plot
```

## Optimization

The unified optimization entry point is `scripts/run_optimization.py`.

```bash
# Uses inputs/iter_flattop_tglfnn.json by default
python scripts/run_optimization.py

# Explicit scalar-control case
python scripts/run_optimization.py \
  --input-file inputs/iter_flattop_tglfnn.json

# Waveform-control optimization
python scripts/run_optimization.py \
  --input-file inputs/iter_waveform_transition.json

# Reactor design optimization
python scripts/run_optimization.py \
  --input-file inputs/reactor_design.json
```

Scalar controls can include `Ip_MA`, `Bt`, `P_aux_MW`,
`greenwald_fraction_target`, `R0`, `a`, `kappa`, and `delta`. Objectives can
combine fusion gain `Q` and fusion power `Pfus`. Constraints are differentiable
quadratic penalties on quantities such as `q0`, `q95`, `q_edge`, `beta_N`, and
the Greenwald fraction.

Control bounds are enforced through a smooth sigmoid parameterization. Scalar
optimization uses a JIT-compiled objective and `value_and_grad`; the machine and
actuator configurations are dynamic JAX PyTrees while `SimulationConfig`
remains static. The first iteration includes XLA compilation, while subsequent
iterations reuse the compiled executable.

Unless `--no-post-simulation` is supplied, optimization writes an
`optimized_*.json` input and runs a forward simulation of the optimized case.

## Streamlit application

```bash
streamlit run app_streamlit.py
```
<p align="center">
  <img src="https://github.com/jaem-seo/tokagrad/blob/main/images/fig2.png">
</p>

The application provides:

1. live 1.5D or 0.5D simulation with machine and actuator controls;
2. AD sensitivities of selected plasma metrics;
3. scalar AD optimization with configurable bounds and constraints.

Chunked live updates call importable JIT wrappers in
`src/tokagrad/app_runtime.py`, allowing compiled executables to survive normal
Streamlit reruns.

## Input-file structure

Inputs are JSON files with `machine`, `actuator`, and `simulation` sections.

```json
{
  "machine": {
    "R0": 6.2,
    "a": 2.0,
    "Bt": 5.3,
    "Ip_MA": 15.0,
    "kappa": 1.8,
    "delta": 0.33,
    "plasma_species": "DT"
  },
  "actuator": {
    "P_aux_MW": 50.0,
    "greenwald_fraction_target": 0.9,
    "heat_center": 0.2,
    "heat_width": 0.3
  },
  "simulation": {
    "simulation_model": "1.5d",
    "nr": 32,
    "ntheta": 32,
    "dt": 0.001,
    "end_time_s": 10.0,
    "diffusion_scheme": "semi_implicit",
    "equilibrium_model": "geqdsk_prescribed",
    "geqdsk_path": "ITERbaseline.eqdsk",
    "transport_model": "tglfnn_jax",
    "neoclassical_transport_model": "neonn_jax",
    "pedestal_model": "eped1_nn_jax",
    "current_evolution_model": "psi_diffusion"
  }
}
```

Adding a top-level `waveform` section selects waveform simulation. Controls use
piecewise-linear interpolation by default; `P_aux_MW` defaults to zero-order
hold. See `inputs/README.md` and the waveform examples for the complete syntax.

Adding an `optimization` section selects objectives, optimized controls,
bounds, constraints, optimizer settings, endpoint constraints, and waveform
regularization.

## Integrated model flow

The 1.5D state is

```text
PlasmaState = (Te[rho], Ti[rho], ne20[rho], psi_ind, psi_edge,
               Phi_b_prev, dV_drho_prev[rho])
```

The previous toroidal flux and volume-derivative profile are carried so the
particle and heat equations can discretize the conservative transients
`V'*n` and `V'^(5/3)*n*T` across time-dependent geometry.

At each transport step the solver:

1. constructs or refreshes the fixed-boundary equilibrium and flux metrics;
2. evaluates heating, radiation, particle, and pedestal sources;
3. evaluates turbulent and neoclassical transport coefficients;
4. advances temperature and density with the configured diffusion scheme;
5. applies the selected density and pedestal enforcement models;
6. advances the poloidal-flux/current state;
7. derives current components, q, fusion power, beta, confinement, and other
   diagnostics.

Expensive equilibrium, source, transport, and pedestal calculations can be
cached for several transport steps using the corresponding `*_skip_steps`
settings.

## Physics and numerical models

### Transport

The supported mainline form is diffusive transport. Available turbulent
closures include `bohm_gyrobohm`, `fusion_surrogates`, and `tglfnn_jax`.
Neoclassical transport can use the reduced Angioni-Sauter implementation,
the positive scalar Chang-Hinton analytic fit, or `neonn_jax`.
Surrogate diffusivities use `chi_clip_min/max`; the analytic
Bohm--gyroBohm closure uses its `bohm_chi_min/max` bounds.
The `bohm_gyrobohm` closure follows the TORAX analytic form: its Bohm term is
proportional to `r_mid*q^2*abs(grad(p_e))/(B0*n_e)`, its gyro-Bohm term to
`sqrt(T_e)*abs(grad(T_e))/B0^2`, and its particle diffusivity uses the harmonic
`chi_e*chi_i/(chi_e+chi_i)` form. The four species/scale coefficients and the
core/edge particle weights are configured by the `bohm_chi_*_coeff` and
`bohm_particle_c1/c2` fields in `SimulationConfig`.

### Equilibrium

`reduced_fixed_boundary` builds Miller/moment surfaces with an analytic
Shafranov-shift closure. `geqdsk_prescribed` reads the bundled or user-supplied
G-EQDSK boundary and nested surfaces. Both modes are fast geometry closures;
neither solves a free-boundary Grad-Shafranov problem.
The reduced geometry metrics retain the calibrated Miller/moment closure used
by the transport and current-diffusion operators. When q is reconstructed from
evolving current, it uses the circular expression multiplied by the ITER
elongation/triangularity/aspect-ratio shape factor. Prescribed G-EQDSK q remains
the source of truth when selected.
The initial total-current profile is selected by
`initial_current_profile_model`: `saturated_components` (default) constructs
the saturated Ohmic + bootstrap + driven-current split, while
`total_current_shape` normalizes `current.initial_current_shape()` directly to
the requested plasma current.

### Pedestal

The default JAX-native EPED1-NN ensemble predicts pedestal quantities, with an
alpha-critical reduced fallback. Martin or Delabie L-H threshold scalings gate
pedestal formation. `tanh_underlay`, `tanh_blend`, and `soft_source` select how
the pedestal target is coupled to the evolved profiles.

### Current

The default current model evolves a poloidal-flux state with neoclassical
resistive diffusion. The total current contains inductive/Ohmic, bootstrap, and
externally driven components. A fixed-Ip Neumann boundary is available, as is
an edge-flux/loop-voltage boundary. A saturated-conductivity model is provided
for the fast 0.5D path.

## Time-step convergence and current-profile precision

Always compare different `dt` values at the same physical end time. Setting
`end_time_s` lets the input loader derive `n_steps` and, by default, slightly
adjust `dt` so the requested end time is reached exactly.

The semi-implicit transport and current solvers are first order in time, so
finite differences between successive `dt` values are expected. No missing
`dt` factor is known in the current update. There are, however, three important
practical qualifications:

1. `equilibrium_skip_steps`, `source_skip_steps`, `transport_skip_steps`, and
   `pedestal_skip_steps` are step counts, not physical times. Changing `dt`
   therefore changes their physical refresh interval `skip_steps * dt`. For a
   clean convergence study set all four values to `1`, or rescale each skip
   count to keep that product constant.
2. Pedestal underlay/projection, density rescaling, hard bounds, and numerical
   limiters are algebraic per-step operations. If one of these is active, a dt
   scan also changes the operator-splitting frequency. Disable nonessential
   enforcement for a strict PDE convergence test.
3. JAX uses float32 by default. Current density is reconstructed from a second
   radial derivative of the poloidal-flux state. At very small `dt`, the
   per-step float32 change in `psi` approaches roundoff and the second
   derivative amplifies that quantization into visible cell-to-cell current
   jitter. For small-dt current-convergence studies, run in float64:

```bash
JAX_ENABLE_X64=1 python scripts/run_simulation.py \
  --input-file inputs/steady_full_diffusive.json
```

Reducing `dt` beyond the useful float32 range does not necessarily improve the
current profile. Increasing precision, checking radial-grid convergence, and
examining the enclosed-current or `psi` profile are more informative than
continuing to reduce `dt` alone.

## Repository layout

```text
src/tokagrad/
  config.py                 machine, actuator, and simulation configuration
  solver.py                 integrated 1.5D evolution
  zero_d.py                 fast 0.5D energy-balance evolution
  equilibrium.py            reduced and prescribed fixed-boundary geometry
  transport.py              turbulent + neoclassical transport dispatch
  heating.py                heating, exchange, fusion, and radiation
  current.py                current reconstruction and flux diffusion
  pedestal.py               pedestal targets, L-H gate, enforcement
  diagnostics.py            integrated plasma metrics
  controls.py               differentiable PyTree controls
  optim_differentiable.py   scalar and waveform optimization
  *_jax.py                  JAX-native neural surrogate inference

scripts/
  run_simulation.py         unified forward-simulation CLI
  run_optimization.py       unified optimization CLI
  check_*.py                surrogate and AD checks

inputs/                     example static, waveform, and optimization cases
external_models/            bundled public BrainFUSE neural model assets
app_streamlit.py            interactive application
```

## Model checks

```bash
python scripts/check_tglfnn_jax.py
python scripts/check_neonn_jax.py
python scripts/check_eped1nn_jax.py
python scripts/check_autodiff_gradients.py
```

The external `fusion_surrogates` adapter can be checked separately with:

```bash
python scripts/check_fusion_surrogates.py
```
