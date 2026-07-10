"""Configuration schema for TokaGrad physics and numerical model switches.

References are attached to the implementing modules rather than repeated for
every tuning parameter here.  In particular see ``current``, ``equilibrium``,
``heating``, ``neoclassical``, ``pedestal``, ``transport``, and ``zero_d``.
Parameters labelled proxy, multiplier, clip, blend, or relaxation are reduced
model/calibration choices unless an implementing docstring states otherwise.
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class MachineConfig:
    """Machine, equilibrium-shape, and plasma-composition parameters."""
    R0: float = 6.2
    # Major radius [m].
    a: float = 2.0
    # Minor radius [m].
    kappa: float = 1.8
    # Edge elongation used by the fast fixed-boundary geometry.
    delta: float = 0.33
    # Edge triangularity used by the fast fixed-boundary geometry.
    Bt: float = 5.3
    # Vacuum toroidal field on axis [T].
    Ip: float = 15.0e6
    # Prescribed total plasma current [A].
    Zeff: float = 1.8
    # Effective ion charge used by resistivity, collisionality, and radiation proxies.
    lnLambda: float = 17.0
    # Coulomb logarithm used by collisional exchange and resistivity.
    plasma_species: str = "DT"
    # Main ion species: "H", "D", "T", "He", or "DT".
    dt_fraction_D: float = 0.5
    # Deuterium fraction for plasma_species="DT".
    dt_fraction_T: float = 0.5
    # Tritium fraction for plasma_species="DT".
    impurity_Z: float = 74.0
    # Main impurity charge number used for dilution and radiation proxies.

@dataclass(frozen=True)
class ActuatorConfig:
    """External actuators and edge boundary values."""
    P_aux_MW: float = 50.0
    # Total auxiliary heating power [MW].
    f_e_heat: float = 0.6
    # Fraction of auxiliary heating deposited into the electron channel
    # when aux_partition_model="fixed".
    aux_partition_model: str = "slowing_down"
    # Auxiliary heat partition: "fixed" uses f_e_heat, "slowing_down" treats
    # P_aux as fast-ion/NBI power and splits it by classical slowing-down theory.
    nbi_birth_energy_MeV: float = 1.0
    # Birth energy for fast-ion/NBI auxiliary heating [MeV].
    nbi_fast_ion_A: float = 2.0
    # Fast auxiliary ion mass number; default is deuterium beam.
    nbi_fast_ion_Z: float = 1.0
    # Fast auxiliary ion charge number.
    heat_center: float = 0.2
    # Center of the Gaussian auxiliary-heating profile in rho.
    heat_width: float = 0.3
    # Width of the Gaussian auxiliary-heating profile in rho.
    greenwald_fraction_target: float = 0.9
    # Target Greenwald fraction f_G = <n_e>/n_G.
    greenwald_edge_density_fraction: float = 0.15
    # Edge density as a fraction of target volume-averaged Greenwald density.
    edge_Te_keV: float = 0.25
    # Fixed electron temperature boundary value at the edge [keV].
    edge_Ti_keV: float = 0.25
    # Fixed ion temperature boundary value at the edge [keV].
    cd_fraction: float = 0.0
    # Manual current-drive fraction of Ip; if <=0, use P_aux_MW * cd_efficiency_20.
    cd_efficiency_20: float = 0.3
    # Current-drive efficiency eta20 = n20 * I_CD[A] * R0[m] / P_aux[W].
    cd_fraction_max: float = 0.5
    # Maximum allowed automatically driven current fraction of Ip.
    cd_center: float = 0.2
    # Center of the Gaussian current-drive profile in rho.
    cd_width: float = 0.3
    # Width of the Gaussian current-drive profile in rho.

@dataclass(frozen=True)
class SimulationConfig:
    """Numerical settings and reduced-model switches."""
    nr: int = 32
    # Number of radial grid cells.
    radial_grid: str = "uniform"
    # Radial grid: "uniform" or "edge_cluster_sqrt".
    edge_cluster_power: float = 2.0
    # Power p in rho=sqrt(1-(1-x)^p) for edge_cluster_sqrt.
    dt: float = 1.0e-3
    # Time step [s] if dt_mode="fixed".  If dt_mode="cfl", this is overwritten 
    # at input-load time from cfl_number * min(dr)^2 / cfl_diffusivity_ref_m2_s.
    n_steps: int = 500
    # Number of time steps per forward simulation.  If end_time_s>0, this is
    # derived from end_time_s and the resolved dt.
    end_time_s: float = 0.0
    # Optional final simulation time [s].  If >0, n_steps is set from this value.
    dt_mode: str = "fixed"
    # Time-step mode: "fixed" uses dt, "cfl" derives dt from a diffusion CFL estimate.
    cfl_number: float = 0.25
    # Dimensionless CFL coefficient used when dt_mode="cfl".
    cfl_diffusivity_ref_m2_s: float = 1.0
    # Reference diffusivity D_ref [m^2/s] for dt ≈ CFL * dx^2 / D_ref before a run.
    adjust_dt_to_end_time: bool = True
    # If end_time_s>0, reduce dt slightly so n_steps*dt equals end_time_s exactly.
    save_every: int = 10
    # Reserved history downsampling cadence; current solver stores every step.
    save_results_enabled: bool = False
    # If true, scripts/run_simulation.py saves selected time slices to disk.
    save_result_times_s: tuple = ()
    # Requested physical times [s] to save.  The nearest available simulation frames are used.
    save_result_file: str = "outputs/tokagrad_results.npz"
    # Output filename for saved time slices.  Recommended format is compressed NumPy .npz.
    save_result_format: str = "npz"
    # Result format.  Currently "npz" is supported; metadata is written next to it as JSON.
    save_result_include_metadata_json: bool = True
    # If true, write a small human-readable .json file with keys, units, and selected times.
    plot_diffusivity: bool = False
    # If true, the simulation script plots diffusivity panel for debuging.
    plot_psi: bool = False
    # If true, scripts/run_simulation.py includes the poloidal-flux psi profile.

    # Expensive-submodel update cadences. A value of 1 recomputes every transport
    # time step (legacy behavior). A value N>1 recomputes at steps 0, N, 2N, ...
    # and reuses the previous value in between. These are intended for speed scans
    # with heavy equilibrium/source/transport/pedestal models.
    equilibrium_skip_steps: int = 1
    # Recompute fixed-boundary equilibrium / geometry every N transport steps.
    source_skip_steps: int = 1
    # Recompute local source terms, heating diagnostics, and density source every N steps.
    transport_skip_steps: int = 1
    # Recompute transport coefficients every N steps.
    pedestal_skip_steps: int = 1
    # Recompute L-H gate and pedestal target profiles every N steps. The cheap
    # source/projection enforcement is still applied every transport step using
    # the cached target to avoid pedestal erosion/oscillation between refreshes.

    # Numerical scheme for diffusion solver
    diffusion_scheme: str = "semi_implicit"
    # Mainline transport time stepping. Supported values:
    # "semi_implicit" (old coefficients, implicit linear diffusion),
    # "full_implicit"/"picard" (nonlinear Picard iterations),
    # "predictor_corrector" (predictor-corrector),
    # "newton"/"newton_raphson" (dense Newton solve for the nonlinear profile fixed point),
    # and "explicit" (legacy/experimental).
    implicit_nonlinear_iters: int = 3
    # Picard/fixed-point iterations for diffusion_scheme="full_implicit".
    implicit_relaxation: float = 1.0
    # Relaxation factor for full-implicit Picard profile updates. 1.0 means no damping.
    predictor_corrector_relaxation: float = 1.0
    # Blend between predictor and corrector for predictor-corrector stepping; 1.0 returns the corrector.
    newton_iters: int = 4
    # Newton iterations for diffusion_scheme="newton".
    newton_damping: float = 0.7
    # Newton update damping factor.
    newton_jacobian_regularization: float = 1.0e-6
    # Diagonal regularization added to the profile fixed-point Jacobian in Newton mode.
    density_evolution_model: str = "greenwald_feedback"
    # Density model: "diffusive", "fixed_initial", "reflective", "greenwald_feedback",
    # "greenwald_feedback_source", "greenwald_rescale_initial_shape", 
    # or "greenwald_rescale_tanh".
    # The rescale modes bypass particle diffusion and enforce the target Greenwald
    # fraction using either the initial density shape or a pedestal-width-aware
    # tanh+core density shape.
    freeze_temperature_profiles: bool = False
    # If true, Te and Ti are reset to their initial profiles after each step.
    differentiable_smooth_mode: bool = False
    # If true, replace selected hard clips/limiters by smooth approximations and
    # disable direct pedestal projection by default for AD/JVP experiments.
    smooth_clip_width: float = 1.0e-2
    # Transition width used by smooth clipping functions in physical units.
    smooth_rate_width: float = 1.0e-2
    # Transition width used by smooth per-step source/rate limiters.
    source_implicitness: float = 1.0
    # Local source treatment: 0 explicit, 1 semi-implicit sink-limited update.
    source_max_delta_keV: float = 2.5
    # Maximum temperature change per step from local source terms [keV].


    # Initial profile condition
    initial_beta_N_target: float = 1.8
    # Target total beta_N (thermal + reduced fast-ion beta) used to rescale
    # the ITER-like initial temperature profile across machines.
    initial_temperature_rescale_to_beta_N: bool = True
    # If true, rescale initial Te/Ti excess above edge to keep beta_N approximately fixed.
    initial_profile_quadrature_points: int = 32
    # Sub-cell quadrature points used to initialize coarse-grid profiles from smooth shapes.
    greenwald_tanh_core_factor: float = 1.10
    # Core density factor used by density_evolution_model="greenwald_rescale_tanh".
    # The profile is rescaled afterward to match the requested Greenwald fraction.
    greenwald_tanh_pedestal_factor: float = 1.00
    # Pedestal-top density factor used by greenwald_rescale_tanh before rescaling.
    greenwald_tanh_transition_fraction: float = 0.25
    # Pedestal transition width as a fraction of the computed pedestal width for
    # greenwald_rescale_tanh.  A floor of pedestal_transition_sharpness is applied.
    greenwald_tanh_default_width: float = 0.04
    # Fallback normalized pedestal width for greenwald_rescale_tanh when the
    # selected pedestal model is "none" or cannot provide a width.
    density_feedback_tau: float = 1.0e-3
    # Global particle feedback time [s] for greenwald_feedback mode.
    density_boundary_source_width: float = 0.1
    # Width in rho of edge particle source for greenwald_feedback mode.
    density_source_max_delta: float = 0.25
    # Maximum per-step density change [1e20 m^-3] from density source/control.
    initial_profile_model: str = "h_mode"
    # Initial condition model: "h_mode" or "parabolic".
    initial_current_profile_model: str = "saturated_components"
    # Initial total-current profile: "saturated_components" builds the
    # conductivity-saturated Ohmic + bootstrap + current-drive split;
    # "total_current_shape" normalizes current.initial_current_shape() to Ip
    # and uses it directly as the initial total current density.


    # 0.5D fast energy-balance mode
    simulation_model: str = "1.5d"
    # Simulation model: "1.5d" for radial diffusion or "0d_fast"/"0.5d" for
    # scalar energy evolution with reconstructed radial profiles.
    zero_d_H_factor: float = 1.0
    # Confinement enhancement factor multiplying IPB98(y,2) H-mode tau_E.
    zero_d_L_factor: float = 1.0
    # Confinement multiplier applied to ITER89-P L-mode tau_E in 0.5D L-H switching.
    zero_d_dt_fraction_tauE: float = 0.10
    # Recommended pseudo timestep fraction of the instantaneous tau_E.
    zero_d_tauE_min: float = 1.0e-3
    # Lower bound for tau_E [s].
    zero_d_tauE_max: float = 20.0
    # Upper bound for tau_E [s].
    zero_d_core_shape_power: float = 2.0
    # Exponent for 0D reconstructed core temperature excess shape.
    zero_d_temperature_max_keV: float = 80.0
    # Safety ceiling for reconstructed Te=Ti profiles.
    zero_d_alpha_birth_energy_MeV: float = 3.52
    # Alpha-particle birth energy used by the 0.5D fast-ion slowing-down proxy [MeV].
    zero_d_fast_nbi_fraction: float = 1.0
    # Fraction of auxiliary heating treated as fast-ion source power in the 0.5D fast-beta proxy.
    zero_d_use_lax_scan: bool = True
    # Use jax.lax.scan for 0.5D rollouts instead of a Python loop.
    # This removes per-step host dispatch overhead and makes 0D genuinely fast.
    zero_d_profile_reconstruction_fast_path: bool = True
    # In 0.5D reconstruction, reuse the previous state as the dummy profile
    # instead of rebuilding full 1.5D initial profiles every step.
    zero_d_reconstruct_current: bool = True
    # If false, keep the previous psi/current state during 0.5D reconstruction.
    # Useful for very fast energy/profile scans where q/current diagnostics are
    # not needed at every step.


    # Transport closure
    transport_mode: str = "diffusive"
    # Mainline transport equation form. "diffusive" uses chi/D and is the supported
    # default; "flux" is retained only for legacy/experimental tests.
    transport_model: str = "tglfnn_jax"
    # Transport closure: "bohm_gyrobohm", "fusion_surrogates", or "tglfnn_jax".
    transport_surrogate_scale: float = 1.0
    # Multiplicative scale for surrogate-derived diffusivities.
    transport_flux_scale: float = 1.0
    # Multiplicative scale for direct flux-divergence mode.
    transport_flux_clip: float = 50.0
    # Clip for normalized direct turbulent fluxes before divergence.
    transport_flux_max_delta_keV: float = 2.5
    # Per-step limiter for explicit transport flux temperature updates [keV].
    transport_flux_max_delta_ne20: float = 0.25
    # Per-step limiter for explicit transport flux density updates [1e20 m^-3].
    fusion_surrogates_fail_mode: str = "fallback"
    # If external model call fails: "fallback" to bohm_gyrobohm or "raise".
    chi_clip_min: float = 0.03
    # Lower bound for effective diffusivities [m^2/s].
    chi_clip_max: float = 30.0
    # Upper bound for effective diffusivities [m^2/s].
    qlknn_gb_to_chi: float = 1.0
    # Conversion scale from QLKNN gyroBohm-normalized fluxes to effective chi [m^2/s].
    gradient_floor: float = 0.5
    # Minimum denominator for converting fluxes to effective diffusivities.
    tglfnn_model_dir: str = "external_models/neural"
    # TGLFNN model location. Accepts the neural repo root, tglfnn model parent,
    # or the DIIID_ion_stiffness_60_rotation directory itself.
    tglfnn_model_name: str = "DIIID_ion_stiffness_60_rotation"
    # Public BrainFUSE TGLFNN model subdirectory name.
    tglfnn_fail_mode: str = "fallback"
    # If TGLFNN-JAX fails: "fallback" to bohm_gyrobohm or "raise".
    tglfnn_jax_max_nets: int = 10
    # Maximum TGLFNN ensemble members for JAX backend. 0 uses all nets.
    tglfnn_gb_to_chi: float = 1.0
    # Additional calibration multiplier applied after gyroBohm flux-to-chi conversion.
    tglfnn_use_gyrobohm_scale: bool = True
    # If true, convert normalized TGLFNN fluxes with chi_gB=rho_s^2 c_s/a before dividing by a/L gradients.
    tglfnn_clip_inputs_to_training: bool = True
    # Clip TGLFNN input features to a broad mean +/- N*sigma envelope from the BrainFUSE model.
    tglfnn_training_clip_sigma: float = 5.0
    # Number of training standard deviations used by the TGLFNN input clipping envelope.
    tglfnn_vpar_1: float = 0.0
    # Optional normalized parallel velocity proxy VPAR_1.
    tglfnn_vpar_shear_1: float = 0.0
    # Optional normalized parallel-velocity shear proxy VPAR_SHEAR_1.
    tglfnn_vexb_shear: float = 0.0
    # Optional normalized ExB shear proxy VEXB_SHEAR.

    # Simple empirical transport parameters
    chi_e_base: float = 0.6
    # Baseline empirical electron heat diffusivity [m^2/s].
    chi_i_base: float = 0.45
    # Baseline empirical ion heat diffusivity [m^2/s].
    Dn_base: float = 0.15
    # Baseline empirical particle diffusivity [m^2/s].
    chi_edge_mult: float = 4.0
    # Edge multiplier for empirical diffusivity, proportional to rho^2.
    stiffness: float = 1.5
    # Strength of critical-gradient-like empirical transport stiffness.
    grad_crit_keV_per_m: float = 3.0
    # Temperature-gradient threshold for empirical stiffness [keV/m].
    bohm_chi_e_bohm_coeff: float = 8.0e-5
    # Coefficient alpha_e,B multiplying the electron Bohm scale.
    bohm_chi_e_gyrobohm_coeff: float = 5.0e-6
    # Coefficient alpha_e,gB multiplying the electron gyroBohm scale.
    bohm_chi_i_bohm_coeff: float = 8.0e-5
    # Coefficient alpha_i,B multiplying the ion Bohm scale.
    bohm_chi_i_gyrobohm_coeff: float = 5.0e-6
    # Coefficient alpha_i,gB multiplying the ion gyroBohm scale.
    bohm_particle_c1: float = 1.0
    # Core weighting in D_e=eta*chi_e*chi_i/(chi_e+chi_i).
    bohm_particle_c2: float = 0.3
    # Edge weighting in D_e=eta*chi_e*chi_i/(chi_e+chi_i).
    bohm_chi_min: float = 0.05
    # Lower bound for Bohm-gyroBohm heat diffusivities [m^2/s].
    bohm_chi_max: float = 100.0
    # Upper bound for Bohm-gyroBohm heat diffusivities [m^2/s].
    bohm_particle_min: float = 0.05
    # Lower bound for Bohm-gyroBohm electron particle diffusivity [m^2/s].
    bohm_particle_max: float = 100.0
    # Upper bound for Bohm-gyroBohm electron particle diffusivity [m^2/s].

    # Fast fixed-boundary equilibrium / geometry
    ntheta: int = 16
    # Number of poloidal grid points for flux-surface geometry.  The default is
    # intentionally small for interactive runs; with the default cardinal theta
    # grid it includes outboard, top, inboard, and bottom Miller points.
    theta_grid_alignment: str = "auto"
    # Poloidal grid alignment: "auto", "cardinal", or "lower_xpoint".
    # "auto" uses cardinal uniform points for fixed-boundary Miller geometry and
    # a uniformly spaced grid shifted to include the lower X-point/min-Z boundary
    # point for prescribed GEQDSK geometry, preserving uniform theta spacing.
    shafranov_shift_max: float = 0.2
    # Maximum allowed analytic Shafranov shift Delta0/a.
    triangularity_profile_power: float = 1.0
    # Power-law exponent for triangularity ramp from axis to edge.
    elongation_profile_power: float = 1.0
    # Power-law exponent for elongation ramp from axis to edge.

    # Flux-coordinate transport geometry
    transport_geometry: str = "flux"
    # Fixed to flux-coordinate finite-volume transport. Legacy cylindrical mode removed.
    phi_dot_over_phi: float = 0.0
    # Optional manual normalized toroidal-flux-boundary rate Phi_b_dot/Phi_b [1/s].
    # Added to the automatically inferred value when auto_phi_dot_over_phi=True.
    auto_phi_dot_over_phi: bool = True
    # If true, infer Phi_b_dot/Phi_b from changes in reduced Phi_b between steps.
    phi_dot_over_phi_clip: float = 5.0
    # Clip |Phi_b_dot/Phi_b| for numerical robustness in interactive shape changes.
    reduced_geometry_metrics: bool = True
    # If true, geometry factors g0...g3 are obtained via simplified assumptions.
    torax_circular_psi_geometry: bool = False
    # Match TORAX's ad-hoc circular-geometry definitions of toroidal flux and
    # g2g3_over_rhon in the psi/current equation. Intended for controlled TORAX
    # benchmarks; leave false for the ordinary Miller/GEQDSK convention.
    # If False, they are obtained via flux-surface average.
    heat_convection_model: str = "none"
    # Heat-convection q_conv model: "none" or "constant".
    heat_convection_e_base: float = 0.0
    # Constant electron heat-convection coefficient used when heat_convection_model="constant".
    heat_convection_i_base: float = 0.0
    # Constant ion heat-convection coefficient used when heat_convection_model="constant".
    particle_convection_model: str = "none"
    # Particle convection/pinch V_e model: "none" or "constant".
    particle_convection_base: float = 0.0
    # Constant particle-convection coefficient used when particle_convection_model="constant".
    equilibrium_model: str = "reduced_fixed_boundary"
    # Equilibrium source: "reduced_fixed_boundary" or "geqdsk_prescribed".
    geqdsk_path: str = ""
    # Path to G-EQDSK/g-file used when equilibrium_model="geqdsk_prescribed".
    geqdsk_use_pressure: bool = False
    # If true, use pressure profile from G-EQDSK; default uses evolving kinetic pressure.
    geqdsk_q_profile_source: str = "eqdsk"
    # q profile used in G-EQDSK geometry mode: "eqdsk" keeps the prescribed
    # qpsi interpolation from the g-file, while "current" reconstructs q from
    # the evolving total current density / psi using the extracted flux-surface
    # geometry, analogous to the reduced fixed-boundary equilibrium.
    geqdsk_flux_coordinate: str = "sqrt_psi"
    # Radial coordinate mapping for G-EQDSK: currently "sqrt_psi".
    geqdsk_surface_geometry: str = "psirz_contours"
    # G-EQDSK flux-surface geometry: "psirz_contours" extracts nested surfaces
    # directly from psirz; "axis_to_boundary" retains the old straight-line
    # interpolation from magnetic axis to LCFS.
    geqdsk_inner_contour_fallback: str = "axis_hessian"
    # Fallback for very small flux levels where contour extraction is under-resolved:
    # "axis_hessian", "nearest_contour", or "axis_to_boundary".
    geqdsk_lcfs_margin: float = 0.0
    # In psirz_contours mode, optionally evaluate all requested cell-centered
    # contour levels at (1 - margin)*rho.  This does not remap the outermost
    # cell center to the LCFS; the true LCFS is plotted from the G-EQDSK boundary
    # separately, while volume/surface integrals remain cell-centered and avoid
    # double-counting the edge shell at small nr.


    # Pedestal model
    pedestal_model: str = "eped1_nn_jax"
    # Pedestal closure: "alpha_critical", "eped1_nn_jax", or "none".
    # "eped1_nn_jax" parses GA BrainFUSE/FANN brainfuse_*.net files and
    # evaluates the ensemble as a differentiable JAX MLP.
    eped1nn_model_dir: str = "external_models/neural"
    # Local path containing cloned gafusion/neural repo for optional EPED1-NN.
    eped1nn_model_name: str = "EPED1_H_superH"
    # GA EPED1-NN model subfolder/name to use when available.
    eped1nn_fail_mode: str = "fallback"
    # If EPED1-NN call fails: "fallback" to alpha_critical or "raise".
    eped1nn_output_scale: float = 1.0
    # Multiplicative scale applied to EPED1-NN predicted pedestal pressure.
    eped1nn_jax_max_nets: int = 1
    # Maximum number of brainfuse_*.net ensemble members used by the JAX
    # EPED1-NN backend. 0 means use all files found in the model directory.
    pedestal_kbm_width_coeff: float = 0.076
    # KBM width coefficient for width estimation in alpha-critical model.
    pedestal_width_min: float = 0.025
    # Minimum pedestal width in normalized rho/psi proxy.
    pedestal_width_max: float = 0.15
    # Maximum pedestal width in normalized rho/psi proxy.
    pedestal_alpha_scale: float = 1.0
    # Multiplicative factor for alpha-critical pedestal model.
    pedestal_alpha_min: float = 1.0
    # Lower bound for alpha_critical.
    pedestal_alpha_max: float = 4.0
    # Upper bound for alpha_critical.
    pedestal_pressure_relax_tau: float = 0.005
    # Relaxation time toward pedestal pressure/temperature targets [s].
    pedestal_density_height20: float = 0.25
    # Nominal density pedestal height [1e20 m^-3].
    pedestal_te_fraction: float = 0.5
    # Fraction of pedestal pressure assigned to electron temperature channel.
    pedestal_transition_sharpness: float = 0.012
    # Sharpness of tanh pedestal transition in rho.
    pedestal_enforcement: str = "tanh_underlay"
    # Pedestal enforcement mode: "tanh_blend" bypasses stiff pedestal source terms
    # and smoothly blends only the pedestal/edge region toward the target tanh pedestal;
    # "tanh_underlay" imposes the target tanh pedestal as a smooth whole-profile
    # floor/underlay so the core profile can sit on top without a pedestal-top dip;
    # "soft_source" keeps the legacy relaxation-source/projection path.
    pedestal_blend_fraction: float = 1.0
    # Fraction of the smooth tanh pedestal correction applied per step for
    # tanh_blend enforcement. 1.0 is an imposed pedestal shape.
    pedestal_blend_core_width_factor: float = 0.5
    # Blend/source mask begins this many smoothing widths inside rho_top.
    pedestal_blend_mask_sharpness: float = 0.02
    # Smoothness of the tanh blend mask in rho; <=0 uses pedestal_transition_sharpness.
    pedestal_blend_include_density: bool = True
    # If true, tanh_blend also blends ne toward the pedestal target.
    pedestal_blend_max_delta_keV: float = 2.0
    # Optional per-step limiter for direct tanh_blend temperature correction; <=0 disables.
    pedestal_blend_max_delta_ne20: float = 0.1
    # Optional per-step limiter for direct tanh_blend density correction; <=0 disables.
    pedestal_underlay_smooth_keV: float = 0.05
    # Smooth-max width [keV] for tanh_underlay temperature floors.
    pedestal_underlay_smooth_ne20: float = 0.005
    # Smooth-max width [1e20 m^-3] for tanh_underlay density floor.

    pedestal_alpha_resolution_guard: bool = True
    # If true, smoothly damps the discrete alpha-target calibration when the
    # pedestal width is under-resolved on a coarse radial grid. Enabled by
    # default because it prevents large coarse-grid pedestal overshoots; disable
    # for high-resolution studies that need the raw alpha calibration.
    pedestal_alpha_calibration_min_width_cells: float = 0.75
    # Pedestal width, in radial-cell units, below which the alpha calibration is
    # mostly disabled when pedestal_alpha_resolution_guard=True.
    pedestal_alpha_calibration_width_cells: float = 0.25
    # Smooth transition width, in cell units, for the unresolved-pedestal guard.
    pedestal_alpha_discrete_scale_max: float = 5.0
    # Maximum discrete alpha calibration multiplier when the guard is enabled.

    pedestal_alpha_tracking: bool = True
    # If true, pedestal source rescales target pressure so actual alpha approaches alpha_crit.
    pedestal_alpha_tracking_gain: float = 1.5
    # Gain for pressure-height correction from alpha_actual toward alpha_crit.
    pedestal_source_max_delta_keV: float = 0.5
    # Maximum per-step pedestal temperature correction [keV] before global source limiting.
    pedestal_min_temperature_keV: float = 0.25
    # Floor used when converting pedestal pressure target to Te/Ti targets.
    pedestal_projection_fraction: float = 0.8
    # Per-step direct blending toward alpha-critical pedestal target after transport solve.
    pedestal_lh_threshold_model: str = "delabie"
    # Threshold power scaling: "martin" for ITPA08, "delabie" for ITPA26, or "none" for no control.
    # If not "none", switch pedestal source/projection on only when P_sep exceeds L-H threshold.
    pedestal_lh_power_basis: str = "net_separatrix"
    # Power basis: "net_separatrix" = Paux+Pohm+Palpha-Prad, "absorbed_heating" = Paux+Pohm+Palpha.
    pedestal_lh_threshold_scale: float = 1.0
    # Multiplier applied to Martin L-H threshold.
    pedestal_lh_margin: float = 0.01
    # Smooth transition margin as fraction of P_LH.
    pedestal_lh_min_gate: float = 0.0
    # Residual raw L-H gate below threshold.  This value is reported in diagnostics;
    # direct pedestal enforcement may apply an additional cutoff/target-strength rule.
    pedestal_lh_gate_cutoff: float = 1.0e-2
    # Raw L-H gate values below this are treated as no-pedestal for direct
    # tanh_blend/tanh_underlay enforcement.  Set to 0 for a fully continuous tail.
    martin_lh_coeff: float = 0.0488
    # Martin/ITPA08 P_LH coefficient, P[MW] = C n20^0.717 Bt^0.803 S^0.941.
    martin_lh_n_exp: float = 0.717
    # Density exponent in Martin L-H threshold scaling.
    martin_lh_bt_exp: float = 0.803
    # Toroidal-field exponent in Martin L-H threshold scaling.
    martin_lh_s_exp: float = 0.941
    # Plasma surface-area exponent in Martin L-H threshold scaling.
    delabie_lh_coeff: float = 0.0441
    # Delabie/ITPA26 P_LH coefficient, P[MW] = C n20^1.08 B_T^0.580 (2/M_eff^0.975) D S.
    delabie_lh_n_exp: float = 1.08
    # Density exponent in Delabie L-H threshold scaling.
    delabie_lh_bt_exp: float = 0.580
    # Toroidal-field exponent in Delabie L-H threshold scaling.
    delabie_lh_meff_exp: float = 0.975
    # M_eff exponent in Delabie L-H threshold scaling.
    delabie_lh_d_exp: float = 1.0
    # Divertor configuration parameter in Delabie L-H threshold scaling (1 for HT-like and 1.93 for VT-like).
    delabie_lh_s_exp: float = 1.0
    # Plasma surface-area exponent in Delabie L-H threshold scaling.


    # Poloidal-flux / current diffusion parameters
    current_evolution_model: str = "psi_diffusion"
    # Current evolution: "saturated_conductivity" or "psi_diffusion".
    # Legacy alias "new_psi_diffusion" is still accepted by the code.
    resistivity_multiplier: float = 1.0
    # Multiplier on eta_neo/mu0 in the psi diffusion equation.
    saturated_current_conductivity_power: float = 1.0
    # j_ind shape proportional to sigma_neo**power in saturated-current mode.
    saturated_current_sigma_floor: float = 0.02
    # Floor as fraction of mean conductivity to avoid exactly zero edge current.
    saturated_current_smooth: float = 0.15
    # Optional radial smoothing strength for saturated-current profile.
    psi_solver_form: str = "face_fv"
    # Poloidal-flux solver: "cell_centered" or "face_fv".  face_fv is more
    # conservative but can be more sensitive on coarse grids.
    psi_state_grid: str = "face"
    # Grid location of psi_ind state: "cell" uses nr cell centers; "face" uses nr+1 faces.
    new_psi_boundary_model: str = "fixed_ip_neumann"
    # Boundary for current_evolution_model="psi_diffusion": "fixed_ip_neumann"
    # imposes the LCFS flux-gradient corresponding to Ip;
    # "edge_psi_dirichlet" uses an explicit edge-psi/loop-voltage boundary.
    current_source_smoothing_passes: int = 1
    # Light binomial smoothing passes applied to the explicit non-inductive
    # source in the psi diffusion equation. This damps grid-scale bootstrap/CD
    # source noise without changing the fixed-Ip boundary condition. Set to 0
    # for strict equation comparisons.
    current_enclosed_smoothing_passes: int = 0
    # Optional post-step smoothing passes on the enclosed-current face profile
    # for psi_diffusion. Endpoints I(0)=0 and I(1)=Ip are preserved, so this
    # is a conservative numerical roughness filter; use 0 for strict physics.
    current_axis_regularization_faces: int = 2
    # Replace the first N inner enclosed-current faces by the smooth-axis
    # expansion I(rho)=a*rho**2+b*rho**4 inferred from neighboring faces. This
    # removes a face-grid axis artifact in saturated psi_diffusion runs while
    # preserving I(0)=0 and I(LCFS)=Ip. Set 0 for strict unfiltered comparison.
    current_diagnostic_smoothing_passes: int = 1
    # Conservative smoothing used only when reconstructing current components
    # from the psi state.  The psi state is visually smooth, but the
    # current density is a second finite-volume derivative of psi; one light
    # pass on the enclosed-current profile removes grid-scale reconstruction
    # noise while preserving I(0)=0 and I(LCFS)=Ip.
    psi_edge_robin_length_m: float = 0.01
    # Extra penetration length [m] in edge Robin condition to avoid one-cell current sheets.
    

    # Neoclassical model
    neoclassical_transport_model: str = "neonn_jax"
    # Neoclassical diffusivity contribution: "angioni", "neonn_jax", or "none".
    neonn_model_dir: str = "external_models/neural"
    # NEOjbs-NN model location. Accepts neural root or neonn/jbsnn directory.
    neonn_model_name: str = "jbsnn"
    # Public BrainFUSE NEOjbs-NN model directory.
    neonn_fail_mode: str = "fallback"
    # If NEOjbs-NN-JAX fails: "fallback" to Angioni or "raise".
    neonn_jax_max_nets: int = 1
    # Maximum NEOjbs-NN ensemble members for JAX backend. 0 uses all nets.
    neonn_transport_scale: float = 1.0
    # Calibration scale applied to the reduced NEOjbs-NN transport mapping.
    neoclassical_chi_scale: float = 1.0
    # Multiplier for neoclassical heat diffusivities.
    neoclassical_D_scale: float = 1.0
    # Multiplier for reduced neoclassical particle diffusivity.
    neoclassical_chi_max: float = 5.0
    # Maximum neoclassical diffusivity contribution [m^2/s].
    neoclassical_shaing_ion_mode: str = "off"
    # Optional Shaing near-axis ion heat transport correction. Options:
    # "off"/"none", "blend", "add" (localized additive),
    # "add_full" (global additive), or "replace"/"shaing". Applies only to chi_i.
    neoclassical_shaing_ion_multiplier: float = 1.8
    # Multiplicative calibration applied to the Shaing ion diffusivity.
    neoclassical_shaing_blend_start: float = 0.2
    # Rho location where blend mode is half Shaing and half Angioni-Sauter.
    neoclassical_shaing_blend_rate: float = 5.0
    # Steepness of the Shaing/Angioni transition in rho for blend/add modes.

    # Edge loop-voltage feedback represented as edge-generated inductive current.
    current_feedback_model: str = "psi_boundary_loop_voltage"
    # Current feedback model: "psi_boundary_loop_voltage" or legacy "edge_current_source".
    # Not used when new_psi_boundary_model="fixed_ip_neumann".
    loop_voltage_gain: float = 0.4
    # Proportional loop-voltage feedback gain [V/MA].
    loop_voltage_max: float = 4.0
    # Maximum absolute loop voltage [V].

    # Bootstrap-current model
    bootstrap_model: str = "sauter"
    # Bootstrap-current closure. "sauter" uses the Sauter 1999 analytic fit.
    # "neonn_jax" attempts to use a NEO/BrainFUSE model output if the loaded
    # model exposes a recognized bootstrap-current output; otherwise the action
    # follows neonn_bootstrap_fail_mode.
    neonn_bootstrap_fail_mode: str = "fallback"
    # If bootstrap_model="neonn_jax" and no bootstrap output is available:
    # "fallback" uses Sauter, "raise" raises an error.
    neonn_bootstrap_output_name: str = ""
    # Optional explicit NEO/BrainFUSE output name for bootstrap current. Empty
    # means auto-detect names containing jbs/jboot/bootstrap.
    neonn_bootstrap_output_units: str = "auto"
    # Output units for a detected bootstrap output: "auto", "A_m2", or
    # "MA_m2". Unknown dimensionless coefficient outputs intentionally fall
    # back unless the units are specified.
    neonn_bootstrap_scale: float = 1.0
    # Extra multiplier applied to a NEO/NN bootstrap-current prediction.
    bootstrap_multiplier: float = 1.0
    # Calibration factor for reduced bootstrap closure.
    bootstrap_btheta_floor: float = 0.08
    # Floor for |Btheta| [T] to avoid singular bootstrap-current estimates.

    # Heating and loss physics
    include_ohmic_heating: bool = True
    # Include eta_neo * j_ind^2 electron heating.
    include_ei_exchange: bool = True
    # Include collisional electron-ion heat exchange.
    include_radiation_losses: bool = True
    # Include bremsstrahlung, line radiation, and synchrotron electron losses.
    include_alpha_heating: bool = True
    # Include D-T alpha heating when plasma_species="DT".
    bremsstrahlung_scale: float = 1.0
    # Multiplicative scale for reduced bremsstrahlung loss.
    line_radiation_scale: float = 1.0
    # Multiplicative scale for reduced impurity line-radiation loss.
    line_radiation_Lz_ref_W_m3: float = 3.0e-35
    # Reference coronal line-cooling coefficient [W m^3] for a C-like impurity around the low-keV edge.
    line_radiation_Tcut_keV: float = 1.5
    # Temperature scale [keV] for the reduced line-radiation cooling window.
    line_radiation_Z_scaling_power: float = 0.5
    # Weak reduced scaling of the cooling coefficient with impurity charge relative to carbon.
    line_radiation_edge_power: float = 3.0
    # Optional edge localization factor rho**power for the reduced line-radiation proxy.
    synchrotron_scale: float = 1.0
    # Multiplicative scale for reduced synchrotron loss.
    synchrotron_wall_reflectivity: float = 0.8
    # Wall reflectivity for synchrotron radiation; only (1-R) is lost.
    synchrotron_escape_fraction: float = 0.05
    # Reduced optical escape/self-absorption factor multiplying optically thin synchrotron emission.
    alpha_heating_scale: float = 1.0
    # Multiplicative scale for D-T alpha heating.
    alpha_partition_model: str = "slowing_down"
    # Alpha electron/ion partition: "mikkelsen" reproduces TORAX, while
    # "slowing_down" uses the composition-dependent critical-energy integral;
    # "fixed" uses the fractions below.
    alpha_electron_fraction_fixed: float = 0.85
    # Fixed electron fraction used only when alpha_partition_model="fixed".
    alpha_ion_fraction_fixed: float = 0.15
    # Fixed ion fraction used only when alpha_partition_model="fixed".
    ei_exchange_multiplier: float = 1.0
    # Scale factor for electron-ion collisional exchange.
