# Optimization capability matrix

Each entry below is an executable scientific acceptance program. The task axis
is the seven-class taxonomy. The response axis separates stable equilibria from
adjusted seasonal cycles, giving 35 cells. Nine additional experiments cover
focused scientific cases and infrastructure.

The [implementation requirements](../../doc/design/optimization-capabilities.md)
describe the numerical features and validation needed to complete the suite.
The [implementation plan](../../doc/design/optimization-implementation-plan.md)
sets out their delivery order and acceptance gates.
The [AMOC resilience proposal](../../doc/design/amoc-resilience-research.md)
provides a scientific target that combines several matrix cells.

| Scientific task | Local / one step | Finite trajectory | Stable equilibrium | Seasonal cycle | Climate statistics |
| --- | --- | --- | --- | --- | --- |
| Sensitivity and mechanisms | `sensitivity_local` | `sensitivity_trajectory` | `sensitivity_equilibrium` | `sensitivity_seasonal` | `sensitivity_statistics` |
| State and forcing estimation | `estimation_local` | `estimation_trajectory` | `estimation_equilibrium` | `estimation_seasonal` | `estimation_statistics` |
| Physical parameter calibration | `calibration_local` | `calibration_trajectory` | `calibration_equilibrium` | `calibration_seasonal` | `calibration_statistics` |
| Optimal perturbations | `perturbation_local` | `perturbation_trajectory` | `perturbation_equilibrium` | `perturbation_seasonal` | `perturbation_statistics` |
| Learning parameterizations | `closure_local` | `closure_trajectory` | `closure_equilibrium` | `closure_seasonal` | `closure_statistics` |
| Identifiability and observation design | `observation_local` | `observation_trajectory` | `observation_equilibrium` | `observation_seasonal` | `observation_statistics` |
| Geometry and discretization | `geometry_local` | `geometry_trajectory` | `geometry_equilibrium` | `geometry_seasonal` | `geometry_statistics` |

Export the matrix, research questions, and required extension contracts with:

```sh
python -m benchmarks.ad_demos --matrix
python -m benchmarks.ad_demos --demo matrix --response local --mode smoke
python -m benchmarks.ad_demos --demo matrix --task geometry --mode smoke
VEROS_ASSET_DIR=/path/to/assets python -m benchmarks.ad_demos --demo matrix \
  --setup global_4deg --mode acceptance --output /tmp/ad-matrix-acceptance
```

`--task` and `--response` select matrix cases. `--demo all` includes the focused
fixtures. Acceptance requires the global configuration for these scientific
cases. Small ACC runs exercise their numerical workflow. The opt-in pytest lane
contains all 44 demos and treats every unmet requirement as an ordinary failure.

## Scientific questions and experiments

The machine-readable registry in `matrix.py` assigns stable question IDs to
cases. A unit test requires every question and every matrix cell to have a
registered executable program.

| Question | Experiment and acceptance evidence |
| --- | --- |
| Regional heat sensitivity and Southern Ocean wind response | Heat, salinity, and wind controls are differentiated against North Atlantic heat content, heat transport, ACC transport, and depth-coordinate overturning. The finite-trajectory case checks 1, 7, 30, 90, and 360 days and a withheld starting season. Climate cases differentiate means, variances, and smooth exceedance frequencies. |
| Initial stratification versus surface forcing | Joint temperature/salinity initial corrections and heat/wind/salinity forcing are estimated from northern profiles and transports. Southern profiles, a changed starting season, and a longer rollout are withheld. Recovery is required in the numerically resolved observation-Jacobian subspace, and null directions are reported. |
| Seasonal forcing reconstruction | Adjusted-cycle and annual-trajectory inverse problems fit mean and annual-harmonic forcing corrections. Equilibrium estimation infers steady forcing. Climate estimation uses distributional observations. Adjusted-state cases exclude transient initial fields as inferential controls. |
| Isopycnal mixing, thermocline structure, and transport | Joint log-scaled isopycnal/vertical mixing and wind controls fit tracer profiles and heat/overturning transport. Held-out diagnostics must improve, and no held-out transport diagnostic may deteriorate. |
| Wind errors versus mixing errors | The joint calibration and observation-design cases expose both control types. Singular values identify resolved combinations. Observation selection targets future-transport variance. |
| Budget-balanced heat redistribution | Projected forcing optimization maximizes regional heat response under a unit control norm. Prescribed and applied heat/salinity budgets are checked. Local and trajectory cases additionally optimize inventory-balanced initial perturbations. |
| Minimum-cost overturning reduction within ten years | `perturbation_trajectory` minimizes scaled buoyancy-forcing cost subject to a 10% overturning reduction at year ten and a control-norm bound. It checks feasibility, stationarity, and complementarity. These checks establish local constrained optimality, not a global optimum. |
| Positive closure fitted to reference tendencies | All closure cells first fit a positive state-dependent diffusivity against conservatively projected, independently refined Veros tendencies. Closure-only heat/salt budgets and withheld tendency fits are required. |
| Closure transfer to seasons and circulation | The tendency-trained weights are used without refitting in withheld rollout, adjusted-cycle, equilibrium, or statistical objectives against the independent refined reference. The experiment checks transfer to tracer profiles and circulation. |
| Observable controls and useful T/S measurements | JVP/VJP transpose and finite-prediction checks validate the observation Jacobian. Greedy T/S selection must reduce uncertainty in a future overturning diagnostic by 20%. A nonlinear forward ensemble checks predicted variance. |
| Sill depth and efficient bathymetric modifications | Geometry cases fit depth/geometry controls and optimize bounded depth redistribution for exchange transport. The existing partial-cell basin remains a smaller fixed-topology test. |
| Grid placement and changing topology | A vertical-grid coordinate is optimized against an independently refined solution with other design coordinates held fixed. Conservative remapping and changed pressure operators are required. A relaxed channel-opening design must retain its transport in a realized discrete geometry, evaluated forward. |
| Climatological circulation calibration | Adjusted and statistical calibration cases jointly constrain hydrography, heat transport, depth-coordinate overturning, and ACC transport. State and sensitivity convergence are checked separately. |
| Climate means, variability, and extremes | Statistical cases require an invariant-response method, uncertainty estimates, effective sample size, split-window convergence, and agreement with independent long forward ensembles. The perturbation case also optimizes a smooth warm-event probability. |

## Controls, diagnostics, and windows

Controls are dimensionless. Physical scales are 1 K and 0.1 g/kg for initial
tracers, 20 W/m² for heat flux, 0.02 N/m² for wind stress, and
`3.5e-6 (g/kg) m/s` for salinity flux. Initial corrections conserve each level's
wet-volume tracer inventory. Heat and salinity forcing patterns have zero
wet-area integral. Ice blocking can still break the applied budget and fail the
experiment. Wind changes also update the TKE surface input.

Mixing coefficients use positive log coordinates. The closure diffusivity is
`1e-5 * exp(w0 + w1 * tanh(Nsqr / 1e-5))` m²/s. Geometry coordinates control two
bounded depth redistributions around Drake Passage and the North Atlantic,
fixed-count vertical-level placement, and a
positive continuous channel opening of `100 * exp(0.5 * z)` m. This keeps central
differences inside the admissible depth domain. The grid and topology providers must conservatively
remap state and rebuild geometry-dependent operators inside differentiation.

Global observations use North Atlantic and Southern Ocean wet-volume profiles,
North Atlantic upper-1000-m heat content, advective heat transport at 26°N,
Drake Passage transport, and Atlantic depth-coordinate overturning across 1000
m at 26°N. Heat and overturning transport include the meridional GM bolus
contribution. Actual partial-cell overlap is used at 1000 m. The observation
scales are 0.05 K, 0.02 g/kg, `1e21` J, `1e14` W, and `1e6` m³/s. ACC smoke
cases use domain-relative regions and sections, with the same units and
operators. Inverse recovery uses a numerical singular-value cutoff of
`max(1e-6, largest_singular_value * 1e-6)`. A resolved direction can still be
weakly constrained by the observations.

Local cases differentiate one step at four forcing phases from a prepared state.
They use twelve preparation steps by default. `--spinup-steps` replaces this
preparation. Preparation is held fixed during local differentiation. `--steps`
must be one for local cases. Trajectory acceptance uses at least 30 days for
sensitivity and geometry, a year for estimation/calibration/closure/observation
design, and exactly ten 360-day years for the overturning-target experiment.
Short `--steps` overrides retain these acceptance checks.

The periodic response uses the setup's 360-day year and twelve observation
phases. Statistical fixtures use a twenty-year burn-in, annual windows, monthly
samples, and sixteen windows in acceptance mode, with doubled windows for
transfer checks. These are initial experimental budgets, not proven sufficient
adjustment or sampling lengths. Convergence criteria may fail and demand longer
experiments. Event thresholds are frozen from baseline climate mean plus two
standard deviations, with a floor of one observation-scale unit. Exceedances
use a sigmoid of width one such unit. They are smooth event proxies.

## Extension contracts

The names below specify proposed extensions to `PreparedIntegrator`.
`responses.py` and `research.py` contain their callers. Missing callables are
reported as failed requirements before expensive model setup. After the
callables exist, the same program proceeds through model evaluations,
differentiation, optimization, and independent validation using the same
scientific criteria.

All response methods accept `carry`, `parameters`, and `observe`. Observations
are normalized by the experiment operator. Both JVP and VJP must propagate
through returned `observations`, including adjustment of the starting state.
Return dictionaries have the following required entries:

| Contract | Arguments beyond carry/parameters/observe | Required result or behavior |
| --- | --- | --- |
| `equilibrium_response` | `tolerance`, `max_steps` | `observations` of shape `(features,)`, `residual_kind="full_state"`, `residual`, `spectral_radius`, `sensitivity_residual`. Steady forcing is fixed at annual phase zero. Residuals include prognostic fields and multistep histories with appropriate physical scales. |
| `periodic_response` | `period_steps`, `phases`, `tolerance`, `max_cycles` | The same convergence/stability entries, with observations of shape `(12, features)`. A phase-consistent Poincare solve must include sensitivity of its phase-zero state. |
| `statistical_response` | `thresholds`, `window_steps`, `burnin_steps`, `n_windows`, `sample_stride`, `seed` | `observations` of shape `(3, features)` containing means, variances, and smooth exceedance frequencies; same-shaped `standard_error` and `window_drift`; scalar `effective_sample_size`; `method` identifying a justified invariant response. A plain finite-time AD estimator is rejected. |
| `with_tracer_closure` | `carry`, `law`, `weights` | Return a carry installing dynamic weights and conservative face fluxes from the pure JAX diffusivity law inside every timestep. The two research weight variables are declared before preparation. |
| `reference_tracer_tendencies` | `carry`, `n_states`, `seed`, `refinement`, `projection`, `withheld_seed` | `source="refined_veros"`, `refinement>=2`, `train_states`, `train_tendencies`, `heldout_states`, `heldout_tendencies`, `tendency_scale`, and pure callbacks `flux_divergence(carry, diffusivity)`, `budget_residual(weights)`, `response_target(kind, steps, heldout)`. Tendencies and rollout truth come from the independent refined model, not the fitted law. |
| `with_vertical_grid` | `carry`, `widths`, `conserve` | Return a carry with rebuilt levels, metrics, masks, density, pressure coefficients, and conservative remapping of fields and histories. The number of levels and total depth remain fixed. |
| `with_wet_geometry` | `carry`, `opening_m`, `channel`, `realized`, `conserve` | Return a carry with a continuous channel design or its discrete realization. Rebuild all affected geometry/operators. The relaxed design must retain its response when realized. |
| `grid_reference_error` | `carry`, `parameters`, `refinement`, `response`, `steps`, `forward=False`, `seed=0` | Differentiable scalar `error`, `remap_budget_residual`, and `operator_change_norm` against an independently refined geometry-aware solution. Statistical grid objectives use invariant sensitivities. `forward=True` evaluates independent forward ensembles for derivative validation. |

Stable adjustment requires full-state residual below `1e-7`, spectral radius
below one on the admissible prognostic tangent space, and sensitivity residual
below `1e-5`. The clock and prescribed controls are excluded from this stability
operator. Conserved inventories, solver gauges, and seasonal phase must be fixed.
Any removed neutral modes must be declared and their constraints respected by
controls and derivatives. Changed initial states and
larger solve budgets must agree in both observations and derivatives. A stable
equilibrium column alone does not satisfy these ocean contracts.

Statistical validation uses separate forward ensembles, not the provider's own
reported uncertainty alone. Directional derivatives must agree with resolved
ensemble finite differences at two perturbation scales within sampling
uncertainty. Deterministic Taylor-order requirements are not applied to noisy
climate estimates. Forward/reverse agreement is still required.

Passing the suite establishes these numerical workflows for its configurations,
control families, synthetic inverse targets, and reference data. Real-world
inference still requires suitable observations, historical forcing, noise and
model-discrepancy models, and resolution checks. A converged optimizer cannot
guarantee that a proposed physical perturbation is feasible or uniquely
identifiable. Those scientific failures remain visible in the reports.
