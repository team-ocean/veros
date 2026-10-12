# Differentiable research acceptance demos

Forty-four executable experiments define acceptance criteria for scientific uses
of Veros AD. Unmet criteria are reported as failures.

The [optimization capability matrix](MATRIX.md) adds 35 scientific cases covering
all seven tasks across local, trajectory, equilibrium, seasonal, and statistical
responses. It records the research-question coverage and proposed APIs needed
to make currently unsupported studies run. Export it with `--matrix`, run it
with `--demo matrix`, and select cells using `--task` and `--response`.

The seven focused scientific cases are bathymetry optimization, forcing
response, joint state/forcing estimation, mixing calibration, observation
design, closure learning, and equilibrium sensitivity. Gradient-horizon and
checkpoint-replay cases exercise shared infrastructure. The closure and
equilibrium experiments are explicitly column prototypes using Veros' TDMA
primitive. Their results do not establish global ocean closure learning or
equilibrium sensitivity.

## Running

Use a Python environment with Veros' JAX dependencies installed, from the
repository root:

```sh
python -m benchmarks.ad_demos --list
python -m benchmarks.ad_demos --mode smoke --output /tmp/ad-smoke
python -m benchmarks.ad_demos --mode acceptance --output /tmp/ad-acceptance
python -m benchmarks.ad_demos --demo forcing_response --setup global_4deg \
  --mode acceptance --output /tmp/ad-global-forcing
```

Each case runs in a fresh serial process with a timeout. The parent continues
after failures and exits with a nonzero code if any case fails, errors, or
times out. Each experiment produces a JSON report and a log. `summary.json`
collects all results. JSON retains measured values, tolerances, failed checks,
configuration, versions, elapsed time, and process peak RSS where available.
Checks are saved incrementally so completed measurements survive a timeout or
process failure. Nonfinite measurements serialize as `null` and fail checks. A
previous report is removed before rerunning its case. Supply a separate output
directory for each configuration you want to retain.

Statuses distinguish `passed`/`failed` acceptance runs from
`smoke_passed`/`smoke_failed`, execution `error`, and `timeout`. **A smoke pass
is not scientific acceptance.** The focused ocean fixtures use six smoke steps
and default to 30 days for acceptance, with a minimum-window check. `--steps`
overrides the integration length without bypassing that check. Matrix cases use
the preparation and response windows in [MATRIX.md](MATRIX.md), including
annual and ten-year trajectory experiments. These acceptance windows still
require scientific validation. The column closure acceptance window is 96
steps, with a four-times-longer held-out rollout. Column equilibrium acceptance
uses 512, 2048, and 8192 daily steps to test convergence to the adjusted state
and its sensitivity.

Small ACC runs use an 8 × 8 × 15 grid and 600-second momentum/tracer steps. This
is a numerical fixture with a restricted geographic domain, not a realistic ACC
simulation. Global runs retain the stock 90 × 40 × 15 grid and its 86400-second
tracer/1800-second momentum timesteps. Global setup needs its external assets.
`--spinup-steps` advances the starting state before constructing controls and
records its initial time. This state is held fixed during differentiation.
Spinup parameter dependence is not included.

Initial-temperature controls update all three tracer slots and recompute density,
stratification, and enthalpy fields. Velocity and tendency histories remain fixed.
The control does not construct a dynamically balanced initial state.

`--physics default` preserves enabled setup physics. `--physics fixed` disables
TKE/EKE/IDEMIX and uses constant mixing as a separate controlled experiment.
Its background vertical diffusivity is set to `1e-5` m²/s so positive log-scaled
calibration controls have a nonzero baseline.
Neutral and skew mixing stay enabled. Results from this configuration do not
validate the default parameterizations. `--device gpu` selects GPU execution.
These experiments initialize no MPI communicator, including in MPI-enabled
installations. `--timeout` sets a per-case limit in seconds (default 600).

## Scientific cases

| Case | Controls and observations | Required behavior |
| --- | --- | --- |
| `bathymetry_optimization` | A 20 m sill-depth mode in an 8 × 10 × 4 basin with partial bottom cells, constant mixing, stable stratification, and surface pressure. Synthetic sea surface height and surface velocity sensors. | AD, finite-difference, and Taylor checks pass; training misfit falls by 90%; maximum depth error is below 1 m; a double-length held-out trajectory improves by 50%; RAM/disk replay gradients agree. |
| `forcing_response` | Two latitude cosine patterns, each with zero wet-area mean and unit area-weighted RMS; 20 W/m² per unit control. Objective is regional heat-content change, in joules, normalized by an imposed heat-input scale. | AD checks pass; a projected update increases regional heat content; control norm stays below one; both prescribed and actually applied added flux remain globally balanced. |
| `state_forcing_twin` | A 1 K upper-ocean initial-temperature mode and two flux modes. Synthetic regional T/S profiles and section transport at three times. Fit the first two times and hold out the last. | Training misfit falls by 90%; held-out misfit falls by 50%; scaled controls recover within Euclidean distance 0.15. Failure of recovery can expose an unobservable control or insufficient optimization. |
| `mixing_calibration` | Positive log-scaled `c_k`/`c_eps` with TKE, or `kappaH_0`/`K_iso_0` with fixed mixing. Fit regional T/S profiles and transport against a known-parameter trajectory. | Training misfit falls by 90%; double-length rollout error falls by 50%; log-parameter recovery error is below 0.1. |
| `observation_design` | Whitened observation Jacobian of the twin experiment. A unit Gaussian prior on scaled controls; greedy selection targets the first flux-mode variance. | Transpose identity and finite perturbation prediction hold; observations reduce target variance by 20%; smallest Jacobian singular value exceeds 0.1; a sampled local posterior reproduces its predicted variance. |
| `closure_learning` | Conservative implicit column diffusion with positive face diffusivity `1e-3 * exp(z0 + z1 * tanh(dT/dz / 0.02))`, in m²/s. | Training misfit falls by 90%; a different initial profile and four-times-longer rollout improve by 50%; diffusivity stays positive; column heat is conserved; parameters recover within 0.15. |
| `equilibrium_sensitivity` | A diffusively coupled column with bottom heat input and 30-day surface restoring. Compare a stationary TDMA solution, a residual-equation adjoint, and finite adjustment trajectories. | Stationary residual is below tolerance; adjoint and solver AD agree; adjusted objective and its gradient converge to equilibrium within `1e-4`. |

The ocean observation operator uses interior wet-cell volume weights, excludes
halos, and computes upper-1000-m meridional transport in Sv at the middle
V-grid section, including partial-layer overlap. It is not a density-binned
overturning diagnostic. The focused inverse fixtures generate synthetic truth
with the same model. Matrix closure and grid-design cases instead require
independently refined Veros references. Passing demonstrates the
numerical workflow under those assumptions, not performance against real
observations or model discrepancy.

The twin/design uncertainty scales are 0.05 K for temperature, 0.02 g/kg for
salinity, and 0.5 Sv for transport. Mixing calibration instead normalizes each
observation by its truth/baseline difference, floored at `1e-4` in that
observation's units. These explicit numerical fixtures need reassessment before
using real data. Observation-design covariance is a local linear-Gaussian
calculation, not a validated nonlinear posterior.
Its small Jacobian is assembled from individual JVPs, so this experiment does
not require batching support in every numerical primitive.

Global added flux is applied after the stock forcing calculation and blocked
where the combined flux would cool water below the setup's -1.8 °C ice threshold.
Ice blocking can break the prescribed zero-integral flux budget. The applied
budget check intentionally exposes this. The heat budget check covers the added
flux, not the entire ocean heat budget or density-related energy conversions.

## Shared validation

`gradient_horizon` checks up to three increasing lengths up to the requested window.
Each length compares primal values, forward/reverse directional derivatives,
central differences at five perturbation scales, Taylor remainders, and an
actual finite descent step. It starts from the configured state. Repeat with
other spinup lengths and forcing/physics configurations before asserting a
supported horizon. It does not choose a model-wide horizon from one objective.

The Taylor check requires an observed order between 1.7 and 2.3 in at least one
resolved interval, or all remainders below `1e-9 * max(1, abs(loss))` for a
response whose curvature is unresolved at the tested scales. This also admits
affine objectives. Central differences must agree at two or more scales.
Branch-dependent objectives may fail these conditions. The report retains
the measurements and failed checks.

`checkpoint_replay` compares composed AD with RAM/disk replay using one or two
RAM slots. It checks loss, control-map VJP, runtime parameter gradient, and storage
budgets. It uses a terminal objective. Diagnostic-window replay is covered
separately in `test/checkpointing_test.py`.

The bathymetry case uses the partial-cell API described in
`doc/tutorial/differentiation.rst`. It runs its own basin fixture independently
of `--setup` and `--physics`, starts from rest, and rejects nonzero spinup.
Its acceptance window is 96 steps at 300 seconds, or eight hours. The held-out
trajectory is sixteen hours. These lengths test the inverse workflow rather
than a general ocean sensitivity horizon. Observation residuals use one
truth/baseline response scale per sensor type, floored at `1e-8` in that type's
units. Geometry changes retain initial concentrations and zero velocity
histories. The experiment checks local fixed-topology depth controls. It does
not validate moving coastlines or global bathymetry estimation.

The demos use the default neutral-slope policy and derivative conventions.
Distinguish runner and import errors from numerical or scientific failures.

## Pytest acceptance lane

Ordinary tests check the harness without requiring the scientific cases to pass.
The expensive acceptance lane is opt-in:

```sh
python -m pytest test/ad_demo_test.py --backend jax
VEROS_RUN_AD_ACCEPTANCE=1 python -m pytest test/ad_acceptance_test.py
```

`VEROS_AD_SETUP`, `VEROS_AD_PHYSICS`, and `VEROS_AD_TIMEOUT` select the acceptance
configuration. Column cases retain their column scope regardless of the ocean
setup selection. Matrix cases require global geometry in acceptance mode and
include explicit failures for missing ocean closure, adjustment, statistical,
and geometry capabilities. Existing column fixtures retain their smaller scope.
Ordinary coverage tests and response-contract tests are in `test/ad_matrix_test.py`.
