# Features required for the optimization demos

The [35-cell capability matrix](../../benchmarks/ad_demos/MATRIX.md) defines
scientific acceptance experiments. Making each experiment execute and producing
a scientifically defensible result are separate requirements. Some experiments
can legitimately reveal an unidentifiable control, an infeasible target, or an
inadequate closure family.

The [implementation plan](optimization-implementation-plan.md) defines delivery
milestones, dependencies, and acceptance gates for the seven systems below.

## Current baseline

The 2026-10-11 ACC smoke runs completed all 35 cases without execution errors or
timeouts. Four passed and 31 failed. These are short numerical experiments, not
global scientific acceptance. Global 4-degree local sensitivity also passed its
smoke checks.

Eight absent interfaces gate 25 cases before ocean setup. The remaining ten
execute. Six fail recovery, transfer, optimization, or observation-information
criteria. Their failures do not by themselves establish a defect in AD.

Compiled timesteps, dynamic floating setting overrides, implicit TDMA
derivatives, differentiable pressure solves, and RAM/disk adjoint
checkpointing already exist in this checkout. Fixed-topology partial-cell depth
controls also exist. [Verification](compiled-compatibility-validation.md) covers
CPU, GPU, and MPI derivatives and checkpoint replay on small, short physical
cases. Each has a narrower validated scope than the full matrix.

## Required systems

### 1. Numerically reliable derivatives through the intended physics

The differentiation mechanisms and short-trajectory validation are implemented.
The [unstable-column investigation](tke-gradient-investigation.md) and separate
[surface-stencil investigation](neutral-parity-investigation.md) already isolate
neutral-slope sensitivities and demonstrate candidate policy controls. Their
scopes differ, and default physics remains unchanged. The remaining work is to
decide the intended mixing policies, validate any adopted changes, and establish
support at the scientific configurations and horizons.

Reuse the existing investigations and numerical checks. Extend them when a
target case exposes a new failure or contradicts the established explanation.

Deliver repeatable primal, JVP, VJP, finite-difference, and finite-update checks
over scientifically relevant horizons, starting states, seasons, and enabled
physics. Passing one local gradient does not establish a general AD horizon.

### 2. Affordable long integrations and derivatives

Compiled timesteps and bounded-memory RAM/disk adjoints are implemented, with
recorded forcing replay, accumulated objectives, parameter gradients, memory
accounting, and CPU/GPU/MPI verification. The remaining integration gap is that
the matrix calls jax.grad and jax.jvp around response values, while the host
CheckpointedIntegrator.value_and_grad cannot itself be transformed with JAX.

Adapt scientific objectives to the existing checkpoint gradient interface and
provide response JVP/VJP operations for observation and implicit-solve workflows.
Connect closure and geometry inputs as they are implemented. Measure the
requested horizons and ensemble costs using the existing execution path.
Per-step JIT is the execution baseline. Reconsider compiled blocks for a
substantial, repeatable end-to-end performance gain well beyond a few percent
or a demonstrated material UX improvement. Establish whether the user workflow
benefit requires blocks, and account for compilation, memory, and implementation
complexity. Blocks remain outside the required milestones.

### 3. Inverse, constrained-optimization, and uncertainty tools

Replace the small normalized-gradient optimizer with convergence-controlled
solvers, appropriate scaling/preconditioning, and matrix-free Jacobian products.
Provide residual/JVP/VJP interfaces for least-squares problems, constrained
optimization with feasibility and stationarity checks, and multiple starting
points where local minima matter.

Separate numerical Jacobian rank from practical identifiability under specified
noise and priors. Report weak and null combinations, parameter tradeoffs, and
uncertainty in predictions. Extend the local Gaussian observation-design
calculation with correlated observation errors, sampling costs, and nonlinear
posterior validation when those assumptions are required by a study.

Diagnose whether a failed inverse case is limited by optimization, observations,
control expressiveness, or model discrepancy. A one-step observation set may
not constrain future transport. A requested 10% overturning reduction may be
infeasible within the forcing budget. Neither failure should be repaired by
weakening acceptance tolerances without changing the scientific question.

### 4. Adjusted-state solves and their implicit derivatives

Implement equilibrium_response and periodic_response. They serve all seven
equilibrium and all seven seasonal cells, respectively, with additional closure
or geometry requirements in those rows.

For an equilibrium x = P(x, theta), solve the scaled prognostic residual and
the tangent system (I - P_x) dx = P_theta dtheta. The adjoint uses the transpose.
Use matrix-free JVP/VJP operators. Include
multistep histories, appropriate preconditioners, and sensitivity residuals.

For a seasonal cycle, use the full annual map at a fixed phase, solve for its
phase-consistent starting state, and propagate its derivative into monthly
observations. Fix conserved quantities and gauges and exclude the clock and
prescribed controls from stability operators.

Check state and derivative convergence separately, with changed initial states
and larger solve budgets. Detect branch dependence and loss of stability. A
configuration with chaotic variability may have no attracting annual cycle.
Periodic forcing alone does not establish one.

### 5. Conservative trainable closures and independent references

Implement with_tracer_closure and reference_tracer_tendencies for all five
closure cells. Install the pure trainable law in the actual timestep, specify
which mixing process it replaces or augments, and carry its weights through
continuation and checkpoints. Require positive diffusivity, conservative face
fluxes, and appropriate energy accounting.

Build a reproducible refined-Veros reference pipeline with conservative
projection, a defined unresolved-tendency target, training and withheld states,
normalization, closure-only budgets, and independent rollout targets. The
reference must not be generated by the law being fitted.

Validate tendency fits and transfer to seasons, long circulation, and statistics
separately. A two-parameter positive vertical diffusivity may be unable to
represent the refined-model correction. That is a possible scientific result.

### 6. Geometry, remapping, and compatible physics

Implement with_vertical_grid, with_wet_geometry, and grid_reference_error for
all five geometry cells. Rebuild metrics, cell/face fractions, density-related
fields, pressure operators, and solver coefficients inside the numerical map.
Conservatively remap prognostic fields and multistep histories and expose the
associated derivatives. Define how physical inventory is handled when the
domain volume changes. Fixed concentrations do not conserve inventory.

Support continuously varying channel geometry with fixed array shapes, then
evaluate the realized discrete geometry forward. Reassess null spaces and
boundary conditions when connectivity changes. Validate resting states,
transport and tracer budgets, operator derivatives, and independent refined-grid
accuracy. A successful relaxed design need not survive discrete realization.

The current partial-cell setting validation rejects TKE, EKE, IDEMIX, neutral
and skew mixing, energy conservation, and several friction options. The geometry
demos retain setup physics that intersects these exclusions, even with fixed
vertical mixing. These physical discretizations must be extended and validated
before the geometry row can run as currently specified. Removing the guards
would not implement those discretizations.

### 7. Statistical responses with uncertainty and convergence evidence

Implement statistical_response for all seven statistics cells. Provide means,
variances, and smooth exceedance frequencies, together with autocorrelation-aware
uncertainty, effective sample size, and convergence diagnostics. Test adjustment,
window length, ensemble size, starting state, and initial forcing phase.

Choose a derivative estimator justified for the configuration. Stable equilibria
and periodic cycles can supply statistical responses through their adjusted
solutions. Chaotic configurations may require ensemble response, shadowing, or
other linear-response methods whose assumptions are checked. Long-time
statistical sensitivity is a distinct problem from differentiating a single
finite trajectory. See
[Wang, Hu, and Blonigan, 2014](https://doi.org/10.1016/j.jcp.2014.03.002).
No particular estimator is established for these Veros cases yet.

The [shadowing evaluation plan](shadowing-evaluation-plan.md) makes NILSS/NILSAS
an early feasibility investigation, including validation of potential bias from
an omitted unstable-response contribution.

Validate derivatives against independently simulated ensemble finite differences
with resolved signal and sampling uncertainty. Extend the same estimator to
closure-transfer and grid-accuracy objectives. Define attractor conditioning and
transition sampling explicitly when multiple circulation regimes coexist.
Neither a fixed twenty-year burn-in nor sixteen annual windows guarantees
adequate adjustment or sampling.

## Delivery order

1. Reuse existing AD, checkpoint capabilities, and neutral-mixing investigations.
   Connect scientific objectives to checkpoint replay, follow up the established
   physics findings where they limit a target case, and improve
   convergence diagnostics for the ten currently executing cells and measure
   global finite-window performance.
2. Build the common matrix-free adjusted-state solver infrastructure, then
   equilibrium and seasonal response methods.
3. Build the independent-reference pipeline and conservative closure hook.
   Develop geometry/remapping and compatible partial-cell physics as a separate
   substantial numerical project.
4. Develop and validate statistical-response estimators first on understood
   stable configurations, then on chaotic ocean configurations.
5. Connect the capabilities to real observations, historical forcing, realistic
   error/prior models, and resolution/model-form validation for scientific studies.

The eight interface names specify demo contracts. Analysis and reference-data
tools can use separate adapters that preserve the acceptance criteria.

The [AMOC resilience proposal](amoc-resilience-research.md) additionally needs
observation-conditioned searches for divergent futures and nonlinear branch or
transition analysis. The matrix provides foundations for that study but does
not yet contain the complete inference experiment.
