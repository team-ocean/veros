# Implementation plan for the 35 optimization demos

This plan implements the seven systems in
[Features required for the optimization demos](optimization-capabilities.md).
The [capability matrix](../../benchmarks/ad_demos/MATRIX.md) supplies the
scientific acceptance programs. The
[AMOC resilience proposal](amoc-resilience-research.md) is a subsequent integration
study.

Systems 1 and 2 already have implemented numerical foundations and
[CPU, GPU, and MPI verification](compiled-compatibility-validation.md). Their
remaining work is targeted physics validation, integration with the scientific
drivers, and measurement at the requested horizons. Systems 3–7 contain the
main new scientific tooling and numerical extensions.

Execution retains one compiled numerical timestep with host orchestration and
checkpoint replay. Block compilation is outside the implementation milestones.
Reconsider it only with a substantial, repeatable end-to-end benefit well beyond
a few percent, accounting for compilation, memory, and implementation complexity.

## Completion criteria

Completion requires all 35 cases to run through their numerical, optimization,
and independent-validation stages on their declared configurations. It also
requires evidence that the response and derivative estimators are appropriate
for those configurations. A missing API, execution error, or an unvalidated
derivative is an implementation limitation.

Scientific checks remain ordinary failures. Unidentifiability, infeasibility,
branch dependence, and insufficient closure expressiveness can be defensible
outcomes, provided they are investigated and supported by evidence. Preserve
the existing checks and report these diagnoses alongside them. A solver's failure
to find a solution alone does not establish infeasibility or nonidentifiability.

Use identifiable inverse problems, feasible constrained problems, and understood
response systems as positive controls in supporting tests. Keep the scientific
demos and their independent references as separate acceptance gates.

## Execution order and dependencies

| Milestone | Work | Dependencies | Evidence required to advance |
| --- | --- | --- | --- |
| M0 | Record baseline, resource costs, and failure mechanisms | Existing suite | Reproducible results and explicit blockers for every cell |
| M1 | Connect existing checkpoint adjoints to demos (2), build finite inverse tools (3), and apply established physics findings where they limit a case (1) | M0 | Validated finite responses at requested horizons and diagnosed outcomes for the ten executing cells |
| M2 | Equilibrium and seasonal responses (4); closure hook and reference pipeline (5) | Relevant M1 numerical and execution gates | Adjusted responses and closure studies validated on suitable physical configurations |
| M3 | Geometry, conservative remapping, and compatible partial-cell physics (6) | Relevant M1 gates; reference projection from 5 | Local/trajectory geometry validated, then combined with adjusted responses |
| M4 | Statistical response in stable and chaotic configurations (7) | Relevant M1 execution gates; M2 for stable adjusted responses | Independent statistical and derivative convergence evidence |
| M5 | Integrate all combinations and establish scientific configurations | Applicable gates from 1–7 | Every cell reaches its independent validation stage; results and limitations documented |

Start the forward reference-data work in 5 and the estimator feasibility study
in 7 during M1. The milestone order governs integration and acceptance. Those
investigations need not wait for all preceding work. Size later work using M0
profiling and the statistical feasibility results. Geometry physics and chaotic
response are substantial workstreams with uncertain effort.

## M0: Establish the baseline

- Reuse existing verification and smoke evidence. Rerun cases when implementation,
  configuration, or reference changes require it. Retain versions, hardware,
  configuration, assets, seeds, checks, memory, and wall time in persistent
  artifacts. Keep expensive global runs in the opt-in lane.
- Establish global finite-window baselines for representative objectives and
  controls. Measure increasing horizons and preparation lengths.
- Diagnose the six currently executing but failing cases. Distinguish optimizer
  stopping, poorly scaled objectives, weak observations, control restrictions,
  and model-response failures.
- Record the additional blockers hidden behind missing-API gates, especially
  incompatible partial-cell physics and absent adjusted-state convergence.
- Preserve scientific criteria when reorganizing demo adapters. Add solver
  convergence and diagnostic metadata without interpreting a numerical failure
  as a scientific conclusion.

Deliver a baseline report and a dependency checklist indexed by matrix case.

## 1. Reliable derivatives through the intended physics

The differentiation mechanisms and short-trajectory checks already exist,
including full-physics, runtime-parameter, and replay verification. The relevant
neutral-mixing investigations have also established reproducible causes and
candidate policies:

- [Unstable interior columns](tke-gradient-investigation.md) expose a nearly
  singular slope denominator and tangent/adjoint disagreement. The measured
  states do not support near-zero TKE as the cause. An experimental zero-slope
  policy restores the tested derivative agreement.
- [Surface quadrature](neutral-parity-investigation.md) uses a zero vertical
  gradient even above stably stratified interior faces. One-ULP horizontal tracer
  differences can switch the taper. A surface-only zero-slope control restores
  the tested coefficient parity.

The unresolved work is the intended mixing-policy decision and scientific
validation of any adopted change. These two controls have different scopes.
Retain the documented configuration/horizon limits while the default policy
remains unchanged.

### Implementation slices

1. Use the existing probes, results, and policy controls. For an affected demo,
   establish whether its limiting behavior matches one of the documented cases.
2. Resolve the intended surface and unstable-interior mixing policies. If a
   change is adopted, implement that specific policy with regression checks and
   validate forward fields, tracer/energy budgets, and resolved circulation.
3. Add investigation only for a new observed failure or evidence that the
   established explanation does not apply. Reuse existing limiter, turbulence,
   arithmetic, and solver verification elsewhere.
4. Extend horizon validation to the control families and default/fixed physics
   used by the matrix. Reuse verified CPU/GPU/MPI paths and repeat checks at the
   requested scientific horizons.

### Acceptance gate

Primal, tangent, adjoint, finite-difference, and finite-update results agree within
declared numerical tolerances in their applicable regimes. Results survive
checkpoint replay. Any altered discretization also passes forward-physics and
budget checks. Report supported configurations and horizons explicitly.

Primary locations are `veros/core/isoneutral`, `veros/core/operators.py`, the
turbulence/advection kernels, and the pressure/TDMA solvers. Existing derivative
tests and the gradient probe provide regression coverage. Add targeted cases
where the audit exposes a gap.

## 2. Affordable long integrations and derivatives

Compiled timesteps, bounded RAM/disk checkpoint schedules, recorded forcing
replay, accumulated objectives, parameter gradients, memory accounting, and MPI
adjoints are implemented. The scientific demos currently differentiate
PreparedIntegrator.advance directly instead of using the host checkpoint driver.
Connecting those interfaces is the immediate remaining work.

### Implementation slices

1. Adapt scalar scientific objectives to the existing host checkpoint gradient
   interface. Pull initial-carry and runtime-parameter cotangents back through
   the control transforms. Accumulate observation objectives through existing
   numerical records where appropriate.
2. Expose streamed tangent propagation and response JVP/VJP operations for
   observation Jacobians and implicit solves. Preserve existing replay semantics
   and validate the adapters against direct short-trajectory AD.
3. Connect new closure and geometry inputs as those features are implemented.
   Reuse the existing carried state, forcing records, and checkpoint storage.
4. Benchmark the requested annual and ten-year windows and repeated ensembles
   using the existing single-step execution. Keep per-step JIT as the baseline.
   Consider blocks only after evidence of a substantial, repeatable end-to-end
   gain well beyond a few percent. Include compilation and memory costs in that
   comparison and justify the additional complexity.

### Acceptance gate

Short direct AD and checkpoint-backed scientific objectives agree in values and
derivatives. Memory follows the declared checkpoint/storage budget as horizons
increase. Compiled timestep signatures are reused independently of total horizon.
Reports expose wall time, replay cost, memory, and failure status. Run the
minimum scientific windows for the finite-trajectory cells within documented
resource budgets.

Primary locations are `veros/integration.py`, `veros/checkpointing.py`, and the
response adapters in `benchmarks/ad_demos`.

## 3. Inverse, constrained-optimization, and uncertainty tools

### Implementation slices

1. Separate control transforms, whitened residuals, priors, observations, and
   optimization from the numerical response provider. Keep this analysis layer
   outside the timestep implementation.
2. Add convergence-controlled least-squares solvers with scaling, matrix-free
   JVP/VJP products, and appropriate preconditioning. Record stopping criteria
   and multiple-start results before interpreting failed recovery.
3. Add constrained optimization with explicit feasibility restoration, physical
   budgets, admissible control domains, and stationarity/complementarity checks.
4. Diagnose observable combinations relative to observation error and priors.
   Distinguish numerical rank, weak information, and predictive uncertainty.
5. Extend observation design with correlated errors and measurement costs.
   Validate its local covariance prediction using nonlinear forward ensembles.
   Add broader posterior exploration when local approximations fail.

### Acceptance gate

Positive controls demonstrate recovery and constrained optimality. The matrix
inverse cases report converged solutions or an investigated limiting mechanism,
with withheld-region and longer-response validation. Observation selection is
checked against future transport, not parameter variance alone. Preserve the
existing 90%/50% misfit, recovery, and transport checks while diagnosing failures.

Primary locations are the analysis helpers and scientific drivers under
`benchmarks/ad_demos`. A general production analysis module should be extracted
only when these workflows demonstrate a shared stable interface.

## 4. Equilibrium and seasonal response

### Implementation slices

1. Define a scaled prognostic residual that includes multistep histories and
   excludes clocks and prescribed controls. Specify conservation constraints,
   gauges, phase conventions, and admissible tangent spaces.
2. Implement the equilibrium forward solve with convergence and branch metadata.
   Use matrix-free residual products and preconditioners.
3. Implement implicit JVP/VJP solves, including direct parameter dependence of
   observations. Check the sensitivity residual independently of the state
   residual.
4. Build the annual Poincare map using the long-integration operations. Solve for
   a phase-consistent starting state, differentiate its adjustment, and return
   the twelve observation phases.
5. Estimate stability on the constrained prognostic space. Investigate changed
   initial states, larger budgets, nearby controls, and alternate branches.

### Acceptance gate

The `equilibrium_response` and `periodic_response` contracts satisfy the matrix's
state, stability, sensitivity, and independence checks. Supporting manufactured
systems establish numerical correctness. Ocean cases establish physical scope.
If the intended configuration lacks a stable equilibrium or annual cycle,
document that result and identify a physically justified configuration for the
corresponding workflow. Do not substitute a climatological mean for a periodic
orbit.

Deliver equilibrium responses before seasonal responses. Use a shared
matrix-free solve layer and keep adjustment drivers outside core timestep
orchestration where possible.

## 5. Trainable closures and independent references

### Implementation slices

1. Build the refined-model run and conservative projection pipeline. Define the
   coarse-grid resolved tendency, unresolved correction, boundary contributions,
   normalization, and reference provenance before fitting a law.
2. Add the closure hook at the intended conservative tracer-flux calculation.
   Specify which baseline process it replaces or augments. Support dynamic
   weights, continuation, and checkpoint replay.
3. Validate positivity, closure-only tracer budgets, and the relevant energy
   exchange. Fit projected tendencies on one set of states and validate on
   independent states and forcing conditions.
4. Run the trained weights without refitting against refined-model circulation
   targets. Add equilibrium, seasonal, and statistical transfer as their response
   providers become available.

### Acceptance gate

The `with_tracer_closure` and `reference_tracer_tendencies` contracts are exercised
inside an ocean timestep. Independent data support both tendency and rollout
checks. A closure family that cannot represent the reference correction remains
a documented scientific failure. Broaden its physics only through an explicit
modeling decision.

Keep reference generation and datasets separate from the numerical hook. Reuse
the conservative projection operators in geometry validation.

## 6. Geometry and compatible partial-cell physics

### Implementation slices

1. Define inventory and momentum remapping for vertical-level movement and
   volume changes, including histories. Preserve existing fixed-concentration
   bathymetry semantics unless a caller explicitly requests a different policy.
2. Implement `with_vertical_grid`, rebuilding coordinates, metrics, density,
   fractions, and pressure coefficients. Test conservative remapping and
   operator derivatives before optimizing a grid.
3. Extend partial-cell discretizations required by the actual demo settings in
   small slices: tracer/neutral/skew fluxes, compatible friction, turbulence
   energy equations, energy exchanges, and needed diagnostics. Retain guards
   for configurations whose physics remains unsupported.
4. Implement continuous channel design with fixed array shapes. Rebuild
   connectivity-sensitive boundary conditions and gauges, handle shallow/dry
   limits, and evaluate discrete realizations forward.
5. Implement `grid_reference_error` using independent refined runs and
   conservative projection. Combine grid-accuracy, bathymetric-exchange, and
   realization validation with each response provider.

### Acceptance gate

Resting states remain at rest. Tracer and momentum remapping conserve the
declared quantities. Compatible physics closes its budgets. Operator JVP/VJP
and finite differences agree. The five geometry cases reach both refined-grid
and discrete-realization checks. An optimum in the relaxed geometry must still
be evaluated in its physical realization.

Primary locations are `veros/core/geometry.py`, numerical metric construction,
pressure solvers, affected physics kernels, and their diagnostics. Removing
partial-cell setting guards is the final consequence of validated support.

## 7. Statistical response

### Implementation slices

1. During M1, establish a feasibility benchmark with an understood stable
   system, a chaotic test system, and a representative ocean configuration.
   Measure transient growth, adjustment, autocorrelation, and finite-difference
   signal versus sampling noise.
2. Implement statistical observations and uncertainty accounting independently
   of the sensitivity estimator. Check means, variances, and smooth exceedance
   frequencies, including the sampling precision of each observable.
3. Provide invariant responses from validated stable equilibria and cycles.
   These are a baseline for the statistics interface, with a limited physical
   scope.
4. Compare candidate chaotic-response estimators against independent ensemble
   finite differences on the feasibility benchmark. Select a method based on
   validity, bias, variance, memory, and cost. Record assumptions and conditions
   where the response may not be differentiable.
5. Implement the selected estimator using shared JVP/VJP and integration tools.
   Validate changed initial states, seeds, phases, longer windows, and larger
   ensembles. Define which attractor is sampled when regimes coexist.
6. Connect `statistical_response` to all seven tasks, including independently
   evaluated closure-transfer and grid-accuracy objectives.

### Acceptance gate

Statistical values and derivatives agree with independent forward ensembles
within resolved sampling uncertainty. Forward/reverse consistency accompanies
this check but does not replace it. Report convergence, effective sample size,
uncertainty, and estimator scope. A successful stable-system test does not
establish chaotic-ocean sensitivity. If no candidate is validated for the ocean
configuration, retain an explicit capability limitation and continue estimator
research.

## M5: Integrate and review all 35 cells

- Exercise every task/response combination, including closure and geometry
  composed with adjusted and statistical providers. These combinations need
  validation even when the components pass separately.
- Reuse the matrix CLI filters and opt-in acceptance lane. Keep fast numerical
  tests separate from expensive global experiments, and retain failure evidence
  and reproducible reference assets.
- Review supported horizon, starting-state preparation, physical observation
  definitions, budgets, priors, and sampling precision for each case. Fixed
  initial preparation must remain distinct from differentiation of adjustment.
- Record whether each failed scientific criterion reflects a numerical problem,
  unresolved evidence, or an investigated scientific limitation. Maintain
  failed checks. Do not replace them with expected-failure markers.
- Publish a support matrix linking each cell to its measurements and limitations.
  Extend to real observations and the AMOC resilience experiment only after the
  relevant numerical and inference gates are satisfied.

Each implementation slice should contain one coherent feature, its targeted
verification, and updated support evidence. Extend the existing numerical
capabilities. Start with connecting scientific objectives to the existing
checkpoint adjoint and improving inverse-solver convergence. Follow up the
established neutral-mixing findings where they limit a target case. Adopt and
validate a specific physical policy before extending its claimed support.
