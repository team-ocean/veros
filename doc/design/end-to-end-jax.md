# End-to-end JIT and differentiation

Design proposal based on commit `97f261a`. The implementation is documented in [Compiled integration and differentiation](../tutorial/differentiation.rst).

Veros can plausibly support a compiled, differentiable integration over many time steps without rewriting its numerical core. The proposed change is a pure integration API alongside the existing runner, built from the current routines and kernels. It needs an explicit carry for model variables and diagnostic accumulators, a small tracing mode for routine wrappers, and a host driver for output and restarts.

The main work is making the surrounding state and effects safe under JAX transformations. Solver derivatives and finite gradients through the parameterizations need separate validation. Compiling a trajectory successfully would establish only part of the goal.

## Intended behavior

After ordinary setup, one compiled call advances a fixed number of steps, including forcing, physics, compatible plugin hooks, sanity checks, diagnostic sampling, diagnostic finalization, and time-index rotation. It returns the continuation state, requested diagnostic records, and failure information. Calling it again with the returned state continues the same simulation.

Users can differentiate a scalar objective of the final state, an accumulated diagnostic, or emitted diagnostic records with respect to selected floating inputs. Initial tracers, forcing arrays, and explicitly supported physical coefficients are useful targets. Integer indices, masks, island topology, enabled schemes, dimensions, and output schedules are fixed for differentiation.

NetCDF/HDF5 writes, console messages, progress bars, signals, and file reads remain host operations. A production driver advances one compiled timestep, checks its status, and writes its results. Pure trajectories compose these steps in Python, subject to reverse-mode memory limits. File writing is outside the differentiated function.

Deliver this first on one CPU or GPU. MPI forward execution and MPI differentiation are additional acceptance gates described below. Single-device support does not establish distributed support. Setup remains eager. Differentiating through grid construction, island discovery, asset loading, or a spin-up performed outside the integration API is outside the first implementation.

## Baseline at commit `97f261a`

| Location | Existing behavior | Required change |
| --- | --- | --- |
| `veros/state.py` | `VerosVariables` and `VerosState` are registered PyTrees. Only `_variables` is dynamic in the state tree. Diagnostics, settings, timers, and plugins are auxiliary data. | Carry diagnostic arrays explicitly and keep mutable host objects outside computation. Audit reconstruction under AD. |
| `veros/routines.py` | Routines mutate variables, return nothing, run timers/logging, flush the backend, and sometimes gather/scatter variables. Kernels return updates and are individually jitted. | Reuse routines on detached state shells with instrumentation disabled during tracing. Keep existing kernel interfaces. |
| `veros/veros.py:VerosSetup.step` | A step combines restart writes, numerical updates, hooks, Python sanity branching, diagnostics, output, and time-index rotation. | Extract shared numerical orchestration and add a pure adapter. |
| `veros/veros.py:VerosSetup.run` | Python loops until elapsed time reaches `runlen`. | Compile a pure timestep and compose its derivatives across a host-managed trajectory. |
| `veros/diagnostics/api.py` | Python branches on `vs.time % frequency < dt_tracer`. | Express due events with array predicates and functional branches. |
| Diagnostic classes | Accumulators live on mutable diagnostic instances. Several `output()` methods compute, normalize, write, and reset together. | Separate numerical sampling/finalization from writing and explicitly return updated accumulators. |
| `core/external/solvers/__init__.py` | `best` selects host SciPy on CPU; solver instances are memoized by state identity. | Select the JAX solver explicitly and prepare it before tracing. Avoid initializing or caching solvers on traced state shells. |
| `core/external/solvers/scipy_jax.py` | JAX BiCGSTAB operates on matrix coefficients captured during solver construction. It ignores the returned `info`. | Retain this solver for fixed geometry, add residual checks, and validate its adjoint. |
| `core/special/tdma_.py` | The accelerated TDMA primitive has implementations, abstract evaluation, and CPU/GPU lowerings, but no JVP/transpose rules. | Initially use the pure JAX TDMA path; add an implicit derivative for accelerated solves separately. |
| `core/external/solve_pressure.py` | Python branches on `vs.itt == 0`. | Replace this branch with `lax.cond` or an equivalent array selection. |

Most core branches select schemes using static settings and can stay as Python branches. Likewise, fixed loops over stencil offsets or two neighboring levels need no general conversion. The salt check in `calc_initial_conditions` and island discovery are setup operations, so they do not block the proposed integration boundary.

## Smallest useful architecture

### 1. Carry existing variable containers explicitly

Add `veros/integration.py` with a carry containing:

- Model `VerosVariables`, including all three time levels, tendencies, forcing data, `time`, `itt`, and `tau`/`taup1`/`taum1`.
- A fixed dictionary of diagnostic `VerosVariables` for diagnostics with internal state.
- Numerical status, including the first failed iteration and a failure code. Add an explicit RNG key only when a supported stochastic hook needs one.

Use the existing registered variable PyTree and ordinary tuples/dictionaries. Initially carry all model arrays, including geometry. Separating prognostic fields from grid fields can wait for evidence that it helps compilation or memory use.

Preparation captures an isolated configuration snapshot containing settings, manifested dimensions and metadata, diagnostic selections and frequencies, compatible hooks, and the prepared solver. Rebuilding this prepared object is required after configuration changes. Changing only array values must reuse the executable for the same shapes, dtypes, step count, and record selection.

For each numerical step, construct a detached `VerosState` shell from this configuration and the carry's model leaves. The shell supports existing `state.variables`, `state.settings`, `state.dimensions`, and timer access. Attribute assignment during tracing stays local to this shell, and the adapter returns all updated arrays. Never run the pure path on `simulation.state` itself or mutate a diagnostic instance captured in auxiliary data.

Numerical diagnostics receive their accumulators as explicit arguments. Snapshot records read the shell's variables directly. They must not use `Snapshot.variables`, which aliases the original variable container today. No new state-container hierarchy is needed, and the existing `VerosState` PyTree need not absorb all diagnostic objects.

Keep metadata identities and lock flags stable across timesteps. The current variable PyTree includes `__locked__` in its auxiliary data, so the adapter must restore locks before returning the carry. Use stable no-op timer objects for nested kernel calls.

Make `veros_variables_pytree_unflatten` install leaves without dtype coercion or allocation. Its current use of `setattr` invokes validation and `asarray`, which needs checking against abstract leaves, AD placeholders, and integer `float0` tangents. Keep shape/dtype validation at ordinary assignment and API boundaries. Apply the same audit to the distributed wrapper when enabling MPI.

### 2. Reuse mutable routines behind a pure boundary

Introduce a scoped tracing flag in the existing thread-local routine context. In that mode:

- `VerosRoutine` unlocks the local variables and invokes the routine body without timers, logging, or `flush()`.
- `VerosKernel` skips instrumentation. Keep nested kernel JIT initially. Remove it within this mode only if measurements justify the change.
- Explicit `with state.timers[...]` blocks use no-op timers. Setting `timer_context.active = False` alone is insufficient because it still records start and last times.
- Calls requiring host-only operations are rejected or handled outside the pure boundary. For MPI, `dist_safe=False` needs its own validated compiled implementation.

This context controls Python tracing behavior, not device execution. It must be restored after exceptions and must not carry numerical data. Wrapper mutation, solver caches, diagnostic instances, and timers must not retain tracers after a trace or a retrace.

Extract the existing numerical sequence from `step()` into one shared helper. Keep the ordering and routine bodies. The existing host step and the new adapter should both use that helper, with their own effect handling. Avoid maintaining a second copy of the momentum/thermodynamics/EKE/TKE/IDEMIX sequence.

Preserve this order exactly:

1. The host runner handles any restart due at the incoming state, matching the current pre-step restart point.
2. Forcing, diffusivities, momentum, thermodynamics, energy integrations, boundaries, vertical velocity, and plugin run hooks execute.
3. `itt` and `time` advance, then `after_timestep` executes.
4. Sanity checking and the isoneutral diagnostic streamfunction execute, followed by sampling and output-event finalization.
5. The three time indices rotate.

Today diagnostics read `vs.tau` before rotation even though many updates have written `vs.taup1`. Preserve that behavior in parity tests. Revising which physical time level diagnostics report would be a separate numerical change. Emitted records must capture the intended arrays and timestamp before rotation. A host writer cannot reconstruct them from the rotated state by using its new `tau`.

### 3. Compose compiled timesteps

The prepared integrator exposes a pure, jitted `step(carry)` operation. Its body reconstructs the local shell, applies the shared numerical sequence, updates diagnostics, rotates indices, and returns the new carry and event records. `advance(carry, n_steps=...)` composes this operation in a Python loop and optionally stacks records.

JAX differentiates through successive calls to the compiled step. The step executable is independent of the horizon. A host-managed checkpoint schedule can replay timesteps and stitch one-step VJPs while limiting saved-state storage.

```python
prepared = simulation.prepare_integrator()
carry0 = prepared.pack(simulation.state)

def loss(alpha):
    # ACC basic's restoring target is already a dynamic model variable.
    carry = prepared.replace_variables(
        carry0, t_star=alpha * carry0.variables.t_star
    )
    final, records = prepared.advance(carry, n_steps=96)
    return objective(final, records)

value, gradient = jax.value_and_grad(loss)(1.0)
```

`replace_variables` creates a new container. `objective` must select floating arrays and return a scalar. Differentiating a whole carry would encounter integer and Boolean leaves. The supported pattern differentiates selected floating inputs. Perturbing initial tracers additionally requires recomputing consistent density/enthalpy fields and specifying how their multiple time levels and tendency history are initialized.

## Diagnostics that work across steps

Add numerical sampling and finalization helpers to the existing diagnostic modules. A helper takes `(state, diagnostic_variables)` and returns updated variables and, for finalization, an output payload. The ordinary diagnostic methods can call these same helpers and then use their existing writers.

| Diagnostic | Numerical behavior to preserve |
| --- | --- |
| Averages | Increment `average_nitts`, sum selected fields using the pre-rotation index, emit a guarded mean, then reset sums and count. |
| Energy | Reuse `diagnose_kernel`, sum its results, apply the existing `rho_0` normalization at output, then reset selected fields and count. Replace `float(nitts or 1)` with array-safe division. |
| Overturning | Reuse `diagnose_kernel`; carry `sigma`, `zarea`, accumulated transports and streamfunctions, and count. Preserve the current output-variable selection and reset rules. |
| Tracer monitor | Move reductions and updates of previous tracer contents into a numerical helper triggered by the output event. Return values and differences for host formatting. |
| CFL monitor | Move computations from `output()` into a numerical helper, use `npx.maximum` instead of Python `max`, and return CFL values and NaN status. |
| Snapshot | Capture selected model arrays at the output event. No duplicated model-state accumulator. |

Keep sampling before output finalization when both are due. Use `lax.cond` for conditional sampling/finalization so expensive diagnostics are skipped when not due. Both branches must return the same PyTree and array types, and each branch must operate on its own local containers. An inactive branch must not mutate the other branch's state during tracing.

Preserve the existing due predicate for positive frequencies:

```python
due = time % frequency < dt_tracer
```

Handle disabled/zero frequencies statically before any modulo operation. Preserve this predicate because nonintegral frequencies, restart offsets, and floating-point precision affect event timing. Initially keep `dt_tracer` and frequencies fixed and do not differentiate the event schedule.

### Output records and memory

Start with two supported output strategies:

- The pure API returns selected per-step payloads and validity masks, with fixed shapes. Callers explicitly choose retained diagnostics. Disable snapshot histories by default. Full-resolution snapshots at every timestep would dominate memory.
- The production host driver installs each timestep, checks status, writes finalized output events, and updates progress. Sampling and accumulation happen inside the compiled step. This path retains only the current step records.

Both strategies use the same event and finalization code. Accumulators reset inside computation regardless of diskless mode or when the writer runs, so a diagnostic objective sees the same mathematics in either mode. Preserve a differentiable copy of an emitted mean before resetting its accumulator. Record selection and layout are static compilation options.

Adapt the NetCDF writer to accept explicit payload arrays and event times. Retain metadata, masking, scaling, ghost removal, and dimension order. In particular, do not call an existing `output()` method on already finalized results, which would normalize and reset them again.

Restart boundaries save the continuation carry, including unfinished diagnostic windows and rotated time indices. Reuse existing restart field metadata and file groups. Installation into host diagnostic instances occurs only outside transformations. Derive schedules from restored time. Final forced restart and orderly shutdown remain host behavior.

Include `itt` in continuation restart data. Its current variable metadata does not set `write_to_restart=True`, although pressure mode uses it to detect the first step. Define a legacy-file migration explicitly so resumed pressure runs do not accidentally execute first-step initialization. Recovering an iteration count from time is valid only when the timestep and time origin are known.

Avoid putting file writes inside compiled-step callbacks. Pure callbacks can be omitted or repeated, and I/O callbacks have AD restrictions. Returning numerical payloads keeps checkpoint recomputation and repeated gradient evaluations free of writes. See [JAX's callback transformation rules](https://docs.jax.dev/en/latest/201/callbacks.html).

## Differentiation beyond a compilable trajectory

### Pressure and streamfunction solves

Require `linear_solver="scipy_jax"` when preparing this mode on CPU or GPU. Either select it before setup or fail preparation with a specific configuration error. `best` is not sufficient. Prepare the solver on concrete state and attach its immutable handle to the detached shell. Add a narrow prepared-handle path to `get_linear_solver` before its identity cache. Pressure mode also needs explicit preparation because setup does not initialize its solver through `streamfunction_init`.

The current JAX solver already uses BiCGSTAB, whose derivatives are implemented through implicit differentiation and an additional solve. Both primal and adjoint solves must converge for accurate gradients. No bespoke Poisson adjoint is needed for the first fixed-geometry implementation. See [JAX BiCGSTAB](https://docs.jax.dev/en/latest/_autosummary/jax.scipy.sparse.linalg.bicgstab.html).

Compute residual norms explicitly because the documented `info` is currently a placeholder. Validate the original boundary-constrained equation as well as the scaled equation, and test transpose solves on nonsymmetric configurations. The hard-coded `atol=1e-8` needs checking against dtype and RHS scale. Expose suitable tolerances if those tests require them.

Captured matrix coefficients establish gradients through RHS and boundary values for fixed geometry. They do not establish gradients through grid/depth changes or coefficients affecting the operator. Supporting those targets requires passing matrix coefficients as dynamic arrays and constructing a differentiable matvec. Island masks and island count remain fixed even then.

### Tridiagonal solves

Initially require `use_special_tdma=False`. The existing Thomas algorithm implemented with two JAX scans is the reference for values and gradients. Check land rows, bottom edge rows, shallow columns, and both precisions.

Use `lax.custom_linear_solve` for accelerated TDMA derivatives. Define the masked tridiagonal matvec in JAX, use the existing primitive for the primal solve, and transpose the off-diagonals with the correct shifts. Include derivatives of all four floating inputs. For `Ax=d`, the differential is `A dx = dd - dA x`. Masks and column depths are fixed.

Do not mark the operator symmetric unless proved. The transpose must respect bottom coupling and land-row semantics, including the CPU primitive's column-depth handling. Validate JVP/VJP dot products and finite differences before enabling the accelerated path. [Custom linear solve](https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html) supports implicit derivatives without differentiating the custom kernel itself.

### Coefficients and parameterizations

`VerosSettings` coerces values to Python types and enters static PyTree auxiliary data. Changing a setting is not an optimizable input. Start with existing dynamic initial/forcing arrays. For each requested physical coefficient, register a scalar control as a model variable during setup and change only the relevant numerical reads to use it, falling back to the ordinary setting for setups without that control. `kappaH_0`/`kappaM_0` in `tke.py` or `K_iso_0`/`K_gm_0` in `eke.py` are small first targets. Controls overwritten from settings each step would lose their gradients.

Seed controls from their configured settings during setup and mark persistent controls for restart output. The selected control must also reach any derived fields or solver coefficients it affects. Changing a leaf after a setup-time computation does not retroactively differentiate that computation.

Add broader dynamic parameter plumbing only when several supported controls make it worthwhile. Any such parameters must be dynamic children across nested kernel calls. Hiding them in static auxiliary objects or mutable settings is unsafe.

Audit finite gradients on representative physical states. TKE/EKE contain expressions such as `sqrt(maximum(0, energy))`. Zero-energy cells can have singular derivatives, and masking after an invalid operation need not repair its gradient. Use safe operands or a documented local derivative convention where needed, with dedicated value and gradient checks. Do not apply global smoothing without evaluating its numerical effect.

Advection limiters, clipping, ice masks, and density-bin membership introduce nonsmooth behavior. Automatic differentiation follows the implemented branches and does not differentiate Boolean membership changes. Overturning in density coordinates uses hard threshold masks, so it cannot promise a useful gradient for moving density across bins. Depth-coordinate overturning, smooth tracer averages, and energy measures are better initial objectives. Smooth binning would be an explicit additional diagnostic option with different numerical semantics.

### Reverse-mode memory

Returning only the final carry avoids an explicit trajectory, but reverse-mode AD still saves intermediates. Start by measuring memory with and without `jax.checkpoint` on the step body. Recomputing a pure step is safe because it performs no output. See [JAX gradient checkpointing](https://docs.jax.dev/en/latest/gradient-checkpointing.html).

Checkpointing each step does not imply constant-memory gradients. Long integrations use a deliberate trajectory checkpoint schedule. RAM or disk storage retains continuation states, replay recomputes missing states, and the backward traversal applies a compiled one-step VJP to each incoming cotangent. Parameter cotangents sum over timesteps. The checkpoint slot budget excludes live states, adjoints, step-local differentiation buffers, and executable caches.

## Hooks, failures, and MPI

Forcing, `after_timestep`, and plugin run entrypoints may reuse existing routine syntax if they only modify declared model arrays and use traceable operations. Their captured configuration is fixed. Move file access, logging, and asset preparation to the host/setup path. Hidden mutable `self` fields, Python branching on time, and undeclared state updates need individual adaptation. Preparation validates known capabilities and a representative trace. Arbitrary Python hooks cannot be guaranteed compatible by decorator inspection alone. Reject unsupported plugins or diagnostics with specific errors.

Replace the Python sanity exception with carried status. Preserve the current sanity predicate first, record the first failure, and freeze the carry on subsequent iterations. Fill subsequent records as invalid. The host driver raises with the recorded iteration after the timestep returns. The pure API reports status and callers must reject failed trajectories when evaluating objectives. Failure and event masks are discrete. Optional broader finiteness checks should be separate from parity work.

MPI needs a targeted audit of `veros/distributed.py`, not a replacement with JAX sharding:

- Pin a compatible JAX/mpi4jax pair and verify signatures, effect ordering, and step differentiation. The repository pins JAX in `requirements_jax.txt`, while solver and custom primitive APIs are version-sensitive.
- Prove JVP/VJP support for every communication operation on the numerical path. The documented allreduce differentiation support is for `MPI.SUM`. Monitor `MAX`/logical reductions should be separated from differentiable objectives. See the [mpi4jax API](https://mpi4jax.readthedocs.io/en/latest/api.html).
- Audit halos and their adjoints, including corner overlap. The backward exchange must accumulate cotangents for duplicated values correctly. Validate the existing send/recv-based gather/scatter used by the root-rank JAX pressure solve. Use custom derivative rules or compatible alternatives only where required.
- Ensure diagnostic branches and failure decisions are consistent across ranks. All ranks must participate in matching communication during forward execution, adjoint solves, and checkpoint recomputation.
- Keep communicator construction and host barriers outside tracing. Use separate communicators for compiled numerical traffic and host I/O/control. The mpi4jax documentation describes ordering hazards when sharing a communicator with mpi4py in its [usage cautions](https://mpi4jax.readthedocs.io/en/latest/sharp-bits.html).
- Compare distributed gradients against a single-rank reference and account for replicated global objectives so sensitivities are not multiplied by the number of ranks.

Retain the root-rank JAX pressure solve for the first distributed version if it passes these gates. A distributed elliptic solver is a scalability project and should not be bundled into this implementation. Reverse-mode MPI and checkpoint recomputation remain the least certain parts of the proposal.

## Delivery and acceptance

| Increment | Main files | Acceptance gate |
| --- | --- | --- |
| Pure compiled timestep | New `integration.py`; `veros.py`, `routines.py`, `timer.py`, `state.py`; solver preparation and pressure branch | Six calls reuse one compiled timestep, cover every index rotation twice, preserve their inputs, and match the existing runner. Repeated calls do not leak tracers or initialize additional solvers. |
| Working diagnostics and host output | Diagnostic API/base and six modules; small NetCDF writer adapter; restart/runner boundary | All built-in diagnostics match values, counts, sampling/output times, resets, file records, and continuation across step/restart boundaries. |
| Verified single-device gradients | Solver residual reporting; targeted numerical fixes; selected controls | Jitted JVP/VJP and scalar gradients pass solver and short-trajectory checks, including a diagnostic objective and a composition of compiled timesteps. |
| Practical longer trajectories | Integration checkpoint policy and benchmark | Measured compile time, throughput, and backward memory for representative ACC and global configurations, with a documented tested horizon. |
| Accelerated TDMA and MPI | TDMA wrapper; targeted distributed adapters | CPU/GPU accelerated gradients match the reference; multiple ranks pass numerical, communication-adjoint, and checkpoint tests. |

Add focused tests in `test/integration_test.py` and `test/diagnostics_test.py`, and extend existing state, linear-solver, restart, and distributed tests where appropriate:

- Compare all active arrays after one step, six steps, and split runs. Include first-step pressure behavior, streamfunction islands, cyclic boundaries, and enabled EKE/TKE/IDEMIX branches. Use tolerance-based comparisons because fusion and iterative solves can change rounding.
- Exercise independent inputs and prepared instances, cold/warm tracing, `eval_shape`, `jit`, JVP/VJP, checkpointing, and exception cleanup. Ensure unchanged signatures reuse compiled execution, and no host writes occur during tracing, execution, or differentiation.
- For each diagnostic, include disabled sampling, output without samples, simultaneous sample/output, non-integral frequencies, restored time offsets, reset windows, and consecutive timestep events. Compare NetCDF payload values, timestamps, masks, dimensions, and record counts to eager output.
- Check finite-difference directional derivatives of smooth short-run losses in float64 over several perturbation sizes. Include a nonzero forcing sensitivity, an accumulated average spanning output resets, solver boundary sensitivity, and masked TDMA coefficient derivatives. Use JVP/VJP dot-product tests alongside finite differences.
- Test real zero-energy/land cells separately from smooth fixtures. Characterize density-bin and limiter gradients with checks appropriate to their nonsmooth behavior.
- Benchmark compilation separately from synchronized warm execution and backward passes. Compare against the existing kernel-jitted runner at identical precision, solver settings, diagnostics, and step counts. Record peak memory and test that forward and reverse timestep executables are reused across horizons.

As a planning estimate for one contributor familiar with Veros, allow roughly 1–2 engineer-weeks for a single-device proof with a smooth gradient, another 2–3 for all built-in diagnostics/output/restart parity, and another 1–3 for parameterization gradient fixes, representative coverage, and memory measurements. Accelerated TDMA and MPI could add several weeks and should be estimated after primitive-level probes. These are scope estimates, not measured implementation costs or speedup predictions.

The first prototype should cover the detached-state timestep, prepared JAX solver, pure TDMA, and averages, then test composition through the wrappers and PyTree metadata. This proposal is based on source inspection and JAX/mpi4jax documentation. Runtime compatibility requires separate verification.
