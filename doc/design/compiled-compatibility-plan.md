# Single-step execution compatibility plan

The implementation provides automatic single-step JIT for supported JAX configurations, legacy host diagnostics, and first-order checkpointed gradients. The ordinary runner provides forward compatibility for configurations that need unrestricted Python hooks, NumPy, or the existing SciPy solver. Execution selection happens during preparation and reports its result and reason.

Acceptance checks cover short full-physics CPU and GPU trajectories and distributed first-order replay on one, two, and four CPU ranks and one and two GPU ranks. The [verification report](compiled-compatibility-validation.md) records the tested configurations, memory peaks, and limits, including the separate long-horizon neutral-mixing sensitivity.

| Capability | Delivered support | Completion criterion |
| --- | --- | --- |
| Native JAX numerical hooks | One fused compiled timestep | Forward states agree with ordinary execution and supported gradients are validated |
| Arbitrary Python hooks and plugins | Ordinary-runner fallback selected during preparation | Existing behavior remains available and selection is reported |
| Legacy custom diagnostics | Host execution for diagnostics that observe model state | Sampling, output, resets, restart state, and diskless behavior agree |
| Continuous controls and forcing | Explicit step inputs and carried state | Original execution and replay consume identical inputs |
| Structural configuration changes | Explicit trajectory segments and repreparation | Changes take effect after a documented state handoff |
| NumPy and the existing SciPy solver | Ordinary execution | Explicit backend and solver requests are honored |
| Timing and profiling | Timestep and host-operation timers, device traces, ordinary-runner routine timers | Measurements identify the execution path they describe |
| Root-only MPI hooks | Ordinary execution for configurations requiring them | Existing gather/root/scatter behavior remains available |
| Checkpointed gradients | First-order replay through individual compiled steps | Gradients agree with reference derivatives within bounded storage |
| Distributed checkpoint replay | Coordinated first-order replay with local checkpoints on each rank | States and gradients agree with a single-process reference |

1. **Select execution during preparation.** Provide automatic selection, an explicit compiled mode, and an explicit ordinary/debugging mode. Automatic selection uses the compiled path when the backend, solver, numerical hooks, plugins, and diagnostics meet its supported contracts. Otherwise it selects the ordinary runner for the run and reports the reason. Explicit compiled requests fail with an actionable explanation when unsupported.

   Keep eligibility checks focused on the existing extension points. Numerical hooks intended for compiled execution must explicitly declare that they support native JAX tracing. Successful tracing alone cannot establish that a Python hook has safe side effects. Use focused adapters for workloads that need them.

   Differentiation has a separate capability check. A gradient request must fail before a long traversal if an operation affecting the trajectory lacks a derivative rule. Execution selection must never silently substitute a solver or fall back to an unsupported gradient path.

2. **Restore legacy host diagnostics and per-timestep reporting.** Permit existing `diagnose` and `output` methods through a focused compatibility path for diagnostics that observe model state and mutate only their diagnostic accumulators. Preserve the state and time indices seen by ordinary execution, including the pre-rotation view. Reuse its frequency predicates and preserve initialization, accumulator resets, restart fields, diskless behavior, and error propagation. Share array storage where possible to avoid complete state copies each timestep.

   Diagnostics that mutate model variables require ordinary execution unless specifically adapted. Diagnostics used in a differentiated objective use the pure sampling/finalization interface. Host diagnostic side effects, progress, output, and persistent restart writes run on the original forward traversal and are excluded from adjoint recomputation.

   Restore lifecycle logs and timings for whole timesteps and host operations. Use device trace annotations for compiled numerical profiling, with the existing ordinary runner available for routine-level timing. Compilation time should be distinguishable from steady-state execution. See [JAX profiling](https://docs.jax.dev/en/latest/profiling.html).

3. **Expose continuous controls and reproducible forcing inputs.** Keep supported continuous parameters as numerical inputs. Allow timestep-dependent forcing through an explicit input interface. Randomness and any numerical history needed by the next timestep belong in the carried state or explicit inputs. Keep structural configuration fixed within each prepared trajectory.

   File-derived forcing should be supplied as recorded values or immutable, verified references, so replay does not reread changed data. Budget large forcing records across RAM and disk. Differentiated runs require explicit inputs and carried numerical state.

   Changing physics choices, numerical hooks, solver, or array structure ends the current trajectory segment. The caller hands off state and reprepares explicitly. Document this boundary and reject unsupported changes within a prepared segment. Automatic differentiation across structural transitions is outside the initial milestone.

4. **Validate first-order checkpointed gradients and memory accounting.** Retain host orchestration of reusable compiled timesteps and their VJPs. Checkpoint placement and recomputation intervals are measured in timesteps and remain independent of compilation. The first milestone supports gradients with respect to supported continuous parameters and initial state, including terminal and differentiable diagnostic objectives.

   Validate schedule behavior, replay inputs, parameter accumulation, and RAM/disk storage with small recurrence problems. Use finite differences at several perturbation sizes and JVP/VJP dot products to check step derivatives. Compare checkpointed gradients with direct trajectory gradients on small ocean cases, including both pressure formulations, TKE, neutral/skew mixing, diagnostics, and supported solvers. Validate every external numerical primitive needed by those cases.

   Track checkpoint storage, forcing records, live state, adjoints, step-local buffers, and executable caches separately. Checkpoint slot limits do not bound total memory, so include byte estimates and measured peaks in acceptance results. Acceptance applies to the configurations and horizons in the verification report. Scientific applications require their own trajectory checks.

5. **Release automatic execution selection after compatibility checks pass.** Compare ordinary and compiled runs using native hook fixtures and small ocean configurations. Cover legacy diagnostic schedules and resets, restart recovery, per-timestep output, forcing inputs, failure handling, and fallback selection for arbitrary hooks, NumPy, explicit SciPy requests, and root-only MPI routines. Verify that forward observers run once and that replay uses the original numerical inputs.

   Run numerical checks individually in fresh processes under a process-tree memory guard. Start with a 3 GiB process limit and a 4 GiB available-system-memory reserve, and bound GPU allocation. Adjust limits only after measuring the workload. A watchdog stop is an incomplete check. Full 1-degree experiments wait for adequate hardware.

   Make automatic selection the default after these forward checks and the first-order gradient release gate pass. Report the selected execution path and differentiation capabilities. Preserve explicit ordinary execution for compatibility and debugging.

6. **Add distributed first-order checkpoint replay for large-model workloads.** Reuse the existing distributed numerical step and its derivative rules. Have every rank follow the same schedule while storing its local state, with consistent configuration and forcing inputs. Handle rank-local controls, shared-parameter reductions, halo adjoints, collective failures, and checkpoint cleanup explicitly.

   Validate on small CPU problems using one, two, and four ranks before GPU tests. Compare forward states and gradients with a single-process reference and include global diagnostic objectives. This milestone supports numerical operations with existing derivative contracts. Arbitrary root-only Python hooks continue to use ordinary execution.

The following capabilities are outside the planned milestones. They can be reconsidered when a concrete workload establishes their value.

| Capability | Scope decision and alternative |
| --- | --- |
| Compact outer JIT around an entire trajectory | Omit from the initial scope. Use host orchestration of compiled steps and stitched gradients. |
| General replay of arbitrary stateful Python hooks | Avoid a general capture framework. Use explicit inputs and carried state for differentiated runs, or ordinary execution for legacy hooks. |
| Transparent structural changes during a trajectory | Defer. Use explicit segments, state handoff, and repreparation. |
| The legacy SciPy algorithm inside JIT with gradients | Defer a dedicated adapter. Preserve ordinary SciPy execution and use supported JAX/PETSc solvers for compiled trajectories. |
| A special compiled mode reproducing routine-level timers | Omit. Use timestep/host timers, device profiling, and the ordinary runner's routine timers. |
| Arbitrary higher-order AD and batching through RAM/disk replay | Defer. Deliver first-order checkpointed gradients, then consider an explicit Hessian-vector API for an application that needs it. Existing native JAX transformation support remains available where it already works. |
| General differentiation through arbitrary root-only MPI hooks | Defer. Preserve forward behavior through ordinary execution and add specific numerical derivative adapters when needed. |
