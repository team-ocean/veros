Compiled integration and differentiation
========================================

The JAX backend compiles one numerical timestep, including forcing, physics,
diagnostic sampling, and diagnostic resets. Python composes those timesteps
for execution and differentiation. Setup and file output run on the host.
:meth:`veros.VerosSetup.run` selects execution automatically and reports the
selected path.

Running a compiled simulation
-----------------------------

Initialize your setup as usual. ``run()`` selects compiled execution for supported
JAX configurations and ordinary execution otherwise. Use ``run(execution="eager")``
for ordinary execution or ``run(execution="compiled")`` to require compilation.
For an explicit timestep count, call ``run_compiled``::

    from veros import runtime_settings
    runtime_settings.backend = "jax"

    from veros.setups.acc_basic import ACCBasicSetup
    simulation = ACCBasicSetup(
        override=dict(nx=16, ny=16, nz=15, dt_mom=600, dt_tracer=600)
    )
    simulation.setup()
    simulation.run_compiled(n_steps=24)

The driver installs results into ``simulation.state`` and writes configured
diagnostics and restarts. It checks status and updates progress after each
step. Snapshot records use the ordinary runner's time slot. ``n_steps=None``
advances for ``settings.runlen``, rounded up to a whole step. A final restart
is written even when execution is interrupted.

The corresponding command line invocation is::

    veros run acc_basic.py --backend jax

Use ``--execution eager`` for ordinary execution. ``--execution compiled`` or
``--compiled`` requires a supported JAX configuration. Explicit SciPy solver
requests use ordinary execution. Compiled solver overrides remain explicit.

Use the repository's ``requirements_jax.txt`` for the tested JAX version. MPI
integration additionally requires ``mpi4py>=4.1`` and ``mpi4jax==0.9.1.post2``.
These are available through the ``jax-mpi`` installation extra. Native TDMA
extensions are optional. Both the JAX implementation and accelerated CPU/GPU
implementations support forward and reverse differentiation.

Numerical setup hooks and plugin entrypoints must declare their compilation
contract with ``@veros_routine(supports_compilation=True)``. This declaration
promises native JAX operations and no Python side effects during timesteps.
Unmarked numerical hooks select ordinary execution without being traced.
Root-only numerical hooks also select ordinary execution under MPI.
Setups overriding ``step()`` use ordinary execution so their custom timestep
behavior is preserved.

Legacy diagnostics can declare ``observes_model = True`` to run their existing
``diagnose`` and ``output`` methods on the host. They may update their own
accumulators but must preserve model variables and settings. Their sampling
frequencies and pre-rotation time indices match ordinary execution. Losses
using diagnostic values require the pure sampling/finalization interface.
To retain compiled finalization, a subclass overriding a compiled diagnostic's
``output()`` must declare its own ``supports_compilation = True`` contract and
provide the equivalent behavior through ``finalize`` and ``write_finalized``.

The driver reports the first timestep separately, including compilation. Its
steady-state timers cover synchronized timesteps, host forcing providers,
diagnostics, and I/O, excluding the first iteration. Device profiles include a
trace annotation for each timestep. Use ordinary execution when routine-level
host timings are needed.

The :download:`compatibility verification report <../design/compiled-compatibility-validation.md>`
records the CPU, GPU, MPI, restart, replay, and memory checks and their limits.

A pure numerical API
--------------------

For differentiation, prepare an integrator and keep its continuation state
inside the JAX computation::

    import jax
    import jax.numpy as jnp

    averages = simulation.state.diagnostics["averages"]
    averages.sampling_frequency = simulation.state.settings.dt_tracer
    averages.output_frequency = 3 * simulation.state.settings.dt_tracer
    prepared = simulation.prepare_integrator(
        record_diagnostics=("averages",),
        checkpoint=True,
    )
    initial = prepared.pack(simulation.state)

    def objective(scale, initial):
        carry = prepared.replace_variables(
            initial, t_star=scale * initial.variables.t_star
        )
        final, records = prepared.advance(carry, n_steps=24)
        temperatures = records["averages"].values["temp"]
        emitted_means = temperatures[:, 2:-2, 2:-2].sum(axis=(1, 2, 3))
        loss = jnp.sum(jnp.where(records["averages"].valid, emitted_means, 0))
        return loss, final

    (value, final), gradient = jax.value_and_grad(objective, has_aux=True)(
        1.0, initial
    )
    prepared.check_status(final)

``t_star`` is the surface restoring target used by the ACC basic setup. Other
setups have their own forcing variables. Configure ``output_variables`` before
setup so the diagnostic allocates the required arrays. Schedules can also be
changed before preparation, as above.

``IntegrationState`` contains model variables, diagnostic accumulators, a
failure code, and the first failed iteration. ``advance`` returns a new carry
and a dictionary of ``DiagnosticRecord`` objects. Each record contains ``time``,
``iteration``, ``valid``, and a dictionary of output arrays. Histories have a
leading step dimension, including invalid records filled with zeros. The
integrator never installs results or writes files during numerical execution.
Use ``prepared.check_status(final)`` on the host to reject failed trajectories,
then ``prepared.install(final)`` and ``prepared.write_records(records)`` when
output is wanted.

All six built-in diagnostics support this path. By default, records retain
active outputs except snapshots. Sampling and window resets still run for
unrecorded diagnostics. Select snapshots explicitly when their arrays are
needed. ``final_only=True`` avoids a full output history and returns the record
for the last step, which may be invalid. ``prepared.step(carry)`` returns one
compiled timestep and its records without a leading step dimension. The host
driver uses this operation and handles every output event immediately.
``None`` and zero disable a schedule. Infinite output intervals allow
accumulation without periodic output.

``advance`` loops in Python and reuses the same compiled timestep for every
horizon. ``jax.grad`` and ``jax.jvp`` compose derivatives through these calls
without requiring an outer ``jax.jit``. Jitting the whole trajectory would
unroll the Python loop and create a horizon-specific executable. For long
horizons, use the checkpoint driver below to control saved-state memory.
Pure continuation calls compose inside a differentiated function::

    def advance_twice(carry):
        middle, first_records = prepared.advance(carry, n_steps=12)
        final, second_records = prepared.advance(middle, n_steps=12)
        return final, (first_records, second_records)

Packing, installation, file writes, and restart reads are host boundaries.
Loading a restart continues its diagnostic windows and time indices but does
not reconnect gradients to the computation that produced the file. Legacy
restarts without an iteration count recover it from ``time / dt_tracer`` only
when that ratio is integral.

Dynamic inputs and fixed configuration
--------------------------------------

Floating model variables and scalar floating settings can be dynamic inputs.
Use ``replace_variables`` to change initial tracers or forcing arrays. Pass
setting overrides separately to ``advance``::

    parameters = {"c_k": jnp.array(0.1), "c_eps": jnp.array(0.7)}

    def parameter_loss(parameters, initial):
        final, _ = prepared.advance(
            initial, n_steps=24, parameters=parameters, final_only=True
        )
        vs = final.variables
        return jnp.sum(vs.temp[2:-2, 2:-2, :, vs.tau] ** 2), final

    (value, final), parameter_gradient = jax.value_and_grad(
        parameter_loss, has_aux=True
    )(parameters, initial)
    prepared.check_status(final)

Kernels and hooks read these values through their existing ``state.settings``
attributes. Overrides do not mutate host settings or add model variables.
Omitted names use the values captured at preparation. Changing values with the
same parameter names, shapes, and normalized dtype reuses compiled execution.
Changing the set of names creates a different compiled signature. A parameter
unused by the enabled physics has zero sensitivity.
Pass overrides to each continuation call. The carry does not retain them.

Array dimensions, enabled physics, diagnostic schedules, masks, and island
topology remain fixed. Solver coefficients are fixed unless partial cells are
enabled as described below. Overrides reject
nonfloating settings, nonscalar values, and settings that define frozen grid,
reference data, or execution schedules: ``pi``, ``radius``, ``degtom``,
``omega``, ``x_origin``, ``y_origin``, ``rho_0``, ``grav``, ``dt_mom``,
``dt_tracer``, ``runlen``, and ``restart_frequency``. Setup and plugin authors
must also list settings used to construct cached data in
``prepare_integrator(fixed_parameters=(...))``.

Overrides differentiate integration from the supplied initial carry. They do
not rerun initialization, rebuild the pressure operator, or differentiate
setup-time preprocessing. Prepare again when fixed data changes. Update related
forcing leaves together when a setup computes derived forcing during
initialization. A positive parameter can be optimized in log coordinates by
constructing its override with ``jnp.exp`` inside the objective.

Restart-persistent scalar controls can also be registered at construction::

    simulation = ACCBasicSetup(
        control_parameters=("kappaH_0", "kappaM_0", "K_iso_0", "K_gm_0")
    )

These create restart-persistent scalar variables named ``control_kappaH_0`` and
so on. Their initial values come from the configured settings. Replace a
control leaf in the carry to differentiate its use in the corresponding
parameterization. A control has no sensitivity when the selected physics
branch does not use it. For example, the constant diffusivity controls are
used when their prognostic TKE/EKE parameterization is disabled.
An override of a setting with an existing ``control_*`` variable is rejected.
Choose one input path for that setting.

Forcing, ``after_timestep``, and plugin run hooks may update declared model
variables with JAX operations. They must avoid Python branches on array values,
file access, logging, and hidden mutable state, and leave time and rotating
indices to the integrator. Preparation traces a step and reports transformation
errors before execution. Hook authors must ensure purity because tracing cannot
detect every Python side effect. Diagnostics used in differentiated objectives
must declare ``supports_compilation=True`` and provide numerical ``diagnose``
and ``finalize`` methods. ``finalize`` returns output arrays and resets the
accumulator. ``write_finalized`` handles host output. Host observers use the
legacy compatibility path described above. Other active diagnostics and
root-only distributed routines select ordinary execution in automatic mode and
are rejected by explicit preparation.

Bathymetry controls
-------------------

``enable_partial_cells=True`` enables continuous bottom-depth controls within
fixed wet cells. Configure this before ``setup()``. The supported configuration
uses serial JAX execution, the ``scipy_jax`` pressure solver, surface pressure,
constant mixing, and ``enable_conserve_energy=False``. TKE, EKE, IDEMIX, neutral
and skew diffusion, TEM friction, horizontal momentum friction, biharmonic
mixing, streamfunction mode, and active energy/overturning diagnostics are
rejected. Tracer advection, horizontal tracer diffusion, vertical mixing,
vertical momentum friction, wind forcing, and quadratic bottom drag use partial-cell
volumes or open face areas. Tracer-monitor volumes and vertical CFL estimates
also use the occupied thicknesses.

The reference z levels and vertical difference distances remain fixed.
Only the bottom cell's occupied thickness changes. Density and hydrostatic
pressure use the reference levels. This is a partial-step discretization,
not a terrain-following coordinate system. Neighboring velocity faces use the
minimum occupied thickness of their adjacent tracer cells. Depth-dependent
pressure coefficients are explicit numerical inputs to the implicit linear
solve, so derivatives include changes in both its operator and right-hand side.

Build candidate geometry inside the differentiated objective::

    initial = prepared.pack(simulation.state)
    lower, upper = prepared.bathymetry_bounds(initial, minimum_fraction=0.1)

    def bathymetry_loss(control):
        depth = lower + (upper - lower) * jax.nn.sigmoid(control)
        candidate = prepared.with_bathymetry(initial, depth)
        final, _ = prepared.advance(candidate, n_steps=96)
        vs = final.variables
        misfit = jnp.mean((vs.psi[2:-2, 2:-2, vs.tau] / 9.81 - observed_ssh) ** 2)
        return jnp.where(final.failure_code == 0, misfit, jnp.nan), final

    (value, final), gradient = jax.value_and_grad(
        bathymetry_loss, has_aux=True
    )(control)
    prepared.check_status(final)

Depth arrays contain the interior grid, excluding halos, and use positive
meters below the surface. Land depths must be zero. The lower bound preserves
at least ``minimum_fraction`` of each wet bottom cell. The upper bound fills
that cell completely. Invalid depths set failure code 4 and freeze the
trajectory. Geometry persists in the carry through continuation, installation,
restart output, and RAM/disk checkpoints.
Restarts require the same reference grid and wet-cell indices as the saved run.

``with_bathymetry`` preserves tracer concentrations, velocities, and tendency
histories. If initial profiles or balanced velocities depend on depth, construct
them inside the objective after updating geometry. Changing cell volume at
fixed concentration changes total tracer content. A conservative remapping is
a different initialization policy and must be supplied by the experiment.
For host-managed adjoints, apply the control-map VJP to ``result.gradient`` to
obtain the depth-control gradient, as for other initial-state controls.

Within fixed topology, gradients are local derivatives of this discretization.
The minimum of neighboring depths is nonsmooth where the shallower side
changes. Bottom-cell transitions, moving coastlines, and opening or closing
passages require a new topology and preparation. This API does not differentiate
those discrete changes. Validation of a particular optimization also requires
appropriate initial conditions, observations, and depth priors.

The executable ``bathymetry_optimization`` acceptance demo fits a known sill
depth perturbation in an 8 x 10 x 4 basin. It checks forward/reverse AD, finite
differences, Taylor remainders, depth recovery, a double-length held-out
trajectory, and RAM/disk replay. Run it with::

    python -m benchmarks.ad_demos --demo bathymetry_optimization \
        --mode acceptance --output /tmp/bathymetry-acceptance

Its eight-hour acceptance window establishes this numerical inverse workflow.
It does not establish bathymetry identifiability in the global 4-degree model.

Gradient and memory behavior
----------------------------

The JAX pressure solver uses implicit derivatives of its linear operator. Each
primal, tangent, and adjoint right-hand side is normalized and solved with a
fresh relative tolerance. BiCGSTAB uses a zero initial guess, with GMRES as a
fallback when its residual check fails. The integrator reports failed forward
solves. TDMA derivatives include all three coefficient arrays, the right-hand
side, and the transposed solve. Zero-energy, zero-wind, and zero-salinity
square roots use a zero local derivative at zero while preserving forward
values. IDEMIX's clipped inverse hyperbolic cosine uses the same convention at
its lower boundary. Small-denominator slope ratios use an equivalent division
derivative that avoids underflow when squaring their denominator in reduced
precision.

Clipping, advection limiters, ice masks, and density bins remain nonsmooth.
Automatic differentiation follows their implemented branches. Density-coordinate
overturning does not differentiate movement across bin boundaries. Use
depth-coordinate overturning or smooth tracer objectives when that sensitivity
matters. Finite-difference agreement can deteriorate when a perturbation changes
one of these branches. In testing, ACC basic with its default TKE failed
forward/reverse derivative agreement at 96 steps. The failure is tied to
neutral slopes in unstably stratified columns, where the slope denominator
becomes ``-1e-20`` and floating-point differences can change the taper sharply.
An experimental policy that sets these slopes to zero restored derivative
agreement in this configuration, but it changes the mixing calculation and is
not the model default. The
:download:`investigation report <../design/tke-gradient-investigation.md>`
records the evidence and reproducer. The raw measurements below retain the
failed check separately from the validated runs. Check both derivative modes
and several finite-difference scales for the configuration and horizon you
intend to use.

``checkpoint=True`` recomputes intermediates within each step in the backward
pass. It does not bound the number of saved trajectory states. Snapshot
histories and requested diagnostic records also consume memory. For long
reverse-mode computations, the host-managed checkpoint driver stitches
one-step VJPs with RAM or disk replay under an explicit storage budget.

The :download:`PETSc performance report <../design/petsc-performance.md>`
compares CPU, GPU, and MPI execution, native KSP cost, and TDMA backends. Its
:download:`raw results <../../benchmarks/petsc-performance-results.json>`
include repeated warm timings and scientific parity failures. The
:download:`surface neutral-slope investigation
<../design/neutral-parity-investigation.md>` and its :download:`measurements
<../../benchmarks/neutral-parity-investigation.json>` trace those failures to
coefficient sensitivity at the surface boundary. Native CPU TDMA kernels avoid
per-column Python exception checks. The automatic ``use_special_tdma=None``
setting selects an available extension. Explicit ``True`` and ``False``
overrides select native and pure JAX implementations. Benchmark both choices
for your configuration.

Run ``benchmarks/integration_benchmark.py`` to measure preparation, first-call
compilation, synchronized forward/backward execution, eager execution, peak
process RSS, and device allocator memory where available. Use separate processes
when comparing checkpoint policies. Compilation can dominate short runs.

Measured examples
~~~~~~~~~~~~~~~~~

The following runs used JAX/jaxlib 0.11.2, float64, and accelerated TDMA.
CPU runs used four logical CPUs of an AMD Ryzen 5 5600. The GPU was an RTX
3060 with 12 GB. Each configuration ran in a fresh process. Every run checked
all model variables against eager execution, finite differences at two scales,
and forward/reverse derivative agreement.

.. list-table:: Synchronized warm execution and estimated backward buffers
   :header-rows: 1

   * - Setup/device
     - Steps
     - Checkpointing
     - Forward (s)
     - Value + gradient (s)
     - Buffers (MiB)
   * - ACC (CPU)
     - 24
     - none
     - 0.121
     - 0.500
     - 557.8
   * - ACC (CPU)
     - 24
     - step + block 6
     - 0.121
     - 0.602
     - 82.7
   * - ACC, fixed mixing (CPU)
     - 96
     - step + block 8
     - 0.405
     - 1.955
     - 92.8
   * - Global 4-degree (CUDA)
     - 24
     - step + block 6
     - 0.498
     - 1.867
     - 1078.6

ACC grids were 16 x 16 x 15 with 600-second steps. The global grid was
90 x 40 x 15 with 3600-second tracer steps and 1800-second momentum steps.
The 96-step ACC case used fixed vertical mixing with TKE disabled.
Buffer sizes are XLA estimates for arguments, outputs, and temporaries after
alias accounting. They exclude compilation and executable caches. Process RSS
and device allocator peaks in the
:download:`raw measurements <../../benchmarks/integration-results.json>`
include compilation and validation. The benchmark script reports cold-call
costs separately. These examples took substantially longer to compile than to
execute.

Explicit forcing inputs
-----------------------

``prepared.step(carry, step_inputs=forcing)`` accepts a mapping of forcing
variable names to arrays with the prepared shapes. These arrays replace the
corresponding forcing fields after ``set_forcing`` and before physics. Supported
fields include surface momentum, heat and salt fluxes, TKE/IDEMIX forcing, and
temperature/salinity sources when enabled. ``advance`` accepts a sequence with
one mapping per timestep. Its pure trajectory can differentiate these arrays::

    forcing = {"forc_temp_surface": jnp.zeros_like(initial.variables.forc_temp_surface)}
    final, records = prepared.advance(
        initial, n_steps=3, step_inputs=[forcing, forcing, forcing]
    )

Keep structural settings fixed within a prepared trajectory. Changing physics,
grid dimensions, timestep settings, or solver requires explicit state handoff
and repreparation. Continuous setting overrides remain available through
``parameters``. Native hooks needing randomness or numerical history must carry
that state explicitly.

Disk and RAM checkpoint schedules
---------------------------------

``veros.checkpointing.CheckpointedIntegrator`` performs first-order reverse
integration with a host-managed schedule. It reuses one compiled forward step
and one compiled step VJP, propagating the state cotangent backward and summing
parameter cotangents. Checkpoint placement and storage determine replay::

    from pathlib import Path
    from veros.checkpointing import CheckpointedIntegrator, PeriodicDiskSchedule

    def terminal_objective(final):
        return jnp.sum(final.variables.temp[2:-2, 2:-2, :, final.variables.tau])

    directory = Path("adjoint-checkpoints")
    directory.mkdir(exist_ok=True)
    driver = CheckpointedIntegrator(
        prepared, terminal_objective=terminal_objective
    )
    schedule = PeriodicDiskSchedule(
        24, disk_interval=4, ram_checkpoints=2
    )
    result = driver.value_and_grad(
        initial, n_steps=24, schedule=schedule, directory=directory
    )
    forcing_scale_gradient = jnp.sum(
        result.gradient.variables.t_star * initial.variables.t_star
    )

For externally supplied forcing, pass an iterable through ``step_inputs``.
The driver consumes it once on the original forward traversal and records the
normalized arrays for replay. Set ``input_storage="disk"`` and provide
``directory`` to offload these records. The recorded inputs are exogenous to
this first-order driver. Use the pure trajectory API for input-array gradients.
``observer(index, carry)`` runs once per original forward timestep, including
for observations or host diagnostics, and is excluded from recomputation.

``max_checkpoint_ram_bytes`` and ``max_input_ram_bytes`` limit host storage
before allocating each RAM record. Statistics include checkpoint peaks,
``state_bytes``, ``adjoint_bytes``, ``parameter_bytes``, and recorded-input peaks.
``forward_memory`` and ``reverse_memory`` report XLA estimates for arguments,
outputs, aliases, temporaries, and generated code, when available. These
estimates describe a compiled signature and exclude the host compilation cache.
Live states, compiled executables, and step-local differentiation buffers need
additional memory beyond those storage limits.

With MPI, each rank stores its local state and follows the same schedule.
Objectives must return the same global scalar on every rank, for example by
using ``global_sum`` inside ``tracing_context()``. The driver divides adjoint
seeds across ranks and sums shared parameter gradients exactly once. Parameters
are shared by default. ``shared_parameters=()`` selects rank-local parameters.
``shared_variables=("control_kappaH_0",)`` similarly reduces cotangents of
replicated initial controls. Other initial-state cotangents are rank-local.
Host validation and storage failures propagate to every rank, and each rank
cleans its owned checkpoint directory.

When differentiating a replicated global loss directly with JAX under MPI,
seed ``loss / runtime_state.proc_num`` on each rank and sum gradients of shared
controls across ranks. Distributed floating-point sums use a summing transpose,
so seeding every copy with one would multiply the shared sensitivity.

The result contains ``value``, ``gradient``, ``final``, ``statistics``, and
``parameter_gradient``.
``gradient`` is the cotangent of the initial carry. Integer and Boolean leaves
have JAX's ``float0`` dtype. Supported inputs and fixed configuration are the
same as for the pure integration API. Pass runtime setting overrides with
``driver.value_and_grad(initial, n_steps=24, parameters=parameters, ...)``.
``result.parameter_gradient`` has the same keys and includes their contribution
from every reverse step. It is an empty dictionary when no overrides are
provided. Objective callbacks may accept a ``parameters`` keyword argument for
direct parameter dependence, such as a regularization term::

    def terminal_objective(final, *, parameters):
        vs = final.variables
        return (jnp.sum(vs.temp[2:-2, 2:-2, :, vs.tau] ** 2)
                + 0.01 * parameters["c_k"] ** 2)

Callbacks must be pure. If the same optimization parameters also map to the
initial carry, apply that initialization function's VJP to ``result.gradient``
and add it to the runtime setting gradient. Parameters are fixed throughout a
replay traversal and are not written into carry checkpoint files or model
restarts. Replay accumulates objective values only during the original forward
traversal and parameter cotangents only during the reverse traversal.

``benchmarks/dynamic_parameters_demo.py`` demonstrates parameter estimation
with synthetic observations, derivative checks, one optimization update, and
RAM/disk replay for ACC or the global 4-degree setup. Run it with
``PYTHONPATH=. VEROS_BACKEND=jax python benchmarks/dynamic_parameters_demo.py --setup global_4deg``.
Its finite-difference checks use float64 and relative parameter perturbations
of ``1e-6`` and ``1e-7``. Larger perturbations can cross limiter or mixing
thresholds. On a 12 GB GPU, the native CUDA TDMA implementation and
``XLA_PYTHON_CLIENT_PREALLOCATE=false`` keep this six-step example within
available memory.

To include diagnostic losses, also pass ``record_objective(records)`` to the
driver. It receives records with a singleton leading step axis, matching
``advance(n_steps=1)``, and returns a scalar contribution for that timestep.
The driver sums these contributions across the original forward traversal.
Diagnostic accumulators, schedules, and resets are replayed, including windows
crossing disk intervals. Replay performs no diagnostic file writes and does
not install the final carry into the simulation.

``BinomialSchedule(n_steps, ram_checkpoints=..., disk_checkpoints=...)`` uses
binomial splits with fixed budgets for both tiers, including the initial
checkpoint. Disk slots hold the outer splits and RAM slots the deeper splits.
For example, ``ram_checkpoints=2, disk_checkpoints=3`` keeps at most two RAM and
three disk checkpoints regardless of the horizon. Without disk slots, this
strategy uses only host RAM.
``PeriodicDiskSchedule`` stores interval starts on disk and uses binomial RAM
checkpoints inside each interval. Its disk checkpoints do not consume RAM slots.
Zero RAM slots are supported, with more recomputation from disk interval starts.
Both intervals and schedule indices count model timesteps.

Each checkpoint preserves every carry leaf without compression or dtype
conversion. Leaves must have explicit dtypes, as in a packed Veros carry.
These are adjoint checkpoints, separate from simulation restart files.
Their tree metadata remains in the running process, so they are not a
persistent recovery format. RAM copies live on the host. Disk leaves are written
individually to NumPy archives in an owned subdirectory, which is cleaned on
normal exit or a Python exception. Use a filesystem backed by disk. A ``tmpfs``
directory still consumes host RAM.

Checkpoint slot budgets exclude the additional memory costs described above,
even with zero RAM slots. Statistics report checkpoint counts,
peak stored bytes, disk traffic, and ``forward_steps``, ``recomputed_steps``,
and ``reverse_steps``. Recomputed steps include each reverse step's forward
evaluation. They exclude rematerialization inside its JAX computation.
Forward and reverse executables are independent of the trajectory length.

Custom scheduling strategies provide ``n_steps``, ``ram_checkpoints``, and
an ``actions()`` method returning a fresh iterator of ``CheckpointAction``
objects. Available actions are ``advance``, ``save``, ``restore``, ``drop``,
``turn``, and ``reverse``. ``turn`` separates the original forward traversal
from reverse replay. ``reverse`` consumes one timestep in backward order.
``save`` selects ``ram`` or ``disk``. A restore/drop/save sequence can move a
checkpoint between tiers. ``validate_schedule`` checks ordering, availability,
RAM budgets, and complete release without executing the model. An optional
``disk_checkpoints`` budget is also validated and enforced by the store.
The action iterator must produce the same schedule on each traversal.
Custom schedules can account for measured I/O and recomputation costs.
The supplied strategies do not guarantee minimum wall time for a storage hierarchy.

The driver supports CPU and GPU execution, including coordinated MPI replay.
It is a host operation and cannot be wrapped in ``jax.jit`` or ``jax.grad``.
Higher-order differentiation and batching are unsupported. The numerical
gradient limitations described above also apply to replay.

MPI execution
-------------

Compiled MPI execution uses duplicated numerical communicators separate from
host control and file I/O. PETSc receives the existing numerical communicator,
and its callbacks share mpi4jax's ordering token. Veros does not create an
additional communicator for PETSc. Halo exchanges and pressure solves have
explicit linear JVP and transpose rules. Checkpoint recomputation repeats these
numerical operations, so all ranks must use identical call lengths and options.
Distributed PETSc verification covers CPU execution on one, two, and four ranks
and GPU execution on two ranks, using PETSc/petsc4py 3.26.0 with the JAX and
mpi4jax versions listed above. Single-rank CPU and GPU tests cover six-step
trajectories with default TKE, all six diagnostics, checkpoint replay, and
forward/reverse derivative agreement. CPU tests also cover diagnostic files
and restart continuation.
MPI file writes were not exercised in this verification. Parallel file output
requires MPI-enabled HDF5, as in the ordinary runner.

With ``linear_solver="petsc"``, preparation retains a distributed PETSc operator
and preconditioner. ``linear_solver="best"`` selects this adapter for MPI when
PETSc is available. Override the compiled selection independently of setup::

    prepared = simulation.prepare_integrator(linear_solver="petsc")
    # Or use the host driver:
    simulation.run_compiled(n_steps=24, linear_solver="petsc")

The CLI exposes the same choice through ``--compiled-solver petsc``.
For a GPU setup using a CPU-only PETSc installation, initialize with the JAX
solver and select PETSc for compiled execution, for example::

    VEROS_LINEAR_SOLVER=scipy_jax veros run acc_basic.py --backend jax --device gpu --compiled --compiled-solver petsc

Callbacks receive local interior right-hand sides and return local solutions.
The adjoint uses a separately prepared transposed matrix and
preconditioner, which add local memory and preparation costs. The solve includes
RHS and boundary-value sensitivities for fixed geometry. Primal, tangent, and
adjoint RHSs are normalized globally, and failed solves return numerical failure
information. MPI must provide
``MPI_THREAD_MULTIPLE`` because callbacks can run on worker threads.

Callbacks use CPU PETSc vectors. A GPU model transfers its local pressure
arrays to and from the host at each solve. This path does not require a
CUDA-enabled PETSc installation, but it does introduce synchronization and
transfer costs. Setup and the ordinary runner retain their existing solver
selection and GPU requirements.

``linear_solver="scipy_jax"`` remains available as a reference and fallback.
Its global pressure operator and solve are replicated on every rank, with a
global memory and compute cost on each rank.

Global diagnostic values are replicated. MPI SUM reductions use mpi4jax's
replicated-output derivative convention. For a parameter replicated on all
ranks, each rank receives its local sensitivity and the shared-parameter
gradient is their sum. Do not sum the replicated objective itself again.
Forward JVPs with identical parameter tangents already give the total shared
sensitivity. Root-only gather/scatter are host operations and are rejected
inside compiled hooks.
