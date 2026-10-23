Compiled integration and differentiation
========================================

The JAX backend can compile a numerical timestep, including forcing, physics,
and diagnostic sampling and finalization. Setup and file output run on the host.
The pure API supports CPU and GPU execution with JAX or PETSc pressure solvers,
including distributed execution through MPI.

Running a simulation
--------------------

``simulation.run()`` selects compiled execution for supported JAX configurations
and reports the selected path. NumPy, explicit SciPy solver selection, unmarked
numerical hooks, and setups overriding ``step()`` use ordinary execution.
``run(execution="compiled")`` requires compilation and raises when the model is
unsupported. ``run(execution="eager")`` selects ordinary execution explicitly.

To run a fixed number of timesteps with diagnostics, restarts, and progress,
call ``simulation.run_compiled(n_steps=24)`` after setup. Without ``n_steps``,
the driver uses ``settings.runlen``, rounded up to a whole timestep. Results are
installed into ``simulation.state`` after each step. Numerical failures raise
on the host, and the driver writes a final restart when execution stops.

The corresponding command line options are::

    veros run acc_basic.py --backend jax
    veros run acc_basic.py --backend jax --execution compiled
    veros run acc_basic.py --backend jax --execution eager

``--compiled`` is an alias for ``--execution compiled``. A compiled run can
select ``--compiled-solver scipy_jax``, ``--compiled-solver petsc``, or
``--compiled-solver best``. An explicit
solver override must be honored by compiled execution. An automatic fallback
raises if ordinary execution would use a different solver.

The first timestep's reported duration includes compilation. Subsequent timers
measure synchronized numerical steps, input preparation, diagnostics, and I/O.
Use ordinary execution for timings of individual numerical routines.

A pure numerical API
--------------------

Initialize a model and prepare its integrator::

    from veros import runtime_settings
    runtime_settings.backend = "jax"

    from veros.setups.acc_basic import ACCBasicSetup
    simulation = ACCBasicSetup(
        override=dict(nx=16, ny=16, nz=15, dt_mom=600, dt_tracer=600)
    )
    simulation.setup()
    prepared = simulation.prepare_integrator(record_diagnostics=("averages",))
    initial = prepared.pack(simulation.state)
    final, records = prepared.advance(initial, n_steps=24)
    prepared.check_status(final)

``IntegrationState`` carries model variables, diagnostic accumulators, and the
first numerical failure. ``step(carry)`` returns one timestep and event records.
``advance`` composes the same compiled timestep in a Python loop. Its records
have a leading timestep axis and a ``valid`` mask for diagnostic output events.
``final_only=True`` returns only the last timestep's records. Failed trajectories
freeze after their first failure.

Differentiate an objective through the continuation state::

    import jax

    def objective(scale):
        carry = prepared.replace_variables(
            initial, t_star=scale * initial.variables.t_star
        )
        final, records = prepared.advance(carry, n_steps=24)
        return final.variables.temp[..., final.variables.tau].sum()

    sensitivity = jax.grad(objective)(1.0)

``t_star`` is the restoring target used by the ACC basic setup. Other setups have
their own forcing variables. Integer indices, grid topology, fixed settings,
diagnostic schedules, and hooks are captured at preparation. Reprepare after
changing them. Floating step-time settings can be overridden as described below.
For the same shapes and dtypes, changing model arrays reuses the compiled step.
``checkpoint=True`` recomputes step intermediates during reverse differentiation.

Numerical hooks and plugin entrypoints must declare
``@veros_routine(supports_compilation=True)``. This promises JAX operations
without host side effects during a timestep. Diagnostics used inside pure
numerical execution must provide sampling and ``finalize`` methods. All built-in
diagnostics support this contract.
Configure output variables before setup and frequencies before preparation.

Packing, installation, and output are host operations. Call
``prepared.install(final)`` to update the model state and
``prepared.write_records(records)`` to write concrete diagnostic output. Numerical
integration and differentiation do not install results or write files.

Host diagnostic observers
-------------------------

Legacy diagnostics can declare ``observes_model = True`` to run their existing
``diagnose`` and ``output`` methods on the host during compiled execution. They
may update their own accumulators and write output, but must preserve model
variables and settings. Sampling schedules and rotating time indices match
ordinary execution. Diagnostic values used in differentiated objectives require
pure sampling and ``finalize`` methods with ``supports_compilation = True``.

A subclass overriding a compiled diagnostic's ``output()`` must declare its own
``supports_compilation = True`` contract and provide equivalent behavior through
``finalize`` and ``write_finalized``. Otherwise automatic selection uses ordinary
execution to preserve the overridden method.

Explicit forcing inputs
-----------------------

``prepared.step`` accepts a ``step_inputs`` mapping of forcing names to arrays.
``prepared.advance`` accepts a sequence containing one mapping per timestep::

    import jax.numpy as jnp

    forcing = jnp.full_like(initial.variables.forc_temp_surface, 0.01)
    final, records = prepared.advance(
        initial,
        n_steps=2,
        step_inputs=[{"forc_temp_surface": forcing}] * 2,
    )

Inputs replace forcing values after the setup's native forcing hook. Each array
must match the prepared variable's full shape, including horizontal halos, and
is converted to its dtype. Accepted names are ``surface_taux``, ``surface_tauy``,
``forc_temp_surface``, ``forc_salt_surface``, ``forc_tke_surface``,
``forc_iw_surface``, ``forc_iw_bottom``, ``temp_source``, and ``salt_source``.
The corresponding variable must be active in the configured model. Forcing
arrays can depend on differentiable inputs inside an objective.

``simulation.run_compiled(step_inputs=...)`` also accepts an iterator. Its
provider runs on the host before each timestep and must yield exactly one
mapping per timestep. Prepare forcing values outside numerical hooks when they
require file reads or other host operations.

Restart continuation
--------------------

Restart files preserve iteration counts, time indices, and diagnostic windows.
Legacy files without an iteration count recover it from ``time / dt_tracer``
when that ratio is integral. Restart reads and writes are host operations and
do not reconnect derivatives to the computation that produced the file.
Changing restart paths, the identifier, or description does not require
repreparation.

Dynamic physical parameters
---------------------------

Pass scalar floating setting overrides to a pure timestep or trajectory::

    import jax.numpy as jnp

    def loss(parameters):
        final, _ = prepared.advance(
            initial, n_steps=24, parameters=parameters, final_only=True
        )
        vs = final.variables
        return jnp.sum(vs.temp[2:-2, 2:-2, :, vs.tau] ** 2)

    parameters = {"c_k": jnp.array(0.1), "c_eps": jnp.array(0.7)}
    sensitivity = jax.grad(loss)(parameters)

Kernels and hooks read overrides through ``state.settings``. Host settings and
the continuation state retain their original values. Pass overrides to each
continuation call. The same names, shapes, and normalized dtype reuse the
compiled step. Changing the names creates a new signature. A setting unused by
the enabled physics has zero sensitivity.

Overrides reject nonfloating settings, nonscalar values, and settings that
define frozen grid data, reference constants, or schedules: ``pi``, ``radius``,
``degtom``, ``omega``, ``x_origin``, ``y_origin``, ``rho_0``, ``grav``,
``dt_mom``, ``dt_tracer``, ``runlen``, and ``restart_frequency``. Setup and
plugin authors must list additional settings used to prepare cached data in
``prepare_integrator(fixed_parameters=(...))``. Overrides do not rerun setup
or rebuild a fixed pressure operator.

Register restart-persistent scalar controls at construction when they should
travel with the model state::

    simulation = ACCBasicSetup(
        control_parameters=("kappaH_0", "kappaM_0", "K_iso_0", "K_gm_0")
    )

These create variables named ``control_kappaH_0`` and so on, initialized from
the corresponding settings. Replace those leaves with ``replace_variables``
to differentiate the parameterization. Constant diffusivity controls are used
when their prognostic TKE/EKE parameterization is disabled. A setting with an
existing ``control_*`` variable cannot also receive a setting override.


MPI execution
-------------

Install the ``jax-mpi`` extra for the supported JAX and MPI Python dependencies.
All ranks must use the same timestep count and integration options. Compiled
hooks must support local execution; root-only gather and scatter are host
operations and are rejected during numerical tracing.

Run a model on a 2 by 2 process grid with::

    mpiexec -n 4 veros run acc_basic.py --backend jax --device cpu \
        --num-proc 2 2 --compiled --compiled-solver petsc

``linear_solver="best"`` selects PETSc for distributed compiled integration
when available and otherwise uses replicated JAX solves. Select either solver
explicitly through ``prepare_integrator(linear_solver=...)`` or
``run_compiled(linear_solver=...)``. The replicated JAX solver stores the global
pressure operator and solves the global system on every rank.

PETSc callbacks share an ordering token with numerical MPI transfers on
communicators prepared separately from host operations. Halo exchanges and
collectives provide forward and reverse derivatives. PETSc derivatives solve
the fixed operator and its separately prepared transpose, including
right-hand-side and boundary-value sensitivities. Numerical right-hand sides
are normalized globally; failed solves set the integration failure status.
PETSc callbacks require ``MPI_THREAD_MULTIPLE`` and use CPU vectors. GPU models
transfer their local pressure arrays to the host for these solves.

Global diagnostic values are replicated across ranks. Seed a replicated scalar
objective once, for example by differentiating ``loss / runtime_state.proc_num``
on each rank. Sum the resulting local sensitivities to obtain the gradient
for a shared parameter. Identical forward tangents on every rank already
produce the shared directional derivative. Parallel file output requires
MPI-enabled HDF5, as in ordinary execution.

Disk and RAM checkpoint schedules
---------------------------------

``CheckpointedIntegrator`` computes first-order initial-state and runtime-setting
gradients using a host schedule, compiled forward steps, and one-step pullbacks::

    from pathlib import Path
    from veros.checkpointing import CheckpointedIntegrator, PeriodicDiskSchedule

    def terminal_objective(final):
        vs = final.variables
        return jnp.sum(vs.temp[2:-2, 2:-2, :, vs.tau])

    directory = Path("adjoint-checkpoints")
    directory.mkdir(exist_ok=True)
    driver = CheckpointedIntegrator(prepared, terminal_objective=terminal_objective)
    schedule = PeriodicDiskSchedule(24, disk_interval=4, ram_checkpoints=2)
    result = driver.value_and_grad(
        initial, n_steps=24, schedule=schedule, directory=directory
    )
    forcing_scale_gradient = jnp.sum(
        result.gradient.variables.t_star * initial.variables.t_star
    )

The result contains ``value``, the initial-carry cotangent ``gradient``, ``final``,
``statistics``, and ``parameter_gradient``. Integer and Boolean cotangent leaves
have JAX's ``float0`` dtype. Runtime overrides use the ``parameters`` argument.
Both objectives may accept a ``parameters`` keyword for direct dependence on
settings. If optimization parameters also construct the initial carry, apply
that construction's VJP to ``result.gradient`` and combine it with
``result.parameter_gradient``. Supported inputs and fixed configuration follow the pure API.

For diagnostic losses, pass ``record_objective(records)``. It returns a scalar
contribution per original forward step and receives a singleton leading step
axis, matching ``advance(n_steps=1)``. Numerical diagnostic windows and resets
are replayed across checkpoint intervals. Objective callbacks must be pure.
Replay writes no diagnostic files and leaves the simulation state unchanged.

An iterable ``step_inputs`` supplies external forcing. The driver reads and
records each input once, then uses the recorded arrays during replay. Set
``input_storage="disk"`` to save them under ``directory``. Input-array derivatives
use the pure trajectory API. ``observer(index, carry)`` runs once per original
forward step and is excluded from recomputation.

``BinomialSchedule(n_steps, ram_checkpoints=..., disk_checkpoints=...)`` bounds
slots in both tiers, including the initial checkpoint. Disk holds outer splits
and RAM deeper splits. ``PeriodicDiskSchedule`` stores interval starts on disk
and uses binomial RAM replay inside each interval. Zero RAM slots are supported.
Intervals and action indices count model timesteps.

Stores preserve every carry leaf's shape and explicit dtype. RAM copies live
on the host. Disk archives occupy an owned temporary subdirectory and are
removed on success or a Python exception. Tree metadata stays in the process,
so archives cannot serve as persistent simulation restarts. Use a disk-backed
filesystem when offloading memory, since ``tmpfs`` consumes host RAM.

``max_checkpoint_ram_bytes`` and ``max_input_ram_bytes`` bound stored RAM arrays.
Slot and byte limits exclude live states, adjoints, compiled executables, and
step-local differentiation buffers. Statistics report storage peaks, disk
traffic, original/recomputed/reverse step counts, and available XLA memory
estimates. Reverse steps include their own forward reevaluation in the
recomputation count. Executables are reused across trajectory lengths.

MPI ranks store local states and follow the same schedule. Objectives must
return the same global scalar on each rank, for example through ``global_sum``
inside ``tracing_context()``. The driver divides adjoint seeds across ranks and
sums shared parameter gradients once. ``shared_parameters=()`` selects rank-local
parameters. ``shared_variables`` names replicated initial controls whose
cotangents should be summed. Other initial-state cotangents remain rank-local.
Host validation and storage failures propagate across ranks before the next
numerical operation.

Custom schedules expose ``n_steps``, ``ram_checkpoints``, optional
``disk_checkpoints``, and a repeatable ``actions()`` iterator. Actions are
``advance``, ``save``, ``restore``, ``drop``, ``turn``, and ``reverse``. ``turn``
ends the original forward traversal. Reverse actions consume one timestep in
backward order. ``validate_schedule`` checks ordering, availability, budgets,
and complete release before model execution.

The driver is a host operation that cannot be transformed with ``jax.jit`` or
``jax.grad``. It supports CPU, GPU, and coordinated MPI replay. The pure API's
numerical gradient limitations still apply. ``benchmarks/dynamic_parameters_demo.py``
provides synthetic parameter fitting, derivative checks, and RAM/disk replay.

Partial-cell bathymetry
-----------------------

Set ``enable_partial_cells=True`` before setup to vary bottom depths within
fixed wet cells. The supported configuration uses serial JAX execution, the
``scipy_jax`` pressure solver, and surface pressure. Constant mixing, TKE, EKE,
energy exchanges, and energy diagnostic records are supported. IDEMIX, TEM
friction, biharmonic mixing/friction, streamfunction mode, and active
overturning diagnostics are rejected. Turbulence support retains the native
scheme's sign limitations and can produce negative energy states.

Tracer and momentum transport, vertical mixing, wind stress, and quadratic
bottom drag use occupied cell thicknesses or open face areas.
Horizontal harmonic friction integrates stress over shared wet faces and
divides its divergence by occupied momentum-cell volumes. Free-slip boundaries
conserve momentum; no-slip wall stresses dissipate kinetic energy. Momentum-source
and drag work use occupied velocity volumes. Energy inventories use native
U/V volumes for kinetic energy and occupied T volumes for enthalpy; diagnostic
work fields retain their nominal W-grid quadrature.
Neutral and skew tracer diffusion use
open horizontal faces and occupied tracer volumes. Their vertical cross and
diagonal quadrature terms carry the matching horizontal face fractions. The matching factors retain the paired tensor structure. The implicit Redi solve also uses occupied tracer thickness.
Reference vertical difference distances remain fixed. Choose the timestep for
the smallest occupied cells in the admissible geometry range.

Tracer-monitor volumes and vertical CFL estimates also use occupied thicknesses.
Bathymetry controls hold reference vertical levels and difference distances
fixed. Neighboring velocity faces use the smaller occupied thickness of their adjacent tracer
cells. Pressure solves rebuild depth-dependent numerical coefficients so
operator derivatives include changes in geometry.

For a configured partial-cell model, construct candidate geometry inside the
objective::

    import jax
    import jax.numpy as jnp

    prepared = simulation.prepare_integrator(record_diagnostics=())
    initial = prepared.pack(simulation.state)
    lower, upper = prepared.bathymetry_bounds(initial, minimum_fraction=0.1)

    def objective(control):
        depth = lower + (upper - lower) * jax.nn.sigmoid(control)
        candidate = prepared.with_bathymetry(initial, depth)
        final, _ = prepared.advance(candidate, n_steps=24)
        vs = final.variables
        misfit = jnp.mean((vs.psi[2:-2, 2:-2, vs.tau] / 9.81 - observed_ssh) ** 2)
        return jnp.where(final.failure_code == 0, misfit, jnp.nan), final

    (value, final), gradient = jax.value_and_grad(objective, has_aux=True)(control)
    prepared.check_status(final)

Depth arrays contain the interior grid, excluding halos, in positive meters
below the surface. Land depths must be zero. The lower bound preserves at least
``minimum_fraction`` of each wet bottom cell. The upper bound fills that cell
completely. Invalid depths set failure code 4 and freeze the trajectory.
Geometry persists through continuation, installation, restart output, and
adjoint checkpoint replay.
Restart reads require the same reference grid and wet-cell indices.

``with_bathymetry`` preserves tracer concentrations, velocities, and tendency
histories. Construct depth-dependent initial profiles or balanced velocities
inside the objective after updating geometry. Changing volume at fixed
concentration changes total tracer content, so an experiment needing
conservative remapping must supply that initialization policy.

Gradients describe the discretization within fixed topology. The minimum of
neighboring depths is nonsmooth where the shallower side changes. Bottom-cell
transitions, moving coastlines, and opening or closing passages require new
topology and preparation.

For a host-managed adjoint, retain the bathymetry construction's VJP when
building the initial candidate::

    candidate, pullback = jax.vjp(
        lambda depth: prepared.with_bathymetry(initial, depth), depth
    )

After checkpoint replay, ``pullback(result.gradient)[0]`` returns the
depth-control gradient.


Optional turbulence and neutral-slope policies
----------------------------------------------

``enable_tke_positive_diffusion=True`` selects implicit positive-part TKE
diffusion and damping, a nonnegative surface-energy constraint, and continuous
global allocation of numerical work. It is disabled by default. Interior stored
energy can remain negative after forcing and advection. PyOM transfer rejects
this option because the Fortran model does not implement these equations.

``enable_neutral_nonstable_zero_slope=True`` sets neutral slopes to zero at
nonstable and surface quadrature stencils. Stable stencils retain their
original formula. This option is disabled by default and requires independent
validation for each physical configuration.
