Compiled integration and differentiation
========================================

The JAX backend can compile a numerical timestep, including forcing, physics,
and diagnostic sampling and finalization. Setup and file output run on the host.
The pure API supports serial CPU and GPU execution with the JAX pressure solver.

Running a simulation
--------------------

``simulation.run()`` selects compiled execution for supported JAX configurations
and reports the selected path. NumPy, explicit SciPy solver selection, unmarked
numerical hooks, and setups overriding ``step()`` use ordinary execution.
``run(execution="compiled")`` requires compilation and raises when the model is
unsupported. ``run(execution="eager")`` selects ordinary execution explicitly.
Compiled integration currently requires serial execution.

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
select ``--compiled-solver scipy_jax`` or ``--compiled-solver best``. An explicit
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
their own forcing variables. Integer indices, grid topology, settings, diagnostic
schedules, and hooks are fixed at preparation. Reprepare after changing them.
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
