Compiled integration and differentiation
========================================

The JAX backend can compile a numerical timestep, including forcing, physics,
and diagnostic sampling and finalization. Setup and file output run on the host.
The pure API supports serial CPU and GPU execution with the JAX pressure solver.

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
without host side effects during a timestep. Diagnostics must provide pure
sampling and ``finalize`` methods. All built-in diagnostics support this contract.
Configure output variables before setup and frequencies before preparation.

Packing, installation, and output are host operations. Call
``prepared.install(final)`` to update the model state and
``prepared.write_records(records)`` to write concrete diagnostic output. Numerical
integration and differentiation do not install results or write files.
