"""Pure compiled timesteps and differentiable trajectory composition."""

from collections import namedtuple
from copy import copy
import math
import operator

from veros import runtime_settings, runtime_state
from veros.routines import tracing_context

IntegrationState = namedtuple("IntegrationState", "variables diagnostics failure_code failed_iteration")
DiagnosticRecord = namedtuple("DiagnosticRecord", "time iteration valid values")


class CompiledCompatibilityError(RuntimeError):
    """A configuration needs ordinary execution."""


def compilation_reasons(simulation, linear_solver=None):
    """Inspect declarations without executing user hooks."""
    from veros import VerosSetup

    reasons = []
    if runtime_settings.backend != "jax":
        reasons.append("compiled integration requires the JAX backend")
    if runtime_state.proc_num > 1:
        reasons.append("compiled integration currently requires serial execution")
    solver = runtime_settings.linear_solver if linear_solver is None else linear_solver
    if solver not in ("best", "scipy", "petsc", "scipy_jax"):
        raise ValueError("Unrecognized compiled linear_solver")
    if solver not in ("best", "scipy_jax"):
        reasons.append(f"solver {solver!r} requires ordinary execution")
    if simulation.__dict__.get("step", getattr(type(simulation), "step")) is not VerosSetup.step:
        reasons.append("overridden step requires ordinary execution")
    for name in ("_step_dynamics", "set_forcing", "after_timestep"):
        routine = simulation.__dict__.get(name, getattr(type(simulation), name))
        if not getattr(routine, "supports_compilation", False):
            reasons.append(f"{name} must declare supports_compilation=True")
        elif not getattr(routine, "dist_safe", True) and runtime_state.proc_num > 1:
            reasons.append(f"{name} requires host gathering")
    for plugin in simulation._plugin_interfaces:
        routine = plugin.run_entrypoint
        if not getattr(routine, "supports_compilation", False):
            reasons.append(f"plugin {plugin.name} must declare supports_compilation=True")
        elif not getattr(routine, "dist_safe", True) and runtime_state.proc_num > 1:
            reasons.append(f"plugin {plugin.name} requires host gathering")
    for name, diagnostic in simulation.state.diagnostics.items():
        if not diagnostic.sampling_frequency and not diagnostic.output_frequency:
            continue
        if diagnostic.supports_compilation:
            declaration = next(cls for cls in type(diagnostic).__mro__ if "supports_compilation" in vars(cls))
            output = diagnostic.__dict__.get("output", type(diagnostic).output)
            if output is not declaration.output:
                reasons.append(f"diagnostic {name} overrides output without declaring a compiled finalization contract")
        if not diagnostic.supports_compilation:
            reasons.append(f"diagnostic {name} must support compiled sampling and finalization")
    return tuple(reasons)


class PreparedIntegrator:
    """Integrate an initialized model with fixed configuration.

    Configuration and hooks are captured at preparation. Floating model
    variables remain dynamic.

    Args:
        simulation: An initialized :class:`veros.VerosSetup`.
        record_diagnostics: Names whose output events should be retained. By
            default, retain enabled diagnostics except snapshots.
        checkpoint: Recompute step intermediates during differentiation.
        linear_solver: Override the compiled solver selection with
            ``scipy_jax`` or ``best``. Defaults to the runtime selection.
    """

    def __init__(
        self,
        simulation,
        *,
        record_diagnostics=None,
        checkpoint=False,
        linear_solver=None,
    ):
        simulation._ensure_setup_done()
        reasons = compilation_reasons(simulation, linear_solver)
        if reasons:
            raise CompiledCompatibilityError("; ".join(reasons))
        import jax
        from veros.core.external.solvers import get_compiled_solver

        if not math.isfinite(simulation.state.settings.dt_tracer) or simulation.state.settings.dt_tracer <= 0:
            raise ValueError("dt_tracer must be finite and positive")
        self.simulation = simulation
        self._step_dynamics = copy(simulation)._step_dynamics
        self._template = copy(simulation.state)
        self._template._settings = copy(simulation.state.settings)
        self._template._dimensions = dict(simulation.state.dimensions)
        self._template._diagnostics = {}
        self._template._linear_solver = get_compiled_solver(simulation.state, linear_solver)
        for name, diag in simulation.state.diagnostics.items():
            for frequency in (diag.sampling_frequency, diag.output_frequency):
                if frequency is not None and (math.isnan(frequency) or frequency < 0):
                    raise ValueError(f"Diagnostic {name} has an invalid frequency: {frequency}")
        self._diagnostics = {
            name: copy(diag) for name, diag in simulation.state.diagnostics.items() if diag.supports_compilation
        }
        for diag in self._diagnostics.values():
            diag.output_variables = tuple(diag.output_variables or ())

        if record_diagnostics is None:
            record_diagnostics = tuple(
                name for name, diag in self._diagnostics.items() if diag.output_frequency and name != "snapshot"
            )
        self.record_diagnostics = tuple(record_diagnostics)
        for name in self.record_diagnostics:
            if name not in self._diagnostics:
                raise ValueError(f"Unknown diagnostic {name}")

        initial = self.pack(simulation.state)
        self._empty_values = {}
        with tracing_context():
            state = self._make_state(initial.variables)
            for name, diag in self._diagnostics.items():
                if not diag.output_frequency and name not in self.record_diagnostics:
                    continue
                local = self._make_diagnostic(name, initial.diagnostics)
                if name == "snapshot":
                    local.variables = state.variables
                values = local.finalize(state)
                self._empty_values[name] = jax.tree.map(
                    lambda value: runtime_state.backend_module.zeros_like(value), values
                )

        self._compiled_step = jax.jit(jax.checkpoint(self._step) if checkpoint else self._step)
        try:
            jax.eval_shape(self._step, initial)
        except Exception as error:
            raise CompiledCompatibilityError(
                "Cannot trace the model step; forcing, hooks, and diagnostics must use JAX operations"
            ) from error
        self._configuration = dict(simulation.state.settings.items())
        self._runtime_configuration = (
            runtime_settings.backend,
            runtime_settings.device,
            runtime_settings.float_type,
            runtime_settings.linear_solver,
            runtime_settings.num_proc,
        )
        self._hooks = {
            name: simulation.__dict__.get(name, getattr(type(simulation), name))
            for name in ("step", "_step_dynamics", "set_forcing", "after_timestep")
        }
        self._plugins = simulation._plugin_interfaces
        self._diagnostic_configuration = self._diagnostic_signature()
        self._diagnostic_outputs = self._diagnostic_output_signature()

    def _diagnostic_signature(self):
        return tuple(
            (
                name,
                type(diag),
                diag.supports_compilation,
                diag.observes_model,
                diag.sampling_frequency,
                diag.output_frequency,
                tuple(diag.output_variables or ()),
            )
            for name, diag in self.simulation.state.diagnostics.items()
        )

    def _diagnostic_output_signature(self):
        return tuple(
            (name, diag.__dict__.get("output", type(diag).output))
            for name, diag in self.simulation.state.diagnostics.items()
        )

    def check_configuration(self):
        """Require explicit repreparation after structural setting changes."""
        changed = [
            name
            for name, value in self._configuration.items()
            if getattr(self.simulation.state.settings, name) != value
        ]
        for name, hook in self._hooks.items():
            if self.simulation.__dict__.get(name, getattr(type(self.simulation), name)) is not hook:
                changed.append(name)
        if self.simulation._plugin_interfaces != self._plugins:
            changed.append("plugins")
        if self._diagnostic_signature() != self._diagnostic_configuration:
            changed.append("diagnostic schedules/variables")
        if self._diagnostic_output_signature() != self._diagnostic_outputs:
            changed.append("diagnostic output methods")
        current_runtime = (
            runtime_settings.backend,
            runtime_settings.device,
            runtime_settings.float_type,
            runtime_settings.linear_solver,
            runtime_settings.num_proc,
        )
        if current_runtime != self._runtime_configuration:
            changed.append("runtime backend/solver/decomposition")
        if changed:
            raise CompiledCompatibilityError(f"Configuration changed ({', '.join(changed)}); reprepare the integrator")

    def pack(self, state):
        """Copy model and diagnostic containers into a continuation state."""
        import jax.numpy as jnp

        diagnostic_variables = {
            name: copy(diag.variables)
            for name, diag in state.diagnostics.items()
            if name in self._diagnostics and name != "snapshot" and getattr(diag, "variables", None) is not None
        }
        return IntegrationState(copy(state.variables), diagnostic_variables, jnp.int32(0), jnp.int32(-1))

    @staticmethod
    def replace_variables(carry, **values):
        """Return a new continuation state with validated variable replacements."""
        variables = copy(carry.variables)
        with variables.unlock():
            variables.update(values)
        return carry._replace(variables=variables)

    def install(self, carry, state=None):
        """Install concrete results into a host state without writing output."""
        if state is None:
            state = self.simulation.state
        with state.variables.unlock():
            state.variables.update(carry.variables)
        for name, variables in carry.diagnostics.items():
            state.diagnostics[name].variables.update(variables)
        if "snapshot" in state.diagnostics:
            state.diagnostics["snapshot"].variables = state.variables
        return state

    def _make_state(self, variables):
        state = copy(self._template)
        state._variables = copy(variables)
        state._solver_residual_ok = True
        return state

    def _make_diagnostic(self, name, variables):
        diagnostic = copy(self._diagnostics[name])
        if name in variables:
            diagnostic.variables = copy(variables[name])
        return diagnostic

    def _empty_records(self, carry):
        import jax.numpy as jnp

        return {
            name: DiagnosticRecord(
                carry.variables.time, carry.variables.itt, jnp.bool_(False), self._empty_values[name]
            )
            for name in self.record_diagnostics
        }

    def _update_diagnostics(self, state, variables):
        import jax
        import jax.numpy as jnp
        from veros.distributed import global_and

        variables = dict(variables)
        records = {}
        valid_cfl = jnp.bool_(True)
        for name, diagnostic in self._diagnostics.items():
            if diagnostic.sampling_frequency:

                def sample(current):
                    local = self._make_diagnostic(name, {name: current} if current is not None else {})
                    local.diagnose(state)
                    return getattr(local, "variables", None)

                current = variables.get(name)
                due = state.variables.time % diagnostic.sampling_frequency < state.settings.dt_tracer
                updated = jax.lax.cond(due, sample, lambda current: current, current)
                if updated is not None:
                    variables[name] = updated

            if not diagnostic.output_frequency:
                if name in self.record_diagnostics:
                    records[name] = DiagnosticRecord(
                        state.variables.time, state.variables.itt, jnp.bool_(False), self._empty_values[name]
                    )
                continue

            def finalize(current):
                local = self._make_diagnostic(name, {name: current} if current is not None else {})
                if name == "snapshot":
                    local.variables = state.variables
                values = local.finalize(state)
                return (None if name == "snapshot" else getattr(local, "variables", None)), values

            current = variables.get(name)
            due = state.variables.time % diagnostic.output_frequency < state.settings.dt_tracer
            updated, values = jax.lax.cond(due, finalize, lambda current: (current, self._empty_values[name]), current)
            if updated is not None:
                variables[name] = updated
            if name == "cfl_monitor":
                valid_cfl = global_and(jnp.isfinite(values["cfl"]) & jnp.isfinite(values["wcfl"]))
            if name in self.record_diagnostics:
                records[name] = DiagnosticRecord(state.variables.time, state.variables.itt, due, values)
        return variables, records, valid_cfl

    def _step(self, carry):
        import jax
        import jax.numpy as jnp
        from veros.core import numerics, isoneutral

        def advance(current):
            state = self._make_state(current.variables)
            with state.variables.unlock():
                self._step_dynamics(state)
                sane = numerics.sanity_check(state)
                solver_ok = jnp.asarray(state._solver_residual_ok, dtype="bool")
                ok = sane & solver_ok
                isoneutral.isoneutral_diag_streamfunction(state)

                def diagnose(variables):
                    return self._update_diagnostics(state, variables)

                diagnostic_variables, records, cfl_ok = jax.lax.cond(
                    ok,
                    diagnose,
                    lambda variables: (variables, self._empty_records(current), jnp.bool_(True)),
                    current.diagnostics,
                )
                code = jnp.where(~sane, 1, jnp.where(~solver_ok, 2, jnp.where(~cfl_ok, 3, 0))).astype("int32")
                records = {name: record._replace(valid=record.valid & (code == 0)) for name, record in records.items()}
                vs = state.variables
                vs.taum1, vs.tau, vs.taup1 = vs.tau, vs.taup1, vs.taum1
            return (
                IntegrationState(
                    vs, diagnostic_variables, code, jnp.where(code != 0, vs.itt, current.failed_iteration)
                ),
                records,
            )

        with tracing_context():
            return jax.lax.cond(
                carry.failure_code == 0,
                advance,
                lambda current: (current, self._empty_records(current)),
                carry,
            )

    def step(self, carry):
        """Return one compiled timestep and event records without a leading step axis."""
        self.check_configuration()
        return self._compiled_step(carry)

    def advance(self, carry, *, n_steps, final_only=False):
        """Compose compiled timesteps in a Python loop and return output records.

        Records have a leading step axis unless ``final_only=True``. Their
        ``valid`` masks identify output events. Failed trajectories carry a
        nonzero failure code and freeze after the first failed step.
        JAX can differentiate this composition without jitting the whole loop.
        Every horizon reuses the same compiled timestep.
        """
        n_steps = operator.index(n_steps)
        if n_steps < 0:
            raise ValueError("n_steps must be nonnegative")
        self.check_configuration()
        import jax
        import jax.numpy as jnp

        records = self._empty_records(carry)
        history = []
        for _ in range(n_steps):
            carry, records = self._compiled_step(carry)
            if not final_only:
                history.append(records)
        if not final_only:
            records = (
                jax.tree.map(lambda *values: jnp.stack(values), *history)
                if history
                else jax.tree.map(lambda value: jnp.empty((0,) + value.shape, dtype=value.dtype), records)
            )
        return carry, records

    @staticmethod
    def check_status(carry):
        """Raise on the host if the numerical trajectory failed."""
        code = int(carry.failure_code)
        if code:
            reason = {
                1: "solution diverged",
                2: "linear solver did not converge",
                3: "CFL number is not finite",
            }[code]
            raise RuntimeError(f"{reason} at iteration {int(carry.failed_iteration)}")

    def write_records(self, records, state=None):
        """Write concrete event records, accepting histories or boundary records."""
        import jax
        import numpy as np

        if state is None:
            state = self.simulation.state
        for name, record in records.items():
            valid = np.asarray(record.valid)
            indices = [None] if valid.ndim == 0 and valid else np.flatnonzero(valid) if valid.ndim else []
            for index in indices:
                event = record if index is None else jax.tree.map(lambda value: value[index], record)
                shell = self._make_state(state.variables)
                with shell.variables.unlock():
                    shell.variables.time = event.time
                    shell.variables.itt = event.iteration
                self._diagnostics[name].write_finalized(shell, event.values, time_value=event.time)
