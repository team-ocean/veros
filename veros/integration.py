"""A compiled numerical timestep, differentiable composition, and a host driver."""

from collections import namedtuple
from copy import copy
import math
import operator
from collections.abc import Mapping

from veros import runtime_settings, runtime_state
from veros.routines import tracing_context

IntegrationState = namedtuple("IntegrationState", "variables diagnostics failure_code failed_iteration")
DiagnosticRecord = namedtuple("DiagnosticRecord", "time iteration valid values")

# These settings affect frozen grid, reference data, operators, or execution schedules.
FIXED_PARAMETERS = {
    "pi",
    "radius",
    "degtom",
    "omega",
    "x_origin",
    "y_origin",
    "rho_0",
    "grav",
    "dt_mom",
    "dt_tracer",
    "runlen",
    "restart_frequency",
}


class CompiledCompatibilityError(RuntimeError):
    """A configuration needs ordinary execution."""


def compilation_reasons(simulation, linear_solver=None):
    """Inspect declarations without executing user hooks."""
    from veros import VerosSetup

    reasons = []
    if runtime_settings.backend != "jax":
        reasons.append("compiled integration requires the JAX backend")
    if simulation.state.settings.enable_partial_cells and runtime_state.proc_num != 1:
        reasons.append("Partial cells currently require serial JAX execution")
    solver = runtime_settings.linear_solver if linear_solver is None else linear_solver
    if simulation.state.settings.enable_partial_cells and solver == "petsc":
        raise ValueError("Partial cells require the variable-coefficient scipy_jax solver")
    if solver not in ("best", "scipy", "petsc", "scipy_jax"):
        raise ValueError("Unrecognized compiled linear_solver")
    if solver not in ("best", "petsc", "scipy_jax"):
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
        if not diagnostic.supports_compilation and not diagnostic.observes_model:
            reasons.append(f"diagnostic {name} must declare observes_model=True for host execution")
    return tuple(reasons)


class PreparedIntegrator:
    """Integrate captured configuration with dynamic floating variables and settings.

    Args:
        simulation: An initialized :class:`veros.VerosSetup`.
        record_diagnostics: Names whose output events should be retained. By
            default, retain enabled diagnostics except snapshots.
        checkpoint: Recompute step intermediates during differentiation.
        linear_solver: Override the compiled solver selection with ``petsc``,
            ``scipy_jax``, or ``best``. Defaults to the runtime selection.
        fixed_parameters: Additional setting names used to prepare cached data
            by this setup or its plugins. These cannot be overridden at runtime.
    """

    def __init__(
        self,
        simulation,
        *,
        record_diagnostics=None,
        checkpoint=False,
        linear_solver=None,
        fixed_parameters=(),
    ):
        simulation._ensure_setup_done()
        reasons = compilation_reasons(simulation, linear_solver)
        if reasons:
            raise CompiledCompatibilityError("; ".join(reasons))
        import jax
        from veros.core.external.solvers import get_compiled_solver
        from veros.core.geometry import validate_diagnostics

        validate_diagnostics(simulation.state)

        if not math.isfinite(simulation.state.settings.dt_tracer) or simulation.state.settings.dt_tracer <= 0:
            raise ValueError("dt_tracer must be finite and positive")
        if runtime_state.proc_num > 1:
            from veros._distributed_jax import prepare_communicator
            from veros.distributed import _mpi_comm_along_axis, proc_rank_to_index

            prepare_communicator(runtime_settings.mpi_comm)
            index = proc_rank_to_index(runtime_state.proc_rank)
            for axis in (0, 1):
                comm = _mpi_comm_along_axis(runtime_settings.mpi_comm, index[1 - axis], runtime_state.proc_rank, axis)
                prepare_communicator(comm)
        self.simulation = simulation
        self._step_dynamics = copy(simulation)._step_dynamics
        self._template = copy(simulation.state)
        self._template._settings = copy(simulation.state.settings)
        self._fixed_parameters = FIXED_PARAMETERS | set(fixed_parameters)
        unknown = self._fixed_parameters - set(self._template.settings.fields()) - FIXED_PARAMETERS
        if unknown:
            raise ValueError(f"Unknown fixed parameters: {sorted(unknown)}")
        self._template._dimensions = dict(simulation.state.dimensions)
        self._template._diagnostics = {}
        self._template._linear_solver = get_compiled_solver(simulation.state, linear_solver)
        for name, diag in simulation.state.diagnostics.items():
            for frequency in (diag.sampling_frequency, diag.output_frequency):
                if frequency is not None and (math.isnan(frequency) or frequency < 0):
                    raise ValueError(f"Diagnostic {name} has an invalid frequency: {frequency}")
        self.host_diagnostics = tuple(
            name
            for name, diag in simulation.state.diagnostics.items()
            if not diag.supports_compilation and (diag.sampling_frequency or diag.output_frequency)
        )
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
        if self._template.settings.enable_partial_cells and "overturning" in self.record_diagnostics:
            raise ValueError("Partial cells do not yet support overturning records")
        for name in self.record_diagnostics:
            if name in self.host_diagnostics:
                raise TypeError(f"Diagnostic {name} needs pure sampling/finalization for differentiated records")
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
        self._configuration = {
            name: value
            for name, value in simulation.state.settings.items()
            if simulation.state.settings.__metadata__[name].type in (bool, int) or name in self._fixed_parameters
        }
        self._runtime_configuration = (
            runtime_settings.backend,
            runtime_settings.device,
            runtime_settings.float_type,
            runtime_settings.linear_solver,
            runtime_settings.num_proc,
        )
        self._hooks = {
            name: simulation.__dict__.get(name, getattr(type(simulation), name, None))
            for name in (
                "step",
                "_step_dynamics",
                "set_forcing",
                "after_timestep",
                "rebuild_vertical_forcing",
                "rebuild_wet_forcing",
                "wet_geometry_channel",
            )
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
            if self.simulation.__dict__.get(name, getattr(type(self.simulation), name, None)) is not hook:
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

    def validate_parameters(self, parameters):
        """Normalize scalar floating overrides without inspecting traced values."""
        import jax.numpy as jnp

        if parameters is None:
            return {}
        if not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping of setting names to scalar values")
        metadata = self._template.settings.__metadata__
        normalized = {}
        for name, value in parameters.items():
            if name not in metadata:
                raise ValueError(f"Unknown setting {name!r}")
            if metadata[name].type is not float:
                raise TypeError(f"Setting {name!r} is not a floating-point setting")
            if name in self._fixed_parameters:
                raise ValueError(f"Setting {name!r} depends on fixed preparation data or schedules")
            if f"control_{name}" in self._template.variables:
                raise ValueError(f"Setting {name!r} already has a variable control; use only one control path")
            value = jnp.asarray(value)
            if value.ndim or not jnp.issubdtype(value.dtype, jnp.floating):
                raise TypeError(f"Parameter {name!r} must be a floating-point scalar")
            normalized[name] = jnp.asarray(value, dtype=runtime_settings.float_type)
        return normalized

    def with_vertical_grid(self, carry, *, widths, conserve=("temp", "salt", "momentum")):
        """Move fixed-count levels while conserving tracer, momentum, and energy inventories."""
        from veros.remapping import vertical_grid

        self.check_configuration()
        return vertical_grid(self, carry, widths, conserve=conserve)

    def grid_reference_error(self, carry, *, parameters=None, refinement, response, steps, forward=False, seed=0):
        """Evaluate a setup-declared refined reference and differentiate its scalar error."""
        self.check_configuration()
        builder = getattr(self.simulation, "build_grid_reference", None)
        if not callable(builder):
            raise ValueError("The setup must declare an independent build_grid_reference provider")
        return builder(
            carry,
            parameters=self.validate_parameters(parameters),
            refinement=refinement,
            response=response,
            steps=steps,
            forward=forward,
            seed=seed,
        )

    def with_wet_geometry(
        self,
        carry,
        *,
        opening_m,
        channel,
        realized=False,
        conserve=("temp", "salt", "momentum"),
        minimum_fraction=0.1,
    ):
        """Open selected dry columns by pooling inventories from a fixed closed reference."""
        from veros.wet_geometry import opening

        self.check_configuration()
        return opening(
            self,
            carry,
            opening_m,
            channel,
            realized=realized,
            conserve=conserve,
            minimum_fraction=minimum_fraction,
        )

    def bathymetry_bounds(self, carry, *, minimum_fraction=0.1):
        """Return interior depth bounds in meters for the fixed bottom-cell indices."""
        from veros.core.geometry import bathymetry_bounds

        if not self._template.settings.enable_partial_cells:
            raise ValueError("Bathymetry controls require enable_partial_cells=True before setup")
        if not 0 < minimum_fraction <= 1:
            raise ValueError("minimum_fraction must lie in (0, 1]")
        with tracing_context():
            lower, upper = bathymetry_bounds(self._make_state(carry.variables), minimum_fraction)
        return lower[2:-2, 2:-2], upper[2:-2, 2:-2]

    def with_bathymetry(self, carry, depth, *, minimum_fraction=0.1, conserve=None):
        """Set fixed-topology depths, optionally conserving column inventories."""
        import jax.numpy as jnp
        from veros.core.geometry import bathymetry_geometry
        from veros.core.operators import update, at

        if conserve is not None:
            if set(conserve) != {"temp", "salt", "momentum"}:
                raise ValueError("Bathymetry remapping requires temp, salt, and momentum conservation")
            self.check_configuration()
        original = carry
        lower, upper = self.bathymetry_bounds(carry, minimum_fraction=minimum_fraction)
        depth = jnp.asarray(depth, dtype=carry.variables.dzt.dtype)
        if depth.shape != upper.shape:
            raise ValueError(f"depth must have interior shape {upper.shape}, got {depth.shape}")
        valid = jnp.all(jnp.isfinite(depth) & (depth >= lower) & (depth <= upper))
        # Keep geometry finite when invalid inputs freeze the trajectory.
        safe_depth = jnp.where(valid, depth, upper)
        full_depth = update(carry.variables.ht, at[2:-2, 2:-2], safe_depth)
        with tracing_context():
            fields = bathymetry_geometry(self._make_state(carry.variables), full_depth)
        carry = self.replace_variables(carry, **fields._asdict())
        if conserve is not None:
            from veros.remapping import remap_carry

            carry = remap_carry(self, original, carry, stretch=True)
        return carry._replace(
            failure_code=jnp.where((carry.failure_code == 0) & ~valid, 4, carry.failure_code),
            failed_iteration=jnp.where((carry.failure_code == 0) & ~valid, carry.variables.itt, carry.failed_iteration),
        )

    def validate_step_inputs(self, step_inputs):
        """Normalize explicit forcing arrays to the prepared variable schema."""
        import jax.numpy as jnp

        if step_inputs is None:
            return {}
        if not isinstance(step_inputs, Mapping):
            raise TypeError("step_inputs must be a mapping of forcing variable names to arrays")
        allowed = {
            "surface_taux",
            "surface_tauy",
            "forc_temp_surface",
            "forc_salt_surface",
            "forc_tke_surface",
            "forc_iw_surface",
            "forc_iw_bottom",
            "temp_source",
            "salt_source",
        }
        values = {}
        for name, value in step_inputs.items():
            if name not in allowed or name not in self._template.variables:
                raise ValueError(f"Unsupported forcing input {name!r}")
            reference = self._template.variables.get(name)
            value = jnp.asarray(value, dtype=reference.dtype)
            if value.shape != reference.shape:
                raise ValueError(f"Forcing {name} must have shape {reference.shape}, got {value.shape}")
            values[name] = value
        return values

    def _make_state(self, variables, parameters=None, step_inputs=None):
        state = copy(self._template)
        state._variables = copy(variables)
        if parameters:
            state._settings = state.settings._with_parameters(parameters)
        state._solver_residual_ok = True
        state._forcing_inputs = step_inputs or {}
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
                valid_cfl = global_and(jnp.all(jnp.stack([jnp.isfinite(value) for value in values.values()])))
            if name in self.record_diagnostics:
                records[name] = DiagnosticRecord(state.variables.time, state.variables.itt, due, values)
        return variables, records, valid_cfl

    def _step(self, carry, parameters=None, step_inputs=None):
        import jax
        import jax.numpy as jnp
        from veros.core import numerics, isoneutral

        def advance(current):
            state = self._make_state(current.variables, parameters, step_inputs)
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

    def step(self, carry, *, parameters=None, step_inputs=None):
        """Return one compiled timestep and event records without a leading step axis."""
        self.check_configuration()
        return self._compiled_step(carry, self.validate_parameters(parameters), self.validate_step_inputs(step_inputs))

    def advance(self, carry, *, n_steps, final_only=False, parameters=None, step_inputs=None):
        """Compose compiled timesteps in a Python loop and return output records.

        Records have a leading step axis unless ``final_only=True``. Their
        ``valid`` masks identify output events. Failed trajectories carry a
        nonzero failure code and freeze after the first failed step.
        ``parameters`` overrides floating step-time settings through
        ``state.settings`` without modifying the model or continuation state.
        JAX can differentiate this composition without jitting the whole loop.
        Every horizon reuses the same compiled timestep.
        """
        n_steps = operator.index(n_steps)
        if n_steps < 0:
            raise ValueError("n_steps must be nonnegative")
        self.check_configuration()
        import jax
        import jax.numpy as jnp

        parameters = self.validate_parameters(parameters)
        records = self._empty_records(carry)
        history = []
        if step_inputs is not None and len(step_inputs) != n_steps:
            raise ValueError("step_inputs must contain one mapping per timestep")
        for index in range(n_steps):
            inputs = self.validate_step_inputs(None if step_inputs is None else step_inputs[index])
            carry, records = self._compiled_step(carry, parameters, inputs)
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
                4: "bathymetry is outside fixed-topology depth bounds",
                5: "vertical grid violates positive widths or staggered spacings, fixed depth, or fixed-topology bounds",
                6: "wet geometry violates channel selection or admissible depth bounds",
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

    def write_host_diagnostics(self, carry):
        """Run observer diagnostics on the concrete pre-rotation state."""
        if not self.host_diagnostics:
            return
        state = copy(self.simulation.state)
        state._settings = copy(state.settings)
        state._variables = copy(carry.variables)
        with state.variables.unlock():
            vs = state.variables
            vs.taum1, vs.tau, vs.taup1 = vs.taup1, vs.taum1, vs.tau
        before = dict(state.variables.items())
        before_settings = dict(state.settings.items())
        for name in self.host_diagnostics:
            diag = state.diagnostics[name]
            if diag.sampling_frequency and float(vs.time) % diag.sampling_frequency < state.settings.dt_tracer:
                diag.diagnose(state)
        for name in self.host_diagnostics:
            diag = state.diagnostics[name]
            if diag.output_frequency and float(vs.time) % diag.output_frequency < state.settings.dt_tracer:
                diag.output(state)
        if any(state.variables.get(name) is not value for name, value in before.items()):
            raise RuntimeError("Host observer diagnostic mutated model variables; use ordinary execution")
        if dict(state.settings.items()) != before_settings:
            raise RuntimeError("Host observer diagnostic mutated model settings; use ordinary execution")


def run_compiled(
    simulation,
    *,
    n_steps=None,
    checkpoint=False,
    show_progress_bar=None,
    linear_solver=None,
    step_inputs=None,
    prepared=None,
):
    """Run compiled timesteps with host diagnostics, restarts, and progress."""
    import time as wall_time
    import jax
    from veros import progress, restart, signals, logger, time
    from veros.timer import timer_context

    simulation._ensure_setup_done()
    settings = simulation.state.settings
    if not math.isfinite(settings.dt_tracer) or settings.dt_tracer <= 0:
        raise ValueError("dt_tracer must be finite and positive")
    n_steps = math.ceil(settings.runlen / settings.dt_tracer) if n_steps is None else operator.index(n_steps)
    if n_steps < 0:
        raise ValueError("n_steps must be nonnegative")
    names = tuple(
        name
        for name, diag in simulation.state.diagnostics.items()
        if diag.output_frequency and diag.supports_compilation
    )
    prepared = prepared or PreparedIntegrator(
        simulation,
        record_diagnostics=names,
        checkpoint=checkpoint,
        linear_solver=linear_solver,
    )
    carry = prepared.pack(simulation.state)
    inputs = None if step_inputs is None else iter(step_inputs)
    length, unit = time.format_time(n_steps * settings.dt_tracer)
    logger.info(f"\nStarting integration for {length:.1f} {unit} (compiled)")
    pbar = progress.get_progress_bar(
        simulation.state, use_tqdm=show_progress_bar, total=float(carry.variables.time) + n_steps * settings.dt_tracer
    )
    try:
        timer_context.active = False
        with signals.signals_to_exception(), pbar:
            for index in range(n_steps):
                with simulation.state.timers["diagnostics"]:
                    restart.write_restart(simulation.state)
                with simulation.state.timers["forcing"]:
                    try:
                        step_input = None if inputs is None else next(inputs)
                    except StopIteration:
                        raise ValueError("step_inputs must contain one mapping per timestep") from None
                started = wall_time.perf_counter()
                with jax.profiler.StepTraceAnnotation("veros", step_num=index), simulation.state.timers["main"]:
                    carry, records = prepared.step(carry, step_inputs=step_input)
                    jax.block_until_ready(carry)
                if index == 0:
                    logger.info(
                        "First compiled timestep (including compilation): {:.2f}s", wall_time.perf_counter() - started
                    )
                prepared.install(carry)
                prepared.check_status(carry)
                with simulation.state.timers["diagnostics"]:
                    prepared.write_records(records)
                    prepared.write_host_diagnostics(carry)
                logger.debug(" Time step took {:.2f}s", simulation.state.timers["main"].last_time)
                timer_context.active = True
                pbar.advance_time(settings.dt_tracer)
            marker = object()
            if inputs is not None and next(inputs, marker) is not marker:
                raise ValueError("step_inputs contains more mappings than timesteps")
    except BaseException:
        logger.critical("Stopping integration at iteration {}", simulation.state.variables.itt)
        raise
    else:
        logger.success("Integration done\n")
    finally:
        timer_context.active = True
        restart.write_restart(simulation.state, force=True)
        logger.debug(
            "Compiled timing: timesteps {:.2f}s; host forcing {:.2f}s; diagnostics and I/O {:.2f}s "
            "(excluding first iteration)",
            simulation.state.timers["main"].total_time,
            simulation.state.timers["forcing"].total_time,
            simulation.state.timers["diagnostics"].total_time,
        )
    return carry
