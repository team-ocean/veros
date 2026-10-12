"""Execution selection, host observers, and explicit forcing contracts."""

from copy import copy

import numpy as np
import pytest

from veros import runtime_settings, veros_routine
from veros.integration import CompiledCompatibilityError, compilation_reasons


@pytest.fixture(autouse=True)
def execution_options():
    old = runtime_settings.diskless_mode, runtime_settings.linear_solver
    object.__setattr__(runtime_settings, "diskless_mode", True)
    object.__setattr__(runtime_settings, "linear_solver", "scipy_jax" if runtime_settings.backend == "jax" else "scipy")
    yield
    object.__setattr__(runtime_settings, "diskless_mode", old[0])
    object.__setattr__(runtime_settings, "linear_solver", old[1])


def make_simulation():
    from veros.setups.acc_basic import ACCBasicSetup
    from veros.core.operators import numpy as npx, update, at

    class Simulation(ACCBasicSetup):
        @veros_routine
        def set_diagnostics(self, state):
            for diag in state.diagnostics.values():
                diag.sampling_frequency = diag.output_frequency = None

        @veros_routine(supports_compilation=True)
        def _step_dynamics(self, state):
            vs = state.variables
            self.set_forcing(state)
            vs.update(getattr(state, "_forcing_inputs", {}))
            vs.temp = update(vs.temp, at[..., vs.taup1], vs.temp[..., vs.tau] + vs.forc_temp_surface[..., None])
            vs.itt = vs.itt + npx.int32(1)
            vs.time = vs.time + state.settings.dt_tracer
            self.after_timestep(state)

    sim = Simulation(override=dict(nx=8, ny=8, dt_mom=1, dt_tracer=1, runlen=3))
    sim.setup()
    return sim


def require_jax():
    if runtime_settings.backend != "jax":
        pytest.skip("requires JAX")


def test_selection_and_ordinary_fallback():
    sim = make_simulation()
    sim.run(show_progress_bar=False)
    assert sim.execution_mode == ("compiled" if runtime_settings.backend == "jax" else "eager")
    assert int(sim.state.variables.itt) == 3
    events = []

    @veros_routine
    def host_hook(state):
        events.append(int(state.variables.itt))

    sim.after_timestep = host_hook
    sim.run(show_progress_bar=False)
    assert sim.execution_mode == "eager"
    assert "after_timestep" in sim.execution_reason
    assert events == [4, 5, 6]
    with pytest.raises(CompiledCompatibilityError):
        sim.run(execution="compiled")
    assert events == [4, 5, 6]
    sim.run(show_progress_bar=False, execution="eager")
    assert events == [4, 5, 6, 7, 8, 9]
    with pytest.raises(ValueError, match="execution"):
        sim.run(execution="unknown")
    object.__setattr__(runtime_settings, "linear_solver", "scipy")
    assert any("solver 'scipy'" in reason for reason in compilation_reasons(sim))
    if runtime_settings.backend == "jax":
        from veros.core.external.solvers import get_compiled_solver

        with pytest.raises(ValueError, match="Compiled linear_solver"):
            get_compiled_solver(sim.state)


@pytest.mark.parametrize("override_location", ("class", "instance"))
def test_step_override_uses_ordinary_execution(monkeypatch, override_location):
    sim = make_simulation()
    prepared = sim.prepare_integrator(record_diagnostics=()) if runtime_settings.backend == "jax" else None
    base_step = sim.step
    events = []

    @veros_routine
    def custom_step(self, state):
        events.append(int(state.variables.itt))
        base_step(state)

    if override_location == "class":
        monkeypatch.setattr(type(sim), "step", custom_step)
    else:
        monkeypatch.setattr(sim, "step", custom_step.__get__(sim, type(sim)))

    if prepared is not None:
        with pytest.raises(CompiledCompatibilityError, match="step"):
            prepared.step(prepared.pack(sim.state))
    sim.run(show_progress_bar=False)
    assert sim.execution_mode == "eager"
    assert "step" in sim.execution_reason
    assert events == [0, 1, 2]
    with pytest.raises(CompiledCompatibilityError, match="step"):
        sim.run(execution="compiled")
    assert events == [0, 1, 2]


@pytest.mark.parametrize("override_location", ("class", "instance"))
def test_diagnostic_output_override_uses_ordinary_execution(monkeypatch, override_location):
    from veros.diagnostics.tracer_monitor import TracerMonitor

    sim = make_simulation()
    events = []

    if override_location == "class":

        class CustomTracerMonitor(TracerMonitor):
            def output(self, state):
                events.append(int(state.variables.itt))

        diagnostic = CustomTracerMonitor(sim.state)
        diagnostic.initialize(sim.state)
        sim.state.diagnostics["tracer_monitor"] = diagnostic
        diagnostic.output_frequency = 1.0
    else:
        diagnostic = sim.state.diagnostics["tracer_monitor"]
        diagnostic.output_frequency = 1.0
        prepared = sim.prepare_integrator(record_diagnostics=()) if runtime_settings.backend == "jax" else None
        monkeypatch.setattr(diagnostic, "output", lambda state: events.append(int(state.variables.itt)))
        if prepared is not None:
            with pytest.raises(CompiledCompatibilityError, match="diagnostic"):
                prepared.step(prepared.pack(sim.state))

    sim.run(show_progress_bar=False)
    assert sim.execution_mode == "eager"
    assert "output" in sim.execution_reason
    assert events == [1, 2, 3]
    with pytest.raises(CompiledCompatibilityError, match="output"):
        sim.run(execution="compiled")
    assert events == [1, 2, 3]


def test_diagnostic_with_declared_finalization_contract_compiles():
    require_jax()
    from veros.diagnostics.tracer_monitor import TracerMonitor

    events = []

    class CustomTracerMonitor(TracerMonitor):
        supports_compilation = True

        def output(self, state):
            self.write_finalized(state, self.finalize(state))

        def write_finalized(self, state, values, time_value=None):
            events.append(int(state.variables.itt))

    sim = make_simulation()
    diagnostic = CustomTracerMonitor(sim.state)
    diagnostic.output_frequency = 1.0
    diagnostic.initialize(sim.state)
    sim.state.diagnostics["tracer_monitor"] = diagnostic
    sim.run(show_progress_bar=False)
    assert sim.execution_mode == "compiled"
    assert events == [1, 2, 3]


def test_declarations_and_structural_changes(monkeypatch):
    require_jax()
    sim = make_simulation()
    prepared = sim.prepare_integrator(record_diagnostics=())
    carry = prepared.pack(sim.state)
    with sim.state.settings.unlock():
        sim.state.settings.enable_tke = False
    with pytest.raises(CompiledCompatibilityError, match="reprepare"):
        prepared.step(carry)
    reconfigured = sim.prepare_integrator(record_diagnostics=())
    current = reconfigured.pack(sim.state)
    reconfigured.step(current)
    sim.state.diagnostics["averages"].output_frequency = 2
    with pytest.raises(CompiledCompatibilityError, match="diagnostic schedules"):
        reconfigured.step(current)
    sim.state.diagnostics["averages"].output_frequency = None
    monkeypatch.setattr(type(sim).set_forcing, "dist_safe", False)
    from veros import runtime_state

    monkeypatch.setattr(type(runtime_state), "proc_num", property(lambda self: 2))
    assert any("requires host gathering" in reason for reason in compilation_reasons(sim))
    from types import SimpleNamespace

    sim._plugin_interfaces = (SimpleNamespace(name="host_plugin", run_entrypoint=lambda state: None),)
    assert any("host_plugin" in reason for reason in compilation_reasons(sim))


def test_legacy_observer_frequency_rotation_and_resets(tmp_path):
    require_jax()
    from veros.diagnostics.base import VerosDiagnostic
    from veros.variables import Variable

    class Counter(VerosDiagnostic):
        name = "counter"
        observes_model = True
        sampling_frequency = 1.0
        output_frequency = 2.5
        var_meta = {
            "count": Variable("Count", None, write_to_restart=True),
            "total": Variable("Total", None, write_to_restart=True),
        }

        def initialize(self, state):
            self.initialize_variables(state)
            self.samples, self.outputs = [], []

        def diagnose(self, state):
            self.samples.append((int(state.variables.itt), int(state.variables.tau)))
            self.variables.count += 1
            self.variables.total += state.variables.temp[..., state.variables.tau].sum()

        def output(self, state):
            self.outputs.append((float(state.variables.time), float(self.variables.total / self.variables.count)))
            self.variables.count = self.variables.count * 0
            self.variables.total = self.variables.total * 0

    sim = make_simulation()
    counter = Counter(sim.state)
    counter.initialize(sim.state)
    sim.state.diagnostics["counter"] = counter
    with sim.state.variables.unlock():
        sim.state.variables.time = 0.5
    prepared = sim.prepare_integrator(record_diagnostics=())
    initial = prepared.pack(sim.state)
    expected, _ = prepared.advance(initial, n_steps=5)
    assert counter.samples == []
    from veros import restart

    with sim.state.settings.unlock():
        sim.state.settings.restart_output_filename = str(tmp_path / "observer-restart.h5")
        sim.state.settings.restart_input_filename = str(tmp_path / "observer-restart.h5")
    object.__setattr__(runtime_settings, "diskless_mode", False)
    sim.run_compiled(n_steps=3, show_progress_bar=False)
    assert float(counter.variables.count) == 1
    counter.variables.count = counter.variables.count * 0
    counter.variables.total = counter.variables.total * 0
    with sim.state.variables.unlock():
        sim.state.variables.time = 0
        sim.state.variables.itt = 0
    restart.read_restart(sim.state)
    assert float(counter.variables.count) == 1
    assert float(sim.state.variables.time) == 3.5
    sim.run_compiled(n_steps=2, show_progress_bar=False)
    assert counter.samples == [(1, 1), (2, 2), (3, 0), (4, 1), (5, 2)]
    assert [time for time, _ in counter.outputs] == [2.5, 5.5]
    assert float(counter.variables.count) == 0
    actual = copy(sim.state.variables)
    with sim.state.variables.unlock():
        sim.state.variables.update(initial.variables)
    counter.samples.clear()
    counter.outputs.clear()
    for _ in range(5):
        sim.step(sim.state)
    np.testing.assert_allclose(actual.temp, sim.state.variables.temp)
    np.testing.assert_allclose(expected.variables.temp, actual.temp)
    assert counter.samples == [(1, 1), (2, 2), (3, 0), (4, 1), (5, 2)]
    assert [time for time, _ in counter.outputs] == [2.5, 5.5]
    with pytest.raises(TypeError, match="pure sampling"):
        sim.prepare_integrator(record_diagnostics=("counter",))
    counter.observes_model = False
    assert any("counter" in reason for reason in compilation_reasons(sim))


def test_forcing_schema_and_pure_derivative():
    require_jax()
    import jax
    import jax.numpy as jnp

    sim = make_simulation()
    prepared = sim.prepare_integrator(record_diagnostics=())
    initial = prepared.pack(sim.state)
    shape = initial.variables.forc_temp_surface.shape

    def loss(scale):
        inputs = [{"forc_temp_surface": jnp.full(shape, scale * index)} for index in (1, 2, 3)]
        final, _ = prepared.advance(initial, n_steps=3, step_inputs=inputs)
        return final.variables.temp[..., final.variables.tau].sum()

    derivative = jax.grad(loss)(1.0)
    np.testing.assert_allclose(derivative, 6 * np.prod(shape) * initial.variables.temp.shape[2])
    for epsilon in (1e-3, 1e-4):
        np.testing.assert_allclose(derivative, (loss(1 + epsilon) - loss(1 - epsilon)) / (2 * epsilon), rtol=1e-8)
    with pytest.raises(ValueError, match="one mapping"):
        prepared.advance(initial, n_steps=2, step_inputs=[{}])
    with pytest.raises(ValueError, match="shape"):
        prepared.step(initial, step_inputs={"forc_temp_surface": jnp.ones(2)})
    with pytest.raises(ValueError, match="Unsupported"):
        prepared.step(initial, step_inputs={"temp": initial.variables.temp})
    with pytest.raises(ValueError, match="Unsupported"):
        prepared.step(initial, step_inputs={"forc_iw_surface": jnp.ones(shape)})
    with pytest.raises(ValueError, match="one mapping"):
        sim.run_compiled(n_steps=2, step_inputs=[{}], show_progress_bar=False)
    with pytest.raises(ValueError, match="more mappings"):
        sim.run_compiled(n_steps=1, step_inputs=[{}, {}], show_progress_bar=False)


def test_real_step_applies_forcing_after_native_hook():
    require_jax()
    import jax.numpy as jnp
    from veros.setups.acc_basic import ACCBasicSetup

    sim = ACCBasicSetup(
        override=dict(
            nx=8,
            ny=8,
            dt_mom=600,
            dt_tracer=600,
            enable_tke=False,
            enable_neutral_diffusion=False,
            enable_skew_diffusion=False,
        )
    )
    sim.setup()
    prepared = sim.prepare_integrator(record_diagnostics=())
    initial = prepared.pack(sim.state)
    forcing = jnp.full_like(initial.variables.forc_temp_surface, 0.01)
    final, _ = prepared.step(initial, step_inputs={"forc_temp_surface": forcing})
    prepared.check_status(final)
    np.testing.assert_array_equal(final.variables.forc_temp_surface, forcing)
    assert prepared._compiled_step._cache_size() == 1
