import numpy as np
import pytest

from veros import runtime_settings, veros_routine


@pytest.fixture(autouse=True)
def integration_options():
    if runtime_settings.backend != "jax":
        pytest.skip("compiled integration requires JAX")
    original = runtime_settings.diskless_mode
    original_solver = runtime_settings.linear_solver
    object.__setattr__(runtime_settings, "diskless_mode", True)
    object.__setattr__(runtime_settings, "linear_solver", "scipy_jax")
    yield
    object.__setattr__(runtime_settings, "diskless_mode", original)
    object.__setattr__(runtime_settings, "linear_solver", original_solver)


def make_simulation(**overrides):
    from veros.setups.acc_basic import ACCBasicSetup

    class Simulation(ACCBasicSetup):
        @veros_routine
        def set_diagnostics(self, state):
            super().set_diagnostics(state)
            for diagnostic in state.diagnostics.values():
                diagnostic.output_frequency = state.settings.dt_tracer * 3
                if diagnostic.name in ("averages", "energy", "overturning"):
                    diagnostic.sampling_frequency = state.settings.dt_tracer
            state.diagnostics["snapshot"].output_variables = ("temp", "u", "dxt")

    return Simulation(override=dict(nx=8, ny=8, dt_tracer=600, dt_mom=600, **overrides))


def assert_variables_equal(actual, expected, *, rtol=2e-8, atol=2e-7):
    assert actual.fields() == expected.fields()
    for key in actual.fields():
        np.testing.assert_allclose(actual.get(key), expected.get(key), rtol=rtol, atol=atol, err_msg=key)


@pytest.mark.parametrize("streamfunction", (True, False))
def test_step_diagnostics_and_continuation(streamfunction, monkeypatch):
    import jax

    sim = make_simulation(enable_streamfunction=streamfunction)
    sim.setup()
    names = tuple(sim.state.diagnostics)
    prepared = sim.prepare_integrator(record_diagnostics=names)
    initial = prepared.pack(sim.state)
    unchanged, empty = prepared.advance(initial, n_steps=0)
    assert unchanged is initial
    for record in empty.values():
        assert record.valid.shape == (0,)
        assert all(value.shape[0] == 0 for value in record.values.values())
    _, boundary = prepared.advance(initial, n_steps=0, final_only=True)
    assert all(record.valid.shape == () and not record.valid for record in boundary.values())
    saved = {key: np.array(value) for key, value in initial.variables.items()}
    captured = {name: [] for name in names}
    for name, diagnostic in sim.state.diagnostics.items():

        def capture(state, values, time_value=None, name=name):
            captured[name].append(
                (float(state.variables.time), {key: np.array(value) for key, value in values.items()})
            )

        monkeypatch.setattr(diagnostic, "write_finalized", capture)

    final, records = prepared.advance(initial, n_steps=6)
    prepared.check_status(final)
    first, _ = prepared.advance(initial, n_steps=2)
    split, _ = prepared.advance(first, n_steps=4)
    assert_variables_equal(final.variables, split.variables)
    for name in initial.diagnostics:
        assert_variables_equal(final.diagnostics[name], split.diagnostics[name])
    for _ in range(6):
        sim.step(sim.state)
    assert_variables_equal(final.variables, sim.state.variables)
    for name in final.diagnostics:
        assert_variables_equal(final.diagnostics[name], sim.state.diagnostics[name].variables)
    for name, record in records.items():
        valid = np.flatnonzero(record.valid)
        assert len(valid) == len(captured[name]) == 2
        for index, (time_value, values) in zip(valid, captured[name]):
            assert float(record.time[index]) == time_value
            for key, value in values.items():
                np.testing.assert_allclose(
                    record.values[key][index], value, rtol=2e-7, atol=2e-6, err_msg=f"{name}.{key}"
                )
    for key, value in saved.items():
        np.testing.assert_array_equal(initial.variables.get(key), value)
    assert not any(isinstance(value, jax.core.Tracer) for value in sim.state.variables.values())
    assert (
        jax.eval_shape(lambda carry: prepared.advance(carry, n_steps=6), initial)[0].variables.temp.shape
        == saved["temp"].shape
    )
    assert not prepared._template.profile_timers[next(iter(prepared._template.profile_timers))].__dict__.get("tracer")


@pytest.mark.parametrize("checkpoint", (False, True))
def test_trajectory_gradient(checkpoint):
    import jax
    import jax.numpy as jnp

    sim = make_simulation()
    sim.setup()
    prepared = sim.prepare_integrator(record_diagnostics=("averages",), checkpoint=checkpoint)
    initial = prepared.pack(sim.state)

    def objective(scale, split=False):
        carry = prepared.replace_variables(initial, t_star=scale * initial.variables.t_star)
        if split:
            carry, a = prepared.advance(carry, n_steps=3)
            carry, b = prepared.advance(carry, n_steps=3)
            records = jax.tree.map(lambda a, b: jnp.concatenate((a, b)), a, b)
        else:
            carry, records = prepared.advance(carry, n_steps=6)
        mean = records["averages"].values["temp"][:, 2:-2, 2:-2].sum(axis=(1, 2, 3))
        return jnp.sum(jnp.where(records["averages"].valid, mean, 0))

    value, gradient = jax.value_and_grad(objective)(1.0)
    assert np.isfinite(value) and np.isfinite(gradient) and abs(gradient) > 1e-6
    for step in (1e-3, 1e-4):
        finite_difference = (objective(1 + step) - objective(1 - step)) / (2 * step)
        np.testing.assert_allclose(gradient, finite_difference, rtol=5e-5, atol=1e-5)
    jax.clear_caches()
    _, tangent = jax.jvp(objective, (jnp.array(1.0),), (jnp.array(1.0),))
    np.testing.assert_allclose(tangent, gradient, rtol=1e-8)
    jax.clear_caches()
    split_gradient = jax.grad(lambda a: objective(a, True))(1.0)
    np.testing.assert_allclose(split_gradient, gradient, rtol=1e-8)


def test_failures_zero_steps_and_configuration():
    import jax
    from veros.routines import CURRENT_CONTEXT

    sim = make_simulation()
    sim.setup()
    prepared = sim.prepare_integrator(record_diagnostics=())
    initial = prepared.pack(sim.state)

    @veros_routine
    def replaced_hook(state):
        raise AssertionError("prepared integrator must retain its hook configuration")

    sim.after_timestep = replaced_hook
    with pytest.raises(RuntimeError, match="reprepare"):
        prepared.step(initial)
    del sim.after_timestep
    unchanged, records = prepared.advance(initial, n_steps=0)
    assert records == {}
    assert_variables_equal(initial.variables, unchanged.variables)
    with pytest.raises(ValueError, match="nonnegative"):
        prepared.advance(initial, n_steps=-1)
    with pytest.raises(ValueError, match="Unknown diagnostic"):
        sim.prepare_integrator(record_diagnostics=("missing",))
    invalid = prepared.replace_variables(initial, u=initial.variables.u * np.nan)
    failed, _ = prepared.advance(invalid, n_steps=3)
    assert int(failed.failed_iteration) == 1
    assert float(failed.variables.time) == sim.state.settings.dt_tracer
    with pytest.raises(RuntimeError, match="iteration 1"):
        prepared.check_status(failed)
    compiled_signatures = prepared._compiled_step._cache_size()
    changed = prepared.replace_variables(initial, t_star=initial.variables.t_star * 1.01)
    valid, _ = prepared.advance(changed, n_steps=3)
    prepared.check_status(valid)
    assert prepared._compiled_step._cache_size() == compiled_signatures
    assert not CURRENT_CONTEXT.tracing
    single, event = prepared.step(initial)
    history, events = prepared.advance(initial, n_steps=1)
    assert_variables_equal(single.variables, history.variables)
    assert not event and not events
    prepared.advance(initial, n_steps=6, final_only=True)
    assert prepared._compiled_step._cache_size() == compiled_signatures
    sim.state.settings.__dict__["dt_tracer"] = 300.0
    assert prepared._template.settings.dt_tracer == 600.0
    assert not any(isinstance(value, jax.core.Tracer) for value in vars(prepared._template).values())


def test_control_gradients():
    import jax
    from veros.setups.acc_basic import ACCBasicSetup

    sim = ACCBasicSetup(
        override=dict(nx=8, ny=8, dt_mom=600, dt_tracer=600, enable_tke=False),
        control_parameters=("kappaH_0", "K_iso_0"),
    )
    sim.setup()
    prepared = sim.prepare_integrator(record_diagnostics=())
    initial = prepared.pack(sim.state)

    def objective(coefficient):
        carry = prepared.replace_variables(initial, control_kappaH_0=coefficient)
        final, _ = prepared.advance(carry, n_steps=3)
        return final.variables.temp[2:-2, 2:-2, -1, final.variables.tau].sum()

    x = float(initial.variables.control_kappaH_0)
    gradient = jax.grad(objective)(x)
    step = max(x * 1e-3, 1e-7)
    difference = (objective(x + step) - objective(x - step)) / (2 * step)
    assert np.isfinite(gradient) and abs(gradient) > 1e-5
    np.testing.assert_allclose(gradient, difference, rtol=1e-4, atol=1e-4)


def test_control_parameterization_inputs():
    import jax
    import jax.numpy as jnp
    from veros.core import eke, tke
    from veros.routines import tracing_context
    from veros.setups.acc_basic import ACCBasicSetup

    controls = ("kappaH_0", "kappaM_0", "K_iso_0", "K_gm_0")
    sim = ACCBasicSetup(
        override=dict(nx=8, ny=8, dt_mom=600, dt_tracer=600, enable_tke=False), control_parameters=controls
    )
    sim.setup()
    prepared = sim.prepare_integrator(record_diagnostics=())
    initial = prepared.pack(sim.state)

    def coefficients(values):
        carry = prepared.replace_variables(initial, **{f"control_{name}": values[i] for i, name in enumerate(controls)})
        state = prepared._make_state(carry.variables)
        with tracing_context(), state.variables.unlock():
            state.variables.Nsqr = jnp.ones_like(state.variables.Nsqr)
            tke.set_tke_diffusivities(state)
            eke.set_eke_diffusivities(state)
        return jnp.array([jnp.mean(state.variables.get(name)) for name in ("kappaH", "kappaM", "K_iso", "K_gm")])

    point = jnp.arange(1.0, 5.0)
    np.testing.assert_allclose(jax.jit(coefficients)(point), point)
    np.testing.assert_allclose(jax.jit(jax.jacfwd(coefficients))(point), np.eye(4), atol=1e-12)
    np.testing.assert_allclose(jax.jit(jax.jacrev(coefficients))(point), np.eye(4), atol=1e-12)


def test_output_and_restart(tmp_path, monkeypatch):
    import h5netcdf
    from veros import restart
    from veros.integration import PreparedIntegrator

    host_iterations = []
    write_records = PreparedIntegrator.write_records

    def capture_records(self, records, state=None):
        host_iterations.append(int(self.simulation.state.variables.itt))
        return write_records(self, records, state)

    monkeypatch.setattr(PreparedIntegrator, "write_records", capture_records)

    sim = make_simulation()
    sim.setup()
    for name, diagnostic in sim.state.diagnostics.items():
        if name not in ("tracer_monitor", "cfl_monitor"):
            diagnostic.output_path = str(tmp_path / f"{name}.nc")
    with sim.state.settings.unlock():
        sim.state.settings.restart_output_filename = str(tmp_path / "restart.h5")
        sim.state.settings.restart_frequency = 1200.0
    prepared = sim.prepare_integrator(record_diagnostics=tuple(sim.state.diagnostics))
    initial = prepared.pack(sim.state)
    expected, records = prepared.advance(initial, n_steps=6)
    object.__setattr__(runtime_settings, "diskless_mode", False)
    sim.run_compiled(n_steps=4, show_progress_bar=False)
    partial, _ = prepared.advance(initial, n_steps=4)
    with sim.state.settings.unlock():
        sim.state.settings.restart_input_filename = str(tmp_path / "restart.h5")
    with sim.state.variables.unlock():
        sim.state.variables.time = 0
        sim.state.variables.itt = 0
    restart.read_restart(sim.state)
    restored_partial = prepared.pack(sim.state)
    assert_variables_equal(partial.variables, restored_partial.variables)
    assert float(restored_partial.diagnostics["averages"].average_nitts) == 1
    for name in partial.diagnostics:
        for key, meta in partial.diagnostics[name].__metadata__.items():
            if meta.write_to_restart and meta.active:
                np.testing.assert_allclose(
                    restored_partial.diagnostics[name].get(key), partial.diagnostics[name].get(key)
                )
    sim.run_compiled(n_steps=2, show_progress_bar=False)
    assert host_iterations == list(range(1, 7))
    assert_variables_equal(expected.variables, sim.state.variables)
    for name in ("averages", "energy", "overturning", "snapshot"):
        with h5netcdf.File(tmp_path / f"{name}.nc") as f:
            np.testing.assert_allclose(f.variables["Time"][:], np.asarray(records[name].time)[[2, 5]] / 86400)
            assert len(f.dimensions["Time"]) == 2
    with h5netcdf.File(tmp_path / "snapshot.nc") as f:
        value = np.asarray(records["snapshot"].values["temp"])[[2, 5], 2:-2, 2:-2]
        mask = np.asarray(initial.variables.maskT)[2:-2, 2:-2]
        value = np.where(mask, value, -1e18).transpose(0, 3, 2, 1)
        np.testing.assert_allclose(f.variables["temp"][:], value, rtol=1e-8, atol=1e-7)
    with sim.state.variables.unlock():
        sim.state.variables.time = 0
        sim.state.variables.itt = 0
    with sim.state.settings.unlock():
        sim.state.settings.restart_input_filename = str(tmp_path / "restart.h5")
    restart.read_restart(sim.state)
    restored = prepared.pack(sim.state)
    assert_variables_equal(expected.variables, restored.variables)
    import h5py

    with h5py.File(tmp_path / "restart.h5", "r+") as restart_file:
        del restart_file["core"]["itt"]
    with sim.state.variables.unlock():
        sim.state.variables.itt = 0
    restart.read_restart(sim.state)
    assert int(sim.state.variables.itt) == int(expected.variables.itt)
    for name in expected.diagnostics:
        for key, meta in restored.diagnostics[name].__metadata__.items():
            if meta.write_to_restart and meta.active:
                np.testing.assert_allclose(restored.diagnostics[name].get(key), expected.diagnostics[name].get(key))
    continued, _ = prepared.advance(restored, n_steps=2)
    direct, _ = prepared.advance(initial, n_steps=8)
    assert_variables_equal(direct.variables, continued.variables)


def test_parameterizations_and_initial_tracers():
    import jax
    import jax.numpy as jnp

    sim = make_simulation(enable_eke=True, enable_idemix=True, eq_of_state_type=5)
    sim.setup()
    prepared = sim.prepare_integrator(record_diagnostics=(), checkpoint=True)
    initial = prepared.pack(sim.state)
    final, _ = prepared.advance(initial, n_steps=6)
    prepared.check_status(final)
    for _ in range(6):
        sim.step(sim.state)
    assert_variables_equal(final.variables, sim.state.variables, rtol=2e-7, atol=2e-7)
    jax.clear_caches()

    def objective(shift):
        carry = prepared.replace_variables(
            initial,
            temp=initial.variables.temp + shift[0] * initial.variables.maskT[..., None],
            salt=initial.variables.salt + shift[1] * initial.variables.maskT[..., None],
        )
        result, _ = prepared.advance(carry, n_steps=6, final_only=True)
        vs = result.variables
        return (vs.temp[2:-2, 2:-2, :, vs.tau] + 0.1 * vs.salt[2:-2, 2:-2, :, vs.tau]).sum()

    point = jnp.array([0.0, 0.0])
    direction = jnp.array([0.1, 0.01])
    gradient = jax.grad(objective)(point)
    assert np.isfinite(gradient).all()
    assert np.all(np.abs(gradient) > 1e-6)
    jax.clear_caches()
    _, tangent = jax.jvp(objective, (point,), (direction,))
    np.testing.assert_allclose(jnp.vdot(gradient, direction), tangent, rtol=1e-7)
    jax.clear_caches()
    for step in (1e-3, 1e-4):
        difference = (objective(point + step * direction) - objective(point - step * direction)) / (2 * step)
        np.testing.assert_allclose(tangent, difference, rtol=5e-4, atol=1e-4)


def test_hook_trace_errors_restore_context():
    from veros.routines import CURRENT_CONTEXT

    sim = make_simulation()
    sim.setup()

    @veros_routine(supports_compilation=True)
    def invalid_hook(state):
        if state.variables.time > 0:
            state.variables.time += 1

    sim.after_timestep = invalid_hook
    with pytest.raises(RuntimeError, match="Cannot trace the model step") as error:
        sim.prepare_integrator(record_diagnostics=())
    assert "boolean conversion" in str(error.value.__cause__).lower()
    assert not CURRENT_CONTEXT.tracing


def test_traceable_plugin_and_hook():
    from types import SimpleNamespace
    from veros.variables import Variable
    from veros.setups.acc_basic import ACCBasicSetup
    import jax

    @veros_routine
    def initialize(state):
        pass

    @veros_routine(supports_compilation=True)
    def run_plugin(state):
        state.variables.plugin_total += state.variables.temp[2:-2, 2:-2, :, state.variables.tau].sum()

    plugin = SimpleNamespace(
        __name__="integration_plugin",
        __VEROS_INTERFACE__=dict(
            setup_entrypoint=initialize,
            run_entrypoint=run_plugin,
            variables=dict(plugin_total=Variable("Plugin total", None)),
        ),
    )

    class PluginSetup(ACCBasicSetup):
        __veros_plugins__ = (plugin,)

        @veros_routine(supports_compilation=True)
        def after_timestep(self, state):
            state.variables.plugin_total += state.variables.time

    sim = PluginSetup(override=dict(nx=8, ny=8, dt_mom=600, dt_tracer=600))
    sim.setup()
    prepared = sim.prepare_integrator(record_diagnostics=())
    initial = prepared.pack(sim.state)
    final, _ = prepared.advance(initial, n_steps=3)
    for _ in range(3):
        sim.step(sim.state)
    assert_variables_equal(final.variables, sim.state.variables)
    gradient = jax.grad(
        lambda a: prepared.advance(prepared.replace_variables(initial, t_star=a * initial.variables.t_star), n_steps=3)[
            0
        ].variables.plugin_total
    )(1.0)
    assert np.isfinite(gradient) and abs(gradient) > 1e-6


@pytest.mark.parametrize("streamfunction", (True, False))
def test_solver_boundary_and_rhs_gradients(streamfunction):
    import jax
    import jax.numpy as jnp
    from veros.routines import tracing_context

    sim = make_simulation(enable_streamfunction=streamfunction)
    sim.setup()
    prepared = sim.prepare_integrator(record_diagnostics=())
    initial = prepared.pack(sim.state)
    rng = np.random.default_rng(31)
    shape = initial.variables.psi.shape[:2]
    rhs = jnp.asarray(rng.normal(size=shape)) * 1e-14
    boundary = jnp.asarray(rng.normal(size=shape))
    weights = jnp.asarray(rng.normal(size=shape))

    def objective(scale):
        state = prepared._make_state(initial.variables)
        with tracing_context():
            solution = state._linear_solver.solve(state, scale[0] * rhs, jnp.ones(shape), scale[1] * boundary)
        return jnp.vdot(solution, weights)

    point = jnp.array([1.0, 1.0])
    direction = jnp.array([0.3, 0.1])
    gradient = jax.grad(objective)(point)
    _, tangent = jax.jit(lambda x: jax.jvp(objective, (x,), (direction,)))(point)
    np.testing.assert_allclose(jnp.vdot(gradient, direction), tangent, rtol=1e-6, atol=1e-6)
    assert np.isfinite(gradient).all() and np.all(np.abs(gradient) > 1e-8)
    for step in (1e-3, 1e-4):
        difference = (objective(point + step * direction) - objective(point - step * direction)) / (2 * step)
        np.testing.assert_allclose(tangent, difference, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("streamfunction", (True, False))
def test_solver_small_rhs_and_zero_tangent(streamfunction):
    import jax
    import jax.numpy as jnp
    from veros.routines import tracing_context

    sim = make_simulation(enable_streamfunction=streamfunction)
    sim.setup()
    prepared = sim.prepare_integrator(record_diagnostics=())
    initial = prepared.pack(sim.state)
    shape = initial.variables.psi.shape[:2]
    rng = np.random.default_rng(13)
    rhs = jnp.asarray(rng.normal(size=shape)) * 1e-30
    weights = jnp.asarray(rng.normal(size=shape))

    def solve(value):
        state = prepared._make_state(initial.variables)
        with tracing_context():
            return state._linear_solver.solve(state, value, jnp.ones(shape), jnp.zeros(shape))

    result, zero_tangent = jax.jit(lambda value: jax.jvp(solve, (value,), (jnp.zeros_like(value),)))(rhs)
    np.testing.assert_array_equal(zero_tangent, 0)
    assert np.isfinite(result).all()
    _, tangent = jax.jit(lambda value: jax.jvp(solve, (value,), (value,)))(rhs)
    gradient = jax.jit(jax.grad(lambda value: jnp.vdot(solve(value), weights)))(rhs)
    np.testing.assert_allclose(jnp.vdot(tangent, weights), jnp.vdot(rhs, gradient), rtol=1e-8, atol=1e-30)
    np.testing.assert_allclose(tangent, result, rtol=1e-8, atol=1e-30)
