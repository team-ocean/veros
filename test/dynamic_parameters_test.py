"""Runtime setting derivatives through model trajectories and checkpoint replay."""

from copy import copy

import numpy as np
import pytest

from veros import runtime_settings, veros_routine, veros_kernel


@pytest.fixture(autouse=True)
def integration_options():
    if runtime_settings.backend != "jax":
        pytest.skip("dynamic integration parameters require JAX")
    import jax

    previous = runtime_settings.diskless_mode, runtime_settings.linear_solver
    object.__setattr__(runtime_settings, "diskless_mode", True)
    object.__setattr__(runtime_settings, "linear_solver", "scipy_jax")
    yield
    jax.clear_caches()
    object.__setattr__(runtime_settings, "diskless_mode", previous[0])
    object.__setattr__(runtime_settings, "linear_solver", previous[1])


def simulation(**overrides):
    from veros.setups.acc_basic import ACCBasicSetup

    class Simulation(ACCBasicSetup):
        @veros_routine
        def set_diagnostics(self, state):
            for diagnostic in state.diagnostics.values():
                diagnostic.output_frequency = diagnostic.sampling_frequency = None
            averages = state.diagnostics["averages"]
            averages.output_variables = ("temp",)
            averages.sampling_frequency = state.settings.dt_tracer
            averages.output_frequency = 3 * state.settings.dt_tracer

    sim = Simulation(
        override=dict(
            nx=8,
            ny=8,
            dt_mom=600,
            dt_tracer=600,
            enable_neutral_diffusion=False,
            enable_skew_diffusion=False,
            **overrides,
        )
    )
    sim.setup()
    return sim


def terminal(final):
    import jax.numpy as jnp

    vs = final.variables
    return (
        jnp.sum(vs.temp[2:-2, 2:-2, :, vs.tau] ** 2)
        + 1e5 * jnp.sum(vs.tke[2:-2, 2:-2, :, vs.tau])
        + 1e4 * jnp.sum(vs.u[2:-2, 2:-2, :, vs.tau] ** 2)
    )


def record_loss(records):
    import jax.numpy as jnp

    record = records["averages"]
    values = jnp.sum(record.values["temp"][:, 2:-2, 2:-2] ** 2, axis=(1, 2, 3))
    return jnp.sum(jnp.where(record.valid, values, 0))


def parameters(sim):
    import jax.numpy as jnp

    return {name: jnp.asarray(getattr(sim.state.settings, name)) for name in ("c_k", "c_eps", "A_h", "r_bot")}


def assert_tree_close(actual, expected, **tolerances):
    import jax

    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        if a.dtype == jax.dtypes.float0:
            assert b.dtype == jax.dtypes.float0
        else:
            np.testing.assert_allclose(a, b, **tolerances)


def test_settings_pytree_and_nested_kernel():
    import jax
    import jax.numpy as jnp

    sim = simulation()
    initial_settings = sim.state.settings
    point = {"c_k": jnp.array(0.2), "c_eps": jnp.array(0.6)}

    @veros_kernel
    def kernel(state):
        return state.settings.c_k * state.settings.c_eps + state.settings.nx

    def evaluate(values):
        state = copy(sim.state)
        state._settings = initial_settings._with_parameters(values)
        return kernel(state)

    compiled = jax.jit(evaluate)
    np.testing.assert_allclose(compiled(point), 8.12)
    cache_size = compiled._cache_size()
    np.testing.assert_allclose(compiled(dict(point, c_k=jnp.array(0.3))), 8.18)
    assert compiled._cache_size() == cache_size
    expected = {"c_k": 0.6, "c_eps": 0.2}
    for derivative in (jax.grad, jax.jacfwd, jax.jacrev):
        actual = derivative(evaluate)(point)
        for name in expected:
            np.testing.assert_allclose(actual[name], expected[name])
    batch = {"c_k": jnp.array([0.2, 0.3]), "c_eps": jnp.array([0.6, 0.6])}
    np.testing.assert_allclose(jax.jit(jax.vmap(evaluate))(batch), [8.12, 8.18])
    view = initial_settings._with_parameters(point)
    shaped = jax.eval_shape(lambda settings: settings, view)
    assert shaped.c_k.shape == ()
    jacobian = jax.jacfwd(lambda settings: settings.c_k * settings.c_eps)(view)
    np.testing.assert_allclose(jacobian.c_k, 0.6)
    assert initial_settings.c_k == 0.1
    assert not hasattr(initial_settings, "_parameter_names")
    np.testing.assert_allclose(kernel(sim.state), 8.07)
    with initial_settings.unlock():
        initial_settings.c_k = 0.4
    np.testing.assert_allclose(kernel(sim.state), 8.28)
    np.testing.assert_allclose(compiled(point), 8.12)


@pytest.mark.parametrize("checkpoint", (False, True))
def test_trajectory_derivatives_and_compilation_reuse(checkpoint):
    import jax

    sim = simulation()
    prepared = sim.prepare_integrator(record_diagnostics=("averages",), checkpoint=checkpoint)
    initial = prepared.pack(sim.state)
    point = parameters(sim)

    def objective(values):
        final, records = prepared.advance(initial, n_steps=6, parameters=values)
        return terminal(final) + record_loss(records)

    backward = jax.value_and_grad(objective)
    value, gradient = backward(point)
    prepared.check_status(prepared.advance(initial, n_steps=6, parameters=point)[0])
    direction = {name: coefficient * 0.1 for name, coefficient in point.items()}
    expected = sum(gradient[name] * direction[name] for name in point)
    assert np.isfinite(value) and np.isfinite(expected)
    assert sum(abs(float(gradient[name] * point[name])) > 1e-8 for name in point) >= 2
    cache_size = prepared._compiled_step._cache_size()
    changed = {name: coefficient * 1.01 for name, coefficient in point.items()}
    backward(changed)
    assert prepared._compiled_step._cache_size() == cache_size
    forward_value, tangent = jax.jvp(objective, (point,), (direction,))
    np.testing.assert_allclose(forward_value, value, rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(tangent, expected, rtol=1e-7, atol=1e-7)
    evaluate = objective
    for epsilon in (1e-3, 1e-4):
        plus = jax.tree.map(lambda a, b: a + epsilon * b, point, direction)
        minus = jax.tree.map(lambda a, b: a - epsilon * b, point, direction)
        difference = (evaluate(plus) - evaluate(minus)) / (2 * epsilon)
        np.testing.assert_allclose(difference, expected, rtol=2e-4, atol=2e-5)
    assert not any(name.startswith("control_") for name in initial.variables.fields())
    assert sim.state.settings.c_k == 0.1
    assert not any(isinstance(leaf, jax.core.Tracer) for leaf in vars(prepared._template.settings).values())


def test_plugin_setting_and_all_runtime_float_settings():
    import jax
    import jax.numpy as jnp
    from types import SimpleNamespace
    from veros.settings import Setting
    from veros.setups.acc_basic import ACCBasicSetup
    from veros.variables import Variable

    @veros_routine
    def initialize(state):
        pass

    @veros_kernel
    def accumulate(state):
        return state.variables.plugin_total + state.settings.plugin_rate * state.settings.dt_tracer

    @veros_routine(supports_compilation=True)
    def run(state):
        state.variables.plugin_total = accumulate(state)

    plugin = SimpleNamespace(
        __name__="parameter_probe",
        __VEROS_INTERFACE__={
            "setup_entrypoint": initialize,
            "run_entrypoint": run,
            "settings": {"plugin_rate": Setting(0.25, float, "Accumulation rate")},
            "variables": {"plugin_total": Variable("Accumulated value", None, initial=0.0)},
        },
    )

    class Simulation(ACCBasicSetup):
        __veros_plugins__ = (plugin,)

        @veros_routine
        def set_diagnostics(self, state):
            for diagnostic in state.diagnostics.values():
                diagnostic.output_frequency = diagnostic.sampling_frequency = None

    sim = Simulation(override=dict(nx=8, ny=8, dt_mom=600, dt_tracer=600))
    sim.setup()
    prepared = sim.prepare_integrator(record_diagnostics=())
    initial = prepared.pack(sim.state)
    values = {
        name: jnp.asarray(getattr(sim.state.settings, name))
        for name, meta in sim.state.settings.__metadata__.items()
        if meta.type is float and name not in prepared._fixed_parameters
    }
    final, _ = prepared.advance(initial, n_steps=3, parameters=values)
    prepared.check_status(final)
    for _ in range(3):
        sim.step(sim.state)
    assert_tree_close(final.variables, sim.state.variables, rtol=2e-7, atol=2e-7)
    derivative = jax.grad(
        lambda rate: prepared.advance(initial, n_steps=3, parameters={"plugin_rate": rate})[0].variables.plugin_total
    )(jnp.array(0.25))
    np.testing.assert_allclose(derivative, 1800.0)
    assert sim.state.settings.plugin_rate == 0.25


def test_override_parity_continuation_and_default_isolation():
    import jax

    sim = simulation()
    prepared = sim.prepare_integrator(record_diagnostics=("averages",), checkpoint=True)
    initial = prepared.pack(sim.state)
    point = {name: coefficient * 1.2 for name, coefficient in parameters(sim).items()}
    final, records = prepared.advance(initial, n_steps=6, parameters=point)
    first, _ = prepared.advance(initial, n_steps=2, parameters=point)
    split, _ = prepared.advance(first, n_steps=4, parameters=point)
    assert_tree_close(final, split, rtol=1e-8, atol=1e-7)
    with sim.state.settings.unlock():
        sim.state.settings.update({name: float(value) for name, value in point.items()})
    for _ in range(6):
        sim.step(sim.state)
    assert_tree_close(final.variables, sim.state.variables, rtol=2e-7, atol=2e-7)
    default, _ = prepared.advance(initial, n_steps=6)
    explicit_default, _ = prepared.advance(
        initial, n_steps=6, parameters={name: value / 1.2 for name, value in point.items()}
    )
    assert_tree_close(default, explicit_default, rtol=1e-8, atol=1e-7)
    assert not np.allclose(final.variables.tke, default.variables.tke, rtol=1e-6, atol=1e-12)
    unchanged, empty = prepared.advance(initial, n_steps=0, parameters=point)
    assert_tree_close(unchanged, initial)
    assert all(leaf.shape[0] == 0 for leaf in jax.tree.leaves(empty))
    assert records["averages"].valid.sum() == 2


def test_eke_and_idemix_parameter_derivatives():
    import jax
    import jax.numpy as jnp

    sim = simulation(enable_eke=True, enable_idemix=True, eq_of_state_type=5)
    prepared = sim.prepare_integrator(record_diagnostics=(), checkpoint=True)
    initial = prepared.pack(sim.state)
    initial = prepared.replace_variables(
        initial,
        E_iw=initial.variables.E_iw + 1e-5 * initial.variables.maskW[..., None],
        eke=initial.variables.eke + 1e-3 * initial.variables.maskW[..., None],
    )
    point = {name: jnp.asarray(getattr(sim.state.settings, name)) for name in ("mu0", "gamma", "eke_c_k", "eke_c_eps")}

    def loss(values):
        final, _ = prepared.advance(initial, n_steps=3, parameters=values, final_only=True)
        vs = final.variables
        return 1e10 * jnp.sum(vs.E_iw[2:-2, 2:-2, :, vs.tau] ** 2) + 1e6 * jnp.sum(vs.eke[2:-2, 2:-2, :, vs.tau] ** 2)

    value, gradient = jax.value_and_grad(loss)(point)
    assert np.isfinite(value)
    assert all(np.isfinite(coefficient) for coefficient in gradient.values())
    assert abs(float(gradient["mu0"])) > 1e-8
    assert abs(float(gradient["eke_c_eps"])) > 1e-8
    direction = jax.tree.map(lambda parameter: parameter * 0.1, point)
    expected = sum(gradient[name] * direction[name] for name in point)
    _, tangent = jax.jvp(loss, (point,), (direction,))
    np.testing.assert_allclose(tangent, expected, rtol=1e-7, atol=1e-8)
    evaluate = loss
    for epsilon in (1e-3, 1e-4):
        plus = jax.tree.map(lambda parameter, delta: parameter + epsilon * delta, point, direction)
        minus = jax.tree.map(lambda parameter, delta: parameter - epsilon * delta, point, direction)
        difference = (evaluate(plus) - evaluate(minus)) / (2 * epsilon)
        np.testing.assert_allclose(difference, expected, rtol=2e-4, atol=1e-6)
    prepared.check_status(prepared.advance(initial, n_steps=3, parameters=point, final_only=True)[0])


def test_parameter_validation_and_legacy_control_conflict():
    import jax
    import jax.numpy as jnp

    sim = simulation()
    prepared = sim.prepare_integrator(record_diagnostics=(), fixed_parameters=("c_eps",))
    initial = prepared.pack(sim.state)
    invalid = (
        ({"missing": 1.0}, ValueError, "Unknown setting"),
        ({"nx": 8.0}, TypeError, "not a floating"),
        ({"enable_tke": 1.0}, TypeError, "not a floating"),
        ({"identifier": 1.0}, TypeError, "not a floating"),
        ({"c_k": jnp.ones(2)}, TypeError, "scalar"),
        ({"c_k": 1}, TypeError, "scalar"),
        ({"c_k": 1j}, TypeError, "scalar"),
        ([0.1], TypeError, "mapping"),
    )
    for value, exception, message in invalid:
        with pytest.raises(exception, match=message):
            prepared.advance(initial, n_steps=0, parameters=value)
    for name in ("omega", "radius", "rho_0", "grav", "dt_mom", "dt_tracer", "runlen", "c_eps"):
        with pytest.raises(ValueError, match="fixed preparation"):
            jax.jit(lambda value: prepared.advance(initial, n_steps=0, parameters={name: value}))(jnp.array(1.0))
    with pytest.raises(ValueError, match="Unknown fixed"):
        sim.prepare_integrator(fixed_parameters=("missing",))
    from veros.setups.acc_basic import ACCBasicSetup

    legacy = ACCBasicSetup(
        override=dict(nx=8, ny=8, dt_mom=600, dt_tracer=600, enable_tke=False), control_parameters=("kappaH_0",)
    )
    legacy.setup()
    prepared = legacy.prepare_integrator(record_diagnostics=())
    with pytest.raises(ValueError, match="variable control"):
        prepared.advance(prepared.pack(legacy.state), n_steps=0, parameters={"kappaH_0": 1e-4})


@pytest.mark.parametrize("steps", (0, 7))
def test_checkpointed_parameter_and_initial_gradients(tmp_path, steps):
    import jax
    from veros.checkpointing import CheckpointedIntegrator, BinomialSchedule, PeriodicDiskSchedule

    sim = simulation()
    prepared = sim.prepare_integrator(record_diagnostics=("averages",), checkpoint=True)
    initial = prepared.pack(sim.state)
    point = parameters(sim)

    def terminal_loss(final, *, parameters):
        return terminal(final) + parameters["c_k"] ** 2

    def records_loss(records, *, parameters):
        return parameters["c_eps"] * record_loss(records)

    def reference(carry, values):
        final, records = prepared.advance(carry, n_steps=steps, parameters=values)
        return terminal_loss(final, parameters=values) + records_loss(records, parameters=values)

    value, (expected_initial, expected_parameters) = jax.value_and_grad(reference, argnums=(0, 1), allow_int=True)(
        initial, point
    )
    driver = CheckpointedIntegrator(prepared, terminal_objective=terminal_loss, record_objective=records_loss)
    schedules = (
        BinomialSchedule(steps, ram_checkpoints=1, disk_checkpoints=2),
        PeriodicDiskSchedule(steps, disk_interval=2, ram_checkpoints=1),
    )
    for schedule in schedules:
        result = driver.value_and_grad(initial, n_steps=steps, parameters=point, schedule=schedule, directory=tmp_path)
        np.testing.assert_allclose(result.value, value, rtol=1e-10, atol=1e-7)
        assert_tree_close(result.parameter_gradient, expected_parameters, rtol=2e-7, atol=1e-5)
        assert_tree_close(result.gradient, expected_initial, rtol=2e-6, atol=1e-4)
        assert not list(tmp_path.iterdir())
    sizes = driver._forward._cache_size(), driver._reverse._cache_size()
    changed = jax.tree.map(lambda coefficient: coefficient * 1.01, point)
    driver.value_and_grad(initial, n_steps=steps, parameters=changed)
    assert sizes == (driver._forward._cache_size(), driver._reverse._cache_size())
    with pytest.raises(RuntimeError, match="host operation"):
        jax.grad(lambda value: driver.value_and_grad(initial, n_steps=steps, parameters={"c_k": value}).value)(0.1)
