import math

import numpy as np
import pytest

from veros import runtime_settings
from veros.checkpointing import (
    BinomialSchedule,
    CheckpointAction,
    CheckpointedIntegrator,
    CheckpointStore,
    PeriodicDiskSchedule,
    validate_schedule,
)


def require_jax():
    if runtime_settings.backend != "jax":
        pytest.skip("adjoint checkpoint execution requires JAX")
    from veros import runtime_state

    runtime_state.backend_module


@pytest.mark.parametrize("length", (0, 1, 2, 7, 16, 37, 2000))
def test_schedule_budgets_and_recomputation(length):
    for slots in (1, 2, 4):
        counts = validate_schedule(BinomialSchedule(length, ram_checkpoints=slots))
        assert counts["forward_steps"] == counts["reverse_steps"] == length
    for ram, disk in ((0, 1), (0, 3), (1, 2), (2, 2)):
        counts = validate_schedule(BinomialSchedule(length, ram_checkpoints=ram, disk_checkpoints=disk))
        assert counts["forward_steps"] == counts["reverse_steps"] == length
    for slots in (0, 1, 3):
        counts = validate_schedule(PeriodicDiskSchedule(length, disk_interval=7, ram_checkpoints=slots))
        assert counts["forward_steps"] == counts["reverse_steps"] == length
    linear = validate_schedule(BinomialSchedule(37, ram_checkpoints=1))
    binomial = validate_schedule(BinomialSchedule(37, ram_checkpoints=4))
    assert binomial["recomputed_steps"] < linear["recomputed_steps"] / 2


def test_invalid_schedules():
    class CustomSchedule:
        n_steps = 2
        ram_checkpoints = 1

        def __init__(self, actions):
            self._actions = actions

        def actions(self):
            return iter(self._actions)

    cases = (
        ([CheckpointAction("restore", 0)], "missing checkpoint"),
        ([CheckpointAction("advance", 0, 3)], "within the schedule"),
        ([CheckpointAction("turn", 0)], "end of the forward"),
        ([CheckpointAction("save", 0, storage="gpu")], "ram or disk"),
        (
            [
                CheckpointAction("save", 0, storage="ram"),
                CheckpointAction("advance", 0, 1),
                CheckpointAction("save", 1, storage="ram"),
            ],
            "RAM checkpoint budget",
        ),
        ([CheckpointAction("advance", 0, 2), CheckpointAction("turn", 2)], "finish the reverse"),
        (
            [
                CheckpointAction("save", 0, storage="ram"),
                CheckpointAction("advance", 0, 2),
                CheckpointAction("turn", 2),
                CheckpointAction("restore", 0),
                CheckpointAction("reverse", 0, 1),
            ],
            "backward order",
        ),
    )
    for actions, match in cases:
        with pytest.raises(ValueError, match=match):
            validate_schedule(CustomSchedule(actions))
    with pytest.raises(ValueError):
        BinomialSchedule(2, ram_checkpoints=0)
    with pytest.raises(ValueError):
        PeriodicDiskSchedule(2, disk_interval=0)


def test_storage_roundtrip_and_cleanup(tmp_path):
    require_jax()
    import jax
    import jax.numpy as jnp

    template = dict(
        values=jnp.array([1.25, -2.5]),
        diagnostic=(jnp.array(3, dtype="int32"), jnp.array(True)),
        scalar=jnp.array(2.0, dtype="float32"),
    )
    marker = tmp_path / "unrelated.txt"
    marker.write_text("keep")
    store = CheckpointStore(template, ram_checkpoints=1, disk_checkpoints=1, directory=tmp_path)
    with store:
        store.save(0, template, storage="ram")
        store.save(1, template, storage="disk")
        assert len(list(tmp_path.glob("veros-ad-*/*.npz"))) == 1
        for index in (0, 1):
            restored = store.restore(index)
            for actual, expected in zip(jax.tree.leaves(restored), jax.tree.leaves(template)):
                np.testing.assert_array_equal(actual, expected)
                assert actual.dtype == expected.dtype
        with pytest.raises(ValueError, match="budget"):
            store.save(2, template, storage="ram")
        with pytest.raises(ValueError, match="budget"):
            store.save(2, template, storage="disk")
        with pytest.raises(ValueError, match="already exists"):
            store.save(1, template, storage="disk")
        with pytest.raises(ValueError, match="shapes, or dtypes"):
            store.save(2, dict(template, values=jnp.ones(3)), storage="disk")
        store.drop(0)
        store.drop(1)
        assert not list(tmp_path.glob("veros-ad-*/*.npz"))
    assert sorted(path.name for path in tmp_path.iterdir()) == [marker.name]
    assert store.statistics["peak_ram_checkpoints"] == store.statistics["peak_disk_checkpoints"] == 1
    assert store.statistics["disk_bytes_written"] == store.statistics["disk_bytes_read"] > 0
    with pytest.raises(RuntimeError, match="context manager"):
        store.restore(0)
    with pytest.raises(RuntimeError, match="intentional"):
        with CheckpointStore(template, ram_checkpoints=0, directory=tmp_path) as store:
            store.save(0, template, storage="disk")
            raise RuntimeError("intentional")
    assert sorted(path.name for path in tmp_path.iterdir()) == [marker.name]


class ToyIntegrator:
    """A smooth recurrence with diagnostic window resets and discrete leaves."""

    def step(self, carry):
        import jax.numpy as jnp

        self.traces = getattr(self, "traces", 0) + 1
        x = jnp.sin(carry.variables["x"] * carry.variables["forcing"]) + 0.1
        iteration = carry.variables["iteration"] + 1
        accumulation = carry.diagnostics["sum"] + x
        due = iteration % 3 == 0
        updated = carry._replace(
            variables=dict(carry.variables, x=x, iteration=iteration),
            diagnostics={"sum": jnp.where(due, 0, accumulation)},
            failure_code=jnp.where(jnp.all(jnp.isfinite(x)), 0, 1).astype("int32"),
            failed_iteration=iteration,
        )
        return updated, dict(valid=due, values=jnp.where(due, accumulation, 0))

    def advance(self, initial, *, n_steps):
        import jax
        import jax.numpy as jnp

        records = []
        for _ in range(n_steps):
            initial, event = self.step(initial)
            records.append(event)
        return initial, jax.tree.map(lambda *values: jnp.stack(values), *records)

    def check_status(self, carry):
        from veros.integration import PreparedIntegrator

        PreparedIntegrator.check_status(carry)


def toy_initial():
    import jax.numpy as jnp
    from veros.integration import IntegrationState

    return IntegrationState(
        dict(
            x=jnp.array([0.2, 0.3]), forcing=jnp.array(0.9, dtype=runtime_settings.float_type), iteration=jnp.int32(0)
        ),
        {"sum": jnp.zeros(2)},
        jnp.int32(0),
        jnp.int32(-1),
    )


def test_memory_reports_reuse_profiles_and_preserve_shape_accounting(monkeypatch):
    require_jax()
    import jax
    import jax.numpy as jnp

    kernel = jax.jit(lambda x: x**2)
    lower = kernel.lower
    calls = []

    def counted_lower(*arguments):
        calls.append(1)
        return lower(*arguments)

    monkeypatch.setattr(kernel, "lower", counted_lower)
    driver = CheckpointedIntegrator(ToyIntegrator(), terminal_objective=lambda c: jnp.sum(c.variables["x"]))
    first = driver._memory_analysis(kernel, jnp.ones(4, dtype=jnp.float64))
    repeated = driver._memory_analysis(kernel, jnp.zeros(4, dtype=jnp.float64))
    assert repeated == first and len(calls) == 1
    larger = driver._memory_analysis(kernel, jnp.ones(8, dtype=jnp.float64))
    assert len(calls) == 2
    assert first["argument_bytes"] == 32 and larger["argument_bytes"] == 64


@pytest.mark.parametrize("slots", (0, 1, 3))
def test_mixed_gradient_diagnostics_and_step_reuse(tmp_path, slots):
    require_jax()
    import jax
    import jax.numpy as jnp

    prepared = ToyIntegrator()
    initial = toy_initial()

    def terminal(carry):
        return jnp.sum(carry.variables["x"] ** 2)

    def records_loss(records):
        return jnp.sum(records["values"] ** 2)

    driver = CheckpointedIntegrator(prepared, terminal_objective=terminal, record_objective=records_loss)
    steps = 11
    schedule = PeriodicDiskSchedule(steps, disk_interval=4, ram_checkpoints=slots)
    result = driver.value_and_grad(initial, n_steps=steps, schedule=schedule, directory=tmp_path)

    def reference(carry):
        final, records = prepared.advance(carry, n_steps=steps)
        return terminal(final) + records_loss(records)

    value, gradient = jax.jit(jax.value_and_grad(reference, allow_int=True))(initial)
    np.testing.assert_allclose(result.value, value, rtol=1e-12)
    for actual, expected in zip(jax.tree.leaves(result.gradient), jax.tree.leaves(gradient)):
        if actual.dtype == jax.dtypes.float0:
            assert expected.dtype == jax.dtypes.float0
        else:
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    final, _ = prepared.advance(initial, n_steps=steps)
    for actual, expected in zip(jax.tree.leaves(result.final), jax.tree.leaves(final)):
        np.testing.assert_array_equal(actual, expected)
    assert result.statistics["peak_ram_checkpoints"] <= slots
    assert result.statistics["peak_disk_checkpoints"] == math.ceil(schedule.n_steps / schedule.disk_interval)
    assert result.statistics["disk_bytes_written"] > 0 and result.statistics["disk_bytes_read"] > 0
    assert not list(tmp_path.iterdir())
    # Reusing the driver must reuse its step executables and own new files.
    signatures = (driver._forward._cache_size(), driver._reverse._cache_size())
    second = driver.value_and_grad(initial, n_steps=steps, schedule=schedule, directory=tmp_path)
    np.testing.assert_array_equal(second.gradient.variables["forcing"], result.gradient.variables["forcing"])
    assert (driver._forward._cache_size(), driver._reverse._cache_size()) == signatures
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("steps", (0, 7))
def test_terminal_cotangent_composes_with_objective(tmp_path, steps):
    require_jax()
    import jax
    import jax.numpy as jnp

    initial, prepared = toy_initial(), ToyIntegrator()

    def terminal(carry):
        return jnp.sum(carry.variables["x"] ** 2)

    def downstream(carry):
        return jnp.sum(0.3 * carry.variables["x"]) + 0.2 * carry.variables["forcing"]

    final = initial
    for _ in range(steps):
        final, _ = prepared.step(final)
    seed = jax.grad(downstream, allow_int=True)(final)
    driver = CheckpointedIntegrator(prepared, terminal_objective=terminal)
    result = driver.value_and_grad(initial, n_steps=steps, terminal_cotangent=seed)

    def reference(carry):
        final = carry
        for _ in range(steps):
            final, _ = prepared.step(final)
        return terminal(final) + downstream(final)

    expected = jax.grad(reference, allow_int=True)(initial)
    np.testing.assert_allclose(result.value, terminal(final), rtol=1e-12)
    for actual, wanted in zip(jax.tree.leaves(result.gradient), jax.tree.leaves(expected)):
        if wanted.dtype == jax.dtypes.float0:
            assert actual.dtype == jax.dtypes.float0
        else:
            np.testing.assert_allclose(actual, wanted, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="structure"):
        driver.value_and_grad(initial, n_steps=steps, terminal_cotangent={})
    malformed = seed._replace(variables=dict(seed.variables, x=jnp.zeros(3)))
    with pytest.raises(ValueError, match="shapes or dtypes"):
        driver.value_and_grad(initial, n_steps=steps, terminal_cotangent=malformed)


def test_zero_steps_ram_and_errors(tmp_path):
    require_jax()
    import jax
    import jax.numpy as jnp

    initial = toy_initial()
    prepared = ToyIntegrator()
    driver = CheckpointedIntegrator(prepared, terminal_objective=lambda carry: jnp.sum(carry.variables["x"] ** 2))
    result = driver.value_and_grad(initial, n_steps=0)
    np.testing.assert_allclose(result.gradient.variables["x"], 2 * initial.variables["x"])
    assert result.statistics["ram_saves"] == result.statistics["disk_saves"] == 0
    ram = driver.value_and_grad(initial, n_steps=9, schedule=BinomialSchedule(9, ram_checkpoints=1))
    assert ram.statistics["disk_saves"] == 0 and ram.statistics["peak_ram_checkpoints"] == 1
    with pytest.raises(ValueError, match="Schedule length"):
        driver.value_and_grad(initial, n_steps=9, schedule=BinomialSchedule(2))
    with pytest.raises(RuntimeError, match="host operation"):
        jax.jit(lambda state: driver.value_and_grad(state, n_steps=2))(initial)
    invalid = initial._replace(
        variables=dict(initial.variables, x=jnp.full(2, jnp.nan, dtype=initial.variables["x"].dtype))
    )
    with pytest.raises(RuntimeError, match="solution diverged"):
        driver.value_and_grad(invalid, n_steps=2, schedule=PeriodicDiskSchedule(2, disk_interval=1), directory=tmp_path)
    assert not list(tmp_path.iterdir())
    nonscalar = CheckpointedIntegrator(prepared, record_objective=lambda records: records["values"])
    with pytest.raises(ValueError, match="scalar"):
        nonscalar.value_and_grad(initial, n_steps=1)


@pytest.mark.parametrize("storage", ("ram", "disk"))
def test_stitched_parameter_and_initial_gradients(tmp_path, storage):
    require_jax()
    import jax
    import jax.numpy as jnp

    class ParameterIntegrator(ToyIntegrator):
        def validate_parameters(self, parameters):
            return {name: jnp.asarray(value) for name, value in parameters.items()}

        def step(self, carry, *, parameters):
            forcing = carry.variables["forcing"]
            controlled = carry._replace(variables=dict(carry.variables, forcing=forcing * parameters["scale"]))
            final, records = super().step(controlled)
            return final._replace(variables=dict(final.variables, forcing=forcing)), records

    prepared = ParameterIntegrator()
    initial = toy_initial()
    parameters = {"scale": jnp.array(0.8)}
    steps = 11

    def terminal(carry, *, parameters):
        return jnp.sum(carry.variables["x"] ** 2) + parameters["scale"] ** 2

    def record_loss(records, *, parameters):
        return parameters["scale"] * jnp.sum(records["values"] ** 2)

    def reference(carry, parameters):
        value = jnp.array(0.0)
        for _ in range(steps):
            carry, records = prepared.step(carry, parameters=parameters)
            value = value + record_loss(records, parameters=parameters)
        return value + terminal(carry, parameters=parameters)

    value, (gradient, parameter_gradient) = jax.value_and_grad(reference, argnums=(0, 1), allow_int=True)(
        initial, parameters
    )
    driver = CheckpointedIntegrator(prepared, terminal_objective=terminal, record_objective=record_loss)
    schedule = (
        BinomialSchedule(steps, ram_checkpoints=2)
        if storage == "ram"
        else PeriodicDiskSchedule(steps, disk_interval=4, ram_checkpoints=1)
    )
    result = driver.value_and_grad(initial, n_steps=steps, parameters=parameters, schedule=schedule, directory=tmp_path)
    np.testing.assert_allclose(result.value, value, rtol=1e-12)
    np.testing.assert_allclose(result.parameter_gradient["scale"], parameter_gradient["scale"], rtol=1e-12)
    for actual, expected in zip(jax.tree.leaves(result.gradient), jax.tree.leaves(gradient)):
        if actual.dtype == jax.dtypes.float0:
            assert expected.dtype == jax.dtypes.float0
        else:
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("materialization", ("periodic", "bounded_mixed", "disk_only"))
def test_long_mixed_replay(tmp_path, materialization):
    require_jax()
    import jax
    import jax.numpy as jnp

    initial = toy_initial()
    prepared = ToyIntegrator()
    driver = CheckpointedIntegrator(prepared, terminal_objective=lambda carry: jnp.sum(carry.variables["x"]))
    if materialization == "periodic":
        schedule = PeriodicDiskSchedule(129, disk_interval=11, ram_checkpoints=2)
        ram_slots, disk_slots = 2, 12
    else:
        ram_slots, disk_slots = (2, 2) if materialization == "bounded_mixed" else (0, 4)
        schedule = BinomialSchedule(129, ram_checkpoints=ram_slots, disk_checkpoints=disk_slots)
    result = driver.value_and_grad(initial, n_steps=129, schedule=schedule, directory=tmp_path)

    def reference(forcing):
        carry = initial._replace(variables=dict(initial.variables, forcing=forcing))
        return jnp.sum(prepared.advance(carry, n_steps=129)[0].variables["x"])

    expected = jax.jit(jax.grad(reference))(initial.variables["forcing"])
    np.testing.assert_allclose(result.gradient.variables["forcing"], expected, rtol=1e-12)
    finite_difference = (
        reference(initial.variables["forcing"] + 1e-5) - reference(initial.variables["forcing"] - 1e-5)
    ) / 2e-5
    np.testing.assert_allclose(result.gradient.variables["forcing"], finite_difference, rtol=1e-8)
    assert result.statistics["peak_ram_checkpoints"] == ram_slots
    assert result.statistics["peak_disk_checkpoints"] == disk_slots
    assert result.statistics["forward_steps"] == result.statistics["reverse_steps"] == 129
    assert not list(tmp_path.iterdir())


def test_custom_materialization_schedule_and_terminal_failure(tmp_path):
    require_jax()
    import jax
    import jax.numpy as jnp

    class CustomSchedule:
        n_steps = 3
        ram_checkpoints = 1

        def actions(self):
            return iter(
                (
                    CheckpointAction("save", 0, storage="ram"),
                    CheckpointAction("drop", 0),
                    CheckpointAction("save", 0, storage="disk"),
                    CheckpointAction("advance", 0, 2),
                    CheckpointAction("save", 2, storage="ram"),
                    CheckpointAction("advance", 2, 3),
                    CheckpointAction("turn", 3),
                    CheckpointAction("restore", 2),
                    CheckpointAction("reverse", 2, 3),
                    CheckpointAction("drop", 2),
                    CheckpointAction("restore", 0),
                    CheckpointAction("advance", 0, 1),
                    CheckpointAction("save", 1, storage="ram"),
                    CheckpointAction("reverse", 1, 2),
                    CheckpointAction("drop", 1),
                    CheckpointAction("restore", 0),
                    CheckpointAction("reverse", 0, 1),
                    CheckpointAction("drop", 0),
                )
            )

    prepared = ToyIntegrator()
    initial = toy_initial()

    def terminal(carry):
        return jnp.sum(carry.variables["x"])

    driver = CheckpointedIntegrator(prepared, terminal_objective=terminal)
    result = driver.value_and_grad(initial, n_steps=3, schedule=CustomSchedule(), directory=tmp_path)

    def reference(carry):
        return terminal(prepared.advance(carry, n_steps=3)[0])

    gradient = jax.grad(reference, allow_int=True)(initial)
    np.testing.assert_allclose(result.gradient.variables["forcing"], gradient.variables["forcing"], rtol=1e-12)
    assert result.statistics["ram_saves"] == 3 and result.statistics["disk_saves"] == 1
    assert not list(tmp_path.iterdir())

    def failure(carry):
        raise ValueError("terminal failed")

    driver = CheckpointedIntegrator(prepared, terminal_objective=failure)
    with pytest.raises(ValueError, match="terminal failed"):
        driver.value_and_grad(initial, n_steps=3, schedule=CustomSchedule(), directory=tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("physics", ("fixed_mixing", "full"))
def test_veros_checkpointed_gradient(tmp_path, physics):
    require_jax()
    import jax
    import jax.numpy as jnp
    from veros import veros_routine
    from veros.setups.acc_basic import ACCBasicSetup

    class Simulation(ACCBasicSetup):
        @veros_routine
        def set_diagnostics(self, state):
            super().set_diagnostics(state)
            for diagnostic in state.diagnostics.values():
                diagnostic.output_frequency = diagnostic.sampling_frequency = None
            averages = state.diagnostics["averages"]
            averages.sampling_frequency = state.settings.dt_tracer
            averages.output_frequency = 3 * state.settings.dt_tracer

    original_diskless, original_solver = runtime_settings.diskless_mode, runtime_settings.linear_solver
    object.__setattr__(runtime_settings, "diskless_mode", True)
    object.__setattr__(runtime_settings, "linear_solver", "scipy_jax")
    try:
        nx, ny = (6, 6) if runtime_settings.device == "gpu" else (8, 8)
        nz = 15
        override = dict(nx=nx, ny=ny, nz=nz, dt_tracer=600, dt_mom=600)
        if physics == "fixed_mixing":
            override.update(enable_tke=False, enable_neutral_diffusion=False, enable_skew_diffusion=False)
        sim = Simulation(override=override)
        sim.setup()
        prepared = sim.prepare_integrator(record_diagnostics=("averages",), checkpoint=True)
        initial = prepared.pack(sim.state)

        def terminal(carry):
            return jnp.sum(carry.variables.temp[2:-2, 2:-2, :, carry.variables.tau])

        def records_loss(records):
            record = records["averages"]
            means = record.values["temp"][:, 2:-2, 2:-2].sum(axis=(1, 2, 3))
            return jnp.sum(jnp.where(record.valid, means, 0))

        def reference(scale):
            carry = prepared.replace_variables(initial, t_star=scale * initial.variables.t_star)
            final, records = prepared.advance(carry, n_steps=7)
            return terminal(final) + records_loss(records)

        value, gradient = jax.value_and_grad(reference)(1.0)
        final, _ = prepared.advance(initial, n_steps=7)
        jax.block_until_ready((value, gradient, final))
        jax.clear_caches()
        driver = CheckpointedIntegrator(prepared, terminal_objective=terminal, record_objective=records_loss)
        schedule = PeriodicDiskSchedule(7, disk_interval=4, ram_checkpoints=1)
        bounded = BinomialSchedule(7, ram_checkpoints=1, disk_checkpoints=2)
        for strategy in (schedule, bounded):
            result = driver.value_and_grad(initial, n_steps=7, schedule=strategy, directory=tmp_path)
            sensitivity = jnp.sum(result.gradient.variables.t_star * initial.variables.t_star)
            np.testing.assert_allclose(result.value, value, rtol=1e-10)
            np.testing.assert_allclose(sensitivity, gradient, rtol=2e-7, atol=1e-6)
            for actual, expected in zip(jax.tree.leaves(result.final), jax.tree.leaves(final)):
                np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-7)
            assert int(result.final.diagnostics["averages"].average_nitts) == 1
            assert result.statistics["peak_ram_checkpoints"] == 1
            assert result.statistics["peak_disk_checkpoints"] == 2
        assert int(sim.state.variables.itt) == 0
        assert not list(tmp_path.iterdir())
        jax.clear_caches()
    finally:
        object.__setattr__(runtime_settings, "diskless_mode", original_diskless)
        object.__setattr__(runtime_settings, "linear_solver", original_solver)


@pytest.mark.parametrize("storage", ("ram", "disk"))
def test_recorded_inputs_observers_and_byte_budgets(tmp_path, storage):
    require_jax()
    import jax.numpy as jnp

    class InputIntegrator(ToyIntegrator):
        def validate_step_inputs(self, inputs):
            return {name: jnp.asarray(value, dtype="float64") for name, value in inputs.items()}

        def step(self, carry, *, step_inputs):
            forcing = carry.variables["forcing"]
            controlled = carry._replace(variables=dict(carry.variables, forcing=forcing + step_inputs["forcing"]))
            final, records = super().step(controlled)
            return final._replace(variables=dict(final.variables, forcing=forcing)), records

    source = tmp_path / "forcing.txt"
    source.write_text("1.0")
    reads, observed = [], []
    steps = 5
    prepared = InputIntegrator()
    initial = toy_initial()

    def inputs():
        for index in range(steps):
            reads.append(index)
            yield {"forcing": jnp.full(2, float(source.read_text()), dtype="float64")}

    def observer(index, carry):
        observed.append(index)
        if index == steps - 1:
            source.write_text("99.0")

    def terminal(carry):
        return jnp.sum(carry.variables["x"] ** 2)

    driver = CheckpointedIntegrator(prepared, terminal_objective=terminal)
    result = driver.value_and_grad(
        initial,
        n_steps=steps,
        schedule=BinomialSchedule(steps, ram_checkpoints=1),
        directory=tmp_path,
        step_inputs=inputs(),
        input_storage=storage,
        observer=observer,
        max_input_ram_bytes=steps * 16,
    )
    assert reads == observed == list(range(steps))
    assert result.statistics["input_peak_ram_bytes"] == (steps * 16 if storage == "ram" else 0)
    if storage == "disk":
        assert result.statistics["input_peak_disk_bytes"] > 0
    import jax

    def reference(carry):
        for _ in range(steps):
            carry, _ = prepared.step(carry, step_inputs={"forcing": jnp.ones(2, dtype="float64")})
        return terminal(carry)

    value, gradient = jax.value_and_grad(reference, allow_int=True)(initial)
    np.testing.assert_allclose(result.value, value, rtol=1e-12)
    np.testing.assert_allclose(result.gradient.variables["forcing"], gradient.variables["forcing"], rtol=1e-12)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["forcing.txt"]
    with pytest.raises(ValueError, match="byte budget"):
        driver.value_and_grad(initial, n_steps=steps, step_inputs=inputs(), max_input_ram_bytes=16)
    with pytest.raises(ValueError, match="byte budget"):
        driver.value_and_grad(initial, n_steps=steps, step_inputs=inputs(), max_checkpoint_ram_bytes=1)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["forcing.txt"]


def test_unsupported_derivative_fails_before_forward(tmp_path):
    require_jax()
    import jax
    import jax.numpy as jnp

    class OpaqueIntegrator(ToyIntegrator):
        def step(self, carry):
            x = carry.variables["x"]
            opaque = jax.pure_callback(lambda value: np.asarray(value), jax.ShapeDtypeStruct(x.shape, x.dtype), x)
            return carry._replace(variables=dict(carry.variables, x=opaque)), {}

    observed = []
    driver = CheckpointedIntegrator(OpaqueIntegrator(), terminal_objective=lambda carry: jnp.sum(carry.variables["x"]))
    with pytest.raises(ValueError, match="Pure callbacks do not support JVP"):
        driver.value_and_grad(
            toy_initial(),
            n_steps=10,
            directory=tmp_path,
            schedule=PeriodicDiskSchedule(10, disk_interval=2),
            observer=lambda index, carry: observed.append(index),
        )
    assert not observed and not list(tmp_path.iterdir())
