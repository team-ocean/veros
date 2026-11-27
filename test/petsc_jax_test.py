import numpy as np
import pytest

from veros import runtime_settings


@pytest.fixture(autouse=True)
def petsc_options():
    if runtime_settings.backend != "jax":
        pytest.skip("compiled PETSc requires JAX")
    pytest.importorskip("petsc4py.PETSc")
    previous = runtime_settings.linear_solver, runtime_settings.diskless_mode
    object.__setattr__(runtime_settings, "linear_solver", "petsc")
    object.__setattr__(runtime_settings, "diskless_mode", True)
    yield
    object.__setattr__(runtime_settings, "linear_solver", previous[0])
    object.__setattr__(runtime_settings, "diskless_mode", previous[1])


def setup_simulation(streamfunction):
    from integration_test import make_simulation
    from veros.core.external.solvers import get_linear_solver

    if runtime_settings.device == "gpu":
        object.__setattr__(runtime_settings, "linear_solver", "scipy_jax")
    sim = make_simulation(enable_streamfunction=streamfunction)
    sim.setup()
    get_linear_solver(sim.state)
    return sim


@pytest.mark.parametrize("streamfunction", (True, False))
def test_petsc_operator_derivatives(streamfunction, monkeypatch):
    import jax
    import jax.numpy as jnp

    sim = setup_simulation(streamfunction)
    prepared = sim.prepare_integrator(record_diagnostics=(), linear_solver="petsc")
    solver = prepared._template._linear_solver
    import gc
    import weakref
    from veros.core.external.solvers.petsc_jax import CompiledPETScSolver

    dense = solver._matrix.copy().convert("dense")
    matrix = dense.getDenseArray().copy()
    shape = (sim.state.settings.nx, sim.state.settings.ny)
    rng = np.random.default_rng(17)
    rhs, direction, weight = (jnp.asarray(rng.normal(size=shape)) for _ in range(3))
    tolerance = 1e-5 if rhs.dtype == np.float32 else 1e-8
    temporary = CompiledPETScSolver(sim.state)
    reference = weakref.ref(temporary)
    jax.block_until_ready(temporary._compiled_solve(rhs))
    jax.effects_barrier()
    jax.clear_caches()
    del temporary
    gc.collect()
    assert reference() is None

    def solve(value):
        return solver._compiled_solve(value)[0]

    value, tangent = jax.jit(lambda a, b: jax.jvp(solve, (a,), (b,)))(rhs, direction)
    _, pullback = jax.vjp(jax.jit(solve), rhs)
    cotangent = pullback(weight)[0]
    for actual, operator, expected in (
        (value, matrix, rhs),
        (tangent, matrix, direction),
        (cotangent, matrix.T, weight),
    ):
        np.testing.assert_allclose(
            actual,
            np.linalg.solve(operator, np.asarray(expected).ravel(order="F")).reshape(shape, order="F"),
            rtol=tolerance,
            atol=tolerance,
        )
    np.testing.assert_allclose(jnp.vdot(tangent, weight), jnp.vdot(direction, cotangent), rtol=tolerance)
    second = jax.jit(jax.grad(jax.grad(lambda scale: jnp.sum(solve(scale * rhs) ** 2))))(0.7)
    np.testing.assert_allclose(second, 2 * jnp.sum(value**2), rtol=10 * tolerance)
    small = jax.jit(solve)(rhs * 1e-30)
    np.testing.assert_allclose(small / 1e-30, value, rtol=tolerance, atol=tolerance)
    _, zero = jax.jit(lambda a: jax.jvp(solve, (a,), (jnp.zeros_like(a),)))(rhs)
    np.testing.assert_array_equal(zero, np.zeros(shape))
    zero, converged = jax.jit(solver._compiled_solve)(jnp.zeros_like(rhs))
    np.testing.assert_array_equal(zero, np.zeros(shape))
    assert bool(converged)

    from veros.routines import tracing_context

    initial = prepared.pack(sim.state)
    full_rhs = jnp.zeros_like(initial.variables.surface_taux)
    boundary_weight = jnp.asarray(rng.normal(size=full_rhs.shape))

    def boundary_loss(boundary):
        state = prepared._make_state(initial.variables)
        with tracing_context():
            result = solver.solve(state, full_rhs, full_rhs, boundary_val=boundary)
        return jnp.sum(result * boundary_weight)

    evaluate = jax.jit(jax.value_and_grad(boundary_loss))
    _, derivative = evaluate(0.3)
    steps = (1e-2, 1e-3) if rhs.dtype == np.float32 else (1e-3, 1e-4)
    for step in steps:
        plus, _ = evaluate(0.3 + step)
        minus, _ = evaluate(0.3 - step)
        np.testing.assert_allclose(
            derivative, (plus - minus) / (2 * step), rtol=2e-3 if rhs.dtype == np.float32 else 1e-5, atol=1e-5
        )

    calls = []
    original = solver._solve_callback

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(solver, "_solve_callback", counted)

    def unused(value):
        solve(value)
        return jnp.array(0)

    jax.jit(unused)(rhs).block_until_ready()
    jax.effects_barrier()
    assert calls == [1]
    solver._ksp.setTolerances(max_it=0)
    failed, converged = jax.jit(solver._compiled_solve)(rhs)
    assert not bool(converged)
    assert np.isnan(failed).all()
    failed, converged = jax.jit(solver._compiled_solve)(rhs.at[0, 0].set(jnp.nan))
    assert not bool(converged)
    assert np.isnan(failed).all()
    solver._ksp.setTolerances(max_it=1000)
    solver._transpose_ksp.setTolerances(max_it=0)
    _, pullback = jax.vjp(jax.jit(solve), rhs)
    assert np.isnan(pullback(weight)[0]).all()


@pytest.mark.parametrize("streamfunction", (True, False))
def test_petsc_trajectory(streamfunction):
    import jax
    import jax.numpy as jnp
    from integration_test import assert_variables_equal

    sim = setup_simulation(streamfunction)
    prepared = sim.prepare_integrator(
        record_diagnostics=tuple(sim.state.diagnostics), checkpoint=True, linear_solver="petsc"
    )
    initial = prepared.pack(sim.state)
    with pytest.raises(ValueError, match="linear_solver"):
        sim.prepare_integrator(linear_solver="unknown")
    final, _ = prepared.advance(initial, n_steps=6)
    prepared.check_status(final)
    for _ in range(6):
        sim.step(sim.state)
    assert_variables_equal(final.variables, sim.state.variables)
    for name in final.diagnostics:
        assert_variables_equal(final.diagnostics[name], sim.state.diagnostics[name].variables)
    jax.clear_caches()

    def objective(scale):
        carry = prepared.replace_variables(initial, t_star=scale * initial.variables.t_star)
        final, records = prepared.advance(carry, n_steps=6)
        means = records["averages"].values["temp"][:, 2:-2, 2:-2].sum(axis=(1, 2, 3))
        return jnp.sum(jnp.where(records["averages"].valid, means, 0))

    _, gradient = jax.jit(jax.value_and_grad(objective))(1.0)
    assert np.isfinite(gradient) and abs(gradient) > 1e-6
    for step in (1e-3, 1e-4):
        difference = (objective(1 + step) - objective(1 - step)) / (2 * step)
        np.testing.assert_allclose(gradient, difference, rtol=5e-5, atol=1e-5)
    jax.clear_caches()
    _, tangent = jax.jit(lambda a: jax.jvp(objective, (a,), (jnp.ones_like(a),)))(jnp.array(1.0))
    np.testing.assert_allclose(tangent, gradient, rtol=1e-8)


def test_petsc_output_and_restart(tmp_path, monkeypatch):
    from integration_test import test_output_and_restart

    if runtime_settings.device == "gpu":
        pytest.skip("host output and restart checks run on CPU")
    test_output_and_restart(tmp_path, monkeypatch)
