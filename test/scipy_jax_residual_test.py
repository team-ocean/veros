"""The iterative fallback must satisfy both physical and sensitivity accuracy."""

import numpy as np
import pytest


@pytest.fixture
def require_jax():
    from veros import runtime_settings, runtime_state

    if runtime_settings.backend != "jax":
        pytest.skip("JAX linear solver requires the JAX backend")
    runtime_state.backend_module


def identity_solver(monkeypatch, bias, fallback_bias=0.0):
    import jax.numpy as jnp
    from jax.scipy.sparse import linalg
    from veros.core.external.solvers.scipy_jax import JAXSciPySolver

    def approximate(operator, value, **kwargs):
        return value + bias, None

    def fallback(operator, value, **kwargs):
        return value + fallback_bias, None

    monkeypatch.setattr(linalg, "bicgstab", approximate)
    monkeypatch.setattr(linalg, "gmres", fallback)
    solver = object.__new__(JAXSciPySolver)
    coefficients = ((jnp.ones((2, 2)),), jnp.ones((2, 2), dtype=bool), jnp.ones((2, 2)))
    solver._set_coefficients(coefficients[0], ((0, 0),), coefficients[1], coefficients[2])
    return solver, coefficients


@pytest.mark.parametrize("amplitude,bias", ((1e8, 1e-10), (1e-100, 1e-5), (1e-200, 1e-5), (0.0, 1e-10)))
def test_fallback_preserves_absolute_and_small_rhs_accuracy(require_jax, monkeypatch, amplitude, bias):
    import jax
    import jax.numpy as jnp
    from veros.routines import tracing_context

    solver, coefficients = identity_solver(monkeypatch, bias)

    def solution(point):
        rhs = jnp.full((2, 2), amplitude) * point
        with tracing_context():
            value, converged = solver._linear_solve(rhs, jnp.zeros_like(rhs), coefficients)
        return value, converged

    value, converged = solution(jnp.asarray(1.0))
    assert bool(converged)
    np.testing.assert_allclose(value, amplitude, rtol=1e-12, atol=0)
    _, tangent = jax.jvp(lambda point: solution(point)[0], (jnp.asarray(1.0),), (jnp.asarray(1.0),))
    np.testing.assert_allclose(tangent, amplitude, rtol=1e-12, atol=0)
    transpose = jax.vjp(lambda point: solution(point)[0], jnp.asarray(1.0))[1](jnp.ones((2, 2)))[0]
    np.testing.assert_allclose(transpose, 4 * amplitude, rtol=1e-8, atol=0)


def test_unsuccessful_fallback_retains_the_failed_residual_gate(require_jax, monkeypatch):
    import jax.numpy as jnp
    from veros.routines import tracing_context

    solver, coefficients = identity_solver(monkeypatch, 1e-10, fallback_bias=1e-10)
    rhs = jnp.full((2, 2), 1e8)
    with tracing_context():
        _, converged = solver._linear_solve(rhs, jnp.zeros_like(rhs), coefficients)
    assert not bool(converged)
