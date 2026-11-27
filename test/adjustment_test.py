"""Implicit adjustment, stability, and independent sensitivity residuals."""

import numpy as np
import pytest

from veros.adjustment import AdjustmentError, ImplicitResponse


def recurrence_response(matrix, *, initial=10, iterations=32):
    import jax
    import jax.numpy as jnp

    matrix = jnp.asarray(matrix)

    def mapping(x, u):
        return matrix @ x + jnp.array([u[0], u[1] ** 2])

    def observe(x, u):
        return jnp.array([x[0] + 2 * x[1] + u[0] ** 2, x[0] * x[1]])

    return ImplicitResponse(
        lambda u: np.full(2, initial, dtype=float),
        mapping,
        lambda x, u, dx, du: jax.jvp(mapping, (x, u), (dx, du))[1],
        lambda x, u, seed: jax.vjp(mapping, x, u)[1](seed),
        observe,
        lambda x, u, dx, du: jax.jvp(observe, (x, u), (dx, du))[1],
        lambda x, u, seed: jax.vjp(observe, x, u)[1](seed),
        tolerance=1e-10,
        max_iterations=iterations,
        warmup=2,
    )


@pytest.mark.usefixtures("jax_backend")
def test_fixed_point_implicit_products_include_adjustment_and_direct_dependence():
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    matrix = jnp.array([[0.8, 0.1], [0.0, 0.6]])
    engine = recurrence_response(matrix)
    point, direction, seed = jnp.array([0.1, 0.2]), jnp.array([0.3, -0.1]), jnp.array([0.7, -0.2])

    def exact(u):
        state = jnp.linalg.solve(jnp.eye(2) - matrix, jnp.array([u[0], u[1] ** 2]))
        return jnp.array([state[0] + 2 * state[1] + u[0] ** 2, state[0] * state[1]])

    function = engine.transformed(point, (2,))
    np.testing.assert_allclose(function(point), exact(point), atol=1e-10)
    np.testing.assert_allclose(jax.jit(function)(point), exact(point), atol=1e-10)
    np.testing.assert_allclose(
        jax.jvp(function, (point,), (direction,))[1], jax.jvp(exact, (point,), (direction,))[1], atol=1e-10
    )
    np.testing.assert_allclose(jax.vjp(function, point)[1](seed)[0], jax.vjp(exact, point)[1](seed)[0], atol=1e-10)
    np.testing.assert_allclose(
        jax.vmap(function)(jnp.stack((point, point * 2))), jax.vmap(exact)(jnp.stack((point, point * 2))), atol=1e-10
    )
    result = engine.diagnostics(point)
    assert result["converged"] and result["residual"] < 1e-10
    assert result["sensitivity_residual"] < 1e-8 and result["stability_residual"] < 1e-10
    np.testing.assert_allclose(result["spectral_radius"], 0.8, atol=1e-10)
    other = recurrence_response(matrix, initial=-20, iterations=64)
    np.testing.assert_allclose(other.solve(point)["observations"], result["observations"], atol=1e-10)
    np.testing.assert_allclose(other.jvp(point, direction), engine.jvp(point, direction), atol=1e-10)


@pytest.mark.parametrize("scale", (1.0, 1e-14))
@pytest.mark.parametrize("transpose", (False, True))
@pytest.mark.usefixtures("jax_backend")
def test_implicit_linear_solve_preserves_small_sensitivity_units(scale, transpose):
    import jax

    jax.config.update("jax_enable_x64", True)
    matrix = np.array([[0.8, 0.1], [0.0, 0.6]])
    engine = recurrence_response(matrix)
    rhs = scale * np.array([1.2, -0.3])
    solution, residual, info = engine.linear_solve(np.zeros(2), np.array([0.1, 0.2]), rhs, transpose=transpose)
    operator = np.eye(2) - matrix
    expected = np.linalg.solve(operator.T if transpose else operator, rhs)
    assert info == 0 and residual < 1e-9
    np.testing.assert_allclose(solution / scale, expected / scale, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("scale", (1.0, 1e-14))
@pytest.mark.parametrize("transpose", (False, True))
@pytest.mark.usefixtures("jax_backend")
def test_polynomial_fallback_checks_the_original_equation(monkeypatch, scale, transpose):
    import scipy.sparse.linalg as sparse

    matrix = np.array([[0.8, 0.1], [0.0, 0.6]])
    engine = recurrence_response(matrix)

    def polynomial(x, u, vector, *, transpose=False):
        operator = matrix.T if transpose else matrix
        current, total = vector.copy(), np.zeros_like(vector)
        for _ in range(16):
            total += current
            current = operator @ current
        return current, total

    engine.polynomial_product = polynomial
    ordinary_gmres, calls = sparse.gmres, []

    def gmres(operator, rhs, **kwargs):
        calls.append(operator)
        if len(calls) == 1:
            return np.zeros_like(rhs), 1
        return ordinary_gmres(operator, rhs, **kwargs)

    monkeypatch.setattr(sparse, "gmres", gmres)
    rhs = scale * np.array([1.2, -0.3])
    solution, residual, info = engine.linear_solve(np.zeros(2), np.zeros(2), rhs, transpose=transpose)
    operator = np.eye(2) - (matrix.T if transpose else matrix)
    np.testing.assert_allclose(solution / scale, np.linalg.solve(operator, rhs / scale), rtol=1e-10, atol=1e-10)
    assert info == 0 and residual < 1e-9 and len(calls) == 2
    assert engine.last_linear_solve["attempts"][0]["info"] == 1
    assert engine.last_linear_solve["attempts"][1]["method"] == "polynomial GMRES"


@pytest.mark.usefixtures("jax_backend")
def test_polynomial_fallback_cannot_hide_an_invalid_original_residual(monkeypatch):
    import scipy.sparse.linalg as sparse

    engine = recurrence_response([[0.8, 0.1], [0.0, 0.6]])
    engine.polynomial_product = lambda x, u, v, **kwargs: (np.zeros_like(v), v)
    ordinary_gmres, calls = sparse.gmres, []

    def gmres(operator, rhs, **kwargs):
        calls.append(operator)
        if len(calls) == 1:
            return np.zeros_like(rhs), 1
        return ordinary_gmres(operator, rhs, **kwargs)

    monkeypatch.setattr(sparse, "gmres", gmres)
    with pytest.raises(AdjustmentError, match="Implicit sensitivity solve unresolved"):
        engine.linear_solve(np.zeros(2), np.zeros(2), np.array([1.2, -0.3]))
    assert len(calls) == 2
    assert engine.last_linear_solve["attempts"][1]["info"] == 0
    assert engine.last_linear_solve["residual"] > 1e-5


@pytest.mark.parametrize("transpose", (False, True))
def test_wet_krylov_coordinates_follow_inputs_and_preserve_full_equation(monkeypatch, transpose):
    import scipy.sparse.linalg as sparse

    def indices(u):
        return np.array([0, 2]) if u[0] == 0 else np.array([1, 3])

    def matrix(u):
        values = np.zeros((4, 4))
        selected = indices(u)
        values[np.ix_(selected, selected)] = [[0.8, 0.1], [0, 0.6]]
        return values

    engine = ImplicitResponse(
        lambda u: np.zeros(4),
        lambda x, u: matrix(u) @ x,
        lambda x, u, dx, du: matrix(u) @ dx,
        lambda x, u, seed: (matrix(u).T @ seed, np.zeros_like(u)),
        None,
        None,
        None,
        state_mask=lambda u: np.isin(np.arange(4), indices(u)),
    )
    original, dimensions = sparse.gmres, []

    def measured(operator, rhs, **kwargs):
        dimensions.append(operator.shape)
        return original(operator, rhs, **kwargs)

    monkeypatch.setattr(sparse, "gmres", measured)
    for u in (np.zeros(1), np.ones(1)):
        rhs = 1e-14 * np.arange(1, 5, dtype=float)
        operator = np.eye(4) - (matrix(u).T if transpose else matrix(u))
        solution, residual, info = engine.linear_solve(np.zeros(4), u, rhs, transpose=transpose)
        np.testing.assert_allclose(solution / 1e-14, np.linalg.solve(operator, rhs / 1e-14), atol=1e-12)
        assert info == 0 and residual < 1e-12
        radius, residual = engine.stability(np.zeros(4), u)
        np.testing.assert_allclose(radius, 0.8, atol=1e-12)
        assert residual < 1e-12 and engine.last_stability["omitted_zero_eigenvalues"] == 2
    assert dimensions == [(2, 2), (2, 2)]


@pytest.mark.usefixtures("jax_backend")
def test_invalid_wet_projection_cannot_hide_full_native_residual():
    engine = recurrence_response([[0.8, 0.1], [0.0, 0.6]])
    engine.state_mask = lambda u: np.array([True, False])
    with pytest.raises(AdjustmentError, match="Implicit sensitivity solve unresolved"):
        engine.linear_solve(np.zeros(2), np.zeros(2), np.ones(2))
    # A missing outgoing map row is exposed by the full eigenpair residual.
    engine = recurrence_response([[0.8, 0], [0.1, 0.6]])
    engine.state_mask = lambda u: np.array([True, False])
    _, residual = engine.stability(np.zeros(2), np.zeros(2))
    assert residual > 0.09


def test_wet_polynomial_fallback_retains_dry_rhs_and_full_residual(monkeypatch):
    import scipy.sparse.linalg as sparse

    matrix = np.diag([0.8, 0, 0.6, 0])
    engine = ImplicitResponse(
        None,
        None,
        lambda x, u, dx, du: matrix @ dx,
        None,
        None,
        None,
        None,
        state_mask=lambda u: np.array([True, False, True, False]),
    )
    engine.polynomial_product = lambda x, u, v, **kwargs: (matrix @ matrix @ v, v + matrix @ v)
    original, calls = sparse.gmres, []

    def gmres(operator, rhs, **kwargs):
        calls.append(operator.shape)
        return (np.zeros_like(rhs), 1) if len(calls) == 1 else original(operator, rhs, **kwargs)

    monkeypatch.setattr(sparse, "gmres", gmres)
    rhs = np.arange(1, 5, dtype=float)
    solution, residual, info = engine.linear_solve(np.zeros(4), np.zeros(1), rhs)
    np.testing.assert_allclose(solution, np.linalg.solve(np.eye(4) - matrix, rhs), atol=1e-12)
    assert info == 0 and residual < 1e-12 and calls == [(2, 2), (2, 2)]


def test_stability_resolves_clustered_complex_modes_with_passive_histories():
    matrix = np.zeros((192, 192))
    radii = np.linspace(0.9965, 0.98, 80)
    for i, (radius, angle) in enumerate(zip(radii, np.linspace(0.003, 0.03, 80))):
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        matrix[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = radius * rotation
    matrix[160:, :32] = 0.3 * np.eye(32)
    matrix[160:, 160:] = 0.3 * np.eye(32, k=-1)
    engine = ImplicitResponse(
        lambda u: np.zeros(192),
        lambda x, u: matrix @ x,
        lambda x, u, dx, du: matrix @ dx,
        lambda x, u, seed: (matrix.T @ seed, np.zeros_like(u)),
        lambda x, u: x[:2],
        lambda x, u, dx, du: dx[:2],
        lambda x, u, seed: (np.pad(seed, (0, 190)), np.zeros_like(u)),
    )
    radius, residual = engine.stability(np.zeros(192), np.zeros(2))
    np.testing.assert_allclose(radius, radii[0], atol=1e-5)
    assert residual < 1e-5 and engine.last_stability["stopping"] == "converged"


@pytest.mark.usefixtures("jax_backend")
def test_unstable_fixed_point_and_unresolved_adjustment_are_distinct():
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    point = jnp.array([0.1, 0.2])
    unstable = recurrence_response([[1.1, 0], [0, 0.6]])
    result = unstable.diagnostics(point)
    assert result["converged"] and result["spectral_radius"] > 1
    limited = recurrence_response([[0.8, 0], [0, 0.6]], iterations=0)
    assert not limited.solve(point)["converged"]
    with pytest.raises(AdjustmentError, match="Adjusted state unresolved"):
        limited.jvp(point, np.ones(2))
    with pytest.raises(jax.errors.JaxRuntimeError, match="Adjusted state unresolved"):
        limited.transformed(point, (2,))(point).block_until_ready()


@pytest.mark.usefixtures("jax_backend")
def test_periodic_response_includes_the_phase_zero_state_derivative():
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    matrix = jnp.array([[0.85, 0.04], [0, 0.7]])
    period = 12

    def forcing(u, phase):
        return jnp.array(
            [u[0] + u[1] * jnp.cos(2 * jnp.pi * phase / period), u[2] * jnp.sin(2 * jnp.pi * phase / period)]
        )

    def mapping(x, u):
        for phase in range(period):
            x = matrix @ x + forcing(u, phase)
        return x

    def observe(x, u):
        phases = []
        for phase in range(period):
            phases.append(x)
            x = matrix @ x + forcing(u, phase)
        return jnp.stack(phases)

    engine = ImplicitResponse(
        lambda u: np.array([2.0, -1.0]),
        mapping,
        lambda x, u, dx, du: jax.jvp(mapping, (x, u), (dx, du))[1],
        lambda x, u, seed: jax.vjp(mapping, x, u)[1](seed),
        observe,
        lambda x, u, dx, du: jax.jvp(observe, (x, u), (dx, du))[1],
        lambda x, u, seed: jax.vjp(observe, x, u)[1](seed),
        tolerance=1e-10,
    )

    def exact(u):
        cycle = jnp.linalg.matrix_power(matrix, period)
        state = jnp.linalg.solve(jnp.eye(2) - cycle, mapping(jnp.zeros(2), u))
        return observe(state, u)

    point, direction = jnp.array([0.2, 0.05, 0.03]), jnp.array([0.1, -0.2, 0.3])
    function = engine.transformed(point, (period, 2))
    np.testing.assert_allclose(function(point), exact(point), atol=1e-10)
    tangent = jax.jvp(function, (point,), (direction,))[1]
    np.testing.assert_allclose(tangent, jax.jvp(exact, (point,), (direction,))[1], atol=1e-10)
    seed = jnp.sin(jnp.arange(2 * period)).reshape(period, 2)
    np.testing.assert_allclose(jax.vjp(function, point)[1](seed)[0], jax.vjp(exact, point)[1](seed)[0], atol=1e-10)
    frozen = jax.jvp(lambda u: observe(engine.solve(point)["state"], u), (point,), (direction,))[1]
    assert np.linalg.norm(tangent - frozen) > 0.1
