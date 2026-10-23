"""Provided spectra must pass independent native-map checks."""

import numpy as np
import pytest

from veros.adjustment import AdjustmentError, ImplicitResponse


def response(matrix, solver):
    size = matrix.shape[0]
    return ImplicitResponse(
        lambda u: np.zeros(size),
        lambda x, u: matrix @ x + u,
        lambda x, u, dx, du: matrix @ dx + du,
        lambda x, u, seed: (matrix.T @ seed, seed),
        lambda x, u: x,
        lambda x, u, dx, du: dx,
        lambda x, u, seed: (seed, np.zeros_like(u)),
        tolerance=1e-11,
        warmup=0,
        stability_solver=solver,
    )


@pytest.mark.parametrize("size", (2, 4))
@pytest.mark.parametrize("complex_pairs", (False, True))
def test_small_stability_multipliers_retain_native_precision(size, complex_pairs):
    if complex_pairs:
        matrix = 1e-30 * np.kron(np.diag(np.linspace(1, 0.5, size // 2)), [[0.1, -0.9], [0.9, 0.1]])
        expected = 1e-30 * np.hypot(0.1, 0.9)
    else:
        matrix = 1e-30 * np.diag(np.linspace(0.5, 0.9, size))
        expected = 9e-31
    engine = response(matrix, None)
    radius, residual = engine.stability(np.zeros(size), np.zeros(size))
    np.testing.assert_allclose(radius, expected, rtol=1e-12, atol=0)
    assert residual < 1e-40


def test_provided_spectrum_is_checked_on_the_actual_map():
    matrix = np.array([[0.9, 0.3, 0, 0], [0, 0.8, 0, 0], [0, 0, 0.4, 0], [0, 0, 0, 0.2]])
    calls = []

    def solver(operator, *, k):
        calls.append(k)
        actual = np.column_stack([operator @ column for column in np.eye(4)])
        np.testing.assert_allclose(actual, matrix, atol=1e-14)
        values, vectors = np.linalg.eig(actual)
        selected = np.argsort(abs(values))[::-1][:k]
        return values[selected], vectors[:, selected]

    engine = response(matrix, solver)
    diagnostics = engine.diagnostics(np.zeros(4))
    assert calls == [2]
    np.testing.assert_allclose(diagnostics["spectral_radius"], 0.9, atol=1e-14)
    assert diagnostics["stability_residual"] < 1e-12
    assert diagnostics["sensitivity_residual"] < 1e-10
    assert len(engine.last_stability["eigenpair_residuals"]) == 2
    assert engine.last_stability["operator_calls"] == 6


def test_provided_spectrum_checks_every_returned_pair():
    engine = response(np.diag([0.9, 0.8, 0.4, 0.2]), lambda operator, k: (np.array([0.9, 0.5]), np.eye(4)[:, :2]))
    radius, error = engine.stability(np.zeros(4), np.zeros(4))
    assert radius == 0.9
    np.testing.assert_allclose(error, 0.3, atol=1e-14)


def test_provided_spectrum_retains_full_operator_with_a_wet_mask():
    matrix = np.diag([0.9, 0, 0.6, 0])

    def solver(operator, *, k):
        assert operator.shape == (4, 4) and k == 2
        actual = np.column_stack([operator @ column for column in np.eye(4)])
        np.testing.assert_array_equal(actual, matrix)
        return np.array([0.9, 0.6]), np.eye(4)[:, [0, 2]]

    engine = response(matrix, solver)
    engine.state_mask = lambda u: np.array([True, False, True, False])
    radius, residual = engine.stability(np.zeros(4), np.zeros(4))
    np.testing.assert_allclose(radius, 0.9, atol=1e-14)
    assert residual < 1e-12 and engine.last_stability["method"] == "provided eigensolver"


def test_provided_spectrum_cannot_hide_an_unstable_mode():
    engine = response(np.diag([1.02, 0.8, 0.4, 0.2]), lambda operator, k: (np.array([1.02, 0.8]), np.eye(4)[:, :2]))
    radius, error = engine.stability(np.zeros(4), np.zeros(4))
    assert radius > 1 and error < 1e-12


def test_nonfinite_native_pair_residual_is_unresolved():
    engine = response(np.diag([0.9, 0.8, 0.4, 0.2]), lambda operator, k: (np.array([0.9, 0.8]), np.eye(4)[:, :2]))
    original = engine.product
    engine.product = lambda x, u, dx, du: (np.full(4, np.nan) if dx[1] else original(x, u, dx, du))
    radius, error = engine.stability(np.zeros(4), np.zeros(4))
    assert radius == 0.9 and np.isinf(error)


@pytest.mark.parametrize("problem", ("missing", "shape", "values", "vectors", "zero", "dtype"))
def test_invalid_provided_spectra_are_rejected(problem):
    values, vectors = np.array([0.9, 0.8]), np.eye(4)[:, :2]
    if problem == "missing":
        values, vectors = values[:1], vectors[:, :1]
    elif problem == "shape":
        vectors = vectors[:3]
    elif problem == "values":
        values[0] = np.nan
    elif problem == "vectors":
        vectors[0, 0] = np.nan
    elif problem == "zero":
        vectors[:, 1] = 0
    else:
        values = values.astype(str)
    engine = response(np.diag([0.9, 0.8, 0.4, 0.2]), lambda operator, k: (values, vectors))
    with pytest.raises(AdjustmentError):
        engine.stability(np.zeros(4), np.zeros(4))


def test_stability_provider_must_be_callable():
    with pytest.raises(TypeError, match="callable"):
        response(np.eye(4), 3)


@pytest.mark.parametrize("kind", ("equilibrium", "periodic", "statistics"))
@pytest.mark.usefixtures("jax_backend")
def test_public_native_response_separates_stability_providers(kind):
    import jax.numpy as jnp
    from veros import runtime_settings
    from bathymetry_helpers import make_simulation

    object.__setattr__(runtime_settings, "diskless_mode", True)
    object.__setattr__(runtime_settings, "linear_solver", "scipy_jax")
    simulation = make_simulation(partial_cells=False, wind=0)
    prepared = simulation.prepare_integrator(record_diagnostics=())
    carry = prepared.pack(simulation.state)
    vs = carry.variables
    carry = prepared.replace_variables(carry, temp=jnp.full_like(vs.temp, 10) * vs.maskT[..., None])
    calls = []

    def first(operator, *, k):
        calls.append(("first", k))
        return -np.arange(1, k + 1, dtype=float), np.eye(operator.shape[0], k)

    def second(operator, *, k):
        calls.append(("second", k))
        return -np.arange(1, k + 1, dtype=float), np.eye(operator.shape[0], k)

    def observe(current):
        return current.variables.temp[5, 5, -1, current.variables.tau]

    arguments = dict(observe=observe, diagnostics=False)
    function = prepared.equilibrium_response if kind == "equilibrium" else prepared.periodic_response
    if kind == "periodic":
        arguments.update(period_steps=1, phases=(0,))
    if kind == "statistics":
        function = prepared.statistical_response
        arguments.update(
            thresholds=jnp.asarray(10.0),
            period_steps=1,
            sample_stride=1,
            window_steps=1,
            burnin_steps=0,
            n_windows=1,
            method="invariant_periodic",
        )
    for solver in (first, second):
        value = function(carry, stability_solver=solver, **arguments)
        np.testing.assert_allclose(value["observations"], [10, 0, 0.5] if kind == "statistics" else 10, atol=1e-10)
    entries = list(prepared._adjusted_responses.values())
    assert len(entries) == 2 and entries[0][0] is entries[1][0]
    for _, engine in entries:
        inputs = next(iter(engine.cache.values()))["inputs"]
        state = engine.solve(inputs)["state"]
        _, residual = engine.stability(state, inputs)
        assert residual > 1e-5
    assert [name for name, _ in calls] == ["first", "second"]
