"""Recoverable optimizer progress without additional numerical evaluations."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("jax_backend")


def test_descent_progress_preserves_evaluations_and_restarts_from_an_accepted_point():
    import jax.numpy as jnp
    from benchmarks.ad_demos.common import descend

    calls = []

    def objective(z):
        calls.append(1)
        return jnp.sum((z - jnp.array([0.1, 0.2])) ** 2)

    point = jnp.array([0.8, -0.1])
    expected, history = descend(objective, point, iterations=4, radius=1)
    expected_calls = len(calls)
    calls.clear()
    events = []
    actual, observed_history = descend(objective, point, iterations=4, radius=1, on_iteration=events.append)
    assert len(calls) == expected_calls
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(observed_history, history)
    checkpoint = next(event for event in events if event["stage"] == "accepted" and event["iteration"] == 2)
    final, remainder = descend(
        objective, jnp.asarray(checkpoint["controls"]), iterations=4 - checkpoint["iteration"], radius=1
    )
    np.testing.assert_array_equal(final, expected)
    np.testing.assert_array_equal(checkpoint["history"] + remainder, history)
    assert events[-1]["stage"] == "complete"
    assert len(checkpoint["history"]) == 2


def test_constrained_progress_does_not_evaluate_the_constraint_again():
    import jax.numpy as jnp
    from benchmarks.ad_demos.inverse import constrained_minimum

    calls = []

    def constraint(z):
        calls.append(1)
        return z[0] + 2 * z[1] - 0.5

    point = jnp.zeros(2)
    expected, before = constrained_minimum(constraint, point)
    count = len(calls)
    calls.clear()
    events = []
    evaluations = []
    actual, after = constrained_minimum(constraint, point, on_iteration=events.append, on_evaluation=evaluations.append)
    assert len(calls) == count
    np.testing.assert_array_equal(actual, expected)
    assert after == before
    assert events and all(event["stage"] in ("feasibility", "cost") for event in events)
    assert any(event["slack_sv"] is not None for event in events)
    assert evaluations
    for evaluation in evaluations:
        z = evaluation["controls"]
        np.testing.assert_allclose(evaluation["value"], z[0] + 2 * z[1] - 0.5)
        np.testing.assert_array_equal(evaluation["gradient"], [1, 2])


def test_constraint_line_search_uses_values_without_differentiating_rejected_trials():
    import jax
    import jax.numpy as jnp
    from scipy.optimize import brentq
    from benchmarks.ad_demos.inverse import constrained_minimum

    values, gradients = [], []

    def expression(z):
        return 0.2 - (1 - z[0]) ** 2 - 100 * (z[1] - z[0] ** 2) ** 2

    def constraint(z):
        values.append(np.asarray(z).copy())
        return expression(z)

    def differentiate(z):
        gradients.append(np.asarray(z).copy())
        return jax.value_and_grad(expression)(z)

    point, diagnostic = constrained_minimum(constraint, jnp.zeros(2), value_and_grad_fn=differentiate)
    assert diagnostic["success"] and diagnostic["feasible"]

    def boundary(x):
        root = np.sqrt(0.2 - (1 - x) ** 2)
        y = x**2 - root / 10
        return y, 2 * x - (1 - x) / (10 * root)

    reference_x = brentq(lambda x: x + boundary(x)[0] * boundary(x)[1], 1 - np.sqrt(0.2) + 1e-12, 0.7)
    np.testing.assert_allclose(point, [reference_x, boundary(reference_x)[0]], atol=1e-7)
    assert any(not any(np.array_equal(value, gradient) for gradient in gradients) for value in values)
    assert len(values) > len(gradients)
    assert diagnostic["value_evaluations"] == len(values)
    assert diagnostic["gradient_evaluations"] == len(gradients)


def test_forward_gradient_preserves_descent_and_constrained_solution():
    import jax
    import jax.numpy as jnp
    from benchmarks.ad_demos.common import descend
    from benchmarks.ad_demos.finite_response import forward_value_and_grad
    from benchmarks.ad_demos.inverse import constrained_minimum

    point = jnp.array([0.2, -0.3, 0.1])

    def objective(z):
        return jnp.sum(jnp.sin(z + 0.2) ** 2) + 0.1 * jnp.sum(z) ** 2

    actual = forward_value_and_grad(objective)(point)
    expected = jax.value_and_grad(objective)(point)
    for left, right in zip(actual, expected):
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12)
    fit, history = descend(objective, point, radius=1)
    forward_fit, forward_history = descend(
        objective, point, radius=1, value_and_grad_fn=forward_value_and_grad(objective)
    )
    np.testing.assert_allclose(forward_fit, fit, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(forward_history, history, rtol=1e-12, atol=1e-12)

    def constraint(z):
        return jnp.sin(z[0]) + 2 * z[1] - z[2] ** 2 - 0.5

    optimum, diagnostic = constrained_minimum(constraint, point)
    forward_optimum, forward_diagnostic = constrained_minimum(
        constraint, point, value_and_grad_fn=forward_value_and_grad(constraint)
    )
    np.testing.assert_allclose(forward_optimum, optimum, rtol=1e-12, atol=1e-12)
    assert diagnostic["success"] and forward_diagnostic["success"]
    for name in ("slack_sv", "stationarity_residual", "complementarity"):
        np.testing.assert_allclose(forward_diagnostic[name], diagnostic[name], rtol=1e-10, atol=1e-12)


def test_completed_heat_stage_is_saved_before_a_later_optimizer_fails(monkeypatch):
    import jax.numpy as jnp
    from benchmarks.ad_demos.common import Checks
    from benchmarks.ad_demos import research

    count = 0

    def descent(function, point, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("interrupted event optimizer")
        kwargs["diagnostics"]["stopping"] = "iteration_limit"
        return jnp.array([0.1, 0, 0, 0]), [0.0, -0.1]

    monkeypatch.setattr(research, "descend", descent)
    response = SimpleNamespace(kind="statistics", indices=jnp.arange(4), full=lambda z: z)
    response.ocean = SimpleNamespace(heat_index=0, amoc_index=1, check_controls=lambda z, checks: None)
    response.values = lambda z, **kwargs: jnp.array([[z[0], 1.0], [0, 0], [0, 0]])
    response.function = lambda fn: lambda z: fn(response.values(z), z)
    response.audit = lambda fn, point, checks, forward: (float(fn(point)), jnp.ones_like(point))
    checks = Checks()
    snapshots = []
    checks.on_update = lambda: snapshots.append(deepcopy(checks.metrics))
    with pytest.raises(RuntimeError, match="interrupted event"):
        research.perturbation(response, checks)
    assert checks.metrics["heat_controls"] == [0.1, 0, 0, 0]
    assert checks.metrics["heat_loss_history"] == [0.0, -0.1]
    assert snapshots[0]["heat_optimizer"]["stopping"] == "iteration_limit"


def test_constraint_evaluations_survive_an_optimizer_interruption(monkeypatch):
    import jax.numpy as jnp
    from benchmarks.ad_demos.common import Checks
    from benchmarks.ad_demos import research

    monkeypatch.setattr(research, "descend", lambda function, point, **kwargs: (point, [0.0, -0.1]))

    def interrupted(constraint, point, **kwargs):
        kwargs["on_evaluation"](dict(stage="evaluation", controls=point.tolist(), value=-0.1, gradient=[1, 2, 3, 4]))
        kwargs["on_evaluation"](dict(stage="evaluation", controls=[0.1, 0, 0, 0], value=0.0, gradient=[2, 3, 4, 5]))
        raise RuntimeError("interrupted constraint optimizer")

    monkeypatch.setattr(research, "constrained_minimum", interrupted)
    response = SimpleNamespace(kind="statistics", indices=jnp.arange(4), full=lambda z: z)
    response.ocean = SimpleNamespace(heat_index=0, amoc_index=1, check_controls=lambda z, checks: None)
    response.values = lambda z, **kwargs: jnp.array([[z[0], 1.0], [0, 0], [0.2, 0]])
    response.function = lambda function: lambda z: function(response.values(z), z)
    response.audit = lambda function, point, checks, forward: (float(function(point)), jnp.ones_like(point))
    checks = Checks()
    snapshots = []
    checks.on_update = lambda: snapshots.append(deepcopy(checks.metrics))
    with pytest.raises(RuntimeError, match="interrupted constraint"):
        research.perturbation(response, checks)
    records = checks.metrics["perturbation_evaluations"]["constraint"]
    assert len(records) == 2
    assert records[0]["gradient"] == [1, 2, 3, 4]
    assert records[1]["controls"] == [0.1, 0, 0, 0]
    assert snapshots[-1]["perturbation_evaluations"]["constraint"] == records


@pytest.mark.parametrize("failure_code", (4, 5, 6))
def test_geometry_backtracking_rejects_inadmissible_trials_before_reference_allocation(failure_code):
    from types import MethodType
    import jax
    import jax.numpy as jnp
    from benchmarks.ad_demos import research
    from benchmarks.ad_demos.common import Checks
    from benchmarks.ad_demos.responses import Response

    def controls(full):
        coordinate = full[14]
        code = jnp.where((coordinate > 0) & (coordinate < 0.1), failure_code, 0)
        return SimpleNamespace(coordinate=coordinate, failure_code=code), {}

    def reference(carry, **kwargs):
        if not isinstance(carry.failure_code, jax.core.Tracer):
            assert int(carry.failure_code) == 0, "An inadmissible trial allocated a reference"
        return dict(error=1 + carry.coordinate**2, remap_budget_residual=0, operator_change_norm=1)

    ocean = SimpleNamespace(
        controls=controls,
        prepared=SimpleNamespace(grid_reference_error=reference),
        config=dict(setup="native_basin"),
        acc_index=0,
    )
    response = SimpleNamespace(kind="local", steps=1, ocean=ocean, config=ocean.config)
    response.full = lambda z: jnp.zeros(16).at[12:16].set(z)
    response.values = lambda z, **kwargs: jnp.asarray([[1 + z[0], 2 + z[1]]])
    response.function = lambda fn: SimpleNamespace(
        map_controls=lambda mapping: lambda point: fn(response.values(mapping(point)), mapping(point))
    )
    response.audit = MethodType(Response.audit, response)
    checks = Checks()
    research.geometry_validation(response, jnp.asarray([0.1, 0.2, 0.31666666667, 0.4]), checks)
    rejected = checks.metrics["grid_accuracy_rejected_controls"]
    assert rejected and all(row["failure_code"] == failure_code for row in rejected)
    assert all(0 < row["controls"][0] < 0.1 for row in rejected)
    assert checks.metrics["grid_accuracy_history"][-1] < checks.metrics["grid_accuracy_history"][0]
    assert all(item["passed"] for item in checks.items)


@pytest.mark.parametrize("matrix_free", (False, True))
def test_inverse_backtracking_rejects_inadmissible_controls_without_evaluating_them(matrix_free):
    import jax
    import jax.numpy as jnp
    from benchmarks.ad_demos.inverse import least_squares

    def function(z):
        return (z - 0.8) * jnp.exp(-z)

    class Residual:
        def __call__(self, z):
            assert not 0.4 < float(z[0]) < 0.6, "An inadmissible trial evaluated the model"
            return function(z)

        def jacobian(self, z):
            return jax.jacfwd(function)(z)

        def jvp(self, z, direction):
            return jax.jvp(function, (z,), (direction,))

        def vjp(self, z, seed):
            return function(z), jax.vjp(function, z)[1](seed)[0]

    rejected = []

    def admissible(z):
        valid = not 0.4 < float(z[0]) < 0.6
        if not valid:
            rejected.append(float(z[0]))
        return valid

    fit, history, diagnostic = least_squares(
        Residual(), jnp.zeros(1), prior_precision=0, matrix_free=matrix_free, admissible=admissible
    )
    assert rejected and diagnostic["converged"]
    np.testing.assert_allclose(fit, [0.8], atol=1e-6)
    assert history[-1] < 1e-12
    _, _, invalid = least_squares(Residual(), jnp.asarray([0.5]), admissible=admissible)
    assert invalid["stopping"] == "inadmissible_initial_point" and not invalid["converged"]


def test_gradient_start_escapes_a_weakly_identified_nonlinear_inverse_basin():
    import jax
    import jax.numpy as jnp
    from benchmarks.ad_demos.inverse import JaxResidual, gradient_start, least_squares

    def residual(z):
        x = z[0]
        return jnp.asarray([(x - 0.01) * (x - 0.305), 1e-3 * (x - 0.305)])

    def objective(z):
        values = residual(z)
        return 0.5 * (values @ values)

    point = jnp.zeros(1)
    function = JaxResidual(residual)
    cold, cold_history, cold_diagnostic = least_squares(function, point, prior_precision=0, max_iterations=8)
    assert float(cold[0]) < 0.02 and cold_diagnostic["converged"]
    assert cold_history[-1] > 1e-14
    value, gradient = jax.value_and_grad(objective)(point)
    events = []
    start, trials = gradient_start(objective, point, value, gradient, on_trial=events.append)
    assert events == trials and float(objective(start)) < float(value)
    fit, history, diagnostic = least_squares(function, start, prior_precision=0, zero_loss_tolerance=1e-14)
    assert diagnostic["converged"] and history[-1] < 1e-14 < cold_history[-1]
    np.testing.assert_allclose(fit, [0.305], atol=1e-6)
    assert all(np.linalg.norm(trial["controls"]) <= 1 for trial in trials)


def test_gradient_start_skips_invalid_trials_and_preserves_a_stationary_point():
    import jax.numpy as jnp
    from benchmarks.ad_demos.inverse import gradient_start

    calls = []

    def function(z):
        calls.append(float(z[0]))
        assert float(z[0]) <= 0.5
        return (z[0] - 0.3) ** 2

    point = jnp.zeros(1)
    selected, trials = gradient_start(
        function, point, 0.09, jnp.asarray([-0.6]), admissible=lambda z: float(z[0]) <= 0.5
    )
    assert trials[0]["admissible"] is False and calls == [0.3, 0.1]
    np.testing.assert_allclose(selected, [0.3])
    unchanged, empty = gradient_start(function, point, 0.09, jnp.zeros(1))
    np.testing.assert_array_equal(unchanged, point)
    assert empty == [] and calls == [0.3, 0.1]
