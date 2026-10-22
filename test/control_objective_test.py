"""Small control-space transposes agree with independent reverse derivatives."""

import numpy as np
import pytest


def test_control_objective_matches_analytic_gradient_jvp_vjp_and_optimizer():
    import jax
    import jax.numpy as jnp
    from benchmarks.ad_demos.finite_response import ControlObjective, value_and_grad, jvp
    from benchmarks.ad_demos.common import Checks, gradient_checks, descend

    def loss(z):
        return (jnp.exp(z[0]) - 2) ** 2 + (z[1] - 0.3) ** 2 + 0.1

    point = jnp.asarray([0.1, 0.2])
    objective = ControlObjective(loss)
    value, gradient = value_and_grad(objective)(point)
    expected = jnp.asarray([2 * (jnp.exp(point[0]) - 2) * jnp.exp(point[0]), 2 * (point[1] - 0.3)])
    np.testing.assert_allclose(gradient, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(gradient, jax.grad(loss)(point), rtol=1e-13, atol=1e-13)
    direction = jnp.asarray([0.4, -0.2])
    _, tangent = jvp(objective, point, direction)
    _, transpose = objective.vjp(point, jnp.asarray(0.7))
    np.testing.assert_allclose(0.7 * tangent, transpose @ direction, atol=1e-13)
    np.testing.assert_allclose(value, loss(point), atol=1e-15)
    checks = Checks()
    gradient_checks(objective, point, checks)
    assert all(c["passed"] for c in checks.items)
    _, history = descend(objective, point, iterations=8, radius=2)
    assert history[-1] < history[0] * 0.2


def test_control_objective_rejects_vector_output():
    import jax.numpy as jnp
    from benchmarks.ad_demos.finite_response import ControlObjective

    with pytest.raises(ValueError, match="scalar objective"):
        ControlObjective(lambda z: z**2).value_and_grad(jnp.ones(2))


def test_statistical_audit_uses_explicit_control_products(monkeypatch):
    from types import SimpleNamespace
    import jax.numpy as jnp
    from benchmarks.ad_demos.common import Checks
    from benchmarks.ad_demos.finite_response import ControlObjective
    from benchmarks.ad_demos.responses import Response

    calls = []
    objective = ControlObjective(lambda z: 2 * z[0] + 0.5)
    original = objective.value_and_grad

    def product(point):
        calls.append("gradient")
        return original(point)

    monkeypatch.setattr(objective, "value_and_grad", product)
    checks = Checks()
    value, gradient = Response.audit(
        SimpleNamespace(kind="statistics"),
        objective,
        jnp.asarray([0.3]),
        checks,
        lambda point, seed: 2 * point[0] + 0.5,
    )
    assert calls == ["gradient"]
    np.testing.assert_allclose(value, 1.1)
    np.testing.assert_allclose(gradient, [2.0])
    assert len(checks.items) == 6 and all(item["passed"] for item in checks.items)
