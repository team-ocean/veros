"""Finite geometry reference host products agree with native forward differences."""

import numpy as np
import pytest
from geometry_reference_initial_test import owner

pytestmark = pytest.mark.usefixtures("jax_backend")


def test_geometry_reference_public_error_checkpoint_and_transformed_products():
    import jax
    import jax.numpy as jnp
    from benchmarks.ad_demos.geometry_reference import evaluate

    ocean = owner()
    ocean.sim.build_grid_reference = lambda carry, **kwargs: evaluate(ocean, carry, **kwargs)

    def result(z):
        widths = ocean.initial.variables.dzt + z * jnp.asarray([4.0, -2.0, -3.0, 1.0])
        carry = ocean.prepared.with_vertical_grid(ocean.initial, widths=widths)
        return ocean.prepared.grid_reference_error(carry, refinement=2, response="trajectory", steps=2)

    point = jnp.asarray(0.3)
    initial = result(point)
    assert np.isfinite(initial["error"]) and initial["error"] > 0
    assert initial["projection_geometry_residual"] < 1e-12
    assert initial["remap_budget_residual"] < 1e-12
    assert initial["operator_change_norm"] > 1e-8
    engine = next(iter(ocean._geometry_errors.values()))
    coarse, fine, parameters, coarse_final, fine_final = engine.ends
    for model, start, actual, steps in (
        (engine.coarse, coarse, coarse_final, 2),
        (engine.fine, fine, fine_final, 4),
    ):
        expected = model.advance(start, n_steps=steps, parameters=parameters, final_only=True)[0]
        model.check_status(expected)
        for name in actual.variables.fields():
            np.testing.assert_array_equal(
                getattr(actual.variables, name), getattr(expected.variables, name), err_msg=name
            )

    def fun(z):
        return result(z)["error"]

    value, tangent = jax.jvp(fun, (point,), (jnp.asarray(1.0),))
    gradient = jax.grad(fun)(point)
    assert np.isfinite(value) and abs(float(tangent)) > 1e-10
    np.testing.assert_allclose(gradient, tangent, rtol=1e-7, atol=1e-10)
    from benchmarks.ad_demos.finite_response import ControlObjective

    # The control transpose must agree with the independent native checkpoint VJP.
    control_value, control_gradient = ControlObjective(lambda z: fun(z[0])).value_and_grad(point[None])
    np.testing.assert_allclose(control_gradient[0], gradient, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(control_value, value, rtol=1e-12, atol=1e-12)
    for epsilon in (1e-3, 1e-4):
        fd = (fun(point + epsilon) - fun(point - epsilon)) / (2 * epsilon)
        np.testing.assert_allclose(tangent, fd, rtol=2e-3, atol=1e-8)
    np.testing.assert_allclose(jax.jit(fun)(point), value, rtol=1e-12, atol=1e-12)
    engine = next(iter(ocean._geometry_errors.values()))
    assert len(engine.statistics) == 2
    assert all(s["replay_relative_error"] < 1e-10 for s in engine.statistics)
    assert [s["reference_steps"] for s in engine.statistics] == [2, 4]
    from benchmarks.ad_demos.geometry_reference import release

    release(ocean)
    np.testing.assert_allclose(fun(point), value, rtol=1e-12, atol=1e-12)


def test_geometry_reference_missing_provider_and_unvalidated_responses_stop_early():
    import jax.numpy as jnp
    from benchmarks.ad_demos.geometry_reference import evaluate

    ocean = owner()
    with pytest.raises(ValueError, match="build_grid_reference"):
        ocean.prepared.grid_reference_error(ocean.initial, refinement=2, response="local", steps=1)
    with pytest.raises(NotImplementedError, match="validated invariant_periodic"):
        evaluate(ocean, ocean.initial, parameters={}, refinement=2, response="statistics", steps=1)
    assert not hasattr(ocean, "_geometry_errors")
    with pytest.raises(ValueError, match="one coarse step"):
        evaluate(ocean, ocean.initial, parameters={}, refinement=2, response="local", steps=2)
    assert np.all(np.isfinite(jnp.asarray(ocean.initial.variables.temp)))
