"""Small closure fits retain their stopping criterion through curvature refinement."""

import numpy as np


def test_converged_fit_does_not_evaluate_refinement():
    from benchmarks.ad_demos.research import refine_tendency_fit

    point, history, solver = np.zeros(2), [0.1], dict(converged=True)

    def unused(_):
        raise AssertionError("Converged fit was evaluated again")

    result = refine_tendency_fit(unused, point, history, solver)
    assert result[0] is point and result[1] is history and result[2] is solver


def test_nonzero_residual_fit_resolves_original_gradient_tolerance():
    import jax
    import jax.numpy as jnp
    from benchmarks.ad_demos.research import refine_tendency_fit

    def loss(z):
        return (jnp.exp(z[0]) - 2) ** 2 + (z[1] - 0.5) ** 2 + 0.1

    initial = jnp.asarray([0.5, 0.3])
    history = [float(loss(initial))]
    prior = dict(
        converged=False,
        stopping="iteration_limit",
        gradient_tolerance=1e-9,
        max_iterations=64,
        radius=None,
        prior_precision=0,
    )
    point, refined, solver = refine_tendency_fit(loss, initial, history, prior)
    np.testing.assert_allclose(point, [np.log(2), 0.5], atol=1e-9)
    assert solver["converged"]
    assert solver["gauss_newton"] is prior
    assert not prior["converged"] and len(history) == 1
    assert solver["gradient_tolerance"] == prior["gradient_tolerance"]
    assert np.linalg.norm(jax.grad(loss)(point)) <= prior["gradient_tolerance"]
    assert refined[-1] < refined[0]
    assert np.all(np.diff(refined) <= 1e-14)
