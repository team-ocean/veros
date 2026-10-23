"""Periodic invariant products against an independently relaxed forced system."""

import numpy as np
import pytest


@pytest.mark.usefixtures("jax_backend")
def test_periodic_statistics_differentiate_orbit_adjustment_and_event_thresholds():
    import jax
    import jax.numpy as jnp
    from veros.adjustment import ImplicitResponse
    from veros.statistics import periodic_statistics

    jax.config.update("jax_enable_x64", True)
    forcing = jnp.array([[0.1, 0.2], [-0.2, 0.1], [0.3, -0.1], [0.2, 0.4]])

    def cycle(x, u, *, sample=False):
        states = []
        for phase in range(4):
            states.append(x + u[1] ** 2)
            x = (0.4 + 0.1 * u[1]) * x + u[0] * forcing[phase]
        return jnp.stack(states) if sample else x

    def mapping(x, u):
        return cycle(x, u)

    def observe(x, u):
        return cycle(x, u, sample=True)

    engine = ImplicitResponse(
        lambda u: np.ones(2),
        mapping,
        lambda x, u, dx, du: jax.jvp(mapping, (x, u), (dx, du))[1],
        lambda x, u, s: jax.vjp(mapping, x, u)[1](s),
        observe,
        lambda x, u, dx, du: jax.jvp(observe, (x, u), (dx, du))[1],
        lambda x, u, s: jax.vjp(observe, x, u)[1](s),
        tolerance=1e-11,
    )

    class Prepared:
        def periodic_response(self, carry, *, parameters, phases, **kwargs):
            assert phases == (0, 0.25, 0.5, 0.75)
            return dict(observations=engine.transformed(parameters, (4, 2))(parameters))

    def function(u):
        return periodic_statistics(
            Prepared(),
            None,
            parameters=u,
            observe=None,
            thresholds=jnp.array([0.2, -0.1]) + 0.1 * u[0],
            period_steps=4,
            sample_stride=1,
            tolerance=1e-11,
            max_cycles=32,
            ram_checkpoints=4,
            conserved=(),
            diagnostics=False,
        )["observations"]

    def independent(u):
        state = np.ones(2) * -10
        for _ in range(100):
            for phase in range(4):
                state = (0.4 + 0.1 * u[1]) * state + u[0] * np.asarray(forcing[phase])
        samples = []
        for phase in range(4):
            samples.append(state + u[1] ** 2)
            state = (0.4 + 0.1 * u[1]) * state + u[0] * np.asarray(forcing[phase])
        samples = np.array(samples)
        return np.stack(
            (
                samples.mean(0),
                samples.var(0),
                (1 / (1 + np.exp(-(samples - np.array([0.2, -0.1]) - 0.1 * u[0])))).mean(0),
            )
        )

    point, direction = jnp.array([0.3, 0.2]), jnp.array([0.1, -0.2])
    value, tangent = jax.jvp(function, (point,), (direction,))
    np.testing.assert_allclose(value, independent(np.asarray(point)), atol=1e-11)
    np.testing.assert_allclose(jax.jit(function)(point), value, atol=1e-11)
    for epsilon in (1e-3, 1e-4):
        difference = (independent(point + epsilon * direction) - independent(point - epsilon * direction)) / (
            2 * epsilon
        )
        np.testing.assert_allclose(tangent, difference, rtol=1e-6, atol=1e-9)
    seed = jnp.arange(6).reshape(3, 2) / 10
    np.testing.assert_allclose(
        jnp.vdot(seed, tangent), jnp.vdot(jax.vjp(function, point)[1](seed)[0], direction), atol=1e-11
    )


def test_statistical_api_requires_an_explicit_supported_estimator():
    from veros.integration import PreparedIntegrator, CompiledCompatibilityError
    from types import SimpleNamespace

    with pytest.raises(CompiledCompatibilityError, match="No validated chaotic"):
        PreparedIntegrator.statistical_response(
            SimpleNamespace(),
            None,
            observe=None,
            thresholds=0,
            period_steps=4,
            window_steps=4,
            burnin_steps=0,
            n_windows=4,
            sample_stride=1,
            method="shadowing",
        )


@pytest.mark.usefixtures("jax_backend")
def test_periodic_statistics_do_not_publish_an_unstable_or_unresolved_orbit():
    import jax.numpy as jnp
    from veros.statistics import periodic_statistics

    class Prepared:
        def periodic_response(self, carry, **kwargs):
            return dict(
                observations=jnp.ones((4, 2)),
                residual=1e-12,
                spectral_radius=1.01,
                stability_residual=1e-12,
                sensitivity_residual=1e-12,
            )

    result = periodic_statistics(
        Prepared(),
        None,
        parameters={},
        observe=None,
        thresholds=jnp.zeros(2),
        period_steps=4,
        sample_stride=1,
        tolerance=1e-9,
        max_cycles=32,
        ram_checkpoints=4,
        conserved=(),
        diagnostics=True,
    )
    assert not np.isfinite(result["observations"]).any()
