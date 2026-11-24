"""Check mixing with a known squared shear from the preceding momentum step."""

import numpy as np
import pytest


def test_preceding_shear_moments_through_upward_crossings(pytestconfig):
    if pytestconfig.getoption("--backend") != "jax":
        pytest.skip("requires the JAX backend")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from benchmarks.ad_demos.prepared_column_events import (
        differentiable_preceding_shear_moments,
        integrate_preceding_shear,
        MOMENT_FIELDS,
    )

    N, ck, ce, prandtl = 1e-4, 0.1, 0.7, 2.0
    energy, top_energy = 6e-6, 2e-5
    shear, bg, fraction, ratio = 6.6 * N / prandtl, 1.2e-4, 0.6, 2.0
    floor, source, duration = 2e-4, 2e-7, 300.0
    a, decay = ck * np.sqrt(2 / N), ce * np.sqrt(N / 2)
    floor_crossing, background_crossing = floor / a, bg * prandtl / a

    def exact_moments(beta):
        # This case has one growing, uncoupled layer with two known crossings.
        # Solve the affine scalar equation on each interval and integrate its moments.
        E = energy * np.exp(beta)
        left = duration
        integral = np.zeros(4, dtype=np.asarray(E).dtype)
        for threshold, mu_slope, mu_constant, kh_slope, kh_constant in (
            (floor_crossing, 0, floor, (1 - fraction) * a / prandtl, fraction * bg),
            (background_crossing, a, 0, (1 - fraction) * a / prandtl, fraction * bg),
            (None, a, 0, a / prandtl, 0),
        ):
            loss = decay + ratio * N * kh_slope
            stationary = (source - ratio * N * kh_constant) / loss
            dt = left if threshold is None else np.log((stationary - E) / (stationary - threshold)) / loss
            area = stationary * dt - (E - stationary) * np.expm1(-loss * dt) / loss
            E = stationary + (E - stationary) * np.exp(-loss * dt)
            integral += (area, mu_slope * area + mu_constant * dt, kh_slope * area + kh_constant * dt, decay * area)
            left -= dt
        return np.asarray([E, *(integral / duration)])

    def arguments(beta):
        return tuple(
            jnp.asarray(x, dtype=jnp.float64)
            for x in (
                [N, N],
                np.zeros((2, 2)),
                [0, 0],
                [bg, 0],
                fraction,
                ratio,
                jnp.stack((energy * jnp.exp(beta), top_energy)),
                [shear, 0],
                [100, 100],
                [source, source],
                duration,
                ck,
                ce,
                1e-4,
                floor,
                1,
            )
        )

    @jax.jit
    def audit(beta):
        result = integrate_preceding_shear(*arguments(beta))
        return jnp.stack([result[key][0] for key in MOMENT_FIELDS]), result

    beta = jnp.float64(0)
    moments, result = audit(beta)
    assert int(result["status"]) == 0
    assert int(result["segment_count"]) == 3
    np.testing.assert_allclose(result["segment_event_threshold"][:2, 0], [floor_crossing, background_crossing])
    np.testing.assert_allclose(moments, exact_moments(0.0), rtol=1e-11, atol=1e-16)

    def reverse_moments(beta):
        return jnp.stack([field[0] for field in differentiable_preceding_shear_moments(*arguments(beta))])

    derivative = jax.jit(jax.jacrev(reverse_moments))(beta)
    np.testing.assert_allclose(derivative, np.imag(exact_moments(1e-30j)) / 1e-30, rtol=1e-11, atol=1e-16)
