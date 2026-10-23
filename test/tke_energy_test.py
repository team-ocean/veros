"""Independent complementarity and global-work checks for the TKE policy."""

import numpy as np
import pytest


def columns():
    rng = np.random.default_rng(3219)
    bottom = np.array([0, 2, 4, 5, 1, 0, 3, 0])
    nz = 5
    width = rng.uniform(0.2, 2.0, size=(len(bottom), nz))
    water = np.arange(nz) >= bottom[:, None]
    edge = np.arange(nz) == bottom[:, None]
    a, b, c = np.zeros_like(width), np.ones_like(width), np.zeros_like(width)
    for row, first in enumerate(bottom):
        for k in range(first, nz - 1):
            conductance = rng.uniform(0.1, 2.0)
            b[row, k] += conductance / width[row, k]
            c[row, k] = -conductance / width[row, k]
            b[row, k + 1] += conductance / width[row, k + 1]
            a[row, k + 1] = -conductance / width[row, k + 1]
    rhs = rng.normal(size=width.shape) * water
    rhs[0] = -abs(rhs[0])
    rhs[1] = abs(rhs[1])
    rhs[2] = 0
    rates = rng.uniform(0.0, 2.0, size=width.shape)
    return (a, b, c, rhs, water, edge, b.copy(), rates), width


def quadratic_reference(arguments, width):
    from scipy.optimize import nnls

    a, b, c, rhs, water, edge, b_edge, rates = arguments
    energies, corrections = [], []
    for row in range(len(rhs)):
        wet = np.flatnonzero(water[row])
        energy = np.zeros_like(rhs[row])
        if wet.size:
            matrix = np.diag(b[row] + rates[row]) + np.diag(a[row, 1:], -1) + np.diag(c[row, :-1], 1)
            matrix = matrix[np.ix_(wet, wet)]
            volume = width[row, wet]
            factor = np.linalg.cholesky(volume[:, None] * matrix)
            positive = nnls(factor.T, np.linalg.solve(factor, volume * rhs[row, wet]))[0]
            energy[wet] = rhs[row, wet] - (matrix - np.eye(len(wet))) @ positive
        correction = max(-energy[-1], 0)
        energy[-1] = max(energy[-1], 0)
        energies.append(energy)
        corrections.append(correction)
    return np.array(energies), np.array(corrections)


@pytest.mark.parametrize("scale", (1.0, 1e-100))
def test_positive_diffusion_matches_independent_quadratic_and_conserves_signed_energy(scale):
    from veros.core.tke_energy import positive_diffusion_column
    from veros.core.operators import numpy as npx
    from veros.routines import tracing_context

    arguments, width = columns()
    expected, expected_correction = quadratic_reference(arguments, width)
    values = [npx.asarray(value[:, None]) for value in arguments]
    values[3] = values[3] * scale
    with tracing_context():
        energy, correction = positive_diffusion_column(*values)
    energy, correction = np.asarray(energy[:, 0]) / scale, np.asarray(correction[:, 0]) / scale
    np.testing.assert_allclose(energy, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(correction, expected_correction, rtol=1e-12, atol=1e-12)
    rhs, water, rates = arguments[3], arguments[4], arguments[7]
    positive = np.maximum(energy, 0)
    lhs = np.sum(width * water * (energy - rhs + rates * positive), axis=1)
    np.testing.assert_allclose(lhs, width[:, -1] * correction, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("power", (0.0, 1.3, -1.3, 1e-100, -1e-100, 1e5, -1e5))
def test_work_matches_independent_root_and_preserves_power(power):
    from scipy.optimize import brentq
    from veros.core.tke_energy import redistribute
    from veros.core.operators import numpy as npx
    from veros.routines import tracing_context

    rng = np.random.default_rng(3220)
    energy = rng.uniform(-0.2, 0.5, size=(4, 3, 5))
    energy[1, :, 2] = 0
    volume = rng.uniform(0.1, 3, size=energy.shape)
    dt = 2.0
    with tracing_context():
        actual = np.asarray(redistribute(npx.asarray(power), npx.asarray(energy), npx.asarray(volume), dt))
    positive = np.maximum(energy[..., :-1], 0)
    if power == 0:
        np.testing.assert_array_equal(actual, 0)
        return
    upper = abs(power) / np.sum(volume[..., -1])

    def fraction(s):
        ratio = positive / np.where(positive + dt * upper * s > 0, positive + dt * upper * s, 1)
        return s * (np.sum(volume[..., -1]) + np.sum(volume[..., :-1] * ratio)) / np.sum(volume[..., -1]) - 1

    rate = np.sign(power) * upper * brentq(fraction, 0, 1, xtol=1e-14)
    expected = np.full_like(energy, rate)
    expected[..., :-1] = rate * positive / np.where(positive + dt * abs(rate) > 0, positive + dt * abs(rate), 1)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-112)
    np.testing.assert_allclose(np.sum(volume * actual) / power, 1, rtol=1e-12)
    assert np.all(dt * abs(actual[..., :-1]) <= positive + 1e-14)


def test_empty_domain_has_zero_work_and_rejects_nonzero_power():
    from veros.core.tke_energy import redistribute
    from veros.core.operators import numpy as npx
    from veros.routines import tracing_context

    empty = npx.zeros((2, 2, 3))
    with tracing_context():
        zero = redistribute(npx.asarray(0.0), empty, empty, 2.0)
        invalid = redistribute(npx.asarray(1.0), empty, empty, 2.0)
    np.testing.assert_array_equal(zero, 0)
    assert np.isnan(invalid).all()


def test_native_products_include_geometry_sources_and_mixing_coefficients():
    from veros import runtime_settings

    if runtime_settings.backend != "jax":
        pytest.skip("derivative products require JAX")
    import jax
    import jax.numpy as jnp
    from veros.core.tke_energy import positive_diffusion_column, redistribute
    from veros.routines import tracing_context

    arguments, width = columns()
    arguments = tuple(jnp.asarray(value[:, None]) for value in arguments)
    energy = jnp.linspace(-0.2, 0.5, 120).reshape(4, 6, 5)
    volume = jnp.linspace(0.1, 3, energy.size).reshape(energy.shape)

    def function(z):
        a, b, c, rhs, water, edge, b_edge, rates = arguments
        factor = jnp.exp(z[1])
        with tracing_context():
            result, correction = positive_diffusion_column(
                a * factor,
                1 + (b - 1) * factor,
                c * factor,
                rhs * jnp.exp(z[0]),
                water,
                edge,
                1 + (b_edge - 1) * factor,
                rates * jnp.exp(z[2]),
            )
            work = redistribute(0.7 + z[0], energy + 0.01 * z[1], volume * jnp.exp(z[2]), 2.0)
        return jnp.concatenate((result.ravel(), correction.ravel(), work.ravel()))

    point, direction = jnp.zeros(3), jnp.asarray([0.2, -0.3, 0.1])
    value, product = jax.jit(lambda z, dz: jax.jvp(function, (z,), (dz,)))(point, direction)
    seed = jnp.sin(jnp.arange(value.size))
    transpose = jax.jit(lambda z, seed: jax.vjp(function, z)[1](seed)[0])(point, seed)
    np.testing.assert_allclose(jnp.vdot(product, seed), jnp.vdot(direction, transpose), rtol=1e-11, atol=1e-12)
    for epsilon in (1e-4, 1e-5):
        difference = (function(point + epsilon * direction) - function(point - epsilon * direction)) / (2 * epsilon)
        np.testing.assert_allclose(difference, product, rtol=2e-3, atol=1e-8)
    empty = jnp.zeros((2, 2, 3))

    def dry(power):
        with tracing_context():
            return jnp.sum(redistribute(power, empty, empty, 2.0))

    np.testing.assert_array_equal(jax.jit(jax.grad(dry))(0.0), 0)
