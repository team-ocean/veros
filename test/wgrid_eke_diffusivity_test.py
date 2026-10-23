"""EKE stratification quadrature clips native weights to occupied W intervals."""

from copy import copy

import numpy as np
import pytest

from wgrid_geometry_test import basin as basin, candidate, reference_width  # noqa: F401
from wgrid_energy_test import energy_state

pytestmark = pytest.mark.usefixtures("jax_backend")


def diffusivity_state(base):
    import jax.numpy as jnp

    state = energy_state(base)
    vs = state.variables
    rng = np.random.default_rng(1941)
    with vs.unlock():
        nsqr = jnp.asarray(rng.uniform(1e-6, 2e-5, size=vs.maskW.shape))
        nsqr = nsqr.at[3, 4, 2].set(-1e-6)
        vs.Nsqr = jnp.broadcast_to(nsqr[..., None], vs.Nsqr.shape)
        vs.coriolis_t = jnp.full_like(vs.coriolis_t, 1e-4).at[4].set(0)
        vs.beta = jnp.full_like(vs.beta, 1e-11)
    return state


def calculate(state):
    from veros.core.eke import set_eke_diffusivities_kernel
    from veros.routines import tracing_context

    with tracing_context():
        return set_eke_diffusivities_kernel(state)


@pytest.mark.parametrize("fraction", (1.0, 0.8, 0.3, 0.12))
def test_rossby_radius_uses_independent_clipped_intervals(basin, fraction):  # noqa: F811
    state = diffusivity_state(candidate(basin, fraction))
    vs, settings = state.variables, state.settings
    occupied = reference_width(state)
    lower = np.asarray(vs.zt)
    nominal = np.append(lower[1:], 0) - lower
    # The stratification rule retains native dzw weights, including the surface.
    fraction = occupied / nominal
    integral = np.sum(
        np.sqrt(np.maximum(0, np.asarray(vs.Nsqr[..., vs.tau]))) * np.asarray(vs.dzw) * fraction / settings.pi,
        axis=2,
    )
    expected = np.minimum(
        integral / np.maximum(abs(np.asarray(vs.coriolis_t)), 1e-16),
        np.sqrt(integral / np.maximum(2 * np.asarray(vs.beta), 1e-16)),
    )
    actual = calculate(state)
    np.testing.assert_allclose(actual.L_rossby, expected, rtol=1e-13, atol=1e-12)
    np.testing.assert_array_equal(actual.L_rossby[np.asarray(vs.kbot) == 0], 0)
    assert np.all(np.isfinite(actual.K_gm))
    assert np.min(actual.K_gm) >= 0


def test_eke_diffusivities_retain_full_cell_values(basin):  # noqa: F811
    partial = diffusivity_state(candidate(basin, 1.0))
    full = copy(partial)
    full._settings = copy(partial.settings)
    with full.settings.unlock():
        full.settings.enable_partial_cells = False
    first, second = calculate(partial), calculate(full)
    for name in first._fields:
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))


def test_eke_diffusivity_geometry_products(basin):  # noqa: F811
    import jax
    import jax.numpy as jnp

    rng = np.random.default_rng(1942)
    fractions = jnp.asarray(rng.uniform(0.22, 0.28, size=(8, 10)))
    patterns = jnp.asarray(rng.uniform(-0.01, 0.01, size=(2, 8, 10)))

    def function(z):
        state = diffusivity_state(candidate(basin, fractions + jnp.einsum("i,ijk->jk", z, patterns)))
        output = calculate(state)
        return jnp.concatenate((output.L_rossby[2:-2, 2:-2].ravel(), output.K_gm[2:-2, 2:-2].ravel()))

    point = jnp.asarray([0.1, -0.1])
    direction = jnp.asarray([0.3, -0.7])
    tangent = jax.jvp(function, (point,), (direction,))[1]
    seed = jnp.cos(jnp.arange(tangent.size))
    transpose = jax.vjp(function, point)[1](seed)[0]
    assert np.linalg.norm(tangent) > 1
    np.testing.assert_allclose(jnp.vdot(seed, tangent), jnp.vdot(transpose, direction), rtol=1e-10, atol=1e-10)
    for epsilon in (1e-3, 1e-4):
        difference = (function(point + epsilon * direction) - function(point - epsilon * direction)) / (2 * epsilon)
        np.testing.assert_allclose(difference, tangent, rtol=2e-3, atol=1e-7)
