"""Occupied W volumes and fluxes follow clipped native control-volume intervals."""

from copy import copy

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("jax_backend")


@pytest.fixture(scope="module", params=(False, True))
def basin(request):
    import jax.numpy as jnp
    from veros import veros_routine
    from bathymetry_helpers import make_simulation

    bottom = 1 + (np.arange(8)[:, None] + np.arange(10)[None, :]) % 4
    bottom[3, 4] = 0
    original = make_simulation(bottom_indices=bottom, cyclic_x=request.param, wind=0)

    class Nonuniform(type(original)):
        @veros_routine
        def set_grid(self, state):
            super().set_grid(state)
            state.variables.dzt = jnp.asarray([60.0, 90.0, 120.0, 150.0])

    simulation = Nonuniform()
    simulation.setup()
    prepared = simulation.prepare_integrator(record_diagnostics=())
    initial = prepared.pack(simulation.state)
    return prepared, initial


def candidate(basin, fraction):
    import jax.numpy as jnp

    prepared, initial = basin
    vs = initial.variables
    width = vs.dzt[jnp.maximum(vs.kbot[2:-2, 2:-2] - 1, 0)]
    depth = jnp.where(vs.kbot[2:-2, 2:-2] > 0, vs.ht[2:-2, 2:-2] - (1 - fraction) * width, 0)
    carry = prepared.with_bathymetry(initial, depth)
    return prepared._make_state(carry.variables)


def reference_width(state):
    vs = state.variables
    lower = np.asarray(vs.zt)
    upper = np.append(lower[1:], 0)
    return np.maximum(0, np.minimum(upper, 0) - np.maximum(lower, -np.asarray(vs.ht)[..., None])) * np.asarray(vs.maskW)


@pytest.mark.parametrize("fraction", (1.0, 0.8, 0.3, 0.12))
def test_w_control_volumes_are_independent_interval_intersections(basin, fraction):
    from veros.core.geometry import wgrid_thickness

    state = candidate(basin, fraction)
    actual = np.asarray(wgrid_thickness(state))
    np.testing.assert_allclose(actual, reference_width(state), rtol=0, atol=1e-12)
    assert np.min(actual[np.asarray(state.variables.maskW)]) > 0
    if fraction == 1:
        full = copy(state)
        full._settings = copy(state.settings)
        with full.settings.unlock():
            full.settings.enable_partial_cells = False
        np.testing.assert_array_equal(actual, wgrid_thickness(full))


def fluxes(state, values):
    import jax.numpy as jnp
    from veros.core import utilities
    from veros.routines import tracing_context

    vs = state.variables
    with tracing_context():
        values = utilities.enforce_boundaries(values, state.settings.enable_cyclic_x)
    east = (
        jnp.zeros_like(values)
        .at[:-1]
        .set(100 * (values[1:] - values[:-1]) / (vs.cost[None, :, None] * vs.dxu[:-1, None, None]))
    )
    north = (
        jnp.zeros_like(values)
        .at[:, :-1]
        .set(100 * (values[:, 1:] - values[:, :-1]) / vs.dyu[None, :-1, None] * vs.cosu[None, :-1, None])
    )
    return east, north


def test_lateral_fluxes_conserve_are_self_adjoint_and_dissipate(basin):
    import jax.numpy as jnp
    from veros.core.geometry import wgrid_horizontal_divergence

    state = candidate(basin, 0.3)
    vs = state.variables
    rng = np.random.default_rng(1825)
    first, second = [jnp.asarray(rng.normal(size=vs.maskW.shape)) for _ in range(2)]
    first_rate = wgrid_horizontal_divergence(state, *fluxes(state, first))
    second_rate = wgrid_horizontal_divergence(state, *fluxes(state, second))
    width = reference_width(state)[2:-2, 2:-2]
    volume = np.asarray(vs.area_t[2:-2, 2:-2, None]) * width
    inventory = np.asarray(first_rate) * volume
    assert np.sum(abs(inventory)) > 1
    assert abs(np.sum(inventory)) < 1e-13 * np.sum(abs(inventory))
    forward = np.sum(volume * np.asarray(first[2:-2, 2:-2]) * second_rate)
    reverse = np.sum(volume * np.asarray(second[2:-2, 2:-2]) * first_rate)
    np.testing.assert_allclose(forward, reverse, rtol=1e-12, atol=1e-10)
    work = np.sum(volume * np.asarray(first[2:-2, 2:-2]) * first_rate)
    assert work < 0
    # Independently sum physical face conductances, counting each periodic face once.
    east, north = fluxes(state, first)
    expected = 0.0
    field = np.asarray(first).copy()
    if state.settings.enable_cyclic_x:
        field[:2] = field[-4:-2]
        field[-2:] = field[2:4]
    widths = reference_width(state)
    for i in range(2, field.shape[0] - 2):
        for j in range(2, field.shape[1] - 2):
            for k in range(field.shape[2]):
                expected -= (
                    float(east[i, j, k])
                    * (field[i + 1, j, k] - field[i, j, k])
                    * min(widths[i, j, k], widths[i + 1, j, k])
                    * float(vs.dyt[j])
                )
                expected -= (
                    float(north[i, j, k])
                    * (field[i, j + 1, k] - field[i, j, k])
                    * min(widths[i, j, k], widths[i, j + 1, k])
                    * float(vs.dxt[i])
                )
    np.testing.assert_allclose(work, expected, rtol=1e-12, atol=1e-9)
    constant = wgrid_horizontal_divergence(state, *fluxes(state, jnp.ones_like(first)))
    np.testing.assert_array_equal(constant, 0)


def test_geometry_products_cover_occupied_volumes_and_faces(basin):
    import jax
    import jax.numpy as jnp
    from veros.core.geometry import wgrid_horizontal_divergence

    _, initial = basin
    rng = np.random.default_rng(1826)
    shape = initial.variables.ht[2:-2, 2:-2].shape
    fractions = jnp.asarray(rng.uniform(0.22, 0.30, size=shape))
    direction = jnp.asarray(rng.uniform(-0.01, 0.01, size=shape))
    values = jnp.asarray(rng.normal(size=initial.variables.maskW.shape))
    seed = jnp.cos(jnp.arange(8 * 10 * 4)).reshape(8, 10, 4)

    def function(z):
        state = candidate(basin, fractions + z * direction)
        return wgrid_horizontal_divergence(state, *fluxes(state, values))

    point = jnp.asarray(0.1)
    tangent = jax.jvp(function, (point,), (jnp.asarray(1.0),))[1]
    transpose = jax.vjp(function, point)[1](seed)[0]
    assert np.linalg.norm(tangent) > 1e-8
    np.testing.assert_allclose(jnp.vdot(seed, tangent), transpose, rtol=1e-9, atol=1e-12)
    for epsilon in (1e-3, 1e-4):
        difference = (function(point + epsilon) - function(point - epsilon)) / (2 * epsilon)
        np.testing.assert_allclose(tangent, difference, rtol=2e-3, atol=1e-10)
