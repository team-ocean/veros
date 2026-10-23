"""W-grid transports preserve constant fields and the native energy quadrature."""

from copy import copy

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("jax_backend")


def updated_velocities(state):
    from veros.core import advection
    from veros.routines import tracing_context

    current = copy(state)
    current._variables = copy(state.variables)
    with tracing_context(), current.variables.unlock():
        current.variables.update(advection.calculate_velocity_on_wgrid_kernel(current))
    return current


def quadrature(variables):
    import jax.numpy as jnp

    return jnp.broadcast_to(variables.dzw, variables.maskW.shape).at[..., -1].multiply(0.5)


@pytest.fixture(scope="module", params=(False, True))
def transport_state(request):
    from partial_energy_test import flow_state

    return updated_velocities(flow_state(request.param))


def tendency(state, values, scheme):
    import jax.numpy as jnp
    from veros.core import advection
    from veros.routines import tracing_context

    vs = state.variables
    with tracing_context():
        east, north, top = getattr(advection, "adv_flux_" + scheme + "_wgrid")(state, values)
    horizontal = (east[2:-2, 2:-2] - east[1:-3, 2:-2]) / (vs.cost[None, 2:-2, None] * vs.dxt[2:-2, None, None]) + (
        north[2:-2, 2:-2] - north[2:-2, 1:-3]
    ) / (vs.cost[None, 2:-2, None] * vs.dyt[None, 2:-2, None])
    vertical = top - jnp.pad(top[..., :-1], ((0, 0), (0, 0), (1, 0)))
    result = -(horizontal + vertical[2:-2, 2:-2] / quadrature(vs)[2:-2, 2:-2]) * vs.maskW[2:-2, 2:-2]
    return result, (east, north, top)


def test_wgrid_column_transport_matches_occupied_faces(transport_state):
    import jax.numpy as jnp

    state, vs = transport_state, transport_state.variables
    for name, grid in (("u", "U"), ("v", "V")):
        fraction = getattr(vs, "hfac" + grid) if state.settings.enable_partial_cells else getattr(vs, "mask" + grid)
        expected = jnp.sum(getattr(vs, name)[..., vs.tau] * vs.dzt * fraction, axis=2)
        actual = jnp.sum(getattr(vs, name + "_wgrid") * quadrature(vs), axis=2)
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-13)
    np.testing.assert_allclose(vs.w_wgrid[2:-2, 2:-2, -1], 0, atol=1e-17)


@pytest.mark.parametrize("scheme", ("upwind", "superbee"))
def test_wgrid_constant_field_and_random_inventory(transport_state, scheme):
    import jax.numpy as jnp

    vs = transport_state.variables
    constant, _ = tendency(transport_state, jnp.ones(vs.maskW.shape), scheme)
    np.testing.assert_allclose(constant, 0, atol=1e-18)
    values = jnp.asarray(np.random.default_rng(1725).uniform(0.5, 1.5, size=vs.maskW.shape))
    rate, (east, north, top) = tendency(transport_state, values, scheme)
    thickness = quadrature(vs)[2:-2, 2:-2]
    volume = vs.area_t[2:-2, 2:-2, None] * thickness * vs.maskW[2:-2, 2:-2]
    scale = (
        jnp.sum(abs(east[2:-2, 2:-2]) * vs.dyt[None, 2:-2, None] * thickness)
        + jnp.sum(abs(north[2:-2, 2:-2]) * vs.dxt[2:-2, None, None] * thickness)
        + jnp.sum(abs(top[2:-2, 2:-2]) * vs.area_t[2:-2, 2:-2, None])
    )
    assert abs(float(jnp.sum(rate * volume))) < 1e-13 * float(scale)


def test_occupied_transport_geometry_products(transport_state):
    import jax
    import jax.numpy as jnp
    from veros.core import advection
    from veros.routines import tracing_context

    if not transport_state.settings.enable_partial_cells:
        pytest.skip("occupied face fractions require partial cells")
    base = transport_state.variables
    pattern = jnp.sin(jnp.arange(base.hfacU.size).reshape(base.hfacU.shape))

    def function(z):
        state = copy(transport_state)
        state._variables = copy(base)
        with state.variables.unlock():
            state.variables.hfacU = base.hfacU * (1 + z * pattern)
            state.variables.hfacV = base.hfacV * (1 - z * pattern)
        with tracing_context():
            result = advection.calculate_velocity_on_wgrid_kernel(state)
        return jnp.concatenate([getattr(result, name).ravel() for name in ("u_wgrid", "v_wgrid", "w_wgrid")])

    point = jnp.asarray(0.03)
    seed = jnp.cos(jnp.arange(function(point).size))
    tangent = jax.jvp(function, (point,), (jnp.asarray(1.0),))[1]
    transpose = jax.vjp(function, point)[1](seed)[0]
    assert np.linalg.norm(tangent) > 1e-3
    np.testing.assert_allclose(jnp.vdot(seed, tangent), transpose, atol=1e-12, rtol=1e-10)
    for epsilon in (1e-3, 1e-4):
        difference = (function(point + epsilon) - function(point - epsilon)) / (2 * epsilon)
        np.testing.assert_allclose(tangent, difference, rtol=1e-8, atol=1e-12)


@pytest.fixture(scope="module")
def energy_state():
    import jax.numpy as jnp
    from bathymetry_helpers import make_simulation
    from partial_energy_test import flow_state

    flow = flow_state(False)
    simulation = make_simulation(
        partial_cells=False,
        bottom_indices=np.asarray(flow.variables.kbot[2:-2, 2:-2]),
        wind=0,
        enable_tke=True,
        enable_eke=True,
        enable_tke_superbee_advection=True,
        enable_eke_superbee_advection=True,
        c_eps=0,
        alpha_tke=0,
        eke_c_eps=0,
        alpha_eke=0,
    )
    state = simulation.state
    with state.variables.unlock():
        vs = state.variables
        vs.u, vs.v = flow.variables.u, flow.variables.v
        vs.kappaM = jnp.zeros_like(vs.kappaM)
        vs.mxl = jnp.ones_like(vs.mxl)
        vs.eke_len = jnp.ones_like(vs.eke_len)
        vs.sqrttke = jnp.ones_like(vs.sqrttke)
        vs.sqrteke = jnp.ones_like(vs.sqrteke)
        vs.tke = jnp.broadcast_to(vs.maskW[..., None], vs.tke.shape).astype(vs.tke.dtype)
        vs.eke = jnp.broadcast_to(vs.maskW[..., None], vs.eke.shape).astype(vs.eke.dtype)
    return updated_velocities(state)


@pytest.mark.parametrize("name", ("tke", "eke"))
def test_native_energy_advection_preserves_constant(energy_state, name):
    from veros.core import tke, eke
    from veros.routines import tracing_context

    state, vs = energy_state, energy_state.variables
    module = tke if name == "tke" else eke
    with tracing_context():
        result = getattr(module, "integrate_" + name + "_kernel")(state)
    np.testing.assert_allclose(getattr(result, name)[2:-2, 2:-2, :, vs.taup1], vs.maskW[2:-2, 2:-2], rtol=0, atol=1e-14)


def test_full_cell_limit_uses_the_same_transports(transport_state):
    from veros.routines import tracing_context
    from veros.core import advection

    if not transport_state.settings.enable_partial_cells:
        pytest.skip("the full-cell limit is constructed from a partial-cell state")
    partial = copy(transport_state)
    partial._variables = copy(transport_state.variables)
    with partial.variables.unlock():
        for grid in ("T", "U", "V"):
            setattr(partial.variables, "hfac" + grid, getattr(partial.variables, "mask" + grid))
    full = copy(partial)
    full._settings = copy(partial.settings)
    with full.settings.unlock():
        full.settings.enable_partial_cells = False
    with tracing_context():
        first = advection.calculate_velocity_on_wgrid_kernel(partial)
        second = advection.calculate_velocity_on_wgrid_kernel(full)
    for name in ("u_wgrid", "v_wgrid", "w_wgrid"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
