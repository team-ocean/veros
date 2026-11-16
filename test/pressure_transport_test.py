"""Pressure forcing and stored velocities use the same wet-column transport."""

from decimal import Decimal, localcontext

import numpy as np
import pytest


def exact_transport(value, weight):
    with localcontext() as context:
        context.prec = 80
        return sum(
            (Decimal.from_float(float(v)) * Decimal.from_float(float(w)) for v, w in zip(value, weight)),
            Decimal(0),
        )


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_weighted_sum_retains_cancellation_residual(dtype):
    from veros import runtime_settings
    from veros.core.transport import _weighted_sum
    from veros.core.operators import numpy as npx

    weight = np.asarray([40, 60, 80, 120], dtype=dtype)
    values = np.asarray([[-3.4, -2.2, 0.0004, 2.2330666666666668], [0, 0, 0, 0]], dtype=dtype)
    function = _weighted_sum
    if runtime_settings.backend == "jax":
        import jax

        function = jax.jit(function)
    high, low = function(npx.asarray(values), npx.broadcast_to(npx.asarray(weight), values.shape))
    for value, h, l in zip(values, np.asarray(high), np.asarray(low)):
        expected = exact_transport(value, weight)
        actual = Decimal.from_float(float(h)) + Decimal.from_float(float(l))
        assert abs(expected - actual) <= Decimal("1e-25")


def pressure_state(partial):
    from veros.state import get_default_state
    from veros.core.operators import at, numpy as npx, update

    state = get_default_state()
    with state.settings.unlock():
        state.settings.nx, state.settings.ny, state.settings.nz = 4, 2, 4
        state.settings.dt_mom = state.settings.dt_tracer = 0.87890625
        state.settings.enable_partial_cells = partial
        state.settings.enable_streamfunction = False
        state.settings.enable_cyclic_x = True
    state.initialize_variables()
    vs = state.variables
    with vs.unlock():
        vs.dzt = npx.asarray([40.0, 60.0, 80.0, 120.0])
        vs.dzw = npx.asarray([40.0, 80.0, 80.0, 160.0])
        for name, value in (("dxt", 3125.0), ("dxu", 3125.0), ("dyt", 1e5), ("dyu", 1e5), ("cost", 1.0), ("cosu", 1.0)):
            setattr(vs, name, npx.full_like(getattr(vs, name), value))
        for grid in ("T", "U", "V"):
            mask = update(getattr(vs, "mask" + grid), at[2:-2, 2:-2], True)
            mask = update(mask, at[3, 2, :2], False)
            if grid == "V":
                mask = update(mask, at[:, 3], False)
            setattr(vs, "mask" + grid, mask)
            if partial:
                fraction = npx.where(mask, 0.7, 0.0)
                setattr(vs, "hfac" + grid, fraction)
        profile = npx.asarray([-3.4, -2.2, 0.0004, 2.2330666666666668])
        vs.u = update(vs.u, at[...], profile[None, None, :, None] * vs.maskU[..., None])
        pressure = 1e-6 * npx.sin(npx.arange(vs.psi.shape[0])[:, None]) * npx.ones_like(vs.psi[..., 0])
        vs.psi = update(vs.psi, at[..., vs.taup1], pressure)
    return state


@pytest.mark.parametrize("partial", (False, True))
def test_native_pressure_update_preserves_consumed_transport_and_closed_faces(partial):
    from veros.core.external.solve_pressure import barotropic_velocity_update, prepare_forcing
    from veros.core.geometry import cell_thickness
    from veros.core.operators import at, numpy as npx, update
    from veros.routines import tracing_context

    state = pressure_state(partial)
    vs = state.variables
    with tracing_context():
        _, (_, uloc, vloc) = prepare_forcing(state)
    with vs.unlock():
        pressure = 1e-6 * npx.sin(npx.arange(vs.psi.shape[0])[:, None]) * npx.ones_like(vs.psi[..., 0])
        vs.psi = update(vs.psi, at[..., vs.taup1], pressure)
    with tracing_context():
        result = barotropic_velocity_update(state, uloc, vloc)
    psi = np.asarray(vs.psi[..., vs.taup1], dtype=np.longdouble)
    for name, grid, forcing, gradient in (
        ("u", "U", uloc, (psi[3:-1, 2:-2] - psi[2:-2, 2:-2]) / np.longdouble(3125)),
        ("v", "V", vloc, (psi[2:-2, 3:-1] - psi[2:-2, 2:-2]) / np.longdouble(1e5)),
    ):
        mask = np.asarray(getattr(vs, "mask" + grid)[2:-2, 2:-2])
        weight = np.asarray(cell_thickness(state, grid)[2:-2, 2:-2]) * mask
        value = np.asarray(getattr(result, name)[2:-2, 2:-2, :, vs.taup1])
        target = np.longdouble(state.settings.dt_mom) * (
            np.asarray(forcing[2:-2, 2:-2], dtype=np.longdouble)
            - np.sum(weight.astype(np.longdouble), axis=-1) * gradient
        )
        np.testing.assert_array_equal(value[~mask], 0)
        for velocity, thickness, wanted in zip(value.reshape(-1, 4), weight.reshape(-1, 4), target.ravel()):
            actual = exact_transport(velocity, thickness)
            wet = thickness > 0
            smallest = np.min(np.abs(velocity[wet]) * thickness[wet]) if np.any(wet) else 0
            bound = 8 * np.finfo(value.dtype).eps * max(abs(wanted), smallest, 1e-30)
            assert abs(np.longdouble(str(actual)) - wanted) <= bound


def test_transport_correction_derivatives_match_the_column_constraint():
    from veros import runtime_settings

    if runtime_settings.backend != "jax":
        pytest.skip("requires the JAX backend")
    import jax
    import jax.numpy as jnp
    from veros.core.transport import preserve_transport

    value = jnp.asarray([[-3.4, -2.2, 0.0004, 2.2330666666666668], [0.0, 0.0, 0.0, 0.0]])
    weight = jnp.asarray([[40.0, 60.0, 80.0, 120.0], [0.0, 0.0, 0.0, 0.0]])
    direction = jnp.asarray([[0.2, -0.1, 0.03, 0.05], [0.0, 0.0, 0.0, 0.0]])
    target = jnp.asarray([0.032, 0.0])
    target_direction = jnp.asarray([0.2, 0.0])

    def function(u, t):
        return preserve_transport(u, weight, t)

    _, tangent = jax.jit(lambda: jax.jvp(function, (value, target), (direction, target_direction)))()
    expected = np.asarray(direction).copy()
    expected[0, 2] = (0.2 - (0.2 * 40 - 0.1 * 60 + 0.05 * 120)) / 80
    np.testing.assert_allclose(tangent, expected, rtol=1e-10, atol=1e-12)
    seed = jnp.cos(jnp.arange(value.size).reshape(value.shape))
    cotangents = jax.jit(lambda: jax.vjp(function, value, target)[1](seed))()
    assert all(np.isfinite(v).all() for v in cotangents)
    left = jnp.sum(cotangents[0] * direction) + jnp.sum(cotangents[1] * target_direction)
    np.testing.assert_allclose(left, jnp.sum(seed * tangent), rtol=1e-10, atol=1e-12)
