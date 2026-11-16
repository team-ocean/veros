"""Continuity retains small surface fluxes under vertical cancellation."""

from decimal import Decimal, localcontext
import numpy as np
import pytest


def as_decimal(value):
    return Decimal.from_float(float(value))


def exact_vertical_velocity(args, partial):
    u, v, dz, dx, dy, cost, cosu, mask, fu, fv = args
    out = np.zeros(u.shape, dtype=float)
    with localcontext() as ctx:
        ctx.prec = 80
        for i in range(2, u.shape[0] - 2):
            for j in range(2, u.shape[1] - 2):
                total = Decimal(0)
                for k in range(u.shape[2]):
                    ue, uw, vn, vs = (
                        as_decimal(u[i, j, k]),
                        as_decimal(u[i - 1, j, k]),
                        as_decimal(v[i, j, k]),
                        as_decimal(v[i, j - 1, k]),
                    )
                    if partial:
                        ue *= as_decimal(fu[i, j, k])
                        uw *= as_decimal(fu[i - 1, j, k])
                        vn *= as_decimal(fv[i, j, k])
                        vs *= as_decimal(fv[i, j - 1, k])
                    increment = -as_decimal(dz[k]) * (
                        (ue - uw) / (as_decimal(cost[j]) * as_decimal(dx[i]))
                        + (as_decimal(cosu[j]) * vn - as_decimal(cosu[j - 1]) * vs)
                        / (as_decimal(cost[j]) * as_decimal(dy[j]))
                    )
                    total += increment if partial else increment * as_decimal(mask[i, j, k])
                    out[i, j, k] = float(total * (as_decimal(mask[i, j, k]) if partial else Decimal(1)))
    return out


def continuity_fields(cyclic, kind, spherical):
    nx, ny, nz = 4, 3, 4
    shape = (nx + 4, ny + 4, nz)
    rng = np.random.default_rng(481)
    bottom = np.ones((nx, ny), int) if kind == "full" else np.array([[1, 2, 0], [3, 1, 2], [1, 4, 1], [2, 2, 1]])
    fraction = np.zeros(shape)
    for i in range(nx):
        for j in range(ny):
            if bottom[i, j] > 0:
                fraction[i + 2, j + 2, bottom[i, j] - 1 :] = 1
                if kind == "partial":
                    fraction[i + 2, j + 2, bottom[i, j] - 1] = 0.31 + 0.1 * ((i + j) % 4)

    def fill_periodic_halos(a):
        if cyclic:
            a[:2] = a[-4:-2]
            a[-2:] = a[2:4]
        return a

    fraction = fill_periodic_halos(fraction)
    mask = (fraction > 0).astype(float)
    fu = np.zeros(shape)
    fv = np.zeros(shape)
    fu[:-1] = np.minimum(fraction[:-1], fraction[1:])
    fv[:, :-1] = np.minimum(fraction[:, :-1], fraction[:, 1:])
    profile = np.array([-3.4, -2.2, 0.0004, 2.2330666666666668])
    u = (1 + 0.2 * np.cos(np.arange(shape[0])[:, None, None])) * profile + 1e-8 * rng.normal(size=shape)
    v = (0.3 + 0.1 * np.sin(np.arange(shape[1])[None, :, None])) * profile + 1e-8 * rng.normal(size=shape)
    u = fill_periodic_halos(u) * (fu > 0)
    v = fill_periodic_halos(v) * (fv > 0)
    du = fill_periodic_halos(rng.normal(size=shape)) * (fu > 0)
    dv = fill_periodic_halos(rng.normal(size=shape)) * (fv > 0)
    dz = np.array([40.0, 60.0, 80.0, 120.0])
    dx = np.array([2900.0, 5300.0, 3125.0, 3847.0, 2900.0, 5300.0, 3125.0, 3847.0])
    dy = np.array([90000.0, 100000.0, 100000.0, 110000.0, 90000.0, 90000.0, 90000.0])
    lat = np.deg2rad(np.arange(shape[1]) * 11 - 33)
    cost = np.cos(lat) if spherical else np.ones(shape[1])
    cosu = np.cos(lat + 0.5 * np.deg2rad(11)) if spherical else np.ones(shape[1])
    return (u, v, dz, dx, dy, cost, cosu, mask, fu, fv), (du, dv)


CASES = [
    (False, "full", False),
    (True, "full", False),
    (False, "stepped", False),
    (True, "stepped", False),
    (True, "full", True),
    (False, "stepped", True),
    (True, "partial", False),
    (False, "partial", True),
]


@pytest.fixture
def dtype(request):
    from veros import runtime_settings

    if runtime_settings.backend == "jax":
        import jax

        jax.config.update("jax_enable_x64", True)
    previous = runtime_settings.float_type
    object.__setattr__(runtime_settings, "float_type", request.param)
    try:
        yield request.param
    finally:
        object.__setattr__(runtime_settings, "float_type", previous)


@pytest.mark.parametrize("cyclic,kind,spherical", CASES)
@pytest.mark.parametrize("dtype", ("float32", "float64"), indirect=True)
def test_native_continuity_matches_exact_equation_and_closed_volume(cyclic, kind, spherical, dtype):
    from veros.core.momentum import vertical_velocity_kernel
    from veros.core.operators import numpy as npx
    from veros.routines import tracing_context
    from veros.state import get_default_state

    state = get_default_state()
    with state.settings.unlock():
        state.settings.nx, state.settings.ny, state.settings.nz = 4, 3, 4
        state.settings.enable_partial_cells = kind == "partial"
        state.settings.enable_cyclic_x = cyclic
        state.settings.coord_degree = spherical
    state.initialize_variables()
    arrays, _ = continuity_fields(cyclic, kind, spherical)
    arrays = tuple(value.astype(dtype) for value in arrays)
    u, v = arrays[:2]
    dx, dy, cost = arrays[3:6]
    vs = state.variables
    with vs.unlock():
        vs.u = npx.asarray(np.repeat(u[..., None], 3, axis=-1))
        vs.v = npx.asarray(np.repeat(v[..., None], 3, axis=-1))
        vs.w = npx.full_like(vs.u, 123)
        for name, value in zip(("dzt", "dxt", "dyt", "cost", "cosu", "maskW", "hfacU", "hfacV"), arrays[2:]):
            if name in ("hfacU", "hfacV") and kind != "partial":
                continue
            setattr(vs, name, npx.asarray(value))
    with tracing_context():
        result = vertical_velocity_kernel(state)
    actual = np.asarray(result.w)
    tau = int(vs.taup1)
    physical = (slice(2, -2), slice(2, -2))
    expected = exact_vertical_velocity(arrays, kind == "partial")[physical]
    value = actual[physical][..., tau]
    assert value.dtype == np.dtype(dtype)
    eps = np.finfo(value.dtype).eps
    assert np.max(abs(value.astype(np.longdouble) - expected)) <= 500 * eps * np.max(abs(expected))
    surface = expected[..., -1]
    assert np.max(abs(value[..., -1].astype(np.longdouble) - surface)) <= 500 * eps * np.max(abs(surface))
    area = dx[2:-2, None].astype(np.longdouble) * dy[None, 2:-2] * cost[None, 2:-2]
    flux = value[..., -1].astype(np.longdouble) * area
    assert abs(np.sum(flux)) <= 500 * eps * np.sum(abs(flux))
    np.testing.assert_array_equal(actual[..., (tau + 1) % 3], 123)
    np.testing.assert_array_equal(actual[..., (tau + 2) % 3], 123)
    np.testing.assert_array_equal(actual[0, :, :, tau], 123)
    np.testing.assert_array_equal(actual[:, 0, :, tau], 123)


@pytest.mark.parametrize("cyclic,kind,spherical", CASES)
def test_continuity_velocity_derivatives(cyclic, kind, spherical):
    from veros import runtime_settings

    if runtime_settings.backend != "jax":
        pytest.skip("requires the JAX backend")
    import jax
    import jax.numpy as jnp
    from veros.core.transport import integrate_continuity

    arrays, direction = continuity_fields(cyclic, kind, spherical)
    u, v = map(jnp.asarray, arrays[:2])
    du, dv = map(jnp.asarray, direction)
    geometry = tuple(map(jnp.asarray, arrays[2:8]))

    def function(u, v):
        return integrate_continuity(
            u,
            v,
            *geometry,
            jnp.asarray(arrays[8]) if kind == "partial" else None,
            jnp.asarray(arrays[9]) if kind == "partial" else None,
        )

    _, tangent = jax.jit(lambda: jax.jvp(function, (u, v), (du, dv)))()
    expected = exact_vertical_velocity((direction[0], direction[1], *arrays[2:]), kind == "partial")[1:, 1:]
    np.testing.assert_allclose(tangent[1:-2, 1:-2], expected[1:-2, 1:-2], rtol=1e-10, atol=1e-12)
    seed = jnp.cos(jnp.arange(tangent.size).reshape(tangent.shape))
    gu, gv = jax.jit(lambda: jax.vjp(function, u, v)[1](seed))()
    assert np.isfinite(gu).all() and np.isfinite(gv).all()
    np.testing.assert_allclose(jnp.sum(gu * du) + jnp.sum(gv * dv), jnp.sum(seed * tangent), rtol=1e-10, atol=1e-12)
