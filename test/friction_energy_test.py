"""Momentum work against native W-grid friction diagnostics."""

from copy import copy

import numpy as np
import pytest

CASES = (
    ("rayleigh_friction", "K_diss_bot", False),
    ("linear_bottom_friction", "K_diss_bot", False),
    ("linear_bottom_friction_variable", "K_diss_bot", False),
    ("quadratic_bottom_friction", "K_diss_bot", False),
    ("explicit_vert_friction", "K_diss_v", False),
    ("implicit_vert_friction", "K_diss_v", False),
    ("harmonic_friction", "K_diss_h", False),
    ("harmonic_friction", "K_diss_h", True),
)


def friction_state(partial, noslip, *, variable=False, sources=False):
    import jax.numpy as jnp
    from bathymetry_helpers import make_simulation
    from veros import runtime_settings

    object.__setattr__(runtime_settings, "diskless_mode", True)
    object.__setattr__(runtime_settings, "linear_solver", "scipy_jax")
    bottom = np.tile(np.array([0, 1, 2, 3, 4]), (8, 2))
    simulation = make_simulation(
        partial_cells=partial,
        bottom_indices=bottom,
        wind=0,
        enable_bottom_friction_var=variable,
        enable_momentum_sources=sources,
    )
    prepared = simulation.prepare_integrator(record_diagnostics=())
    carry = prepared.pack(simulation.state)
    if partial:
        depths = carry.variables.ht[2:-2, 2:-2]
        pattern = jnp.cos(jnp.arange(depths.size).reshape(depths.shape))
        carry = prepared.with_bathymetry(carry, jnp.where(depths > 0, depths - 30 - 10 * pattern, 0))
        prepared.check_status(carry)
    state = prepared._make_state(carry.variables)
    state._settings = copy(state.settings)
    with state.settings.unlock():
        state.settings.enable_conserve_energy = True
        state.settings.enable_noslip_lateral = noslip
        state.settings.enable_hor_friction_cos_scaling = True
        state.settings.A_h = 100
        state.settings.r_ray = 1e-5
        state.settings.r_bot = 2e-5
        state.settings.r_quad_bot = 1e-3
    rng = np.random.default_rng(68)
    with state.variables.unlock():
        vs = state.variables
        vs.cost = jnp.linspace(0.8, 1, vs.cost.size)
        vs.cosu = jnp.linspace(0.81, 1.01, vs.cosu.size)
        vs.area_t = vs.dxt[:, None] * vs.dyt[None, :] * vs.cost[None, :]
        vs.area_u = vs.dxu[:, None] * vs.dyt[None, :] * vs.cost[None, :]
        vs.area_v = vs.dxt[:, None] * vs.dyu[None, :] * vs.cosu[None, :]
        for name, mask in (("u", vs.maskU), ("v", vs.maskV)):
            field = 0.01 * jnp.asarray(rng.normal(size=mask.shape)) * mask
            setattr(vs, name, jnp.broadcast_to(field[..., None], getattr(vs, name).shape))
        vs.kappaM = jnp.full_like(vs.kappaM, 1e-3)
        if sources:
            vs.u_source = -2e-6 * vs.u[..., vs.tau]
            vs.v_source = -3e-6 * vs.v[..., vs.tau]
        if variable:
            vs.r_bot_var_u = 1e-5 * (1 + 0.1 * jnp.sin(jnp.arange(vs.r_bot_var_u.size).reshape(vs.r_bot_var_u.shape)))
            vs.r_bot_var_v = 1.4e-5 * (1 + 0.2 * jnp.cos(jnp.arange(vs.r_bot_var_v.size).reshape(vs.r_bot_var_v.shape)))
    return prepared, state


@pytest.mark.parametrize("partial", (False, True))
@pytest.mark.parametrize("kernel,diagnostic,noslip", CASES)
def test_friction_work_matches_energy_diagnostic(partial, kernel, diagnostic, noslip):
    import jax.numpy as jnp
    from veros.core import friction
    from veros.routines import tracing_context

    variable = kernel.endswith("_variable")
    _, state = friction_state(partial, noslip, variable=variable)
    kernel = kernel.removesuffix("_variable")
    vs = state.variables
    with tracing_context():
        result = getattr(friction, kernel)(state)
    work = jnp.asarray(0.0)
    for name, grid, tendency in (("u", "U", result.du_mix), ("v", "V", result.dv_mix)):
        fraction = getattr(vs, "hfac" + grid) if partial else getattr(vs, "mask" + grid)
        volume = getattr(vs, "area_" + grid.lower())[..., None] * vs.dzt * fraction
        work -= jnp.sum((getattr(vs, name)[..., vs.tau] * tendency * volume)[2:-2, 2:-2])
    weights = (vs.area_t[..., None] * vs.dzw * vs.maskW).at[:, :, -1].multiply(0.5)
    dissipated = jnp.sum((getattr(result, diagnostic) * weights)[2:-2, 2:-2])
    assert float(work) > 0
    np.testing.assert_allclose(dissipated, work, rtol=1e-10, atol=1e-10)
    if diagnostic in ("K_diss_bot", "K_diss_h"):
        assert float(jnp.min(getattr(result, diagnostic))) >= 0


@pytest.mark.parametrize("partial", (False, True))
@pytest.mark.parametrize("sign", (-1, 1))
def test_momentum_source_work_is_signed_and_uses_occupied_volume(partial, sign):
    import jax.numpy as jnp
    from veros.core import friction
    from veros.routines import tracing_context

    _, state = friction_state(partial, False, sources=True)
    vs = state.variables
    with vs.unlock():
        vs.u_source *= sign
        vs.v_source *= sign
    with tracing_context():
        result = friction.momentum_sources(state)
    work = jnp.asarray(0.0)
    for name, grid, tendency in (("u", "U", result.du_mix), ("v", "V", result.dv_mix)):
        fraction = getattr(vs, "hfac" + grid) if partial else getattr(vs, "mask" + grid)
        volume = getattr(vs, "area_" + grid.lower())[..., None] * vs.dzt * fraction
        work -= jnp.sum((getattr(vs, name)[..., vs.tau] * tendency * volume)[2:-2, 2:-2])
    weights = (vs.area_t[..., None] * vs.dzw * vs.maskW).at[:, :, -1].multiply(0.5)
    dissipated = jnp.sum((result.K_diss_bot * weights)[2:-2, 2:-2])
    assert float(sign * work) > 0
    np.testing.assert_allclose(dissipated, work, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("partial", (False, True))
def test_energy_diagnostic_uses_native_velocity_and_occupied_tracer_volumes(partial):
    import jax.numpy as jnp
    from veros.core import friction
    from veros.diagnostics.energy import diagnose_kernel
    from veros.routines import tracing_context

    _, state = friction_state(partial, False, sources=True)
    with state.variables.unlock(), tracing_context():
        state.variables.update(friction.momentum_sources(state))
        result = diagnose_kernel(state)
    vs = state.variables
    kinetic, work = jnp.asarray(0.0), jnp.asarray(0.0)
    for name, grid, tendency in (("u", "U", vs.du_mix), ("v", "V", vs.dv_mix)):
        fraction = getattr(vs, "hfac" + grid) if partial else getattr(vs, "mask" + grid)
        volume = getattr(vs, "area_" + grid.lower())[..., None] * vs.dzt * fraction
        velocity = getattr(vs, name)[..., vs.tau]
        kinetic += 0.5 * jnp.sum((velocity**2 * volume)[2:-2, 2:-2])
        work += jnp.sum((velocity * tendency * volume)[2:-2, 2:-2])
    tracer_volume = vs.area_t[..., None] * vs.dzt * (vs.hfacT if partial else vs.maskT)
    enthalpy = jnp.sum((vs.Hd[..., vs.tau] * tracer_volume)[2:-2, 2:-2])
    np.testing.assert_allclose(result.k_m, kinetic, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(result.Hd_m, enthalpy, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(result.dk_m, work, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(result.ke_diss_m, -work, rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize(
    "kernel,diagnostic",
    (
        ("rayleigh_friction", "K_diss_bot"),
        ("quadratic_bottom_friction", "K_diss_bot"),
        ("harmonic_friction", "K_diss_h"),
        ("implicit_vert_friction", "K_diss_v"),
        ("momentum_sources", "K_diss_bot"),
    ),
)
def test_partial_friction_work_products_and_finite_differences(kernel, diagnostic):
    import jax
    import jax.numpy as jnp
    from veros.core import friction
    from veros.routines import tracing_context

    prepared, base_state = friction_state(True, kernel == "harmonic_friction", sources=kernel == "momentum_sources")
    initial = prepared.pack(base_state)
    depths = initial.variables.ht[2:-2, 2:-2]
    pattern = jnp.sin(jnp.arange(depths.size).reshape(depths.shape)) * (depths > 0)

    def function(z):
        moved = prepared.with_bathymetry(initial, depths + z * pattern)
        moved = prepared.replace_variables(
            moved, u=initial.variables.u * (1 + 0.1 * z), v=initial.variables.v * (1 - 0.05 * z)
        )
        state = prepared._make_state(moved.variables)
        state._settings = copy(base_state.settings)
        vs = state.variables
        with tracing_context():
            result = getattr(friction, kernel)(state)
        work = jnp.asarray(0.0)
        for name, grid, tendency in (("u", "U", result.du_mix), ("v", "V", result.dv_mix)):
            volume = getattr(vs, "area_" + grid.lower())[..., None] * vs.dzt * getattr(vs, "hfac" + grid)
            work -= jnp.sum((getattr(vs, name)[..., vs.tau] * tendency * volume)[2:-2, 2:-2])
        weights = (vs.area_t[..., None] * vs.dzw * vs.maskW).at[:, :, -1].multiply(0.5)
        dissipated = jnp.sum((getattr(result, diagnostic) * weights)[2:-2, 2:-2])
        return jnp.stack((dissipated, work))

    point, direction = jnp.asarray(0.1), jnp.asarray(0.3)
    value, tangent = jax.jvp(jax.jit(function), (point,), (direction,))
    np.testing.assert_allclose(value[0], value[1], rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(tangent[0], tangent[1], rtol=1e-10, atol=1e-9)
    assert float(jnp.linalg.norm(tangent)) > 1e-8
    seed = jnp.array([0.7, -0.2])
    transpose = jax.vjp(function, point)[1](seed)[0]
    np.testing.assert_allclose(seed @ tangent, transpose * direction, rtol=1e-10, atol=1e-9)
    for epsilon in (1e-3, 1e-4):
        difference = (function(point + epsilon * direction) - function(point - epsilon * direction)) / (2 * epsilon)
        np.testing.assert_allclose(difference, tangent, rtol=1e-6, atol=1e-8)
