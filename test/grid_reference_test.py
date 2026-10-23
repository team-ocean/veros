"""Nested geometry references must preserve inventories and face transports."""

import numpy as np
import pytest
from benchmarks.ad_demos.grid_reference import project_tracer, project_velocity

pytestmark = pytest.mark.usefixtures("jax_backend")


@pytest.mark.parametrize("spherical", (False, True))
def test_refinement_retains_physical_faces_and_rejects_a_shift_before_advancing(spherical):
    from types import SimpleNamespace
    import jax.numpy as jnp
    from veros.state import get_default_state
    from veros.core.numerics import calc_grid
    from benchmarks.ad_demos.grid_reference import (
        refined_horizontal_origins,
        horizontal_alignment_residual,
        trajectory_error,
    )

    def grid(nx, ny, nz, width, origin, dt):
        state = get_default_state()
        with state.settings.unlock():
            state.settings.nx, state.settings.ny, state.settings.nz = nx, ny, nz
            state.settings.coord_degree = spherical
            state.settings.enable_cyclic_x = False
            state.settings.x_origin, state.settings.y_origin = origin
            state.settings.dt_tracer = state.settings.dt_mom = dt
        state.initialize_variables()
        with state.variables.unlock():
            state.variables.dxt = jnp.full_like(state.variables.dxt, width)
            state.variables.dyt = jnp.full_like(state.variables.dyt, width)
            state.variables.dzt = jnp.full_like(state.variables.dzt, 100.0 / (nz / 4))
            calc_grid(state)
        return state

    width = 4.0 if spherical else 20000.0
    coarse = grid(4, 3, 4, width, (7.5, -20.0), 300.0)
    origins = refined_horizontal_origins(coarse.variables, coarse.settings, 2)
    fine = grid(8, 6, 8, width / 2, (origins["x_origin"], origins["y_origin"]), 150.0)
    error = horizontal_alignment_residual(
        coarse.variables, fine.variables, refinement=2, coord_degree=spherical, metres_per_degree=coarse.settings.degtom
    )
    assert float(error) < 1e-13
    for axis in ("x", "y"):
        # The outer physical faces, not the configured staggered origins, agree.
        parent = np.asarray(getattr(coarse.variables, axis + "u"))[2:-2]
        child = np.asarray(getattr(fine.variables, axis + "u"))[2:-2]
        np.testing.assert_allclose(child[1::2], parent, rtol=0, atol=1e-10)
    shifted = grid(8, 6, 8, width / 2, (7.5, -20.0), 150.0)
    error = horizontal_alignment_residual(
        coarse.variables,
        shifted.variables,
        refinement=2,
        coord_degree=spherical,
        metres_per_degree=coarse.settings.degtom,
    )
    np.testing.assert_allclose(error, 0.5, rtol=0, atol=1e-13)

    def expensive_step(*args, **kwargs):
        raise AssertionError("A mismatched reference must be rejected before native integration")

    a = SimpleNamespace(_template=coarse, advance=expensive_step)
    b = SimpleNamespace(_template=shifted, advance=expensive_step)
    with pytest.raises(ValueError, match="different physical domain"):
        trajectory_error(
            a,
            SimpleNamespace(variables=coarse.variables),
            b,
            SimpleNamespace(variables=shifted.variables),
            steps=2,
            refinement=2,
        )


def columns():
    import jax.numpy as jnp

    coarse = jnp.broadcast_to(jnp.array([0.0, 0.4, 1.2, 0.8]), (2, 3, 4))
    fine = jnp.broadcast_to(jnp.array([0.0, 0.0, 0.2, 0.2, 0.6, 0.6, 0.4, 0.4]), (4, 6, 8))
    return coarse, fine


def test_nested_tracer_histories_constants_inventories_and_geometry_mismatch():
    import jax
    import jax.numpy as jnp

    coarse, fine = columns()
    area = jnp.arange(1, 7, dtype=coarse.dtype).reshape(2, 3)
    fine_area = jnp.repeat(jnp.repeat(area / 4, 2, axis=0), 2, axis=1)
    values = jnp.asarray(np.random.default_rng(93).uniform(0.1, 2, fine.shape + (3,)))
    result = jax.jit(project_tracer)(values, fine, fine_area, coarse, area)
    assert result["values"].shape == coarse.shape + (3,)
    assert np.min(result["values"]) >= 0
    assert result["inventory_residual"] < 1e-14
    assert result["geometry_residual"] < 1e-14
    constant = project_tracer(jnp.ones_like(fine), fine, fine_area, coarse, area)
    np.testing.assert_allclose(constant["values"], coarse > 0, atol=1e-14)
    mismatch = project_tracer(values, fine, fine_area, coarse, 1.01 * area)
    assert mismatch["geometry_residual"] > 0.009
    mismatch = project_tracer(values, fine, fine_area, coarse * 0.99, area)
    assert mismatch["geometry_residual"] > 0.009
    assert mismatch["inventory_residual"] > 1e-4


@pytest.mark.parametrize("normal_axis", (0, 1))
def test_nested_face_transport_uses_aligned_faces_and_tangential_width(normal_axis):
    import jax
    import jax.numpy as jnp

    coarse, fine = columns()
    values = jnp.asarray(np.random.default_rng(94).normal(size=fine.shape + (3,)))
    width = jnp.ones(coarse.shape[:2]) * 2
    result = jax.jit(
        lambda v: project_velocity(v, fine, jnp.ones(fine.shape[:2]), coarse, width, normal_axis=normal_axis)
    )(values)
    assert result["inventory_residual"] < 1e-14
    assert result["geometry_residual"] < 1e-14
    flux = values * fine[..., None]
    if normal_axis == 0:
        expected = flux[1::2].reshape(2, 3, 2, 8, 3).sum(axis=(2, 3))
    else:
        expected = flux[:, 1::2].reshape(2, 2, 3, 8, 3).sum(axis=(1, 3))
    actual = jnp.sum(result["values"] * (width[..., None] * coarse)[..., None], axis=2)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    constant = project_velocity(
        jnp.ones_like(fine), fine, jnp.ones(fine.shape[:2]), coarse, width, normal_axis=normal_axis
    )
    np.testing.assert_allclose(constant["values"], coarse > 0, atol=1e-14)


@pytest.mark.parametrize("grid", ("tracer", "U", "V"))
def test_candidate_grid_projection_has_resolved_products(grid):
    import jax
    import jax.numpy as jnp

    coarse, fine = columns()
    values = jnp.asarray(np.random.default_rng(95).uniform(0.2, 2, fine.shape))
    direction = jnp.broadcast_to(jnp.array([0, 0.1, -0.15, 0.05]), coarse.shape)
    seed = jnp.cos(jnp.arange(coarse.size)).reshape(coarse.shape)

    def function(z):
        target = coarse + z * direction
        if grid == "tracer":
            return project_tracer(values, fine, jnp.ones(fine.shape[:2]), target, 4 * jnp.ones(coarse.shape[:2]))[
                "values"
            ]
        return project_velocity(
            values,
            fine,
            jnp.ones(fine.shape[:2]),
            target,
            2 * jnp.ones(coarse.shape[:2]),
            normal_axis=0 if grid == "U" else 1,
        )["values"]

    point = jnp.asarray(0.3)
    tangent = jax.jvp(function, (point,), (jnp.asarray(1.0),))[1]
    transpose = jax.vjp(function, point)[1](seed)[0]
    assert np.linalg.norm(tangent) > 1e-3
    np.testing.assert_allclose(jnp.vdot(seed, tangent), transpose, rtol=1e-10, atol=1e-12)
    for epsilon in (1e-3, 1e-4):
        difference = (function(point + epsilon) - function(point - epsilon)) / (2 * epsilon)
        np.testing.assert_allclose(tangent, difference, rtol=2e-3, atol=1e-8)


def test_native_refined_endpoint_error_differentiates_candidate_grid():
    import jax
    import jax.numpy as jnp
    from veros import veros_routine, runtime_settings
    from veros.core.numerics import calc_initial_conditions_kernel
    from veros.routines import tracing_context
    from bathymetry_helpers import make_simulation
    from benchmarks.ad_demos.grid_reference import trajectory_error, refined_horizontal_origins

    object.__setattr__(runtime_settings, "diskless_mode", True)
    object.__setattr__(runtime_settings, "linear_solver", "scipy_jax")
    simulation = make_simulation(wind=0)

    class Refined(type(simulation)):
        @veros_routine
        def set_grid(self, state):
            vs = state.variables
            vs.dxt = jnp.full_like(vs.dxt, 10000.0)
            vs.dyt = jnp.full_like(vs.dyt, 10000.0)
            vs.dzt = jnp.full_like(vs.dzt, 50.0)

    refined = Refined(
        override=dict(
            nx=16,
            ny=20,
            nz=8,
            dt_tracer=150.0,
            dt_mom=150.0,
            **refined_horizontal_origins(simulation.state.variables, simulation.state.settings, 2),
        )
    )
    refined.setup()

    @veros_routine(supports_compilation=True)
    def forcing(state):
        vs = state.variables
        vs.surface_tauy = (
            0.02
            * (1 + 0.1 * jnp.sin(2 * jnp.pi * vs.xt[:, None] / 160000 + 2 * jnp.pi * vs.yt[None, :] / 200000))
            * vs.maskV[..., -1]
        )

    def prepare(sim):
        sim.set_forcing = forcing
        prepared = sim.prepare_integrator(record_diagnostics=())
        carry = prepared.pack(sim.state)
        carry = prepared.with_bathymetry(carry, jnp.full((sim.state.settings.nx, sim.state.settings.ny), 360.0))
        vs = carry.variables
        thickness = vs.dzt * vs.hfacT
        top = -vs.ht[..., None] + jnp.cumsum(thickness, axis=2)
        bottom = top - thickness
        average = 400 * (jnp.exp(top / 400) - jnp.exp(bottom / 400)) / jnp.where(thickness > 0, thickness, 1)
        field = (
            10 + 0.002 * (top + bottom) / 2 + 0.03 * jnp.cos(2 * jnp.pi * vs.xt[:, None, None] / 160000) * average
        ) * vs.maskT
        carry = prepared.replace_variables(carry, temp=jnp.broadcast_to(field[..., None], vs.temp.shape))
        state = prepared._make_state(carry.variables)
        with tracing_context(), state.variables.unlock():
            state.variables.update(calc_initial_conditions_kernel(state))
        return prepared, carry._replace(variables=state.variables)

    coarse, initial = prepare(simulation)
    fine, fine_initial = prepare(refined)

    def result(z):
        widths = initial.variables.dzt + 10 * z * jnp.linspace(-1, 1, 4)
        candidate = coarse.with_vertical_grid(initial, widths=widths, conserve=("temp", "salt", "momentum"))
        return trajectory_error(coarse, candidate, fine, fine_initial, steps=2, refinement=2)

    point = jnp.asarray(0.3)
    current = result(point)
    assert np.isfinite(current["error"]) and current["error"] > 0
    assert current["remap_budget_residual"] < 1e-12
    assert current["projection_geometry_residual"] < 1e-12

    def function(z):
        return result(z)["error"]

    gradient = jax.grad(function)(point)
    tangent = jax.jvp(function, (point,), (jnp.asarray(1.0),))[1]
    assert abs(gradient) > 1e-8
    np.testing.assert_allclose(gradient, tangent, rtol=1e-7, atol=1e-10)
    for epsilon in (0.01, 0.003):
        difference = (function(point + epsilon) - function(point - epsilon)) / (2 * epsilon)
        np.testing.assert_allclose(gradient, difference, rtol=2e-3, atol=1e-8)
    with pytest.raises(ValueError, match="distinct"):
        trajectory_error(coarse, initial, coarse, initial, steps=2, refinement=2)
    with pytest.raises(ValueError, match="dimensions"):
        trajectory_error(coarse, initial, fine, fine_initial, steps=2, refinement=3)
