"""Nested spherical reference measures retain physical areas and native budgets."""

import numpy as np
import pytest


def grid(nx, ny, nz, width, origin, spherical):
    import jax.numpy as jnp
    from veros.state import get_default_state
    from veros.core.numerics import calc_grid

    state = get_default_state()
    with state.settings.unlock():
        state.settings.nx, state.settings.ny, state.settings.nz = nx, ny, nz
        state.settings.coord_degree = spherical
        state.settings.x_origin, state.settings.y_origin = origin
    state.initialize_variables()
    with state.variables.unlock():
        state.variables.dxt = jnp.full_like(state.variables.dxt, width)
        state.variables.dyt = jnp.full_like(state.variables.dyt, width)
        state.variables.dzt = jnp.full_like(state.variables.dzt, 400 / nz)
        calc_grid(state)
    return state


@pytest.mark.parametrize("spherical", (False, True))
def test_exact_physical_area_adds_under_refinement_without_changing_native_metrics(spherical):
    import jax.numpy as jnp
    from benchmarks.ad_demos.grid_reference import (
        physical_horizontal_measures,
        refined_horizontal_origins,
        project_tracer,
        project_velocity,
    )

    width = 4.0 if spherical else 20000.0
    coarse = grid(4, 3, 4, width, (7.5, -20.0), spherical)
    origins = refined_horizontal_origins(coarse.variables, coarse.settings, 2)
    fine = grid(8, 6, 8, width / 2, (origins["x_origin"], origins["y_origin"]), spherical)
    native_coarse = np.asarray(coarse.variables.area_t).copy()
    native_fine = np.asarray(fine.variables.area_t).copy()
    a = physical_horizontal_measures(coarse.variables, coarse.settings)
    b = physical_horizontal_measures(fine.variables, fine.settings)
    np.testing.assert_allclose(b["area_t"].reshape(4, 2, 3, 2).sum(axis=(1, 3)), a["area_t"], rtol=2e-14)
    if spherical:
        lat = np.asarray(coarse.variables.yu[2:-2])
        r = coarse.settings.radius
        exact = r * r * np.deg2rad(width) * (np.sin(np.deg2rad(lat)) - np.sin(np.deg2rad(lat - width)))
        np.testing.assert_allclose(a["area_t"], np.broadcast_to(exact, (4, 3)), rtol=1e-14)
        native_ratio = native_fine[2:-2, 2:-2].reshape(4, 2, 3, 2).sum(axis=(1, 3)) / native_coarse[2:-2, 2:-2]
        np.testing.assert_allclose(native_ratio, np.cos(np.deg2rad(width / 4)), rtol=2e-14)
    np.testing.assert_array_equal(coarse.variables.area_t, native_coarse)
    np.testing.assert_array_equal(fine.variables.area_t, native_fine)
    thickness = jnp.full((4, 3, 4), 100.0)
    refined = jnp.full((8, 6, 8), 50.0)
    projected = project_tracer(jnp.full(refined.shape, 35.0), refined, b["area_t"], thickness, a["area_t"])
    np.testing.assert_allclose(projected["values"], 35.0, rtol=2e-14)
    assert projected["geometry_residual"] < 1e-13 and projected["inventory_residual"] < 1e-13
    if spherical:
        native = project_tracer(
            jnp.full(refined.shape, 35.0),
            refined,
            fine.variables.area_t[2:-2, 2:-2],
            thickness,
            coarse.variables.area_t[2:-2, 2:-2],
        )
        assert native["geometry_residual"] > 1e-4
    for normal in (0, 1):
        key = "width_u" if normal == 0 else "width_v"
        projected = project_velocity(
            jnp.full(refined.shape, 0.2), refined, b[key], thickness, a[key], normal_axis=normal
        )
        np.testing.assert_allclose(projected["values"], 0.2, rtol=2e-14)
        assert projected["geometry_residual"] < 1e-13 and projected["inventory_residual"] < 1e-13


@pytest.mark.parametrize("refinement", (2, 3, 4))
def test_nonuniform_native_vertical_refinement_preserves_faces_and_positive_spacings(refinement):
    from types import SimpleNamespace
    import jax.numpy as jnp
    from veros.core.numerics import u_centered_grid
    from benchmarks.ad_demos.grid_reference import refined_vertical_widths
    from veros.routines import tracing_context

    original = jnp.asarray(
        [50.0, 70.0, 100.0, 140.0, 190.0, 240.0, 290.0, 340.0, 390.0, 440.0, 490.0, 540.0, 590.0, 640.0, 690.0]
    )[::-1]

    def coordinates(width):
        with tracing_context():
            dzw, zt, zw = u_centered_grid(width, jnp.zeros_like(width), jnp.zeros_like(width), jnp.zeros_like(width))
        return dzw, zt - zw[-1], zw - zw[-1]

    spacing, centre, faces = coordinates(original)
    assert np.min(spacing) > 0
    naive, _, _ = coordinates(jnp.repeat(original / refinement, refinement))
    if refinement % 2 == 0:
        assert np.min(naive) <= 0
    width = refined_vertical_widths(SimpleNamespace(dzt=original, dzw=spacing), refinement)
    fine_spacing, fine_centre, fine_faces = coordinates(width)
    assert np.min(width) > 0 and np.min(fine_spacing) > 0
    np.testing.assert_allclose(width.reshape(-1, refinement).sum(axis=-1), original, rtol=1e-14, atol=2e-12)
    np.testing.assert_allclose(fine_faces[refinement - 1 :: refinement], faces, rtol=0, atol=2e-12)
    np.testing.assert_allclose(fine_spacing[refinement - 1 :: refinement], spacing / refinement, rtol=1e-13, atol=2e-12)
    np.testing.assert_allclose(fine_centre[-1], centre[-1] / refinement, rtol=1e-13, atol=2e-12)
