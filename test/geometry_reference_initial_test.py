"""Independent refined geometry initialization preserves physical inventories."""

from types import SimpleNamespace
import numpy as np
import pytest


def owner():
    import jax.numpy as jnp
    from bathymetry_helpers import make_simulation
    from veros import veros_routine

    bottom = np.ones((8, 10), int)
    bottom[3, 4] = 0
    sim = make_simulation(bottom_indices=bottom, cyclic_x=True, wind=0)

    @veros_routine(supports_compilation=True)
    def wet_forcing(state):
        pass

    sim.rebuild_wet_forcing = wet_forcing
    prepared = sim.prepare_integrator(record_diagnostics=(), checkpoint=True)
    initial = prepared.with_bathymetry(prepared.pack(sim.state), jnp.where(jnp.asarray(bottom) > 0, 365.0, 0.0))
    v = initial.variables
    for name, grid in (("u", "U"), ("v", "V"), ("du", "U"), ("dv", "V")):
        values = (0.02 if name in ("u", "v") else 1e-7) * jnp.broadcast_to(
            getattr(v, "mask" + grid)[..., None], getattr(v, name).shape
        )
        initial = prepared.replace_variables(initial, **{name: values})
    return SimpleNamespace(sim=sim, prepared=prepared, initial=initial, config=dict(setup="native_basin"))


def test_refined_geometry_reconstructs_depths_new_wet_cells_and_face_transports():
    import jax
    import jax.numpy as jnp
    from benchmarks.ad_demos.geometry_reference import RefinedGeometry
    from benchmarks.ad_demos.grid_reference import (
        physical_horizontal_measures,
        project_velocity,
        horizontal_alignment_residual,
    )

    ocean = owner()
    reference = RefinedGeometry(ocean, 2)
    initial = ocean.initial
    channel = jnp.zeros((8, 10), bool).at[3, 4].set(True)
    initial = ocean.prepared.with_wet_geometry(initial, opening_m=75.0, channel=channel)
    coarse = initial.variables
    fine = jax.jit(reference.initial)(initial)
    reference.prepared.check_status(fine)
    refined = fine.variables
    assert (
        float(
            horizontal_alignment_residual(
                coarse, refined, refinement=2, coord_degree=False, metres_per_degree=ocean.sim.state.settings.degtom
            )
        )
        < 1e-13
    )
    depth = np.asarray(refined.ht[2:-2, 2:-2]).reshape(8, 2, 10, 2)
    np.testing.assert_allclose(
        depth, np.broadcast_to(np.asarray(coarse.ht[2:-2, 2:-2])[:, None, :, None], depth.shape), atol=1e-12
    )
    assert np.all(np.asarray(refined.maskT)[2 + 6 : 2 + 8, 2 + 8 : 2 + 10, -1])
    a = physical_horizontal_measures(coarse, ocean.sim.state.settings)
    b = physical_horizontal_measures(refined, reference.simulation.state.settings)
    for name in ("temp", "salt", "dtemp", "dsalt", "dHd"):
        old = np.sum(
            np.asarray(getattr(coarse, name)[2:-2, 2:-2])
            * np.asarray(
                a["area_t"][..., None, None] * coarse.dzt[None, None, :, None] * coarse.hfacT[2:-2, 2:-2, :, None]
            ),
            axis=(0, 1, 2),
        )
        new = np.sum(
            np.asarray(getattr(refined, name)[2:-2, 2:-2])
            * np.asarray(
                b["area_t"][..., None, None] * refined.dzt[None, None, :, None] * refined.hfacT[2:-2, 2:-2, :, None]
            ),
            axis=(0, 1, 2),
        )
        np.testing.assert_allclose(new, old, rtol=3e-13, atol=1e-9)
    for name, grid, normal in (("u", "U", 0), ("v", "V", 1), ("du", "U", 0), ("dv", "V", 1)):
        width = "width_u" if normal == 0 else "width_v"
        coarse_t = coarse.dzt * getattr(coarse, "hfac" + grid)[2:-2, 2:-2]
        fine_t = refined.dzt * getattr(refined, "hfac" + grid)[2:-2, 2:-2]
        result = project_velocity(
            getattr(refined, name)[2:-2, 2:-2], fine_t, b[width], coarse_t, a[width], normal_axis=normal
        )
        assert result["geometry_residual"] < 1e-12
        expected = np.sum(
            np.asarray(getattr(coarse, name)[2:-2, 2:-2]) * np.asarray(coarse_t[..., None] * a[width][..., None, None]),
            axis=2,
        )
        actual = np.sum(
            np.asarray(result["values"]) * np.asarray(coarse_t[..., None] * a[width][..., None, None]), axis=2
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)
    final, _ = reference.prepared.advance(fine, n_steps=2, final_only=True)
    reference.prepared.check_status(final)
    assert np.isfinite(final.variables.temp).all()


def test_refined_geometry_initial_grid_products_match_finite_differences():
    import jax
    import jax.numpy as jnp
    from benchmarks.ad_demos.geometry_reference import RefinedGeometry

    ocean = owner()
    reference = RefinedGeometry(ocean, 2)

    def scalar(z):
        widths = ocean.initial.variables.dzt + z * jnp.asarray([4.0, -2.0, -3.0, 1.0])
        carry = ocean.prepared.with_vertical_grid(ocean.initial, widths=widths)
        fine = reference.initial(carry)
        weights = jnp.sin(jnp.arange(fine.variables.temp[..., 0].size)).reshape(fine.variables.temp[..., 0].shape)
        return jnp.sum(fine.variables.temp[..., fine.variables.tau] * weights)

    point = jnp.asarray(0.3)
    value, tangent = jax.jvp(scalar, (point,), (jnp.asarray(1.0),))
    gradient = jax.grad(scalar)(point)
    assert np.isfinite(value) and abs(float(tangent)) > 1e-6
    np.testing.assert_allclose(tangent, gradient, rtol=1e-9, atol=1e-9)
    for epsilon in (1e-3, 1e-4):
        fd = (scalar(point + epsilon) - scalar(point - epsilon)) / (2 * epsilon)
        np.testing.assert_allclose(fd, tangent, rtol=2e-3, atol=1e-8)


def test_refined_geometry_rejects_noninteger_refinement():
    from benchmarks.ad_demos.geometry_reference import RefinedGeometry

    with pytest.raises(ValueError, match="integer refinement"):
        RefinedGeometry(None, 1.5)
