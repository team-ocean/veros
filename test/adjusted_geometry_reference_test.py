"""Adjusted geometry fields and invariant errors use independent native grids."""

from types import SimpleNamespace
import numpy as np
import pytest


def owner(*, annual=False):
    import jax.numpy as jnp
    from veros import veros_routine
    from benchmarks.ad_demos.adjusted_basin import AdjustedBasinSetup

    timestep = 86400.0 if annual else 600.0
    forcing_period = 360 * 86400.0 if annual else 2400.0
    relaxation = 5 * 86400.0 if annual else 3000.0

    class Damped(AdjustedBasinSetup):
        @veros_routine
        def set_coriolis(self, state):
            state.variables.coriolis_t = jnp.full_like(state.variables.coriolis_t, 1e-5 * 600 / timestep)

        @veros_routine(supports_compilation=True)
        def rebuild_vertical_forcing(self, state):
            vs = state.variables
            vs.basin_temperature_target = (10 - 1e-6 * vs.zt[None, None, :] ** 2) * vs.maskT
            vs.basin_salinity_target = 35 * vs.maskT

        @veros_routine(supports_compilation=True)
        def rebuild_wet_forcing(self, state):
            self.rebuild_vertical_forcing(state)

        @veros_routine(supports_compilation=True)
        def set_forcing(self, state):
            vs = state.variables
            target = vs.basin_temperature_target + 0.05 * jnp.cos(2 * jnp.pi * vs.time / forcing_period)
            vs.temp_source = (target - vs.temp[..., vs.tau]) * vs.maskT / relaxation
            vs.salt_source = (vs.basin_salinity_target - vs.salt[..., vs.tau]) * vs.maskT / relaxation
            for name in ("surface_taux", "surface_tauy", "forc_temp_surface", "forc_salt_surface"):
                setattr(vs, name, jnp.zeros_like(getattr(vs, name)))

    sim = Damped(
        override=dict(
            nx=4,
            ny=4,
            nz=2,
            dt_tracer=timestep,
            dt_mom=timestep,
            enable_partial_cells=True,
            enable_conserve_energy=False,
            enable_neutral_diffusion=False,
            enable_skew_diffusion=False,
            eq_of_state_type=1,
            r_ray=0.2 / timestep,
            K_h=1e5 * 600 / timestep,
            A_h=1e5 * 600 / timestep,
            kappaH_0=100.0,
            kappaM_0=100.0,
        )
    )
    sim.setup()
    prepared = sim.prepare_integrator(record_diagnostics=(), checkpoint=True)
    initial = prepared.with_bathymetry(prepared.pack(sim.state), jnp.full((4, 4), 1900.0))
    return SimpleNamespace(
        sim=sim,
        prepared=prepared,
        initial=initial,
        config=dict(setup="adjusted_basin", mode="smoke", ram_checkpoints=4),
    )


@pytest.mark.parametrize("kind", ("equilibrium", "seasonal", "statistics"))
def test_adjusted_geometry_error_matches_jvp_vjp_and_finite_differences(kind):
    import jax
    import jax.numpy as jnp
    from benchmarks.ad_demos.geometry_reference import RefinedGeometry
    from benchmarks.ad_demos.adjusted_geometry_reference import AdjustedGeometryError

    ocean = owner()
    reference = RefinedGeometry(ocean, 2)
    engine = AdjustedGeometryError(reference, ocean.initial, {}, kind, 4, period_steps=4)
    if kind == "equilibrium":
        from benchmarks.ad_demos.geometry_reference import evaluate

        ocean._geometry_references = {2: reference}
        ocean._geometry_errors = {}
        ocean.sim.build_grid_reference = lambda carry, **kwargs: evaluate(ocean, carry, **kwargs)

    def point(z):
        carry = ocean.prepared.with_vertical_grid(
            ocean.initial, widths=ocean.initial.variables.dzt + z * jnp.asarray([40.0, -40.0])
        )
        return engine.codec.pack((carry, {}))

    def objective(z):
        if kind == "equilibrium":
            carry, _ = engine.codec.unpack(point(z))
            return ocean.prepared.grid_reference_error(carry, refinement=2, response=kind, steps=4)["error"]
        return engine.values(point(z))[0]

    z = jnp.asarray(0.3)
    values = engine.values(point(z))
    assert bool(jnp.isfinite(values).all()) and float(values[0]) > 0
    assert float(values[1]) < 1e-10 and float(values[2]) < 1e-10
    assert float(values[8]) > 1e-8
    np.testing.assert_allclose(objective(z), values[0], rtol=1e-10, atol=1e-12)
    value, tangent = jax.jvp(objective, (z,), (jnp.ones_like(z),))
    gradient = jax.grad(objective)(z)
    np.testing.assert_allclose(value, values[0], rtol=1e-10, atol=1e-12)
    assert abs(float(tangent)) > 1e-8
    np.testing.assert_allclose(gradient, tangent, rtol=1e-5, atol=1e-8)
    for epsilon in (1e-3, 3e-4):
        difference = (objective(z + epsilon) - objective(z - epsilon)) / (2 * epsilon)
        np.testing.assert_allclose(tangent, difference, rtol=2e-3, atol=1e-8)
    np.testing.assert_allclose(jax.jit(objective)(z), value, rtol=1e-10, atol=1e-12)
    if kind == "statistics":
        for seed in (4, 5):
            forward = engine.forward(point(z), seed)
            np.testing.assert_allclose(forward[0], value, rtol=1e-3, atol=1e-5)


def test_statistical_geometry_requires_declared_invariant_method_before_solving():
    from benchmarks.ad_demos.adjusted_geometry_reference import AdjustedGeometryError

    reference = SimpleNamespace(ocean=SimpleNamespace(config=dict(setup="global_4deg")))
    with pytest.raises(NotImplementedError, match="validated invariant_periodic"):
        AdjustedGeometryError(reference, None, {}, "statistics", 30)


@pytest.mark.parametrize("kind", ("seasonal", "statistics"))
def test_annual_geometry_reference_uses_native_year_and_resolved_products(kind):
    import jax
    import jax.numpy as jnp
    from benchmarks.ad_demos.geometry_reference import RefinedGeometry, evaluate
    from benchmarks.ad_demos.research_ocean import YEAR

    ocean = owner(annual=True)
    reference = RefinedGeometry(ocean, 2)
    ocean._geometry_references = {2: reference}
    ocean._geometry_errors = {}
    ocean.sim.build_grid_reference = lambda carry, **kwargs: evaluate(ocean, carry, **kwargs)

    def objective(z, forward=False, seed=0):
        carry = ocean.prepared.with_vertical_grid(
            ocean.initial, widths=ocean.initial.variables.dzt + z * jnp.asarray([40.0, -40.0])
        )
        return ocean.prepared.grid_reference_error(
            carry, refinement=2, response=kind, steps=360, forward=forward, seed=seed
        )["error"]

    point = jnp.asarray(0.3)
    value = objective(point)
    (engine,) = ocean._geometry_errors.values()
    assert engine.period_steps == 360
    assert engine.period_steps * ocean.prepared._template.settings.dt_tracer == YEAR
    assert engine.phases == tuple(index / 12 for index in range(12))
    assert np.isfinite(value) and value > 0
    jax.clear_caches()
    _, tangent = jax.jvp(objective, (point,), (jnp.ones_like(point),))
    jax.clear_caches()
    gradient = jax.grad(objective)(point)
    assert np.isfinite(tangent) and abs(tangent) > 1e-8
    np.testing.assert_allclose(gradient, tangent, rtol=1e-5, atol=1e-8)
    jax.clear_caches()
    for epsilon in (1e-3, 3e-4):
        difference = (objective(point + epsilon) - objective(point - epsilon)) / (2 * epsilon)
        np.testing.assert_allclose(tangent, difference, rtol=2e-3, atol=1e-8)
    jax.clear_caches()
    np.testing.assert_allclose(jax.jit(objective)(point), value, rtol=1e-10, atol=1e-12)
    if kind == "statistics":
        for seed in (4, 5):
            np.testing.assert_allclose(objective(point, True, seed), value, rtol=1e-3, atol=1e-5)
