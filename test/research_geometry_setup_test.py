"""Named research channels extend forcing from fixed physical ocean donors."""

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("jax_backend")


@pytest.fixture(scope="module", params=(False, True))
def coastline(request):
    import jax.numpy as jnp
    from bathymetry_helpers import make_simulation
    from veros import veros_routine
    from veros.variables import Variable
    from benchmarks.ad_demos.geometry_setup import (
        add_reference_variables,
        initialize_reference,
        extend_forcing,
        MONTHLY_FIELDS,
    )

    bottom = np.ones((8, 10), dtype=int)
    bottom[3, 2] = 0
    original = make_simulation(bottom_indices=bottom, cyclic_x=True, wind=0)
    spherical = request.param

    class Coastline(type(original)):
        @veros_routine
        def set_parameter(self, state):
            super().set_parameter(state)
            state.settings.coord_degree = spherical
            state.settings.x_origin, state.settings.y_origin = -92.0, 4.0
            state.dimensions["nmonths"] = 12
            add_reference_variables(state)
            for name in MONTHLY_FIELDS:
                state.var_meta[name] = Variable(name, ("xt", "yt", "nmonths"), time_dependent=False)
            state.var_meta["geometry_test_budget"] = Variable("budget", None, initial=7.0)

        @veros_routine
        def set_grid(self, state):
            super().set_grid(state)
            if spherical:
                state.variables.dxt = jnp.full_like(state.variables.dxt, 4.0)
                state.variables.dyt = jnp.full_like(state.variables.dyt, 4.0)

        @veros_routine
        def set_initial_conditions(self, state):
            super().set_initial_conditions(state)
            vs = state.variables
            x, y = jnp.indices(vs.maskT.shape[:2])
            for index, name in enumerate(MONTHLY_FIELDS):
                setattr(
                    vs,
                    name,
                    (1000 * index + 100 * x[..., None] + 10 * y[..., None] + jnp.arange(12)) * vs.maskT[..., -1, None],
                )
            initialize_reference(state)

        @veros_routine(supports_compilation=True)
        def rebuild_wet_forcing(self, state):
            extend_forcing(state)

        def wet_geometry_channel(self, state, name):
            from benchmarks.ad_demos.geometry_setup import channel_mask

            return channel_mask(state, "global_4deg", name)

        wet_geometry_channel.supports_compilation = True

    simulation = Coastline()
    simulation.setup()
    prepared = simulation.prepare_integrator(record_diagnostics=())
    return simulation, prepared, prepared.pack(simulation.state)


def test_nearest_donors_are_reference_interior_ocean_cells(coastline):
    simulation, _, initial = coastline
    vs = initial.variables
    x, y = map(np.asarray, (vs.geometry_forcing_x, vs.geometry_forcing_y))
    wet = np.asarray(vs.geometry_reference_wet)
    dry = ~wet
    assert np.all(wet[x[dry], y[dry]])
    assert np.all((x[dry] >= 2) & (x[dry] < wet.shape[0] - 2))
    assert np.all((y[dry] >= 2) & (y[dry] < wet.shape[1] - 2))
    ocean = np.argwhere(wet[2:-2, 2:-2]) + 2
    i, j = 5, 4
    lon, lat = np.asarray(vs.xt), np.asarray(vs.yt)
    if simulation.state.settings.coord_degree:

        def distance(a, b):
            phi, phi0 = np.deg2rad(lat[b]), np.deg2rad(lat[j])
            return 2 * np.arcsin(
                np.sqrt(
                    np.sin((phi - phi0) / 2) ** 2
                    + np.cos(phi) * np.cos(phi0) * np.sin(np.deg2rad(lon[a] - lon[i]) / 2) ** 2
                )
            )

    else:

        def distance(a, b):
            return np.hypot(lon[a] - lon[i], lat[b] - lat[j])

    assert distance(x[i, j], y[i, j]) <= np.min(distance(ocean[:, 0], ocean[:, 1])) + 1e-10


def test_new_wet_monthly_fields_keep_phase_and_old_values_and_budgets(coastline):
    import jax
    import jax.numpy as jnp
    from benchmarks.ad_demos.geometry_setup import MONTHLY_FIELDS

    _, prepared, initial = coastline
    mask = jnp.zeros((8, 10), bool).at[3, 2].set(True)
    result = jax.jit(lambda depth: prepared.with_wet_geometry(initial, opening_m=depth, channel=mask))(
        jnp.asarray(60.0)
    )
    prepared.check_status(result)
    old, new = initial.variables, result.variables
    x, y = int(old.geometry_forcing_x[5, 4]), int(old.geometry_forcing_y[5, 4])
    original_wet = np.asarray(old.geometry_reference_wet)
    for name in MONTHLY_FIELDS:
        np.testing.assert_array_equal(getattr(new, name)[5, 4], getattr(old, name)[x, y])
        np.testing.assert_array_equal(getattr(new, name)[original_wet], getattr(old, name)[original_wet])
        assert np.all(np.diff(getattr(new, name)[5, 4]) == 1)
    np.testing.assert_array_equal(new.geometry_test_budget, old.geometry_test_budget)
    for name in ("geometry_reference_wet", "geometry_forcing_x", "geometry_forcing_y"):
        np.testing.assert_array_equal(getattr(new, name), getattr(old, name))


@pytest.mark.parametrize("coastline", (True,), indirect=True)
def test_panama_box_is_declared_in_physical_coordinates(coastline):
    import jax
    import jax.numpy as jnp
    from benchmarks.ad_demos.geometry_setup import channel_mask

    simulation, prepared, initial = coastline
    assert simulation.state.settings.coord_degree
    selected = channel_mask(prepared._make_state(initial.variables), "global_4deg", "Panama")
    expected = np.zeros((8, 10), bool)
    expected[3, 2] = True
    np.testing.assert_array_equal(selected, expected)
    opened = jax.jit(lambda depth: prepared.with_wet_geometry(initial, opening_m=depth, channel="Panama"))(
        jnp.asarray(60.0)
    )
    prepared.check_status(opened)
    np.testing.assert_allclose(opened.variables.ht[5, 4], 60.0)
    with pytest.raises(ValueError, match="No research channel"):
        channel_mask(simulation.state, "global_4deg", "unknown")


def test_declared_adjusted_basin_cannot_open_a_new_wet_sill():
    import jax.numpy as jnp
    from benchmarks.ad_demos.matrix import CASES
    from benchmarks.ad_demos.research_ocean import ResearchOcean

    ocean = ResearchOcean(
        dict(setup="adjusted_basin", physics="fixed", neutral_policy="nonstable", spinup_steps=0),
        CASES["geometry_equilibrium"],
    )
    assert np.all(ocean.initial.variables.geometry_reference_wet[2:-2, 2:-2])
    with pytest.raises(ValueError, match="no dry interior columns"):
        ocean.prepared.with_wet_geometry(ocean.initial, opening_m=jnp.asarray(100.0), channel="sill")
