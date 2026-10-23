"""Fresh public partial-cell turbulence setups retain budgets and resting states."""

import numpy as np
import pytest

from wgrid_turbulence_helpers import channel_simulation, configure_turbulence
from wgrid_wet_geometry_test import inventory

pytestmark = pytest.mark.usefixtures("jax_backend")


def public_basin(tke, eke):
    import jax.numpy as jnp
    from bathymetry_helpers import make_simulation
    from veros import veros_routine, runtime_settings

    object.__setattr__(runtime_settings, "diskless_mode", True)

    bottom = np.ones((8, 10), dtype=int)
    bottom[4, :] = 0
    original = make_simulation(bottom_indices=bottom, cyclic_x=True, wind=0)

    class PublicBasin(type(original)):
        @veros_routine
        def set_parameter(self, state):
            super().set_parameter(state)
            configure_turbulence(state.settings)
            state.settings.enable_tke, state.settings.enable_eke = tke, eke

        @veros_routine
        def set_grid(self, state):
            super().set_grid(state)
            state.variables.dzt = jnp.asarray([60.0, 90.0, 120.0, 150.0])

        @veros_routine
        def set_coriolis(self, state):
            state.variables.coriolis_t = 1e-4 + 1e-11 * jnp.broadcast_to(
                state.variables.yt, state.variables.coriolis_t.shape
            )

        @veros_routine
        def set_initial_conditions(self, state):
            vs = state.variables
            vs.temp = 10 * jnp.broadcast_to(vs.maskT[..., None], vs.temp.shape)
            vs.salt = 35 * jnp.broadcast_to(vs.maskT[..., None], vs.salt.shape)
            for name, value in (("tke", 1e-3), ("eke", 0.02)):
                if name in vs:
                    setattr(vs, name, value * jnp.broadcast_to(vs.maskW[..., None], getattr(vs, name).shape))

        @veros_routine(supports_compilation=True)
        def rebuild_wet_forcing(self, state):
            if state.settings.enable_tke:
                state.variables.forc_tke_surface = 1e-5 * state.variables.maskW[..., -1]

        @veros_routine(supports_compilation=True)
        def set_forcing(self, state):
            super().set_forcing(state)
            self.rebuild_wet_forcing(state)

        @veros_routine
        def set_diagnostics(self, state):
            super().set_diagnostics(state)
            diagnostic = state.diagnostics["energy"]
            diagnostic.output_variables = ("tke_m", "eke_m", "dtke_m", "deke_m")
            diagnostic.sampling_frequency = diagnostic.output_frequency = state.settings.dt_tracer

    simulation = PublicBasin()
    simulation.setup()
    return simulation


@pytest.mark.parametrize("tke,eke", ((True, False), (False, True), (True, True)))
def test_public_turbulence_combinations_and_wet_rest(tke, eke):
    import jax.numpy as jnp

    simulation = public_basin(tke, eke)
    prepared = simulation.prepare_integrator(record_diagnostics=("energy",))
    initial = prepared.pack(simulation.state)
    moved = prepared.with_vertical_grid(initial, widths=initial.variables.dzt + jnp.asarray([-0.4, -0.2, 0.2, 0.4]))
    prepared.check_status(moved)
    channel = jnp.zeros((8, 10), bool).at[4, 5].set(True)
    opened = prepared.with_wet_geometry(moved, opening_m=60.0, channel=channel)
    prepared.check_status(opened)
    for name in ("tke", "eke", "dtke", "deke", "K_diss_v"):
        if name not in initial.variables or (name == "K_diss_v" and not tke):
            continue
        grid = "work" if name == "K_diss_v" else "W"
        np.testing.assert_allclose(inventory(initial, name, grid), inventory(moved, name, grid), rtol=1e-12, atol=1e-8)
        np.testing.assert_allclose(inventory(initial, name, grid), inventory(opened, name, grid), rtol=1e-12, atol=1e-8)
    final, records = prepared.advance(opened, n_steps=3)
    prepared.check_status(final)
    assert np.all(records["energy"].valid)
    for value in records["energy"].values.values():
        assert np.all(np.isfinite(value))
    for name in ("u", "v", "w"):
        np.testing.assert_allclose(getattr(final.variables, name), 0, rtol=0, atol=1e-11)
    for name in ("temp", "salt"):
        np.testing.assert_allclose(
            getattr(final.variables, name), getattr(opened.variables, name), rtol=1e-12, atol=1e-12
        )
    for name in ("tke", "eke"):
        if name in final.variables:
            assert np.all(np.isfinite(getattr(final.variables, name)))


def test_public_setup_matches_explicitly_allocated_state():
    import jax.numpy as jnp

    _, native, original = channel_simulation(cyclic=True)
    _, public, initial = channel_simulation(cyclic=True, public=True)
    channel = jnp.zeros((8, 10), bool).at[4, 5].set(True)
    first = native.with_wet_geometry(original, opening_m=60.0, channel=channel)
    second = public.with_wet_geometry(initial, opening_m=60.0, channel=channel)
    assert set(first.variables.fields()) == set(second.variables.fields())
    for name in first.variables.fields():
        np.testing.assert_allclose(
            getattr(first.variables, name), getattr(second.variables, name), rtol=1e-12, atol=1e-14, err_msg=name
        )
    for name, value in native._template.settings.items():
        assert value == getattr(public._template.settings, name), name
    a, _ = native.advance(first, n_steps=2)
    b, _ = public.advance(second, n_steps=2)
    native.check_status(a)
    public.check_status(b)
    for name in a.variables.fields():
        np.testing.assert_allclose(
            getattr(a.variables, name), getattr(b.variables, name), rtol=1e-10, atol=1e-12, err_msg=name
        )
