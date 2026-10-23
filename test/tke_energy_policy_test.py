"""The public TKE policy retains occupied-volume budgets and geometry products."""

import numpy as np
import pytest

from wgrid_energy_test import basin  # noqa: F401
from wgrid_geometry_test import candidate
import wgrid_energy_test


@pytest.fixture(autouse=True)
def positive_diffusion_policy(monkeypatch):
    original = wgrid_energy_test.energy_state

    def state_with_policy(base):
        state = original(base)
        with state.settings.unlock():
            state.settings.enable_tke_positive_diffusion = True
        return state

    monkeypatch.setattr(wgrid_energy_test, "energy_state", state_with_policy)


@pytest.mark.parametrize("surface_flux", (1e-5, -1e-3))
def test_occupied_turbulence_energy_budget(basin, surface_flux):  # noqa: F811
    wgrid_energy_test.test_native_energy_sources_sinks_and_mixing_close_in_occupied_volumes(basin, surface_flux)


def test_occupied_geometry_products(basin):  # noqa: F811
    wgrid_energy_test.test_native_energy_kernel_geometry_products(basin)


@pytest.mark.parametrize("surface_flux", (1e-5, -1e-3))
def test_full_cell_limit(basin, surface_flux):  # noqa: F811
    wgrid_energy_test.test_energy_full_cell_limit_retains_native_kernel_values(basin, surface_flux)


@pytest.mark.parametrize("sign", (-1.0, 1.0))
def test_native_partial_work_conservation_capacity_and_geometry_products(basin, monkeypatch, sign):  # noqa: F811
    import jax
    import jax.numpy as jnp
    from veros.core import diffusion, thermodynamics
    from veros.core.geometry import wgrid_thickness
    from veros.routines import tracing_context

    def manufactured(state, dissipation, bottom):
        vs = state.variables
        pattern = 1 + 0.2 * jnp.sin(jnp.arange(vs.maskW.shape[-1]))
        return 2e-8 * vs.forc_tke_surface[2, 2] * pattern * vs.maskW

    monkeypatch.setattr(diffusion, "dissipation_on_wgrid", manufactured)

    def make_state(z):
        state = wgrid_energy_test.energy_state(candidate(basin, 0.3 + 0.02 * z))
        vs = state.variables
        pattern = jnp.arange(vs.maskW.shape[0])[:, None, None] % 3
        energy = jnp.where(pattern == 0, -1e-10, jnp.where(pattern == 1, 1e-8, 1e-3)) * vs.maskW
        with vs.unlock():
            vs.tke = jnp.broadcast_to(energy[..., None], vs.tke.shape)
            vs.forc_tke_surface = jnp.full_like(vs.forc_tke_surface, sign)
        return state

    def allocation(z):
        state = make_state(z)
        with tracing_context():
            result = thermodynamics.advect_temp_salt_enthalpy(state)
        return result.P_diss_adv[2:-2, 2:-2]

    state = make_state(jnp.asarray(0.0))
    vs = state.variables
    volume = vs.area_t[2:-2, 2:-2, None] * vs.dzw * vs.maskW[2:-2, 2:-2]
    volume = volume.at[..., -1].multiply(0.5)
    target = jnp.sum(volume * manufactured(state, None, None)[2:-2, 2:-2])
    value, tangent = jax.jvp(allocation, (jnp.asarray(0.0),), (jnp.asarray(1.0),))
    np.testing.assert_allclose(jnp.sum(volume * value), target, rtol=1e-12, atol=1e-7)
    if sign > 0:
        nominal = jnp.broadcast_to(vs.dzw, vs.maskW.shape).at[..., -1].multiply(0.5)
        occupied = wgrid_thickness(state)[2:-2, 2:-2]
        available = jnp.maximum(vs.tke[2:-2, 2:-2, :, vs.tau], 0) * occupied / nominal[2:-2, 2:-2]
        excess = state.settings.dt_mom * value[..., :-1] - available[..., :-1]
        assert float(jnp.max(excess)) < 1e-15
    assert float(jnp.linalg.norm(tangent)) > 1e-14
    np.testing.assert_allclose(jnp.sum(volume * tangent), 0, atol=1e-7)
    seed = jnp.cos(jnp.arange(value.size)).reshape(value.shape)
    transpose = jax.vjp(allocation, jnp.asarray(0.0))[1](seed)[0]
    np.testing.assert_allclose(jnp.vdot(seed, tangent), transpose, rtol=1e-8, atol=1e-18)
    for epsilon in (1e-3, 1e-4):
        difference = (allocation(epsilon) - allocation(-epsilon)) / (2 * epsilon)
        np.testing.assert_allclose(tangent, difference, rtol=2e-3, atol=1e-17)
