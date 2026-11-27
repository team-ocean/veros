"""Neutral-slope policy derivatives, conservative fluxes, and energy exchange."""

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("jax_backend")


def test_nonstable_slopes_have_zero_products_and_stable_slopes_are_unchanged():
    import jax
    import jax.numpy as jnp
    from veros.core.isoneutral.isoneutral import neutral_slope

    horizontal = jnp.array([0.0, 2e-21, 1e-5, 2e-5])
    vertical = jnp.array([0.0, 0.001, -0.001, -0.002])

    def filtered(x):
        return neutral_slope(x, vertical, True)

    actual = filtered(horizontal)
    np.testing.assert_array_equal(actual[:2], 0)
    np.testing.assert_array_equal(actual[2:], neutral_slope(horizontal, vertical)[2:])
    tangent = jax.jvp(filtered, (horizontal,), (jnp.ones_like(horizontal),))[1]
    reverse = jax.grad(lambda x: jnp.sum(filtered(x)))(horizontal)
    np.testing.assert_array_equal(tangent[:2], 0)
    np.testing.assert_array_equal(reverse[:2], 0)
    np.testing.assert_allclose(tangent, reverse, rtol=1e-12)


@pytest.mark.parametrize("policy", [False, True])
@pytest.mark.parametrize("iso", [False, True])
def test_neutral_tracer_and_energy_budgets(policy, iso):
    import jax.numpy as jnp
    from benchmarks.ad_demos.bathymetry import make_simulation
    from veros.core.isoneutral.isoneutral import isoneutral_diffusion_pre
    from veros.core.isoneutral.diffusion import isoneutral_diffusion_kernel
    from veros.core.numerics import calc_initial_conditions_kernel
    from veros.core.utilities import enforce_boundaries

    simulation = make_simulation(
        partial_cells=False,
        wind=0,
        enable_neutral_diffusion=True,
        enable_skew_diffusion=True,
        enable_conserve_energy=True,
        enable_neutral_nonstable_zero_slope=policy,
        K_iso_0=500,
        K_iso_steep=100,
        K_gm_0=500,
        eq_of_state_type=3,
    )
    state, settings = simulation.state, simulation.state.settings
    with state.variables.unlock():
        vs = state.variables
        perturbation = jnp.sin(jnp.arange(settings.nx)[:, None] + jnp.arange(settings.ny)[None, :])
        vs.temp = vs.temp.at[2:-2, 2:-2, :, :].add(0.05 * perturbation[..., None, None])
        # Include an unstable interior stencil alongside the surface boundary.
        vs.temp = vs.temp.at[5, 5, 1, :].add(1.0)
        vs.temp = enforce_boundaries(vs.temp, settings.enable_cyclic_x)
        vs.update(calc_initial_conditions_kernel(state))
        vs.K_iso = jnp.full_like(vs.K_iso, 500)
        vs.K_gm = jnp.full_like(vs.K_gm, 500)
        vs.update(isoneutral_diffusion_pre(state))
        volume = vs.area_t[2:-2, 2:-2, None] * vs.dzt * vs.maskT[2:-2, 2:-2]
        volume_w = vs.area_t[2:-2, 2:-2, None] * vs.dzw * vs.maskW[2:-2, 2:-2]
        volume_w = volume_w.at[..., -1].multiply(0.5)
        enthalpy_tendency = 0.0
        for name, istemp in (("temp", True), ("salt", False)):
            before = getattr(vs, name)
            integral = getattr(vs, "int_drhod" + ("T" if istemp else "S"))[..., vs.tau]
            out = isoneutral_diffusion_kernel(state, before, istemp, iso=iso)
            vs.update(out)
            change = (getattr(vs, name)[..., vs.taup1] - before[..., vs.taup1])[2:-2, 2:-2]
            np.testing.assert_allclose(jnp.sum(change * volume) / jnp.sum(volume), 0, atol=1e-12)
            enthalpy_tendency += float(
                jnp.sum(-settings.grav / settings.rho_0 * integral[2:-2, 2:-2] * change / settings.dt_tracer * volume)
            )
        exchange = float(jnp.sum(getattr(vs, "P_diss_iso" if iso else "P_diss_skew")[2:-2, 2:-2] * volume_w))
        np.testing.assert_allclose(enthalpy_tendency, exchange, rtol=1e-7, atol=1e-5)


def test_global_forcing_without_tke():
    import jax.numpy as jnp
    from veros.state import get_default_state
    from veros.variables import Variable
    from veros.setups.global_4deg.global_4deg import set_forcing_kernel

    state = get_default_state()
    with state.settings.unlock():
        state.settings.update(nx=2, ny=3, nz=2, enable_tke=False)
    state.dimensions["nmonths"] = 12
    for name in ("sss_clim", "sst_clim", "qnec", "qnet", "taux", "tauy"):
        state.var_meta[name] = Variable(name, ("xt", "yt", "nmonths"))
    state.initialize_variables()
    with state.variables.unlock():
        vs = state.variables
        vs.maskT = jnp.ones_like(vs.maskT)
        vs.dzt = jnp.ones_like(vs.dzt)
        vs.sss_clim = jnp.full_like(vs.sss_clim, 35)
        vs.salt = jnp.full_like(vs.salt, 35)
        vs.qnet = jnp.full_like(vs.qnet, 20)
        vs.taux = jnp.full_like(vs.taux, 0.1)
        assert "forc_tke_surface" not in vs
        output = set_forcing_kernel(state)
        assert "forc_tke_surface" not in output._asdict()
        np.testing.assert_allclose(output.forc_temp_surface, 20 / (3991.86795711963 * state.settings.rho_0))
        np.testing.assert_allclose(output.forc_salt_surface, 0)
        np.testing.assert_allclose(output.surface_taux, 0.1)
