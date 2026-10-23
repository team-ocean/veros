"""Coupled nonlinear EOS, neutral/skew mixing, and split-timestep grid derivatives."""

from copy import copy

import numpy as np

from wgrid_geometry_test import basin as basin, candidate as geometry_candidate  # noqa: F401
from wgrid_energy_test import energy_state
import pytest

pytestmark = pytest.mark.usefixtures("jax_backend")


def test_partial_cell_turbulence_and_grid_derivatives(basin):  # noqa: F811
    import jax
    import jax.numpy as jnp
    from veros.diagnostics.energy import Energy
    from veros.settings import check_setting_conflicts
    from veros.checkpointing import BinomialSchedule, CheckpointedIntegrator

    simulation = copy(basin[0].simulation)
    from veros.state import get_default_state

    base = energy_state(geometry_candidate(basin, 0.3))
    state = get_default_state()
    state._settings = copy(base.settings)
    with state.settings.unlock():
        settings = state.settings
        settings.eq_of_state_type = 5
        settings.enable_neutral_diffusion = settings.enable_skew_diffusion = True
        settings.K_iso_0 = settings.K_iso_steep = 1000.0
        settings.iso_dslope, settings.iso_slopec = 4e-3, 1e-3
        settings.enable_hor_friction = settings.enable_hor_friction_cos_scaling = True
        settings.A_h, settings.hor_friction_cosPower = 100.0, 1
        settings.c_k, settings.c_eps, settings.alpha_tke = 0.1, 0.7, 30.0
        settings.mxl_min, settings.tke_mxl_choice = 1e-8, 2
        settings.kappaM_min, settings.kappaH_min = 2e-4, 2e-5
        settings.enable_kappaH_profile = True
        settings.enable_tke_superbee_advection = settings.enable_eke_superbee_advection = True
        settings.eke_k_max, settings.eke_c_k, settings.eke_c_eps = 1e4, 0.4, 0.5
        settings.eke_cross, settings.eke_crhin, settings.eke_lmin = 2.0, 1.0, 100.0
        settings.enable_eke_diss_surfbot, settings.eke_diss_surfbot_frac = True, 0.2
        settings.dt_mom = 150.0
    state.initialize_variables()
    with state.variables.unlock():
        for name, value in base.variables.items():
            if name in state.variables:
                setattr(state.variables, name, value)
    check_setting_conflicts(state.settings)
    diagnostic = Energy(state)
    diagnostic.initialize_variables(state)
    state._diagnostics = {"energy": diagnostic}
    simulation.state = state
    state, settings = simulation.state, simulation.state.settings
    with state.variables.unlock():
        vs = state.variables
        profile = 1 + 0.1 * jnp.arange(settings.nz)
        for name, scale in (("tke", 1e-3), ("eke", 0.02)):
            values = scale * profile * vs.maskW
            setattr(vs, name, jnp.broadcast_to(values[..., None], getattr(vs, name).shape))
        vs.K_diss_v = 2e-7 * vs.maskW
        vs.coriolis_t = jnp.full_like(vs.coriolis_t, 1e-4)
        vs.beta = jnp.full_like(vs.beta, 1e-11)
    diagnostic = state.diagnostics["energy"]
    diagnostic.sampling_frequency = diagnostic.output_frequency = settings.dt_tracer
    diagnostic.output_variables = ("tke_m", "eke_m", "dtke_m", "deke_m")
    prepared = simulation.prepare_integrator(record_diagnostics=("energy",))
    initial = prepared.pack(state)

    def controlled(z):
        return prepared.with_vertical_grid(
            initial, widths=initial.variables.dzt + z * jnp.array([-2.0, 1.0, 3.0, -2.0])
        )

    point = jnp.asarray(0.2)
    candidate = controlled(point)
    prepared.check_status(candidate)
    first, records = prepared.step(candidate)
    prepared.check_status(first)
    assert bool(records["energy"].valid)

    def energy_integral(carry, name):
        vs = carry.variables
        # Independently integrate the native centre-bounded W intervals.
        upper = jnp.concatenate((vs.zt[1:], jnp.zeros(1)))
        widths = jnp.maximum(0.0, upper - jnp.maximum(vs.zt, -vs.ht[..., None]))
        volume = vs.area_t[2:-2, 2:-2, None] * widths[2:-2, 2:-2] * vs.maskW[2:-2, 2:-2]
        return jnp.sum(volume * getattr(vs, name)[2:-2, 2:-2, :, vs.tau])

    for name, tendency, dt in (("tke", "dtke_m", settings.dt_mom), ("eke", "deke_m", settings.dt_tracer)):
        before, after = energy_integral(candidate, name), energy_integral(first, name)
        np.testing.assert_allclose(records["energy"].values[name + "_m"], settings.rho_0 * before, rtol=1e-12)
        np.testing.assert_allclose(
            records["energy"].values[tendency], settings.rho_0 * (after - before) / dt, rtol=1e-10
        )

    def objective(carry):
        vs = carry.variables
        return jnp.mean(vs.tke[2:-2, 2:-2, :, vs.tau]) + 0.1 * jnp.mean(vs.eke[2:-2, 2:-2, :, vs.tau])

    def function(z):
        final, _ = prepared.advance(controlled(z), n_steps=2)
        return objective(final)

    value, tangent = jax.jvp(function, (point,), (jnp.asarray(1.0),))
    assert abs(float(tangent)) > 1e-8
    for epsilon in (1e-3, 1e-4):
        difference = (function(point + epsilon) - function(point - epsilon)) / (2 * epsilon)
        np.testing.assert_allclose(difference, tangent, rtol=2e-3, atol=1e-10)
    driver = CheckpointedIntegrator(prepared, terminal_objective=objective)
    result = driver.value_and_grad(candidate, n_steps=2, schedule=BinomialSchedule(2, ram_checkpoints=2))
    gradient = jax.vjp(controlled, point)[1](result.gradient)[0]
    np.testing.assert_allclose(result.value, value, rtol=1e-12)
    np.testing.assert_allclose(gradient, tangent, rtol=1e-8, atol=1e-10)
