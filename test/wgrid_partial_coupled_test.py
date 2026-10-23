"""Coupled partial-cell energy budgets and vertical-grid derivatives."""

from copy import copy

import numpy as np

from wgrid_geometry_test import basin as basin, candidate as geometry_candidate  # noqa: F401
from wgrid_energy_test import energy_state


def test_partial_cell_energy_and_grid_derivatives(basin):  # noqa: F811
    import jax
    import jax.numpy as jnp
    from veros.diagnostics.energy import Energy
    from veros.settings import check_setting_conflicts
    from veros.checkpointing import BinomialSchedule, CheckpointedIntegrator

    simulation = copy(basin[0].simulation)
    state = energy_state(geometry_candidate(basin, 0.3))
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
