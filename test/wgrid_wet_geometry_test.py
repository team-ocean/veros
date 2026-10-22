"""Coupled energy histories survive continuous and discrete channel openings."""

import numpy as np
import pytest

from wgrid_turbulence_helpers import channel_simulation


def volume(carry, grid):
    import jax.numpy as jnp

    vs = carry.variables
    if grid in ("W", "work"):
        if grid == "W":
            upper = jnp.concatenate((vs.zt[1:], jnp.zeros(1)))
            width = jnp.maximum(0.0, upper - jnp.maximum(vs.zt, -vs.ht[..., None])) * vs.maskW
        else:
            width = jnp.broadcast_to(vs.dzw, vs.maskW.shape).at[..., -1].multiply(0.5) * vs.maskW
        area = vs.area_t
    else:
        width = vs.dzt * getattr(vs, "hfac" + grid)
        area = getattr(vs, "area_" + grid.lower())
    return area[2:-2, 2:-2, None] * width[2:-2, 2:-2]


def inventory(carry, name, grid):
    import jax.numpy as jnp

    weights = volume(carry, grid)
    values = getattr(carry.variables, name)[2:-2, 2:-2]
    if values.ndim == 4:
        weights = weights[..., None]
    return jnp.sum(weights * values, axis=(0, 1, 2))


@pytest.mark.parametrize("cyclic", (False, True))
def test_turbulent_channel_budgets_derivatives_replay_and_realization(cyclic):
    import jax
    import jax.numpy as jnp
    from veros.checkpointing import BinomialSchedule, CheckpointedIntegrator
    from veros.core import numerics, momentum, eke, tke
    from veros.routines import tracing_context
    from veros.settings import check_setting_conflicts

    simulation, prepared, initial = channel_simulation(cyclic=cyclic)
    check_setting_conflicts(simulation.state.settings)
    channel = jnp.zeros((8, 10), bool).at[4, 5].set(True)

    def controlled(depth):
        return prepared.with_wet_geometry(initial, opening_m=depth, channel=channel)

    point = jnp.asarray(60.0)
    candidate = controlled(point)
    prepared.check_status(candidate)
    assert int(candidate.variables.kbot[6, 7]) == 4
    np.testing.assert_allclose(candidate.variables.ht[6, 7], point)
    for name, grid in (
        ("temp", "T"),
        ("salt", "T"),
        ("tke", "W"),
        ("eke", "W"),
        ("dtke", "W"),
        ("deke", "W"),
        ("K_diss_v", "work"),
    ):
        np.testing.assert_allclose(
            inventory(candidate, name, grid), inventory(initial, name, grid), rtol=2e-12, atol=1e-7
        )
        expected = inventory(initial, name, grid) / jnp.sum(volume(candidate, grid))
        np.testing.assert_allclose(getattr(candidate.variables, name)[6, 7, -1], expected, rtol=1e-12, atol=1e-14)
    first, records = prepared.step(candidate)
    prepared.check_status(first)
    assert bool(records["energy"].valid)
    for name, dt in (("tke", simulation.state.settings.dt_mom), ("eke", simulation.state.settings.dt_tracer)):
        rho = simulation.state.settings.rho_0
        before = inventory(candidate, name, "W")[candidate.variables.tau]
        after = inventory(first, name, "W")[first.variables.tau]
        np.testing.assert_allclose(records["energy"].values[name + "_m"], rho * before, rtol=1e-12)
        np.testing.assert_allclose(records["energy"].values["d" + name + "_m"], rho * (after - before) / dt, rtol=1e-10)

    def objective(carry):
        vs = carry.variables
        return jnp.mean(vs.tke[2:-2, 2:-2, :, vs.tau]) + 0.1 * jnp.mean(vs.eke[2:-2, 2:-2, :, vs.tau])

    def function(depth):
        return objective(prepared.advance(controlled(depth), n_steps=2)[0])

    value, tangent = jax.jvp(function, (point,), (jnp.asarray(1.0),))
    assert abs(float(tangent)) > 1e-9
    for epsilon in (1e-3, 1e-4):
        difference = (function(point + epsilon) - function(point - epsilon)) / (2 * epsilon)
        np.testing.assert_allclose(difference, tangent, rtol=2e-3, atol=1e-10)
    driver = CheckpointedIntegrator(prepared, terminal_objective=objective)
    result = driver.value_and_grad(candidate, n_steps=2, schedule=BinomialSchedule(2, ram_checkpoints=2))
    gradient = jax.vjp(controlled, point)[1](result.gradient)[0]
    np.testing.assert_allclose(result.value, value, rtol=1e-12)
    np.testing.assert_allclose(gradient, tangent, rtol=1e-8, atol=1e-10)

    realized = prepared.with_wet_geometry(initial, opening_m=100.0, channel=channel, realized=True)
    prepared.check_status(realized)
    np.testing.assert_allclose(realized.variables.ht[6, 7], 150.0)
    final, _ = prepared.advance(realized, n_steps=2)
    prepared.check_status(final)
    if not cyclic:
        return
    # The comparison setup constructs the open topology directly. Reconstruct its
    # declared pool from domain inventories without calling the remap primitive.
    other_simulation, other, independent = channel_simulation(cyclic=True, opened=True)
    fields = {}
    for name, grid in (
        ("temp", "T"),
        ("salt", "T"),
        ("tke", "W"),
        ("eke", "W"),
        ("dtke", "W"),
        ("deke", "W"),
        ("K_diss_v", "work"),
    ):
        old_volume, new_volume = volume(initial, grid), volume(independent, grid)
        retained = jnp.sum(jnp.where(jnp.sum(old_volume, axis=2, keepdims=True) > 0, new_volume, 0))
        total = jnp.sum(new_volume)
        values = getattr(initial.variables, name)
        replacement = values * retained / total
        replacement = replacement.at[6, 7, -1].set(inventory(initial, name, grid) / total)
        fields[name] = replacement
    independent = other.replace_variables(independent, **fields)
    with tracing_context():
        state = other._make_state(independent.variables)
        with state.variables.unlock():
            other_simulation.rebuild_wet_forcing(state)
            state.variables.update(numerics.calc_initial_conditions_kernel(state))
            state.variables.update(momentum.vertical_velocity_kernel(state))
            eke.set_eke_diffusivities(state)
            tke.set_tke_diffusivities(state)
        independent = independent._replace(variables=state.variables)
    for name in ("kbot", "hfacT", "hfacU", "hfacV", "temp", "salt", "tke", "eke", "dtke", "deke", "K_diss_v"):
        np.testing.assert_allclose(
            getattr(realized.variables, name), getattr(independent.variables, name), rtol=1e-12, atol=1e-14
        )
    other_final, _ = other.advance(independent, n_steps=2)
    other.check_status(other_final)
    for name in ("temp", "salt", "u", "v", "tke", "eke"):
        np.testing.assert_allclose(
            getattr(final.variables, name), getattr(other_final.variables, name), rtol=1e-10, atol=1e-12
        )
