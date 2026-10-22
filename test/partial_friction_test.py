"""Partial-cell horizontal stress budgets and geometry derivatives."""

import numpy as np
import pytest


@pytest.mark.parametrize("noslip", (False, True))
@pytest.mark.parametrize("cos_scaling", (False, True))
def test_partial_horizontal_stress_dissipates_and_conserves_free_slip_momentum(noslip, cos_scaling):
    import jax
    import jax.numpy as jnp
    from veros import runtime_settings
    from bathymetry_helpers import make_simulation
    from veros.core.friction import harmonic_friction
    from veros.routines import tracing_context

    object.__setattr__(runtime_settings, "diskless_mode", True)
    object.__setattr__(runtime_settings, "linear_solver", "scipy_jax")
    simulation = make_simulation(A_h=100, enable_noslip_lateral=noslip, enable_hor_friction_cos_scaling=cos_scaling)
    prepared = simulation.prepare_integrator(record_diagnostics=())
    initial = prepared.pack(simulation.state)
    vs = initial.variables
    rng = np.random.default_rng(25)
    cost, cosu = jnp.linspace(0.8, 1, vs.cost.size), jnp.linspace(0.81, 1.01, vs.cosu.size)
    initial = prepared.replace_variables(
        initial,
        u=jnp.asarray(rng.normal(size=vs.u.shape)) * 0.01 * vs.maskU[..., None],
        v=jnp.asarray(rng.normal(size=vs.v.shape)) * 0.01 * vs.maskV[..., None],
        cost=cost,
        cosu=cosu,
        area_u=vs.dxu[:, None] * vs.dyt[None, :] * cost[None, :],
        area_v=vs.dxt[:, None] * vs.dyu[None, :] * cosu[None, :],
    )
    depth = 335 + 0.4 * jnp.arange(8)[:, None] + 0.2 * jnp.arange(10)[None, :]
    pattern = jnp.sin(jnp.arange(8)[:, None] + jnp.arange(10)[None, :])

    def products(z):
        carry = prepared.with_bathymetry(initial, depth + z * pattern)
        with tracing_context():
            result = harmonic_friction(prepared._make_state(carry.variables))
        outputs = []
        work = jnp.asarray(0.0)
        for name, grid, tendency, area in (
            ("u", "U", result.du_mix, carry.variables.area_u),
            ("v", "V", result.dv_mix, carry.variables.area_v),
        ):
            velocity = getattr(carry.variables, name)[2:-2, 2:-2, :, carry.variables.tau]
            volume = area[2:-2, 2:-2, None] * carry.variables.dzt * getattr(carry.variables, "hfac" + grid)[2:-2, 2:-2]
            flux = tendency[2:-2, 2:-2] * volume
            outputs.append((jnp.sum(flux), jnp.sum(abs(flux))))
            work += jnp.sum(velocity * flux)
        return work, outputs

    point = jnp.asarray(0.2)
    work, budgets = products(point)
    assert float(work) < 0
    if not noslip:
        for total, magnitude in budgets:
            assert abs(float(total)) <= 1e-10 * float(magnitude)

    def objective(z):
        return products(z)[0]

    gradient = jax.grad(objective)(point)
    assert abs(float(gradient)) > 1e-8
    for epsilon in (1e-3, 1e-4):
        difference = (objective(point + epsilon) - objective(point - epsilon)) / (2 * epsilon)
        np.testing.assert_allclose(gradient, difference, rtol=2e-3, atol=1e-8)


def test_full_cell_horizontal_friction_matches_ordinary_rollout():
    from bathymetry_helpers import make_simulation
    from veros import runtime_settings

    object.__setattr__(runtime_settings, "diskless_mode", True)
    object.__setattr__(runtime_settings, "linear_solver", "scipy_jax")
    finals = []
    for partial in (False, True):
        simulation = make_simulation(partial_cells=partial, enable_hor_friction=True, A_h=100.0)
        prepared = simulation.prepare_integrator(record_diagnostics=())
        final, _ = prepared.advance(prepared.pack(simulation.state), n_steps=4)
        prepared.check_status(final)
        finals.append(final)
    for name in finals[0].variables.fields():
        np.testing.assert_allclose(
            finals[0].variables.get(name), finals[1].variables.get(name), rtol=1e-10, atol=1e-10, err_msg=name
        )
