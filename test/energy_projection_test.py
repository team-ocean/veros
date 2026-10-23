"""Column work identities for source transfers to the native W quadrature."""

from copy import copy

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("jax_backend")


def source_basin(*, partial=False, widths=(100.0, 100.0, 100.0, 100.0)):
    import jax.numpy as jnp
    from bathymetry_helpers import make_simulation
    from veros import runtime_settings
    from veros.core.numerics import u_centered_grid, calc_initial_conditions_kernel
    from veros.routines import tracing_context

    object.__setattr__(runtime_settings, "diskless_mode", True)
    object.__setattr__(runtime_settings, "linear_solver", "scipy_jax")
    bottom = np.tile(np.array([0, 1, 2, 3, 4]), (8, 2))
    simulation = make_simulation(partial_cells=partial, bottom_indices=bottom, wind=0, enable_tempsalt_sources=True)
    state = simulation.state
    widths = jnp.asarray(widths)
    with state.variables.unlock():
        with tracing_context():
            dzw, zt, zw = u_centered_grid(widths, jnp.zeros(4), jnp.zeros(4), jnp.zeros(4))
        vs = state.variables
        vs.dzt, vs.dzw, vs.zt, vs.zw = widths, dzw, zt - zw[-1], zw - zw[-1]
        vs.ht = jnp.sum(vs.maskT * widths, axis=2)
        with tracing_context():
            vs.update(calc_initial_conditions_kernel(state))
    prepared = simulation.prepare_integrator(record_diagnostics=())
    carry = prepared.pack(state)
    if partial:
        depths = carry.variables.ht[2:-2, 2:-2]
        bottom_width = widths[jnp.maximum(carry.variables.kbot[2:-2, 2:-2] - 1, 0)]
        carry = prepared.with_bathymetry(carry, jnp.where(depths > 0, depths - 0.4 * bottom_width, 0))
        prepared.check_status(carry)
    return prepared, carry


@pytest.mark.parametrize("widths", ((100.0, 100.0, 100.0, 100.0), (80.0, 120.0, 90.0, 110.0)))
def test_w_projection_preserves_every_column_including_one_wet_layer(widths):
    import jax.numpy as jnp
    from veros.core.diffusion import dissipation_on_wgrid
    from veros.routines import tracing_context

    prepared, carry = source_basin(widths=widths)
    state = prepared._make_state(carry.variables)
    vs = state.variables
    source = jnp.asarray(np.random.default_rng(61).normal(size=vs.maskT.shape)) * vs.maskT
    with tracing_context():
        projected = dissipation_on_wgrid(state, source, vs.kbot)
    before = jnp.sum(source * vs.dzt, axis=2)
    weights = (vs.maskW * vs.dzw).at[:, :, -1].multiply(0.5)
    after = jnp.sum(projected * weights, axis=2)
    np.testing.assert_allclose(after, before, rtol=1e-13, atol=1e-13)
    with tracing_context():
        positive = dissipation_on_wgrid(state, jnp.abs(source), vs.kbot)
    assert float(jnp.min(positive)) >= 0


def test_w_projection_metric_derivatives_preserve_the_column_work_identity():
    import jax
    import jax.numpy as jnp
    from veros.core.diffusion import dissipation_on_wgrid
    from veros.core.numerics import u_centered_grid
    from veros.routines import tracing_context

    prepared, carry = source_basin(widths=(80.0, 120.0, 90.0, 110.0))
    source = jnp.asarray(np.random.default_rng(62).normal(size=carry.variables.maskT.shape)) * carry.variables.maskT
    pattern = jnp.array([1.0, -0.5, 0.7, -1.2])

    def function(z):
        widths = carry.variables.dzt + z * pattern
        with tracing_context():
            dzw, _, _ = u_centered_grid(widths, jnp.zeros(4), jnp.zeros(4), jnp.zeros(4))
        moved = prepared.replace_variables(carry, dzt=widths, dzw=dzw)
        state = prepared._make_state(moved.variables)
        with tracing_context():
            projected = dissipation_on_wgrid(state, source, state.variables.kbot)
        weights = (state.variables.maskW * dzw).at[:, :, -1].multiply(0.5)
        return jnp.sum(projected * weights, axis=2)

    point = jnp.asarray(0.1)
    direction = jnp.asarray(0.3)
    value, tangent = jax.jvp(jax.jit(function), (point,), (direction,))
    expected = jnp.sum(source * pattern, axis=2) * direction
    np.testing.assert_allclose(tangent, expected, rtol=1e-12, atol=1e-12)
    seed = jnp.sin(jnp.arange(value.size)).reshape(value.shape)
    transpose = jax.vjp(function, point)[1](seed)[0]
    np.testing.assert_allclose(jnp.vdot(seed, tangent), transpose * direction, rtol=1e-12, atol=1e-12)
    difference = (function(point + 1e-3 * direction) - function(point - 1e-3 * direction)) / 2e-3
    np.testing.assert_allclose(difference, tangent, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("partial", (False, True))
def test_tracer_source_work_matches_occupied_dynamic_enthalpy_change(partial):
    import jax.numpy as jnp
    from veros.core import density
    from veros.core.diffusion import tempsalt_sources
    from veros.routines import tracing_context

    prepared, carry = source_basin(partial=partial, widths=(80.0, 120.0, 90.0, 110.0))
    state = prepared._make_state(carry.variables)
    # Enable this diagnostic independently of the other energy exchanges.
    state._settings = copy(state.settings)
    with state.settings.unlock():
        state.settings.enable_conserve_energy = True
    with state.variables.unlock():
        vs = state.variables
        pattern = jnp.sin(jnp.arange(vs.maskT.size).reshape(vs.maskT.shape))
        vs.temp_source = 1e-4 * pattern * vs.maskT
        vs.salt_source = -2e-5 * pattern * vs.maskT
    vs = state.variables
    with tracing_context():
        output = tempsalt_sources(state)
        before = density.get_dyn_enthalpy(state, vs.salt[..., vs.tau], vs.temp[..., vs.tau], jnp.abs(vs.zt))
        after = density.get_dyn_enthalpy(state, output.salt[..., vs.taup1], output.temp[..., vs.taup1], jnp.abs(vs.zt))
    fraction = vs.hfacT if partial else vs.maskT
    work = jnp.sum((after - before) / state.settings.dt_tracer * vs.dzt * fraction, axis=2)
    weights = (vs.maskW * vs.dzw).at[:, :, -1].multiply(0.5)
    projected = jnp.sum(output.P_diss_sources * weights, axis=2)
    np.testing.assert_allclose(projected[2:-2, 2:-2], work[2:-2, 2:-2], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("partial", (False, True))
def test_horizontal_tracer_work_matches_projected_face_dissipation(partial):
    import jax.numpy as jnp
    from veros.core.diffusion import horizontal_diffusion, compute_dissipation, dissipation_on_wgrid
    from veros.routines import tracing_context

    prepared, carry = source_basin(partial=partial, widths=(80.0, 120.0, 90.0, 110.0))
    state = prepared._make_state(carry.variables)
    vs = state.variables
    tracer = jnp.asarray(np.random.default_rng(63).normal(size=vs.maskT.shape)) * vs.maskT
    with tracing_context():
        tendency, fe, fn = horizontal_diffusion(state, tracer, 10.0)
        diss = compute_dissipation(state, tracer, fe, fn)
        projected = dissipation_on_wgrid(state, diss, vs.kbot)
    fraction = vs.hfacT if partial else vs.maskT
    volume = vs.area_t[..., None] * vs.dzt * fraction
    work = -state.settings.grav / state.settings.rho_0 * tracer * tendency * volume
    weights = (vs.area_t[..., None] * vs.maskW * vs.dzw).at[:, :, -1].multiply(0.5)
    actual = jnp.sum((projected * weights)[2:-2, 2:-2])
    expected = jnp.sum(work[2:-2, 2:-2])
    assert float(expected) > 0
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)
    assert float(jnp.min(projected)) >= 0


def test_partial_source_work_depth_products_and_finite_differences():
    import jax
    import jax.numpy as jnp
    from veros.core import density
    from veros.core.diffusion import tempsalt_sources
    from veros.routines import tracing_context

    prepared, carry = source_basin(partial=True)
    pattern = jnp.sin(jnp.arange(carry.variables.maskT.size).reshape(carry.variables.maskT.shape))
    carry = prepared.replace_variables(
        carry, temp_source=1e-4 * pattern * carry.variables.maskT, salt_source=-2e-5 * pattern * carry.variables.maskT
    )
    depths = carry.variables.ht[2:-2, 2:-2]
    depth_direction = jnp.cos(jnp.arange(depths.size).reshape(depths.shape)) * (depths > 0)

    def function(z):
        moved = prepared.with_bathymetry(carry, depths + z * depth_direction)
        state = prepared._make_state(moved.variables)
        state._settings = copy(state.settings)
        with state.settings.unlock():
            state.settings.enable_conserve_energy = True
        vs = state.variables
        with tracing_context():
            output = tempsalt_sources(state)
            before = density.get_dyn_enthalpy(state, vs.salt[..., vs.tau], vs.temp[..., vs.tau], jnp.abs(vs.zt))
            after = density.get_dyn_enthalpy(
                state, output.salt[..., vs.taup1], output.temp[..., vs.taup1], jnp.abs(vs.zt)
            )
        expected = jnp.sum((after - before) / state.settings.dt_tracer * vs.dzt * vs.hfacT, axis=2)
        weights = (vs.maskW * vs.dzw).at[:, :, -1].multiply(0.5)
        actual = jnp.sum(output.P_diss_sources * weights, axis=2)
        return jnp.stack((actual[2:-2, 2:-2], expected[2:-2, 2:-2]))

    point, direction = jnp.asarray(0.1), jnp.asarray(0.3)
    value, tangent = jax.jvp(jax.jit(function), (point,), (direction,))
    np.testing.assert_allclose(value[0], value[1], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(tangent[0], tangent[1], rtol=1e-10, atol=1e-12)
    assert float(jnp.linalg.norm(tangent)) > 1e-8
    seed = jnp.sin(jnp.arange(value.size)).reshape(value.shape)
    transpose = jax.vjp(function, point)[1](seed)[0]
    np.testing.assert_allclose(jnp.vdot(seed, tangent), transpose * direction, rtol=1e-10, atol=1e-12)
    difference = (function(point + 1e-3 * direction) - function(point - 1e-3 * direction)) / 2e-3
    np.testing.assert_allclose(difference, tangent, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("partial", (False, True))
def test_one_layer_source_budget_in_a_native_step_and_checkpoint_pullback(partial):
    import jax
    import jax.numpy as jnp
    from bathymetry_helpers import make_simulation
    from veros import runtime_settings
    from veros.checkpointing import CheckpointedIntegrator

    object.__setattr__(runtime_settings, "diskless_mode", True)
    object.__setattr__(runtime_settings, "linear_solver", "scipy_jax")
    bottom = np.tile(np.array([0, 1, 2, 3, 4]), (8, 2))
    simulation = make_simulation(
        partial_cells=partial,
        bottom_indices=bottom,
        wind=0,
        enable_tempsalt_sources=True,
        enable_conserve_energy=not partial,
        kappaH_0=0,
        kappaM_0=0,
        K_h=0,
    )
    if partial:
        # Enable energy exchanges for the source-only trajectory.
        simulation.state._settings = copy(simulation.state.settings)
        with simulation.state.settings.unlock():
            simulation.state.settings.enable_conserve_energy = True
    prepared = simulation.prepare_integrator(record_diagnostics=())
    initial = prepared.pack(simulation.state)
    if partial:
        depths = initial.variables.ht[2:-2, 2:-2]
        initial = prepared.with_bathymetry(initial, jnp.where(depths > 0, depths - 40, 0))
        prepared.check_status(initial)
    vs = initial.variables
    pattern = (1 + 0.2 * jnp.sin(jnp.arange(vs.maskT.size).reshape(vs.maskT.shape))) * vs.maskT
    fraction = vs.hfacT if partial else vs.maskT
    volume = vs.area_t[..., None] * vs.dzt * fraction
    weights = (vs.area_t[..., None] * vs.dzw * vs.maskW).at[:, :, -1].multiply(0.5)

    def controls(z):
        return prepared.replace_variables(initial, temp_source=1e-4 * z * pattern, salt_source=-2e-5 * z * pattern)

    def objective(carry):
        return jnp.sum((carry.variables.P_diss_sources * weights)[2:-2, 2:-2])

    def function(z):
        final = prepared.step(controls(z))[0]
        hd_change = final.variables.Hd[..., final.variables.tau] - initial.variables.Hd[..., initial.variables.tau]
        work = jnp.sum((hd_change * volume)[2:-2, 2:-2]) / simulation.state.settings.dt_tracer
        return jnp.stack((objective(final), work))

    point = jnp.asarray(0.2)
    final = prepared.step(controls(point))[0]
    prepared.check_status(final)
    for name in ("u", "v", "w"):
        np.testing.assert_allclose(getattr(final.variables, name), 0, atol=1e-12)
    value, tangent = jax.jvp(function, (point,), (jnp.asarray(1.0),))
    np.testing.assert_allclose(value[0], value[1], rtol=1e-9)
    np.testing.assert_allclose(tangent[0], tangent[1], rtol=1e-10)
    difference = (function(point + 1e-3) - function(point - 1e-3)) / 2e-3
    np.testing.assert_allclose(difference, tangent, rtol=1e-8)
    gradient = jax.grad(lambda z: function(z)[0])(point)
    result = CheckpointedIntegrator(prepared, terminal_objective=objective).value_and_grad(controls(point), n_steps=1)
    replay = jax.vjp(controls, point)[1](result.gradient)[0]
    np.testing.assert_allclose(replay, gradient, rtol=1e-10)


@pytest.mark.parametrize("partial", (False, True))
@pytest.mark.parametrize("superbee", (False, True))
def test_enthalpy_advection_preserves_occupied_inventory(partial, superbee):
    import jax.numpy as jnp
    from veros.core.thermodynamics import advect_temp_salt_enthalpy
    from veros.routines import tracing_context

    prepared, carry = source_basin(partial=partial)
    state = prepared._make_state(carry.variables)
    state._settings = copy(state.settings)
    with state.settings.unlock():
        state.settings.enable_conserve_energy = True
        state.settings.enable_superbee_advection = superbee
    rng = np.random.default_rng(65)
    with state.variables.unlock():
        vs = state.variables
        enthalpy = jnp.asarray(rng.normal(size=vs.maskT.shape)) * vs.maskT
        vs.Hd = jnp.broadcast_to(enthalpy[..., None], vs.Hd.shape)
        for name, mask in (("u", vs.maskU), ("v", vs.maskV), ("w", vs.maskW)):
            value = getattr(vs, name)
            setattr(vs, name, 1e-3 * jnp.asarray(rng.normal(size=value.shape)) * mask[..., None])
    with tracing_context():
        result = advect_temp_salt_enthalpy(state)
    fraction = vs.hfacT if partial else vs.maskT
    volume = vs.area_t[..., None] * vs.dzt * fraction
    contribution = (result.dHd[..., vs.tau] * volume)[2:-2, 2:-2]
    assert float(jnp.sum(jnp.abs(contribution))) > 1
    assert abs(float(jnp.sum(contribution))) < 1e-12 * float(jnp.sum(jnp.abs(contribution)))


@pytest.mark.parametrize("superbee", (False, True))
def test_partial_enthalpy_advection_depth_products_and_budget(superbee):
    import jax
    import jax.numpy as jnp
    from veros.core.thermodynamics import advect_temp_salt_enthalpy
    from veros.routines import tracing_context

    prepared, carry = source_basin(partial=True)
    rng = np.random.default_rng(67)
    replacements = {}
    for name, mask in (("u", carry.variables.maskU), ("v", carry.variables.maskV), ("w", carry.variables.maskW)):
        value = getattr(carry.variables, name)
        replacements[name] = 1e-3 * jnp.asarray(rng.normal(size=value.shape)) * mask[..., None]
    carry = prepared.replace_variables(carry, **replacements)
    frozen = jnp.asarray(rng.normal(size=carry.variables.Hd.shape)) * carry.variables.maskT[..., None]
    depths = carry.variables.ht[2:-2, 2:-2]
    pattern = jnp.cos(jnp.arange(depths.size).reshape(depths.shape)) * (depths > 0)

    def function(z):
        moved = prepared.with_bathymetry(carry, depths + z * pattern)
        moved = prepared.replace_variables(moved, Hd=frozen)
        state = prepared._make_state(moved.variables)
        state._settings = copy(state.settings)
        with state.settings.unlock():
            state.settings.enable_conserve_energy = True
            state.settings.enable_superbee_advection = superbee
        with tracing_context():
            result = advect_temp_salt_enthalpy(state)
        vs = state.variables
        tendency = result.dHd[2:-2, 2:-2, :, vs.tau]
        volume = (vs.area_t[..., None] * vs.dzt * vs.hfacT)[2:-2, 2:-2]
        budget = jnp.sum(tendency * volume) / jnp.sum(volume)
        return jnp.concatenate((tendency.reshape(-1), budget[None]))

    point, direction = jnp.asarray(0.1), jnp.asarray(0.3)
    value, tangent = jax.jvp(jax.jit(function), (point,), (direction,))
    np.testing.assert_allclose(value[-1], 0, atol=1e-14)
    np.testing.assert_allclose(tangent[-1], 0, atol=1e-14)
    assert float(jnp.linalg.norm(tangent)) > 1e-12
    seed = jnp.sin(jnp.arange(value.size))
    transpose = jax.vjp(function, point)[1](seed)[0]
    np.testing.assert_allclose(jnp.vdot(seed, tangent), transpose * direction, rtol=1e-10, atol=1e-14)
    for epsilon in (1e-3, 1e-4):
        difference = (function(point + epsilon * direction) - function(point - epsilon * direction)) / (2 * epsilon)
        np.testing.assert_allclose(difference, tangent, rtol=1e-5, atol=1e-14)
