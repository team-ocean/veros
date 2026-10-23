"""Inventory, positivity, and derivatives at coincident vertical interfaces."""

import numpy as np
import pytest

from veros.remapping import remap_cell_averages, remap_column_inventories

pytestmark = pytest.mark.usefixtures("jax_backend")


def test_moving_levels_preserve_each_history_inventory_and_constant_fields():
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    old = jnp.array([[0.0, 1.0, 2.0, 1.0], [1.0, 2.0, 1.0, 0.0]])
    new = jnp.array([[0.0, 0.8, 2.1, 1.1], [1.2, 1.7, 1.1, 0.0]])
    values = jnp.asarray(np.random.default_rng(4).uniform(0.1, 2, (2, 4, 3)))
    mapped = jax.jit(lambda v: remap_cell_averages(v, old, new, axis=1))(values)
    np.testing.assert_allclose(
        jnp.sum(mapped * new[..., None], axis=1), jnp.sum(values * old[..., None], axis=1), atol=1e-14
    )
    assert np.min(mapped) >= 0
    constant = remap_cell_averages(jnp.ones_like(values) * (old > 0)[..., None], old, new, axis=1)
    np.testing.assert_allclose(constant, jnp.broadcast_to((new > 0)[..., None], values.shape), atol=1e-14)
    np.testing.assert_allclose(remap_cell_averages(values, old, old, axis=1), values * (old > 0)[..., None], atol=1e-14)


def test_remap_jvp_vjp_and_finite_differences_at_coincident_interfaces():
    import jax
    import jax.numpy as jnp

    old = jnp.array([1.0, 2.0, 1.0])
    values = jnp.array([0.2, 1.0, 3.0])
    direction, seed = jnp.array([0.1, -0.3, 0.2]), jnp.array([0.4, -0.2, 0.1])

    def function(widths):
        return remap_cell_averages(values, old, widths)

    tangent = jax.jvp(function, (old,), (direction,))[1]
    transpose = jax.vjp(function, old)[1](seed)[0]
    np.testing.assert_allclose(seed @ tangent, transpose @ direction, atol=1e-14)
    errors = []
    for epsilon in (1e-4, 1e-5, 1e-6):
        difference = (function(old + epsilon * direction) - function(old - epsilon * direction)) / (2 * epsilon)
        errors.append(float(jnp.linalg.norm(difference - tangent)))
    # The primitive is C1 across an old face; central errors there are O(epsilon).
    assert errors[1] < errors[0] / 5 and errors[2] < errors[1] / 5
    np.testing.assert_allclose(difference, tangent, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(jnp.sum(tangent * old + values * direction), 0, atol=1e-14)


def test_conservative_projection_between_different_vertical_level_counts():
    import jax
    import jax.numpy as jnp

    old = jnp.array([0.5, 0.5, 1.0, 1.0, 1.0])
    new = jnp.array([1.0, 2.0, 1.0])
    values = jnp.array([[1.0, 0.1], [0.2, 0.4], [0.3, 0.8], [2.0, 1.2], [1.0, 0.5]])
    mapped = jax.jit(lambda v: remap_cell_averages(v, old, new, axis=0))(values)
    assert mapped.shape == (3, 2)
    np.testing.assert_allclose(
        jnp.sum(mapped * new[:, None], axis=0), jnp.sum(values * old[:, None], axis=0), atol=1e-14
    )
    assert np.min(mapped) >= 0


def test_volume_change_retains_column_inventories_and_has_resolved_products():
    import jax
    import jax.numpy as jnp

    old = jnp.array([[0.0, 1.0, 2.0, 1.0], [1.0, 2.0, 1.0, 0.0]])
    new = jnp.array([[0.0, 0.8, 1.8, 1.0], [1.2, 2.1, 1.1, 0.0]])
    values = jnp.asarray(np.random.default_rng(61).uniform(0.1, 2, (2, 4, 3)))
    mapped = jax.jit(lambda widths: remap_column_inventories(values, old, widths, axis=1))(new)
    np.testing.assert_allclose(
        jnp.sum(mapped * new[..., None], axis=1), jnp.sum(values * old[..., None], axis=1), atol=1e-14
    )
    assert np.min(mapped) >= 0
    constant = remap_column_inventories((old > 0).astype(old.dtype), old, new)
    expected = jnp.sum(old, axis=-1) / jnp.sum(new, axis=-1)
    np.testing.assert_allclose(constant, (new > 0) * expected[..., None], atol=1e-14)
    direction = jnp.array([[0, 0.1, -0.2, 0.15], [0.05, -0.1, 0.2, 0]])
    seed = jnp.sin(jnp.arange(values.size)).reshape(values.shape)

    def function(widths):
        return remap_column_inventories(values, old, widths, axis=1)

    tangent = jax.jvp(function, (new,), (direction,))[1]
    transpose = jax.vjp(function, new)[1](seed)[0]
    np.testing.assert_allclose(jnp.vdot(seed, tangent), jnp.vdot(transpose, direction), atol=1e-13)
    errors = []
    for epsilon in (1e-4, 1e-5, 1e-6):
        difference = (function(new + epsilon * direction) - function(new - epsilon * direction)) / (2 * epsilon)
        errors.append(float(jnp.linalg.norm(difference - tangent)))
    assert errors[1] < errors[0] / 5 and errors[2] < errors[1] / 5
    np.testing.assert_allclose(tangent, difference, rtol=1e-5, atol=1e-9)


def test_bathymetry_inventory_policy_trajectory_and_checkpoint_pullback():
    import jax
    import jax.numpy as jnp
    from bathymetry_helpers import make_simulation
    from veros import runtime_settings
    from veros.checkpointing import CheckpointedIntegrator

    object.__setattr__(runtime_settings, "diskless_mode", True)
    object.__setattr__(runtime_settings, "linear_solver", "scipy_jax")
    simulation = make_simulation(wind=0, enable_hor_friction=True, A_h=100.0)
    prepared = simulation.prepare_integrator(record_diagnostics=())
    initial = prepared.pack(simulation.state)
    initial = prepared.with_bathymetry(initial, jnp.full((8, 10), 360.0))
    rng = np.random.default_rng(62)
    replacements = {}
    for name, grid, scale in (
        ("u", "U", 0.01),
        ("v", "V", 0.01),
        ("du", "U", 1e-5),
        ("dv", "V", 1e-5),
        ("dtemp", "T", 1e-7),
        ("dsalt", "T", 1e-8),
    ):
        field = getattr(initial.variables, name)
        replacements[name] = (
            scale * jnp.asarray(rng.normal(size=field.shape)) * getattr(initial.variables, "mask" + grid)[..., None]
        )
    initial = prepared.replace_variables(initial, **replacements)
    depth = 355 + 0.2 * jnp.arange(8)[:, None] + 0.1 * jnp.arange(10)[None, :]
    pattern = jnp.sin(jnp.arange(8)[:, None] + jnp.arange(10)[None, :])

    def controlled(z):
        return prepared.with_bathymetry(initial, depth + z * pattern, conserve=("temp", "salt", "momentum"))

    point = jnp.asarray(0.2)
    candidate = jax.jit(controlled)(point)
    prepared.check_status(candidate)
    for name, grid in (
        ("temp", "T"),
        ("salt", "T"),
        ("u", "U"),
        ("v", "V"),
        ("du", "U"),
        ("dv", "V"),
        ("dtemp", "T"),
        ("dsalt", "T"),
    ):
        before = getattr(initial.variables, name)[2:-2, 2:-2]
        after = getattr(candidate.variables, name)[2:-2, 2:-2]
        old_volume = initial.variables.dzt * getattr(initial.variables, "hfac" + grid)[2:-2, 2:-2]
        new_volume = candidate.variables.dzt * getattr(candidate.variables, "hfac" + grid)[2:-2, 2:-2]
        np.testing.assert_allclose(
            jnp.sum(before * old_volume[..., None], axis=2), jnp.sum(after * new_volume[..., None], axis=2), atol=1e-10
        )
    default = prepared.with_bathymetry(initial, depth + point * pattern)
    for name in ("temp", "salt", "u", "v", "du", "dv", "dtemp", "dsalt"):
        np.testing.assert_array_equal(getattr(default.variables, name), getattr(initial.variables, name))
    assert float(jnp.min(candidate.variables.salt)) >= 0

    def objective(carry):
        return jnp.mean(carry.variables.v[2:-2, 2:-2, 0, carry.variables.tau])

    def trajectory(z):
        return objective(prepared.advance(controlled(z), n_steps=2)[0])

    gradient = jax.grad(trajectory)(point)
    assert abs(float(gradient)) > 1e-8
    for epsilon in (1e-3, 1e-4):
        difference = (trajectory(point + epsilon) - trajectory(point - epsilon)) / (2 * epsilon)
        np.testing.assert_allclose(gradient, difference, rtol=2e-3, atol=1e-8)
    driver = CheckpointedIntegrator(prepared, terminal_objective=objective)
    result = driver.value_and_grad(controlled(point), n_steps=2)
    replay = jax.vjp(controlled, point)[1](result.gradient)[0]
    np.testing.assert_allclose(replay, gradient, rtol=1e-7, atol=1e-9)
    with pytest.raises(ValueError, match="temp, salt, and momentum"):
        prepared.with_bathymetry(initial, depth, conserve=("temp",))


def test_vertical_grid_operator_budgets_derivatives_rest_and_replay():
    import jax
    import jax.numpy as jnp
    from bathymetry_helpers import make_simulation
    from veros import runtime_settings
    from veros.checkpointing import CheckpointedIntegrator

    object.__setattr__(runtime_settings, "diskless_mode", True)
    object.__setattr__(runtime_settings, "linear_solver", "scipy_jax")
    simulation = make_simulation(wind=0, kappaH_0=0.0, enable_hor_friction=True, A_h=100.0)
    prepared = simulation.prepare_integrator(record_diagnostics=())
    carry = prepared.pack(simulation.state)
    carry = prepared.with_bathymetry(carry, jnp.full((8, 10), 360.0))
    pattern = jnp.array([0.1, -0.2, 0.15, -0.05])

    def controlled(z):
        return prepared.with_vertical_grid(carry, widths=carry.variables.dzt * (1 + z * pattern))

    updated = jax.jit(controlled)(jnp.asarray(0.1))
    prepared.check_status(updated)
    rng = np.random.default_rng(24)
    moving = prepared.replace_variables(
        carry,
        u=0.01 * jnp.asarray(rng.normal(size=carry.variables.u.shape)) * carry.variables.maskU[..., None],
        du=1e-5 * jnp.asarray(rng.normal(size=carry.variables.du.shape)) * carry.variables.maskU[..., None],
    )
    moving_result = prepared.with_vertical_grid(moving, widths=carry.variables.dzt * (1 + 0.1 * pattern))
    for name, grid in (("temp", "T"), ("salt", "T"), ("u", "U"), ("v", "V"), ("dtemp", "T"), ("du", "U")):
        old = getattr(moving.variables, name)[2:-2, 2:-2]
        new = getattr(moving_result.variables, name)[2:-2, 2:-2]
        weights = carry.variables.dzt * getattr(carry.variables, "hfac" + grid)[2:-2, 2:-2]
        new_weights = updated.variables.dzt * getattr(updated.variables, "hfac" + grid)[2:-2, 2:-2]
        np.testing.assert_allclose(
            jnp.sum(old * weights[..., None], axis=2), jnp.sum(new * new_weights[..., None], axis=2), atol=1e-10
        )
    final, _ = prepared.advance(updated, n_steps=3)
    prepared.check_status(final)
    for name in ("u", "v", "w"):
        np.testing.assert_allclose(getattr(final.variables, name), 0, atol=1e-12)
    np.testing.assert_allclose(final.variables.temp, updated.variables.temp, atol=1e-12)
    np.testing.assert_allclose(final.variables.salt, updated.variables.salt, atol=1e-12)

    def objective(c):
        return jnp.sum(c.variables.temp[2:-2, 2:-2, :, c.variables.tau] * c.variables.zt)

    def function(z):
        return objective(prepared.advance(controlled_moving(z), n_steps=2)[0])

    def controlled_moving(z):
        return prepared.with_vertical_grid(moving, widths=carry.variables.dzt * (1 + z * pattern))

    point = jnp.asarray(0.1)
    gradient = jax.grad(function)(point)
    epsilon = 1e-4
    difference = (function(point + epsilon) - function(point - epsilon)) / (2 * epsilon)
    np.testing.assert_allclose(gradient, difference, rtol=2e-3, atol=1e-8)
    driver = CheckpointedIntegrator(prepared, terminal_objective=objective)
    initial = controlled_moving(point)
    result = driver.value_and_grad(initial, n_steps=2)
    # The host checkpoint pullback composes with the public remapping transform.
    seed = result.gradient
    replay = jax.vjp(controlled_moving, point)[1](seed)[0]
    np.testing.assert_allclose(replay, gradient, rtol=1e-8, atol=1e-8)
    invalid = prepared.with_vertical_grid(carry, widths=jnp.array([0.0, 100.0, 100.0, 200.0]))
    assert int(invalid.failure_code) == 5
    with pytest.raises(RuntimeError, match="vertical grid"):
        prepared.check_status(invalid)


@pytest.mark.parametrize("partial_cells", (False, True))
def test_positive_widths_with_negative_staggered_spacing_are_rejected(partial_cells):
    import jax
    import jax.numpy as jnp
    from bathymetry_helpers import make_simulation
    from veros import runtime_settings

    object.__setattr__(runtime_settings, "diskless_mode", True)
    object.__setattr__(runtime_settings, "linear_solver", "scipy_jax")
    simulation = make_simulation(partial_cells=partial_cells, wind=0)
    prepared = simulation.prepare_integrator(record_diagnostics=())
    initial = prepared.pack(simulation.state)
    updated = jax.jit(lambda widths: prepared.with_vertical_grid(initial, widths=widths))(
        jnp.array([20.0, 340.0, 20.0, 20.0])
    )
    assert int(updated.failure_code) == 5
    for name in ("dzt", "dzw", "zt", "zw"):
        np.testing.assert_array_equal(getattr(updated.variables, name), getattr(initial.variables, name))
    for name in ("temp", "salt", "Nsqr"):
        assert np.all(np.isfinite(getattr(updated.variables, name)))
    with pytest.raises(RuntimeError, match="staggered"):
        prepared.check_status(updated)


def test_vertical_restoring_targets_follow_physical_coordinates_and_hook_changes_require_reprepare():
    import jax.numpy as jnp
    from benchmarks.ad_demos.adjusted_basin import AdjustedBasinSetup
    from veros import runtime_settings
    from veros.integration import CompiledCompatibilityError

    object.__setattr__(runtime_settings, "diskless_mode", True)
    object.__setattr__(runtime_settings, "linear_solver", "scipy_jax")
    simulation = AdjustedBasinSetup()
    simulation.setup()
    prepared = simulation.prepare_integrator(record_diagnostics=())
    initial = prepared.pack(simulation.state)
    carry = prepared.with_vertical_grid(initial, widths=jnp.array([450.0, 510.0, 500.0, 540.0]))
    vs = carry.variables
    expected = (
        10
        + 0.002 * vs.zt[None, None, :]
        + 0.15 * jnp.cos(jnp.pi * vs.yt[None, :, None] / 200000)
        + 0.01 * jnp.sin(2 * jnp.pi * vs.xt[:, None, None] / 160000)
    ) * vs.maskT
    np.testing.assert_allclose(vs.basin_temperature_target, expected, atol=1e-12)
    assert float(jnp.max(abs(vs.basin_temperature_target - initial.variables.basin_temperature_target))) > 0.01
    simulation.rebuild_vertical_forcing = lambda state: None
    with pytest.raises(CompiledCompatibilityError, match="rebuild_vertical_forcing"):
        prepared.with_vertical_grid(initial, widths=initial.variables.dzt)


@pytest.mark.parametrize("stretch", (False, True))
def test_nonbinary_coincident_faces_preserve_constant_field_shape_derivative(stretch):
    import jax
    import jax.numpy as jnp

    rng = np.random.default_rng(781)
    widths = jnp.asarray(rng.uniform(10, 400, size=(6, 15)))
    direction = jnp.asarray(rng.normal(size=widths.shape))
    if not stretch:
        direction = direction - direction.mean(axis=-1, keepdims=True)
    histories = jnp.array([0.2, 1.0, 35.0])
    values = jnp.ones(widths.shape + (3,)) * histories
    remap = remap_column_inventories if stretch else remap_cell_averages
    function = jax.jit(lambda w: remap(values, widths, w, axis=1))
    _, tangent = jax.jvp(function, (widths,), (direction,))
    if stretch:
        expected = -direction.sum(axis=-1) / widths.sum(axis=-1)
        expected = jnp.broadcast_to(expected[:, None, None] * histories, tangent.shape)
    else:
        expected = jnp.zeros_like(tangent)
    np.testing.assert_allclose(tangent, expected, atol=2e-13, rtol=1e-11)
    seed = jnp.asarray(rng.normal(size=values.shape))
    transpose = jax.vjp(function, widths)[1](seed)[0]
    np.testing.assert_allclose(jnp.vdot(seed, tangent), jnp.vdot(transpose, direction), atol=1e-12)
    for epsilon in (1e-4, 1e-5):
        fd = (function(widths + epsilon * direction) - function(widths - epsilon * direction)) / (2 * epsilon)
        np.testing.assert_allclose(fd, tangent, atol=2e-7, rtol=1e-6)
