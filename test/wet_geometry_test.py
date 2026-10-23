"""Conservative dry-column openings and independent whole-cell realizations."""

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("jax_backend")


def channel_basin(*, opened=False, declare_forcing=True, cyclic=False):
    import jax.numpy as jnp
    from bathymetry_helpers import make_simulation
    from veros import runtime_settings, veros_routine
    from veros.core.numerics import calc_initial_conditions_kernel

    object.__setattr__(runtime_settings, "diskless_mode", True)
    object.__setattr__(runtime_settings, "linear_solver", "scipy_jax")
    indices = np.ones((8, 10), dtype=int)
    indices[4, :] = 0
    if opened:
        indices[4, 5] = 4
    sim = make_simulation(bottom_indices=indices, wind=0, cyclic_x=cyclic)
    if declare_forcing:

        @veros_routine(supports_compilation=True)
        def rebuild(state):
            pass

        sim.rebuild_wet_forcing = rebuild
    with sim.state.variables.unlock():
        vs = sim.state.variables
        vs.temp = 10 * jnp.broadcast_to(vs.maskT[..., None], vs.temp.shape)
        vs.salt = 35 * jnp.broadcast_to(vs.maskT[..., None], vs.salt.shape)
        vs.update(calc_initial_conditions_kernel(sim.state))
    prepared = sim.prepare_integrator(record_diagnostics=())
    channel = jnp.zeros((8, 10), bool).at[4, 5].set(True)
    return sim, prepared, prepared.pack(sim.state), channel


def test_opening_preserves_signed_histories_and_nonnegative_tracers():
    import jax
    import jax.numpy as jnp

    sim, prepared, initial, channel = channel_basin()
    rng = np.random.default_rng(41)
    fields = {}
    for name, grid in [("u", "U"), ("v", "V"), ("du", "U"), ("dv", "V"), ("dtemp", "T"), ("dsalt", "T"), ("dHd", "T")]:
        if name not in initial.variables:
            continue
        fields[name] = (
            jnp.asarray(rng.normal(size=getattr(initial.variables, name).shape))
            * getattr(initial.variables, "mask" + grid)[..., None]
        )
    initial = prepared.replace_variables(initial, **fields)

    def control(z):
        return prepared.with_wet_geometry(initial, opening_m=z, channel=channel)

    result = jax.jit(control)(jnp.asarray(75.0))
    prepared.check_status(result)
    assert int(result.variables.kbot[6, 7]) == 4
    np.testing.assert_allclose(result.variables.ht[6, 7], 75)
    for name, grid in [
        ("temp", "T"),
        ("salt", "T"),
        ("u", "U"),
        ("v", "V"),
        ("du", "U"),
        ("dv", "V"),
        ("dtemp", "T"),
        ("dsalt", "T"),
        ("dHd", "T"),
    ]:
        if name not in initial.variables:
            continue

        def inventory(c):
            vs = c.variables
            volume = (
                getattr(vs, "area_" + grid.lower())[2:-2, 2:-2, None] * vs.dzt * getattr(vs, "hfac" + grid)[2:-2, 2:-2]
            )
            return jnp.sum(getattr(vs, name)[2:-2, 2:-2] * volume[..., None], axis=(0, 1, 2))

        np.testing.assert_allclose(inventory(result), inventory(initial), rtol=1e-12, atol=1e-3, err_msg=name)
    assert bool(jnp.all(result.variables.salt[result.variables.maskT] > 0))
    assert bool(jnp.all(result.variables.temp[result.variables.maskT] > 0))
    closed = control(jnp.asarray(0.0))
    prepared.check_status(closed)
    for name in ("temp", "salt", "u", "v", "du", "dv", "dtemp", "dsalt", "kbot", "hfacT", "hfacU", "hfacV"):
        np.testing.assert_allclose(
            getattr(closed.variables, name), getattr(initial.variables, name), atol=1e-12, err_msg=name
        )
    for depth in (-1.0, 1e-300, 5.0, 405.0, float("nan")):
        invalid = control(jnp.asarray(depth))
        assert int(invalid.failure_code) == 6
        with pytest.raises(RuntimeError, match="wet geometry"):
            prepared.check_status(invalid)
        assert bool(jnp.all(jnp.isfinite(invalid.variables.temp)))


@pytest.mark.parametrize("cyclic", (False, True))
def test_channel_rest_and_discrete_realization_match_independent_setup(cyclic):
    import jax.numpy as jnp
    from veros.core.numerics import calc_initial_conditions_kernel
    from veros.routines import tracing_context

    sim, prepared, initial, channel = channel_basin(cyclic=cyclic)
    candidate = prepared.with_wet_geometry(initial, opening_m=75.0, channel=channel)
    final = prepared.advance(candidate, n_steps=3)[0]
    prepared.check_status(final)
    for name in ("u", "v", "w"):
        np.testing.assert_allclose(getattr(final.variables, name), 0, atol=1e-12)
    realized = prepared.with_wet_geometry(initial, opening_m=75.0, channel=channel, realized=True)
    other_sim, other, independent, _ = channel_basin(opened=True, cyclic=cyclic)
    # The independently constructed topology receives the declared pooled initialization.
    for name in ("temp", "salt"):
        concentration = realized.variables.get(name)[2, 2, -1, 0]
        independent = other.replace_variables(
            independent,
            **{
                name: jnp.broadcast_to(
                    independent.variables.maskT[..., None], getattr(independent.variables, name).shape
                )
                * concentration
            },
        )
    with tracing_context():
        update = calc_initial_conditions_kernel(other._make_state(independent.variables))
    independent = other.replace_variables(independent, **update._asdict())
    a, b = prepared.advance(realized, n_steps=3)[0], other.advance(independent, n_steps=3)[0]
    for c, p in ((a, prepared), (b, other)):
        p.check_status(c)
    for name in a.variables.fields():
        np.testing.assert_allclose(
            getattr(a.variables, name), getattr(b.variables, name), atol=1e-10, rtol=1e-10, err_msg=name
        )
    prepared.install(a)
    np.testing.assert_array_equal(sim.state.variables.kbot, a.variables.kbot)


@pytest.mark.parametrize("layers", (0, 2, 15))
def test_nonuniform_realization_keeps_selected_whole_layers_and_advances(layers):
    import jax
    import jax.numpy as jnp
    from bathymetry_helpers import make_simulation
    from veros import veros_routine

    indices = np.ones((8, 10), dtype=int)
    indices[4, :] = 0
    template = make_simulation(bottom_indices=indices, wind=0, nz=15)
    stock = jnp.asarray(
        [50.0, 70.0, 100.0, 140.0, 190.0, 240.0, 290.0, 340.0, 390.0, 440.0, 490.0, 540.0, 590.0, 640.0, 690.0]
    )[::-1]

    class Nonuniform(type(template)):
        @veros_routine
        def set_grid(self, state):
            super().set_grid(state)
            state.variables.dzt = stock

        @veros_routine(supports_compilation=True)
        def rebuild_wet_forcing(self, state):
            pass

    simulation = Nonuniform(override=dict(nz=15))
    simulation.setup()
    prepared = simulation.prepare_integrator(record_diagnostics=(), checkpoint=True)
    initial = prepared.pack(simulation.state)
    depth = jnp.sum(stock) - 0.3 * stock[0]
    initial = prepared.with_bathymetry(initial, jnp.where(jnp.asarray(indices) > 0, depth, 0))
    widths = stock * jnp.exp(0.15 * 0.316666666672144 * jnp.linspace(-1, 1, stock.size))
    widths *= jnp.sum(stock) / jnp.sum(widths)
    initial = prepared.with_vertical_grid(initial, widths=widths)
    channel = jnp.zeros((8, 10), bool).at[4, 5].set(True)
    opening = 100 * np.exp(0.5 * 0.40000005243559084) if layers == 2 else (0 if layers == 0 else float(jnp.sum(widths)))
    realized = jax.jit(
        lambda value: prepared.with_wet_geometry(initial, opening_m=value, channel=channel, realized=True)
    )(jnp.asarray(opening))
    prepared.check_status(realized)
    assert int(realized.variables.kbot[6, 7]) == (16 - layers if layers else 0)
    fractions = np.asarray(realized.variables.hfacT[6, 7])
    np.testing.assert_array_equal(fractions, np.r_[np.zeros(15 - layers), np.ones(layers)])
    old_wet = np.asarray(initial.variables.maskT[..., -1])
    np.testing.assert_allclose(realized.variables.ht[old_wet], initial.variables.ht[old_wet], rtol=0, atol=1e-10)
    expected_depth = float(jnp.sum(widths[-layers:])) if layers else 0
    np.testing.assert_allclose(realized.variables.ht[6, 7], expected_depth, rtol=0, atol=1e-10)
    final = prepared.advance(realized, n_steps=1, final_only=True)[0]
    prepared.check_status(final)
    assert int(final.variables.itt) == int(initial.variables.itt) + 1


def test_opening_geometry_products_and_checkpoint_chain():
    import jax
    import jax.numpy as jnp
    from veros.checkpointing import CheckpointedIntegrator, BinomialSchedule

    sim, prepared, initial, channel = channel_basin()
    pattern = jnp.sin(jnp.arange(8)[:, None] + jnp.arange(10)[None, :])
    initial = prepared.replace_variables(
        initial,
        temp=initial.variables.temp.at[2:-2, 2:-2].add(
            0.01 * pattern[..., None, None] * initial.variables.maskT[2:-2, 2:-2, :, None]
        ),
    )

    def control(z):
        return prepared.with_wet_geometry(initial, opening_m=z, channel=channel)

    def terminal(c):
        return jnp.sum(c.variables.temp[2:-2, 2:-2, :, c.variables.tau] * pattern[..., None])

    def objective(z):
        return terminal(prepared.advance(control(z), n_steps=2)[0])

    point = jnp.asarray(75.0)
    gradient = jax.grad(objective)(point)
    assert abs(float(gradient)) > 1e-8
    np.testing.assert_allclose(jax.jvp(objective, (point,), (jnp.asarray(1.0),))[1], gradient, rtol=1e-8)
    for epsilon in (0.01, 0.001):
        np.testing.assert_allclose(
            (objective(point + epsilon) - objective(point - epsilon)) / (2 * epsilon), gradient, rtol=2e-3, atol=1e-8
        )
    result = CheckpointedIntegrator(prepared, terminal_objective=terminal).value_and_grad(
        control(point), n_steps=2, schedule=BinomialSchedule(2, ram_checkpoints=1)
    )
    replay = jax.vjp(control, point)[1](result.gradient)[0]
    np.testing.assert_allclose(replay, gradient, rtol=1e-8, atol=1e-10)


def test_new_wet_forcing_and_channel_declarations_require_repreparation():
    import jax.numpy as jnp
    from veros import veros_routine
    from veros.integration import CompiledCompatibilityError

    sim, prepared, initial, channel = channel_basin(declare_forcing=False)
    with pytest.raises(ValueError, match="rebuild_wet_forcing"):
        prepared.with_wet_geometry(initial, opening_m=75.0, channel=channel)

    @veros_routine(supports_compilation=True)
    def rebuild(state):
        pass

    sim.rebuild_wet_forcing = rebuild
    with pytest.raises(CompiledCompatibilityError, match="rebuild_wet_forcing"):
        prepared.with_wet_geometry(initial, opening_m=75.0, channel=channel)
    prepared = sim.prepare_integrator(record_diagnostics=())
    with pytest.raises(ValueError, match="wet_geometry_channel"):
        prepared.with_wet_geometry(initial, opening_m=75.0, channel="missing")

    def channel_mask(state, name):
        assert name == "test"
        return channel

    channel_mask.supports_compilation = True
    sim.wet_geometry_channel = channel_mask
    prepared = sim.prepare_integrator(record_diagnostics=())
    result = prepared.with_wet_geometry(initial, opening_m=jnp.asarray(75.0), channel="test")
    prepared.check_status(result)
