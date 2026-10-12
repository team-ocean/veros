"""Conservation, topology, and operator derivatives for partial bottom cells."""

from copy import copy

import numpy as np
import pytest

from veros import runtime_settings


@pytest.fixture(autouse=True)
def options():
    if runtime_settings.backend != "jax":
        pytest.skip("bathymetry integration requires JAX")
    previous = runtime_settings.diskless_mode, runtime_settings.linear_solver
    object.__setattr__(runtime_settings, "diskless_mode", True)
    object.__setattr__(runtime_settings, "linear_solver", "scipy_jax")
    yield
    object.__setattr__(runtime_settings, "diskless_mode", previous[0])
    object.__setattr__(runtime_settings, "linear_solver", previous[1])


def test_full_cells_match_existing_model():
    from benchmarks.ad_demos.bathymetry import make_simulation

    finals = []
    for partial in (False, True):
        sim = make_simulation(partial_cells=partial)
        prepared = sim.prepare_integrator(record_diagnostics=())
        initial = prepared.pack(sim.state)
        if partial:
            initial = prepared.with_bathymetry(initial, initial.variables.ht[2:-2, 2:-2])
        final, _ = prepared.advance(initial, n_steps=4)
        prepared.check_status(final)
        finals.append(final)
    for name in finals[0].variables.fields():
        np.testing.assert_allclose(
            finals[0].variables.get(name), finals[1].variables.get(name), rtol=1e-10, atol=1e-10, err_msg=name
        )


def test_geometry_bounds_faces_and_purity():
    import jax
    import jax.numpy as jnp
    from benchmarks.ad_demos.bathymetry import BathymetryBasin

    basin = BathymetryBasin()
    initial = basin.initial
    carry = basin.controls(jnp.array([0.3]))
    vs = carry.variables
    np.testing.assert_allclose(vs.ht[2:-2, 2:-2], basin.reference + 6 * basin.sill)
    np.testing.assert_allclose(vs.hfacU[:-1], np.minimum(vs.hfacT[:-1], vs.hfacT[1:]) * vs.maskU[:-1])
    np.testing.assert_allclose(vs.hfacV[:, :-1], np.minimum(vs.hfacT[:, :-1], vs.hfacT[:, 1:]) * vs.maskV[:, :-1])
    np.testing.assert_array_equal(vs.kbot, initial.variables.kbot)
    np.testing.assert_array_equal(vs.maskT, initial.variables.maskT)
    np.testing.assert_array_equal(initial.variables.ht, basin.sim.state.variables.ht)
    np.testing.assert_allclose(initial.variables.hfacT, initial.variables.maskT)
    lower, upper = basin.prepared.bathymetry_bounds(carry)
    for invalid in (upper + 1, lower - 1, jnp.full(upper.shape, jnp.nan)):
        failed = jax.jit(basin.prepared.with_bathymetry)(initial, invalid)
        assert int(failed.failure_code) == 4
        with pytest.raises(RuntimeError, match="bathymetry"):
            basin.prepared.check_status(failed)
        frozen, _ = basin.prepared.advance(failed, n_steps=2)
        assert int(frozen.variables.itt) == 0
    with pytest.raises(ValueError, match="interior shape"):
        basin.prepared.with_bathymetry(initial, jnp.ones(3))
    with pytest.raises(ValueError, match="minimum_fraction"):
        basin.prepared.bathymetry_bounds(initial, minimum_fraction=0)


def test_resting_basin_stays_at_rest():
    import jax.numpy as jnp
    from benchmarks.ad_demos.bathymetry import make_simulation

    sim = make_simulation(wind=0, kappaH_0=0.0)
    prepared = sim.prepare_integrator(record_diagnostics=())
    carry = prepared.pack(sim.state)
    depth = 335 + 30 * jnp.arange(8)[:, None] / 7 + jnp.zeros((8, 10))
    initial = prepared.with_bathymetry(carry, depth)
    final, _ = prepared.advance(initial, n_steps=8)
    prepared.check_status(final)
    for name in ("u", "v", "w"):
        np.testing.assert_allclose(final.variables.get(name), 0, atol=1e-12)
    for name in ("temp", "salt"):
        np.testing.assert_allclose(final.variables.get(name), initial.variables.get(name), atol=1e-12)


@pytest.mark.parametrize("cyclic", (False, True))
def test_mixed_bottom_indices_land_and_periodic_faces(cyclic):
    from benchmarks.ad_demos.bathymetry import make_simulation

    indices = np.broadcast_to(np.arange(10) % 5, (8, 10)).copy()
    sim = make_simulation(bottom_indices=indices, cyclic_x=cyclic, wind=0, kappaH_0=0.0)
    prepared = sim.prepare_integrator(record_diagnostics=())
    carry = prepared.pack(sim.state)
    lower, upper = prepared.bathymetry_bounds(carry)
    depth = (lower + upper) / 2
    carry = prepared.with_bathymetry(carry, depth)
    vs = carry.variables
    np.testing.assert_allclose(vs.ht[2:-2, 2:-2], depth)
    np.testing.assert_allclose(vs.hu[:-1], np.minimum(vs.ht[:-1], vs.ht[1:]) * np.any(vs.maskU[:-1], axis=2))
    np.testing.assert_allclose(
        vs.hv[:, :-1], np.minimum(vs.ht[:, :-1], vs.ht[:, 1:]) * np.any(vs.maskV[:, :-1], axis=2)
    )
    if cyclic:
        np.testing.assert_allclose(vs.hfacT[:2], vs.hfacT[-4:-2])
        np.testing.assert_allclose(vs.hfacU[-2:], vs.hfacU[2:4])
    invalid = depth.at[0, 0].set(1)
    assert int(prepared.with_bathymetry(carry, invalid).failure_code) == 4
    final, _ = prepared.advance(carry, n_steps=3)
    prepared.check_status(final)
    np.testing.assert_allclose(final.variables.u, 0, atol=1e-12)
    np.testing.assert_allclose(final.variables.v, 0, atol=1e-12)
    np.testing.assert_allclose(final.variables.temp, carry.variables.temp, atol=1e-12)


@pytest.mark.parametrize("superbee", (False, True))
def test_tracer_fluxes_and_vertical_mixing_conserve_content(superbee):
    import jax.numpy as jnp
    from benchmarks.ad_demos.bathymetry import BathymetryBasin
    from veros.core import diffusion, thermodynamics
    from veros.core.geometry import cell_thickness
    from veros.routines import tracing_context

    basin = BathymetryBasin()
    carry = basin.controls(jnp.zeros(1))
    state = basin.prepared._make_state(carry.variables)
    state._settings = copy(state.settings)
    with state.settings.unlock():
        state.settings.enable_superbee_advection = superbee
    rng = np.random.default_rng(17)
    vs = state.variables
    tracer = jnp.asarray(rng.uniform(0, 1, vs.maskT.shape)) * vs.maskT
    with vs.unlock():
        for name, mask in (("u", vs.maskU), ("v", vs.maskV), ("w", vs.maskW)):
            setattr(vs, name, jnp.asarray(rng.uniform(-1e-3, 1e-3, getattr(vs, name).shape)) * mask[..., None])
        vs.temp = jnp.broadcast_to(tracer[..., None], vs.temp.shape)
        vs.salt = jnp.broadcast_to(tracer[..., None], vs.salt.shape)
        vs.kappaH = jnp.full(vs.maskT.shape, 1e-3)
    volume = vs.area_t[2:-2, 2:-2, None] * cell_thickness(state)[2:-2, 2:-2] * vs.maskT[2:-2, 2:-2]
    with tracing_context():
        advected = thermodynamics.advect_tracer(state, tracer)
        diffused, _, _ = diffusion.horizontal_diffusion(state, tracer, 10.0)
        mixed = thermodynamics.vertmix_tempsalt(state)
    for tendency in (advected, diffused):
        contribution = np.asarray(tendency[2:-2, 2:-2] * volume)
        assert abs(contribution.sum()) < 1e-12 * np.abs(contribution).sum()
    np.testing.assert_allclose(
        jnp.sum(mixed.temp[2:-2, 2:-2, :, vs.taup1] * volume), jnp.sum(tracer[2:-2, 2:-2] * volume), rtol=1e-12
    )


def test_variable_operator_derivatives():
    import jax
    import jax.numpy as jnp
    from benchmarks.ad_demos.bathymetry import BathymetryBasin
    from veros.routines import tracing_context

    basin = BathymetryBasin()
    rhs = jnp.zeros(basin.initial.variables.ht.shape).at[3:8, 4:9].set(1e-6)
    seed = jnp.sin(jnp.arange(rhs.size).reshape(rhs.shape))

    def objective(z):
        state = basin.prepared._make_state(basin.controls(z).variables)
        with tracing_context():
            solution = state._linear_solver.solve(state, rhs, jnp.zeros_like(rhs))
        return jnp.sum(solution * seed)

    point = jnp.array([0.2])
    gradient = jax.jit(jax.grad(objective))(point)
    _, tangent = jax.jit(lambda z: jax.jvp(objective, (z,), (jnp.ones_like(z),)))(point)
    assert np.isfinite(gradient).all() and abs(float(gradient[0])) > 1e-8
    np.testing.assert_allclose(tangent, gradient[0], rtol=1e-8)
    evaluate = jax.jit(objective)
    for epsilon in (1e-3, 1e-4):
        finite = (evaluate(point + epsilon) - evaluate(point - epsilon)) / (2 * epsilon)
        np.testing.assert_allclose(gradient[0], finite, rtol=1e-5, atol=1e-10)


@pytest.mark.parametrize("implicit", (False, True))
def test_vertical_momentum_mixing_conserves_column_transport(implicit):
    import jax.numpy as jnp
    from benchmarks.ad_demos.bathymetry import BathymetryBasin
    from veros.core import friction
    from veros.core.geometry import cell_thickness
    from veros.routines import tracing_context

    basin = BathymetryBasin()
    state = basin.prepared._make_state(basin.controls(jnp.zeros(1)).variables)
    vs = state.variables
    rng = np.random.default_rng(23)
    with vs.unlock():
        vs.kappaM = jnp.full(vs.maskT.shape, 1e-3)
        vs.u = jnp.asarray(rng.uniform(-0.1, 0.1, vs.u.shape)) * vs.maskU[..., None]
        vs.v = jnp.asarray(rng.uniform(-0.1, 0.1, vs.v.shape)) * vs.maskV[..., None]
    with tracing_context():
        result = (friction.implicit_vert_friction if implicit else friction.explicit_vert_friction)(state)
    for grid, name, tendency in (("U", "u", "du_mix"), ("V", "v", "dv_mix")):
        thickness = cell_thickness(state, grid) * getattr(vs, "mask" + grid)
        np.testing.assert_allclose(jnp.sum(getattr(result, tendency) * thickness, axis=2), 0, atol=1e-12)


def test_unsupported_physics_and_solver_are_rejected():
    from benchmarks.ad_demos.bathymetry import BathymetryBasin
    from veros.settings import check_setting_conflicts

    basin = BathymetryBasin()
    settings = copy(basin.sim.state.settings)
    with settings.unlock():
        settings.enable_tke = True
    with pytest.raises(ValueError, match="enable_tke"):
        check_setting_conflicts(settings)
    with pytest.raises(ValueError, match="variable-coefficient"):
        basin.sim.prepare_integrator(linear_solver="petsc")
    with pytest.raises(ValueError, match="records"):
        basin.sim.prepare_integrator(record_diagnostics=("energy",))


def test_stress_and_drag_in_a_partial_surface_cell():
    import jax
    import jax.numpy as jnp
    from benchmarks.ad_demos.bathymetry import make_simulation
    from veros.core import friction, momentum
    from veros.routines import tracing_context

    sim = make_simulation(bottom_indices=np.full((8, 10), 4), r_quad_bot=0.003)
    prepared = sim.prepare_integrator(record_diagnostics=())
    full = prepared.pack(sim.state)
    lower, upper = prepared.bathymetry_bounds(full)
    partial = prepared.with_bathymetry(full, (lower + upper) / 2)

    def evaluate(carry, speed):
        state = prepared._make_state(carry.variables)
        vs = state.variables
        with vs.unlock():
            vs.u = speed * jnp.broadcast_to(vs.maskU[..., None], vs.u.shape)
            vs.v = speed * jnp.broadcast_to(vs.maskV[..., None], vs.v.shape)
        with tracing_context():
            return momentum.tend_tauxyf(state), friction.quadratic_bottom_friction(state)

    full_wind, full_drag = evaluate(full, 0.1)
    partial_wind, partial_drag = evaluate(partial, 0.1)
    # Equal applied stresses impart equal column transport increments.
    for grid, name in (("U", "du"), ("V", "dv")):
        fraction = getattr(partial.variables, "hfac" + grid)[2:-2, 2:-2]
        np.testing.assert_allclose(
            getattr(partial_wind, name)[2:-2, 2:-2, :, full.variables.tau] * fraction,
            getattr(full_wind, name)[2:-2, 2:-2, :, full.variables.tau],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            getattr(partial_drag, name + "_mix")[2:-2, 2:-2] * fraction,
            getattr(full_drag, name + "_mix")[2:-2, 2:-2],
            atol=1e-12,
        )

    def drag_objective(speed):
        _, drag = evaluate(partial, speed)
        return jnp.sum(drag.du_mix) + jnp.sum(drag.dv_mix)

    np.testing.assert_allclose(jax.grad(drag_objective)(0.0), 0, atol=1e-12)


def test_restart_preserves_candidate_geometry(tmp_path):
    import jax.numpy as jnp
    from benchmarks.ad_demos.bathymetry import BathymetryBasin
    from veros import restart
    from veros.io_tools import hdf5 as h5tools

    basin = BathymetryBasin()
    final, _ = basin.prepared.advance(basin.controls(jnp.array([0.4])), n_steps=2)
    basin.prepared.check_status(final)
    basin.prepared.install(final)
    filename = str(tmp_path / "bathymetry.h5")
    settings = basin.sim.state.settings
    with settings.unlock():
        settings.restart_output_filename = settings.restart_input_filename = filename
    object.__setattr__(runtime_settings, "diskless_mode", False)
    try:
        restart.write_restart(basin.sim.state, force=True)
        h5tools._wait_for_disk(filename)
    finally:
        object.__setattr__(runtime_settings, "diskless_mode", True)
    basin.prepared.install(basin.initial)
    restart.read_restart(basin.sim.state)
    for name in ("hfacT", "hfacU", "hfacV", "ht", "hu", "hv", "hur", "hvr", "temp", "u"):
        np.testing.assert_allclose(
            basin.sim.state.variables.get(name), final.variables.get(name), rtol=1e-12, atol=1e-12, err_msg=name
        )
    restored = basin.prepared.pack(basin.sim.state)
    resumed, _ = basin.prepared.advance(restored, n_steps=2)
    reference, _ = basin.prepared.advance(final, n_steps=2)
    np.testing.assert_allclose(resumed.variables.psi, reference.variables.psi, rtol=1e-12, atol=1e-12)
