"""Occupied-volume Redi/GM budgets, tensor structure, and geometry products."""

import numpy as np
import pytest


def neutral_basin(*, partial=True, policy=True, eos=1, stepped=False):
    import jax.numpy as jnp
    from bathymetry_helpers import make_simulation
    from veros import runtime_settings
    from veros.core.numerics import calc_initial_conditions_kernel
    from veros.core.utilities import enforce_boundaries

    object.__setattr__(runtime_settings, "diskless_mode", True)
    object.__setattr__(runtime_settings, "linear_solver", "scipy_jax")
    indices = None
    if stepped:
        indices = np.ones((8, 10), dtype=int)
        indices[:2, :3] = 2
        indices[6:, 7:] = 3
        indices[0, 0] = 0
    simulation = make_simulation(
        partial_cells=partial,
        bottom_indices=indices,
        wind=0,
        enable_neutral_diffusion=True,
        enable_skew_diffusion=True,
        enable_neutral_nonstable_zero_slope=policy,
        eq_of_state_type=eos,
        K_iso_0=500,
        K_iso_steep=0,
        K_gm_0=500,
    )
    state = simulation.state
    with state.variables.unlock():
        vs = state.variables
        pattern = jnp.sin(jnp.arange(8)[:, None] + 2 * jnp.arange(10)[None, :])
        vs.temp = vs.temp.at[2:-2, 2:-2].add(0.002 * pattern[..., None, None] * vs.maskT[2:-2, 2:-2, :, None])
        vs.temp = enforce_boundaries(vs.temp, False)
        vs.update(calc_initial_conditions_kernel(state))
        vs.K_iso = jnp.full_like(vs.K_iso, 500)
        vs.K_gm = jnp.full_like(vs.K_gm, 500)
    prepared = simulation.prepare_integrator(record_diagnostics=())
    initial = prepared.pack(state)
    if partial:
        depth = jnp.sum(initial.variables.maskT[2:-2, 2:-2] * initial.variables.dzt, axis=2)
        depth = jnp.where(depth > 0, depth - 20 - 10 * pattern, 0)
        initial = prepared.with_bathymetry(initial, depth)
        prepared.check_status(initial)
    return prepared, initial


@pytest.mark.parametrize("iso", (False, True))
@pytest.mark.parametrize("policy", (False, True))
def test_partial_neutral_frozen_tensor_and_implicit_budgets(iso, policy):
    import jax
    import jax.numpy as jnp
    from veros.core.isoneutral.isoneutral import isoneutral_diffusion_pre
    from veros.core.isoneutral.diffusion import _calc_tracer_fluxes, _calc_explicit_part, isoneutral_diffusion_tracer
    from veros.routines import tracing_context

    prepared, initial = neutral_basin(policy=policy, stepped=True)
    state = prepared._make_state(initial.variables)
    with state.variables.unlock():
        state.variables.update(isoneutral_diffusion_pre(state))
    vs = state.variables
    mask = np.asarray(vs.maskT[2:-2, 2:-2], dtype=bool)
    indices = np.flatnonzero(mask.ravel())
    volumes = np.asarray(vs.area_t[2:-2, 2:-2, None] * vs.dzt * vs.hfacT[2:-2, 2:-2]).ravel()[indices]

    def tracer(vector):
        values = jnp.zeros(mask.size).at[indices].set(vector).reshape(mask.shape)
        full = jnp.zeros_like(vs.maskT, dtype=vs.temp.dtype).at[2:-2, 2:-2].set(values)
        return jnp.broadcast_to(full[..., None], vs.temp.shape)

    def generator(vector):
        tr = tracer(vector)
        with tracing_context():
            fe, fn, ft = _calc_tracer_fluxes(
                state,
                tr,
                vs.K_iso if iso else jnp.zeros_like(vs.K_iso),
                jnp.zeros_like(vs.K_gm) if iso else vs.K_gm,
                iso=iso,
            )
            if iso:
                ft = ft.at[:, :, :-1].add(
                    vs.K_33[:, :, :-1]
                    * (tr[:, :, 1:, vs.tau] - tr[:, :, :-1, vs.tau])
                    / vs.dzw[None, None, :-1]
                    * vs.maskW[:, :, :-1]
                )
            tendency = _calc_explicit_part(state, fe, fn, ft)
        return tendency[2:-2, 2:-2].ravel()[indices]

    matrix = np.asarray(jax.jit(jax.vmap(generator))(jnp.eye(len(indices)))).T
    norm = np.linalg.norm(matrix)
    np.testing.assert_allclose(matrix @ np.ones(len(indices)), 0, atol=1e-14 * norm)
    assert np.linalg.norm(volumes @ matrix) < 1e-14 * np.linalg.norm(volumes) * norm
    scaled = np.sqrt(volumes)[:, None] * matrix / np.sqrt(volumes)[None, :]
    if iso:
        assert np.linalg.norm(scaled - scaled.T) < 1e-12 * np.linalg.norm(scaled)
    else:
        assert np.linalg.norm(scaled + scaled.T) < 1e-12 * np.linalg.norm(scaled)
    assert np.linalg.eigvalsh((scaled + scaled.T) / 2)[-1] < 1e-12 * np.linalg.norm(scaled)

    # Test the actual explicit/implicit update independently of the generator.
    for values in (jnp.ones(len(indices)), jnp.asarray(np.random.default_rng(6).normal(size=len(indices)))):
        tr = tracer(values)
        with tracing_context():
            updated, *_ = isoneutral_diffusion_tracer(
                state, tr, jnp.zeros_like(vs.maskT, dtype=vs.temp.dtype), iso=iso, skew=not iso
            )
        change = (updated[2:-2, 2:-2, :, vs.taup1] - tr[2:-2, 2:-2, :, vs.taup1]).ravel()[indices]
        assert abs(float(jnp.vdot(volumes, change))) < 1e-12 * float(jnp.sum(volumes * jnp.maximum(abs(values), 1)))
        if bool(jnp.all(values == 1)):
            np.testing.assert_allclose(change, 0, atol=1e-12)


@pytest.mark.parametrize("policy", (False, True))
def test_partial_neutral_native_geometry_derivative_and_checkpoint(policy):
    import jax
    import jax.numpy as jnp
    from veros.checkpointing import CheckpointedIntegrator

    prepared, initial = neutral_basin(policy=policy, eos=3)
    pattern = jnp.sin(jnp.arange(8)[:, None] + 2 * jnp.arange(10)[None, :])
    depth = initial.variables.ht[2:-2, 2:-2]

    def controls(z):
        return prepared.with_bathymetry(initial, depth + z * pattern)

    def objective(carry):
        vs = carry.variables
        return jnp.sum(vs.temp[2:-2, 2:-2, :, vs.tau] * pattern[..., None])

    def response(z):
        return objective(prepared.advance(controls(z), n_steps=3)[0])

    point = jnp.asarray(0.2)
    final = prepared.advance(controls(point), n_steps=3)[0]
    prepared.check_status(final)
    gradient = jax.grad(response)(point)
    tangent = jax.jvp(response, (point,), (jnp.asarray(1.0),))[1]
    np.testing.assert_allclose(tangent, gradient, rtol=1e-8, atol=1e-10)
    assert abs(float(gradient)) > 1e-10
    for epsilon in (1e-2, 1e-3):
        finite = (response(point + epsilon) - response(point - epsilon)) / (2 * epsilon)
        np.testing.assert_allclose(gradient, finite, rtol=2e-3, atol=1e-9)
    checkpointed = CheckpointedIntegrator(prepared, terminal_objective=objective).value_and_grad(
        controls(point), n_steps=3
    )
    replay = jax.vjp(controls, point)[1](checkpointed.gradient)[0]
    np.testing.assert_allclose(replay, gradient, rtol=1e-8, atol=1e-10)


def test_partial_neutral_full_cell_trajectory_parity():
    finals = []
    for partial in (False, True):
        prepared, initial = neutral_basin(partial=partial)
        if partial:
            full_depth = np.sum(initial.variables.maskT[2:-2, 2:-2] * initial.variables.dzt, axis=2)
            initial = prepared.with_bathymetry(initial, full_depth)
        final = prepared.advance(initial, n_steps=3)[0]
        prepared.check_status(final)
        finals.append(final)
    for name in finals[0].variables.fields():
        np.testing.assert_allclose(
            finals[0].variables.get(name), finals[1].variables.get(name), rtol=1e-10, atol=1e-10, err_msg=name
        )


@pytest.mark.parametrize("partial", (False, True))
@pytest.mark.parametrize("iso", (False, True))
def test_neutral_projected_work_matches_the_frozen_tracer_functional(partial, iso):
    from copy import copy
    import jax.numpy as jnp
    from veros.core.isoneutral.isoneutral import isoneutral_diffusion_pre
    from veros.core.isoneutral.diffusion import isoneutral_diffusion_kernel
    from veros.routines import tracing_context

    prepared, initial = neutral_basin(partial=partial, stepped=True)
    state = prepared._make_state(initial.variables)
    state._settings = copy(state.settings)
    with state.settings.unlock():
        state.settings.enable_conserve_energy = True
    with state.variables.unlock():
        vs = state.variables
        with tracing_context():
            vs.update(isoneutral_diffusion_pre(state))
        functional = jnp.asarray(np.random.default_rng(64).normal(size=vs.maskT.shape)) * vs.maskT
        vs.temp = jnp.broadcast_to(functional[..., None], vs.temp.shape)
        vs.int_drhodT = jnp.broadcast_to(functional[..., None], vs.int_drhodT.shape)
        vs.dtemp_iso = jnp.zeros_like(vs.dtemp_iso)
        vs.P_diss_iso = jnp.zeros_like(vs.P_diss_iso)
        vs.P_diss_skew = jnp.zeros_like(vs.P_diss_skew)
    with tracing_context():
        result = isoneutral_diffusion_kernel(state, vs.temp, istemp=True, iso=iso)
    fraction = vs.hfacT if partial else vs.maskT
    volume = vs.area_t[..., None] * vs.dzt * fraction
    work = -state.settings.grav / state.settings.rho_0 * functional * result.dtemp_iso * volume
    weights = (vs.area_t[..., None] * vs.maskW * vs.dzw).at[:, :, -1].multiply(0.5)
    dissipation = result.P_diss_iso if iso else result.P_diss_skew
    actual = float(jnp.sum((dissipation * weights)[2:-2, 2:-2]))
    expected = float(jnp.sum(work[2:-2, 2:-2]))
    scale = max(float(jnp.sum(jnp.abs(work[2:-2, 2:-2]))), 1)
    assert abs(actual - expected) < 1e-10 * scale
    if iso:
        assert expected > 0
    else:
        assert abs(expected) < 1e-10 * scale
