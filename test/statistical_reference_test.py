"""Refined periodic targets require independent ordinary forward agreement."""

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("jax_backend")


def test_native_periodic_reference_validates_all_moments_and_rejects_short_burnin():
    import jax.numpy as jnp
    from veros import veros_routine, runtime_settings
    from veros.core.numerics import calc_initial_conditions_kernel
    from veros.routines import tracing_context
    from bathymetry_helpers import make_simulation
    from benchmarks.ad_demos.statistical_reference import periodic_reference
    from scipy.sparse.linalg import eigs

    object.__setattr__(runtime_settings, "diskless_mode", True)
    object.__setattr__(runtime_settings, "linear_solver", "scipy_jax")
    indices = np.ones((8, 10), int)
    indices[4, :] = 0
    simulation = make_simulation(
        bottom_indices=indices,
        wind=0,
        enable_tempsalt_sources=True,
        enable_ray_friction=True,
        r_ray=1 / 3000,
    )
    period = 24

    @veros_routine(supports_compilation=True)
    def forcing(state):
        vs = state.variables
        target = (
            10 + 0.001 * vs.ht[..., None] + 0.01 * jnp.cos(2 * jnp.pi * vs.time / (period * state.settings.dt_tracer))
        )
        vs.temp_source = (target - vs.temp[..., vs.tau]) * vs.maskT / 3000
        vs.salt_source = (35 - vs.salt[..., vs.tau]) * vs.maskT / 3000

    @veros_routine(supports_compilation=True)
    def rebuild(state):
        pass

    simulation.set_forcing = forcing
    simulation.rebuild_wet_forcing = rebuild
    prepared = simulation.prepare_integrator(record_diagnostics=())
    initial = prepared.pack(simulation.state)
    channel = jnp.zeros((8, 10), bool).at[4, 5].set(True)
    candidate = prepared.with_wet_geometry(initial, opening_m=75.0, channel=channel)

    def observe(carry):
        return carry.variables.temp[6, 7, -1, carry.variables.tau]

    spectra = []

    def stability(operator, *, k):
        spectra.append((operator.shape, k))
        return eigs(
            operator,
            k=k,
            which="LM",
            tol=1e-6,
            maxiter=1000,
            ncv=min(120, operator.shape[0]),
            v0=np.random.default_rng(19).normal(size=operator.shape[0]),
        )

    options = dict(
        observe=observe,
        thresholds=jnp.asarray(10.08),
        period_steps=period,
        sample_stride=6,
        window_steps=period,
        burnin_steps=20 * period,
        n_windows=2,
        seed=97,
        ram_checkpoints=32,
        stability_solver=stability,
    )
    values, audit = periodic_reference(prepared, candidate, **options)
    assert values.shape == (3,) and np.isfinite(values).all()
    assert spectra and all(count == 4 for _, count in spectra)
    assert values[1] > 1e-6 and 0 < values[2] < 1
    assert audit["independent_statistics_passed"] and audit["starting_state_passed"]
    assert audit["phase_samples"] == 4 and audit["independent_seeds"] == [97, 98, 99, 100]
    assert np.asarray(audit["independent_forward_statistics"]).shape == (4, 3)
    perturbed = prepared.replace_variables(
        candidate, temp=candidate.variables.temp + 5 * candidate.variables.maskT[..., None]
    )
    state = prepared._make_state(perturbed.variables)
    with tracing_context(), state.variables.unlock():
        state.variables.update(calc_initial_conditions_kernel(state))
    perturbed = perturbed._replace(variables=state.variables)
    with pytest.raises(RuntimeError, match="forward agreement"):
        periodic_reference(prepared, perturbed, **dict(options, burnin_steps=0, n_windows=1))
