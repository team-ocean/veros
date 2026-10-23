"""Native closure execution, energy parity, and conservative reference operators."""

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("jax_backend")


def closure_basin():
    import jax.numpy as jnp
    from veros import veros_routine
    from veros.variables import Variable
    from benchmarks.ad_demos.adjusted_basin import AdjustedBasinSetup

    class Basin(AdjustedBasinSetup):
        @veros_routine
        def set_parameter(self, state):
            super().set_parameter(state)
            for name in ("research_closure_scale", "research_closure_state"):
                state.var_meta[name] = Variable(name, None)

    simulation = Basin(override=dict(nx=4, ny=4, enable_tempsalt_sources=False))
    simulation.setup()
    prepared = simulation.prepare_integrator(record_diagnostics=(), checkpoint=True)
    carry = prepared.pack(simulation.state)

    def law(variables, weights):
        return jnp.full_like(variables.kappaH, 1e-5 * jnp.exp(weights[0]))

    return simulation, prepared, carry, law


def test_constant_closure_matches_native_diffusion_and_energy_exchange():
    import jax.numpy as jnp
    from veros.integration import PreparedIntegrator

    simulation, prepared, carry, law = closure_basin()
    weights = jnp.array([0.2, 0.0])
    carry = prepared.with_tracer_closure(carry, law=law, weights=weights)
    ordinary = PreparedIntegrator(simulation, record_diagnostics=(), checkpoint=True)
    augmented, _ = prepared.advance(carry, n_steps=3, final_only=True)
    expected, _ = ordinary.advance(
        carry,
        n_steps=3,
        final_only=True,
        parameters={"kappaH_0": simulation.state.settings.kappaH_0 + 1e-5 * jnp.exp(weights[0])},
    )
    prepared.check_status(augmented)
    for name in ("temp", "salt", "u", "v", "rho", "Hd", "Nsqr", "P_diss_v", "P_diss_nonlin"):
        np.testing.assert_allclose(
            getattr(augmented.variables, name), getattr(expected.variables, name), rtol=1e-10, atol=1e-11
        )
    np.testing.assert_allclose(
        (augmented.variables.kappaH - expected.variables.kappaH) * carry.variables.maskW, 0, atol=1e-15
    )
    volume = carry.variables.area_t[2:-2, 2:-2, None] * carry.variables.dzt * carry.variables.maskT[2:-2, 2:-2]
    for name in ("temp", "salt"):
        before = getattr(carry.variables, name)[2:-2, 2:-2, :, carry.variables.tau]
        after = getattr(augmented.variables, name)[2:-2, 2:-2, :, augmented.variables.tau]
        np.testing.assert_allclose(np.sum((after - before) * volume) / np.sum(volume), 0, atol=1e-11)


def test_dynamic_closure_weights_survive_checkpoint_adjoint():
    import jax
    import jax.numpy as jnp
    from veros.checkpointing import CheckpointedIntegrator, BinomialSchedule

    _, prepared, carry, law = closure_basin()
    point = jnp.array([0.2, 0.0])
    prepared.with_tracer_closure(carry, law=law, weights=point)

    def observe(c):
        return jnp.sum(c.variables.temp[2:-2, 2:-2, -1, c.variables.tau] ** 2)

    def function(weights):
        initial = prepared.with_tracer_closure(carry, law=law, weights=weights)
        final, _ = prepared.advance(initial, n_steps=3, final_only=True)
        return observe(final)

    value, expected = jax.value_and_grad(function)(point)
    initial, pullback = jax.vjp(lambda w: prepared.with_tracer_closure(carry, law=law, weights=w), point)
    driver = CheckpointedIntegrator(prepared, terminal_objective=observe)
    result = driver.value_and_grad(initial, n_steps=3, schedule=BinomialSchedule(3, ram_checkpoints=1))
    np.testing.assert_allclose(result.value, value, rtol=1e-12)
    gradient = pullback(result.gradient)[0]
    np.testing.assert_allclose(gradient, expected, rtol=1e-8, atol=1e-10)
    direction = jnp.array([1.0, 0.0])
    np.testing.assert_allclose(jax.jvp(function, (point,), (direction,))[1], expected @ direction, rtol=1e-8)
    epsilon = 1e-3
    np.testing.assert_allclose(
        (function(point + epsilon * direction) - function(point - epsilon * direction)) / (2 * epsilon),
        expected @ direction,
        rtol=2e-3,
        atol=1e-8,
    )


def test_inventory_projection_and_positive_flux_limit():
    import jax.numpy as jnp
    from types import SimpleNamespace
    from benchmarks.ad_demos.tracer_reference import (
        nested_inventory,
        positive_flux_lower_bound,
        vertical_flux_tendencies,
    )

    fine = jnp.arange(64.0).reshape(4, 4, 4)
    volume = 1 + fine / 100
    projected = nested_inventory(fine, volume, (2, 2, 2))
    np.testing.assert_allclose(projected.sum(), (fine * volume).sum(), rtol=1e-12)
    mask = jnp.ones((5, 5, 3), bool)
    variables = SimpleNamespace(
        temp=jnp.broadcast_to(jnp.array([1.0, 2.0, 4.0])[None, None, :, None], (5, 5, 3, 3)),
        salt=jnp.broadcast_to(jnp.array([30.0, 31.0, 33.0])[None, None, :, None], (5, 5, 3, 3)),
        tau=0,
        maskT=mask,
        maskW=mask,
        dzt=jnp.array([2.0, 3.0, 4.0]),
        dzw=jnp.array([2.5, 3.5, 4.0]),
    )

    # The helper accepts the variable container's membership protocol.
    class Variables(SimpleNamespace):
        def __contains__(self, key):
            return hasattr(self, key)

    variables = Variables(**vars(variables))
    tendency = vertical_flux_tendencies(variables, jnp.ones(mask.shape))
    np.testing.assert_allclose(np.sum(tendency[:, 2:-2, 2:-2] * variables.dzt, axis=-1), 0, atol=1e-12)
    state = SimpleNamespace(variables=variables)
    scales = np.ones(2)
    assert positive_flux_lower_bound([state], [np.asarray(tendency)], scales) < 1e-25
    certificate = {}
    bound = positive_flux_lower_bound([state], [-np.asarray(tendency)], scales, diagnostics=certificate)
    assert bound > 1e-5
    assert certificate["primal_dual_gap"] < 1e-12
    assert bound <= certificate["primal_loss"] + 1e-15


def test_refined_adjusted_targets_match_phase_and_reject_unresolved_stability(monkeypatch, tmp_path):
    import jax.numpy as jnp
    import pytest
    from benchmarks.ad_demos.matrix import CASES
    from benchmarks.ad_demos.research_ocean import ResearchOcean
    from veros.integration import PreparedIntegrator

    config = dict(
        setup="adjusted_basin",
        physics="fixed",
        mode="study",
        steps=None,
        spinup_steps=0,
        neutral_policy="nonstable",
        artifact_directory=str(tmp_path),
    )
    ocean = ResearchOcean(config, CASES["closure_equilibrium"])
    reference = ocean.prepared.reference_tracer_tendencies(
        ocean.initial, n_states=1, seed=31, refinement=2, projection="conservative", withheld_seed=47
    )
    with np.load(reference["metadata"]["dataset"]) as data:
        temperature = data["fine_basin_temperature_target"]
        salinity = data["fine_basin_salinity_target"]
        x, y, z = data["fine_xt"], data["fine_yt"], data["fine_zt"]
        meridional = np.cos(np.pi * y[None, :, None] / 200000)
        expected = (
            10 + 0.002 * z[None, None, :] + 0.15 * meridional + 0.01 * np.sin(2 * np.pi * x[:, None, None] / 160000)
        )
        np.testing.assert_allclose(temperature, expected, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(
            np.diff(temperature, axis=-1), np.broadcast_to(0.002 * np.diff(z), temperature[..., 1:].shape), atol=1e-13
        )
        np.testing.assert_allclose(salinity, np.broadcast_to(35 - 0.005 * meridional, salinity.shape), atol=1e-13)
    settings = reference["metadata"]["native_fine_settings"]
    assert settings["r_ray"] == ocean.sim.state.settings.r_ray
    assert settings["dt_tracer"] == ocean.sim.state.settings.dt_tracer / 2
    assert settings["dt_mom"] == ocean.sim.state.settings.dt_mom / 2
    calls = []
    stability_error = [1e-10]

    def adjusted(prepared, initial, *, observe, **kwargs):
        calls.append(float(initial.variables.time))
        value = observe(initial)
        if "phases" in kwargs:
            value = jnp.stack([value] * len(kwargs["phases"]))
        return dict(
            observations=value,
            residual_kind="full_state",
            residual=1e-12,
            spectral_radius=0.5,
            stability_residual=stability_error[0],
            sensitivity_residual=1e-9,
        )

    monkeypatch.setattr(PreparedIntegrator, "equilibrium_response", adjusted)
    monkeypatch.setattr(PreparedIntegrator, "periodic_response", adjusted)
    for kind in ("equilibrium", "seasonal"):
        for heldout in (False, True):
            target = reference["response_target"](kind, steps=1, heldout=heldout)
            assert np.isfinite(target).all()
    np.testing.assert_allclose(calls, float(ocean.initial.variables.time))
    assert len(reference["metadata"]["adjusted_responses"]) == 4
    stability_error[0] = 1e-3
    with pytest.raises(RuntimeError, match="resolved a stable branch"):
        reference["response_target"]("equilibrium", steps=2, heldout=False)

    from benchmarks.ad_demos import statistical_reference

    statistical_calls = []

    def statistical_target(prepared, initial, *, observe, **kwargs):
        assert prepared._template.settings.nx == 2 * ocean.sim.state.settings.nx
        assert np.isfinite(observe(initial)).all()
        statistical_calls.append(kwargs)
        return jnp.zeros((3, ocean.observe(ocean.initial).size)), dict(method="invariant_periodic")

    monkeypatch.setattr(statistical_reference, "periodic_reference", statistical_target)
    with pytest.raises(ValueError, match="declared thresholds"):
        reference["response_target"]("statistics", steps=1, heldout=True)
    budgets = dict(
        thresholds=jnp.ones(ocean.observe(ocean.initial).size),
        sample_stride=120,
        window_steps=1440,
        burnin_steps=28800,
        n_windows=32,
    )
    for _ in range(2):
        target = reference["response_target"]("statistics", steps=1, heldout=True, statistics=budgets)
        assert target.shape == (3, ocean.observe(ocean.initial).size)
    assert len(statistical_calls) == 1
    call = statistical_calls[0]
    assert (call["period_steps"], call["sample_stride"], call["window_steps"], call["burnin_steps"]) == (
        2880,
        240,
        2880,
        57600,
    )
    assert call["n_windows"] == 32 and call["seed"] == 47
    reference["response_target"](
        "statistics", steps=1, heldout=True, statistics=dict(budgets, thresholds=budgets["thresholds"] + 1)
    )
    assert len(statistical_calls) == 2


def test_unresolved_fitted_closure_stops_before_refined_transfer():
    import jax.numpy as jnp
    import pytest
    from types import SimpleNamespace
    from benchmarks.ad_demos.common import Checks
    from benchmarks.ad_demos.research import closure_training
    from benchmarks.ad_demos.research_ocean import closure_law
    from benchmarks.ad_demos.responses import Response

    states = [
        SimpleNamespace(variables=SimpleNamespace(Nsqr=jnp.full((1, 1, 1, 3), value), tau=0)) for value in (1e-6, 3e-6)
    ]
    truth = jnp.array([0.2, 0.3])
    targets = [closure_law(c.variables, truth) for c in states]
    transferred = []

    def target(*args, **kwargs):
        transferred.append(True)
        raise AssertionError("An unresolved fitted branch must stop before reference transfer")

    reference = dict(
        source="refined_veros",
        refinement=2,
        metadata=dict(
            projection_inventory_relative_error=0.0,
            projection_horizontal_relative_error=0.0,
            positive_flux_loss_lower_bound=0.0,
        ),
        train_states=states,
        heldout_states=states,
        train_tendencies=targets,
        heldout_tendencies=targets,
        tendency_scale=1e-5,
        flux_divergence=lambda c, k: k,
        budget_residual=lambda w: jnp.zeros(2),
        response_target=target,
    )

    class Unresolved:
        kind = "statistics"
        config = dict(setup="adjusted_basin")
        ocean = SimpleNamespace(
            initial=None,
            prepared=SimpleNamespace(reference_tracer_tendencies=lambda *args, **kwargs: reference),
        )
        check_adjustment = Response.check_adjustment

        def adjusted(self, point):
            return dict(
                residual_kind="full_state",
                residual=1e-12,
                spectral_radius=1.1,
                stability_residual=1e-10,
                sensitivity_residual=1e-9,
            )

    checks = Checks()
    with pytest.raises(RuntimeError, match="fitted closure.*stable adjusted branch"):
        closure_training(Unresolved(), checks)
    assert not transferred
    assert checks.metrics["closure_optimizer"]["converged"]
    assert not next(c["passed"] for c in checks.items if c["name"] == "tendency_weights stable adjusted branch")
