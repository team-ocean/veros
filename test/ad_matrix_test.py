"""Coverage, response semantics, and physical diagnostic acceptance checks."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.ad_demos import DEMOS
from benchmarks.ad_demos.common import Checks
from benchmarks.ad_demos.matrix import CASES, CONTRACTS, QUESTIONS, RESPONSES, TASKS, missing_capabilities
from benchmarks.ad_demos.research import control_indices
from benchmarks.ad_demos.research_ocean import CP, CONTROL_NAMES, ResearchOcean
from benchmarks.ad_demos.responses import Response


def test_every_cell_and_question_has_an_executable_case():
    assert {(case.task, case.response) for case in CASES.values()} == {
        (task, response) for task in TASKS for response in RESPONSES
    }
    assert len(CASES) == 35
    assert set(CASES) <= set(DEMOS)
    assert set(QUESTIONS) == {question for case in CASES.values() for question in case.questions}
    for name, case in CASES.items():
        assert name == case.name
        assert case.questions
        assert set(case.requirements) <= set(CONTRACTS)
        indices = control_indices(case)
        assert len(indices) == len(set(indices))
        assert all(0 <= index < len(CONTROL_NAMES) for index in indices)
        if case.response not in ("local", "trajectory"):
            assert not set(indices) & {
                0,
                1,
            }  # Adjusted-state estimation does not infer transient initial fields.


def test_missing_capability_checks_are_callable_and_complete():
    class PartialProvider:
        statistical_response = "not implemented"

        def with_vertical_grid(self):
            pass

    missing = missing_capabilities(PartialProvider, CASES["geometry_statistics"])
    assert set(missing) == {"statistical_response", "with_wet_geometry", "grid_reference_error"}
    assert all(missing.values())


def test_cli_exports_matrix_and_filters(monkeypatch, capsys, tmp_path):
    import subprocess
    import sys
    from benchmarks.ad_demos.__main__ import main, write_result

    monkeypatch.setattr(sys, "argv", ["demos", "--matrix"])
    assert main() == 0
    assert len(json.loads(capsys.readouterr().out)["cases"]) == 35
    selected = []

    def child(command, **kwargs):
        name = command[command.index("--demo") + 1]
        selected.append(name)
        write_result(tmp_path / f"{name}.json", dict(status="failed"))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", child)
    monkeypatch.setattr(sys, "argv", ["demos", "--demo", "matrix", "--task", "geometry", "--output", str(tmp_path)])
    assert main() == 1
    assert selected == [f"geometry_{response}" for response in RESPONSES]


@pytest.mark.parametrize("policy", (None, "default", "positive_diffusion"))
def test_cli_records_and_forwards_explicit_tke_policy(monkeypatch, tmp_path, policy):
    import subprocess
    import sys
    from benchmarks.ad_demos.__main__ import main, write_result

    def child(command, **kwargs):
        if policy is None:
            assert "--tke-policy" not in command
        else:
            assert command[command.index("--tke-policy") + 1] == policy
        write_result(tmp_path / "perturbation_local.json", dict(status="smoke_passed"))
        return SimpleNamespace(returncode=0)

    arguments = ["demos", "--demo", "perturbation_local", "--output", str(tmp_path)]
    if policy is not None:
        arguments.extend(("--tke-policy", policy))
    monkeypatch.setattr(subprocess, "run", child)
    monkeypatch.setattr(sys, "argv", arguments)
    assert main() == 0
    config = json.loads((tmp_path / "summary.json").read_text())["config"]
    assert config.get("tke_policy") == policy
    if policy is None:
        assert "tke_policy" not in config


def test_unknown_tke_policy_rejected_before_model_setup():
    with pytest.raises(ValueError, match="Unknown TKE policy"):
        ResearchOcean(dict(setup="acc", physics="default", tke_policy="typo"), CASES["perturbation_local"])


class Variables(SimpleNamespace):
    def __contains__(self, key):
        return key in vars(self)


@pytest.mark.parametrize("partial", [False, True])
def test_physical_observations_exclude_halos_and_use_actual_layer_overlap(partial):
    jnp = pytest.importorskip("jax.numpy")
    shape = (6, 6, 2)
    variables = Variables(
        tau=0,
        temp=jnp.full(shape + (3,), 10.0),
        salt=jnp.full(shape + (3,), 35.0),
        u=jnp.ones(shape + (3,)),
        v=jnp.ones(shape + (3,)),
        maskT=jnp.ones(shape, bool),
        maskU=jnp.ones(shape, bool),
        maskV=jnp.ones(shape, bool),
        area_t=jnp.full((6, 6), 100.0),
        dxt=jnp.full(6, 2.0),
        dyt=jnp.full(6, 3.0),
        dzt=jnp.array([800.0, 400.0]),
        zt=jnp.array([-800.0, -200.0]),
        cosu=jnp.ones(6),
    )
    if partial:
        fractions = jnp.ones(shape).at[..., 0].set(0.5)
        variables.hfacT = variables.hfacU = variables.hfacV = fractions
    ocean = object.__new__(ResearchOcean)
    ocean.sim = SimpleNamespace(
        state=SimpleNamespace(settings=SimpleNamespace(rho_0=1024, enable_skew_diffusion=False))
    )
    ocean.north, ocean.south = jnp.ones((2, 2), bool), jnp.zeros((2, 2), bool)
    ocean.section = ocean.drake = 3
    ocean.atlantic = ocean.drake_latitudes = jnp.ones(2, bool)
    ocean.observation_scales = jnp.ones(12)
    observed = np.asarray(ocean.observe(SimpleNamespace(variables=variables)))
    np.testing.assert_allclose(observed[:4], [10, 10, 35, 35])
    depth, upper, below = (800, 800, 0) if partial else (1200, 1000, 200)
    expected = [
        CP * 1024 * 10 * 4 * 100 * upper,
        CP * 1024 * 10 * 2 * 2 * depth,
        2 * 3 * depth,
        -2 * 2 * below,
    ]
    np.testing.assert_allclose(observed[-4:], expected, rtol=1e-6)
    variables.temp = variables.temp.at[:2].set(1e6).at[-2:].set(1e6)
    np.testing.assert_allclose(ocean.observe(SimpleNamespace(variables=variables)), observed, rtol=1e-6)


class AdjustedOcean:
    """Analytic stable recurrence for testing adjusted-response contracts."""

    dt = 86400.0

    def __init__(self, kind, *, method="invariant_equilibrium"):
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        self.case = CASES[f"sensitivity_{kind}"]
        self.initial = (jnp.array(10.0), jnp.zeros(len(CONTROL_NAMES)))
        self.prepared = self
        self.method = method
        self.adjustment_budgets = []

    def controls(self, full, **kwargs):
        return (10 + full[0], full), {}

    @staticmethod
    def observe(carry):
        import jax.numpy as jnp

        return jnp.stack((carry[0], 2 * carry[0]))

    @staticmethod
    def advance(carry, *, n_steps, **kwargs):
        x, full = carry
        target = 10 + full[2] + 0.5 * full[4]
        return ((target + 0.8**n_steps * (x - target), full), None)

    @staticmethod
    def check_status(carry):
        pass

    def equilibrium_response(self, carry, **kwargs):
        self.adjustment_budgets.append(kwargs.get("max_steps", kwargs.get("max_cycles")))
        full = carry[1]
        return dict(
            observations=self.observe((10 + full[2] + 0.5 * full[4], full)),
            residual_kind="full_state",
            residual=0.0,
            spectral_radius=0.8,
            sensitivity_residual=0.0,
        )

    def periodic_response(self, carry, **kwargs):
        import jax.numpy as jnp

        result = self.equilibrium_response(carry, **kwargs)
        result["observations"] = jnp.broadcast_to(result["observations"], (len(kwargs["phases"]), 2))
        return result

    def statistical_response(self, carry, **kwargs):
        import jax
        import jax.numpy as jnp

        mean = self.equilibrium_response(carry)["observations"]
        values = jnp.stack((mean, jnp.zeros_like(mean), jax.nn.sigmoid(mean - kwargs["thresholds"])))
        return dict(
            observations=values,
            method=self.method,
            standard_error=jnp.zeros_like(values),
            effective_sample_size=128,
            window_drift=jnp.zeros_like(values),
        )


@pytest.mark.parametrize("kind", ["equilibrium", "seasonal", "statistics"])
def test_adjusted_response_contract_runs_beyond_capability_detection(kind):
    jnp = pytest.importorskip("jax.numpy")
    ocean = AdjustedOcean(kind)
    response = Response(ocean, (2, 4), dict(mode="smoke", steps=None, setup="acc"))
    point, checks = jnp.array([0.2, 0.1]), Checks()
    response.validate(point, checks)

    def loss(z):
        return jnp.sum(response.values(z))

    response.audit(loss, point, checks, lambda z, seed: jnp.sum(response.values(z, forward=True, seed=seed)))
    assert all(item["passed"] for item in checks.items), checks.items
    if kind in ("equilibrium", "seasonal"):
        assert max(ocean.adjustment_budgets) == 2 * min(ocean.adjustment_budgets)


def test_finite_time_ad_cannot_claim_climate_acceptance():
    jnp = pytest.importorskip("jax.numpy")
    response = Response(
        AdjustedOcean("statistics", method="finite_time_ad"),
        (2, 4),
        dict(mode="smoke", steps=None, setup="acc"),
    )
    checks = Checks()
    response.validate(jnp.zeros(2), checks)
    assert not next(item["passed"] for item in checks.items if item["name"] == "invariant response method")


def test_observation_ensemble_uses_the_predicted_season():
    import jax
    import jax.numpy as jnp
    from types import SimpleNamespace
    from benchmarks.ad_demos.common import gradient_checks
    from benchmarks.ad_demos.inverse import JaxResidual
    from benchmarks.ad_demos.research import observation_design

    class Function(JaxResidual):
        def value_and_grad(self, point):
            return jax.value_and_grad(self.function)(point)

    class Seasonal:
        kind = "seasonal"
        indices = jnp.arange(2)
        ocean = SimpleNamespace(nz=1, amoc_index=7)
        config = {}

        def __init__(self):
            self.ensemble_phases = []

        def values(self, point, *, heldout=False, forward=False, seed=0):
            self.ensemble_phases.append(seed)
            observations = jnp.array([point[0], point[1], point[0], point[1], 0, 0, 0, 2 + point[0] + seed * point[1]])
            return jnp.broadcast_to(observations, (12, 8))

        def function(self, function, *, heldout=False):
            return Function(lambda point: function(self.values(point, heldout=heldout), point))

        def audit(self, function, point, checks, forward):
            return gradient_checks(function, point, checks)

    response, checks = Seasonal(), Checks()
    observation_design(response, checks)
    assert set(response.ensemble_phases) == {0}
    assert next(
        check["passed"] for check in checks.items if check["name"] == "nonlinear future-transport ensemble variance"
    )
    samples = np.asarray(checks.metrics["nonlinear_ensemble_controls"])
    covariance = np.asarray(checks.metrics["posterior_control_covariance"])
    expected = 0.01 * np.random.default_rng(27).normal(size=(64, 2)) @ np.linalg.cholesky(covariance).T
    np.testing.assert_array_equal(samples, expected)
    bound = checks.metrics["observation_information_bound"]["all_candidate_variance"]
    assert bound <= checks.metrics["future_transport_variance"][-1] + 1e-12
