"""Solved physical column responses retain independent reference accuracy."""
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def response(pytestconfig):
    if pytestconfig.getoption("--backend") != "jax":
        pytest.skip("requires the JAX backend")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from benchmarks.ad_demos.prepared_column_response import linear_response

    jax.config.update("jax_enable_x64", True)
    fixture = json.loads((Path(__file__).parent / "reference_data/prepared_column_events.json").read_text())
    native, reference = fixture["native_preparation"], fixture["physical_response"]
    values = tuple(jnp.asarray(native["inputs"][key], dtype=jnp.float64) for key in native["argument_order"])
    weights = jnp.asarray(reference["work_weights"])
    return reference, values, weights, linear_response(values, weights)


def assert_fields(actual, expected):
    for key, value in expected.items():
        if key == "work":
            assert abs(np.longdouble(actual[key]) - np.longdouble(value)) <= 0.01
        else:
            np.testing.assert_allclose(np.asarray(actual[key]), value, rtol=1e-11, atol=1e-16)


def test_implicit_response_against_independent_variations(response):
    reference, _, _, local = response
    np.testing.assert_allclose(local["jacobian"], reference["jacobian"], rtol=1e-11, atol=1e-16)
    np.testing.assert_allclose(local["log_mean_response"], reference["log_mean_response"], rtol=1e-11, atol=1e-16)
    assert_fields(local["tangent"], reference["tangent"])
    assert abs(local["work_direct"] - local["work_adjoint"]) <= 0.01
    assert np.max(abs(local["tangent"]["residual"])) <= 1e-13


@pytest.fixture(scope="module")
def physical_points(response):
    from benchmarks.ad_demos.prepared_column_response import solve

    reference, values, weights, local = response
    return {
        case["beta"]: solve(
            values,
            weights,
            local["jacobian"],
            log_energy=case["beta"],
            initial=case["beta"] * local["log_mean_response"],
        )
        for case in reference["finite"]
    }


@pytest.mark.parametrize("beta", [0.001, -0.001, 0.0003, -0.0003])
def test_solved_physical_points_against_independent_reconstruction(response, physical_points, beta):
    reference, _, weights, _ = response
    point = physical_points[beta]
    expected = next(case for case in reference["finite"] if case["beta"] == beta)
    assert_fields(point["observations"], expected["fields"])
    np.testing.assert_allclose(point["log_means"], expected["log_means"], rtol=1e-8, atol=1e-13)
    assert point["evaluations"] <= 16
    assert point["tighter_work_change_J"] <= 0.01
    event, output = point["event"], point["observations"]
    assert int(event["status"]) == 0
    assert np.max(abs(np.asarray(output["residual"]))) <= 1e-13
    cn_work = np.sum(
        np.asarray(weights, dtype=np.longdouble) * np.asarray(output["N"]) * np.asarray(output["input_kh"])
    )
    assert abs(np.longdouble(output["work"]) - cn_work) <= 0.01


def test_physical_response_finite_scales_and_prediction(response, physical_points):
    reference, _, _, local = response
    for radius in (0.001, 0.0003):
        positive = physical_points[radius]["observations"]
        negative = physical_points[-radius]["observations"]
        for key in reference["tangent"]:
            estimate = (np.asarray(positive[key]) - np.asarray(negative[key])) / (2 * radius)
            tangent = local["tangent"][key]
            assert np.max(abs(estimate - tangent)) <= 1e-8 + 0.002 * np.max(abs(tangent))
    base, slope = np.longdouble(local["primal"]["work"]), np.longdouble(local["work_direct"])
    for sign in (1, -1):
        errors = [
            abs(np.longdouble(physical_points[sign * radius]["observations"]["work"]) - base - sign * radius * slope)
            for radius in (0.001, 0.0003)
        ]
        assert min(errors) > 500 * np.finfo(float).eps * abs(base)
        assert np.log(errors[0] / errors[1]) / np.log(0.001 / 0.0003) >= 1.8
    achieved = np.longdouble(physical_points[0.001]["observations"]["work"]) - base
    assert abs(achieved - 0.001 * slope) <= 0.1 * abs(achieved)


def test_response_rejects_uncertified_column(response):
    from benchmarks.ad_demos.prepared_column_response import solve

    _, values, weights, local = response
    changed = list(values)
    changed[9] = -values[9]
    with pytest.raises(ValueError, match="certified event domain"):
        solve(tuple(changed), weights, local["jacobian"])


def test_response_stops_at_evaluation_cap(response):
    from benchmarks.ad_demos.prepared_column_response import solve

    _, values, weights, local = response
    with pytest.raises(RuntimeError, match="evaluation cap"):
        solve(values, weights, local["jacobian"], log_energy=0.001, max_evaluations=2)


def test_response_requires_accepted_reference_means(response):
    from benchmarks.ad_demos.prepared_column_response import linear_response

    _, values, weights, _ = response
    changed = list(values)
    changed[2] = values[2] * 1.01
    with pytest.raises(ValueError, match="accepted reference mean"):
        linear_response(tuple(changed), weights)
