"""Prepared-column events retain independent values and finite-change derivatives."""

import json
from pathlib import Path

import numpy as np
import pytest

NAMES = (
    "N",
    "Q",
    "M",
    "bg",
    "fraction",
    "ratio",
    "energy",
    "previous",
    "h",
    "eke_rate",
    "total",
    "ck",
    "ce",
    "mxl_min",
    "mu_floor",
    "mu_max",
)


@pytest.fixture(scope="module")
def case(pytestconfig):
    if pytestconfig.getoption("--backend") != "jax":
        pytest.skip("requires the JAX backend")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from benchmarks.ad_demos.prepared_column_events import integrate

    jax.config.update("jax_enable_x64", True)
    reference = json.loads((Path(__file__).parent / "reference_data/prepared_column_events.json").read_text())
    inputs = tuple(jnp.asarray(reference["inputs"][name], dtype=jnp.float64) for name in NAMES)
    return reference, inputs, jax.jit(integrate)


def assert_moments(actual, expected):
    from benchmarks.ad_demos.prepared_column_events import MOMENT_FIELDS

    for key in MOMENT_FIELDS:
        np.testing.assert_allclose(
            np.asarray(actual[key]), np.asarray(expected[key], dtype=float), rtol=1e-11, atol=1e-16
        )


def test_actual_column_values_and_native_history(case):
    reference, inputs, integrate = case
    result = integrate(*inputs)
    assert int(result["status"]) == 0
    assert bool(result["continuous_branch_certified"])
    assert int(result["segment_count"]) == 4
    assert_moments(result, reference["moments"])
    prandtl = np.asarray(result["segment_Pr"])[:4]
    assert np.min(prandtl) == 1
    assert np.max(prandtl) == 10
    assert np.any((prandtl > 1) & (prandtl < 10))


def test_rising_energies_can_remain_inside_their_branches(case):
    reference, inputs, integrate = case
    changed = list(inputs)
    changed[6] = changed[6] * np.exp(-0.001)
    result = integrate(*changed)
    assert int(result["status"]) == 0
    assert bool(result["continuous_branch_certified"])
    assert not bool(result["monotonicity_certified"])
    assert_moments(result, reference["negative_energy_moments"])


@pytest.fixture(scope="module")
def derivatives(case):
    import jax
    import jax.numpy as jnp
    from benchmarks.ad_demos.prepared_column_events import integrate, differentiable_moments, MOMENT_FIELDS

    reference, inputs, _ = case
    direction = tuple(value * coefficient for value, coefficient in zip(inputs, reference["direction_coefficients"]))
    weights = jnp.asarray(reference["work_weights"], dtype=jnp.float64)
    forward = jax.jit(lambda *values: jax.jvp(integrate, values, direction))
    primal, tangent = forward(*inputs)

    def objective(*values):
        return jnp.vdot(weights, differentiable_moments(*values)[3])

    value, gradient = jax.jit(jax.value_and_grad(objective, argnums=tuple(range(16))))(*inputs)
    jax.block_until_ready((primal, tangent, value, gradient))
    return dict(
        tangent={k: np.asarray(tangent[k]) for k in MOMENT_FIELDS},
        gradient=gradient,
        value=value,
        direction=direction,
        weights=np.asarray(weights),
    )


def test_full_input_derivatives_against_independent_variation(case, derivatives):
    reference, inputs, _ = case
    assert_moments(derivatives["tangent"], reference["mixed_tangent"])
    assert all(np.isfinite(np.asarray(value)).all() for value in derivatives["gradient"])
    forward = np.sum(derivatives["weights"].astype(np.longdouble) * derivatives["tangent"]["mean_kh"])
    reverse = sum(
        np.sum(np.asarray(g, dtype=np.longdouble) * np.asarray(t, dtype=np.longdouble))
        for g, t in zip(derivatives["gradient"], derivatives["direction"])
    )
    assert abs(reverse - forward) <= 1e-9 + 1e-5 * max(abs(forward), abs(reverse))
    assert abs(reverse - np.longdouble(reference["mixed_work_reference_J"])) <= 0.01
    for index in (8, 13, 15):
        np.testing.assert_array_equal(np.asarray(derivatives["gradient"][index]), 0)


def test_original_signed_finite_scales_and_second_order_error(case, derivatives):
    from benchmarks.ad_demos.prepared_column_events import MOMENT_FIELDS

    reference, inputs, integrate = case
    cases = {}
    for radius in (0.001, 0.0003):
        for sign in (1, -1):
            alpha = sign * radius
            result = integrate(
                *(
                    value * np.exp(alpha * coefficient)
                    for value, coefficient in zip(inputs, reference["direction_coefficients"])
                )
            )
            assert int(result["status"]) == 0
            cases[alpha] = {key: np.asarray(result[key]) for key in MOMENT_FIELDS}
    for radius in (0.001, 0.0003):
        for key in MOMENT_FIELDS:
            estimate = (cases[radius][key] - cases[-radius][key]) / (2 * radius)
            tangent = derivatives["tangent"][key]
            assert np.max(abs(estimate - tangent)) <= 1e-8 + 0.002 * np.max(abs(tangent))
    weights = derivatives["weights"].astype(np.longdouble)
    initial = np.sum(weights * np.asarray(reference["moments"]["mean_kh"], dtype=np.longdouble))
    slope = sum(
        np.sum(np.asarray(g, dtype=np.longdouble) * np.asarray(t, dtype=np.longdouble))
        for g, t in zip(derivatives["gradient"], derivatives["direction"])
    )
    for sign in (1, -1):
        errors = [
            abs(np.sum(weights * cases[sign * radius]["mean_kh"]) - initial - sign * radius * slope)
            for radius in (0.001, 0.0003)
        ]
        assert min(errors) > 500 * np.finfo(float).eps * abs(initial)
        assert np.log(errors[0] / errors[1]) / np.log(0.001 / 0.0003) >= 1.8


@pytest.mark.parametrize("index,factor", [(6, -1), (4, -1), (9, -1)])
def test_rejects_outside_physical_input_domain(case, index, factor):
    _, inputs, integrate = case
    changed = list(inputs)
    changed[index] = factor * changed[index]
    result = integrate(*changed)
    assert int(result["status"]) != 0
    assert not bool(result["continuous_branch_certified"])


def test_rejects_non_metzler_transport(case):
    _, inputs, integrate = case
    changed = list(inputs)
    changed[1] = changed[1].at[0, 1].set(-1e-6)
    assert int(integrate(*changed)["status"]) != 0


def test_reverse_rejects_invalid_column(case):
    import jax
    import jax.numpy as jnp
    from benchmarks.ad_demos.prepared_column_events import differentiable_moments

    _, inputs, _ = case
    changed = list(inputs)
    changed[6] = -changed[6]

    def objective(energy):
        return jnp.sum(differentiable_moments(*changed[:6], energy, *changed[7:])[3])

    value, gradient = jax.jit(jax.value_and_grad(objective))(changed[6])
    assert not np.isfinite(np.asarray(value))
    assert np.isnan(np.asarray(gradient)).all()


def test_requires_float64(case):
    import jax.numpy as jnp
    from benchmarks.ad_demos.prepared_column_events import integrate

    _, inputs, _ = case
    with pytest.raises(ValueError, match="float64"):
        integrate(*(value.astype(jnp.float32) for value in inputs))
