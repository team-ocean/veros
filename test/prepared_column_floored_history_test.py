"""Explicit friction-history choice retains independent physical and tangent checks."""
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def cases(pytestconfig):
    if pytestconfig.getoption("--backend") != "jax":
        pytest.skip("requires the JAX backend")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from benchmarks.ad_demos.prepared_column_native import prepare

    jax.config.update("jax_enable_x64", True)
    fixture = json.loads((Path(__file__).parent / "reference_data/prepared_column_events.json").read_text())
    prepared = {}
    for case in fixture["floored_history"]["cases"]:
        args = tuple(jnp.asarray(case["inputs"][k], dtype=jnp.float64) for k in case["argument_order"])
        _, _, event_args = jax.jit(prepare)(*args)
        prepared[case["name"]] = case, event_args
    return prepared


@pytest.mark.parametrize("name", ["active", "fresh_positive"])
def test_floored_history_matches_independent_physical_moments(cases, name):
    import jax
    from benchmarks.ad_demos.prepared_column_events import integrate_floored_history

    case, args = cases[name]
    out = jax.jit(integrate_floored_history)(*args)
    assert int(out["status"]) == 0
    assert bool(out["continuous_branch_certified"])
    assert np.all(np.asarray(out["end"]) > 0)
    for key, expected in case["moments"].items():
        np.testing.assert_allclose(np.asarray(out[key]), expected, rtol=1e-11, atol=1e-16)
    if name == "active":
        work = np.sum(
            np.asarray(case["work_weights"], dtype=np.longdouble) * np.asarray(args[0]) * np.asarray(out["mean_kh"])
        )
        assert abs(work - np.longdouble(case["mixing_work_J"])) <= 0.01


def test_floored_history_incoming_energy_derivative(cases):
    import jax
    import jax.numpy as jnp
    from benchmarks.ad_demos.prepared_column_events import (
        MOMENT_FIELDS,
        integrate_floored_history,
        differentiable_floored_history_moments,
    )

    case, args = cases["active"]
    weights = jnp.asarray(case["work_weights"])

    def forward(beta):
        changed = list(args)
        changed[6] = args[6] * jnp.exp(beta)
        out = integrate_floored_history(*changed)
        return tuple(out[k] for k in MOMENT_FIELDS)

    _, tangent = jax.jit(lambda: jax.jvp(forward, (jnp.asarray(0.0),), (jnp.asarray(1.0),)))()
    for key, actual in zip(MOMENT_FIELDS, tangent):
        np.testing.assert_allclose(np.asarray(actual), case["incoming_energy_tangent"][key], rtol=1e-11, atol=1e-16)

    def objective(beta):
        changed = list(args)
        changed[6] = args[6] * jnp.exp(beta)
        return jnp.vdot(weights, args[0] * differentiable_floored_history_moments(*changed)[3])

    reverse = jax.jit(jax.grad(objective))(jnp.asarray(0.0))
    direct = np.sum(np.asarray(weights, dtype=np.longdouble) * np.asarray(args[0]) * np.asarray(tangent[3]))
    assert abs(direct - np.longdouble(case["independent_partial_work_J"])) <= 0.01
    assert abs(np.longdouble(reverse) - np.longdouble(case["independent_partial_work_J"])) <= 0.01


def test_floored_history_intermediate_prandtl_branch(cases):
    import jax
    import jax.numpy as jnp
    from benchmarks.ad_demos.prepared_column_events import (
        MOMENT_FIELDS,
        integrate_floored_history,
        differentiable_floored_history_moments,
    )

    N = np.array([1e-4, 2e-4])
    energy = np.array([7e-6, 5e-6])
    previous = np.array([3.3e-8, 1e-8])
    floor, ck, ce, ratio, total = 2e-4, 0.1, 0.7, 48.0, 300.0
    a = ck * np.sqrt(2 / N)
    Pr = np.clip(6.6 * N * floor / previous, 1, 10)
    slope = np.array([a[0] / Pr[0], 0.0])
    dslope = np.array([-1.5 * slope[0], 0.0])
    gamma = ce * np.sqrt(N / 2)
    rate = ratio * N * slope + gamma
    drate = ratio * N * (slope + dslope) + gamma / 2
    end = energy * np.exp(-rate * total)
    dend = -total * drate * end
    mean = energy * (-np.expm1(-rate * total)) / (rate * total)
    dmean = -dend / (rate * total) - mean * drate / rate
    expected = (end, mean, np.full(2, floor), slope * mean, gamma * mean)
    derivative = (dend, dmean, np.zeros(2), dslope * mean + slope * dmean, gamma / 2 * mean + gamma * dmean)
    args = tuple(
        jnp.asarray(x, dtype=jnp.float64)
        for x in (
            N,
            np.zeros((2, 2)),
            np.zeros(2),
            [0.001, 0.0],
            0.0,
            ratio,
            energy,
            previous,
            [5.0, 5.0],
            np.zeros(2),
            total,
            ck,
            ce,
            1e-8,
            floor,
            0.1,
        )
    )

    def values(beta, reverse=False):
        changed = list(args)
        changed[0] = args[0] * jnp.exp(beta)
        if reverse:
            return differentiable_floored_history_moments(*changed)
        out = integrate_floored_history(*changed)
        return tuple(out[k] for k in MOMENT_FIELDS)

    primal, tangent = jax.jit(lambda: jax.jvp(values, (jnp.asarray(0.0),), (jnp.asarray(1.0),)))()
    for actual, true in zip(primal, expected):
        np.testing.assert_allclose(np.asarray(actual), true, rtol=1e-11, atol=1e-16)
    for actual, true in zip(tangent, derivative):
        np.testing.assert_allclose(np.asarray(actual), true, rtol=1e-11, atol=1e-16)
    reverse = jax.jit(jax.grad(lambda beta: sum(jnp.sum(x) for x in values(beta, reverse=True))))(jnp.asarray(0.0))
    np.testing.assert_allclose(np.asarray(reverse), sum(np.sum(x) for x in derivative), rtol=1e-11, atol=1e-16)
    assert 1 < Pr[0] < 10
    assert np.all(a * energy < floor)


def test_floored_history_rejects_empty_incoming_energy(cases):
    import jax
    import jax.numpy as jnp
    from benchmarks.ad_demos.prepared_column_events import (
        integrate_floored_history,
        differentiable_floored_history_moments,
    )

    _, args = cases["active"]
    changed = list(args)
    changed[6] = jnp.zeros_like(args[6])
    assert int(jax.jit(integrate_floored_history)(*changed)["status"]) != 0
    out = jax.jit(differentiable_floored_history_moments)(*changed)
    assert all(np.all(~np.isfinite(np.asarray(x))) for x in out)
