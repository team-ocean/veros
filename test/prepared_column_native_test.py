"""Native column preparation retains independent physical and derivative checks."""

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def case(pytestconfig):
    if pytestconfig.getoption("--backend") != "jax":
        pytest.skip("requires the JAX backend")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from benchmarks.ad_demos.prepared_column_native import conditional

    jax.config.update("jax_enable_x64", True)
    reference = json.loads((Path(__file__).parent / "reference_data/prepared_column_events.json").read_text())[
        "native_preparation"
    ]
    values = tuple(jnp.asarray(reference["inputs"][key], dtype=jnp.float64) for key in reference["argument_order"])
    return reference, values, jax.jit(conditional)


def assert_reference(actual, expected):
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected, dtype=float), rtol=1e-11, atol=1e-16)


def test_native_conditional_means_and_energy(case):
    from benchmarks.ad_demos.prepared_column_events import MOMENT_FIELDS

    reference, values, conditional = case
    result = conditional(*values)
    event, prepared = result["event"], result["prepared"]
    assert int(event["status"]) == 0
    assert np.max(abs(np.asarray(result["residual"]))) <= 1e-13
    for key in MOMENT_FIELDS:
        assert_reference(event[key], reference["moments"][key])
    host = {key: np.asarray(value, dtype=np.longdouble) for key, value in zip(reference["argument_order"], values)}
    event = {key: np.asarray(value) for key, value in event.items()}
    prepared = {key: np.asarray(value, dtype=np.longdouble) for key, value in prepared.items()}
    flow = host["dt_mom"] * (prepared["Q"] @ event["mean"].astype(np.longdouble))
    source = host["dt_mom"] * prepared["M"] * event["mean_mu"]
    donor = host["dt_mom"] * host["eke_rate"]
    work = host["dt_tracer"] * host["kh"] * prepared["N"]
    heat = host["dt_mom"] * event["heat"]
    residual = event["end"] - host["energy"] - flow - source - donor + work + heat
    scale = np.max(host["energy"] + source + donor + np.maximum(flow, 0))
    assert np.max(abs(residual)) <= 500 * np.finfo(float).eps * scale
    assert abs(np.sum(flow * host["width"])) <= 500 * np.finfo(float).eps * np.sum(abs(flow * host["width"]))
    support = prepared["source0"][:-1] - prepared["ratio"] * prepared["fraction"] * host["bg"][:-1] * prepared["N"][:-1]
    assert np.min(support) >= 0


@pytest.fixture(scope="module")
def products(case):
    import gc
    import jax
    import jax.numpy as jnp
    from benchmarks.ad_demos.prepared_column_events import differentiable_moments, MOMENT_FIELDS
    from benchmarks.ad_demos.prepared_column_native import prepare

    reference, values, conditional = case
    n = len(values[2])
    seed = jnp.asarray(reference["direction"]["direction"])
    cotangent = jnp.asarray(reference["direction"]["cotangent"])
    weights = jnp.asarray(reference["direction"]["work_weights"])
    work_scale = jnp.asarray(reference["direction"]["work_scale"])
    q0 = jnp.zeros(2 * n - 1, dtype=jnp.float64)

    def varied(q):
        changed = list(values)
        changed[2] = values[2] * jnp.exp(q[:n])
        changed[3] = jnp.concatenate((values[3][:-1] * jnp.exp(q[n:]), jnp.zeros(1, dtype=q.dtype)))
        return tuple(changed)

    def forward(q):
        result = conditional(*varied(q))
        cn, prepared, event = result["cn"], result["prepared"], result["event"]
        work = jnp.vdot(weights, prepared["N"] * event["mean_kh"])
        output = {key: event[key] for key in MOMENT_FIELDS}
        output.update({key: prepared[key] for key in ("N", "Nnew", "M", "Q", "fraction", "source0")})
        output.update(
            cn_delta=cn["delta"],
            residual=result["residual"],
            work=work,
            phi=jnp.vdot(cotangent, result["residual"]) + work / work_scale,
        )
        return output

    primal, tangent = jax.jit(lambda q: jax.jvp(forward, (q,), (seed,)))(q0)
    jax.block_until_ready((primal, tangent))
    primal = {key: np.asarray(value) for key, value in primal.items()}
    tangent = {key: np.asarray(value) for key, value in tangent.items()}
    jax.clear_caches()
    gc.collect()

    def objective(q):
        changed = varied(q)
        _, prepared, event_args = prepare(*changed)
        _, _, mean_mu, mean_kh, _ = differentiable_moments(*event_args)
        residual = jnp.concatenate((jnp.log(mean_mu / changed[2]), jnp.log(mean_kh[:-1] / changed[3][:-1])))
        return jnp.vdot(cotangent, residual) + jnp.vdot(weights, prepared["N"] * mean_kh) / work_scale

    value, gradient = jax.jit(jax.value_and_grad(objective))(q0)
    jax.block_until_ready((value, gradient))
    return dict(
        primal=primal,
        tangent=tangent,
        value=np.asarray(value),
        gradient=np.asarray(gradient),
        varied=varied,
        seed=np.asarray(seed),
        cotangent=np.asarray(cotangent),
        weights=np.asarray(weights),
        work_scale=np.asarray(work_scale),
    )


def test_native_preparation_and_event_tangents(case, products):
    reference, _, _ = case
    for key, expected in reference["preparation_tangent"].items():
        assert_reference(products["tangent"][key], expected)
    for key, expected in reference["moment_tangent"].items():
        assert_reference(products["tangent"][key], expected)
    assert_reference(products["tangent"]["residual"], reference["observer_tangent"]["residual"])
    assert (
        abs(np.longdouble(products["tangent"]["work"]) - np.longdouble(reference["observer_tangent"]["work"])) <= 0.01
    )


def test_native_preparation_reverse_against_independent_variation(case, products):
    reference, _, _ = case
    assert np.isfinite(products["gradient"]).all()
    contraction = np.sum(products["gradient"].astype(np.longdouble) * products["seed"].astype(np.longdouble))
    forward = np.longdouble(products["tangent"]["phi"])
    assert abs(contraction - forward) <= 1e-9 + 1e-5 * max(abs(contraction), abs(forward))
    assert abs(products["value"] - products["primal"]["phi"]) <= 1e-12
    residual = np.asarray(reference["observer_tangent"]["residual"], dtype=np.longdouble)
    residual_part = np.sum(products["cotangent"].astype(np.longdouble) * residual)
    reverse_work = (contraction - residual_part) * np.longdouble(products["work_scale"])
    assert abs(reverse_work - np.longdouble(reference["observer_tangent"]["work"])) <= 0.01


def test_native_prepared_original_finite_scales(case, products):
    from benchmarks.ad_demos.prepared_column_events import MOMENT_FIELDS

    _, _, conditional = case
    cases = {}
    for radius in (0.001, 0.0003):
        for sign in (1, -1):
            result = conditional(*products["varied"](sign * radius * products["seed"]))
            event, prepared = result["event"], result["prepared"]
            assert int(event["status"]) == 0
            output = {key: np.asarray(event[key]) for key in MOMENT_FIELDS}
            output["residual"] = np.asarray(result["residual"])
            work = np.sum(products["weights"].astype(np.longdouble) * np.asarray(prepared["N"]) * output["mean_kh"])
            output["phi"] = (
                np.sum(products["cotangent"].astype(np.longdouble) * output["residual"]) + work / products["work_scale"]
            )
            cases[sign * radius] = output
    for radius in (0.001, 0.0003):
        for key in (*MOMENT_FIELDS, "residual", "phi"):
            estimate = (cases[radius][key] - cases[-radius][key]) / (2 * radius)
            tangent = products["tangent"][key]
            assert np.max(abs(estimate - tangent)) <= 1e-8 + 0.002 * np.max(abs(tangent))
    slope = np.sum(products["gradient"].astype(np.longdouble) * products["seed"].astype(np.longdouble))
    for sign in (1, -1):
        errors = [
            abs(cases[sign * radius]["phi"] - products["primal"]["phi"] - sign * radius * slope)
            for radius in (0.001, 0.0003)
        ]
        assert min(errors) > 500 * np.finfo(float).eps * abs(products["primal"]["phi"])
        assert np.log(errors[0] / errors[1]) / np.log(0.001 / 0.0003) >= 1.8


@pytest.mark.parametrize("shape", [(3, 14), (4, 1)])
def test_native_preparation_rejects_unsupported_shapes(case, shape):
    from benchmarks.ad_demos.prepared_column_native import prepare

    _, values, _ = case
    changed = list(values)
    changed[0] = values[0][: shape[0], : shape[1]]
    changed[1] = values[1][: shape[0], : shape[1]]
    with pytest.raises(ValueError, match="four paired fields"):
        prepare(*changed)


def test_native_preparation_requires_float64(case):
    import jax.numpy as jnp
    from benchmarks.ad_demos.prepared_column_native import prepare

    _, values, _ = case
    with pytest.raises(ValueError, match="float64"):
        prepare(*(value.astype(jnp.float32) for value in values))
