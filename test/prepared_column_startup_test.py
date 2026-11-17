"""Empty-energy native startup retains independent physical and defect checks."""

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def startup(pytestconfig):
    if pytestconfig.getoption("--backend") != "jax":
        pytest.skip("requires the JAX backend")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from benchmarks.ad_demos.prepared_column_native import prepare
    from benchmarks.ad_demos.prepared_column_startup import integrate_empty

    jax.config.update("jax_enable_x64", True)
    r = json.loads((Path(__file__).parent / "reference_data/prepared_column_events.json").read_text())["zero_start"]
    args = tuple(jnp.asarray(r["inputs"][k], dtype=jnp.float64) for k in r["argument_order"])
    cn, z, event_args = jax.jit(prepare)(*args)
    weights = jnp.asarray(r["work_weights"])
    function = jax.jit(integrate_empty)
    event = function(*event_args, weights)
    return r, args, cn, z, event_args, weights, event, function


def test_empty_startup_matches_full_native_law(startup):
    r, args, _, _, _, _, event, _ = startup
    assert int(event["status"]) == 0
    assert np.all(np.asarray(args[9]) == 0)
    assert np.all(np.asarray(args[10]) == 0)
    assert np.all(np.asarray(event["end"]) >= 0)
    residual = np.r_[
        np.log(np.asarray(event["mean_mu"]) / np.asarray(args[2])),
        np.log(np.asarray(event["mean_kh"][:-1]) / np.asarray(args[3][:-1])),
    ]
    assert np.max(abs(residual)) <= 1e-13
    for key, value in r["moments"].items():
        np.testing.assert_allclose(np.asarray(event[key]), value, rtol=1e-11, atol=1e-16)


def test_empty_startup_error_bounds_are_conservative(startup):
    r, _, _, _, _, _, event, _ = startup
    for key in ("end_error", "mean_error", "kh_error", "heat_error"):
        assert np.all(np.asarray(event[key]) >= np.asarray(r["independent_bounds"][key]))
    assert np.all(np.asarray(event["bank_defect_bound"]) >= r["independent_bounds"]["bank_error"])
    assert np.all(np.asarray(event["upper_energy"]) >= r["independent_bounds"]["upper"])
    assert r["work_error_bound_J"] <= float(event["work_error_bound_J"]) <= 0.01


def test_empty_startup_physical_energy_and_source(startup):
    from benchmarks.ad_demos.prepared_column_energy import observers

    r, args, cn, z, _, weights, event, _ = startup
    h, d, pressure, width = (np.asarray(args[j], dtype=np.longdouble) for j in (4, 5, 6, 7))
    tracer, dt = (np.longdouble(args[j]) for j in (13, 12))
    bank_weight = np.asarray(weights, dtype=np.longdouble) / tracer
    rhoarea = bank_weight[0] / width[0]
    mass = np.asarray(rhoarea * h, dtype=float)
    energy = np.asarray(
        observers(
            cn["delta"], args[3], z["N"], args[0], args[1], args[5], args[6], args[13], mass, float(rhoarea), weights
        ),
        dtype=np.longdouble,
    )
    assert abs(energy[0] - r["finite_enthalpy_J"]) <= r["PE_recovery_bound_J"]
    assert abs(energy[3] - r["mixing_work_J"]) <= 0.01
    mean, end, kh, heat = (np.asarray(event[k], dtype=np.longdouble) for k in ("mean", "end", "mean_kh", "heat"))
    Q, M, N = (np.asarray(z[k], dtype=np.longdouble) for k in ("Q", "M", "N"))
    flow = dt * (Q @ mean)
    source = dt * M * np.asarray(event["mean_mu"], dtype=np.longdouble)
    debit = dt * np.asarray(args[11], dtype=np.longdouble)
    bank = end - flow - source - debit + tracer * kh * N + dt * heat
    scale = np.max(source + debit + np.maximum(flow, 0))
    assert (
        np.max(abs(bank) + np.asarray(event["bank_defect_bound"], dtype=np.longdouble))
        <= 500 * np.finfo(float).eps * scale
    )
    assert abs(np.sum(flow * bank_weight)) <= 500 * np.finfo(float).eps * np.sum(abs(flow * bank_weight))
    old = np.asarray(args[0], dtype=np.longdouble)
    du, dv = np.asarray(cn["delta"][2:], dtype=np.longdouble)
    dKE = rhoarea / 2 * np.sum((du * (2 * old[2] + du) + dv * (2 * old[3] + dv)) * h)
    assert abs(dKE + np.sum(source * bank_weight)) <= r["source_bound_J"]
    total = dKE + np.sum((end - debit + dt * heat) * bank_weight) + energy[0] - energy[2]
    assert abs(total) <= r["PE_recovery_bound_J"]


@pytest.mark.parametrize("field", [6, 7])
def test_empty_startup_rejects_incoming_energy_or_history(startup, field):
    import jax.numpy as jnp

    _, _, _, _, event_args, weights, _, function = startup
    changed = list(event_args)
    changed[field] = jnp.full_like(changed[field], 1e-8)
    assert int(function(*changed, weights)["status"]) != 0


def test_empty_startup_rejects_unfunded_background(startup):
    _, _, _, _, event_args, weights, _, function = startup
    changed = list(event_args)
    changed[4] = changed[4] * 2
    assert int(function(*changed, weights)["status"]) != 0
