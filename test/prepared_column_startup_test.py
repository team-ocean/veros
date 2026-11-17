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


def test_startup_bound_covers_source_cancellation(startup):
    import jax
    import jax.numpy as jnp
    from benchmarks.ad_demos.prepared_column_native import prepare

    r, args, _, _, _, weights, _, function = startup
    case = r["half_step_certificate"]
    changed = list(args)
    changed[2], changed[3] = (jnp.asarray(case[key]) for key in ("mu", "kh"))
    changed[12], changed[13] = (jnp.asarray(case[key]) for key in ("dt_mom", "dt_tracer"))
    _, _, event_args = jax.jit(prepare)(*changed)
    event = function(*event_args, weights / 2)
    assert int(event["status"]) == 0
    assert np.all(np.asarray(event["upper_energy"]) >= case["upper_energy"])


@pytest.fixture(scope="module")
def physical_startup_points(startup):
    import jax.numpy as jnp
    from benchmarks.ad_demos.prepared_column_startup import solve

    r, args, _, _, _, weights, _, _ = startup
    points = {}
    for case in r["physical_solve"]["cases"]:
        changed = list(args)
        for i in (0, 1):
            changed[i] = args[i].at[2:].multiply(jnp.exp(case["alpha"]))
        points[case["alpha"]] = solve(tuple(changed), weights, initial=case["initial_log_means"])
    return points


@pytest.mark.parametrize("alpha", [0.001, -0.001, 0.0003, -0.0003])
def test_empty_physical_solve_matches_independent_trials(startup, physical_startup_points, alpha):
    r, _, _, _, _, _, _, _ = startup
    case = next(x for x in r["physical_solve"]["cases"] if x["alpha"] == alpha)
    point = physical_startup_points[alpha]
    out = point["observations"]
    assert point["evaluations"] <= 16
    assert point["tighter_work_change_J"] <= 0.01
    assert point["physical_bank_error"] <= point["physical_bank_bound"]
    assert np.max(abs(np.asarray(out["residual"]))) <= 1e-13
    assert abs(np.longdouble(out["work"]) - np.longdouble(case["mixing_work_J"])) <= 0.01
    for key, value in case["moments"].items():
        np.testing.assert_allclose(np.asarray(out[key]), value, rtol=1e-11, atol=1e-16)
    for i, key in enumerate(("T", "S", "U", "V")):
        carried = np.asarray(out["cn_high"][i], dtype=np.longdouble) + np.asarray(out["cn_low"][i], dtype=np.longdouble)
        np.testing.assert_allclose(carried, case["CN"][key], rtol=1e-12, atol=1e-11)


def test_empty_solve_rejects_energy_invalid_near_root(startup):
    import jax
    import jax.numpy as jnp
    from benchmarks.ad_demos.prepared_column_startup import evaluate, solve

    r, args, _, _, _, weights, _, _ = startup
    case = r["physical_solve"]["energy_rejection"]
    changed = list(args)
    for i in (0, 1):
        changed[i] = args[i].at[2:].multiply(jnp.exp(case["alpha"]))
    changed = tuple(changed)
    out, event = jax.jit(evaluate)(jnp.asarray(case["near_root_log_means"]), changed, weights)
    assert int(event["status"]) == 0
    assert np.max(abs(np.asarray(out["residual"]))) <= 1e-13
    cn_work = np.sum(np.asarray(weights, dtype=np.longdouble) * np.asarray(out["N"]) * np.asarray(out["input_kh"]))
    assert abs(np.longdouble(out["work"]) - cn_work) <= 0.01
    assert case["original_combined_bank_error"] > case["original_bank_bound"]
    with pytest.raises(RuntimeError, match="evaluation cap"):
        solve(changed, weights, initial=case["near_root_log_means"], max_evaluations=2)
    point = solve(changed, weights, initial=case["near_root_log_means"])
    assert point["evaluations"] > 2
    assert point["physical_bank_error"] <= point["physical_bank_bound"]


@pytest.mark.parametrize("field", [9, 10])
def test_empty_solve_rejects_energy_or_history(startup, field):
    import jax.numpy as jnp
    from benchmarks.ad_demos.prepared_column_startup import solve

    _, args, _, _, _, weights, _, _ = startup
    changed = list(args)
    changed[field] = jnp.full_like(changed[field], 1e-8)
    with pytest.raises(ValueError, match="certified empty-start domain"):
        solve(tuple(changed), weights)


def test_empty_solve_requires_floor_viscosity(startup):
    from benchmarks.ad_demos.prepared_column_startup import solve

    _, args, _, _, _, weights, _, _ = startup
    changed = list(args)
    changed[2] = args[2] * 1.01
    with pytest.raises(ValueError, match="viscosity at its floor"):
        solve(tuple(changed), weights)
