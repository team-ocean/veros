"""Native energy responses agree with independent pressure integrals."""
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def energy_case(pytestconfig):
    if pytestconfig.getoption("--backend") != "jax":
        pytest.skip("requires the JAX backend")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from benchmarks.ad_demos.prepared_column_native import prepare
    from benchmarks.ad_demos.prepared_column_energy import observers

    jax.config.update("jax_enable_x64", True)
    fixture = json.loads((Path(__file__).parent / "reference_data/prepared_column_events.json").read_text())
    native, physical, reference = (fixture[k] for k in ("native_preparation", "physical_response", "energy_response"))
    args = tuple(jnp.asarray(native["inputs"][k], dtype=jnp.float64) for k in native["argument_order"])
    weights, mass = (jnp.asarray(v) for v in (physical["work_weights"], reference["mass"]))
    n = len(args[2])

    def observed(q):
        values = list(args)
        values[2] = args[2] * jnp.exp(q[:n])
        values[3] = jnp.concatenate((args[3][:-1] * jnp.exp(q[n:]), jnp.zeros(1, dtype=q.dtype)))
        cn, z, _ = prepare(*values)
        return observers(
            cn["delta"],
            values[3],
            z["N"],
            args[0],
            args[1],
            args[5],
            args[6],
            args[13],
            mass,
            reference["area_rho"],
            weights,
        )

    q0 = jnp.zeros(2 * n - 1, dtype=jnp.float64)
    qbeta = jnp.asarray(physical["log_mean_response"])
    primal, tangent = jax.jit(lambda q: jax.jvp(observed, (q,), (qbeta,)))(q0)
    pullback = jax.jit(lambda cot: jax.vjp(lambda q: observed(q) / 1e12, q0)[1](cot)[0])
    gradients = np.stack([np.asarray(pullback(jnp.asarray(cot))) for cot in np.eye(4)])
    points = {c["beta"]: np.asarray(jax.jit(observed)(jnp.asarray(c["log_means"]))) for c in physical["finite"]}
    return dict(
        reference=reference,
        physical=physical,
        primal=np.asarray(primal),
        tangent=np.asarray(tangent),
        gradients=gradients,
        points=points,
    )


def assert_energy_precision(actual, expected, reference):
    errors = abs(np.asarray(actual, dtype=np.longdouble) - np.asarray(expected, dtype=np.longdouble))
    assert np.max(errors[:3]) <= reference["energy_precision_J"]
    assert errors[3] <= reference["work_precision_J"]


def test_energy_values_and_forward_variation(energy_case):
    c = energy_case
    assert_energy_precision(c["primal"], c["reference"]["primal"], c["reference"])
    assert_energy_precision(c["tangent"], c["reference"]["tangent"], c["reference"])
    for point in c["reference"]["finite"]:
        assert_energy_precision(c["points"][point["beta"]], point["values"], c["reference"])


def test_complete_energy_root_adjoints(energy_case):
    c = energy_case
    J = np.asarray(c["physical"]["jacobian"])
    partial = np.asarray(c["reference"]["incoming_residual_partial"])
    values = []
    for gradient in c["gradients"]:
        adjoint = np.linalg.solve(J.T, gradient)
        residual = J.T @ adjoint - gradient
        scale = np.linalg.norm(J.T, np.inf) * np.linalg.norm(adjoint, np.inf) + np.linalg.norm(gradient, np.inf)
        assert np.max(abs(residual)) <= 1e-13 * scale
        values.append(-np.sum(adjoint.astype(np.longdouble) * partial.astype(np.longdouble)) * 1e12)
    assert_energy_precision(values, c["tangent"], c["reference"])
    assert_energy_precision(values, c["reference"]["tangent"], c["reference"])


def test_energy_finite_changes_and_prediction(energy_case):
    c = energy_case
    for radius in (0.001, 0.0003):
        estimate = (c["points"][radius] - c["points"][-radius]) / (2 * radius)
        assert np.all(abs(estimate - c["tangent"]) <= 1e-8 + 0.002 * abs(c["tangent"]))
    for sign in (1, -1):
        errors = [abs(c["points"][sign * r] - c["primal"] - sign * r * c["tangent"]) for r in (0.001, 0.0003)]
        floor = (
            500
            * np.finfo(float).eps
            * np.maximum(abs(c["primal"]), np.maximum(abs(c["points"][0.001]), abs(c["points"][-0.001])))
        )
        assert np.all(np.minimum(*errors) > floor)
        assert np.all(np.log(errors[0] / errors[1]) / np.log(0.001 / 0.0003) >= 1.8)
    achieved = c["points"][0.001] - c["primal"]
    assert np.all(abs(achieved - 0.001 * c["tangent"]) <= 0.1 * abs(achieved))
