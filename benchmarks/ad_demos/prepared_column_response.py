"""Experimental mean solve and incoming-energy response for a native column.

Inputs follow prepared_column_native.prepare, with SI work weights
rho * area * width * dt_tracer. Host helpers retain the event domain.
"""

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.ad_demos.prepared_column_native import conditional
from benchmarks.ad_demos.prepared_column_events import MOMENT_FIELDS


def evaluate(log_means, log_energy, args, weights):
    """Evaluate mean equations and mixing work in log coordinates."""
    n = args[2].shape[0]
    values = list(args)
    values[2] = args[2] * jnp.exp(log_means[:n])
    values[3] = jnp.concatenate((args[3][:-1] * jnp.exp(log_means[n:]), jnp.zeros(1, dtype=log_means.dtype)))
    values[9] = args[9] * jnp.exp(log_energy)
    result = conditional(*values)
    cn, prepared, event = result["cn"], result["prepared"], result["event"]
    output = {key: event[key] for key in MOMENT_FIELDS}
    output.update({key: prepared[key] for key in ("N", "Nold", "Nnew", "M", "Q", "fraction", "source0")})
    output.update(
        residual=result["residual"],
        work=jnp.vdot(weights, prepared["N"] * event["mean_kh"]),
        cn_delta=cn["delta"],
        cn_high=cn["high"],
        cn_low=cn["low"],
        input_mu=values[2],
        input_kh=values[3],
        incoming_energy=values[9],
    )
    return output, event


def _require_accepted(output, event):
    if int(event["status"]) != 0 or not bool(event["continuous_branch_certified"]):
        raise ValueError("column is outside the certified event domain")
    if not all(np.isfinite(np.asarray(value)).all() for value in output.values()):
        raise ValueError("column observations must be finite")


def linear_response(args, weights):
    """Return the local implicit incoming-energy response and its direct/adjoint work."""
    size = 2 * args[2].shape[0] - 1
    zero = jnp.zeros(size, dtype=jnp.float64)
    beta = jnp.asarray(0.0, dtype=zero.dtype)
    primal, event = jax.jit(evaluate)(zero, beta, args, weights)
    _require_accepted(primal, event)
    if np.max(abs(np.asarray(primal["residual"]))) > 1e-13:
        raise ValueError("linear response requires an accepted reference mean solution")

    def observations(q, b):
        return evaluate(q, b, args, weights)[0]

    product = jax.jit(lambda direction, amplitude: jax.jvp(observations, (zero, beta), (direction, amplitude))[1])
    basis = [product(jnp.asarray(direction), beta) for direction in np.eye(size)]
    partial = product(zero, jnp.asarray(1.0, dtype=zero.dtype))
    columns = {key: np.stack([np.asarray(value[key]) for value in basis], axis=-1) for key in partial}
    partial = {key: np.asarray(value) for key, value in partial.items()}
    jacobian = columns["residual"]
    log_mean_response = np.linalg.solve(jacobian, -partial["residual"])
    tangent = {key: partial[key] + columns[key] @ log_mean_response for key in partial}
    adjoint = np.linalg.solve(jacobian.T, columns["work"])
    return dict(
        primal=primal,
        jacobian=jacobian,
        log_mean_response=log_mean_response,
        tangent=tangent,
        work_direct=float(tangent["work"]),
        work_adjoint=float(partial["work"] - adjoint @ partial["residual"]),
    )


def solve(args, weights, jacobian, *, log_energy=0.0, initial=None, max_evaluations=16):
    """Solve one nearby physical column with a fixed reference Jacobian and bounded work."""
    if max_evaluations < 2:
        raise ValueError("at least two evaluations are required for the tighter-work check")
    size = 2 * args[2].shape[0] - 1
    q = np.zeros(size) if initial is None else np.asarray(initial, dtype=float).copy()
    function = jax.jit(evaluate)
    previous_work = None
    for count in range(1, max_evaluations + 1):
        output, event = function(jnp.asarray(q), jnp.asarray(log_energy), args, weights)
        _require_accepted(output, event)
        residual = np.asarray(output["residual"])
        cn_work = np.sum(
            np.asarray(weights, dtype=np.longdouble)
            * np.asarray(output["input_kh"], dtype=np.longdouble)
            * np.asarray(output["N"], dtype=np.longdouble)
        )
        work = np.longdouble(output["work"])
        eligible = np.max(abs(residual)) <= 1e-13 and abs(work - cn_work) <= 0.01
        if eligible and previous_work is not None and abs(work - previous_work) <= 0.01:
            return dict(
                log_means=q,
                observations=output,
                event=event,
                evaluations=count,
                tighter_work_change_J=float(abs(work - previous_work)),
            )
        previous_work = work if eligible else None
        q = q - np.linalg.solve(jacobian, residual)
    raise RuntimeError("physical column mean solve exhausted its evaluation cap")
