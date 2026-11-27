"""Experimental moments from an empty turbulent-energy column.

Status zero checks bounded startup error. The host solve also checks the energy
balance using the diffusivity supplied to the native tracer calculation.
"""

import jax
import jax.numpy as jnp
import numpy as np
from benchmarks.ad_demos.prepared_column_native import prepare
from benchmarks.ad_demos.prepared_column_events import MOMENT_FIELDS, _propagate


def integrate_empty(
    N, Q, M, bg, fraction, ratio, energy, previous, h, eke_rate, total, ck, ce, mxl_min, mu_floor, mu_max, work_weights
):
    if not jax.config.x64_enabled or N.dtype != jnp.float64:
        raise ValueError("empty-column startup requires JAX float64")
    eps = jnp.finfo(N.dtype).eps
    roundoff_margin = 1 + 4096 * eps
    zero = jnp.zeros_like(N)
    a = ck * jnp.sqrt(2 / N)
    gamma = ce * jnp.sqrt(N / 2)
    slope = ((1 - fraction) * a / 10).at[-1].set(0)
    constant = (fraction * bg).at[-1].set(0)
    threshold = N * mxl_min**2 / 2
    A = Q - jnp.diag(ratio * N * slope + gamma)
    source = M * mu_floor + eke_rate
    forcing = source - ratio * N * constant
    mixing_scale = ratio * N * slope * threshold
    heat_scale = gamma * threshold
    discriminant = jnp.sqrt((mixing_scale + heat_scale) ** 2 - 3 * heat_scale * mixing_scale)
    x1 = jnp.clip(((mixing_scale + heat_scale) - discriminant) / (3 * heat_scale), 0, 1)
    x2 = jnp.clip(((mixing_scale + heat_scale) + discriminant) / (3 * heat_scale), 0, 1)

    def rhs_defect(x):
        return -heat_scale * x**3 + (mixing_scale + heat_scale) * x**2 - mixing_scale * x

    defect = jnp.maximum(abs(rhs_defect(x1)), abs(rhs_defect(x2))) * roundoff_margin
    equilibrium = -jnp.linalg.solve(A, forcing)
    error_equilibrium = -jnp.linalg.solve(A, defect)
    forcing_roundoff = 8 * eps * (abs(M * mu_floor) + abs(eke_rate) + abs(ratio * N * constant))
    equilibrium_roundoff = -jnp.linalg.solve(A, forcing_roundoff)
    upper = (equilibrium + error_equilibrium + equilibrium_roundoff) * roundoff_margin
    raw = ck * jnp.maximum(mxl_min, jnp.sqrt(2 * upper / N)) * jnp.sqrt(upper)
    end, integral, ok = _propagate(A, forcing, zero, total)
    end_error, error_integral, bound_ok = _propagate(A, defect, zero, total)
    mean = integral / total
    mean_error = error_integral / total * roundoff_margin
    end_error = end_error * roundoff_margin
    mean_mu = jnp.full_like(N, mu_floor)
    mean_kh = constant + slope * mean
    heat = gamma * mean
    kh_error = (slope * mean_error + slope * threshold / 4) * roundoff_margin
    heat_error = (gamma * mean_error + 4 * heat_scale / 27) * roundoff_margin
    work_bound = jnp.vdot(work_weights, N * kh_error) * roundoff_margin
    bank_defect = (
        end_error + total * (jnp.abs(Q) @ mean_error) + total * ratio * N * kh_error + total * heat_error
    ) * roundoff_margin
    flow = total * (Q @ mean)
    source_amount = total * source
    bank_bound = 500 * eps * jnp.max(source_amount + jnp.maximum(flow, 0))
    bank = end - flow - source_amount + total * ratio * N * mean_kh + total * heat
    moments_ok = (
        jnp.all(end_error <= 1e-16 + 1e-11 * abs(end))
        & jnp.all(mean_error <= 1e-16 + 1e-11 * abs(mean))
        & jnp.all(kh_error <= 1e-16 + 1e-11 * abs(mean_kh))
        & jnp.all(heat_error <= 1e-16 + 1e-11 * abs(heat))
    )
    inputs = (
        jnp.all(M >= 0)
        & jnp.all(eke_rate >= 0)
        & jnp.all(bg >= 0)
        & (bg[-1] == 0)
        & jnp.all(h > 0)
        & jnp.all(work_weights > 0)
        & (fraction >= 0)
        & (fraction <= 1)
        & (ratio > 0)
        & (total > 0)
        & (ck > 0)
        & (ce > 0)
        & (mxl_min > 0)
        & (mu_floor > 0)
        & (mu_max > mu_floor)
    )
    domain = (
        inputs
        & jnp.all(energy == 0)
        & jnp.all(previous == 0)
        & jnp.all(N > 1e-12)
        & jnp.all(forcing >= 0)
        & jnp.all(A - jnp.diag(jnp.diag(A)) >= 0)
        & jnp.all(jnp.sum(A, axis=1) < 0)
        & jnp.all(equilibrium >= 0)
        & jnp.all(error_equilibrium >= 0)
        & jnp.all(raw < mu_floor)
        & jnp.all(raw[:-1] / 10 < bg[:-1])
        & (jnp.max(jnp.sqrt(2 * upper / N)) < jnp.min(h))
    )
    certified = (
        domain
        & ok
        & bound_ok
        & moments_ok
        & jnp.all(end >= 0)
        & (work_bound <= 0.01)
        & jnp.all(abs(bank) + bank_defect <= bank_bound)
    )
    return dict(
        end=end,
        mean=mean,
        mean_mu=mean_mu,
        mean_kh=mean_kh,
        heat=heat,
        status=jnp.where(certified, 0, 1),
        minimum_length_error_certified=certified,
        work_error_bound_J=work_bound,
        bank_defect_bound=bank_defect,
        bank_bound=bank_bound,
        upper_energy=upper,
        end_error=end_error,
        mean_error=mean_error,
        kh_error=kh_error,
        heat_error=heat_error,
    )


def evaluate(log_means, args, weights):
    """Evaluate an empty column with thermal means in log coordinates."""
    values = list(args)
    values[3] = jnp.concatenate((args[3][:-1] * jnp.exp(log_means), jnp.zeros(1, dtype=log_means.dtype)))
    cn, prepared, event_args = prepare(*values)
    event = integrate_empty(*event_args, weights)
    output = {key: event[key] for key in MOMENT_FIELDS}
    output.update({key: prepared[key] for key in ("N", "Nold", "Nnew", "M", "Q", "fraction", "source0")})
    output.update(
        residual=jnp.log(event["mean_kh"][:-1] / values[3][:-1]),
        work=jnp.vdot(weights, prepared["N"] * event["mean_kh"]),
        cn_delta=cn["delta"],
        cn_high=cn["high"],
        cn_low=cn["low"],
        input_mu=values[2],
        input_kh=values[3],
        incoming_energy=values[9],
    )
    return output, event


def _physical_bank(output, event, args):
    dt, tracer = (np.longdouble(args[j]) for j in (12, 13))
    flow = dt * (np.asarray(output["Q"], dtype=np.longdouble) @ np.asarray(output["mean"], dtype=np.longdouble))
    source = dt * np.asarray(output["M"], dtype=np.longdouble) * np.asarray(output["mean_mu"], dtype=np.longdouble)
    debit = dt * np.asarray(args[11], dtype=np.longdouble)
    work = tracer * np.asarray(output["input_kh"], dtype=np.longdouble) * np.asarray(output["N"], dtype=np.longdouble)
    heat = dt * np.asarray(output["heat"], dtype=np.longdouble)
    residual = np.asarray(output["end"], dtype=np.longdouble) - flow - source - debit + work + heat
    bound = 500 * np.finfo(float).eps * np.max(source + debit + np.maximum(flow, 0))
    error = np.max(abs(residual)) + np.max(np.asarray(event["bank_defect_bound"], dtype=np.longdouble))
    return error, bound


def solve(args, weights, *, initial=None, max_evaluations=16):
    """Solve nearby empty-column thermal means at fixed floor viscosity."""
    if max_evaluations < 2:
        raise ValueError("at least two evaluations are required for the tighter-work check")
    if not np.all(abs(np.log(np.asarray(args[2]) / np.asarray(args[19]))) <= 1e-13):
        raise ValueError("empty-column mean solve requires viscosity at its floor")
    q = np.zeros(args[2].shape[0] - 1) if initial is None else np.asarray(initial, dtype=float).copy()
    function = jax.jit(evaluate)
    previous_work = None
    for count in range(1, max_evaluations + 1):
        output, event = function(jnp.asarray(q), args, weights)
        if int(event["status"]) != 0 or not bool(event["minimum_length_error_certified"]):
            raise ValueError("column is outside the certified empty-start domain")
        if not all(np.isfinite(np.asarray(value)).all() for value in output.values()):
            raise ValueError("column observations must be finite")
        residual = np.asarray(output["residual"])
        cn_work = np.sum(
            np.asarray(weights, dtype=np.longdouble)
            * np.asarray(output["input_kh"], dtype=np.longdouble)
            * np.asarray(output["N"], dtype=np.longdouble)
        )
        work = np.longdouble(output["work"])
        bank_error, bank_bound = _physical_bank(output, event, args)
        eligible = np.max(abs(residual)) <= 1e-13 and abs(work - cn_work) <= 0.01 and bank_error <= bank_bound
        if eligible and previous_work is not None and abs(work - previous_work) <= 0.01:
            return dict(
                log_means=q,
                observations=output,
                event=event,
                evaluations=count,
                tighter_work_change_J=float(abs(work - previous_work)),
                physical_bank_error=float(bank_error),
                physical_bank_bound=float(bank_bound),
            )
        previous_work = work if eligible else None
        q = q + residual
    raise RuntimeError("empty-column mean solve exhausted its evaluation cap")
