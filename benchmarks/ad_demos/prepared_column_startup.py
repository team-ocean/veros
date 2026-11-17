"""Experimental moments from an empty turbulent-energy column.

Status zero checks bounded startup error. Only forward moments are verified.
"""

import jax
import jax.numpy as jnp
from benchmarks.ad_demos.prepared_column_events import _propagate


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
