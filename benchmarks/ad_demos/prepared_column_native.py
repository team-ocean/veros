"""Experimental native preparation of a prescribed-column mixing mean equation.

Four carried fields (T, S, U, V) use residual-form Crank-Nicolson increments and
compensated endpoint sums. Common-pressure EOS5 density differences use a fixed
eight-point integral of the native density differential along vertical intervals.
Mean shear, conservative energy transport and available background funding then
supply the event component.
"""
import jax
import jax.numpy as jnp
import numpy as np
from veros.core.operators import solve_tridiagonal_jax
from veros.core.density.gsw import gsw_rho
from benchmarks.ad_demos.prepared_column_events import integrate


def _increments(high, low, coefficient, h, d, dt):
    """Solve four carried fields using the native Veros tridiagonal primitive."""
    q = coefficient[:, :-1] / d[None, :-1]
    lower = -dt[:, None] / 2 * q / h[None, 1:]
    upper = -dt[:, None] / 2 * q / h[None, :-1]
    diagonal = jnp.ones_like(high).at[:, :-1].add(-upper).at[:, 1:].add(-lower)
    difference = high[:, 1:] - high[:, :-1] + (low[:, 1:] - low[:, :-1])
    flux = q * difference
    rhs = jnp.zeros_like(high).at[:, :-1].add(dt[:, None] * flux / h[None, :-1])
    rhs = rhs.at[:, 1:].add(-dt[:, None] * flux / h[None, 1:])
    zero = jnp.zeros((high.shape[0], 1), dtype=high.dtype)
    a = jnp.concatenate((zero, lower), axis=1)[:, None, :]
    b = diagonal[:, None, :]
    c = jnp.concatenate((upper, zero), axis=1)[:, None, :]
    mask = jnp.ones_like(b, dtype=bool)
    edge = jnp.zeros_like(mask).at[:, :, 0].set(True)
    delta = solve_tridiagonal_jax(a, b, c, rhs[:, None, :], mask, edge)[:, 0, :]
    endpoint_high = jax.lax.optimization_barrier(high + delta)
    rounded_delta = jax.lax.optimization_barrier(endpoint_high - high)
    endpoint_low = high - (endpoint_high - rounded_delta) + (delta - rounded_delta) + low
    return dict(delta=delta, high=endpoint_high, low=endpoint_low)


_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(8)
_NODES = (_NODES + 1) / 2
_WEIGHTS = _WEIGHTS / 2


def _stability(old_high, old_low, new_high, new_low, pressure, d, gravity, rho0):
    """Integrate the density differential along each carried vertical interval."""
    theta = jnp.asarray(_NODES)[:, None]
    weights = jnp.asarray(_WEIGHTS)[:, None]
    results = []
    for high, low in ((old_high, old_low), (new_high, new_low)):
        delta = high[:, 1:] - high[:, :-1] + (low[:, 1:] - low[:, :-1])
        T = high[0, :-1][None, :] + theta * delta[0][None, :] + low[0, :-1][None, :]
        S = high[1, :-1][None, :] + theta * delta[1][None, :] + low[1, :-1][None, :]
        tangent = (jnp.broadcast_to(delta[1], S.shape), jnp.broadcast_to(delta[0], T.shape))
        _, density_difference = jax.jvp(lambda S, T: gsw_rho(S, T, pressure[:-1][None, :]), (S, T), tangent)
        Nsqr = -gravity / rho0 * jnp.sum(weights * density_difference, axis=0) / d[:-1]
        results.append(jnp.concatenate((Nsqr, Nsqr[-1:])))
    return dict(Nold=results[0], Nnew=results[1], Nmid=(results[0] + results[1]) / 2)


def prepare(
    high,
    low,
    mu,
    kh,
    h,
    d,
    pressure,
    width,
    bg,
    energy,
    previous,
    eke_rate,
    dt_mom,
    dt_tracer,
    gravity,
    rho0,
    ck,
    ce,
    mxl_min,
    mu_floor,
    mu_max,
    alpha_tke,
):
    """Return carried CN fields, physical coefficients and arguments for the event component."""
    if high.ndim != 2 or high.shape[0] != 4 or low.shape != high.shape or (high.shape[1] < 2):
        raise ValueError("native preparation requires four paired fields and at least two levels")
    if not jax.config.x64_enabled or high.dtype != jnp.float64:
        raise ValueError("native preparation requires JAX float64")
    coefficient = jnp.stack((kh, kh, mu, mu))
    dt = jnp.stack((dt_tracer, dt_tracer, dt_mom, dt_mom))
    cn = _increments(high, low, coefficient, h, d, dt)
    stratification = _stability(high, low, cn["high"], cn["low"], pressure, d, gravity, rho0)
    N = stratification["Nmid"]
    difference = (
        high[2:, 1:] - high[2:, :-1] + (low[2:, 1:] - low[2:, :-1]) + (cn["delta"][2:, 1:] - cn["delta"][2:, :-1]) / 2
    )
    M = jnp.concatenate((jnp.sum(difference**2, axis=0) / d[:-1] ** 2, jnp.zeros(1, dtype=N.dtype)))
    ratio = dt_tracer / dt_mom
    source0 = M * mu_floor + eke_rate
    fraction = jnp.minimum(1, jnp.min(source0[:-1] / (ratio * bg[:-1] * N[:-1])))
    fraction = jnp.where(fraction < 1, fraction * (1 - 8 * jnp.finfo(N.dtype).eps), fraction)
    conductance = alpha_tke * 0.5 * (mu[:-1] + mu[1:]) / h[1:]
    upper = conductance / width[:-1]
    lower = conductance / width[1:]
    diagonal = jnp.zeros_like(N).at[:-1].add(-upper).at[1:].add(-lower)
    Q = jnp.diag(diagonal) + jnp.diag(upper, 1) + jnp.diag(lower, -1)
    prepared = dict(
        N=N,
        Nold=stratification["Nold"],
        Nnew=stratification["Nnew"],
        M=M,
        Q=Q,
        source0=source0,
        fraction=fraction,
        ratio=ratio,
    )
    return (
        cn,
        prepared,
        (N, Q, M, bg, fraction, ratio, energy, previous, h, eke_rate, dt_mom, ck, ce, mxl_min, mu_floor, mu_max),
    )


def conditional(
    high,
    low,
    mu,
    kh,
    h,
    d,
    pressure,
    width,
    bg,
    energy,
    previous,
    eke_rate,
    dt_mom,
    dt_tracer,
    gravity,
    rho0,
    ck,
    ce,
    mxl_min,
    mu_floor,
    mu_max,
    alpha_tke,
):
    """Evaluate the conditional mean residual without solving for its zero."""
    cn, z, event_args = prepare(
        high,
        low,
        mu,
        kh,
        h,
        d,
        pressure,
        width,
        bg,
        energy,
        previous,
        eke_rate,
        dt_mom,
        dt_tracer,
        gravity,
        rho0,
        ck,
        ce,
        mxl_min,
        mu_floor,
        mu_max,
        alpha_tke,
    )
    event = integrate(*event_args)
    residual = jnp.concatenate((jnp.log(event["mean_mu"] / mu), jnp.log(event["mean_kh"][:-1] / kh[:-1])))
    return dict(cn=cn, prepared=z, event=event, residual=residual)
