"""Experimental finite column energy in joules, using paired CN increments.

Return finite enthalpy change, its flux form, nonlinear conversion and mixing work.
"""
import jax
import jax.numpy as jnp
import numpy as np
from veros.core.density.gsw import gsw_dHdT, gsw_dHdS

_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(8)
_THETA = (_NODES + 1) / 2
_WEIGHT = _WEIGHTS / 2


def observers(delta, kh, N, high, low, d, pressure, dt_tracer, mass, area_rho, work_weights):
    if not jax.config.x64_enabled or high.dtype != jnp.float64:
        raise ValueError("finite column energy requires JAX float64")
    theta = jnp.asarray(_THETA)[:, None]
    weight = jnp.asarray(_WEIGHT)[:, None]
    T = high[0][None, :] + (low[0][None, :] + theta * delta[0][None, :])
    S = high[1][None, :] + (low[1][None, :] + theta * delta[1][None, :])
    AT = jnp.sum(weight * gsw_dHdT(S, T, pressure[None, :]), axis=0)
    AS = jnp.sum(weight * gsw_dHdS(S, T, pressure[None, :]), axis=0)
    finite = jnp.vdot(mass, AT * delta[0] + AS * delta[1])
    difference = high[:2, 1:] - high[:2, :-1] + (low[:2, 1:] - low[:2, :-1]) + (delta[:2, 1:] - delta[:2, :-1]) / 2
    flux = kh[None, :-1] * difference / d[None, :-1]
    flux_energy = area_rho * dt_tracer * jnp.sum((AT[:-1] - AT[1:]) * flux[0] + (AS[:-1] - AS[1:]) * flux[1])
    work = jnp.vdot(work_weights, N * kh)
    return jnp.stack((finite, flux_energy, flux_energy - work, work))
