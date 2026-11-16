"""Preserve transport through rounded pressure and continuity updates."""

from veros import runtime_settings
from veros.core.operators import for_loop, scan, numpy as npx

if runtime_settings.backend == "jax":
    from jax.lax import optimization_barrier as _barrier
else:

    def _barrier(value):
        return value


def _two_sum(a, b):
    total = _barrier(a + b)
    back = _barrier(total - a)
    apart = _barrier(total - back)
    return total, _barrier(_barrier(a - apart) + _barrier(b - back))


def _two_product(a, b):
    product = _barrier(a * b)
    split = npx.asarray(2 ** ((npx.finfo(a.dtype).nmant + 2) // 2) + 1, dtype=a.dtype)
    ca, cb = _barrier(split * a), _barrier(split * b)
    ah, bh = _barrier(ca - _barrier(ca - a)), _barrier(cb - _barrier(cb - b))
    al, bl = _barrier(a - ah), _barrier(b - bh)
    error = _barrier(_barrier(_barrier(_barrier(ah * bh) - product) + _barrier(ah * bl)) + _barrier(al * bh))
    return product, _barrier(error + _barrier(al * bl))


def _weighted_sum(value, weight):
    def accumulate(index, carry):
        high, low = carry
        product, error = _two_product(value[..., index], weight[..., index])
        high, increment = _two_sum(high, product)
        return _two_sum(high, _barrier(low + _barrier(increment + error)))

    zero = npx.zeros_like(value[..., 0])
    return for_loop(0, value.shape[-1], accumulate, (zero, zero))


def _subtract_pair(a, b):
    high, error = _two_sum(a[0], -b[0])
    return _two_sum(high, _barrier(error + _barrier(a[1] - b[1])))


def _scale_pair(pair, value):
    high, error = _two_product(pair[0], value)
    return _two_sum(high, _barrier(error + _barrier(pair[1] * value)))


def _divide_pair(pair, denominator):
    quotient = _barrier(pair[0] / denominator)
    product, error = _two_product(quotient, denominator)
    residual, tail = _two_sum(pair[0], -product)
    remainder = _barrier(residual + _barrier(_barrier(tail + pair[1]) - error))
    return _two_sum(quotient, _barrier(remainder / denominator))


def integrate_continuity(u, v, dzt, dxt, dyt, cost, cosu, maskW, hfacU=None, hfacV=None):
    """Integrate horizontal divergence while retaining transport cancellation."""
    partial = hfacU is not None
    dx = _barrier(cost[None, 1:] * dxt[1:, None])
    dy = _barrier(cost[None, 1:] * dyt[None, 1:])

    def accumulate(carry, level):
        def transport(value, fraction):
            if fraction is None:
                return _two_product(value, dzt[level])
            high, low = _two_product(value, fraction)
            return _scale_pair((high, low), dzt[level])

        east = transport(u[1:, 1:, level], hfacU[1:, 1:, level] if partial else None)
        west = transport(u[:-1, 1:, level], hfacU[:-1, 1:, level] if partial else None)
        north = transport(v[1:, 1:, level], hfacV[1:, 1:, level] if partial else None)
        south = transport(v[1:, :-1, level], hfacV[1:, :-1, level] if partial else None)
        zonal = _divide_pair(_subtract_pair(east, west), dx)
        meridional = _divide_pair(
            _subtract_pair(_scale_pair(north, cosu[None, 1:]), _scale_pair(south, cosu[None, :-1])), dy
        )
        high, error = _two_sum(zonal[0], meridional[0])
        high, low = _two_sum(high, _barrier(error + _barrier(zonal[1] + meridional[1])))
        if not partial:
            high, low = high * maskW[1:, 1:, level], low * maskW[1:, 1:, level]
        total, error = _two_sum(carry[0], -high)
        total, residual = _two_sum(total, _barrier(carry[1] + _barrier(error - low)))
        value = _barrier(total + residual)
        if partial:
            value = value * maskW[1:, 1:, level]
        return (total, residual), value

    zero = npx.zeros_like(u[1:, 1:, 0])
    _, values = scan(accumulate, (zero, zero), npx.arange(u.shape[-1]))
    return npx.moveaxis(values, 0, -1)


def preserve_transport(value, weight, target):
    """Reconcile represented wet-column transport with the pressure forcing."""
    weight = npx.broadcast_to(weight, value.shape)
    wet = weight > 0
    index = npx.argmin(npx.where(wet, npx.abs(value), npx.inf), axis=-1)
    selected_weight = npx.take_along_axis(weight, index[..., None], axis=-1)[..., 0]
    selected_weight = npx.where(selected_weight > 0, selected_weight, 1)
    selected = (npx.arange(value.shape[-1]) == index[..., None]) & wet

    def compensate(_, current):
        # Barriers retain rounding residuals under XLA algebraic simplification.
        high, low = _weighted_sum(current, weight)
        residual = _barrier(_barrier(target - high) - low)
        velocity = npx.take_along_axis(current, index[..., None], axis=-1)[..., 0]
        corrected = velocity + residual / selected_weight
        return npx.where(selected, corrected[..., None], current)

    return for_loop(0, 2, compensate, value)
