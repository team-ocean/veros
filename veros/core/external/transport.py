"""Preserve column transport through rounded velocity updates."""

from veros import runtime_settings
from veros.core.operators import for_loop, numpy as npx

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
