"""Positive-part TKE diffusion and conservative numerical-work allocation."""

from veros import runtime_settings, veros_kernel
from veros.core import utilities
from veros.core.operators import numpy as npx, update, at
from veros.distributed import global_sum


def _while_loop(condition, body, initial):
    if runtime_settings.backend == "jax":
        from jax.lax import while_loop

        return while_loop(condition, body, initial)
    result = initial
    while condition(result):
        result = body(result)
    return result


@veros_kernel
def redistribute(power, energy, volume, dt):
    """Allocate global power across interior and surface W cells.

    Inputs cover each rank's owned cells. ``power`` is already globally summed.
    ``volume`` uses the work quadrature, including its surface half cell. The
    bounded interior rate sends the remaining power through the surface cells.
    """
    positive = npx.where(energy[..., :-1] > 0, energy[..., :-1], 0.0)
    interior_volume = volume[..., :-1]
    surface_volume = global_sum(npx.sum(volume[..., -1]))
    has_surface = surface_volume > 0
    safe_surface = npx.where(has_surface, surface_volume, 1.0)

    def weight(rate):
        denominator = positive + dt * npx.abs(rate)
        return positive / npx.where(denominator > 0, denominator, 1.0)

    def equation(rate):
        weighted_volume = global_sum(npx.sum(interior_volume * weight(rate)))
        return npx.where(has_surface, rate * (surface_volume + weighted_volume) - power, rate)

    def solve(function, initial):
        magnitude = npx.abs(power)
        sign = npx.sign(power)
        upper = magnitude / safe_surface
        tolerance = 64 * npx.finfo(energy.dtype).eps * magnitude

        def value_and_slope(rate):
            ratio = weight(rate)
            value = rate * (surface_volume + global_sum(npx.sum(interior_volume * ratio))) - magnitude
            slope = surface_volume + global_sum(npx.sum(interior_volume * ratio**2))
            return npx.where(has_surface, value, rate), npx.where(has_surface, slope, 1.0)

        def condition(state):
            count, rate, residual = state
            return (count < 32) & (npx.abs(residual) > tolerance)

        def iteration(state):
            count, rate, residual = state
            _, slope = value_and_slope(rate)
            rate = npx.clip(rate - residual / slope, 0, upper)
            residual, _ = value_and_slope(rate)
            return count + 1, rate, residual

        guess = npx.abs(initial)
        residual, _ = value_and_slope(guess)
        _, rate, residual = _while_loop(condition, iteration, (0, guess, residual))
        return npx.where(npx.abs(residual) <= tolerance, sign * rate, npx.nan)

    initial_volume = surface_volume + global_sum(npx.sum(interior_volume * (positive > 0)))
    initial = power / npx.where(initial_volume > 0, initial_volume, 1.0)
    if runtime_settings.backend == "jax":
        from jax.lax import custom_root

        rate = custom_root(equation, initial, solve, lambda linear, rhs: rhs / linear(npx.ones_like(rhs)))
    else:
        rate = solve(equation, initial)
    work = npx.zeros_like(energy)
    work = update(work, at[..., :-1], rate * weight(rate))
    work = update(work, at[..., -1], rate)
    valid = has_surface | (power == 0)
    return npx.where(valid, npx.where(has_surface, work, 0.0), npx.nan)


def _positive(active, arguments):
    a, b, c, d, water, edge, b_edge, rates = arguments
    return utilities.solve_implicit(
        npx.where(active, a, 0.0),
        npx.where(active, b + rates, 1.0),
        npx.where(active, c, 0.0),
        npx.where(active, d, 0.0),
        water,
        b_edge=npx.where(active, b_edge + rates, 1.0),
        edge_mask=edge,
    )


def _raw(positive, arguments):
    a, b, c, d, water, edge, b_edge, rates = arguments
    diagonal = npx.where(edge, b_edge, b) + rates
    result = d - (diagonal - 1) * positive
    lower = npx.where(edge, 0.0, a)
    result = update(result, at[..., 1:], result[..., 1:] - lower[..., 1:] * positive[..., :-1])
    result = update(result, at[..., :-1], result[..., :-1] - c[..., :-1] * positive[..., 1:])
    return npx.where(water, result, 0.0)


@veros_kernel
def positive_diffusion_column(a, b, c, d, water, edge, b_edge, rates):
    """Solve E + (L + R) E+ = rhs with a nonnegative surface constraint.

    ``b`` and ``b_edge`` contain I + L, and ``rates`` contains damping R.
    The positive part uses a monotone M-matrix complementarity solve.
    Signed interior energy and advection can leave negative timestep values.
    """
    arguments = (a, b, c, d, water, edge, b_edge, rates)
    if runtime_settings.backend == "jax":
        import jax

        frozen = jax.tree.map(jax.lax.stop_gradient, arguments)
    else:
        frozen = arguments
    initial = (frozen[3] > 0) & water

    def condition(state):
        count, active, changed = state
        return (count < d.shape[-1] + 1) & changed

    def iteration(state):
        count, active, _ = state
        positive = _positive(active, frozen)
        raw = _raw(positive, frozen)
        updated = (active | (raw > 0)) & water
        return count + 1, updated, npx.any(updated != active)

    _, active, changed = _while_loop(condition, iteration, (0, initial, npx.asarray(True)))
    positive = _positive(active, arguments)
    raw = npx.where(active, positive, _raw(positive, arguments))
    correction = npx.where(water[..., -1], npx.maximum(-raw[..., -1], 0.0), 0.0)
    energy = update(raw, at[..., -1], npx.maximum(raw[..., -1], 0.0))
    consistent = (~changed) & npx.all(positive >= 0) & npx.all(active | (~water) | (_raw(positive, arguments) <= 0))
    return npx.where(consistent, energy, npx.nan), npx.where(consistent, correction, npx.nan)
