"""Invariant statistics on a validated stable, periodically forced orbit."""


def periodic_statistics(
    prepared,
    carry,
    *,
    parameters,
    observe,
    thresholds,
    period_steps,
    sample_stride,
    tolerance,
    max_cycles,
    ram_checkpoints,
    conserved,
    diagnostics,
    stability_solver=None,
):
    import math
    import operator
    import jax
    import jax.numpy as jnp

    period_steps, sample_stride = operator.index(period_steps), operator.index(sample_stride)
    if period_steps < 1 or sample_stride < 1:
        raise ValueError("Period and sampling stride must be positive integers")
    count = period_steps // math.gcd(period_steps, sample_stride)
    phases = tuple(index / count for index in range(count))
    result = prepared.periodic_response(
        carry,
        parameters=parameters,
        observe=observe,
        period_steps=period_steps,
        phases=phases,
        tolerance=tolerance,
        max_cycles=max_cycles,
        ram_checkpoints=ram_checkpoints,
        conserved=conserved,
        diagnostics=diagnostics,
        stability_solver=stability_solver,
    )
    samples = result["observations"]
    mean = jnp.mean(samples, axis=0)
    variance = jnp.mean((samples - mean) ** 2, axis=0)
    observations = jnp.stack((mean, variance, jnp.mean(jax.nn.sigmoid(samples - thresholds), axis=0)))
    if diagnostics:
        margin = 1 - result["spectral_radius"]
        valid = (
            (result["residual"] < tolerance)
            & (margin > 0)
            & (result["stability_residual"] < jnp.minimum(1e-5, margin))
            & (result["sensitivity_residual"] < 1e-5)
        )
        observations = jnp.where(valid, observations, jnp.nan)
    return dict(
        result,
        observations=observations,
        method="invariant_periodic",
        standard_error=jnp.zeros_like(observations),
        effective_sample_size=float("inf"),
        window_drift=jnp.zeros_like(observations),
        phase_samples=count,
        uncertainty_scope="Finite periodic-orbit population evaluated in full; no Monte Carlo sampling error. State/sensitivity tolerances and phase-sampling convention remain separate numerical limits.",
        estimator_scope="Stable periodic orbit for the declared period and sampled discrete map; no chaotic invariant-response estimator.",
    )
