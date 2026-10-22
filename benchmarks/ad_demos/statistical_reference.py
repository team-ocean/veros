"""Independent forward validation of a refined stable periodic climate target."""

import numpy as np


def periodic_reference(
    prepared,
    carry,
    *,
    observe,
    thresholds,
    period_steps,
    sample_stride,
    window_steps,
    burnin_steps,
    n_windows,
    seed,
    ram_checkpoints=64,
    stability_solver=None,
):
    import jax
    import jax.numpy as jnp
    from veros.core.numerics import calc_initial_conditions_kernel
    from veros.routines import tracing_context

    options = dict(
        observe=observe,
        thresholds=thresholds,
        period_steps=period_steps,
        sample_stride=sample_stride,
        window_steps=window_steps,
        burnin_steps=burnin_steps,
        n_windows=n_windows,
        method="invariant_periodic",
        ram_checkpoints=ram_checkpoints,
    )
    if stability_solver is not None:
        options["stability_solver"] = stability_solver
    result = prepared.statistical_response(carry, **options)
    names = ("residual", "spectral_radius", "stability_residual", "sensitivity_residual")
    audit = {name: float(result[name]) for name in names}
    margin = 1 - audit["spectral_radius"]
    valid = (
        result["residual_kind"] == "full_state"
        and np.isfinite(list(audit.values())).all()
        and audit["residual"] < 1e-7
        and margin > 0
        and audit["stability_residual"] < min(1e-5, margin)
        and audit["sensitivity_residual"] < 1e-5
        and np.isfinite(result["observations"]).all()
    )
    if not valid:
        raise RuntimeError(f"Independent statistical reference has unresolved adjustment: {audit}")
    advance = jax.jit(lambda c, n: jax.lax.fori_loop(0, n, lambda i, x: prepared.step(x)[0], c))
    period_seconds = period_steps * prepared._template.settings.dt_tracer

    def changed(shift, phase):
        vs = carry.variables
        current = prepared.replace_variables(
            carry,
            temp=vs.temp + shift * vs.maskT[..., None],
            time=vs.time + phase,
        )
        state = prepared._make_state(current.variables)
        with tracing_context(), state.variables.unlock():
            state.variables.update(calc_initial_conditions_kernel(state))
        return current._replace(variables=state.variables)

    ensembles = []
    for index in range(4):
        rng = np.random.default_rng(seed + index)
        current = advance(changed(rng.uniform(-0.3, 0.3), rng.uniform(0, period_seconds)), burnin_steps)
        samples = []
        for _ in range(12 * n_windows):
            current = advance(current, sample_stride)
            samples.append(observe(current))
        prepared.check_status(current)
        samples = jnp.stack(samples)
        mean = jnp.mean(samples, axis=0)
        ensembles.append(
            np.asarray(
                jnp.stack(
                    (
                        mean,
                        jnp.mean((samples - mean) ** 2, axis=0),
                        jnp.mean(jax.nn.sigmoid(samples - thresholds), axis=0),
                    )
                )
            )
        )
    ensembles = np.stack(ensembles)
    standard_error = np.std(ensembles, axis=0, ddof=1) / 2
    provider = np.asarray(result["observations"])
    independent = np.all(np.abs(np.mean(ensembles, axis=0) - provider) <= 3 * standard_error + 1e-4)
    repeated = prepared.statistical_response(
        changed(0.3, period_seconds / 2),
        **options,
        diagnostics=False,
    )
    start_error = float(np.max(np.abs(np.asarray(repeated["observations"]) - provider)))
    audit.update(
        method=result["method"],
        phase_samples=int(result["phase_samples"]),
        sample_stride=sample_stride,
        window_steps=window_steps,
        burnin_steps=burnin_steps,
        n_windows=n_windows,
        independent_seeds=list(range(seed, seed + 4)),
        independent_forward_statistics=ensembles.tolist(),
        standard_error=standard_error.tolist(),
        starting_state_error=start_error,
        independent_statistics_passed=bool(independent),
        starting_state_passed=start_error <= 1e-4,
        scope="Stable periodic invariant reference with four independent ordinary forward climates and a changed-start solve.",
    )
    if not independent or not start_error <= 1e-4:
        raise RuntimeError(f"Independent statistical reference has unresolved forward agreement: {audit}")
    return result["observations"], audit
