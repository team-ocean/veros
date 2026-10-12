"""Research experiments using the finite-trajectory Veros API."""

import tempfile

import numpy as np

from .common import Checks, Ocean, descend, gradient_checks


def forcing_response(ocean, steps, checks):
    import jax.numpy as jnp

    zero = jnp.zeros(3)
    baseline = ocean.advance(zero, steps)
    reference = ocean.heat(baseline)
    scale = 20 * jnp.sum(ocean.area) * steps * ocean.dt

    def loss(z):
        final = ocean.advance(z, steps)
        return jnp.where(final.failure_code == 0, -(ocean.heat(final) - reference) / scale, jnp.nan)

    # The two heat-flux coordinates leave the initial state fixed.
    def objective(forcing):
        return loss(jnp.concatenate((jnp.zeros(1), forcing)))

    value, gradient = gradient_checks(objective, jnp.zeros(2), checks)
    optimized, history = descend(objective, jnp.zeros(2), radius=1)
    final = ocean.advance(jnp.concatenate((jnp.zeros(1), optimized)), steps)
    ocean.prepared.check_status(final)
    flux = 20 * jnp.einsum("k,kij->ij", optimized, ocean.patterns)
    checks.close("prescribed heat-flux budget", jnp.sum(flux * ocean.area) / jnp.sum(ocean.area), 0)
    checks.close("applied heat-flux budget", final.variables.ad_flux_integral / scale, 0, atol=1e-8)
    checks.require("control norm", float(jnp.linalg.norm(optimized)) <= 1 + 1e-12, optimized.tolist())
    checks.require("finite response improves", history[-1] < value - 1e-6, history)
    checks.metrics.update(
        loss_history=history,
        controls=optimized.tolist(),
        sensitivity=gradient.tolist(),
        heat_content_change_joules=float(ocean.heat(final) - reference),
    )


def twin_problem(ocean, steps):
    import jax.numpy as jnp

    truth = jnp.array([0.4, 0.6, -0.3])
    target, final = ocean.observations(truth, steps)
    ocean.prepared.check_status(final)
    # Explicit synthetic observation uncertainty: 0.05 K, 0.02 g/kg, 0.5 Sv.
    nz = ocean.sim.state.settings.nz
    scale = jnp.tile(jnp.concatenate((jnp.full(nz, 0.05), jnp.full(nz, 0.02))), 2)
    scale = jnp.concatenate((scale, jnp.array([0.5])))
    baseline, _ = ocean.observations(jnp.zeros(3), steps)

    def response(z):
        observations, final = ocean.observations(z, steps)
        return jnp.where(final.failure_code == 0, (observations - baseline) / scale, jnp.nan)

    normalized_target = (target - baseline) / scale
    return truth, response, normalized_target


def state_forcing_twin(ocean, steps, checks):
    import jax.numpy as jnp

    truth, response, target = twin_problem(ocean, steps)

    def loss(z):
        residual = response(z)[:-1] - target[:-1]
        return jnp.mean(residual**2)

    initial_loss, _ = gradient_checks(loss, jnp.zeros(3), checks)
    optimized, history = descend(loss, jnp.zeros(3), iterations=20)
    initial_holdout = float(jnp.mean(target[-1] ** 2))
    holdout = float(jnp.mean((response(optimized)[-1] - target[-1]) ** 2))
    checks.require("training misfit reduced by 90%", history[-1] < initial_loss * 0.1, history)
    checks.require("held-out time improves", holdout < initial_holdout * 0.5, [initial_holdout, holdout])
    checks.require(
        "observable controls recovered", float(jnp.linalg.norm(optimized - truth)) < 0.15, optimized.tolist()
    )
    checks.metrics.update(
        truth=truth.tolist(), controls=optimized.tolist(), loss_history=history, held_out_loss=holdout
    )


def mixing_calibration(ocean, steps, checks):
    import jax.numpy as jnp

    if ocean.sim.state.settings.enable_tke:
        names = ("c_k", "c_eps")
    else:
        names = ("kappaH_0", "K_iso_0")
    base = jnp.array([getattr(ocean.sim.state.settings, name) for name in names])
    truth = jnp.log(jnp.array([1.3, 0.8]))

    def parameters(z):
        return dict(zip(names, base * jnp.exp(z)))

    zero = jnp.zeros(3)
    target, final = ocean.observations(zero, steps, parameters=parameters(truth))
    ocean.prepared.check_status(final)
    baseline, _ = ocean.observations(zero, steps, parameters=parameters(jnp.zeros(2)))
    scales = jnp.maximum(jnp.abs(target - baseline), 1e-4)

    def loss(z):
        observed, final = ocean.observations(zero, steps, parameters=parameters(z))
        residual = (observed - target) / scales
        return jnp.where(final.failure_code == 0, jnp.mean(residual**2), jnp.nan)

    initial_loss, _ = gradient_checks(loss, jnp.zeros(2), checks)
    optimized, history = descend(loss, jnp.zeros(2), iterations=20)
    long_target = ocean.advance(zero, 2 * steps, parameters=parameters(truth))
    long_initial = ocean.advance(zero, 2 * steps, parameters=parameters(jnp.zeros(2)))
    long_fit = ocean.advance(zero, 2 * steps, parameters=parameters(optimized))
    for final in (long_target, long_initial, long_fit):
        ocean.prepared.check_status(final)
    long_scale = jnp.maximum(jnp.abs(ocean.observe(long_target) - ocean.observe(long_initial)), 1e-4)
    before = float(jnp.mean(((ocean.observe(long_initial) - ocean.observe(long_target)) / long_scale) ** 2))
    after = float(jnp.mean(((ocean.observe(long_fit) - ocean.observe(long_target)) / long_scale) ** 2))
    checks.require("training misfit reduced by 90%", history[-1] < initial_loss * 0.1, history)
    checks.require("longer-rollout improvement", after < before * 0.5, [before, after])
    checks.require("parameter recovery", float(jnp.linalg.norm(optimized - truth)) < 0.1, optimized.tolist())
    checks.metrics.update(
        parameter_names=names,
        truth=parameters(truth),
        fitted=parameters(optimized),
        loss_history=history,
        longer_rollout_loss=after,
    )


def observation_design(ocean, steps, checks):
    import jax
    import jax.numpy as jnp

    _, response, _ = twin_problem(ocean, steps)
    point = jnp.zeros(3)

    def observe(z):
        return response(z).reshape(-1)

    direction = jnp.array([0.2, -0.3, 0.4])
    values, pullback = jax.vjp(observe, point)
    seed = jnp.sin(jnp.arange(values.size, dtype=values.dtype))
    _, tangent = jax.jvp(observe, (point,), (direction,))
    checks.close("observation transpose identity", jnp.vdot(tangent, seed), jnp.vdot(direction, pullback(seed)[0]))

    def linearize(direction):
        return jax.jvp(observe, (point,), (direction,))[1]

    jacobian = np.stack([np.asarray(linearize(basis)) for basis in jnp.eye(3)], axis=1)
    checks.require("finite observation Jacobian", np.isfinite(jacobian).all(), dict(shape=jacobian.shape))
    if not np.isfinite(jacobian).all():
        return
    # Whitened observation noise and a unit covariance prior in scaled control coordinates.
    covariance = np.eye(3)
    selected, variances = [], [float(covariance[1, 1])]
    for _ in range(min(6, jacobian.shape[0])):
        reductions = []
        for index, row in enumerate(jacobian):
            reduction = (covariance[1] @ row) ** 2 / (1 + row @ covariance @ row)
            reductions.append(-np.inf if index in selected else reduction)
        index = int(np.argmax(reductions))
        row = jacobian[index]
        vector = covariance @ row
        covariance -= np.outer(vector, vector) / (1 + row @ vector)
        selected.append(index)
        variances.append(float(covariance[1, 1]))
    checks.require("target-control variance reduced by 20%", variances[-1] < 0.8 * variances[0], variances)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    checks.require("all controls informed beyond prior scale", singular[-1] > 0.1, singular.tolist())
    checks.close(
        "finite perturbation prediction",
        observe(point + 1e-3 * direction) - observe(point),
        1e-3 * tangent,
        rtol=1e-2,
        atol=1e-6,
    )
    rng = np.random.default_rng(17)
    samples = rng.normal(size=(4000, 3)) @ np.linalg.cholesky(covariance).T
    checks.close("local posterior ensemble variance", np.var(samples[:, 1]), covariance[1, 1], rtol=0.08)
    checks.metrics.update(
        singular_values=singular.tolist(),
        selected_observation_indices=selected,
        variance_history=variances,
        uncertainty_scope="linearized response, independent unit-noise observations, unit control prior",
    )


def gradient_horizon(ocean, steps, checks):
    import jax
    import jax.numpy as jnp

    results = []
    checks.metrics["horizons"] = results
    for length in sorted({max(1, steps // 16), max(1, steps // 4), steps}):

        def record_check():
            item = local.items[-1]
            checks.require(f"{length} steps: {item['name']}", item["passed"], item["detail"])

        local = Checks(on_update=record_check)

        def objective(z):
            observations, final = ocean.observations(z, length)
            return jnp.where(
                final.failure_code == 0, jnp.mean(observations[..., : ocean.sim.state.settings.nz] ** 2), jnp.nan
            )

        value, gradient = gradient_checks(objective, jnp.zeros(3), local)
        final = ocean.advance(jnp.zeros(3), length)
        ocean.prepared.check_status(final)
        _, history = descend(objective, jnp.zeros(3), iterations=1)
        local.require("finite gradient step improves", history[-1] < history[0], history)
        results.append(dict(steps=length, value=value, gradient=gradient.tolist()))
        jax.clear_caches()


def checkpoint_replay(ocean, steps, checks):
    import jax
    import jax.numpy as jnp
    from veros.checkpointing import BinomialSchedule, CheckpointedIntegrator

    point = jnp.array([0.2, 0.3, -0.1])
    parameter_name = "c_k" if ocean.sim.state.settings.enable_tke else "kappaH_0"
    parameters = {parameter_name: jnp.asarray(getattr(ocean.sim.state.settings, parameter_name))}
    checks.metrics["replay_parameter"] = parameter_name

    def terminal(final):
        return jnp.mean(ocean.observe(final) ** 2)

    def reference(z, values):
        return terminal(ocean.advance(z, steps, parameters=values))

    value, (control_gradient, parameter_gradient) = jax.value_and_grad(reference, argnums=(0, 1))(point, parameters)
    carry, initialize_vjp = jax.vjp(ocean.controls, point)
    jax.clear_caches()
    for ram_slots in (1, 2):
        driver = CheckpointedIntegrator(ocean.prepared, terminal_objective=terminal)
        schedule = BinomialSchedule(steps, ram_checkpoints=ram_slots, disk_checkpoints=1)
        with tempfile.TemporaryDirectory(prefix="veros-ad-acceptance-") as directory:
            result = driver.value_and_grad(
                carry, n_steps=steps, parameters=parameters, schedule=schedule, directory=directory
            )
        actual = initialize_vjp(result.gradient)[0]
        checks.close(f"RAM slots {ram_slots}: objective", result.value, value, rtol=1e-9)
        checks.close(f"RAM slots {ram_slots}: control gradient", actual, control_gradient, rtol=1e-5)
        checks.close(
            f"RAM slots {ram_slots}: parameter gradient",
            result.parameter_gradient[parameter_name],
            parameter_gradient[parameter_name],
            rtol=1e-5,
        )
        checks.require(
            f"RAM slots {ram_slots}: RAM budget",
            result.statistics["peak_ram_checkpoints"] <= ram_slots,
            result.statistics,
        )
        checks.require(
            f"RAM slots {ram_slots}: disk budget", result.statistics["peak_disk_checkpoints"] <= 1, result.statistics
        )
        jax.clear_caches()


def run(name, config, checks):
    ocean = Ocean(config)
    steps = config["steps"] or (6 if config["mode"] == "smoke" else round(30 * 86400 / ocean.dt))
    checks.metrics["ocean"] = ocean.metadata(steps)
    globals()[name](ocean, steps, checks)
    if config["mode"] == "acceptance":
        checks.require("30-day ocean window", steps * ocean.dt >= 30 * 86400, ocean.metadata(steps))
    return checks
