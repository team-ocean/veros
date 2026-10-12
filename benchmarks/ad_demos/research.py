"""End-to-end scientific experiments for every task/response pair."""

import numpy as np

from .common import descend
from .matrix import CASES, QUESTIONS, missing_capabilities
from .research_ocean import CONTROL_NAMES, ResearchOcean, closure_law
from .responses import Response


def control_indices(case):
    seasonal = case.response != "equilibrium"
    forcing = (2, 3, 4, 5, 6, 7) if seasonal else (2, 4, 6)
    initial = (0, 1) if case.response in ("local", "trajectory") else ()
    return {
        "sensitivity": forcing,
        "estimation": initial + forcing,
        "calibration": (8, 9, 4),
        "perturbation": (2, 3, 6, 7) if seasonal else (2, 6),
        "closure": (10, 11),
        "observation": initial + (2, 4, 6, 8, 9),
        "geometry": (12, 13, 14, 15),
    }[case.task]


def mean_feature(values, index, kind):
    import jax.numpy as jnp

    if kind == "statistics":
        return values[0, index]
    if kind == "trajectory":
        return values[-1, index]
    return jnp.mean(values[..., index])


def jacobian(function, point):
    """Small control-space Jacobian assembled without batching numerical primitives."""
    import jax
    import jax.numpy as jnp

    return np.stack(
        [
            np.asarray(jax.jvp(function, (point,), (basis,))[1]).reshape(-1)
            for basis in jnp.eye(point.size, dtype=point.dtype)
        ],
        axis=1,
    )


def sensitivity(response, checks):
    import jax
    import jax.numpy as jnp

    point = jnp.zeros(response.indices.size)
    ocean = response.ocean
    indices = (ocean.heat_index, ocean.heat_transport_index, ocean.acc_index, ocean.amoc_index)

    def diagnostics(z, forward=False, seed=0):
        values = response.values(z, forward=forward, seed=seed)
        if response.kind == "statistics":
            return values[:, jnp.asarray(indices)].reshape(-1)
        return jnp.stack([mean_feature(values, index, response.kind) for index in indices])

    def loss(z):
        return jnp.sum(diagnostics(z))

    response.audit(loss, point, checks, lambda z, seed: jnp.sum(diagnostics(z, True, seed)))
    matrix = jacobian(diagnostics, point)
    checks.require(
        "resolved diagnostic sensitivity",
        np.isfinite(matrix).all() and np.linalg.norm(matrix) > 1e-8,
        matrix.tolist(),
    )
    checks.metrics["diagnostic_jacobian"] = matrix.tolist()
    if response.kind == "trajectory" and response.config["mode"] == "acceptance":
        horizons = []
        original = response.steps
        for days in (1, 7, 30, 90, 360):
            response.steps = round(days * 86400 / ocean.dt)
            _, gradient = response.audit(loss, point, checks)
            horizons.append(dict(days=days, gradient=gradient.tolist()))
        response.steps = original
        checks.metrics["wind_and_heat_horizons"] = horizons

    # Validate the derivative at the withheld starting season.
    def withheld(z):
        values = response.values(z, heldout=True)
        return mean_feature(values, ocean.acc_index, response.kind)

    if response.kind != "statistics":
        response.audit(withheld, point, checks)
    checks.metrics["held_out_transport_gradient"] = jax.grad(withheld)(point).tolist()


def fit_controls(response, checks, *, geometry=False, initial_point=None):
    import jax.numpy as jnp

    point = jnp.zeros(response.indices.size) if initial_point is None else initial_point
    truth = jnp.linspace(0.15, 0.4, point.size)
    target = response.values(truth)
    baseline = response.values(point)
    scale = jnp.maximum(jnp.abs(target - baseline), 1e-3)
    # Fit northern profiles and heat transport, withholding southern profiles and ACC.
    nz, ocean = response.ocean.nz, response.ocean
    train_indices = jnp.concatenate(
        (jnp.arange(2 * nz), jnp.asarray([ocean.heat_transport_index, ocean.amoc_index]))
    )
    if response.kind == "local":
        train_indices = jnp.arange(4 * nz + 4)
    hold_indices = jnp.concatenate((jnp.arange(2 * nz, 4 * nz), jnp.asarray([ocean.acc_index])))

    def observation_loss(z, forward=False, seed=0):
        values = response.values(z, forward=forward, seed=seed)
        residual = (values[..., train_indices] - target[..., train_indices]) / scale[..., train_indices]
        return jnp.mean(residual**2)

    def loss(z):
        return observation_loss(z) + 5e-4 * jnp.sum(z**2)

    initial, _ = response.audit(
        loss, point, checks, lambda z, seed: observation_loss(z, True, seed) + 5e-4 * jnp.sum(z**2)
    )
    fit, history = descend(loss, point, iterations=24, radius=1)
    checks.require(
        "training misfit reduced by 90%",
        float(observation_loss(fit)) < 0.1 * float(observation_loss(point)),
        history,
    )
    held_target = response.values(truth, heldout=True)
    held_base = response.values(point, heldout=True)
    held_fit = response.values(fit, heldout=True)
    held_scale = jnp.maximum(jnp.abs(held_target - held_base), 1e-3)

    def held_loss(values):
        return float(
            jnp.mean(
                ((values[..., hold_indices] - held_target[..., hold_indices]) / held_scale[..., hold_indices])
                ** 2
            )
        )

    before, after = held_loss(held_base), held_loss(held_fit)
    checks.require("withheld region and response improve by 50%", after < 0.5 * before, [before, after])
    matrix = jacobian(lambda z: response.values(z)[..., train_indices], point)
    _, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    informed = singular > max(1e-6, singular[0] * 1e-6)
    recovery = np.linalg.norm(vt[informed] @ np.asarray(fit - truth))
    checks.require(
        "observable combinations recovered",
        informed.any() and recovery < 0.15,
        dict(singular_values=singular.tolist(), observable_error=float(recovery)),
    )
    checks.require(
        "no transport diagnostic deteriorates",
        bool(
            jnp.all(
                jnp.abs(held_fit[..., 4 * nz + 1 :] - held_target[..., 4 * nz + 1 :])
                <= jnp.abs(held_base[..., 4 * nz + 1 :] - held_target[..., 4 * nz + 1 :]) + 1e-5
            )
        ),
        dict(baseline=held_base[..., 4 * nz + 1 :], fitted=held_fit[..., 4 * nz + 1 :]),
    )
    ocean.check_controls(response.full(fit), checks)
    checks.metrics.update(
        truth=truth.tolist(),
        fitted=fit.tolist(),
        total_loss_history=history,
        observation_loss=float(observation_loss(fit)),
        prior_loss=float(5e-4 * jnp.sum(fit**2)),
        held_out_loss=after,
        initial_total_loss=initial,
    )
    if geometry:
        geometry_validation(response, fit, checks)
    return fit


def perturbation(response, checks):
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize, nnls

    ocean, point = response.ocean, jnp.zeros(response.indices.size)
    baseline_values = response.values(point)
    heat_baseline = mean_feature(baseline_values, ocean.heat_index, response.kind)

    def heat_loss(z, forward=False, seed=0):
        return -(
            mean_feature(response.values(z, forward=forward, seed=seed), ocean.heat_index, response.kind)
            - heat_baseline
        )

    initial, _ = response.audit(heat_loss, point, checks, lambda z, seed: heat_loss(z, True, seed))
    fit, history = descend(heat_loss, point, radius=1)
    checks.require("bounded heat redistribution improves", history[-1] < initial - 1e-6, history)
    checks.require("admissible forcing norm", float(jnp.linalg.norm(fit)) <= 1 + 1e-9, fit.tolist())
    ocean.check_controls(response.full(fit), checks)
    if response.kind in ("local", "trajectory"):
        initial_response = Response(ocean, (0, 1), response.config)
        zero_initial = jnp.zeros(2)
        initial_baseline = mean_feature(
            initial_response.values(zero_initial), ocean.heat_index, response.kind
        )

        def initial_state_loss(z):
            return -(
                mean_feature(initial_response.values(z), ocean.heat_index, response.kind) - initial_baseline
            )

        initial_response.audit(initial_state_loss, zero_initial, checks)
        initial_fit, initial_history = descend(initial_state_loss, zero_initial, radius=1)
        checks.require(
            "balanced initial perturbation improves",
            initial_history[-1] < initial_history[0] - 1e-6,
            initial_history,
        )
        ocean.check_controls(initial_response.full(initial_fit), checks)
        checks.metrics.update(
            initial_perturbation=initial_fit.tolist(), initial_perturbation_history=initial_history
        )
    if response.kind == "statistics":

        def event_loss(z, forward=False, seed=0):
            return response.values(z, forward=forward, seed=seed)[2, ocean.heat_index]

        response.audit(lambda z: event_loss(z), point, checks, lambda z, seed: event_loss(z, True, seed))
        event_fit, event_history = descend(lambda z: event_loss(z), point, radius=1)
        checks.require(
            "warm-event probability improves", event_history[-1] < event_history[0] - 1e-6, event_history
        )
        ocean.check_controls(response.full(event_fit), checks)
        checks.metrics.update(climate_event_controls=event_fit.tolist(), climate_event_history=event_history)
    baseline = float(mean_feature(baseline_values, ocean.amoc_index, response.kind))
    checks.require("resolved positive overturning", baseline > 1e-5, dict(baseline_sv=baseline))
    target = 0.9 * baseline

    def constraint(z, forward=False, seed=0):
        return target - mean_feature(
            response.values(z, forward=forward, seed=seed), ocean.amoc_index, response.kind
        )

    response.audit(constraint, point, checks, lambda z, seed: constraint(z, True, seed))
    derivative = jax.grad(constraint)
    optimization = minimize(
        lambda z: 0.5 * z @ z,
        np.zeros(point.size),
        jac=lambda z: z,
        method="SLSQP",
        constraints=(
            dict(
                type="ineq",
                fun=lambda z: float(constraint(jnp.asarray(z))),
                jac=lambda z: np.asarray(derivative(jnp.asarray(z))),
            ),
            dict(type="ineq", fun=lambda z: 1 - z @ z, jac=lambda z: -2 * z),
        ),
        options=dict(maxiter=24, ftol=1e-9),
    )
    optimum = jnp.asarray(optimization.x)
    slack = float(constraint(optimum))
    checks.require(
        "ten-percent overturning target",
        bool(optimization.success) and slack >= -1e-5,
        dict(message=str(optimization.message), slack_sv=slack, controls=optimization.x.tolist()),
    )
    gradient = np.asarray(derivative(optimum))
    # Check first-order constrained stationarity.
    columns = np.stack((gradient, -2 * optimization.x), axis=1)
    multipliers, residual = nnls(columns, optimization.x)
    checks.require(
        "minimum-cost stationarity",
        residual < 1e-3,
        dict(residual=residual, multipliers=multipliers.tolist()),
    )
    checks.require(
        "constraint complementarity",
        abs(multipliers[0] * slack) < 1e-3
        and abs(multipliers[1] * (1 - optimization.x @ optimization.x)) < 1e-3,
        multipliers.tolist(),
    )
    ocean.check_controls(response.full(optimum), checks)
    final = ocean.advance(response.full(optimum), response.steps)
    ocean.prepared.check_status(final)
    area = float(jnp.sum(ocean.area))
    checks.close(
        "applied heat redistribution budget",
        final.variables.research_heat_input / (20 * area * response.steps * ocean.dt),
        0,
        atol=1e-8,
    )
    checks.close(
        "applied salinity redistribution budget",
        final.variables.research_salt_input / (3.5e-6 * area * response.steps * ocean.dt),
        0,
        atol=1e-8,
    )
    checks.metrics.update(
        heat_controls=fit.tolist(),
        heat_loss_history=history,
        target_overturning_sv=target,
        minimum_cost_controls=optimization.x.tolist(),
        minimum_scaled_cost=float(0.5 * optimization.x @ optimization.x),
    )


def observation_design(response, checks):
    import jax
    import jax.numpy as jnp

    ocean, point = response.ocean, jnp.zeros(response.indices.size)

    def candidates(z):
        return response.values(z)[..., : 4 * ocean.nz].reshape(-1)

    def target(z):
        return mean_feature(response.values(z, heldout=True), ocean.amoc_index, response.kind)

    matrix = jacobian(candidates, point)
    values, pullback = jax.vjp(candidates, point)
    direction = jnp.arange(1, point.size + 1, dtype=point.dtype)
    seed = jnp.sin(jnp.arange(values.size, dtype=point.dtype))
    tangent = jax.jvp(candidates, (point,), (direction,))[1]
    checks.close(
        "observation transpose identity", jnp.vdot(tangent, seed), jnp.vdot(direction, pullback(seed)[0])
    )
    response.audit(
        target,
        point,
        checks,
        lambda z, seed: mean_feature(
            response.values(z, heldout=True, forward=True, seed=seed), ocean.amoc_index, response.kind
        ),
    )
    target_gradient = np.asarray(jax.grad(target)(point))
    covariance = np.eye(point.size)
    initial_variance = float(target_gradient @ covariance @ target_gradient)
    selected, variances = [], [initial_variance]
    for _ in range(min(12, matrix.shape[0])):
        reductions = [
            (
                (target_gradient @ covariance @ row) ** 2 / (1 + row @ covariance @ row)
                if index not in selected
                else -np.inf
            )
            for index, row in enumerate(matrix)
        ]
        index = int(np.argmax(reductions))
        row = matrix[index]
        projected = covariance @ row
        covariance -= np.outer(projected, projected) / (1 + row @ projected)
        selected.append(index)
        variances.append(float(target_gradient @ covariance @ target_gradient))
    checks.require(
        "future-transport variance reduced by 20%",
        initial_variance > 1e-10 and variances[-1] < 0.8 * initial_variance,
        variances,
    )
    checks.close(
        "finite T/S observation prediction",
        candidates(point + 1e-3 * direction) - values,
        1e-3 * tangent,
        rtol=1e-2,
        atol=1e-5,
    )
    rng = np.random.default_rng(27)
    samples = 0.01 * rng.normal(size=(64, point.size)) @ np.linalg.cholesky(covariance).T
    # Validate transfer with nonlinear forward predictions.
    predictions = np.array(
        [
            float(
                mean_feature(
                    response.values(
                        jnp.asarray(sample), heldout=True, forward=response.kind == "statistics", seed=71
                    ),
                    ocean.amoc_index,
                    response.kind,
                )
            )
            for sample in samples
        ]
    )
    from scipy.stats import chi2

    bounds = 1e-4 * variances[-1] * chi2.ppf([0.025, 0.975], len(samples) - 1) / (len(samples) - 1)
    variance = float(np.var(predictions, ddof=1))
    checks.require(
        "nonlinear future-transport ensemble variance",
        bool(bounds[0] <= variance <= bounds[1]),
        dict(empirical=variance, confidence_bounds=bounds.tolist()),
    )
    singular = np.linalg.svd(matrix, compute_uv=False)
    checks.metrics.update(
        selected_T_S_indices=selected,
        singular_values=singular.tolist(),
        future_transport_variance=variances,
        target_gradient=target_gradient.tolist(),
        uncertainty_scope="local Gaussian control prior and T/S noise; nonlinear forward check",
    )


def closure_training(response, checks):
    import jax.numpy as jnp

    ocean = response.ocean
    reference = ocean.prepared.reference_tracer_tendencies(
        ocean.initial,
        n_states=8,
        seed=31,
        refinement=2,
        projection="conservative",
        withheld_seed=47,
    )
    checks.require(
        "independent refined reference",
        reference["source"] == "refined_veros" and reference["refinement"] >= 2,
        dict(source=reference["source"], refinement=reference["refinement"]),
    )

    # Fit closure tendencies on projected independent reference states.
    def tendency_loss(weights, subset):
        residuals = []
        for carry, target in zip(reference[subset + "_states"], reference[subset + "_tendencies"]):
            diffusivity = closure_law(carry.variables, weights)
            tendency = reference["flux_divergence"](carry, diffusivity)
            residuals.append(jnp.mean(((tendency - target) / reference["tendency_scale"]) ** 2))
        return jnp.mean(jnp.stack(residuals))

    from .common import gradient_checks

    initial, _ = gradient_checks(lambda z: tendency_loss(z, "train"), jnp.zeros(2), checks)
    fit, history = descend(lambda z: tendency_loss(z, "train"), jnp.zeros(2), iterations=24, radius=1)
    checks.require("reference-tendency fit reduced by 90%", history[-1] < 0.1 * initial, history)
    before, after = float(tendency_loss(jnp.zeros(2), "heldout")), float(tendency_loss(fit, "heldout"))
    checks.require("withheld reference tendencies improve", after < 0.5 * before, [before, after])
    checks.close("closure-only heat/salt conservation", reference["budget_residual"](fit), 0, atol=1e-8)
    # Test transfer against an independent refined-model rollout.
    reference_rollout = reference["response_target"](response.kind, steps=response.steps, heldout=True)
    baseline = response.values(jnp.zeros(2), heldout=True)
    learned = response.values(fit, heldout=True)
    scale = jnp.maximum(jnp.abs(reference_rollout - baseline), 1e-3)
    before = float(jnp.mean(((baseline - reference_rollout) / scale) ** 2))
    after = float(jnp.mean(((learned - reference_rollout) / scale) ** 2))
    checks.require("tendency-trained closure transfers to circulation", after < 0.5 * before, [before, after])

    def rollout_loss(z, forward=False, seed=0):
        return jnp.mean(
            ((response.values(z, heldout=True, forward=forward, seed=seed) - reference_rollout) / scale) ** 2
        )

    response.audit(rollout_loss, fit, checks, lambda z, seed: rollout_loss(z, True, seed))
    ocean.check_controls(response.full(fit), checks)
    checks.metrics.update(
        tendency_training_history=history, tendency_weights=fit.tolist(), reference_rollout_error=after
    )


def geometry_validation(response, fit, checks):
    import jax.numpy as jnp

    ocean, full = response.ocean, response.full(fit)
    carry, parameters = ocean.controls(full)
    reference = ocean.prepared.grid_reference_error(
        carry, parameters=parameters, refinement=2, response=response.kind, steps=response.steps
    )
    baseline, base_parameters = ocean.controls(jnp.zeros(len(CONTROL_NAMES)))
    base_error = ocean.prepared.grid_reference_error(
        baseline, parameters=base_parameters, refinement=2, response=response.kind, steps=response.steps
    )
    checks.require(
        "finite refined-grid reference error",
        bool(np.isfinite(reference["error"]) and np.isfinite(base_error["error"])),
        dict(initial=base_error["error"], fitted=reference["error"]),
    )
    checks.close("grid remapping inventories", reference["remap_budget_residual"], 0, atol=1e-8)
    checks.require(
        "geometry-dependent operator changes",
        float(reference["operator_change_norm"]) > 1e-8,
        float(reference["operator_change_norm"]),
    )
    # Discrete realization is evaluated forward, never differentiated as a boolean mask.
    relaxed = response.values(fit, forward=response.kind == "statistics", seed=77)
    realized = response.values(fit, realized=True, forward=response.kind == "statistics", seed=77)
    relaxed_transport, realized_transport = relaxed[..., ocean.acc_index :], realized[..., ocean.acc_index :]
    checks.close("realized channel transport", realized_transport, relaxed_transport, rtol=0.1, atol=1e-3)

    def accuracy(grid_coordinate, forward=False, seed=0):
        z = fit.at[2].set(grid_coordinate[0])
        state, params = ocean.controls(response.full(z))
        return ocean.prepared.grid_reference_error(
            state,
            parameters=params,
            refinement=2,
            response=response.kind,
            steps=response.steps,
            forward=forward,
            seed=seed,
        )["error"]

    grid_point = fit[2:3]
    response.audit(lambda z: accuracy(z), grid_point, checks, lambda z, seed: accuracy(z, True, seed))
    grid_fit, history = descend(accuracy, grid_point, iterations=8, radius=1)
    checks.require("vertical-grid optimization improves accuracy", history[-1] < history[0], history)

    def exchange(depth_coordinates, forward=False, seed=0):
        z = fit.at[:2].set(depth_coordinates)
        return -mean_feature(response.values(z, forward=forward, seed=seed), ocean.acc_index, response.kind)

    response.audit(lambda z: exchange(z), fit[:2], checks, lambda z, seed: exchange(z, True, seed))
    depth_fit, exchange_history = descend(lambda z: exchange(z), fit[:2], iterations=8, radius=1)
    checks.require(
        "bounded bathymetric exchange improves", exchange_history[-1] < exchange_history[0], exchange_history
    )
    checks.require(
        "admissible depth-design norm", float(jnp.linalg.norm(depth_fit)) <= 1 + 1e-9, depth_fit.tolist()
    )
    checks.metrics.update(grid_accuracy_history=history, grid_accuracy_controls=grid_fit.tolist())
    checks.metrics.update(depth_design_controls=depth_fit.tolist(), exchange_design_history=exchange_history)


def run(name, config, checks):
    from veros.integration import PreparedIntegrator

    case = CASES[name]
    indices = control_indices(case)
    checks.metrics.update(
        task=case.task,
        response=case.response,
        research_questions={key: QUESTIONS[key] for key in case.questions},
        controls=[CONTROL_NAMES[index] for index in indices],
        experiment_scope=(
            "global 4-degree research acceptance"
            if config["setup"] == "global_4deg"
            else "small ACC numerical fixture"
        ),
        extension_contracts=case.requirements,
    )
    missing = missing_capabilities(PreparedIntegrator, case)
    if missing:
        for method, contract in missing.items():
            checks.require("required capability: " + method, False, contract)
        checks.metrics["missing_capabilities"] = missing
        return
    ocean = ResearchOcean(config, case)
    checks.metrics["prepared_time_seconds"] = float(ocean.initial.variables.time)
    response = Response(ocean, indices, config)
    import jax.numpy as jnp

    point = jnp.zeros(len(indices))
    response.validate(point, checks)
    {
        "sensitivity": sensitivity,
        "estimation": fit_controls,
        "calibration": fit_controls,
        "perturbation": perturbation,
        "closure": closure_training,
        "observation": observation_design,
        "geometry": lambda r, c: fit_controls(r, c, geometry=True),
    }[case.task](response, checks)
