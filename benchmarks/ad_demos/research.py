"""End-to-end scientific experiments for every task/response pair."""

import numpy as np

from .common import descend
from .matrix import CASES, QUESTIONS, missing_capabilities
from .research_ocean import CONTROL_NAMES, ResearchOcean, closure_law
from .responses import Response
from .finite_response import value_and_grad
from .inverse import JaxResidual, multistart_least_squares, constrained_minimum, observation_covariance


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

    if hasattr(function, "jacobian"):
        return function.jacobian(point)
    return np.stack(
        [
            np.asarray(jax.jvp(function, (point,), (basis,))[1]).reshape(-1)
            for basis in jnp.eye(point.size, dtype=point.dtype)
        ],
        axis=1,
    )


def sensitivity(response, checks):
    import jax.numpy as jnp

    point = jnp.zeros(response.indices.size)
    ocean = response.ocean
    indices = (ocean.heat_index, ocean.heat_transport_index, ocean.acc_index, ocean.amoc_index)

    def diagnostics(z, forward=False, seed=0):
        values = response.values(z, forward=forward, seed=seed)
        if response.kind == "statistics":
            return values[:, jnp.asarray(indices)].reshape(-1)
        return jnp.stack([mean_feature(values, index, response.kind) for index in indices])

    diagnostics_product = response.function(
        lambda values, z: (
            values[:, jnp.asarray(indices)].reshape(-1)
            if response.kind == "statistics"
            else jnp.stack([mean_feature(values, index, response.kind) for index in indices])
        )
    )
    loss = response.function(lambda values, z: jnp.sum(diagnostics_product.function(values, z)))

    response.audit(loss, point, checks, lambda z, seed: jnp.sum(diagnostics(z, True, seed)))
    matrix = jacobian(diagnostics_product, point)
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
    withheld = response.function(lambda values, z: mean_feature(values, ocean.acc_index, response.kind), heldout=True)

    if response.kind != "statistics":
        response.audit(withheld, point, checks)
    checks.metrics["held_out_transport_gradient"] = value_and_grad(withheld)(point)[1].tolist()


def fit_controls(response, checks, *, geometry=False, initial_point=None):
    import jax.numpy as jnp

    point = jnp.zeros(response.indices.size) if initial_point is None else initial_point
    truth = jnp.linspace(0.15, 0.4, point.size)
    target = response.values(truth)
    baseline = response.values(point)
    scale = jnp.maximum(jnp.abs(target - baseline), 1e-3)
    # Fit northern profiles and heat transport, withholding southern profiles and ACC.
    nz, ocean = response.ocean.nz, response.ocean
    train_indices = jnp.concatenate((jnp.arange(2 * nz), jnp.asarray([ocean.heat_transport_index, ocean.amoc_index])))
    if response.kind == "local":
        train_indices = jnp.arange(4 * nz + 4)
    hold_indices = jnp.concatenate((jnp.arange(2 * nz, 4 * nz), jnp.asarray([ocean.acc_index])))

    def observation_loss(z, forward=False, seed=0):
        values = response.values(z, forward=forward, seed=seed)
        residual = (values[..., train_indices] - target[..., train_indices]) / scale[..., train_indices]
        return jnp.mean(residual**2)

    residual = response.function(
        lambda values, z: (
            (values[..., train_indices] - target[..., train_indices]) / scale[..., train_indices]
        ).reshape(-1)
        * jnp.sqrt(2 / target[..., train_indices].size)
    )
    # Exact synthetic observations test recovery without shrinkage bias.
    loss = response.function(lambda values, z: 0.5 * jnp.sum(residual.function(values, z) ** 2))

    initial, initial_gradient = response.audit(loss, point, checks, lambda z, seed: observation_loss(z, True, seed))

    solver_options = {}
    if geometry:

        def admissible(z):
            carry, _ = ocean.controls(response.full(z))
            code = int(carry.failure_code)
            if code in (4, 5, 6):
                checks.metrics.setdefault("inverse_rejected_controls", []).append(
                    dict(controls=z.tolist(), failure_code=code)
                )
                if checks.on_update is not None:
                    checks.on_update()
                return False
            ocean.prepared.check_status(carry)
            return True

        solver_options["admissible"] = admissible

    optimizer_point = point
    if geometry and response.kind == "trajectory":
        from .inverse import gradient_start

        search = dict(initial_controls=point.tolist(), initial_value=float(initial), trials=[])
        checks.metrics["inverse_initial_gradient_search"] = search

        def record_trial(trial):
            search["trials"].append(trial)
            if checks.on_update is not None:
                checks.on_update()

        optimizer_point, _ = gradient_start(
            loss, point, initial, initial_gradient, admissible=admissible, on_trial=record_trial
        )
        search["selected_controls"] = optimizer_point.tolist()

    def progress(measurement):
        checks.metrics["inverse_progress"] = measurement
        if measurement["stage"] != "Jacobian":
            checks.metrics.setdefault("inverse_progress_history", []).append(measurement)
        if checks.on_update is not None:
            checks.on_update()

    fit, history, solver = multistart_least_squares(
        residual,
        optimizer_point,
        prior_precision=0,
        radius=1,
        on_iteration=progress,
        zero_loss_tolerance=1e-14,
        **solver_options,
    )
    if not np.array_equal(optimizer_point, point):
        history = [float(initial)] + history
        solver["initialization"] = "bounded normalized-gradient search"
    checks.require("inverse solver convergence", solver["converged"], solver)
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
                ((values[..., hold_indices] - held_target[..., hold_indices]) / held_scale[..., hold_indices]) ** 2
            )
        )

    before, after = held_loss(held_base), held_loss(held_fit)
    checks.require("withheld region and response improve by 50%", after < 0.5 * before, [before, after])
    matrix = jacobian(response.function(lambda values, z: values[..., train_indices]), point)
    _, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    informed = singular > max(1e-6, singular[0] * 1e-6)
    recovery = np.linalg.norm(vt[informed] @ np.asarray(fit - truth))
    observation_singular = np.linalg.svd(matrix, compute_uv=False)
    objective_matrix = matrix / np.asarray(scale[..., train_indices]).reshape(-1, 1) / np.sqrt(matrix.shape[0])
    objective_singular = np.linalg.svd(objective_matrix, compute_uv=False)
    _, _, combinations = np.linalg.svd(matrix, full_matrices=True)
    rank = int(np.sum(informed))
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
        prior_loss=0.0,
        inverse_target="noise-free synthetic observations; unregularized recovery within the admissible domain",
        held_out_loss=after,
        initial_total_loss=initial,
        optimizer=solver,
        truth_observation_loss=float(observation_loss(truth)),
        truth_total_loss=float(loss(truth)),
        truth_control_norm=float(jnp.linalg.norm(truth)),
        identifiability=dict(
            numerical_rank=rank,
            numerical_cutoff=max(1e-6, singular[0] * 1e-6),
            observation_singular_values=observation_singular.tolist(),
            objective_singular_values=objective_singular.tolist(),
            weak_observation_combinations=combinations[: singular.size][singular < 1].tolist(),
            null_combinations=combinations[rank:].tolist(),
            observation_error_scope="physical observation scales in ResearchOcean.observe",
            posterior_covariance=np.linalg.inv(np.eye(point.size) + matrix.T @ matrix).tolist(),
            combinations_above_observation_noise=int(np.sum(observation_singular > 1)),
            objective_scale_scope="truth/baseline difference floored at 1e-3; mean squared residual",
            prior_weight=0.0,
        ),
    )
    if geometry:
        geometry_validation(response, fit, checks)
    return fit


def perturbation(response, checks):
    import jax.numpy as jnp

    def progress(stage):
        def record(measurement):
            checks.metrics["perturbation_progress"] = dict(optimization_stage=stage, **measurement)
            if measurement["stage"] == "evaluation":
                checks.metrics.setdefault("perturbation_evaluations", {}).setdefault(stage, []).append(measurement)
            if measurement["stage"] in ("gradient", "accepted", "complete"):
                checks.metrics.setdefault("perturbation_checkpoints", {})[stage] = measurement
            if checks.on_update is not None:
                checks.on_update()

        return record

    ocean, point = response.ocean, jnp.zeros(response.indices.size)
    baseline_values = response.values(point)
    heat_baseline = mean_feature(baseline_values, ocean.heat_index, response.kind)

    def heat_loss(z, forward=False, seed=0):
        return -(
            mean_feature(response.values(z, forward=forward, seed=seed), ocean.heat_index, response.kind)
            - heat_baseline
        )

    heat_forward = heat_loss
    heat_loss = (
        response.terminal_feature(ocean.heat_index, scale=-1, offset=heat_baseline)
        if response.kind == "trajectory"
        else response.function(
            lambda values, z: -(mean_feature(values, ocean.heat_index, response.kind) - heat_baseline)
        )
    )
    initial, _ = response.audit(heat_loss, point, checks, lambda z, seed: heat_forward(z, True, seed))
    heat_optimizer = {}
    fit, history = descend(heat_loss, point, radius=1, diagnostics=heat_optimizer, on_iteration=progress("heat"))
    checks.metrics.update(heat_controls=fit.tolist(), heat_loss_history=history, heat_optimizer=heat_optimizer)
    checks.require("bounded heat redistribution improves", history[-1] < initial - 1e-6, history)
    checks.require("admissible forcing norm", float(jnp.linalg.norm(fit)) <= 1 + 1e-9, fit.tolist())
    ocean.check_controls(response.full(fit), checks)
    if response.kind in ("local", "trajectory"):
        initial_response = Response(ocean, (0, 1), response.config)
        zero_initial = jnp.zeros(2)
        initial_baseline = mean_feature(initial_response.values(zero_initial), ocean.heat_index, response.kind)

        initial_state_loss = (
            initial_response.terminal_feature(ocean.heat_index, scale=-1, offset=initial_baseline)
            if response.kind == "trajectory"
            else initial_response.function(
                lambda values, z: -(mean_feature(values, ocean.heat_index, response.kind) - initial_baseline)
            )
        )

        initial_response.audit(initial_state_loss, zero_initial, checks)
        initial_optimizer = {}
        initial_fit, initial_history = descend(
            initial_state_loss, zero_initial, radius=1, diagnostics=initial_optimizer, on_iteration=progress("initial")
        )
        checks.require(
            "balanced initial perturbation improves",
            initial_history[-1] < initial_history[0] - 1e-6,
            initial_history,
        )
        ocean.check_controls(initial_response.full(initial_fit), checks)
        checks.metrics.update(
            initial_perturbation=initial_fit.tolist(),
            initial_perturbation_history=initial_history,
            initial_perturbation_optimizer=initial_optimizer,
        )
    if response.kind == "statistics":

        def event_loss(z, forward=False, seed=0):
            return response.values(z, forward=forward, seed=seed)[2, ocean.heat_index]

        response.audit(lambda z: event_loss(z), point, checks, lambda z, seed: event_loss(z, True, seed))
        event_fit, event_history = descend(lambda z: event_loss(z), point, radius=1, on_iteration=progress("event"))
        checks.metrics.update(climate_event_controls=event_fit.tolist(), climate_event_history=event_history)
        checks.require("warm-event probability improves", event_history[-1] < event_history[0] - 1e-6, event_history)
        ocean.check_controls(response.full(event_fit), checks)
        checks.metrics.update(climate_event_controls=event_fit.tolist(), climate_event_history=event_history)
    baseline = float(mean_feature(baseline_values, ocean.amoc_index, response.kind))
    checks.require("resolved positive overturning", baseline > 1e-5, dict(baseline_sv=baseline))
    target = 0.9 * baseline
    checks.metrics["target_overturning_sv"] = target

    def constraint(z, forward=False, seed=0):
        return target - mean_feature(response.values(z, forward=forward, seed=seed), ocean.amoc_index, response.kind)

    constraint_forward = constraint
    constraint = (
        response.terminal_feature(ocean.amoc_index, scale=-1, offset=target)
        if response.kind == "trajectory"
        else response.function(lambda values, z: target - mean_feature(values, ocean.amoc_index, response.kind))
    )
    response.audit(constraint, point, checks, lambda z, seed: constraint_forward(z, True, seed))
    optimum, optimization = constrained_minimum(
        constraint, point, on_iteration=progress("constraint"), on_evaluation=progress("constraint")
    )
    checks.metrics.update(
        minimum_cost_controls=optimum.tolist(),
        minimum_scaled_cost=float(0.5 * optimum @ optimum),
        constrained_optimizer=optimization,
    )
    slack = optimization["slack_sv"]
    checks.require(
        "ten-percent overturning target",
        optimization["success"] and slack >= -1e-5,
        optimization,
    )
    checks.require("minimum-cost stationarity", optimization["stationarity_residual"] < 1e-3, optimization)
    checks.require("constraint complementarity", np.max(np.abs(optimization["complementarity"])) < 1e-3, optimization)
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
        heat_optimizer=heat_optimizer,
        target_overturning_sv=target,
        minimum_cost_controls=optimum.tolist(),
        minimum_scaled_cost=float(0.5 * optimum @ optimum),
        constrained_optimizer=optimization,
    )


def observation_design(response, checks):
    import jax.numpy as jnp

    ocean, point = response.ocean, jnp.zeros(response.indices.size)

    candidates = response.function(lambda values, z: values[..., : 4 * ocean.nz].reshape(-1))
    target = response.function(lambda values, z: mean_feature(values, ocean.amoc_index, response.kind), heldout=True)

    matrix = jacobian(candidates, point)
    values = candidates(point)
    direction = jnp.arange(1, point.size + 1, dtype=point.dtype)
    seed = jnp.sin(jnp.arange(values.size, dtype=point.dtype))
    tangent = candidates.jvp(point, direction)[1]
    transpose = candidates.vjp(point, seed)[1]
    checks.close("observation transpose identity", jnp.vdot(tangent, seed), jnp.vdot(direction, transpose))
    response.audit(
        target,
        point,
        checks,
        lambda z, seed: mean_feature(
            response.values(z, heldout=True, forward=True, seed=seed), ocean.amoc_index, response.kind
        ),
    )
    target_gradient = np.asarray(target.value_and_grad(point)[1])
    covariance = np.eye(point.size)
    initial_variance = float(target_gradient @ covariance @ target_gradient)
    selected, variances, total_costs = [], [initial_variance], []
    errors = np.asarray(response.config.get("observation_errors", np.eye(matrix.shape[0])))
    costs = np.asarray(response.config.get("observation_costs", np.ones(matrix.shape[0])))
    for _ in range(min(12, matrix.shape[0])):
        proposals = [
            (
                observation_covariance(matrix, selected + [index], errors=errors, costs=costs)
                if index not in selected
                else (None, None)
            )
            for index in range(matrix.shape[0])
        ]
        reductions = [
            (
                (variances[-1] - target_gradient @ candidate @ target_gradient) / costs[index]
                if candidate is not None
                else -np.inf
            )
            for index, (candidate, _) in enumerate(proposals)
        ]
        index = int(np.argmax(reductions))
        covariance, cost = proposals[index]
        selected.append(index)
        variances.append(float(target_gradient @ covariance @ target_gradient))
        total_costs.append(cost)
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
    all_covariance, _ = observation_covariance(matrix, list(range(matrix.shape[0])), errors=errors, costs=costs)
    all_variance = float(target_gradient @ all_covariance @ target_gradient)
    singular = np.linalg.svd(matrix, compute_uv=False)
    checks.metrics.update(
        selected_T_S_indices=selected,
        measurement_costs=total_costs,
        correlated_errors=not np.allclose(errors, np.diag(np.diag(errors))),
        singular_values=singular.tolist(),
        future_transport_variance=variances,
        target_gradient=target_gradient.tolist(),
        posterior_control_covariance=covariance.tolist(),
        nonlinear_ensemble_controls=samples.tolist(),
        nonlinear_ensemble_seed=27,
        observation_information_bound=dict(
            all_candidate_variance=all_variance,
            maximum_variance_reduction=1 - all_variance / initial_variance if initial_variance > 0 else None,
            scope="All candidate observations, without the selection budget, in the declared local Gaussian model.",
        ),
        uncertainty_scope="local Gaussian control prior and T/S noise; nonlinear forward check",
    )
    if checks.on_update is not None:
        checks.on_update()
    # Validate transfer with nonlinear forward predictions.
    predictions = np.array(
        [
            float(
                mean_feature(
                    response.values(
                        jnp.asarray(sample),
                        heldout=True,
                        forward=response.kind == "statistics",
                        seed=71 if response.kind == "statistics" else 0,
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


def refine_tendency_fit(loss, point, history, solver):
    """Resolve nonzero-residual curvature in the small, unconstrained closure fit."""
    if solver["converged"]:
        return point, history, solver
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize

    differentiate = jax.jit(jax.value_and_grad(loss))
    hessian = jax.jit(jax.hessian(loss))
    history = list(history)

    def evaluate(weights):
        value, gradient = differentiate(jnp.asarray(weights))
        return float(value), np.asarray(gradient)

    result = minimize(
        evaluate,
        np.asarray(point),
        jac=True,
        hess=lambda weights: np.asarray(hessian(jnp.asarray(weights))),
        method="trust-exact",
        callback=lambda weights: history.append(evaluate(weights)[0]),
        options=dict(gtol=solver["gradient_tolerance"], maxiter=solver["max_iterations"]),
    )
    value, gradient = evaluate(result.x)
    norm = float(np.linalg.norm(gradient))
    converged = bool(np.isfinite(value) and np.isfinite(norm) and norm <= solver["gradient_tolerance"])
    diagnostic = dict(
        solver,
        gauss_newton=solver,
        method="scaled Gauss-Newton with exact-curvature refinement",
        stopping="projected_gradient_tolerance" if converged else "curvature_refinement_unresolved",
        converged=converged,
        projected_gradient_norm=norm,
        gradient_norm=norm,
        control_norm=float(np.linalg.norm(result.x)),
        curvature_refinement=dict(method="trust-exact", iterations=int(result.nit), message=str(result.message)),
    )
    return jnp.asarray(result.x), history, diagnostic


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
    checks.metrics["refined_reference"] = reference["metadata"]
    horizontal_error = reference["metadata"].get("projection_horizontal_relative_error", float("inf"))
    independent = (
        reference["source"] == "refined_veros"
        and reference["refinement"] >= 2
        and np.isfinite(horizontal_error)
        and 0 <= horizontal_error <= 1e-10
    )
    checks.require(
        "independent refined reference",
        independent,
        dict(
            source=reference["source"], refinement=reference["refinement"], horizontal_relative_error=horizontal_error
        ),
    )
    if not independent:
        raise ValueError("Independent reference horizontal geometry is unresolved")
    checks.require(
        "conservative refined tracer projection",
        reference["metadata"]["projection_inventory_relative_error"] < 1e-11,
        reference["metadata"]["projection_inventory_relative_error"],
    )

    # Fit closure tendencies on projected independent reference states.
    def tendency_residual(weights, subset):
        residuals = []
        for carry, target in zip(reference[subset + "_states"], reference[subset + "_tendencies"]):
            diffusivity = closure_law(carry.variables, weights)
            tendency = reference["flux_divergence"](carry, diffusivity)
            residuals.append((tendency - target) / reference["tendency_scale"])
        return jnp.stack(residuals)

    def tendency_loss(weights, subset):
        return jnp.mean(tendency_residual(weights, subset) ** 2)

    from .common import gradient_checks

    initial, _ = gradient_checks(lambda z: tendency_loss(z, "train"), jnp.zeros(2), checks)
    size = tendency_residual(jnp.zeros(2), "train").size
    residual = JaxResidual(lambda weights: tendency_residual(weights, "train").reshape(-1) * jnp.sqrt(2 / size))
    fit, history, solver = multistart_least_squares(residual, jnp.zeros(2), prior_precision=0, radius=None)
    fit, history, solver = refine_tendency_fit(lambda weights: tendency_loss(weights, "train"), fit, history, solver)
    checks.require("closure solver convergence", solver["converged"], solver)
    if not solver["converged"]:
        raise RuntimeError("Closure tendency optimization has not converged")
    lower_bound = reference["metadata"]["positive_flux_loss_lower_bound"]
    checks.metrics["closure_representability"] = dict(
        unrestricted_nonnegative_flux_loss_lower_bound=lower_bound,
        initial_training_loss=initial,
        lower_bound_fraction=lower_bound / max(initial, 1e-30),
        diagnosis=(
            "nonnegative added diffusion cannot reach the original training threshold"
            if lower_bound >= 0.1 * initial
            else "relaxed flux bound does not exclude the training threshold"
        ),
        scope="independent nonnegative diffusivity at every face, state, and column; a superset of the fitted law",
    )
    checks.require("reference-tendency fit reduced by 90%", history[-1] < 0.1 * initial, history)
    before, after = float(tendency_loss(jnp.zeros(2), "heldout")), float(tendency_loss(fit, "heldout"))
    checks.require("withheld reference tendencies improve", after < 0.5 * before, [before, after])
    checks.close("closure-only heat/salt conservation", reference["budget_residual"](fit), 0, atol=1e-8)
    checks.metrics["closure_optimizer"] = solver
    checks.metrics["tendency_weights"] = fit.tolist()
    if response.kind in ("equilibrium", "seasonal") or (
        response.kind == "statistics"
        and (
            response.config["setup"] == "adjusted_basin"
            or response.config.get("statistical_method") == "invariant_periodic"
        )
    ):
        first = len(checks.items)
        response.check_adjustment(fit, response.adjusted(fit), checks, label="tendency_weights")
        if any(not check["passed"] for check in checks.items[first:]):
            raise RuntimeError("The fitted closure has not resolved a stable adjusted branch")
    # Test transfer against an independent refined-model rollout.
    target_options = dict(steps=response.steps, heldout=True)
    if response.kind == "statistics":
        target_options["statistics"] = dict(
            thresholds=response.thresholds,
            sample_stride=max(1, response.window_steps // 12),
            window_steps=response.window_steps,
            burnin_steps=response.burnin_steps,
            n_windows=2 * response.windows,
        )
    reference_rollout = reference["response_target"](response.kind, **target_options)
    baseline = response.values(jnp.zeros(2), heldout=True)
    learned = response.values(fit, heldout=True)
    scale = jnp.maximum(jnp.abs(reference_rollout - baseline), 1e-3)
    before = float(jnp.mean(((baseline - reference_rollout) / scale) ** 2))
    after = float(jnp.mean(((learned - reference_rollout) / scale) ** 2))
    checks.metrics.update(
        tendency_training_history=history,
        tendency_weights=fit.tolist(),
        reference_rollout_error=after,
        closure_rollout=dict(reference=reference_rollout, baseline=baseline, learned=learned, scale=scale),
    )
    checks.require("tendency-trained closure transfers to circulation", after < 0.5 * before, [before, after])

    def rollout_loss(z, forward=False, seed=0):
        return jnp.mean(
            ((response.values(z, heldout=True, forward=forward, seed=seed) - reference_rollout) / scale) ** 2
        )

    rollout_forward = rollout_loss
    rollout_loss = response.function(
        lambda values, z: jnp.mean(((values - reference_rollout) / scale) ** 2), heldout=True
    )
    response.audit(
        rollout_loss, fit, checks, lambda z, seed: rollout_forward(z, True, seed), coarser_scales=(0.1, 0.03)
    )
    ocean.check_controls(response.full(fit), checks)


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
    geometry_grid_design(response, fit, checks)
    geometry_depth_design(response, fit, checks)


def geometry_grid_design(response, fit, checks):
    import jax.numpy as jnp

    ocean = response.ocean

    def accuracy(grid_coordinate, forward=False, seed=0):
        import jax

        z = fit.at[2].set(grid_coordinate[0])
        state, params = ocean.controls(response.full(z))
        if not isinstance(state.failure_code, jax.core.Tracer) and int(state.failure_code) in (4, 5, 6):
            checks.metrics.setdefault("grid_accuracy_rejected_controls", []).append(
                dict(controls=grid_coordinate.tolist(), failure_code=int(state.failure_code))
            )
            if checks.on_update is not None:
                checks.on_update()
            return jnp.asarray(jnp.inf, dtype=grid_coordinate.dtype)
        return ocean.prepared.grid_reference_error(
            state,
            parameters=params,
            refinement=2,
            response=response.kind,
            steps=response.steps,
            forward=forward,
            seed=seed,
        )["error"]

    from .finite_response import ControlObjective

    grid_point = fit[2:3]
    accuracy_objective = ControlObjective(accuracy)
    checks.metrics["grid_accuracy_gradient_method"] = "streamed control Jacobian and explicit transpose"
    response.audit(accuracy_objective, grid_point, checks, lambda z, seed: accuracy(z, True, seed))
    grid_fit, history = descend(accuracy_objective, grid_point, iterations=8, radius=1)
    checks.metrics.update(grid_accuracy_history=history, grid_accuracy_controls=grid_fit.tolist())
    checks.require("vertical-grid optimization improves accuracy", history[-1] < history[0], history)

    from .geometry_reference import release

    release(ocean)


def geometry_depth_design(response, fit, checks):
    import jax.numpy as jnp

    ocean = response.ocean

    def exchange(depth_coordinates, forward=False, seed=0):
        z = fit.at[:2].set(depth_coordinates)
        return -mean_feature(response.values(z, forward=forward, seed=seed), ocean.acc_index, response.kind)

    exchange_objective = response.function(
        lambda values, z: -mean_feature(values, ocean.acc_index, response.kind)
    ).map_controls(lambda depth: fit.at[:2].set(depth))
    response.audit(exchange_objective, fit[:2], checks, lambda z, seed: exchange(z, True, seed))
    depth_fit, exchange_history = descend(exchange_objective, fit[:2], iterations=8, radius=1)
    checks.require(
        "bounded bathymetric exchange improves", exchange_history[-1] < exchange_history[0], exchange_history
    )
    checks.require("admissible depth-design norm", float(jnp.linalg.norm(depth_fit)) <= 1 + 1e-9, depth_fit.tolist())
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
        experiment_scope={
            "global_4deg": "global 4-degree research acceptance",
            "acc": "small ACC numerical fixture",
            "adjusted_basin": "forced dissipative three-dimensional basin study",
        }[config["setup"]],
        extension_contracts=case.requirements,
    )
    missing = missing_capabilities(PreparedIntegrator, case)
    if missing:
        for method, contract in missing.items():
            checks.require("required capability: " + method, False, contract)
        checks.metrics["missing_capabilities"] = missing
        return
    if case.task == "geometry":
        from .geometry_reference import require_response

        require_response(case.response, config)
    ocean = ResearchOcean(config, case)
    checks.metrics["prepared_time_seconds"] = float(ocean.initial.variables.time)
    checks.metrics["model_settings"] = dict(ocean.sim.state.settings.items())
    checks.metrics["grid_shape"] = [ocean.sim.state.settings.nx, ocean.sim.state.settings.ny, ocean.nz]
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
    if response.kind in ("equilibrium", "seasonal") or (
        response.kind == "statistics"
        and (config["setup"] == "adjusted_basin" or config.get("statistical_method") == "invariant_periodic")
    ):
        selected = {
            "estimation": ("fitted",),
            "calibration": ("fitted",),
            "perturbation": ("heat_controls", "minimum_cost_controls"),
        }.get(case.task, ())
        for name in selected:
            controls = jnp.asarray(checks.metrics[name])
            response.check_adjustment(controls, response.adjusted(controls), checks, label=name)
    if hasattr(ocean.prepared, "_adjusted_responses"):
        checks.metrics["adjusted_execution"] = [
            dict(
                period_steps=adapter.period_steps,
                state_dimension=adapter.state.size,
                checkpoint=adapter.checkpoint_statistics,
                compiled_equilibrium_transposes=adapter.equilibrium_transpose_calls,
                checkpoint_scope="shared physical response map across solver budgets",
                solver_max_iterations=engine.max_iterations,
                solver_warmup=engine.warmup,
                linear_solve_count=engine.linear_solve_count,
                last_linear_solve=engine.last_linear_solve,
                stability_solver=engine.last_stability,
                cached_states=len(engine.cache),
            )
            for adapter, engine in ocean.prepared._adjusted_responses.values()
        ]
    if response.finite is not None and response.finite.statistics:
        statistics = response.finite.statistics
        checks.metrics["checkpoint_response"] = dict(
            evaluations=len(statistics),
            forward_steps=sum(s["forward_steps"] for s in statistics),
            recomputed_steps=sum(s["recomputed_steps"] for s in statistics),
            reverse_steps=sum(s["reverse_steps"] for s in statistics),
            last_vjp_seconds=response.finite.last_seconds,
            product_cache_releases=response.finite.cache_releases,
            peak_checkpoint_ram_bytes=max(s["peak_ram_bytes"] for s in statistics),
            peak_checkpoint_disk_bytes=max(s["peak_disk_bytes"] for s in statistics),
            boundary_carries_per_window=2 if response.kind == "local" else 4,
            control_window_count=4 if response.kind == "local" else 1,
            last_window=statistics[-1],
        )
