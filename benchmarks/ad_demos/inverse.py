"""Finite inverse solvers and local uncertainty outside numerical timesteps."""

import numpy as np


class JaxResidual:
    """Products for a pure JAX residual outside numerical integration."""

    def __init__(self, function):
        import jax

        self.function = jax.jit(function)

    def __call__(self, point):
        return self.function(point)

    def jvp(self, point, direction):
        import jax

        return jax.jvp(self.function, (point,), (direction,))

    def vjp(self, point, seed):
        import jax

        value, pullback = jax.vjp(self.function, point)
        return value, pullback(seed)[0]

    def jacobian(self, point):
        import jax.numpy as jnp

        return np.column_stack(
            [np.asarray(self.jvp(point, direction)[1]).reshape(-1) for direction in jnp.eye(point.size)]
        )


def project(point, radius):
    return point if radius is None else point * min(1.0, radius / max(np.linalg.norm(point), 1e-30))


def gradient_start(
    function, point, value, gradient, *, radius=1, steps=(1.0, 0.3, 0.1), admissible=None, on_trial=None
):
    """Choose an improving bounded start along an independently computed gradient."""
    import jax.numpy as jnp

    point, gradient = np.asarray(point, dtype=float), np.asarray(gradient, dtype=float)
    norm = np.linalg.norm(gradient)
    best, best_value, trials = point, float(value), []
    if not np.isfinite(norm) or norm == 0:
        return jnp.asarray(best), trials
    for step in steps:
        candidate = project(point - step * gradient / norm, radius)
        slope = gradient @ (candidate - point)
        if slope >= 0:
            continue
        valid = admissible is None or admissible(jnp.asarray(candidate))
        trial = dict(step=step, controls=candidate.tolist(), admissible=bool(valid))
        if valid:
            trial_value = float(function(jnp.asarray(candidate)))
            trial["value"] = trial_value
            if np.isfinite(trial_value) and trial_value < best_value and trial_value <= value + 1e-4 * slope:
                best, best_value = candidate, trial_value
        trials.append(trial)
        if on_trial is not None:
            on_trial(trial)
    return jnp.asarray(best), trials


def least_squares(
    residual,
    point,
    *,
    prior_precision=1e-3,
    radius=1,
    max_iterations=64,
    tolerance=1e-7,
    zero_loss_tolerance=None,
    matrix_free=False,
    on_iteration=None,
    admissible=None,
):
    """Scaled Gauss-Newton with a norm-constrained subproblem and Armijo search.

    Residuals are whitened by the caller. The isotropic Gaussian prior is in
    physical control coordinates. Matrix-free products use the response JVP/VJP.
    An explicit zero-loss tolerance can certify unregularized exact fits using
    the nonnegative objective's global lower bound.
    """
    import jax.numpy as jnp
    from scipy.optimize import minimize
    from scipy.sparse.linalg import LinearOperator, lsmr

    point = project(np.asarray(point, dtype=float), radius)
    history, stopping = [], "iteration_limit"
    iterations = 0
    for iteration in range(max_iterations + 1):
        if iteration == 0 and admissible is not None and not admissible(jnp.asarray(point)):
            history.append(float("inf"))
            gradient, projected_norm = np.full_like(point, np.nan), float("inf")
            stopping = "inadmissible_initial_point"
            break
        residual_values = np.asarray(residual(jnp.asarray(point)))
        r = residual_values.reshape(-1)
        loss = 0.5 * (r @ r + prior_precision * (point @ point))
        history.append(float(loss))
        exact_candidate = (
            prior_precision == 0
            and zero_loss_tolerance is not None
            and np.isfinite(loss)
            and loss <= zero_loss_tolerance
            and hasattr(residual, "vjp")
        )
        if exact_candidate:
            # Check gradient validity with one transpose at the nonnegative loss bound.
            gradient = np.asarray(residual.vjp(jnp.asarray(point), jnp.asarray(residual_values))[1])
            projected_norm = np.linalg.norm(point - project(point - gradient, radius))
            stopping = "zero_loss_tolerance" if np.isfinite(gradient).all() else "nonfinite_response"
            if on_iteration is not None:
                on_iteration(
                    dict(
                        iteration=iteration,
                        loss=float(loss),
                        controls=point.tolist(),
                        stage="stopping check",
                        projected_gradient_norm=float(projected_norm),
                    )
                )
            break
        if on_iteration is not None:
            on_iteration(dict(iteration=iteration, loss=float(loss), controls=point.tolist(), stage="Jacobian"))
        if matrix_free:
            scaling = np.sqrt(
                np.maximum(
                    np.array(
                        [
                            np.sum(np.asarray(residual.jvp(jnp.asarray(point), basis)[1]) ** 2)
                            for basis in jnp.eye(point.size)
                        ]
                    )
                    + prior_precision,
                    1e-12,
                )
            )
            operator = LinearOperator(
                (r.size + point.size, point.size),
                matvec=lambda v: np.concatenate(
                    (
                        np.asarray(residual.jvp(jnp.asarray(point), jnp.asarray(v / scaling))[1]).reshape(-1),
                        np.sqrt(prior_precision) * v / scaling,
                    )
                ),
                rmatvec=lambda w: (
                    np.asarray(
                        residual.vjp(jnp.asarray(point), jnp.asarray(w[: r.size].reshape(residual_values.shape)))[1]
                    )
                    + np.sqrt(prior_precision) * w[r.size :]
                )
                / scaling,
            )
            augmented = np.concatenate((r, np.sqrt(prior_precision) * point))
            gradient = operator.rmatvec(augmented) * scaling
            # LSMR never forms a dense observation Jacobian.
            step = lsmr(operator, -augmented, atol=1e-10, btol=1e-10, maxiter=4 * point.size)[0] / scaling
            hessian = None
        else:
            matrix = residual.jacobian(jnp.asarray(point))
            gradient = matrix.T @ r + prior_precision * point
            hessian = matrix.T @ matrix + prior_precision * np.eye(point.size)
            scaling = np.sqrt(np.maximum(np.diag(hessian), 1e-12))
            augmented = np.vstack((matrix, np.sqrt(prior_precision) * np.eye(point.size))) / scaling
            rhs = -np.concatenate((r, np.sqrt(prior_precision) * point))
            step = np.linalg.lstsq(augmented, rhs, rcond=1e-12)[0] / scaling
        projected_norm = np.linalg.norm(point - project(point - gradient, radius))
        if on_iteration is not None:
            on_iteration(
                dict(
                    iteration=iteration,
                    loss=float(loss),
                    controls=point.tolist(),
                    stage="stopping check",
                    projected_gradient_norm=float(projected_norm),
                )
            )
        if not np.isfinite(loss) or not np.isfinite(gradient).all():
            stopping = "nonfinite_response"
            break
        if prior_precision == 0 and zero_loss_tolerance is not None and loss <= zero_loss_tolerance:
            stopping = "zero_loss_tolerance"
            break
        if projected_norm <= tolerance:
            stopping = "projected_gradient_tolerance"
            break
        if iteration == max_iterations:
            break
        if radius is not None and np.linalg.norm(point + step) > radius:
            # Solve the convex local model inside the physical control domain.
            quadratic = (
                (lambda v: hessian @ v)
                if hessian is not None
                else (lambda v: operator.rmatvec(operator.matvec(v * scaling)) * scaling)
            )
            subproblem = minimize(
                lambda v: gradient @ v + 0.5 * v @ quadratic(v),
                project(point + step, radius) - point,
                jac=lambda v: gradient + quadratic(v),
                method="SLSQP",
                constraints=dict(
                    type="ineq", fun=lambda v: radius**2 - (point + v) @ (point + v), jac=lambda v: -2 * (point + v)
                ),
                options=dict(maxiter=100, ftol=1e-12),
            )
            step = subproblem.x
        accepted = False
        for power in range(24):
            candidate = project(point + 0.5**power * step, radius)
            if admissible is not None and not admissible(jnp.asarray(candidate)):
                continue
            trial = np.asarray(residual(jnp.asarray(candidate))).reshape(-1)
            trial_loss = 0.5 * (trial @ trial + prior_precision * (candidate @ candidate))
            if np.isfinite(trial_loss) and trial_loss <= loss + 1e-4 * gradient @ (candidate - point):
                point, accepted = candidate, True
                iterations += 1
                break
        if not accepted:
            stopping = "line_search_exhausted"
            break
    diagnostics = dict(
        method="scaled Gauss-Newton",
        matrix_free=matrix_free,
        stopping=stopping,
        converged=stopping in ("projected_gradient_tolerance", "zero_loss_tolerance"),
        accepted_iterations=iterations,
        max_iterations=max_iterations,
        projected_gradient_norm=float(projected_norm),
        gradient_norm=float(np.linalg.norm(gradient)),
        gradient_tolerance=tolerance,
        zero_loss_tolerance=zero_loss_tolerance,
        control_norm=float(np.linalg.norm(point)),
        radius=radius,
        prior_precision=prior_precision,
    )
    return jnp.asarray(point), history, diagnostics


def multistart_least_squares(residual, point, *, zero_loss_tolerance=None, **kwargs):
    import jax.numpy as jnp

    rng = np.random.default_rng(42)
    starts = [np.asarray(point)] + [
        project(rng.normal(size=point.size) * scale, kwargs.get("radius", 1)) for scale in (0.1, 0.3)
    ]
    callback = kwargs.pop("on_iteration", None)
    results = []
    for index, start in enumerate(starts):
        notify = None if callback is None else lambda progress, index=index: callback(dict(progress, start=index))
        results.append(
            least_squares(
                residual, jnp.asarray(start), on_iteration=notify, zero_loss_tolerance=zero_loss_tolerance, **kwargs
            )
        )
        if notify is not None:
            fit, history, diagnostic = results[-1]
            notify(dict(stage="start complete", controls=fit.tolist(), loss=history[-1], optimizer=diagnostic))
        if (
            zero_loss_tolerance is not None
            and results[-1][2]["converged"]
            and results[-1][1][-1] <= zero_loss_tolerance
        ):
            break
    selected = min(range(len(results)), key=lambda i: results[i][1][-1])
    tied_converged = [
        i
        for i, (_, history, diagnostic) in enumerate(results)
        if diagnostic["converged"] and history[-1] <= results[selected][1][-1] + 1e-12
    ]
    if tied_converged:
        selected = min(tied_converged, key=lambda i: results[i][1][-1])
    fit, history, diagnostic = results[selected]
    diagnostic.update(
        selected_start=selected,
        starts=[dict(d, initial=s.tolist(), final=f.tolist(), loss=h[-1]) for s, (f, h, d) in zip(starts, results)],
        zero_loss_tolerance=zero_loss_tolerance,
    )
    return fit, history, diagnostic


def constrained_minimum(
    constraint,
    point,
    *,
    radius=1,
    tolerance=1e-8,
    max_iterations=100,
    on_iteration=None,
    on_evaluation=None,
    value_and_grad_fn=None,
):
    """Restore feasibility before minimizing control cost, retaining failed searches."""
    import jax.numpy as jnp
    from scipy.optimize import minimize, nnls
    from .finite_response import value_and_grad

    cached_point, cached_pair = None, None
    value_point, cached_value = None, None
    evaluations = dict(values=0, gradients=0)
    differentiate = value_and_grad(constraint) if value_and_grad_fn is None else value_and_grad_fn

    def scalar(z):
        nonlocal value_point, cached_value
        if cached_point is not None and np.array_equal(z, cached_point):
            return cached_pair[0]
        if value_point is not None and np.array_equal(z, value_point):
            return cached_value
        value_point, cached_value = np.array(z, copy=True), float(constraint(jnp.asarray(z)))
        evaluations["values"] += 1
        return cached_value

    def pair(z):
        nonlocal cached_point, cached_pair, value_point, cached_value
        if cached_point is not None and np.array_equal(z, cached_point):
            return cached_pair
        value, gradient = differentiate(jnp.asarray(z))
        cached_point, cached_pair = np.array(z, copy=True), (float(value), np.asarray(gradient))
        value_point, cached_value = cached_point.copy(), cached_pair[0]
        evaluations["gradients"] += 1
        if on_evaluation is not None:
            on_evaluation(
                dict(
                    stage="evaluation",
                    controls=cached_point.tolist(),
                    value=cached_pair[0],
                    gradient=cached_pair[1].tolist(),
                )
            )
        return cached_pair

    rng = np.random.default_rng(43)
    searches = []
    objective_scale = max(np.linalg.norm(pair(np.asarray(point))[1]), 1e-8)
    restoration_tolerance = max(tolerance, np.sqrt(2 * tolerance) * objective_scale)
    ball = dict(type="ineq", fun=lambda z: radius**2 - z @ z, jac=lambda z: -2 * z)

    def violation(z):
        deficit = min(scalar(z), 0.0)
        return 0.5 * (deficit / objective_scale) ** 2

    def violation_gradient(z):
        slack, gradient = pair(z)
        deficit = min(slack, 0.0)
        return deficit * gradient / objective_scale**2

    def progress(stage, start):
        iteration = 0

        def record(z):
            nonlocal iteration
            iteration += 1
            if on_iteration is not None:
                same_point = cached_point is not None and np.array_equal(z, cached_point)
                known_value = value_point is not None and np.array_equal(z, value_point)
                on_iteration(
                    dict(
                        stage=stage,
                        start=start,
                        iteration=iteration,
                        controls=np.asarray(z).tolist(),
                        control_norm=float(np.linalg.norm(z)),
                        slack_sv=cached_pair[0] if same_point else (cached_value if known_value else None),
                    )
                )

        return record

    for index, start in enumerate(
        [np.asarray(point)] + [project(rng.normal(size=point.size), radius) for _ in range(2)]
    ):
        feasible = minimize(
            violation,
            start,
            jac=violation_gradient,
            method="SLSQP",
            constraints=ball,
            options=dict(maxiter=max_iterations, ftol=tolerance),
            callback=progress("feasibility", index) if on_iteration is not None else None,
        )
        searches.append(
            dict(
                success=bool(feasible.success),
                message=str(feasible.message),
                iterations=int(feasible.nit),
                controls=project(feasible.x, radius).tolist(),
                slack=scalar(project(feasible.x, radius)),
            )
        )
        if searches[-1]["slack"] >= -restoration_tolerance:
            break
    best = max(searches, key=lambda result: result["slack"])
    optimization = (
        None
        if best["slack"] < -restoration_tolerance
        else minimize(
            lambda z: 0.5 * z @ z,
            np.asarray(best["controls"]),
            jac=lambda z: z,
            method="SLSQP",
            constraints=(
                dict(
                    type="ineq", fun=lambda z: scalar(z) / objective_scale, jac=lambda z: pair(z)[1] / objective_scale
                ),
                ball,
            ),
            options=dict(maxiter=max_iterations, ftol=min(tolerance, tolerance / objective_scale)),
            callback=progress("cost", None) if on_iteration is not None else None,
        )
    )
    # Retain an admissible feasible-search result if the cost solve fails.
    chosen = (
        optimization.x
        if optimization is not None and optimization.success and np.linalg.norm(optimization.x) <= radius + 1e-9
        else np.asarray(best["controls"])
    )
    slack, gradient = pair(chosen)
    columns = np.stack((gradient, -2 * chosen), axis=1)
    multipliers, stationarity = nnls(columns, chosen)
    complementarity = multipliers * [slack, radius**2 - chosen @ chosen]
    diagnostic = dict(
        method="SLSQP with feasibility restoration",
        success=optimization is not None and bool(optimization.success) and slack >= -tolerance,
        message=(
            "feasibility restoration did not reach the target" if optimization is None else str(optimization.message)
        ),
        iterations=0 if optimization is None else int(optimization.nit),
        feasibility_objective_scale=float(objective_scale),
        feasibility_ftol=tolerance,
        restoration_tolerance=float(restoration_tolerance),
        max_iterations=max_iterations,
        ftol=tolerance,
        feasibility_searches=searches,
        slack_sv=slack,
        control_norm=float(np.linalg.norm(chosen)),
        constraint_gradient=gradient.tolist(),
        stationarity_residual=float(stationarity),
        multipliers=multipliers.tolist(),
        complementarity=complementarity.tolist(),
        feasible=slack >= -tolerance and np.linalg.norm(chosen) <= radius + tolerance,
        value_evaluations=evaluations["values"],
        gradient_evaluations=evaluations["gradients"],
        diagnosis="feasible" if slack >= -tolerance else "target not reached by bounded multistart searches",
    )
    return jnp.asarray(chosen), diagnostic


def observation_covariance(matrix, selected, *, errors=None, costs=None, prior=None):
    """Condition a Gaussian prior using correlated measurement errors."""
    dimension = matrix.shape[1]
    prior = np.eye(dimension) if prior is None else np.asarray(prior)
    errors = np.eye(matrix.shape[0]) if errors is None else np.asarray(errors)
    indices = np.asarray(selected, dtype=int)
    if not indices.size:
        return prior.copy(), 0.0
    rows = matrix[indices]
    covariance = np.linalg.inv(np.linalg.inv(prior) + rows.T @ np.linalg.solve(errors[np.ix_(indices, indices)], rows))
    cost = float(indices.size if costs is None else np.sum(np.asarray(costs)[indices]))
    return covariance, cost
