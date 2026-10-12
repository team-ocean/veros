"""Column closure and equilibrium experiments using Veros' TDMA solver."""

from .common import descend, gradient_checks


class Column:
    def __init__(self, *, dt=600.0, restoring=False):
        import jax.numpy as jnp

        self.dz, self.dt, self.restoring = 10.0, dt, restoring
        self.water = jnp.ones((1, 1, 15), dtype=bool)
        self.edge = jnp.zeros_like(self.water).at[..., 0].set(True)
        self.initial = jnp.linspace(10, 15, 15)

    def coefficients(self, temperature, z):
        import jax.numpy as jnp

        gradient = jnp.diff(temperature) / self.dz
        diffusivity = 1e-3 * jnp.exp(z[0] + z[1] * jnp.tanh(gradient / 0.02))
        faces = diffusivity / self.dz**2
        a = -jnp.concatenate((jnp.zeros(1), faces))
        c = -jnp.concatenate((faces, jnp.zeros(1)))
        b = -a - c
        if self.restoring:
            b = b.at[-1].add(1 / (30 * 86400))
        return a, b, c, diffusivity

    def source(self):
        import jax.numpy as jnp

        source = jnp.zeros(15)
        if self.restoring:
            source = source.at[0].set(1e-6 / self.dz).at[-1].set(15 / (30 * 86400))
        return source

    def solve(self, a, b, c, rhs):
        from veros.core.operators import solve_tridiagonal

        return solve_tridiagonal(
            a[None, None, :], b[None, None, :], c[None, None, :], rhs[None, None, :], self.water, self.edge
        )[0, 0]

    def step(self, temperature, z):
        a, b, c, _ = self.coefficients(temperature, z)
        return self.solve(self.dt * a, 1 + self.dt * b, self.dt * c, temperature + self.dt * self.source())

    def rollout(self, initial, z, steps):
        import jax

        def step(temperature, _):
            return self.step(temperature, z), None

        return jax.lax.scan(jax.checkpoint(step), initial, None, length=steps)[0]


def closure_learning(config, checks):
    import jax.numpy as jnp

    column = Column()
    steps = config["steps"] or (6 if config["mode"] == "smoke" else 96)
    truth = jnp.array([jnp.log(2.0), 0.4])
    target = column.rollout(column.initial, truth, steps)
    baseline = column.rollout(column.initial, jnp.zeros(2), steps)
    scale = jnp.maximum(jnp.abs(target - baseline), 1e-5)

    def loss(z):
        return jnp.mean(((column.rollout(column.initial, z, steps) - target) / scale) ** 2)

    initial_loss, _ = gradient_checks(loss, jnp.zeros(2), checks)
    fit, history = descend(loss, jnp.zeros(2), iterations=25)
    unseen = column.initial + 0.7 * jnp.sin(jnp.linspace(0, jnp.pi, 15))
    holdout_target = column.rollout(unseen, truth, 4 * steps)
    holdout_initial = column.rollout(unseen, jnp.zeros(2), 4 * steps)
    holdout_fit = column.rollout(unseen, fit, 4 * steps)
    before = float(jnp.linalg.norm(holdout_initial - holdout_target))
    after = float(jnp.linalg.norm(holdout_fit - holdout_target))
    _, _, _, diffusivity = column.coefficients(holdout_fit, fit)
    checks.require(
        "positive finite diffusivity",
        bool(jnp.all(jnp.isfinite(diffusivity) & (diffusivity > 0))),
        diffusivity.tolist(),
    )
    checks.close("closed-column heat conservation", jnp.sum(holdout_fit - unseen) * column.dz, 0, atol=1e-8)
    checks.require("training misfit reduced by 90%", history[-1] < initial_loss * 0.1, history)
    checks.require("unseen profile and longer rollout improve", after < before * 0.5, [before, after])
    checks.require("closure parameters recovered", float(jnp.linalg.norm(fit - truth)) < 0.15, fit.tolist())
    checks.metrics.update(
        scope="column prototype",
        steps=steps,
        dt=column.dt,
        truth=truth.tolist(),
        fitted=fit.tolist(),
        loss_history=history,
    )
    return checks


def equilibrium_sensitivity(config, checks):
    import jax
    import jax.numpy as jnp

    column = Column(dt=86400.0, restoring=True)
    point = jnp.array([0.4])

    def coefficients(z):
        return column.coefficients(column.initial, jnp.concatenate((z, jnp.zeros(1))))[:3]

    def equilibrium(z):
        return column.solve(*coefficients(z), column.source())

    def residual(temperature, z):
        a, b, c = coefficients(z)
        previous = jnp.concatenate((jnp.zeros(1), temperature[:-1]))
        following = jnp.concatenate((temperature[1:], jnp.zeros(1)))
        return a * previous + b * temperature + c * following - column.source()

    def objective(z):
        return jnp.mean(equilibrium(z)[:5])

    value, gradient = gradient_checks(objective, point, checks)
    temperature = equilibrium(point)
    matrix = jax.jacfwd(residual, argnums=0)(temperature, point)
    sensitivity = jax.jacfwd(residual, argnums=1)(temperature, point)
    weights = jnp.concatenate((jnp.full(5, 0.2), jnp.zeros(10)))
    adjoint = jnp.linalg.solve(matrix.T, weights)
    implicit = -adjoint @ sensitivity
    checks.close("equilibrium residual", residual(temperature, point), 0, atol=1e-12)
    checks.close("implicit adjoint versus solver AD", implicit, gradient, rtol=1e-8)
    lengths = (16, 32, 64) if config["mode"] == "smoke" else (512, 2048, 8192)
    if config["steps"]:
        lengths = tuple(sorted({max(1, config["steps"] // 4), max(1, config["steps"] // 2), config["steps"]}))
    errors = []
    for length in lengths:

        def unrolled(z):
            final = column.rollout(column.initial, jnp.concatenate((z, jnp.zeros(1))), length)
            return jnp.mean(final[:5])

        finite_value, finite_gradient = jax.jit(jax.value_and_grad(unrolled))(point)
        errors.append(
            dict(
                steps=length,
                state_error=float(abs(finite_value - value)),
                gradient_error=float(jnp.linalg.norm(finite_gradient - gradient)),
            )
        )
    checks.require("adjusted state converges", errors[-1]["state_error"] < 1e-4, errors)
    checks.require("unrolled sensitivity converges", errors[-1]["gradient_error"] < 1e-4, errors)
    checks.metrics.update(
        scope="linear column equilibrium prototype",
        equilibrium=temperature.tolist(),
        gradient=gradient.tolist(),
        horizon_errors=errors,
    )
    return checks


def run(name, config, checks):
    return globals()[name](config, checks)
