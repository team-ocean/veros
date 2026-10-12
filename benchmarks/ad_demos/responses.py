"""Response definitions, adjustment checks, and independent statistical validation."""

import numpy as np

from .common import gradient_checks
from .research_ocean import YEAR


class Response:
    def __init__(self, ocean, indices, config):
        import jax.numpy as jnp

        self.ocean, self.indices, self.config = ocean, jnp.asarray(indices), config
        self.kind = ocean.case.response
        if self.kind == "local" and config["steps"] not in (None, 1):
            raise ValueError("Local response demos require exactly one differentiated step")
        self.period_steps = round(YEAR / ocean.dt)
        days = 30 if ocean.case.task == "sensitivity" else 360
        if ocean.case.task == "perturbation":
            days = 3600
        default = 1 if self.kind == "local" else (6 if config["mode"] == "smoke" else round(days * 86400 / ocean.dt))
        self.steps = config["steps"] or default
        self.window_steps = max(self.period_steps, self.steps)
        self.windows = 4 if config["mode"] == "smoke" else 16
        self.burnin_steps = 20 * self.period_steps
        self.thresholds = ocean.observe(ocean.initial) + 1

    def full(self, z):
        import jax.numpy as jnp
        from .research_ocean import CONTROL_NAMES

        return jnp.zeros(len(CONTROL_NAMES), dtype=z.dtype).at[self.indices].set(z)

    def adjusted(self, z, *, phase_days=0, initial_shift=0, doubled=False, realized=False):
        full = self.full(z).at[0].add(initial_shift)
        carry, parameters = self.ocean.controls(full, phase_days=phase_days, realized=realized)
        if self.kind == "equilibrium":
            return self.ocean.prepared.equilibrium_response(
                carry,
                parameters=parameters,
                observe=self.ocean.observe,
                tolerance=1e-7,
                max_steps=max(self.steps, 20000 * round(86400 / self.ocean.dt)) * (2 if doubled else 1),
            )
        if self.kind == "seasonal":
            return self.ocean.prepared.periodic_response(
                carry,
                parameters=parameters,
                observe=self.ocean.observe,
                period_steps=self.period_steps,
                phases=tuple(month / 12 for month in range(12)),
                tolerance=1e-7,
                max_cycles=64 if doubled else 32,
            )
        return self.ocean.prepared.statistical_response(
            carry,
            parameters=parameters,
            observe=self.ocean.observe,
            thresholds=self.thresholds,
            window_steps=self.window_steps,
            burnin_steps=self.burnin_steps,
            n_windows=self.windows * (2 if doubled else 1),
            sample_stride=max(1, self.window_steps // 12),
            seed=int(phase_days),
        )

    def values(self, z, *, heldout=False, seed=0, forward=False, realized=False):
        import jax.numpy as jnp

        full = self.full(z)
        if self.kind == "local":
            # Independent one-step responses sample several states and forcing phases.
            return jnp.stack(
                [
                    self.ocean.observe(
                        self.ocean.advance(
                            full.at[0].add(0.2 if heldout else 0), 1, phase_days=phase, realized=realized
                        )
                    )
                    for phase in ((45, 135, 225, 315) if heldout else (0, 90, 180, 270))
                ]
            )
        if self.kind == "trajectory":
            steps = self.steps * (2 if heldout else 1)
            carry, parameters = self.ocean.controls(full, phase_days=90 if heldout else 0, realized=realized)
            previous, values = 0, []
            for boundary in sorted({max(1, steps // 3), max(1, 2 * steps // 3), steps}):
                carry, _ = self.ocean.prepared.advance(
                    carry, n_steps=boundary - previous, parameters=parameters, final_only=True
                )
                values.append(jnp.where(carry.failure_code == 0, self.ocean.observe(carry), jnp.nan))
                previous = boundary
            return jnp.stack(values)
        if self.kind == "statistics" and forward:
            return self.forward_statistics(z, seed=seed, doubled=heldout, realized=realized)
        result = self.adjusted(
            z, phase_days=seed, initial_shift=0.2 if heldout else 0, doubled=heldout, realized=realized
        )
        return result["observations"]

    def forward_statistics(self, z, *, seed=0, doubled=False, realized=False):
        """Forward ensemble independent of the statistical-sensitivity provider."""
        import jax.numpy as jnp

        rng = np.random.default_rng(seed)
        full = self.full(z).at[0].add(float(rng.uniform(-0.3, 0.3)))
        carry, parameters = self.ocean.controls(full, phase_days=float(rng.uniform(0, 360)), realized=realized)
        carry, _ = self.ocean.prepared.advance(carry, n_steps=self.burnin_steps, parameters=parameters, final_only=True)
        values = []
        # Monthly samples resolve the stock forcing, matching the response provider.
        stride = max(1, self.window_steps // 12)
        for _ in range(12 * self.windows * (2 if doubled else 1)):
            carry, _ = self.ocean.prepared.advance(carry, n_steps=stride, parameters=parameters, final_only=True)
            self.ocean.prepared.check_status(carry)
            values.append(self.ocean.observe(carry))
        samples = jnp.stack(values)
        import jax

        exceedance = jnp.mean(jax.nn.sigmoid(samples - self.thresholds), axis=0)
        return jnp.stack((jnp.mean(samples, axis=0), jnp.var(samples, axis=0), exceedance))

    def validate(self, point, checks):
        import jax
        import jax.numpy as jnp

        checks.metrics.update(
            response=self.kind,
            steps=self.steps,
            period_days=360,
            statistical_windows=self.windows,
            burnin_years=20,
        )
        if self.config["mode"] == "acceptance":
            checks.require("global research configuration", self.config["setup"] == "global_4deg", self.config["setup"])
            if self.kind == "trajectory":
                required_days = (
                    3600
                    if self.ocean.case.task == "perturbation"
                    else (30 if self.ocean.case.task in ("sensitivity", "geometry") else 360)
                )
                within_window = self.steps * self.ocean.dt >= required_days * 86400
                if self.ocean.case.task == "perturbation":
                    within_window = self.steps * self.ocean.dt == required_days * 86400
                checks.require(
                    "scientific response window",
                    within_window,
                    dict(required_days=required_days, simulated_days=self.steps * self.ocean.dt / 86400),
                )
        if self.kind in ("local", "trajectory"):
            values = self.values(point)
            checks.require("finite response", np.isfinite(values).all(), dict(shape=values.shape))
            return
        result = self.adjusted(point)
        if self.kind == "statistics":
            # Freeze event thresholds from the baseline climate, not its startup state.
            baseline_mean, baseline_variance = result["observations"][:2]
            self.thresholds = baseline_mean + 2 * jnp.sqrt(jnp.maximum(baseline_variance, 1))
            result = self.adjusted(point)
            checks.metrics["climate_event_thresholds"] = self.thresholds.tolist()
        checks.require("finite adjusted observations", np.isfinite(result["observations"]).all(), self.kind)
        if self.kind in ("equilibrium", "seasonal"):
            checks.require(
                "full-state residual",
                result["residual_kind"] == "full_state" and float(result["residual"]) < 1e-7,
                dict(kind=result["residual_kind"], residual=float(result["residual"])),
            )
            checks.require(
                "stable adjusted branch",
                float(result["spectral_radius"]) < 1,
                float(result["spectral_radius"]),
            )
            checks.require(
                "adjustment sensitivity residual",
                float(result["sensitivity_residual"]) < 1e-5,
                float(result["sensitivity_residual"]),
            )
            other = self.adjusted(point, initial_shift=0.3, doubled=True)
            checks.close(
                "adjusted-state independence",
                other["observations"],
                result["observations"],
                rtol=1e-5,
                atol=1e-5,
            )
            first = jax.jvp(lambda z: self.adjusted(z)["observations"], (point,), (jnp.ones_like(point),))[1]
            second = jax.jvp(
                lambda z: self.adjusted(z, initial_shift=0.3, doubled=True)["observations"],
                (point,),
                (jnp.ones_like(point),),
            )[1]
            checks.close("adjusted-gradient independence", second, first, rtol=1e-3, atol=1e-5)
            if self.kind == "seasonal":
                checks.require(
                    "twelve annual phases",
                    result["observations"].shape[0] == 12,
                    result["observations"].shape,
                )
            return
        allowed = (
            "shadowing",
            "linear_response",
            "ensemble_response",
            "invariant_equilibrium",
            "invariant_periodic",
        )
        checks.require("invariant response method", result["method"] in allowed, result["method"])
        checks.require(
            "effective sample size",
            float(result["effective_sample_size"]) >= 32,
            float(result["effective_sample_size"]),
        )
        se = np.asarray(result["standard_error"])
        checks.require("finite positive uncertainty", bool(np.isfinite(se).all() and (se >= 0).all()), se.tolist())
        checks.require(
            "split-window convergence",
            bool(np.all(np.abs(result["window_drift"]) <= 3 * se + 1e-5)),
            np.asarray(result["window_drift"]).tolist(),
        )
        empirical = np.stack([np.asarray(self.forward_statistics(point, seed=seed)) for seed in range(4)])
        uncertainty = np.std(empirical, axis=0, ddof=1) / 2
        checks.require(
            "independent climate statistics",
            bool(np.all(np.abs(np.mean(empirical, axis=0) - result["observations"]) <= 3 * (se + uncertainty) + 1e-4)),
            dict(
                provider=result["observations"],
                forward=np.mean(empirical, axis=0),
                standard_error=uncertainty,
            ),
        )
        repeated = self.adjusted(point, initial_shift=0.3, phase_days=180, doubled=True)
        checks.require(
            "invariant starting-state independence",
            bool(
                np.all(
                    np.abs(repeated["observations"] - result["observations"])
                    <= 3 * (se + repeated["standard_error"]) + 1e-4
                )
            ),
            dict(first=result["observations"], second=repeated["observations"]),
        )

    def audit(self, loss, point, checks, forward_loss=None):
        if self.kind != "statistics":
            return gradient_checks(loss, point, checks)
        import jax
        import jax.numpy as jnp

        value, gradient = jax.value_and_grad(loss)(point)
        direction = jnp.arange(1, point.size + 1, dtype=point.dtype)
        direction /= jnp.linalg.norm(direction)
        _, tangent = jax.jvp(loss, (point,), (direction,))
        checks.close("statistical forward/reverse derivative", tangent, jnp.vdot(gradient, direction))
        checks.require("finite statistical gradient", np.isfinite(gradient).all(), gradient.tolist())
        estimates = []
        for epsilon in (0.1, 0.03):
            differences = np.array(
                [
                    (
                        float(forward_loss(point + epsilon * direction, seed))
                        - float(forward_loss(point - epsilon * direction, seed))
                    )
                    / (2 * epsilon)
                    for seed in range(4, 8)
                ]
            )
            mean, se = float(np.mean(differences)), float(np.std(differences, ddof=1) / 2)
            estimates.append(dict(epsilon=epsilon, derivative=mean, standard_error=se))
            checks.require(f"resolved climate finite difference {epsilon}", abs(mean) > 3 * se + 1e-8, estimates[-1])
            checks.require(
                f"climate derivative confidence {epsilon}",
                abs(mean - float(tangent)) <= 3 * se + 0.02 * abs(mean),
                dict(ad=float(tangent), **estimates[-1]),
            )
        return float(value), gradient
