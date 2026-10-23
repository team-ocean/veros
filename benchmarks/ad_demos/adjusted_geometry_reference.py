"""Independent geometry errors on adjusted states and stable periodic climates."""

import math
import numpy as np


def current_fields(carry):
    import jax.numpy as jnp

    vs = carry.variables
    return jnp.stack([getattr(vs, name)[2:-2, 2:-2, :, vs.tau] for name in ("temp", "salt", "u", "v")])


def with_current_fields(prepared, carry, fields):
    return prepared.replace_variables(
        carry,
        **{
            name: getattr(carry.variables, name).at[2:-2, 2:-2, :, carry.variables.tau].set(field)
            for name, field in zip(("temp", "salt", "u", "v"), fields)
        },
    )


class AdjustedGeometryError:
    """Compare native adjusted fields on a common physical grid.

    Seasonal errors average matched annual phases. The statistical error is the
    invariant expectation of the same squared physical-field discrepancy on a
    validated stable joint periodic orbit. Ordinary forward climates use fresh
    initial perturbations and paired forcing phases, without starting from the
    solved orbit. No chaotic invariant-response method is supplied here.
    """

    def __init__(self, reference, carry, parameters, response, steps, *, period_steps=None):
        import jax
        from veros.ocean_response import FloatingCoordinates
        from .research_ocean import YEAR
        from .grid_reference import endpoint_error

        if response not in ("equilibrium", "seasonal", "statistics"):
            raise ValueError("Adjusted geometry requires an equilibrium, seasonal or statistical response")
        if steps < 1 or int(steps) != steps:
            raise ValueError("The response horizon must contain a positive integer number of steps")
        config = reference.ocean.config
        if (
            response == "statistics"
            and config.get("statistical_method", "invariant_periodic" if config["setup"] == "adjusted_basin" else None)
            != "invariant_periodic"
        ):
            raise NotImplementedError("Statistical geometry requires a declared, validated invariant_periodic method")
        self.reference, self.kind, self.steps = reference, response, int(steps)
        self.coarse, self.fine = reference.ocean.prepared, reference.prepared
        self.baseline = reference.ocean.initial
        dt = self.coarse._template.settings.dt_tracer
        if period_steps is None:
            period_steps = 1 if response == "equilibrium" else round(YEAR / dt)
            if response != "equilibrium" and abs(period_steps * dt - YEAR) > 1e-8:
                raise ValueError("The native timestep must divide the declared annual period")
        if period_steps < 1 or int(period_steps) != period_steps:
            raise ValueError("The adjusted period must contain a positive integer number of native timesteps")
        self.period_steps = int(period_steps)
        self.stride = max(1, max(self.period_steps, self.steps) // 12)
        count = (
            self.period_steps // math.gcd(self.period_steps, self.stride)
            if response == "statistics"
            else min(12, self.period_steps)
        )
        self.phases = tuple(index / count for index in range(count))
        self.windows = 4 if config.get("mode") == "smoke" else 16
        self.ram_checkpoints = config.get("ram_checkpoints", 4)
        self.codec = FloatingCoordinates((carry, parameters), dynamic_discrete=True)
        self.initial = jax.jit(reference.initial)
        cs, fs = self.coarse._template.settings, self.fine._template.settings
        self.compare = jax.jit(lambda c, f: endpoint_error(c.variables, f.variables, cs, fs, measure="physical"))

    def adjusted(self, model, carry, parameters, refinement):
        options = dict(parameters=parameters, observe=current_fields, ram_checkpoints=self.ram_checkpoints)
        if self.kind == "equilibrium":
            return model.equilibrium_response(carry, tolerance=1e-11, **options)
        return model.periodic_response(
            carry,
            period_steps=self.period_steps * refinement,
            phases=self.phases,
            tolerance=1e-9,
            **options,
        )

    def comparison(self, carry, fine, coarse_fields, fine_fields):
        return self.compare(
            with_current_fields(self.coarse, carry, coarse_fields),
            with_current_fields(self.fine, fine, fine_fields),
        )

    def initial_budget(self, carry, fine):
        import jax.numpy as jnp
        from .grid_reference import physical_horizontal_measures

        areas = [
            physical_horizontal_measures(c.variables, m._template.settings)["area_t"]
            for c, m in ((carry, self.coarse), (fine, self.fine))
        ]
        residuals = []
        for name in ("temp", "salt"):
            totals = []
            for current, area in zip((carry, fine), areas):
                vs = current.variables
                totals.append(
                    jnp.sum(
                        getattr(vs, name)[2:-2, 2:-2] * (area[..., None] * vs.dzt * vs.hfacT[2:-2, 2:-2])[..., None],
                        axis=(0, 1, 2),
                    )
                )
            residuals.append(jnp.max(abs(totals[1] - totals[0])) / jnp.maximum(jnp.max(abs(totals[0])), 1))
        return jnp.max(jnp.stack(residuals))

    @staticmethod
    def valid(result):
        import jax.numpy as jnp

        margin = 1 - result["spectral_radius"]
        return (
            (result["residual"] < 0.1 * result["state_tolerance"])
            & (margin > 0)
            & (result["stability_residual"] < jnp.minimum(1e-5, margin))
            & (result["sensitivity_residual"] < 1e-5)
        )

    def values(self, point):
        import jax
        import jax.numpy as jnp
        from .geometry_response import FiniteGeometryError

        carry, parameters = self.codec.unpack(point)
        fine = self.initial(carry)
        coarse_result = self.adjusted(self.coarse, carry, parameters, 1)
        fine_result = self.adjusted(self.fine, fine, parameters, self.reference.refinement)
        coarse_fields, fine_fields = coarse_result["observations"], fine_result["observations"]
        if self.kind == "equilibrium":
            comparison = self.comparison(carry, fine, coarse_fields, fine_fields)
        else:
            samples = jax.vmap(lambda a, b: self.comparison(carry, fine, a, b))(coarse_fields, fine_fields)
            comparison = {
                name: jnp.mean(value, axis=0) if name in ("error", "field_errors") else jnp.max(value, axis=0)
                for name, value in samples.items()
            }
        valid = self.valid(coarse_result) & self.valid(fine_result)
        result = jnp.concatenate(
            (
                jnp.stack(
                    (
                        jnp.where(valid, comparison["error"], jnp.nan),
                        jnp.maximum(comparison["remap_budget_residual"], self.initial_budget(carry, fine)),
                        comparison["projection_geometry_residual"],
                        comparison["projection_horizontal_residual"],
                    )
                ),
                comparison["field_errors"],
                jnp.asarray([FiniteGeometryError.operator_change(self, carry)]),
            )
        )
        return jnp.concatenate((result[:1], jax.lax.stop_gradient(result[1:])))

    def transformed(self, example):
        return self.values

    def forward(self, point, seed):
        """Evaluate an independent ordinary climate with the declared phase population."""
        import jax
        import jax.numpy as jnp
        from veros.core.numerics import calc_initial_conditions_kernel
        from veros.routines import tracing_context
        from .geometry_response import FiniteGeometryError

        if self.kind != "statistics":
            raise ValueError("Independent climate evaluation requires the statistical response")
        if any(isinstance(value, jax.core.Tracer) for value in jax.tree.leaves(point)):
            raise ValueError("Ordinary forward climates do not provide invariant-response derivatives")
        if not np.isfinite(np.asarray(self.values(point))[0]):
            raise RuntimeError("Validate the stable periodic reference before running forward climates")
        carry, parameters = self.codec.unpack(point)
        rng = np.random.default_rng(seed)
        phase = int(rng.integers(len(self.phases))) * self.stride % self.period_steps
        vs = carry.variables
        carry = self.coarse.replace_variables(
            carry,
            temp=vs.temp + float(rng.uniform(-0.3, 0.3)) * vs.maskT[..., None],
            time=jnp.asarray(phase * self.coarse._template.settings.dt_tracer, dtype=vs.time.dtype),
        )
        state = self.coarse._make_state(carry.variables)
        with tracing_context(), state.variables.unlock():
            state.variables.update(calc_initial_conditions_kernel(state))
        carry = carry._replace(variables=state.variables)
        fine = self.initial(carry)
        budget = self.initial_budget(carry, fine)
        starts = []
        for model, initial, refinement in ((self.coarse, carry, 1), (self.fine, fine, self.reference.refinement)):
            starts.append(FiniteGeometryError.advance(model, initial, parameters, 20 * self.period_steps * refinement))
        carry, fine = starts
        comparisons = []
        for _ in range(len(self.phases) * self.windows):
            carry = FiniteGeometryError.advance(self.coarse, carry, parameters, self.stride)
            fine = FiniteGeometryError.advance(self.fine, fine, parameters, self.stride * self.reference.refinement)
            comparisons.append(self.compare(carry, fine))
        comparison = {
            name: (jnp.mean if name in ("error", "field_errors") else jnp.max)(
                jnp.stack([row[name] for row in comparisons]), axis=0
            )
            for name in comparisons[0]
        }
        return jnp.concatenate(
            (
                jnp.stack(
                    (
                        comparison["error"],
                        jnp.maximum(budget, comparison["remap_budget_residual"]),
                        comparison["projection_geometry_residual"],
                        comparison["projection_horizontal_residual"],
                    )
                ),
                comparison["field_errors"],
                jnp.asarray([FiniteGeometryError.operator_change(self, self.codec.unpack(point)[0])]),
            )
        )
