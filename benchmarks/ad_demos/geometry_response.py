"""Bounded finite-window products for an independent geometry error objective."""

import hashlib
import numpy as np


class FiniteGeometryError:
    """Expose a differentiable scalar error and nondifferentiable audit metrics."""

    def __init__(self, reference, carry, parameters, steps):
        import jax
        import jax.numpy as jnp
        from veros.ocean_response import FloatingCoordinates
        from veros.checkpointing import CheckpointedIntegrator
        from .grid_reference import endpoint_error

        if steps < 1 or int(steps) != steps:
            raise ValueError("Finite geometry references require a positive integer horizon")
        self.reference, self.steps = reference, int(steps)
        self.coarse, self.fine = reference.ocean.prepared, reference.prepared
        self.device = next(iter(carry.variables.temp.devices()))
        self.codec = FloatingCoordinates((carry, parameters), dynamic_discrete=True)
        self.unpack = jax.jit(self.codec.unpack)
        self.initial = jax.jit(reference.initial)
        cs, fs = self.coarse._template.settings, self.fine._template.settings
        self.compare = jax.jit(lambda c, f: endpoint_error(c.variables, f.variables, cs, fs, measure="physical"))
        self.error = jax.jit(lambda c, f: self.compare(c, f)["error"])
        self.tangent_steps = [
            jax.jit(
                lambda c, p, dc, dp, model=model: jax.jvp(lambda c, p: model.step(c, parameters=p)[0], (c, p), (dc, dp))
            )
            for model in (self.coarse, self.fine)
        ]
        self.drivers = [
            CheckpointedIntegrator(model, terminal_objective=lambda c: jnp.asarray(0.0))
            for model in (self.coarse, self.fine)
        ]
        self.key, self.ends, self.result = None, None, None
        self.functions, self.statistics = {}, []
        self.active_product, self.cache_releases = None, 0
        self.baseline = reference.ocean.initial
        self.progress = getattr(reference.ocean, "geometry_reference_progress", None)

    def mark(self, stage):
        if self.progress is not None:
            self.progress(stage)

    def activate(self, product):
        if self.reference.ocean.config.get("setup") == "global_4deg" and product != self.active_product:
            import ctypes
            import gc
            import sys
            import jax

            jax.clear_caches()
            gc.collect()
            if sys.platform == "linux":
                allocator = ctypes.CDLL(None)
                if hasattr(allocator, "malloc_trim"):
                    allocator.malloc_trim(0)
            self.cache_releases += 1
        self.active_product = product

    def operator_change(self, carry):
        import jax
        import jax.numpy as jnp
        from veros.routines import tracing_context

        solver = self.coarse._template._linear_solver
        with tracing_context():
            current = solver.geometry_coefficients(self.coarse._make_state(carry.variables))
            baseline = solver.geometry_coefficients(self.coarse._make_state(self.baseline.variables))
        differences = [
            jnp.sum((a.astype(float) - b.astype(float)) ** 2)
            for a, b in zip(jax.tree.leaves(current), jax.tree.leaves(baseline))
        ]
        norm = sum(jnp.sum(b.astype(float) ** 2) for b in jax.tree.leaves(baseline))
        vertical = jnp.sum((carry.variables.dzt - self.baseline.variables.dzt) ** 2) / jnp.sum(
            self.baseline.variables.dzt**2
        )
        return jnp.sqrt(sum(differences) / jnp.maximum(norm, 1e-30) + vertical)

    @staticmethod
    def advance(model, carry, parameters, steps):
        import jax

        for _ in range(steps):
            carry, _ = model.step(carry, parameters=parameters)
            # Bound queued native states during host-only reference evaluation.
            jax.block_until_ready(carry)
            model.check_status(carry)
        return carry

    def evaluate(self, inputs):
        import jax.numpy as jnp
        from .grid_reference import physical_horizontal_measures

        key = hashlib.sha256(np.asarray(inputs).tobytes()).digest()
        if key == self.key:
            return self.result
        self.key, self.ends, self.result = None, None, None
        if self.reference.ocean.config.get("setup") == "global_4deg":
            import ctypes
            import gc
            import sys

            gc.collect()
            if sys.platform == "linux":
                allocator = ctypes.CDLL(None)
                if hasattr(allocator, "malloc_trim"):
                    allocator.malloc_trim(0)
        carry, parameters = self.unpack(inputs)
        fine = self.initial(carry)
        self.coarse.check_status(carry)
        self.fine.check_status(fine)
        a = physical_horizontal_measures(carry.variables, self.coarse._template.settings)["area_t"]
        b = physical_horizontal_measures(fine.variables, self.fine._template.settings)["area_t"]
        budgets = []
        for name in ("temp", "salt"):

            def inventory(c, area):
                vs = c.variables
                return jnp.sum(
                    getattr(vs, name)[2:-2, 2:-2] * (area[..., None] * vs.dzt * vs.hfacT[2:-2, 2:-2])[..., None],
                    axis=(0, 1, 2),
                )

            before, after = inventory(carry, a), inventory(fine, b)
            budgets.append(jnp.max(abs(after - before)) / jnp.maximum(jnp.max(abs(before)), 1))
        cfinal = self.advance(self.coarse, carry, parameters, self.steps)
        ffinal = self.advance(self.fine, fine, parameters, self.steps * self.reference.refinement)
        self.coarse.check_status(cfinal)
        self.fine.check_status(ffinal)
        comparison = self.compare(cfinal, ffinal)
        result = jnp.concatenate(
            (
                jnp.stack(
                    (
                        comparison["error"],
                        jnp.maximum(comparison["remap_budget_residual"], jnp.max(jnp.stack(budgets))),
                        comparison["projection_geometry_residual"],
                        comparison["projection_horizontal_residual"],
                    )
                ),
                comparison["field_errors"],
                jnp.asarray([self.operator_change(carry)]),
            )
        )
        self.key, self.ends, self.result = key, (carry, fine, parameters, cfinal, ffinal), np.asarray(result)
        return self.result

    def jvp(self, inputs, direction):
        import jax
        import jax.numpy as jnp

        self.activate("tangent")
        result = self.evaluate(inputs)
        (_, _), (dc, dp) = jax.jvp(self.unpack, (inputs,), (direction,))
        carry, fine, parameters, _, _ = self.ends
        _, df = jax.jvp(self.initial, (carry,), (dc,))
        finals, tangents = [], []
        for model, start, tangent, count, product in zip(
            (self.coarse, self.fine),
            (carry, fine),
            (dc, df),
            (self.steps, self.steps * self.reference.refinement),
            self.tangent_steps,
        ):
            current = start
            for _ in range(count):
                _, tangent = product(current, parameters, tangent, dp)
                current = model.step(current, parameters=parameters)[0]
                jax.block_until_ready((current, tangent))
                model.check_status(current)
            finals.append(current)
            tangents.append(tangent)
        _, product = jax.jvp(self.error, tuple(finals), tuple(tangents))
        return np.asarray(jnp.zeros_like(jnp.asarray(result)).at[0].set(product))

    def vjp(self, inputs, seed):
        import jax
        import jax.numpy as jnp
        from veros.checkpointing import BinomialSchedule
        from .finite_response import add_cotangents

        self.activate("adjoint")
        self.evaluate(inputs)
        carry, fine, parameters, cfinal, ffinal = self.ends
        self.mark("endpoint_pullback")
        _, pullback = jax.vjp(self.error, cfinal, ffinal)
        terminal_seeds = pullback(jnp.asarray(seed)[0])
        jax.block_until_ready(terminal_seeds)
        del pullback
        expected_finals = [cfinal, ffinal]
        self.ends, self.key = None, None
        del cfinal, ffinal
        results = []
        config = self.reference.ocean.config
        for index, (driver, start, cotangent, count, expected_final) in enumerate(
            zip(
                self.drivers,
                (carry, fine),
                terminal_seeds,
                (self.steps, self.steps * self.reference.refinement),
                expected_finals,
            )
        ):
            stage = "coarse_checkpoint" if index == 0 else "fine_checkpoint"
            self.mark(stage)
            self.activate(stage)
            result = driver.value_and_grad(
                start,
                n_steps=count,
                parameters=parameters,
                terminal_cotangent=cotangent,
                schedule=BinomialSchedule(count, ram_checkpoints=config.get("ram_checkpoints", 4), disk_checkpoints=0),
            )
            results.append((result.gradient, result.parameter_gradient))
            jax.block_until_ready(results[-1])
            replay = 0.0
            for actual, expected in zip(jax.tree.leaves(result.final), jax.tree.leaves(expected_final)):
                if np.issubdtype(expected.dtype, np.floating):
                    replay = max(
                        replay,
                        float(np.max(abs(np.asarray(actual) - np.asarray(expected))))
                        / max(float(np.max(abs(np.asarray(expected)))), 1),
                    )
            self.statistics.append(dict(result.statistics, reference_steps=count, replay_relative_error=replay))
            if not np.isfinite(replay) or replay > 1e-10:
                raise RuntimeError("Geometry reference checkpoint primal differs from its ordinary native integration")
            expected_finals[index] = None
            del result, expected_final
        self.mark("initialization_pullback")
        self.activate("initialization_pullback")
        fine_seed = jax.vjp(self.initial, carry)[1](results[1][0])[0]
        state_seed = add_cotangents(results[0][0], fine_seed)
        parameter_seed = add_cotangents(results[0][1], results[1][1])
        gradient = jax.vjp(self.unpack, inputs)[1]((state_seed, parameter_seed))[0]
        gradient = np.asarray(gradient)
        self.mark("reverse_complete")
        self.activate("reverse_complete")
        return gradient

    def transformed(self, example):
        import jax
        from jax.custom_derivatives import linear_call
        from veros.backend import device_callback

        key = (example.shape, str(example.dtype))
        if key not in self.functions:
            output = jax.ShapeDtypeStruct((9,), example.dtype)
            input_spec = jax.ShapeDtypeStruct(example.shape, example.dtype)
            primal_callback = device_callback(self.evaluate, self.device)
            tangent_callback = device_callback(self.jvp, self.device)
            adjoint_callback = device_callback(self.vjp, self.device)

            @jax.custom_jvp
            def function(inputs):
                return jax.pure_callback(primal_callback, output, inputs, vmap_method="sequential")

            @function.defjvp
            def differentiate(primals, tangents):
                (inputs,), (direction,) = primals, tangents

                def forward(inputs, direction):
                    return jax.pure_callback(tangent_callback, output, inputs, direction, vmap_method="sequential")

                def reverse(inputs, seed):
                    return jax.pure_callback(adjoint_callback, input_spec, inputs, seed, vmap_method="sequential")

                return function(inputs), linear_call(forward, reverse, inputs, direction)

            self.functions[key] = function
        return self.functions[key]
