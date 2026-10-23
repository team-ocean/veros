"""Host response products with bounded checkpoint replay and streamed tangents."""

import time
import tempfile
import gc
import ctypes
import sys

import numpy as np

_ACTIVE_PRODUCT = None


def add_cotangents(left, right):
    import jax

    return jax.tree.map(lambda a, b: a if a.dtype == jax.dtypes.float0 else a + b, left, right)


def tangent_codec(tangent):
    """Map floating tangent leaves while preserving float0 integer cotangents."""
    import jax

    leaves, structure = jax.tree.flatten(tangent)
    zeros = [
        np.zeros(leaf.shape, dtype=jax.dtypes.float0) if leaf.dtype == jax.dtypes.float0 else None for leaf in leaves
    ]

    def pack(value):
        return tuple(leaf for leaf in jax.tree.leaves(value) if leaf.dtype != jax.dtypes.float0)

    def unpack(floating):
        iterator = iter(floating)
        return jax.tree.unflatten(structure, [next(iterator) if zero is None else zero for zero in zeros])

    return pack, unpack


class FiniteResponse:
    def __init__(self, response):
        import jax
        import jax.numpy as jnp
        from veros.checkpointing import CheckpointedIntegrator

        self.response = response
        self.prepared = response.ocean.prepared
        self.driver = CheckpointedIntegrator(self.prepared, terminal_objective=lambda carry: jnp.asarray(0.0))
        self.tangent_step = jax.jit(
            lambda c, p, dc, dp: jax.jvp(lambda c, p: self.prepared.step(c, parameters=p)[0], (c, p), (dc, dp))
        )
        self.observe = jax.jit(response.ocean.observe)
        self.observe_vjp = jax.jit(lambda c, seed: jax.vjp(self.observe, c)[1](seed)[0])
        self.statistics = []
        self.active_product = None
        self.cache_releases = 0
        self.matrix_step = None

    def activate(self, product):
        """Release alternate AD executables on the large global configuration."""
        import jax

        global _ACTIVE_PRODUCT
        key = (id(self), product)
        if self.response.config.get("setup") == "global_4deg" and (
            key != _ACTIVE_PRODUCT or self.active_product is None
        ):
            jax.clear_caches()
            gc.collect()
            if sys.platform == "linux":
                allocator = ctypes.CDLL(None)
                if hasattr(allocator, "malloc_trim"):
                    allocator.malloc_trim(0)
            self.cache_releases += 1
            _ACTIVE_PRODUCT = key
        self.active_product = product

    def specifications(self, heldout=False):
        if self.response.kind == "local":
            phases = (45, 135, 225, 315) if heldout else (0, 90, 180, 270)
            return [(phase, 0.2 if heldout else 0, (1,)) for phase in phases]
        steps = self.response.steps * (2 if heldout else 1)
        boundaries = tuple(sorted({max(1, steps // 3), max(1, 2 * steps // 3), steps}))
        return [(90 if heldout else 0, 0, boundaries)]

    def controls(self, z, phase, shift):
        full = self.response.full(z).at[0].add(shift)
        return self.response.ocean.controls(full, phase_days=phase)

    def windows(self, z, heldout=False):
        return [
            (self.controls(z, phase, shift), boundaries) for phase, shift, boundaries in self.specifications(heldout)
        ]

    def values(self, z, *, heldout=False):
        import jax
        import jax.numpy as jnp

        values = []
        for phase, shift, boundaries in self.specifications(heldout):
            carry, parameters = self.controls(z, phase, shift)
            previous = 0
            for boundary in boundaries:
                carry, _ = self.prepared.advance(
                    carry, n_steps=boundary - previous, parameters=parameters, final_only=True
                )
                if not isinstance(carry.failure_code, jax.core.Tracer):
                    self.prepared.check_status(carry)
                values.append(jnp.where(carry.failure_code == 0, self.observe(carry), jnp.nan))
                previous = boundary
        return jnp.stack(values)

    def jvp(self, z, direction, *, heldout=False):
        import jax
        import jax.numpy as jnp

        self.activate("tangent")
        values, products = [], []
        for phase, shift, boundaries in self.specifications(heldout):
            (carry, parameters), (dc, dp) = jax.jvp(
                lambda point: self.controls(point, phase, shift), (z,), (direction,)
            )
            carry, parameters = self.controls(z, phase, shift)
            previous = 0
            for boundary in boundaries:
                for _ in range(boundary - previous):
                    _, dc = self.tangent_step(carry, parameters, dc, dp)
                    carry, _ = self.prepared.step(carry, parameters=parameters)
                    self.prepared.check_status(carry)
                _, tangent = jax.jvp(self.observe, (carry,), (dc,))
                value = self.observe(carry)
                values.append(value)
                products.append(tangent)
                previous = boundary
        return jnp.stack(values), jnp.stack(products)

    def jacobian(self, z, *, heldout=False):
        """Share one primal linearization across control columns at each step."""
        import jax
        import jax.numpy as jnp

        self.activate("matrix_tangent")
        values, products = [], []
        numerical_step = self.prepared.step
        if hasattr(self.prepared, "check_configuration"):
            self.prepared.check_configuration()
        for phase, shift, boundaries in self.specifications(heldout):
            seeds = [
                jax.jvp(lambda point: self.controls(point, phase, shift), (z,), (basis,)) for basis in jnp.eye(z.size)
            ]
            (carry, parameters), (dc, dp) = seeds[0]
            carry, parameters = self.controls(z, phase, shift)
            parameters = self.prepared.validate_parameters(parameters)
            if self.matrix_step is None:
                self.pack_carry, unpack_carry = tangent_codec(dc)
                self.pack_parameters, unpack_parameters = tangent_codec(dp)

                def step(carry, parameters, carry_columns, parameter_columns):
                    _, linearized = jax.linearize(lambda c, p: numerical_step(c, parameters=p)[0], carry, parameters)
                    columns = jax.lax.map(
                        lambda pair: self.pack_carry(linearized(unpack_carry(pair[0]), unpack_parameters(pair[1]))),
                        (carry_columns, parameter_columns),
                    )
                    return columns

                def observe(carry, columns):
                    _, linearized = jax.linearize(self.observe, carry)
                    return jax.lax.map(lambda column: linearized(unpack_carry(column)), columns)

                self.matrix_step, self.matrix_observe = jax.jit(step), jax.jit(observe)
            dc = jax.tree.map(lambda *leaves: jnp.stack(leaves), *[self.pack_carry(seed[1][0]) for seed in seeds])
            dp = jax.tree.map(lambda *leaves: jnp.stack(leaves), *[self.pack_parameters(seed[1][1]) for seed in seeds])
            del seeds
            previous = 0
            for boundary in boundaries:
                for _ in range(boundary - previous):
                    dc = self.matrix_step(carry, parameters, dc, dp)
                    carry, _ = self.prepared.step(carry, parameters=parameters)
                    self.prepared.check_status(carry)
                value, columns = self.observe(carry), self.matrix_observe(carry, dc)
                values.append(value)
                products.append(columns)
                previous = boundary
        return jnp.stack(values), jnp.stack(products, axis=1)

    def vjp(self, z, seed, *, heldout=False):
        import jax
        import jax.numpy as jnp
        from veros.checkpointing import BinomialSchedule

        self.activate("adjoint")
        cursor, gradient = 0, jnp.zeros_like(z)
        started = time.perf_counter()
        for phase, shift, boundaries in self.specifications(heldout):
            initial, parameters = self.controls(z, phase, shift)
            starts, finals, lengths = [], [], []
            carry, previous = initial, 0
            for boundary in boundaries:
                starts.append(carry)
                length = boundary - previous
                carry, _ = self.prepared.advance(carry, n_steps=length, parameters=parameters, final_only=True)
                self.prepared.check_status(carry)
                finals.append(carry)
                lengths.append(length)
                previous = boundary
            adjoint = None
            parameter_gradient = jax.tree.map(jnp.zeros_like, parameters)
            for index in reversed(range(len(boundaries))):
                observation_seed = self.observe_vjp(finals[index], seed[cursor + index])
                adjoint = observation_seed if adjoint is None else add_cotangents(adjoint, observation_seed)
                result = self.driver.value_and_grad(
                    starts[index],
                    n_steps=lengths[index],
                    parameters=parameters,
                    terminal_cotangent=adjoint,
                    directory=tempfile.gettempdir() if self.response.config.get("disk_checkpoints", 0) else None,
                    schedule=BinomialSchedule(
                        lengths[index],
                        ram_checkpoints=self.response.config.get("ram_checkpoints", 4),
                        disk_checkpoints=self.response.config.get("disk_checkpoints", 0),
                    ),
                )
                adjoint = result.gradient
                parameter_gradient = add_cotangents(parameter_gradient, result.parameter_gradient)
                replay_error = np.max(
                    np.abs(np.asarray(self.observe(result.final)) - np.asarray(self.observe(finals[index])))
                )
                self.statistics.append(
                    dict(
                        result.statistics,
                        observation_boundary=boundaries[index],
                        checkpoint_observation_discrepancy=float(replay_error),
                    )
                )
            jax.block_until_ready((adjoint, parameter_gradient))
            _, pullback = jax.vjp(lambda point: self.controls(point, phase, shift), z)
            gradient = gradient + pullback((adjoint, parameter_gradient))[0]
            jax.block_until_ready(gradient)
            del pullback
            del initial, parameters, starts, finals, carry, adjoint, parameter_gradient, observation_seed, result
            if self.response.config.get("setup") == "global_4deg":
                gc.collect()
                if sys.platform == "linux":
                    allocator = ctypes.CDLL(None)
                    if hasattr(allocator, "malloc_trim"):
                        allocator.malloc_trim(0)
            cursor += len(boundaries)
        jax.block_until_ready(gradient)
        self.last_seconds = time.perf_counter() - started
        return gradient


class ObservationFunction:
    """Compose a JAX observation function with explicit host response products."""

    def __init__(self, response, function, *, heldout=False):
        self.response, self.function, self.heldout = response, function, heldout

    def __call__(self, z):
        return self.function(self.response.values(z, heldout=self.heldout), z)

    def map_controls(self, mapping):
        return MappedObservationFunction(self, mapping)

    def value_and_grad(self, z):
        import jax

        values = self.response.values(z, heldout=self.heldout)
        _, (seed, direct) = jax.value_and_grad(self.function, argnums=(0, 1))(values, z)
        return self.function(values, z), self.response.vjp(z, seed, heldout=self.heldout) + direct

    def jvp(self, z, direction):
        import jax

        values, tangent = self.response.jvp(z, direction, heldout=self.heldout)
        _, product = jax.jvp(self.function, (values, z), (tangent, direction))
        return self.function(values, z), product

    def jacobian(self, z):
        import jax
        import jax.numpy as jnp

        if getattr(self.response, "kind", None) in ("local", "trajectory"):
            values, columns = self.response.finite_provider().jacobian(z, heldout=self.heldout)
            return np.stack(
                [
                    np.asarray(jax.jvp(self.function, (values, z), (column, basis))[1]).reshape(-1)
                    for column, basis in zip(columns, jnp.eye(z.size))
                ],
                axis=1,
            )
        return np.stack([np.asarray(self.jvp(z, basis)[1]).reshape(-1) for basis in jnp.eye(z.size)], axis=1)

    def vjp(self, z, seed):
        import jax

        values = self.response.values(z, heldout=self.heldout)
        _, pullback = jax.vjp(self.function, values, z)
        observation_seed, direct = pullback(seed)
        return self.function(values, z), self.response.vjp(z, observation_seed, heldout=self.heldout) + direct


class TerminalFeature(ObservationFunction):
    """Differentiate an affine terminal observation in one checkpoint traversal."""

    gradient_method = "terminal checkpoint adjoint"

    def __init__(self, response, index, *, scale=1.0, offset=0.0, heldout=False):
        from veros.checkpointing import CheckpointedIntegrator

        if response.kind != "trajectory":
            raise ValueError("Terminal features require a trajectory response")
        super().__init__(response, lambda values, z: scale * values[-1, index] + offset, heldout=heldout)
        self.driver = CheckpointedIntegrator(
            response.ocean.prepared,
            terminal_objective=lambda carry: scale * response.ocean.observe(carry)[index] + offset,
        )

    def value_and_grad(self, z):
        import jax
        from veros.checkpointing import BinomialSchedule

        provider = self.response.finite_provider()
        provider.activate("adjoint")
        ((phase, shift, boundaries),) = provider.specifications(self.heldout)
        _, pullback = jax.vjp(lambda point: provider.controls(point, phase, shift), z)
        initial, parameters = provider.controls(z, phase, shift)
        started = time.perf_counter()
        result = self.driver.value_and_grad(
            initial,
            n_steps=boundaries[-1],
            parameters=parameters,
            directory=tempfile.gettempdir() if self.response.config.get("disk_checkpoints", 0) else None,
            schedule=BinomialSchedule(
                boundaries[-1],
                ram_checkpoints=self.response.config.get("ram_checkpoints", 4),
                disk_checkpoints=self.response.config.get("disk_checkpoints", 0),
            ),
        )
        gradient = pullback((result.gradient, result.parameter_gradient))[0]
        jax.block_until_ready(gradient)
        provider.statistics.append(dict(result.statistics, observation_boundary=boundaries[-1], terminal_feature=True))
        provider.last_seconds = time.perf_counter() - started
        return result.value, gradient

    def vjp(self, z, seed):
        value, gradient = self.value_and_grad(z)
        return value, gradient * seed


class MappedObservationFunction:
    """Apply a control map while retaining the explicit response derivative provider."""

    def __init__(self, function, mapping):
        self.function, self.mapping = function, mapping

    def __call__(self, point):
        return self.function(self.mapping(point))

    def value_and_grad(self, point):
        import jax

        full, pullback = jax.vjp(self.mapping, point)
        value, gradient = self.function.value_and_grad(full)
        return value, pullback(gradient)[0]

    def jvp(self, point, direction):
        import jax

        full, tangent = jax.jvp(self.mapping, (point,), (direction,))
        return self.function.jvp(full, tangent)

    def vjp(self, point, seed):
        import jax

        full, pullback = jax.vjp(self.mapping, point)
        value, gradient = self.function.vjp(full, seed)
        return value, pullback(gradient)[0]

    def jacobian(self, point):
        import jax.numpy as jnp

        return np.stack(
            [np.asarray(self.jvp(point, basis)[1]).reshape(-1) for basis in jnp.eye(point.size, dtype=point.dtype)],
            axis=1,
        )


class ControlObjective:
    """Transpose a small control Jacobian assembled from streamed JVPs.

    A scalar grid-design objective has few inputs and a large native state.
    Its complete control gradient needs one JVP per input, independently of
    state dimension, without constructing a full-state reverse trajectory.
    """

    gradient_method = "streamed control Jacobian transpose"

    def __init__(self, function):
        self.function = function

    def __call__(self, point):
        return self.function(point)

    def value_and_grad(self, point):
        value, gradient = forward_value_and_grad(self.function)(point)
        if np.ndim(value):
            raise ValueError("ControlObjective requires a scalar objective")
        return value, gradient

    def jvp(self, point, direction):
        return jvp(self.function, point, direction)

    def vjp(self, point, seed):
        value, gradient = self.value_and_grad(point)
        return value, gradient * seed

    def jacobian(self, point):
        return self.value_and_grad(point)[1]


def value_and_grad(function):
    import jax

    return (
        function.value_and_grad
        if isinstance(function, (ObservationFunction, MappedObservationFunction, ControlObjective))
        else jax.value_and_grad(function)
    )


def jvp(function, point, direction):
    import jax

    if isinstance(function, (ObservationFunction, MappedObservationFunction, ControlObjective)):
        return function.jvp(point, direction)
    return jax.jvp(function, (point,), (direction,))


def forward_value_and_grad(function):
    """Assemble a scalar gradient from sequential control-space products."""
    import jax.numpy as jnp

    def evaluate(point):
        products = [jvp(function, point, basis) for basis in jnp.eye(point.size, dtype=point.dtype)]
        return products[0][0], jnp.stack([product[1] for product in products])

    return evaluate
