"""Host fixed-point solves with implicit first-order response products."""

import hashlib

import numpy as np


class AdjustmentError(RuntimeError):
    """An adjusted state or its sensitivity did not resolve convergence."""


class ImplicitResponse:
    """Differentiate a scaled, constrained host map with explicit JVP/VJP products.

    Transposes use the same coordinates and constraints. Observation products
    include direct input dependence. An optional polynomial product returns
    A**p v and sum(A**k v, k=0..p-1) for the map Jacobian or its transpose.
    Every solve independently checks the original (I-A) equation.

    Supplied stability eigenpairs receive native residual checks. The provider
    establishes coverage of the dominant spectrum. An optional state mask can
    omit coordinates only when their Jacobian rows and columns are identically
    zero for every input. Krylov bases use the active coordinates, and residual
    checks retain the full native operator.
    """

    def __init__(
        self,
        initial,
        mapping,
        product,
        transpose,
        observe,
        observation_product,
        observation_transpose,
        *,
        tolerance=1e-7,
        max_iterations=32,
        warmup=4,
        polynomial_product=None,
        stability_solver=None,
        state_mask=None,
    ):
        self.initial = initial
        self.mapping, self.product, self.transpose = mapping, product, transpose
        self.observe = observe
        self.observation_product, self.observation_transpose = observation_product, observation_transpose
        self.tolerance, self.max_iterations, self.warmup = tolerance, max_iterations, warmup
        self.cache = {}
        self.transformed_functions = {}
        self.linear_solve_count = 0
        self.last_linear_solve = None
        self.last_stability = None
        self.polynomial_product = polynomial_product
        if stability_solver is not None and not callable(stability_solver):
            raise TypeError("stability_solver must be callable")
        self.stability_solver = stability_solver
        self.state_mask = state_mask

    def active_indices(self, state, inputs):
        if self.state_mask is None:
            return None
        mask = np.asarray(self.state_mask(inputs))
        if mask.dtype != bool or mask.shape != state.shape or not np.any(mask):
            raise ValueError("State mask must be a nonempty boolean support matching the state")
        return None if np.all(mask) else np.flatnonzero(mask)

    @staticmethod
    def restricted_operator(operator, indices):
        from scipy.sparse.linalg import LinearOperator

        if indices is None:
            return operator

        def apply(vector):
            full = np.zeros(operator.shape[0], dtype=vector.dtype)
            full[indices] = vector
            return (operator @ full)[indices]

        return LinearOperator((indices.size, indices.size), matvec=apply, dtype=operator.dtype)

    @staticmethod
    def norm(value):
        return float(np.max(np.abs(value), initial=0))

    def operator(self, state, inputs, transpose=False):
        from scipy.sparse.linalg import LinearOperator

        zeros = np.zeros_like(inputs)

        def apply(vector):
            if np.iscomplexobj(vector):
                return apply(vector.real) + 1j * apply(vector.imag)
            mapped = (
                self.transpose(state, inputs, vector)[0] if transpose else self.product(state, inputs, vector, zeros)
            )
            return np.asarray(vector) - np.asarray(mapped)

        return LinearOperator((state.size, state.size), matvec=apply, dtype=state.dtype)

    def linear_solve(self, state, inputs, rhs, *, transpose=False, strict=True, linear_tolerance=1e-9):
        from scipy.sparse.linalg import LinearOperator, gmres

        operator = self.operator(state, inputs, transpose)
        indices = self.active_indices(state, inputs)

        def solve(system, right_hand_side, **kwargs):
            if indices is None:
                return gmres(system, right_hand_side, **kwargs)
            reduced = self.restricted_operator(system, indices)
            solution = np.array(right_hand_side, copy=True)
            solution[indices], info = gmres(reduced, right_hand_side[indices], **kwargs)
            return solution, info

        if not np.any(rhs):
            return np.zeros_like(rhs), 0.0, 0
        scale = self.norm(rhs)
        solution, info = solve(operator, rhs / scale, rtol=linear_tolerance, atol=1e-12, restart=120, maxiter=40)
        solution *= scale
        residual = self.norm(operator @ solution - rhs) / scale
        attempts = [dict(method="GMRES", residual=residual, info=int(info))]
        if self.polynomial_product is not None and (
            info != 0 or not np.isfinite(residual) or residual > max(1e-5, 10 * linear_tolerance)
        ):
            transformed_rhs = np.asarray(self.polynomial_product(state, inputs, rhs, transpose=transpose)[1])
            transformed_scale = self.norm(transformed_rhs)
            if np.isfinite(transformed_scale) and transformed_scale > 0:
                transformed = LinearOperator(
                    operator.shape,
                    matvec=lambda v: v - np.asarray(self.polynomial_product(state, inputs, v, transpose=transpose)[0]),
                    dtype=state.dtype,
                )
                candidate, candidate_info = solve(
                    transformed,
                    transformed_rhs / transformed_scale,
                    rtol=min(linear_tolerance, 1e-9),
                    atol=1e-12,
                    restart=120,
                    maxiter=5,
                )
                candidate *= transformed_scale
                candidate_residual = self.norm(operator @ candidate - rhs) / scale
                attempts.append(dict(method="polynomial GMRES", residual=candidate_residual, info=int(candidate_info)))
                if np.isfinite(candidate_residual) and (not np.isfinite(residual) or candidate_residual < residual):
                    solution, residual, info = candidate, candidate_residual, candidate_info
        self.linear_solve_count += 1
        self.last_linear_solve = dict(transpose=transpose, residual=residual, info=int(info), attempts=attempts)
        if indices is not None:
            self.last_linear_solve.update(state_coordinates=state.size, krylov_coordinates=indices.size)
        if strict and (info != 0 or not np.isfinite(residual) or residual > 1e-5):
            raise AdjustmentError(f"Implicit sensitivity solve unresolved: info={info}, residual={residual}")
        return solution, residual, int(info)

    def stability(self, state, inputs):
        from scipy.sparse.linalg import LinearOperator, eigs, ArpackNoConvergence

        zeros = np.zeros_like(inputs)
        calls = 0

        def apply(vector):
            nonlocal calls
            calls += 1
            if np.iscomplexobj(vector):
                return np.asarray(self.product(state, inputs, vector.real, zeros)) + 1j * np.asarray(
                    self.product(state, inputs, vector.imag, zeros)
                )
            return np.asarray(self.product(state, inputs, vector, zeros))

        operator = LinearOperator((state.size, state.size), matvec=apply, dtype=state.dtype)
        requested = state.size if state.size <= 3 else min(4, state.size - 2)
        self.last_stability = dict(
            requested_eigenpairs=requested, arnoldi_vectors=min(120, state.size), max_iterations=1000
        )
        if self.stability_solver is not None:
            provider = self.stability_solver
            self.last_stability = dict(
                requested_eigenpairs=requested,
                method="provided eigensolver",
                provider=getattr(provider, "__qualname__", type(provider).__qualname__),
            )
            values, vectors = provider(operator, k=requested)
            values, vectors = np.asarray(values), np.asarray(vectors)
            if (
                values.ndim != 1
                or values.size < requested
                or vectors.shape != (state.size, values.size)
                or not np.issubdtype(values.dtype, np.number)
                or not np.issubdtype(vectors.dtype, np.number)
                or not np.all(np.isfinite(values))
                or not np.all(np.isfinite(vectors))
            ):
                raise AdjustmentError("Stability solver returned invalid eigenpair arrays")
            errors = []
            for value, vector in zip(values, vectors.T):
                scale = np.max(np.abs(vector), initial=0)
                if not np.isfinite(scale) or scale <= 0:
                    raise AdjustmentError("Stability solver returned an invalid eigenvector")
                vector = vector / scale
                errors.append(float(np.linalg.norm(operator @ vector - value * vector) / np.linalg.norm(vector)))
            self.last_stability.update(
                stopping="returned",
                operator_calls=calls,
                eigenvalues=np.column_stack((np.real(values), np.imag(values))).tolist(),
                eigenpair_residuals=errors,
            )
            error = float(np.max(errors)) if np.all(np.isfinite(errors)) else float("inf")
            return float(np.max(np.abs(values))), error
        indices = self.active_indices(state, inputs)
        reduced = self.restricted_operator(operator, indices)
        size = reduced.shape[0]
        requested = size if size <= 3 else min(4, size - 2)
        if indices is not None:
            self.last_stability.update(
                state_coordinates=state.size,
                krylov_coordinates=size,
                omitted_zero_eigenvalues=state.size - size,
                requested_eigenpairs=requested,
                arnoldi_vectors=min(120, size),
            )
        if size <= 3:
            matrix = np.column_stack([reduced @ v for v in np.eye(size)])
            values, vectors = np.linalg.eig(matrix)
        else:
            try:
                values, vectors = eigs(
                    reduced,
                    k=requested,
                    which="LM",
                    tol=1e-6,
                    maxiter=1000,
                    ncv=min(120, size),
                    v0=np.random.default_rng(19).normal(size=size),
                )
            except ArpackNoConvergence as error:
                self.last_stability.update(
                    stopping="iteration_limit",
                    operator_calls=calls,
                    converged_eigenpairs=error.eigenvalues.size,
                    error=str(error),
                )
                return float("inf"), float("inf")
        index = int(np.argmax(np.abs(values)))
        vector, value = vectors[:, index], values[index]
        if indices is not None:
            full = np.zeros(state.size, dtype=vector.dtype)
            full[indices] = vector
            vector = full
        error = np.linalg.norm(operator @ vector - value * vector) / np.linalg.norm(vector)
        self.last_stability.update(
            stopping="converged",
            operator_calls=calls,
            eigenvalues=np.column_stack((np.real(values), np.imag(values))).tolist(),
        )
        return float(abs(value)), float(error)

    def solve(self, inputs):
        inputs = np.asarray(inputs)
        key = hashlib.sha256(inputs.tobytes()).digest()
        if key in self.cache:
            return self.cache[key]
        initial = np.asarray(self.initial(inputs))
        continuations = [
            result
            for result in self.cache.values()
            if result["converged"] and np.array_equal(result["initial_state"], initial)
        ]
        state = (
            min(continuations, key=lambda result: np.linalg.norm(result["inputs"] - inputs))["state"].copy()
            if continuations
            else initial.copy()
        )
        history, stopping, iterations, newton_solves = [], "iteration_limit", 0, []
        for _ in range(0 if continuations else self.warmup):
            state = np.asarray(self.mapping(state, inputs))
        for iteration in range(self.max_iterations + 1):
            mapped = np.asarray(self.mapping(state, inputs))
            residual = state - mapped
            norm = self.norm(residual)
            history.append(norm)
            if not np.isfinite(norm):
                stopping = "nonfinite_state"
                break
            if norm < self.tolerance * 0.1:
                stopping = "full_state_tolerance"
                break
            if iteration == self.max_iterations:
                break
            step, error, info = self.linear_solve(state, inputs, -residual, strict=False, linear_tolerance=1e-3)
            newton_solves.append(dict(iteration=iteration, residual=error, info=info))
            accepted = False
            if np.isfinite(error) and error < 0.1:
                for power in range(16):
                    candidate = state + 0.5**power * step
                    try:
                        candidate_residual = candidate - np.asarray(self.mapping(candidate, inputs))
                    except RuntimeError:
                        continue
                    if self.norm(candidate_residual) < (1 - 1e-4 * 0.5**power) * norm:
                        state, accepted = candidate, True
                        break
            if not accepted:
                # Forward relaxation remains meaningful on a stable branch.
                state = mapped
            iterations += 1
        converged = stopping == "full_state_tolerance"
        result = dict(
            state=state,
            initial_state=initial,
            inputs=inputs.copy(),
            warm_start=bool(continuations),
            observations=np.asarray(self.observe(state, inputs)),
            residual=norm,
            converged=converged,
            stopping=stopping,
            iterations=iterations,
            residual_history=history,
            newton_solves=newton_solves,
        )
        if len(self.cache) >= 3:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = result
        return result

    def diagnostics(self, inputs):
        result = self.solve(inputs)
        if "spectral_radius" not in result:
            state = result["state"]
            radius, error = self.stability(state, inputs) if result["converged"] else (float("inf"), float("inf"))
            probe = np.sin(np.arange(state.size) + 1)
            _, residual, info = (
                self.linear_solve(state, inputs, probe, strict=False)
                if result["converged"]
                else (None, float("inf"), -1)
            )
            result.update(
                spectral_radius=radius,
                stability_residual=error,
                sensitivity_residual=residual,
                sensitivity_info=info,
                stability_solver=self.last_stability,
            )
        return result

    def _solution(self, inputs):
        result = self.solve(inputs)
        if not result["converged"]:
            raise AdjustmentError(f"Adjusted state unresolved: {result['stopping']}, residual={result['residual']}")
        return result

    def _state(self, inputs):
        return self._solution(inputs)["state"]

    def jvp(self, inputs, direction):
        state = self._state(inputs)
        rhs = np.asarray(self.product(state, inputs, np.zeros_like(state), direction))
        tangent, _, _ = self.linear_solve(state, inputs, rhs)
        return np.asarray(self.observation_product(state, inputs, tangent, direction))

    def vjp(self, inputs, seed):
        state = self._state(inputs)
        state_seed, direct = self.observation_transpose(state, inputs, seed)
        adjoint, _, _ = self.linear_solve(state, inputs, np.asarray(state_seed), transpose=True)
        _, indirect = self.transpose(state, inputs, adjoint)
        return np.asarray(direct) + np.asarray(indirect)

    def transformed(self, example_inputs, observation_shape):
        """Expose first-order JVP/VJP and sequential vmap through host callbacks."""
        import jax
        from jax.custom_derivatives import linear_call
        from .backend import device_callback

        device = (
            next(iter(example_inputs.devices()))
            if isinstance(example_inputs, jax.Array) and not isinstance(example_inputs, jax.core.Tracer)
            else jax.devices()[0]
        )
        key = example_inputs.shape, str(example_inputs.dtype), tuple(observation_shape), device
        if key in self.transformed_functions:
            return self.transformed_functions[key]
        output = jax.ShapeDtypeStruct(observation_shape, example_inputs.dtype)
        input_spec = jax.ShapeDtypeStruct(example_inputs.shape, example_inputs.dtype)
        primal_callback = device_callback(lambda u: self._solution(u)["observations"], device)
        tangent_callback = device_callback(self.jvp, device)
        adjoint_callback = device_callback(self.vjp, device)

        @jax.custom_jvp
        def observations(inputs):
            return jax.pure_callback(primal_callback, output, inputs, vmap_method="sequential")

        @observations.defjvp
        def differentiate(primals, tangents):
            (inputs,) = primals
            (direction,) = tangents

            def forward(inputs, direction):
                return jax.pure_callback(tangent_callback, output, inputs, direction, vmap_method="sequential")

            def reverse(inputs, seed):
                return jax.pure_callback(adjoint_callback, input_spec, inputs, seed, vmap_method="sequential")

            return observations(inputs), linear_call(forward, reverse, inputs, direction)

        self.transformed_functions[key] = observations
        return observations
