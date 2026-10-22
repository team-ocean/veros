"""Scaled prognostic coordinates and per-step ocean adjustment products."""

import numpy as np


def canonical(prepared, carry, phase=0):
    import jax.numpy as jnp

    vs = carry.variables
    order = jnp.stack((vs.tau, vs.taum1, vs.taup1))
    changes = {
        name: jnp.take(value, order, axis=meta.dims.index("timesteps"))
        for name, value in vs.items()
        if "timesteps" in ((meta := vs.__metadata__[name]).dims or ())
    }
    changes.update(
        time=jnp.asarray(phase, vs.time.dtype),
        itt=jnp.asarray(3, vs.itt.dtype),
        tau=jnp.asarray(0, vs.tau.dtype),
        taum1=jnp.asarray(1, vs.taum1.dtype),
        taup1=jnp.asarray(2, vs.taup1.dtype),
    )
    return prepared.replace_variables(carry, **changes)


class FloatingCoordinates:
    """Pack dynamic inputs, restoring discrete geometry without differentiating its decisions."""

    def __init__(self, example, *, dynamic_discrete=False):
        import jax

        self.leaves, self.structure = jax.tree.flatten(example)
        self.floating = [dynamic_discrete or np.issubdtype(leaf.dtype, np.floating) for leaf in self.leaves]
        self.leaves = [
            jax.ShapeDtypeStruct(leaf.shape, leaf.dtype) if floating else leaf
            for leaf, floating in zip(self.leaves, self.floating)
        ]

    def pack(self, value):
        import jax
        import jax.numpy as jnp

        return jnp.concatenate(
            [leaf.reshape(-1) for leaf, floating in zip(jax.tree.leaves(value), self.floating) if floating]
        )

    def unpack(self, vector):
        import jax
        import jax.numpy as jnp

        vector = jnp.asarray(vector)
        offset, leaves = 0, []
        for leaf, floating in zip(self.leaves, self.floating):
            if floating:
                leaves.append(vector[offset : offset + leaf.size].reshape(leaf.shape).astype(leaf.dtype))
                offset += leaf.size
            else:
                leaves.append(leaf)
        return jax.tree.unflatten(self.structure, leaves)


class PrognosticCoordinates:
    """Restart prognostic fields and histories, with physical scales and masks."""

    def __init__(self, prepared, example, *, conserved=()):
        self.prepared, self.conserved = prepared, tuple(conserved)
        vs, settings = example.variables, prepared._template.settings
        dt = settings.dt_tracer
        scales = dict(
            temp=1,
            salt=0.1,
            u=0.1,
            v=0.1,
            w=0.01,
            du=0.1 / settings.dt_mom,
            dv=0.1 / settings.dt_mom,
            dtemp=1 / dt,
            dsalt=0.1 / dt,
            dHd=1 / dt,
            tke=1e-3,
            dtke=1e-3 / dt,
            eke=1e-3,
            deke=1e-3 / dt,
            E_iw=1e-3,
            dE_iw=1e-3 / dt,
            K_diss_v=1e-3 / dt,
            psi=1e7 if settings.enable_streamfunction else 4,
            psin=1e7,
            dpsi=1e7 / dt,
            dpsin=1e7 / dt,
        )
        self.fields = []
        self.algebraic_fields = ("rho", "Nsqr", "Hd", "int_drhodT", "int_drhodS")
        for name, value in vs.items():
            meta = vs.__metadata__[name]
            if (
                name in self.algebraic_fields
                or (name == "dHd" and not settings.enable_conserve_energy)
                or (name == "K_diss_v" and not settings.enable_tke)
            ):
                continue
            if (
                name == "time"
                or not meta.time_dependent
                or not meta.write_to_restart
                or not np.issubdtype(value.dtype, np.floating)
            ):
                continue
            mask = meta.get_mask(settings, vs)
            # Partial-cell controls can activate any interior cell. Dry coordinates
            # are projected to zero in pack/unpack using the current input masks.
            mask = (
                np.ones(value.shape, bool)
                if mask is None or settings.enable_partial_cells
                else np.broadcast_to(
                    np.asarray(mask).reshape(np.asarray(mask).shape + (1,) * (value.ndim - np.ndim(mask))), value.shape
                ).copy()
            )
            if len(meta.dims) >= 2 and meta.dims[0] in ("xt", "xu") and meta.dims[1] in ("yt", "yu"):
                mask[:2] = mask[-2:] = False
                mask[:, :2] = mask[:, -2:] = False
            indices = np.flatnonzero(mask.reshape(-1))
            self.fields.append((name, indices, scales.get(name, 1.0)))
        if not self.fields:
            raise ValueError("No prognostic restart fields are available")
        if set(self.conserved) - {"temp", "salt"}:
            raise ValueError("Conserved constraints support temp and salt inventories")
        self.size = sum(indices.size for _, indices, _ in self.fields)

    def constrain(self, carry, reference):
        import jax.numpy as jnp

        vs = carry.variables
        changes = {}
        volume = vs.area_t[..., None] * vs.dzt * vs.maskT
        volume = volume.at[:2].set(0).at[-2:].set(0).at[:, :2].set(0).at[:, -2:].set(0)
        if "hfacT" in vs:
            volume *= vs.hfacT
        for name in self.conserved:
            field = getattr(vs, name)
            target = jnp.sum(getattr(reference.variables, name)[..., 0] * volume) / jnp.sum(volume)
            mean = jnp.sum(field * volume[..., None], axis=(0, 1, 2)) / jnp.sum(volume)
            changes[name] = field + (target - mean) * vs.maskT[..., None]
        if not self.prepared._template.settings.enable_streamfunction:
            weights = vs.area_t * vs.maskT[..., -1]
            weights = weights.at[:2].set(0).at[-2:].set(0).at[:, :2].set(0).at[:, -2:].set(0)
            if self.prepared._template.settings.enable_partial_cells:
                from jax.ops import segment_sum
                from .core.geometry import connected_components

                labels = connected_components(self.prepared._make_state(vs))
                count = labels.size + 1
                area = weights[2:-2, 2:-2].reshape(-1)
                indices = labels.reshape(-1)
                denominator = segment_sum(area, indices, num_segments=count)
                denominator = jnp.where(denominator > 0, denominator, 1)
                mean = (
                    segment_sum(
                        vs.psi[2:-2, 2:-2].reshape((-1, vs.psi.shape[-1])) * area[:, None], indices, num_segments=count
                    )
                    / denominator[:, None]
                )
                target = (
                    segment_sum(reference.variables.psi[2:-2, 2:-2, 0].reshape(-1) * area, indices, num_segments=count)
                    / denominator
                )
                changes["psi"] = vs.psi.at[2:-2, 2:-2].add(
                    (target[labels, None] - mean[labels]) * vs.maskT[2:-2, 2:-2, -1, None]
                )
            else:
                mean = jnp.sum(vs.psi * weights[..., None], axis=(0, 1)) / jnp.sum(weights)
                target = jnp.sum(reference.variables.psi[..., 0] * weights) / jnp.sum(weights)
                changes["psi"] = vs.psi + (target - mean) * vs.maskT[..., -1, None]
        return self.prepared.replace_variables(carry, **changes) if changes else carry

    def pack(self, carry):
        import jax.numpy as jnp

        fields = []
        for name, indices, scale in self.fields:
            field = getattr(carry.variables, name)
            if self.prepared._template.settings.enable_partial_cells:
                mask = carry.variables.__metadata__[name].get_mask(self.prepared._template.settings, carry.variables)
                if mask is not None:
                    field = field * mask.reshape(mask.shape + (1,) * (field.ndim - mask.ndim))
            fields.append(field.reshape(-1)[indices] / scale)
        return jnp.concatenate(fields)

    def active_mask(self, carry):
        """Return exact support of the input-dependent pack/unpack projection."""
        support = []
        for name, indices, _ in self.fields:
            field = getattr(carry.variables, name)
            mask = carry.variables.__metadata__[name].get_mask(self.prepared._template.settings, carry.variables)
            support.append(
                np.ones(indices.size, bool)
                if mask is None
                else np.broadcast_to(
                    np.asarray(mask).reshape(mask.shape + (1,) * (field.ndim - mask.ndim)), field.shape
                ).reshape(-1)[indices]
                != 0
            )
        return np.concatenate(support)

    def unpack(self, vector, reference):
        from veros.core.utilities import enforce_boundaries

        changes, offset = {}, 0
        for name, indices, scale in self.fields:
            field = getattr(reference.variables, name)
            field = (
                field.reshape(-1).at[indices].set(vector[offset : offset + indices.size] * scale).reshape(field.shape)
            )
            meta = reference.variables.__metadata__[name]
            if self.prepared._template.settings.enable_partial_cells:
                mask = meta.get_mask(self.prepared._template.settings, reference.variables)
                if mask is not None:
                    field = field * mask.reshape(mask.shape + (1,) * (field.ndim - mask.ndim))
            if len(meta.dims) >= 2 and meta.dims[0] in ("xt", "xu") and meta.dims[1] in ("yt", "yu"):
                field = enforce_boundaries(field, self.prepared._template.settings.enable_cyclic_x)
            changes[name] = field
            offset += indices.size
        carry = self.prepared.replace_variables(reference, **changes)
        return self.constrain(carry, reference)


class OceanMap:
    def __init__(self, prepared, parameters, observe, *, period_steps=1, phases=(), conserved=(), ram_checkpoints=4):
        import jax
        import jax.numpy as jnp
        from .checkpointing import CheckpointedIntegrator

        self.prepared, self.period_steps, self.phases, self.observe = prepared, period_steps, phases, observe
        self.ram_checkpoints = ram_checkpoints
        example = canonical(prepared, prepared.pack(prepared.simulation.state))
        self.inputs = FloatingCoordinates(
            (example, parameters), dynamic_discrete=prepared._template.settings.enable_partial_cells
        )
        self.state = PrognosticCoordinates(prepared, example, conserved=conserved)
        self.compiled_cycles = period_steps > 1 and self.state.size <= 100000
        self.cycle_product = None
        self.reverse_block = None

        def advance(carry, parameters, steps):
            return jax.lax.fori_loop(
                0, steps, lambda index, current: prepared.step(current, parameters=parameters)[0], carry
            )

        self.cycle_forward = jax.jit(advance)
        self.construct = jax.jit(lambda x, u: self._construct(x, u))
        self.pack_final = jax.jit(
            lambda c, reference: self.state.pack(self.state.constrain(canonical(prepared, c), reference))
        )
        self.step_product = jax.jit(
            lambda c, p, dc, dp: jax.jvp(lambda c, p: prepared.step(c, parameters=p)[0], (c, p), (dc, dp))
        )
        self.equilibrium_product = (
            jax.jit(self._equilibrium_product) if period_steps == 1 and self.state.size <= 100000 else None
        )
        self.equilibrium_transpose = (
            jax.jit(self._equilibrium_transpose) if period_steps == 1 and self.state.size <= 100000 else None
        )
        self.equilibrium_transpose_calls = 0
        self.equilibrium_polynomials = {}
        if self.equilibrium_product is not None:
            for transpose in (False, True):
                self.equilibrium_polynomials[transpose] = jax.jit(
                    lambda x, u, v, transpose=transpose: self._equilibrium_polynomial(x, u, v, transpose=transpose)
                )
        self.observe_product = jax.jit(lambda c, dc: jax.jvp(observe, (c,), (dc,)))
        self.driver = CheckpointedIntegrator(prepared, terminal_objective=lambda c: jnp.asarray(0.0))
        self.checkpoint_statistics = dict(
            evaluations=0,
            hierarchical_evaluations=0,
            forward_steps=0,
            recomputed_steps=0,
            reverse_steps=0,
            peak_ram_checkpoints=0,
            peak_disk_checkpoints=0,
            peak_ram_bytes=0,
            peak_disk_bytes=0,
        )
        self.active_product = None

    def activate(self, product):
        if self.state.size > 100000 and product != self.active_product:
            import jax
            import gc

            jax.clear_caches()
            gc.collect()
        self.active_product = product

    def _construct(self, x, u):
        from veros.core.numerics import calc_initial_conditions_kernel
        from veros.routines import tracing_context

        carry, parameters = self.inputs.unpack(u)
        carry = self.state.unpack(x, carry)
        state = self.prepared._make_state(carry.variables, parameters)
        with tracing_context(), state.variables.unlock():
            state.variables.update(calc_initial_conditions_kernel(state))
        return carry._replace(variables=state.variables), parameters

    def normalize(self, carry, reference):
        return self.state.constrain(canonical(self.prepared, carry), reference)

    def full_cycle(self, x, u):
        initial, parameters = self.construct(x, u)
        final = self.advance(initial, parameters, self.period_steps)
        return initial, final, self.normalize(final, initial), parameters

    def initial(self, u):
        carry, _ = self.inputs.unpack(u)
        return self.state.pack(self.state.constrain(carry, carry))

    def active_state_mask(self, inputs):
        carry, _ = self.inputs.unpack(inputs)
        return self.state.active_mask(carry)

    def mapping(self, x, u):
        initial, parameters = self.construct(x, u)
        carry = initial
        carry = self.advance(carry, parameters, self.period_steps)
        for name in self.state.conserved:
            vs = initial.variables
            volume = np.asarray(vs.area_t[2:-2, 2:-2, None] * vs.dzt * vs.maskT[2:-2, 2:-2])
            if "hfacT" in vs:
                volume = volume * np.asarray(vs.hfacT[2:-2, 2:-2])
            before = np.asarray(getattr(vs, name)[2:-2, 2:-2, :, vs.tau])
            after = np.asarray(getattr(carry.variables, name)[2:-2, 2:-2, :, carry.variables.tau])
            error = float(np.sum((after - before) * volume) / np.sum(volume))
            if abs(error) > 1e-8:
                raise ValueError(f"Declared {name} inventory constraint is violated by the map: {error}")
        return self.pack_final(carry, initial)

    def product(self, x, u, dx, du):
        import jax

        self.activate("tangent")
        if self.equilibrium_product is not None:
            from .integration import IntegrationState

            self.prepared.check_configuration()
            result, code, iteration = self.equilibrium_product(x, u, dx, du)
            self.prepared.check_status(IntegrationState(None, {}, code, iteration))
            return result
        (carry, parameters), (dc, dp) = jax.jvp(self.construct, (x, u), (dx, du))
        carry, parameters = self.construct(x, u)
        initial, initial_tangent = carry, dc
        carry, dc = self.advance_product(carry, parameters, dc, dp, self.period_steps)
        return jax.jvp(self.pack_final, (carry, initial), (dc, initial_tangent))[1]

    def _equilibrium_product(self, x, u, dx, du):
        import jax

        (initial, parameters), (dc, dp) = jax.jvp(self.construct, (x, u), (dx, du))
        initial, parameters = self.construct(x, u)
        _, tangent = self.step_product(initial, parameters, dc, dp)
        final, _ = self.prepared.step(initial, parameters=parameters)
        result = jax.jvp(self.pack_final, (final, initial), (tangent, dc))[1]
        return result, final.failure_code, final.failed_iteration

    def tangent_step(self, carry, parameters, dc, dp):
        _, dc = self.step_product(carry, parameters, dc, dp)
        carry, _ = self.prepared.step(carry, parameters=parameters)
        return carry, dc

    def advance(self, carry, parameters, steps):
        if self.compiled_cycles:
            self.prepared.check_configuration()
            carry = self.cycle_forward(carry, parameters, steps)
        else:
            carry, _ = self.prepared.advance(carry, n_steps=steps, parameters=parameters, final_only=True)
        self.prepared.check_status(carry)
        return carry

    def advance_product(self, carry, parameters, dc, dp, steps):
        import jax

        self.prepared.check_configuration()
        if self.compiled_cycles:
            if self.cycle_product is None:
                self.tangent_coordinates = FloatingCoordinates(dc)

                def advance(current, parameters, tangent, dp, steps):
                    coordinates = self.tangent_coordinates

                    def step(index, values):
                        current, tangent = values
                        current, tangent = self.tangent_step(current, parameters, coordinates.unpack(tangent), dp)
                        return current, coordinates.pack(tangent)

                    return jax.lax.fori_loop(0, steps, step, (current, tangent))

                self.cycle_product = jax.jit(advance)
            carry, tangent = self.cycle_product(carry, parameters, self.tangent_coordinates.pack(dc), dp, steps)
            dc = self.tangent_coordinates.unpack(tangent)
        else:
            for _ in range(steps):
                carry, dc = self.tangent_step(carry, parameters, dc, dp)
                self.prepared.check_status(carry)
        self.prepared.check_status(carry)
        return carry, dc

    def reverse_segment(self, initial, parameters, length, seed):
        from veros import runtime_settings
        from .checkpointing import BinomialSchedule

        slots = (length + 31) // 32 + min(4, length // 8) + 8
        if length == 0:
            import jax

            self.prepared.check_configuration()
            self.prepared.check_status(initial)
            gradient, parameter_gradient = jax.vjp(lambda carry, parameters: carry, initial, parameters)[1](seed)
            measured = {}
        elif (
            self.compiled_cycles
            and runtime_settings.device == "cpu"
            and length > 0
            and length % 8 == 0
            and slots <= self.ram_checkpoints
        ):
            gradient, parameter_gradient, measured = self._hierarchical_reverse(initial, parameters, length, seed)
        else:
            result = self.driver.value_and_grad(
                initial,
                parameters=parameters,
                n_steps=length,
                terminal_cotangent=seed,
                schedule=BinomialSchedule(length, ram_checkpoints=self.ram_checkpoints),
            )
            gradient, parameter_gradient, measured = result.gradient, result.parameter_gradient, result.statistics
        statistics = self.checkpoint_statistics
        statistics["evaluations"] += 1
        for name in ("forward_steps", "recomputed_steps", "reverse_steps", "hierarchical_evaluations"):
            statistics[name] += measured.get(name, 0)
        for name in ("peak_ram_bytes", "peak_disk_bytes", "peak_ram_checkpoints", "peak_disk_checkpoints"):
            statistics[name] = max(statistics[name], measured.get(name, 0))
        return gradient, parameter_gradient

    def _hierarchical_reverse(self, initial, parameters, length, seed):
        """Replay 32-step checkpoints through shared eight-step reverse blocks.

        Reserve eight carry-sized slots for each block's reverse tape. Other
        AD activations, cotangents, and executables add to checkpoint storage.
        """
        import jax
        import jax.numpy as jnp

        if self.reverse_block is None:

            def advance(carry, parameters):
                return jax.lax.scan(
                    lambda current, index: (self.prepared.step(current, parameters=parameters)[0], None),
                    carry,
                    jnp.arange(8),
                )[0]

            self.reverse_block = jax.jit(lambda carry, parameters, seed: jax.vjp(advance, carry, parameters)[1](seed))

        outer, carry = [], initial
        for start in range(0, length, 32):
            steps = min(32, length - start)
            outer.append((carry, steps))
            carry = self.advance(carry, parameters, steps)
        parameter_gradient = jax.tree.map(jnp.zeros_like, parameters)
        for carry, steps in reversed(outer):
            inner = []
            for _ in range(steps // 8):
                inner.append(carry)
                carry = self.advance(carry, parameters, 8)
            for carry in reversed(inner):
                seed, contribution = self.reverse_block(carry, parameters, seed)
                parameter_gradient = jax.tree.map(lambda a, b: a + b, parameter_gradient, contribution)
        slots = len(outer) + min(4, length // 8) + 8
        state_bytes = sum(leaf.size * leaf.dtype.itemsize for leaf in jax.tree.leaves(initial))
        return (
            seed,
            parameter_gradient,
            dict(
                hierarchical_evaluations=1,
                forward_steps=length,
                recomputed_steps=2 * length,
                reverse_steps=length,
                peak_ram_checkpoints=slots,
                peak_ram_bytes=slots * state_bytes,
            ),
        )

    def transpose(self, x, u, seed):
        import jax

        self.activate("adjoint")
        if self.equilibrium_transpose is not None:
            from .integration import IntegrationState

            self.prepared.check_configuration()
            result, code, iteration = self.equilibrium_transpose(x, u, seed)
            self.prepared.check_status(IntegrationState(None, {}, code, iteration))
            self.equilibrium_transpose_calls += 1
            return result
        (initial, parameters), pullback = jax.vjp(self.construct, x, u)
        initial, parameters = self.construct(x, u)
        final = self.advance(initial, parameters, self.period_steps)
        final_seed, reference_seed = jax.vjp(self.pack_final, final, initial)[1](seed)
        dc, dp = self.reverse_segment(initial, parameters, self.period_steps, final_seed)
        dc = jax.tree.map(lambda a, b: a if a.dtype == jax.dtypes.float0 else a + b, dc, reference_seed)
        return pullback((dc, dp))

    def _equilibrium_transpose(self, x, u, seed):
        import jax

        (initial, parameters), pullback = jax.vjp(self.construct, x, u)
        initial, parameters = self.construct(x, u)
        final, _ = self.prepared.step(initial, parameters=parameters)
        final_seed, reference_seed = jax.vjp(self.pack_final, final, initial)[1](seed)
        dc, dp = jax.vjp(lambda c, p: self.prepared.step(c, parameters=p)[0], initial, parameters)[1](final_seed)
        dc = jax.tree.map(lambda a, b: a if a.dtype == jax.dtypes.float0 else a + b, dc, reference_seed)
        return pullback((dc, dp)), final.failure_code, final.failed_iteration

    def _equilibrium_polynomial(self, x, u, vector, *, transpose):
        import jax
        import jax.numpy as jnp

        def step(index, values):
            current, total = values
            if transpose:
                (mapped, _), code, iteration = self.equilibrium_transpose(x, u, current)
            else:
                mapped, code, iteration = self.equilibrium_product(x, u, current, jnp.zeros_like(u))
            return mapped, total + current

        result = jax.lax.fori_loop(0, 16, step, (vector, jnp.zeros_like(vector)))
        initial, parameters = self.construct(x, u)
        final, _ = self.prepared.step(initial, parameters=parameters)
        return result, final.failure_code, final.failed_iteration

    def polynomial_product(self, x, u, vector, *, transpose=False):
        from .integration import IntegrationState

        self.activate("adjoint" if transpose else "tangent")
        self.prepared.check_configuration()
        result, code, iteration = self.equilibrium_polynomials[transpose](x, u, vector)
        self.prepared.check_status(IntegrationState(None, {}, code, iteration))
        return result

    def observations(self, x, u):
        import jax.numpy as jnp

        _, _, carry, parameters = self.full_cycle(x, u)
        if not self.phases:
            return self.observe(carry)
        values, previous = [], 0
        for phase in self.phases:
            boundary = round(phase * self.period_steps)
            carry = self.advance(carry, parameters, boundary - previous)
            values.append(self.observe(carry))
            previous = boundary
        return jnp.stack(values)

    def observation_product(self, x, u, dx, du):
        import jax
        import jax.numpy as jnp

        self.activate("tangent")
        (carry, parameters), (dc, dp) = jax.jvp(self.construct, (x, u), (dx, du))
        carry, parameters = self.construct(x, u)
        initial, initial_tangent = carry, dc
        carry, dc = self.advance_product(carry, parameters, dc, dp, self.period_steps)
        _, dc = jax.jvp(self.normalize, (carry, initial), (dc, initial_tangent))
        carry = self.normalize(carry, initial)
        if not self.phases:
            return self.observe_product(carry, dc)[1]
        values, previous = [], 0
        for phase in self.phases:
            boundary = round(phase * self.period_steps)
            carry, dc = self.advance_product(carry, parameters, dc, dp, boundary - previous)
            values.append(self.observe_product(carry, dc)[1])
            previous = boundary
        return jnp.stack(values)

    def observation_transpose(self, x, u, seed):
        import jax
        import jax.numpy as jnp

        self.activate("adjoint")
        (initial, parameters), pullback = jax.vjp(self.construct, x, u)
        initial, parameters = self.construct(x, u)
        final = self.advance(initial, parameters, self.period_steps)
        _, normalize_pullback = jax.vjp(self.normalize, final, initial)
        normalized = self.normalize(final, initial)
        if not self.phases:
            adjoint = jax.vjp(self.observe, normalized)[1](seed)[0]
            parameter_gradient = jax.tree.map(jnp.zeros_like, parameters)
        else:
            starts, finals, lengths, carry, previous = [], [], [], normalized, 0
            for phase in self.phases:
                boundary = round(phase * self.period_steps)
                starts.append(carry)
                carry = self.advance(carry, parameters, boundary - previous)
                finals.append(carry)
                lengths.append(boundary - previous)
                previous = boundary
            adjoint = jax.tree.map(
                lambda value: (
                    np.zeros(value.shape, dtype=jax.dtypes.float0)
                    if not jnp.issubdtype(value.dtype, jnp.inexact)
                    else jnp.zeros_like(value)
                ),
                carry,
            )
            parameter_gradient = jax.tree.map(jnp.zeros_like, parameters)
            for index in reversed(range(len(self.phases))):
                observation_seed = jax.vjp(self.observe, finals[index])[1](seed[index])[0]
                adjoint = jax.tree.map(
                    lambda a, b: a if a.dtype == jax.dtypes.float0 else a + b, adjoint, observation_seed
                )
                adjoint, dp = self.reverse_segment(starts[index], parameters, lengths[index], adjoint)
                parameter_gradient = jax.tree.map(lambda a, b: a + b, parameter_gradient, dp)
        final_seed, reference_seed = normalize_pullback(adjoint)
        adjoint, dp = self.reverse_segment(initial, parameters, self.period_steps, final_seed)
        adjoint = jax.tree.map(lambda a, b: a if a.dtype == jax.dtypes.float0 else a + b, adjoint, reference_seed)
        parameter_gradient = jax.tree.map(lambda a, b: a + b, parameter_gradient, dp)
        return pullback((adjoint, parameter_gradient))


def response(
    prepared,
    carry,
    *,
    parameters,
    observe,
    tolerance,
    max_iterations,
    period_steps=1,
    phases=(),
    conserved=(),
    warmup=4,
    diagnostics=True,
    ram_checkpoints=4,
    stability_solver=None,
):
    import jax
    import operator
    from veros import runtime_state
    from .adjustment import ImplicitResponse
    from .integration import CompiledCompatibilityError

    if runtime_state.proc_num != 1:
        raise CompiledCompatibilityError("Implicit ocean responses require serial compiled execution")
    prepared.check_configuration()
    ram_checkpoints = operator.index(ram_checkpoints)
    if ram_checkpoints < 0:
        raise ValueError("RAM checkpoint count must be nonnegative")
    parameters = prepared.validate_parameters(parameters)
    if any(diag.sampling_frequency or diag.output_frequency for diag in prepared._diagnostics.values()):
        raise ValueError("Adjusted responses require disabled diagnostic accumulators")
    if period_steps < 1 or tuple(sorted(set(phases))) != tuple(phases) or any(p < 0 or p >= 1 for p in phases):
        raise ValueError("Period steps must be positive and phases increasing in [0, 1)")
    mapping_key = (
        tuple((name, np.shape(value)) for name, value in parameters.items()),
        observe,
        period_steps,
        phases,
        tuple(conserved),
        ram_checkpoints,
    )
    key = mapping_key, tolerance, max_iterations, warmup, id(stability_solver)
    if not hasattr(prepared, "_adjusted_responses"):
        prepared._adjusted_responses = {}
    if key not in prepared._adjusted_responses:
        ocean = next(
            (adapter for existing, (adapter, _) in prepared._adjusted_responses.items() if existing[0] == mapping_key),
            None,
        )
        if ocean is None:
            ocean = OceanMap(
                prepared,
                parameters,
                observe,
                period_steps=period_steps,
                phases=phases,
                conserved=conserved,
                ram_checkpoints=ram_checkpoints,
            )
        engine = ImplicitResponse(
            ocean.initial,
            ocean.mapping,
            ocean.product,
            ocean.transpose,
            ocean.observations,
            ocean.observation_product,
            ocean.observation_transpose,
            tolerance=tolerance,
            max_iterations=max_iterations,
            warmup=warmup,
            polynomial_product=ocean.polynomial_product if ocean.equilibrium_polynomials else None,
            stability_solver=stability_solver,
            state_mask=ocean.active_state_mask if prepared._template.settings.enable_partial_cells else None,
        )
        prepared._adjusted_responses[key] = (ocean, engine)
    ocean, engine = prepared._adjusted_responses[key]
    inputs = ocean.inputs.pack((canonical(prepared, carry), parameters))
    shape = (len(phases),) + jax.eval_shape(observe, carry).shape if phases else jax.eval_shape(observe, carry).shape
    from .backend import device_callback

    device = (
        next(iter(inputs.devices()))
        if isinstance(inputs, jax.Array) and not isinstance(inputs, jax.core.Tracer)
        else jax.devices()[0]
    )
    metrics = ("residual", "spectral_radius", "stability_residual", "sensitivity_residual")
    direct_gpu = device.platform == "gpu" and not isinstance(inputs, jax.core.Tracer)
    if direct_gpu:
        with jax.default_device(device):
            solution = engine._solution(inputs)
            observations = jax.numpy.asarray(solution["observations"], dtype=inputs.dtype)
            if diagnostics:
                solution = engine.diagnostics(inputs)
                diagnostic_values = jax.numpy.asarray([solution[name] for name in metrics], dtype=inputs.dtype)
    else:
        observations = engine.transformed(inputs, shape)(inputs)
    if not diagnostics:
        return dict(observations=observations)
    if not direct_gpu:
        diagnostic_values = jax.pure_callback(
            device_callback(
                lambda u: np.asarray([engine.diagnostics(u)[name] for name in metrics], dtype=u.dtype), device
            ),
            jax.ShapeDtypeStruct((len(metrics),), inputs.dtype),
            jax.lax.stop_gradient(inputs),
            vmap_method="sequential",
        )
    return dict(
        observations=observations,
        residual_kind="full_state",
        state_tolerance=tolerance,
        ram_checkpoints=ram_checkpoints,
        cycle_execution="compiled" if ocean.compiled_cycles else "per-step",
        **dict(zip(metrics, diagnostic_values)),
        state_fields=[name for name, _, _ in ocean.state.fields],
        algebraic_fields=list(ocean.state.algebraic_fields),
        physical_scales={name: scale for name, _, scale in ocean.state.fields},
        conserved_inventories=list(conserved),
        phase_seconds=0,
        pressure_gauge=(
            "native streamfunction boundaries"
            if prepared._template.settings.enable_streamfunction
            else (
                "fixed initial wet-area mean sea level in each connected component"
                if prepared._template.settings.enable_partial_cells
                else "fixed initial wet-area mean sea level"
            )
        ),
        method="implicit fixed point" if not phases else "implicit Poincare map",
    )
