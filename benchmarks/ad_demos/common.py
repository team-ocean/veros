"""Control maps, physical observations, and checks shared by the experiments."""

import numpy as np

CP = 3991.86795711963


class Checks:
    def __init__(self, on_update=None):
        self.items = []
        self.metrics = {}
        self.on_update = on_update

    def require(self, name, passed, detail):
        self.items.append(dict(name=name, passed=bool(passed), detail=detail))
        if self.on_update is not None:
            self.on_update()

    def close(self, name, actual, expected, *, rtol=1e-5, atol=1e-9):
        actual, expected = np.asarray(actual), np.asarray(expected)
        self.require(
            name,
            np.all(np.isfinite(actual)) and np.allclose(actual, expected, rtol=rtol, atol=atol),
            dict(actual=actual.tolist(), expected=expected.tolist(), rtol=rtol, atol=atol),
        )


def gradient_checks(function, point, checks, *, direction=None, coarser_scales=(), finer_scales=()):
    import jax.numpy as jnp
    from .finite_response import value_and_grad, jvp

    if direction is None:
        direction = jnp.arange(1, point.size + 1, dtype=point.dtype)
        direction = direction / jnp.linalg.norm(direction)
    evaluate = function
    value, gradient = value_and_grad(function)(point)
    forward_value, tangent = jvp(function, point, direction)
    contraction = jnp.vdot(gradient, direction)
    checks.close("forward/reverse primal", forward_value, value, rtol=1e-10)
    checks.close("forward/reverse directional derivative", tangent, contraction)
    checks.require("finite gradient", np.isfinite(gradient).all(), np.asarray(gradient).tolist())
    differences, remainders = [], []
    epsilons = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
    audit = dict(
        point=np.asarray(point).tolist(),
        direction=np.asarray(direction).tolist(),
        value=float(value),
        derivative=float(contraction),
        samples=[],
    )
    checks.metrics.setdefault("gradient_audits", []).append(audit)
    if hasattr(function, "gradient_method"):
        audit["gradient_method"] = function.gradient_method

    def measure(scales):
        for epsilon in scales:
            plus, minus = evaluate(point + epsilon * direction), evaluate(point - epsilon * direction)
            differences.append(float((plus - minus) / (2 * epsilon)))
            remainders.append(float(jnp.abs(plus - value - epsilon * contraction)))
            audit["samples"].append(
                dict(epsilon=epsilon, plus=float(plus), minus=float(minus), central_difference=differences[-1])
            )
            if checks.on_update is not None:
                checks.on_update()

    measure(epsilons)
    initial_matches = sum(np.isclose(differences, float(contraction), rtol=2e-3, atol=1e-8))

    def taylor_orders():
        return [
            float(np.log(a / b) / np.log(ea / eb)) if a > 1e-13 and b > 1e-13 else None
            for a, b, ea, eb in zip(remainders[:-1], remainders[1:], epsilons[:-1], epsilons[1:])
        ]

    def taylor_passed(orders):
        return any(order is not None and 1.7 < order < 2.3 for order in orders) or max(remainders) <= 1e-9 * max(
            1, abs(float(value))
        )

    if initial_matches < 2 or not taylor_passed(taylor_orders()):
        smaller = [3e-5, 1e-5, 3e-6, 1e-6]
        measure(smaller)
        epsilons.extend(smaller)
    if finer_scales and (
        sum(np.isclose(differences, float(contraction), rtol=2e-3, atol=1e-8)) < 2 or not taylor_passed(taylor_orders())
    ):
        measure(finer_scales)
        epsilons.extend(finer_scales)
    if coarser_scales and not taylor_passed(taylor_orders()):
        measure(coarser_scales)
        epsilons.extend(coarser_scales)
    checks.require(
        "finite-difference plateau",
        sum(np.isclose(differences, float(contraction), rtol=2e-3, atol=1e-8)) >= 2,
        dict(
            epsilons=epsilons,
            derivatives=differences,
            ad=float(contraction),
            original_scale_matches=int(initial_matches),
        ),
    )
    orders = taylor_orders()
    checks.require(
        "Taylor remainder",
        taylor_passed(orders),
        dict(
            epsilons=epsilons,
            remainders=remainders,
            orders=orders,
            unresolved_tolerance=1e-9 * max(1, abs(float(value))),
        ),
    )
    return float(value), gradient


def descend(function, point, *, iterations=8, radius=None, diagnostics=None, on_iteration=None, value_and_grad_fn=None):
    """Small projected gradient descent with backtracking and recorded losses."""
    import jax.numpy as jnp
    from .finite_response import value_and_grad

    backward = value_and_grad(function) if value_and_grad_fn is None else value_and_grad_fn
    evaluate = function
    history = []
    stopping = "iteration_limit"
    accepted_steps = []

    def report(stage, iteration, loss, **extra):
        if on_iteration is not None:
            on_iteration(
                dict(
                    stage=stage,
                    iteration=iteration,
                    controls=np.asarray(point).tolist(),
                    loss=float(loss),
                    history=list(history),
                    accepted_steps=list(accepted_steps),
                    **extra,
                )
            )

    for iteration in range(iterations):
        value, gradient = backward(point)
        history.append(float(value))
        norm = float(jnp.linalg.norm(gradient))
        report("gradient", iteration, value, gradient=np.asarray(gradient).tolist(), gradient_norm=norm)
        if not np.isfinite(norm) or norm < 1e-12:
            stopping = "nonfinite_gradient" if not np.isfinite(norm) else "gradient_tolerance"
            break
        accepted = False
        for step in (1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001):
            candidate = point - step * gradient / norm
            if radius is not None:
                candidate *= jnp.minimum(1.0, radius / jnp.maximum(jnp.linalg.norm(candidate), 1e-30))
            report("trial", iteration, value, candidate=np.asarray(candidate).tolist(), step=step)
            trial = float(evaluate(candidate))
            if np.isfinite(trial) and trial < float(value):
                point, accepted = candidate, True
                accepted_steps.append(step)
                report("accepted", iteration + 1, trial)
                break
        if not accepted:
            stopping = "backtracking_exhausted"
            break
    history.append(float(evaluate(point)))
    if diagnostics is not None:
        _, final_gradient = backward(point)
        projected = point - final_gradient
        if radius is not None:
            projected *= jnp.minimum(1.0, radius / jnp.maximum(jnp.linalg.norm(projected), 1e-30))
        diagnostics.update(
            method="normalized gradient descent with finite backtracking",
            stopping=stopping,
            max_iterations=iterations,
            accepted_iterations=len(accepted_steps),
            accepted_steps=accepted_steps,
            gradient_norm=float(jnp.linalg.norm(final_gradient)),
            projected_gradient_norm=float(jnp.linalg.norm(point - projected)),
            gradient_tolerance=1e-12,
            radius=radius,
            control_norm=float(jnp.linalg.norm(point)),
        )
    report("complete", len(history) - 1, history[-1], stopping=stopping)
    return point, history


class Ocean:
    """Fixed-geometry experiment with additive surface heat forcing in W/m²."""

    def __init__(self, config):
        import jax.numpy as jnp
        from veros import veros_routine
        from veros.variables import Variable

        if config["setup"] == "acc":
            from veros.setups.acc_basic import ACCBasicSetup as Setup

            overrides = dict(nx=8, ny=8, dt_tracer=600.0, dt_mom=600.0)
        else:
            from veros.setups.global_4deg import GlobalFourDegreeSetup as Setup

            overrides = {}
        if config["physics"] == "fixed":
            overrides.update(enable_tke=False, enable_eke=False, enable_idemix=False, kappaH_0=1e-5)

        class Simulation(Setup):
            @veros_routine
            def set_parameter(self, state):
                super().set_parameter(state)
                state.var_meta["ad_heat_flux"] = Variable("Added surface heat flux", ("xt", "yt"), "W/m^2")
                state.var_meta["ad_flux_integral"] = Variable("Integrated added heat input", None, "J")

            @veros_routine(supports_compilation=True)
            def set_forcing(self, state):
                super().set_forcing(state)
                vs = state.variables
                flux = vs.ad_heat_flux * vs.maskT[:, :, -1]
                if config["setup"] == "global_4deg":
                    blocked = (vs.temp[:, :, -1, vs.tau] < -1.8) & (
                        vs.forc_temp_surface + flux / (CP * state.settings.rho_0) < 0
                    )
                    flux = jnp.where(blocked, 0, flux)
                vs.forc_temp_surface = vs.forc_temp_surface + flux / (CP * state.settings.rho_0)
                vs.ad_flux_integral = vs.ad_flux_integral + state.settings.dt_tracer * jnp.sum(
                    flux[2:-2, 2:-2] * vs.area_t[2:-2, 2:-2]
                )

            @veros_routine
            def set_diagnostics(self, state):
                for diagnostic in state.diagnostics.values():
                    diagnostic.output_frequency = diagnostic.sampling_frequency = None

        self.sim = Simulation(override=overrides)
        self.sim.setup()
        self.prepared = self.sim.prepare_integrator(record_diagnostics=(), checkpoint=True)
        self.initial = self.prepared.pack(self.sim.state)
        if config["spinup_steps"]:
            self.initial, _ = self.prepared.advance(self.initial, n_steps=config["spinup_steps"], final_only=True)
            self.prepared.check_status(self.initial)
        vs = self.initial.variables
        self.dt = self.sim.state.settings.dt_tracer
        self.area = vs.area_t[2:-2, 2:-2] * vs.maskT[2:-2, 2:-2, -1]
        self.volume = vs.area_t[2:-2, 2:-2, None] * vs.dzt[None, None, :] * vs.maskT[2:-2, 2:-2]
        latitude = vs.yt[2:-2]
        split = (jnp.min(latitude) + jnp.max(latitude)) / 2
        self.region = jnp.broadcast_to(latitude[None, :] > split, self.area.shape)
        coordinate = (latitude - jnp.min(latitude)) / (jnp.max(latitude) - jnp.min(latitude))
        patterns = jnp.stack(
            [jnp.broadcast_to(jnp.cos(k * jnp.pi * coordinate)[None, :], self.area.shape) for k in (1, 2)]
        )
        patterns -= jnp.sum(patterns * self.area, axis=(1, 2))[:, None, None] / jnp.sum(self.area)
        patterns *= vs.maskT[2:-2, 2:-2, -1]
        patterns /= jnp.sqrt(jnp.sum(patterns**2 * self.area, axis=(1, 2)) / jnp.sum(self.area))[:, None, None]
        self.patterns = patterns

    def controls(self, z):
        import jax.numpy as jnp
        from veros.core.numerics import calc_initial_conditions_kernel
        from veros.core.utilities import enforce_boundaries
        from veros.routines import tracing_context

        vs = self.initial.variables
        flux = jnp.zeros_like(vs.ad_heat_flux).at[2:-2, 2:-2].set(20 * jnp.einsum("k,kij->ij", z[1:], self.patterns))
        vertical = jnp.exp(vs.zt / 300.0)
        shift = jnp.zeros_like(vs.maskT, dtype=vs.temp.dtype)
        shift = shift.at[2:-2, 2:-2].set(self.patterns[0, :, :, None] * vertical * vs.maskT[2:-2, 2:-2])
        flux = enforce_boundaries(flux, self.sim.state.settings.enable_cyclic_x)
        temperature = enforce_boundaries(vs.temp + z[0] * shift[..., None], self.sim.state.settings.enable_cyclic_x)
        carry = self.prepared.replace_variables(
            self.initial,
            temp=temperature,
            ad_heat_flux=flux,
            ad_flux_integral=jnp.zeros_like(vs.ad_flux_integral),
        )
        state = self.prepared._make_state(carry.variables)
        with tracing_context(), state.variables.unlock():
            state.variables.update(calc_initial_conditions_kernel(state))
        return carry._replace(variables=state.variables)

    def advance(self, z, steps, *, parameters=None):
        return self.prepared.advance(self.controls(z), n_steps=steps, parameters=parameters, final_only=True)[0]

    def observe(self, final):
        """Regional T/S profiles plus upper-1000-m meridional transport in Sv."""
        import jax.numpy as jnp

        vs = final.variables
        profiles = []
        for region in (self.region, ~self.region):
            weight = self.volume * region[:, :, None]
            denominator = jnp.maximum(jnp.sum(weight, axis=(0, 1)), 1)
            for tracer in (vs.temp, vs.salt):
                profiles.append(jnp.sum(tracer[2:-2, 2:-2, :, vs.tau] * weight, axis=(0, 1)) / denominator)
        section = vs.v.shape[1] // 2
        upper_fraction = jnp.clip((vs.zt + 0.5 * vs.dzt + 1000) / vs.dzt, 0, 1)
        transport = (
            jnp.sum(
                vs.v[2:-2, section, :, vs.tau]
                * vs.dxt[2:-2, None]
                * vs.cosu[section]
                * vs.dzt[None, :]
                * vs.maskV[2:-2, section, :]
                * upper_fraction[None, :]
            )
            / 1e6
        )
        return jnp.concatenate((*profiles, jnp.atleast_1d(transport)))

    def observations(self, z, steps, *, parameters=None):
        import jax.numpy as jnp

        carry = self.controls(z)
        outputs, previous = [], 0
        for boundary in sorted({max(1, steps // 3), max(1, 2 * steps // 3), steps}):
            carry, _ = self.prepared.advance(carry, n_steps=boundary - previous, parameters=parameters, final_only=True)
            outputs.append(self.observe(carry))
            previous = boundary
        return jnp.stack(outputs), carry

    def heat(self, final):
        vs = final.variables
        return (
            CP
            * self.sim.state.settings.rho_0
            * (vs.temp[2:-2, 2:-2, :, vs.tau] * self.volume * self.region[:, :, None]).sum()
        )

    def metadata(self, steps):
        return dict(
            shape=[self.sim.state.settings.nx, self.sim.state.settings.ny, self.sim.state.settings.nz],
            dt_tracer=self.dt,
            dt_mom=self.sim.state.settings.dt_mom,
            steps=steps,
            simulated_days=steps * self.dt / 86400,
            initial_time_seconds=float(self.initial.variables.time),
            initial_condition="finite spinup" if float(self.initial.variables.time) else "setup initial state",
        )
