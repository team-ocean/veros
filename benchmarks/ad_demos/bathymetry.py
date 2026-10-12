"""Recover a continuous sill-depth perturbation in a three-dimensional basin."""

import tempfile

from .common import descend, gradient_checks


def make_simulation(*, partial_cells=True, wind=0.05, bottom_indices=None, cyclic_x=False, **overrides):
    from veros import VerosSetup, veros_routine
    from veros.core.operators import numpy as npx, update, at

    class Basin(VerosSetup):
        @veros_routine
        def set_parameter(self, state):
            settings = state.settings
            settings.identifier = "bathymetry_basin"
            settings.nx, settings.ny, settings.nz = 8, 10, 4
            settings.dt_mom = settings.dt_tracer = 300.0
            settings.runlen = 96 * settings.dt_tracer
            settings.enable_streamfunction = False
            settings.enable_partial_cells = partial_cells
            settings.enable_cyclic_x = cyclic_x
            settings.enable_conserve_energy = False
            settings.enable_implicit_vert_friction = True
            settings.enable_hor_diffusion = True
            settings.K_h = 10.0
            settings.kappaH_0 = 1e-4
            settings.kappaM_0 = 1e-3
            settings.eq_of_state_type = 1

        @veros_routine
        def set_grid(self, state):
            vs = state.variables
            vs.dxt = update(vs.dxt, at[:], 20_000.0)
            vs.dyt = update(vs.dyt, at[:], 20_000.0)
            vs.dzt = update(vs.dzt, at[:], 100.0)

        @veros_routine
        def set_coriolis(self, state):
            pass

        @veros_routine
        def set_topography(self, state):
            vs = state.variables
            indices = 1 if bottom_indices is None else npx.asarray(bottom_indices, dtype=vs.kbot.dtype)
            vs.kbot = update(vs.kbot, at[2:-2, 2:-2], indices)

        @veros_routine
        def set_initial_conditions(self, state):
            vs = state.variables
            vs.temp = update(vs.temp, at[:], (10 + 0.002 * vs.zt[None, None, :, None]) * vs.maskT[..., None])
            vs.salt = update(vs.salt, at[:], 35 * vs.maskT[..., None])

        @veros_routine(supports_compilation=True)
        def set_forcing(self, state):
            vs = state.variables
            x = npx.arange(state.settings.nx)[:, None]
            y = npx.arange(state.settings.ny)[None, :]
            vs.surface_tauy = update(vs.surface_tauy, at[2:-2, 2:-2], wind * (1 + 0.1 * npx.sin(x + y)))

        @veros_routine
        def set_diagnostics(self, state):
            for diagnostic in state.diagnostics.values():
                diagnostic.output_frequency = diagnostic.sampling_frequency = None

        @veros_routine(supports_compilation=True)
        def after_timestep(self, state):
            pass

    simulation = Basin(override=overrides)
    simulation.setup()
    return simulation


class BathymetryBasin:
    def __init__(self):
        import jax.numpy as jnp

        self.sim = make_simulation()
        self.prepared = self.sim.prepare_integrator(record_diagnostics=())
        self.initial = self.prepared.pack(self.sim.state)
        x = jnp.linspace(-1, 1, self.sim.state.settings.nx)[:, None]
        y = jnp.linspace(-1, 1, self.sim.state.settings.ny)[None, :]
        self.sill = jnp.exp(-((y / 0.35) ** 2)) * (1 + 0.1 * x)
        self.reference = 380 - 40 * self.sill + 3 * x + 2 * y

    def controls(self, z):
        return self.prepared.with_bathymetry(self.initial, self.reference + 20 * z[0] * self.sill)

    @staticmethod
    def observe(carry):
        import jax.numpy as jnp

        vs = carry.variables
        # Fixed sensor locations and units avoid direct volume-weight dependence.
        return jnp.concatenate((vs.psi[3:-3, 3:-3, vs.tau].ravel() / 9.81, vs.v[3:-3, 3:-3, -1, vs.tau].ravel()))

    def response(self, z, steps):
        carry = self.controls(z)
        carry, _ = self.prepared.advance(carry, n_steps=steps)
        return self.observe(carry), carry


def run(name, config, checks):
    import jax
    import jax.numpy as jnp
    from veros.checkpointing import BinomialSchedule, CheckpointedIntegrator

    basin = BathymetryBasin()
    steps = config["steps"] or (6 if config["mode"] == "smoke" else 96)
    checks.metrics.update(
        experiment_scope="8 x 10 x 4 Cartesian basin, fixed wet cells, constant mixing, surface pressure",
        observation_units=["sea surface height (m)", "surface meridional velocity (m/s)"],
        steps=steps,
        dt_seconds=basin.sim.state.settings.dt_tracer,
        initialization="fixed tracer concentrations and zero velocity histories",
    )
    if config["mode"] == "acceptance":
        checks.require("minimum eight-hour trajectory", steps >= 96, steps)
    if config["spinup_steps"]:
        raise ValueError("The bathymetry basin starts from rest; spinup-steps must be zero")
    zero, truth = jnp.zeros(1), jnp.array([0.6])
    baseline, final = basin.response(zero, steps)
    basin.prepared.check_status(final)
    target, final = basin.response(truth, steps)
    basin.prepared.check_status(final)
    # One response scale per observation type avoids amplifying nearly unchanged sensors.
    count = target.size // 2
    scale = jnp.concatenate(
        [
            jnp.full(count, jnp.maximum(jnp.max(jnp.abs(delta)), 1e-8))
            for delta in (target[:count] - baseline[:count], target[count:] - baseline[count:])
        ]
    )

    def terminal(carry):
        return jnp.mean(((basin.observe(carry) - target) / scale) ** 2)

    def loss(z):
        _, final = basin.response(z, steps)
        return jnp.where(final.failure_code == 0, terminal(final), jnp.nan)

    initial_loss, gradient = gradient_checks(loss, zero, checks)
    checks.require("nonzero depth sensitivity", float(jnp.linalg.norm(gradient)) > 1e-5, gradient.tolist())
    optimized, history = descend(loss, zero, iterations=20, radius=1)
    checks.require("training misfit reduced by 90%", history[-1] < 0.1 * initial_loss, history)
    checks.require(
        "sill depth recovered within one meter", float(jnp.abs(optimized[0] - truth[0]) * 20) < 1, optimized.tolist()
    )
    long_target, long_final = basin.response(truth, 2 * steps)
    basin.prepared.check_status(long_final)
    long_base, _ = basin.response(zero, 2 * steps)
    long_fit, long_final = basin.response(optimized, 2 * steps)
    basin.prepared.check_status(long_final)
    held_scale = jnp.concatenate(
        [
            jnp.full(count, jnp.maximum(jnp.max(jnp.abs(delta)), 1e-8))
            for delta in (long_target[:count] - long_base[:count], long_target[count:] - long_base[count:])
        ]
    )
    before = float(jnp.mean(((long_base - long_target) / held_scale) ** 2))
    after = float(jnp.mean(((long_fit - long_target) / held_scale) ** 2))
    checks.require("double-length held-out trajectory improves by 50%", after < 0.5 * before, [before, after])

    carry, pullback = jax.vjp(basin.controls, zero)
    for disk in (False, True):
        driver = CheckpointedIntegrator(basin.prepared, terminal_objective=terminal)
        schedule = BinomialSchedule(steps, ram_checkpoints=1, disk_checkpoints=int(disk))
        with tempfile.TemporaryDirectory(prefix="veros-bathymetry-") as directory:
            result = driver.value_and_grad(carry, n_steps=steps, schedule=schedule, directory=directory)
        replay_gradient = pullback(result.gradient)[0]
        label = "RAM/disk" if disk else "RAM"
        checks.close(f"{label} replay loss", result.value, initial_loss)
        checks.close(f"{label} replay bathymetry gradient", replay_gradient, gradient)
    fitted = basin.controls(optimized)
    lower, upper = basin.prepared.bathymetry_bounds(fitted)
    depth = fitted.variables.ht[2:-2, 2:-2]
    checks.require(
        "depth stays within topology bounds",
        bool(jnp.all((depth >= lower) & (depth <= upper))),
        dict(minimum_depth_m=float(jnp.min(depth)), maximum_depth_m=float(jnp.max(depth))),
    )
    checks.metrics.update(
        truth_control=truth.tolist(),
        fitted_control=optimized.tolist(),
        loss_history=history,
        maximum_depth_error_m=float(jnp.max(jnp.abs((optimized[0] - truth[0]) * 20 * basin.sill))),
        held_out_loss=after,
        reference_depth_m=basin.reference.tolist(),
        fitted_depth_m=depth.tolist(),
    )
