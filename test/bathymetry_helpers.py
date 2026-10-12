"""Small basin setups for partial-cell tests."""


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
