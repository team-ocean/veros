"""A forced, dissipative three-dimensional basin for adjusted-response studies."""

from veros import VerosSetup, veros_routine
from veros.variables import Variable
from veros.core.operators import numpy as npx


class AdjustedBasinSetup(VerosSetup):
    @veros_routine
    def set_parameter(self, state):
        state.settings.update(
            nx=8,
            ny=10,
            nz=4,
            dt_tracer=21600.0,
            dt_mom=300.0,
            runlen=0,
            enable_streamfunction=False,
            enable_conserve_energy=True,
            enable_hor_diffusion=True,
            K_h=10.0,
            enable_hor_friction=True,
            A_h=100.0,
            enable_ray_friction=True,
            r_ray=1 / 86400.0,
            enable_implicit_vert_friction=True,
            kappaM_0=1e-2,
            kappaH_0=1e-5,
            enable_neutral_diffusion=True,
            K_iso_0=10.0,
            K_iso_steep=0.0,
            enable_skew_diffusion=True,
            K_gm_0=10.0,
            enable_tempsalt_sources=True,
            eq_of_state_type=5,
        )
        for name in ("basin_temperature_target", "basin_salinity_target"):
            state.var_meta[name] = Variable(name, ("xt", "yt", "zt"), time_dependent=False)

    @veros_routine
    def set_grid(self, state):
        vs, settings = state.variables, state.settings
        vs.dxt = npx.full_like(vs.dxt, 160000 / settings.nx)
        vs.dyt = npx.full_like(vs.dyt, 200000 / settings.ny)
        vs.dzt = npx.full_like(vs.dzt, 2000 / settings.nz)

    @veros_routine
    def set_coriolis(self, state):
        state.variables.coriolis_t = npx.full_like(state.variables.coriolis_t, 1e-5)

    @veros_routine
    def set_topography(self, state):
        state.variables.kbot = state.variables.kbot.at[2:-2, 2:-2].set(1)

    @veros_routine(supports_compilation=True)
    def rebuild_vertical_forcing(self, state):
        vs = state.variables
        meridional = npx.cos(npx.pi * vs.yt[None, :, None] / 200000)
        zonal = npx.sin(2 * npx.pi * vs.xt[:, None, None] / 160000)
        vs.basin_temperature_target = (10 + 0.002 * vs.zt[None, None, :] + 0.15 * meridional + 0.01 * zonal) * vs.maskT
        vs.basin_salinity_target = (35 - 0.005 * meridional) * vs.maskT

    @veros_routine
    def set_initial_conditions(self, state):
        self.rebuild_vertical_forcing(state)
        vs = state.variables
        vs.temp = vs.basin_temperature_target[..., None] * npx.ones(3)
        vs.salt = vs.basin_salinity_target[..., None] * npx.ones(3)

    @veros_routine(supports_compilation=True)
    def set_forcing(self, state):
        vs = state.variables
        seasonal = npx.cos(2 * npx.pi * vs.time / (360 * 86400))
        target = vs.basin_temperature_target + 0.05 * seasonal * npx.cos(npx.pi * vs.yt[None, :, None] / 200000)
        if state.settings.enable_tempsalt_sources:
            vs.temp_source = (target - vs.temp[..., vs.tau]) * vs.maskT / (10 * 86400)
            vs.salt_source = (vs.basin_salinity_target - vs.salt[..., vs.tau]) * vs.maskT / (10 * 86400)
        vs.surface_taux = 0.001 * npx.sin(npx.pi * vs.yt[None, :] / 200000) * vs.maskU[..., -1]
        vs.surface_tauy = npx.zeros_like(vs.surface_tauy)
        vs.forc_temp_surface = npx.zeros_like(vs.forc_temp_surface)
        vs.forc_salt_surface = npx.zeros_like(vs.forc_salt_surface)

    @veros_routine
    def set_diagnostics(self, state):
        for diagnostic in state.diagnostics.values():
            diagnostic.output_frequency = diagnostic.sampling_frequency = None

    @veros_routine(supports_compilation=True)
    def after_timestep(self, state):
        pass
