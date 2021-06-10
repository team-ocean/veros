from benchmark_base import benchmark_cli

from veros.pyom_compat import load_pyom, pyom_from_state, run_pyom


@benchmark_cli
def main(pyom2_lib, timesteps, size):
    from veros import veros_routine
    from veros.setups.acc import ACCSetup
    from veros.variables import Variable, allocate
    from veros.distributed import global_min, global_max
    from veros.core.operators import update, at, numpy as npx

    class ACC2Benchmark(ACCSetup):
        @veros_routine
        def set_parameter(self, state):
            settings = state.settings
            settings.identifier = "acc"
            settings.restart_output_filename = None

            settings.nx, settings.ny, settings.nz = size
            settings.dt_mom = 600
            settings.dt_tracer = 600
            settings.runlen = settings.dt_tracer * timesteps

            settings.x_origin = 0.0
            settings.y_origin = -40.0

            settings.coord_degree = True
            settings.enable_cyclic_x = True

            settings.enable_neutral_diffusion = True
            settings.K_iso_0 = 1000.0
            settings.K_iso_steep = 500.0
            settings.iso_dslope = 0.005
            settings.iso_slopec = 0.01
            settings.enable_skew_diffusion = True

            settings.enable_hor_friction = True
            settings.A_h = (2 * settings.degtom) ** 3 * 2e-11
            settings.enable_hor_friction_cos_scaling = True
            settings.hor_friction_cosPower = 1

            settings.enable_bottom_friction = True
            settings.r_bot = 1e-5

            settings.enable_implicit_vert_friction = True

            settings.enable_tke = True
            settings.c_k = 0.1
            settings.c_eps = 0.7
            settings.alpha_tke = 30.0
            settings.mxl_min = 1e-8
            settings.tke_mxl_choice = 2
            settings.kappaM_min = 2e-4
            settings.kappaH_min = 0.0
            settings.enable_kappaH_profile = False
            settings.enable_Prandtl_tke = False

            settings.K_gm_0 = 1000.0
            settings.enable_eke = True
            settings.eke_k_max = 1e4
            settings.eke_c_k = 0.4
            settings.eke_c_eps = 0.5
            settings.eke_cross = 2.0
            settings.eke_crhin = 1.0
            settings.eke_lmin = 100.0
            settings.enable_eke_superbee_advection = True
            settings.enable_eke_isopycnal_diffusion = True

            settings.enable_idemix = True
            settings.enable_idemix_hor_diffusion = True
            settings.enable_eke_diss_surfbot = True
            settings.eke_diss_surfbot_frac = 0.2
            settings.enable_idemix_superbee_advection = True

            settings.eq_of_state_type = 5

            var_meta = state.var_meta
            var_meta.update(
                t_star=Variable("t_star", ("yt",), "deg C", "Reference surface temperature"),
                t_rest=Variable("t_rest", ("xt", "yt"), "1/s", "Surface temperature restoring time scale"),
            )

        @veros_routine
        def set_grid(self, state):
            vs = state.variables
            settings = state.settings
            vs.dxt = update(vs.dxt, at[...], 120 / settings.nx)
            vs.dyt = update(vs.dyt, at[...], 80 / settings.ny)
            vs.dzt = update(vs.dzt, at[...], 4000 / settings.nz)

        @veros_routine
        def set_coriolis(self, state):
            vs = state.variables
            settings = state.settings
            vs.coriolis_t = update(
                vs.coriolis_t, at[...], 2 * settings.omega * npx.sin(vs.yt[None, :] / 180.0 * settings.pi)
            )

        @veros_routine
        def set_topography(self, state):
            vs = state.variables
            x, y = npx.meshgrid(vs.xt, vs.yt, indexing="ij")
            vs.kbot = npx.logical_or(x > 1.0, y < -20).astype("int")

        @veros_routine
        def set_initial_conditions(self, state):
            vs = state.variables
            settings = state.settings

            # initial conditions
            vs.temp = update(vs.temp, at[...], ((1 - vs.zt[None, None, :] / vs.zw[0]) * 15 * vs.maskT)[..., None])
            vs.salt = update(vs.salt, at[...], 35.0 * vs.maskT[..., None])

            # wind stress forcing
            yt_min = global_min(vs.yt.min())
            yu_min = global_min(vs.yu.min())
            yt_max = global_max(vs.yt.max())
            yu_max = global_max(vs.yu.max())

            taux = allocate(state.dimensions, ("yt",))
            taux = npx.where(vs.yt < -20, 0.1 * npx.sin(settings.pi * (vs.yu - yu_min) / (-20.0 - yt_min)), taux)
            taux = npx.where(vs.yt > 10, 0.1 * (1 - npx.cos(2 * settings.pi * (vs.yu - 10.0) / (yu_max - 10.0))), taux)
            vs.surface_taux = taux * vs.maskU[:, :, -1]

            # surface heatflux forcing
            vs.t_star = allocate(state.dimensions, ("yt",), fill=15)
            vs.t_star = npx.where(vs.yt < -20, 15 * (vs.yt - yt_min) / (-20 - yt_min), vs.t_star)
            vs.t_star = npx.where(vs.yt > 20, 15 * (1 - (vs.yt - 20) / (yt_max - 20)), vs.t_star)
            vs.t_rest = vs.dzt[npx.newaxis, -1] / (30.0 * 86400.0) * vs.maskT[:, :, -1]

            if settings.enable_tke:
                vs.forc_tke_surface = update(
                    vs.forc_tke_surface,
                    at[2:-2, 2:-2],
                    npx.sqrt(
                        (0.5 * (vs.surface_taux[2:-2, 2:-2] + vs.surface_taux[1:-3, 2:-2]) / settings.rho_0) ** 2
                        + (0.5 * (vs.surface_tauy[2:-2, 2:-2] + vs.surface_tauy[2:-2, 1:-3]) / settings.rho_0) ** 2
                    )
                    ** (1.5),
                )

            if settings.enable_idemix:
                vs.forc_iw_bottom = 1e-6 * vs.maskW[:, :, -1]
                vs.forc_iw_surface = 1e-7 * vs.maskW[:, :, -1]

        @veros_routine
        def set_forcing(self, state):
            vs = state.variables
            vs.forc_temp_surface = vs.t_rest * (vs.t_star - vs.temp[:, :, -1, vs.tau])

        @veros_routine
        def set_diagnostics(self, state):
            state.diagnostics.clear()

    sim = ACC2Benchmark()
    sim.setup()

    if not pyom2_lib:
        sim.run()
        return

    pyom_obj = load_pyom(pyom2_lib)
    pyom_obj = pyom_from_state(sim.state, pyom_obj, ignore_attrs=("t_star", "t_rest"))

    # different units in pyom
    pyom_obj.main_module.surface_taux /= 1000

    t_rest = sim.state.variables.t_rest
    t_star = sim.state.variables.t_star

    def set_forcing_pyom(pyom_obj):
        m = pyom_obj.main_module
        m.forc_temp_surface[:] = t_rest * (t_star - m.temp[:, :, -1, m.tau - 1])

    run_pyom(pyom_obj, set_forcing_pyom)


if __name__ == "__main__":
    main()
