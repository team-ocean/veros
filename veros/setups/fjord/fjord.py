from veros import VerosSetup, veros_routine
from veros.variables import allocate, Variable
from veros.core.operators import numpy as npx, update, at


class FjordSetup(VerosSetup):
    """A model using spherical coordinates with a closed domain representing the idealized fjord.

    The bathymetry of the model is idealized to a flat-bottom (with depth of 40 m) over the mouth of the fjord.
    The bathymetry of fjord's inner part has a constant depth slope changing from 40 to 20 m from the mouth
    of the fjord to the end of its inner part (northern boundary of the domain), respectively.
    The horizontal grid has resolution of :math:`0.0005 \\times 0.005` degrees, and the vertical one has 10 levels.

    Wind forcing over the fjord, buoyancy relaxation and sponge layer (salinity) forcing
    drive water masses circulation.

    This setup demonstrates:
     - setting up an idealized fjord geometry
     - modifing surface forcings over selected regions of the domain
     - setting up sponge layer forcing
     - basic usage of diagnostics

    :doc:`Adapted from ACC sector model </reference/setups/acc_sector>`.

    """

    @veros_routine
    def set_parameter(self, state):
        settings = state.settings

        settings.identifier = "fjord"

        settings.nx, settings.ny, settings.nz = 32, 96, 10
        settings.dt_mom = 4.0
        settings.dt_tracer = 4.0
        settings.runlen = 86400.0 * 30

        settings.x_origin = 11.0
        settings.y_origin = 58.0

        settings.coord_degree = True
        settings.enable_cyclic_x = False

        settings.enable_neutral_diffusion = True
        settings.K_iso_0 = 1.0
        settings.K_iso_steep = 1.0
        settings.iso_dslope = 1e-3
        settings.iso_slopec = 4e-3
        settings.enable_skew_diffusion = True

        settings.enable_hor_friction = True
        settings.A_h = 10.0
        settings.enable_hor_friction_cos_scaling = True
        settings.hor_friction_cosPower = 1

        settings.enable_tempsalt_sources = True

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
        settings.kappaH_min = 2e-5
        settings.enable_Prandtl_tke = False
        settings.enable_kappaH_profile = True

        settings.K_gm_0 = 1.0
        settings.enable_eke = False
        settings.eke_k_max = 1e4
        settings.eke_c_k = 0.4
        settings.eke_c_eps = 0.5
        settings.eke_cross = 2.0
        settings.eke_crhin = 1.0
        settings.eke_lmin = 100.0
        settings.enable_eke_superbee_advection = False
        settings.enable_eke_isopycnal_diffusion = False

        settings.enable_idemix = False
        settings.enable_idemix_hor_diffusion = False
        settings.enable_eke_diss_surfbot = False
        settings.eke_diss_surfbot_frac = 0.2
        settings.enable_idemix_superbee_advection = False

        settings.eq_of_state_type = 3

        var_meta = state.var_meta
        var_meta.update(
            sst_star=Variable("sst_star", ("yt",), "deg C", "Reference surface temperature"),
            sst_rest=Variable(
                "sst_rest",
                (
                    "xt",
                    "yt",
                ),
                "m/s",
                "Surface temperature restoring time scale",
            ),
            sss_star=Variable("sss_star", ("yt",), "g/kg", "Reference surface salinity"),
            sss_rest=Variable(
                "sss_rest",
                (
                    "xt",
                    "yt",
                ),
                "m/s",
                "Surface salinity restoring time scale",
            ),
            s_star=Variable(
                "s_star",
                (
                    "xt",
                    "yt",
                    "zt",
                ),
                "g/kg",
                "Salinity sponge layer forcing",
            ),
            rest_tscl=Variable(
                "rest_tscl",
                (
                    "xt",
                    "yt",
                    "zt",
                ),
                "1/s",
                "Forcing restoration time scale",
            ),
        )

    @veros_routine
    def set_grid(self, state):
        vs = state.variables
        settings = state.settings

        # keep total domain size constant when nx or ny changes
        vs.dxt = update(vs.dxt, at[...], 0.0005 * 32 / settings.nx)
        vs.dyt = update(vs.dyt, at[...], 0.005 * 96 / settings.ny)
        vs.dzt = update(vs.dzt, at[...], 4.0 * 10.0 / settings.nz)

    @veros_routine
    def set_coriolis(self, state):
        vs = state.variables
        settings = state.settings
        vs.coriolis_t = update(
            vs.coriolis_t, at[:, :], 2 * settings.omega * npx.sin(vs.yt[None, :] / 180.0 * settings.pi)
        )

    @veros_routine
    def set_topography(self, state):
        vs = state.variables

        x, y = npx.meshgrid(vs.xt, vs.yt, indexing="ij")
        kzt = (20 * (y[y >= 58.2] - 58.2) + 1).astype("int")
        landmass = npx.logical_or((x > 11.0055) & (x < 11.0105), y < 58.2).astype("int")
        landmass_eq_0 = landmass == 0

        vs.kbot = npx.ones(x.shape)
        bathymetry = y >= 58.2
        vs.kbot = update(vs.kbot, at[bathymetry], kzt)
        vs.kbot = update(vs.kbot, at[landmass_eq_0], 0.0)

    @veros_routine(
        dist_safe=False,
        local_variables=[
            "sst_star",
            "s_star",
            "sss_star",
            "sst_rest",
            "rest_tscl",
            "sss_rest",
            "temp",
            "salt",
            "maskT",
            "maskU",
            "maskV",
            "maskW",
            "xt",
            "yt",
            "yu",
            "zt",
            "zw",
            "dzt",
            "forc_tke_surface",
            "surface_taux",
            "surface_tauy",
            "forc_iw_bottom",
            "forc_iw_surface",
        ],
    )
    def set_initial_conditions(self, state):
        vs = state.variables
        settings = state.settings

        # initial conditions
        vs.temp = update(vs.temp, at[...], (5.0 + (1 - vs.zt[None, None, :] / vs.zw[0]) * 11 * vs.maskT)[..., None])
        vs.salt = update(vs.salt, at[...], (npx.linspace(34, 26, settings.nz)[None, None, :] * vs.maskT)[..., None])

        # wind stress forcing
        vs.surface_taux = 5e-2 * vs.maskU[:, :, -1]
        vs.surface_tauy = 2e-2 * vs.maskV[:, :, -1]

        # surface heatflux forcing
        vs.sst_star = allocate(state.dimensions, ("yt",), fill=16.0)

        # surface salinity forcing
        vs.sss_star = allocate(state.dimensions, ("yt",), fill=26.0)
        vs.sst_rest = vs.dzt[-1] / (10.0 * 86400.0) * vs.maskT[:, :, -1]
        vs.sss_rest = vs.dzt[-1] / (30.0 * 86400.0) * vs.maskT[:, :, -1]

        # salinity sponge layer forcing
        _, _yt, _ = npx.meshgrid(vs.xt, vs.yt, vs.zt, indexing="ij")
        sponge_region = _yt <= vs.yt[4]
        vs.s_star = allocate(
            state.dimensions,
            (
                "xt",
                "yt",
                "zt",
            ),
        )
        vs.s_star = update(
            vs.s_star, at[sponge_region], (npx.linspace(34, 26, settings.nz)[None, None, :] * vs.maskT)[sponge_region]
        )
        vs.rest_tscl = allocate(
            state.dimensions,
            (
                "xt",
                "yt",
                "zt",
            ),
        )
        vs.rest_tscl = update(vs.s_star, at[sponge_region], 1.0 / (30.0 * 86400.0))

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
        settings = state.settings

        vs.forc_temp_surface = vs.sst_rest * (vs.sst_star - vs.temp[:, :, -1, vs.tau])
        vs.forc_salt_surface = vs.sss_rest * (vs.sss_star - vs.salt[:, :, -1, vs.tau])

        if settings.enable_tempsalt_sources:
            vs.salt_source = update(
                vs.salt_source, at[...], vs.maskT * vs.rest_tscl * (vs.s_star[:, :, :] - vs.salt[:, :, :, vs.tau])
            )

    @veros_routine
    def set_diagnostics(self, state):
        settings = state.settings
        state.diagnostics["snapshot"].output_frequency = 86400.0
        state.diagnostics["averages"].output_variables = (
            "salt",
            "temp",
            "u",
            "v",
            "w",
            "psi",
            "rho",
            "surface_taux",
            "surface_tauy",
        )
        state.diagnostics["averages"].output_frequency = 86400 / 4.0
        state.diagnostics["averages"].sampling_frequency = settings.dt_tracer * 10
        state.diagnostics["tracer_monitor"].output_frequency = 86400.0 / 4.0
        state.diagnostics["energy"].output_frequency = 86400.0 / 48
        state.diagnostics["energy"].sampling_frequency = settings.dt_tracer * 10

    @veros_routine
    def after_timestep(self, state):
        pass
