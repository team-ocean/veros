import numpy as np

from veros import VerosSetup, veros_routine
from veros.variables import allocate, Variable
from veros.core.operators import numpy as npx, update, at

from veros.pyom_compat import load_pyom, setup_pyom

from test_base import compare_state


yt_start = -39.0
yt_end = 43
yu_start = -40.0
yu_end = 42


def set_parameter_pyom(pyom_obj):
    m = pyom_obj.main_module

    (m.nx, m.ny, m.nz) = (30, 42, 15)
    m.dt_mom = 4800
    m.dt_tracer = 86400 / 2.0
    m.runlen = 86400 * 365

    m.coord_degree = 1
    m.enable_cyclic_x = 1

    m.congr_epsilon = 1e-8
    m.congr_max_iterations = 10_000

    m.ab_eps = 0.1

    i = pyom_obj.isoneutral_module
    i.enable_neutral_diffusion = 1
    i.k_iso_0 = 1000.0
    i.k_iso_steep = 500.0
    i.iso_dslope = 0.005
    i.iso_slopec = 0.01
    i.enable_skew_diffusion = 1

    m.enable_hor_friction = 1
    m.a_h = 2.2e5
    m.enable_hor_friction_cos_scaling = 1
    m.hor_friction_cospower = 1

    m.enable_bottom_friction = 1
    m.r_bot = 1e-5

    m.enable_streamfunction = True
    m.enable_free_surface = False

    m.enable_implicit_vert_friction = 1
    t = pyom_obj.tke_module
    t.enable_tke = 1
    t.c_k = 0.1
    t.c_eps = 0.7
    t.alpha_tke = 30.0
    t.mxl_min = 1e-8
    t.tke_mxl_choice = 2
    t.kappam_min = 2e-4

    i.k_gm_0 = 1000.0
    e = pyom_obj.eke_module
    e.enable_eke = 1
    e.eke_k_max = 1e4
    e.eke_c_k = 0.4
    e.eke_c_eps = 0.5
    e.eke_cross = 2.0
    e.eke_crhin = 1.0
    e.eke_lmin = 100.0
    e.enable_eke_superbee_advection = 1
    e.enable_eke_isopycnal_diffusion = 1

    i = pyom_obj.idemix_module
    i.enable_idemix = 1
    i.enable_idemix_hor_diffusion = 1
    i.enable_eke_diss_surfbot = 1
    i.eke_diss_surfbot_frac = 0.2
    i.enable_idemix_superbee_advection = 1
    i.tau_v = 86400.0
    i.jstar = 10.0
    i.mu0 = 4.0 / 3.0
    i.gamma = 1.57

    m.eq_of_state_type = 3


def set_grid_pyom(pyom_obj):
    m = pyom_obj.main_module
    ddz = [50.0, 70.0, 100.0, 140.0, 190.0, 240.0, 290.0, 340.0, 390.0, 440.0, 490.0, 540.0, 590.0, 640.0, 690.0]
    m.dxt[:] = 2.0
    m.dyt[:] = 2.0
    m.x_origin = 0.0
    m.y_origin = -40.0
    m.dzt[:] = ddz[::-1]
    m.dzt[:] *= 1 / 2.5


def set_coriolis_pyom(pyom_obj):
    m = pyom_obj.main_module
    m.coriolis_t[:, :] = 2 * m.omega * np.sin(m.yt[None, :] / 180.0 * np.pi)


def set_topography_pyom(pyom_obj):
    m = pyom_obj.main_module
    (X, Y) = np.meshgrid(m.xt, m.yt)
    X = X.transpose()
    Y = Y.transpose()
    m.kbot[...] = (X > 1.0) | (Y < -20)


def set_initial_conditions_pyom(pyom_obj):
    m = pyom_obj.main_module

    # initial conditions
    m.temp[:, :, :, :] = ((1 - m.zt[None, None, :] / m.zw[0]) * 15 * m.maskt)[..., None]
    m.salt[:, :, :, :] = 35.0 * m.maskt[..., None]

    # wind stress forcing
    taux = np.zeros(m.ny + 1)
    yt = m.yt[2 : m.ny + 3]
    taux = (0.1e-3 * np.sin(np.pi * (m.yu[2 : m.ny + 3] - yu_start) / (-20.0 - yt_start))) * (yt < -20) + (
        0.1e-3 * (1 - np.cos(2 * np.pi * (m.yu[2 : m.ny + 3] - 10.0) / (yu_end - 10.0)))
    ) * (yt > 10)
    m.surface_taux[:, 2 : m.ny + 3] = taux * m.masku[:, 2 : m.ny + 3, -1]

    t = pyom_obj.tke_module
    t.forc_tke_surface[2:-2, 2:-2] = (
        np.sqrt(
            (0.5 * (m.surface_taux[2:-2, 2:-2] + m.surface_taux[1:-3, 2:-2])) ** 2
            + (0.5 * (m.surface_tauy[2:-2, 2:-2] + m.surface_tauy[2:-2, 1:-3])) ** 2
        )
        ** 1.5
    )


def set_forcing_pyom(pyom_obj):
    m = pyom_obj.main_module
    t_star = (
        15 * np.invert((m.yt < -20) | (m.yt > 20))
        + 15 * (m.yt - yt_start) / (-20 - yt_start) * (m.yt < -20)
        + 15 * (1 - (m.yt - 20) / (yt_end - 20)) * (m.yt > 20.0)
    )
    t_rest = m.dzt[None, -1] / (30.0 * 86400.0) * m.maskt[:, :, -1]
    m.forc_temp_surface = t_rest * (t_star - m.temp[:, :, -1, m.tau - 1])


class ACCSetup(VerosSetup):
    @veros_routine
    def set_parameter(self, state):
        settings = state.settings
        settings.identifier = "acc"

        settings.nx, settings.ny, settings.nz = 30, 42, 15
        settings.dt_mom = 4800
        settings.dt_tracer = 86400 / 2.0
        settings.runlen = 86400 * 365

        settings.x_origin = 0.0
        settings.y_origin = -40.0

        settings.coord_degree = True
        settings.enable_cyclic_x = True

        settings.enable_streamfunction = True
        settings.enable_free_surface = False

        settings.enable_neutral_diffusion = True
        settings.K_iso_0 = 1000.0
        settings.K_iso_steep = 500.0
        settings.iso_dslope = 0.005
        settings.iso_slopec = 0.01
        settings.enable_skew_diffusion = True

        settings.enable_hor_friction = True
        settings.A_h = 2.2e5
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

        settings.enable_idemix = 1
        settings.enable_idemix_hor_diffusion = 1
        settings.enable_eke_diss_surfbot = 1
        settings.eke_diss_surfbot_frac = 0.2
        settings.enable_idemix_superbee_advection = 1
        settings.tau_v = 86400.0
        settings.jstar = 10.0
        settings.mu0 = 4.0 / 3.0

        settings.eq_of_state_type = 3

        var_meta = state.var_meta
        var_meta.update(
            t_star=Variable("t_star", ("yt",), "deg C", "Reference surface temperature"),
            t_rest=Variable("t_rest", ("xt", "yt"), "1/s", "Surface temperature restoring time scale"),
        )

    @veros_routine
    def set_grid(self, state):
        vs = state.variables
        ddz = npx.array(
            [50.0, 70.0, 100.0, 140.0, 190.0, 240.0, 290.0, 340.0, 390.0, 440.0, 490.0, 540.0, 590.0, 640.0, 690.0]
        )
        vs.dxt = update(vs.dxt, at[...], 2.0)
        vs.dyt = update(vs.dyt, at[...], 2.0)
        vs.dzt = update(vs.dzt, at[...], ddz[::-1] / 2.5)

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
        taux = allocate(state.dimensions, ("yt",))
        taux = npx.where(vs.yt < -20, 0.1e-3 * npx.sin(settings.pi * (vs.yu - yu_start) / (-20.0 - yt_start)), taux)
        taux = npx.where(vs.yt > 10, 0.1e-3 * (1 - npx.cos(2 * settings.pi * (vs.yu - 10.0) / (yu_end - 10.0))), taux)
        vs.surface_taux = taux * vs.maskU[:, :, -1]

        # surface heatflux forcing
        vs.t_star = allocate(state.dimensions, ("yt",), fill=15)
        vs.t_star = npx.where(vs.yt < -20, 15 * (vs.yt - yt_start) / (-20 - yt_start), vs.t_star)
        vs.t_star = npx.where(vs.yt > 20, 15 * (1 - (vs.yt - 20) / (yt_end - 20)), vs.t_star)
        vs.t_rest = vs.dzt[npx.newaxis, -1] / (30.0 * 86400.0) * vs.maskT[:, :, -1]

        if settings.enable_tke:
            vs.forc_tke_surface = update(
                vs.forc_tke_surface,
                at[2:-2, 2:-2],
                npx.sqrt(
                    (0.5 * (vs.surface_taux[2:-2, 2:-2] + vs.surface_taux[1:-3, 2:-2])) ** 2
                    + (0.5 * (vs.surface_tauy[2:-2, 2:-2] + vs.surface_tauy[2:-2, 1:-3])) ** 2
                )
                ** (1.5),
            )

    @veros_routine
    def set_forcing(self, state):
        vs = state.variables
        vs.forc_temp_surface = vs.t_rest * (vs.t_star - vs.temp[:, :, -1, vs.tau])

    @veros_routine
    def set_diagnostics(self, state):
        pass

    @veros_routine
    def after_timestep(self, state):
        pass


def test_acc_setup(pyom2_lib):
    pyom_obj = load_pyom(pyom2_lib)
    setup_pyom(
        pyom_obj,
        set_parameter_pyom,
        set_grid_pyom,
        set_coriolis_pyom,
        set_topography_pyom,
        set_initial_conditions_pyom,
        set_forcing_pyom,
    )

    sim = ACCSetup()
    sim.setup()

    compare_state(sim.state, pyom_obj, atol=1e-5)
