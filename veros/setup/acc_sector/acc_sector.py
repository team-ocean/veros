#!/usr/bin/env python

import veros
import veros.tools


class ACCSector(veros.Veros):
    """A model using spherical coordinates with a partially closed domain representing the narrow sector of Atlantic and ACC.

    The bathymetry of the model is idealized to a flat-bottom (with depth of 4000 m) over the majority of the domain,
    except a half depth appended within the confines of the circumpolar channel at the inflow and outflow regions.
    The horizontal grid has resolution of :math:`2 \\times 2` degrees, and the vertical one has 40 levels.

    Wind forcing over the sector part and buoyancy relaxation drive a large-scale meridional overturning circulation.

    This setup demonstrates:
     - setting up an idealized geometry after `(Munday et al., 2013) <https://doi.org/10.1175/JPO-D-12-095.1>`_.
     - modifing surface forcings over selected regions of the domain
     - sensitivity of circumpolar transport and meridional overturning 
       to changes in Southern Ocean wind stress and buoyancy anomalies
     - basic usage of diagnostics

    :doc:`Adapted from ACC channel model </reference/setups/acc>`.

    """
    
    max_depth = 4000.
    so_wind_factor = 1.

    @veros.veros_method
    def set_parameter(self):
        self.identifier = "acc_sector"

        self.nx, self.ny, self.nz = 15, 62, 40
        self.dt_mom = 3600.
        self.dt_tracer = 3600.
        self.runlen = 86400 * 365

        self.coord_degree = True
        self.enable_cyclic_x = True

        self.congr_epsilon = 1e-12
        self.congr_max_iterations = 5000

        self.enable_neutral_diffusion = True
        self.K_iso_0 = 1000.0
        self.K_iso_steep = 500.0
        self.iso_dslope = 0.005
        self.iso_slopec = 0.01
        self.enable_skew_diffusion = True

        self.enable_hor_friction = True
        self.A_h = 5e4 * 2 
        self.enable_hor_friction_cos_scaling = True
        self.hor_friction_cosPower = 1

        self.enable_bottom_friction = True
        self.r_bot = 1e-5

        self.enable_implicit_vert_friction = True

        self.enable_tke = True
        self.c_k = 0.1
        self.c_eps = 0.7
        self.alpha_tke = 30.0
        self.mxl_min = 1e-8
        self.tke_mxl_choice = 2
        self.kappaM_min = 2e-4
        self.kappaH_min = 2e-5
        # self.enable_tke_superbee_advection = True
        self.enable_Prandtl_tke = False

        self.K_gm_0 = 1300.0
        self.enable_eke = False
        self.eke_k_max = 1e4
        self.eke_c_k = 0.4
        self.eke_c_eps = 0.5
        self.eke_cross = 2.
        self.eke_crhin = 1.0
        self.eke_lmin = 100.0
        self.enable_eke_superbee_advection = False
        self.enable_eke_isopycnal_diffusion = False

        self.enable_idemix = False
        self.enable_idemix_hor_diffusion = False
        self.enable_eke_diss_surfbot = False
        self.eke_diss_surfbot_frac = 0.2
        self.enable_idemix_superbee_advection = False

        self.eq_of_state_type = 3

    @veros.veros_method
    def set_grid(self):
        self.dxt[...] = 2.0
        self.dyt[...] = 2.0
        self.x_origin = 0.0
        self.y_origin = -60.0
        self.dzt[...] = veros.tools.get_vinokur_grid_steps(self.nz, self.max_depth, 10., refine_towards="lower")

    @veros.veros_method
    def set_coriolis(self):
        self.coriolis_t[:, :] = 2 * self.omega * np.sin(self.yt[None, :] / 180. * self.pi)

    @veros.veros_method
    def set_topography(self):
        x, y = np.meshgrid(self.xt, self.yt, indexing="ij")
        self.kbot = np.logical_or( (x > 1.0) & (x < 27), y < -40).astype(np.int)

        # A half depth (ridge) is appended to the domain within the confines
        # of the circumpolar channel at the inflow and outflow regions
        bathymetry = np.logical_or( ((x <= 1.0) & (y < -40)), ((x >= 27) & (y < -40)) )
        kzt2000 = np.sum((self.zt < -2000.).astype(np.int))
        self.kbot[bathymetry] = kzt2000

    @veros.veros_method
    def set_initial_conditions(self):
        # initial conditions
        self.temp[:, :, :, 0:2] = ((1 - self.zt[None, None, :] / self.zw[0]) * 15 * self.maskT)[..., None]
        self.salt[:, :, :, 0:2] = 35.0 * self.maskT[..., None]

        # wind stress forcing
        taux = np.zeros(self.ny + 4, dtype=self.default_float_type)
        north = self.yt > 30
        subequatorial_north_n = (self.yt >= 15) & (self.yt < 30)
        subequatorial_north_s = (self.yt > 0) & (self.yt < 15)
        equator = (self.yt > -5) & (self.yt < 5)
        subequatorial_south_n = (self.yt > -15) & (self.yt < 0)
        subequatorial_south_s = (self.yt <= -15) & (self.yt > -30)
        south = self.yt < -30

        taux[north] = -5e-2 * np.sin(np.pi * (self.yu[north] - self.yu.max()) / (self.yt.max() - 30.))
        taux[subequatorial_north_s] =  5e-2 * np.sin(np.pi * (self.yu[subequatorial_north_s] - 30.) / 30.)
        taux[subequatorial_north_n] = 5e-2 * np.sin(np.pi * (self.yt[subequatorial_north_n] - 30.) / 30.)
        taux[subequatorial_south_n] =  -5e-2 * np.sin(np.pi * (self.yu[subequatorial_south_n] - 30.) / 30.)
        taux[subequatorial_south_s] = -5e-2 * np.sin(np.pi * (self.yt[subequatorial_south_s] - 30.) / 30.)
        taux[equator] = -1.5e-2 * np.cos(np.pi * (self.yu[equator] - 10.) / 10.) - 2.5e-2
        taux[south] = 15e-2 * np.sin(np.pi * (self.yu[south] - self.yu.min()) / (-30. - self.yt.min()))
        self.surface_taux[:, :] = taux * self.maskU[:, :, -1]

        # surface heatflux forcing
        DELTA_T, TS, TN = 25., 0., 5.
        self._t_star = DELTA_T * np.ones(self.ny + 4, dtype=self.default_float_type)
        self._t_star[self.yt<0] = TS + DELTA_T * np.sin(np.pi * (self.yt[self.yt<0] + 60.) / np.abs(2 * self.y_origin))
        self._t_star[self.yt>0] = TN + (DELTA_T + TS - TN) * np.sin(np.pi * (self.yt[self.yt>0] + 60.) / np.abs(2 * self.y_origin))
        self._t_rest = self.dzt[np.newaxis, -1] / (10. * 86400.) * self.maskT[:, :, -1]

        if self.enable_tke:
            self.forc_tke_surface[2:-2, 2:-2] = np.sqrt((0.5 * (self.surface_taux[2:-2, 2:-2] + self.surface_taux[1:-3, 2:-2]) / self.rho_0)**2
                                                        + (0.5 * (self.surface_tauy[2:-2, 2:-2] + self.surface_tauy[2:-2, 1:-3]) / self.rho_0)**2)**(1.5)

        if self.enable_idemix:
            self.forc_iw_bottom[...] = 1e-6 * self.maskW[:, :, -1]
            self.forc_iw_surface[...] = 1e-7 * self.maskW[:, :, -1]

    @veros.veros_method
    def set_forcing(self):
        self.forc_temp_surface[...] = self._t_rest * (self._t_star - self.temp[:, :, -1, self.tau])

    @veros.veros_method
    def set_diagnostics(self):
        self.diagnostics["snapshot"].output_frequency = 86400 * 10
        self.diagnostics["averages"].output_variables = (
            "salt", "temp", "u", "v", "w", "psi", "rho", "surface_taux", "surface_tauy"
        )
        self.diagnostics["averages"].output_frequency = 365 * 86400.
        self.diagnostics["averages"].sampling_frequency = self.dt_tracer * 10
        self.diagnostics["overturning"].output_frequency = 365 * 86400. / 48.
        self.diagnostics["overturning"].sampling_frequency = self.dt_tracer * 10
        self.diagnostics["tracer_monitor"].output_frequency = 365 * 86400. / 12.
        self.diagnostics["energy"].output_frequency = 365 * 86400. / 48
        self.diagnostics["energy"].sampling_frequency = self.dt_tracer * 10

    def after_timestep(self):
        pass


@veros.tools.cli
def run(*args, **kwargs):
    simulation = ACCSector(*args, **kwargs)
    simulation.setup()
    simulation.run()


if __name__ == "__main__":
    run()
