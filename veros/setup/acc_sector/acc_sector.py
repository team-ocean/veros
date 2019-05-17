#!/usr/bin/env python

from veros import VerosSetup, veros_method, runtime_settings as rs
import veros.tools
from veros.variables import allocate
from veros.distributed import global_min, global_max


class ACCSectorSetup(VerosSetup):
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

    @veros_method
    def set_parameter(self, vs):
        vs.identifier = 'acc_sector'

        vs.nx, vs.ny, vs.nz = 15, 62, 40
        vs.dt_mom = 3600.
        vs.dt_tracer = 3600.
        vs.runlen = 86400 * 365

        vs.coord_degree = True
        vs.enable_cyclic_x = True

        vs.congr_epsilon = 1e-12
        vs.congr_max_iterations = 5000

        vs.enable_neutral_diffusion = True
        vs.K_iso_0 = 1000.0
        vs.K_iso_steep = 500.0
        vs.iso_dslope = 0.005
        vs.iso_slopec = 0.01
        vs.enable_skew_diffusion = True

        vs.enable_hor_friction = True
        vs.A_h = 5e4 * 2
        vs.enable_hor_friction_cos_scaling = True
        vs.hor_friction_cosPower = 1

        vs.enable_bottom_friction = True
        vs.r_bot = 1e-5

        vs.enable_implicit_vert_friction = True

        vs.enable_tke = True
        vs.c_k = 0.1
        vs.c_eps = 0.7
        vs.alpha_tke = 30.0
        vs.mxl_min = 1e-8
        vs.tke_mxl_choice = 2
        vs.kappaM_min = 2e-4
        vs.kappaH_min = 2e-5
        # vs.enable_tke_superbee_advection = True
        vs.enable_Prandtl_tke = False

        vs.K_gm_0 = 1300.0
        vs.enable_eke = False
        vs.eke_k_max = 1e4
        vs.eke_c_k = 0.4
        vs.eke_c_eps = 0.5
        vs.eke_cross = 2.
        vs.eke_crhin = 1.0
        vs.eke_lmin = 100.0
        vs.enable_eke_superbee_advection = False
        vs.enable_eke_isopycnal_diffusion = False

        vs.enable_idemix = False
        vs.enable_idemix_hor_diffusion = False
        vs.enable_eke_diss_surfbot = False
        vs.eke_diss_surfbot_frac = 0.2
        vs.enable_idemix_superbee_advection = False

        vs.eq_of_state_type = 3

    @veros_method
    def set_grid(self, vs):
        # keep total domain size constant when nx or ny changes
        vs.dxt[...] = 2.0 * 15 / vs.nx
        vs.dyt[...] = 2.0 * 62 / vs.ny

        vs.x_origin = 0.0
        vs.y_origin = -60.0

        vs.dzt[...] = veros.tools.get_vinokur_grid_steps(vs.nz, self.max_depth, 10., refine_towards='lower')

    @veros_method
    def set_coriolis(self, vs):
        vs.coriolis_t[:, :] = 2 * vs.omega * np.sin(vs.yt[None, :] / 180. * vs.pi)

    @veros_method
    def set_topography(self, vs):
        x, y = np.meshgrid(vs.xt, vs.yt, indexing='ij')
        vs.kbot = np.logical_or((x > 1.0) & (x < 27), y < -40).astype(np.int)

        # A half depth (ridge) is appended to the domain within the confines
        # of the circumpolar channel at the inflow and outflow regions
        bathymetry = np.logical_or(((x <= 1.0) & (y < -40)), ((x >= 27) & (y < -40)))
        kzt2000 = np.sum((vs.zt < -2000.).astype(np.int))
        vs.kbot[bathymetry] = kzt2000

    @veros_method
    def set_initial_conditions(self, vs):
        # initial conditions
        vs.temp[:, :, :, 0:2] = ((1 - vs.zt[None, None, :] / vs.zw[0]) * 15 * vs.maskT)[..., None]
        vs.salt[:, :, :, 0:2] = 35.0 * vs.maskT[..., None]

        # wind stress forcing
        yt_min = global_min(vs, vs.yt.min())
        yu_min = global_min(vs, vs.yu.min())
        yt_max = global_max(vs, vs.yt.max())
        yu_max = global_max(vs, vs.yu.max())

        taux = allocate(vs, ('yt',))
        north = vs.yt > 30
        subequatorial_north_n = (vs.yt >= 15) & (vs.yt < 30)
        subequatorial_north_s = (vs.yt > 0) & (vs.yt < 15)
        equator = (vs.yt > -5) & (vs.yt < 5)
        subequatorial_south_n = (vs.yt > -15) & (vs.yt < 0)
        subequatorial_south_s = (vs.yt <= -15) & (vs.yt > -30)
        south = vs.yt < -30

        taux[north] = -5e-2 * np.sin(np.pi * (vs.yu[north] - yu_max) / (yt_max - 30.))
        taux[subequatorial_north_s] =  5e-2 * np.sin(np.pi * (vs.yu[subequatorial_north_s] - 30.) / 30.)
        taux[subequatorial_north_n] = 5e-2 * np.sin(np.pi * (vs.yt[subequatorial_north_n] - 30.) / 30.)
        taux[subequatorial_south_n] =  -5e-2 * np.sin(np.pi * (vs.yu[subequatorial_south_n] - 30.) / 30.)
        taux[subequatorial_south_s] = -5e-2 * np.sin(np.pi * (vs.yt[subequatorial_south_s] - 30.) / 30.)
        taux[equator] = -1.5e-2 * np.cos(np.pi * (vs.yu[equator] - 10.) / 10.) - 2.5e-2
        taux[south] = 15e-2 * np.sin(np.pi * (vs.yu[south] - yu_min) / (-30. - yt_min))
        vs.surface_taux[:, :] = taux * vs.maskU[:, :, -1]

        # surface heatflux forcing
        DELTA_T, TS, TN = 25., 0., 5.
        vs._t_star = allocate(vs, ('yt',), fill=DELTA_T)
        vs._t_star[vs.yt<0] = TS + DELTA_T * np.sin(np.pi * (vs.yt[vs.yt<0] + 60.) / np.abs(2 * vs.y_origin))
        vs._t_star[vs.yt>0] = TN + (DELTA_T + TS - TN) * np.sin(np.pi * (vs.yt[vs.yt>0] + 60.) / np.abs(2 * vs.y_origin))
        vs._t_rest = vs.dzt[None, -1] / (10. * 86400.) * vs.maskT[:, :, -1]

        if vs.enable_tke:
            vs.forc_tke_surface[2:-2, 2:-2] = np.sqrt((0.5 * (vs.surface_taux[2:-2, 2:-2] + vs.surface_taux[1:-3, 2:-2]) / vs.rho_0)**2
                                                      + (0.5 * (vs.surface_tauy[2:-2, 2:-2] + vs.surface_tauy[2:-2, 1:-3]) / vs.rho_0)**2)**(1.5)

        if vs.enable_idemix:
            vs.forc_iw_bottom[...] = 1e-6 * vs.maskW[:, :, -1]
            vs.forc_iw_surface[...] = 1e-7 * vs.maskW[:, :, -1]

    @veros_method
    def set_forcing(self, vs):
        vs.forc_temp_surface[...] = vs._t_rest * (vs._t_star - vs.temp[:, :, -1, vs.tau])

    @veros_method
    def set_diagnostics(self, vs):
        vs.diagnostics['snapshot'].output_frequency = 86400 * 10
        vs.diagnostics['averages'].output_variables = (
            'salt', 'temp', 'u', 'v', 'w', 'psi', 'rho', 'surface_taux', 'surface_tauy'
        )
        vs.diagnostics['averages'].output_frequency = 365 * 86400.
        vs.diagnostics['averages'].sampling_frequency = vs.dt_tracer * 10
        vs.diagnostics['overturning'].output_frequency = 365 * 86400. / 48.
        vs.diagnostics['overturning'].sampling_frequency = vs.dt_tracer * 10
        vs.diagnostics['tracer_monitor'].output_frequency = 365 * 86400. / 12.
        vs.diagnostics['energy'].output_frequency = 365 * 86400. / 48
        vs.diagnostics['energy'].sampling_frequency = vs.dt_tracer * 10

    def after_timestep(self, vs):
        pass


@veros.tools.cli
def run(*args, **kwargs):
    simulation = ACCSectorSetup(*args, **kwargs)
    simulation.setup()
    simulation.run()


if __name__ == '__main__':
    run()
