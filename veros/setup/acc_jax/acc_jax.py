#!/usr/bin/env python

from veros import VerosSetup
from veros.tools import cli, get_vinokur_grid_steps
from veros.variables import allocate
from veros.distributed import global_min, global_max


class ACCSetup(VerosSetup):
    """A model using spherical coordinates with a partially closed domain representing the Atlantic and ACC.

    Wind forcing over the channel part and buoyancy relaxation drive a large-scale meridional overturning circulation.

    This setup demonstrates:
     - setting up an idealized geometry
     - updating surface forcings
     - basic usage of diagnostics

    `Adapted from pyOM2 <https://wiki.cen.uni-hamburg.de/ifm/TO/pyOM2/ACC%202>`_.
    """
    def set_parameter(self, vs):
        vs.identifier = 'acc'

        vs.nx, vs.ny, vs.nz = 180, 360, 80
        vs.dt_mom = 60
        vs.dt_tracer = 60
        vs.runlen = vs.dt_tracer * 10

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
        vs.A_h = (2 * vs.degtom)**3 * 2e-11
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
        vs.tke_mxl_choice = 1
        vs.kappaM_min = 2e-4
        vs.kappaH_min = 2e-5
        vs.enable_kappaH_profile = True
        # vs.enable_tke_superbee_advection = True

        vs.K_gm_0 = 1000.0
        vs.enable_eke = True
        vs.eke_k_max = 1e4
        vs.eke_c_k = 0.4
        vs.eke_c_eps = 0.5
        vs.eke_cross = 2.
        vs.eke_crhin = 1.0
        vs.eke_lmin = 100.0
        vs.enable_eke_superbee_advection = True
        vs.enable_eke_isopycnal_diffusion = True

        vs.enable_idemix = False

        vs.eq_of_state_type = 5

        vs.diskless_mode = True

    def set_grid(self, vs):
        from veros.core.operators import numpy as np, update, at
        vs.dzt = np.array(get_vinokur_grid_steps(vs.nz, 4000, 10, refine_towards='lower'))
        vs.dxt = update(vs.dxt, at[...], .2)
        vs.dyt = update(vs.dyt, at[...], .2)
        vs.x_origin = 0.0
        vs.y_origin = -40.0

    def set_coriolis(self, vs):
        from veros.core.operators import numpy as np, update, at
        vs.coriolis_t = update(vs.coriolis_t, at[...], 2 * vs.omega * np.sin(vs.yt[None, :] / 180. * vs.pi))

    def set_topography(self, vs):
        from veros.core.operators import numpy as np, update, at
        x, y = np.meshgrid(vs.xt, vs.yt, indexing='ij')
        vs.kbot = update(vs.kbot, at[...], np.logical_or(x > 1.0, y < -20).astype('int'))

    def set_initial_conditions(self, vs):
        from veros.core.operators import numpy as np, update, at
        # initial conditions
        vs.temp = update(vs.temp, at[:, :, :, 0:2], ((1 - vs.zt[None, None, :] / vs.zw[0]) * 15 * vs.maskT)[..., None])
        vs.salt = update(vs.salt, at[:, :, :, 0:2], 35.0 * vs.maskT[..., None])

        # wind stress forcing
        yt_min = global_min(vs.yt.min())
        yu_min = global_min(vs.yu.min())
        yt_max = global_max(vs.yt.max())
        yu_max = global_max(vs.yu.max())

        mask_south = vs.yt < -20
        mask_north = vs.yt > 10
        taux = (
            mask_south * (1e-4 * np.sin(vs.pi * (vs.yu - yu_min) / (-20.0 - yt_min)))
            + mask_north * (1e-4 * (1 - np.cos(2 * vs.pi * (vs.yu - 10.0) / (yu_max - 10.0))))
        )
        vs.surface_taux = taux * vs.maskU[:, :, -1]

        # surface heatflux forcing
        mask_south = vs.yt < -20
        mask_north = vs.yt > 20
        vs._t_star = (
            15 * (~mask_south & ~mask_north)
            + mask_south * (15 * (vs.yt - yt_min) / (-20 - yt_min))
            + mask_north * 15 * (1 - (vs.yt - 20) / (yt_max - 20))
        )
        vs._t_rest = vs.dzt[None, -1] / (30. * 86400.) * vs.maskT[:, :, -1]

        if vs.enable_tke:
            vs.forc_tke_surface = update(vs.forc_tke_surface, at[2:-2, 2:-2],
                np.sqrt((0.5 * (vs.surface_taux[2:-2, 2:-2] + vs.surface_taux[1:-3, 2:-2]) / vs.rho_0)**2
                + (0.5 * (vs.surface_tauy[2:-2, 2:-2] + vs.surface_tauy[2:-2, 1:-3]) / vs.rho_0)**2)**(1.5)
            )

    def set_forcing(self, vs):
        from veros.core.operators import update, at
        vs.forc_temp_surface = update(vs.forc_temp_surface, at[...], vs._t_rest * (vs._t_star - vs.temp[:, :, -1, vs.tau]))

    def set_diagnostics(self, vs):
        vs.diagnostics = {}

    def after_timestep(self, vs):
        pass


@cli
def run(*args, **kwargs):
    simulation = ACCSetup(*args, **kwargs)
    simulation.setup()
    simulation.run()


if __name__ == '__main__':
    run()
