#!/usr/bin/env python

import os
import h5netcdf

from veros import VerosSetup, tools, time, veros_kernel
from veros.variables import Variable, allocate

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_FILES = tools.get_assets('global_1deg', os.path.join(BASE_PATH, 'assets.yml'))


@veros_kernel
def set_forcing_kernel(f1, f2, n1, n2, surface_taux, surface_tauy, taux, tauy, enable_tke, forc_tke_surface, rho_0,
                       t_star, qnec, qnet, forc_temp_surface, temp, tau, cp_0, s_star, forc_salt_surface,
                       t_rest, salt, maskT, dzt, enable_tempsalt_sources, temp_source, divpen_shortwave, qsol):
    from veros.core.operators import numpy as np, update, at

    # linearly interpolate wind stress and shift from MITgcm U/V grid to this grid
    surface_taux = update(surface_taux, at[:-1, :], f1 * taux[1:, :, n1] + f2 * taux[1:, :, n2])
    surface_tauy = update(surface_tauy, at[:, :-1], f1 * tauy[:, 1:, n1] + f2 * tauy[:, 1:, n2])

    if settings.enable_tke:
        forc_tke_surface = update(forc_tke_surface, at[1:-1, 1:-1], np.sqrt((0.5 * (surface_taux[1:-1, 1:-1]
                                                                                            + surface_taux[:-2, 1:-1]) / rho_0) ** 2
                                                                                    + (0.5 * (surface_tauy[1:-1, 1:-1]
                                                                                            + surface_tauy[1:-1, :-2]) / rho_0) ** 2) ** (3. / 2.))

    # W/m^2 K kg/J m^3/kg = K m/s
    t_star_cur = f1 * t_star[..., n1] + f2 * t_star[..., n2]
    qqnec = f1 * qnec[..., n1] + f2 * qnec[..., n2]
    qqnet = f1 * qnet[..., n1] + f2 * qnet[..., n2]
    forc_temp_surface = (qqnet + qqnec * (t_star_cur - temp[..., -1, tau])) * maskT[..., -1] / cp_0 / rho_0
    s_star_cur = f1 * s_star[..., n1] + f2 * s_star[..., n2]
    forc_salt_surface = 1. / t_rest * (s_star_cur - salt[..., -1, tau]) * maskT[..., -1] * dzt[-1]

    # apply simple ice mask
    mask1 = temp[:, :, -1, tau] * maskT[:, :, -1] > -1.8
    mask2 = forc_temp_surface > 0
    ice = np.logical_or(mask1, mask2)
    forc_temp_surface *= ice
    forc_salt_surface *= ice

    # solar radiation
    if settings.enable_tempsalt_sources:
        temp_source = ((f1 * qsol[..., n1, None] + f2 * qsol[..., n2, None])
                                * divpen_shortwave[None, None, :] * ice[..., None]
                                * maskT[..., :] / cp_0 / rho_0)

    return surface_taux, surface_tauy, forc_tke_surface, forc_temp_surface, forc_salt_surface, temp_source


class GlobalOneDegreeSetup(VerosSetup):
    """Global 1 degree model with 115 vertical levels.

    `Adapted from pyOM2 <https://wiki.zmaw.de/ifm/TO/pyOM2/1x1%20global%20model>`_.
    """

    def set_parameter(self, settings, var_meta, objects):
        """
        set main parameters
        """
        settings.nx = 360
        settings.ny = 160
        settings.nz = 115
        settings.dt_mom = 1800.0
        settings.dt_tracer = 1800.0
        settings.runlen = 10 * settings.dt_tracer

        settings.coord_degree = True
        settings.enable_cyclic_x = True

        settings.congr_epsilon = 1e-10
        settings.congr_max_iterations = 10000

        settings.enable_hor_friction = True
        settings.A_h = 5e4
        settings.enable_hor_friction_cos_scaling = True
        settings.hor_friction_cosPower = 1
        settings.enable_tempsalt_sources = True
        settings.enable_implicit_vert_friction = True

        settings.eq_of_state_type = 5

        # isoneutral
        settings.enable_neutral_diffusion = True
        settings.K_iso_0 = 1000.0
        settings.K_iso_steep = 50.0
        settings.iso_dslope = 0.005
        settings.iso_slopec = 0.005
        settings.enable_skew_diffusion = True

        # tke
        settings.enable_tke = True
        settings.c_k = 0.1
        settings.c_eps = 0.7
        settings.alpha_tke = 30.0
        settings.mxl_min = 1e-8
        settings.tke_mxl_choice = 1
        settings.kappaM_min = 2e-4
        settings.kappaH_min = 2e-5
        settings.enable_kappaH_profile = True
        settings.enable_tke_superbee_advection = True

        # eke
        settings.enable_eke = True
        settings.eke_k_max = 1e4
        settings.eke_c_k = 0.4
        settings.eke_c_eps = 0.5
        settings.eke_cross = 2.
        settings.eke_crhin = 1.0
        settings.eke_lmin = 100.0
        settings.enable_eke_superbee_advection = True
        settings.enable_eke_isopycnal_diffusion = True

        # idemix
        settings.enable_idemix = False
        settings.enable_eke_diss_surfbot = True
        settings.eke_diss_surfbot_frac = 0.2
        settings.enable_idemix_superbee_advection = True
        settings.enable_idemix_hor_diffusion = True

        # custom variables
        objects.nmonths = 12
        var_meta.update(
            t_star=Variable('t_star', ('xt', 'yt', 'nmonths'), '', '', time_dependent=False),
            s_star=Variable('s_star', ('xt', 'yt', 'nmonths'), '', '', time_dependent=False),
            qnec=Variable('qnec', ('xt', 'yt', 'nmonths'), '', '', time_dependent=False),
            qnet=Variable('qnet', ('xt', 'yt', 'nmonths'), '', '', time_dependent=False),
            qsol=Variable('qsol', ('xt', 'yt', 'nmonths'), '', '', time_dependent=False),
            divpen_shortwave=Variable('divpen_shortwave', ('zt',), '', '', time_dependent=False),
            taux=Variable('taux', ('xt', 'yt', 'nmonths'), '', '', time_dependent=False),
            tauy=Variable('tauy', ('xt', 'yt', 'nmonths'), '', '', time_dependent=False),
        )

    def _read_forcing(self, vs, var):
        from veros.core.operators import numpy as np
        with h5netcdf.File(DATA_FILES['forcing'], 'r') as infile:
            var = infile.variables[var]
            return np.array(var, dtype=str(var.dtype)).T

    def set_grid(self, vs):
        from veros.core.operators import update, at
        dz_data = self._read_forcing(vs, 'dz')
        vs.dzt = update(vs.dzt, at[...], dz_data[::-1])
        vs.dxt = update(vs.dxt, at[...], 1.0)
        vs.dyt = update(vs.dyt, at[...], 1.0)
        vs.y_origin = -79.
        vs.x_origin = 91.

    def set_coriolis(self, vs):
        from veros.core.operators import numpy as np, update, at
        vs.coriolis_t = update(vs.coriolis_t, at[...], 2 * vs.omega * np.sin(vs.yt[np.newaxis, :] / 180. * vs.pi))

    def set_topography(self, vs):
        import numpy as onp
        from veros.core.operators import numpy as np, update, update_multiply, at
        bathymetry_data = self._read_forcing(vs, 'bathymetry')
        salt_data = self._read_forcing(vs, 'salinity')[:, :, ::-1]

        mask_salt = salt_data == 0.
        vs.kbot = update(vs.kbot, at[2:-2, 2:-2], 1 + np.sum(mask_salt.astype('int'), axis=2))

        mask_bathy = bathymetry_data == 0
        vs.kbot = update_multiply(vs.kbot, at[2:-2, 2:-2], ~mask_bathy)
        vs.kbot = vs.kbot * (vs.kbot < vs.nz)

        # close some channels
        i, j = onp.indices((vs.nx, vs.ny))

        mask_channel = (i >= 207) & (i < 214) & (j < 5)  # i = 208,214; j = 1,5
        vs.kbot = update_multiply(vs.kbot, at[2:-2, 2:-2], ~mask_channel)

        # Aleutian islands
        mask_channel = (i == 104) & (j == 134)  # i = 105; j = 135
        vs.kbot = update_multiply(vs.kbot, at[2:-2, 2:-2], ~mask_channel)

        # Engl channel
        mask_channel = (i >= 269) & (i < 271) & (j == 130)  # i = 270,271; j = 131
        vs.kbot = update_multiply(vs.kbot, at[2:-2, 2:-2], ~mask_channel)

    def set_initial_conditions(self, vs):
        from veros.core.operators import numpy as np, update, at
        rpart_shortwave = 0.58
        efold1_shortwave = 0.35
        efold2_shortwave = 23.0

        # initial conditions
        temp_data = self._read_forcing(vs, 'temperature')
        vs.temp = update(vs.temp, at[2:-2, 2:-2, :, 0], temp_data[..., ::-1] * vs.maskT[2:-2, 2:-2, :])
        vs.temp = update(vs.temp, at[2:-2, 2:-2, :, 1], temp_data[..., ::-1] * vs.maskT[2:-2, 2:-2, :])

        salt_data = self._read_forcing(vs, 'salinity')
        vs.salt = update(vs.salt, at[2:-2, 2:-2, :, 0], salt_data[..., ::-1] * vs.maskT[2:-2, 2:-2, :])
        vs.salt = update(vs.salt, at[2:-2, 2:-2, :, 1], salt_data[..., ::-1] * vs.maskT[2:-2, 2:-2, :])

        # wind stress on MIT grid
        vs.taux = update(vs.taux, at[2:-2, 2:-2, :], self._read_forcing(vs, 'tau_x'))
        vs.tauy = update(vs.tauy, at[2:-2, 2:-2, :], self._read_forcing(vs, 'tau_y'))

        qnec_data = self._read_forcing(vs, 'dqdt')
        vs.qnec = update(vs.qnec, at[2:-2, 2:-2, :], qnec_data * vs.maskT[2:-2, 2:-2, -1, np.newaxis])

        qsol_data = self._read_forcing(vs, 'swf')
        vs.qsol = update(vs.qsol, at[2:-2, 2:-2, :], -qsol_data * vs.maskT[2:-2, 2:-2, -1, np.newaxis])

        # SST and SSS
        sst_data = self._read_forcing(vs, 'sst')
        vs.t_star = update(vs.t_star, at[2:-2, 2:-2, :], sst_data * vs.maskT[2:-2, 2:-2, -1, np.newaxis])

        sss_data = self._read_forcing(vs, 'sss')
        vs.s_star = update(vs.s_star, at[2:-2, 2:-2, :], sss_data * vs.maskT[2:-2, 2:-2, -1, np.newaxis])

        if settings.enable_idemix:
            tidal_energy_data = self._read_forcing(vs, 'tidal_energy')
            mask = np.maximum(0, vs.kbot[2:-2, 2:-2] - 1)[:, :, np.newaxis] == np.arange(vs.nz)[np.newaxis, np.newaxis, :]
            tidal_energy_data *= vs.maskW[2:-2, 2:-2, :][mask].reshape(vs.nx, vs.ny) / vs.rho_0
            vs.forc_iw_bottom = update(vs.forc_iw_bottom, at[2:-2, 2:-2], tidal_energy_data)

            wind_energy_data = self._read_forcing(vs, 'wind_energy')
            wind_energy_data *= vs.maskW[2:-2, 2:-2, -1] / vs.rho_0 * 0.2
            vs.forc_iw_surface = update(vs.forc_iw_surface, at[2:-2, 2:-2], wind_energy_data)

        """
        Initialize penetration profile for solar radiation and store divergence in divpen
        note that pen is set to 0.0 at the surface instead of 1.0 to compensate for the
        shortwave part of the total surface flux
        """
        swarg1 = vs.zw / efold1_shortwave
        swarg2 = vs.zw / efold2_shortwave
        pen = rpart_shortwave * np.exp(swarg1) + (1.0 - rpart_shortwave) * np.exp(swarg2)
        pen = update(pen, at[-1], 0.)
        vs.divpen_shortwave = allocate(vs, ('zt',))
        vs.divpen_shortwave = update(vs.divpen_shortwave, at[1:], (pen[1:] - pen[:-1]) / vs.dzt[1:])
        vs.divpen_shortwave = update(vs.divpen_shortwave, at[0], pen[0] / vs.dzt[0])

    def set_forcing(self, vs):
        t_rest = 30. * 86400.
        cp_0 = 3991.86795711963  # J/kg /K

        year_in_seconds = time.convert_time(1., 'years', 'seconds')
        (n1, f1), (n2, f2) = tools.get_periodic_interval(vs.time, year_in_seconds,
                                                         year_in_seconds / 12., 12)

        (
            vs.surface_taux, vs.surface_tauy, vs.forc_tke_surface, vs.forc_temp_surface,
            vs.forc_salt_surface, vs.temp_source
        ) = run_kernel(set_forcing_kernel, vs, n1=n1, f1=f1, n2=n2, f2=f2, t_rest=t_rest, cp_0=cp_0)

    def set_diagnostics(self, vs):
        vs.diagnostics = {}

    def after_timestep(self, vs):
        pass


@tools.cli
def run(*args, **kwargs):
    simulation = GlobalOneDegreeSetup(*args, **kwargs)
    simulation.setup()
    simulation.run()


if __name__ == '__main__':
    run()
