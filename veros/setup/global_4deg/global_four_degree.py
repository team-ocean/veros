#!/usr/bin/env python

import os

import h5netcdf

from veros import VerosSetup, veros_method
from veros.variables import Variable
import veros.tools

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_FILES = veros.tools.get_assets(
    'global_4deg',
    os.path.join(BASE_PATH, 'assets.yml')
)


class GlobalFourDegreeSetup(VerosSetup):
    """Global 4 degree model with 15 vertical levels.

    This setup demonstrates:
     - setting up a realistic model
     - reading input data from external files
     - implementing surface forcings
     - applying a simple ice mask

    `Adapted from pyOM2 <https://wiki.cen.uni-hamburg.de/ifm/TO/pyOM2/4x4%20global%20model>`_.
    """
    @veros_method
    def set_parameter(self, vs):
        vs.identifier = '4deg'

        vs.nx, vs.ny, vs.nz = 90, 40, 15
        vs.dt_mom = 1800.0
        vs.dt_tracer = 86400.0
        vs.runlen = 0.

        vs.coord_degree = True
        vs.enable_cyclic_x = True

        vs.congr_epsilon = 1e-8
        vs.congr_max_iterations = 20000

        vs.enable_neutral_diffusion = True
        vs.K_iso_0 = 1000.0
        vs.K_iso_steep = 1000.0
        vs.iso_dslope = 4. / 1000.0
        vs.iso_slopec = 1. / 1000.0
        vs.enable_skew_diffusion = True

        vs.enable_hor_friction = True
        vs.A_h = (4 * vs.degtom)**3 * 2e-11
        vs.enable_hor_friction_cos_scaling = True
        vs.hor_friction_cosPower = 1

        vs.enable_implicit_vert_friction = True
        vs.enable_tke = True
        vs.c_k = 0.1
        vs.c_eps = 0.7
        vs.alpha_tke = 30.0
        vs.mxl_min = 1e-8
        vs.tke_mxl_choice = 2
        vs.enable_tke_superbee_advection = True

        vs.enable_eke = True
        vs.eke_k_max = 1e4
        vs.eke_c_k = 0.4
        vs.eke_c_eps = 0.5
        vs.eke_cross = 2.
        vs.eke_crhin = 1.0
        vs.eke_lmin = 100.0
        vs.enable_eke_superbee_advection = True

        vs.enable_idemix = False
        vs.enable_idemix_hor_diffusion = True
        vs.enable_eke_diss_surfbot = True
        vs.eke_diss_surfbot_frac = 0.2
        vs.enable_idemix_superbee_advection = True

        vs.eq_of_state_type = 5

        # custom variables
        vs.nmonths = 12
        vs.variables.update(
            sss_clim=Variable('sss_clim', ('xt', 'yt', 'nmonths'), '', '', time_dependent=False),
            sst_clim=Variable('sst_clim', ('xt', 'yt', 'nmonths'), '', '', time_dependent=False),
            qnec=Variable('qnec', ('xt', 'yt', 'nmonths'), '', '', time_dependent=False),
            qnet=Variable('qnet', ('xt', 'yt', 'nmonths'), '', '', time_dependent=False),
            taux=Variable('taux', ('xt', 'yt', 'nmonths'), '', '', time_dependent=False),
            tauy=Variable('tauy', ('xt', 'yt', 'nmonths'), '', '', time_dependent=False),
        )

    @veros_method
    def _read_forcing(self, vs, var):
        with h5netcdf.File(DATA_FILES['forcing'], 'r') as infile:
            var_obj = infile.variables[var]
            return np.array(var_obj, dtype=str(var_obj.dtype)).T

    @veros_method
    def set_grid(self, vs):
        ddz = np.array([50., 70., 100., 140., 190., 240., 290., 340.,
                        390., 440., 490., 540., 590., 640., 690.])
        vs.dzt[:] = ddz[::-1]
        vs.dxt[:] = 4.0
        vs.dyt[:] = 4.0
        vs.y_origin = -76.0
        vs.x_origin = 4.0

    @veros_method
    def set_coriolis(self, vs):
        vs.coriolis_t[...] = 2 * vs.omega * np.sin(vs.yt[np.newaxis, :] / 180. * vs.pi)

    @veros_method(dist_safe=False, local_variables=[
        'kbot'
    ])
    def set_topography(self, vs):
        bathymetry_data = self._read_forcing(vs, 'bathymetry')
        salt_data = self._read_forcing(vs, 'salinity')[:, :, ::-1]
        mask_salt = salt_data == 0.
        vs.kbot[2:-2, 2:-2] = 1 + np.sum(mask_salt.astype(np.int), axis=2)
        mask_bathy = bathymetry_data == 0
        vs.kbot[2:-2, 2:-2][mask_bathy] = 0
        vs.kbot[vs.kbot == vs.nz] = 0

    @veros_method(dist_safe=False, local_variables=[
        'taux', 'tauy', 'qnec', 'qnet', 'sss_clim', 'sst_clim',
        'temp', 'salt', 'taux', 'tauy', 'area_t', 'maskT',
        'forc_iw_bottom', 'forc_iw_surface'
    ])
    def set_initial_conditions(self, vs):
        # initial conditions for T and S
        temp_data = self._read_forcing(vs, 'temperature')[:, :, ::-1]
        vs.temp[2:-2, 2:-2, :, :2] = temp_data[:, :, :, np.newaxis] * \
            vs.maskT[2:-2, 2:-2, :, np.newaxis]

        salt_data = self._read_forcing(vs, 'salinity')[:, :, ::-1]
        vs.salt[2:-2, 2:-2, :, :2] = salt_data[..., np.newaxis] * vs.maskT[2:-2, 2:-2, :, np.newaxis]

        # use Trenberth wind stress from MITgcm instead of ECMWF (also contained in ecmwf_4deg.cdf)
        vs.taux[2:-2, 2:-2, :] = self._read_forcing(vs, 'tau_x')
        vs.tauy[2:-2, 2:-2, :] = self._read_forcing(vs, 'tau_y')

        # heat flux
        with h5netcdf.File(DATA_FILES['ecmwf'], 'r') as ecmwf_data:
            qnec_var = ecmwf_data.variables['Q3']
            vs.qnec[2:-2, 2:-2, :] = np.array(qnec_var, dtype=str(qnec_var.dtype)).transpose()
            vs.qnec[vs.qnec <= -1e10] = 0.0

        q = self._read_forcing(vs, 'q_net')
        vs.qnet[2:-2, 2:-2, :] = -q
        vs.qnet[vs.qnet <= -1e10] = 0.0

        fxa = np.sum(vs.qnet[2:-2, 2:-2, :] * vs.area_t[2:-2, 2:-2, np.newaxis]) \
              / 12 / np.sum(vs.area_t[2:-2, 2:-2])
        print(' removing an annual mean heat flux imbalance of %e W/m^2' % fxa)
        vs.qnet[...] = (vs.qnet - fxa) * vs.maskT[:, :, -1, np.newaxis]

        # SST and SSS
        vs.sst_clim[2:-2, 2:-2, :] = self._read_forcing(vs, 'sst')
        vs.sss_clim[2:-2, 2:-2, :] = self._read_forcing(vs, 'sss')

        if vs.enable_idemix:
            vs.forc_iw_bottom[2:-2, 2:-2] = self._read_forcing(vs, 'tidal_energy') / vs.rho_0
            vs.forc_iw_surface[2:-2, 2:-2] = self._read_forcing(vs, 'wind_energy') / vs.rho_0 * 0.2

    @veros_method
    def set_forcing(self, vs):
        year_in_seconds = 360 * 86400.
        (n1, f1), (n2, f2) = veros.tools.get_periodic_interval(
            vs.time, year_in_seconds, year_in_seconds / 12., 12
        )

        # wind stress
        vs.surface_taux[...] = (f1 * vs.taux[:, :, n1] + f2 * vs.taux[:, :, n2])
        vs.surface_tauy[...] = (f1 * vs.tauy[:, :, n1] + f2 * vs.tauy[:, :, n2])

        # tke flux
        if vs.enable_tke:
            vs.forc_tke_surface[1:-1, 1:-1] = np.sqrt((0.5 * (vs.surface_taux[1:-1, 1:-1] \
                                                                + vs.surface_taux[:-2, 1:-1]) / vs.rho_0)**2
                                                      + (0.5 * (vs.surface_tauy[1:-1, 1:-1] \
                                                                + vs.surface_tauy[1:-1, :-2]) / vs.rho_0)**2)**(3. / 2.)
        # heat flux : W/m^2 K kg/J m^3/kg = K m/s
        cp_0 = 3991.86795711963
        sst = f1 * vs.sst_clim[:, :, n1] + f2 * vs.sst_clim[:, :, n2]
        qnec = f1 * vs.qnec[:, :, n1] + f2 * vs.qnec[:, :, n2]
        qnet = f1 * vs.qnet[:, :, n1] + f2 * vs.qnet[:, :, n2]
        vs.forc_temp_surface[...] = (qnet + qnec * (sst - vs.temp[:, :, -1, vs.tau])) \
                                       * vs.maskT[:, :, -1] / cp_0 / vs.rho_0

        # salinity restoring
        t_rest = 30 * 86400.0
        sss = f1 * vs.sss_clim[:, :, n1] + f2 * vs.sss_clim[:, :, n2]
        vs.forc_salt_surface[:] = 1. / t_rest * \
            (sss - vs.salt[:, :, -1, vs.tau]) * vs.maskT[:, :, -1] * vs.dzt[-1]

        # apply simple ice mask
        mask = np.logical_and(vs.temp[:, :, -1, vs.tau] * vs.maskT[:, :, -1] < -1.8,
                              vs.forc_temp_surface < 0.)
        vs.forc_temp_surface[mask] = 0.0
        vs.forc_salt_surface[mask] = 0.0

    @veros.veros_method
    def set_diagnostics(self, vs):
        vs.diagnostics['snapshot'].output_frequency = 360 * 86400.
        vs.diagnostics['overturning'].output_frequency = 360 * 86400.
        vs.diagnostics['overturning'].sampling_frequency = vs.dt_tracer
        vs.diagnostics['energy'].output_frequency = 360 * 86400.
        vs.diagnostics['energy'].sampling_frequency = 86400
        average_vars = ['temp', 'salt', 'u', 'v', 'w', 'surface_taux', 'surface_tauy', 'psi']
        vs.diagnostics['averages'].output_variables = average_vars
        vs.diagnostics['averages'].output_frequency = 360 * 86400.
        vs.diagnostics['averages'].sampling_frequency = 86400

    @veros_method
    def after_timestep(self, vs):
        pass


@veros.tools.cli
def run(*args, **kwargs):
    simulation = GlobalFourDegreeSetup(*args, **kwargs)
    simulation.setup()
    simulation.run()


if __name__ == '__main__':
    run()
