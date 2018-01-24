#!/usr/bin/env python

import os
import logging
from netCDF4 import Dataset

import veros
import veros.tools

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_FILES = veros.tools.get_assets("global_4deg", os.path.join(BASE_PATH, "assets.yml"))


class GlobalFourDegree(veros.Veros):
    """Global 4 degree model with 15 vertical levels.

    This setup demonstrates:
     - setting up a realistic model
     - reading input data from external files
     - implementing surface forcings
     - applying a simple ice mask

    `Adapted from pyOM2 <https://wiki.cen.uni-hamburg.de/ifm/TO/pyOM2/4x4%20global%20model>`_.
    """
    @veros.veros_method
    def set_parameter(self):
        self.identifier = "4deg"

        self.nx, self.ny, self.nz = 90, 40, 15
        self.dt_mom = 1800.0
        self.dt_tracer = 86400.0
        self.runlen = 0.

        self.coord_degree = True
        self.enable_cyclic_x = True

        self.congr_epsilon = 1e-8
        self.congr_max_iterations = 20000

        self.enable_neutral_diffusion = True
        self.K_iso_0 = 1000.0
        self.K_iso_steep = 1000.0
        self.iso_dslope = 4. / 1000.0
        self.iso_slopec = 1. / 1000.0
        self.enable_skew_diffusion = True

        self.enable_hor_friction = True
        self.A_h = (4 * self.degtom)**3 * 2e-11
        self.enable_hor_friction_cos_scaling = True
        self.hor_friction_cosPower = 1

        self.enable_implicit_vert_friction = True
        self.enable_tke = True
        self.c_k = 0.1
        self.c_eps = 0.7
        self.alpha_tke = 30.0
        self.mxl_min = 1e-8
        self.tke_mxl_choice = 2
        self.enable_tke_superbee_advection = True

        self.enable_eke = True
        self.eke_k_max = 1e4
        self.eke_c_k = 0.4
        self.eke_c_eps = 0.5
        self.eke_cross = 2.
        self.eke_crhin = 1.0
        self.eke_lmin = 100.0
        self.enable_eke_superbee_advection = True

        self.enable_idemix = True
        self.enable_idemix_hor_diffusion = True
        self.enable_eke_diss_surfbot = True
        self.eke_diss_surfbot_frac = 0.2
        self.enable_idemix_superbee_advection = True

        self.eq_of_state_type = 5

    @veros.veros_method
    def _read_forcing(self, var):
        with Dataset(DATA_FILES["forcing"], "r") as infile:
            return infile.variables[var][...].T

    @veros.veros_method
    def set_grid(self):
        ddz = np.array([50., 70., 100., 140., 190., 240., 290., 340.,
                        390., 440., 490., 540., 590., 640., 690.])
        self.dzt[:] = ddz[::-1]
        self.dxt[:] = 4.0
        self.dyt[:] = 4.0
        self.y_origin = -76.0
        self.x_origin = 4.0

    @veros.veros_method
    def set_coriolis(self):
        self.coriolis_t[...] = 2 * self.omega * np.sin(self.yt[np.newaxis, :] / 180. * self.pi)

    @veros.veros_method
    def set_topography(self):
        bathymetry_data = self._read_forcing("bathymetry")
        salt_data = self._read_forcing("salinity")[:, :, ::-1]
        mask_salt = salt_data == 0.
        self.kbot[2:-2, 2:-2] = 1 + np.sum(mask_salt.astype(np.int), axis=2)
        mask_bathy = bathymetry_data == 0
        self.kbot[2:-2, 2:-2][mask_bathy] = 0
        self.kbot[self.kbot == self.nz] = 0

    @veros.veros_method
    def set_initial_conditions(self):
        self.taux, self.tauy, self.qnec, self.qnet, self.sss_clim, self.sst_clim = (
            np.zeros((self.nx + 4, self.ny + 4, 12), dtype=self.default_float_type) for _ in range(6))

        # initial conditions for T and S
        temp_data = self._read_forcing("temperature")[:, :, ::-1]
        self.temp[2:-2, 2:-2, :, :2] = temp_data[:, :, :, np.newaxis] * \
            self.maskT[2:-2, 2:-2, :, np.newaxis]

        salt_data = self._read_forcing("salinity")[:, :, ::-1]
        self.salt[2:-2, 2:-2, :, :2] = salt_data[..., np.newaxis] * self.maskT[2:-2, 2:-2, :, np.newaxis]

        # use Trenberth wind stress from MITgcm instead of ECMWF (also contained in ecmwf_4deg.cdf)
        self.taux[2:-2, 2:-2, :] = self._read_forcing("tau_x") / self.rho_0
        self.tauy[2:-2, 2:-2, :] = self._read_forcing("tau_y") / self.rho_0

        # heat flux
        with Dataset(DATA_FILES["ecmwf"], "r") as ecmwf_data:
            self.qnec[2:-2, 2:-2, :] = np.array(ecmwf_data.variables["Q3"]).transpose()
            self.qnec[self.qnec <= -1e10] = 0.0

        q = self._read_forcing("q_net")
        self.qnet[2:-2, 2:-2, :] = -q
        self.qnet[self.qnet <= -1e10] = 0.0

        fxa = np.sum(self.qnet[2:-2, 2:-2, :] * self.area_t[2:-2, 2:-2, np.newaxis]) \
              / 12 / np.sum(self.area_t[2:-2, 2:-2])
        logging.info(" removing an annual mean heat flux imbalance of %e W/m^2" % fxa)
        self.qnet[...] = (self.qnet - fxa) * self.maskT[:, :, -1, np.newaxis]

        # SST and SSS
        self.sst_clim[2:-2, 2:-2, :] = self._read_forcing("sst")
        self.sss_clim[2:-2, 2:-2, :] = self._read_forcing("sss")

        if self.enable_idemix:
            self.forc_iw_bottom[2:-2, 2:-2] = self._read_forcing("tidal_energy") / self.rho_0
            self.forc_iw_surface[2:-2, 2:-2] = self._read_forcing("wind_energy") / self.rho_0 * 0.2

    @veros.veros_method
    def set_forcing(self):
        year_in_seconds = 360 * 86400.
        (n1, f1), (n2, f2) = veros.tools.get_periodic_interval(
            self.time, year_in_seconds, year_in_seconds / 12., 12
        )

        # wind stress
        self.surface_taux[...] = (f1 * self.taux[:, :, n1] + f2 * self.taux[:, :, n2])
        self.surface_tauy[...] = (f1 * self.tauy[:, :, n1] + f2 * self.tauy[:, :, n2])

        # tke flux
        if self.enable_tke:
            self.forc_tke_surface[1:-1, 1:-1] = np.sqrt((0.5 * (self.surface_taux[1:-1, 1:-1] \
                                                                + self.surface_taux[:-2, 1:-1]))**2
                                                      + (0.5 * (self.surface_tauy[1:-1, 1:-1] \
                                                                + self.surface_tauy[1:-1, :-2]))**2)**(3. / 2.)
        # heat flux : W/m^2 K kg/J m^3/kg = K m/s
        cp_0 = 3991.86795711963
        sst = f1 * self.sst_clim[:, :, n1] + f2 * self.sst_clim[:, :, n2]
        qnec = f1 * self.qnec[:, :, n1] + f2 * self.qnec[:, :, n2]
        qnet = f1 * self.qnet[:, :, n1] + f2 * self.qnet[:, :, n2]
        self.forc_temp_surface[...] = (qnet + qnec * (sst - self.temp[:, :, -1, self.tau])) \
                                       * self.maskT[:, :, -1] / cp_0 / self.rho_0

        # salinity restoring
        t_rest = 30 * 86400.0
        sss = f1 * self.sss_clim[:, :, n1] + f2 * self.sss_clim[:, :, n2]
        self.forc_salt_surface[:] = 1. / t_rest * \
            (sss - self.salt[:, :, -1, self.tau]) * self.maskT[:, :, -1] * self.dzt[-1]

        # apply simple ice mask
        mask = np.logical_and(self.temp[:, :, -1, self.tau] * self.maskT[:, :, -1] < -1.8,
                              self.forc_temp_surface < 0.)
        self.forc_temp_surface[mask] = 0.0
        self.forc_salt_surface[mask] = 0.0

    @veros.veros_method
    def set_diagnostics(self):
        self.diagnostics["cfl_monitor"].output_frequency = 360 * 86400. / 24.
        self.diagnostics["snapshot"].output_frequency = 360 * 86400. / 24.
        self.diagnostics["overturning"].output_frequency = 360 * 86400. / 24.
        self.diagnostics["overturning"].sampling_frequency = self.dt_tracer
        self.diagnostics["energy"].output_frequency = 360 * 86400. / 24.
        self.diagnostics["energy"].sampling_frequency = 86400
        average_vars = ["temp", "salt", "u", "v", "w", "surface_taux",
                        "surface_tauy", "psi"]
        self.diagnostics["averages"].output_variables = average_vars
        self.diagnostics["averages"].output_frequency = 86400. * 30
        self.diagnostics["averages"].sampling_frequency = 86400

    @veros.veros_method
    def after_timestep(self):
        pass


@veros.tools.cli
def run(*args, **kwargs):
    simulation = GlobalFourDegree(*args, **kwargs)
    simulation.setup()
    simulation.run()


if __name__ == "__main__":
    run()
