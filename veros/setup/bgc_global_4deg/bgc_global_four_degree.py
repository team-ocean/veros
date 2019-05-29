#!/usr/bin/env python

import os
import logging
from netCDF4 import Dataset

import veros
import veros.tools
import ruamel.yaml as yaml

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_FILES = veros.tools.get_assets("bgc_global_4deg", os.path.join(BASE_PATH, "assets.yml"))


class GlobalFourDegreeBGC(veros.Veros):
    """Global 4 degree model with 15 vertical levels and biogeochemistry.
    """
    collapse_AMOC = False

    @veros.veros_method
    def set_parameter(self):
        self.identifier = "4deg"

        self.nx, self.ny, self.nz = 90, 40, 15
        self.dt_mom = 1800
        self.dt_tracer = 86400.0
        self.dt_bio = self.dt_tracer // 4
        self.runlen = 200 * 360 * self.dt_tracer
        self.trcmin = 0

        self.enable_npzd = True
        self.enable_nitrogen = False
        self.enable_calcifiers = False
        self.enable_carbon = True
        self.enable_iron = False
        self.enable_oxygen = False

        with open(os.path.join(BASE_PATH, "npzd.yml")) as yaml_file:
            cfg = yaml.safe_load(yaml_file)["npzd"]
            self.npzd_selected_rules = cfg["selected_rules"]

        self.kappaH_min = 9e-5 # NOTE usually 2e-5

        self.nud0 = 0.07 / 86400
        self.bbio = 1.038
        self.cbio = 1.0
        self.abio_P = 0.27 / 86400
        self.gbio = 0.13 / 86400
        self.nupt0 = 0.027 / 86400
        self.specific_mortality_phytoplankton = 0.03 / 86400
        self.quadric_mortality_zooplankton = 0.06 / 86400
        self.zooplankton_growth_efficiency = 0.70
        self.assimilation_efficiency = 0.6
        self.wd0 = 2 / 86400
        self.mwz = 1000
        self.mw = 0.02 / 86400
        self.dcaco3 = 2500

        self.coord_degree = True
        self.enable_cyclic_x = True

        self.congr_epsilon = 1e-8
        self.congr_max_iterations = 20000

        self.enable_neutral_diffusion = True
        self.K_iso_0 = 1000.0
        self.K_iso_steep = 500.0
        self.iso_dslope = 0.001
        self.iso_slopec = 0.001
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
        self.kappaM_min = 2e-4
        # self.kappaH_min =7e-5 # NOTE usually 2e-5
        self.enable_tke_superbee_advection = True
        self.enable_Prandtl_tke = False

        # eke
        self.K_gm_0 = 1000.0
        self.enable_eke = False
        self.eke_k_max = 1e4
        self.eke_c_k = 0.4
        self.eke_c_eps = 0.5
        self.eke_cross = 2.
        self.eke_crhin = 1.0
        self.eke_lmin = 100.0
        self.enable_eke_superbee_advection = False
        self.enable_eke_isopycnal_diffusion = False

        # idemix
        self.enable_idemix = False
        self.enable_idemix_hor_diffusion = False
        self.enable_eke_diss_surfbot = False
        self.eke_diss_surfbot_frac = 0.2  #  fraction which goes into bottom
        self.enable_idemix_superbee_advection = False

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
        self.y_origin = -78.0
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

        if self.enable_npzd:
            with Dataset(DATA_FILES["corev2"], "r") as infile:
                self.swr_initial = infile.variables["SWDN_MOD"][...].T

            phytoplankton = 0.14 * np.exp(self.zw) * self.maskT
            zooplankton = 0.014 * np.exp(self.zw) * self.maskT

            self.phytoplankton[:, :, :, :] = phytoplankton[..., np.newaxis]
            self.zooplankton[:, :, :, :] = zooplankton[..., np.newaxis]
            self.detritus[:, :, :, :] = 1e-4 * self.maskT[..., np.newaxis]

            self.po4[:, :, :, :] = 2.2
            self.po4[:, :, -1, :] = 0.5
            self.po4[...] *= self.maskT[..., np.newaxis]

        if self.enable_nitrogen:
            self.diazotroph[...] = self.phytoplankton / 10
            self.no3[...] = self.po4[...] * 10
            self.don[...] = 20 * self.maskT[..., np.newaxis]
            self.dop[...] = 20 * self.maskT[..., np.newaxis]

        if self.enable_carbon:
            self.dic[...] = 2300
            self.atmospheric_co2[...] = 280
            self.alkalinity[...] = 2400

            self.hSWS[...] = 5e-7

        if self.enable_calcifiers:
            self.coccolitophore[...] = self.phytoplankton / 10
            self.caco3[...] = 0.01 * self.maskT[..., np.newaxis]

        if self.enable_iron:
            self.fe[2:-2, 2:-2, -2:] = 8
            self.particulate_fe[2:-2, 2:-2, -2:] = 9

        if self.enable_oxygen:
            self.o2[2:-2, 2:-2, -2:] = 20

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
            self.forc_tke_surface[1:-1, 1:-1] = np.sqrt((0.5 * (self.surface_taux[1:-1, 1:-1]
                                                                + self.surface_taux[:-2, 1:-1]))**2
                                                        + (0.5 * (self.surface_tauy[1:-1, 1:-1]
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

        # NOTE this is used to collapse the AMOC. Should only be enabled if you wish to collapse it
        if self.collapse_AMOC:
            east_AMOC_border = (self.xt <= 50) + (self.xt >= 260)
            north_AMOC_border = self.yt >= 50
            AMOC_border = east_AMOC_border.reshape(-1, 1) * north_AMOC_border.reshape(1, -1)
            sss[:] += -1 * AMOC_border

        self.forc_salt_surface[:] = 1. / t_rest * \
            (sss - self.salt[:, :, -1, self.tau]) * self.maskT[:, :, -1] * self.dzt[-1]

        # apply simple ice mask
        mask = np.logical_and(self.temp[:, :, -1, self.tau] * self.maskT[:, :, -1] < -1.8,
                              self.forc_temp_surface < 0.)
        self.forc_temp_surface[mask] = 0.0
        self.forc_salt_surface[mask] = 0.0

        if self.enable_npzd:
            # incomming shortwave radiation for plankton production
            self.swr[2:-2, 2:-2] = (f1 * self.swr_initial[:, :, n1] + f2 * self.swr_initial[:, :, n2])

    @veros.veros_method
    def set_diagnostics(self):
        self.diagnostics["snapshot"].output_frequency = 360 * 86400

        self.diagnostics["averages"].output_variables = ["phytoplankton", "po4", "zooplankton", "detritus", "wind_speed", "dic", "alkalinity",
                                                         "temp", "salt", "u", "v", "surface_tauy", "surface_taux", "kappaH", "psi", "rho", "pCO2", "dco2star", "cflux", "detritus_export"]

        self.diagnostics["averages"].output_frequency = 360 * 86400
        self.diagnostics["averages"].sampling_frequency = self.dt_tracer * 10

        self.diagnostics["overturning"].output_frequency = 360 * 86400
        self.diagnostics["overturning"].sampling_frequency = self.dt_tracer * 10

        self.diagnostics["npzd"].output_frequency = 10 * 86400
        self.diagnostics["npzd"].save_graph = False
        self.diagnostics["npzd"].surface_out = ["po4", "phytoplankton"]

    @veros.veros_method
    def after_timestep(self):
        pass


@veros.tools.cli
def run(*args, **kwargs):
    simulation = GlobalFourDegreeBGC(*args, **kwargs)
    simulation.setup()
    simulation.run()


if __name__ == "__main__":
    run()
