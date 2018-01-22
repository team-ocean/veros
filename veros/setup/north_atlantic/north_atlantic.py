#!/usr/bin/env python

import os
from netCDF4 import Dataset
from PIL import Image
import numpy as np
import scipy.spatial
import scipy.ndimage

import veros
import veros.tools

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_FILES = veros.tools.get_assets("north_atlantic", os.path.join(BASE_PATH, "assets.yml"))


class NorthAtlantic(veros.Veros):
    """ A regional model of the North Atlantic, inspired by `Smith et al., 2000`_.

    Forcing and initial conditions are taken from the FLAME PyOM2 setup. Bathymetry
    data from ETOPO1 (resolution of 1 arcmin).

    Boundary forcings are implemented via sponge layers in the Greenland Sea, by the
    Strait of Gibraltar, and in the South Atlantic. This setup runs with arbitrary resolution;
    upon changing the number of grid cells, all forcing files will be interpolated to
    the new grid. Default resolution corresponds roughly to :math:`0.5 \\times 0.25` degrees.

    .. _Smith et al., 2000:
       http://journals.ametsoc.org/doi/10.1175/1520-0485%282000%29030%3C1532%3ANSOTNA%3E2.0.CO%3B2
    """

    @veros.veros_method
    def set_parameter(self):
        self.identifier = "na"

        self.nx, self.ny, self.nz = 250, 350, 50
        self.x_origin = -98.
        self.y_origin = -18.
        self._x_boundary = 17.2
        self._y_boundary = 70.
        self._max_depth = 5800.

        self.dt_mom = 3600. / 2.
        self.dt_tracer = 3600. / 2.
        self.runlen = 0.

        self.runlen = 86400 * 365. * 10.

        self.coord_degree = True

        self.congr_epsilon = 1e-10
        self.congr_max_iterations = 20000

        self.enable_neutral_diffusion = True
        self.enable_skew_diffusion = True
        self.K_iso_0 = 1000.0
        self.K_iso_steep = 200.0
        self.iso_dslope = 1. / 1000.0
        self.iso_slopec = 4. / 1000.0

        self.enable_hor_friction = True
        self.A_h = 1e3
        self.enable_hor_friction_cos_scaling = True
        self.hor_friction_cosPower = 1
        self.enable_tempsalt_sources = True

        self.enable_implicit_vert_friction = True
        self.enable_tke = True
        self.c_k = 0.1
        self.c_eps = 0.7
        self.alpha_tke = 30.0
        self.mxl_min = 1e-8
        self.tke_mxl_choice = 2

        self.K_gm_0 = 1000.0

        self.enable_eke = False
        self.enable_idemix = False
        self.enable_idemix_hor_diffusion = True

        self.eq_of_state_type = 5

    def set_grid(self):
        self.dxt[2:-2] = (self._x_boundary - self.x_origin) / self.nx
        self.dyt[2:-2] = (self._y_boundary - self.y_origin) / self.ny
        self.dzt[...] = veros.tools.get_vinokur_grid_steps(self.nz, self._max_depth, 10., refine_towards="lower")

    def set_coriolis(self):
        self.coriolis_t[:, :] = 2 * self.omega * np.sin(self.yt[np.newaxis, :] / 180. * self.pi)

    def set_topography(self):
        with Dataset(DATA_FILES["topography"], "r") as topography_file:
            topo_x, topo_y, topo_bottom_depth = (
                topography_file.variables[k][...].T for k in ("x", "y", "z"))
        topo_mask = np.flipud(np.asarray(Image.open("topo_mask.png"))).T
        topo_bottom_depth *= 1 - topo_mask
        topo_bottom_depth = scipy.ndimage.gaussian_filter(
            topo_bottom_depth, sigma=(len(topo_x) / self.nx, len(topo_y) / self.ny)
        )
        interp_coords = np.meshgrid(self.xt[2:-2], self.yt[2:-2], indexing="ij")
        interp_coords = np.rollaxis(np.asarray(interp_coords), 0, 3)
        z_interp = scipy.interpolate.interpn((topo_x, topo_y), topo_bottom_depth, interp_coords,
                                             method="nearest", bounds_error=False, fill_value=0)
        self.kbot[2:-2, 2:-2] = np.where(z_interp < 0., 1 + np.argmin(np.abs(z_interp[:, :, np.newaxis]
                                         - self.zt[np.newaxis, np.newaxis, :]), axis=2), 0)
        self.kbot *= self.kbot < self.nz

    def set_initial_conditions(self):
        with Dataset(DATA_FILES["forcing"], "r") as forcing_file:
            t_hor = (self.xt[2:-2], self.yt[2:-2])
            t_grid = (self.xt[2:-2], self.yt[2:-2], self.zt)
            forc_coords = [forcing_file.variables[k][...].T for k in ("xt", "yt", "zt")]
            forc_coords[0][...] += -360
            forc_coords[2][...] = -0.01 * forc_coords[2][::-1]
            temp = veros.tools.interpolate(
                forc_coords, forcing_file.variables["temp_ic"][::-1, ...].T, t_grid, missing_value=-1e20
            )
            self.temp[2:-2, 2:-2, :, self.tau] = self.maskT[2:-2, 2:-2, :] * temp
            salt = 35. + 1000 * veros.tools.interpolate(
                forc_coords, forcing_file.variables["salt_ic"][::-1, ...].T, t_grid, missing_value=-1e20
            )
            self.salt[2:-2, 2:-2, :, self.tau] = self.maskT[2:-2, 2:-2, :] * salt

            self._taux = np.zeros((self.nx + 4, self.ny + 4, 12), dtype=self.default_float_type)
            self._tauy = np.zeros((self.nx + 4, self.ny + 4, 12), dtype=self.default_float_type)
            forc_u_coords_hor = [forcing_file.variables[k][...].T for k in ("xu", "yu")]
            forc_u_coords_hor[0][...] += -360
            for k in range(12):
                self._taux[2:-2, 2:-2, k] = veros.tools.interpolate(
                    forc_u_coords_hor, forcing_file.variables["taux"][k, ...].T, t_hor, missing_value=-1e20
                ) / 10. / self.rho_0
                self._tauy[2:-2, 2:-2, k] = veros.tools.interpolate(
                    forc_u_coords_hor, forcing_file.variables["tauy"][k, ...].T, t_hor, missing_value=-1e20
                ) / 10. / self.rho_0

            # heat flux and salinity restoring
            self._sst_clim = np.zeros((self.nx + 4, self.ny + 4, 12), dtype=self.default_float_type)
            self._sss_clim = np.zeros((self.nx + 4, self.ny + 4, 12), dtype=self.default_float_type)
            self._sst_rest = np.zeros((self.nx + 4, self.ny + 4, 12), dtype=self.default_float_type)
            self._sss_rest = np.zeros((self.nx + 4, self.ny + 4, 12), dtype=self.default_float_type)

            sst_clim, sss_clim, sst_rest, sss_rest = [
                forcing_file.variables[k][...].T for k in ("sst_clim", "sss_clim", "sst_rest", "sss_rest")
            ]
            for k in range(12):
                self._sst_clim[2:-2, 2:-2, k] = veros.tools.interpolate(
                    forc_coords[:-1], sst_clim[..., k], t_hor, missing_value=-1e20)
                self._sss_clim[2:-2, 2:-2, k] = veros.tools.interpolate(
                    forc_coords[:-1], sss_clim[..., k], t_hor, missing_value=-1e20) * 1000 + 35
                self._sst_rest[2:-2, 2:-2, k] = veros.tools.interpolate(
                    forc_coords[:-1], sst_rest[..., k], t_hor, missing_value=-1e20) * 41868.
                self._sss_rest[2:-2, 2:-2, k] = veros.tools.interpolate(
                    forc_coords[:-1], sss_rest[..., k], t_hor, missing_value=-1e20) / 100.

        with Dataset(DATA_FILES["restoring"], "r") as restoring_file:
            rest_coords = [restoring_file.variables[k][...].T for k in ("xt", "yt", "zt")]
            rest_coords[0][...] += -360

            # sponge layers
            self._t_star = np.zeros((self.nx + 4, self.ny + 4, self.nz, 12), dtype=self.default_float_type)
            self._s_star = np.zeros((self.nx + 4, self.ny + 4, self.nz, 12), dtype=self.default_float_type)
            self._rest_tscl = np.zeros((self.nx + 4, self.ny + 4, self.nz), dtype=self.default_float_type)

            self._rest_tscl[2:-2, 2:-2, :] = veros.tools.interpolate(
                rest_coords, restoring_file.variables["tscl"][0, ...].T, t_grid)
            for k in range(12):
                self._t_star[2:-2, 2:-2, :, k] = veros.tools.interpolate(
                    rest_coords, restoring_file.variables["t_star"][k, ...].T, t_grid, missing_value=0.
                )
                self._s_star[2:-2, 2:-2, :, k] = veros.tools.interpolate(
                    rest_coords, restoring_file.variables["s_star"][k, ...].T, t_grid, missing_value=0.
                )

        if self.enable_idemix:
            f = np.load("tidal_energy.npy") / self.rho_0
            self.forc_iw_bottom[2:-2, 2:-2] = veros.tools.interpolate(forc_coords[:-1], f, t_hor)
            f = np.load("wind_energy.npy") / self.rho_0 * 0.2
            self.forc_iw_surface[2:-2, 2:-2] = veros.tools.interpolate(forc_coords[:-1], f, t_hor)

    @veros.veros_method
    def set_forcing(self):
        year_in_seconds = 360 * 86400.0
        (n1, f1), (n2, f2) = veros.tools.get_periodic_interval(self.time, year_in_seconds,
                                                               year_in_seconds / 12., 12)

        self.surface_taux[...] = (f1 * self._taux[:, :, n1] + f2 * self._taux[:, :, n2])
        self.surface_tauy[...] = (f1 * self._tauy[:, :, n1] + f2 * self._tauy[:, :, n2])

        if self.enable_tke:
            self.forc_tke_surface[1:-1, 1:-1] = np.sqrt((0.5 * (self.surface_taux[1:-1, 1:-1] + self.surface_taux[:-2, 1:-1]))**2
                                                        + (0.5 * (self.surface_tauy[1:-1, 1:-1] + self.surface_tauy[1:-1, :-2]))**2
                                                        ) ** (3. / 2.)
        cp_0 = 3991.86795711963
        self.forc_temp_surface[...] = (f1 * self._sst_rest[:, :, n1] + f2 * self._sst_rest[:, :, n2]) * \
                                      (f1 * self._sst_clim[:, :, n1] + f2 * self._sst_clim[:, :, n2]
                                       - self.temp[:, :, -1, self.tau]) * self.maskT[:, :, -1] / cp_0 / self.rho_0
        self.forc_salt_surface[...] = (f1 * self._sss_rest[:, :, n1] + f2 * self._sss_rest[:, :, n2]) * \
                                      (f1 * self._sss_clim[:, :, n1] + f2 * self._sss_clim[:, :, n2]
                                       - self.salt[:, :, -1, self.tau]) * self.maskT[:, :, -1]

        ice_mask = (self.temp[:, :, -1, self.tau] * self.maskT[:, :, -1] <= -1.8) & (self.forc_temp_surface <= 0.0)
        self.forc_temp_surface[...] *= ~ice_mask
        self.forc_salt_surface[...] *= ~ice_mask

        if self.enable_tempsalt_sources:
            self.temp_source[...] = self.maskT * self._rest_tscl \
                * (f1 * self._t_star[:, :, :, n1] + f2 * self._t_star[:, :, :, n2] - self.temp[:, :, :, self.tau])
            self.salt_source[...] = self.maskT * self._rest_tscl \
                * (f1 * self._s_star[:, :, :, n1] + f2 * self._s_star[:, :, :, n2] - self.salt[:, :, :, self.tau])

    def set_diagnostics(self):
        self.diagnostics["snapshot"].output_frequency = 3600. * 24 * 10
        self.diagnostics["averages"].output_frequency = 3600. * 24 * 10
        self.diagnostics["averages"].sampling_frequency = self.dt_tracer
        self.diagnostics["averages"].output_variables = ["temp", "salt", "u", "v", "w",
                                                         "surface_taux", "surface_tauy", "psi"]
        self.diagnostics["cfl_monitor"].output_frequency = self.dt_tracer * 10

    def after_timestep(self):
        pass


@veros.tools.cli
def run(*args, **kwargs):
    simulation = NorthAtlantic(*args, **kwargs)
    simulation.setup()
    simulation.run()


if __name__ == "__main__":
    run()
