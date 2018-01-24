#!/usr/bin/env python

import os
import logging

import numpy as np
from netCDF4 import Dataset
from PIL import Image
import scipy.ndimage

import veros
import veros.tools
import veros.core.cyclic

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_FILES = veros.tools.get_assets("wave_propagation", os.path.join(BASE_PATH, "assets.yml"))


class WavePropagation(veros.Veros):
    """
    Global model with flexible resolution and idealized geometry in the
    Atlantic to examine coastal wave propagation.
    """
    # settings for north atlantic
    na_shelf_depth = 250
    na_shelf_distance = 0
    na_slope_length = 600e3
    na_max_depth = 4000

    # global settings
    max_depth = 5600.
    equatorial_grid_spacing = 0.5
    polar_grid_spacing = None

    # southern ocean wind modifier
    so_wind_interval = (-69., -27.)
    so_wind_factor = 1.

    @veros.veros_method
    def set_parameter(self):
        self.identifier = "wp"

        self.nx = 180
        self.ny = 180
        self.nz = 60
        self.dt_mom = self.dt_tracer = 0
        self.runlen = 86400 * 10

        self.coord_degree = True
        self.enable_cyclic_x = True

        # streamfunction
        self.congr_epsilon = 1e-6
        self.congr_max_iterations = 10000

        # friction
        self.enable_hor_friction = True
        self.A_h = 5e4
        self.enable_hor_friction_cos_scaling = True
        self.hor_friction_cosPower = 1
        self.enable_tempsalt_sources = True
        self.enable_implicit_vert_friction = True

        self.eq_of_state_type = 5

        # isoneutral
        self.enable_neutral_diffusion = True
        self.K_iso_0 = 1000.0
        self.K_iso_steep = 50.0
        self.iso_dslope = 0.005
        self.iso_slopec = 0.005
        self.enable_skew_diffusion = True

        # tke
        self.enable_tke = True
        self.c_k = 0.1
        self.c_eps = 0.7
        self.alpha_tke = 30.0
        self.mxl_min = 1e-8
        self.tke_mxl_choice = 2
        self.enable_tke_superbee_advection = True

        # eke
        self.enable_eke = True
        self.eke_k_max = 1e4
        self.eke_c_k = 0.4
        self.eke_c_eps = 0.5
        self.eke_cross = 2.
        self.eke_crhin = 1.0
        self.eke_lmin = 100.0
        self.enable_eke_superbee_advection = True
        self.enable_eke_isopycnal_diffusion = True

        # idemix
        self.enable_idemix = False
        self.enable_eke_diss_surfbot = True
        self.eke_diss_surfbot_frac = 0.2
        self.enable_idemix_superbee_advection = True
        self.enable_idemix_hor_diffusion = True

    def _get_data(self, var):
        with Dataset(DATA_FILES["forcing"], "r") as forcing_file:
            return forcing_file.variables[var][...].T

    @veros.veros_method
    def set_grid(self):
        if self.ny % 2:
            raise ValueError("ny has to be an even number of grid cells")
        self.dxt[...] = 360. / self.nx
        self.dyt[2:-2] = veros.tools.get_vinokur_grid_steps(
            self.ny, 160., self.equatorial_grid_spacing, upper_stepsize=self.polar_grid_spacing, two_sided_grid=True
        )
        self.dzt[...] = veros.tools.get_vinokur_grid_steps(self.nz, self.max_depth, 10., refine_towards="lower")
        self.y_origin = -80.
        self.x_origin = 90.

    @veros.veros_method
    def set_coriolis(self):
        self.coriolis_t[...] = 2 * self.omega * np.sin(self.yt[np.newaxis, :] / 180. * self.pi)

    @veros.veros_method
    def _shift_longitude_array(self, lon, arr):
        wrap_i = np.where((lon[:-1] < self.xt.min()) & (lon[1:] >= self.xt.min()))[0][0]
        new_lon = np.concatenate((lon[wrap_i:-1], lon[:wrap_i] + 360.))
        new_arr = np.concatenate((arr[wrap_i:-1, ...], arr[:wrap_i, ...]))
        return new_lon, new_arr

    @veros.veros_method
    def set_topography(self):
        with Dataset(DATA_FILES["topography"], "r") as topography_file:
            topo_x, topo_y, topo_z = (topography_file.variables[k][...].T.astype(np.float) for k in ("x", "y", "z"))
        topo_z[topo_z > 0] = 0.
        topo_mask = (np.flipud(Image.open("topography_idealized.png")).T / 255).astype(np.bool)
        gaussian_sigma = (0.5 * len(topo_x) / self.nx, 0.5 * len(topo_y) / self.ny)
        topo_z_smoothed = scipy.ndimage.gaussian_filter(topo_z, sigma=gaussian_sigma)
        topo_z_smoothed[~topo_mask & (topo_z_smoothed >= 0.)] = -100.
        topo_masked = np.where(topo_mask, 0., topo_z_smoothed)

        na_mask_image = np.flipud(Image.open("na_mask.png")).T / 255.
        topo_x_shifted, na_mask_shifted = self._shift_longitude_array(topo_x, na_mask_image)
        coords = (self.xt[2:-2], self.yt[2:-2])
        self.na_mask = np.zeros((self.nx + 4, self.ny + 4), dtype=np.bool)
        self.na_mask[2:-2, 2:-2] = np.logical_not(veros.tools.interpolate(
            (topo_x_shifted, topo_y), na_mask_shifted, coords, kind="nearest", fill=False
        ).astype(np.bool))

        topo_x_shifted, topo_masked_shifted = self._shift_longitude_array(topo_x, topo_masked)
        z_interp = veros.tools.interpolate(
            (topo_x_shifted, topo_y), topo_masked_shifted, coords, kind="nearest", fill=False
        )
        z_interp[self.na_mask[2:-2, 2:-2]] = -self.na_max_depth

        grid_coords = np.meshgrid(*coords, indexing="ij")
        coastline_distance = veros.tools.get_coastline_distance(
            grid_coords, z_interp >= 0, spherical=True, radius=self.radius
        )
        if self.na_slope_length:
            slope_distance = coastline_distance - self.na_shelf_distance
            slope_mask = np.logical_and(self.na_mask[2:-2, 2:-2], slope_distance < self.na_slope_length)
            z_interp[slope_mask] = -(self.na_shelf_depth + slope_distance[slope_mask] / self.na_slope_length \
                                                           * (self.na_max_depth - self.na_shelf_depth))
        if self.na_shelf_distance:
            shelf_mask = np.logical_and(self.na_mask[2:-2, 2:-2], coastline_distance < self.na_shelf_distance)
            z_interp[shelf_mask] = -self.na_shelf_depth

        depth_levels = 1 + np.argmin(np.abs(z_interp[:, :, np.newaxis] - self.zt[np.newaxis, np.newaxis, :]), axis=2)
        self.kbot[2:-2, 2:-2] = np.where(z_interp < 0., depth_levels, 0)
        self.kbot *= self.kbot < self.nz

    @veros.veros_method
    def _fix_north_atlantic(self, arr):
        """Calculate zonal mean forcing over masked area (na_mask)."""
        newaxes = (slice(2, -2), slice(2, -2)) + (np.newaxis,) * (arr.ndim - 2)
        arr_masked = np.ma.masked_where(np.logical_or(~self.na_mask[newaxes], arr == 0.), arr)
        zonal_mean_na = arr_masked.mean(axis=0)
        return np.where(~arr_masked.mask, zonal_mean_na[np.newaxis, ...], arr)

    @veros.veros_method
    def set_initial_conditions(self):
        self._t_star = np.zeros((self.nx + 4, self.ny + 4, 12), dtype=self.default_float_type)
        self._s_star = np.zeros((self.nx + 4, self.ny + 4, 12), dtype=self.default_float_type)
        self._qnec = np.zeros((self.nx + 4, self.ny + 4, 12), dtype=self.default_float_type)
        self._qnet = np.zeros((self.nx + 4, self.ny + 4, 12), dtype=self.default_float_type)
        self._qsol = np.zeros((self.nx + 4, self.ny + 4, 12), dtype=self.default_float_type)
        self._divpen_shortwave = np.zeros(self.nz, dtype=self.default_float_type)
        self._taux = np.zeros((self.nx + 4, self.ny + 4, 12), dtype=self.default_float_type)
        self._tauy = np.zeros((self.nx + 4, self.ny + 4, 12), dtype=self.default_float_type)

        rpart_shortwave = 0.58
        efold1_shortwave = 0.35
        efold2_shortwave = 23.0

        t_grid = (self.xt[2:-2], self.yt[2:-2], self.zt)
        xt_forc, yt_forc, zt_forc = (self._get_data(k) for k in ("xt", "yt", "zt"))
        zt_forc = zt_forc[::-1]

        # initial conditions
        temp_data = veros.tools.interpolate((xt_forc, yt_forc, zt_forc), self._get_data("temperature")[:, :, ::-1],
                                      t_grid, missing_value=0.)
        self.temp[2:-2, 2:-2, :, 0] = temp_data * self.maskT[2:-2, 2:-2, :]
        self.temp[2:-2, 2:-2, :, 1] = temp_data * self.maskT[2:-2, 2:-2, :]

        salt_data = veros.tools.interpolate((xt_forc, yt_forc, zt_forc), self._get_data("salinity")[:, :, ::-1],
                                       t_grid, missing_value=0.)
        self.salt[2:-2, 2:-2, :, 0] = salt_data * self.maskT[2:-2, 2:-2, :]
        self.salt[2:-2, 2:-2, :, 1] = salt_data * self.maskT[2:-2, 2:-2, :]

        # wind stress on MIT grid
        time_grid = (self.xt[2:-2], self.yt[2:-2], np.arange(12))
        taux_data = veros.tools.interpolate((xt_forc, yt_forc, np.arange(12)),
                                      self._get_data("tau_x"), time_grid,
                                      missing_value=0.)
        self._taux[2:-2, 2:-2, :] = taux_data / self.rho_0
        mask = np.logical_and(self.yt > self.so_wind_interval[0], self.yt < self.so_wind_interval[1])[..., np.newaxis]
        self._taux *= 1. + mask * (self.so_wind_factor - 1.) * np.sin(np.pi * (self.yt[np.newaxis, :, np.newaxis] - self.so_wind_interval[0]) \
                                                                            / (self.so_wind_interval[1] - self.so_wind_interval[0]))

        tauy_data = veros.tools.interpolate((xt_forc, yt_forc, np.arange(12)),
                                      self._get_data("tau_y"), time_grid,
                                      missing_value=0.)
        self._tauy[2:-2, 2:-2, :] = tauy_data / self.rho_0

        if self.enable_cyclic_x:
            veros.core.cyclic.setcyclic_x(self._taux)
            veros.core.cyclic.setcyclic_x(self._tauy)

        # Qnet and dQ/dT and Qsol
        qnet_data = veros.tools.interpolate((xt_forc, yt_forc, np.arange(12)),
                                      self._get_data("q_net"), time_grid, missing_value=0.)
        self._qnet[2:-2, 2:-2, :] = -qnet_data * self.maskT[2:-2, 2:-2, -1, np.newaxis]

        qnec_data = veros.tools.interpolate((xt_forc, yt_forc, np.arange(12)),
                                       self._get_data("dqdt"), time_grid, missing_value=0.)
        self._qnec[2:-2, 2:-2, :] = qnec_data * self.maskT[2:-2, 2:-2, -1, np.newaxis]

        qsol_data = veros.tools.interpolate((xt_forc, yt_forc, np.arange(12)),
                                       self._get_data("swf"), time_grid, missing_value=0.)
        self._qsol[2:-2, 2:-2, :] = -qsol_data * self.maskT[2:-2, 2:-2, -1, np.newaxis]

        # SST and SSS
        sst_data = veros.tools.interpolate((xt_forc, yt_forc, np.arange(12)),
                                     self._get_data("sst"), time_grid, missing_value=0.)
        self._t_star[2:-2, 2:-2, :] = sst_data * self.maskT[2:-2, 2:-2, -1, np.newaxis]

        sss_data = veros.tools.interpolate((xt_forc, yt_forc, np.arange(12)),
                                     self._get_data("sss"), time_grid, missing_value=0.)
        self._s_star[2:-2, 2:-2, :] = sss_data * self.maskT[2:-2, 2:-2, -1, np.newaxis]

        if self.enable_idemix:
            tidal_energy_data = veros.tools.interpolate(
                (xt_forc, yt_forc), self._get_data("tidal_energy"), t_grid[:-1], missing_value=0.
            )
            mask_x, mask_y = (i + 2 for i in np.indices((self.nx, self.ny)))
            mask_z = np.maximum(0, self.kbot[2:-2, 2:-2] - 1)
            tidal_energy_data[:, :] *= self.maskW[mask_x, mask_y, mask_z] / self.rho_0
            self.forc_iw_bottom[2:-2, 2:-2] = tidal_energy_data

        # average variables in North Atlantic
        na_average_vars = [self._taux, self._tauy, self._qnet, self._qnec, self._qsol,
                           self._t_star, self._s_star, self.salt, self.temp]
        if self.enable_idemix:
            na_average_vars += [self.forc_iw_bottom]
        for k in na_average_vars:
            k[2:-2, 2:-2, ...] = self._fix_north_atlantic(k[2:-2, 2:-2, ...])

        """
        Initialize penetration profile for solar radiation and store divergence in divpen
        note that pen is set to 0.0 at the surface instead of 1.0 to compensate for the
        shortwave part of the total surface flux
        """
        swarg1 = self.zw / efold1_shortwave
        swarg2 = self.zw / efold2_shortwave
        pen = rpart_shortwave * np.exp(swarg1) + (1.0 - rpart_shortwave) * np.exp(swarg2)
        pen[-1] = 0.
        self._divpen_shortwave[1:] = (pen[1:] - pen[:-1]) / self.dzt[1:]
        self._divpen_shortwave[0] = pen[0] / self.dzt[0]

    @veros.veros_method
    def set_forcing(self):
        self.set_timestep()

        t_rest = 30. * 86400.
        cp_0 = 3991.86795711963  # J/kg /K

        year_in_seconds = 360 * 86400.
        (n1, f1), (n2, f2) = veros.tools.get_periodic_interval(
            self.time, year_in_seconds, year_in_seconds / 12., 12
        )

        self.surface_taux[...] = f1 * self._taux[:, :, n1] + f2 * self._taux[:, :, n2]
        self.surface_tauy[...] = f1 * self._tauy[:, :, n1] + f2 * self._tauy[:, :, n2]

        if self.enable_tke:
            self.forc_tke_surface[1:-1, 1:-1] = np.sqrt((0.5 * (self.surface_taux[1:-1, 1:-1] + self.surface_taux[:-2, 1:-1])) ** 2
                                                      + (0.5 * (self.surface_tauy[1:-1, 1:-1] + self.surface_tauy[1:-1, :-2])) ** 2) ** (3. / 2.)

        # W/m^2 K kg/J m^3/kg = K m/s
        fxa = f1 * self._t_star[..., n1] + f2 * self._t_star[..., n2]
        self._qqnec = f1 * self._qnec[..., n1] + f2 * self._qnec[..., n2]
        self._qqnet = f1 * self._qnet[..., n1] + f2 * self._qnet[..., n2]
        self.forc_temp_surface[...] = (self._qqnet + self._qqnec * (fxa - self.temp[..., -1, self.tau])) \
            * self.maskT[..., -1] / cp_0 / self.rho_0
        fxa = f1 * self._s_star[..., n1] + f2 * self._s_star[..., n2]
        self.forc_salt_surface[...] = 1. / t_rest * \
            (fxa - self.salt[..., -1, self.tau]) * self.maskT[..., -1] * self.dzt[-1]

        # apply simple ice mask
        mask1 = self.temp[:, :, -1, self.tau] * self.maskT[:, :, -1] <= -1.8
        mask2 = self.forc_temp_surface <= 0
        ice = ~(mask1 & mask2)
        self.forc_temp_surface[...] *= ice
        self.forc_salt_surface[...] *= ice

        # solar radiation
        if self.enable_tempsalt_sources:
            self.temp_source[..., :] = (f1 * self._qsol[..., n1, None] + f2 * self._qsol[..., n2, None]) \
                * self._divpen_shortwave[None, None, :] * ice[..., None] \
                * self.maskT[..., :] / cp_0 / self.rho_0

    @veros.veros_method
    def set_diagnostics(self):
        self.diagnostics["cfl_monitor"].output_frequency = 86400.
        self.diagnostics["tracer_monitor"].output_frequency = 86400.
        self.diagnostics["snapshot"].output_frequency = 10 * 86400.
        self.diagnostics["overturning"].output_frequency = 360 * 86400
        self.diagnostics["overturning"].sampling_frequency = 10 * 86400
        self.diagnostics["energy"].output_frequency = 360 * 86400
        self.diagnostics["energy"].sampling_frequency = 86400.
        self.diagnostics["averages"].output_frequency = 360 * 86400
        self.diagnostics["averages"].sampling_frequency = 86400.

        average_vars = ["surface_taux", "surface_tauy", "forc_temp_surface", "forc_salt_surface",
                        "psi", "temp", "salt", "u", "v", "w", "Nsqr", "Hd", "rho",
                        "K_diss_v", "P_diss_v", "P_diss_nonlin", "P_diss_iso", "kappaH"]
        if self.enable_skew_diffusion:
            average_vars += ["B1_gm", "B2_gm"]
        if self.enable_TEM_friction:
            average_vars += ["kappa_gm", "K_diss_gm"]
        if self.enable_tke:
            average_vars += ["tke", "Prandtlnumber", "mxl", "tke_diss",
                             "forc_tke_surface", "tke_surf_corr"]
        if self.enable_idemix:
            average_vars += ["E_iw", "forc_iw_surface", "forc_iw_bottom", "iw_diss",
                             "c0", "v0"]
        if self.enable_eke:
            average_vars += ["eke", "K_gm", "L_rossby", "L_rhines"]
        self.diagnostics["averages"].output_variables = average_vars

        self.variables["na_mask"] = veros.variables.Variable(
            "Mask for North Atlantic", ("xt", "yt"), "", "Mask for North Atlantic",
            dtype="int", time_dependent=False, output=True
        )
        self.diagnostics["snapshot"].output_variables.append("na_mask")

    @veros.veros_method
    def set_timestep(self):
        if self.time < 90 * 86400:
            if self.dt_tracer != 1800.:
                self.dt_tracer = self.dt_mom = 1800.
                logging.info("Setting time step to 30m")
        else:
            if self.dt_tracer != 3600.:
                self.dt_tracer = self.dt_mom = 3600.
                logging.info("Setting time step to 1h")

    def after_timestep(self):
        pass


@veros.tools.cli
def run(*args, **kwargs):
    simulation = WavePropagation(*args, **kwargs)
    simulation.setup()
    simulation.run()


if __name__ == "__main__":
    run()
