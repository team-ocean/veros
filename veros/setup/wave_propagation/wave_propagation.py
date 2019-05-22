#!/usr/bin/env python

import os

import numpy as np
import h5netcdf
from PIL import Image
import scipy.ndimage
from loguru import logger

from veros import veros_method, VerosSetup
from veros.variables import Variable
import veros.tools
from veros.core.utilities import enforce_boundaries

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_FILES = veros.tools.get_assets('wave_propagation', os.path.join(BASE_PATH, 'assets.yml'))
TOPO_MASK_FILE = os.path.join(BASE_PATH, 'topography_idealized.png')
NA_MASK_FILE = os.path.join(BASE_PATH, 'na_mask.png')


class WavePropagationSetup(VerosSetup):
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

    @veros_method
    def set_parameter(self, vs):
        vs.identifier = 'wp'

        vs.nx = 180
        vs.ny = 180
        vs.nz = 60
        vs.dt_mom = vs.dt_tracer = 0
        vs.runlen = 86400 * 10

        vs.coord_degree = True
        vs.enable_cyclic_x = True

        # streamfunction
        vs.congr_epsilon = 1e-6
        vs.congr_max_iterations = 10000

        # friction
        vs.enable_hor_friction = True
        vs.A_h = 5e4
        vs.enable_hor_friction_cos_scaling = True
        vs.hor_friction_cosPower = 1
        vs.enable_tempsalt_sources = True
        vs.enable_implicit_vert_friction = True

        vs.eq_of_state_type = 5

        # isoneutral
        vs.enable_neutral_diffusion = True
        vs.K_iso_0 = 1000.0
        vs.K_iso_steep = 50.0
        vs.iso_dslope = 0.005
        vs.iso_slopec = 0.005
        vs.enable_skew_diffusion = True

        # tke
        vs.enable_tke = True
        vs.c_k = 0.1
        vs.c_eps = 0.7
        vs.alpha_tke = 30.0
        vs.mxl_min = 1e-8
        vs.tke_mxl_choice = 2
        vs.enable_tke_superbee_advection = True

        # eke
        vs.enable_eke = True
        vs.eke_k_max = 1e4
        vs.eke_c_k = 0.4
        vs.eke_c_eps = 0.5
        vs.eke_cross = 2.
        vs.eke_crhin = 1.0
        vs.eke_lmin = 100.0
        vs.enable_eke_superbee_advection = True
        vs.enable_eke_isopycnal_diffusion = True

        # idemix
        vs.enable_idemix = False
        vs.enable_eke_diss_surfbot = True
        vs.eke_diss_surfbot_frac = 0.2
        vs.enable_idemix_superbee_advection = True
        vs.enable_idemix_hor_diffusion = True

        # custom variables
        vs.nmonths = 12
        vs.variables.update(
            t_star=Variable('t_star', ('xt', 'yt', 'nmonths'), '', '', time_dependent=False),
            s_star=Variable('s_star', ('xt', 'yt', 'nmonths'), '', '', time_dependent=False),
            qnec=Variable('qnec', ('xt', 'yt', 'nmonths'), '', '', time_dependent=False),
            qnet=Variable('qnet', ('xt', 'yt', 'nmonths'), '', '', time_dependent=False),
            qsol=Variable('qsol', ('xt', 'yt', 'nmonths'), '', '', time_dependent=False),
            divpen_shortwave=Variable('divpen_shortwave', ('zt',), '', '', time_dependent=False),
            taux=Variable('taux', ('xt', 'yt', 'nmonths'), '', '', time_dependent=False),
            tauy=Variable('tauy', ('xt', 'yt', 'nmonths'), '', '', time_dependent=False),
            na_mask=Variable(
                'Mask for North Atlantic', ('xt', 'yt'), '', 'Mask for North Atlantic',
                dtype='bool', time_dependent=False, output=True
            )
        )

    @veros_method(inline=True)
    def _get_data(self, vs, var):
        with h5netcdf.File(DATA_FILES['forcing'], 'r') as forcing_file:
            var_obj = forcing_file.variables[var]
            return np.array(var_obj, dtype=str(var_obj.dtype)).T

    @veros_method(dist_safe=False, local_variables=[
        'dxt', 'dyt', 'dzt'
    ])
    def set_grid(self, vs):
        if vs.ny % 2:
            raise ValueError('ny has to be an even number of grid cells')
        vs.dxt[...] = 360. / vs.nx
        vs.dyt[2:-2] = veros.tools.get_vinokur_grid_steps(
            vs.ny, 160., self.equatorial_grid_spacing, upper_stepsize=self.polar_grid_spacing, two_sided_grid=True
        )
        vs.dzt[...] = veros.tools.get_vinokur_grid_steps(vs.nz, self.max_depth, 10., refine_towards='lower')
        vs.y_origin = -80.
        vs.x_origin = 90.

    @veros_method
    def set_coriolis(self, vs):
        vs.coriolis_t[...] = 2 * vs.omega * np.sin(vs.yt[np.newaxis, :] / 180. * vs.pi)

    @veros_method
    def _shift_longitude_array(self, vs, lon, arr):
        wrap_i = np.where((lon[:-1] < vs.xt.min()) & (lon[1:] >= vs.xt.min()))[0][0]
        new_lon = np.concatenate((lon[wrap_i:-1], lon[:wrap_i] + 360.))
        new_arr = np.concatenate((arr[wrap_i:-1, ...], arr[:wrap_i, ...]))
        return new_lon, new_arr

    @veros_method(dist_safe=False, local_variables=[
        'kbot', 'xt', 'yt', 'zt', 'na_mask'
    ])
    def set_topography(self, vs):
        with h5netcdf.File(DATA_FILES['topography'], 'r') as topography_file:
            topo_x, topo_y, topo_z = (
                np.array(topography_file.variables[k], dtype='float').T
                for k in ('x', 'y', 'z')
            )
        topo_z[topo_z > 0] = 0.
        topo_mask = (np.flipud(Image.open(TOPO_MASK_FILE)).T / 255).astype(np.bool)
        gaussian_sigma = (0.5 * len(topo_x) / vs.nx, 0.5 * len(topo_y) / vs.ny)
        topo_z_smoothed = scipy.ndimage.gaussian_filter(topo_z, sigma=gaussian_sigma)
        topo_z_smoothed[~topo_mask & (topo_z_smoothed >= 0.)] = -100.
        topo_masked = np.where(topo_mask, 0., topo_z_smoothed)

        na_mask_image = np.flipud(Image.open(NA_MASK_FILE)).T / 255.
        topo_x_shifted, na_mask_shifted = self._shift_longitude_array(vs, topo_x, na_mask_image)
        coords = (vs.xt[2:-2], vs.yt[2:-2])
        vs.na_mask[2:-2, 2:-2] = np.logical_not(veros.tools.interpolate(
            (topo_x_shifted, topo_y), na_mask_shifted, coords, kind='nearest', fill=False
        ).astype(np.bool))

        topo_x_shifted, topo_masked_shifted = self._shift_longitude_array(vs, topo_x, topo_masked)
        z_interp = veros.tools.interpolate(
            (topo_x_shifted, topo_y), topo_masked_shifted, coords, kind='nearest', fill=False
        )
        z_interp[vs.na_mask[2:-2, 2:-2]] = -self.na_max_depth

        grid_coords = np.meshgrid(*coords, indexing='ij')
        coastline_distance = veros.tools.get_coastline_distance(
            grid_coords, z_interp >= 0, spherical=True, radius=vs.radius
        )
        if self.na_slope_length:
            slope_distance = coastline_distance - self.na_shelf_distance
            slope_mask = np.logical_and(vs.na_mask[2:-2, 2:-2], slope_distance < self.na_slope_length)
            z_interp[slope_mask] = -(self.na_shelf_depth + slope_distance[slope_mask] / self.na_slope_length \
                                                           * (self.na_max_depth - self.na_shelf_depth))
        if self.na_shelf_distance:
            shelf_mask = np.logical_and(vs.na_mask[2:-2, 2:-2], coastline_distance < self.na_shelf_distance)
            z_interp[shelf_mask] = -self.na_shelf_depth

        depth_levels = 1 + np.argmin(np.abs(z_interp[:, :, np.newaxis] - vs.zt[np.newaxis, np.newaxis, :]), axis=2)
        vs.kbot[2:-2, 2:-2] = np.where(z_interp < 0., depth_levels, 0)
        vs.kbot *= vs.kbot < vs.nz

    @veros_method
    def _fix_north_atlantic(self, vs, arr):
        """Calculate zonal mean forcing over masked area (na_mask)."""
        newaxes = (slice(2, -2), slice(2, -2)) + (np.newaxis,) * (arr.ndim - 2)
        arr_masked = np.ma.masked_where(np.logical_or(~vs.na_mask[newaxes], arr == 0.), arr)
        zonal_mean_na = arr_masked.mean(axis=0)
        return np.where(~arr_masked.mask, zonal_mean_na[np.newaxis, ...], arr)

    @veros_method(dist_safe=False, local_variables=[
        'qnet', 'temp', 'salt', 'maskT', 'taux', 'tauy', 'xt', 'yt', 'zt',
        'qnec', 'qsol', 't_star', 's_star', 'na_mask',
        'maskW', 'divpen_shortwave', 'dzt', 'zw',
    ])
    def set_initial_conditions(self, vs):
        rpart_shortwave = 0.58
        efold1_shortwave = 0.35
        efold2_shortwave = 23.0

        t_grid = (vs.xt[2:-2], vs.yt[2:-2], vs.zt)
        xt_forc, yt_forc, zt_forc = (self._get_data(vs, k) for k in ('xt', 'yt', 'zt'))
        zt_forc = zt_forc[::-1]

        # initial conditions
        temp_data = veros.tools.interpolate((xt_forc, yt_forc, zt_forc), self._get_data(vs, 'temperature')[:, :, ::-1],
                                      t_grid, missing_value=0.)
        vs.temp[2:-2, 2:-2, :, 0] = temp_data * vs.maskT[2:-2, 2:-2, :]
        vs.temp[2:-2, 2:-2, :, 1] = temp_data * vs.maskT[2:-2, 2:-2, :]

        salt_data = veros.tools.interpolate((xt_forc, yt_forc, zt_forc), self._get_data(vs, 'salinity')[:, :, ::-1],
                                       t_grid, missing_value=0.)
        vs.salt[2:-2, 2:-2, :, 0] = salt_data * vs.maskT[2:-2, 2:-2, :]
        vs.salt[2:-2, 2:-2, :, 1] = salt_data * vs.maskT[2:-2, 2:-2, :]

        # wind stress on MIT grid
        time_grid = (vs.xt[2:-2], vs.yt[2:-2], np.arange(12))
        taux_data = veros.tools.interpolate((xt_forc, yt_forc, np.arange(12)),
                                      self._get_data(vs, 'tau_x'), time_grid,
                                      missing_value=0.)
        vs.taux[2:-2, 2:-2, :] = taux_data
        mask = np.logical_and(vs.yt > self.so_wind_interval[0], vs.yt < self.so_wind_interval[1])[..., np.newaxis]
        vs.taux *= 1. + mask * (self.so_wind_factor - 1.) * np.sin(np.pi * (vs.yt[np.newaxis, :, np.newaxis] - self.so_wind_interval[0]) \
                                                                            / (self.so_wind_interval[1] - self.so_wind_interval[0]))

        tauy_data = veros.tools.interpolate((xt_forc, yt_forc, np.arange(12)),
                                      self._get_data(vs, 'tau_y'), time_grid,
                                      missing_value=0.)
        vs.tauy[2:-2, 2:-2, :] = tauy_data

        enforce_boundaries(vs, vs.taux)
        enforce_boundaries(vs, vs.tauy)

        # Qnet and dQ/dT and Qsol
        qnet_data = veros.tools.interpolate((xt_forc, yt_forc, np.arange(12)),
                                      self._get_data(vs, 'q_net'), time_grid, missing_value=0.)
        vs.qnet[2:-2, 2:-2, :] = -qnet_data * vs.maskT[2:-2, 2:-2, -1, np.newaxis]

        qnec_data = veros.tools.interpolate((xt_forc, yt_forc, np.arange(12)),
                                       self._get_data(vs, 'dqdt'), time_grid, missing_value=0.)
        vs.qnec[2:-2, 2:-2, :] = qnec_data * vs.maskT[2:-2, 2:-2, -1, np.newaxis]

        qsol_data = veros.tools.interpolate((xt_forc, yt_forc, np.arange(12)),
                                       self._get_data(vs, 'swf'), time_grid, missing_value=0.)
        vs.qsol[2:-2, 2:-2, :] = -qsol_data * vs.maskT[2:-2, 2:-2, -1, np.newaxis]

        # SST and SSS
        sst_data = veros.tools.interpolate((xt_forc, yt_forc, np.arange(12)),
                                     self._get_data(vs, 'sst'), time_grid, missing_value=0.)
        vs.t_star[2:-2, 2:-2, :] = sst_data * vs.maskT[2:-2, 2:-2, -1, np.newaxis]

        sss_data = veros.tools.interpolate((xt_forc, yt_forc, np.arange(12)),
                                     self._get_data(vs, 'sss'), time_grid, missing_value=0.)
        vs.s_star[2:-2, 2:-2, :] = sss_data * vs.maskT[2:-2, 2:-2, -1, np.newaxis]

        if vs.enable_idemix:
            tidal_energy_data = veros.tools.interpolate(
                (xt_forc, yt_forc), self._get_data(vs, 'tidal_energy'), t_grid[:-1], missing_value=0.
            )
            mask_x, mask_y = (i + 2 for i in np.indices((vs.nx, vs.ny)))
            mask_z = np.maximum(0, vs.kbot[2:-2, 2:-2] - 1)
            tidal_energy_data[:, :] *= vs.maskW[mask_x, mask_y, mask_z] / vs.rho_0
            vs.forc_iw_bottom[2:-2, 2:-2] = tidal_energy_data

        # average variables in North Atlantic
        na_average_vars = [vs.taux, vs.tauy, vs.qnet, vs.qnec, vs.qsol,
                           vs.t_star, vs.s_star, vs.salt, vs.temp]

        for k in na_average_vars:
            k[2:-2, 2:-2, ...] = self._fix_north_atlantic(vs, k[2:-2, 2:-2, ...])

        """
        Initialize penetration profile for solar radiation and store divergence in divpen
        note that pen is set to 0.0 at the surface instead of 1.0 to compensate for the
        shortwave part of the total surface flux
        """
        swarg1 = vs.zw / efold1_shortwave
        swarg2 = vs.zw / efold2_shortwave
        pen = rpart_shortwave * np.exp(swarg1) + (1.0 - rpart_shortwave) * np.exp(swarg2)
        pen[-1] = 0.
        vs.divpen_shortwave[1:] = (pen[1:] - pen[:-1]) / vs.dzt[1:]
        vs.divpen_shortwave[0] = pen[0] / vs.dzt[0]

    @veros_method
    def set_forcing(self, vs):
        self.set_timestep(vs)

        t_rest = 30. * 86400.
        cp_0 = 3991.86795711963  # J/kg /K

        year_in_seconds = 360 * 86400.
        (n1, f1), (n2, f2) = veros.tools.get_periodic_interval(
            vs.time, year_in_seconds, year_in_seconds / 12., 12
        )

        vs.surface_taux[...] = f1 * vs.taux[:, :, n1] + f2 * vs.taux[:, :, n2]
        vs.surface_tauy[...] = f1 * vs.tauy[:, :, n1] + f2 * vs.tauy[:, :, n2]

        if vs.enable_tke:
            vs.forc_tke_surface[1:-1, 1:-1] = np.sqrt((0.5 * (vs.surface_taux[1:-1, 1:-1] + vs.surface_taux[:-2, 1:-1]) / vs.rho_0) ** 2
                                                        + (0.5 * (vs.surface_tauy[1:-1, 1:-1] + vs.surface_tauy[1:-1, :-2]) / vs.rho_0) ** 2) ** (3. / 2.)

        # W/m^2 K kg/J m^3/kg = K m/s
        fxa = f1 * vs.t_star[..., n1] + f2 * vs.t_star[..., n2]
        vs.qqnec = f1 * vs.qnec[..., n1] + f2 * vs.qnec[..., n2]
        vs.qqnet = f1 * vs.qnet[..., n1] + f2 * vs.qnet[..., n2]
        vs.forc_temp_surface[...] = (vs.qqnet + vs.qqnec * (fxa - vs.temp[..., -1, vs.tau])) \
            * vs.maskT[..., -1] / cp_0 / vs.rho_0
        fxa = f1 * vs.s_star[..., n1] + f2 * vs.s_star[..., n2]
        vs.forc_salt_surface[...] = 1. / t_rest * \
            (fxa - vs.salt[..., -1, vs.tau]) * vs.maskT[..., -1] * vs.dzt[-1]

        # apply simple ice mask
        mask1 = vs.temp[:, :, -1, vs.tau] * vs.maskT[:, :, -1] <= -1.8
        mask2 = vs.forc_temp_surface <= 0
        ice = ~(mask1 & mask2)
        vs.forc_temp_surface[...] *= ice
        vs.forc_salt_surface[...] *= ice

        # solar radiation
        if vs.enable_tempsalt_sources:
            vs.temp_source[..., :] = (f1 * vs.qsol[..., n1, None] + f2 * vs.qsol[..., n2, None]) \
                * vs.divpen_shortwave[None, None, :] * ice[..., None] \
                * vs.maskT[..., :] / cp_0 / vs.rho_0

    @veros_method
    def set_diagnostics(self, vs):
        vs.diagnostics['cfl_monitor'].output_frequency = 86400.
        vs.diagnostics['tracer_monitor'].output_frequency = 86400.
        vs.diagnostics['snapshot'].output_frequency = 10 * 86400.
        vs.diagnostics['overturning'].output_frequency = 360 * 86400
        vs.diagnostics['overturning'].sampling_frequency = 10 * 86400
        vs.diagnostics['energy'].output_frequency = 360 * 86400
        vs.diagnostics['energy'].sampling_frequency = 86400.
        vs.diagnostics['averages'].output_frequency = 360 * 86400
        vs.diagnostics['averages'].sampling_frequency = 86400.

        average_vars = ['surface_taux', 'surface_tauy', 'forc_temp_surface', 'forc_salt_surface',
                        'psi', 'temp', 'salt', 'u', 'v', 'w', 'Nsqr', 'Hd', 'rho',
                        'K_diss_v', 'P_diss_v', 'P_diss_nonlin', 'P_diss_iso', 'kappaH']
        if vs.enable_skew_diffusion:
            average_vars += ['B1_gm', 'B2_gm']
        if vs.enable_TEM_friction:
            average_vars += ['kappa_gm', 'K_diss_gm']
        if vs.enable_tke:
            average_vars += ['tke', 'Prandtlnumber', 'mxl', 'tke_diss',
                             'forc_tke_surface', 'tke_surf_corr']
        if vs.enable_idemix:
            average_vars += ['E_iw', 'forc_iw_surface', 'iw_diss',
                             'c0', 'v0']
        if vs.enable_eke:
            average_vars += ['eke', 'K_gm', 'L_rossby', 'L_rhines']
        vs.diagnostics['averages'].output_variables = average_vars

        vs.diagnostics['snapshot'].output_variables.append('na_mask')

    @veros_method
    def set_timestep(self, vs):
        if vs.time < 90 * 86400:
            if vs.dt_tracer != 1800.:
                vs.dt_tracer = vs.dt_mom = 1800.
                logger.info('Setting time step to 30m')
        else:
            if vs.dt_tracer != 3600.:
                vs.dt_tracer = vs.dt_mom = 3600.
                logger.info('Setting time step to 1h')

    def after_timestep(self, vs):
        pass


@veros.tools.cli
def run(*args, **kwargs):
    simulation = WavePropagationSetup(*args, **kwargs)
    simulation.setup()
    simulation.run()


if __name__ == '__main__':
    run()
