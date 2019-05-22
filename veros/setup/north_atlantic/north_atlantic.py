#!/usr/bin/env python

import os

import h5netcdf
from PIL import Image
import numpy as np
import scipy.spatial
import scipy.ndimage

from veros import VerosSetup, veros_method
from veros.variables import Variable
import veros.tools

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_FILES = veros.tools.get_assets('north_atlantic', os.path.join(BASE_PATH, 'assets.yml'))
TOPO_MASK_FILE = os.path.join(BASE_PATH, 'topo_mask.png')


class NorthAtlanticSetup(VerosSetup):
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

    @veros_method
    def set_parameter(self, vs):
        vs.identifier = 'na'

        vs.nx, vs.ny, vs.nz = 250, 350, 50
        vs.x_origin = -98.
        vs.y_origin = -18.
        vs._x_boundary = 17.2
        vs._y_boundary = 70.
        vs._max_depth = 5800.

        vs.dt_mom = 3600. / 2.
        vs.dt_tracer = 3600. / 2.
        vs.runlen = 86400 * 365. * 10.

        vs.coord_degree = True

        vs.congr_epsilon = 1e-10
        vs.congr_max_iterations = 20000

        vs.enable_neutral_diffusion = True
        vs.enable_skew_diffusion = True
        vs.K_iso_0 = 1000.0
        vs.K_iso_steep = 200.0
        vs.iso_dslope = 1. / 1000.0
        vs.iso_slopec = 4. / 1000.0

        vs.enable_hor_friction = True
        vs.A_h = 1e3
        vs.enable_hor_friction_cos_scaling = True
        vs.hor_friction_cosPower = 1
        vs.enable_tempsalt_sources = True

        vs.enable_implicit_vert_friction = True
        vs.enable_tke = True
        vs.c_k = 0.1
        vs.c_eps = 0.7
        vs.alpha_tke = 30.0
        vs.mxl_min = 1e-8
        vs.tke_mxl_choice = 2

        vs.K_gm_0 = 1000.0

        vs.enable_eke = False
        vs.enable_idemix = False

        vs.eq_of_state_type = 5

        vs.variables.update({
            'sss_clim': Variable('sss_clim', ('xt', 'yt', 12), 'g/kg', 'Monthly sea surface salinity'),
            'sst_clim': Variable('sst_clim', ('xt', 'yt', 12), 'deg C', 'Monthly sea surface temperature'),
            'sss_rest': Variable('sss_rest', ('xt', 'yt', 12), 'g/kg', 'Monthly sea surface salinity restoring'),
            'sst_rest': Variable('sst_rest', ('xt', 'yt', 12), 'deg C', 'Monthly sea surface temperature restoring'),
            't_star': Variable('t_star', ('xt', 'yt', 'zt', 12), 'deg C', 'Temperature sponge layer forcing'),
            's_star': Variable('s_star', ('xt', 'yt', 'zt', 12), 'g/kg', 'Salinity sponge layer forcing'),
            'rest_tscl': Variable('rest_tscl', ('xt', 'yt', 'zt'), '1/s', 'Forcing restoration time scale'),
            'taux': Variable('taux', ('xt', 'yt', 12), 'N/s^2', 'Monthly zonal wind stress'),
            'tauy': Variable('tauy', ('xt', 'yt', 12), 'N/s^2', 'Monthly meridional wind stress'),
        })

    @veros_method
    def set_grid(self, vs):
        vs.dxt[2:-2] = (vs._x_boundary - vs.x_origin) / vs.nx
        vs.dyt[2:-2] = (vs._y_boundary - vs.y_origin) / vs.ny
        vs.dzt[...] = veros.tools.get_vinokur_grid_steps(vs.nz, vs._max_depth, 10., refine_towards='lower')

    @veros_method
    def set_coriolis(self, vs):
        vs.coriolis_t[:, :] = 2 * vs.omega * np.sin(vs.yt[np.newaxis, :] / 180. * vs.pi)

    @veros_method(dist_safe=False, local_variables=[
        'kbot', 'xt', 'yt', 'zt'
    ])
    def set_topography(self, vs):
        with h5netcdf.File(DATA_FILES['topography'], 'r') as topo_file:
            topo_x, topo_y, topo_bottom_depth = (
                self._get_data(vs, topo_file, k) for k in ('x', 'y', 'z')
            )
        topo_mask = np.flipud(np.asarray(Image.open(TOPO_MASK_FILE))).T
        topo_bottom_depth *= 1 - topo_mask
        topo_bottom_depth = scipy.ndimage.gaussian_filter(
            topo_bottom_depth, sigma=(len(topo_x) / vs.nx, len(topo_y) / vs.ny)
        )
        interp_coords = np.meshgrid(vs.xt[2:-2], vs.yt[2:-2], indexing='ij')
        interp_coords = np.rollaxis(np.asarray(interp_coords), 0, 3)
        z_interp = scipy.interpolate.interpn((topo_x, topo_y), topo_bottom_depth, interp_coords,
                                             method='nearest', bounds_error=False, fill_value=0)
        vs.kbot[2:-2, 2:-2] = np.where(z_interp < 0., 1 + np.argmin(np.abs(z_interp[:, :, np.newaxis]
                                       - vs.zt[np.newaxis, np.newaxis, :]), axis=2), 0)
        vs.kbot *= vs.kbot < vs.nz

    @veros_method(inline=True)
    def _get_data(self, vs, f, var):
        """Retrieve variable from h5netcdf file"""
        var_obj = f.variables[var]
        return np.array(var_obj[...].astype(str(var_obj.dtype))).T

    @veros_method(dist_safe=False, local_variables=[
        'xt', 'yt', 'zt', 'temp', 'maskT', 'salt', 'taux', 'tauy',
        'sst_clim', 'sss_clim', 'sst_rest', 'sss_rest', 't_star', 's_star', 'rest_tscl'
    ])
    def set_initial_conditions(self, vs):
        with h5netcdf.File(DATA_FILES['forcing'], 'r') as forcing_file:
            t_hor = (vs.xt[2:-2], vs.yt[2:-2])
            t_grid = (vs.xt[2:-2], vs.yt[2:-2], vs.zt)

            forc_coords = [self._get_data(vs, forcing_file, k) for k in ('xt', 'yt', 'zt')]
            forc_coords[0][...] += -360
            forc_coords[2][...] = -0.01 * forc_coords[2][::-1]

            temp_raw = self._get_data(vs, forcing_file, 'temp_ic')[..., ::-1]
            temp = veros.tools.interpolate(
                forc_coords, temp_raw, t_grid, missing_value=-1e20
            )
            vs.temp[2:-2, 2:-2, :, vs.tau] = vs.maskT[2:-2, 2:-2, :] * temp

            salt_raw = self._get_data(vs, forcing_file, 'salt_ic')[..., ::-1]
            salt = 35. + 1000 * veros.tools.interpolate(
                forc_coords, salt_raw, t_grid, missing_value=-1e20
            )
            vs.salt[2:-2, 2:-2, :, vs.tau] = vs.maskT[2:-2, 2:-2, :] * salt

            forc_u_coords_hor = [self._get_data(vs, forcing_file, k) for k in ('xu', 'yu')]
            forc_u_coords_hor[0][...] += -360

            taux = self._get_data(vs, forcing_file, 'taux')
            tauy = self._get_data(vs, forcing_file, 'tauy')
            for k in range(12):
                vs.taux[2:-2, 2:-2, k] = veros.tools.interpolate(
                    forc_u_coords_hor, taux[..., k], t_hor, missing_value=-1e20
                ) / 10.
                vs.tauy[2:-2, 2:-2, k] = veros.tools.interpolate(
                    forc_u_coords_hor, tauy[..., k], t_hor, missing_value=-1e20
                ) / 10.

            # heat flux and salinity restoring

            sst_clim, sss_clim, sst_rest, sss_rest = [
                forcing_file.variables[k][...].T for k in ('sst_clim', 'sss_clim', 'sst_rest', 'sss_rest')
            ]

        for k in range(12):
            vs.sst_clim[2:-2, 2:-2, k] = veros.tools.interpolate(
                forc_coords[:-1], sst_clim[..., k], t_hor, missing_value=-1e20)
            vs.sss_clim[2:-2, 2:-2, k] = veros.tools.interpolate(
                forc_coords[:-1], sss_clim[..., k], t_hor, missing_value=-1e20) * 1000 + 35
            vs.sst_rest[2:-2, 2:-2, k] = veros.tools.interpolate(
                forc_coords[:-1], sst_rest[..., k], t_hor, missing_value=-1e20) * 41868.
            vs.sss_rest[2:-2, 2:-2, k] = veros.tools.interpolate(
                forc_coords[:-1], sss_rest[..., k], t_hor, missing_value=-1e20) / 100.

        with h5netcdf.File(DATA_FILES['restoring'], 'r') as restoring_file:
            rest_coords = [self._get_data(vs, restoring_file, k) for k in ('xt', 'yt', 'zt')]
            rest_coords[0][...] += -360

            # sponge layers

            vs.rest_tscl[2:-2, 2:-2, :] = veros.tools.interpolate(
                rest_coords, self._get_data(vs, restoring_file, 'tscl')[..., 0], t_grid)

            t_star = self._get_data(vs, restoring_file, 't_star')
            s_star = self._get_data(vs, restoring_file, 's_star')
            for k in range(12):
                vs.t_star[2:-2, 2:-2, :, k] = veros.tools.interpolate(
                    rest_coords, t_star[..., k], t_grid, missing_value=0.
                )
                vs.s_star[2:-2, 2:-2, :, k] = veros.tools.interpolate(
                    rest_coords, s_star[..., k], t_grid, missing_value=0.
                )

    @veros_method
    def set_forcing(self, vs):
        year_in_seconds = 360 * 86400.0
        (n1, f1), (n2, f2) = veros.tools.get_periodic_interval(vs.time, year_in_seconds,
                                                               year_in_seconds / 12., 12)

        vs.surface_taux[...] = (f1 * vs.taux[:, :, n1] + f2 * vs.taux[:, :, n2])
        vs.surface_tauy[...] = (f1 * vs.tauy[:, :, n1] + f2 * vs.tauy[:, :, n2])

        if vs.enable_tke:
            vs.forc_tke_surface[1:-1, 1:-1] = np.sqrt((0.5 * (vs.surface_taux[1:-1, 1:-1] + vs.surface_taux[:-2, 1:-1]) / vs.rho_0)**2
                                                      + (0.5 * (vs.surface_tauy[1:-1, 1:-1] + vs.surface_tauy[1:-1, :-2]) / vs.rho_0)**2
                                                      ) ** (3. / 2.)
        cp_0 = 3991.86795711963
        vs.forc_temp_surface[...] = (f1 * vs.sst_rest[:, :, n1] + f2 * vs.sst_rest[:, :, n2]) * \
                                    (f1 * vs.sst_clim[:, :, n1] + f2 * vs.sst_clim[:, :, n2]
                                     - vs.temp[:, :, -1, vs.tau]) * vs.maskT[:, :, -1] / cp_0 / vs.rho_0
        vs.forc_salt_surface[...] = (f1 * vs.sss_rest[:, :, n1] + f2 * vs.sss_rest[:, :, n2]) * \
                                    (f1 * vs.sss_clim[:, :, n1] + f2 * vs.sss_clim[:, :, n2]
                                     - vs.salt[:, :, -1, vs.tau]) * vs.maskT[:, :, -1]

        ice_mask = (vs.temp[:, :, -1, vs.tau] * vs.maskT[:, :, -1] <= -1.8) & (vs.forc_temp_surface <= 0.0)
        vs.forc_temp_surface[...] *= ~ice_mask
        vs.forc_salt_surface[...] *= ~ice_mask

        if vs.enable_tempsalt_sources:
            vs.temp_source[...] = vs.maskT * vs.rest_tscl \
                * (f1 * vs.t_star[:, :, :, n1] + f2 * vs.t_star[:, :, :, n2] - vs.temp[:, :, :, vs.tau])
            vs.salt_source[...] = vs.maskT * vs.rest_tscl \
                * (f1 * vs.s_star[:, :, :, n1] + f2 * vs.s_star[:, :, :, n2] - vs.salt[:, :, :, vs.tau])

    @veros_method
    def set_diagnostics(self, vs):
        vs.diagnostics['snapshot'].output_frequency = 3600. * 24 * 10
        vs.diagnostics['averages'].output_frequency = 3600. * 24 * 10
        vs.diagnostics['averages'].sampling_frequency = vs.dt_tracer
        vs.diagnostics['averages'].output_variables = ['temp', 'salt', 'u', 'v', 'w',
                                                       'surface_taux', 'surface_tauy', 'psi']
        vs.diagnostics['cfl_monitor'].output_frequency = vs.dt_tracer * 10

    @veros_method
    def after_timestep(self, vs):
        pass


@veros.tools.cli
def run(*args, **kwargs):
    simulation = NorthAtlanticSetup(*args, **kwargs)
    simulation.setup()
    simulation.run()


if __name__ == '__main__':
    run()
