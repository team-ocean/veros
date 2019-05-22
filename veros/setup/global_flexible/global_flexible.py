#!/usr/bin/env python

import os

import numpy as np
import h5netcdf
import scipy.ndimage
import scipy.interpolate

from veros import veros_method, VerosSetup, runtime_settings as rs, runtime_state as rst
from veros.variables import Variable, allocate
import veros.tools
from veros.core.utilities import enforce_boundaries

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_FILES = veros.tools.get_assets('global_flexible', os.path.join(BASE_PATH, 'assets.yml'))


class GlobalFlexibleResolutionSetup(VerosSetup):
    """
    Global model with flexible resolution.
    """
    # global settings
    min_depth = 10.
    max_depth = 5400.
    equatorial_grid_spacing_factor = 0.5
    polar_grid_spacing_factor = None

    @veros_method
    def set_parameter(self, vs):
        vs.identifier = 'UNNAMED'

        vs.nx = 360
        vs.ny = 160
        vs.nz = 60
        vs.dt_mom = vs.dt_tracer = 900
        vs.runlen = 86400 * 10

        vs.coord_degree = True
        vs.enable_cyclic_x = True

        # streamfunction
        vs.congr_epsilon = 1e-10
        vs.congr_max_iterations = 1000

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
        )

    @veros_method(inline=True)
    def _get_data(self, vs, var, idx=None):
        if idx is None:
            idx = Ellipsis
        else:
            idx = idx[::-1]

        kwargs = {}
        if rst.proc_num > 1:
            kwargs.update(
                driver='mpio',
                comm=rs.mpi_comm,
            )

        with h5netcdf.File(DATA_FILES['forcing'], 'r', **kwargs) as forcing_file:
            var_obj = forcing_file.variables[var]
            return np.array(var_obj[idx].astype(str(var_obj.dtype))).T

    @veros_method(dist_safe=False, local_variables=[
        'dxt', 'dyt', 'dzt'
    ])
    def set_grid(self, vs):
        if vs.ny % 2:
            raise ValueError('ny has to be an even number of grid cells')

        vs.dxt[...] = 360. / vs.nx

        if self.equatorial_grid_spacing_factor is not None:
            eq_spacing = self.equatorial_grid_spacing_factor * 160. / vs.ny
        else:
            eq_spacing = None

        if self.polar_grid_spacing_factor is not None:
            polar_spacing = self.polar_grid_spacing_factor * 160. / vs.ny
        else:
            polar_spacing = None

        vs.dyt[2:-2] = veros.tools.get_vinokur_grid_steps(
            vs.ny, 160., eq_spacing, upper_stepsize=polar_spacing, two_sided_grid=True
        )
        vs.dzt[...] = veros.tools.get_vinokur_grid_steps(vs.nz, self.max_depth, self.min_depth, refine_towards='lower')
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
        'kbot', 'xt', 'yt', 'zt'
    ])
    def set_topography(self, vs):
        with h5netcdf.File(DATA_FILES['topography'], 'r') as topography_file:
            topo_x, topo_y, topo_z = (
                np.array(topography_file.variables[k], dtype='float').T
                for k in ('x', 'y', 'z')
            )
        topo_z[topo_z > 0] = 0.

        # smooth topography to match grid resolution
        gaussian_sigma = (0.5 * len(topo_x) / vs.nx, 0.5 * len(topo_y) / vs.ny)
        topo_z_smoothed = scipy.ndimage.gaussian_filter(topo_z, sigma=gaussian_sigma)
        topo_z_smoothed[topo_z >= -1] = 0

        topo_x_shifted, topo_z_shifted = self._shift_longitude_array(vs, topo_x, topo_z_smoothed)
        coords = (vs.xt[2:-2], vs.yt[2:-2])
        z_interp = allocate(vs, ('xt', 'yt'))
        z_interp[2:-2, 2:-2] = veros.tools.interpolate(
            (topo_x_shifted, topo_y), topo_z_shifted, coords, kind='nearest', fill=False
        )

        depth_levels = 1 + np.argmin(np.abs(z_interp[:, :, np.newaxis] - vs.zt[np.newaxis, np.newaxis, :]), axis=2)
        vs.kbot[2:-2, 2:-2] = np.where(z_interp < 0., depth_levels, 0)[2:-2, 2:-2]
        vs.kbot *= vs.kbot < vs.nz

        enforce_boundaries(vs, vs.kbot)

        # remove marginal seas
        # (dilate to close 1-cell passages, fill holes, undo dilation)
        marginal = (
            scipy.ndimage.binary_erosion(
                scipy.ndimage.binary_fill_holes(
                    scipy.ndimage.binary_dilation(vs.kbot == 0)
                )
            )
        )

        vs.kbot[marginal] = 0

    @veros_method
    def set_initial_conditions(self, vs):
        rpart_shortwave = 0.58
        efold1_shortwave = 0.35
        efold2_shortwave = 23.0

        t_grid = (vs.xt[2:-2], vs.yt[2:-2], vs.zt)
        xt_forc, yt_forc, zt_forc = (self._get_data(vs, k) for k in ('xt', 'yt', 'zt'))
        zt_forc = zt_forc[::-1]

        # coordinates must be monotonous for this to work
        assert np.diff(xt_forc).all() > 0
        assert np.diff(yt_forc).all() > 0

        # determine slice to read from forcing file
        data_subset = (
            slice(
                max(0, int(np.argmax(xt_forc >= vs.xt.min())) - 1),
                len(xt_forc) - max(0, int(np.argmax(xt_forc[::-1] <= vs.xt.max())) - 1)
            ),
            slice(
                max(0, int(np.argmax(yt_forc >= vs.yt.min())) - 1),
                len(yt_forc) - max(0, int(np.argmax(yt_forc[::-1] <= vs.yt.max())) - 1)
            ),
            Ellipsis
        )

        xt_forc = xt_forc[data_subset[0]]
        yt_forc = yt_forc[data_subset[1]]

        # initial conditions
        temp_raw = self._get_data(vs, 'temperature', idx=data_subset)[..., ::-1]
        temp_data = veros.tools.interpolate((xt_forc, yt_forc, zt_forc), temp_raw,
                                            t_grid)
        vs.temp[2:-2, 2:-2, :, 0] = temp_data * vs.maskT[2:-2, 2:-2, :]
        vs.temp[2:-2, 2:-2, :, 1] = temp_data * vs.maskT[2:-2, 2:-2, :]

        salt_raw = self._get_data(vs, 'salinity', idx=data_subset)[..., ::-1]
        salt_data = veros.tools.interpolate((xt_forc, yt_forc, zt_forc), salt_raw,
                                            t_grid)
        vs.salt[2:-2, 2:-2, :, 0] = salt_data * vs.maskT[2:-2, 2:-2, :]
        vs.salt[2:-2, 2:-2, :, 1] = salt_data * vs.maskT[2:-2, 2:-2, :]

        # wind stress on MIT grid
        time_grid = (vs.xt[2:-2], vs.yt[2:-2], np.arange(12))
        taux_raw = self._get_data(vs, 'tau_x', idx=data_subset)
        taux_data = veros.tools.interpolate((xt_forc, yt_forc, np.arange(12)),
                                            taux_raw, time_grid)
        vs.taux[2:-2, 2:-2, :] = taux_data / vs.rho_0

        tauy_raw = self._get_data(vs, 'tau_y', idx=data_subset)
        tauy_data = veros.tools.interpolate((xt_forc, yt_forc, np.arange(12)),
                                            tauy_raw, time_grid)
        vs.tauy[2:-2, 2:-2, :] = tauy_data / vs.rho_0

        enforce_boundaries(vs, vs.taux)
        enforce_boundaries(vs, vs.tauy)

        # Qnet and dQ/dT and Qsol
        qnet_raw = self._get_data(vs, 'q_net', idx=data_subset)
        qnet_data = veros.tools.interpolate((xt_forc, yt_forc, np.arange(12)),
                                            qnet_raw, time_grid)
        vs.qnet[2:-2, 2:-2, :] = -qnet_data * vs.maskT[2:-2, 2:-2, -1, np.newaxis]

        qnec_raw = self._get_data(vs, 'dqdt', idx=data_subset)
        qnec_data = veros.tools.interpolate((xt_forc, yt_forc, np.arange(12)),
                                            qnec_raw, time_grid)
        vs.qnec[2:-2, 2:-2, :] = qnec_data * vs.maskT[2:-2, 2:-2, -1, np.newaxis]

        qsol_raw = self._get_data(vs, 'swf', idx=data_subset)
        qsol_data = veros.tools.interpolate((xt_forc, yt_forc, np.arange(12)),
                                            qsol_raw, time_grid)
        vs.qsol[2:-2, 2:-2, :] = -qsol_data * vs.maskT[2:-2, 2:-2, -1, np.newaxis]

        # SST and SSS
        sst_raw = self._get_data(vs, 'sst', idx=data_subset)
        sst_data = veros.tools.interpolate((xt_forc, yt_forc, np.arange(12)),
                                           sst_raw, time_grid)
        vs.t_star[2:-2, 2:-2, :] = sst_data * vs.maskT[2:-2, 2:-2, -1, np.newaxis]

        sss_raw = self._get_data(vs, 'sss', idx=data_subset)
        sss_data = veros.tools.interpolate((xt_forc, yt_forc, np.arange(12)),
                                           sss_raw, time_grid)
        vs.s_star[2:-2, 2:-2, :] = sss_data * vs.maskT[2:-2, 2:-2, -1, np.newaxis]

        if vs.enable_idemix:
            tidal_energy_raw = self._get_data(vs, 'tidal_energy', idx=data_subset)
            tidal_energy_data = veros.tools.interpolate(
                (xt_forc, yt_forc), tidal_energy_raw, t_grid[:-1]
            )
            mask_x, mask_y = (i + 2 for i in np.indices((vs.nx, vs.ny)))
            mask_z = np.maximum(0, vs.kbot[2:-2, 2:-2] - 1)
            tidal_energy_data[:, :] *= vs.maskW[mask_x, mask_y, mask_z] / vs.rho_0
            vs.forc_iw_bottom[2:-2, 2:-2] = tidal_energy_data

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
        t_rest = 30. * 86400.
        cp_0 = 3991.86795711963  # J/kg /K

        year_in_seconds = 360 * 86400.
        (n1, f1), (n2, f2) = veros.tools.get_periodic_interval(
            vs.time, year_in_seconds, year_in_seconds / 12., 12
        )

        vs.surface_taux[...] = f1 * vs.taux[:, :, n1] + f2 * vs.taux[:, :, n2]
        vs.surface_tauy[...] = f1 * vs.tauy[:, :, n1] + f2 * vs.tauy[:, :, n2]

        if vs.enable_tke:
            vs.forc_tke_surface[1:-1, 1:-1] = np.sqrt((0.5 * (vs.surface_taux[1:-1, 1:-1] + vs.surface_taux[:-2, 1:-1])) ** 2
                                                      + (0.5 * (vs.surface_tauy[1:-1, 1:-1] + vs.surface_tauy[1:-1, :-2])) ** 2) ** (3. / 2.)

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
        vs.diagnostics['cfl_monitor'].output_frequency = vs.dt_tracer * 100
        vs.diagnostics['tracer_monitor'].output_frequency = 86400.
        vs.diagnostics['snapshot'].output_frequency = 86400.
        vs.diagnostics['overturning'].output_frequency = 360 * 86400
        vs.diagnostics['overturning'].sampling_frequency = 86400.
        vs.diagnostics['energy'].output_frequency = 360 * 86400
        vs.diagnostics['energy'].sampling_frequency = 86400.
        vs.diagnostics['averages'].output_frequency = 10 * 86400
        vs.diagnostics['averages'].sampling_frequency = 3600.

        average_vars = ['surface_taux', 'surface_tauy', 'forc_temp_surface', 'forc_salt_surface',
                        'psi', 'temp', 'salt', 'u', 'v', 'w', 'Nsqr', 'Hd', 'rho', 'kappaH']
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

    @veros_method
    def after_timestep(self, vs):
        pass


@veros.tools.cli
def run(*args, **kwargs):
    simulation = GlobalFlexibleResolutionSetup(*args, **kwargs)
    simulation.setup()
    simulation.run()


if __name__ == '__main__':
    run()
