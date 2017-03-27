from netCDF4 import Dataset
from PIL import Image
import numpy as np
import scipy.interpolate
import scipy.spatial
import scipy.ndimage

from climate.pyom import PyOM, pyom_method
from climate.pyom.core import cyclic

class NorthAtlantic(PyOM):
    """ Inspired by
    http://journals.ametsoc.org/doi/10.1175/1520-0485%282000%29030%3C1532%3ANSOTNA%3E2.0.CO%3B2
    """

    @pyom_method
    def set_parameter(self):
        """set main parameter
        """
        self.nx, self.ny, self.nz = 250, 200, 50
        self.x_origin = -98.
        self.y_origin = -18.
        self._x_boundary = 17.2
        self._y_boundary = 70.
        self._max_depth = 5800.

        self.dt_mom = 3600. / 2.
        self.dt_tracer = 3600. / 2.

        self.runlen = 86400 * 365. * 10.

        self.coord_degree = True

        self.congr_epsilon = 1e-10
        self.congr_max_iterations = 20000
        self.enable_streamfunction = True

        self.enable_neutral_diffusion = True
        self.enable_skew_diffusion = True
        self.K_iso_0 = 1000.0
        self.K_iso_steep = 200.0
        self.iso_dslope = 1./1000.0
        self.iso_slopec = 4./1000.0

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

    def _gaussian(self, x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    def _normalize_sum(self, var, sum_value, minimum_value=0.):
        var[0] = 0.
        var *= (sum_value - len(var) * minimum_value) / var.sum()
        return var + minimum_value

    def set_grid(self):
        self.dxt[2:-2] = (self._x_boundary - self.x_origin) / self.nx
        self.dyt[2:-2] = (self._y_boundary - self.y_origin) / self.ny
        ddzt = self._gaussian(np.arange(self.nz), 0.5 * self.nz, 0.125 * self.nz)
        self.dzt[...] = self._normalize_sum(np.cumsum(ddzt), self._max_depth, 10.)[::-1]

    def set_coriolis(self):
        self.coriolis_t[:,:] = 2 * self.omega * np.sin(self.yt[np.newaxis, :] / 180. * self.pi)

    def _interpolate_along_axis(self, coords, arr, interp_coords, axis):
        assert len(coords) == arr.shape[axis]

        diff = coords[np.newaxis, :] - interp_coords[:, np.newaxis]
        diff_m = np.where(diff <= 0, diff, np.inf)

        i_m = np.argmin(np.abs(diff_m), axis=1)
        i_p = np.minimum(len(coords) - 1, i_m + 1)

        full_shape = [np.newaxis] * arr.ndim
        full_shape[axis] = slice(None)
        s = [slice(None)] * arr.ndim
        s[axis] = i_m
        mask = np.isnan(arr[s])
        i_m_full = np.where(mask, i_p[full_shape], i_m[full_shape])
        s[axis] = i_p
        mask = np.isnan(arr[s])
        i_p_full = np.where(mask, i_m[full_shape], i_p[full_shape])

        pos = np.where(i_p_full == i_m_full, 1., ((coords[i_p] - interp_coords) \
                        / (coords[i_p] - coords[i_m] + 1e-20))[full_shape])

        indices_p, indices_m = np.indices(i_m_full.shape), np.indices(i_m_full.shape)
        indices_p[axis] = i_p_full
        indices_m[axis] = i_m_full
        return arr[tuple(indices_p)] * (1-pos) + arr[tuple(indices_m)] * pos

    def _fill_holes(self, data):
        data = data.copy()
        shape = data.shape
        dim = data.ndim
        flag = np.zeros(shape, dtype=bool)
        t_ct = int(data.size/5)
        flag[~np.isnan(data)] = True

        slcs = [slice(None)] * dim

        while np.any(~flag): # as long as there are any False's in flag
            for i in range(dim): # do each axis
                # make slices to shift view one element along the axis
                slcs1 = slcs[:]
                slcs2 = slcs[:]
                slcs1[i] = slice(0, -1)
                slcs2[i] = slice(1, None)

                # replace from the right
                repmask = np.logical_and(~flag[slcs1], flag[slcs2])
                data[slcs1][repmask] = data[slcs2][repmask]
                flag[slcs1][repmask] = True

                # replace from the left
                repmask = np.logical_and(~flag[slcs2], flag[slcs1])
                data[slcs2][repmask] = data[slcs1][repmask]
                flag[slcs2][repmask] = True
        return data

    def _interpolate(self, coords, var, grid=None, missing_value=-1e20):
        if grid is None:
            grid = (self.xt[2:-2], self.yt[2:-2])
            if len(coords) == 3:
                grid += (self.zt,)

        var = np.array(var)
        invalid_mask = var == missing_value
        var[invalid_mask] = np.nan

        interp_values = var
        for i, (x, x_new) in enumerate(zip(coords, grid)):
            interp_values = self._interpolate_along_axis(x, interp_values, x_new, i)
        interp_values = self._fill_holes(interp_values)
        return interp_values

    def set_topography(self):
        with Dataset("ETOPO1_Bed_g_gmt4_NA.nc","r") as topography_file:
            topo_x, topo_y, topo_bottom_depth = (topography_file.variables[k][...].T for k in ("x","y","z"))
        topo_mask = np.flipud(np.asarray(Image.open("topo_mask.png"))).T
        topo_bottom_depth *= 1 - topo_mask
        topo_bottom_depth = scipy.ndimage.gaussian_filter(topo_bottom_depth, sigma=(len(topo_x) / self.nx, len(topo_y) / self.ny))
        interp_coords = np.meshgrid(self.xt[2:-2], self.yt[2:-2], indexing="ij")
        interp_coords = np.rollaxis(np.asarray(interp_coords),0,3)
        z_interp = scipy.interpolate.interpn((topo_x, topo_y), topo_bottom_depth, interp_coords,
                                              method="nearest", bounds_error=False, fill_value=0)
        self.kbot[2:-2, 2:-2] = np.where(z_interp < 0., 1 + np.argmin(np.abs(z_interp[:, :, np.newaxis] \
                                                     - self.zt[np.newaxis, np.newaxis, :]), axis=2), 0)
        self.kbot *= self.kbot < self.nz

    def set_initial_conditions(self):
        with Dataset("forcing.cdf", "r") as forcing_file:
            forc_coords = [forcing_file.variables[k][...].T for k in ("xt","yt","zt")]
            forc_coords[0][...] += -360
            forc_coords[2][...] = -0.01 * forc_coords[2][::-1]
            temp = self._interpolate(forc_coords, forcing_file.variables["temp_ic"][::-1, ...].T)
            self.temp[2:-2, 2:-2, :, self.tau] = self.maskT[2:-2, 2:-2, :] * temp
            salt = 35. + 1000 * self._interpolate(forc_coords, forcing_file.variables["salt_ic"][::-1, ...].T)
            self.salt[2:-2, 2:-2, :, self.tau] = self.maskT[2:-2, 2:-2, :] * salt

            self._taux = np.zeros((self.nx+4, self.ny+4, 12))
            self._tauy = np.zeros((self.nx+4, self.ny+4, 12))
            forc_u_coords_hor = [forcing_file.variables[k][...].T for k in ("xu","yu")]
            forc_u_coords_hor[0][...] += -360
            for k in xrange(12):
                self._taux[2:-2, 2:-2, k] = self._interpolate(forc_u_coords_hor, forcing_file.variables["taux"][k,...].T,
                                                              grid=(self.xu[2:-2], self.yt[2:-2])) / 10. / self.rho_0
                self._tauy[2:-2, 2:-2, k] = self._interpolate(forc_u_coords_hor, forcing_file.variables["tauy"][k,...].T,
                                                              grid=(self.xt[2:-2], self.yu[2:-2])) / 10. / self.rho_0

            for t in (self._taux, self._tauy):
                if self.enable_cyclic_x:
                    cyclic.setcyclic_x(t)

            # heat flux and salinity restoring
            self._sst_clim = np.zeros((self.nx+4, self.ny+4, 12))
            self._sss_clim = np.zeros((self.nx+4, self.ny+4, 12))
            self._sst_rest = np.zeros((self.nx+4, self.ny+4, 12))
            self._sss_rest = np.zeros((self.nx+4, self.ny+4, 12))

            sst_clim, sss_clim, sst_rest, sss_rest = [forcing_file.variables[k][...].T for k in ("sst_clim","sss_clim","sst_rest","sss_rest")]
            for k in xrange(12):
                self._sst_clim[2:-2,2:-2,k] = self._interpolate(forc_coords[:-1], sst_clim[...,k])
                self._sss_clim[2:-2,2:-2,k] = self._interpolate(forc_coords[:-1], sss_clim[...,k]) * 1000 + 35
                self._sst_rest[2:-2,2:-2,k] = self._interpolate(forc_coords[:-1], sst_rest[...,k]) * 41868.
                self._sss_rest[2:-2,2:-2,k] = self._interpolate(forc_coords[:-1], sss_rest[...,k]) / 100.0

        with Dataset("restoring_zone.cdf", "r") as restoring_file:
            rest_coords = [restoring_file.variables[k][...].T for k in ("xt","yt","zt")]
            rest_coords[0][...] += -360

            # sponge layers
            self._t_star = np.zeros((self.nx+4, self.ny+4, self.nz, 12))
            self._s_star = np.zeros((self.nx+4, self.ny+4, self.nz, 12))
            self._rest_tscl = np.zeros((self.nx+4, self.ny+4, self.nz))

            self._rest_tscl[2:-2, 2:-2, :] = self._interpolate(rest_coords, restoring_file.variables["tscl"][0, ...].T)
            for k in xrange(12):
                self._t_star[2:-2, 2:-2, :, k] = self._interpolate(rest_coords, restoring_file.variables["t_star"][k,...].T, missing_value=0.)
                self._s_star[2:-2, 2:-2, :, k] = self._interpolate(rest_coords, restoring_file.variables["s_star"][k,...].T, missing_value=0.)

        if self.enable_idemix:
            f = np.load("tidal_energy.npy") / self.rho_0
            self.forc_iw_bottom[2:-2, 2:-2] = self._interpolate(forc_coords[:-1], f)
            f = np.load("wind_energy.npy") / self.rho_0 * 0.2
            self.forc_iw_surface[2:-2,2:-2] = self._interpolate(forc_coords[:-1], f)

    def _get_periodic_interval(self,currentTime,cycleLength,recSpacing,nbRec):
        """  interpolation routine taken from mitgcm
        """
        locTime = currentTime - recSpacing * 0.5 + cycleLength * (2 - round(currentTime/cycleLength))
        tmpTime = locTime % cycleLength
        tRec1 = 1 + int(tmpTime/recSpacing)
        tRec2 = 1 + tRec1 % int(nbRec)
        wght2 = (tmpTime - recSpacing*(tRec1 - 1)) / recSpacing
        wght1 = 1.0 - wght2
        return (tRec1-1, wght1), (tRec2-1, wght2)

    @pyom_method
    def set_forcing(self):
        year_in_seconds = 360 * 86400.0
        (n1,f1), (n2,f2) = self._get_periodic_interval(self.itt * self.dt_tracer, year_in_seconds, year_in_seconds / 12., 12)

        self.surface_taux[...] = (f1 * self._taux[:,:,n1] + f2 * self._taux[:,:,n2])
        self.surface_tauy[...] = (f1 * self._tauy[:,:,n1] + f2 * self._tauy[:,:,n2])

        if self.enable_tke:
            self.forc_tke_surface[1:-1,1:-1] = np.sqrt((0.5 * (self.surface_taux[1:-1,1:-1] + self.surface_taux[:-2,1:-1]))**2 \
                                                     + (0.5 * (self.surface_tauy[1:-1,1:-1] + self.surface_tauy[1:-1,:-2]))**2 \
                                                     ) ** (3./2.)
        cp_0 = 3991.86795711963
        self.forc_temp_surface[...] = (f1 * self._sst_rest[:,:,n1] + f2 * self._sst_rest[:,:,n2]) * \
                                      (f1 * self._sst_clim[:,:,n1] + f2 * self._sst_clim[:,:,n2] \
                                        - self.temp[:,:,-1,self.tau]) * self.maskT[:,:,-1] / cp_0 / self.rho_0
        self.forc_salt_surface[...] = (f1 * self._sss_rest[:,:,n1] + f2 * self._sss_rest[:,:,n2]) * \
                                      (f1 * self._sss_clim[:,:,n1] + f2 * self._sss_clim[:,:,n2] \
                                        - self.salt[:,:,-1,self.tau]) * self.maskT[:,:,-1]

        ice_mask = (self.temp[:,:,-1,self.tau] * self.maskT[:,:,-1] <= -1.8) & (self.forc_temp_surface <= 0.0)
        self.forc_temp_surface[...] *= ~ice_mask
        self.forc_salt_surface[...] *= ~ice_mask

        if self.enable_tempsalt_sources:
            self.temp_source[...] = self.maskT * self._rest_tscl \
                                * (f1 * self._t_star[:,:,:,n1] + f2 * self._t_star[:,:,:,n2] - self.temp[:,:,:,self.tau])
            self.salt_source[...] = self.maskT * self._rest_tscl \
                                * (f1 * self._s_star[:,:,:,n1] + f2 * self._s_star[:,:,:,n2] - self.salt[:,:,:,self.tau])

    def set_diagnostics(self):
        self.diagnostics["snapshot"].output_frequency = 3600. * 24 * 10
        average_vars = {"temp", "salt", "u", "v", "w", "surface_taux", "surface_tauy", "psi"}
        for var in average_vars:
            self.variables[var].average = True


if __name__ == "__main__":
    sim = NorthAtlantic()
    sim.run()
