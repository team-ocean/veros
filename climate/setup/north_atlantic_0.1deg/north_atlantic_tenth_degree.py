from netCDF4 import Dataset
import scipy.interpolate
import scipy.spatial

from climate.pyom import PyOM, pyom_method, cyclic

class NorthAtlanticHighResolution(PyOM):
    """ Based on
    http://journals.ametsoc.org/doi/10.1175/1520-0485%282000%29030%3C1532%3ANSOTNA%3E2.0.CO%3B2
    """
    interpolator_initialized = False

    @pyom_method
    def set_parameter(self):
        """set main parameter
        """
        self.nx, self.ny, self.nz = (150,160,45)#(87,89,45)
        self.x_origin = -98.
        self.y_origin = -20.
        self._x_boundary = 17.2
        self._y_boundary = 72.6

        self.dt_mom = 3600.0
        self.dt_tracer = 3600.0

        self.runlen = 86400 * 365.0
        self.enable_diag_snapshots = True

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
        self.A_h = 5e4
        self.enable_hor_friction_cos_scaling = True
        self.hor_friction_cosPower = 3
        self.enable_tempsalt_sources = True

        self.enable_implicit_vert_friction = True
        self.enable_tke = False
        self.c_k = 0.1
        self.c_eps = 0.7
        self.alpha_tke = 30.0
        self.mxl_min = 1e-8
        self.tke_mxl_choice = 2

        self.K_gm_0 = 1000.0

        self.enable_idemix = False
        self.enable_idemix_hor_diffusion = True

        self.eq_of_state_type = 5

    @pyom_method
    def set_grid(self):
        with Dataset("forcing.cdf", "r") as forcing_file:
            self.dxt[2:-2] = (self._x_boundary - self.x_origin) / self.nx
            self.dyt[2:-2] = (self._y_boundary - self.y_origin) / self.ny
            self.dzt[:] = forcing_file.variables["dzt"][::-1] / 100.0

    @pyom_method
    def set_coriolis(self):
        self.coriolis_t[:,:] = 2 * self.omega * np.sin(self.yt[np.newaxis, :] / 180. * self.pi)

    @pyom_method
    def _init_interpolator(self, coords, mask):
        data_coords = tuple(c[mask] for c in np.meshgrid(*coords, indexing="ij"))
        delaunay = scipy.spatial.Delaunay(np.vstack([i.flatten() for i in data_coords]).T)
        self.linear_interpolator = lambda val: scipy.interpolate.LinearNDInterpolator(delaunay, val)


    @pyom_method
    def _interpolate(self, coords, var, grid=None, fill_value="nan"):
        if grid is None:
            if len(coords) == 2:
                interp_coords = np.meshgrid(self.xt[2:-2], self.yt[2:-2], self.zt[-1:], indexing="ij")
            elif len(coords) == 3:
                interp_coords = np.meshgrid(self.xt[2:-2], self.yt[2:-2], self.zt, indexing="ij")
        else:
            interp_coords = np.meshgrid(*grid, indexing="ij")
        interp_coords = np.rollaxis(np.asarray(interp_coords),0,len(interp_coords)+1)
        return self.linear_interpolator(var[var > -1e19])(interp_coords)
        # if fill_value == "nan":
        #     fill_value = np.nan
        # if grid is None:
        #     if len(coords) == 2:
        #         interp_coords = np.meshgrid(self.xt[2:-2], self.yt[2:-2], indexing="ij")
        #     elif len(coords) == 3:
        #         interp_coords = np.meshgrid(self.xt[2:-2], self.yt[2:-2], self.zt, indexing="ij")
        #else:
        #    interp_coords = np.meshgrid(*grid, indexing="ij")
        #interp_coords = np.rollaxis(np.asarray(interp_coords),0,len(coords)+1)
        #
        # var = np.array(var)
        # invalid_mask = var < -1e19
        # if np.any(invalid_mask):
        #     var[invalid_mask] = np.nan
        # interp_values = scipy.interpolate.interpn(coords, var, interp_coords,
        #                 method="linear", bounds_error=False, fill_value=fill_value)
        # mask = np.isnan(interp_values)
        # if np.any(mask):
        #     ccoords = np.meshgrid(*coords, indexing="ij")
        #     interp_values[mask] = scipy.interpolate.griddata(tuple(c[~invalid_mask] for c in ccoords), var[~invalid_mask], interp_coords,
        #                           method="nearest")[mask]
        # return interp_values

    @pyom_method
    def set_topography(self):
        with Dataset("forcing.cdf","r") as forcing_file:
            topo_x, topo_y, topo_kmt, topo_depth = (forcing_file.variables[k][...].T for k in ("xt","yt","kmt","depth_t"))
            topo_bottom_depth = np.where(topo_kmt > 0, -topo_depth[topo_kmt - 1], 100)
            interp_coords = np.meshgrid(self.xt[2:-2], self.yt[2:-2], indexing="ij")
            interp_coords = np.rollaxis(np.asarray(interp_coords),0,3)
            z_interp = scipy.interpolate.interpn((topo_x - 360, topo_y), topo_bottom_depth, interp_coords,
                                                  method="linear", bounds_error=False, fill_value=None)
        self.kbot[2:-2, 2:-2] = np.where(z_interp < 0., 1 + np.argmin(np.abs(z_interp[:, :, np.newaxis] - self.zt[np.newaxis, np.newaxis, :]), axis=2), 0)
        self.kbot *= self.kbot < self.nz


    @pyom_method
    def set_initial_conditions(self):
        with Dataset("forcing.cdf", "r") as forcing_file:
            forc_coords = [forcing_file.variables[k][...].T for k in ("xt","yt","zt")]
            forc_coords[0][...] += -360
            forc_coords[2][...] = -.01 * forc_coords[2][::-1]
            self._init_interpolator(forc_coords, forcing_file.variables["temp_ic"][::-1, :, :].T > -1e19)
            temp = self._interpolate(forc_coords, forcing_file.variables["temp_ic"][::-1, :, :].T)
            self.temp[2:-2, 2:-2, :, self.tau] = self.maskT[2:-2, 2:-2, :] * temp
            salt = 35. + 1000 * self._interpolate(forc_coords, forcing_file.variables["salt_ic"][::-1, :, :].T)
            self.salt[2:-2, 2:-2, :, self.tau] = self.maskT[2:-2, 2:-2, :] * salt

            self._taux = np.zeros((self.nx+4, self.ny+4, 12))
            self._tauy = np.zeros((self.nx+4, self.ny+4, 12))
            for k in xrange(12):
                self._taux[2:-2, 2:-2, k] = self._interpolate(forc_coords[:-1], forcing_file.variables["taux"][k,...].T, grid=(self.xu[2:-2], self.yt[2:-2])) / 10. / self.rho_0
                self._tauy[2:-2, 2:-2, k] = self._interpolate(forc_coords[:-1], forcing_file.variables["tauy"][k,...].T, grid=(self.xt[2:-2], self.yu[2:-2])) / 10. / self.rho_0

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
            for d in (self._sst_clim, self._sss_clim, self._sst_rest, self._sss_rest):
                d[...] *= d > -1e10

        with Dataset("restoring_zone.cdf", "r") as restoring_file:
            rest_coords = [restoring_file.variables[k][...].T for k in ("xt","yt","zt")]
            rest_coords[0][...] += -360
            rest_coords[2][...] = -.01 * forc_coords[2][::-1]

            # sponge layers
            self._t_star = np.zeros((self.nx+4, self.ny+4, self.nz, 12))
            self._s_star = np.zeros((self.nx+4, self.ny+4, self.nz, 12))
            self._rest_tscl = np.zeros((self.nx+4, self.ny+4, self.nz))

            self._rest_tscl[2:-2, 2:-2, :] = self._interpolate(rest_coords, restoring_file.variables["tscl"][0, ...].T)
            for k in xrange(12):
                self._t_star[2:-2, 2:-2, :, k] = self._interpolate(rest_coords, restoring_file.variables["t_star"][k,...].T)
                self._s_star[2:-2, 2:-2, :, k] = self._interpolate(rest_coords, restoring_file.variables["s_star"][k,...].T)
            for d in (self._rest_tscl, self._t_star, self._s_star):
                d[...] *= d > -1e10

        if self.enable_idemix:
            f = np.fromfile("tidal_energy.bin", dtype =">f").reshape(self.nx, self.ny, order="F") / self.rho_0
            self.forc_iw_bottom[2:-2, 2:-2] = f
            f = np.fromfile("wind_energy.bin", dtype =">f").reshape(self.nx, self.ny, order="F") / self.rho_0 * 0.2
            self.forc_iw_surface[2:-2,2:-2] = f

    @pyom_method
    def _get_periodic_interval(self,currentTime,cycleLength,recSpacing,nbRec):
        """  interpolation routine taken from mitgcm
        """
        locTime = currentTime - recSpacing * 0.5 + cycleLength * (2 - round(currentTime/cycleLength))
        tmpTime = locTime % cycleLength
        tRec1 = int(tmpTime/recSpacing)
        tRec2 = tRec1 % int(nbRec)
        wght2 = (tmpTime - recSpacing*(tRec1 - 1)) / recSpacing
        wght1 = 1.0 - wght2
        return (tRec1,wght1), (tRec2,wght2)

    @pyom_method
    def set_forcing(self):
        year_in_seconds = 365 * 86400.0
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

    @pyom_method
    def set_diagnostics(self):
        average_vars = {"temp", "salt", "u", "v", "w", "surface_taux", "surface_tauy", "psi"}
        for var in average_vars:
            self.variables[var].average = True


if __name__ == "__main__":
    sim = NorthAtlanticHighResolution()
    sim.run(snapint = 86400.0)
