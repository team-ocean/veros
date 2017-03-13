import os
import logging
from netCDF4 import Dataset

from climate.pyom import PyOMLegacy, diagnostics, pyom_method

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_FILES = dict(
    dz = "dz.bin",
    temperature = "lev_t.bin",
    salt = "lev_s.bin",
    sss = "lev_sss.bin",
    sst = "lev_sst.bin",
    tau_x = "trenberth_taux.bin",
    tau_y = "trenberth_tauy.bin",
    q_net = "ncep_qnet.bin",
    bathymetry = "bathymetry.bin",
    tidal_energy = "tidal_energy.bin",
    wind_energy = "wind_energy.bin",
    ecmwf = "ecmwf_4deg_monthly.cdf",
)
DATA_FILES = {key: os.path.join(BASE_PATH, val) for key, val in DATA_FILES.items()}


class GlobalFourDegree(PyOMLegacy):
    """
    global 4 deg model with 15 levels
    """
    @pyom_method
    def set_parameter(self):
        m=self.fortran.main_module

        (m.nx,m.ny,m.nz) = (90,40,15)
        m.dt_mom = 1800.0
        m.dt_tracer = 86400.0

        m.coord_degree = True
        m.enable_cyclic_x = True

        m.congr_epsilon = 1e-8
        m.congr_max_iterations = 20000
        m.enable_streamfunction = True

        m.enable_diag_ts_monitor = True
        m.ts_monint = 365*86400./24.
        m.enable_diag_snapshots  = True
        m.snapint  =  365*86400. /24.0
        m.enable_diag_overturning= True
        m.overint  =  365*86400./24.0
        m.overfreq = m.dt_tracer
        m.enable_diag_energy     = True;
        m.energint = 365*86400./24.
        m.energfreq = 86400
        m.enable_diag_averages   = True
        m.aveint  = 86400.*30
        m.avefreq = 86400

        I=self.fortran.isoneutral_module
        I.enable_neutral_diffusion = True
        I.K_iso_0 = 1000.0
        I.K_iso_steep = 1000.0
        I.iso_dslope=4./1000.0
        I.iso_slopec=1./1000.0
        I.enable_skew_diffusion = True

        m.enable_hor_friction  = True
        m.A_h = (4*m.degtom)**3*2e-11
        m.enable_hor_friction_cos_scaling = True
        m.hor_friction_cosPower = 1

        m.enable_implicit_vert_friction = True
        T=self.fortran.tke_module
        T.enable_tke = True
        T.c_k = 0.1
        T.c_eps = 0.7
        T.alpha_tke = 30.0
        T.mxl_min = 1e-8
        T.tke_mxl_choice = 2
        T.enable_tke_superbee_advection = True

        E=self.fortran.eke_module
        E.enable_eke = True
        E.eke_k_max  = 1e4
        E.eke_c_k    = 0.4
        E.eke_c_eps  = 0.5
        E.eke_cross  = 2.
        E.eke_crhin  = 1.0
        E.eke_lmin   = 100.0
        E.enable_eke_superbee_advection = True

        I=self.fortran.idemix_module
        I.enable_idemix = True
        I.enable_idemix_hor_diffusion = True
        I.enable_eke_diss_surfbot = True
        I.eke_diss_surfbot_frac = 0.2 # fraction which goes into bottom
        I.enable_idemix_superbee_advection = True

        m.eq_of_state_type = 5

    @pyom_method
    def _read_binary(self, var, shape=(-1,), dtype=">f"):
        data = np.array(np.fromfile(DATA_FILES[var], dtype=dtype, count=np.prod(shape)).reshape(shape, order="F"), dtype=np.float)
        return data

    @pyom_method
    def set_grid(self):
        m=self.fortran.main_module
        ddz = np.array([50.,70.,100.,140.,190.,240.,290.,340.,390.,440.,490.,540.,590.,640.,690.])
        m.dzt[:] = ddz[::-1]
        m.dxt[:] = 4.0
        m.dyt[:] = 4.0
        m.y_origin = -76.0
        m.x_origin = 4.0

    @pyom_method
    def set_coriolis(self):
        m=self.fortran.main_module
        m.coriolis_t[...] = 2*m.omega*np.sin(m.yt[np.newaxis,:]/180.*m.pi)

    @pyom_method
    def set_topography(self):
        m = self.main_module
        bathymetry_data = self._read_binary("bathymetry", (m.nx, m.ny), dtype=">i")
        salt_data = self._read_binary("salt", (m.nx, m.ny, m.nz))[:,:,::-1]
        for k in xrange(m.nz-1, -1, -1):
            mask_salt = salt_data[:,:,k] != 0.
            m.kbot[2:-2, 2:-2][mask_salt] = k+1
        mask_bathy = bathymetry_data == 0
        m.kbot[2:-2, 2:-2][mask_bathy] = 0
        m.kbot[m.kbot == m.nz] = 0

    @pyom_method
    def set_initial_conditions(self):
        """ setup initial conditions
        """
        m = self.fortran.main_module
        self.taux, self.tauy, self.qnec, self.qnet, self.sss_clim, self.sst_clim = (np.zeros((m.nx+4,m.ny+4,12)) for _ in range(6))

        # initial conditions for T and S
        temp_data = self._read_binary("temperature", shape=(m.nx,m.ny,m.nz))[:,:,::-1]
        m.temp[2:-2,2:-2,:,:2] = temp_data[...,np.newaxis] * m.maskT[2:-2,2:-2,:,np.newaxis]

        salt_data = self._read_binary("salt", shape=(m.nx,m.ny,m.nz))[:,:,::-1]
        m.salt[2:-2,2:-2,:,:2] = salt_data[...,np.newaxis] * m.maskT[2:-2,2:-2,:,np.newaxis]

        # use Trenberth wind stress from MITgcm instead of ECMWF (also contained in ecmwf_4deg.cdf)
        self.taux[2:-2,2:-2,:] = self._read_binary("tau_x",shape=(m.nx,m.ny,12)) / m.rho_0
        self.tauy[2:-2,2:-2,:] = self._read_binary("tau_y",shape=(m.nx,m.ny,12)) / m.rho_0

        # heat flux
        with Dataset(DATA_FILES["ecmwf"],"r") as ecmwf_data:
            self.qnec[2:-2,2:-2,:] = np.array(ecmwf_data.variables["Q3"]).transpose()
            self.qnec[self.qnec <= -1e10] = 0.0

        q = self._read_binary("q_net",shape=(m.nx,m.ny,12))
        self.qnet[2:-2,2:-2,:] = -q
        self.qnet[self.qnet <= -1e10] = 0.0

        fxa = np.sum(self.qnet[2:-2, 2:-2, :] * m.area_t[2:-2, 2:-2, np.newaxis]) / 12 / np.sum(m.area_t[2:-2, 2:-2])
        logging.info(" removing an annual mean heat flux imbalance of %e W/m^2" % fxa)
        self.qnet[...] = (self.qnet - fxa) * m.maskT[: ,: , -1, np.newaxis]

        # SST and SSS
        self.sst_clim[2:-2,2:-2,:] = self._read_binary("sst",shape=(m.nx,m.ny,12))
        self.sss_clim[2:-2,2:-2,:] = self._read_binary("sss",shape=(m.nx,m.ny,12))

        idm = self.fortran.idemix_module
        if idm.enable_idemix:
            idm.forc_iw_bottom[2:-2,2:-2] = self._read_binary("tidal_energy", shape=(m.nx,m.ny)) / m.rho_0
            idm.forc_iw_surface[2:-2,2:-2] = self._read_binary("wind_energy", shape=(m.nx,m.ny)) / m.rho_0 * 0.2

    @pyom_method
    def _get_periodic_interval(self, currentTime, cycleLength, recSpacing, nbrec):
        # interpolation routine taken from mitgcm
        locTime = currentTime - recSpacing * 0.5 + cycleLength * (2 - round(currentTime / cycleLength))
        tmpTime = locTime % cycleLength
        tRec1 = int(tmpTime / recSpacing)
        tRec2 = int(tmpTime % nbrec)
        wght2 = (tmpTime - recSpacing * tRec1) / recSpacing
        wght1 = 1. - wght2
        return (tRec1, wght1), (tRec2, wght2)

    @pyom_method
    def set_forcing(self):
        m=self.fortran.main_module

        (n1,f1), (n2,f2) = self._get_periodic_interval((m.itt - 1) * m.dt_tracer, 365*86400.0, 365*86400./12., 12)

        # wind stress
        m.surface_taux[:]=(f1*self.taux[:,:,n1] + f2*self.taux[:,:,n2])
        m.surface_tauy[:]=(f1*self.tauy[:,:,n1] + f2*self.tauy[:,:,n2])

        # tke flux
        tkm=self.fortran.tke_module
        if tkm.enable_tke:
            tkm.forc_tke_surface[1:-1,1:-1] = np.sqrt((0.5*(m.surface_taux[1:-1,1:-1] + m.surface_taux[:-2,1:-1]))**2  \
                                            + (0.5 * (m.surface_tauy[1:-1,1:-1] + m.surface_tauy[1:-1,:-2]))**2)**(3./2.)
        # heat flux : W/m^2 K kg/J m^3/kg = K m/s
        cp_0 = 3991.86795711963
        sst  =  f1*self.sst_clim[:,:,n1] + f2*self.sst_clim[:,:,n2]
        qnec =  f1*self.qnec[:,:,n1] + f2*self.qnec[:,:,n2]
        qnet =  f1*self.qnet[:,:,n1] + f2*self.qnet[:,:,n2]
        m.forc_temp_surface[:] = (qnet + qnec * (sst - m.temp[:,:,-1,self.get_tau()])) * m.maskT[:,:,-1] / cp_0 / m.rho_0

        # salinity restoring
        t_rest= 30 * 86400.0
        sss = f1*self.sss_clim[:,:,n1] + f2*self.sss_clim[:,:,n2]
        m.forc_salt_surface[:] = 1. / t_rest * (sss - m.salt[:,:,-1,self.get_tau()]) * m.maskT[:,:,-1] * m.dzt[-1]

        # apply simple ice mask
        mask = (m.temp[:,:,-1,self.get_tau()] * m.maskT[:,:,-1] <= -1.8) & (m.forc_temp_surface <= 0.0)
        m.forc_temp_surface[mask] = 0.0
        m.forc_salt_surface[mask] = 0.0

        if m.enable_tempsalt_sources:
            m.temp_source[:] = m.maskT * self.rest_tscl * (f1*self.t_star[:,:,:,n1] + f2*self.t_star[:,:,:,n2] - m.temp[:,:,:,self.get_tau()])
            m.salt_source[:] = m.maskT * self.rest_tscl * (f1*self.s_star[:,:,:,n1] + f2*self.s_star[:,:,:,n2] - m.salt[:,:,:,self.get_tau()])

    @pyom_method
    def set_diagnostics(self):
        m = self.fortran.main_module
        for var in ("temp", "salt", "u", "v", "w", "surface_taux", "surface_tauy", "psi"):
            m.variables[var].average = True

if __name__ == "__main__":
    GlobalFourDegree().run(snapint= 86400.0*10, runlen = 86400.*365)
