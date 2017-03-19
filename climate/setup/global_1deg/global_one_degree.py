import os

from climate.pyom import PyOMLegacy, cyclic, pyom_method

MAIN_OPTIONS = dict(
    nx = 360,
    ny = 160,
    nz = 115,
    dt_mom = 3600.0,
    dt_tracer = 3600.0,

    coord_degree = True,
    enable_cyclic_x = True,

    enable_diag_ts_monitor = True,
    ts_monint = 86400.0,
    enable_diag_snapshots = True,

    enable_diag_overturning = True,
    overint = 365 * 86400,
    overfreq = 365 * 86400 / 24.,
    enable_diag_energy = True,
    energint = 365 * 86400,
    energfreq = 365 * 86400 / 24.,
    enable_diag_averages = True,
    aveint = 365 * 86400,
    avefreq = 365 * 86400 / 24.,

    congr_epsilon = 1e-10,
    congr_max_iterations = 10000,
    enable_streamfunction = True,

    enable_hor_friction = True,
    A_h = 5e4,
    enable_hor_friction_cos_scaling = True,
    hor_friction_cosPower = 1,
    enable_tempsalt_sources = True,
    enable_implicit_vert_friction = True,

    eq_of_state_type = 3,
)

ISONEUTRAL_OPTIONS = dict(
    enable_neutral_diffusion = True,
    K_iso_0 = 1000.0,
    K_iso_steep = 50.0,
    iso_dslope = 0.005,
    iso_slopec = 0.005,
    enable_skew_diffusion = True,
)

TKE_OPTIONS = dict(
    enable_tke = True,
    c_k = 0.1,
    c_eps = 0.7,
    alpha_tke = 30.0,
    mxl_min = 1e-8,
    tke_mxl_choice = 2,
    enable_tke_superbee_advection = True,
)

EKE_OPTIONS = dict(
    enable_eke = True,
    eke_k_max = 1e4,
    eke_c_k = 0.4,
    eke_c_eps = 0.5,
    eke_cross = 2.,
    eke_crhin = 1.0,
    eke_lmin = 100.0,
    enable_eke_superbee_advection = True,
    enable_eke_isopycnal_diffusion = True,
)

IDEMIX_OPTIONS = dict(
    enable_idemix = True,
    enable_eke_diss_surfbot = True,
    eke_diss_surfbot_frac = 0.2, # fraction which goes into bottom
    enable_idemix_superbee_advection = True,
    enable_idemix_hor_diffusion = True,
    #np = 17+2,
    #enable_idemix_M2 = True,
    #enable_idemix_niw = True,
    #omega_M2 = 2*np.pi / (12*60*60 + 25.2 * 60), # M2 frequency in 1/s
)

BASE_PATH = os.path.dirname(os.path.realpath(__file__))

DATA_FILES = dict(
    dz = "dz.npy",
    temperature = "lev_clim_temp.npy",
    salt = "lev_clim_salt.npy",
    sss = "lev_sss.npy",
    tau_x = "ECMWFBB_taux.npy",
    tau_y = "ECMWFBB_tauy.npy",
    q_net = "ECMWFBB_qnet.npy",
    dqdt = "ECMWFBB_dqdt.npy",
    swf = "ECMWFBB_swf.npy",
    sst = "ECMWFBB_target_sst.npy",
    bathymetry = "bathymetry.npy",
    tidal_energy = "tidal_energy.npy",
    wind_energy = "wind_energy_ncep.npy"
)
DATA_FILES = {key: os.path.join(BASE_PATH, val) for key, val in DATA_FILES.items()}

class GlobalOneDegree(PyOMLegacy):
    """
    Global 1 degree model with 115 vertical levels, translated from setup1.f90.

    https://wiki.zmaw.de/ifm/TO/pyOM2/1x1%20global%20model
    """

    @pyom_method
    def set_parameter(self):
        """
        set main parameters
        """
        for name, filepath in DATA_FILES.items():
            if not os.path.isfile(filepath):
                raise RuntimeError("{} data file {} not found".format(name,filepath))

        self._set_parameters(self.main_module, MAIN_OPTIONS)
        self._set_parameters(self.isoneutral_module, ISONEUTRAL_OPTIONS)
        self._set_parameters(self.tke_module, TKE_OPTIONS)
        self._set_parameters(self.eke_module, EKE_OPTIONS)
        self._set_parameters(self.idemix_module, IDEMIX_OPTIONS)

    @pyom_method
    def _set_parameters(self,module,parameters):
        for key, attribute in parameters.items():
            setattr(module,key,attribute)

    @pyom_method
    def _read_binary(self, var):
        return np.load(DATA_FILES[var])

    @pyom_method
    def set_grid(self):
        dz_data = self._read_binary("dz")
        m = self.main_module
        m.dzt[...] = dz_data[::-1]
        m.dxt[...] = 1.0
        m.dyt[...] = 1.0
        m.y_origin = -79.
        m.x_origin = 91.

    @pyom_method
    def set_coriolis(self):
        m = self.main_module
        m.coriolis_t[...] = 2 * m.omega * np.sin(m.yt[np.newaxis, :] / 180. * m.pi)

    @pyom_method
    def set_initial_conditions(self):
        m = self.main_module
        self.t_star = np.zeros((m.nx+4, m.ny+4, 12))
        self.s_star = np.zeros((m.nx+4, m.ny+4, 12))
        self.qnec = np.zeros((m.nx+4, m.ny+4, 12))
        self.qnet = np.zeros((m.nx+4, m.ny+4, 12))
        self.qsol = np.zeros((m.nx+4, m.ny+4, 12))
        self.divpen_shortwave = np.zeros(m.nz)
        self.taux = np.zeros((m.nx+4, m.ny+4, 12))
        self.tauy = np.zeros((m.nx+4, m.ny+4, 12))

        rpart_shortwave = 0.58
        efold1_shortwave = 0.35
        efold2_shortwave = 23.0

        # initial conditions
        temp_data = self._read_binary("temperature")
        m.temp[2:-2, 2:-2, :, 0] = temp_data[..., ::-1] * m.maskT[2:-2, 2:-2, :]
        m.temp[2:-2, 2:-2, :, 1] = temp_data[..., ::-1] * m.maskT[2:-2, 2:-2, :]

        salt_data = self._read_binary("salt")
        m.salt[2:-2, 2:-2, :, 0] = salt_data[..., ::-1] * m.maskT[2:-2, 2:-2, :]
        m.salt[2:-2, 2:-2, :, 1] = salt_data[..., ::-1] * m.maskT[2:-2, 2:-2, :]

        # wind stress on MIT grid
        taux_data = self._read_binary("tau_x")
        self.taux[2:-2, 2:-2, :] = taux_data / m.rho_0
        self.taux[self.taux < -99.9] = 0.

        tauy_data = self._read_binary("tau_y")
        self.tauy[2:-2, 2:-2, :] = tauy_data / m.rho_0
        self.tauy[self.tauy < -99.9] = 0.

        if m.enable_cyclic_x:
            cyclic.setcyclic_x(self.taux)
            cyclic.setcyclic_x(self.tauy)

        # Qnet and dQ/dT and Qsol
        qnet_data = self._read_binary("q_net")
        self.qnet[2:-2, 2:-2, :] = -qnet_data * m.maskT[2:-2, 2:-2, -1, np.newaxis]

        qnec_data = self._read_binary("dqdt")
        self.qnec[2:-2, 2:-2, :] = qnec_data * m.maskT[2:-2, 2:-2, -1, np.newaxis]

        qsol_data = self._read_binary("swf")
        self.qsol[2:-2, 2:-2, :] = -qsol_data * m.maskT[2:-2, 2:-2, -1, np.newaxis]

        # SST and SSS
        sst_data = self._read_binary("sst")
        self.t_star[2:-2, 2:-2, :] = sst_data * m.maskT[2:-2, 2:-2, -1, np.newaxis]

        sss_data = self._read_binary("sss")
        self.s_star[2:-2, 2:-2, :] = sss_data * m.maskT[2:-2, 2:-2, -1, np.newaxis]

        idm = self.idemix_module
        if idm.enable_idemix:
            tidal_energy_data = self._read_binary("tidal_energy")
            mask_x, mask_y = (i+2 for i in np.indices((m.nx, m.ny)))
            mask_z = np.maximum(0, m.kbot[2:-2, 2:-2] - 1)
            tidal_energy_data[:, :] *= m.maskW[mask_x, mask_y, mask_z] / m.rho_0

            if idm.enable_idemix_M2:
                idm.forc_M2[2:-2, 2:-2, 1:-1] = 0.5 * tidal_energy_data[..., np.newaxis] / (2*m.pi)
                idm.forc_iw_bottom[2:-2, 2:-2] = 0.5 * tidal_energy_data
            else:
                idm.forc_iw_bottom[2:-2, 2:-2] = tidal_energy_data

            wind_energy_data = self._read_binary("wind_energy")
            wind_energy_data[...] *= m.maskW[2:-2, 2:-2, -1] / m.rho_0 * 0.2

            if idm.enable_idemix_niw:
                idm.forc_niw[2:-2, 2:-2, :m.np-1] = 1.0 * wind_energy_data[..., np.newaxis] / (2*m.pi)
                idm.forc_iw_surface[2:-2, 2:-2] = 0.
            else:
                idm.forc_iw_surface[2:-2, 2:-2] = wind_energy_data

            if idm.enable_idemix_niw:
                idm.omega_niw[2:-2] = np.maximum(1e-8, np.abs(1.05 * m.coriolis_t))

            if idm.enable_idemix_niw or idm.enable_idemix_M2:
                """
                 if enable_idemix_niw .or. enable_idemix_M2:
                  iret = nf_open("hrms_1deg.nc",NF_nowrite,ncid)
                  iret = nf_inq_varid(ncid,"HRMS",id)
                  iret = nf_get_vara_double(ncid,id,(/is_pe,js_pe/),(/ie_pe-is_pe+1,je_pe-js_pe+1/),topo_hrms(is_pe:ie_pe,js_pe:je_pe))
                  iret = nf_inq_varid(ncid,"LAM",id)
                  iret = nf_get_vara_double(ncid,id,(/is_pe,js_pe/),(/ie_pe-is_pe+1,je_pe-js_pe+1/),topo_lam(is_pe:ie_pe,js_pe:je_pe))
                  ncclos (ncid, iret)

                  border_exchg_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,topo_hrms)
                  setcyclic_xy   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,topo_hrms)
                  border_exchg_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,topo_lam)
                  setcyclic_xy   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,topo_lam)
                 endif
                 """
                # NOTE: file hrms_1deg.nc needed
                raise NotImplementedError

        """
        Initialize penetration profile for solar radiation
        and store divergence in divpen
        note that pen(nz) is set 0.0 instead of 1.0 to compensate for the
        shortwave part of the total surface flux
        """
        swarg1 = m.zw / efold1_shortwave
        swarg2 = m.zw / efold2_shortwave
        pen = rpart_shortwave * np.exp(swarg1) + (1.0 - rpart_shortwave) * np.exp(swarg2)
        self.divpen_shortwave = np.zeros(m.nz)
        self.divpen_shortwave[1:] = (pen[1:] - pen[:-1]) / m.dzt[1:]
        self.divpen_shortwave[0] = pen[0] / m.dzt[0]

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
        t_rest = 30. * 86400.
        cp_0 = 3991.86795711963 # J/kg /K
        fxa = 365 * 86400.

        m = self.main_module
        (n1, f1), (n2, f2) = self._get_periodic_interval((m.itt-1) * m.dt_tracer, fxa, fxa / 12., 12)

        # linearly interpolate wind stress and shift from MITgcm U/V grid to this grid
        m.surface_taux[...] = f1 * self.taux[:, :, n1] + f2 * self.taux[:, :, n2]
        m.surface_tauy[...] = f1 * self.tauy[:, :, n1] + f2 * self.tauy[:, :, n2]

        tkm = self.tke_module
        if tkm.enable_tke:
            tkm.forc_tke_surface[1:-1, 1:-1] = np.sqrt((0.5 * (m.surface_taux[1:-1, 1:-1] + m.surface_taux[:-2, 1:-1])) ** 2 \
                                                      +(0.5 * (m.surface_tauy[1:-1, 1:-1] + m.surface_tauy[1:-1, :-2])) ** 2) ** (3./2.)

        # W/m^2 K kg/J m^3/kg = K m/s
        fxa = f1 * self.t_star[..., n1] + f2 * self.t_star[..., n2]
        self.qqnec = f1 * self.qnec[..., n1] + f2 * self.qnec[..., n2]
        self.qqnet = f1 * self.qnet[..., n1] + f2 * self.qnet[..., n2]
        m.forc_temp_surface[...] = (self.qqnet + self.qqnec * (fxa - m.temp[..., -1, self.get_tau()])) \
                                            * m.maskT[..., -1] / cp_0 / m.rho_0
        fxa = f1 * self.s_star[..., n1] + f2 * self.s_star[..., n2]
        m.forc_salt_surface[...] = 1. / t_rest * (fxa - m.salt[..., -1, self.get_tau()]) * m.maskT[..., -1] * m.dzt[-1]

        # apply simple ice mask
        ice = np.ones((m.nx+4, m.ny+4), dtype=np.uint8)
        mask1 = m.temp[:, :, -1, self.get_tau()] * m.maskT[:, :, -1] <= -1.8
        mask2 = m.forc_temp_surface <= 0
        mask = ~(mask1 & mask2)
        m.forc_temp_surface *= mask
        m.forc_salt_surface *= mask
        ice *= mask

        # solar radiation
        m.temp_source[..., :] = (f1 * self.qsol[..., n1, None] + f2 * self.qsol[..., n2, None]) \
                                        * self.divpen_shortwave[None, None, :] * ice[..., None] \
                                        * m.maskT[..., :] / cp_0 / m.rho_0

    @pyom_method
    def set_topography(self):
        m = self.main_module

        bathymetry_data = self._read_binary("bathymetry")
        salt_data = self._read_binary("salt")[:,:,::-1]

        mask_salt = salt_data == 0.
        m.kbot[2:-2, 2:-2] = 1 + np.sum(mask_salt.astype(np.int), axis=2)

        mask_bathy = bathymetry_data == 0
        m.kbot[2:-2, 2:-2][mask_bathy] = 0

        m.kbot *= m.kbot < m.nz

        # close some channels
        i, j = np.indices((m.nx,m.ny))

        mask_channel = (i >= 207) & (i < 214) & (j < 5) # i = 208,214; j = 1,5
        m.kbot[2:-2, 2:-2][mask_channel] = 0

        # Aleutian islands
        mask_channel = (i == 104) & (j == 134) # i = 105; j = 135
        m.kbot[2:-2, 2:-2][mask_channel] = 0

        # Engl channel
        mask_channel = (i >= 269) & (i < 271) & (j == 130) # i = 270,271; j = 131
        m.kbot[2:-2, 2:-2][mask_channel] = 0

    @pyom_method
    def set_diagnostics(self):
        m = self.main_module
        idm = self.idemix_module
        tkm = self.tke_module
        ekm = self.eke_module
        average_vars = ("surface_taux", "surface_tauy", "forc_temp_surface", "forc_salt_surface",
                        "psi", "temp", "salt", "u", "v", "w", "Nsqr", "Hd", "rho",
                        "K_diss_v", "P_diss_v", "P_diss_nonlin", "P_diss_iso", "kappaH")
        if m.enable_skew_diffusion:
            average_vars += ("B1_gm", "B2_gm")
        if m.enable_TEM_friction:
            average_vars += ("kappa_gm", "K_diss_gm")
        if tkm.enable_tke:
            average_vars += ("tke", "Prandtlnumber", "mxl", "tke_diss",
                             "forc_tke_surface", "tke_surf_corr")
        if idm.enable_idemix:
            average_vars += ("E_iw", "forc_iw_surface", "forc_iw_bottom", "iw_diss",
                             "c0", "v0")
        if ekm.enable_eke:
            average_vars += ("eke", "K_gm", "L_rossby", "L_rhines")
        if idm.enable_idemix_M2:
            average_vars += ("E_M2", "cg_M2", "tau_M2", "alpha_M2_cont")
        if idm.enable_idemix_niw:
            average_vars += ("E_niw", "cg_niw", "tau_niw")
        for var in average_vars:
            self.variables[var].average = True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortran","-f", default=None, help="Path to fortran library")
    args, _ = parser.parse_known_args()
    simulation = GlobalOneDegree(fortran=args.fortran)
    simulation.run(runlen=86400., snapint=86400./2.)
