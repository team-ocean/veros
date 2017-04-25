import os
from netCDF4 import Dataset

from veros import VerosLegacy, veros_method, time, tools
from veros.core import cyclic

MAIN_OPTIONS = dict(
    nx = 360,
    ny = 160,
    nz = 115,
    dt_mom = 1800.0,
    dt_tracer = 1800.0,

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
    eke_diss_surfbot_frac = 0.2,
    enable_idemix_superbee_advection = True,
    enable_idemix_hor_diffusion = True,
)

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
FORCING_FILE = os.path.join(BASE_PATH, "forcing_1deg_global.nc")

class GlobalOneDegree(VerosLegacy):
    """
    Global 1 degree model with 115 vertical levels, translated from setup1.f90.

    https://wiki.zmaw.de/ifm/TO/pyOM2/1x1%20global%20model
    """

    @veros_method
    def set_parameter(self):
        """
        set main parameters
        """
        if not os.path.isfile(FORCING_FILE):
            raise RuntimeError("{} data file {} not found".format(name,filepath))

        self._set_parameters(self.main_module, MAIN_OPTIONS)
        self._set_parameters(self.isoneutral_module, ISONEUTRAL_OPTIONS)
        self._set_parameters(self.tke_module, TKE_OPTIONS)
        self._set_parameters(self.eke_module, EKE_OPTIONS)
        self._set_parameters(self.idemix_module, IDEMIX_OPTIONS)

    @veros_method
    def _set_parameters(self,module,parameters):
        for key, attribute in parameters.items():
            setattr(module,key,attribute)

    @veros_method
    def _read_forcing(self, var):
        with Dataset(FORCING_FILE, "r") as infile:
            return infile.variables[var][...].T

    @veros_method
    def set_grid(self):
        dz_data = self._read_forcing("dz")
        m = self.main_module
        m.dzt[...] = dz_data[::-1]
        m.dxt[...] = 1.0
        m.dyt[...] = 1.0
        m.y_origin = -79.
        m.x_origin = 91.

    @veros_method
    def set_coriolis(self):
        m = self.main_module
        m.coriolis_t[...] = 2 * m.omega * np.sin(m.yt[np.newaxis, :] / 180. * m.pi)

    @veros_method
    def set_topography(self):
        m = self.main_module

        bathymetry_data = self._read_forcing("bathymetry")
        salt_data = self._read_forcing("salinity")[:,:,::-1]

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

    @veros_method
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
        temp_data = self._read_forcing("temperature")
        m.temp[2:-2, 2:-2, :, 0] = temp_data[..., ::-1] * m.maskT[2:-2, 2:-2, :]
        m.temp[2:-2, 2:-2, :, 1] = temp_data[..., ::-1] * m.maskT[2:-2, 2:-2, :]

        salt_data = self._read_forcing("salinity")
        m.salt[2:-2, 2:-2, :, 0] = salt_data[..., ::-1] * m.maskT[2:-2, 2:-2, :]
        m.salt[2:-2, 2:-2, :, 1] = salt_data[..., ::-1] * m.maskT[2:-2, 2:-2, :]

        # wind stress on MIT grid
        taux_data = self._read_forcing("tau_x")
        self.taux[2:-2, 2:-2, :] = taux_data / m.rho_0

        tauy_data = self._read_forcing("tau_y")
        self.tauy[2:-2, 2:-2, :] = tauy_data / m.rho_0

        if m.enable_cyclic_x:
            cyclic.setcyclic_x(self.taux)
            cyclic.setcyclic_x(self.tauy)

        # Qnet and dQ/dT and Qsol
        qnet_data = self._read_forcing("q_net")
        self.qnet[2:-2, 2:-2, :] = -qnet_data * m.maskT[2:-2, 2:-2, -1, np.newaxis]

        qnec_data = self._read_forcing("dqdt")
        self.qnec[2:-2, 2:-2, :] = qnec_data * m.maskT[2:-2, 2:-2, -1, np.newaxis]

        qsol_data = self._read_forcing("swf")
        self.qsol[2:-2, 2:-2, :] = -qsol_data * m.maskT[2:-2, 2:-2, -1, np.newaxis]

        # SST and SSS
        sst_data = self._read_forcing("sst")
        self.t_star[2:-2, 2:-2, :] = sst_data * m.maskT[2:-2, 2:-2, -1, np.newaxis]

        sss_data = self._read_forcing("sss")
        self.s_star[2:-2, 2:-2, :] = sss_data * m.maskT[2:-2, 2:-2, -1, np.newaxis]

        idm = self.idemix_module
        if idm.enable_idemix:
            tidal_energy_data = self._read_forcing("tidal_energy")
            mask_x, mask_y = (i+2 for i in np.indices((m.nx, m.ny)))
            mask_z = np.maximum(0, m.kbot[2:-2, 2:-2] - 1)
            tidal_energy_data[:, :] *= m.maskW[mask_x, mask_y, mask_z] / m.rho_0
            idm.forc_iw_bottom[2:-2, 2:-2] = tidal_energy_data

            wind_energy_data = self._read_forcing("wind_energy")
            wind_energy_data[...] *= m.maskW[2:-2, 2:-2, -1] / m.rho_0 * 0.2
            idm.forc_iw_surface[2:-2, 2:-2] = wind_energy_data

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

    @veros_method
    def set_forcing(self):
        t_rest = 30. * 86400.
        cp_0 = 3991.86795711963 # J/kg /K

        m = self.main_module
        year_in_seconds = time.convert_time(m, 1., "years", "seconds")
        (n1, f1), (n2, f2) = tools.get_periodic_interval(time.current_time(m, "seconds"), year_in_seconds, year_in_seconds / 12., 12)

        # linearly interpolate wind stress and shift from MITgcm U/V grid to this grid
        m.surface_taux[:-1, :] = f1 * self.taux[1:, :, n1] + f2 * self.taux[1:, :, n2]
        m.surface_tauy[:, :-1] = f1 * self.tauy[:, 1:, n1] + f2 * self.tauy[:, 1:, n2]

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

    @veros_method
    def set_diagnostics(self):
        m = self.main_module
        idm = self.idemix_module
        tkm = self.tke_module
        ekm = self.eke_module
        average_vars = ["surface_taux", "surface_tauy", "forc_temp_surface", "forc_salt_surface",
                        "psi", "temp", "salt", "u", "v", "w", "Nsqr", "Hd", "rho",
                        "K_diss_v", "P_diss_v", "P_diss_nonlin", "P_diss_iso", "kappaH"]
        if m.enable_skew_diffusion:
            average_vars += ["B1_gm", "B2_gm"]
        if m.enable_TEM_friction:
            average_vars += ["kappa_gm", "K_diss_gm"]
        if tkm.enable_tke:
            average_vars += ["tke", "Prandtlnumber", "mxl", "tke_diss",
                             "forc_tke_surface", "tke_surf_corr"]
        if idm.enable_idemix:
            average_vars += ["E_iw", "forc_iw_surface", "forc_iw_bottom", "iw_diss",
                             "c0", "v0"]
        if ekm.enable_eke:
            average_vars += ["eke", "K_gm", "L_rossby", "L_rhines"]
        if idm.enable_idemix_M2:
            average_vars += ["E_M2", "cg_M2", "tau_M2", "alpha_M2_cont"]
        if idm.enable_idemix_niw:
            average_vars += ["E_niw", "cg_niw", "tau_niw"]
        self.diagnostics["averages"].output_variables = average_vars


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortran","-f", default=None, help="Path to fortran library")
    args, _ = parser.parse_known_args()
    simulation = GlobalOneDegree(fortran=args.fortran)
    simulation.run(runlen=86400., snapint=86400./2.)
