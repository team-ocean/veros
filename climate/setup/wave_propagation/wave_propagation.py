import os
import numpy as np
from netCDF4 import Dataset
from PIL import Image

from climate import tools
from climate.pyom import PyOM, pyom_method
from climate.pyom.core import cyclic

BASE_PATH = os.path.dirname(os.path.realpath(__file__))

DATA_FILES = dict(
    temperature = "lev_clim_temp.npy",
    salt = "lev_clim_salt.npy",
    sss = "lev_sss.npy",
    tau_x = "ECMWFBB_taux.npy",
    tau_y = "ECMWFBB_tauy.npy",
    q_net = "ECMWFBB_qnet.npy",
    dqdt = "ECMWFBB_dqdt.npy",
    swf = "ECMWFBB_swf.npy",
    sst = "ECMWFBB_target_sst.npy",
    tidal_energy = "tidal_energy.npy",
)
DATA_FILES = {key: os.path.join(BASE_PATH, val) for key, val in DATA_FILES.items()}

class WavePropagation(PyOM):
    """
    Global model with flexible resolution and idealized geometry in the
    Atlantic to examine coastal wave propagation.
    """

    @pyom_method
    def set_parameter(self):
        self.nx = 360
        self.ny = 160
        self.nz = 115
        self._max_depth = 8000
        self.dt_mom = 3600.0
        self.dt_tracer = 3600.0

        self.coord_degree = True
        self.enable_cyclic_x = True

        self.congr_epsilon = 1e-10
        self.congr_max_iterations = 10000
        self.enable_streamfunction = True

        self.enable_hor_friction = True
        self.A_h = 5e4
        self.enable_hor_friction_cos_scaling = True
        self.hor_friction_cosPower = 1
        self.enable_tempsalt_sources = True
        self.enable_implicit_vert_friction = True

        self.eq_of_state_type = 3

        self.enable_neutral_diffusion = True
        self.K_iso_0 = 1000.0
        self.K_iso_steep = 50.0
        self.iso_dslope = 0.005
        self.iso_slopec = 0.005
        self.enable_skew_diffusion = True

        self.enable_tke = True
        self.c_k = 0.1
        self.c_eps = 0.7
        self.alpha_tke = 30.0
        self.mxl_min = 1e-8
        self.tke_mxl_choice = 2
        self.enable_tke_superbee_advection = True

        self.enable_eke = True
        self.eke_k_max = 1e4
        self.eke_c_k = 0.4
        self.eke_c_eps = 0.5
        self.eke_cross = 2.
        self.eke_crhin = 1.0
        self.eke_lmin = 100.0
        self.enable_eke_superbee_advection = True
        self.enable_eke_isopycnal_diffusion = True

        self.enable_idemix = True
        self.enable_eke_diss_surfbot = True
        self.eke_diss_surfbot_frac = 0.2
        self.enable_idemix_superbee_advection = True
        self.enable_idemix_hor_diffusion = True

    def _set_parameters(self,module,parameters):
        for key, attribute in parameters.items():
            setattr(module,key,attribute)

    def _read_binary(self, var):
        return np.load(DATA_FILES[var])

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

    def set_grid(self):
        self.dzt[...] = tools.gaussian_spacing(self.nz, self._max_depth, min_spacing=10.)
        self.dxt[...] = 360. / self.nx
        self.dyt[...] = 160. / self.ny
        self.y_origin = -80. + 160. / self.ny
        self.x_origin = 90. + 360. / self.nx

    def set_coriolis(self):
        self.coriolis_t[...] = 2 * self.omega * np.sin(self.yt[np.newaxis, :] / 180. * self.pi)

    @pyom_method
    def set_topography(self):
        bathymetry_data = self._read_binary("bathymetry")
        salt_data = self._read_binary("salt")[:,:,::-1]

        mask_salt = salt_data == 0.
        self.kbot[2:-2, 2:-2] = 1 + np.sum(mask_salt.astype(np.int), axis=2)

        mask_bathy = bathymetry_data == 0
        self.kbot[2:-2, 2:-2][mask_bathy] = 0

        self.kbot *= self.kbot < self.nz

        # close some channels
        i, j = np.indices((self.nx,self.ny))

        mask_channel = (i >= 207) & (i < 214) & (j < 5) # i = 208,214; j = 1,5
        self.kbot[2:-2, 2:-2][mask_channel] = 0

        # Aleutian islands
        mask_channel = (i == 104) & (j == 134) # i = 105; j = 135
        self.kbot[2:-2, 2:-2][mask_channel] = 0

        # Engl channel
        mask_channel = (i >= 269) & (i < 271) & (j == 130) # i = 270,271; j = 131
        self.kbot[2:-2, 2:-2][mask_channel] = 0

    def set_initial_conditions(self):
        self._t_star = np.zeros((self.nx+4, self.ny+4, 12))
        self._s_star = np.zeros((self.nx+4, self.ny+4, 12))
        self._qnec = np.zeros((self.nx+4, self.ny+4, 12))
        self._qnet = np.zeros((self.nx+4, self.ny+4, 12))
        self._qsol = np.zeros((self.nx+4, self.ny+4, 12))
        self._divpen_shortwave = np.zeros(self.nz)
        self._taux = np.zeros((self.nx+4, self.ny+4, 12))
        self._tauy = np.zeros((self.nx+4, self.ny+4, 12))

        rpart_shortwave = 0.58
        efold1_shortwave = 0.35
        efold2_shortwave = 23.0

        # initial conditions
        temp_data = self._read_binary("temperature")
        self.temp[2:-2, 2:-2, :, 0] = temp_data[..., ::-1] * self.maskT[2:-2, 2:-2, :]
        self.temp[2:-2, 2:-2, :, 1] = temp_data[..., ::-1] * self.maskT[2:-2, 2:-2, :]

        salt_data = self._read_binary("salt")
        self.salt[2:-2, 2:-2, :, 0] = salt_data[..., ::-1] * self.maskT[2:-2, 2:-2, :]
        self.salt[2:-2, 2:-2, :, 1] = salt_data[..., ::-1] * self.maskT[2:-2, 2:-2, :]

        # wind stress on MIT grid
        taux_data = self._read_binary("tau_x")
        self._taux[2:-2, 2:-2, :] = taux_data / self.rho_0
        self._taux[self._taux < -99.9] = 0.

        tauy_data = self._read_binary("tau_y")
        self._tauy[2:-2, 2:-2, :] = tauy_data / self.rho_0
        self._tauy[self._tauy < -99.9] = 0.

        if self.enable_cyclic_x:
            cyclic.setcyclic_x(self._taux)
            cyclic.setcyclic_x(self._tauy)

        # Qnet and dQ/dT and Qsol
        qnet_data = self._read_binary("q_net")
        self._qnet[2:-2, 2:-2, :] = -qnet_data * self.maskT[2:-2, 2:-2, -1, np.newaxis]

        qnec_data = self._read_binary("dqdt")
        self._qnec[2:-2, 2:-2, :] = qnec_data * self.maskT[2:-2, 2:-2, -1, np.newaxis]

        qsol_data = self._read_binary("swf")
        self._qsol[2:-2, 2:-2, :] = -qsol_data * self.maskT[2:-2, 2:-2, -1, np.newaxis]

        # SST and SSS
        sst_data = self._read_binary("sst")
        self._t_star[2:-2, 2:-2, :] = sst_data * self.maskT[2:-2, 2:-2, -1, np.newaxis]

        sss_data = self._read_binary("sss")
        self._s_star[2:-2, 2:-2, :] = sss_data * self.maskT[2:-2, 2:-2, -1, np.newaxis]

        if self.enable_idemix:
            tidal_energy_data = self._read_binary("tidal_energy")
            mask_x, mask_y = (i+2 for i in np.indices((self.nx, self.ny)))
            mask_z = np.maximum(0, self.kbot[2:-2, 2:-2] - 1)
            tidal_energy_data[:, :] *= self.maskW[mask_x, mask_y, mask_z] / self.rho_0
            self.forc_iw_bottom[2:-2, 2:-2] = tidal_energy_data

        """
        Initialize penetration profile for solar radiation
        and store divergence in divpen
        note that pen(nz) is set 0.0 instead of 1.0 to compensate for the
        shortwave part of the total surface flux
        """
        swarg1 = self.zw / efold1_shortwave
        swarg2 = self.zw / efold2_shortwave
        pen = rpart_shortwave * np.exp(swarg1) + (1.0 - rpart_shortwave) * np.exp(swarg2)
        self._divpen_shortwave = np.zeros(self.nz)
        self._divpen_shortwave[1:] = (pen[1:] - pen[:-1]) / self.dzt[1:]
        self._divpen_shortwave[0] = pen[0] / self.dzt[0]

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

        (n1, f1), (n2, f2) = self._get_periodic_interval((self.itt-1) * self.dt_tracer, fxa, fxa / 12., 12)

        # linearly interpolate wind stress and shift from MITgcm U/V grid to this grid
        self.surface_taux[...] = f1 * self._taux[:, :, n1] + f2 * self._taux[:, :, n2]
        self.surface_tauy[...] = f1 * self._tauy[:, :, n1] + f2 * self._tauy[:, :, n2]

        if self.enable_tke:
            self.forc_tke_surface[1:-1, 1:-1] = np.sqrt((0.5 * (self.surface_taux[1:-1, 1:-1] + self.surface_taux[:-2, 1:-1])) ** 2 \
                                                      +(0.5 * (self.surface_tauy[1:-1, 1:-1] + self.surface_tauy[1:-1, :-2])) ** 2) ** (3./2.)

        # W/m^2 K kg/J m^3/kg = K m/s
        fxa = f1 * self._t_star[..., n1] + f2 * self._t_star[..., n2]
        self._qqnec = f1 * self._qnec[..., n1] + f2 * self._qnec[..., n2]
        self._qqnet = f1 * self._qnet[..., n1] + f2 * self._qnet[..., n2]
        self.forc_temp_surface[...] = (self._qqnet + self._qqnec * (fxa - self.temp[..., -1, self.tau])) \
                                            * self.maskT[..., -1] / cp_0 / self.rho_0
        fxa = f1 * self._s_star[..., n1] + f2 * self._s_star[..., n2]
        self.forc_salt_surface[...] = 1. / t_rest * (fxa - self.salt[..., -1, self.tau]) * self.maskT[..., -1] * self.dzt[-1]

        # apply simple ice mask
        ice = np.ones((self.nx+4, self.ny+4), dtype=np.uint8)
        mask1 = self.temp[:, :, -1, self.get_tau()] * self.maskT[:, :, -1] <= -1.8
        mask2 = self.forc_temp_surface <= 0
        mask = ~(mask1 & mask2)
        self.forc_temp_surface[...] *= mask
        self.forc_salt_surface[...] *= mask
        ice *= mask

        # solar radiation
        self.temp_source[..., :] = (f1 * self._qsol[..., n1, None] + f2 * self._qsol[..., n2, None]) \
                                        * self._divpen_shortwave[None, None, :] * ice[..., None] \
                                        * self.maskT[..., :] / cp_0 / self.rho_0

    @pyom_method
    def set_diagnostics(self):
        self.diagnostics["cfl_monitor"].output_frequency = 86400.0
        self.diagnostics["snapshots"].output_frequency = 0.5 * 86400.
        self.diagnostics["overturning"].output_frequency = 365 * 86400
        self.diagnostics["overturning"].sampling_frequency = 365 * 86400 / 24.
        self.diagnostics["energy"].output_frequency = 365 * 86400
        self.diagnostics["energy"].sampling_frequency = 365 * 86400 / 24.
        self.diagnostics["averages"].output_frequency = 365 * 86400
        self.diagnostics["averages"].sampling_frequency = 365 * 86400 / 24.

        average_vars = ("surface_taux", "surface_tauy", "forc_temp_surface", "forc_salt_surface",
                        "psi", "temp", "salt", "u", "v", "w", "Nsqr", "Hd", "rho",
                        "K_diss_v", "P_diss_v", "P_diss_nonlin", "P_diss_iso", "kappaH")
        if self.enable_skew_diffusion:
            average_vars += ("B1_gm", "B2_gm")
        if self.enable_TEM_friction:
            average_vars += ("kappa_gm", "K_diss_gm")
        if self.enable_tke:
            average_vars += ("tke", "Prandtlnumber", "mxl", "tke_diss",
                             "forc_tke_surface", "tke_surf_corr")
        if self.enable_idemix:
            average_vars += ("E_iw", "forc_iw_surface", "forc_iw_bottom", "iw_diss",
                             "c0", "v0")
        if self.enable_eke:
            average_vars += ("eke", "K_gm", "L_rossby", "L_rhines")

        for var in average_vars:
            self.variables[var].average = True


if __name__ == "__main__":
    WavePropagation(fortran=args.fortran).run(runlen=86400.)
