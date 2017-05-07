import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from test_base import VerosTest
from veros import VerosLegacy, veros_method

yt_start = -39.0
yt_end = 43
yu_start = -40.0
yu_end = 42


class ACC2(VerosLegacy):
    """
    A simple global model with a Southern Ocean and Atlantic part
    """
    @veros_method
    def set_parameter(self):
        self.identifier = "acc2_test"

        m = self.main_module

        (m.nx, m.ny, m.nz) = (30, 42, 15)
        m.dt_mom = 4800
        m.dt_tracer = 86400 / 2.
        m.runlen = 86400 * 365

        m.coord_degree = 1
        m.enable_cyclic_x = 1

        m.congr_epsilon = 1e-12
        m.congr_max_iterations = 5000

        m.enable_diag_snapshots = True
        m.snapint = 86400 * 10
        m.enable_diag_averages = True
        m.aveint = 365 * 86400.
        m.avefreq = m.dt_tracer * 10
        m.enable_diag_overturning = True
        m.overint = 365 * 86400. / 48.
        m.overfreq = m.dt_tracer * 10
        m.enable_diag_ts_monitor = True
        m.ts_monint = 365 * 86400. / 12.
        m.enable_diag_energy = True
        m.energint = 365 * 86400. / 48
        m.energfreq = m.dt_tracer * 10

        i = self.isoneutral_module
        i.enable_neutral_diffusion = 1
        i.K_iso_0 = 1000.0
        i.K_iso_steep = 500.0
        i.iso_dslope = 0.005
        i.iso_slopec = 0.01
        i.enable_skew_diffusion = 1

        m.enable_hor_friction = 1
        m.A_h = (2 * m.degtom)**3 * 2e-11
        m.enable_hor_friction_cos_scaling = 1
        m.hor_friction_cosPower = 1

        m.enable_bottom_friction = 1
        m.r_bot = 1e-5

        m.enable_implicit_vert_friction = 1
        t = self.tke_module
        t.enable_tke = 1
        t.c_k = 0.1
        t.c_eps = 0.7
        t.alpha_tke = 30.0
        t.mxl_min = 1e-8
        t.tke_mxl_choice = 2
        # t.enable_tke_superbee_advection = 1

        i.K_gm_0 = 1000.0
        e = self.eke_module
        e.enable_eke = 1
        e.eke_k_max = 1e4
        e.eke_c_k = 0.4
        e.eke_c_eps = 0.5
        e.eke_cross = 2.
        e.eke_crhin = 1.0
        e.eke_lmin = 100.0
        e.enable_eke_superbee_advection = 1
        e.enable_eke_isopycnal_diffusion = 1

        i = self.idemix_module
        i.enable_idemix = 1
        i.enable_idemix_hor_diffusion = 1
        i.enable_eke_diss_surfbot = 1
        i.eke_diss_surfbot_frac = 0.2
        i.enable_idemix_superbee_advection = 1

        m.eq_of_state_type = 3

    @veros_method
    def set_grid(self):
        m = self.main_module
        ddz = [50., 70., 100., 140., 190., 240., 290., 340.,
               390., 440., 490., 540., 590., 640., 690.]
        m.dxt[:] = 2.0
        m.dyt[:] = 2.0
        m.x_origin = 0.0
        m.y_origin = -40.0
        m.dzt[:] = ddz[::-1]
        m.dzt[:] *= 1 / 2.5

    @veros_method
    def set_coriolis(self):
        m = self.main_module
        m.coriolis_t[:, :] = 2 * m.omega * np.sin(m.yt[None, :] / 180. * np.pi)

    @veros_method
    def set_topography(self):
        m = self.main_module
        (X, Y) = np.meshgrid(m.xt, m.yt)
        X = X.transpose()
        Y = Y.transpose()
        m.kbot[...] = (X > 1.0) | (Y < -20)

    @veros_method
    def set_initial_conditions(self):
        m = self.main_module

        # initial conditions
        m.temp[:, :, :, 0:2] = ((1 - m.zt[None, None, :] / m.zw[0]) * 15 * m.maskT)[..., None]
        m.salt[:, :, :, 0:2] = 35.0 * m.maskT[..., None]

        # wind stress forcing
        taux = np.zeros(m.ny + 1)
        yt = m.yt[2:m.ny + 3]
        taux = (.1e-3 * np.sin(np.pi * (m.yu[2:m.ny + 3] - yu_start) / (-20.0 - yt_start))) * (yt < -20) \
            + (.1e-3 * (1 - np.cos(2 * np.pi *
                                   (m.yu[2:m.ny + 3] - 10.0) / (yu_end - 10.0)))) * (yt > 10)
        m.surface_taux[:, 2:m.ny + 3] = taux * m.maskU[:, 2:m.ny + 3, -1]

        # surface heatflux forcing
        self.t_star = 15 * np.invert((m.yt < -20) | (m.yt > 20)) \
            + 15 * (m.yt - yt_start) / (-20 - yt_start) * (m.yt < -20) \
            + 15 * (1 - (m.yt - 20) / (yt_end - 20)) * (m.yt > 20.)
        self.t_rest = m.dzt[None, -1] / (30. * 86400.) * m.maskT[:, :, -1]

        t = self.tke_module
        if t.enable_tke:
            t.forc_tke_surface[2:-2, 2:-2] = np.sqrt((0.5 * (m.surface_taux[2:-2, 2:-2] + m.surface_taux[1:-3, 2:-2]))**2
                                                     + (0.5 * (m.surface_tauy[2:-2, 2:-2] + m.surface_tauy[2:-2, 1:-3]))**2)**(1.5)

        i = self.idemix_module
        if i.enable_idemix:
            i.forc_iw_bottom[:] = 1.0e-6 * m.maskW[:, :, -1]
            i.forc_iw_surface[:] = 0.1e-6 * m.maskW[:, :, -1]

    @veros_method
    def set_forcing(self):
        m = self.main_module
        m.forc_temp_surface[:] = self.t_rest * (self.t_star - m.temp[:, :, -1, self.get_tau()])

    @veros_method
    def set_diagnostics(self):
        self.diagnostics["averages"].output_variables = (
            "salt", "temp", "u", "v", "w", "psi", "surface_taux", "surface_tauy")


class ACC2Test(VerosTest):
    timesteps = 5

    def __init__(self, *args, **kwargs):
        try:
            self.fortran = kwargs["fortran"]
        except KeyError:
            try:
                self.fortran = sys.argv[1]
            except IndexError:
                raise RuntimeError("Path to fortran library must be given via keyword argument or command line")

    def _check_all_objects(self):
        differing_scalars = self.check_scalar_objects()
        differing_arrays = self.check_array_objects()
        passed = True
        if differing_scalars or differing_arrays:
            print("The following attributes do not match between old and new veros:")
            for s, (v1, v2) in differing_scalars.items():
                print("{}, {}, {}".format(s,v1,v2))
            for a, (v1, v2) in differing_arrays.items():
                if v1 is None:
                    print(a, v1, "")
                    continue
                if v2 is None:
                    print(a, "", v2)
                    continue
                if "salt" in a or a in ("B1_gm","B2_gm"): # salt and isoneutral streamfunctions aren't used by this example
                    continue
                if a in ("psi", "psin"): # top row contains noise and is not part of the solution
                    v1[2,:] = 0.
                    v2[2,:] = 0.
                passed = self.check_variable(a,atol=1e-5,data=(v1,v2)) and passed
        plt.show()
        return passed

    def run(self):
        self.veros_new = ACC2()
        self.veros_new.diskless_mode = True
        self.veros_new.pyom_compatibility_mode = True
        self.veros_new.setup()

        self.veros_legacy = ACC2(fortran=self.fortran)
        self.veros_legacy.diskless_mode = True
        self.veros_legacy.setup()

        # integrate for some time steps and compare
        if self.timesteps > 0:
            self.veros_new.runlen = self.timesteps * 86400. / 2
            self.veros_new.run()
            self.veros_legacy.fortran.main_module.runlen = self.timesteps * 86400. / 2
            self.veros_legacy.run()
        return self._check_all_objects()

if __name__ == "__main__":
    passed = ACC2Test().run()
    sys.exit(int(not passed))
